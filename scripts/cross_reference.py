#!/usr/bin/env python3
"""
Cross-reference plan items against git evidence.

Gathers evidence for each plan item — where patterns were found (diff, current
files, or not at all). Does NOT make implementation verdicts; that's the LLM's
job. The script provides structured evidence the LLM uses to evaluate each item.

Evidence levels per item:
- ✅ IN_DIFF: all patterns found in diff added lines (strong signal of new work)
- 🔍 MIXED: some in diff, others only in current files or not found
- ⚠️ PRE_EXISTING: patterns exist in codebase but NOT in diff (name match only)
- ❌ NOT_FOUND: patterns not found anywhere
- ⏭️ SKIPPED: no file target or patterns to verify mechanically

Also performs grep-level dead-code detection across the full codebase.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

from languages import detect_language


# Evidence level constants — describe WHERE patterns were found, not whether
# the plan was implemented. Implementation verdicts are the LLM's job.
IN_DIFF = '✅'          # All patterns found in diff added lines
MIXED = '🔍'           # Some in diff, others only in current files or not found
PRE_EXISTING = '⚠️'    # Pattern names found in codebase but NOT in diff
NOT_FOUND = '❌'       # Not found anywhere
SKIPPED = '⏭️'         # Nothing to verify mechanically


def find_file_in_evidence(file_pattern: Optional[str], modified_files: list[str],
                          file_diffs: dict[str, str]) -> Optional[str]:
    """Find the actual file path matching a plan's file_pattern."""
    if not file_pattern:
        return None

    # Exact match
    if file_pattern in file_diffs:
        return file_pattern

    # Basename match
    basename = Path(file_pattern).name
    for f in modified_files:
        if Path(f).name == basename:
            return f

    # Partial path match (plan might say "normalizers/base.ts",
    # actual path is "src/core/normalizers/base.ts")
    clean = file_pattern.lstrip('./')
    for f in modified_files:
        if f.endswith(clean):
            return f

    return None


def _extract_added_lines(diff_text: str) -> tuple[list[str], str]:
    """Extract added lines from a unified diff. Returns (lines, joined_text)."""
    added_lines = []
    for line in diff_text.split('\n'):
        if line.startswith('+') and not line.startswith('+++'):
            added_lines.append(line[1:])  # strip the leading +
    return added_lines, '\n'.join(added_lines)


def _find_pattern_in_text(pattern: str, lines: list[str], joined_text: str,
                          snippet_prefix: str = '') -> tuple[bool, Optional[str]]:
    """Search for a pattern in text lines, trying exact then case-insensitive.

    Returns (found, evidence_snippet).
    """
    if pattern in joined_text:
        for line in lines:
            if pattern in line:
                snippet = line.strip()
                if len(snippet) > 100:
                    snippet = snippet[:100] + '...'
                return True, f'{snippet_prefix}{snippet}'
        return True, None

    escaped = re.escape(pattern)
    if re.search(escaped, joined_text, re.IGNORECASE):
        for line in lines:
            if re.search(escaped, line, re.IGNORECASE):
                snippet = line.strip()
                if len(snippet) > 100:
                    snippet = snippet[:100] + '...'
                return True, f'{snippet_prefix}{snippet} (case-insensitive match)'
        return True, None

    return False, None


def check_pattern_in_diff(pattern: str, diff_text: str,
                          _cache: Optional[tuple[list[str], str]] = None) -> tuple[bool, Optional[str]]:
    """Check if a pattern appears in added lines of a diff.

    Returns (found, evidence_snippet).
    Pass _cache=(added_lines, added_text) to avoid re-parsing the same diff.
    """
    if _cache is not None:
        added_lines, added_text = _cache
    else:
        added_lines, added_text = _extract_added_lines(diff_text)

    return _find_pattern_in_text(pattern, added_lines, added_text, snippet_prefix='+')


def check_pattern_in_file(pattern: str, file_content: str) -> tuple[bool, Optional[str]]:
    """Check if a pattern exists in current file contents.

    Returns (found, evidence_snippet).
    """
    lines = file_content.split('\n')
    return _find_pattern_in_text(pattern, lines, file_content)


def _looks_like_string_literal(pattern: str) -> bool:
    """Check if a pattern looks like a string/enum value rather than an identifier."""
    if re.match(r'^[a-z][a-z0-9]*(_[a-z0-9]+)+$', pattern):
        return True
    return False


def _build_dead_code_texts(current_files: dict[str, str],
                           declaring_file: str) -> tuple[str, str, str]:
    """Build text blobs for dead-code checking. Returns (declaring, other, all)."""
    declaring_text = ''
    other_parts = []
    for fpath, content in current_files.items():
        if Path(fpath).name == Path(declaring_file).name:
            declaring_text = content
        else:
            other_parts.append(content)
    other_files_text = '\n'.join(other_parts)
    return declaring_text, other_files_text, declaring_text + '\n' + other_files_text


def check_dead_code(pattern: str, category: str,
                    declaring_file: Optional[str],
                    declaring_text: str, other_files_text: str,
                    all_text: str) -> tuple[bool, Optional[str]]:
    """Check if a declared symbol is actually used (not dead code).

    Returns (is_dead, explanation).
    """
    if not declaring_file:
        return False, None

    if _looks_like_string_literal(pattern):
        return False, None

    if category == 'type_definition':
        if pattern in other_files_text:
            return False, None
        uses_in_file = len(re.findall(re.escape(pattern), declaring_text))
        if uses_in_file <= 1:
            return True, f'{pattern} declared but not referenced in any other file'
        return False, None

    elif category == 'function':
        lang_spec = detect_language(file_path=declaring_file)
        call_tmpl = lang_spec.get('call_pattern', r'{name}\s*\(')
        call_regex = call_tmpl.replace('{name}', re.escape(pattern))

        all_calls = re.findall(call_regex, all_text)
        if all_calls:
            if len(all_calls) <= 1:
                return True, f'{pattern}() declared but never called'
            return False, None
        else:
            if pattern in all_text:
                return False, None
            return True, f'{pattern} declared but not referenced'

    elif category == 'field':
        lang_spec = detect_language(file_path=declaring_file)
        access_tmpl = lang_spec.get('access_pattern', r'\.{name}\b')

        assign_pattern = rf'{re.escape(pattern)}\s*[:=]'
        read_pattern = access_tmpl.replace('{name}', re.escape(pattern))

        has_assign = bool(re.search(assign_pattern, all_text))
        has_read = bool(re.search(read_pattern, other_files_text))

        if not has_assign and not has_read:
            return True, f'{pattern} declared on type but never assigned or read'
        if not has_assign:
            return True, f'{pattern} read by consumers but never assigned a value'
        if not has_read:
            if not re.search(read_pattern, declaring_text):
                return True, f'{pattern} assigned but never read by any consumer'

        return False, None

    return False, None


def _determine_evidence_level(diff_hits: int, file_hits: int, misses: int) -> str:
    """Determine the evidence level from per-pattern hit counts."""
    total = diff_hits + file_hits + misses
    if total == 0:
        return SKIPPED
    if diff_hits == total:
        return IN_DIFF
    if diff_hits > 0:
        return MIXED
    if file_hits > 0:
        return PRE_EXISTING
    return NOT_FOUND


def _search_pattern(pattern: str, target_file: Optional[str],
                    file_diffs: dict[str, str], added_cache: dict,
                    current_files: dict[str, str],
                    is_test: bool = False) -> tuple[str, str]:
    """Search for a pattern across diffs and current files.

    Returns (hit_type, evidence_string) where hit_type is one of:
    'diff', 'diff_other', 'file', 'file_other', 'miss'.
    """
    # 1. Check target file's diff
    if target_file and target_file in file_diffs:
        found, snippet = check_pattern_in_diff(
            pattern, file_diffs[target_file], _cache=added_cache.get(target_file))
        if found:
            ev = f'"{pattern}" in diff'
            if snippet:
                ev += f': {snippet}'
            return 'diff', ev

    # 2. Check other files' diffs
    for f, diff_text in file_diffs.items():
        if f == target_file:
            continue
        if is_test and not ('test' in f.lower() or 'spec' in f.lower()):
            continue
        found, snippet = check_pattern_in_diff(pattern, diff_text, _cache=added_cache.get(f))
        if found:
            ev = f'"{pattern}" in diff of {f}'
            if target_file:
                ev += f' (expected in {target_file})'
            if snippet:
                ev += f': {snippet}'
            return 'diff_other', ev

    # 3. Check target file's current contents
    if target_file:
        file_content = current_files.get(target_file, '')
        if file_content:
            found, snippet = check_pattern_in_file(pattern, file_content)
            if found:
                ev = f'"{pattern}" exists in {target_file} but NOT in diff — name match only, cannot confirm plan changes'
                if snippet:
                    ev += f': {snippet}'
                return 'file', ev

    # 4. Check other files' current contents
    for f, content in current_files.items():
        if f == target_file:
            continue
        if is_test and not ('test' in f.lower() or 'spec' in f.lower()):
            continue
        found, snippet = check_pattern_in_file(pattern, content)
        if found:
            ev = f'"{pattern}" exists in {f} but NOT in diff — name match only, cannot confirm plan changes'
            if snippet:
                ev += f': {snippet}'
            return 'file_other', ev

    # 5. Not found anywhere
    where = f'{target_file} ' if target_file else ''
    return 'miss', f'"{pattern}" not found in {where}diff or current files'


def cross_reference(plan_items: list[dict], evidence: dict) -> list[dict]:
    """Cross-reference plan items against evidence, producing per-item evidence."""
    file_diffs = evidence.get('file_diffs', {})
    modified_files = evidence.get('modified_files', [])
    current_files = evidence.get('current_files', {})

    # Pre-compute added lines per file to avoid re-parsing diffs
    added_cache: dict[str, tuple[list[str], str]] = {}
    for f, diff_text in file_diffs.items():
        added_cache[f] = _extract_added_lines(diff_text)

    # Pre-compute dead-code text blobs per declaring file
    dead_code_cache: dict[str, tuple[str, str, str]] = {}

    results = []

    for item in plan_items:
        result = {
            'id': item['id'],
            'change_id': item.get('change_id', ''),
            'change_title': item.get('change_title', ''),
            'sub_id': item.get('sub_id', ''),
            'description': item.get('description', ''),
            'file_pattern': item.get('file_pattern'),
            'category': item.get('category', 'wiring'),
            'evidence_level': NOT_FOUND,
            'evidence': [],
            'dead_code_findings': [],
        }

        file_pattern = item.get('file_pattern')
        expected_patterns = item.get('expected_patterns', [])

        # If no file target and no patterns, we can't verify mechanically
        if not file_pattern and not expected_patterns:
            result['evidence_level'] = SKIPPED
            result['evidence'].append('No file target or patterns to verify')
            results.append(result)
            continue

        # Resolve target file in the diff
        actual_file = find_file_in_evidence(file_pattern, modified_files, file_diffs) if file_pattern else None

        # Track file-level evidence
        if file_pattern and not actual_file:
            result['evidence'].append(f'File "{file_pattern}" not modified in diff')

        # If no patterns to check, evidence is just whether the file was touched
        if not expected_patterns:
            if actual_file:
                result['evidence_level'] = IN_DIFF
                result['evidence'].append(f'{actual_file} was modified in diff')
            elif file_pattern:
                # Check if file exists at all
                if file_pattern in current_files or any(
                    Path(f).name == Path(file_pattern).name for f in current_files
                ):
                    result['evidence_level'] = PRE_EXISTING
                    result['evidence'].append(f'{file_pattern} exists but was not modified')
                else:
                    result['evidence_level'] = NOT_FOUND
            else:
                result['evidence_level'] = SKIPPED
                result['evidence'].append('No file target or patterns to verify')
            results.append(result)
            continue

        # Search for each pattern
        is_test = item.get('category') == 'test'
        diff_hits = 0
        file_hits = 0
        misses = 0

        # Pre-build dead-code texts for this declaring file (once per file)
        if actual_file and actual_file not in dead_code_cache:
            dead_code_cache[actual_file] = _build_dead_code_texts(current_files, actual_file)

        for pattern in expected_patterns:
            hit_type, ev_str = _search_pattern(
                pattern, actual_file, file_diffs, added_cache,
                current_files, is_test=is_test,
            )
            result['evidence'].append(ev_str)

            if hit_type == 'diff':
                diff_hits += 1
                # Dead-code check for patterns found in diff
                if actual_file:
                    dc_texts = dead_code_cache.get(actual_file, ('', '', ''))
                    is_dead, dead_reason = check_dead_code(
                        pattern, item.get('category', 'wiring'),
                        actual_file, *dc_texts,
                    )
                    if is_dead:
                        result['dead_code_findings'].append(dead_reason)
            elif hit_type == 'diff_other':
                diff_hits += 1  # still in a diff, just wrong file
            elif hit_type in ('file', 'file_other'):
                file_hits += 1
            else:
                misses += 1

        result['evidence_level'] = _determine_evidence_level(diff_hits, file_hits, misses)

        results.append(result)

    return results


def generate_report(results: list[dict], plan_title: str,
                    report_context: Optional[dict] = None) -> str:
    """Generate the evidence report.

    This report presents WHERE patterns were found — not whether the plan
    was implemented. The LLM evaluates implementation based on this evidence.
    """
    lines = []
    ctx = report_context or {}

    # Title
    audit_date = ctx.get('audit_date', '')
    plan_file = Path(ctx.get('plan_path', '')).name if ctx.get('plan_path') else ''
    date_part = f' {audit_date[:10]}' if audit_date else ''
    file_part = f' ({plan_file})' if plan_file else ''
    clean_title = plan_title.removeprefix('Plan: ').removeprefix('plan: ')
    lines.append(f'# Plan Evidence Report{date_part} for {clean_title}{file_part}')
    lines.append('')

    # Context block
    if ctx:
        lines.append('## Context')
        lines.append('')
        if ctx.get('plan_path'):
            lines.append(f'- **Plan:** `{ctx["plan_path"]}`')
        if ctx.get('plan_mtime_str'):
            lines.append(f'- **Plan last modified:** {ctx["plan_mtime_str"]}')
        if ctx.get('repo'):
            lines.append(f'- **Repository:** `{ctx["repo"]}`')
        if ctx.get('branch'):
            lines.append(f'- **Branch:** `{ctx["branch"]}`')
        if ctx.get('scope'):
            scope_labels = {
                'branch': f'Committed changes vs `{ctx.get("base", "main")}`',
                'plan': 'Changes since plan was last modified',
                'uncommitted': 'Uncommitted changes only',
                'all': f'All changes (committed + uncommitted) vs `{ctx.get("base", "main")}`',
            }
            lines.append(f'- **Scope:** {scope_labels.get(ctx["scope"], ctx["scope"])}')
        if ctx.get('head_sha'):
            sha = ctx['head_sha'][:10]
            subject = ctx.get('head_subject', '')
            lines.append(f'- **HEAD:** `{sha}` {subject}')
        if ctx.get('uncommitted_count') is not None:
            n = ctx['uncommitted_count']
            label = f'{n} file{"s" if n != 1 else ""} with uncommitted changes' if n > 0 else 'clean working tree'
            lines.append(f'- **Working tree:** {label}')
        if ctx.get('audit_date'):
            lines.append(f'- **Audit date:** {ctx["audit_date"]}')
        lines.append('')

    # Evidence summary
    total = len(results)
    level_counts = Counter(r['evidence_level'] for r in results)
    in_diff = level_counts[IN_DIFF]
    mixed = level_counts[MIXED]
    pre_existing = level_counts[PRE_EXISTING]
    not_found = level_counts[NOT_FOUND]
    skipped = level_counts[SKIPPED]
    has_dead = sum(1 for r in results if r['dead_code_findings'])

    lines.append('## Evidence Summary')
    lines.append('')
    lines.append(f'- **{total}** plan items checked')
    lines.append(f'- **{in_diff}** confirmed in diff {IN_DIFF}')
    if mixed:
        lines.append(f'- **{mixed}** partially in diff {MIXED}')
    if pre_existing:
        lines.append(f'- **{pre_existing}** name exists, not in diff {PRE_EXISTING}')
    if not_found:
        lines.append(f'- **{not_found}** not found {NOT_FOUND}')
    if skipped:
        lines.append(f'- **{skipped}** skipped (not mechanically verifiable) {SKIPPED}')
    if has_dead:
        lines.append(f'- **{has_dead}** with dead-code signals')
    lines.append('')

    needs_review = mixed + pre_existing + not_found
    if needs_review > 0:
        lines.append(f'**{needs_review} item(s) need LLM verification** — pattern matching alone cannot confirm implementation.')
    elif in_diff == total - skipped and not has_dead:
        lines.append('All verifiable patterns found in diff added lines.')
    lines.append('')

    # Group results by change_id
    changes = {}
    for r in results:
        cid = r['change_id'] or 'Ungrouped'
        if cid not in changes:
            changes[cid] = {
                'title': r.get('change_title', cid),
                'items': []
            }
        changes[cid]['items'].append(r)

    # Detailed evidence per change
    lines.append('## Detailed Evidence')
    lines.append('')

    for cid, change in changes.items():
        lines.append(f'### {cid}: {change["title"]}')
        lines.append('')
        lines.append('| # | Sub | Item | Evidence | Details |')
        lines.append('|---|-----|------|----------|---------|')

        for r in change['items']:
            level = r['evidence_level']
            sub_id = r.get('sub_id', '') or ''
            desc = r['description'][:60]
            evidence = '; '.join(r['evidence'][:2])  # first 2 pieces
            if len(evidence) > 80:
                evidence = evidence[:80] + '...'
            # Escape pipe characters in table cells
            desc = desc.replace('|', '\\|')
            evidence = evidence.replace('|', '\\|')
            sub_id = sub_id.replace('|', '\\|')
            lines.append(f'| {r["id"]} | {sub_id} | {desc} | {level} | {evidence} |')

        lines.append('')

    # Dead code findings section
    all_dead = [(r, finding) for r in results for finding in r['dead_code_findings']]
    if all_dead:
        lines.append('## Dead Code Signals')
        lines.append('')
        lines.append('These patterns were found in the diff but may not be actively used:')
        lines.append('')
        lines.append('| Signal | File | Item |')
        lines.append('|--------|------|------|')
        for r, finding in all_dead:
            fpath = r.get('file_pattern', '?')
            desc = r['description'][:50].replace('|', '\\|')
            finding_clean = finding.replace('|', '\\|')
            lines.append(f'| {finding_clean} | {fpath} | {desc} |')
        lines.append('')

    # Test coverage section
    test_items = [r for r in results if r['category'] == 'test']
    if test_items:
        lines.append('## Test Coverage')
        lines.append('')
        lines.append('| Plan test item | Evidence | Details |')
        lines.append('|----------------|----------|---------|')
        for r in test_items:
            desc = r['description'][:60].replace('|', '\\|')
            evidence = '; '.join(r['evidence'][:1]).replace('|', '\\|')
            lines.append(f'| {desc} | {r["evidence_level"]} | {evidence} |')
        lines.append('')

    return '\n'.join(lines)


def main():
    """Cross-reference plan items (from stdin JSON) against git evidence."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Cross-reference plan items against git diff.'
    )
    parser.add_argument('--base', default='main',
                        help='Git ref to diff against (default: main)')
    parser.add_argument('--repo', default='.',
                        help='Repository root (default: current directory)')

    args = parser.parse_args()
    repo = Path(args.repo).resolve()

    # Read plan items from stdin
    plan_items = json.load(sys.stdin)

    # Gather evidence
    from gather_evidence import gather_evidence

    plan_files = list(set(
        item.get('file_pattern') for item in plan_items
        if item.get('file_pattern')
    ))
    evidence = gather_evidence(args.base, repo, plan_files)

    if evidence['errors']:
        for err in evidence['errors']:
            print(f'Error: {err}', file=sys.stderr)
        sys.exit(1)

    # Cross-reference
    results = cross_reference(plan_items, evidence)

    # Get plan title
    plan_title = plan_items[0].get('plan_title', 'Unknown Plan') if plan_items else 'Unknown Plan'

    # Generate report
    report = generate_report(results, plan_title)
    print(report)


if __name__ == '__main__':
    main()
