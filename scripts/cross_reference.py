#!/usr/bin/env python3
"""
Cross-reference plan items against git evidence.

For each plan item, determines:
- ✅ Implemented: target file modified AND expected patterns found in diff
- ❌ Not implemented: target file not modified OR patterns missing
- ⚠️ Dead code: patterns present but never used (declared but not assigned/called/read)
- ⚡ Partial: some patterns found, others missing

Also performs grep-level dead-code detection across the full codebase.
"""

import json
import re
import sys
from pathlib import Path
from typing import Optional

from languages import detect_language


# Status constants
IMPLEMENTED = '✅'
NOT_IMPLEMENTED = '❌'
DEAD_CODE = '⚠️'
PARTIAL = '⚡'
SKIPPED = '⏭️'


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

    # Try exact match first
    if pattern in added_text:
        # Find the line containing it for evidence
        for line in added_lines:
            if pattern in line:
                snippet = line.strip()
                if len(snippet) > 100:
                    snippet = snippet[:100] + '...'
                return True, f'+{snippet}'
        return True, None

    # Try case-insensitive for type names that might differ in casing
    if re.search(re.escape(pattern), added_text, re.IGNORECASE):
        for line in added_lines:
            if re.search(re.escape(pattern), line, re.IGNORECASE):
                snippet = line.strip()
                if len(snippet) > 100:
                    snippet = snippet[:100] + '...'
                return True, f'+{snippet} (case-insensitive match)'
        return True, None

    return False, None


def _looks_like_string_literal(pattern: str) -> bool:
    """Check if a pattern looks like a string/enum value rather than an identifier.

    String literals like 'skill_invocation', 'user_authored', 'meta_command'
    should not be checked for function calls or field assignments — they're
    values, not declarations.
    """
    # snake_case with underscores = almost certainly a string/enum value
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
    Pass pre-built text blobs from _build_dead_code_texts() to avoid
    re-concatenating on every call.
    """
    if not declaring_file:
        return False, None

    # String/enum values are never "dead code" in the function/field sense
    if _looks_like_string_literal(pattern):
        return False, None

    if category == 'type_definition':
        # A type should be referenced (imported or used) in other files
        if pattern in other_files_text:
            return False, None
        # Check if it's used in the same file (e.g., in a field declaration)
        uses_in_file = len(re.findall(re.escape(pattern), declaring_text))
        if uses_in_file <= 1:  # 1 = just the declaration itself
            return True, f'{pattern} declared but not referenced in any other file'
        return False, None

    elif category == 'function':
        # Use language-aware call pattern if possible
        lang_spec = detect_language(file_path=declaring_file)
        call_tmpl = lang_spec.get('call_pattern', r'{name}\s*\(')
        call_regex = call_tmpl.replace('{name}', re.escape(pattern))

        all_calls = re.findall(call_regex, all_text)
        if all_calls:
            # Pattern is used as a function — check if it's called more than declared
            if len(all_calls) <= 1:
                return True, f'{pattern}() declared but never called'
            return False, None
        else:
            # Pattern never appears as a call — it's probably not a function name.
            # Fall through to a simple reference check instead of reporting dead code.
            if pattern in all_text:
                return False, None
            return True, f'{pattern} declared but not referenced'

    elif category == 'field':
        # A field should be both assigned and read
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
            # Being assigned but not read in OTHER files might be okay
            # (it could be used in the same file)
            if not re.search(read_pattern, declaring_text):
                return True, f'{pattern} assigned but never read by any consumer'

        return False, None

    return False, None


def cross_reference(plan_items: list[dict], evidence: dict) -> list[dict]:
    """Cross-reference plan items against evidence, producing audit results."""
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
            'status': NOT_IMPLEMENTED,
            'evidence': [],
            'dead_code_findings': [],
        }

        file_pattern = item.get('file_pattern')
        expected_patterns = item.get('expected_patterns', [])

        # If no file target and no patterns, we can't verify mechanically
        if not file_pattern and not expected_patterns:
            result['status'] = SKIPPED
            result['evidence'].append('No file target or patterns to verify')
            results.append(result)
            continue

        # For items with no file target but with patterns (common for test
        # descriptions), search across all relevant files in the diff rather
        # than giving up. For test items, search test files first.
        if not file_pattern and expected_patterns:
            is_test = item.get('category') == 'test'
            search_files = {}
            for f, diff_text in file_diffs.items():
                if is_test:
                    if 'test' in f.lower() or 'spec' in f.lower():
                        search_files[f] = diff_text
                else:
                    search_files[f] = diff_text
            # Fall back to all files if no test files found
            if not search_files:
                search_files = file_diffs

            found_count = 0
            for pattern in expected_patterns:
                for f, diff_text in search_files.items():
                    found, snippet = check_pattern_in_diff(pattern, diff_text, _cache=added_cache.get(f))
                    if found:
                        found_count += 1
                        result['evidence'].append(f'"{pattern}" found in {f}: {snippet}')
                        break
                else:
                    result['evidence'].append(f'"{pattern}" not found in any {"test " if is_test else ""}file')

            total = len(expected_patterns)
            if found_count == total:
                result['status'] = IMPLEMENTED
            elif found_count > 0:
                result['status'] = PARTIAL
            else:
                result['status'] = NOT_IMPLEMENTED

            results.append(result)
            continue

        # Find the actual file in the diff
        actual_file = find_file_in_evidence(file_pattern, modified_files, file_diffs)

        if not actual_file and file_pattern:
            # File wasn't modified at all
            result['status'] = NOT_IMPLEMENTED
            result['evidence'].append(f'File matching "{file_pattern}" not found in diff')

            # But check if patterns appear anywhere in the diff (wrong file?)
            if expected_patterns:
                for pattern in expected_patterns:
                    for f, diff_text in file_diffs.items():
                        found, snippet = check_pattern_in_diff(pattern, diff_text, _cache=added_cache.get(f))
                        if found:
                            result['evidence'].append(
                                f'Pattern "{pattern}" found in {f} instead (wrong file?): {snippet}'
                            )
                            result['status'] = PARTIAL
                            break

            results.append(result)
            continue

        # File was modified — check patterns
        if not expected_patterns:
            # No specific patterns but file was touched
            result['status'] = IMPLEMENTED
            result['evidence'].append(f'{actual_file} was modified')
            results.append(result)
            continue

        diff_text = file_diffs.get(actual_file, '')
        diff_cache = added_cache.get(actual_file)
        found_count = 0
        total_count = len(expected_patterns)

        # Pre-build dead-code texts for this declaring file (once per file)
        if actual_file and actual_file not in dead_code_cache:
            dead_code_cache[actual_file] = _build_dead_code_texts(current_files, actual_file)

        for pattern in expected_patterns:
            found, snippet = check_pattern_in_diff(pattern, diff_text, _cache=diff_cache)
            if found:
                found_count += 1
                evidence_str = f'"{pattern}" found'
                if snippet:
                    evidence_str += f': {snippet}'
                result['evidence'].append(evidence_str)

                # Dead-code check for found patterns
                dc_declaring, dc_other, dc_all = dead_code_cache.get(actual_file, ('', '', ''))
                is_dead, dead_reason = check_dead_code(
                    pattern, item.get('category', 'wiring'),
                    actual_file, dc_declaring, dc_other, dc_all
                )
                if is_dead:
                    result['dead_code_findings'].append(dead_reason)
            else:
                result['evidence'].append(f'"{pattern}" NOT found in {actual_file} diff')

                # Check if it's in the full diff (different file)
                for f, f_diff in file_diffs.items():
                    if f != actual_file:
                        alt_found, alt_snippet = check_pattern_in_diff(pattern, f_diff, _cache=added_cache.get(f))
                        if alt_found:
                            result['evidence'].append(
                                f'  → but found in {f}: {alt_snippet}'
                            )
                            found_count += 0.5  # partial credit
                            break

        # Determine status
        if result['dead_code_findings']:
            result['status'] = DEAD_CODE
        elif found_count == total_count:
            result['status'] = IMPLEMENTED
        elif found_count > 0:
            result['status'] = PARTIAL
        else:
            result['status'] = NOT_IMPLEMENTED

        results.append(result)

    return results


def generate_report(results: list[dict], plan_title: str,
                    report_context: Optional[dict] = None) -> str:
    """Generate the PLAN_REVIEW.md report.

    Args:
        results: Cross-reference results.
        plan_title: Title extracted from the plan.
        report_context: Optional dict with keys: plan_path, scope, base,
            branch, repo, head_sha, head_subject, uncommitted_count,
            plan_mtime_str.
    """
    lines = []
    ctx = report_context or {}

    # Title
    audit_date = ctx.get('audit_date', '')
    plan_file = Path(ctx.get('plan_path', '')).name if ctx.get('plan_path') else ''
    date_part = f' {audit_date[:10]}' if audit_date else ''
    file_part = f' ({plan_file})' if plan_file else ''
    # Strip "Plan: " prefix from title if present
    clean_title = plan_title.removeprefix('Plan: ').removeprefix('plan: ')
    lines.append(f'# Plan Implementation Report{date_part} for {clean_title}{file_part}')
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

    # Summary counts
    total = len(results)
    implemented = sum(1 for r in results if r['status'] == IMPLEMENTED)
    not_impl = sum(1 for r in results if r['status'] == NOT_IMPLEMENTED)
    dead = sum(1 for r in results if r['status'] == DEAD_CODE)
    partial = sum(1 for r in results if r['status'] == PARTIAL)
    skipped = sum(1 for r in results if r['status'] == SKIPPED)

    lines.append('## Summary')
    lines.append('')
    lines.append(f'- **{total}** plan items checked')
    lines.append(f'- **{implemented}** implemented {IMPLEMENTED}')
    lines.append(f'- **{not_impl}** not implemented {NOT_IMPLEMENTED}')
    lines.append(f'- **{dead}** dead code {DEAD_CODE}')
    lines.append(f'- **{partial}** partial {PARTIAL}')
    if skipped:
        lines.append(f'- **{skipped}** skipped (not mechanically verifiable) {SKIPPED}')
    lines.append('')

    if not_impl == 0 and dead == 0:
        lines.append('All verifiable plan items are implemented and live.')
    elif not_impl > 0 or dead > 0:
        lines.append('**Issues detected** — see details below.')
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

    # Detailed checklist per change
    lines.append('## Detailed Checklist')
    lines.append('')

    for cid, change in changes.items():
        lines.append(f'### {cid}: {change["title"]}')
        lines.append('')
        lines.append('| # | Sub | Item | Status | Evidence |')
        lines.append('|---|-----|------|--------|----------|')

        for r in change['items']:
            status = r['status']
            sub_id = r.get('sub_id', '') or ''
            desc = r['description'][:60]
            evidence = '; '.join(r['evidence'][:2])  # first 2 pieces
            if len(evidence) > 80:
                evidence = evidence[:80] + '...'
            # Escape pipe characters in table cells
            desc = desc.replace('|', '\\|')
            evidence = evidence.replace('|', '\\|')
            sub_id = sub_id.replace('|', '\\|')
            lines.append(f'| {r["id"]} | {sub_id} | {desc} | {status} | {evidence} |')

        lines.append('')

    # Dead code findings section
    all_dead = [(r, finding) for r in results for finding in r['dead_code_findings']]
    if all_dead:
        lines.append('## Dead Code Findings')
        lines.append('')
        lines.append('| Finding | File | Item |')
        lines.append('|---------|------|------|')
        for r, finding in all_dead:
            fpath = r.get('file_pattern', '?')
            desc = r['description'][:50].replace('|', '\\|')
            finding_clean = finding.replace('|', '\\|')
            lines.append(f'| {DEAD_CODE} {finding_clean} | {fpath} | {desc} |')
        lines.append('')

    # Test coverage section
    test_items = [r for r in results if r['category'] == 'test']
    if test_items:
        lines.append('## Test Coverage')
        lines.append('')
        lines.append('| Plan test item | Status | Evidence |')
        lines.append('|----------------|--------|----------|')
        for r in test_items:
            desc = r['description'][:60].replace('|', '\\|')
            evidence = '; '.join(r['evidence'][:1]).replace('|', '\\|')
            lines.append(f'| {desc} | {r["status"]} | {evidence} |')
        lines.append('')

    # Narrative placeholder
    lines.append('## Narrative Assessment')
    lines.append('')
    if not_impl > 0 or dead > 0:
        lines.append('<!-- Claude: Fill in this section with analysis of:')
        lines.append('     - WHY items were missed (architectural mismatch? plan assumption wrong?)')
        lines.append('     - Whether dead code indicates a deeper integration gap')
        lines.append('     - Recommended fix approach -->')
        lines.append('')

        # Auto-generate some narrative from the data
        if not_impl > 0:
            missing = [r for r in results if r['status'] == NOT_IMPLEMENTED]
            lines.append(f'{not_impl} plan item(s) were not implemented:')
            lines.append('')
            for r in missing:
                sub = f' ({r["sub_id"]})' if r.get('sub_id') else ''
                lines.append(f'- **{r["change_id"]}{sub}**: {r["description"]}')
                for ev in r['evidence']:
                    lines.append(f'  - {ev}')
            lines.append('')

        if dead > 0:
            dead_items = [r for r in results if r['status'] == DEAD_CODE]
            lines.append(f'{dead} item(s) have dead code — the code exists but is not actually used:')
            lines.append('')
            for r in dead_items:
                sub = f' ({r["sub_id"]})' if r.get('sub_id') else ''
                lines.append(f'- **{r["change_id"]}{sub}**: {r["description"]}')
                for finding in r['dead_code_findings']:
                    lines.append(f'  - {finding}')
            lines.append('')
    else:
        lines.append('All verifiable plan items were implemented and are live in the codebase.')
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
