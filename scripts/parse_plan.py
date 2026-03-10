#!/usr/bin/env python3
"""
Parse a Claude Code /plan markdown file into discrete, verifiable checklist items.

Recognizes the structure CC's /plan produces:
- ## Change N: Title
- **File: path/to/file.ts**
- ### Sub-headings for sub-items
- Code blocks with type/function/field declarations
- Tables (file mappings, behavior matrices)
- ## Tests section

Output: JSON array of plan items to stdout.
"""

import json
import re
import sys
from pathlib import Path
from typing import Optional


_NON_ACTIONABLE_H2 = {
    'context', 'summary', 'problem', 'problem statement', 'current state',
    'technical notes', 'existing code to reuse',
    'what does not change', "what doesn't change", 'thinking blocks',
    'privacy', 'data format comparison', 'default source strategy',
    'graceful degradation', 'requirements',
}

# File-listing H2 headings — skipped when explicit change sections exist,
# but treated as actionable in plans that lack them.
_FILES_H2 = re.compile(
    r'^##\s+Files?\s+(?:to\s+)?(?:Modify|Create|Modified|Deleted|Create/Modify)\b',
    re.IGNORECASE,
)


def _has_explicit_changes(lines: list[str]) -> bool:
    """Check if the plan uses ## Change N: or ## Part N: headings."""
    for line in lines:
        if re.match(r'^##\s+(?:Change|Part)\s+\d+', line):
            return True
    return False


def parse_plan(plan_text: str) -> list[dict]:
    """Parse plan markdown into checklist items."""
    items = []
    lines = plan_text.split('\n')

    # Extract plan title from first H1
    plan_title = ''
    for line in lines:
        if line.startswith('# ') and not line.startswith('## '):
            plan_title = line.lstrip('# ').strip()
            break

    has_explicit = _has_explicit_changes(lines)
    change_counter = 0

    current_change_id = None
    current_change_title = ''
    current_file = None
    current_sub_id = None
    current_sub_title = ''
    in_code_block = False
    code_block_lines = []
    in_tests_section = False
    item_counter = 0

    i = 0
    code_fence_lang = None
    while i < len(lines):
        line = lines[i]

        # Track code blocks
        if line.strip().startswith('```'):
            if in_code_block:
                # End of code block — extract patterns from accumulated lines
                patterns = extract_patterns_from_code(
                    '\n'.join(code_block_lines),
                    file_path=current_file,
                    fence=code_fence_lang,
                )
                if patterns and current_change_id:
                    # Attach patterns to most recent item or create new one
                    if items and items[-1]['change_id'] == current_change_id:
                        items[-1]['expected_patterns'].extend(patterns)
                    else:
                        item_counter += 1
                        items.append({
                            'id': item_counter,
                            'change_id': current_change_id,
                            'change_title': current_change_title,
                            'file_pattern': current_file,
                            'description': f'Code block patterns in {current_sub_title or current_change_title}',
                            'expected_patterns': patterns,
                            'category': categorize_patterns(patterns),
                        })
                in_code_block = False
                code_block_lines = []
                code_fence_lang = None
            else:
                in_code_block = True
                # Extract language from fence: ```typescript, ```python, etc.
                fence_text = line.strip()[3:].strip()
                code_fence_lang = fence_text if fence_text else None
            i += 1
            continue

        if in_code_block:
            code_block_lines.append(line)
            i += 1
            continue

        # --- H2 heading detection ---
        h2_match = re.match(r'^##\s+(.+)', line)
        if h2_match:
            h2_text = h2_match.group(1).strip()
            h2_lower = h2_text.lower()

            # Detect Tests section
            if re.match(r'tests?\b', h2_lower):
                in_tests_section = True
                current_change_id = current_change_id or 'Tests'
                i += 1
                continue

            # Detect Verification section (skip — procedural, not implementable)
            if h2_lower.startswith('verification'):
                in_tests_section = False
                current_change_id = None
                i += 1
                continue

            # Non-actionable headings — skip
            if h2_lower in _NON_ACTIONABLE_H2:
                in_tests_section = False
                current_change_id = None
                i += 1
                continue

            # Files-listing headings — skip only if plan has explicit changes
            if _FILES_H2.match(line):
                if has_explicit:
                    i += 1
                    continue
                # Otherwise fall through and treat as actionable

            in_tests_section = False

            # Detect ## Change N: Title or ## Part N: Title
            numbered_match = re.match(r'(Change|Part)\s+(\d+):\s*(.+)', h2_text)
            if numbered_match:
                current_change_id = f'{numbered_match.group(1)} {numbered_match.group(2)}'
                current_change_title = numbered_match.group(3).strip()
                current_file = None
                current_sub_id = None
                current_sub_title = ''
                i += 1
                continue

            # Detect file-scoped heading: ## Changes to `file.ts`
            file_heading = re.search(r'`([^`]+\.\w+)`', h2_text)
            if file_heading:
                current_file = file_heading.group(1)

            # Generic actionable H2 — assign a change_id
            change_counter += 1
            current_change_id = f'Section {change_counter}'
            current_change_title = h2_text
            current_sub_id = None
            current_sub_title = ''
            i += 1
            continue

        # Detect file targets: **File: path**, **File: `path`**, **File:** `path`
        file_match = re.match(r'^\*\*File:\*?\*?\s*`?([^`*]+)`?', line)
        if file_match:
            current_file = file_match.group(1).strip()
            i += 1
            continue

        # Detect sub-headings: ### 2a. Title or ### Title
        sub_match = re.match(r'^###\s+(\S+\.?\s*)?(.+)', line)
        if sub_match and current_change_id:
            sub_label = (sub_match.group(1) or '').strip().rstrip('.')
            current_sub_title = sub_match.group(2).strip()

            # Extract file path from sub-heading if present
            # e.g., "### 5a. Pattern analysis — `src/core/analyzer.ts`"
            file_in_heading = re.search(r'`([^`]+\.\w+)`', current_sub_title)
            if file_in_heading:
                current_file = file_in_heading.group(1)
            current_sub_id = sub_label if sub_label else current_sub_title[:30]

            # The sub-heading itself is an item
            item_counter += 1
            items.append({
                'id': item_counter,
                'change_id': current_change_id,
                'change_title': current_change_title,
                'sub_id': current_sub_id,
                'file_pattern': current_file,
                'description': current_sub_title,
                'expected_patterns': [],
                'category': 'function' if any(kw in current_sub_title.lower() for kw in ['add method', 'add private', 'add function', 'integrate', 'refactor']) else 'wiring',
            })
            i += 1
            continue

        # Detect list items with substance (not just structural)
        list_match = re.match(r'^(\s*)[-*]\s+(.+)', line)
        if not list_match:
            list_match = re.match(r'^(\s*)\d+\.\s+(.+)', line)

        if list_match and current_change_id:
            indent = len(list_match.group(1))
            content = list_match.group(2).strip()

            # Skip short/structural items
            if len(content) < 15 and not re.search(r'`\w+`', content):
                i += 1
                continue

            # Extract inline code patterns
            inline_patterns = re.findall(r'`([^`]+)`', content)
            # Filter to meaningful patterns (identifiers, not plain English)
            inline_patterns = [p for p in inline_patterns if re.search(r'[A-Za-z_]\w*', p) and len(p) > 2]

            if in_tests_section:
                item_counter += 1
                items.append({
                    'id': item_counter,
                    'change_id': 'Tests',
                    'change_title': 'Tests',
                    'file_pattern': extract_test_file_hint(content),
                    'description': content,
                    'expected_patterns': inline_patterns,
                    'category': 'test',
                })
            elif inline_patterns or len(content) > 30:
                item_counter += 1
                items.append({
                    'id': item_counter,
                    'change_id': current_change_id,
                    'change_title': current_change_title,
                    'sub_id': current_sub_id,
                    'file_pattern': current_file,
                    'description': clean_description(content),
                    'expected_patterns': inline_patterns,
                    'category': categorize_from_description(content),
                })

            i += 1
            continue

        # Detect substantive paragraphs with inline code patterns
        # (catches Change blocks that use prose instead of lists/sub-headings)
        if (current_change_id and not in_tests_section
                and not line.startswith('#') and not line.startswith('|')
                and not line.startswith('-') and not line.startswith('*')
                and not re.match(r'^\d+\.', line.strip())
                and not line.startswith('**File:')):
            # Check for file path references in prose (updates current_file context)
            file_ref = re.search(r'`(src/[^`]+\.\w+)`', line)
            if file_ref:
                # Use as this item's file but also update context for following items
                item_file = file_ref.group(1)
                current_file = item_file
            else:
                # Check for "in path/file.ts" pattern without backticks
                file_ref_plain = re.search(r'\bin\s+(src/\S+\.\w+)', line)
                item_file = file_ref_plain.group(1) if file_ref_plain else current_file

            inline_codes = re.findall(r'`([^`]+)`', line)
            meaningful = [p for p in inline_codes if re.search(r'[A-Za-z_]\w*', p) and len(p) > 2]
            if len(meaningful) >= 2 and len(line.strip()) > 40:
                item_counter += 1
                items.append({
                    'id': item_counter,
                    'change_id': current_change_id,
                    'change_title': current_change_title,
                    'sub_id': current_sub_id,
                    'file_pattern': item_file,
                    'description': clean_description(line.strip()),
                    'expected_patterns': meaningful,
                    'category': categorize_from_description(line),
                })
                i += 1
                continue

        # Detect table rows with plan-item content (behavior matrices, file mappings)
        table_match = re.match(r'^\|(.+)\|$', line)
        if table_match and current_change_id:
            cells = [c.strip() for c in table_match.group(1).split('|')]
            # Skip header/separator rows
            if not all(re.match(r'^[-:]+$', c) for c in cells) and len(cells) >= 2:
                # Check if this looks like a behavior/filter table
                has_exclude = any('exclude' in c.lower() or '**exclude**' in c.lower() for c in cells)
                has_include = any('include' in c.lower() or '**include**' in c.lower() for c in cells)
                if has_exclude or has_include:
                    # This is a behavior specification row
                    consumer = cells[0].strip('* ')
                    if consumer and not consumer.startswith('---'):
                        item_counter += 1
                        # Extract code-like patterns from consumer name
                        # Skip English phrases like "Exemplar selection"
                        consumer_name = consumer.split('(')[0].strip() if '(' in consumer else ''
                        consumer_patterns = []
                        if consumer_name and re.match(r'^[A-Za-z_]\w*$', consumer_name):
                            consumer_patterns = [consumer_name]

                        items.append({
                            'id': item_counter,
                            'change_id': current_change_id,
                            'change_title': current_change_title,
                            'file_pattern': current_file,
                            'description': f'Behavior for {consumer}: {" | ".join(cells[1:])}',
                            'expected_patterns': consumer_patterns,
                            'category': 'filter_logic',
                        })

        i += 1

    # Post-process: attach plan title and clean up patterns
    for item in items:
        item['plan_title'] = plan_title
        # Clean all patterns: extract identifiers from signatures, XML tags, etc.
        if item.get('expected_patterns'):
            cleaned = []
            for p in item['expected_patterns']:
                cleaned.append(_clean_pattern(p))
            # Deduplicate after cleaning (signatures may reduce to same name)
            seen = set()
            unique = []
            for p in cleaned:
                if p not in seen and len(p) > 2:
                    seen.add(p)
                    unique.append(p)
            item['expected_patterns'] = unique

    return items


def extract_patterns_from_code(code: str, file_path: Optional[str] = None,
                               fence: Optional[str] = None) -> list[str]:
    """Extract verifiable patterns from a code block.

    Uses language-specific regexes when the language can be detected from
    the file path or code fence. Falls back to a generic extractor.
    """
    from languages import detect_language, extract_patterns
    lang_spec = detect_language(file_path=file_path, fence=fence)
    return extract_patterns(code, lang_spec)


def categorize_patterns(patterns: list[str]) -> str:
    """Guess category from extracted patterns."""
    text = ' '.join(patterns).lower()
    if any(kw in text for kw in ['type', 'interface', 'enum']):
        return 'type_definition'
    if any(kw in text for kw in ['test', 'spec', 'describe', 'expect']):
        return 'test'
    return 'wiring'


def categorize_from_description(desc: str) -> str:
    """Categorize a plan item from its description text."""
    lower = desc.lower()
    if any(kw in lower for kw in ['add type', 'add to interface', 'type alias', 'add field']):
        return 'type_definition'
    if any(kw in lower for kw in ['add method', 'add function', 'add private', 'refactor']):
        return 'function'
    if any(kw in lower for kw in ['filter', 'exclude', 'skip', 'include']):
        return 'filter_logic'
    if any(kw in lower for kw in ['test', 'verify', 'assert']):
        return 'test'
    if any(kw in lower for kw in ['wire', 'integrate', 'call', 'pass', 'render']):
        return 'wiring'
    if any(kw in lower for kw in ['field', 'property', 'optional']):
        return 'field'
    return 'wiring'


def extract_test_file_hint(desc: str) -> Optional[str]:
    """Try to extract a test file path hint from a test description."""
    # Look for paths in backticks
    paths = re.findall(r'`([^`]*(?:test|spec)[^`]*)`', desc, re.IGNORECASE)
    if paths:
        return paths[0]
    # Look for file-like references
    paths = re.findall(r'(\S+\.(?:test|spec)\.\w+)', desc)
    if paths:
        return paths[0]
    return None


def _clean_pattern(pattern: str) -> str:
    """Clean a pattern extracted from inline backticks.

    Handles full method signatures by extracting just the name:
      "extractFoo(arg: Type): ReturnType" → "extractFoo"
      "foo()" → "foo"
      "someField" → "someField" (unchanged)
      "turnOrigin?: TurnOrigin" → "turnOrigin"
      "<local-command-caveat>...</local-command-caveat>" → "local-command-caveat"
      "<command-name>/code-review</command-name>" → "command-name"
    """
    # XML tags — extract the tag name
    xml_match = re.match(r'^<([\w-]+)[>/]', pattern)
    if xml_match:
        return xml_match.group(1)
    # If it contains '(' — extract the function/method name before it
    paren_match = re.match(r'^(\w+)\s*\(', pattern)
    if paren_match:
        return paren_match.group(1)
    # If it's a field declaration "name?: Type" or "name: Type" — extract the name
    field_match = re.match(r'^(\w+)\??\s*:', pattern)
    if field_match:
        return field_match.group(1)
    return pattern


def clean_description(text: str) -> str:
    """Clean up a description string."""
    # Remove markdown bold/italic
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    # Remove backtick formatting but keep content
    text = re.sub(r'`([^`]+)`', r'\1', text)
    return text.strip()


def main():
    if len(sys.argv) < 2:
        print("Usage: parse_plan.py <plan-file>", file=sys.stderr)
        print("", file=sys.stderr)
        print("Parses a Claude Code /plan markdown file into verifiable checklist items.", file=sys.stderr)
        print("Output: JSON array to stdout.", file=sys.stderr)
        sys.exit(1)

    plan_path = Path(sys.argv[1])
    if not plan_path.exists():
        print(f"Error: Plan file {plan_path} does not exist", file=sys.stderr)
        sys.exit(1)

    plan_text = plan_path.read_text(encoding='utf-8')
    items = parse_plan(plan_text)

    json.dump(items, sys.stdout, indent=2)
    print()  # trailing newline


if __name__ == '__main__':
    main()
