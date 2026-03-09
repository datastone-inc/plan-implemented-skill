#!/usr/bin/env python3
"""
Plan Implementation Audit — main orchestrator.

Parses a Claude Code /plan file, gathers git diff evidence, cross-references
each plan item against the diff, and generates PLAN_REVIEW.md.

Usage:
    review.py <plan-file> [--base main] [--repo .]
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Allow importing sibling modules
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from parse_plan import parse_plan
from gather_evidence import gather_evidence, run_git
from cross_reference import cross_reference, generate_report, NOT_IMPLEMENTED, DEAD_CODE


def find_plans_directory(repo: Path) -> Path | None:
    """Discover the plans directory using CC's settings hierarchy.

    Search order:
    1. Project .claude/settings.local.json → plansDirectory
    2. Project .claude/settings.json → plansDirectory
    3. User ~/.claude/settings.json → plansDirectory
    4. Project .claude/plans/ (if exists)
    5. ~/.claude/plans/ (global default)
    """
    # Check project settings first, then user settings
    settings_files = [
        repo / '.claude' / 'settings.local.json',
        repo / '.claude' / 'settings.json',
        Path.home() / '.claude' / 'settings.json',
    ]

    for sf in settings_files:
        if sf.exists():
            try:
                data = json.loads(sf.read_text(encoding='utf-8'))
                plans_dir = data.get('plansDirectory')
                if plans_dir:
                    p = Path(plans_dir)
                    # Resolve relative paths from repo root
                    if not p.is_absolute():
                        p = repo / p
                    p = p.expanduser().resolve()
                    if p.exists():
                        return p
            except (json.JSONDecodeError, OSError):
                pass

    # Fallback: check common locations
    candidates = [
        repo / '.claude' / 'plans',
        Path.home() / '.claude' / 'plans',
    ]
    for c in candidates:
        if c.exists() and any(c.glob('*.md')):
            return c

    return None


def list_plans(plans_dir: Path) -> None:
    """Print a numbered list of plans with timestamps and titles."""
    md_files = sorted(plans_dir.glob('*.md'), key=lambda f: f.stat().st_mtime,
                      reverse=True)
    if not md_files:
        print(f'No .md files in {plans_dir}')
        return

    print(f'Plans in {plans_dir}:\n')
    for idx, f in enumerate(md_files, 1):
        mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
        title = f.stem
        try:
            for line in f.read_text(encoding='utf-8').split('\n'):
                if line.startswith('# ') and not line.startswith('## '):
                    title = line.lstrip('# ').strip()
                    break
        except OSError:
            pass
        print(f'  {idx}. [{mtime}]  {f.name}')
        print(f'               {title}')


def main():
    parser = argparse.ArgumentParser(
        description='Audit whether a Claude Code /plan was fully implemented.'
    )
    parser.add_argument('plan_file', nargs='?', default=None,
                        help='Path to plan markdown file. If omitted, discovers '
                             'the most recent plan from CC settings or default locations.')
    parser.add_argument('--base', default='main',
                        help='Git ref to diff against (default: main)')
    parser.add_argument('--scope', default='plan',
                        choices=['branch', 'plan', 'uncommitted', 'all'],
                        help='What changes to review: '
                             'branch=committed vs base branch, '
                             'plan=changes since plan was created/updated (default), '
                             'uncommitted=only staged+unstaged vs HEAD, '
                             'all=committed+uncommitted vs base branch')
    parser.add_argument('--repo', default='.',
                        help='Repository root (default: current directory)')
    parser.add_argument('--output', default=None,
                        help='Output file path (default: PLAN_REVIEW.md in repo root)')
    parser.add_argument('--json', action='store_true',
                        help='Output raw JSON results instead of markdown')
    parser.add_argument('--list', action='store_true',
                        help='List available plans and exit')

    args = parser.parse_args()

    repo = Path(args.repo).resolve()

    # Handle --list independently (works whether or not a plan file is given)
    if args.list:
        plans_dir = find_plans_directory(repo)
        if not plans_dir:
            print('No plans directory found.', file=sys.stderr)
            sys.exit(1)
        list_plans(plans_dir)
        sys.exit(0)

    # Discover or validate plan file
    if args.plan_file:
        plan_path = Path(args.plan_file)
    else:
        plans_dir = find_plans_directory(repo)
        if not plans_dir:
            print('Error: No plans directory found.', file=sys.stderr)
            print('  Checked: .claude/settings.json, ~/.claude/settings.json,',
                  file=sys.stderr)
            print('           .claude/plans/, ~/.claude/plans/', file=sys.stderr)
            print('  Specify a plan file explicitly or configure plansDirectory.',
                  file=sys.stderr)
            sys.exit(1)

        md_files = sorted(plans_dir.glob('*.md'), key=lambda f: f.stat().st_mtime,
                          reverse=True)
        if not md_files:
            print(f'Error: No .md files found in {plans_dir}', file=sys.stderr)
            sys.exit(1)

        if len(md_files) == 1:
            plan_path = md_files[0]
            print(f'Auto-discovered plan: {plan_path}')
        else:
            # Multiple plans — show them so the caller can confirm
            print(f'Found {len(md_files)} plans in {plans_dir}:\n')
            list_plans(plans_dir)
            print(f'\nUsing most recent: {md_files[0].name}')
            plan_path = md_files[0]

    # Validate inputs
    if not plan_path.exists():
        print(f'Error: Plan file {plan_path} does not exist', file=sys.stderr)
        sys.exit(1)

    if not (repo / '.git').exists():
        print(f'Error: {repo} is not a git repository', file=sys.stderr)
        sys.exit(1)

    # Phase 1: Parse plan
    print(f'Parsing plan: {plan_path}')
    plan_text = plan_path.read_text(encoding='utf-8')
    plan_items = parse_plan(plan_text)
    print(f'  Found {len(plan_items)} verifiable items')

    if not plan_items:
        print('Error: No verifiable items found in plan', file=sys.stderr)
        print('  Is this a Claude Code /plan file with "## Change N:" headings?',
              file=sys.stderr)
        sys.exit(1)

    # Phase 2: Gather evidence
    plan_mtime = plan_path.stat().st_mtime
    scope_desc = {
        'branch': f'git diff {args.base}..HEAD (committed only)',
        'plan': f'changes since plan was modified ({datetime.fromtimestamp(plan_mtime).strftime("%Y-%m-%d %H:%M")})',
        'uncommitted': 'uncommitted changes only (staged + unstaged)',
        'all': f'all changes vs {args.base} (committed + uncommitted)',
    }
    print(f'Gathering evidence: {scope_desc[args.scope]}')
    plan_files = list(set(
        item.get('file_pattern') for item in plan_items
        if item.get('file_pattern')
    ))
    evidence = gather_evidence(
        args.base, repo, plan_files,
        scope=args.scope, plan_mtime=plan_mtime,
    )

    if evidence['errors']:
        for err in evidence['errors']:
            print(f'  Error: {err}', file=sys.stderr)
        sys.exit(1)

    file_count = len(evidence.get('file_diffs', {}))
    print(f'  {file_count} files in diff, {len(evidence.get("current_files", {}))} files read')

    # Phase 3: Cross-reference
    print('Cross-referencing plan items against diff...')
    results = cross_reference(plan_items, evidence)

    # Tally
    statuses = {}
    for r in results:
        s = r['status']
        statuses[s] = statuses.get(s, 0) + 1

    print(f'  Results: {statuses}')

    # Gather context for report
    # Current branch
    _, branch_name, _ = run_git(['rev-parse', '--abbrev-ref', 'HEAD'], repo)
    branch_name = branch_name.strip()

    # HEAD commit
    _, head_log, _ = run_git(['log', '-1', '--format=%H%n%s'], repo)
    head_lines = head_log.strip().split('\n', 1)
    head_sha = head_lines[0]
    head_subject = head_lines[1] if len(head_lines) > 1 else ''

    # Uncommitted file count
    _, dirty_files, _ = run_git(['status', '--porcelain'], repo)
    uncommitted_count = len([l for l in dirty_files.strip().split('\n') if l.strip()]) if dirty_files.strip() else 0

    report_context = {
        'plan_path': str(plan_path),
        'plan_mtime_str': datetime.fromtimestamp(plan_mtime).strftime('%Y-%m-%d %H:%M:%S'),
        'repo': str(repo),
        'branch': branch_name,
        'scope': args.scope,
        'base': args.base,
        'head_sha': head_sha,
        'head_subject': head_subject,
        'uncommitted_count': uncommitted_count,
        'audit_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    # Output
    if args.json:
        json.dump(results, sys.stdout, indent=2)
        print()
    else:
        plan_title = plan_items[0].get('plan_title', 'Unknown Plan') if plan_items else 'Unknown Plan'
        report = generate_report(results, plan_title, report_context=report_context)

        # Add metadata header
        header = f'<!-- Generated by plan-implemented on {report_context["audit_date"]} -->\n'
        header += f'<!-- Plan: {plan_path} -->\n'
        header += f'<!-- Scope: {args.scope} | Base: {args.base} -->\n\n'
        report = header + report

        # Write to file
        output_path = Path(args.output) if args.output else repo / 'PLAN_REVIEW.md'
        output_path.write_text(report, encoding='utf-8')
        print(f'\nReport saved to: {output_path}')

        # Print summary
        has_issues = any(r['status'] in (NOT_IMPLEMENTED, DEAD_CODE) for r in results)
        if has_issues:
            print('\n⚠️  Issues detected — review PLAN_REVIEW.md for details')
            sys.exit(1)
        else:
            print('\n✅ All verifiable plan items implemented')
            sys.exit(0)


if __name__ == '__main__':
    main()
