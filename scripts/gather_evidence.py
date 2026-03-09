#!/usr/bin/env python3
"""
Gather implementation evidence from git for plan verification.

Collects:
- Full git diff (base..HEAD)
- Per-file diffs keyed by path
- Current file contents for files mentioned in the plan
- List of all modified files

Output: JSON dict to stdout.
"""

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def run_git(args: list[str], repo: Path) -> tuple[int, str, str]:
    """Run a git command and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        ['git'] + args,
        capture_output=True,
        text=True,
        cwd=str(repo),
        timeout=30,
    )
    return result.returncode, result.stdout, result.stderr


def find_commit_at_time(timestamp: float, repo: Path) -> Optional[str]:
    """Find the commit hash closest to (but before) a given unix timestamp."""
    iso_time = datetime.fromtimestamp(timestamp).isoformat()
    rc, out, _ = run_git(
        ['log', f'--before={iso_time}', '-1', '--format=%H', '--all'],
        repo,
    )
    if rc == 0 and out.strip():
        return out.strip()
    return None


def get_diff(base: str, repo: Path, scope: str = 'branch',
             plan_mtime: Optional[float] = None) -> Optional[str]:
    """Get diff according to the requested scope.

    Scopes:
      branch      — committed changes: base..HEAD (original behavior)
      plan        — changes since the plan was created/updated, including uncommitted
      uncommitted — only staged + unstaged changes vs HEAD
      all         — committed + uncommitted changes vs base branch
    """
    if scope == 'uncommitted':
        # Staged + unstaged against HEAD
        rc, out, _ = run_git(['diff', 'HEAD'], repo)
        if rc == 0 and out.strip():
            return out
        # Maybe everything is staged
        rc, out, _ = run_git(['diff', '--cached'], repo)
        if rc == 0 and out.strip():
            return out
        return None

    if scope == 'plan' and plan_mtime:
        # Find the commit at or before the plan's mtime
        anchor = find_commit_at_time(plan_mtime, repo)
        if anchor:
            # Include uncommitted changes: diff anchor against working tree
            rc, out, _ = run_git(['diff', anchor], repo)
            if rc == 0 and out.strip():
                return out
        # Fall through to branch behavior if plan-anchoring fails

    if scope == 'all':
        # Committed + uncommitted against base (working tree vs base)
        rc, out, _ = run_git(['diff', base], repo)
        if rc == 0 and out.strip():
            return out

    # Default / branch scope: committed only
    rc, out, _ = run_git(['diff', f'{base}..HEAD'], repo)
    if rc == 0 and out.strip():
        return out

    # Fallback: staged + unstaged against base
    rc, out, _ = run_git(['diff', base], repo)
    if rc == 0 and out.strip():
        return out

    # Last resort: diff against HEAD~1
    rc, out, _ = run_git(['diff', 'HEAD~1..HEAD'], repo)
    if rc == 0:
        return out

    return None


def parse_diff_by_file(full_diff: str) -> dict[str, str]:
    """Split a unified diff into per-file sections."""
    files = {}
    current_file = None
    current_lines = []

    for line in full_diff.split('\n'):
        # Detect new file in diff
        if line.startswith('diff --git'):
            if current_file and current_lines:
                files[current_file] = '\n'.join(current_lines)
            current_lines = [line]
            # Extract b-side path
            match = re.search(r'b/(.+)$', line)
            current_file = match.group(1) if match else None
        else:
            current_lines.append(line)

    # Don't forget the last file
    if current_file and current_lines:
        files[current_file] = '\n'.join(current_lines)

    return files


def get_modified_files(base: str, repo: Path, scope: str = 'branch',
                       plan_mtime: Optional[float] = None) -> list[str]:
    """Get list of modified files matching the diff scope."""
    def _parse(out: str) -> list[str]:
        return [f.strip() for f in out.strip().split('\n') if f.strip()]

    if scope == 'uncommitted':
        rc, out, _ = run_git(['diff', '--name-only', 'HEAD'], repo)
        if rc == 0 and out.strip():
            return _parse(out)
        rc, out, _ = run_git(['diff', '--name-only', '--cached'], repo)
        if rc == 0 and out.strip():
            return _parse(out)
        return []

    if scope == 'plan' and plan_mtime:
        anchor = find_commit_at_time(plan_mtime, repo)
        if anchor:
            rc, out, _ = run_git(['diff', '--name-only', anchor], repo)
            if rc == 0 and out.strip():
                return _parse(out)

    if scope == 'all':
        rc, out, _ = run_git(['diff', '--name-only', base], repo)
        if rc == 0 and out.strip():
            return _parse(out)

    # Default / branch
    rc, out, _ = run_git(['diff', '--name-only', f'{base}..HEAD'], repo)
    if rc == 0 and out.strip():
        return _parse(out)

    rc, out, _ = run_git(['diff', '--name-only', base], repo)
    if rc == 0 and out.strip():
        return _parse(out)

    return []


def read_current_files(file_patterns: list[str], repo: Path) -> dict[str, str]:
    """Read current content of files matching the given patterns."""
    contents = {}

    for pattern in file_patterns:
        if not pattern:
            continue

        # Try exact path first
        exact = repo / pattern
        try:
            contents[pattern] = exact.read_text(encoding='utf-8')
            continue
        except (FileNotFoundError, IsADirectoryError, OSError):
            pass

        # Try as a filename suffix match (plan often has partial paths)
        for p in repo.rglob('*'):
            if p.is_file() and p.name == Path(pattern).name:
                rel = str(p.relative_to(repo))
                if 'node_modules' not in rel and '.git' not in rel:
                    try:
                        contents[rel] = p.read_text(encoding='utf-8')
                    except (FileNotFoundError, OSError):
                        pass
                    break  # take first match

    return contents


def gather_evidence(base: str, repo: Path, plan_files: list[str],
                    scope: str = 'branch',
                    plan_mtime: Optional[float] = None) -> dict:
    """Gather all evidence needed for cross-referencing.

    Args:
        base: Git ref to diff against (for branch/all scopes).
        repo: Repository root path.
        plan_files: File paths mentioned in the plan.
        scope: One of 'branch', 'plan', 'uncommitted', 'all'.
        plan_mtime: Plan file's modification time (unix timestamp), used by 'plan' scope.
    """
    evidence = {
        'base': base,
        'scope': scope,
        'repo': str(repo),
        'full_diff': None,
        'file_diffs': {},
        'modified_files': [],
        'current_files': {},
        'errors': [],
    }

    # Check repo is a git repo
    rc, _, _ = run_git(['rev-parse', '--git-dir'], repo)
    if rc != 0:
        evidence['errors'].append(f'{repo} is not a git repository')
        return evidence

    # For scopes that need a base ref, verify it exists
    if scope in ('branch', 'all'):
        rc, _, _ = run_git(['rev-parse', '--verify', base], repo)
        if rc != 0:
            evidence['errors'].append(f'Base ref "{base}" does not exist')
            return evidence

    # Get full diff
    full_diff = get_diff(base, repo, scope=scope, plan_mtime=plan_mtime)
    if full_diff is None:
        evidence['errors'].append('Could not get diff — no changes found')
        return evidence

    evidence['full_diff'] = full_diff
    evidence['file_diffs'] = parse_diff_by_file(full_diff)
    evidence['modified_files'] = list(evidence['file_diffs'].keys())

    # Read current file contents for plan-mentioned files
    evidence['current_files'] = read_current_files(plan_files, repo)

    # Also read current state of all modified files (for dead-code detection)
    for f in evidence['modified_files']:
        if f not in evidence['current_files']:
            try:
                evidence['current_files'][f] = (repo / f).read_text(encoding='utf-8')
            except (FileNotFoundError, IsADirectoryError, OSError):
                pass

    return evidence


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Gather git evidence for plan verification.'
    )
    parser.add_argument('--base', default='main',
                        help='Git ref to diff against (default: main)')
    parser.add_argument('--repo', default='.',
                        help='Repository root (default: current directory)')
    parser.add_argument('--plan-files', nargs='*', default=[],
                        help='File paths mentioned in the plan (for reading current state)')

    args = parser.parse_args()
    repo = Path(args.repo).resolve()

    evidence = gather_evidence(args.base, repo, args.plan_files)

    # Output without full file contents to keep it manageable on stdout
    # (cross_reference.py calls gather_evidence directly as a module)
    summary = {
        'base': evidence['base'],
        'repo': evidence['repo'],
        'modified_files': evidence['modified_files'],
        'file_count': len(evidence['file_diffs']),
        'diff_size_bytes': len(evidence['full_diff']) if evidence['full_diff'] else 0,
        'current_files_read': list(evidence['current_files'].keys()),
        'errors': evidence['errors'],
    }
    json.dump(summary, sys.stdout, indent=2)
    print()


if __name__ == '__main__':
    main()
