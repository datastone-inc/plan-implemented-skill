<!-- markdownlint-disable MD013 MD033 MD041 -->
<p align="center"><img src="assets/datastone_logo.png" width="300" alt="dataStone logo" /></p>
<!-- markdownlint-enable MD033 MD041 -->

# plan-implemented skill for Claude Code

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Claude Code](https://img.shields.io/badge/Claude%20Code-Skill-blueviolet.svg)](https://docs.anthropic.com/en/docs/claude-code)

Post-implementation cross-check skill for Claude Code `/plan`. Verifies that a `/plan` was fully implemented by comparing plan items against actual code changes. Catches phantom completions (tasks marked complete that were never done), dead code, and missing tests, then interactively walks you through gaps and offers to fix them.

> [!NOTE]
>
> - Works on committed and uncommitted work. Run before or after committing.
> - Complements code review but doesn't replace it. Code review checks whether the code that was written is correct. This skill checks whether the code that was supposed to be written actually exists. See Phantom Completions in AI-Assisted Development for the theory behind why this gap occurs.
> - Suggested to run this **before** `/simplify`, `/refactor`, or similar commands. If there's a gap between the plan and the current code, those commands may remove or restructure code that appears unused or dead, but is only that way because the wiring is missing. Close the phantom completions and implementation gaps first, then simplify.

## Installation

### Clone into the global skills folder

```bash
mkdir -p ~/.claude/skills
git clone https://github.com/datastone-inc/plan-implemented-skill ~/.claude/skills/plan-implemented
```

This installs the skill globally, available in all your projects. That covers most people.

**Project-scoped install (niche):** If you need the skill scoped to a single repo, avoid cloning into it, as that creates a nested git repo which Git handles poorly. Copy from the global install instead:

```bash
mkdir -p .claude/skills
cp -r ~/.claude/skills/plan-implemented .claude/skills/plan-implemented
```

Note: the copy won't receive updates via `git pull`. You'll need to re-copy after updating the global install.

Restart Claude Code or start a new session.

### Keeping it up-to-date

```bash
cd ~/.claude/skills/plan-implemented && git pull
```

## Requirements

- Python 3.10+
- Git

No pip dependencies. Uses only the Python standard library.

## Usage

In Claude Code:

``` text
# Auto-discover most recent plan, review all changes (committed + uncommitted) since it was written
/plan-implemented

# Specify a plan file
/plan-implemented .claude/plans/my-plan.md

# Review only uncommitted work
/plan-implemented --scope uncommitted

# Review committed changes only vs a branch
/plan-implemented --scope branch --base develop

# Review everything (committed + uncommitted) vs main
/plan-implemented --scope all

# List available plans
/plan-implemented --list

# Or just say it naturally
> was the plan actually followed?
> check if the plan was implemented
> I haven't committed yet, did I cover the plan?

# You can also run the skill with ad-hoc AI augmentation, e.g.:
> /plan-implemented list all the plans in .claude/plans in reverse chronological order with a brief summary title. run the plan-implemented skill on the ones I select in parallel.
```

### Scopes

| Scope | What it diffs | When to use |
| ------ | ------------ | ----------- |
| `plan` (default) | Changes since the plan was last modified, including uncommitted | Best general-purpose option |
| `branch` | Committed changes only: `base..HEAD` | Clean committed-only view |
| `uncommitted` | Staged + unstaged vs HEAD | Just finished implementing, haven't committed |
| `all` | Committed + uncommitted vs base branch | Complete picture |

### CLI Reference

Direct script usage (for automation or debugging):

```text
usage: review.py [-h] [--base BASE] [--scope {branch,plan,uncommitted,all}]
                 [--repo REPO] [--output OUTPUT] [--json] [--list]
                 [plan_file]

Audit whether a Claude Code /plan was fully implemented.

positional arguments:
  plan_file             Path to plan markdown file. If omitted, discovers the
                        most recent plan from CC settings or default
                        locations.

options:
  -h, --help            show this help message and exit
  --base BASE           Git ref to diff against (default: main)
  --scope {branch,plan,uncommitted,all}
                        What changes to review: branch=committed vs base
                        branch, plan=changes since plan was created/updated
                        (default), uncommitted=only staged+unstaged vs HEAD,
                        all=committed+uncommitted vs base branch
  --repo REPO           Repository root (default: current directory)
  --output OUTPUT       Output file path (default: PLAN_REVIEW.md in repo
                        root)
  --json                Output raw JSON results instead of markdown
  --list                List available plans and exit
```

**Examples:**

```bash
# Run directly from command line
python3 scripts/review.py examples/sample-plan.md --repo .

# Review uncommitted work only
python3 scripts/review.py --scope uncommitted

# Compare against develop branch instead of main
python3 scripts/review.py --base develop

# Output JSON for further processing
python3 scripts/review.py --json > results.json

# List all available plans
python3 scripts/review.py --list
```

## Interactive flow

1. **Summary**: scorecard and per-Change overview
2. **Walkthrough**: one Change group at a time, explaining gaps, assessing severity, flagging false positives
3. **Fix**: implement missing pieces in place when asked
4. **Re-verify**: re-run the audit after fixes to confirm

Claude pauses at each step for your input. You choose what to fix, skip, or stop.

## Language support

Pattern extraction works across multiple languages, detected automatically from file extensions and code fences:

TypeScript/JS, Python, Rust, Go, Java/Kotlin, C/C++, C#, Ruby, Swift, SQL

To add a new language, add an entry to `scripts/languages.py`. No other code changes needed. See the existing entries for the pattern format.

## How it works

1. **Parse**: reads the plan markdown and extracts verifiable items (types, functions, fields, filters, tests) using language-aware pattern extraction
2. **Evidence**: gathers changes according to the chosen scope (branch diff, plan-anchored, uncommitted, or all)
3. **Cross-reference**: checks each item against the diff; language-aware dead-code detection finds declared-but-unused symbols. This cross-reference step is what catches phantom completions, where the agent marked a plan item complete without performing the specified work.
4. **Interactive**: Claude presents findings, walks through gaps, and offers to fix them

## Architecture

The skill operates as a 4-stage pipeline. Each stage is independent and testable:

```text
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Parse      │ -> │   Evidence   │ -> │     Cross    │ -> │ Interactive  │
│   Plan       │    │   Gather     │    │  Reference   │    │   Review     │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

### Stage 1: Parse Plan ([`scripts/parse_plan.py`](scripts/parse_plan.py))

Reads a Claude Code `/plan` markdown file and extracts verifiable items:

- Recognizes `## Change N:` headings to group related work
- Extracts code blocks and applies language-specific regex patterns from [`scripts/languages.py`](scripts/languages.py)
- Finds inline code mentions in prose (e.g., "Update the `handleRequest` function")
- Categorizes items: `type_definition`, `function`, `field`, `test`, `filter_logic`, `wiring`
- Outputs structured JSON: `{id, change_id, change_title, file_pattern, expected_patterns, category}`

### Stage 2: Evidence Gather ([`scripts/gather_evidence.py`](scripts/gather_evidence.py))

Collects git diff and file contents according to scope:

- **Scopes**:
  - `plan` (default): Changes since plan file was last modified
  - `branch`: Committed changes only (`base..HEAD`)
  - `uncommitted`: Staged + unstaged vs HEAD
  - `all`: Committed + uncommitted vs base
- Parses unified diffs by file
- Reads current file contents for pattern searching
- Handles exact path and basename matching

### Stage 3: Cross-Reference ([`scripts/cross_reference.py`](scripts/cross_reference.py))

Matches plan patterns against diff evidence:

- **Evidence levels**:
  - ✅ `IN_DIFF`: All patterns found in added diff lines
  - 🔍 `MIXED`: Some in diff, others pre-existing
  - ⚠️ `PRE_EXISTING`: Pattern exists but not in diff
  - ❌ `NOT_FOUND`: Not found anywhere
  - ⏭️ `SKIPPED`: No mechanically verifiable patterns
- **Dead-code detection**:
  - Type definitions: searches for references elsewhere
  - Functions: searches for calls using language-specific `call_pattern`
  - Fields: searches for assignments and reads using `access_pattern`
- Generates markdown report with evidence table and dead-code signals

### Stage 4: Interactive Review (orchestrated by Claude Code)

Claude Code reads the generated report and walks through findings with you:

1. Presents summary scorecard and per-Change overview
2. Reviews gaps one Change at a time, explaining evidence and severity
3. Offers to implement missing pieces when you confirm
4. Re-runs the audit after fixes to verify completion

The scripts provide evidence; Claude Code makes the verdicts and suggests fixes.

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines. In particular, you may want to focus on adding language support or improving the cross-reference logic.

For bug reports and feature requests, please open an issue.

## Authors

**Dave Sharpe** ([dave.sharpe@datastone.ca](mailto:dave.sharpe@datastone.ca)) at **[dataStone Inc.](https://github.com/datastone-inc)**: concept, design, and development

**[Claude (Anthropic)](https://www.anthropic.com)**: co-developed the implementation, scripts, and multi-language support via Claude Code

## License

[MIT](LICENSE)
