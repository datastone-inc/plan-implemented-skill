---
name: plan-implemented
description: "Post-implementation audit of Claude Code /plan output against actual code changes. Use when verifying a plan was implemented, checking if a plan was followed, auditing implementation completeness, finding dead code from a plan, or reviewing what was missed after Claude Code finished executing a plan. Supports reviewing committed changes, uncommitted work, or everything since the plan was created. Also trigger when the user says 'was the plan actually followed', 'did CC implement everything', 'check the plan against the diff', 'what did we miss', 'audit the plan', 'review uncommitted changes against the plan', or 'what changed since the plan'. Works with .claude/plans/ markdown files. Use this even when the user just wants a general post-implementation check — they don't need to say 'plan' explicitly if there's a plan in the project."
argument-hint: "[plan-file] [--base branch] [--scope branch|plan|uncommitted|all]"
allowed-tools: Bash(python:*), Bash(git:*), Read, Grep, Glob, Edit, AskUserQuestion
---

# Plan Implementation Audit

You are auditing whether a Claude Code `/plan` was fully implemented. The audit script gathers structured evidence (where patterns were found), and **you** make all implementation verdicts by reading the actual code. The script never says "implemented" or "not implemented" — that's your job.

## Step 1: Determine the plan file

If `$ARGUMENTS` includes a path, use that as the plan file.

If no plan file is specified, discover it automatically:

1. Check `.claude/settings.local.json`, `.claude/settings.json`, and `~/.claude/settings.json` for a `plansDirectory` setting
2. If not configured, check `.claude/plans/` in the repo, then `~/.claude/plans/`
3. Pick the most recently modified `.md` file in whichever directory is found
4. Tell the user which plan you found and confirm before proceeding

To list available plans for the user:
```bash
python <skill-path>/scripts/review.py --list --repo .
```

## Step 2: Determine the diff scope

The `--scope` flag controls which changes to review. If `$ARGUMENTS` includes `--scope <value>`, use it. Otherwise, choose intelligently:

| Scope | What it diffs | When to use |
|-------|--------------|-------------|
| `plan` (default) | Changes since the plan file was last modified, including uncommitted | Best general-purpose option — captures everything done since the plan was written |
| `branch` | Committed changes only: `base..HEAD` | When you want a clean committed-only view |
| `uncommitted` | Staged + unstaged vs HEAD | When the user just finished implementing and hasn't committed yet |
| `all` | Committed + uncommitted vs base branch | When you want the complete picture against the base |

If the user says things like "check my uncommitted work" or "I haven't committed yet", use `--scope uncommitted`. If they say "what changed since the plan", use `--scope plan`. If they just say "audit the plan", the default `plan` scope is usually right.

## Step 3: Determine the base branch

If `$ARGUMENTS` includes `--base <branch>`, use that branch.

Otherwise, default to `main`. If the repo uses a different default branch, detect it:
```bash
git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's@^refs/remotes/origin/@@'
```

Note: `--base` is only used by the `branch` and `all` scopes. The `plan` scope anchors to the plan's modification time instead, and `uncommitted` diffs against HEAD.

## Step 4: Run the evidence-gathering script

Output to a temp file — the user doesn't need to open it, you present the results interactively.

Use a temp filename based on the plan's basename so parallel runs don't collide:
```bash
python <skill-path>/scripts/review.py <plan-file> --base <branch> --scope <scope> --repo . --output /tmp/plan-review-<plan-basename>.md
```
For example, if the plan file is `giggly-snacking-tarjan.md`, use `--output /tmp/plan-review-giggly-snacking-tarjan.md`.

The script runs three phases:
1. **Parse** — extracts discrete checkable items from the plan (types, functions, fields, filters, tests, wiring)
2. **Evidence** — gathers diffs and current file contents according to the chosen scope
3. **Cross-reference** — for each plan item, records WHERE its patterns were found (diff, current files, or nowhere)

The script produces **evidence levels**, not implementation verdicts:
- ✅ `IN_DIFF` — all patterns found in diff added lines (strong signal)
- 🔍 `MIXED` — some in diff, others only in current files or not found
- ⚠️ `PRE_EXISTING` — pattern names exist in codebase but NOT in diff (name match only — cannot confirm changes)
- ❌ `NOT_FOUND` — not found anywhere
- ⏭️ `SKIPPED` — nothing to verify mechanically

The script exits 0 if all items are IN_DIFF with no dead-code signals, 1 otherwise.

**After reading the temp file**, delete only that specific file:
```bash
python -c "import os; os.remove('/tmp/plan-review-<plan-basename>.md')"
```

If the user explicitly asks to save the report (e.g. `--output path`), write it there instead and do NOT delete it.

### Fallback: script fails to parse

If the script exits with "No verifiable items found in plan" or extracts 0 items, **do not give up**. Fall back to a manual audit:

1. Read the plan file yourself
2. Get the diff using the appropriate scope:
   - `plan` scope: `git log --before=<plan-mtime-iso> -1 --format=%H --all` to find the anchor commit, then `git diff <anchor>`
   - `uncommitted`: `git diff HEAD`
   - `branch`: `git diff <base>..HEAD`
   - `all`: `git diff <base>`
3. Break the plan into discrete checkable items yourself — look for: file targets, function/type/field names in backticks, numbered steps, sub-headings describing work
4. For each item, read the relevant source files and verify the plan's *specific changes* were made
5. **CRITICAL: Do not confuse "name exists" with "plan work was done."** If the plan says "optimize `classifyBatch` to use batch API" and you find a `classifyBatch` function, you must verify it contains the *specific changes* the plan describes (batch API calls, new parameters, etc.), not just that a function with that name exists. Pre-existing code matching a plan item's name is NOT evidence of implementation.
6. Continue to Step 5 with your findings

## Step 5: Evaluate the evidence and present your verdict

Read the temp file. For each plan item, **you** decide whether it was implemented by reading the evidence and, when needed, the actual source files.

### Your evaluation process

1. **IN_DIFF items** — patterns found in diff added lines. These are likely implemented, but skim the evidence to confirm the diff lines match the plan's intent (not just a coincidental name match in an unrelated change).

2. **MIXED items** — some patterns in diff, some not. Read the source files to determine: were the missing patterns implemented under different names, inlined, or genuinely missing?

3. **PRE_EXISTING items** — pattern names exist in the codebase but NOT in the diff. **These are the most dangerous for false positives.** You MUST read the actual code and compare it against the plan's description. A function named `classifyBatch` existing does NOT mean the plan's "optimize classifyBatch to use batch API" was implemented. Check for the specific changes the plan describes.

4. **NOT_FOUND items** — not found anywhere. Likely not implemented, but check if the work was done under different names or restructured.

5. **Dead-code signals** — patterns found in diff but grep suggests they're never called/assigned/read. Verify whether they're actually wired up.

### Present your verdict

Present a concise summary to the user:

1. **Context** — which plan, scope used, branch, HEAD commit
2. **Your scorecard** — your verdicts: implemented, not implemented, needs investigation
3. **Per-Change overview** — one line per Change group. For example:
   ```
   Change 1: Add TurnOrigin type — 3 items, all implemented
   Change 2: Claude Code normalizer — 8 items, 6 implemented, 2 missing
   Change 5: Downstream consumers — 5a done, 5b-5d have gaps
   ```
   Use the sub_id (5a, 5b, etc.) when available to make cross-referencing with the plan easy.

For items you couldn't fully verify from the evidence alone, say so: "I verified X by reading the source; Y needs your confirmation."

If everything is implemented, say so and you're done.

**STOP.** You MUST call `AskUserQuestion` NOW to ask the user how they want to proceed. Suggested options:
- Walk through gaps (default — go Change by Change through missing and uncertain items, with code-level analysis showing what's missing and why)
- Jump to a specific Change group
- Fix everything automatically
- Just show me the details (dump the full evidence without interactive flow)

Do NOT proceed to Step 6 until the user responds.

## Step 6: Interactive walkthrough

Walk through gaps one Change group at a time, starting with the most impactful. For each group with issues:

1. **Show your analysis.** Read the relevant source files and diff sections. Explain what the plan expected vs what the code actually does. Be specific — quote code, not just pattern names.

2. **Assess each gap:**
   - **Implemented differently** — the intent is covered but under a different name or structure
   - **Blocker** — feature is broken without this
   - **Gap** — feature works but incomplete
   - **Nice-to-have** — plan item that isn't critical

3. **Be explicit about your reasoning.** For example: "The plan expected `classifyTurnOrigin()` as a separate method, but the implementation inlined the logic into `buildTurns()` at line 142 — the intent is covered."

**STOP.** After presenting each Change group, you MUST call `AskUserQuestion` to ask the user what to do. Suggested options:
- **Fix** — implement the missing pieces for this Change group
- **Skip** — move on to the next Change group
- **Stop** — end the walkthrough

Do NOT move to the next Change group or start fixing until the user responds. Wait for their input on every group.

## Step 7: Fix and re-verify

When the user asks you to fix a gap:

1. Read the relevant plan section and source files
2. Implement the fix using Edit
3. After all requested fixes in a group are done:

**STOP.** You MUST call `AskUserQuestion` to ask: "Fixes applied. Want me to re-run the audit to verify, or move on to the next Change group?"

Do NOT continue until the user responds.

4. If they want re-verification, re-run with `--scope uncommitted` (to catch the just-made changes) and present an updated summary. Then continue the walkthrough with remaining Change groups.
5. If they want to move on, proceed to the next Change group (back to Step 6).

## Step 8: Clean up completed plan

After a successful audit — either everything is implemented in Step 5, or the user finishes the walkthrough and all gaps are resolved — offer to delete the plan file.

Only offer this for plans in `~/.claude/plans/` (the global auto-generated folder). Do not offer to delete plans stored inside a repo (e.g. `.claude/plans/`), as those are project-scoped and version-controlled.

**STOP.** You MUST call `AskUserQuestion` to ask: "This plan is fully implemented. Want me to delete `<plan-file>`?"

Do NOT delete the file until the user confirms. If the user says no, leave it.

## What the script provides

| Evidence | Meaning |
|----------|---------|
| Pattern in diff | Identifier appears in added lines of the diff — strong signal of new work |
| Pattern in current file | Identifier exists in the codebase but not in the diff — name match only, does NOT confirm changes |
| File modified | Target file was touched in the diff |
| Dead-code signal | Pattern found but grep suggests it's never called/assigned/read |

The script does NOT make pass/fail judgments. It provides structured evidence for your evaluation. Dead-code detection is grep-based — it catches "never assigned" and "never called" but not "condition always false" or "unreachable branch."

## Example invocations

User types: `/plan-implemented`
-> Auto-discover most recent plan, gather evidence, evaluate and walk through gaps

User types: `/plan-implemented .claude/plans/turn-origin-tagging.md`
-> Use that specific plan file, evaluate changes since it was written

User types: `/plan-implemented --scope uncommitted`
-> Auto-discover plan, evaluate only uncommitted work against it

User types: `/plan-implemented --scope branch --base develop`
-> Auto-discover plan, evaluate committed changes only vs develop

User types: "check if the plan was actually implemented"
-> Auto-trigger, discover the plan, gather evidence, evaluate interactively

User types: "I haven't committed yet, did I cover everything in the plan?"
-> Auto-trigger with `--scope uncommitted`

User types: "fix the gaps from the plan"
-> After an audit has run, jump straight to fixing
