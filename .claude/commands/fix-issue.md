# Fix GitHub Issue

You are tasked with fixing GitHub issue #$ARGUMENTS. Follow this workflow precisely, step by step.

## Step 1: Understand the Issue

Fetch the issue details:

```
gh issue view $ARGUMENTS
```

Read the issue thoroughly. Understand what the bug is, what the expected behavior should be, and any reproduction steps provided.

## Step 2: Research

Investigate the codebase to understand:
- Where the relevant code lives
- What causes the reported behavior
- What a correct fix looks like

Use search, file reads, and any other tools you need. Be thorough — understand the problem fully before writing any code.

## Step 3: Create a Worktree

Create a worktree and switch to it. Use the branch name `fix-$ARGUMENTS`:

```
git worktree add .claude/worktrees/fix-$ARGUMENTS -b fix-$ARGUMENTS
```

All subsequent work MUST happen in the worktree directory: `.claude/worktrees/fix-$ARGUMENTS`.

## Step 4: Write a Failing Test (TDD)

Following the project's TDD convention (see CLAUDE.md — "Bugs and Issues"):

1. Write a test that reproduces the issue. Follow the project's test conventions — register an `OperationTestConfig` in `tests/test_ops.py` if appropriate, or add a test in the relevant test file.
2. Run the test and **verify it fails**. If the test passes, your test does not reproduce the issue — revisit your understanding and fix the test.

## Step 5: Implement the Fix

1. Implement the minimal fix for the issue.
2. Follow all project conventions from CLAUDE.md (PascalCase handlers, single OpRegistry, etc.).
3. Rebuild: `uv pip install -e .` (run this from the worktree directory).
4. Run the failing test again and **verify it now passes**.
5. Run the full test suite (`uv run pytest`) to make sure nothing is broken.

## Step 6: Commit and Push

1. Stage only the relevant files (no `.env`, credentials, or unrelated changes).
2. Write a clear commit message explaining the fix and referencing the issue:
   ```
   Fix #$ARGUMENTS: <concise description>

   <details about what was wrong and how the fix works>

   Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
   ```
3. Push the branch:
   ```
   git push -u origin fix-$ARGUMENTS
   ```

## Step 7: Create a Pull Request

Create a PR that references the issue:

```
gh pr create --title "Fix #$ARGUMENTS: <concise description>" --body "$(cat <<'EOF'
## Summary

Fixes #$ARGUMENTS

<1-3 bullet points explaining the fix>

## Test plan

- [ ] New test reproduces the original issue (fails without fix)
- [ ] New test passes with fix applied
- [ ] Full test suite passes

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

## Step 8: Wait for CI and Copilot Review

Poll the PR checks until they all complete:

```
gh pr checks <PR_NUMBER> --watch
```

If any checks fail, investigate and fix. Commit, push, and wait again.

## Step 9: Address Copilot Review

After CI passes, check for Copilot review comments:

```
gh api repos/{owner}/{repo}/pulls/<PR_NUMBER>/comments
```

For each Copilot comment:

1. **Assess whether the feedback is valid.** Copilot sometimes produces noise — generic suggestions, stylistic nitpicks that don't match this project's conventions, or "improvements" that add unnecessary complexity.

2. **If the feedback is valid and actionable:** implement the change, commit, and push.

3. **If the feedback is noise:** reply to the comment on the PR explaining why you consider it non-actionable, then move on. Example:
   ```
   gh api repos/{owner}/{repo}/pulls/<PR_NUMBER>/comments/<COMMENT_ID>/replies -f body="This suggestion doesn't apply here because <reason>. Moving on."
   ```

After addressing all comments, if you made changes, wait for CI to pass again (repeat Step 8).

## Step 10: Clean Up and Report

Return to the main repository directory and remove the worktree:

```
cd /Users/till/git/jax-mps
git worktree remove .claude/worktrees/fix-$ARGUMENTS
```

Then provide a summary to the user covering:

1. **Diagnosis**: What was the root cause of the issue? Explain the chain of events that led to the bug.
2. **Fix**: What was changed and why? Describe the approach taken.
3. **Improvement opportunities**: Are there related areas of the codebase that could benefit from further work? Performance improvements, robustness enhancements, or related issues that this investigation uncovered.

Report the PR URL at the end.

## Important Rules

- NEVER skip or xfail tests without explicit user approval.
- NEVER push to `main`.
- NEVER use `--no-verify` for commits.
- NEVER delete operations or tests without explicit user approval.
- Use `uv run ...` for all commands.
- Use `uv pip install -e .` to rebuild after code changes.
