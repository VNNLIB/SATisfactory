# Contributing Guide

## Repository Structure

```
/src        - all source code
/docs       - meeting minutes, timesheets, design documents, client communications
/tests      - test files and test logs
```

All project outputs (code, documents, test logs) must be committed here. This repository must remain **private** at all times.

---

## Branches

| Branch | Purpose |
|---|---|
| `main` | Stable, production-ready code. Updated at the end of each sprint only. |
| `develop` | Active integration branch. All feature branches merge here. |
| `feature/...` | Individual work branches. Created from `develop`, merged back into `develop`. |

**Rules:**
- Never push directly to `main`. All changes go through a pull request with at least one approval.
- Create all new branches from `develop`, not `main`.
- Delete your feature branch after it has been merged.

---

## Branch Naming

```
feature/short-description
fix/short-description
hotfix/short-description
docs/short-description
test/short-description
security/short-description
refactor/short-description
```

**Examples:**
```
feature/user-authentication
fix/broken-login-redirect
docs/add-week3-minutes
test/database-unit-tests
security/sanitise-user-inputs
refactor/cleanup-api-calls
hotfix/crash-on-startup
```

Keep names lowercase, use hyphens not underscores, be specific enough that the purpose is clear without reading the code.

---

## Commit Messages

Use the format: `type: short description`

| Type | When to use |
|---|---|
| `feat` | Adding new functionality |
| `fix` | Fixing a bug |
| `docs` | Adding or updating documentation or meeting files |
| `test` | Adding or updating tests |
| `refactor` | Code changes that don't fix a bug or add a feature |
| `security` | Security-related changes |
| `chore` | Maintenance tasks (dependency updates, config changes) |

**Examples:**
```
feat: add login page with form validation
fix: resolve null pointer on user profile load
docs: add Week 3 meeting minutes
test: add unit tests for authentication module
security: sanitise inputs on search endpoint
```

Keep the description under 72 characters. Use present tense ("add" not "added").

---

## Issues

Create an issue for every piece of work before starting it.

**When creating an issue:**
- Write a clear title describing the outcome, not the action ("User can log in" not "work on login")
- Assign it to the person doing the work
- Attach it to the relevant Sprint milestone
- Add it to the Project board
- Apply a label (see below)

**Labels:**
```
feature
bug
documentation
test
security
refactor
blocked
```

**To link a PR to an issue**, include `Closes #<issue-number>` in the PR description. When the PR merges, the issue closes automatically.

---

## Pull Requests

All merges into `main` require a PR with at least **1 approval**. Merges into `develop` do not require a PR but using one is encouraged for significant changes.

**PR title format:** same as branch naming - `type: short description`

**PR description should include:**
- What the change does
- Any testing performed
- `Closes #<issue-number>` to link the issue

**Before opening a PR:**
- Make sure your branch is up to date with `develop`
- Resolve any merge conflicts locally
- Test your changes

**Reviewers:** assign at least one other team member. The reviewer should check the code works and makes sense, not just approve to clear the queue.

---

## GitHub Projects

The team uses a single Project board with two views:

- **Board view** - Kanban columns: Backlog / This Sprint / In Progress / In Review / Done
- **Table view** - flat list of all issues across all sprints

Filter by milestone to view work for a specific sprint. Move your card to the correct column as your work progresses.

---

## Milestones

| Milestone | Purpose |
|---|---|
| Sprint 1 | Issues to be completed by Sprint 1 deadline |
| Sprint 2 | Issues to be completed by Sprint 2 deadline |
| Sprint 3 | Issues to be completed by Sprint 3 deadline |

Every issue must be assigned to a milestone. Do not leave issues unassigned.

---

## Git Identity

Before your first commit, confirm your local Git identity matches your GitHub account:

```bash
git config --global user.name "Your Full Name"
git config --global user.email "your-github-email@example.com"
```

Commits from an unrecognised email will not appear on your GitHub contribution graph and cannot be attributed to you.

---

## Weekly Docs Commit

Each week, the Team Manager commits the following to `/docs` before Sunday evening:

- Meeting minutes (`Minutes_GroupX_Wk<N>.html`)
- Timesheet (`TimeSheet_GroupX_Wk<N>.xlsx`)

Individual members commit their Booked Hours file to `/docs` before sending it to the Team Manager.
