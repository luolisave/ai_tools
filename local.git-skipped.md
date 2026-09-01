# Git Notes: Ignoring Changes to Already-Tracked Files

## Files currently skipped on my local (skip-worktree)

Recorded 2026-09-01. These files are tracked but local changes to them are ignored via `git update-index --skip-worktree`. Verify/list them anytime with `git ls-files -v` (lines starting with `S`).

```text
read_novel_melo_cn/novel.txt
read_novel_melo_cn/novel/0001.txt
read_novel_melo_cn/novel/0002.txt
read_novel_melo_cn/novel/0003.txt
read_novel_melo_cn/novel/0004.txt
read_novel_melo_cn/novel/0005.txt
read_novel_melo_cn/novel_mp3/0001.mp3
read_novel_melo_cn/novel_mp3/0002.mp3
read_novel_melo_cn/novel_mp3/0003.mp3
read_novel_melo_cn/novel_mp3/0004.mp3
read_novel_melo_cn/novel_mp3/0005.mp3
```

> TL;DR: `.gitignore` only affects **untracked** files. Once a file is **tracked** (committed), you use `git update-index --skip-worktree` to keep it in the repo but ignore local changes.

## The problem

Files matched by `.gitignore` still showed up in `git status` / got modified.

**Why:** the files were committed to the repo *before* the `.gitignore` rules were added. `.gitignore` has **no effect on already-tracked files** — it only prevents *untracked* files from being added.

## The solution

Keep the files in the remote repo, but stop tracking local changes to them by marking each one with `--skip-worktree`:

```powershell
# 1. List every tracked file under the target folders
git ls-files read_novel_melo_cn/novel read_novel_melo_cn/novel_mp3 read_novel_melo_cn/novel.txt

# 2. Mark each listed file so Git ignores its local changes
git ls-files read_novel_melo_cn/novel read_novel_melo_cn/novel_mp3 read_novel_melo_cn/novel.txt | ForEach-Object { git update-index --skip-worktree $_ }

# 3. Verify status is clean
git status --short
```

## Command breakdown

- `git ls-files <path...>` — lists **currently tracked** files (in the index/staging area). Needed because `git update-index` works on **individual files**, not whole folders, so we must first gather the file list.
- `|` — pipe: feeds the output line-by-line into the next command.
- `ForEach-Object { ... }` — PowerShell loop; runs the block once for **each** incoming line.
- `$_` — inside the loop, the **current line** (the current filename).
- `git update-index --skip-worktree <file>` — tells Git: *"this file is still tracked, but ignore any local changes to it."*
- `git status --short` (`-s`) — compact one-line-per-change status.

## Related commands

| Goal | Command |
|------|---------|
| **Undo** (start tracking changes again) | `git update-index --no-skip-worktree <file>` |
| See which files are marked | `git ls-files -v` (files starting with `S` = skip-worktree) |
| Old-school cousin (auto-droppable flag) | `git update-index --assume-unchanged <file>` |

### `--skip-worktree` vs `--assume-unchanged`

- `--assume-unchanged` is the older flag; Git may drop it on its own under certain conditions.
- `--skip-worktree` is more persistent and is the right choice for the *"keep in repo but ignore locally"* use case.

## Key concepts

- **`.gitignore`** only prevents **untracked** files from being added. It does nothing to **tracked** files.
- **`git update-index --skip-worktree`** keeps the file in the repo while ignoring local modifications.
- These flags are **local-only** — they are **not pushed** to the remote, so teammates' clones will not have them.
- If you want a rule shared with everyone, the alternative is `git rm --cached` (removes from repo, keeps on disk) — but that **contradicts** the "keep the files in the remote" goal.
- If the file changes on the remote and you `git pull`, Git may complain about local divergence. Un-set the flag temporarily with `git update-index --no-skip-worktree <file>` to resolve.