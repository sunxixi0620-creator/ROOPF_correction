#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 rNN mechanism-slug [remote]" >&2
    exit 2
fi

round="$1"
slug="$2"
remote="${3:-origin}"

if [[ ! "$round" =~ ^r[0-9]{2}$ ]]; then
    echo "Round must match rNN, for example r01." >&2
    exit 2
fi
if [[ ! "$slug" =~ ^[a-z0-9][a-z0-9-]*$ ]]; then
    echo "Mechanism slug must use lowercase letters, digits, and hyphens." >&2
    exit 2
fi

experiment_branch="experiment/${round}-${slug}"
baseline_branch="baseline/${round}-${slug}"
baseline_tag="baseline-${round}-${slug}"
current_branch="$(git branch --show-current)"
head_sha="$(git rev-parse HEAD)"

if [[ "$current_branch" != "$experiment_branch" ]]; then
    echo "Current branch is '$current_branch'; expected '$experiment_branch'." >&2
    exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
    echo "Working tree must be clean before baseline promotion." >&2
    exit 1
fi
if ! git remote get-url "$remote" >/dev/null 2>&1; then
    echo "Remote '$remote' is not configured." >&2
    exit 1
fi

if git show-ref --verify --quiet "refs/heads/$baseline_branch"; then
    [[ "$(git rev-parse "$baseline_branch")" == "$head_sha" ]] || {
        echo "Local baseline branch points to a different commit." >&2
        exit 1
    }
else
    git branch "$baseline_branch" "$head_sha"
fi

if git rev-parse -q --verify "refs/tags/$baseline_tag" >/dev/null; then
    [[ "$(git rev-list -n 1 "$baseline_tag")" == "$head_sha" ]] || {
        echo "Local baseline tag points to a different commit." >&2
        exit 1
    }
else
    git tag -a "$baseline_tag" -m "ROOPF-Agentic baseline $round: $slug" "$head_sha"
fi

git push -u "$remote" "$experiment_branch"
git push "$remote" "$baseline_branch"
git push "$remote" "$baseline_tag"

echo "Promoted $head_sha to $baseline_branch and $baseline_tag on $remote."
