#!/bin/bash
set -euo pipefail

# Get current version from pyproject.toml
CURRENT_VERSION=$(grep '^version' pyproject.toml | cut -d'"' -f2)
TAG="v${CURRENT_VERSION}"

echo "Current version: ${CURRENT_VERSION}"
echo "Tag to create: ${TAG}"

# Check for uncommitted changes
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Error: uncommitted changes present. Commit or stash them first."
    exit 1
fi

# Check we're on main
BRANCH=$(git branch --show-current)
if [[ "$BRANCH" != "main" ]]; then
    echo "Error: not on main branch (currently on ${BRANCH})"
    exit 1
fi

# Check we're in sync with remote
echo "Fetching from origin..."
git fetch origin main
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)
if [[ "$LOCAL" != "$REMOTE" ]]; then
    echo "Error: local main is not in sync with origin/main"
    echo "  Local:  ${LOCAL}"
    echo "  Remote: ${REMOTE}"
    exit 1
fi

# Check tag doesn't already exist
if git rev-parse "$TAG" >/dev/null 2>&1; then
    echo "Error: tag ${TAG} already exists"
    exit 1
fi

# Determine next version (bump patch)
IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT_VERSION"
NEXT_VERSION="${MAJOR}.${MINOR}.$((PATCH + 1))"

echo "Next version: ${NEXT_VERSION}"
echo ""
read -p "Proceed with release? [y/N] " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Create and push tag
echo "Creating tag ${TAG}..."
git tag -a "$TAG" -m "Release ${CURRENT_VERSION}"
git push origin "$TAG"

# Bump version
echo "Bumping version to ${NEXT_VERSION}..."
uv version "$NEXT_VERSION"

# Commit and push version bump
git add pyproject.toml uv.lock
git commit -m "Bump version to ${NEXT_VERSION}"
git push

echo ""
echo "Done! Release ${CURRENT_VERSION} is being built."
echo "Watch progress at: https://github.com/tillahoffmann/jax-mps/actions"
