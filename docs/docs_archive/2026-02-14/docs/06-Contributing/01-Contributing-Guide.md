# Git Flow Configuration - Mímir Framework

This project uses **Git Flow** branching model for development workflow.

## Branch Structure

### Main Branches

- **`main`**
  - Production-ready code
  - Only release and hotfix merges allowed
  - Always stable and deployable
  - Tagged with version numbers (v1.0.0, v1.1.0, etc.)

- **`develop`**
  - Integration branch for features
  - Contains latest development changes
  - Base for feature branches
  - Generally stable but not production-ready

### Supporting Branches

- **`feature/*`**
  - New features and enhancements
  - Branch from: `develop`
  - Merge to: `develop`
  - Naming: `feature/feature-name`
  - Examples: `feature/gpu-optimization`, `feature/new-architecture`

- **`release/*`**
  - Prepare for production release
  - Branch from: `develop`
  - Merge to: `main` and `develop`
  - Naming: `release/X.Y.Z`
  - Examples: `release/1.1.0`, `release/2.0.0`

- **`hotfix/*`**
  - Critical bug fixes for production
  - Branch from: `main`
  - Merge to: `main` and `develop`
  - Naming: `hotfix/X.Y.Z`
  - Examples: `hotfix/1.0.1`, `hotfix/1.1.1`

- **`bugfix/*`**
  - Non-critical bug fixes
  - Branch from: `develop`
  - Merge to: `develop`
  - Naming: `bugfix/bug-description`
  - Examples: `bugfix/memory-leak`, `bugfix/tokenizer-crash`

- **`support/*`**
  - Long-term support for old versions
  - Branch from: tagged version on `main`
  - Naming: `support/X.Y`
  - Examples: `support/1.x`, `support/2.x`

## Workflow Commands

### Starting New Work

```bash
# Start a new feature
git flow feature start my-feature

# Start a bugfix
git flow bugfix start my-bugfix

# Start a release
git flow release start 1.1.0

# Start a hotfix
git flow hotfix start 1.0.1
```

### Finishing Work

```bash
# Finish a feature (merges to develop)
git flow feature finish my-feature

# Finish a bugfix (merges to develop)
git flow bugfix finish my-bugfix

# Finish a release (merges to main and develop, creates tag)
git flow release finish 1.1.0

# Finish a hotfix (merges to main and develop, creates tag)
git flow hotfix finish 1.0.1
```

### Publishing and Pulling

```bash
# Publish feature to remote
git flow feature publish my-feature

# Pull feature from remote
git flow feature pull origin my-feature

# Track a remote feature
git flow feature track my-feature
```

## Versioning Strategy

We follow **Semantic Versioning** (SemVer):

```
MAJOR.MINOR.PATCH

1.0.0 → Initial release
1.1.0 → New features, backward compatible
1.1.1 → Bug fixes, backward compatible
2.0.0 → Breaking changes
```

### Version Increments

- **MAJOR** (X.0.0)
  - Breaking API changes
  - Major architectural changes
  - Incompatible with previous versions

- **MINOR** (1.X.0)
  - New features (backward compatible)
  - New architectures or optimizations
  - Performance improvements

- **PATCH** (1.0.X)
  - Bug fixes
  - Security patches
  - Documentation updates

## Development Workflow

### 1. Feature Development

```bash
# Switch to develop
git checkout develop
git pull origin develop

# Create feature branch
git flow feature start awesome-feature

# Work on feature
# ... make changes ...
git add .
git commit -m "Add awesome feature"

# Finish feature (auto-merges to develop)
git flow feature finish awesome-feature

# Push develop
git push origin develop
```

### 2. Release Process

```bash
# Start release from develop
git flow release start 1.1.0

# Update version numbers
# - Update VERSION file
# - Update documentation
# - Update changelog
git commit -am "Bump version to 1.1.0"

# Finish release (merges to main and develop, creates tag)
git flow release finish 1.1.0

# Push everything
git push origin main
git push origin develop
git push origin --tags
```

### 3. Hotfix Process

```bash
# Start hotfix from main
git flow hotfix start 1.0.1

# Fix the critical bug
git commit -am "Fix critical bug in optimizer"

# Finish hotfix (merges to main and develop, creates tag)
git flow hotfix finish 1.0.1

# Push everything
git push origin main
git push origin develop
git push origin --tags
```

## Commit Message Convention

Follow **Conventional Commits** specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, no logic change)
- **refactor**: Code refactoring
- **perf**: Performance improvements
- **test**: Adding or updating tests
- **build**: Build system changes
- **ci**: CI/CD changes
- **chore**: Other changes (dependencies, etc.)

### Examples

```bash
# Feature
git commit -m "feat(model): add Vision Transformer architecture"

# Bug fix
git commit -m "fix(tokenizer): resolve memory leak in BPE encoding"

# Performance
git commit -m "perf(simd): optimize matrix multiplication with AVX2"

# Documentation
git commit -m "docs(readme): update installation instructions"

# Breaking change
git commit -m "feat(api)!: redesign Lua API for better performance

BREAKING CHANGE: model.create() now requires explicit config parameter"
```

## Release Checklist

Before creating a release:

- [ ] All features merged to `develop`
- [ ] All tests passing
- [ ] Code compiles without warnings
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in relevant files
- [ ] Performance benchmarks run
- [ ] Memory leaks checked
- [ ] Security audit completed (for major releases)

## Branch Protection Rules

### `main` branch
- ✅ Require pull request reviews
- ✅ Require status checks to pass
- ✅ Require signed commits (recommended)
- ❌ No direct pushes allowed
- ✅ Only via release/hotfix merge

### `develop` branch
- ✅ Require pull request reviews (recommended)
- ✅ Require status checks to pass
- ⚠️ Direct pushes allowed (for small fixes)
- ✅ Feature merges via git-flow

## Tagging Strategy

Tags are created automatically during release/hotfix finish:

```bash
# Version tags
v1.0.0, v1.1.0, v2.0.0

# Pre-release tags
v1.1.0-alpha.1
v1.1.0-beta.1
v1.1.0-rc.1
```

To create a pre-release:

```bash
git tag -a v1.1.0-beta.1 -m "Beta release 1.1.0"
git push origin v1.1.0-beta.1
```

## Quick Reference

```bash
# Initialize git-flow (already done)
git flow init

# Feature workflow
git flow feature start NAME
git flow feature finish NAME

# Release workflow
git flow release start VERSION
git flow release finish VERSION

# Hotfix workflow
git flow hotfix start VERSION
git flow hotfix finish VERSION

# View current state
git flow status

# List features/releases/hotfixes
git flow feature list
git flow release list
git flow hotfix list
```

## Resources

- [Git Flow Cheat Sheet](https://danielkummer.github.io/git-flow-cheatsheet/)
- [A Successful Git Branching Model](https://nvie.com/posts/a-successful-git-branching-model/)
- [Semantic Versioning](https://semver.org/)
- [Conventional Commits](https://www.conventionalcommits.org/)

---

**Current Version:** 1.0.0  
**Last Updated:** November 24, 2025
