---
name: sdlc-release
description: Ship step of the SDLC. Run the full gate, finish the changelog, draft the PR description and, only when explicitly asked, commit, push, open the PR or publish a release. Use when review is done and the user wants to commit, open a PR, or cut a release.
disable-model-invocation: true
---

# Ship

Precondition: independent review finished with no open finding. This step has outward-facing actions, so each one needs the
operator's explicit ask **in this turn**; an earlier "yes" does not carry over (`REVIEW.md` section 4).

## 1. Gate (always)

```bash
python3 scripts/sdlc_check.py --base main
```

All of `compile`, `tests`, `changelog` and `docs` must be `PASS` (`docs` may be `SKIP` when MkDocs is not installed; say so). On
`FAIL`, fix the cause and rerun. Never edit the script or skip a check to get green. Report the actual output.

## 2. Plan, changelog, version

- The plan items are ticked and the phase `Status:` line is current.
- User-visible changes have an entry under `## [Unreleased]` in `CHANGELOG.md` (Added / Changed / Fixed), written for users.
- A version bump (`prism/__init__.py` `__version__`) and moving `[Unreleased]` under a dated heading happen **only** when the
  operator is cutting a release.

## 3. PR description

Fill `.github/PULL_REQUEST_TEMPLATE.md`: *What and why* from the plan item and `spec.md`, *How it was tested* from the gate output
and the tests added. Show it to the operator. Create the PR (`gh pr create`) only if asked.

## 4. Commit and push (only if asked)

One logical change per commit, `type: description` as in `git log`. Never force-push, never push to `main` directly, never skip
hooks. If commits are GPG-signed and the agent waits for a passphrase, stop and ask the operator to unlock it; do not disable
signing.

## 5. Publishing (only if asked, step by step)

Follow `docs/development.md#releasing`: TestPyPI first, then a signed tag `v<__version__>`, then PyPI. PyPI versions are permanent.
Stop and ask before each of: pushing a tag, running the Publish workflow, creating a GitHub release. Issues and comments on GitHub
are outward actions as well.

## Exit criterion

Gate green in this session, and every outward step the operator asked for is done and reported.
