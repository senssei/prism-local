# Review policy (`REVIEW.md`)

## 1. Who decides

Agents propose; the **operator decides** what merges and what ships. A change that passes every gate is not done until it has been
reviewed, and the review is never done by the session that wrote the change.

```text
Plan approved -> test red -> code green -> gate exit 0 -> independent review -> operator decision -> commit / release
```

## 2. Independent review

Review with fresh context: a separate subagent or a new session that is given only `intent.md`, `spec.md`, the `plan.md` items
under review, this file, and the diff (`git diff $(git merge-base main HEAD)` plus untracked files). It reports findings ranked by
severity, each with `file:line` and a concrete failing input, and no praise. The author fixes findings test-first; the reviewer
does not edit code.

## 3. Checklist

### A. Evidence
- [ ] `python3 scripts/sdlc_check.py` exited `0` in this session (compile, tests, changelog, docs).
- [ ] Every new or changed behavior has a test that was seen failing first (`sdlc_check.py --red`).
- [ ] No test was weakened, skipped or deleted to get green.

### B. Spec
- [ ] The behavior is written in `spec.md` (or the normative doc it points to) and the code matches it.
- [ ] No invariant I1 to I8 is broken. Touching one needs operator approval.
- [ ] A silent hardware fallback is a defect (I1).

### C. Patch
- [ ] The diff contains only what the plan item covers; no drive-by refactors.
- [ ] No new runtime dependency (I4); optional engines stay lazy.
- [ ] Errors keep the JSON error shape and status codes of `docs/api.md`.

### D. Security and cleanliness
- [ ] No injection into shell or paths, no path traversal in model or directory handling, no secrets in logs (`docs/security.md`).
- [ ] Defaults stay safe (I5).

### E. Docs
- [ ] `docs/` updated where user-visible; `CHANGELOG.md` has an entry under `[Unreleased]`.

## 4. Operator gates

Stop and ask, every time, before: changing `intent.md`; changing an invariant; a version bump, tag, PyPI or GitHub release; pushing;
opening or commenting on an issue or PR; anything outside the repository. Approval for one of these does not carry over to the next.

## 5. Commands

```bash
python3 scripts/sdlc_check.py                        # the gate
python3 scripts/sdlc_check.py --red tests.test_x.TestY.test_z
git diff $(git merge-base main HEAD)                 # what the reviewer sees
git config core.hooksPath .githooks                  # opt in to the pre-commit gate
```
