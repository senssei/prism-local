---
name: sdlc-implement
description: Stages 4 and 5 of the SDLC. For each approved plan.md item, write a hermetic test and prove it fails for the right reason (sdlc_check.py --red), make the smallest change that turns it green, run the gate, and tick the box. Use when a plan item is approved, or when the user says to implement or continue implementing.
---

# Stages 4 and 5: test, code

Precondition: the item is in `plan.md`, `spec.md` describes its behavior, and the operator approved the plan. If not, go back to
`sdlc-plan`. A **bug fix** may start here: reproduce it with a failing test first, and update `spec.md` if the intended behavior
was undefined.

## Loop, once per plan item

1. Take the first unticked item. Put a one-line note in the phase's `Status:` in `plan.md` (what you are on).
2. **Stage 4, test first.** Write or extend a hermetic test (`tests/fakes.py`, temp dirs, `create_server(port=0)`; no GPU,
   network or Ollama). Then prove it is red:
   ```bash
   python3 scripts/sdlc_check.py --red tests.test_module.TestClass.test_name [more ids...]
   ```
   It exits 0 only if every named test **fails or errors** now, and prints the exception line of each failure (test output is
   swallowed, each test has a 60 s limit). Read those lines: the reason must be the missing behavior (`AttributeError: run_red`, a
   wrong value), not a typo in the test. It exits 1 for a test that already passes (proves nothing), is skipped or
   `expectedFailure`, times out, or does not exist. Fix the test until it is red for the right reason.
3. **Stage 5, code.** Make the smallest change that passes. Match the surrounding style, naming and comment density. Do not
   refactor unrelated code and do not add runtime dependencies.
4. Run the tests you wrote, then the whole gate: `python3 scripts/sdlc_check.py`. Fix regressions before moving on.
5. **Tick the box** in `plan.md` only now, when the gate exited 0. Update `Status:` in one line.
6. Commit only if the operator asked. When they do: one logical change per commit, `type: description` as in `git log`.

## Project-specific traps

- Anything that may change which hardware runs the model must be **reported**, never silently fallen back (invariant I1). Test it.
- New CLI subcommand goes in `docs/cli.md`, new route in `docs/api.md`; `tests/test_docs.py` fails otherwise.
- Do not weaken or delete a failing test to get green. If a test is wrong, say so and fix it in its own step.
- If the spec or plan turns out to be wrong, stop, fix the artifact, and tell the operator. Do not silently change scope.
- On the reference workstation never run two model loads at once without the resource guard (`plan.md` Phase 1).

## Exit criterion

All items of the change are ticked, the gate exited 0 in this session, and `git diff` contains nothing that is not in the plan.
Continue with `sdlc-review`.
