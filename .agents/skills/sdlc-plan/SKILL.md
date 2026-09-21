---
name: sdlc-plan
description: Stages 1 to 3 of the SDLC. Check the change against intent.md, write the new behavior into spec.md, and add plan.md items that name their files and tests, then get the operator's approval before any code. Use when a feature, bugfix or refactor is requested and the spec or an approved plan item does not exist yet.
---

# Stages 1 to 3: intent, spec, plan

Goal: agree on what "done" means before touching code. The output is edits to `intent.md` (only if needed), `spec.md` and
`plan.md`. Do not edit `prism/` or `tests/` in this phase.

## Steps

1. **Restate the request** in one or two sentences. If it is ambiguous in a way that changes the design, ask one focused question
   now. Do not guess, and do not ask what the code can answer.
2. **Explore.** Read the code that owns the behavior (`prism/`), its tests (`tests/`), the doc that describes it (`docs/*.md`),
   the relevant part of `spec.md`, and `CHANGELOG.md`. Read files, do not skim names. For a wide search use a read-only
   exploration subagent if the harness has one.
3. **Stage 1, intent.** Does the change fit `intent.md` (problem, outcome, constraints, non-goals)? If it contradicts it, stop:
   propose the edit to `intent.md` and get the operator's approval before continuing. Changing an invariant (I1 to I8 in
   `spec.md`) is the same kind of change.
4. **Stage 2, spec.** Write the behavior into `spec.md`: what must stay true, the failure modes and their HTTP status or exit
   code, the environment variables, and which normative doc changes (`docs/api.md`, `docs/cli.md`). Planned behavior goes under
   "Planned behavior" until it is built. Prefer pointing at a normative doc over repeating its tables.
5. **Stage 3, plan.** Add `- [ ]` items to `plan.md` under a phase (create the phase if needed). One item is one commit-sized
   step. Each names the files it touches and the test that proves it. Add items for docs and the `CHANGELOG.md` entry when the
   change is user-visible. Put risks and open questions in the phase, especially hardware behavior that a hermetic test cannot
   cover and anything near invariant I1.
6. **Check the project rules** in `AGENTS.md` against the plan: standard library only in the runtime, hermetic tests, docs and
   changelog included.
7. **Present the diff** of the artifacts (not the whole exploration) and ask for approval. Do not start implementing in the same
   turn.
8. On approval add `Status: approved by the operator, not started.` under the phase heading in `plan.md`.

## Exit criterion

`spec.md` describes the behavior, every plan item names its files and its test, and the operator approved. If the scope changes,
edit the artifacts; do not just remember it.
