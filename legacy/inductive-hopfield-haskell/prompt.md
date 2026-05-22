# Agent Operating Contract: Inductive Hopfield Networks

You are an AI coding agent working on the inductive Hopfield subproject.
This file is your standing instruction set. Read it fully before writing
any code or making any edits.

---

## 0. Governance

Before any substantive work, read and follow:

- **`PLAYBOOK.md`** (repo root) — workspace-wide agent rules, quality gates,
  falsifiability requirements, glossary/notebook/literature protocols.
- **This file** — subproject-specific workflow.

All PLAYBOOK rules apply here. If anything in this file conflicts with
PLAYBOOK.md, the PLAYBOOK wins.

---

## 1. Startup Checks (Every Session)

Run these checks at the start of every working session. If any check fails,
stop and report the failure to the researcher before proceeding.

### 1.1 Toolchain

```
ghc --version          # Must be GHC 9.6+ (or whatever resolver requires)
cabal --version        # Must be cabal-install 3.10+
cabal build all        # Must compile cleanly (if code exists)
cabal test             # Must pass (if tests exist)
```

If `ghc` or `cabal` are not found, do NOT attempt to install them silently.
Report the missing tool and stop.

### 1.2 Formatter and Linter (Optional but Recommended)

```
fourmolu --version     # Haskell formatter
hlint --version        # Haskell linter
```

If these are not available, proceed but note the gap in your lab notebook entry.
Do not let missing formatters block implementation.

### 1.3 Research Access

Verify you can retrieve information from at least one of these paths:

- Built-in web search / fetch tools
- MCP servers for arXiv, Semantic Scholar, or Haskell package docs
- Files already present in the repo (literature matrix, glossary, READMEs)

If you cannot access any external research source and the information you need
is not already in the repo, stop and report. Do NOT fabricate citations, API
signatures, or mathematical results from memory alone.

### 1.4 Touch Set Declaration

Before editing any file, declare your intended touch set in your first message
to the researcher and in your lab notebook entry. Follow the async coordination
rules in PLAYBOOK.md Section 2.8.

---

## 2. The Inductive Loop

This subproject builds Hopfield networks one neuron count at a time. The loop
for each value of N is:

### Step 1: Implement

Write or extend the Haskell modules to handle the current N:

- `src/Hopfield/Classical/Types.hs` — types for N-neuron network
- `src/Hopfield/Classical/Hebbian.hs` — Hebbian weight imprinting
- `src/Hopfield/Classical/Update.hs` — asynchronous sign-rule update
- `src/Hopfield/Classical/Energy.hs` — energy function computation
- `src/Hopfield/Classical/Capacity.hs` — capacity measurement utilities

For the first iteration (n=2), write the modules from scratch with explicit,
inspectable types and pure functions. Do NOT use matrix libraries yet; use
lists or small vectors so the equations are visible in the code.

For subsequent iterations, generalize the existing code. Track what changes
between n and n+1 in a comment block at the top of the relevant module.

### Step 2: Test

Write or extend tests in `test/Hopfield/`:

- `ClassicalSpec.hs` — unit tests: store a pattern, retrieve it, verify match
- `InductiveSpec.hs` — verify that the n+1 network is a valid extension of n
- `CapacitySpec.hs` — store increasing numbers of random patterns, measure
  recall accuracy

Run:
```
cabal test --test-show-details=direct
```

All tests must pass before proceeding.

### Step 3: Measure

Run capacity experiments for the current N:

- Store 1 pattern. Retrieve it. Record success/failure.
- Store 2 patterns. Retrieve each. Record.
- Continue until recall fails or interference dominates.
- Record: N, patterns_stored, patterns_recalled, failure_mode, notes.

Append results to `experiments/runs/n<N>.md` (create if needed).

### Step 4: Reflect

Write a brief summary of what changed mathematically from the previous step:

- How did the weight matrix structure change?
- Did the energy landscape change qualitatively?
- Is the code for this step essentially the same pattern as the last?
- Are you seeing diminishing conceptual returns?

Include this summary in your lab notebook entry.

### Step 5: Decide

If the classical scaling is still producing structural insight, advance to n+1
and repeat from Step 1.

If the classical regime has clearly plateaued (see transition criteria in
README.md), stop the classical loop, freeze the baseline, document the limit,
and begin work under `src/Hopfield/Modern/`.

### Step 6: Commit

After each completed inductive step, commit with a message of the form:
```
hopfield(classical): implement and test n=<N>, capacity=<max patterns recalled>
```

---

## 3. Classical-to-Modern Transition

The transition gate is passed when ALL of these hold:

- Empirical storage capacity is confirmed near ~0.138N (Amit et al., 1985).
- Added neurons yield only incremental memory, not conceptual insight.
- The n+1 code path is essentially mechanical repetition.
- Recall failures are dominated by interference, not implementation bugs.

When the gate is passed:

1. Add a final experiment entry documenting the classical ceiling.
2. Update `experiments/README.md` with a summary table of all classical runs.
3. Begin implementing `src/Hopfield/Modern/` starting with the continuous
   update rule: x_new = softmax(beta * X^T * xi).
4. The modern phase goal is an explicit storage/retrieval API that agents can
   test as associative memory, not markdown scratch files.

---

## 4. Modern Hopfield Phase

### 4.1 Implementation Targets

- `src/Hopfield/Modern/Types.hs` — continuous state vectors, stored pattern matrix
- `src/Hopfield/Modern/Update.hs` — softmax update rule with inverse temperature beta
- `src/Hopfield/Modern/Energy.hs` — modern energy function (log-sum-exp form)
- `src/Hopfield/Modern/Memory.hs` — store/retrieve API for structured facts

### 4.2 Testing

- Store and retrieve toy symbolic patterns.
- Measure capacity: how many patterns before retrieval degrades?
- Compare empirically against the classical baseline at the same N.
- Test whether an AI agent can use the Memory API to store and recall facts
  during a coding session. Record what works and what fails.

### 4.3 Transition to Equation Discovery

Once the modern implementation is working, the next goal is to identify or
derive the compact equation for modern Hopfield networks. Search for:

- Existing closed-form derivations in the literature (Ramsauer et al., 2021;
  Krotov & Hopfield, 2016).
- Whether the inductive pattern from the classical phase generalizes.
- Whether a proof of the modern update rule's correctness can be expressed
  as a Haskell type or property test.

If a compact form exists in the literature, cite it. If not, document the
gap and attempt to derive or disprove it. Do NOT claim a novel result without
independent verification.

---

## 5. Quality Gates (Per Session)

Before ending any working session, verify:

- [ ] All new references are in `02_LITERATURE_MATRIX.md`
- [ ] All new technical terms are in `GLOSSARY.md`
- [ ] All decisions are recorded in `01_LAB_NOTEBOOK.md`
- [ ] No unfalsifiable claims introduced
- [ ] All tests pass (`cabal test`)
- [ ] Experiment results are recorded in `experiments/runs/`
- [ ] Touch set matches what was actually edited
- [ ] Commit made with descriptive message

---

## 6. What NOT To Do

- Do NOT skip straight to large N. The inductive structure is the point.
- Do NOT use matrix libraries for the first classical steps. Visibility first.
- Do NOT fabricate mathematical results or citations.
- Do NOT modify files outside your declared touch set without approval.
- Do NOT treat experiment failures as bugs to hide. Capacity limits and
  interference patterns are first-class findings.
- Do NOT generate unfalsifiable claims about Hopfield network properties.
  If a property holds, test it. If it fails, document the failure.
