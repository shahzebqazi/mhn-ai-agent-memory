# Experiment Results

Results from inductive capacity experiments are stored here as per-N markdown files.

## File Naming

Each inductive step produces a file: `runs/n<N>.md`

Example: `runs/n2.md`, `runs/n3.md`, `runs/n4.md`

## Entry Format

Each file should contain:

```markdown
# N=<N> Capacity Experiment

**Date:** YYYY-MM-DD
**Agent:** [model name or "researcher"]

## Results

| Patterns Attempted | Patterns Recalled | Accuracy | Failure Mode |
|---|---|---|---|
| 1 | 1 | 1.00 | none |
| 2 | 1 | 0.50 | interference |
| ... | ... | ... | ... |

## Max Reliable Recall

<number> patterns at 100% accuracy

## Observations

<what changed from N-1; mathematical notes; anything surprising>
```

## Summary Table

Agents should maintain a running summary here as steps are completed:

| N | Max Patterns (100% recall) | Theoretical Limit (0.138N) | Notes |
|---|---|---|---|
| 2 | (pending) | 0.28 | |
| 3 | (pending) | 0.41 | |
| 4 | (pending) | 0.55 | |
| ... | ... | ... | |
