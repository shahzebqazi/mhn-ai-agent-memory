# Glossary — self-reconfiguring / adaptive models

**Conditional computation**  
Using only a subset of parameters or subnetworks for each input (routing, gating, sparse experts).

**Mixture of experts (MoE)**  
Several parallel “expert” modules (often FFN copies); a router sends each token or sample to one or a few experts so **total parameters** can exceed **active FLOPs**.

**Sparse activation**  
Large parameter count but small fraction executed per forward pass (MoE, sparse attention patterns).

**Dense model**  
Standard transformer: all layer parameters participate in each forward pass for that path (no expert skipping).

**Router / gating**  
Learned or heuristic mechanism that decides which experts or paths run for a given input.

**Dynamic neural network**  
At inference, the effective graph or parameter use **depends on the input** (instance-wise, spatial, or temporal adaptivity). See Han et al., *Dynamic Neural Networks: A Survey* (arXiv:2102.04906).

**Dynamic sparse attention**  
Attention patterns that are sparse and **input-dependent** (e.g. Dynamic Sparse Attention for transformers, arXiv:2110.11299).

**MoEfication / dense → MoE conversion**  
Post-hoc or training-time replacement of dense FFN blocks with MoE layers (e.g. D2DMoE, arXiv:2310.04361).

**Dynamic-k routing**  
Selecting a **variable number** of experts per token (vs fixed top-k).

**Symbolic-MoE (heterogeneous)**  
Routing among **separate pretrained LLMs** by skills/tags, then aggregating outputs (arXiv:2503.05641) — modular at the **system** level, not only inside one weight tensor.

**Adaptive computation time (ACT)**  
Halting or varying **depth/steps** per input (Graves, arXiv:1603.08983). Related: Universal Transformer, PonderNet (arXiv:2107.05407).

**AdaTape**  
Adaptive **elastic input sequence**: variable extra “tape” tokens per sample (arXiv:2301.13195); combines adaptive function type and compute budget.

**Early exit / dynamic depth**  
Stop forward pass when an internal classifier is confident enough; depth varies by input difficulty.

**Inductive bias (adaptive compute)**  
Letting easy inputs use less compute can match task structure (e.g. hierarchical depth in arithmetic).
