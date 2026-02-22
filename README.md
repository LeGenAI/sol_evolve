# 🚀 SolEvolve
**LLM-Driven Evolutionary Search for SAT-Verified Constructions in Discrete Mathematics**

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![SAT Solver](https://img.shields.io/badge/SAT-Kissat_4.0.3-orange.svg?style=for-the-badge)](https://github.com/arminbiere/kissat)
[![AI](https://img.shields.io/badge/AI-LangGraph_DeepAgents-purple.svg?style=for-the-badge)](https://langchain.com)

**Discovering Optimal Error-Correcting Codes and Combinatorial Structures using Autonomous LLM Agents and SAT Solvers**

[📖 Overview](#-overview) • [🛠️ Installation](#️-installation) • [🧪 Reproducing Results](#-reproducing-paper-results) • [🤖 Architecture](#-framework-architecture-langgraph-deep-agents) • [📄 Citation](#-citation)

</div>

---

## 📖 Overview

**SolEvolve** is a neuro-symbolic algorithmic discovery framework that combines the semantic reasoning capabilities of Large Language Models (LLMs) with the rigorous verification power of SAT solvers. By closing the loop with an autonomous "Reflector" agent, SolEvolve synthesizes novel SAT formulations to solve long-standing problems in discrete mathematics.

### 🏆 Key Achievements

| Domain | Problem | SolEvolve Result | Details |
| :--- | :--- | :--- | :--- |
| **Binary Linear Codes** | Generator Matrix Search | `[32,14,8]` code found in **16.85s** | Single CPU thread, matching best-known theoretical bounds. |
| **Self-Orthogonal Embeddings** | Shortest Length Minimization | Achieved Gram-rank lower limits | Constructive proofs across $\mathbb{F}_2, \mathbb{F}_3, \mathbb{F}_4$, and $\mathbb{F}_5$. |
| **Generalized Lucas Cubes** | Non-Linear Perfect Partitions | Extended Mollard's boundaries | e.g., verified $\Lambda_{15}(1^{12})$ model via constraint solving. |

---

## 🛠️ Installation

### 1. Python Environment

Clone the repository and install the dependencies (Python 3.12+ recommended):

```bash
git clone https://github.com/LeGenAI/sol_evolve.git
cd sol_evolve
pip3 install -r requirements.txt
```

### 2. SAT Solver (Kissat) Setup

SolEvolve relies heavily on the [Kissat](https://github.com/arminbiere/kissat) SAT solver for its verification engine. You must compile Kissat locally before running the experiments:

```bash
# 1. Clone the Kissat repository into the project directory
git clone https://github.com/arminbiere/kissat.git

# 2. Build Kissat
cd kissat
./configure
make

# 3. Verify the executable is at kissat/build/kissat
cd ..
./kissat/build/kissat --version
```
> **Note:** SolEvolve expects the binary at `./kissat/build/kissat` by default. If your binary is located elsewhere, you can specify its path in the execution scripts.

---

## 🧪 Reproducing Paper Results

To ensure transparency and reproducibility, all scripts and intermediate logs from the paper are provided. 

### Binary Error-Correcting Codes (e.g., [32,14,8])

To reproduce the rapid discovery of optimal binary codes, use the sequential intermediate challenges script. This script utilizes the `engines/code_generator.py` module to iteratively build SAT instances.

```bash
python3 intermediate_challenges.py
```
*   **Interactive Menu**: Select option `m` (Multiple solution search mode) to find distinct generator matrices.
*   **Parameters**: Input `n=32`, `k=14`, `d=8` when prompted.
*   **Result**: The solver will rapidly output solving times and verification metrics.

### Self-Orthogonal Embeddings (Table 1)

The discovery logs and verification matrices achieving the Gram-rank theoretical limit ($t_{\min}$) across $\mathbb{F}_2, \mathbb{F}_3, \mathbb{F}_4$, and $\mathbb{F}_5$ are stored in the `results/` directory.

*   **Ternary BCH [20,7,9]**: `results/ternary_bch/`
*   **Quaternary Hermitian [15,5,6]**: `results/gf4_hermitian/`
*   **Quinary Dot-Product [18,6,8]**: `results/gf5_so/`

Each directory contains the parsed results, log histories, and the explicit matrices confirming the shortest implementations matching the Grassl bounds.

### Generalized Lucas Cubes (Table 2)

The discovery of perfect partitions for complex irregular topologies, such as the $\Lambda_{15}(1^{12})$ model, can be reproduced or reviewed using the shift scripts:

```bash
python3 shift_sat_n15.py
```
*   Unsatisfiability proofs (UNSAT) for limits like $s=4$ and SAT assignments for non-uniform ball sizes are detailed within the script logs and the `results/` folder outputs.

---

## 🤖 Framework Architecture (LangGraph Deep Agents)

Beyond static scripts, SolEvolve provides the complete LLM-driven scaffold that generated these approaches. The multi-agent pipeline connects a **Generator**, **Evolver**, **Verifier**, and **Reflector** communicating over an OpenRouter LLM endpoint.

### Running the Agent Workflow
Set your LLM credentials:
```bash
export OPENROUTER_API_KEY="YOUR_KEY"
# optional: export OPENROUTER_MODEL="anthropic/claude-3.5-sonnet"
```

Run a sample LLM workflow:
```bash
python3 run_demo.py "Goal: outline CNF encoding steps for [32,14,8]."
```
*   **Human-in-the-loop**: Use `python3 hitl_demo.py "..."` to pause and approve agent tool calls manually.
*   **Agents Files**: Check `agents/` for the exact system prompts used in the paper.

---

## 📄 Citation

If you use SolEvolve or its generated SAT formulations, please consider citing our paper:
```bibtex
@article{solevolve2026,
  title={{SolEvolve: LLM-Driven Evolutionary Search for SAT-Verified Constructions in Discrete Mathematics}},
  author={Kim, Jon-Lark and Baek, Jae-Hyun},
  journal={Neurocomputing},
  year={2026}
}
```
