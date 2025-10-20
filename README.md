# TITAN: Trajectory-Informed Freezing for Large-Scale VQE

TITAN is a learning-to-freeze framework that **predicts and manages parameter freezing** in Variational Quantum Eigensolvers (VQE). It combines a lightweight encoder with an **Adaptive Parameter Freezing & Activation (APFA)** mechanism that tracks gradient statistics (EMA) and toggles parameters between *active* and *frozen* using dynamic thresholds—cutting shots and classical optimization time while preserving solution quality. :contentReference[oaicite:0]{index=0}

> **At a glance**
> - **APFA state machine** with per-parameter EMA gradients, global magnitude signals, and patience counters for freeze/reactivate. :contentReference[oaicite:1]{index=1}  
> - **Orthogonal to existing VQE improvements** (measurement grouping/shadows, ansatz design, advanced optimizers). :contentReference[oaicite:2]{index=2}  
> - **Layer-wise training option (LGGD)** and Gaussian initialization supported. :contentReference[oaicite:3]{index=3}  
> - Demonstrated on **HEA isotropic/anisotropic Hamiltonians** and **molecular benchmarks** (e.g., LiH, H₂O). :contentReference[oaicite:4]{index=4}

---

## Table of Contents
- [What is TITAN?](#what-is-titan)
- [Key Ideas](#key-ideas)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Reproduce Paper-style Setups](#reproduce-paper-style-setups)
- [Project Layout](#project-layout)
- [Configuration](#configuration)
- [Visualizations](#visualizations)
- [Limitations](#limitations)
- [Citing](#citing)
- [License](#license)
- [Contact](#contact)

---

## What is TITAN?

TITAN proactively **shrinks the VQE search space** by predicting which parameters will remain low-saliency and freezing them early, then **re-activating** if their saliency rises. This reduces both **trainable dimension** and **measurement overhead**, and composes cleanly with other techniques (e.g., shadow-based measurement allocation, natural-gradient optimizers). See the **APFA architecture diagram** (Appendix, *p.3*, Fig. 1) for the freeze/activate state machine. :contentReference[oaicite:5]{index=5}

---

## Key Ideas

- **APFA**: Maintain an **EMA of |grad|** per parameter; compute global magnitude and a decay ratio to **scale freeze/activate thresholds**; use **patience counters** to change states. Only **active** parameters are updated (Hadamard-masked gradient step). (*Appendix B*, Eqs. (2)–(6)). :contentReference[oaicite:6]{index=6}  
- **Orthogonality**: TITAN complements (i) measurement-cost mitigation, (ii) ansatz design/compression, (iii) advanced classical optimizers. (*Appendix A*). :contentReference[oaicite:7]{index=7}  
- **Layer-wise option (LGGD)**: Deterministic depth growth with exact gradients for the active prefix; reduced per-iteration cost. (*Appendix E.1*). :contentReference[oaicite:8]{index=8}  
- **Benchmarks**: HEA on isotropic (XX+YY+ZZ) and anisotropic models; **molecular LiH** and others with second-quantized Hamiltonians and Givens rotations. (*Appendix F*). :contentReference[oaicite:9]{index=9}

---

## Installation

```bash
# 1) Create env (Python 3.10+ recommended)
conda create -n titan python=3.10 -y
conda activate titan

# 2) Install dependencies
pip install -U pip wheel setuptools
pip install numpy scipy matplotlib pandas pyyaml tqdm
pip install torch torchvision  # for the ResNet18+2D attention backbone
pip install pennylane qiskit

# (optional) dev tools
pip install pre-commit black isort mypy
