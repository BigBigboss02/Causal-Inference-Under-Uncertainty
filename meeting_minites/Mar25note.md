# Meeting Notes - March 25

## Action Items

### 1. Plots for CCN Abstract

Create a figure with **8 histograms** comparing model performance vs children.

**Row 1: LLM-based models**
- Qwen (one-particle LLM)
- GPT-5.1 (one-particle LLM)
- REACT (Qwen)
- REACT (GPT-5.1)

**Row 2: SMC-S (4 panels)**
- (1) High skill + very low randomness  
  → model performs very well (lesioned randomness & skill)
- (2) Lesioned skill + moderate randomness (0.5–0.7)
- (3) Lower skill + low randomness
- (4) Both skill and randomness vary  
  → aim to match children’s distribution

**Note:**
- Fix true prior to a small value (e.g. 0.05)
- Rationale: children rarely discover the true rule, but it remains possible

---

### 2. Parameter Fitting

- Fit **randomness** and **skill** parameters using **MAP estimation**
- Target: match the distribution of children’s attempts
- Apply to each model variant

---

### 3. One-Particle LLM Experiment (Jeffrey)

**Modify rule-testing function:**
- Introduce probabilistic success for correct rule:
  - e.g. 0.95, 0.9, 0.8, 0.5

**Experiment:**
- Plot histogram over **10 trials**
- Measure number of attempts needed

**If performance collapses:**
- Add prompt:
  > "Remember that the keys and boxes are physical objects, and sometimes the correct key might fail to open the box due to a mechanical failure"