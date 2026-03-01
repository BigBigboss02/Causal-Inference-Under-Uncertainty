# Model Fitting Plan – Notes from Marta

## 1. Model Variants (Lesioned + Full)

We will fit simplified (lesioned) versions before the full model:

1. **No Generator**
   - Proposal contains only hardcoded hypotheses.
   - Tests whether hypothesis generation is necessary.

2. **No Skill**
   - Removes the key skill parameter.
   - Behaviour explained only through priors.

3. **No Skill + No Generator**
   - Minimal baseline model.
   - Pure prior-driven rule competition.

4. **Full Model**
   - Includes all parameters below.

The aim is to identify which components are necessary.

---

## 2. Three Core Parameters

The three parameters to consider are:

### 2.1 Key Skill (θ)

- Modelled as a **Beta distribution**.
- Controls generalisation.
- Central parameter.
- Fitted individually per participant.

Interpretation:
Represents learning efficiency / rule extraction strength.

---

### 2.2 Prior for the True Rule

- Initial prior probability assigned to the correct rule.
- Influences early learning dynamics.
- Affects how quickly the true rule dominates.

---

### 2.3 Prior on Colour-Matching Rules

- Prior probability mass placed on colour-matching rules.
- Captures bias toward perceptual shortcut strategies.
- Models systematic suboptimal reasoning.

---

## 3. Fitting Strategy

We increase complexity gradually.

### Step 1: One-Parameter Model
- Fit **skill only**.
- Inspect likelihood surface per participant.
- Check identifiability.

### Step 2: Two-Parameter Model
- Skill + one prior.
- Use **grid search**.
- Compute **MAP estimate**.
- Inspect likelihood surface.

### Step 3: Three-Parameter Model
- Skill + true rule prior + colour prior.
- Grid search no longer feasible.
- Use **random search + local refinement**.

---

## 4. Likelihood Surface Analysis

Before increasing dimensionality:
- Visualise likelihood surfaces.
- Check for:
  - Flat regions
  - Multi-modality
  - Parameter trade-offs
  - Identifiability issues

Understanding the surface is essential before adding complexity.

---

## 5. Implementation Principle

- Hypotheses must be implemented as **executable programs**.
- Natural language may elicit hypotheses.
- Validation must occur through code execution.
- World modelling should use programmatic structures.

The LLM component should demonstrate hypotheses as code, not merely describe them.