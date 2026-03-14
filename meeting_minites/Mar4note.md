# Meeting Notes - March 4

## Three ways how we can manipulate model behaviour via changing parametres

1. **Manipulating skill prior**  
   High prior → faster box opening, skill expectation stays high for longer, fewer repeated box opening attempts

2. **Manipulating prior for the correct rule**  
   (Number of shapes on the box)  
   High prior → the correct hypothesis discovered quickly

3. **Manipulating prior on the generator (random SoC)**  
   Higher prior → more random to begin with, less explicit rule testing


---

## Visually comparing models and children

Plot a histogram of **Number of Attempts to open all boxes**  
(for children who opened all boxes)

Plot a histogram of **Number of Attempts needed to open all boxes by the model**

**On the same plot**

A model run needs to produce **a log of opening attempts**

### Comparison Methods

1. **The simplest comparison — histograms**

2. **For each child**  
   On trial **N**, examine how many boxes were opened.  
   This is the variable we can fit.

If we run the same model for multiple runs, we can estimate the probability that:

```
(1,2,3,4,5) boxes will be opened on trial n
```

---

## Likelihood Method (Method 3 – Harder)

The likelihood that the child is explained by the given model is the likelihood that the same number of boxes was opened.

On each trial that the model simulates, we would need to produce **a distribution of probabilities across each key–box combination**.

Then computing a likelihood for a child:

```
P(childs_data | model_i)
```

entails:

For a given child:

1. Instantiate the **model_i**
2. Produce a distribution of probabilities over **ALL possible (k, b) combinations**

Then the likelihood on the first trial is the probability assigned to the child’s actual attempt by that model.

```
P(child_action_1 | model_i)
```

is the probability that the model assigns to that action on the first trial.

---

## The Sampling Hypothesis

Tell the model to open the **combination that the child has opened**

As long as the probability of that combination is **not zero**

Let the model **update beliefs after the child’s evidence is given**

In the end, select the model with the **Maximal Posterior Likelihood**

**MAP**

(Not ideal, but once we implement this we will consider cross-validation)

---

## Example

Prior to:

```
(9,1)
```

Expectation would be:

```
0.9
```

Then:

```
P(red, red, true) = 0.9
```

---

## Parameter Estimation

For **each child**, and **each model lesion**, we want to estimate the **best fitting parameters**.

In the paper, we would like to have **a bar plot showing the log likelihoods for the 4 model lesions**.

The log likelihoods used for these bars are the log likelihood for **each child and each model**, assuming **MAP parametrization of each model for each child**.

---

## Two Main Model Features

1. **Generator in the proposal**
2. **Skill**

---

## Lesioned Models

### Lesion (1)

No generator — only hardcoded rules in the proposal.

No skill learning.

Expectation of theta:

```
1
```

Always.

Possible approximation:

```
theta prior = (1000000, 1)
```

---

### Lesion (2)

Skill learning allowed, but **no generator**

---

### Lesion (3)

Generator allowed, but **no skill learning**

---

### Lesion (4)

**Full model**

Both generator **and** skill learning included.

---

## Example Hypothesis

```python
def h1(k,b):
    return true
```

---

## Parameter Grid

5 levels of **skill prior**

5 levels of **generator probability**

Grid size:

```
25
```

---

## Target Conferences

### CCN (Computational Cognitive Neuroscience)

https://2026.ccneuro.org/

**April 2** — CCN abstract submission

Conference in **New York in August**

---

### NeurIPS

Submission deadline: **early May**

---

# Action Items — March 4

### Farzin

Run **LLM REACT** and **Jeffrey's LLM model with DeepSeek**

Keep **API receipts**

---

### Zhuangfei

SoC model fitting

1. Method 1
2. Method 2
3. Method 3

---

### Jeffrey

Basic **LLM baseline for generating programmatic hypotheses as particles**

If task-to-program approaches do not work:

Try first generating hypotheses in **natural language**, then ask the LLM to **translate them to code**

Reference:

https://arxiv.org/pdf/2309.05660