# Meeting Notes — March 18

## Action Items (from Marta Kryven, Slack)

---

## 1. SMC-S (Jeffrey)

### Prior Simplification
- Simplify the SMC-S prior to aid debugging

Define:
- `g`: fraction of prior allocated to the generator (0 < g < 1)
- `t`: fraction of prior allocated to the true prior (0 < t < 1 - g)
- Remaining prior mass is divided **equally among other hypotheses**

Requirement:
- Log priors on each run to track allocation

---

### Test Cases

#### Test Case 1 (should open in few attempts)
1. Large theta (19,1), relatively large prior for true rule (number) 0.2, g = 0.8  
2. Large theta (19,1), relatively large prior for true rule (number) 0.2, g = 0.1  

#### Test Case 2 (should need more attempts)
1. Large theta (19,1), relatively small prior for true rule 0.001, g = 0.8  
2. Large theta (19,1), relatively small prior for true rule 0.001, g = 0.2  

#### Test Case 3 (random sampler)
- Large theta (19,1), tiny prior for true rule 0.001, g = 0.9  

#### Test Case 4 (effect of theta)
1. Small theta (1,1), relatively large prior for number 0.2, g = 0.8  
2. Small theta (1,1), relatively large prior for number 0.2, g = 0.1  

#### Test Case 5 (likely poor performance)
- Small theta (1,1), tiny prior for true rule 0.001, g = 0.9  

---

### Additional
- It may be interesting to test performance with **fewer particles**

---

## 2. Web App (Farzin)

### Goal
- Human experiment with adults simulating the Box task with **full observability**

### Setup
- Participants see **5 doors**
- Doors share the same:
  - Colour
  - Symbols
  - Numbers  
  as in the Box task

### Data Collection
- Log:
  - Which keys were tried on which boxes
  - Time taken per attempt
  - Number of attempts
  - Demographics

### Generalization Test
- After all boxes are opened:
  - Test whether subjects can select the correct key for a **previously unseen door**
- Details to be discussed later (meeting or Slack)

---

### Implementation Notes
- For now: focus on **logic, not visual quality**
- Need to:
  - Draw 4 doors
  - Draw icons for key buttons underneath

### Suggested Tools
- Keynote (create + screenshot)
- Inkscape
- GIMP
- https://app.diagrams.net

If graphics are an issue:
- Another RA can assist

---

### Hosting & Backend
- Create a **Google Cloud account** (1-year free hosting)

- Use provided starter code:
  - JavaScript + PHP (~2016)
  - Includes:
    - Saving data on server
    - Basic experiment workflow

- You may use other libraries if needed, but:
  - Core logic must work
  - Must save user data properly

---

## 3. Other LLMs (Zhuangfei)

### Observation
- A one-particle LLM model can solve the Box task quickly when using GPT-5.1 backend
- This suggests GPT-5.1 no longer requires SMC-S
- Reason: GPT-5.1 is optimized for code synthesis

### Constraints
- GPT-5.1 API is:
  - Slow
  - Expensive

### Immediate Task
- Run at least **10 trials with GPT-5.1**
- Produce:
  - Histogram
  - Performance demonstration  
- (Jeffrey will handle this)

---

### Paper Direction
Compare LLM-based hypothesis discovery across:
- GPT-4.1  
- GPT-5.1  
- DeepSeek family (Zhunangfei picks deepseek chat, very cheap around 1/4 cost of qwen plus)
- Qwen (latest version, Zhuangfei picks qwen plus $0.115 input / $0.287 output per 1M tokens)
- Possibly LLaMA  

---

### Next Step
- Run Jeffrey’s **one-particle LLM model with Qwen**
- Begin experimentation

---

## References / Links

- Meeting notes archive (Google Doc):  
  https://docs.google.com/document/d/1t9y-OtLh541OqYkhd3z9xZNHsjRSqjBJlTnVGR1s0DI/edit?usp=sharing

- Qwen API access (Alibaba Cloud):  
  https://www.alibabacloud.com/help/en/model-studio/models#4c74a7d1841hr

- LLaMA API alternative:  
  https://www.together.ai/

- Diagram tool:  
  https://app.diagrams.net