
## Hypothesis

SE.LLMA's goal of building strong Swedish language models makes generation reliability a relevant target. The paper *Sparks of Weight-Space Programming* argues that neural weights can sometimes be inspected and edited directly, and reports a large improvement on a mathematical benchmark after a small edit to a language model. Its most important methodological distinction is between (i) inferring behavior from weights without running the network and (ii) using forward-pass feedback to guide edits. These are different claims.

I test a narrow hypothesis: **A targeted weight-space edit can reduce repetitive text generation in Swedish without significantly degrading Swedish linguistic quality or English generation.**

## Experiment

The experiment uses Qwen2.5-0.5B-Instruct, a small multilingual causal model, on CPU. 
Eight Swedish development prompts identify repeated 3-token sequences. For each later occurrence, the final hidden state immediately before the repeated sequence is captured. The output row for the sequence's first token is adjusted by a rank-one, activation-conditioned subtraction. The output projection is untied from input embeddings before editing. Four conditions are compared on ten held-out Swedish and five English prompts: Original model, Model with targeted weight edit, Model with random weight edit, Original model with repetition penalty.

## Results

| Condition | Swedish Adherence (↑) | Code-Switch Rate (↓) | Loop Rate (↓) | Distinct 4-gram (↑) | Regression Keyword Acc (↑) |English/Math Acc (↑)|
| :--- | :---: | :---: | :---: | :---: | :---: |:---:|
| baseline | 0.90    | 20% (2/10)|    40% (4/10)|    0.76    |100% (10/10)|    100% (5/5)|
| prompt_control | 1.00|    0% (0/10)    |20% (2/10)    |0.78    |60% (6/10)    |100% (5/5)|
| activation_steering | 0.87|    10% (1/10)    |30% (3/10)    |0.75    |80% (8/10)    |100% (5/5)|
| random_weight_edit | 0.90|    10% (1/10)    |40% (4/10)    |0.71    |100% (10/10)|    100% (5/5)|
| targeted_weight_edit | 0.90|    10% (1/10)    |40% (4/10)    |0.76    |90% (9/10)    |100% (5/5)|




| Condition | Swedish repeated 3-grams (↓) |  Swedish Distinct-2 (↑)| 
| :--- | :---: | :---: | 
| Baseline | 6.30%     | 0.8915|    
| Targeted weight edit  | 6.52% |   0.8915    |
| Random weight edit | 6.52%|    0.8915  |
| Repetition penalty | 1.09%|    0.9809   |



English results:

| Condition | Repeated 3-grams (↓) |  Distinct-2 (↑) |
| :--- | :---: | :---: | 
| Baseline | 0.00%     | 0.9787|    
| Targeted weight edit  | 0.00% |   0.9787    |
| Random weight edit | 0.00%|    0.9787  |
| Repetition penalty | 0.00%|    1.0000   |


The targeted weight edit did not improve Swedish generation. Repetition increased slightly from 6.30% to 6.52%. Therefore, our initial hypothesis is not supported by these results.
The targeted edit did not outperform random editing. Both methods produced identical aggregate metrics. This is important: we have not demonstrated that the model's learned internal behavior was successfully identified and corrected. The inference-time repetition penalty performed substantially better. It reduced repeated 3-grams
from 6.30% to 1.09%, an approximately 83% relative reduction. This suggests that, for this particular model and task, a simple decoding intervention is more effective than our weight-editing method.
No measured English regression. English repetition remained at zero, while Distinct-2 increased slightly. This is encouraging, although it does not establish that English fluency was preserved.
The experiment gives a useful negative result:
• The feedback-informed weight edit did not demonstrate an advantage over the baseline.
• The random-edit control performed identically at the aggregate level.
• A simple inference-time repetition penalty produced a much larger improvement.


## Recommendation

This supports a cautious recommendation: SE.LLMA should not prioritize this particular weightediting technique over simpler decoding methods without stronger evidence. It does not prove that weight-space programming is ineffective in general. The most informative next experiment is to have a frontier LLM propose the edit from a description
of the activations and compare it with this data-derived edit under identical controls, on a real Swedish benchmark (for example EuroEval) and with a LoRA baseline.


## References 

1. Takehi et al. (2026), Sparks of Weight-Space Programming, https://rikiyat.github.io/blog/weight-spaceprogramming/
2. Qwen model card, https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
