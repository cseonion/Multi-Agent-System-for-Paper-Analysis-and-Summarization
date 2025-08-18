# Pure RL for Better Reasoning in LLMs: What DeepSeek-R1 Teaches Us (and How to Try It)

If you’ve ever wondered whether large language models can “teach themselves” to reason without hand-holding, DeepSeek-R1 is a milestone worth studying. The work shows that pure reinforcement learning (RL)—with simple, rule-based rewards—can unlock strong, generalizable reasoning, and that this reasoning can be distilled into smaller, cheaper models.

In this post, you’ll learn:
- How DeepSeek-R1 uses RL (not supervised fine-tuning) to incentivize chain-of-thought reasoning
- Why group-based RL (GRPO) and rule-based rewards are practical and robust
- How a four-stage pipeline combines RL and SFT to stabilize and scale reasoning
- How to distill reasoning into smaller models
- A minimal Python snippet to prototype a GRPO-style loop with rule-based rewards

## TL;DR: What is DeepSeek-R1?

DeepSeek-R1 is a training framework that:
- Uses pure RL to discover reasoning behavior in a base LLM (DeepSeek-R1-Zero)
- Adds a small cold-start SFT step to improve readability and stability, then continues RL
- Generates a large, high-quality reasoning dataset and distills it into smaller open-source models (no RL needed for the small models)
- Achieves strong results on reasoning, coding, and open-ended benchmarks—often rivaling proprietary systems—while remaining open and reproducible

## Why Reasoning Needs a Different Approach

Most open models lag behind top proprietary systems in complex reasoning—multi-step math, coding with planning, and long-context analysis. Traditional approaches like SFT and heuristic search don’t fully close the gap. DeepSeek-R1 shows that:
- Pure RL can incentivize the model to produce longer, more careful, and self-checking reasoning traces
- Carefully designed rewards and output templates matter more than you might think
- Once discovered, reasoning can be distilled efficiently into smaller models

## The Training Blueprint

### Stage A: Pure RL Reasoning Discovery (DeepSeek-R1-Zero)
- No SFT, no human labels—just RL from scratch
- Group Relative Policy Optimization (GRPO): updates use group-based baselines for stability and cost efficiency
- Rewards:
  - Correctness: e.g., boxed final answers in math; passing test cases for code
  - Format adherence: consistent structure like “Reasoning” then “Final Answer”
- Output template: enforce reasoning first, then answer, making reasoning verifiable

What emerged: longer chains of thought, self-verification, reflection, and strategy exploration—without being explicitly taught.

### Stage B: Enhanced RL with Cold-Start Data (DeepSeek-R1)
- Cold-start SFT with a small, high-quality set of chain-of-thought examples to seed readability and stabilize RL
- Four-stage pipeline:
  1) Cold-Start SFT with curated CoT
  2) RL with dual rewards: accuracy + language consistency
  3) Rejection sampling + SFT: harvest the model’s best reasoning traces and mix with general SFT data (writing, QA, etc.)
  4) Final RL alignment: rule-based rewards for reasoning; reward models for general helpfulness/harmlessness

### Distillation to Smaller Models
- Use DeepSeek-R1 to generate ~800k curated reasoning samples
- Fine-tune smaller open models (e.g., Qwen, Llama) via SFT only
- No RL required at this stage—distillation alone transfers strong reasoning

### Evaluation
- Benchmarks: AIME 2024, MATH-500, MMLU, Codeforces, LiveCodeBench, FRAMES, SimpleQA, and more
- Metrics: pass@k, consensus@64, non-zero-temperature sampling, careful prompting
- Highlights:
  - Pure RL (R1-Zero) boosted AIME 2024 pass@1 from 15.6% to 71.0%, and to 86.7% with majority voting
  - Cold-start SFT further improved readability and stability, with results comparable to o1-1217
  - Distilled models (e.g., Qwen-14B/32B, Llama-70B) set new open-source baselines for dense models

## Design Choices That Matter

- Group-based RL (GRPO): Stabilizes updates by normalizing rewards within sampled groups—cheaper and simpler than training a separate value function
- Rule-based rewards: Easy to verify and hard to game (e.g., exact answer checks, test cases). Avoids reward hacking common with neural reward models
- Explicit output template: Forces “Reasoning → Final Answer,” making evaluation and filtering easier
- Dual rewards: Accuracy + language/format consistency produces readable, verifiable traces
- Rejection sampling: A practical way to bootstrap a high-quality CoT dataset from your own model

## Minimal Prototype: GRPO with Rule-Based Rewards

Below is a simplified Python sketch to help you reason about implementing GRPO-style updates with rule-based rewards for math QA. It assumes:
- A model.generate() API that returns multiple candidate outputs
- A function to check correctness by extracting a boxed answer (e.g., “\boxed{42}”)
- Group-based baselines to compute advantages

This is illustrative; adapt it to your actual model and training stack.

```python
import re
import math
import random
from typing import List, Dict

# ---- Utilities ----

def extract_boxed_answer(text: str) -> str:
    m = re.search(r"\\boxed\{([^{}]+)\}", text)
    return m.group(1).strip() if m else ""

def format_is_valid(text: str) -> bool:
    # Require "Reasoning:" and "Final Answer:" sections
    return ("Reasoning:" in text) and ("Final Answer:" in text) and ("\\boxed{" in text)

def correctness_reward(pred: str, gold: str) -> float:
    return 1.0 if extract_boxed_answer(pred) == gold else 0.0

def format_reward(pred: str) -> float:
    return 0.2 if format_is_valid(pred) else 0.0

def total_reward(pred: str, gold: str) -> float:
    # Rule-based dual reward: correctness + format
    return correctness_reward(pred, gold) + format_reward(pred)

# ---- Fake model API for illustration ----

class ToyModel:
    def __init__(self):
        self.temperature = 0.7
        self.theta = 0.0  # pretend parameter

    def logprob(self, prompt: str, output: str) -> float:
        # Dummy logprob based on a pretend "match" heuristic
        score = 1.0 if "Final Answer:" in output else 0.2
        score += 0.5 if "\\boxed{" in output else 0.0
        score += 0.3 if "Reasoning:" in output else 0.0
        return math.log(max(1e-6, 0.1 * score))

    def generate(self, prompt: str, num_samples: int) -> List[str]:
        # Return a mix of valid/invalid formats and random answers
        samples = []
        for _ in range(num_samples):
            ans = str(random.choice([12, 13, 42, 7]))
            if random.random() < 0.7:
                out = f"Reasoning: ... some steps ...\nFinal Answer: \\boxed{{{ans}}}"
            else:
                out = f"Answer: {ans}"
            samples.append(out)
        return samples

    def step(self, grads: float, lr: float = 1e-3):
        # Toy "update": move theta in the direction of grads
        self.theta += lr * grads

# ---- GRPO-like training loop ----

def grpo_train_step(model: ToyModel, batch: List[Dict], group_size: int = 8, lr: float = 1e-3):
    """
    batch: list of dicts with {prompt, gold_answer}
    For each prompt, sample 'group_size' outputs, compute rewards,
    use group mean as baseline to form advantages, and update with a
    PPO-like surrogate (here simplified to advantage-weighted logprobs).
    """
    total_grad_signal = 0.0
    for ex in batch:
        prompt, gold = ex["prompt"], ex["gold_answer"]

        # 1) Sample a group of candidates
        cands = model.generate(prompt, num_samples=group_size)

        # 2) Compute rewards and baseline
        rewards = [total_reward(c, gold) for c in cands]
        baseline = sum(rewards) / len(rewards)

        # 3) Compute advantages (group-relative)
        advantages = [r - baseline for r in rewards]

        # 4) Surrogate objective: sum_i A_i * logpi(o_i | prompt)
        # Here we treat grad ~ d/dtheta (A * logpi) and accumulate as a scalar
        for cand, adv in zip(cands, advantages):
            lp = model.logprob(prompt, cand)
            total_grad_signal += adv * lp  # toy gradient signal

    # 5) Update model parameters
    model.step(grads=total_grad_signal, lr=lr)

# ---- Example usage ----

if __name__ == "__main__":
    random.seed(3)
    model = ToyModel()

    # Tiny "math" dataset with gold boxed answers
    dataset = [
        {"prompt": "Compute 3*4.", "gold_answer": "12"},
        {"prompt": "What is 6+7?", "gold_answer": "13"},
        {"prompt": "Life, the universe, and everything?", "gold_answer": "42"},
    ]

    # Train for a few steps
    for step in range(20):
        grpo_train_step(model, batch=dataset, group_size=6, lr=5e-3)
    print("Training complete. Theta:", model.theta)
```

How this maps to the paper:
- Group baseline: advantages computed relative to the group mean reward
- Rule-based rewards: exact final-answer checks and format consistency
- Template: “Reasoning” → “Final Answer” with a boxed answer
- PPO-ish surrogate: we weight log-probabilities by advantages (simplified here)

Tip: In a real system, replace ToyModel with your LLM (e.g., vLLM/HF Transformers), plug in exact match/test-case rewards, and wrap the loss in a proper PPO/GRPO implementation.

## Distillation in Practice (Simple Sketch)

1) Use your RL-trained “teacher” to generate high-quality CoT traces with strict format and correctness filters  
2) Curate ~100k–1M samples (the paper used ~800k)  
3) Fine-tune smaller models via SFT on this dataset

Pseudocode outline:
- Prompt teacher with zero-shot, format-specified instructions
- Generate multiple samples per question
- Filter by correctness/format/test cases
- Deduplicate and balance by domain
- Fine-tune small models on the accepted pairs (prompt → CoT → final answer)

## What Worked (and What Didn’t)

Highlights:
- Pure RL incentivized reasoning: AIME 2024 pass@1 jumped from 15.6% to 71.0%; majority vote reached 86.7%
- Cold-start SFT improved readability and convergence; results rivaled strong proprietary models (e.g., o1-1217)
- Distillation transferred reasoning: smaller Qwen/Llama variants outperformed much larger non-reasoning baselines

Limitations to watch:
- General capabilities: function calling, multi-turn chat, role-play, strict JSON output can lag behind non-reasoning-optimized models
- Multilingual consistency: traces may mix languages outside Chinese/English
- Prompt sensitivity: few-shot prompts can hurt; prefer zero-shot with explicit format
- Safety vs. coverage: safety alignment may lower willingness/accuracy in some domains (e.g., non-English QA)
- RL cost: long-horizon tasks (e.g., software engineering) are expensive to evaluate and optimize
- Small-model RL: less efficient than distillation; may underperform without heavy compute

## Practical Tips to Reproduce

- Start simple: rule-based rewards for correctness and format go a long way
- Use explicit templates: always ask for “Reasoning” then “Final Answer”
- Prefer zero-shot prompts with strong format constraints
- Use group-based sampling for stable RL updates; avoid training a separate reward model unless necessary
- Mine your own high-quality data via rejection sampling before SFT
- Distill early: small models benefit greatly from the teacher’s curated traces

## Key Takeaways

- You can unlock reasoning with pure RL, using simple, verifiable rewards
- A small cold-start SFT greatly stabilizes and accelerates RL training
- Rejection sampling yields high-quality CoT data you can reuse for SFT
- Distillation scales reasoning to smaller, cheaper models without RL
- Design choices—templates, rewards, sampling—matter as much as architecture

## Potential Applications

- STEM tutoring and exam prep (math/physics solutions with step-by-step reasoning)
- Code generation with planning and self-verification via unit tests
- Long-context document analysis and extraction with explicit rationale
- Safer assistants that explain decisions and can be audited post-hoc
- On-device or low-cost reasoning via distilled small models

Curious to try this? Start with a small RL loop, enforce a strict output template, and add rule-based rewards. Once your model starts “thinking out loud,” you’re ready to harvest traces and distill them into a model you can ship.