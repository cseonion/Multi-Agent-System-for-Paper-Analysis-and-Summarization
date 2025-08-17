# Transformers, BERT, and RoBERTa: What Changed, What Matters, and When to Use Each

Based on the multi-paper analysis report

Introduction
Modern NLP is built on three pivotal ideas: attention-only sequence modeling (Transformer), bidirectional encoder pretraining (BERT), and “it’s the training recipe, not the architecture” (RoBERTa). In this post, you’ll learn how these models differ in goals, training, and practical use, and you’ll leave with a small code snippet to try dynamic masking (the trick that helped RoBERTa pull ahead).

## Executive snapshot (one paragraph each)

- Transformer (Attention Is All You Need): Introduces the encoder–decoder architecture powered entirely by multi-head self-attention. It replaces recurrence/convolution, boosts parallelism, models long-range dependencies better, and achieves state-of-the-art machine translation with faster training.

- BERT: Repurposes the Transformer encoder for self-supervised pretraining (Masked Language Modeling + Next Sentence Prediction). It learns deeply bidirectional representations that fine-tune well on diverse NLU tasks with minimal task-specific engineering.

- RoBERTa: Keeps BERT’s encoder but shows that scale and training strategy dominate performance. It removes NSP, uses dynamic masking, increases data, batch size, and training time, and delivers stronger NLU results without changing the architecture.

## Problem settings and goals

- Transformer (AIAIY): Supervised sequence-to-sequence (e.g., WMT’14 machine translation). Goals: parallel training, long-range modeling, faster convergence vs. RNN/CNN.

- BERT: General-purpose language understanding via self-supervised pretraining, then fine-tuning for GLUE, SQuAD, SWAG. Goal: exploit deep bidirectional context.

- RoBERTa: Re-evaluates BERT’s pretraining to isolate what truly matters. Goal: maximize downstream accuracy via optimized data and training, not architectural tweaks.

## What’s inside: architectures and objectives

- Transformer (encoder–decoder)
  - Fixed sinusoidal positional encodings.
  - Multi-head attention + residuals + feed-forward layers.
  - Decoder uses masked self-attention; cross-attention connects encoder to decoder.
  - Objective: supervised token-level cross-entropy with autoregressive decoding.

- BERT (encoder-only)
  - Learned positional embeddings; WordPiece tokenization.
  - Special tokens: [CLS], [SEP]; segment embeddings.
  - Objectives: Masked Language Modeling (15% tokens; mixed replacement) + Next Sentence Prediction.

- RoBERTa (encoder-only, BERT-compatible)
  - Same architecture as BERT; byte-level BPE (50k) instead of WordPiece.
  - Objective: MLM only; no NSP.
  - Dynamic masking (new masks each epoch), larger batches, longer training, more diverse data.

Positional handling: Transformer uses fixed sinusoidal; BERT/RoBERTa use learned positions.

## Data, scale, and compute (why training matters)

- Transformer: Trained on WMT’14 En–De/En–Fr with 8×P100; hours–days; efficient vs. RNN/CNN.

- BERT: BooksCorpus + Wikipedia; BASE and LARGE variants; heavy pretraining (for its time).

- RoBERTa: ~160GB of text (BookCorpus, Wikipedia, CC-News, OpenWebText, Stories), very large batches (up to ~8k sequences), long schedules (up to ~500k steps), heavy distributed training.

## Results at a glance

- Transformer: New SOTA BLEU on WMT’14 En–De (28.4) and En–Fr (41.0); strong transfer to parsing.

- BERT: SOTA across GLUE, SQuAD v1.1, SWAG; simple fine-tuning works broadly.

- RoBERTa: Improves over BERT on GLUE (dev/test), SQuAD v1.1/v2.0, RACE; competitive with XLNet despite no NSP and unchanged architecture.

## Strengths and limitations

- Transformer (AIAIY)
  - Strengths: Attention-only seq2seq with massive parallelism; efficient training; interpretable heads; great for generative transduction.
  - Limits: O(n^2) attention cost; needs parallel data; autoregressive decoding is sequential at inference.

- BERT
  - Strengths: Universal bidirectional representations; strong fine-tuning baselines; minimal task-specific work.
  - Limits: Expensive pretraining; NSP may not help; domain mismatch without continued pretraining; MLM is sample-inefficient.

- RoBERTa
  - Strengths: Shows data/compute and dynamic masking drive gains; robust NLU baselines; removes NSP without loss.
  - Limits: Higher compute/data demands; English-centric; still O(n^2); not tailored for generative seq2seq.

## Conceptual impact

- Transformer: Establishes attention-only as a general backbone across NLP, vision, speech; encoder/decoder foundations for later models.

- BERT: Popularizes pretrain-then-finetune; proves importance of bidirectionality; standardizes [CLS]/[SEP] formatting.

- RoBERTa: Recalibrates the field toward training/data optimization; challenges NSP; provides stronger, reproducible baselines.

## Assumptions and trade-offs

- Transformer: Trades quadratic memory for global receptive field and throughput; best fit for seq2seq generation.

- BERT: Assumes masked-token recovery yields transferable semantics; trades generative ability for discriminative strength.

- RoBERTa: Assumes scaling and recipe refinements beat architectural novelty; trades accessibility for peak accuracy.

## Practical guidance: when to use which

- Need machine translation, summarization, data-to-text, or any seq2seq with cross-attention? Use a Transformer encoder–decoder (AIAIY-style). Modern variants add relative positions and better decoding, but the core holds.

- Need strong NLU (classification, NER, extractive QA) with little task-specific engineering? Use a pretrained encoder. Prefer RoBERTa for stronger results if you can afford it; choose BERT for compatibility, smaller checkpoints, or tighter resource budgets.

- Domain-specific tasks (biomedical, legal, code): Continue pretraining (domain-adaptive pretraining) on in-domain data for BERT/RoBERTa before fine-tuning.

## Hands-on: dynamic masking for MLM in practice (RoBERTa vs. BERT)

The snippet below shows a minimal masked language modeling setup using Hugging Face Transformers. Switching the model name flips between BERT and RoBERTa; DataCollatorForLanguageModeling enables dynamic masking (RoBERTa-style).

```python
# pip install transformers datasets accelerate -q
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForMaskedLM,
                          DataCollatorForLanguageModeling, Trainer, TrainingArguments)

# Toggle between 'roberta-base' (byte-level BPE, no NSP) and 'bert-base-uncased' (WordPiece)
MODEL_NAME = "roberta-base"  # or "bert-base-uncased"

# 1) Load a small text corpus (subset for a quick demo)
raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, padding=False)

tok = raw.map(tokenize, batched=True, remove_columns=["text"])

# 2) Dynamic masking: new masks each batch/epoch (key to RoBERTa’s gains)
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# 3) Model and quick training loop (few steps for illustration)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

args = TrainingArguments(
    output_dir="mlm-demo",
    per_device_train_batch_size=8,
    learning_rate=5e-5,
    num_train_epochs=1,
    max_steps=100,       # keep tiny for demo
    logging_steps=20,
    report_to="none",
)

trainer = Trainer(model=model, args=args, train_dataset=tok, data_collator=collator)
trainer.train()

# Tip:
# - Using 'roberta-base' mirrors RoBERTa-style MLM (no NSP) with dynamic masking.
# - Using 'bert-base-uncased' still benefits from dynamic masking here,
#   but original BERT pretraining also included Next Sentence Prediction.
```

Notes:
- NSP is not part of RoBERTa’s recipe. The Trainer above uses MLM only, which aligns with RoBERTa.
- For task fine-tuning (e.g., SST-2, MRPC), swap the model head to AutoModelForSequenceClassification and load a GLUE dataset.

## Reproducibility and resources

- All three released code/models; RoBERTa also shared data resources and detailed training recipes.
- Exact RoBERTa reproduction needs substantial hardware. Even so, smaller-scale replicas benefit from its recipe: dynamic masking, longer training, and larger batches (within your limits).

## Open problems and extensions

- Long-context efficiency: O(n^2) attention spurs sparse/linear attention and memory-augmented approaches.
- Multilingual and low-resource transfer: mBERT and XLM-R extend the recipe; domain shifts remain tricky.
- Generative pretraining: Encoder-only excels at NLU; encoder–decoder or decoder-only pretraining (e.g., T5, GPT) better suit generation—bridging both is active research.
- Interpretability and robustness: Attention helps, but robust generalization under distribution shifts and adversarial inputs is still an open challenge.

## Key takeaways

- Transformer (AIAIY) unlocked parallel, attention-only seq2seq and set the blueprint for modern architectures.
- BERT proved bidirectional encoder pretraining plus simple fine-tuning yields strong, general NLU performance.
- RoBERTa showed that training/data scale and dynamic masking, not architectural changes, drove further NLU gains.

## Potential applications

- Transformer: Machine translation, abstractive summarization, code generation, speech recognition with seq2seq.
- BERT/RoBERTa: Text classification, NER, extractive QA, sentence similarity, retrieval re-ranking.
- With domain-adaptive pretraining: Biomedical/clinical NLP, legal document understanding, code intelligence, financial analysis.