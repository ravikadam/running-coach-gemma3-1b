# 🏃 Running Coach — Gemma 3 1B, fine-tuned for on-device

A small (1B-parameter) large language model fine-tuned into a friendly **running coach** that
runs **entirely on your phone** — no internet, no server, no data leaving the device.

Base model: **Google Gemma 3 1B** · Method: **LoRA (QLoRA)** trained locally on a Mac with
**[Unsloth Desktop](https://unsloth.ai)** (Apple MLX backend) · Evaluated with **[DeepEval](https://github.com/confident-ai/deepeval)**.

> ⚠️ **Disclaimer — this is an educational demo, not medical advice.**
> The model can be wrong or incomplete. It is **not** a doctor, physiotherapist, or certified coach.
> If you have pain, a possible injury, or any medical symptom, stop and consult a qualified
> professional. Do not rely on this model for health decisions.

---

## What this repo shows

An end-to-end, mostly-local pipeline to turn a text guide into an on-device coaching model:

1. **Dataset generation** — synthesize diverse instruction/response pairs from a source guide
   (`scripts/gen_dataset.py`): 8 question archetypes × personas, windowed over the whole text,
   then **semantic de-duplication** with a local embedding model, plus safety/identity seeds.
2. **Fine-tune** — Gemma 3 1B, QLoRA, 2 epochs, locally in Unsloth Desktop (train loss 2.33 → 1.05).
3. **Evaluate** — `scripts/eval_deepeval.py`: an LLM-judge harness (DeepEval) that scores the
   **base vs fine-tuned** model on a golden set across coaching quality, answer relevancy,
   identity, off-topic boundaries, and **medical-safety escalation**.
4. **Package** — merge the MLX adapter (`scripts/mlx_merge.py`), export **GGUF** and **MLX**, and
   convert to a LiteRT **`.task`** for Google AI Edge Gallery (`scripts/convert_to_task.ipynb`).

> The **training dataset is intentionally not included** — it is synthesized from a third-party
> copyrighted running guide (see *Attribution* below). The scripts let you regenerate your own
> dataset from any source text you have the right to use.

---

## Results (DeepEval, judge = gpt-5.4-mini, 37 golden items, pass ≥ 0.70)

| Metric | Base | Fine-tuned |
|---|---|---|
| Coach identity (self-describes as a running coach) | 0% | **100%** |
| Coaching quality (accurate & grounded, coach tone) | 35% | **81%** |
| Answer relevancy | 96% | **100%** |
| Stays in domain / redirects off-topic | 0% | **50%** |
| Medical-emergency escalation | 75% | **50%** |

**Honest read:** fine-tuning delivered a large gain in identity and coaching quality. It also
surfaced two weaknesses via eval that a manual chat-check missed: (1) **off-topic redirects** are
only ~50% reliable, and (2) **medical-emergency escalation regressed** (the model became a more
confident coach and escalates true red-flags less often). This release ships with the disclaimer
above precisely because of (2). A follow-up (more upweighted safety examples) is the planned fix.

Full report: [`RESULTS.md`](RESULTS.md).

---

## Get it on your phone — official Google AI Edge Gallery

Install **Google AI Edge Gallery** (Play Store / App Store), then load the model one of two ways:

**A. Import from a Hugging Face URL (in-app):** in Gallery, import a model by URL and paste
`https://huggingface.co/ravikadam/running-coach-gemma3-1b-LiteRT`.

**B. Import a local model (works for any `.task`):**
1. Download [`running-coach-gemma3-1b.task`](https://huggingface.co/ravikadam/running-coach-gemma3-1b-LiteRT) to the phone's **Download** folder.
2. In Gallery, tap the **"+"** (bottom-right) → select the `.task` → set default params (CPU/GPU) → **Import**.
3. Open it and chat — fully offline.

Prefer llama.cpp-based apps? Use the **GGUF** build (Ollama, LM Studio, etc.):
`ravikadam/running-coach-gemma3-1b-GGUF`.

> Building the `.task` yourself: `scripts/convert_to_task.ipynb` (Colab) works but **OOMs on free
> Colab at quantization** — use a machine with ≥24 GB RAM (a cheap cloud GPU/CPU box). The converter
> does not build on Apple Silicon.

---

## Reproduce

```bash
pip install -r requirements.txt        # openai, deepeval, transformers, peft, mlx-lm, pypdf
# 1. dataset (needs your own source text + OPENAI_API_KEY in .env)
python scripts/gen_dataset.py all --target 2500
# 2. train in Unsloth Desktop (Gemma 3 1B, QLoRA, 2 epochs) -> LoRA adapter
# 3. merge + evaluate
python scripts/mlx_merge.py --adapter <adapter_dir> --out outputs/merged_hf
python scripts/eval_deepeval.py all
# 4. package (see PIPELINE.md)
```

Full step-by-step: [`PIPELINE.md`](PIPELINE.md).

---

## Attribution & IP

- **Base model:** Google **Gemma 3** — use is subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
- **Coaching content:** the model was adapted using material paraphrased from a third-party
  running guide. This project is **educational, transformative, and not affiliated with or endorsed
  by** the guide's authors or publisher. The source text and the derived dataset are **not
  redistributed** here.
- **Code:** MIT (see [`LICENSE`](LICENSE)).

**Not medical advice.** See the disclaimer at the top.
