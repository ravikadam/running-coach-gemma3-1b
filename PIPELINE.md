# Rebel Running Coach → Gemma 3 1B → Edge Gallery

End-to-end pipeline: book → dataset → LoRA fine-tune → `.task` → phone.

## 0. Status
- [x] Dataset generated: `data/train.jsonl` (1,126) + `data/val.jsonl` (59), `messages` format.
- [ ] Train Gemma 3 1B (Unsloth Desktop, local, QLoRA)
- [ ] Export merged 16-bit model
- [ ] Convert to LiteRT `.task` (Colab/Linux — `ai-edge-torch`)
- [ ] Sideload into Google AI Edge Gallery

---

## 1. Train in Unsloth Desktop (local, on your M2 Pro)

**Train tab → Configure:**

| Field | Value |
|---|---|
| Model | `unsloth/gemma-3-1b-it` (search "gemma-3-1b" in the model dropdown) |
| Method | **QLoRA** (fits comfortably in 16 GB) |
| HF Token | paste your token — needed, Gemma is gated. It's saved at `~/.cache/huggingface/token` |
| Dataset | **Browse** → `runtrain/data/train.jsonl` |
| Eval/validation | `runtrain/data/val.jsonl` (if the app exposes an eval field) |

**Hyperparameters (effective + anti-overfit for ~1.1k examples):**

| Param | Value | Why |
|---|---|---|
| Epochs | **2** (max 3) | Small dataset — more epochs memorize & degrade general ability |
| LoRA rank (r) | **16** | Enough capacity for a narrow domain |
| LoRA alpha | 32 | 2× rank, standard |
| LoRA dropout | 0.05 | mild regularization |
| Learning rate | **2e-4** | standard LoRA LR |
| LR schedule | cosine, warmup ~5% | |
| Max seq length | 1024 | our examples are short |
| Batch / grad-accum | whatever fits; effective batch ~16–32 | |
| Target modules | all-linear (q,k,v,o + gate,up,down) | best quality |

**Watch the val loss.** If train loss keeps dropping while val loss rises → overfitting; stop at 1–2 epochs. If both still falling at epoch 2 → we have room, bump dataset target and regenerate.

**Quick sanity chat** after training (app's chat tab): ask "my knee hurts below the kneecap after running" and "who are you?" — confirm coach persona + medical hedge.

---

## 2. Export for conversion

In the app, export the fine-tuned model as **Merged 16-bit (safetensors)** — this is what the LiteRT converter needs. (GGUF export is for llama.cpp/Ollama, NOT for Edge Gallery.)
If the app only exports the LoRA adapter, that's fine — the Colab script below can merge base + adapter.

Put the exported folder somewhere you can upload (e.g. zip it).

---

## 3. Convert to `.task` (Colab / any Linux + NVIDIA — NOT possible on macOS-ARM)

Edge Gallery runs **LiteRT `.task`** bundles via MediaPipe LLM Inference. The converter
(`ai-edge-torch`) does not build on Apple Silicon, so this one step runs on Colab.

Use `scripts/convert_to_task.ipynb` (Colab). It:
1. installs `ai-edge-torch`, `mediapipe`, `ai-edge-litert`
2. loads your merged Gemma 3 1B checkpoint
3. converts → quantizes (int4 dynamic, ~550 MB → runs on-device)
4. bundles into `running-coach-gemma3-1b.task`

Download the `.task`.

---

## 4. Sideload into AI Edge Gallery

1. Install **Google AI Edge Gallery** APK on the phone.
2. Copy `running-coach-gemma3-1b.task` to the device (USB / Drive / `adb push`).
3. Gallery → LLM / "Local models" → **+ / Import local model** → pick the `.task`.
4. Chat with your Rebel Running Coach.

---

## Regenerate dataset (if needed)
```bash
cd runtrain
python scripts/gen_dataset.py all --target 2500 --per-call 10 --concurrency 8
# knobs: --model gpt-4o-mini  --sim 0.90 (lower = more aggressive dedup)
```
