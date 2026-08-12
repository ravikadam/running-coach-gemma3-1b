# Convert a fine-tuned Gemma 3 1B -> LiteRT .task for Google AI Edge Gallery
# RUN ON GOOGLE COLAB (GPU runtime) or a Linux+NVIDIA box. Does NOT build on macOS-ARM.
# Paste cells into a Colab notebook (each `# %%` is a cell).
#
# NOTE: ai-edge-torch changes fast. Verify current API against:
#   https://github.com/google-ai-edge/ai-edge-torch/tree/main/ai_edge_torch/generative/examples/gemma3
# We'll confirm exact calls together right before running.

# %% [1] install (pin nightly; versions matter)
# !pip -q install ai-edge-torch-nightly ai-edge-litert-nightly mediapipe huggingface_hub

# %% [2] get your fine-tuned checkpoint
# Option A: you exported a MERGED 16-bit model from Unsloth Desktop -> upload/unzip it here.
# Option B: you have base + LoRA adapter -> merge first:
#   from peft import PeftModel; from transformers import AutoModelForCausalLM, AutoTokenizer
#   base = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it", torch_dtype="bfloat16")
#   model = PeftModel.from_pretrained(base, "ADAPTER_DIR").merge_and_unload()
#   model.save_pretrained("merged_gemma3_1b"); AutoTokenizer.from_pretrained("google/gemma-3-1b-it").save_pretrained("merged_gemma3_1b")
CKPT = "merged_gemma3_1b"   # local path to merged HF checkpoint

# %% [3] convert HF -> LiteRT (.tflite) via ai-edge-torch generative API, int4/int8 quant
# The generative examples ship a Gemma3 builder + a converter utility.
from ai_edge_torch.generative.examples.gemma3 import gemma3
from ai_edge_torch.generative.utilities import converter
from ai_edge_torch.generative.utilities.export_config import ExportConfig

pytorch_model = gemma3.build_model_1b(f"{CKPT}")     # loads weights from the checkpoint
output_tflite = "gemma3_1b_int4.tflite"
converter.convert_to_tflite(
    pytorch_model,
    output_path=".",
    output_name_prefix="gemma3_1b",
    prefill_seq_len=1024,
    kv_cache_max_len=1280,
    quantize="dynamic_int4",       # int4 -> ~550MB; use "dynamic_int8" for more quality/size
    export_config=ExportConfig(),
)

# %% [4] bundle .tflite + tokenizer -> .task for MediaPipe / Edge Gallery
from mediapipe.tasks.python.genai import bundler
config = bundler.BundleConfig(
    tflite_model="gemma3_1b_int4.tflite",
    tokenizer_model=f"{CKPT}/tokenizer.model",   # Gemma sentencepiece tokenizer
    start_token="<bos>",
    stop_tokens=["<eos>", "<end_of_turn>"],
    output_filename="running-coach-gemma3-1b.task",
    enable_bytes_to_unicode_mapping=False,
    prompt_prefix="<start_of_turn>user\n",
    prompt_suffix="<end_of_turn>\n<start_of_turn>model\n",
)
bundler.create_bundle(config)
print("wrote running-coach-gemma3-1b.task")

# %% [5] download
# from google.colab import files; files.download("running-coach-gemma3-1b.task")
