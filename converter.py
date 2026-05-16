import os
import subprocess
from pathlib import Path


def find_llama_convert():
    candidates = [
        "./llama.cpp/convert_hf_to_gguf.py",
        "../llama.cpp/convert_hf_to_gguf.py",
        os.path.expanduser("~/llama.cpp/convert_hf_to_gguf.py"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def find_llama_quantize():
    candidates = [
        "./llama.cpp/llama-quantize",
        "../llama.cpp/llama-quantize",
        os.path.expanduser("~/llama.cpp/llama-quantize"),
        "./llama.cpp/build/bin/llama-quantize",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def merge_lora(adapter_dir: str, base_model: str, merged_dir: str, log):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    log("🔀 LoRA weights birlashtirilmoqda (bu uzoq vaqt olishi mumkin)...")

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

    log(f"   Base model yuklanmoqda: {base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    log("   LoRA qo'llanilmoqda...")
    model = PeftModel.from_pretrained(base, adapter_dir)
    model = model.merge_and_unload()

    Path(merged_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    log(f"✅ Merged model saqlandi: {merged_dir}")


def to_gguf(merged_dir: str, gguf_path: str, log) -> str:
    convert_script = find_llama_convert()
    if not convert_script:
        log("⚠️  llama.cpp topilmadi — GGUF conversion o'tkazib yuborildi.")
        log("   O'rnatish uchun: bash setup_llama.sh")
        return None

    log("🔄 GGUF formatiga o'tkazilmoqda...")
    result = subprocess.run(
        ["python3", convert_script, merged_dir, "--outfile", gguf_path, "--outtype", "f16"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        log(f"⚠️  GGUF xato: {result.stderr[-300:]}")
        return None

    log(f"✅ GGUF: {gguf_path}")

    quantize_bin = find_llama_quantize()
    if quantize_bin:
        q_path = gguf_path.replace(".gguf", "-q4.gguf")
        log("⚡ Quantization (Q4_K_M)...")
        r = subprocess.run([quantize_bin, gguf_path, q_path, "Q4_K_M"],
                           capture_output=True, text=True)
        if r.returncode == 0 and Path(q_path).exists():
            log(f"✅ Quantized: {q_path}")
            return q_path

    return gguf_path


def register_in_ollama(gguf_path: str, model_name: str, log) -> str:
    modelfile = f"""FROM {os.path.abspath(gguf_path)}
SYSTEM "Siz o'zbek tilidagi AI yordamchisiz. Foydalanuvchi savollariga o'zbek tilida javob bering."
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.15
"""
    mf_path = "Modelfile"
    with open(mf_path, "w") as f:
        f.write(modelfile)

    log(f"📤 Ollama ga yuklanmoqda: {model_name}...")
    result = subprocess.run(
        ["ollama", "create", model_name, "-f", mf_path],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        log(f"✅ Ollama modeli tayyor: ollama run {model_name}")
        return model_name
    else:
        log(f"⚠️  Ollama xato: {result.stderr[-300:]}")
        return "uzbek-gpt-local"


def convert_to_ollama(adapter_dir: str, base_model: str, merged_dir: str,
                       gguf_path: str, log) -> str:
    merge_lora(adapter_dir, base_model, merged_dir, log)

    final_gguf = to_gguf(merged_dir, gguf_path, log)
    if not final_gguf:
        log("ℹ️  GGUF yo'q — transformers backend bilan suhbat ishlaydi.")
        return "uzbek-gpt-local"

    return register_in_ollama(final_gguf, "uzbek-gpt", log)
