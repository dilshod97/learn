"""
╔══════════════════════════════════════════════════════════════╗
║         O'ZBEK GPT-20B — QLoRA FINE-TUNING (TO'LIQ)        ║
╚══════════════════════════════════════════════════════════════╝

O'rnatish:
    pip3 install transformers datasets peft bitsandbytes accelerate trl sentencepiece

Ishga tushirish:
    # Faqat dataset tayyorlash:
    python lora_finetune.py --mode prepare --txt uzbek_data.txt

    # Fine-tuning:
    python lora_finetune.py --mode train --txt uzbek_data.txt

    # Savol-javob:
    python lora_finetune.py --mode chat

    # Hammasi ketma-ket:
    python3 main.py --mode all --txt uzbek_data.txt
"""

import os
import re
import json
import argparse
import torch
from pathlib import Path

# ══════════════════════════════════════════════════════════════
#  SOZLAMALAR — faqat shu yerda o'zgartiring
# ══════════════════════════════════════════════════════════════
CONFIG = {
    # Model
    "base_model"   : "EleutherAI/gpt-neox-20b",  # 20B model
    # "base_model"  : "mistralai/Mistral-7B-v0.1", # Kichikroq (7B) variant

    # Fayllar
    "txt_file"     : "uzbek_data.txt",
    "dataset_file" : "uzbek_dataset.json",
    "output_dir"   : "./uzbek-gpt-lora",

    # Dataset
    "chunk_size"   : 300,    # Har bir parchada necha so'z

    # LoRA
    "lora_r"       : 16,     # Rank: 8=tez, 16=muvozanat, 32=sifatli
    "lora_alpha"   : 32,     # Scaling (odatda r*2)
    "lora_dropout" : 0.05,

    # Training
    "epochs"       : 3,
    "batch_size"   : 2,
    "grad_accum"   : 8,      # Effective batch = batch_size * grad_accum = 16
    "learning_rate": 2e-4,
    "max_seq_len"  : 1024,

    # Generatsiya
    "max_new_tokens"    : 512,
    "temperature"       : 0.7,
    "top_p"             : 0.9,
    "repetition_penalty": 1.15,
}


# ══════════════════════════════════════════════════════════════
#  1. DATASET TAYYORLASH
# ══════════════════════════════════════════════════════════════
def prepare_dataset(txt_file: str, output_file: str, chunk_size: int = 300):
    print("\n" + "═"*50)
    print("📖  DATASET TAYYORLANMOQDA")
    print("═"*50)

    # --- Fayl o'qish ---
    with open(txt_file, "r", encoding="utf-8") as f:
        raw = f.read()
    print(f"✅ Fayl o'qildi: {len(raw):,} belgi, {len(raw.split()):,} so'z")

    # --- Tozalash ---
    text = re.sub(r'[ \t]+', ' ', raw)          # Ko'p bo'shliqlar
    text = re.sub(r'\n{3,}', '\n\n', text)       # Ko'p bo'sh qatorlar
    text = re.sub(r'[^\w\s.,!?;:\-\'"()\n]', '', text)  # Keraksiz belgilar
    text = text.strip()

    # --- Chunklarga bo'lish ---
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, cur, cur_len = [], [], 0

    for sent in sentences:
        words = sent.split()
        if cur_len + len(words) > chunk_size and cur:
            chunks.append(" ".join(cur))
            cur, cur_len = words, len(words)
        else:
            cur.extend(words)
            cur_len += len(words)
    if cur:
        chunks.append(" ".join(cur))

    # Juda qisqa chunklar olib tashlanadi
    chunks = [c for c in chunks if len(c.split()) > 20]
    print(f"✂️  {len(chunks)} ta chunk (har biri ~{chunk_size} so'z)")

    # --- Q&A juftlari yasash ---
    dataset = []
    for chunk in chunks:
        words = chunk.split()
        half  = len(words) // 2

        # Format 1: To'liq matn o'rganish
        dataset.append({
            "instruction": "Quyidagi o'zbek matnini o'qi va eslаb qol.",
            "input"      : chunk,
            "output"     : chunk
        })

        # Format 2: Matnni davom ettir
        if half > 15:
            dataset.append({
                "instruction": "Quyidagi matnni davom ettir.",
                "input"      : " ".join(words[:half]),
                "output"     : " ".join(words[half:])
            })

        # Format 3: Mazmunini ayt
        dataset.append({
            "instruction": "Quyidagi matn haqida qisqacha ma'lumot ber.",
            "input"      : chunk,
            "output"     : " ".join(words[:40]) + "..."
        })

        # Format 4: Sof generatsiya (input yo'q)
        dataset.append({
            "instruction": "O'zbek tilida ma'lumot ber:",
            "input"      : "",
            "output"     : chunk
        })

    print(f"📚 Jami: {len(dataset)} ta misol ({len(dataset)//len(chunks)}x ko'paytirildi)")

    # --- Saqlash ---
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"✅ Saqlandi: {output_file}")

    # Namuna
    print("\n--- Namuna ---")
    sample = dataset[1]
    print(f"instruction : {sample['instruction']}")
    print(f"input       : {sample['input'][:80]}...")
    print(f"output      : {sample['output'][:80]}...")

    return output_file


# ══════════════════════════════════════════════════════════════
#  2. FINE-TUNING
# ══════════════════════════════════════════════════════════════
def train(cfg: dict):
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        BitsAndBytesConfig, TrainingArguments
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    from datasets import load_dataset

    print("\n" + "═"*50)
    print("🚀  FINE-TUNING BOSHLANMOQDA")
    print("═"*50)
    print(f"Model     : {cfg['base_model']}")
    print(f"Dataset   : {cfg['dataset_file']}")
    print(f"LoRA rank : {cfg['lora_r']}")
    print(f"Epochs    : {cfg['epochs']}")

    # --- 4-bit Quantization ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # --- Tokenizer ---
    print("\n📦 Tokenizer yuklanmoqda...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- Model ---
    print("📦 Model yuklanmoqda (4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # --- LoRA ---
    # GPT-NeoX-20B uchun target_modules = "query_key_value"
    # Mistral/LLaMA uchun = ["q_proj","v_proj"]
    target = (
        ["query_key_value"]
        if "neox" in cfg["base_model"].lower()
        else ["q_proj", "v_proj"]
    )

    lora_cfg = LoraConfig(
        r             = cfg["lora_r"],
        lora_alpha    = cfg["lora_alpha"],
        target_modules= target,
        lora_dropout  = cfg["lora_dropout"],
        bias          = "none",
        task_type     = "CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # --- Dataset ---
    dataset = load_dataset("json", data_files=cfg["dataset_file"], split="train")
    print(f"📂 Dataset: {len(dataset)} ta misol")

    def fmt(ex):
        """Alpaca prompt formati"""
        if ex["input"]:
            return (
                f"### Ko'rsatma:\n{ex['instruction']}\n\n"
                f"### Kirish:\n{ex['input']}\n\n"
                f"### Javob:\n{ex['output']}"
            )
        return (
            f"### Ko'rsatma:\n{ex['instruction']}\n\n"
            f"### Javob:\n{ex['output']}"
        )

    # --- Training ---
    args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        gradient_checkpointing=True,
        optim="adamw_torch",  # ← "paged_adamw_8bit" o'rniga
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_steps=10,  # ← "warmup_ratio" o'rniga
        logging_steps=10,
        save_steps=100,
        bf16=False,  # ← MPS bf16 ni qo'llab-quvvatlamaydi
        fp16=False,
        max_grad_norm=0.3,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        formatting_func=fmt,
        processing_class=tokenizer,  # ← "tokenizer" o'rniga "processing_class"
        args=args
    )

    trainer.train()

    # --- Saqlash ---
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"\n✅ Model saqlandi: {cfg['output_dir']}")


# ══════════════════════════════════════════════════════════════
#  3. SAVOL-JAVOB (CHAT)
# ══════════════════════════════════════════════════════════════
def chat(cfg: dict):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    print("\n" + "═"*50)
    print("💬  SAVOL-JAVOB REJIMI")
    print("═"*50)

    if not Path(cfg["output_dir"]).exists():
        print(f"❌ Model topilmadi: {cfg['output_dir']}")
        print("   Avval '--mode train' bilan train qiling.")
        return

    # --- Model yuklash ---
    print("📦 Model yuklanmoqda (4-bit)...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["output_dir"])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, cfg["output_dir"])
    model.eval()

    def ask(question: str, context: str = "") -> str:
        if context:
            # Training formatiga mos: Ko'rsatma → Kirish → Javob
            prompt = (
                f"### Ko'rsatma:\nQuyidagi matn asosida savolga javob ber: {question}\n\n"
                f"### Kirish:\n{context}\n\n"
                f"### Javob:\n"
            )
        else:
            prompt = (
                f"### Ko'rsatma:\n{question}\n\n"
                f"### Javob:\n"
            )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens     = cfg["max_new_tokens"],
                temperature        = cfg["temperature"],
                top_p              = cfg["top_p"],
                repetition_penalty = cfg["repetition_penalty"],
                do_sample          = True,
                eos_token_id       = tokenizer.eos_token_id,
                pad_token_id       = tokenizer.eos_token_id,
            )
        # Faqat yangi tokenlar (promptdan keyingi qism)
        new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        answer = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        return answer if answer else "(bo'sh javob)"

    # --- Interaktiv loop ---
    print("\n✅ Tayyor! ('exit' — chiqish, 'clear' — tozalash)\n")
    while True:
        q = input("❓ Savol: ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit", "chiq"):
            print("👋 Xayr!")
            break
        if q.lower() == "clear":
            os.system("cls" if os.name == "nt" else "clear")
            continue

        ctx = input("📄 Kontekst (ixtiyoriy, bo'sh — Enter): ").strip()
        print("\n💬 Javob:")
        print(ask(q, ctx))
        print("─" * 50)


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="O'zbek GPT LoRA Fine-Tuning")
    parser.add_argument(
        "--mode", choices=["prepare", "train", "chat", "all"],
        default="all", help="Rejim tanlash"
    )
    parser.add_argument(
        "--txt", default=CONFIG["txt_file"],
        help="TXT fayl yo'li"
    )
    args = parser.parse_args()

    CONFIG["txt_file"] = args.txt

    if args.mode in ("prepare", "all"):
        prepare_dataset(
            CONFIG["txt_file"],
            CONFIG["dataset_file"],
            CONFIG["chunk_size"]
        )

    if args.mode in ("train", "all"):
        train(CONFIG)

    if args.mode in ("chat", "all"):
        chat(CONFIG)