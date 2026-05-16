import os
import re
import json
import torch
from pathlib import Path


CONFIG = {
    "base_model"   : "openai/gpt-oss-20b",
    "dataset_file" : "uzbek_dataset.json",
    "output_dir"   : "./uzbek-gpt-lora",
    "merged_dir"   : "./uzbek-gpt-merged",
    "gguf_path"    : "./uzbek-gpt.gguf",
    "chunk_size"   : 300,
    "lora_r"       : 16,
    "lora_alpha"   : 32,
    "lora_dropout" : 0.05,
    "epochs"       : 3,
    "batch_size"   : 2,
    "grad_accum"   : 8,
    "learning_rate": 2e-4,
    "max_seq_len"  : 1024,
}


class Trainer:
    def __init__(self, state: dict):
        self.state = state
        self.cfg = CONFIG.copy()

    def log(self, msg: str):
        print(msg, flush=True)
        self.state.setdefault("logs", []).append(msg)

    def run(self):
        try:
            self.state.update({"status": "preparing", "logs": [], "progress": 0})

            files = self.state.get("uploaded_files", [])
            self.prepare_dataset(files)
            self.state["progress"] = 20

            self.state["status"] = "training"
            self.train()
            self.state["progress"] = 80

            self.state["status"] = "converting"
            from converter import convert_to_ollama
            model_name = convert_to_ollama(
                self.cfg["output_dir"],
                self.cfg["base_model"],
                self.cfg["merged_dir"],
                self.cfg["gguf_path"],
                self.log,
            )
            self.state["ollama_model"] = model_name
            self.state["progress"] = 100
            self.state["status"] = "ready"
            self.log(f"✅ Tayyor! Ollama modeli: {model_name}")

        except Exception as e:
            self.state["status"] = "error"
            self.log(f"❌ Xato: {e}")
            raise

    def prepare_dataset(self, files: list):
        self.log("📖 Dataset tayyorlanmoqda...")

        raw = ""
        for fpath in files:
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    raw += f.read() + "\n\n"
                self.log(f"   ✓ {Path(fpath).name}")
            except Exception as e:
                self.log(f"   ✗ {fpath}: {e}")

        if not raw.strip():
            raise ValueError("Yuklangan fayllar bo'sh yoki o'qib bo'lmadi")

        text = re.sub(r'[ \t]+', ' ', raw)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()

        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks, cur, cur_len = [], [], 0
        for sent in sentences:
            words = sent.split()
            if cur_len + len(words) > self.cfg["chunk_size"] and cur:
                chunks.append(" ".join(cur))
                cur, cur_len = words, len(words)
            else:
                cur.extend(words)
                cur_len += len(words)
        if cur:
            chunks.append(" ".join(cur))

        chunks = [c for c in chunks if len(c.split()) > 20]
        self.log(f"✂️  {len(chunks)} ta chunk")

        dataset = []
        for chunk in chunks:
            words = chunk.split()
            half = len(words) // 2

            dataset.append({
                "instruction": "Quyidagi o'zbek matnini o'qi va eslab qol.",
                "input": chunk,
                "output": chunk,
            })
            if half > 15:
                dataset.append({
                    "instruction": "Quyidagi matnni davom ettir.",
                    "input": " ".join(words[:half]),
                    "output": " ".join(words[half:]),
                })
            dataset.append({
                "instruction": "Quyidagi matn haqida qisqacha ma'lumot ber.",
                "input": chunk,
                "output": " ".join(words[:40]) + "...",
            })

        with open(self.cfg["dataset_file"], "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        self.log(f"📚 Jami {len(dataset)} ta misol saqlandi → {self.cfg['dataset_file']}")

    def train(self):
        from transformers import (
            AutoTokenizer, AutoModelForCausalLM,
            BitsAndBytesConfig, TrainingArguments,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer
        from datasets import load_dataset

        cfg = self.cfg
        self.log(f"🚀 Fine-tuning: {cfg['base_model']}")

        model_lower    = cfg["base_model"].lower()
        is_gpt_oss     = "gpt-oss" in model_lower
        is_neox        = "neox" in model_lower

        self.log("📦 Tokenizer yuklanmoqda...")
        tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # gpt-oss-20b — allaqachon MXFP4 quantized, BitsAndBytes shart emas
        load_kwargs = {
            "device_map"       : "auto",
            "trust_remote_code": True,
        }
        if is_gpt_oss:
            self.log("📦 Model yuklanmoqda (MXFP4 native quantization)...")
            load_kwargs["torch_dtype"] = "auto"
        else:
            self.log("📦 Model yuklanmoqda (4-bit QLoRA)...")
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], **load_kwargs)
        # gpt-oss MXFP4 prepare_model_for_kbit_training bilan mos kelmaydi
        if not is_gpt_oss:
            model = prepare_model_for_kbit_training(model)
        else:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()

        # LoRA target modules — model arxitekturasiga qarab
        if is_neox:
            target = ["query_key_value"]
        elif is_gpt_oss:
            target = ["q_proj", "k_proj", "v_proj", "o_proj"]
        else:
            target = ["q_proj", "v_proj"]
        lora_cfg = LoraConfig(
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            target_modules=target,
            lora_dropout=cfg["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.log(f"🎯 Trainable params: {trainable:,}")

        dataset = load_dataset("json", data_files=cfg["dataset_file"], split="train")
        self.log(f"📂 Dataset: {len(dataset)} ta misol")

        def fmt(batch):
            """TRL 0.13+ batch ko'rinishida list qaytarishni talab qiladi."""
            texts = []
            instructions = batch["instruction"]
            inputs       = batch["input"]
            outputs      = batch["output"]
            for instr, inp, out in zip(instructions, inputs, outputs):
                if inp:
                    texts.append(
                        f"### Ko'rsatma:\n{instr}\n\n"
                        f"### Kirish:\n{inp}\n\n"
                        f"### Javob:\n{out}"
                    )
                else:
                    texts.append(
                        f"### Ko'rsatma:\n{instr}\n\n### Javob:\n{out}"
                    )
            return texts

        has_cuda = torch.cuda.is_available()
        training_args = TrainingArguments(
            output_dir=cfg["output_dir"],
            num_train_epochs=cfg["epochs"],
            per_device_train_batch_size=cfg["batch_size"],
            gradient_accumulation_steps=cfg["grad_accum"],
            gradient_checkpointing=True,
            optim="paged_adamw_8bit" if has_cuda else "adamw_torch",
            learning_rate=cfg["learning_rate"],
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            logging_steps=10,
            save_steps=200,
            bf16=has_cuda and torch.cuda.is_bf16_supported(),
            fp16=has_cuda and not torch.cuda.is_bf16_supported(),
            max_grad_norm=0.3,
            report_to="none",
        )

        sft = SFTTrainer(
            model=model,
            train_dataset=dataset,
            formatting_func=fmt,
            processing_class=tokenizer,
            args=training_args,
        )
        sft.train()
        sft.save_model(cfg["output_dir"])
        tokenizer.save_pretrained(cfg["output_dir"])
        self.log(f"✅ LoRA adapter saqlandi: {cfg['output_dir']}")
