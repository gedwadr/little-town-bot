import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

MODEL_PATH  = "./models/qwen2.5-1.5b-instruct"
DATA_PATH   = "./logs/trainLogs/output.jsonl"
OUTPUT_DIR  = "./checkpoints/boardgame-v1"
MAX_SEQ_LEN = 1024 + 256 + 128

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,                    # rank — higher = more capacity but more memory
    lora_alpha=32,           # scaling factor (usually 2x rank)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

dataset = load_dataset("json", data_files=DATA_PATH, split="train")

def format_example(example):
    """Convert {"messages": [...]} to a single string the model trains on."""
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

lengths = [len(tokenizer.apply_chat_template(ex["messages"], tokenize=True)) for ex in dataset]
print(f"max={max(lengths)} mean={sum(lengths)//len(lengths)} p95={sorted(lengths)[int(len(lengths)*0.95)]}")
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,           # 3-5 is typical for SFT
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,  # effective batch = 2*4 = 8
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    fp16=True,                    # use bf16=True if your GPU supports it (RTX 30xx+)
    logging_steps=10,
    save_strategy="epoch",
    max_seq_length=MAX_SEQ_LEN,
    gradient_checkpointing=True,        # ← trades compute for memory
    optim="paged_adamw_8bit",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    formatting_func=format_example,
)

print("training start...")
trainer.train()

model.save_pretrained(OUTPUT_DIR + "/final-adapter")
tokenizer.save_pretrained(OUTPUT_DIR + "/final-adapter")
print("Done. Adapter saved to", OUTPUT_DIR + "/final-adapter")
