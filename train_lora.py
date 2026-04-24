"""
LoRA微调训练脚本
用于对Qwen2.5-1.5B-Instruct模型进行LoRA微调
"""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset, DatasetDict
import os

# 禁用HF离线模式，确保能正常下载模型和数据
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 打印CUDA相关信息
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

# ========== 配置参数 ==========
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # 预训练模型名称
DATA_PATH = "./data.json"  # 训练数据JSON文件路径
OUTPUT_DIR = "./output/lora_train"  # 模型输出目录

BATCH_SIZE = 2  # 批次大小
MAX_LENGTH = 512  # 最大序列长度
GRADIENT_ACCUMULATION = 4  # 梯度累积步数
EPOCHS = 2  # 训练轮数
LEARNING_RATE = 5e-5  # 学习率


def load_data(path):
    """
    加载JSON格式的训练数据
    
    Args:
        path: JSON文件路径
        
    Returns:
        list: 训练数据列表
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def preprocess_function(examples, tokenizer, max_length):
    """
    对训练数据进行预处理，将指令、输入、输出拼接并tokenize
    
    Args:
        examples: 数据批次，包含instruction、input、output字段
        tokenizer: 分词器
        max_length: 最大序列长度
        
    Returns:
        dict: 包含input_ids、attention_mask、labels的tokenized数据
    """
    prompts = []
    for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
        if input_text:
            # 有输入时: Instruction: xxx\nInput: xxx\nOutput: xxx
            text = f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output}{tokenizer.eos_token}"
        else:
            # 无输入时: Instruction: xxx\nOutput: xxx
            text = f"Instruction: {instruction}\nOutput: {output}{tokenizer.eos_token}"
        prompts.append(text)

    # tokenize并padding到固定长度
    tokenized = tokenizer(prompts, max_length=max_length, truncation=True, padding="max_length")
    # labels与input_ids相同，模型需要学习预测input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def main():
    """主训练流程"""
    # 1. 加载训练数据
    print("Loading data...")
    data = load_data(DATA_PATH)
    print(f"Loaded {len(data)} samples")

    # 2. 划分训练集和验证集(9:1划分)
    dataset = Dataset.from_list(data)
    splits = dataset.train_test_split(test_size=0.1, seed=42)
    dataset = DatasetDict({
        "train": splits["train"],
        "eval": splits["test"]
    })

    # 3. 加载tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right"
    )
    # 确保pad_token存在
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. 加载预训练模型(使用bf16精度)
    print("Loading model with bf16...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # 5. 配置LoRA参数并应用到模型
    print("Applying LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # 任务类型:因果语言模型
        r=4,  # LoRA rank维度
        lora_alpha=8,  # LoRA缩放系数
        lora_dropout=0.1,  # Dropout比例
        # 目标模块: QKV和输出投影层
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",  # 不训练bias
        inference_mode=False  # 训练模式
    )

    # 应用LoRA
    model = get_peft_model(model, lora_config)
    # 打印可训练参数信息
    model.print_trainable_parameters()

    # 6. 定义tokenize函数
    def tokenize_function(examples):
        return preprocess_function(examples, tokenizer, MAX_LENGTH)

    # 7. 对训练集和验证集进行tokenize
    print("Tokenizing dataset...")
    tokenized_train = dataset["train"].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing train"
    )
    tokenized_eval = dataset["eval"].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["eval"].column_names,
        desc="Tokenizing eval"
    )

    # 8. 配置训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,  # 梯度累积
        learning_rate=LEARNING_RATE,
        logging_steps=10,  # 日志打印频率
        save_steps=100,  # 保存频率
        save_total_limit=2,  # 最多保存2个checkpoint
        eval_strategy="steps",  # 验证策略
        eval_steps=100,
        warmup_steps=20,  # 热身步数
        bf16=True,  # 启用bf16精度
        fp16=False,
        report_to="none",  # 不上报到任何远程记录服务
        ddp_find_unused_parameters=False,  # DDP相关
        gradient_checkpointing=False,  # 梯度检查点，节省显存
        optim="adamw_torch",  # 优化器
        max_grad_norm=1.0,  # 最大梯度范数
        weight_decay=0.1,  # 权重衰减
        remove_unused_columns=False,
    )

    # 9. 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        processing_class=tokenizer,
    )

    # 10. 开始训练
    print("Starting training...")
    trainer.train()

    # 11. 保存模型和tokenizer
    print("Saving model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()