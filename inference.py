# 推理脚本：加载微调后的LoRA模型进行推理
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_PATH = "./output/lora_train_1/checkpoint-500"  # LoRA权重路径
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # 基础模型
MAX_LENGTH = 512  # 生成最大token数

print(MODEL_PATH)


def load_model():
    """加载基础模型、tokenizer和LoRA权重"""
    print("Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    
    print("Loading LoRA weights...")
    # 使用PeftModel加载LoRA适配器
    model = PeftModel.from_pretrained(
        base_model,
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    model.eval()
    return model, tokenizer


def generate(instruction, input_text="", max_length=MAX_LENGTH):
    """单条指令生成"""
    model, tokenizer = load_model()
    
    if input_text:
        text = f"Instruction: {instruction}\nInput: {input_text}\nOutput:"
    else:
        text = f"Instruction: {instruction}\nOutput:"
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    
    return response.strip()


def chat():
    """交互式聊天模式"""
    model, tokenizer = load_model()
    print("\n=== Chat Started ===")
    print("Type 'quit' to exit\n")
    
    while True:
        instruction = input("User: ").strip()
        if instruction.lower() == "quit":
            break
        
        if not instruction:
            continue
        
        text = f"Instruction: {instruction}\nOutput:"
        
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        print(f"Assistant: {response}\n")


if __name__ == "__main__":
    import sys
    # 命令行参数: python inference.py [instruction] [input] 或 python inference.py --chat
    if len(sys.argv) > 1:
        if sys.argv[1] == "--chat":
            chat()
        else:
            instruction = sys.argv[1]
            input_text = sys.argv[2] if len(sys.argv) > 2 else ""
            result = generate(instruction, input_text)
            print(result)
    else:
        test_instruction = "如果被告人不服判决，有什么权利？"
        result = generate(test_instruction)
        print(f"Q: {test_instruction}")
        print(f"A: {result}")
