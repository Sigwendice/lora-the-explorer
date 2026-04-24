# 🧭 lora-the-explorer

**¡Come on, vámonos! Let's explore LLM fine‑tuning!**  
An expedition into Low‑Rank Adaptation (LoRA) for large language models — lightweight, fast, and fun.  
*Swiper, no swiping! We don’t steal parameters — we adapt them.* 🦊

<img src="./images/lora-the-explorer.jpg" alt="lora-the-explorer logo" width="30%">

![exploring](https://img.shields.io/badge/status-exploring-brightgreen?style=flat&logo=mapbox)

---

## 🌍 About the Expedition

This project is your backpack 🎒 for LoRA fine‑tuning demos with Large Language Models.  
It’s built as a **teaching & learning example** — just swap in your own dataset and you’re off to fine‑tune on *your* data.

No GPU‑melting required. We pack light: a small adapter that fits in your pocket.

---

## 📦 What's Inside?

- **Minimal** fine‑tuning script with HuggingFace `Trainer`
- **Interactive** chat inference to test your model
- **Plug‑and‑play** data format (just replace `data.json`)
- **BF16** mixed‑precision training for speed & memory
- **Pre‑configured** LoRA hyper‑params that work on a single GPU

---

## 🚀 Getting Started

### 1️⃣ Clone & Enter the Backpack

```bash
git clone https://github.com/your-username/lora-the-explorer.git
cd lora-the-explorer
```

### 2️⃣ Install Dependencies (Pack Your Gear)

```bash
pip install -r requirements.txt
```

<details>
<summary>📋 What’s in requirements.txt?</summary>

```
torch>=2.0.0
transformers>=4.45.0
peft>=0.12.0
accelerate>=0.33.0
datasets
```
</details>

### 3️⃣ Prepare Your Map (Data)

Place your dataset as `data.json` in the root folder, or edit `train_lora.py` to point to your own path.  
Format:

```json
[
    {
        "instruction": "Translate the following sentence to Spanish",
        "input": "Where is the nearest library?",
        "output": "¿Dónde está la biblioteca más cercana?",
        "id": 1
    }
]
```

### 4️⃣ Start the Expedition!

```bash
python train_lora.py
```

You’ll see training logs in your terminal. The adapter will be saved into `./output/`.

---

## 💬 Chat with Your Model

```bash
# Single question
python inference.py "What's the capital of France?"

# Interactive chat mode (type 'exit' to quit)
python inference.py --chat
```

Example chat:

```
You: Translate to Spanish: Good morning!
🤖 Model: ¡Buenos días!
```

*(📸 You can add a screenshot of the interactive chat here)*

---

## 🧭 LoRA Compass Settings (Hyperparameters)

| Setting          | Value                          | Meaning                              |
|------------------|--------------------------------|--------------------------------------|
| `r` (rank)       | 4                              | Width of the low‑rank path           |
| `lora_alpha`     | 8                              | Scaling factor for the update        |
| `lora_dropout`   | 0.1                            | Helps avoid overfitting trolls 👹    |
| `target_modules` | `q_proj,k_proj,v_proj,o_proj` | Attention layers to adapt            |
| `bf16`           | `True`                         | Mixed precision for efficient travel |

You can easily modify these at the top of `train_lora.py`.

---

## 🌳 Project Structure

```
lora-the-explorer/
├── train_lora.py          # 🧭 LoRA fine‑tuning script
├── inference.py           # 💬 Interactive chat script
├── data.json              # 🗺️ Your training data (demo included)
├── requirements.txt       # 🎒 Dependencies list
├── images/                # 🖼️ Mascot & illustrations
│   └── lora-the-explorer.jpg
├── output/                # 🎒 Trained LoRA adapters saved here
└── README.md              # 📖 You are here!
```

---

## 🔧 Customisation Tips

- **Change the model**: edit `model_name` in `train_lora.py` (any CausalLM from 🤗 Hub)
- **Use your own data**: replace `data.json` or modify the `load_dataset` call
- **Play with LoRA rank**: `r=8` often boosts quality but increases weight size
- **Add validation**: set `evaluation_strategy="steps"` and a validation split

---

## 🦜 Credits & Inspiration

- This project was inspired by the LoRA paper: [LoRA: Low‑Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- Mascot style: *Dora the Explorer* – always ready to explore new data!
- Built with ❤️ and [🤗 Transformers](https://huggingface.co/docs/transformers) + [PEFT](https://huggingface.co/docs/peft)

---

## 📜 License

MIT © Atrio

---

**Now, vámonos! 🌟**

> *“If you get lost during fine‑tuning, just ask the map!”*