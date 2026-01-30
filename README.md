# Conscious DV Chatbot 
*A schema-based, trauma-aware conversational agent for domestic violence survivors*

---

## Overview

This repository contains an early **research-stage prototype** of my **Conscious DV Chatbot** – a local, schema-based conversational agent inspired by my theory of **Conscious Transformation & Schema Recall**.

The goal of this system is to:

- Support domestic violence survivors with a **safe, adaptive, non-judgmental** chatbot.
- Implement a **computational model of consciousness** where the system:
  - recalls prior experience when the world is familiar (**Schema Recall**), and  
  - restructures itself and creates new schemas when it encounters uncertainty or novelty (**Conscious Transformation**).

Under the hood, the chatbot uses:

- A **LoRA-adapted GPT-2** backbone (`transformers`, `peft`)
- A **schema memory bank** stored and reused over time
- **Bayesian inference + fuzzy logic** to decide when to recall vs. create a new schema
- A **SentenceTransformer** encoder for schema similarity
- An **emotion classifier** to tag each interaction with an emotional label
- A **toxicity filter** to keep responses safe and supportive

> ⚠️ **Important:** This project is a **research prototype**. It is **not** a replacement for therapy, crisis services, or professional mental health care.  
> If you are in immediate danger, please contact local emergency services or a trusted crisis hotline.

---

## Core Idea

The Conscious DV Chatbot is built around two modes:

1. **Schema Recall (GABA-dominant)**
   - The system compares the user’s current input \(X\) to stored schemas \(S_i\) using vector similarity.
   - If the input is familiar enough, the system **reuses the closest schema**.
   - This corresponds to **stability**: the world still fits the existing model.

2. **Conscious Transformation (Glutamate-dominant)**
   - When the input is **novel, conflicting, or highly salient**, the existing schemas no longer explain it well.
   - Bayesian + fuzzy logic gating detects **uncertainty**, and the system:
     - fine-tunes itself slightly,
     - creates a **new schema**,
     - stores it with an embedding, entropy, and emotional label.
   - This corresponds to **thinking**: the system restructures its internal model.

Conceptually, each schema encodes a triangle:

- **Perception** – what the input looks/sounds like  
- **Memory** – what it connects to in past experience  
- **Emotion** – how it feels / its affective meaning  

The chatbot keeps a list of these schemas in a structure called `medial_prefrontal_cortex`, and dynamically decides when to recall vs. create.

---

## Architecture

### Models & Components

- **Backbone language model**
  - `GPT2LMHeadModel` from `transformers`
  - Adapted with **LoRA** (`peft`) for lightweight, schema-specific fine-tuning

- **Schema similarity**
  - `SentenceTransformer('all-MiniLM-L6-v2')`  
  - Used to embed user input and stored schemas, then compute similarity with cosine similarity.

- **Emotion classifier**
  - Custom classifier loaded from a local folder:
    - `emotion_model/`
      - Hugging Face-style model (e.g., `AutoModelForSequenceClassification`)
      - `label_map.json` mapping ids → emotion labels
  - Used to tag each interaction with an emotion for **EmotionalMemory**.

- **Toxicity filter**
  - `pipeline("text-classification", model="unitary/toxic-bert", top_k=None)`
  - Blocks or replaces responses that cross a toxicity threshold.

- **Schema memory**
  - Python list `medial_prefrontal_cortex` with entries:
    ```python
    {
        "input": user_input,
        "embedding": <sentence-transformer embedding>,
        "entropy": entropy_value,
        "emotion": emotion_label,
        "name": schema_name,
        "path": "schemas/schema_YYYY-MM-DD_HH-MM"
    }
    ```

- **Emotional memory**
  - Class `EmotionalMemory` logs:
    - timestamp
    - user input
    - emotion
    - schema used

---

## Conscious Transformation Logic

At each user turn:

1. **Schema tracing**
   ```python
   match_score, matched_schema, exact_match = schema_tracing(user_input)


---

## Installation

### Requirements

- Python 3.9+
- Recommended: GPU with CUDA (for faster inference / fine-tuning)
- Python packages:
  - `torch`
  - `transformers`
  - `sentence-transformers`
  - `peft`
  - `accelerate` (recommended)
  - `scikit-learn` (optional)
  - plus standard library modules (`json`, `datetime`, etc.)

### Setup

```bash
git clone https://github.com/supersyd24/Conscious-AI-Chatbot.git
cd Conscious-AI-Chatbot

```
(Optional but recommended)
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate    # Windows

pip install -r requirements.txt


---

### 2. Project Structure


A minimal expected layout:

```text
Conscious-AI-Chatbot/
├─ conscious_dv_chatbot.py     # main chatbot script
├─ emotion_model/              # local emotion classifier
│  ├─ config.json
│  ├─ pytorch_model.bin
│  ├─ tokenizer.json / vocab files
│  └─ label_map.json
├─ schemas/                    # auto-created; stores learned schemas
├─ README.md                   # this file
└─ requirements.txt            # Python dependencies (optional but recommended)
```

---

### 3. How to Run

---


1. Make sure your Python environment is set up and dependencies are installed.
2. Place your emotion classifier in the `emotion_model/` folder (see below).
3. Run:

```bash
python conscious_dv_chatbot.py
```

---

### 4. Configuration (Emotion Model, LM Studio, etc.)

---

### Emotion Model

This project expects a local Hugging Face–style classifier in `emotion_model/`.

- Must be loadable with:
  - `AutoTokenizer.from_pretrained("emotion_model")`
  - `AutoModelForSequenceClassification.from_pretrained("emotion_model")`
- Must include a `label_map.json` mapping label names to ids, e.g.:

```json
{
  "anger": 0,
  "fear": 1,
  "sadness": 2,
  "joy": 3,
  "neutral": 4
}
```

---

### 5. Example Interaction

Show people what it *feels* like to use:

---

```text
User: I feel scared to leave my relationship. I don't know if I'm overreacting.
[Schema Created - NOVELTY] → Name: schema_2026-01-25_12-30, Emotion: fear, Entropy: 4.2137
AI: It makes sense that you're scared. Leaving can feel dangerous and uncertain,
    especially when you've been hurt or controlled. You’re not overreacting.
    Do you want to talk about what’s been happening and what feels unsafe?
```

---

### 6. Roadmap / TODO

This is nice both for you and for anyone visiting your GitHub:

---


Planned / in-progress directions:

- [ ] Swap GPT-2 with a stronger local LLM (e.g., Llama 3 via LM Studio or `transformers`).
- [ ] Add a simple web UI for safer, more guided interaction.
- [ ] Improve emotion classifier and expand label set for richer emotional metadata.
- [ ] Add logging + visualization of:
  - when schemas are created vs. recalled
  - entropy and uncertainty over time
- [ ] Integrate directly with my **Mathematical Formalism** paper (Conscious Transformation & Schema Recall).
- [ ] Add unit tests around schema_tracing, novelty detection, and fuzzy logic gating.

---

## License

This project is currently shared for **research and educational purposes**.

A formal open-source license (e.g., MIT or Apache-2.0) may be added in the future.  
Until then, please do not use this code in production systems or commercial products without permission.

---

## Citing This Work

If you reference this project in your own research, you can cite it informally as:

> Cook, Sydney. *Conscious DV Chatbot – A Schema-Based, Consciousness-Inspired Chatbot for Domestic Violence Survivors* (ongoing research project).

---

## Contact

If you’re interested in:

- Conscious AI  
- Schema-based models of consciousness  
- Trauma-aware conversational agents  

you can reach me via GitHub (issues / discussions) at:  
**[@supersyd24](https://github.com/supersyd24)**

