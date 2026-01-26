# Conscious DV Chatbot 🧠💬  
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
