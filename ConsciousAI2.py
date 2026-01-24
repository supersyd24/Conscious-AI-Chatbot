import os
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline, AutoTokenizer, AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, PeftConfig, TaskType
from sentence_transformers import SentenceTransformer, util
import torch.optim as optim
import datetime
import json

# -----------------------------
# Models & tokenizers
# -----------------------------
# Load GPT-2 model and tokenizer
base_model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Apply LoRA for fine-tuning
lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(base_model, lora_config)
model.train()
optimizer = optim.Adam(model.parameters(), lr=5e-5)

# Sentence embedder
embedder = SentenceTransformer('all-MiniLM-L6-v2') #For schema similarity

# Toxicity filter
moderation_pipeline = pipeline("text-classification", model="unitary/toxic-bert", top_k=None)

# Load emotion classifier
with open("emotion_model/label_map.json") as f:
    label_map = json.load(f)
id2label = {v: k for k, v in label_map.items()}

emotion_tokenizer = AutoTokenizer.from_pretrained("emotion_model")
emotion_model = AutoModelForSequenceClassification.from_pretrained("emotion_model")
emotion_model.eval()

def predict_emotion(text):
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = emotion_model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        return id2label.get(predicted_class, "unknown")

# -----------------------------
# Memory banks
# -----------------------------
medial_prefrontal_cortex = []

class EmotionalMemory:
    def __init__(self):
        self.entries = []

    def add_entry(self, user_input, emotion, schema_used):
        self.entries.append({
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_input": user_input,
            "emotion": emotion,
            "schema_used": schema_used
        })

    def show_entries(self):
        for entry in self.entries:
            print(f"{entry['timestamp']} | Emotion: {entry['emotion']} | Input: {entry['user_input']} | Schema: {entry['schema_used']}")

emotional_memory = EmotionalMemory()

# -----------------------------
# Novelty detection
# -----------------------------
NOVELTY_SIM_THRESHOLD = 0.75  # if best embedding similarity < this → treat as new
CANON = lambda s: " ".join(s.strip().lower().split())  # normalize whitespace/case

# Track exact inputs we've seen (canonicalized)
seen_inputs = set()

def is_novel_input(user_input: str, best_similarity: float, exact_match: bool) -> bool:
    """
    Returns True if this input should force creation of a new schema.
    Conditions:
      1) Not an exact string match we've saved before, AND
      2) Best cosine similarity below threshold OR we have never seen this exact text.
    """
    c = CANON(user_input)
    if exact_match:
        return False
    if c not in seen_inputs:
        return True
    return best_similarity < NOVELTY_SIM_THRESHOLD

# -----------------------------
# Core helpers
# -----------------------------
def schema_tracing(user_input):
    input_embedding = embedder.encode(user_input, convert_to_tensor=True)
    best_score = 0.0
    best_schema = None
    exact_match = False

    for schema in medial_prefrontal_cortex:
        similarity = float(util.pytorch_cos_sim(input_embedding, schema["embedding"]))
        if user_input.strip().lower() == schema["input"].strip().lower():
            similarity = 1.0
            exact_match = True
        if similarity > best_score:
            best_score = similarity
            best_schema = schema

    return best_score, best_schema, exact_match

def p_x_from_familiarity(score):
    return score if 0.4 < score < 0.85 else 0.0

def bayes_inference(P_X, P_H, P_not_H):
    numerator = P_X * P_H
    denominator = (P_X * P_H) + (P_not_H * (1 - P_H))
    return numerator / denominator if denominator != 0 else 0.0

def fuzzy_logic(P_H_given_E, P_not_H):
    mu_glutamate = P_H_given_E  # μGlutamate = P(H|E)
    mu_gaba = P_not_H           # μGABA = P(¬H)
    return mu_gaba, mu_glutamate

def compute_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy.mean().item()

def is_safe(response):
    results = moderation_pipeline(response)[0]
    for label in results:
        if label["label"].lower() in ["toxic", "insult", "threat"] and label["score"] > 0.5:
            return False
    return True

def save_schema(model, user_input, entropy, emotion="unknown"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    schema_name = f"schema_{timestamp}"
    save_path = f"schemas/{schema_name}"
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    medial_prefrontal_cortex.append({
        "input": user_input,
        "embedding": embedder.encode(user_input, convert_to_tensor=True),
        "entropy": entropy,
        "emotion": emotion,
        "name": schema_name,
        "path": save_path
    })
    # Mark the canonical input as seen
    seen_inputs.add(CANON(user_input))
    return schema_name

def load_peft_model(schema_path):
    peft_config = PeftConfig.from_pretrained(schema_path)
    base = GPT2LMHeadModel.from_pretrained(peft_config.base_model_name_or_path)
    return get_peft_model(base, LoraConfig.from_pretrained(schema_path))

# -----------------------------
# Main loop
# -----------------------------
def main():
    global model, optimizer
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        match_score, matched_schema, exact_match = schema_tracing(user_input)

        P_H = 1.0 - match_score
        P_not_H = match_score
        P_X = p_x_from_familiarity(match_score)
        P_H_given_E = bayes_inference(P_X, P_H, P_not_H)

        inputs = tokenizer(user_input, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model(**inputs, labels=inputs["input_ids"])
        entropy = compute_entropy(outputs.logits)
        mu_gaba, mu_glutamate = fuzzy_logic(P_H_given_E, P_not_H)

        schema_used = "None"
        novelty = is_novel_input(user_input, match_score, exact_match)

        if exact_match and matched_schema:
            # Perfect match → load the matched schema
            model = load_peft_model(matched_schema["path"])
            optimizer = optim.Adam(model.parameters(), lr=5e-5)
            schema_used = matched_schema["name"]

        elif novelty:
            # FORCE schema creation for unseen / low-similarity input
            model.train()
            pseudo_inputs = tokenizer("I'm here to support you.", return_tensors="pt")
            pseudo_inputs = {k: v.to(model.device) for k, v in pseudo_inputs.items()}
            loss = model(**pseudo_inputs, labels=pseudo_inputs["input_ids"]).loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            emotion = predict_emotion(user_input)
            schema_used = save_schema(model, user_input, entropy, emotion)
            emotional_memory.add_entry(user_input, emotion, schema_used)
            print(f"[Schema Created - NOVELTY] → Name: {schema_used}, Emotion: {emotion}, Entropy: {entropy:.4f}")

        elif mu_glutamate > mu_gaba:
            # High uncertainty / evidence for new hypothesis → create schema
            model.train()
            pseudo_inputs = tokenizer("I'm here to support you.", return_tensors="pt")
            pseudo_inputs = {k: v.to(model.device) for k, v in pseudo_inputs.items()}
            loss = model(**pseudo_inputs, labels=pseudo_inputs["input_ids"]).loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            emotion = predict_emotion(user_input)
            schema_used = save_schema(model, user_input, entropy, emotion)
            emotional_memory.add_entry(user_input, emotion, schema_used)
            print(f"[Schema Created] → Name: {schema_used}, Emotion: {emotion}, Entropy: {entropy:.4f}")

        elif mu_gaba > mu_glutamate and matched_schema:
            # Familiar enough → reuse closest schema
            model = load_peft_model(matched_schema["path"])
            optimizer = optim.Adam(model.parameters(), lr=5e-5)
            schema_used = matched_schema["name"]

        # Generate and safety-check response
        response_ids = model.generate(**inputs, max_new_tokens=30)
        response = tokenizer.decode(response_ids[0], skip_special_tokens=True)

        if is_safe(response):
            print("AI:", response)
        else:
            print("AI: I'm here to support you.")

if __name__ == "__main__":
    main()
