from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import json

df = pd.read_csv("emotion_data.csv")
label_list = df["emotion"].unique().tolist()
label2id = {label: idx for idx, label in enumerate(label_list)}
df["label"] = df["emotion"].map(label2id)

# Save label map
with open("emotion_model/label_map.json", "w") as f:
    json.dump(label2id, f)

dataset = Dataset.from_pandas(df[["input", "label"]]).train_test_split(test_size=0.2)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["input"], padding=True, truncation=True)

tokenized = dataset.map(tokenize, batched=True)
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(label_list))

args = TrainingArguments(
    output_dir="emotion_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
)

trainer.train()
model.save_pretrained("emotion_model")
tokenizer.save_pretrained("emotion_model")
