#this script is for testing the roberta_news_classifier
#loads the dataset and generates a confusion matrix after testing

#dont load tensorflow and flax to speed up load times
import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import pandas as pd
import torch
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

MODEL_PATH = "../roberta_news_classifier"
DATA_PATH = "../data/News_Category_Dataset_v3.json"  # Ensure this path is correct

print("Loading and splitting data...")
df = pd.read_json(DATA_PATH, lines=True)
df['text'] = df['headline'] + " " + df['short_description']

mapping = {
    'THE WORLDPOST': 'WORLD NEWS', 'WORLDPOST': 'WORLD NEWS',
    'HEALTHY LIVING': 'WELLNESS', 'FIFTY': 'WELLNESS',
    'PARENTS': 'PARENTING', 'TASTE': 'FOOD & DRINK',
    'MONEY': 'BUSINESS', 'GREEN': 'ENVIRONMENT', 'ENVIRONMENT': 'SCIENCE'
}
df['category'] = df['category'].replace(mapping)

target_categories = [
    'POLITICS', 'WELLNESS', 'ENTERTAINMENT', 'TRAVEL',
    'STYLE & BEAUTY', 'FOOD & DRINK', 'BUSINESS',
    'COMEDY', 'SPORTS', 'WORLD NEWS', 'TECH', 'SCIENCE'
]
df = df[df['category'].isin(target_categories)].copy()

train_df, val_df = train_test_split(
    df[['text', 'category']], # We use the text category name for direct comparison
    test_size=0.2,
    stratify=df['category'],
    random_state=42
)

print("Loading RoBERTa model...")
classifier = pipeline(
    "text-classification",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    device=0 if torch.cuda.is_available() else -1,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

print(f"Running inference on {len(val_df)} samples...")
val_texts = val_df['text'].tolist()
y_true = val_df['category'].tolist()

results = classifier(val_texts, truncation=True, max_length=128, batch_size=32)
y_pred = [res['label'] for res in results]

print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=target_categories))

print("Generating Confusion Matrix...")
fig, ax = plt.subplots(figsize=(12, 10))
cm = confusion_matrix(y_true, y_pred, labels=target_categories)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_categories)

disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("RoBERTa Confusion Matrix: News Category Classification")
plt.tight_layout()

plt.savefig("confusion_matrix.png")
print("Confusion matrix saved as 'confusion_matrix.png'")
plt.show()