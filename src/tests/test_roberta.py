#dont load tensorflow and flax to speed up load times
import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
from transformers import pipeline

classifier = pipeline("text-classification", model="./roberta_news_classifier")
print(classifier("Stock prices fall as interest rates rise"))