from transformers import AutoTokenizer
import re

class ZoraTokenizer:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(self, text):
        return self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    def split_question(self, text):
        # Uses simple heuristics to split multi-part questions
        return re.split(r'(\?|and|or|,)', text)

tokenizer = ZoraTokenizer()