import torch
import torch.nn as nn
from transformers import AutoModel

class ZoraModel(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super(ZoraModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.fc = nn.Linear(self.encoder.config.hidden_size, self.encoder.config.hidden_size)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)  # Confidence score
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        hidden_state = self.fc(outputs.last_hidden_state[:, 0, :])
        confidence = torch.sigmoid(self.classifier(hidden_state))  # Confidence score
        return hidden_state, confidence