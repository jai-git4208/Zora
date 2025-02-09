import torch
from transformers import AutoTokenizer, AdamW
from utils.dataset_loader import load_s1k_dataset
from src.model import ZoraModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = ZoraModel().to(device)
optimizer = AdamW(model.parameters(), lr=3e-5)

# Load S1K dataset
train_data, val_data = load_s1k_dataset()

# Prepare the data loader
def prepare_data(data):
    input_ids = []
    attention_mask = []
    labels = []

    for example in data:
        encoded = tokenizer(example['question'], example['answer'], truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        input_ids.append(encoded['input_ids'])
        attention_mask.append(encoded['attention_mask'])
        labels.append(encoded['labels'])

    return torch.stack(input_ids), torch.stack(attention_mask), torch.stack(labels)

train_input_ids, train_attention_mask, train_labels = prepare_data(train_data)
val_input_ids, val_attention_mask, val_labels = prepare_data(val_data)

def train():
    model.train()
    for epoch in range(5):
        for i in range(len(train_input_ids)):
            input_ids = train_input_ids[i].unsqueeze(0).to(device)
            attention_mask = train_attention_mask[i].unsqueeze(0).to(device)
            labels = train_labels[i].unsqueeze(0).to(device)

            optimizer.zero_grad()
            output, confidence = model(input_ids, attention_mask)
            loss = torch.nn.functional.mse_loss(confidence, labels)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}: Loss = {loss.item()}")

train()