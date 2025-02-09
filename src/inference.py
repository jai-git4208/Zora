import torch
from transformers import AutoTokenizer
from src.model import ZoraModel
from src.reasoning import ZoraReasoning
from src.memory import ZoraMemory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = ZoraModel().to(device)
model.load_state_dict(torch.load("models/zora_model.pt"))
model.eval()

reasoning = ZoraReasoning(model)
memory = ZoraMemory()

def ask_zora(prompt):
    # Process input
    tokens = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    
    # Run through model
    with torch.no_grad():
        response, confidence = model(tokens["input_ids"], tokens["attention_mask"])
    
    # Self-check & refine
    refined_response = reasoning.self_check_response(response, memory.history)

    # Store interaction
    memory.store_interaction(prompt, refined_response)

    return refined_response

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    print(f"ZORA: {ask_zora(user_input)}")