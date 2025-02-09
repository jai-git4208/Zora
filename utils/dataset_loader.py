from datasets import load_dataset

def load_s1k_dataset():
    # Load the S1K dataset from Hugging Face
    dataset = load_dataset("s1k")
    
    # Return the train and validation sets
    return dataset["train"], dataset["validation"]