import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

def download_and_load_model(model_name="t5-small", device =torch.device("mps") ):
    """
    Downloads and loads a T5 model and its tokenizer.
    
    Args:
        model_name (str): Pretrained T5 model name.
        
    Returns:
        model: Loaded T5 model
        tokenizer: Corresponding tokenizer
        device: torch.device used
    """

    print(f"Downloading and loading the {model_name} model...")

    try:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
    except Exception as e:
        print("Error loading tokenizer:", e)
        raise

    try:
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        model.to(device)
    except Exception as e:
        print("Error loading model:", e)
        raise

    print("Model and tokenizer loaded successfully.")
    return model, tokenizer, device
