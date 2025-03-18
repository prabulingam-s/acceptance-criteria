from transformers import T5ForConditionalGeneration, T5Tokenizer
import os
import torch

def generate_acceptance_criteria(description):
    # Load the fine-tuned model and tokenizer
    model_path = "../models/fine-tuned/fine-tuned-model"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Fine-tuned model not found at {model_path}. Ensure training completed successfully.")

    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path, model_max_length=512)

    model.eval()  # Ensure model is in evaluation mode

    # If using GPU, move model to CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Prepare input prompt
    input_text = f"generate acceptance criteria in English with Scenario, Given, When, Then format: {description}"
    print("Input Prompt:", input_text)

    # Tokenize input
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

    # Generate output using beam search
    outputs = model.generate(
        input_ids,
        max_length=150,  # Adjusted to a reasonable max length
        min_length=50,  # Prevent short, incomplete outputs
        num_beams=5,  # Beam search for better structured results
        early_stopping=True,
        no_repeat_ngram_size=3,  # Prevent excessive repetition
    )

    # Decode the generated output
    acceptance_criteria = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"Generated Output: {acceptance_criteria}")

    return acceptance_criteria

if __name__ == "__main__":
    # Example test description
    description = "As a data engineer, I want to create an external Hive table to read data stored in S3 without moving it"

    # Generate acceptance criteria
    criteria = generate_acceptance_criteria(description)
    print("Generated Acceptance Criteria:")
    print(criteria)
