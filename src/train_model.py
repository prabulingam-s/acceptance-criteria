from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments, T5Tokenizer
from datasets import load_from_disk

def train_model():
    # Load the tokenized dataset
    tokenized_dataset = load_from_disk('../data/processed/tokenized_dataset')

    # Load the pre-trained T5 model
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="../models/fine-tuned/fine-tuned-model",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-4,  # Slightly increased for better fine-tuning
        per_device_train_batch_size=4,  # Reduce batch size if memory is an issue
        per_device_eval_batch_size=4,
        num_train_epochs=15,  # Increased epochs if model was undertrained
        weight_decay=0.01,
        save_total_limit=2,
        logging_steps=10,
        report_to="none",
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model and tokenizer
    trainer.save_model("../models/fine-tuned/fine-tuned-model")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    tokenizer.save_pretrained("../models/fine-tuned/fine-tuned-model")

    # Debug: Print final loss
    eval_results = trainer.evaluate()
    print(f"Final Evaluation Loss: {eval_results['eval_loss']}")

    print("Model and tokenizer saved successfully.")

if __name__ == "__main__":
    train_model()
