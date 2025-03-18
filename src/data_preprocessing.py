import csv
from datasets import Dataset
from transformers import T5Tokenizer

def preprocess_data():
    # Load the dataset
    data = []
    with open('../data/raw/jira_dataset.csv', mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='|')
        for row in reader:
            data.append({
                'Description': f"generate acceptance criteria in English with Scenario, Given, When, Then format: {row['Description']}",
                'Acceptance Criteria': row['Acceptance Criteria'] + " </s>",  # Ensure EOS token
            })

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_dict({
        'Description': [row['Description'] for row in data],
        'Acceptance Criteria': [row['Acceptance Criteria'] for row in data],
    })

    # Split dataset into train (80%) and test (20%)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    # Initialize the tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # Tokenization function
    def tokenize_function(examples):
        inputs = tokenizer(examples['Description'], max_length=512, truncation=True, padding="max_length")
        targets = tokenizer(examples['Acceptance Criteria'], max_length=512, truncation=True, padding="max_length")

        labels = targets["input_ids"]
        labels = [[-100 if token == tokenizer.pad_token_id else token for token in seq] for seq in labels]  # Fix pad token issue

        return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'], 'labels': labels}

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Debugging: Check tokenization results
    print("Sample Tokenized Input:", tokenizer.decode(tokenized_dataset['train'][0]['input_ids'], skip_special_tokens=True))
    print("Sample Tokenized Output:", tokenizer.decode(
        [t for t in tokenized_dataset['train'][0]['labels'] if t != -100], skip_special_tokens=True))

    # Save the tokenized dataset
    tokenized_dataset.save_to_disk('../data/processed/tokenized_dataset')
    print("Dataset preprocessed and saved successfully.")

if __name__ == "__main__":
    preprocess_data()
