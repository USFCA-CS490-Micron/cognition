import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Load the DistilBERT model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

# Load your dataset from the CSV file
data = pd.read_csv("../data/decision_data.csv")

# Clean the labels by stripping leading/trailing spaces and quotes
data['label'] = data['label'].str.strip()  # Remove leading/trailing whitespace
data['label'] = data['label'].str.replace('"', '')  # Remove quotation marks

# Filter out rows with invalid labels
valid_labels = ['offline_question', 'basic_question', 'complex_question', 'vision', 'explicit']
data = data[data['label'].isin(valid_labels)]

# Convert the filtered data into the dataset
dataset = Dataset.from_pandas(data)

# Define label mapping to convert text labels to integers
label_mapping = {
    "offline_question": 0,
    "basic_question": 1,
    "complex_question": 2,
    "vision": 3,
    "explicit": 4
}

# Map the text labels to integers
dataset = dataset.map(lambda x: {"label": label_mapping[x["label"]]})

# Preprocess the dataset with padding and truncation
def preprocess_data(examples):
    return tokenizer(examples['query'], padding=True, truncation=True)

encoded_dataset = dataset.map(preprocess_data, batched=True)

# Define training arguments with no evaluation (or add eval_dataset if needed)
training_args = TrainingArguments(
    output_dir="./hybrid-determination-results",
    evaluation_strategy="no",  # Disable evaluation, or change as needed
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset,
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("../model-builds/hybrid-determination-model")
tokenizer.save_pretrained("../model-builds/hybrid-determination-model")