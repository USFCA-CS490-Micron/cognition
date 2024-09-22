from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import test_reader


# Load the fine-tuned model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("../training/model-builds/hybrid-determination-model")
tokenizer = AutoTokenizer.from_pretrained("../training/model-builds/hybrid-determination-model")

offline_q = 'offline_question'
basic_q = 'basic_question'
complex_q = 'complex_question'
vision = 'vision'
explicit = 'explicit'

# Define labels
labels = [offline_q, basic_q, complex_q, vision, explicit]

questions = test_reader.load_queries("./tests/determination.csv")

hits = 0
total = len(questions)

print(f"\nGetting answers...\n")
for question in questions:
    # Tokenize the input
    inputs = tokenizer(question['query'], return_tensors="pt")

    # Run the model for classification
    with torch.no_grad():
        logits = model(**inputs).logits

    # Get the predicted label
    predicted_label = torch.argmax(logits, dim=-1).item()

    correct = False
    if labels[predicted_label] == question['label']:
        correct = True
        hits += 1

    # Print the result
    print(
        f"{str("\033[92m" +"PASS" + "\033[0m") if correct else str("\033[91m" +"FAIL" + "\033[0m")} Question: {question['query']}"
        f"\n\t\tPredicted label: {labels[predicted_label]}"
        f"\n\t\tExpected label:  {question['label']}"
    )

print(f"\nHits: {hits}, Misses: {total - hits}")

