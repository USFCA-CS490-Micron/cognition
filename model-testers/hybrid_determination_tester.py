from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from test_reader import load_queries

# Load the fine-tuned model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("../training/model-builds/hybrid-determination-model")
tokenizer = AutoTokenizer.from_pretrained("../training/model-builds/hybrid-determination-model")

# Define labels
labels = ['offline_question', 'basic_question', 'complex_question', 'vision', 'explicit']

questions = load_queries("./tests/determination_test_data.csv")
counts = {
    "offline_question": {"correct": 0, "failures": 0},
    "basic_question": {"correct": 0, "failures": 0},
    "complex_question": {"correct": 0, "failures": 0},
    "vision": {"correct": 0, "failures": 0},
    "explicit": {"correct": 0, "failures": 0}
}


print(f"\nTesting...")
for question in questions:
    # Tokenize the input
    inputs = tokenizer(question['query'], return_tensors="pt")

    # Run the model for classification
    with torch.no_grad():
        logits = model(**inputs).logits

    # Get the predicted label
    predicted_label = torch.argmax(logits, dim=-1).item()

    predicted_label_str = labels[predicted_label]
    expected_label = question['label'].strip()

    correct = False
    if predicted_label_str == expected_label:
        counts[expected_label]['correct'] += 1
        correct = True
    else:
        counts[expected_label]['failures'] += 1

    if not correct:
        print(
            f"\n{str("\033[92m" +"PASS" + "\033[0m") if correct else str("\033[91m" +"FAIL" + "\033[0m")} Question: {question['query']}"
            f"\n\t\tPredicted label: {predicted_label_str}"
            f"\n\t\tExpected label:  {expected_label}"
        )


print()
for count in counts:
    print(
        f"{count}:"
        f"\n\t{str("\033[92m" +"CORRECT" + "\033[0m")}: {counts[count]['correct']}"
        f"\n\t{str("\033[91m" +"FAILURES" + "\033[0m")}: {counts[count]['failures']}"
    )

total_correct = sum(count["correct"] for count in counts.values())
total_failures = sum(count["failures"] for count in counts.values())
total_responses = total_correct + total_failures

print(
    f"\nTotal Correct: {total_correct}"
    f"\nTotal Failures: {total_failures}"
    f"\nAccuracy: {(total_correct / total_responses) * 100:.1f}%"
)

