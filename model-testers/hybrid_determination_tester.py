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
counts = {
    "offline_question": {"correct": 0, "failures": 0},
    "basic_question": {"correct": 0, "failures": 0},
    "complex_question": {"correct": 0, "failures": 0},
    "vision": {"correct": 0, "failures": 0},
    "explicit": {"correct": 0, "failures": 0}
}


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
        counts[question['label']]['correct'] += 1
        correct = True
    else:
        counts[question['label']]['failures'] += 1

    if not correct:
        print(
            f"{str("\033[92m" +"PASS" + "\033[0m") if correct else str("\033[91m" +"FAIL" + "\033[0m")} Question: {question['query']}"
            f"\n\t\tPredicted label: {labels[predicted_label]}"
            f"\n\t\tExpected label:  {question['label']}"
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
    f"\nPercent correct: {(total_correct / total_responses) * 100:.1f}%"
)

