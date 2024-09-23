import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from .test_reader import load_queries

def test(passes=10):

    current_directory = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(current_directory, "../training/model-builds/hybrid-determination")
    test_path = os.path.join(current_directory, "./tests/determination_test_data.csv")

    # Load the fine-tuned model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Define labels
    labels = ['offline_question', 'basic_question', 'complex_question', 'vision', 'explicit']

    num_repeats = passes if passes >= 1 else 10
    questions = load_queries(test_path)
    counts = {
        "offline_question": {"correct": 0, "failures": 0},
        "basic_question": {"correct": 0, "failures": 0},
        "complex_question": {"correct": 0, "failures": 0},
        "vision": {"correct": 0, "failures": 0},
        "explicit": {"correct": 0, "failures": 0}
    }

    print(f"\nTesting..."
          f"\nPlease wait, this may take a while."
          f"\nRunning {len(questions)} tests {num_repeats} times ({len(questions) * num_repeats} iterations)."
          f"\n(I apologize if your computer sets on fire.)\n")
    for i in range(num_repeats):
        print(f"Pass {i + 1}\nStarting...", end='')
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
        print(f"Done.")
        print(f"Finished pass {i + 1} ({(i + 1) * len(questions)} tests complete).\n")


    print(f"\nResults:")
    for count in counts:
        print(
            f"\t{count}:"
            f"\n\t\t{str("\033[92m" +"CORRECT" + "\033[0m")}: {counts[count]['correct']}"
            f"\n\t\t{str("\033[91m" +"FAILURES" + "\033[0m")}: {counts[count]['failures']}"
        )

    total_correct = sum(count["correct"] for count in counts.values())
    total_failures = sum(count["failures"] for count in counts.values())
    total_responses = total_correct + total_failures

    print(
        f"Cumulative Results:"
        f"\n\tTotal Correct: {total_correct}"
        f"\n\tTotal Failures: {total_failures}"
        f"\n\tAccuracy: {(total_correct / total_responses) * 100:.1f}%"
        f"\t{str("\033[92m" +"ACCEPTABLE" + "\033[0m") if (total_correct / total_responses) >= .98
        else str("\033[91m" +"CONTINUE TRAINING" + "\033[0m")}"
    )


if __name__ == '__main__':
    test()