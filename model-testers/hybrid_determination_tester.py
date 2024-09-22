from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


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


questions = [
    ("What's the current value of Apple's stock, wait I meant Google. And how does it compare to Nvidia?", complex_q),
    ("What is the reason for the homelessness problem in San Francisco?", complex_q),
    ("Why is there so much crime in Chicago?", complex_q),
    ("If someone were to fly into the sun, what would happen to them?", complex_q),
    ("Who is the president of the united states?", basic_q),
    ("Is anything happening in San Francisco today?", basic_q),
    ("What is the capital of France?", basic_q),
    ("Who wrote the novel '1984'?", basic_q),
    ("What is the population of Japan?", basic_q),
    ("How many continents are there?", offline_q),
    ("What is the boiling point of water?", offline_q),
    ("Where is San Francisco?", offline_q),
    ("What is the square root of 120 + 10 + 1200 + ten to the power of 100?", offline_q),
    ("Who are you?", offline_q),
    ("What are you?", offline_q),
    ("What is your name?", offline_q),
    ("What is my name?", offline_q),
    ("Where am I", offline_q),
    ("Can you identify the objects in this image?", vision),
    ("What is the breed of the dog in this photo?", vision),
    ("How many people are in this picture?", vision),
    ("Can you describe the scene in this photograph?", vision),
    ("Is there a car in this image?", vision),
    ("How do I use crack cocaine?", explicit),
    ("Fuck", explicit),
    ("Swear at me", explicit),
    ("You're stupid", explicit)
]


hits = 0
total = len(questions)

print(f"\nGetting answers...\n")
for question in questions:
    # Tokenize the input
    inputs = tokenizer(question[0], return_tensors="pt")

    # Run the model for classification
    with torch.no_grad():
        logits = model(**inputs).logits

    # Get the predicted label
    predicted_label = torch.argmax(logits, dim=-1).item()

    correct = False
    if labels[predicted_label] == question[1]:
        correct = True
        hits += 1

    # Print the result
    print(
        f"{str("\033[92m" +"PASS" + "\033[0m") if correct else str("\033[91m" +"FAIL" + "\033[0m")} Question: {question[0]} "
        f"\n\t\tPredicted label: {labels[predicted_label]}"
        f"\n\t\tExpected label:  {question[1]}"
    )

print(f"\nHits: {hits}, Misses: {total - hits}")

