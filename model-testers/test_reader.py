import csv


def load_queries(file_path):
    questions = []
    try:
        with open(file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if 'query' in row and 'label' in row:
                    question = {'query': row['query'], 'label': row['label']}
                    questions.append(question)
                else:
                    print(f"Missing 'query' or 'label' in row: {row}")
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    return questions

