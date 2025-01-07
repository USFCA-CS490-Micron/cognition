def print_transcription(transcription: str):
    print_headline(label="Transcription", text=transcription)


def print_response(response: str):
    print_headline(label="Response", text=response)


def print_headline(label: str, text: str):
    print(f"{"=" * 24}\n"
          f"{label}: {text}\n"
          f"{"=" * 24}"
          )
