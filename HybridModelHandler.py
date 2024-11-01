from .HybridDeterminationModel import HybridDeterminationModel
from .OllamaConnector import OllamaConnector
from .OpenAIConnector import OpenAIConnector


class HybridModelHandler:
    OFFLINE = 0
    BASIC = 1
    COMPLEX = 2
    VISION = 3
    EXPLICIT = 4
    FAIL = -1

    bindings = {
        "offline_question": OFFLINE,
        "basic_question": BASIC,
        "complex_question": COMPLEX,
        "vision": VISION,
        "explicit": EXPLICIT
    }

    rev_bindings = {
        OFFLINE: "offline_question",
        BASIC: "basic_question",
        COMPLEX: "complex_question",
        VISION: "vision",
        EXPLICIT: "explicit"
    }

    model_bindings = {
        OFFLINE: "Local Llama 3.2 Finetuned QA",
        BASIC: "OpenAI GPT 4o-mini",
        COMPLEX: "OpenAI GPT 4o",
        VISION: "Google Cloud Vision (not yet implemented)",
        EXPLICIT: "Query bounced"
    }

    def __init__(self):
        self.model = HybridDeterminationModel()
        self.ollama = OllamaConnector()
        self.openai = OpenAIConnector()

    def query(self, query: str):
        kind = self.model.determine(query)

        print(f"HybridModelHandler received query: {query}\n"
              f"Binding assigned: {self.rev_bindings[kind]}\n"
              f"Sending query to provider: {self.model_bindings[kind]}")

        if kind == self.OFFLINE:
            return self.ollama.send_query(query, "llama_qa", stream=False)
        elif kind == self.BASIC:
            return self.openai.send_query(query, model="4o-mini")
        elif kind == self.COMPLEX:
            return self.openai.send_query(query, model="4o-mini")
        elif kind == self.VISION:
            return "Not yet supported."
        elif kind == self.EXPLICIT:
            return "I'm sorry, I will not answer that."


if __name__ == '__main__':
    hmh = HybridModelHandler()
    hmh.query("Hello!")
