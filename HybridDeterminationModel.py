from .OllamaConnector import OllamaConnector


class HybridDeterminationModel:
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

    def __init__(self, ollama_connector: OllamaConnector = None):
        self.connector = ollama_connector if ollama_connector is not None else OllamaConnector()
        # self.connector = OllamaConnector()

    def determine(self, query: str) -> int:
        result = self.connector.send_query(query=query, model="HybridDetermination", stream=False)
        if result in self.bindings.keys():
            return self.bindings[result]
        else:
            print("FAIL!")
            return self.FAIL

    def test(self):
        queries = [
            "Analyze the causes and consequences of the 2008 global financial crisis. What role did subprime mortgages and financial derivatives play in the collapse, and how have regulations changed since then?",
            "Are the effects of caffeine worse than nicotine?",
            "Are you human?",
            "Are you dumb?",
            "Can you see any visual changes between these two art pieces?"
        ]

        for query in queries:
            self.determine(query=query)


if __name__ == '__main__':
    hddm = HybridDeterminationModel()
    hddm.test()
