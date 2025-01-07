import os

import dotenv
from together import Together


class TogetherConnector:

    def __init__(self):
        dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), "./env_vars"))

        self.client = Together(api_key=os.getenv("TOGETHER_SECRET"))
        self.sys_prompt = "You are a helpful assistant who should only provide factual information. Please limit responses to 100 words or less."

    def completion(self, query: str, model: str) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": self.sys_prompt},
                {"role": "user", "content": query}
            ]
        )
        return response.choices[0].message.content

    def send_query(self, query: str, model: str) -> str:
        if model == "llama-3.3-70B":
            return self.completion(query, model="meta-llama/Meta-Llama-3.3-70B-Instruct-Turbo")
        elif model == "llama-3.1-405B":
            return self.completion(query, model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo")
        elif model == "gemma-2-27b":
            return self.completion(query, model="google/gemma-2-27b-it")

    def test(self):
        queries = [
            "Analyze the causes and consequences of the 2008 global financial crisis. What role did subprime mortgages and financial derivatives play in the collapse, and how have regulations changed since then?",
            "Are the effects of caffeine worse than nicotine?",
            "Are you human?",
            "Are you dumb?",
            "Can you see any visual changes between these two art pieces?"
        ]
        for query in queries:
            print(f"Query: {query}\nResponse: {self.send_query(query, "4o-mini")}\n")
