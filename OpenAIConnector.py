import dotenv
import os

import openai


class OpenAIConnector:

    def __init__(self):
        dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), "./env_vars"))

        self.client = openai.Client(api_key=os.getenv("OPENAI_SECRET"))
        self.sys_prompt = "You are a helpful assistant who should only provide factual information. Please limit responses to 100 words or shorter."

    def query_mini(self, query: str):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.sys_prompt},
                {"role": "user", "content": query}
            ]
        )
        return response.choices[0].message.content

    def send_query(self, query: str, model: str) -> str:
        if model == "4o-mini":
            return self.query_mini(query)
        # todo add support for other models (for funsies)

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


if __name__ == '__main__':
    connector = OpenAIConnector()
    connector.test()
