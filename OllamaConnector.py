import json
import ollama


class OllamaConnector:
    def __init__(self):
        self.client = ollama.Client(host="http://localhost:11434")
        self.models = self.read_models_json("models.json")

    @staticmethod
    def read_models_json(json_loc: str = "models.json"):
        try:
            with open(json_loc, "rt") as models_json:
                models_dict = json.load(models_json)
                return models_dict
        except OSError as e:
            print(f"Failed to read in {json_loc}:\n\t{e}")

    def does_model_exist(self, model_name: str) -> bool:
        return model_name in self.models.keys()

    def is_model_installed(self, model_name: str) -> bool:
        return self.models[model_name]["is_installed"]

    def is_model_disabled(self, model_name: str) -> bool:
        return self.models[model_name]["is_disabled"]

    def is_model_available(self, model_name: str) -> bool:
        return self.does_model_exist(model_name=model_name) and self.is_model_installed(
            model_name=model_name) and not self.is_model_disabled(model_name=model_name)

    def is_model_custom(self, model_name: str) -> bool:
        return self.models[model_name]["is_custom"]

    def install_model(self, model_name: str) -> bool:
        if not self.does_model_exist(model_name=model_name):
            print(f"Model {model_name} does not exist in self.models.")
            return False
        if self.is_model_custom(model_name=model_name) and not self.is_model_installed(model_name=model_name):
            print(f"Attempting to install custom model {model_name}")
            modelfile = self.models[model_name]["modelfile"]
            try:
                created_model_status = self.client.create(model=model_name, modelfile=modelfile)
                self.models[model_name]["is_installed"] = True if created_model_status is not None else False
                if self.is_model_installed(model_name=model_name): print(f"Model {model_name} was installed.")
                return True
            except ollama.ResponseError as e:
                print(f"Failed to install custom model {model_name}: {e}")
                return False
        else:
            try:
                return True if self.client.pull(model_name) is not None else False
            except ollama.ResponseError as e:
                print(f"Failed to pull model {model_name}, error: {e}")
                return False

    def populate_modelfiles_for_custom_models(self):
        for model_name in self.models.keys():
            if self.models[model_name]["is_file_based"]:
                print(f"Populating `modelfile` for {model_name}")
                try:
                    with open(self.models[model_name]["modelfile_loc"], "rt") as modelfile_wrapper:
                        modelfile = modelfile_wrapper.read()
                        self.models[model_name]["modelfile"] = modelfile
                except OSError as e:
                    print(f"OS-level error occurred while populating modelfile for {model_name}: {e}")

    def validate_all_models(self):
        for model_name in self.models.keys():
            print(f"Validating model: {model_name}")

            if not self.models[model_name]["is_installed"]:
                try:
                    print(f"\tChecking if model {model_name} installed...")
                    try:
                        # This will jump to `except ollama.ResponseError` if model is not installed
                        self.client.chat(model=model_name)
                        self.models[model_name]["is_installed"] = True
                        print(f"\tModel {model_name} is already installed.")
                    except ollama.ResponseError as e:
                        if e.status_code == 404:
                            self.install_model(model_name=model_name)
                except OSError as e:
                    self.models[model_name]["is_installed"] = False
                    self.models[model_name]["is_disabled"] = True
                    print(f"Failed to install model {model_name}, disabled.\n\tException: {e}")
                except ollama.ResponseError as e:
                    self.models[model_name]["is_installed"] = False
                    self.models[model_name]["is_disabled"] = True
                    print(f"Failed to install model {model_name}, disabled.\n\tException: {e}")

            if not self.does_model_exist(model_name=model_name):
                print(f"Model {model_name} does not exist")
                install_status = self.install_model(model_name=model_name)
                print(f"Install status for model {model_name}: {install_status}")

    def send_query(self, query: str, model: str) -> str | None:
        try:
            if model not in self.models.keys():
                print(f"Model {model} is not in self.models.")
                return None
            response = self.client.chat(model=model, messages=[{
                "role": "user",
                "content": query
            }])["message"]["content"]
            return response
        except ollama.ResponseError as e:
            print(f"Query to model {model} failed:\n\t{e}")
            return None

    # update to use 'stream = True' (this will allow text to be written to display as it comes)
    def send_query_qa(self, query: str) -> str | None:
        response = self.send_query(query=query, model="llama_qa")
        return response

    def test_all_models(self):
        model_names = self.models.keys()
        print(f"Testing all models: {", ".join(model_names)}")
        for model in model_names:
            query = self.models[model]["test_query"]
            response = self.send_query(query, model)
            print(f"\n====Response from model {model}====\n{response}\n====End of response from model {model}====")


def main():
    connector = OllamaConnector()
    connector.populate_modelfiles_for_custom_models()
    connector.validate_all_models()
    connector.test_all_models()


if __name__ == '__main__':
    main()
