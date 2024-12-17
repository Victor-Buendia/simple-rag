from yaml import safe_load

config = safe_load(open("./config.yaml"))

chat_model = config["ollama"]["chat_model"]
embeddings_model = config["ollama"]["embeddings_model"]
