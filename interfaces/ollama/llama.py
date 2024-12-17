from config import ollama_host
from ollama import Client

ollama_client = Client(ollama_host)

def stream_ollama(stream):
    for chunk in stream:
        yield chunk["message"]["content"]
