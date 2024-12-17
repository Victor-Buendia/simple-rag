def stream_ollama(stream):
    for chunk in stream:
        yield chunk["message"]["content"]
