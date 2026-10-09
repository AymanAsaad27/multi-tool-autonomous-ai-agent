import os

from dotenv import load_dotenv

load_dotenv()


def get_llm(temperature: float = 0):
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
            temperature=temperature,
            base_url="http://127.0.0.1:11434",
            sync_client_kwargs={
                "trust_env": False,
            },
            async_client_kwargs={
                "trust_env": False,
            },
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        temperature=temperature,
        max_retries=3,
    )