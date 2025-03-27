import os
from typing import Any, TypeVar, TypedDict
from dotenv import load_dotenv
from openai import OpenAI

# ucitavamo env
load_dotenv(override=True)

OpenAIClient = TypeVar('OpenAiClient', bound=OpenAI)

# APi kljucevi
openai_api_key = os.getenv("OPENAI_API_KEY")


def verify_api_keys():
    if openai_api_key:
        print(f"Open AI api key exists and starts with {openai_api_key[:8]}")
    else:
        print("Open AI api key missing.")


# init clients

open_ai_client: OpenAIClient = OpenAI()


class AIClients(TypedDict):
    openai: OpenAIClient


def Clients() -> AIClients:
    return {
        "openai": open_ai_client
    }
