import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY", "").strip('"\' ')

for model in ["open-mistral-7b", "mistral-tiny", "mistral-small", "open-mixtral-8x7b", "codestral-latest"]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Hello!"}
        ],
        "max_tokens": 30
    }
    try:
        resp = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload, timeout=10)
        print(f"Model {model} -> Status: {resp.status_code}")
        if resp.status_code == 200:
            print("  Result:", resp.json()["choices"][0]["message"]["content"])
        else:
            print("  Error:", resp.text[:100])
    except Exception as e:
        print(f"Model {model} exception:", e)
