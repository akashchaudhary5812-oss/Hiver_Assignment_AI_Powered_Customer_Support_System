import os, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY", "").strip('"\' ')

prompt = """You are an expert customer support agent for Amazon on Twitter (@AmazonHelp).
Your job is to draft a personalized, empathetic, concise (under 280 characters), and actionable Twitter response to the customer's specific issue.
Guidelines:
- Never ask for passwords, full credit card numbers, or sensitive PII on public Twitter.
- Direct customers to send a secure Direct Message (DM) or use official Amazon secure links (e.g. https://amzn.to/help or https://amzn.to/returns).
- Directly acknowledge and empathize with their specific situation and emotional tone.
- If they threaten legal action, report severe delays, or report broken/stolen goods, explicitly state that you are escalating this to a senior specialist team.

Customer Tweet: "I recieved My package broken i will file a case against you as i have been messaging you since 4 days!"

Draft the tweet reply directly with no extra commentary:"""

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
payload = {
    "model": "open-mistral-7b",
    "messages": [
        {"role": "user", "content": prompt}
    ],
    "temperature": 0.3,
    "max_tokens": 120
}

resp = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload, timeout=15)
if resp.status_code == 200:
    print("MISTRAL GENERATED REPLY:\n", resp.json()["choices"][0]["message"]["content"].strip())
else:
    print("Error:", resp.text)
