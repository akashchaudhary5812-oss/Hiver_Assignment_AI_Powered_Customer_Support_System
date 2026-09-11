import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from src.agent import SupportAgent

agent = SupportAgent()

queries = [
    "I ordered a Kindle paperwhite but the screen is cracked and won't turn on!",
    "My package was supposed to arrive 3 days ago for my daughter's birthday party and it is still not here!",
    "Can you tell me how to check where my package is right now?",
    "I was double charged $79 on my credit card statement for my order!"
]

for q in queries:
    print("=" * 70)
    print("Customer:", q)
    res = agent.process(q)
    print("ML Model Predicted Intent:", res["intent"])
    print("ML Confidence Score:", f"{res['intent_confidence']*100:.1f}%")
    print("Escalation:", res["escalate"], f"({res['risk_level']}) - {res['escalation_reason']}")
    print("Engine:", res["generation_engine"])
    print("Dynamic Reply:\n" + res["draft_reply"])
