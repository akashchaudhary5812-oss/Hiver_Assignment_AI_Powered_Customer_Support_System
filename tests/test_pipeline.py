"""
Automated Unit and Integration Tests for AI Support Agent Pipeline.
"""

import pytest
import os
import json
from src.data_loader import clean_tweet_text, parse_conversation_turns, INTENT_LABELS
from src.indexer import ResolutionIndexer
from src.agent import SupportAgent
from src.baselines import TrivialBaseline, SimpleMLBaseline
from src.evaluator import LLMJudge, EvaluationHarness
from src.model_trainer import BANKING77_REPO, load_dataset_2_banking77


def test_clean_tweet_text():
    raw = "@AmazonHelp &amp; @115821 My package arrived broken! https://t.co/xyz ^JM"
    cleaned = clean_tweet_text(raw)
    assert "@AmazonHelp" not in cleaned
    assert "@115821" not in cleaned
    assert "^JM" not in cleaned
    assert "&" in cleaned
    assert "My package arrived broken!" in cleaned


def test_intent_classification():
    agent = SupportAgent()
    
    # 1. Order tracking
    res1 = agent.classify_intent("Where is my order? Can you give me the tracking link?")
    assert res1["intent"] == "ORDER_TRACKING_AND_STATUS"
    
    # 2. Account security
    res2 = agent.classify_intent("My account got hacked and there is an unauthorized order!")
    assert res2["intent"] == "ACCOUNT_AND_SECURITY"
    
    # 3. Damaged item
    res3 = agent.classify_intent("The package box was ripped and the item inside is shattered and broken.")
    assert res3["intent"] == "DAMAGED_OR_WRONG_ITEM"
    
    # 4. Late / Missing delivery
    res4 = agent.classify_intent("It says delivered 3 days ago but I never received anything on my porch.")
    assert res4["intent"] == "LATE_OR_MISSING_DELIVERY"
    
    # 5. Prime subscription
    res5 = agent.classify_intent("How do I cancel my Amazon Prime membership annual subscription renewal?")
    assert res5["intent"] == "PRIME_AND_SUBSCRIPTION"
    
    # 6. Payment & Promo
    res6 = agent.classify_intent("Why is my gift card balance withheld? I was charged twice.")
    assert res6["intent"] == "PAYMENT_AND_PROMOTIONS"
    
    # 7. Refund & Return
    res7 = agent.classify_intent("How long does it take for my return dropoff to be refunded to my credit card?")
    assert res7["intent"] == "REFUND_AND_RETURN_STATUS"


def test_escalation_rules():
    agent = SupportAgent()
    
    # Security must ALWAYS escalate
    esc1 = agent.evaluate_escalation(
        "I was locked out and suspect a scam phishing hack on my account", 
        "ACCOUNT_AND_SECURITY"
    )
    assert esc1["escalate"] is True
    assert esc1["risk_level"] == "CRITICAL"
    assert esc1["status"] == "ESCALATE_IMMEDIATELY"
    assert "credentials" in esc1["next_action"].lower()
    
    # Double charge financial dispute must escalate
    esc2 = agent.evaluate_escalation(
        "I was charged twice on my credit card and the money was stolen", 
        "PAYMENT_AND_PROMOTIONS"
    )
    assert esc2["escalate"] is True
    
    # Severe customer churn / legal threat
    esc3 = agent.evaluate_escalation(
        "This is unacceptable and you are stealing my money, I am filing a lawsuit!", 
        "GENERAL_FEEDBACK_OR_CHITCHAT"
    )
    assert esc3["escalate"] is True
    assert esc3["risk_level"] == "CRITICAL"
    
    # Routine order tracking must NOT escalate
    esc4 = agent.evaluate_escalation(
        "Could you send me the tracking link to check my order status?", 
        "ORDER_TRACKING_AND_STATUS"
    )
    assert esc4["escalate"] is False
    assert esc4["risk_level"] == "LOW"
    assert esc4["status"] == "AUTO_HANDLE"
    assert "Routine order tracking" in esc4["reason"]


def test_indexer_retrieval():
    indexer = ResolutionIndexer(kb_path="data/historical_resolutions.json")
    results = indexer.retrieve("package missing never arrived", top_k=3)
    assert len(results) == 3
    assert all("customer_query" in r for r in results)
    assert all("support_reply" in r for r in results)


def test_baselines():
    b1 = TrivialBaseline()
    res_b1 = b1.process("My item is broken")
    assert res_b1["intent"] == "ORDER_TRACKING_AND_STATUS"
    assert res_b1["escalate"] is False
    
    b2 = SimpleMLBaseline(kb_path="data/historical_resolutions.json")
    res_b2 = b2.process("Where is my package?")
    assert res_b2["intent"] in INTENT_LABELS
    assert isinstance(res_b2["draft_reply"], str)


def test_llm_judge_and_rubric():
    judge = LLMJudge()
    eval_res = judge.evaluate_reply(
        customer_query="My package was delayed by 3 days",
        draft_reply="We're sorry for the delay! Please check your order status at https://amzn.to/orders or let us know if you need further help.",
        reference_reply="Sorry for the wait! You can view live tracking at https://amzn.to/help.",
        intent="LATE_OR_MISSING_DELIVERY",
        escalate=False
    )
    assert 1.0 <= eval_res["grounding"] <= 5.0
    assert 1.0 <= eval_res["empathy"] <= 5.0
    assert 1.0 <= eval_res["actionability"] <= 5.0
    assert 1.0 <= eval_res["overall"] <= 5.0


def test_golden_dataset_structure():
    with open("data/golden_set.json", "r", encoding="utf-8") as f:
        golden = json.load(f)
    assert len(golden) == 200
    for g in golden:
        assert "id" in g
        assert "customer_query" in g
        assert "ground_truth_intent" in g
        assert g["ground_truth_intent"] in INTENT_LABELS
        assert "ground_truth_escalate" in g
        assert isinstance(g["ground_truth_escalate"], bool)
        assert "escalation_reason" in g
        assert "reference_response" in g
        assert "human_judge_calibration" in g


def test_official_banking77_intent_source_is_loaded():
    samples = load_dataset_2_banking77(max_per_target_intent=2)
    assert BANKING77_REPO == "PolyAI/banking77"
    assert samples
    assert all(sample["source"].startswith("PolyAI/banking77:") for sample in samples)
    assert all(sample["intent"] in INTENT_LABELS for sample in samples)
