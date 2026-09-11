"""
Data loading, cleaning, and preprocessing utilities for Customer Support dialogues.
"""

import re
import html
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

# 8 Domain-derived e-commerce support intents
INTENT_DEFINITIONS = {
    "ORDER_TRACKING_AND_STATUS": {
        "description": "Inquiries regarding tracking numbers, carrier status, dispatch timelines, and delivery estimation.",
        "keywords": ["track", "tracking", "status", "dispatch", "where is my order", "shipped", "estimated delivery", "courier", "package status", "carrier"],
        "typical_resolution": "Provide order lookup link, explain standard shipping SLA, prompt customer to check tracking portal with tracking ID."
    },
    "LATE_OR_MISSING_DELIVERY": {
        "description": "Shipments marked delivered but not received, severe carrier delays, lost parcels, or missed promised delivery windows.",
        "keywords": ["not received", "late", "delayed", "said delivered", "never arrived", "missing", "still waiting", "past delivery date", "stolen", "lost"],
        "typical_resolution": "Empathize with delay, check carrier tracking window (e.g. allow 36h buffer or investigate with carrier), escalate if past threshold for replacement."
    },
    "REFUND_AND_RETURN_STATUS": {
        "description": "Requests for product returns, refund processing timelines, return pickup scheduling, and balance reversals.",
        "keywords": ["refund", "return", "send back", "drop off", "money back", "reimbursement", "pickup", "credited", "reversal", "return label"],
        "typical_resolution": "Explain return window policy, provide return center link, clarify 3-5 business day refund timeline post-inspection."
    },
    "DAMAGED_OR_WRONG_ITEM": {
        "description": "Defective merchandise, broken seals, expired items, wrong size/color/product delivered, or package tampering.",
        "keywords": ["damaged", "broken", "wrong item", "different item", "faulty", "tampered", "opened box", "shattered", "defective", "missing parts"],
        "typical_resolution": "Acknowledge inconvenience, offer immediate replacement or free return with return label via Online Returns Center."
    },
    "ACCOUNT_AND_SECURITY": {
        "description": "Unauthorized account access, compromised passwords, 2FA issues, suspended accounts, or suspected phishing.",
        "keywords": ["hacked", "unauthorized", "locked out", "password reset", "2fa", "security", "scam", "phishing", "compromised", "login error"],
        "typical_resolution": "High priority safety escalation: Do not ask for credentials publicly; direct to secure account recovery and security verification portal."
    },
    "PRIME_AND_SUBSCRIPTION": {
        "description": "Amazon Prime membership fees, auto-renewal queries, student discounts, Prime Video/Music/Kindle streaming access issues.",
        "keywords": ["prime", "membership", "subscription", "annual fee", "renew", "cancel prime", "prime video", "kindle unlimited", "music unlimited"],
        "typical_resolution": "Guide user to 'Manage Prime Membership' page to cancel, view renewal dates, or manage benefits."
    },
    "PAYMENT_AND_PROMOTIONS": {
        "description": "Gift card balance holds, declined cards, double charging, invalid promo/coupon codes, or invoice disputes.",
        "keywords": ["gift card", "promo code", "coupon", "charged twice", "double charge", "payment declined", "balance withheld", "voucher", "billing"],
        "typical_resolution": "Explain gift card terms, clarify authorization holds vs actual charges, route to secure billing support if dispute persists."
    },
    "GENERAL_FEEDBACK_OR_CHITCHAT": {
        "description": "Customer appreciation, general complaints not tied to an active order, UI feedback, or conversational chit-chat.",
        "keywords": ["thanks", "thank you", "great job", "terrible service", "suggestion", "hello", "hi", "hey", "just saying", "bye"],
        "typical_resolution": "Politely acknowledge sentiment, thank for feedback or offer general assistance."
    }
}

INTENT_LABELS = list(INTENT_DEFINITIONS.keys())


def clean_tweet_text(text: str) -> str:
    """
    Cleans raw customer/support tweet text:
    - Normalizes HTML entities (&amp; -> &, &gt; -> >)
    - Strips handle anonymizations (@115821, @AmazonHelp, @sprintcar)
    - Normalizes URLs (preserves placeholder token or clean format)
    - Strips trailing agent signatures (^JM, ^AF, ^CD)
    - Normalizes extra whitespace
    """
    if not isinstance(text, str):
        return ""
    
    # Decode HTML
    cleaned = html.unescape(text)
    
    # Remove agent signatures like ^JM, ^AF, ^CD at end of support tweets
    cleaned = re.sub(r'\^[A-Z]{2,3}\b', '', cleaned)
    
    # Remove Twitter handles (e.g. @AmazonHelp, @115821, @username)
    cleaned = re.sub(r'@[A-Za-z0-9_]+', '', cleaned)
    
    # Normalize multiple whitespaces and newlines
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned


def parse_conversation_turns(raw_conv_text: str) -> List[Dict[str, str]]:
    """
    Parses a conversation string into structured turns:
    Customer: ...
    Support: ...
    """
    turns = []
    lines = raw_conv_text.split('\n')
    current_role = None
    current_text = []

    for line in lines:
        line_s = line.strip()
        if line_s.startswith("Customer:"):
            if current_role and current_text:
                turns.append({"role": current_role, "text": " ".join(current_text)})
                current_text = []
            current_role = "customer"
            current_text.append(line_s.replace("Customer:", "").strip())
        elif line_s.startswith("Support:"):
            if current_role and current_text:
                turns.append({"role": current_role, "text": " ".join(current_text)})
                current_text = []
            current_role = "support"
            current_text.append(line_s.replace("Support:", "").strip())
        else:
            if current_role and line_s:
                current_text.append(line_s)
                
    if current_role and current_text:
        turns.append({"role": current_role, "text": " ".join(current_text)})

    return turns


def extract_qa_pairs(raw_conv_text: str) -> Optional[Dict[str, Any]]:
    """
    Extracts the primary initial customer inquiry and the immediate grounded support reply.
    """
    turns = parse_conversation_turns(raw_conv_text)
    if not turns:
        return None
    
    # Find first customer message and subsequent support message
    first_cust_idx = -1
    for idx, t in enumerate(turns):
        if t["role"] == "customer" and len(clean_tweet_text(t["text"])) > 10:
            first_cust_idx = idx
            break
            
    if first_cust_idx == -1:
        return None
        
    first_supp_idx = -1
    for idx in range(first_cust_idx + 1, len(turns)):
        if turns[idx]["role"] == "support" and len(clean_tweet_text(turns[idx]["text"])) > 10:
            first_supp_idx = idx
            break
            
    if first_supp_idx == -1:
        return None
        
    cust_query = clean_tweet_text(turns[first_cust_idx]["text"])
    supp_reply = clean_tweet_text(turns[first_supp_idx]["text"])
    
    # Multi-turn context before current turn (if any)
    history = []
    for idx in range(first_cust_idx):
        history.append(f"{turns[idx]['role'].capitalize()}: {clean_tweet_text(turns[idx]['text'])}")
        
    return {
        "customer_query": cust_query,
        "support_reply": supp_reply,
        "history": history,
        "raw_turns_count": len(turns)
    }
