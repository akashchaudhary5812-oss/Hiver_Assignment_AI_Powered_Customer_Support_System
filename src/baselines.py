"""
Baselines for comparative benchmarking:
1. Baseline 1 (Trivial): Majority-class intent + static canned macro response + never-escalate heuristic.
2. Baseline 2 (Simple ML): TF-IDF + Logistic Regression / Naive Bayes classifier + simple keyword escalation + 1-NN historical retrieval reply.
"""

import json
import os
from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
from src.data_loader import clean_tweet_text, INTENT_LABELS


class TrivialBaseline:
    """
    Baseline 1: Trivial Baseline.
    - Intent: Predicts majority class ('ORDER_TRACKING_AND_STATUS') for every ticket.
    - Escalation: Static heuristic (never escalates).
    - Reply: Fixed canned macro template.
    """
    def __init__(self, majority_intent: str = "ORDER_TRACKING_AND_STATUS"):
        self.majority_intent = majority_intent
        self.canned_reply = "Thank you for reaching out to Amazon Help! Please visit https://amzn.to/help or send us a DM with your order details."

    def process(self, query: str, history: Optional[List[str]] = None) -> Dict[str, Any]:
        return {
            "input_query": query,
            "intent": self.majority_intent,
            "intent_confidence": 0.50,
            "escalate": False,
            "escalation_reason": "Trivial baseline heuristic: auto-handle all tickets.",
            "draft_reply": self.canned_reply,
            "grounded_sources": []
        }


class SimpleMLBaseline:
    """
    Baseline 2: Simple ML Baseline.
    - Intent: Trained TF-IDF + Multinomial Naive Bayes classifier on historical dataset.
    - Escalation: Simple sentiment word count threshold.
    - Reply: Direct 1-Nearest Neighbor historical tweet copy-paste.
    """
    def __init__(self, kb_path: str = "data/historical_resolutions.json"):
        self.kb_path = kb_path
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        self.clf = MultinomialNB()
        self.historical_items: List[Dict[str, Any]] = []
        self.tfidf_matrix = None
        self._fit()

    def _fit(self):
        if not os.path.exists(self.kb_path):
            raise FileNotFoundError(f"Knowledge base not found at {self.kb_path}")
            
        with open(self.kb_path, "r", encoding="utf-8") as f:
            self.historical_items = json.load(f)
            
        queries = [clean_tweet_text(item["customer_query"]) for item in self.historical_items]
        intents = [item["intent"] for item in self.historical_items]
        
        X = self.vectorizer.fit_transform(queries)
        self.clf.fit(X, intents)
        self.tfidf_matrix = X

    def process(self, query: str, history: Optional[List[str]] = None) -> Dict[str, Any]:
        cleaned = clean_tweet_text(query)
        if not cleaned:
            cleaned = "help"
            
        q_vec = self.vectorizer.transform([cleaned])
        pred_intent = self.clf.predict(q_vec)[0]
        probs = self.clf.predict_proba(q_vec)[0]
        conf = float(np.max(probs))
        
        # Simple sentiment word count escalation
        text_lower = cleaned.lower()
        negative_count = sum(1 for w in ["angry", "bad", "terrible", "worst", "unacceptable", "broken", "scam", "late", "hate"] if w in text_lower)
        escalate = negative_count >= 2
        esc_reason = "Simple ML Baseline: Triggered escalation due to multiple negative sentiment keywords." if escalate else "Simple ML Baseline: Negative keywords below threshold."
        
        # 1-NN retrieval copy-paste
        sims = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        top_idx = int(np.argmax(sims))
        nn_reply = self.historical_items[top_idx]["support_reply"]
        
        return {
            "input_query": query,
            "intent": pred_intent,
            "intent_confidence": conf,
            "escalate": escalate,
            "escalation_reason": esc_reason,
            "draft_reply": nn_reply,
            "grounded_sources": [self.historical_items[top_idx]["id"]]
        }
