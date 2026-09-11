"""
ML training pipeline for AmazonHelp intent classification, augmented with the
official PolyAI/Banking77 dataset only where labels transfer across domains.
"""

import os
import csv
import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from urllib.request import urlretrieve

from src.data_loader import clean_tweet_text, INTENT_LABELS

# Cross-domain weak-supervision map.  Banking77 is a banking dataset, so its
# labels are *not* treated as Amazon labels.  Only concepts that genuinely
# transfer to the AmazonHelp taxonomy are included, and each target class is
# capped during training so that Banking77 cannot dominate the Twitter data.
# Source label spellings intentionally match PolyAI/banking77 exactly.
BANKING77_MAPPING = {
    "card_arrival": "ORDER_TRACKING_AND_STATUS",
    "card_delivery_estimate": "ORDER_TRACKING_AND_STATUS",
    "tracking_delivery": "ORDER_TRACKING_AND_STATUS",
    "lost_or_stolen_card": "ACCOUNT_AND_SECURITY",
    "compromised_card": "ACCOUNT_AND_SECURITY",
    "verify_top_up": "ACCOUNT_AND_SECURITY",
    "pin_blocked": "ACCOUNT_AND_SECURITY",
    "verify_source_of_funds": "ACCOUNT_AND_SECURITY",
    "passcode_forgotten": "ACCOUNT_AND_SECURITY",
    "refund_not_showing_up": "REFUND_AND_RETURN_STATUS",
    "declined_transfer": "PAYMENT_AND_PROMOTIONS",
    "declined_card_payment": "PAYMENT_AND_PROMOTIONS",
    "card_payment_fee_charged": "PAYMENT_AND_PROMOTIONS",
    "card_payment_wrong_exchange_rate": "PAYMENT_AND_PROMOTIONS",
    "transfer_fee_charged": "PAYMENT_AND_PROMOTIONS",
    "top_up_failed": "PAYMENT_AND_PROMOTIONS",
    "top_up_reverted": "PAYMENT_AND_PROMOTIONS",
    "balance_not_updated_after_cheque_or_cash_deposit": "PAYMENT_AND_PROMOTIONS",
    "extra_charge_on_statement": "PAYMENT_AND_PROMOTIONS",
    "transfer_not_received_by_recipient": "LATE_OR_MISSING_DELIVERY",
    "pending_transfer": "LATE_OR_MISSING_DELIVERY",
    "transfer_timing": "LATE_OR_MISSING_DELIVERY",
    "failed_transfer": "PAYMENT_AND_PROMOTIONS",
    "transaction_charged_twice": "PAYMENT_AND_PROMOTIONS",
    "card_payment_not_recognised": "ACCOUNT_AND_SECURITY",
    "cash_withdrawal_not_recognised": "ACCOUNT_AND_SECURITY",
    "direct_debit_payment_not_recognised": "ACCOUNT_AND_SECURITY",
    "lost_or_stolen_phone": "ACCOUNT_AND_SECURITY",
    "unable_to_verify_identity": "ACCOUNT_AND_SECURITY",
    "verify_my_identity": "ACCOUNT_AND_SECURITY",
    "why_verify_identity": "ACCOUNT_AND_SECURITY",
    "request_refund": "REFUND_AND_RETURN_STATUS",
    "reverted_card_payment?": "REFUND_AND_RETURN_STATUS",
    "card_not_working": "PAYMENT_AND_PROMOTIONS",
    "pending_card_payment": "PAYMENT_AND_PROMOTIONS",
    "cash_withdrawal_charge": "PAYMENT_AND_PROMOTIONS",
    "top_up_by_card_charge": "PAYMENT_AND_PROMOTIONS",
    "top_up_by_bank_transfer_charge": "PAYMENT_AND_PROMOTIONS",
}

BANKING77_REPO = "PolyAI/banking77"
# These are the exact URLs specified by the official dataset loading script.
BANKING77_TRAIN_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    "master/banking_data/train.csv"
)
BANKING77_CACHE_PATH = Path("data/banking77_train.csv")


def load_dataset_1_twitter_amazon() -> List[Dict[str, str]]:
    """Loads and formats Twitter Customer Support dataset (AmazonHelp)."""
    kb_path = "data/historical_resolutions.json"
    if not os.path.exists(kb_path):
        from scripts.build_dataset import main as build_kb
        build_kb()
        
    with open(kb_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    samples = []
    for item in data:
        text = clean_tweet_text(item.get("customer_query", ""))
        intent = item.get("intent")
        if text and intent in INTENT_LABELS:
            samples.append({"text": text, "intent": intent, "source": "Twitter-AmazonHelp"})
    return samples


def _get_official_banking77_train(path: Path = BANKING77_CACHE_PATH) -> Path:
    """Return a cached copy of the official PolyAI Banking77 training CSV."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            urlretrieve(BANKING77_TRAIN_URL, path)
        except Exception as exc:
            raise RuntimeError(
                "Could not download the official PolyAI/Banking77 train split. "
                "Download it once from the URL in README.md to "
                f"{path}, then rerun training. Original error: {exc}"
            ) from exc
    return path


def load_dataset_2_banking77(max_per_target_intent: int = 350) -> List[Dict[str, str]]:
    """Load official Banking77 labels and conservatively map transferable intents.

    This auxiliary data is used only for intent representation learning. Amazon
    reply retrieval and the golden evaluation set remain Twitter-only.
    """
    path = _get_official_banking77_train()
    by_target: Dict[str, List[Dict[str, str]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            label_text = (row.get("category") or "").strip()
            target_intent = BANKING77_MAPPING.get(label_text)
            raw_text = clean_tweet_text(row.get("text", ""))
            if target_intent and raw_text:
                by_target.setdefault(target_intent, []).append({
                    "text": raw_text,
                    "intent": target_intent,
                    "source": f"{BANKING77_REPO}:{label_text}",
                })

    # Input is ordered by category; deterministic subsampling keeps training
    # reproducible while preventing cross-domain labels from swamping Twitter.
    samples: List[Dict[str, str]] = []
    for intent in sorted(by_target):
        samples.extend(by_target[intent][:max_per_target_intent])
    return samples


def train_intent_model(save_path: str = "data/trained_intent_model.joblib") -> Dict[str, Any]:
    """
    Trains TF-IDF + balanced Logistic Regression on Twitter data plus bounded,
    mapped Banking77 auxiliary examples.
    """
    print("[1/3] Loading Dataset 1: Customer Support on Twitter (AmazonHelp)...")
    d1 = load_dataset_1_twitter_amazon()
    print(f"      -> Loaded {len(d1)} examples from Twitter dataset.")
    
    print("[2/3] Loading Dataset 2: official PolyAI/Banking77...")
    d2 = load_dataset_2_banking77()
    print(f"      -> Loaded {len(d2)} mapped examples from Banking77 dataset.")
    
    all_data = d1 + d2
    texts = [sample["text"] for sample in all_data]
    intents = [sample["intent"] for sample in all_data]
    print(f"[3/3] Training ML Model on total {len(all_data)} multi-dataset queries across 8 intents...")
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=15000,
            sublinear_tf=True,
            stop_words='english'
        )),
        ('clf', LogisticRegression(
            C=3.0,
            max_iter=1000,
            class_weight='balanced'
        ))
    ])
    
    pipeline.fit(texts, intents)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(pipeline, save_path)
    print(f"[+] Trained ML Intent Model saved successfully to {save_path}!")
    
    return {
        "total_samples": len(all_data),
        "twitter_samples": len(d1),
        "banking77_samples": len(d2),
        "banking77_source": BANKING77_REPO,
        "banking77_cache": str(BANKING77_CACHE_PATH),
        "banking77_max_per_target_intent": 350,
        "classes": list(pipeline.classes_)
    }


class MLIntentScorer:
    """
    Inference scoring engine for the trained ML model.
    Generates class probabilities, confidence scores, and top candidate intents.
    """
    def __init__(self, model_path: str = "data/trained_intent_model.joblib"):
        self.model_path = model_path
        if not os.path.exists(model_path):
            train_intent_model(model_path)
        self.pipeline = joblib.load(model_path)
        self.classes = [str(label) for label in self.pipeline.classes_]

    def score(self, text: str) -> Dict[str, Any]:
        """
        Runs ML model scoring on incoming customer query:
        Returns predicted intent, confidence score, and full class distribution.
        """
        cleaned = clean_tweet_text(text)
        if not cleaned:
            cleaned = "help"
            
        probs = self.pipeline.predict_proba([cleaned])[0]
        top_idx = int(np.argmax(probs))
        pred_intent = self.classes[top_idx]
        confidence = float(probs[top_idx])
        
        # Sort full probability distribution
        distribution = {str(cls): round(float(prob), 4) for cls, prob in zip(self.classes, probs)}
        sorted_dist = dict(sorted(distribution.items(), key=lambda item: item[1], reverse=True))
        
        return {
            "predicted_intent": pred_intent,
            "ml_confidence_score": round(confidence, 4),
            "top_classes": list(sorted_dist.items())[:3],
            "full_distribution": sorted_dist
        }


if __name__ == "__main__":
    train_intent_model()
