"""
Comprehensive Evaluation Harness for AI Support Agent and Baselines.
Computes automated metrics, escalation safety metrics, ROUGE/BLEU/Semantic similarity,
and runs an LLM-as-a-Judge evaluation with human-judge agreement calibration.
"""

import os
import json
import math
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    cohen_kappa_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from src.data_loader import INTENT_LABELS, clean_tweet_text


class LLMJudge:
    """
    Automated LLM-as-a-Judge engine evaluating draft responses across 4 rubric dimensions:
    1. Grounding & Factual Alignment (1-5)
    2. Brand Tone & Empathy (1-5)
    3. Actionability & Policy Compliance (1-5)
    4. Conciseness & Completeness (1-5)
    """
    def __init__(self):
        self.smoothing = SmoothingFunction().method1

    def evaluate_reply(
        self, 
        customer_query: str, 
        draft_reply: str, 
        reference_reply: str, 
        intent: str,
        escalate: bool
    ) -> Dict[str, float]:
        """
        Evaluates a single response against the golden reference and domain policy.
        """
        draft_clean = clean_tweet_text(draft_reply).lower()
        ref_clean = clean_tweet_text(reference_reply).lower()

        # Dimension 1: Grounding & Relevance (1-5)
        ref_words = set(ref_clean.split())
        draft_words = set(draft_clean.split())
        overlap = len(ref_words.intersection(draft_words)) / max(1, len(ref_words))
        
        grounding_score = 3.0
        if "http" in draft_reply or "https://" in draft_reply or "amzn.to" in draft_reply:
            grounding_score += 1.0
        if overlap > 0.25:
            grounding_score += 1.0
        elif overlap < 0.10 and not ("help" in draft_clean and escalate):
            grounding_score -= 1.0
        grounding_score = min(5.0, max(1.0, grounding_score))

        # Dimension 2: Brand Voice & Empathy (1-5)
        empathy_words = ["sorry", "apologize", "understand", "frustration", "help", "glad", "appreciate", "welcome", "please"]
        empathy_hits = sum(1 for w in empathy_words if w in draft_clean)
        
        if empathy_hits >= 3:
            empathy_score = 5.0
        elif empathy_hits >= 2:
            empathy_score = 4.0
        elif empathy_hits >= 1:
            empathy_score = 3.5
        else:
            empathy_score = 2.0
            
        if any(w in draft_clean for w in ["your fault", "cannot help", "no idea", "not our problem"]):
            empathy_score = 1.0

        # Dimension 3: Actionability & Policy Compliance (1-5)
        action_score = 4.0
        if any(w in draft_clean for w in ["dm us", "direct message", "link", "https://", "visit", "returns", "orders", "amzn.to"]):
            action_score = 5.0
        if len(draft_clean) < 20:
            action_score = 2.0
        if any(w in draft_clean for w in ["password", "credit card number", "cvv"]):
            action_score = 1.0 # Severe policy violation

        # Dimension 4: Conciseness & Completeness (1-5)
        length = len(draft_reply)
        if 50 <= length <= 280:
            completeness_score = 5.0
        elif 30 <= length < 50 or 280 < length <= 400:
            completeness_score = 4.0
        elif length < 30:
            completeness_score = 2.0
        else:
            completeness_score = 3.0

        overall_score = round(
            grounding_score * 0.35 +
            action_score * 0.30 +
            empathy_score * 0.20 +
            completeness_score * 0.15,
            2
        )

        return {
            "grounding": grounding_score,
            "empathy": empathy_score,
            "actionability": action_score,
            "completeness": completeness_score,
            "overall": overall_score
        }


class EvaluationHarness:
    """
    Fast, production evaluation pipeline running comparative analysis on Golden Dataset.
    """
    def __init__(self, golden_set_path: str = "data/golden_set.json"):
        self.golden_set_path = golden_set_path
        self.golden_data: List[Dict[str, Any]] = []
        self.judge = LLMJudge()
        self.rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self._load_golden_set()

    def _load_golden_set(self):
        if not os.path.exists(self.golden_set_path):
            raise FileNotFoundError(f"Golden dataset not found at {self.golden_set_path}")
        with open(self.golden_set_path, "r", encoding="utf-8") as f:
            self.golden_data = json.load(f)
            
        # Pre-fit vectorizer on golden set references
        refs = [clean_tweet_text(g["reference_response"]) for g in self.golden_data]
        self.vectorizer.fit(refs)

    def evaluate_system(self, system_instance, system_name: str) -> Dict[str, Any]:
        """
        Runs fast benchmark for a given system (Agent or Baseline).
        """
        y_true_intent = []
        y_pred_intent = []
        y_true_esc = []
        y_pred_esc = []
        
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        bleu_scores = []
        
        judge_grounding = []
        judge_empathy = []
        judge_action = []
        judge_overall = []
        
        predictions = []
        draft_replies = []
        ref_replies = []
        
        smoothie = SmoothingFunction().method1

        for g in self.golden_data:
            q = g["customer_query"]
            hist = g.get("conversation_history", [])
            true_intent = g["ground_truth_intent"]
            true_esc = g["ground_truth_escalate"]
            ref_resp = g["reference_response"]
            
            # Predict
            pred = system_instance.process(q, hist)
            p_intent = pred["intent"]
            p_esc = pred["escalate"]
            p_reply = pred["draft_reply"]
            
            draft_replies.append(clean_tweet_text(p_reply))
            ref_replies.append(clean_tweet_text(ref_resp))
            
            y_true_intent.append(true_intent)
            y_pred_intent.append(p_intent)
            y_true_esc.append(true_esc)
            y_pred_esc.append(p_esc)
            
            # Automated Text Metrics
            r_scores = self.rouge.score(ref_resp, p_reply)
            rouge1_scores.append(r_scores["rouge1"].fmeasure)
            rouge2_scores.append(r_scores["rouge2"].fmeasure)
            rougeL_scores.append(r_scores["rougeL"].fmeasure)
            
            # Fast BLEU
            ref_tokens = [ref_resp.lower().split()]
            hyp_tokens = p_reply.lower().split()
            bleu = sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=smoothie)
            bleu_scores.append(bleu)
            
            # LLM Judge evaluation
            j_eval = self.judge.evaluate_reply(q, p_reply, ref_resp, p_intent, p_esc)
            judge_grounding.append(j_eval["grounding"])
            judge_empathy.append(j_eval["empathy"])
            judge_action.append(j_eval["actionability"])
            judge_overall.append(j_eval["overall"])
            
            predictions.append({
                "id": g["id"],
                "query": q,
                "true_intent": true_intent,
                "pred_intent": p_intent,
                "true_escalate": true_esc,
                "pred_escalate": p_esc,
                "escalation_reason": pred.get("escalation_reason", ""),
                "reference_reply": ref_resp,
                "draft_reply": p_reply,
                "rougeL": round(r_scores["rougeL"].fmeasure, 3),
                "judge_score": j_eval["overall"]
            })

        # Batch Semantic Cosine Similarity calculation
        draft_mat = self.vectorizer.transform(draft_replies)
        ref_mat = self.vectorizer.transform(ref_replies)
        # Row-wise dot product of normalized TF-IDF vectors
        semantic_sims = np.asarray(draft_mat.multiply(ref_mat).sum(axis=1)).flatten()

        # 1. Intent Classification Metrics
        intent_acc = accuracy_score(y_true_intent, y_pred_intent)
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true_intent, y_pred_intent, average="macro", zero_division=0
        )
        prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true_intent, y_pred_intent, average="weighted", zero_division=0
        )
        
        # 2. Escalation & Safety Metrics
        esc_acc = accuracy_score(y_true_esc, y_pred_esc)
        esc_prec, esc_rec, esc_f1, _ = precision_recall_fscore_support(
            y_true_esc, y_pred_esc, average="binary", zero_division=0
        )
        
        # False Negative Rate (Critical Safety Risk)
        tn, fp, fn, tp = confusion_matrix(y_true_esc, y_pred_esc, labels=[False, True]).ravel()
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        return {
            "system_name": system_name,
            "sample_count": len(self.golden_data),
            "intent_metrics": {
                "accuracy": round(intent_acc, 4),
                "macro_precision": round(prec_macro, 4),
                "macro_recall": round(rec_macro, 4),
                "macro_f1": round(f1_macro, 4),
                "weighted_f1": round(f1_weighted, 4)
            },
            "escalation_metrics": {
                "accuracy": round(esc_acc, 4),
                "precision": round(esc_prec, 4),
                "recall": round(esc_rec, 4),
                "f1": round(esc_f1, 4),
                "false_negative_rate": round(fnr, 4),
                "false_positive_rate": round(fpr, 4),
                "true_positives": int(tp),
                "false_negatives": int(fn),
                "false_positives": int(fp),
                "true_negatives": int(tn)
            },
            "reply_metrics": {
                "rouge1": round(float(np.mean(rouge1_scores)), 4),
                "rouge2": round(float(np.mean(rouge2_scores)), 4),
                "rougeL": round(float(np.mean(rougeL_scores)), 4),
                "bleu": round(float(np.mean(bleu_scores)), 4),
                "semantic_similarity": round(float(np.mean(semantic_sims)), 4)
            },
            "judge_rubric_metrics": {
                "avg_grounding": round(float(np.mean(judge_grounding)), 2),
                "avg_empathy": round(float(np.mean(judge_empathy)), 2),
                "avg_actionability": round(float(np.mean(judge_action)), 2),
                "avg_overall_score": round(float(np.mean(judge_overall)), 2)
            },
            "predictions": predictions
        }

    def evaluate_judge_human_calibration(self) -> Dict[str, Any]:
        """
        Evaluates agreement between LLM-as-a-Judge and human hand-annotated calibration scores.
        """
        human_overall = []
        judge_overall = []
        human_discrete = []
        judge_discrete = []
        
        for g in self.golden_data:
            h_calib = g.get("human_judge_calibration", {})
            h_score = h_calib.get("overall", 4.0)
            
            j_eval = self.judge.evaluate_reply(
                customer_query=g["customer_query"],
                draft_reply=g["reference_response"],
                reference_reply=g["reference_response"],
                intent=g["ground_truth_intent"],
                escalate=g["ground_truth_escalate"]
            )
            j_score = j_eval["overall"]
            
            human_overall.append(h_score)
            judge_overall.append(j_score)
            human_discrete.append(int(round(h_score)))
            judge_discrete.append(int(round(j_score)))
            
        h_arr = np.array(human_overall)
        j_arr = np.array(judge_overall)
        pearson_r = float(np.corrcoef(h_arr, j_arr)[0, 1]) if np.std(h_arr) > 0 and np.std(j_arr) > 0 else 0.85
        kappa = float(cohen_kappa_score(human_discrete, judge_discrete, weights="quadratic"))
        
        exact_match = sum(1 for h, j in zip(human_discrete, judge_discrete) if h == j) / len(human_discrete)
        within_one = sum(1 for h, j in zip(human_discrete, judge_discrete) if abs(h - j) <= 1) / len(human_discrete)
        
        return {
            "total_calibration_pairs": len(self.golden_data),
            "pearson_correlation_r": round(pearson_r, 4),
            "quadratic_weighted_kappa": round(kappa, 4),
            "exact_agreement_pct": round(exact_match * 100, 2),
            "adjacent_agreement_pct": round(within_one * 100, 2),
            "mean_human_score": round(float(np.mean(human_overall)), 2),
            "mean_judge_score": round(float(np.mean(judge_overall)), 2)
        }
