"""
AI Customer Support Agent:
1. ML model (trained on Twitter support plus bounded official Banking77 auxiliary data) computes intent and confidence.
2. Vector Indexer retrieves real-world historical resolution context from the datasets.
3. Multi-signal Triage Engine determines Human Escalation vs Auto-Resolution with explicit reason.
4. Mistral LLM ingests ML scores + dataset context to dynamically generate a tailored, non-static reply.
"""

import os
import re
import json
import requests
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv

from src.data_loader import INTENT_DEFINITIONS, INTENT_LABELS, clean_tweet_text
from src.indexer import ResolutionIndexer
from src.model_trainer import MLIntentScorer

load_dotenv()


class SupportAgent:
    """
    Production AI Support Agent for AmazonHelp Twitter Customer Service.
    Seamlessly unites:
    - ML intent classifier (Twitter + bounded official Banking77 auxiliary data)
    - Few-Shot Historical Dataset RAG Context
    - Risk & Safety Escalation Triage Engine
    - Mistral-7B LLM for dynamic, non-static response synthesis
    """
    def __init__(
        self, 
        indexer: Optional[ResolutionIndexer] = None, 
        scorer: Optional[MLIntentScorer] = None,
        model_name: str = "open-mistral-7b"
    ):
        self._indexer = indexer
        self.scorer = scorer or MLIntentScorer()
        self.api_key = os.getenv("MISTRAL_API_KEY", "").strip('"\' ')
        self.model_name = model_name

    @property
    def indexer(self) -> ResolutionIndexer:
        if self._indexer is None:
            self._indexer = ResolutionIndexer()
        return self._indexer

    def classify_intent(self, query: str, history: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Classifies incoming customer inquiry using the trained ML model + safety overrides.
        Generates ML intent, calibrated confidence score, and distribution.
        """
        text = clean_tweet_text(query)
        full_context = " ".join([clean_tweet_text(h) for h in (history or [])] + [text])
        
        # 1. Run trained ML model on query text
        ml_res = self.scorer.score(full_context)
        pred_intent = ml_res["predicted_intent"]
        confidence = ml_res["ml_confidence_score"]
        
        # 2. Critical Safety Overrides (e.g. urgent account security / hacked flags)
        text_lower = full_context.lower()
        if any(w in text_lower for w in ["hacked", "unauthorized order", "stolen account", "phishing", "scam", "compromised", "account locked"]):
            pred_intent = "ACCOUNT_AND_SECURITY"
            confidence = max(confidence, 0.96)
            
        elif any(w in text_lower for w in ["damaged", "broken", "shattered", "opened box", "tampered", "defective", "faulty", "wrong item", "different item"]):
            pred_intent = "DAMAGED_OR_WRONG_ITEM"
            confidence = max(confidence, 0.94)

        return {
            "intent": pred_intent,
            "confidence": confidence,
            "top_classes": ml_res["top_classes"],
            "full_distribution": ml_res["full_distribution"],
            "rationale": (
                "Classified by TF-IDF Logistic Regression trained on AmazonHelp "
                "plus bounded Banking77 auxiliary examples; "
                f"top-class probability is {confidence*100:.1f}%."
            )
        }

    def evaluate_escalation(
        self, 
        query: str, 
        intent: str, 
        history: Optional[List[str]] = None,
        confidence: float = 1.0
    ) -> Dict[str, Any]:
        """
        Multi-signal Escalation Engine:
        Evaluates risk factors, customer sentiment, financial liability, and ML confidence.
        """
        text = clean_tweet_text(query).lower()
        full_context = " ".join([clean_tweet_text(h).lower() for h in (history or [])] + [text])
        
        def decision(
            *, escalate: bool, status: str, risk_level: str, reason: str,
            next_action: str, matched_trigger: str = ""
        ) -> Dict[str, Any]:
            """Create an auditable routing result for the API, CLI, and UI."""
            return {
                "escalate": escalate,
                "status": status,
                "risk_level": risk_level,
                "reason": reason,
                "next_action": next_action,
                "matched_trigger": matched_trigger,
            }

        # Rule 1: Legal action / Severe hostility / Extended unresolved support attempts
        if any(w in full_context for w in ["lawsuit", "lawyer", "attorney", "file a case", "court", "consumer court", "unacceptable", "incompetent", "stealing", "police", "furious", "disgusted", "4 days", "5 days", "a week", "messaging you"]):
            return decision(
                escalate=True, status="ESCALATE_IMMEDIATELY", risk_level="CRITICAL",
                reason="Potential legal, safety, or repeated-unresolved-service signal detected; an automated reply must not close this case.",
                next_action="Route to a senior human agent with the conversation history attached.",
            )

        # Rule 2: Account Security & Fraud
        if intent == "ACCOUNT_AND_SECURITY" or any(w in full_context for w in ["hacked", "stolen account", "unauthorized order", "compromised", "fraud"]):
            return decision(
                escalate=True, status="ESCALATE_IMMEDIATELY", risk_level="CRITICAL",
                reason="Suspected account compromise, fraud, or unauthorized access requires secure identity verification.",
                next_action="Route to the account-security specialist; do not request credentials or payment details publicly.",
            )
            
        # Rule 3: Financial dispute / Double charging / Withheld funds
        if intent == "PAYMENT_AND_PROMOTIONS" and any(w in full_context for w in ["charged twice", "double charge", "stolen", "balance withheld", "scam", "unauthorized", "bank dispute", "fraud"]):
            return decision(
                escalate=True, status="ESCALATE_TO_BILLING", risk_level="HIGH",
                reason="A disputed or duplicate charge needs account-specific payment verification that this agent cannot perform.",
                next_action="Route to a billing specialist for a secure transaction review.",
            )
            
        # Rule 4: High value or tampered damage
        if intent == "DAMAGED_OR_WRONG_ITEM" and any(w in full_context for w in ["tampered", "opened box", "stolen item", "expensive", "laptop", "phone", "tv", "shattered", "broken", "case against"]):
            return decision(
                escalate=True, status="ESCALATE_TO_FULFILMENT", risk_level="HIGH",
                reason="Potential tampering, loss, or high-value-item damage requires a fulfilment investigation before a remedy is promised.",
                next_action="Route to a fulfilment specialist and preserve the delivery details.",
            )
            
        # Rule 5: Critical delivery delay or repeated broken SLA
        if intent == "LATE_OR_MISSING_DELIVERY" and any(w in full_context for w in ["3rd time", "4th time", "days late", "weeks late", "urgent", "medication", "ruined", "birthday", "lied to me", "promised yesterday"]):
            return decision(
                escalate=True, status="ESCALATE_TO_LOGISTICS", risk_level="HIGH",
                reason="The message indicates a repeated, severe, or time-critical delivery failure beyond routine tracking guidance.",
                next_action="Route to logistics support for a delivery investigation.",
            )
            
        # Rule 6: Ambiguous / Low ML confidence
        if confidence < 0.50:
            return decision(
                escalate=True, status="HUMAN_REVIEW_REQUIRED", risk_level="MEDIUM",
                reason=f"Intent confidence is only {confidence*100:.1f}%, below the 50% auto-handle threshold.",
                next_action="Route to a generalist human agent to identify the issue before responding.",
            )
            
        # Default: Standard self-service resolution
        return decision(
            escalate=False, status="AUTO_HANDLE", risk_level="LOW",
            reason=f"Routine {intent.replace('_', ' ').lower()} request with no detected security, financial-dispute, or severe-service trigger.",
            next_action="Send grounded self-service guidance and retain the case for follow-up if the customer replies.",
        )

    def _generate_with_mistral(
        self,
        query: str,
        intent: str,
        confidence: float,
        escalate: bool,
        escalation_reason: str,
        retrieved_examples: List[Dict[str, Any]],
        history: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Calls Mistral LLM to dynamically synthesize a unique, non-static, brand-grounded response.
        Ingests ML model confidence score, intent, escalation triage, and historical dataset context.
        """
        if not self.api_key:
            return None

        # Build few-shot historical dataset context from real Twitter support resolutions
        examples_str = ""
        for idx, ex in enumerate(retrieved_examples[:2], 1):
            examples_str += f"\nDataset Resolution Exemplar #{idx} (Similarity: {ex.get('similarity_score', 0):.2f}):\n"
            examples_str += f"Customer: \"{ex.get('customer_query')}\"\n"
            examples_str += f"Historical Resolution: \"{ex.get('support_reply')}\"\n"

        history_str = ""
        if history:
            history_str = "Prior Conversation History:\n" + "\n".join(history[-4:]) + "\n\n"

        # Empathy sentence based on escalation flag
        empathy_line = (
            "I’m really sorry you’re experiencing this issue."
            if not escalate else
            "I’m sorry this problem is causing you frustration; we’ll prioritize a solution."
        )

        action_instruction = (
            "ESCALATE TO HUMAN: The customer has an urgent, high-friction, or safety-critical issue. Explicitly acknowledge their specific frustration, include a warm empathetic line, confirm immediate escalation to our specialist team for review, and provide a secure Direct Message (DM) link: https://amzn.to/help"
            if escalate else
            f"AUTO-RESOLVE: The ML model identified this as standard '{intent}'. Include an empathetic opening line acknowledging their situation, and provide an immediate, helpful, customized step-by-step solution with the appropriate link (e.g., https://amzn.to/track, https://amzn.to/returns, https://amzn.to/prime-manage, or https://amzn.to/help)."
        )

        system_prompt = (
            "You are an expert, highly empathetic customer support agent for Amazon on Twitter (@AmazonHelp).\n"
            "You draft dynamic, customized, non-static, and actionable Twitter replies grounded in real Amazon resolution trajectories.\n\n"
            "CORE RULES:\n"
            "1. MUST be under 280 characters (standard Twitter limit).\n"
            "2. NEVER request passwords, full credit cards, or private PII on public Twitter.\n"
            "3. Directly address the customer's specific items, exact complaints, and emotional state.\n"
            "4. NEVER output boilerplate or robotic canned text. Make every response feel human, warm, and authentic.\n"
            "5. Output ONLY the response tweet text."
        )

        user_content = (
            f"{history_str}"
            f"{empathy_line}\n\n"
            f"Incoming Customer Tweet: \"{query}\"\n\n"
            f"[ML Model Analytics (AmazonHelp + bounded Banking77 auxiliary data)]:\n"
            f"• ML Predicted Intent: {intent}\n"
            f"• ML Model Confidence: {confidence*100:.1f}%\n"
            f"• Triage Decision: {'🚨 ESCALATE TO HUMAN' if escalate else '✅ AUTO-RESOLVE'}\n"
            f"• Decision Rationale: {escalation_reason}\n\n"
            f"[Dataset Context & Historical Grounding]:\n{examples_str}\n"
            f"Action Directive: {action_instruction}\n\n"
            f"Synthesize the optimal dynamic Twitter response:"
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.4,
            "max_tokens": 130
        }

        try:
            resp = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload, timeout=12)
            if resp.status_code == 200:
                reply = resp.json()["choices"][0]["message"]["content"].strip().strip('"')
                return reply
            elif resp.status_code == 429 and self.model_name != "open-mistral-7b":
                self.model_name = "open-mistral-7b"
                payload["model"] = "open-mistral-7b"
                resp = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload, timeout=12)
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"].strip().strip('"')
        except Exception:
            pass

        return None

    def generate_reply(
        self,
        query: str,
        intent: str,
        confidence: float,
        escalate: bool,
        escalation_reason: str,
        history: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Drafts a grounded, empathetic, policy-compliant Twitter support response.
        Prioritizes dynamic Mistral LLM generation; falls back to deterministic grounded RAG.
        """
        retrieved = self.indexer.retrieve(query, top_k=2, intent_filter=intent, history=history)
        
        # 1. Attempt dynamic generation with Mistral AI
        mistral_reply = self._generate_with_mistral(
            query=query,
            intent=intent,
            confidence=confidence,
            escalate=escalate,
            escalation_reason=escalation_reason,
            retrieved_examples=retrieved,
            history=history
        )

        if mistral_reply:
            mistral_reply = mistral_reply[:280].rstrip()
            return {
                "reply": mistral_reply,
                "grounded_sources": [r.get("id") for r in retrieved],
                "char_count": len(mistral_reply),
                "engine": "Mistral-7B (Dynamic Hybrid RAG Grounded)"
            }

        # 2. Offline fallback: return the closest historically used resolution,
        # never an intent-specific canned macro. This keeps every reply auditable
        # to a retrieved source even when no generation provider is configured.
        if not retrieved:
            raise RuntimeError("No grounded resolution was retrieved; refusing to emit an ungrounded reply.")
        top_ex = retrieved[0]
        reply = top_ex["support_reply"].strip()
        reply = re.sub(r'\s+', ' ', reply).strip()
        reply = reply[:280].rstrip()

        return {
            "reply": reply,
            "grounded_sources": [r.get("id") for r in retrieved],
            "char_count": len(reply),
            "engine": "Hybrid FAISS RAG (Historical Resolution Retrieval)"
        }

    def process(self, query: str, history: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Complete end-to-end inference pipeline:
        Customer query -> classifier -> triage -> retrieved Twitter context -> reply draft.
        """
        intent_res = self.classify_intent(query, history)
        intent = intent_res["intent"]
        confidence = intent_res["confidence"]
        
        esc_res = self.evaluate_escalation(query, intent, history, confidence)
        escalate = esc_res["escalate"]
        escalation_reason = esc_res["reason"]
        
        reply_res = self.generate_reply(query, intent, confidence, escalate, escalation_reason, history)
        
        return {
            "input_query": query,
            "conversation_history": history or [],
            "intent": intent,
            "intent_confidence": confidence,
            "top_classes": intent_res.get("top_classes", []),
            "intent_rationale": intent_res["rationale"],
            "escalate": escalate,
            "risk_level": esc_res.get("risk_level", "LOW"),
            "escalation_status": esc_res.get("status", "AUTO_HANDLE"),
            "escalation_reason": escalation_reason,
            "next_action": esc_res.get("next_action", ""),
            "matched_trigger": esc_res.get("matched_trigger", ""),
            "draft_reply": reply_res["reply"],
            "grounded_sources": reply_res["grounded_sources"],
            "generation_engine": reply_res.get("engine", "Mistral-7B")
        }
