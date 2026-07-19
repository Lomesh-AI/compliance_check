"""
generator.py — LLM generation for two tasks.

Task 1 — check_compliance_for_rule(rule_subject, rule_triples, relevant_chunks)
    Called once per KG rule-group (19 calls total for our 51-triple KG).
    Receives only the triples relevant to this rule-group + only the chunks
    retrieved as relevant to this rule. NOT all chunks. NOT all triples.
    Returns: PASS | FAIL | INSUFFICIENT_EVIDENCE + reason

Task 2 — answer_query(query, chunks)
    DPDP QA with citation-anchored prompting.

Fixes vs previous version:
  - No longer accepts raw chunk loop (that caused O(n) LLM calls)
  - check_compliance() now takes rule_subject + rule_triples + retrieved_chunks
  - Triples are filtered per rule-group before passing to LLM (#4)
  - Added INSUFFICIENT_EVIDENCE verdict when retrieved context is too thin (#6)
  - Lazy LLM client init retained
"""

import logging
import re
from typing import List, Dict, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import get_llm_client
from src.kg.deontic_kg import DeonticKG

log = logging.getLogger("generator")


# ── Prompt templates ─────────────────────────────────────────────────────────

# SYSTEM_COMPLIANCE = (
#     "You are a compliance expert specializing in India's Digital Personal Data "
#     "Protection Act 2023 (DPDP Act). "
#     "You will receive one regulatory rule-group with deontic typing "
#     "(OBLIGATION = must do, PROHIBITION = must not do, PERMISSION = may do, "
#     "ENTITLEMENT = Data Principal's right) and the most relevant excerpts from "
#     "a business policy document. "
#     "Determine whether the business policy satisfies the rule. "
#     "Use ONLY the provided excerpts. Do not use outside knowledge."
# )

# PROMPT_COMPLIANCE = """\
# Rule category: {rule_subject}

# DPDP Act rules for this category (with deontic type, section, and condition):
# {rule_triples}

# Pre-screening result (rule-based, not LLM):
# {gap_prescreen}

# Most relevant excerpts from the business policy:
# {context}

# Question: Considering the deontic types above, does the business policy satisfy \
# all OBLIGATION and PROHIBITION rules in this category?

# Respond on line 1 with exactly one of: PASS | FAIL | INSUFFICIENT_EVIDENCE
#   PASS                  — policy clearly satisfies all obligations and prohibitions
#   FAIL                  — policy violates or omits one or more obligations/prohibitions
#   INSUFFICIENT_EVIDENCE — the retrieved excerpts do not address this rule category

# On line 2, cite the specific rule violated or satisfied and the relevant policy \
# excerpt (max 40 words)."""

SYSTEM_COMPLIANCE = (
    "You are a pragmatic compliance expert specializing in India's Digital Personal Data "
    "Protection Act 2023 (DPDP Act). "
    "You will receive one regulatory rule-group with deontic typing "
    "(OBLIGATION = must do, PROHIBITION = must not do, PERMISSION = may do, "
    "ENTITLEMENT = Data Principal's right) and the most relevant excerpts from "
    "a business policy document. "
    "Evaluate if the policy SUBSTANTIALLY SATISFIES the core intent of the rule. "
    "Do not fail a policy just because it uses different terminology or omits minor sub-clauses, "
    "as long as the primary obligation is met. "
    "Use ONLY the provided excerpts. Do not use outside knowledge."
)

PROMPT_COMPLIANCE = """\
Rule category: {rule_subject}

DPDP Act rules for this category (with deontic type, section, and condition):
{rule_triples}

Pre-screening result (rule-based, not LLM):
{gap_prescreen}

Most relevant excerpts from the business policy:
{context}

Question: Considering the deontic types above, does the business policy substantially \
satisfy the OBLIGATION and PROHIBITION rules in this category?

Respond on line 1 with exactly one of: PASS | FAIL | INSUFFICIENT_EVIDENCE
  PASS                  — The policy meets the core intent of the obligations and respects prohibitions.
  FAIL                  — The policy explicitly contradicts a prohibition, or completely omits a critical mandatory obligation.
  INSUFFICIENT_EVIDENCE — The retrieved excerpts do not address this rule category at all.

On line 2, cite the specific rule violated or satisfied and the relevant policy \
excerpt (max 40 words)."""

SYSTEM_QA = (
    "You are a legal assistant specializing in India's Digital Personal Data "
    "Protection Act 2023. "
    "Answer questions ONLY using the provided context chunks. "
    "Cite every factual claim with [chunk_N] where N is the chunk id. "
    "If the context lacks enough information, say exactly: "
    "'The provided documents do not contain sufficient information to answer this question.' "
    "Do not use outside knowledge."
)

PROMPT_QA = """\
Context chunks:
{context}

Question: {query}

Answer (with inline [chunk_N] citations):"""


# ── Generator ────────────────────────────────────────────────────────────────

class ComplianceGenerator:

    def __init__(self):
        self._llm = None        # lazy — no API call at import time
        self._deontic_kg = None # lazy

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_llm_client()
        return self._llm

    @property
    def deontic_kg(self):
        if self._deontic_kg is None:
            self._deontic_kg = DeonticKG()
        return self._deontic_kg

    # ── Task 1: Rule-group compliance check ───────────────────────────────────

    def check_compliance_for_rule(
        self,
        rule_subject:    str,
        rule_triples:    List[Dict],
        relevant_chunks: List[Dict],
        policy_text:     str = "",
    ) -> Dict:
        """
        Check whether the business policy satisfies ONE KG rule-group.

        Steps:
          1. Use DeonticKG.format_for_prompt() for richer, typed rule text
          2. Run dynamic gap pre-screening on retrieved chunk text
          3. Send deontic-typed rules + gap prescreen + chunks to LLM

        Args:
            rule_subject    : e.g. 'Consent', 'Data Fiduciary'
            rule_triples    : triples for this subject from the static KG
            relevant_chunks : top-k chunks retrieved for this rule
            policy_text     : full policy text for gap pre-screening (optional)

        Returns:
            verdict        : 'pass' | 'fail' | 'insufficient_evidence' | 'unknown'
            reason         : short explanation
            raw            : raw LLM reply
            gap_prescreen  : dict from rule-based pre-screening
        """
        # 1. Format rules with deontic types (section + condition included)
        rules_text = self.deontic_kg.format_for_prompt(
            rule_triples,
            include_section=True,
            include_condition=True,
            include_consequence=False,
        )

        # 2. Rule-based gap pre-screening on retrieved chunk text
        chunk_text_combined = " ".join(c.get("text", "") for c in relevant_chunks)
        if chunk_text_combined.strip():
            gap = self.deontic_kg.detect_gaps(chunk_text_combined)
            gap_text = (
                f"Rule-based pre-screen: "
                f"covered={gap['summary']['covered']} / "
                f"{gap['summary']['total_obligations']} obligations, "
                f"missing={gap['summary']['missing']}, "
                f"violations={gap['summary']['violations']}"
            )
        else:
            gap      = {}
            gap_text = "Pre-screening skipped (no text retrieved)"

        # 3. LLM check with deontic-enriched prompt
        context = _chunks_to_context(relevant_chunks)
        prompt  = PROMPT_COMPLIANCE.format(
            rule_subject=rule_subject,
            rule_triples=rules_text,
            gap_prescreen=gap_text,
            context=context,
        )

        try:
            reply = self.llm.chat(prompt, system=SYSTEM_COMPLIANCE)
        except Exception as e:
            log.error("LLM call failed for rule '%s': %s", rule_subject, e)
            return {
                "verdict": "unknown", "reason": str(e),
                "raw": "", "gap_prescreen": gap,
            }

        verdict, reason = _parse_compliance_reply(reply)
        return {
            "verdict":       verdict,
            "reason":        reason,
            "raw":           reply,
            "gap_prescreen": gap,
        }

    # ── Task 2: DPDP QA ───────────────────────────────────────────────────────

    def answer_query(self, query: str, chunks: List[Dict]) -> str:
        """Answer a DPDP question from reranked chunks with citation anchoring."""
        context = _chunks_to_context(chunks)
        prompt  = PROMPT_QA.format(context=context, query=query)
        try:
            return self.llm.chat(prompt, system=SYSTEM_QA)
        except Exception as e:
            log.error("LLM call failed: %s", e)
            return f"Error calling LLM: {e}"


# ── Module-level helpers ──────────────────────────────────────────────────────

def _triples_to_text(triples: List[Dict]) -> str:
    lines = []
    for t in triples:
        s = t.get("subject", "").strip()
        p = t.get("predicate", "relatedTo").strip()
        o = t.get("object", "").strip()
        if s and o:
            lines.append(f"- {s} {p} {o}")
    return "\n".join(lines) if lines else "(no rules provided)"


def _chunks_to_context(chunks: List[Dict]) -> str:
    parts = []
    for c in chunks:
        cid   = c.get("id", "?")
        fname = c.get("filename", "unknown")
        text  = c.get("text", "")
        parts.append(f"[chunk_{cid}] ({fname})\n{text}")
    return "\n\n---\n\n".join(parts)


def _parse_compliance_reply(reply: str) -> Tuple[str, str]:
    """Parse PASS / FAIL / INSUFFICIENT_EVIDENCE from LLM reply."""
    lines  = [l.strip() for l in reply.strip().splitlines() if l.strip()]
    if not lines:
        return "unknown", reply[:200]

    first  = lines[0].upper()
    reason = lines[1] if len(lines) > 1 else "N/A"

    if first.startswith("PASS"):
        return "pass", reason
    if first.startswith("FAIL"):
        return "fail", reason
    if first.startswith("INSUFFICIENT"):
        return "insufficient_evidence", reason

    # Heuristic fallback
    low = reply.lower()
    if any(w in low for w in ["violat", "fail", "non-complian", "breach"]):
        return "fail", reply[:300]
    if any(w in low for w in ["insufficient", "not address", "no mention", "unclear"]):
        return "insufficient_evidence", reply[:300]
    return "pass", reply[:300]
