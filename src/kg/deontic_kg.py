"""
deontic_kg.py — Static + Dynamic (Eventic) KG with deontic reasoning.

Fixes in this version:
  - SYNONYM_MAP added: policy language → canonical DPDP object name.
    "delete" / "remove" / "erase" all map to "Erasure".
    "share" / "disclose" map to "Third Party".
    "permission" / "authorise" map to "Consent". Etc.
  - detect_gaps() applies synonym normalisation before matching.
  - extract_dynamic_triples() applies synonym normalisation on policy text.
  - BM25/dense search is upstream; this module operates on canonical terms.
"""

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import KG_DIR

log = logging.getLogger("deontic_kg")

STATIC_KG_PATH = KG_DIR / "dpdp_deontic_kg.json"
LEGACY_KG_PATH = KG_DIR / "dpdp_triples.json"

# ── Deontic word taxonomy ─────────────────────────────────────────────────────
DEONTIC_PATTERNS: Dict[str, List[str]] = {
    "obligation": [
        r"\bmust\b", r"\bshall\b", r"\bis required to\b",
        r"\bis obligated to\b", r"\bis mandated to\b",
        r"\bhas to\b", r"\bwill ensure\b", r"\bwill provide\b",
        r"\bwill obtain\b", r"\bwill notify\b", r"\bcommit to\b",
    ],
    "prohibition": [
        r"\bmust not\b", r"\bshall not\b", r"\bmay not\b",
        r"\bis prohibited from\b", r"\bis forbidden\b",
        r"\bcannot\b", r"\bwill not\b", r"\bdoes not sell\b",
        r"\bdoes not share\b", r"\bdoes not disclose\b",
    ],
    "permission": [
        r"\bmay\b(?!\s+not)", r"\bcan\b(?!\s+not)",
        r"\bis permitted to\b", r"\bis allowed to\b",
        r"\bat its discretion\b",
    ],
    "entitlement": [
        r"\bhas the right to\b", r"\bhave the right to\b",
        r"\bis entitled to\b", r"\bmay request\b",
        r"\bcan request\b", r"\bcan withdraw\b",
    ],
}

# ── Synonym map ───────────────────────────────────────────────────────────────
# Maps common policy language → canonical DPDP object names used in the KG.
# Fixes the brittle matching bug where "delete" would miss "Erasure".
SYNONYM_MAP: Dict[str, str] = {
    # Erasure / deletion
    "delete":        "Erasure",
    "deleted":       "Erasure",
    "deletion":      "Erasure",
    "remove":        "Erasure",
    "removal":       "Erasure",
    "erase":         "Erasure",
    "erased":        "Erasure",
    "erasure":       "Erasure",
    "destroy":       "Erasure",
    "purge":         "Erasure",
    # Consent
    "permission":    "Consent",
    "approve":       "Consent",
    "approval":      "Consent",
    "authorise":     "Consent",
    "authorize":     "Consent",
    "authorisation": "Consent",
    "agree":         "Consent",
    "agreement":     "Consent",
    # Third party / sharing
    "share":         "Third Party",
    "sharing":       "Third Party",
    "disclose":      "Third Party",
    "disclosure":    "Third Party",
    "transfer":      "Third Party",
    "transmit":      "Third Party",
    # Breach
    "breach":        "Data Breach",
    "incident":      "Data Breach",
    "hack":          "Data Breach",
    "leak":          "Data Breach",
    "compromise":    "Data Breach",
    # Security
    "secure":        "Security",
    "security":      "Security",
    "protect":       "Security",
    "safeguard":     "Security",
    "encrypt":       "Encryption",
    "encryption":    "Encryption",
    # Notice
    "inform":        "Notice",
    "notify":        "Notice",
    "notification":  "Notice",
    "inform":        "Notice",
    # Retention
    "retain":        "Retention",
    "retention":     "Retention",
    "store":         "Retention",
    "storage":       "Retention",
    "keep":          "Retention",
    # Children
    "child":         "Children",
    "minor":         "Children",
    "minors":        "Children",
    "underage":      "Children",
    # Access
    "access":        "Access",
    "view":          "Access",
    "inspect":       "Access",
    # Correction
    "correct":       "Correction",
    "update":        "Correction",
    "amend":         "Correction",
    "rectify":       "Correction",
    # Grievance
    "complain":      "Grievance",
    "complaint":     "Grievance",
    "dispute":       "Grievance",
    "appeal":        "Grievance",
    "grievance":     "Grievance",
}

# Canonical object names used in the KG (for entity spotting in policy text)
DPDP_ENTITIES = {
    "data fiduciary", "data principal", "consent manager",
    "data protection officer", "data protection board",
    "significant data fiduciary", "data processor",
    "we", "our company", "the company", "organization", "us",
}

# Extended canonical object set including synonyms
DPDP_OBJECTS = {
    "consent", "notice", "personal data", "sensitive personal data",
    "data breach", "breach", "grievance", "complaint", "erasure",
    "deletion", "correction", "access", "withdrawal", "nomination",
    "penalty", "audit", "security", "encryption", "retention",
    "purpose", "third party", "children", "minors", "cross border",
    "transfer", "delete", "remove", "share", "disclose",
}


def _apply_synonyms(text: str) -> str:
    """
    Replace known policy synonyms with canonical DPDP object names.
    Applied word-by-word so 'delete' → 'Erasure' before matching.
    """
    words = text.lower().split()
    normalised = []
    for w in words:
        clean = re.sub(r'[^\w]', '', w)  # strip punctuation from token
        normalised.append(SYNONYM_MAP.get(clean, w))
    return " ".join(normalised)


class DeonticKG:

    def __init__(self):
        self.static_graph   = nx.MultiDiGraph()
        self.static_triples: List[Dict] = []
        self.by_subject:    Dict[str, List[Dict]] = defaultdict(list)
        self.by_deontic:    Dict[str, List[Dict]] = defaultdict(list)
        self._load_static_kg()

    def _load_static_kg(self):
        path = STATIC_KG_PATH if STATIC_KG_PATH.exists() else LEGACY_KG_PATH
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        for t in raw:
            triple = {
                "subject":     t.get("subject", "").strip(),
                "predicate":   t.get("predicate", "relatedTo").strip(),
                "object":      t.get("object", "").strip(),
                "deontic":     t.get("deontic", "obligation"),
                "section":     t.get("section", ""),
                "condition":   t.get("condition", ""),
                "consequence": t.get("consequence", ""),
            }
            if not triple["subject"] or not triple["object"]:
                continue
            self.static_triples.append(triple)
            self.by_subject[triple["subject"]].append(triple)
            self.by_deontic[triple["deontic"]].append(triple)
            self.static_graph.add_edge(
                triple["subject"], triple["object"],
                key=triple["predicate"],
                **{k: v for k, v in triple.items()
                   if k not in ("subject", "object")},
            )
        log.info(
            "Static KG: %d triples | obligations=%d prohibitions=%d permissions=%d entitlements=%d",
            len(self.static_triples),
            len(self.by_deontic.get("obligation", [])),
            len(self.by_deontic.get("prohibition", [])),
            len(self.by_deontic.get("permission", [])),
            len(self.by_deontic.get("entitlement", [])),
        )

    def get_relevant_triples(
        self, subject: str, deontic_filter: Optional[str] = None,
    ) -> List[Dict]:
        triples = self.by_subject.get(subject, [])
        if deontic_filter:
            triples = [t for t in triples if t["deontic"] == deontic_filter]
        return triples

    def get_all_obligations(self) -> List[Dict]:
        return self.by_deontic.get("obligation", [])

    def get_all_prohibitions(self) -> List[Dict]:
        return self.by_deontic.get("prohibition", [])

    def format_for_prompt(
        self, triples: List[Dict],
        include_section: bool = True,
        include_condition: bool = True,
        include_consequence: bool = False,
    ) -> str:
        lines = []
        for t in triples:
            dtype  = t.get("deontic", "obligation").upper()
            section = t.get("section", "")
            cond    = t.get("condition", "")
            conseq  = t.get("consequence", "")
            header  = f"[{dtype}"
            if include_section and section:
                header += f" {section}"
            header += f"] {t['subject']} {t['predicate']} {t['object']}"
            lines.append(header)
            if include_condition and cond:
                lines.append(f"  Condition: {cond}")
            if include_consequence and conseq and conseq != "N/A":
                lines.append(f"  Consequence: {conseq}")
        return "\n".join(lines) if lines else "(no rules)"

    def extract_dynamic_triples(self, policy_text: str) -> List[Dict]:
        """
        Extract deontic triples from policy text with synonym normalisation.
        """
        sentences = re.split(r'(?<=[.!?])\s+', policy_text)
        dynamic_triples = []
        for sent in sentences:
            if len(sent.split()) < 4:
                continue
            sent_lower = sent.lower().strip()
            sent_normalised = _apply_synonyms(sent_lower)

            detected_deontic = None
            for dtype, patterns in DEONTIC_PATTERNS.items():
                for pat in patterns:
                    if re.search(pat, sent_lower):
                        detected_deontic = dtype
                        break
                if detected_deontic:
                    break
            if not detected_deontic:
                continue

            subject_found = None
            for ent in sorted(DPDP_ENTITIES, key=len, reverse=True):
                if ent in sent_lower:
                    subject_found = ent.title()
                    break

            # Use normalised text for object detection (catches "delete" → "Erasure")
            object_found = None
            for obj in sorted(DPDP_OBJECTS | set(SYNONYM_MAP.keys()), key=len, reverse=True):
                if obj in sent_normalised or obj in sent_lower:
                    canonical = SYNONYM_MAP.get(obj, obj.title())
                    object_found = canonical
                    break

            if subject_found and object_found:
                dynamic_triples.append({
                    "subject":  subject_found,
                    "predicate": detected_deontic,
                    "object":   object_found,
                    "deontic":  detected_deontic,
                    "sentence": sent[:200],
                })
        log.info("Dynamic KG: %d triples extracted", len(dynamic_triples))
        return dynamic_triples

    def detect_gaps(self, policy_text: str) -> Dict:
        """
        Rule-based pre-screening with synonym normalisation.
        Returns covered/missing/violations for ALL static obligations and prohibitions.
        """
        dynamic = self.extract_dynamic_triples(policy_text)

        # Build lookup: canonical_object.lower() → list of dynamic triples
        dynamic_by_object: Dict[str, List[Dict]] = defaultdict(list)
        for dt in dynamic:
            dynamic_by_object[dt["object"].lower()].append(dt)
            # Also index the synonym-normalised form
            canonical = SYNONYM_MAP.get(dt["object"].lower(), dt["object"].lower())
            dynamic_by_object[canonical.lower()].append(dt)

        covered, missing, violations = [], [], []

        for triple in self.by_deontic.get("obligation", []):
            obj_lower = triple["object"].lower()
            canonical = SYNONYM_MAP.get(obj_lower, obj_lower)
            matches   = dynamic_by_object.get(obj_lower, []) + dynamic_by_object.get(canonical, [])
            matches   = list({id(m): m for m in matches}.values())  # dedup

            if matches:
                compatible = [m for m in matches
                              if m["deontic"] in ("obligation", "entitlement", "permission")]
                if compatible:
                    covered.append({"triple": triple, "matched": compatible[0]["sentence"]})
                else:
                    violations.append({"triple": triple, "matched": matches[0]["sentence"]})
            else:
                missing.append({"triple": triple})

        for triple in self.by_deontic.get("prohibition", []):
            obj_lower = triple["object"].lower()
            canonical = SYNONYM_MAP.get(obj_lower, obj_lower)
            matches   = dynamic_by_object.get(obj_lower, []) + dynamic_by_object.get(canonical, [])
            perm_matches = [m for m in matches if m["deontic"] == "permission"]
            if perm_matches:
                violations.append({"triple": triple, "matched": perm_matches[0]["sentence"]})

        return {
            "covered":    covered,
            "missing":    missing,
            "violations": violations,
            "summary": {
                "total_obligations": len(self.by_deontic.get("obligation", [])),
                "covered":           len(covered),
                "missing":           len(missing),
                "violations":        len(violations),
            },
        }

    def summarise(self) -> str:
        return (
            f"Static DPDP KG: {len(self.static_triples)} triples | "
            f"obligations={len(self.by_deontic.get('obligation',[]))} "
            f"prohibitions={len(self.by_deontic.get('prohibition',[]))} "
            f"permissions={len(self.by_deontic.get('permission',[]))} "
            f"entitlements={len(self.by_deontic.get('entitlement',[]))}"
        )
