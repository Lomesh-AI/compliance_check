"""
build_gold.py — Build clean gold evaluation datasets.

Fixes in this version:
  #5  Eval set expanded: 50 retrieval queries, 15 generation QA pairs
  #6  Circular eval addressed: gold labels built using a DIFFERENT model
      (BAAI/bge-base-en-v1.5, larger) than the one used at query time
      (BAAI/bge-small-en-v1.5). Still requires manual verification —
      a TODO comment marks every auto-assigned ID for human review.
      The script writes manual_verification_needed: true on each entry.
  #8  Compliance gold set added: gold_compliance.json with 10 labeled
      (query, expected_verdict, expected_rule) triples for eval_compliance().

NOTE on circular eval: the only real fix is manual verification.
After running this script, open data/eval/gold_retrieval.json and for each
entry check that the chunk_ids listed actually contain the answer text.
Set "manual_verified": true on each entry you have checked.
The eval harness will warn if unverified entries are used.

RUN:
    python src/eval/build_gold.py
"""

import json
import logging
from pathlib import Path
from typing import List, Dict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import CHUNKS_PATH, FAISS_PATH, EVAL_DIR

log = logging.getLogger("build_gold")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Gold labeling model: intentionally DIFFERENT from query-time model (bge-small)
# Using bge-base (larger) to reduce circularity. Still not a perfect fix —
# see manual_verification_needed flag.
GOLD_EMBED_MODEL = "BAAI/bge-base-en-v1.5"
GOLD_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ── 50 DPDP retrieval queries ─────────────────────────────────────────────────
# Each has a synthetic_answer written from the Act text.
# Used only for chunk labeling — not stored in the gold file.

RETRIEVAL_QUERIES = [
    # Consent (10)
    {"query": "What are the requirements for valid consent under DPDP Act?",
     "synthetic_answer": "Consent must be free, specific, informed, unconditional, and unambiguous. It must be given through a clear affirmative action. Consent obtained by fraud, misrepresentation or coercion is invalid."},
    {"query": "How can a Data Principal withdraw consent?",
     "synthetic_answer": "A Data Principal can withdraw consent at any time through the same ease with which it was given. Withdrawal does not affect the lawfulness of processing before withdrawal. The Data Fiduciary must stop processing within reasonable time."},
    {"query": "What is a Consent Manager and what is its role?",
     "synthetic_answer": "A Consent Manager is an entity registered with the Data Protection Board that enables Data Principals to give, manage, review, and withdraw consent through an accessible and transparent platform. It is accountable to the Data Principal."},
    {"query": "Can consent be bundled with other consent or terms of service?",
     "synthetic_answer": "No. Consent must be specific and unconditional. Bundling consent with other agreements or making it a precondition for service is not permitted under the DPDP Act."},
    {"query": "What happens to processing if a Data Principal withdraws consent?",
     "synthetic_answer": "Upon withdrawal of consent, the Data Fiduciary must cease processing the personal data within a reasonable timeframe. Prior processing remains lawful. The Data Fiduciary may also cease providing services if the processing was essential."},
    {"query": "Must consent be in writing under the DPDP Act?",
     "synthetic_answer": "The DPDP Act does not require written consent but requires a clear affirmative action. Digital consent through clicking, ticking a box, or similar unambiguous action is valid. Pre-ticked boxes and silence do not constitute consent."},
    {"query": "What language must consent notices be given in?",
     "synthetic_answer": "Notices must be given in English or any language specified in the Eighth Schedule of the Constitution so that the Data Principal understands them. The policy must be accessible and in plain language."},
    {"query": "Can a Data Fiduciary process data beyond the consented purpose?",
     "synthetic_answer": "No. A Data Fiduciary can process personal data only for the purpose for which consent was obtained. Processing beyond the stated purpose requires fresh consent."},
    {"query": "What must a notice contain before collecting personal data?",
     "synthetic_answer": "Before collecting data a Data Fiduciary must provide notice describing: the personal data to be collected, purpose of processing, how to exercise rights, and how to contact the grievance officer and file a complaint with the Data Protection Board."},
    {"query": "Who can give consent on behalf of a child?",
     "synthetic_answer": "A parent or lawful guardian must give verifiable consent on behalf of a child. A child means a person under 18 years of age. The Data Fiduciary must verify that the person giving consent is indeed the parent or guardian."},

    # Data Fiduciary obligations (10)
    {"query": "What security safeguards must a Data Fiduciary implement?",
     "synthetic_answer": "Data Fiduciaries must implement appropriate technical and organisational measures to protect personal data including encryption of data at rest and in transit, access controls, and regular security audits to prevent unauthorised access, disclosure, or loss."},
    {"query": "What must a Data Fiduciary do when it no longer needs personal data?",
     "synthetic_answer": "When the purpose of processing is fulfilled or consent is withdrawn the Data Fiduciary must erase personal data and cause the Data Processor to erase it within the period specified by the Board."},
    {"query": "What are the obligations of a Data Fiduciary regarding data accuracy?",
     "synthetic_answer": "Data Fiduciaries must make reasonable efforts to ensure personal data is complete, accurate, and consistent especially when used to make decisions affecting the Data Principal or when disclosed to other fiduciaries."},
    {"query": "Must a Data Fiduciary appoint a grievance officer?",
     "synthetic_answer": "Yes. Every Data Fiduciary must publish the contact details of a grievance officer on their website who will address complaints from Data Principals within a prescribed period. The officer must resolve grievances before the Principal can approach the Board."},
    {"query": "What are the additional obligations of a Significant Data Fiduciary?",
     "synthetic_answer": "A Significant Data Fiduciary must appoint a Data Protection Officer based in India, conduct Data Protection Impact Assessments, perform periodic data audits, and comply with additional obligations prescribed by the Central Government."},
    {"query": "How is a Significant Data Fiduciary designated?",
     "synthetic_answer": "The Central Government designates a Significant Data Fiduciary based on volume and sensitivity of data processed, risk to rights of Data Principals, potential impact on national security, sovereignty, and public order."},
    {"query": "Can a Data Fiduciary appoint a Data Processor?",
     "synthetic_answer": "Yes. A Data Fiduciary can appoint a Data Processor to process data on its behalf through a valid contract. The Data Fiduciary remains responsible for compliance and must ensure the Processor follows all obligations under the Act."},
    {"query": "What records must a Data Fiduciary maintain?",
     "synthetic_answer": "Data Fiduciaries must maintain compliance records as prescribed by the Board including records of consent obtained, purposes of processing, data retention periods, and data breach incidents."},
    {"query": "Can a Data Fiduciary transfer personal data outside India?",
     "synthetic_answer": "Personal data may be transferred to countries or territories notified by the Central Government. Transfer to blacklisted countries is prohibited. Data Fiduciaries must comply with conditions prescribed for cross-border transfers."},
    {"query": "What is purpose limitation under DPDP Act?",
     "synthetic_answer": "Purpose limitation means personal data can only be processed for the specific purpose stated in the notice and consented to by the Data Principal. Data cannot be used for a different purpose without obtaining fresh consent."},

    # Data Principal rights (10)
    {"query": "What is the right to access personal data under DPDP Act?",
     "synthetic_answer": "A Data Principal has the right to obtain confirmation of whether their personal data is being processed, a summary of data being processed, identities of Data Fiduciaries who have received it, and any other prescribed information."},
    {"query": "How can a Data Principal correct inaccurate personal data?",
     "synthetic_answer": "A Data Principal can request the Data Fiduciary to correct inaccurate, incomplete, or misleading personal data. The Data Fiduciary must correct or complete the data within the prescribed time period."},
    {"query": "What is the right to erasure of personal data?",
     "synthetic_answer": "A Data Principal can request erasure of personal data when the purpose for which it was collected is fulfilled or when consent is withdrawn. The Data Fiduciary must erase the data within the prescribed period."},
    {"query": "Can a Data Principal nominate someone to exercise their rights?",
     "synthetic_answer": "Yes. A Data Principal can nominate another person to exercise their rights under the Act in the event of the Principal's death or incapacity. The nominee acts on behalf of the Principal after providing prescribed proof."},
    {"query": "What are the duties of a Data Principal?",
     "synthetic_answer": "A Data Principal must not impersonate another person while providing personal data, must not suppress material information, and must not register a false or frivolous grievance or complaint with a Data Fiduciary or the Data Protection Board."},
    {"query": "How can a Data Principal approach the Data Protection Board?",
     "synthetic_answer": "After exhausting the Data Fiduciary's grievance redressal mechanism, a Data Principal can file a complaint with the Data Protection Board. The Board will inquire into the complaint and may impose penalties or direct remedial action."},
    {"query": "What right does a Data Principal have regarding automated decisions?",
     "synthetic_answer": "Where significant decisions affecting a Data Principal are made through automated means, the Principal has the right to request human review of the decision and to receive an explanation of the decision-making process."},
    {"query": "Can a Data Principal object to processing of their personal data?",
     "synthetic_answer": "A Data Principal can withdraw consent at any time which has the effect of objecting to continued processing. For processing based on legitimate interests the Principal may raise a grievance with the Data Fiduciary."},
    {"query": "What information must a Data Fiduciary provide when a Data Principal requests access?",
     "synthetic_answer": "A Data Fiduciary must provide confirmation of processing, a summary of the data processed, identities of other fiduciaries who received the data, and any other information prescribed by the Board within the prescribed time period."},
    {"query": "Can a Data Principal request deletion of data that is required by law?",
     "synthetic_answer": "No. The right to erasure does not apply where the Data Fiduciary is required to retain personal data by law or court order. Data required for compliance with legal obligations cannot be deleted on the Principal's request alone."},

    # Data breach and penalties (10)
    {"query": "What must a Data Fiduciary do in case of a personal data breach?",
     "synthetic_answer": "In the event of a data breach a Data Fiduciary must notify the Data Protection Board and each affected Data Principal without undue delay. The notification must describe the nature of the breach and remedial measures taken."},
    {"query": "What are the penalties for failing to notify a data breach?",
     "synthetic_answer": "Failure to notify the Data Protection Board or affected Data Principals of a personal data breach may attract a financial penalty of up to two hundred and fifty crore rupees as specified in the Schedule."},
    {"query": "What is the maximum penalty under the DPDP Act?",
     "synthetic_answer": "The Schedule to the DPDP Act specifies penalties for different violations ranging from fifty crore to two hundred and fifty crore rupees. The Data Protection Board determines the penalty after considering severity, intent, and repetition."},
    {"query": "What factors does the Board consider when imposing a penalty?",
     "synthetic_answer": "The Board considers the nature, gravity, and duration of the breach, the type of data affected, whether the breach was intentional or negligent, the steps taken to mitigate the harm, and whether the entity reported the breach voluntarily."},
    {"query": "Can a Data Fiduciary appeal a Board penalty?",
     "synthetic_answer": "Yes. A person aggrieved by any order of the Data Protection Board can appeal to the Appellate Tribunal within the period prescribed. The Tribunal may stay the operation of the order pending the appeal."},
    {"query": "What penalty applies for processing children's data without consent?",
     "synthetic_answer": "Processing children's personal data without verifiable parental consent may attract a penalty of up to two hundred crore rupees as specified in the Schedule to the DPDP Act."},
    {"query": "Can individuals be penalised under the DPDP Act?",
     "synthetic_answer": "The DPDP Act primarily imposes obligations on Data Fiduciaries. However Data Principals who impersonate others or file false complaints may face penalties. Data Protection Officers of significant fiduciaries may also face liability for non-compliance."},
    {"query": "What are the penalties for non-compliance with security safeguards?",
     "synthetic_answer": "Failure to implement reasonable security safeguards resulting in a personal data breach may attract a penalty of up to two hundred and fifty crore rupees as specified in the Schedule."},
    {"query": "Is there a penalty for not providing a grievance mechanism?",
     "synthetic_answer": "Failure to maintain a grievance redressal mechanism or to respond to a Data Principal's complaint may attract penalties under the Act. The Board may also direct the Data Fiduciary to establish an adequate mechanism."},
    {"query": "What is the role of the Data Protection Board in adjudication?",
     "synthetic_answer": "The Data Protection Board adjudicates complaints from Data Principals and references from the Central Government. It has powers of a civil court to summon persons, receive evidence, and impose penalties after due inquiry."},

    # Special categories and exemptions (10)
    {"query": "What additional protections apply to sensitive personal data?",
     "synthetic_answer": "Sensitive personal data including health, financial, biometric, and sexual orientation data requires explicit consent. Data Fiduciaries must implement enhanced security measures and conduct Data Protection Impact Assessments before processing."},
    {"query": "What restrictions exist on processing children's data?",
     "synthetic_answer": "Data Fiduciaries must not track children's behaviour, target advertising at children, or process data in a manner detrimental to children's wellbeing. Verifiable parental consent is mandatory for any processing of children's data."},
    {"query": "Are there exemptions to consent requirements under DPDP Act?",
     "synthetic_answer": "Consent is not required for State processing for subsidies, benefits and services; medical emergencies; employment purposes; national security purposes; or research and archiving with appropriate safeguards."},
    {"query": "Can the government exempt itself from DPDP Act obligations?",
     "synthetic_answer": "The Central Government can exempt instrumentalities of the State from provisions of the Act in the interest of national security, sovereignty, public order, or friendly relations with foreign states. Such exemptions must be in the public interest."},
    {"query": "What is the treatment of personal data for research purposes?",
     "synthetic_answer": "Processing for research, archiving, or statistical purposes may be exempted from certain requirements subject to prescribed safeguards. The data must be anonymised where possible and must not be used for decisions affecting individual Data Principals."},
    {"query": "Does DPDP Act apply to data processed outside India?",
     "synthetic_answer": "The DPDP Act applies to processing of digital personal data within India and to processing outside India if it relates to profiling of Data Principals in India or offering goods and services to persons within India."},
    {"query": "What are the cross-border transfer restrictions under DPDP Act?",
     "synthetic_answer": "Personal data can only be transferred to countries notified by the Central Government. Transfer to blacklisted countries is prohibited. Transfers must comply with additional conditions prescribed for each notified country."},
    {"query": "Are media and journalism exempt from DPDP Act?",
     "synthetic_answer": "The Central Government may prescribe exemptions for processing by the press, research, education, or artistic purposes subject to safeguards. Such exemptions are not blanket and must be balanced against privacy rights."},
    {"query": "What happens to existing data collected before DPDP Act?",
     "synthetic_answer": "Data Fiduciaries who collected data before the Act must obtain fresh consent from Data Principals within the time prescribed by the Central Government. They must also provide notice to existing customers about their rights."},
    {"query": "Is anonymised data covered by DPDP Act?",
     "synthetic_answer": "The DPDP Act applies only to personal data. Anonymised data that cannot be re-identified is not personal data and falls outside the scope of the Act. Data Fiduciaries must ensure anonymisation is irreversible."},
]

# ── 15 generation QA pairs ────────────────────────────────────────────────────
GENERATION_GOLD = [
    {"query": "What is the right of a Data Principal to withdraw consent?",
     "gold_answer": "A Data Principal has the right to withdraw consent given to a Data Fiduciary at any time. Upon withdrawal, the Data Fiduciary must cease processing within a reasonable timeframe. Withdrawal does not affect the lawfulness of prior processing.",
     "section_reference": "Section 6(4), DPDP Act 2023"},
    {"query": "What notice must a Data Fiduciary give before collecting data?",
     "gold_answer": "Before or at the time of collecting data the Data Fiduciary must provide notice in clear and plain language describing: the personal data to be collected, the purpose, how to exercise rights, and how to contact the grievance officer and approach the Data Protection Board.",
     "section_reference": "Section 5, DPDP Act 2023"},
    {"query": "What are a Significant Data Fiduciary's extra obligations?",
     "gold_answer": "A Significant Data Fiduciary must appoint a Data Protection Officer resident in India, conduct Data Protection Impact Assessments, perform periodic data audits, and comply with additional obligations prescribed by the Central Government.",
     "section_reference": "Section 10, DPDP Act 2023"},
    {"query": "What information can a Data Principal access about their data?",
     "gold_answer": "A Data Principal has the right to obtain confirmation of processing, a summary of data processed, the identities of all other Data Fiduciaries who received the data, and any other information prescribed by the Board.",
     "section_reference": "Section 11, DPDP Act 2023"},
    {"query": "When can personal data be processed without consent?",
     "gold_answer": "Personal data may be processed without consent for: State functions providing subsidies and services; medical emergencies; employment purposes; compliance with law; national security; prevention of offences; and research with prescribed safeguards.",
     "section_reference": "Section 7, DPDP Act 2023"},
    {"query": "What restrictions apply to processing children's personal data?",
     "gold_answer": "Data Fiduciaries must obtain verifiable parental consent, must not track children's behaviour, must not target advertising at children, and must not process data in a manner detrimental to children's wellbeing. A child is a person under 18 years.",
     "section_reference": "Section 9, DPDP Act 2023"},
    {"query": "What penalty applies for failure to notify a data breach?",
     "gold_answer": "Failure to notify the Data Protection Board or affected Data Principals of a personal data breach may attract a financial penalty of up to two hundred and fifty crore rupees as specified in the Schedule to the DPDP Act 2023.",
     "section_reference": "Schedule Item 2, DPDP Act 2023"},
    {"query": "How can the State be exempted from DPDP Act provisions?",
     "gold_answer": "The Central Government may exempt any instrumentality of the State from provisions of the Act in the interest of sovereignty, national security, friendly relations with foreign states, or maintenance of public order.",
     "section_reference": "Section 17, DPDP Act 2023"},
    {"query": "What are the conditions for valid consent under DPDP Act?",
     "gold_answer": "Consent must be free, specific, informed, unconditional, and unambiguous. It must be given through a clear affirmative action. It must be separate from other consent. Pre-ticked boxes, silence, and bundled consent are not valid.",
     "section_reference": "Section 6(1), DPDP Act 2023"},
    {"query": "What are the data accuracy obligations of a Data Fiduciary?",
     "gold_answer": "Data Fiduciaries must make reasonable efforts to ensure that personal data is complete, accurate, and consistent, especially when used for decisions affecting the Data Principal or when likely to be disclosed to other Data Fiduciaries.",
     "section_reference": "Section 8(3), DPDP Act 2023"},
    {"query": "What security measures must Data Fiduciaries implement?",
     "gold_answer": "Data Fiduciaries must implement appropriate security safeguards to protect personal data from breach. This includes encryption at rest and in transit, access controls, and regular audits. Failure may attract a penalty of up to INR 250 crore.",
     "section_reference": "Section 8(5), DPDP Act 2023"},
    {"query": "What are the Data Principal's duties under the DPDP Act?",
     "gold_answer": "A Data Principal must not impersonate another person when providing data, must not suppress material information, and must not register false or frivolous grievances. Violations may attract penalties on the Data Principal.",
     "section_reference": "Section 15, DPDP Act 2023"},
    {"query": "What is the right to grievance redressal for a Data Principal?",
     "gold_answer": "A Data Principal can file a complaint with the Data Fiduciary's grievance officer. If unsatisfied, they can approach the Data Protection Board. The Board will inquire and may impose penalties or direct remedial action.",
     "section_reference": "Section 13, DPDP Act 2023"},
    {"query": "Can a Data Principal nominate someone to exercise rights?",
     "gold_answer": "Yes. A Data Principal can nominate another person to exercise their rights in the event of death or incapacity. The nominee must provide prescribed proof of their status before the Data Fiduciary can act on their instructions.",
     "section_reference": "Section 14, DPDP Act 2023"},
    {"query": "What must a Data Fiduciary do when processing is no longer needed?",
     "gold_answer": "A Data Fiduciary must erase personal data and cause the Data Processor to erase it within the period specified by the Board once the purpose of processing is fulfilled or consent is withdrawn. Retention beyond this point is prohibited.",
     "section_reference": "Section 8(7), DPDP Act 2023"},
]

# ── Compliance gold set ───────────────────────────────────────────────────────
# Manually authored expected verdicts for the compliance checker.
# These are the ground truth for eval_compliance(). No auto-labeling.
# Each entry describes what a policy MUST say and the expected verdict.

COMPLIANCE_GOLD = [
    {
        "rule_subject": "Consent",
        "test_policy_excerpt": "We collect your information with your permission and you can withdraw this permission at any time by contacting us.",
        "expected_verdict": "pass",
        "rationale": "Mentions permission (consent synonym), withdrawal mechanism present.",
    },
    {
        "rule_subject": "Consent",
        "test_policy_excerpt": "By using our services you agree to our privacy policy and data collection practices.",
        "expected_verdict": "fail",
        "rationale": "Consent is bundled with terms of service and implied from use — not a clear affirmative action.",
    },
    {
        "rule_subject": "Notice",
        "test_policy_excerpt": "We collect your name, email, and usage data to provide our services and improve your experience.",
        "expected_verdict": "fail",
        "rationale": "States purpose and data types but does not mention rights of Data Principal or grievance officer contact.",
    },
    {
        "rule_subject": "Notice",
        "test_policy_excerpt": "We collect your name and email to process your order. You have the right to access, correct, and delete your data. Contact our DPO at dpo@company.com for any grievances or to exercise your rights.",
        "expected_verdict": "pass",
        "rationale": "States data collected, purpose, Principal rights, and contact for grievances.",
    },
    {
        "rule_subject": "Data Breach",
        "test_policy_excerpt": "In the event of a security incident we will investigate and take appropriate action to protect your data.",
        "expected_verdict": "fail",
        "rationale": "Does not mention notifying affected users or the Data Protection Board.",
    },
    {
        "rule_subject": "Data Breach",
        "test_policy_excerpt": "If a data breach occurs we will notify the Data Protection Board and all affected users without undue delay describing the nature of the breach and steps taken.",
        "expected_verdict": "pass",
        "rationale": "Explicitly mentions Board notification, user notification, and description of breach.",
    },
    {
        "rule_subject": "Children Data",
        "test_policy_excerpt": "Our services are available to all users. We do not knowingly collect data from users under 13.",
        "expected_verdict": "fail",
        "rationale": "DPDP Act sets age at 18, not 13. No mention of verifiable parental consent.",
    },
    {
        "rule_subject": "Children Data",
        "test_policy_excerpt": "We do not collect personal data from anyone under 18. If you are under 18 your parent or guardian must provide verifiable consent before you use our services.",
        "expected_verdict": "pass",
        "rationale": "Correctly identifies 18 as the age threshold and requires verifiable parental consent.",
    },
    {
        "rule_subject": "Storage Limitation",
        "test_policy_excerpt": "We retain your data for as long as necessary for the purposes outlined in this policy.",
        "expected_verdict": "insufficient_evidence",
        "rationale": "Vague — does not specify what 'necessary' means or provide a concrete retention schedule.",
    },
    {
        "rule_subject": "Storage Limitation",
        "test_policy_excerpt": "We retain your personal data for 3 years after your last interaction with us. After this period all personal data is securely deleted unless required by law.",
        "expected_verdict": "pass",
        "rationale": "States concrete retention period and deletion policy.",
    },
]


# ── Labeling function ─────────────────────────────────────────────────────────

# def find_relevant_chunk_ids(
#     synthetic_answer: str,
#     dpdp_chunk_ids: List[int],
#     all_chunks: List[Dict],
#     index: faiss.Index,
#     model: SentenceTransformer,
#     top_k: int = 3,
#     min_score: float = 0.25,
# ) -> List[int]:
#     """
#     HyDE-style: embed synthetic answer, find matching DPDP chunks.
#     Uses a DIFFERENT (larger) model than query time to reduce circularity.
#     Still requires manual verification — every returned ID is flagged.
#     """
#     dpdp_id_set = set(dpdp_chunk_ids)
#     vec = model.encode(
#         [GOLD_QUERY_PREFIX + synthetic_answer],
#         convert_to_numpy=True, normalize_embeddings=True,
#     )
#     scores, ids = index.search(vec, min(top_k * 6, len(all_chunks)))
#     relevant = []
#     for score, idx in zip(scores[0], ids[0]):
#         if idx < 0:
#             continue
#         if int(idx) in dpdp_id_set and float(score) >= min_score:
#             relevant.append(int(idx))
#         if len(relevant) >= top_k:
#             break
#     return relevant


# def main():
#     if not CHUNKS_PATH.exists() or not FAISS_PATH.exists():
#         raise FileNotFoundError("Run python src/ingestion/ingest.py first.")

#     with open(CHUNKS_PATH, encoding="utf-8") as f:
#         all_chunks = json.load(f)

#     index = faiss.read_index(str(FAISS_PATH))
#     assert index.ntotal == len(all_chunks), "Index desync — re-run ingest.py"

#     dpdp_chunks = [c for c in all_chunks if "DigitalPersonalData" in c.get("filename", "")]
#     dpdp_ids    = [c["id"] for c in dpdp_chunks]
#     log.info("DPDP chunks: %d / %d total", len(dpdp_chunks), len(all_chunks))
#     log.info("Loading gold labeling model: %s (different from query-time model)", GOLD_EMBED_MODEL)

#     model = SentenceTransformer(GOLD_EMBED_MODEL)
#     EVAL_DIR.mkdir(parents=True, exist_ok=True)

#     # Retrieval gold (50 queries)
#     log.info("Building gold_retrieval.json (%d queries)...", len(RETRIEVAL_QUERIES))
#     retrieval_gold = []
#     for item in RETRIEVAL_QUERIES:
#         rel_ids = find_relevant_chunk_ids(
#             item["synthetic_answer"], dpdp_ids, all_chunks, index, model, top_k=3,
#         )
#         entry = {
#             "query":                  item["query"],
#             "relevant_chunk_ids":     rel_ids,
#             "manual_verified":        False,  # SET TO TRUE AFTER HUMAN REVIEW
#             "manual_verification_needed": True,
#         }
#         retrieval_gold.append(entry)
#         log.info("  '%s...' → %s", item["query"][:45], rel_ids)

#     gold_ret_path = EVAL_DIR / "gold_retrieval.json"
#     with open(gold_ret_path, "w", encoding="utf-8") as f:
#         json.dump(retrieval_gold, f, ensure_ascii=False, indent=2)
#     log.info("Saved → %s (%d queries)", gold_ret_path, len(retrieval_gold))

#     # Generation gold (15 pairs)
#     log.info("Building gold_generation.json (%d QA pairs)...", len(GENERATION_GOLD))
#     generation_gold = []
#     for item in GENERATION_GOLD:
#         rel_ids = find_relevant_chunk_ids(
#             item["gold_answer"], dpdp_ids, all_chunks, index, model, top_k=3,
#         )
#         generation_gold.append({
#             "query":             item["query"],
#             "gold_answer":       item["gold_answer"],
#             "gold_chunk_ids":    rel_ids,
#             "section_reference": item["section_reference"],
#             "manual_verified":   False,
#         })

#     gold_gen_path = EVAL_DIR / "gold_generation.json"
#     with open(gold_gen_path, "w", encoding="utf-8") as f:
#         json.dump(generation_gold, f, ensure_ascii=False, indent=2)
#     log.info("Saved → %s (%d pairs)", gold_gen_path, len(generation_gold))

#     # Compliance gold (manually authored, no auto-labeling)
#     gold_comp_path = EVAL_DIR / "gold_compliance.json"
#     with open(gold_comp_path, "w", encoding="utf-8") as f:
#         json.dump(COMPLIANCE_GOLD, f, ensure_ascii=False, indent=2)
#     log.info("Saved → %s (%d entries, manually authored)", gold_comp_path, len(COMPLIANCE_GOLD))

#     print(f"\n✓ Gold sets built:")
#     print(f"  Retrieval gold  : {gold_ret_path}  ({len(retrieval_gold)} queries)")
#     print(f"  Generation gold : {gold_gen_path}  ({len(generation_gold)} QA pairs)")
#     print(f"  Compliance gold : {gold_comp_path}  ({len(COMPLIANCE_GOLD)} cases — manually authored)")
#     print(f"\n⚠ IMPORTANT: Open gold_retrieval.json and gold_generation.json.")
#     print(f"  For each entry, verify the relevant_chunk_ids actually contain the answer.")
#     print(f"  Set manual_verified: true on entries you have checked.")
#     print(f"  The eval harness will warn on unverified entries.")
#     print(f"\nNext: python src/eval/run_eval.py")


# if __name__ == "__main__":
#     main()

# ── Labeling function ─────────────────────────────────────────────────────────

def find_relevant_chunk_ids(
    synthetic_answer: str,
    dpdp_chunk_ids: List[int],
    dpdp_chunks: List[Dict],
    model: SentenceTransformer,
    temp_index: faiss.Index,
    top_k: int = 3,
    min_score: float = 0.25,
) -> List[int]:
    """
    HyDE-style: embed synthetic answer, find matching DPDP chunks.
    Uses a DIFFERENT (larger) model than query time to reduce circularity.
    Searches a temporary index built with the SAME larger model to avoid dim mismatch.
    """
    vec = model.encode(
        [GOLD_QUERY_PREFIX + synthetic_answer],
        convert_to_numpy=True, normalize_embeddings=True,
    )
    scores, ids = temp_index.search(vec, min(top_k * 6, len(dpdp_chunks)))
    relevant = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        # The temp index only contains DPDP chunks, so idx maps directly to dpdp_chunk_ids
        if float(score) >= min_score:
            relevant.append(int(dpdp_chunk_ids[idx]))
        if len(relevant) >= top_k:
            break
    return relevant


def main():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError("Run python src/ingestion/ingest.py first.")

    with open(CHUNKS_PATH, encoding="utf-8") as f:
        all_chunks = json.load(f)

    dpdp_chunks = [c for c in all_chunks if "DigitalPersonalData" in c.get("filename", "")]
    if not dpdp_chunks:
        raise ValueError("No DPDP chunks found. Ensure the DPDP Act PDF is ingested.")
        
    dpdp_ids = [c["id"] for c in dpdp_chunks]
    log.info("DPDP chunks: %d / %d total", len(dpdp_chunks), len(all_chunks))
    
    log.info("Loading gold labeling model: %s (different from query-time model)", GOLD_EMBED_MODEL)
    model = SentenceTransformer(GOLD_EMBED_MODEL)
    
    # Build a temporary FAISS index using the LARGER model (bge-base) just for labeling
    log.info("Building temporary 768-dim index for gold labeling...")
    dpdp_texts = [c["text"] for c in dpdp_chunks]
    dpdp_vecs = model.encode(dpdp_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    temp_index = faiss.IndexFlatIP(dpdp_vecs.shape[1])
    temp_index.add(dpdp_vecs)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Retrieval gold (50 queries)
    log.info("Building gold_retrieval.json (%d queries)...", len(RETRIEVAL_QUERIES))
    retrieval_gold = []
    for item in RETRIEVAL_QUERIES:
        rel_ids = find_relevant_chunk_ids(
            item["synthetic_answer"], dpdp_ids, dpdp_chunks, model, temp_index, top_k=3,
        )
        entry = {
            "query":                  item["query"],
            "relevant_chunk_ids":     rel_ids,
            "manual_verified":        False,
            "manual_verification_needed": True,
        }
        retrieval_gold.append(entry)
        log.info("  '%s...' → %s", item["query"][:45], rel_ids)

    gold_ret_path = EVAL_DIR / "gold_retrieval.json"
    with open(gold_ret_path, "w", encoding="utf-8") as f:
        json.dump(retrieval_gold, f, ensure_ascii=False, indent=2)
    log.info("Saved → %s (%d queries)", gold_ret_path, len(retrieval_gold))

    # Generation gold (15 pairs)
    log.info("Building gold_generation.json (%d QA pairs)...", len(GENERATION_GOLD))
    generation_gold = []
    for item in GENERATION_GOLD:
        rel_ids = find_relevant_chunk_ids(
            item["gold_answer"], dpdp_ids, dpdp_chunks, model, temp_index, top_k=3,
        )
        generation_gold.append({
            "query":             item["query"],
            "gold_answer":       item["gold_answer"],
            "gold_chunk_ids":    rel_ids,
            "section_reference": item["section_reference"],
            "manual_verified":   False,
        })

    gold_gen_path = EVAL_DIR / "gold_generation.json"
    with open(gold_gen_path, "w", encoding="utf-8") as f:
        json.dump(generation_gold, f, ensure_ascii=False, indent=2)
    log.info("Saved → %s (%d pairs)", gold_gen_path, len(generation_gold))

    # Compliance gold (manually authored, no auto-labeling)
    gold_comp_path = EVAL_DIR / "gold_compliance.json"
    with open(gold_comp_path, "w", encoding="utf-8") as f:
        json.dump(COMPLIANCE_GOLD, f, ensure_ascii=False, indent=2)
    log.info("Saved → %s (%d entries, manually authored)", gold_comp_path, len(COMPLIANCE_GOLD))

    print(f"\n✓ Gold sets built:")
    print(f"  Retrieval gold  : {gold_ret_path}  ({len(retrieval_gold)} queries)")
    print(f"  Generation gold : {gold_gen_path}  ({len(generation_gold)} QA pairs)")
    print(f"  Compliance gold : {gold_comp_path}  ({len(COMPLIANCE_GOLD)} cases — manually authored)")
    print(f"\n⚠ IMPORTANT: Open gold_retrieval.json and gold_generation.json.")
    print(f"  For each entry, verify the relevant_chunk_ids actually contain the answer.")
    print(f"  Set manual_verified: true on entries you have checked.")
    print(f"  The eval harness will warn on unverified entries.")
    print(f"\nNext: python src/eval/run_eval.py")

if __name__ == "__main__":
    main()