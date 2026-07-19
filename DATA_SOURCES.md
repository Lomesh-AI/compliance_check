# Data Sources for PolicyRAG v2

This document lists every data source to expand the corpus, with exact download
instructions. All sources are free, public, and legally usable for research.

---

## Tier 1 — Core Regulatory Text (add immediately)

### 1. DPDP Rules 2025 (official companion to the Act)
The Rules were notified November 13 2025 and provide operational detail the Act
delegates to subordinate legislation: consent collection procedures, breach
notification timelines, DPIA requirements, cross-border transfer conditions.

- **URL**: https://www.meity.gov.in/dpdp-rules-2025
- **Alternate**: https://www.dpdpa.com/ → "DPDP Rules 2025 PDF"
- **Add to**: `data/docs/DPDPRules2025.pdf`
- **Why**: Doubles the regulatory text. Many company obligations are in the Rules
  not the Act. Without it, the system can only check Act-level obligations.

### 2. IT Act 2000 + IT (Amendment) Act 2008 — Sections 43A and 72A
These sections on reasonable security practices predated DPDP and still apply
concurrently in some contexts.

- **URL**: https://indiacode.nic.in/handle/123456789/1999
- **Add to**: `data/docs/ITAct2000_Sections43A_72A.pdf`
- **Why**: A business policy may satisfy DPDP but violate IT Act. Cross-reference
  coverage matters.

### 3. MeitY Draft Data Governance Framework (2022)
Non-binding but heavily cited in compliance discussions.

- **URL**: https://www.meity.gov.in/data-governance-framework
- **Add to**: `data/docs/MeitY_DataGovernanceFramework.pdf`

---

## Tier 2 — Comparable International Regulations (cross-jurisdiction context)

### 4. GDPR Full Text
EU General Data Protection Regulation. The DPDP Act was modelled partly on GDPR.
Many DPDP concepts (consent, purpose limitation, DPO) map to GDPR counterparts.
Adding GDPR text lets the system explain comparisons and gaps.

- **URL**: https://gdpr-info.eu/ → download full PDF
- **Alternate**: https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679
- **Add to**: `data/docs/GDPR_2016.pdf`

### 5. CCPA + CPRA Full Text
California Consumer Privacy Act. Many Indian tech companies serving US users
must comply with both DPDP and CCPA.

- **URL**: https://oag.ca.gov/privacy/ccpa
- **Add to**: `data/docs/CCPA_CPRA.pdf`

### 6. Singapore PDPA 2012 (amended 2020)
Most similar structure to DPDP in Asia. Useful for comparative analysis.

- **URL**: https://sso.agc.gov.sg/Act/PDPA2012
- **Add to**: `data/docs/Singapore_PDPA.pdf`

---

## Tier 3 — Business Policy Corpus (compliance targets)

### 7. Indian Bank Privacy Policies (manual collection)
Collect PDF privacy policies from major Indian banks and NBFCs. These are the
primary compliance targets. Sources:
- SBI: https://www.sbi.co.in/web/personal-banking/privacy-policy
- HDFC: already in `data/docs/`
- ICICI: https://www.icicibank.com/privacy-policy
- Axis: https://www.axisbank.com/privacy-policy
- Kotak: https://www.kotak.com/privacy-policy
- Bajaj Finserv: https://www.bajajfinserv.in/privacy-policy

**Add to**: `data/docs/business/` (create subdirectory)

### 8. Indian Tech Company Privacy Policies
- Zomato: https://www.zomato.com/privacy
- Paytm: https://paytm.com/company/privacy-policy
- Flipkart: https://www.flipkart.com/pages/privacypolicy
- Ola: https://www.olacabs.com/privacy-policy
- Swiggy: https://www.swiggy.com/privacy-policy

### 9. Indian Government Services (DigiLocker, UMANG, CoWIN)
These process massive volumes of personal data. Their policies vs DPDP is a
high-interest compliance question.

- **DigiLocker**: https://digilocker.gov.in/privacy.html
- **UMANG**: https://web.umang.gov.in/privacy.html

---

## Tier 4 — Research Datasets (for training/eval enrichment)

### 10. C3PA Dataset (CCPA-annotated privacy policies)
48,947 expert-labeled segments from 411 organizations' privacy policies
annotated for CCPA compliance. Directly usable as silver labels.

- **URL**: https://arxiv.org/abs/2410.03925
- **GitHub**: search "C3PA dataset privacy"
- **Use for**: Weak supervision — map CCPA labels to DPDP equivalents

### 11. OPP-115 Corpus (115 annotated privacy policies)
From Carnegie Mellon's Usable Privacy Policy Project. 10 data practice categories.
Already maps partially to GDPR (JURIX 2020 paper).

- **URL**: https://usableprivacy.org/data
- **Download**: https://github.com/citp/privacy-policy-corpus
- **Use for**: Baseline retrieval evaluation only. DO NOT mix with DPDP index
  (we fixed this bug already). Use as a separate test corpus.

### 12. PolicyQA Dataset (reading comprehension for privacy policies)
2,696 QA pairs from 115 privacy policies. Good for eval set augmentation.

- **URL**: https://arxiv.org/abs/2010.02557
- **GitHub**: https://github.com/wasiahmad/PolicyQA
- **Use for**: Additional QA pairs to augment `gold_generation.json`

---

## Download script

Run this to fetch the primary regulatory PDFs automatically:

```bash
python scripts/download_data.py
```

See `scripts/download_data.py` for the implementation.

---

## Index strategy after adding new PDFs

After adding any new PDFs to `data/docs/`:

```bash
# Re-run ingestion — it reads all PDFs in data/docs/
python src/ingestion/ingest.py

# Rebuild gold sets (chunk IDs change after re-ingestion)
python src/eval/build_gold.py
```

The FAISS/chunks sync check will catch any inconsistency automatically.
