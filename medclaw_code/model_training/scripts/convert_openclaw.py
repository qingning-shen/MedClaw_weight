#!/usr/bin/env python3
"""
Rule-based distillation of OpenClaw-Medical-Skills SKILL.md files into SFT training data.
No LLM or API calls required — pure string templates + entity lists.

Usage:
    python convert_openclaw.py --skills_dir /path/to/OpenClaw-Medical-Skills/skills \
                               --output_file ../data/distill/medical_tool_calls.jsonl \
                               --num_samples 2000
"""

import os
import re
import json
import random
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Tool schemas for our 7 callable tools (used in the "tools" field of each sample)
# ---------------------------------------------------------------------------
TOOL_SCHEMAS = {
    "pubmed_search": {
        "type": "function",
        "function": {
            "name": "pubmed_search",
            "description": "Search PubMed for biomedical literature by keywords, disease names, or gene names.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query string"},
                    "max_results": {"type": "integer", "description": "Number of results to return (default 5)"}
                },
                "required": ["query"]
            }
        }
    },
    "drug_lookup": {
        "type": "function",
        "function": {
            "name": "drug_lookup",
            "description": "Look up drug information including mechanism, indications, and side effects.",
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_name": {"type": "string", "description": "Name of the drug (generic or brand name)"}
                },
                "required": ["drug_name"]
            }
        }
    },
    "search_clinical_trials": {
        "type": "function",
        "function": {
            "name": "search_clinical_trials",
            "description": "Search ClinicalTrials.gov for clinical trials by condition or drug.",
            "parameters": {
                "type": "object",
                "properties": {
                    "condition": {"type": "string", "description": "Medical condition or disease name"},
                    "drug": {"type": "string", "description": "Drug or intervention name (optional)"},
                    "status": {"type": "string", "description": "Trial status filter: RECRUITING, COMPLETED, etc. (default RECRUITING)"},
                    "max_results": {"type": "integer", "description": "Number of results (default 5)"}
                },
                "required": ["condition"]
            }
        }
    },
    "query_gene_variant": {
        "type": "function",
        "function": {
            "name": "query_gene_variant",
            "description": "Query ClinVar for gene variant pathogenicity and clinical significance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "gene_symbol": {"type": "string", "description": "Gene symbol (e.g. BRCA1, TP53)"},
                    "variant": {"type": "string", "description": "Specific variant notation (optional, e.g. c.5266dupC)"}
                },
                "required": ["gene_symbol"]
            }
        }
    },
    "get_disease_info": {
        "type": "function",
        "function": {
            "name": "get_disease_info",
            "description": "Retrieve disease summary including symptoms, causes, and treatments from MedlinePlus.",
            "parameters": {
                "type": "object",
                "properties": {
                    "disease_name": {"type": "string", "description": "Name of the disease or medical condition"}
                },
                "required": ["disease_name"]
            }
        }
    },
    "check_drug_interaction": {
        "type": "function",
        "function": {
            "name": "check_drug_interaction",
            "description": "Check for known interactions between two drugs using OpenFDA data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "drug1": {"type": "string", "description": "First drug name"},
                    "drug2": {"type": "string", "description": "Second drug name"}
                },
                "required": ["drug1", "drug2"]
            }
        }
    },
    "lookup_gene_info": {
        "type": "function",
        "function": {
            "name": "lookup_gene_info",
            "description": "Look up gene information including function, location, and associated diseases.",
            "parameters": {
                "type": "object",
                "properties": {
                    "gene_symbol": {"type": "string", "description": "Official gene symbol (e.g. EGFR, KRAS)"}
                },
                "required": ["gene_symbol"]
            }
        }
    },
    "medical_calculator": {
        "type": "function",
        "function": {
            "name": "medical_calculator",
            "description": "Perform standard medical calculations: BMI, eGFR, or BSA.",
            "parameters": {
                "type": "object",
                "properties": {
                    "formula": {"type": "string", "description": "Calculation type: 'bmi', 'egfr', or 'bsa'"},
                    "weight_kg": {"type": "number", "description": "Body weight in kilograms"},
                    "height_cm": {"type": "number", "description": "Height in centimeters"},
                    "age_years": {"type": "integer", "description": "Age in years (required for eGFR)"},
                    "is_female": {"type": "boolean", "description": "True if patient is female (required for eGFR)"},
                    "creatinine_mg_dl": {"type": "number", "description": "Serum creatinine in mg/dL (required for eGFR)"}
                },
                "required": ["formula"]
            }
        }
    }
}

# ---------------------------------------------------------------------------
# Keyword → tool mapping (ordered by specificity, first match wins)
# ---------------------------------------------------------------------------
KEYWORD_TOOL_MAP = [
    (["clinical trial", "nct", "clinicaltrials", "recruiting trial"], "search_clinical_trials"),
    (["drug interaction", "drug-drug", "contraindication", "concurrent medication"], "check_drug_interaction"),
    (["variant", "snp", "mutation", "polymorphism", "pathogenic", "clinvar", "allele"], "query_gene_variant"),
    (["bmi", "body mass", "gfr", "creatinine", "egfr", "renal function", "bsa", "body surface"], "medical_calculator"),
    (["drug", "compound", "molecule", "medication", "pharmaceutical", "chembl", "drugbank", "inhibitor", "agonist"], "drug_lookup"),
    (["pubmed", "paper", "literature", "journal", "study", "article", "publication", "research finding"], "pubmed_search"),
    (["gene", "expression", "rna", "transcription", "geo ", "mrna", "protein function", "gene function"], "lookup_gene_info"),
    (["disease", "condition", "syndrome", "disorder", "diagnosis", "symptom", "treatment"], "get_disease_info"),
]

# ---------------------------------------------------------------------------
# Entity lists for filling query templates
# ---------------------------------------------------------------------------
DISEASES = [
    "breast cancer", "lung cancer", "colorectal cancer", "prostate cancer", "leukemia",
    "type 2 diabetes", "type 1 diabetes", "hypertension", "heart failure", "atrial fibrillation",
    "Alzheimer's disease", "Parkinson's disease", "multiple sclerosis", "epilepsy",
    "COVID-19", "influenza", "tuberculosis", "HIV/AIDS", "hepatitis B", "hepatitis C",
    "rheumatoid arthritis", "systemic lupus erythematosus", "Crohn's disease",
    "asthma", "COPD", "pulmonary fibrosis",
    "glioblastoma", "melanoma", "ovarian cancer", "pancreatic cancer",
    "chronic kidney disease", "acute kidney injury",
]

DRUGS = [
    "metformin", "aspirin", "ibuprofen", "atorvastatin", "lisinopril",
    "warfarin", "clopidogrel", "amoxicillin", "azithromycin", "ciprofloxacin",
    "omeprazole", "metoprolol", "amlodipine", "losartan", "hydrochlorothiazide",
    "levothyroxine", "prednisone", "dexamethasone", "morphine", "tramadol",
    "gefitinib", "erlotinib", "imatinib", "bevacizumab", "trastuzumab",
    "pembrolizumab", "nivolumab", "ipilimumab",
    "insulin glargine", "sitagliptin", "empagliflozin",
    "semaglutide", "liraglutide",
]

GENES = [
    "BRCA1", "BRCA2", "TP53", "KRAS", "EGFR", "ALK", "RET", "BRAF", "PIK3CA",
    "PTEN", "ATM", "CHEK2", "PALB2", "RAD51C", "RAD51D",
    "HER2", "ERBB2", "VEGFA", "MYC", "CDKN2A", "RB1", "APC", "MLH1", "MSH2",
    "MTHFR", "APOE", "COMT", "CYP2D6", "CYP2C19", "CYP3A4",
    "TNF", "IL6", "IL1B", "IFNG", "ACE", "AGT",
]

DRUG_PAIRS = [
    ("warfarin", "aspirin"), ("warfarin", "ibuprofen"), ("metformin", "contrast dye"),
    ("clopidogrel", "omeprazole"), ("statins", "gemfibrozil"), ("digoxin", "amiodarone"),
    ("SSRIs", "tramadol"), ("methotrexate", "NSAIDs"), ("tacrolimus", "fluconazole"),
    ("lithium", "ibuprofen"), ("MAOIs", "serotonin reuptake inhibitors"),
]

CALCULATOR_EXAMPLES = [
    {"formula": "bmi", "weight_kg": 75, "height_cm": 170},
    {"formula": "bmi", "weight_kg": 90, "height_cm": 165},
    {"formula": "bmi", "weight_kg": 55, "height_cm": 158},
    {"formula": "egfr", "creatinine_mg_dl": 1.2, "age_years": 60, "is_female": False},
    {"formula": "egfr", "creatinine_mg_dl": 0.9, "age_years": 45, "is_female": True},
    {"formula": "egfr", "creatinine_mg_dl": 2.5, "age_years": 70, "is_female": False},
    {"formula": "bsa", "weight_kg": 70, "height_cm": 175},
    {"formula": "bsa", "weight_kg": 60, "height_cm": 162},
]

# ---------------------------------------------------------------------------
# Query templates per tool
# ---------------------------------------------------------------------------
QUERY_TEMPLATES = {
    "pubmed_search": [
        "Search for recent papers on {disease}.",
        "Find literature about the treatment of {disease}.",
        "What are the latest studies on {gene} in cancer?",
        "Search PubMed for research about {drug} efficacy.",
        "Find published studies on {drug} and {disease}.",
        "What does the literature say about {gene} mutations?",
        "Search for papers about {disease} and {gene} expression.",
        "Find recent research on {drug} side effects.",
        "What are recent findings about targeted therapy in {disease}?",
        "Look up studies on {gene} as a biomarker for {disease}.",
    ],
    "drug_lookup": [
        "What is {drug}? Tell me about its mechanism of action.",
        "What are the indications and side effects of {drug}?",
        "Look up information about {drug}.",
        "What is the mechanism of {drug} in treating {disease}?",
        "Tell me about {drug} — how does it work?",
        "What drug class does {drug} belong to?",
        "Find information on {drug} pharmacology.",
        "What are the contraindications of {drug}?",
    ],
    "search_clinical_trials": [
        "Are there any clinical trials for {disease}?",
        "Find recruiting clinical trials for {disease}.",
        "Search for trials testing {drug} in {disease}.",
        "What clinical trials are currently open for {disease} patients?",
        "Find ongoing trials for {drug} in {disease} treatment.",
        "Are there any Phase 3 trials for {disease}?",
        "Search ClinicalTrials.gov for {disease} studies.",
        "What trials are available for patients with {disease}?",
    ],
    "query_gene_variant": [
        "What variants of {gene} are associated with disease?",
        "Look up pathogenic variants in {gene}.",
        "What is the clinical significance of mutations in {gene}?",
        "Find ClinVar entries for {gene}.",
        "Are there known pathogenic variants in {gene} linked to cancer?",
        "What is known about {gene} variants in ClinVar?",
        "Look up disease-associated mutations in {gene}.",
    ],
    "get_disease_info": [
        "What is {disease}? Describe its symptoms and treatment.",
        "Tell me about {disease} — causes, symptoms, and management.",
        "Get information about {disease}.",
        "What are the diagnostic criteria for {disease}?",
        "Describe the pathophysiology and treatment of {disease}.",
        "What are the symptoms of {disease}?",
        "What causes {disease} and how is it treated?",
        "Give me a summary of {disease} for a clinical overview.",
    ],
    "check_drug_interaction": [
        "Is there an interaction between {drug1} and {drug2}?",
        "Check if {drug1} interacts with {drug2}.",
        "What happens if a patient takes {drug1} and {drug2} together?",
        "Are {drug1} and {drug2} safe to use concurrently?",
        "Look up the interaction between {drug1} and {drug2}.",
        "Does {drug1} interact with {drug2}?",
        "What are the risks of co-administering {drug1} and {drug2}?",
    ],
    "lookup_gene_info": [
        "What does the {gene} gene do?",
        "Tell me about the function of {gene}.",
        "What is the role of {gene} in human biology?",
        "Look up information about the {gene} gene.",
        "What diseases are associated with {gene}?",
        "What is {gene} and where is it located in the genome?",
        "Describe the function and clinical relevance of {gene}.",
    ],
    "medical_calculator": [
        "Calculate the BMI for a patient weighing {weight_kg} kg and {height_cm} cm tall.",
        "What is the BMI of someone who is {height_cm} cm tall and weighs {weight_kg} kg?",
        "Estimate eGFR for a {age_years}-year-old {'female' if is_female else 'male'} with creatinine {creatinine_mg_dl} mg/dL.",
        "Calculate eGFR: age {age_years}, creatinine {creatinine_mg_dl} mg/dL, {'female' if is_female else 'male'}.",
        "What is the body surface area for a patient weighing {weight_kg} kg and {height_cm} cm tall?",
    ],
}

# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_skill_md(filepath: Path) -> dict:
    """Extract skill name, description, and category keywords from a SKILL.md file."""
    text = filepath.read_text(encoding="utf-8", errors="ignore").lower()
    name = filepath.stem.replace("-", " ").replace("_", " ")

    # Try to extract title from H1 heading
    h1_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    if h1_match:
        name = h1_match.group(1).strip()

    # Extract first substantive paragraph as description
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip() and not p.strip().startswith("#")]
    description = paragraphs[0][:200] if paragraphs else ""

    return {"name": name, "description": description, "raw": text, "path": str(filepath)}


def classify_skill(skill: dict) -> str | None:
    """Map a skill to one of our 7 tools based on keyword matching. Returns tool name or None."""
    combined = skill["name"] + " " + skill["description"] + " " + skill["raw"][:500]
    for keywords, tool in KEYWORD_TOOL_MAP:
        if any(kw in combined for kw in keywords):
            return tool
    return None


# ---------------------------------------------------------------------------
# Sample generators
# ---------------------------------------------------------------------------

def _tool_subset(primary_tool: str, n_distractor: int = 2) -> list[dict]:
    """Return the primary tool schema plus n random distractor schemas."""
    others = [s for k, s in TOOL_SCHEMAS.items() if k != primary_tool]
    selected = random.sample(others, min(n_distractor, len(others)))
    return [TOOL_SCHEMAS[primary_tool]] + selected


def _make_sample(tool: str, query: str, arguments: dict) -> dict:
    """Build one ms-swift sharegpt JSONL record."""
    tools = _tool_subset(tool)
    return {
        "tools": json.dumps(tools),
        "conversations": [
            {"role": "user", "content": query},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": tool,
                            "arguments": json.dumps(arguments, ensure_ascii=False)
                        }
                    }
                ]
            }
        ]
    }


def generate_samples_for_tool(tool: str, skill_count: int, samples_per_skill: int) -> list[dict]:
    """Generate training samples for a given tool using templates."""
    samples = []
    templates = QUERY_TEMPLATES[tool]

    for _ in range(skill_count):
        for _ in range(samples_per_skill):
            tmpl = random.choice(templates)

            if tool == "pubmed_search":
                entity = random.choice(DISEASES + GENES + DRUGS)
                args = {"query": entity, "max_results": random.choice([3, 5, 10])}
                # Fill template with available slots
                query = tmpl.format(disease=random.choice(DISEASES), gene=random.choice(GENES),
                                    drug=random.choice(DRUGS)).split("{")[0]
                query = tmpl
                d, g, dr = random.choice(DISEASES), random.choice(GENES), random.choice(DRUGS)
                query = tmpl.replace("{disease}", d).replace("{gene}", g).replace("{drug}", dr)
                args = {"query": entity, "max_results": 5}

            elif tool == "drug_lookup":
                drug = random.choice(DRUGS)
                query = tmpl.replace("{drug}", drug).replace("{disease}", random.choice(DISEASES))
                args = {"drug_name": drug}

            elif tool == "search_clinical_trials":
                disease = random.choice(DISEASES)
                drug = random.choice(DRUGS)
                query = tmpl.replace("{disease}", disease).replace("{drug}", drug)
                args = {"condition": disease}
                if random.random() > 0.5:
                    args["drug"] = drug
                args["max_results"] = 5

            elif tool == "query_gene_variant":
                gene = random.choice(GENES)
                query = tmpl.replace("{gene}", gene)
                args = {"gene_symbol": gene}

            elif tool == "get_disease_info":
                disease = random.choice(DISEASES)
                query = tmpl.replace("{disease}", disease)
                args = {"disease_name": disease}

            elif tool == "check_drug_interaction":
                d1, d2 = random.choice(DRUG_PAIRS) if random.random() > 0.3 else random.sample(DRUGS, 2)
                query = tmpl.replace("{drug1}", d1).replace("{drug2}", d2)
                args = {"drug1": d1, "drug2": d2}

            elif tool == "lookup_gene_info":
                gene = random.choice(GENES)
                query = tmpl.replace("{gene}", gene)
                args = {"gene_symbol": gene}

            elif tool == "medical_calculator":
                ex = random.choice(CALCULATOR_EXAMPLES)
                if ex["formula"] == "bmi":
                    query = (f"Calculate the BMI for a patient weighing {ex['weight_kg']} kg "
                             f"and {ex['height_cm']} cm tall.")
                elif ex["formula"] == "egfr":
                    sex = "female" if ex.get("is_female") else "male"
                    query = (f"Estimate eGFR for a {ex['age_years']}-year-old {sex} "
                             f"with creatinine {ex['creatinine_mg_dl']} mg/dL.")
                else:
                    query = (f"Calculate body surface area for {ex['weight_kg']} kg, {ex['height_cm']} cm.")
                args = {k: v for k, v in ex.items()}

            else:
                continue

            samples.append(_make_sample(tool, query, args))

    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Distill OpenClaw skills to SFT JSONL.")
    parser.add_argument("--skills_dir", required=True, help="Path to OpenClaw-Medical-Skills/skills directory")
    parser.add_argument("--output_file", default="../data/distill/medical_tool_calls.jsonl")
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    skills_path = Path(args.skills_dir)
    if not skills_path.exists():
        raise FileNotFoundError(f"Skills directory not found: {skills_path}")

    md_files = list(skills_path.rglob("*.md"))
    print(f"Found {len(md_files)} SKILL.md files.")

    # Parse and classify all skills
    tool_buckets: dict[str, int] = {t: 0 for t in TOOL_SCHEMAS}
    for f in md_files:
        skill = parse_skill_md(f)
        tool = classify_skill(skill)
        if tool:
            tool_buckets[tool] += 1

    total_classified = sum(tool_buckets.values())
    print(f"Classified {total_classified} skills across {len(tool_buckets)} tools: {tool_buckets}")

    # Generate samples proportional to skill count, capped at num_samples total
    all_samples = []
    for tool, count in tool_buckets.items():
        if count == 0:
            count = 5  # ensure each tool gets at least some coverage
        fraction = count / max(total_classified, 1)
        budget = max(10, int(args.num_samples * fraction))
        samples_per_skill = max(1, budget // max(count, 1))
        new_samples = generate_samples_for_tool(tool, count, samples_per_skill)
        all_samples.extend(new_samples)

    # Shuffle and trim to target
    random.shuffle(all_samples)
    all_samples = all_samples[:args.num_samples]

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_samples)} samples to {out_path}")

    # Tool distribution in output
    dist: dict[str, int] = {}
    for s in all_samples:
        t = s["conversations"][1]["tool_calls"][0]["function"]["name"]
        dist[t] = dist.get(t, 0) + 1
    print("Output distribution:", dist)


if __name__ == "__main__":
    main()
