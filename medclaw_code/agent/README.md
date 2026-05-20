# agent

smolagents-based agent that orchestrates biomedical tool calls using the fine-tuned MedClaw model.

## Requirements

```bash
pip install -r requirements.txt
```

## Skills (tools available to the agent)

| Skill | API | Description |
|---|---|---|
| `pubmed_search` | NCBI E-utils | Search PubMed literature |
| `drug_lookup` | ChEMBL | Drug mechanism, type, and clinical phase |
| `search_clinical_trials` | ClinicalTrials.gov v2 | Find trials by condition/drug |
| `query_gene_variant` | NCBI ClinVar | Pathogenic variant lookup |
| `get_disease_info` | MedlinePlus | Disease summary and symptoms |
| `check_drug_interaction` | OpenFDA | Drug-drug interaction warnings |
| `lookup_gene_info` | NCBI Gene | Gene function and location |
| `medical_calculator` | Local | BMI, eGFR (CKD-EPI 2021), BSA |

All APIs are free and require no API keys.  
Set `NCBI_EMAIL=your@email.com` in your environment to comply with NCBI usage guidelines.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_SERVER_URL` | `http://localhost:8000/v1` | vLLM server base URL |
| `MODEL_NAME` | `medclaw` | Model name passed to the API |
| `AGENT_MAX_STEPS` | `5` | Maximum reasoning steps before stopping |
| `MEDCLAW_TEST_MODE` | `0` | Set to `1` to return mock responses (no vLLM needed) |
| `NCBI_EMAIL` | `medclaw@example.com` | Email for NCBI API usage |

## Testing skills locally

Each skill can be imported and tested independently without the agent or vLLM:

```python
from skills.pubmed_search import pubmed_search
print(pubmed_search("BRCA1 breast cancer"))

from skills.medical_calculator import medical_calculator
print(medical_calculator("bmi", weight_kg=70, height_cm=175))
print(medical_calculator("egfr", creatinine_mg_dl=1.2, age_years=60, is_female=False))
```

## Testing the full agent (test mode)

```bash
MEDCLAW_TEST_MODE=1 python -c "
from agent import run_agent
result = run_agent('Find papers about BRCA1')
print(result)
"
```
