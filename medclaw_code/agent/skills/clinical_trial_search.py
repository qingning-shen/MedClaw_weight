"""Clinical trial search via ClinicalTrials.gov API v2 (free, no API key required)."""

import json
import urllib.parse
import urllib.request
from smolagents import tool

_BASE = "https://clinicaltrials.gov/api/v2/studies"


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "MedClaw/1.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode())


@tool
def search_clinical_trials(condition: str, drug: str = None, status: str = "RECRUITING", max_results: int = 5) -> str:
    """
    Search ClinicalTrials.gov for clinical studies by condition or drug.

    Args:
        condition: Medical condition or disease name (e.g. "breast cancer", "type 2 diabetes").
        drug: Drug or intervention name to filter by (optional).
        status: Trial status filter — RECRUITING, COMPLETED, or ALL (default RECRUITING).
        max_results: Number of results to return (default 5, max 10).

    Returns:
        List of matching clinical trials with NCT ID, title, phase, and status.
    """
    max_results = min(max(1, max_results), 10)

    params = {
        "query.cond": condition,
        "pageSize": max_results,
        "format": "json",
        "fields": "NCTId,BriefTitle,Phase,OverallStatus,Condition,InterventionName",
    }
    if drug:
        params["query.intr"] = drug
    if status.upper() != "ALL":
        params["filter.overallStatus"] = status.upper()

    url = _BASE + "?" + urllib.parse.urlencode(params)
    data = _get(url)

    studies = data.get("studies", [])
    if not studies:
        return f"No {status.lower()} clinical trials found for: {condition}" + (f" / {drug}" if drug else "")

    lines = [f"Clinical trials for '{condition}'" + (f" + '{drug}'" if drug else "") + f" ({status}):\n"]
    for study in studies:
        proto = study.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        status_mod = proto.get("statusModule", {})
        design = proto.get("designModule", {})

        nct_id = ident.get("nctId", "N/A")
        title = ident.get("briefTitle", "No title")
        phase = design.get("phases", ["N/A"])
        phase_str = ", ".join(phase) if phase else "N/A"
        trial_status = status_mod.get("overallStatus", "N/A")

        lines.append(f"- {nct_id}: {title}\n  Phase: {phase_str} | Status: {trial_status}")

    return "\n".join(lines)
