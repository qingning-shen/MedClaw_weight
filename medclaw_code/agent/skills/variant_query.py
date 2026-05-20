"""Gene variant pathogenicity lookup via NCBI ClinVar E-utils (free, no API key required)."""

import json
import time
import urllib.parse
import urllib.request
from smolagents import tool

_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "MedClaw/1.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode())


@tool
def query_gene_variant(gene_symbol: str, variant: str = None) -> str:
    """
    Query ClinVar for known pathogenic variants in a gene and their clinical significance.

    Args:
        gene_symbol: Official gene symbol (e.g. "BRCA1", "TP53", "EGFR").
        variant: Specific variant notation to narrow the search (optional, e.g. "c.5266dupC").

    Returns:
        List of ClinVar entries with variant name, clinical significance, and review status.
    """
    # Build search term: gene[gene] AND pathogenic[clinical_significance]
    term = f"{gene_symbol}[gene]"
    if variant:
        term += f" AND {variant}"
    else:
        term += " AND (pathogenic[clinical_significance] OR likely_pathogenic[clinical_significance])"

    search_url = (
        f"{_BASE}/esearch.fcgi?db=clinvar"
        f"&term={urllib.parse.quote(term)}"
        f"&retmax=5&retmode=json&sort=relevance"
    )
    search_data = _get(search_url)
    ids = search_data.get("esearchresult", {}).get("idlist", [])

    if not ids:
        return f"No pathogenic ClinVar variants found for gene: {gene_symbol}" + (f" / {variant}" if variant else "")

    time.sleep(0.4)

    id_str = ",".join(ids)
    summary_url = f"{_BASE}/esummary.fcgi?db=clinvar&id={id_str}&retmode=json"
    summary_data = _get(summary_url)
    result = summary_data.get("result", {})

    lines = [f"ClinVar variants for {gene_symbol}" + (f" ({variant})" if variant else "") + ":\n"]
    for uid in ids:
        entry = result.get(uid, {})
        title = entry.get("title", "N/A")
        sig = entry.get("clinical_significance", {})
        significance = sig.get("description", "N/A") if isinstance(sig, dict) else str(sig)
        review = entry.get("review_status", "N/A")
        lines.append(f"- {title}\n  Significance: {significance} | Review: {review}")

    return "\n".join(lines)
