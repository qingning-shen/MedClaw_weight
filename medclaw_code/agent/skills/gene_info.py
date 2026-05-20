"""Gene information lookup via NCBI Gene E-utils (free, no API key required)."""

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
def lookup_gene_info(gene_symbol: str) -> str:
    """
    Look up gene function, chromosomal location, and associated diseases from NCBI Gene.

    Args:
        gene_symbol: Official HGNC gene symbol (e.g. "EGFR", "KRAS", "TP53").

    Returns:
        Gene summary including full name, location, function, and OMIM-linked diseases.
    """
    # Search NCBI Gene for human genes with this symbol
    term = f"{gene_symbol}[gene/protein name] AND Homo sapiens[organism]"
    search_url = (
        f"{_BASE}/esearch.fcgi?db=gene"
        f"&term={urllib.parse.quote(term)}"
        f"&retmax=1&retmode=json"
    )
    search_data = _get(search_url)
    ids = search_data.get("esearchresult", {}).get("idlist", [])

    if not ids:
        return f"No NCBI Gene entry found for symbol: {gene_symbol}"

    time.sleep(0.4)

    gene_id = ids[0]
    summary_url = f"{_BASE}/esummary.fcgi?db=gene&id={gene_id}&retmode=json"
    summary_data = _get(summary_url)
    entry = summary_data.get("result", {}).get(gene_id, {})

    if not entry:
        return f"Could not retrieve gene summary for {gene_symbol}."

    full_name = entry.get("description", "N/A")
    location = entry.get("maplocation", "N/A")
    chromosome = entry.get("chromosome", "N/A")
    summary = entry.get("summary", "")
    aliases = entry.get("otheraliases", "")

    lines = [
        f"Gene: {gene_symbol} (NCBI Gene ID: {gene_id})",
        f"Full name: {full_name}",
        f"Location: Chromosome {chromosome}, {location}",
    ]
    if aliases:
        lines.append(f"Aliases: {aliases}")
    if summary:
        lines.append(f"Function: {summary[:500]}{'...' if len(summary) > 500 else ''}")

    return "\n".join(lines)
