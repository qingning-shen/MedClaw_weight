"""PubMed literature search via NCBI E-utils (free, no API key required)."""

import os
import time
import urllib.parse
import urllib.request
import json
from smolagents import tool

_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_EMAIL = os.getenv("NCBI_EMAIL", "medclaw@example.com")  # set NCBI_EMAIL in environment


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "MedClaw/1.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode())


@tool
def pubmed_search(query: str, max_results: int = 5) -> str:
    """
    Search PubMed for biomedical literature by keywords, disease names, or gene names.

    Args:
        query: Search query string (e.g. "BRCA1 breast cancer treatment").
        max_results: Number of results to return (default 5, max 20).

    Returns:
        Formatted list of papers with title, authors, journal, and publication year.
    """
    max_results = min(max(1, max_results), 20)

    # Step 1: search for PMIDs
    search_url = (
        f"{_BASE}/esearch.fcgi?db=pubmed"
        f"&term={urllib.parse.quote(query)}"
        f"&retmax={max_results}&retmode=json&sort=relevance"
        f"&email={urllib.parse.quote(_EMAIL)}"
    )
    search_data = _get(search_url)
    ids = search_data.get("esearchresult", {}).get("idlist", [])

    if not ids:
        return f"No PubMed results found for: {query}"

    time.sleep(0.4)  # respect NCBI rate limit (3 req/s without API key)

    # Step 2: fetch summaries
    id_str = ",".join(ids)
    summary_url = (
        f"{_BASE}/esummary.fcgi?db=pubmed&id={id_str}&retmode=json"
        f"&email={urllib.parse.quote(_EMAIL)}"
    )
    summary_data = _get(summary_url)
    result_map = summary_data.get("result", {})

    lines = [f"PubMed results for '{query}':\n"]
    for pmid in ids:
        art = result_map.get(pmid, {})
        title = art.get("title", "No title")
        authors = art.get("authors", [])
        author_str = ", ".join(a.get("name", "") for a in authors[:3])
        if len(authors) > 3:
            author_str += " et al."
        source = art.get("source", "")
        pub_date = art.get("pubdate", "")
        lines.append(f"- PMID {pmid}: {title}\n  {author_str} | {source} ({pub_date})")

    return "\n".join(lines)
