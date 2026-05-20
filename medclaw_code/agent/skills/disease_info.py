"""Disease summary from MedlinePlus health topics search API (free, no API key required)."""

import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from smolagents import tool

_BASE = "https://wsearch.nlm.nih.gov/ws/query"


@tool
def get_disease_info(disease_name: str) -> str:
    """
    Retrieve a disease or medical condition summary from MedlinePlus.

    Args:
        disease_name: Name of the disease or medical condition (e.g. "hypertension", "type 2 diabetes").

    Returns:
        Disease summary including description, also-called names, and a MedlinePlus URL.
    """
    params = {
        "db": "healthTopics",
        "term": disease_name,
        "retmax": "3",
    }
    url = _BASE + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "MedClaw/1.0"})

    with urllib.request.urlopen(req, timeout=15) as r:
        raw = r.read().decode("utf-8", errors="ignore")

    # MedlinePlus returns XML
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return f"Could not retrieve information for '{disease_name}'."

    # Namespace used by MedlinePlus web service
    ns = {"nlm": "https://medlineplus.gov/"}

    docs = root.findall(".//document")
    if not docs:
        return f"No MedlinePlus health topic found for: {disease_name}"

    lines = [f"MedlinePlus information for '{disease_name}':\n"]
    for doc in docs[:2]:
        # Title
        title_el = doc.find(".//content[@name='title']")
        title = title_el.text.strip() if title_el is not None and title_el.text else "N/A"

        # Full summary snippet
        snippet_el = doc.find(".//content[@name='FullSummary']")
        if snippet_el is None:
            snippet_el = doc.find(".//content[@name='snippet']")
        snippet = ""
        if snippet_el is not None:
            # Strip embedded HTML tags
            inner = ET.tostring(snippet_el, encoding="unicode", method="text")
            snippet = " ".join(inner.split())[:400]

        # Also-called names
        ac_el = doc.find(".//content[@name='altName']")
        also_called = ac_el.text.strip() if ac_el is not None and ac_el.text else None

        # URL
        url_el = doc.find(".//content[@name='organizationURL']")
        topic_url = url_el.text.strip() if url_el is not None and url_el.text else ""

        lines.append(f"Topic: {title}")
        if also_called:
            lines.append(f"Also called: {also_called}")
        if snippet:
            lines.append(f"Summary: {snippet}...")
        if topic_url:
            lines.append(f"More info: {topic_url}")
        lines.append("")

    return "\n".join(lines).strip()
