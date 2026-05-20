"""Drug interaction check via OpenFDA drug label API (free, no API key required)."""

import json
import urllib.parse
import urllib.request
from smolagents import tool

_BASE = "https://api.fda.gov/drug/label.json"


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "MedClaw/1.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode())


@tool
def check_drug_interaction(drug1: str, drug2: str) -> str:
    """
    Check for known drug-drug interactions using FDA drug label data.

    Args:
        drug1: Name of the first drug (generic name preferred, e.g. "warfarin").
        drug2: Name of the second drug (generic name preferred, e.g. "aspirin").

    Returns:
        Interaction warnings extracted from FDA drug labels, or a note if none found.
    """
    def search_label(primary: str, secondary: str) -> list[str]:
        """Search primary drug's label for mentions of secondary drug in interactions section."""
        query = f'openfda.generic_name:"{primary}" AND drug_interactions:"{secondary}"'
        url = f"{_BASE}?search={urllib.parse.quote(query)}&limit=2"
        try:
            data = _get(url)
        except Exception:
            return []

        results = data.get("results", [])
        snippets = []
        for r in results:
            interactions = r.get("drug_interactions", [])
            for section in interactions:
                text_lower = section.lower()
                # Find the sentence(s) mentioning the secondary drug
                for sentence in section.split("."):
                    if secondary.lower() in sentence.lower():
                        s = sentence.strip()
                        if s and len(s) > 20:
                            snippets.append(s + ".")
                            break
                if snippets:
                    break
        return snippets[:2]

    # Try both directions since the label might be on either drug
    snippets = search_label(drug1, drug2)
    if not snippets:
        snippets = search_label(drug2, drug1)

    if not snippets:
        return (
            f"No explicit interaction warning found in FDA labels for {drug1} + {drug2}.\n"
            "This does not confirm safety — always consult a clinical pharmacist or drug interaction database."
        )

    lines = [f"Interaction warning: {drug1} + {drug2}\n"]
    lines.extend(f"- {s}" for s in snippets)
    lines.append("\nSource: FDA drug label data (OpenFDA)")
    return "\n".join(lines)
