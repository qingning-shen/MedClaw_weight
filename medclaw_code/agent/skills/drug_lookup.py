"""Drug information lookup via ChEMBL REST API (free, no API key required)."""

import json
import urllib.parse
import urllib.request
from smolagents import tool

_BASE = "https://www.ebi.ac.uk/chembl/api/data"


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "MedClaw/1.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode())


@tool
def drug_lookup(drug_name: str) -> str:
    """
    Look up drug information including mechanism of action, indications, and drug type.

    Args:
        drug_name: Generic or brand name of the drug (e.g. "metformin", "aspirin").

    Returns:
        Drug summary including ChEMBL ID, drug type, mechanism, and max clinical phase.
    """
    encoded = urllib.parse.quote(drug_name.strip())

    # Try exact preferred name match first, then fuzzy search
    url = f"{_BASE}/molecule?pref_name__iexact={encoded}&format=json&limit=1"
    data = _get(url)

    molecules = data.get("molecules", [])
    if not molecules:
        # Fall back to free-text search
        url = f"{_BASE}/molecule?q={encoded}&format=json&limit=3"
        data = _get(url)
        molecules = data.get("molecules", [])

    if not molecules:
        return f"No drug information found for '{drug_name}' in ChEMBL."

    mol = molecules[0]
    props = mol.get("molecule_properties") or {}
    hierarchy = mol.get("molecule_hierarchy") or {}

    name = mol.get("pref_name") or drug_name
    chembl_id = mol.get("molecule_chembl_id", "N/A")
    mol_type = mol.get("molecule_type", "N/A")
    max_phase = mol.get("max_phase", "N/A")
    oral = mol.get("oral", None)
    mw = props.get("full_mw", "N/A")

    # Fetch mechanism of action if available
    mech_url = f"{_BASE}/mechanism?molecule_chembl_id={chembl_id}&format=json&limit=3"
    try:
        mech_data = _get(mech_url)
        mechanisms = mech_data.get("mechanisms", [])
        mech_lines = [m.get("mechanism_of_action", "") for m in mechanisms if m.get("mechanism_of_action")]
    except Exception:
        mech_lines = []

    lines = [
        f"Drug: {name} (ChEMBL: {chembl_id})",
        f"Type: {mol_type}",
        f"Max clinical phase: {max_phase}",
        f"Oral bioavailability: {'Yes' if oral else 'No' if oral is not None else 'Unknown'}",
        f"Molecular weight: {mw}",
    ]
    if mech_lines:
        lines.append("Mechanism of action:")
        for m in mech_lines:
            lines.append(f"  - {m}")

    return "\n".join(lines)
