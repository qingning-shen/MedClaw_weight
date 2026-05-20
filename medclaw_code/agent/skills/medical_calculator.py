"""Standard medical calculations — BMI, eGFR (CKD-EPI), BSA. No external API needed."""

import math
from smolagents import tool


@tool
def medical_calculator(
    formula: str,
    weight_kg: float = None,
    height_cm: float = None,
    age_years: int = None,
    is_female: bool = None,
    creatinine_mg_dl: float = None,
) -> str:
    """
    Perform standard medical calculations: BMI, eGFR, or BSA.

    Args:
        formula: Calculation type — one of 'bmi', 'egfr', or 'bsa'.
        weight_kg: Body weight in kilograms (required for bmi and bsa).
        height_cm: Height in centimeters (required for bmi and bsa).
        age_years: Patient age in years (required for egfr).
        is_female: True if patient is female (required for egfr).
        creatinine_mg_dl: Serum creatinine in mg/dL (required for egfr).

    Returns:
        Calculated value with units and clinical interpretation.
    """
    formula = formula.lower().strip()

    # ── BMI ──────────────────────────────────────────────────────────────────
    if formula == "bmi":
        if weight_kg is None or height_cm is None:
            return "BMI requires weight_kg and height_cm."
        if weight_kg <= 0 or height_cm <= 0:
            return "Weight and height must be positive values."
        height_m = height_cm / 100.0
        bmi = weight_kg / (height_m ** 2)

        if bmi < 18.5:
            category = "Underweight"
        elif bmi < 25.0:
            category = "Normal weight"
        elif bmi < 30.0:
            category = "Overweight"
        elif bmi < 35.0:
            category = "Obese (Class I)"
        elif bmi < 40.0:
            category = "Obese (Class II)"
        else:
            category = "Obese (Class III — Severe)"

        return (
            f"BMI = {bmi:.1f} kg/m²\n"
            f"Category: {category}\n"
            f"(Weight: {weight_kg} kg, Height: {height_cm} cm)"
        )

    # ── eGFR — CKD-EPI 2021 (race-free) ─────────────────────────────────────
    elif formula == "egfr":
        if creatinine_mg_dl is None or age_years is None or is_female is None:
            return "eGFR requires creatinine_mg_dl, age_years, and is_female."
        if creatinine_mg_dl <= 0 or age_years <= 0:
            return "Creatinine and age must be positive values."

        # CKD-EPI 2021 equation (race-free)
        sex_kappa = 0.7 if is_female else 0.9
        sex_alpha = -0.241 if is_female else -0.302
        sex_factor = 1.012 if is_female else 1.0

        cr_ratio = creatinine_mg_dl / sex_kappa
        if cr_ratio < 1.0:
            egfr = 142 * (cr_ratio ** sex_alpha) * (0.9938 ** age_years) * sex_factor
        else:
            egfr = 142 * (cr_ratio ** -1.200) * (0.9938 ** age_years) * sex_factor

        # CKD staging
        if egfr >= 90:
            stage = "G1 — Normal or high"
        elif egfr >= 60:
            stage = "G2 — Mildly decreased"
        elif egfr >= 45:
            stage = "G3a — Mildly to moderately decreased"
        elif egfr >= 30:
            stage = "G3b — Moderately to severely decreased"
        elif egfr >= 15:
            stage = "G4 — Severely decreased"
        else:
            stage = "G5 — Kidney failure"

        sex_str = "Female" if is_female else "Male"
        return (
            f"eGFR = {egfr:.1f} mL/min/1.73m² (CKD-EPI 2021)\n"
            f"CKD Stage: {stage}\n"
            f"(Creatinine: {creatinine_mg_dl} mg/dL, Age: {age_years} years, Sex: {sex_str})"
        )

    # ── BSA — Mosteller formula ───────────────────────────────────────────────
    elif formula == "bsa":
        if weight_kg is None or height_cm is None:
            return "BSA requires weight_kg and height_cm."
        if weight_kg <= 0 or height_cm <= 0:
            return "Weight and height must be positive values."
        bsa = math.sqrt((height_cm * weight_kg) / 3600.0)
        return (
            f"BSA = {bsa:.2f} m² (Mosteller formula)\n"
            f"(Weight: {weight_kg} kg, Height: {height_cm} cm)\n"
            "Reference range: adults ~1.6–1.9 m²"
        )

    else:
        return f"Unknown formula '{formula}'. Supported: 'bmi', 'egfr', 'bsa'."
