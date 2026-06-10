"""
Vital Validator — Medical sanity checks for numeric vital signs and lab values.

Prevents physiologically impossible values from entering the clinical state.
All extracted numeric values should pass through validate_single() or
validate_vitals_dict() before being stored in ConversationState.

Problem solved: impossible fever values (e.g. 1024°F), negative heart rates,
and other out-of-range values produced by noisy OCR or LLM hallucination.
"""
import re
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physiological ranges  [min, max]  — inclusive
# ---------------------------------------------------------------------------

_TEMP_F = (90.0, 115.0)
_TEMP_C = (32.0, 46.0)

VITAL_RANGES: dict[str, tuple[float, float]] = {
    "heart_rate":        (20.0, 300.0),
    "respiratory_rate":  ( 4.0,  80.0),
    "oxygen_saturation": (50.0, 100.0),
    "systolic_bp":       (50.0, 300.0),
    "diastolic_bp":      (20.0, 200.0),
    "weight_kg":         ( 0.5, 650.0),
    "height_cm":         (20.0, 280.0),
    "bmi":               ( 5.0, 100.0),
    "blood_glucose":     (20.0, 1000.0),
    "fasting_glucose":   (20.0,  600.0),
    "hemoglobin":        ( 2.0,  25.0),
    "wbc":               ( 0.1,  200.0),
    "platelets":         ( 5.0, 2000.0),
    "creatinine":        ( 0.1,  30.0),
    "sodium":            (100.0, 180.0),
    "potassium":         ( 1.5,   9.0),
    "hba1c":             ( 2.0,  20.0),
    "tsh":               (0.001, 100.0),
    "crp":               ( 0.0,  500.0),
    "esr":               ( 0.0,  200.0),
    "lactate":           ( 0.3,  30.0),
    "alt":               ( 1.0, 5000.0),
    "ast":               ( 1.0, 5000.0),
    "bilirubin":         ( 0.0,  50.0),
    "troponin":          ( 0.0,  50.0),
    "d_dimer":           ( 0.0, 100.0),
    "procalcitonin":     ( 0.0, 1000.0),
    "ferritin":          ( 0.0, 50000.0),
    "uric_acid":         ( 0.5,  30.0),
}

# Key aliases → canonical VITAL_RANGES key
_ALIASES: dict[str, str] = {
    "temp":                 "temperature",
    "temperature":          "temperature",
    "fever_temperature":    "temperature",
    "fever_temp":           "temperature",
    "fever_temp_f":         "temperature",
    "hr":                   "heart_rate",
    "pulse":                "heart_rate",
    "heart_rate":           "heart_rate",
    "rr":                   "respiratory_rate",
    "resp_rate":            "respiratory_rate",
    "respiratory_rate":     "respiratory_rate",
    "spo2":                 "oxygen_saturation",
    "o2_sat":               "oxygen_saturation",
    "o2sat":                "oxygen_saturation",
    "oxygen_saturation":    "oxygen_saturation",
    "sbp":                  "systolic_bp",
    "bp_systolic":          "systolic_bp",
    "systolic_bp":          "systolic_bp",
    "dbp":                  "diastolic_bp",
    "bp_diastolic":         "diastolic_bp",
    "diastolic_bp":         "diastolic_bp",
    "weight":               "weight_kg",
    "weight_kg":            "weight_kg",
    "height":               "height_cm",
    "height_cm":            "height_cm",
    "glucose":              "blood_glucose",
    "blood_sugar":          "blood_glucose",
    "blood_glucose":        "blood_glucose",
    "fasting_glucose":      "fasting_glucose",
    "fbs":                  "fasting_glucose",
    "hb":                   "hemoglobin",
    "hgb":                  "hemoglobin",
    "hemoglobin":           "hemoglobin",
    "wbc":                  "wbc",
    "platelets":            "platelets",
    "creatinine":           "creatinine",
    "sodium":               "sodium",
    "potassium":            "potassium",
    "hba1c":                "hba1c",
    "tsh":                  "tsh",
    "crp":                  "crp",
    "esr":                  "esr",
    "lactate":              "lactate",
    "alt":                  "alt",
    "sgpt":                 "alt",
    "ast":                  "ast",
    "sgot":                 "ast",
    "bilirubin":            "bilirubin",
    "troponin":             "troponin",
    "d_dimer":              "d_dimer",
    "procalcitonin":        "procalcitonin",
    "ferritin":             "ferritin",
    "uric_acid":            "uric_acid",
}


def _to_float(value: Any) -> float | None:
    """Extract a numeric value from int, float, or string."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        m = re.search(r"(\d+(?:\.\d+)?)", str(value))
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return None


def _is_temperature_key(key: str) -> bool:
    k = key.lower()
    return (
        k in ("temperature", "temp", "fever_temperature", "fever_temp", "fever_temp_f")
        or "fever" in k
        or (k == "temp")
    )


def validate_single(
    key: str,
    value: Any,
) -> tuple[Any, bool, str | None]:
    """
    Validate one vital sign or lab value.

    Returns (validated_value, is_valid, warning_message).
      - validated_value : original value if valid, None if rejected
      - is_valid        : True if the value passes the sanity check
      - warning_message : reason string if rejected, else None
    """
    numeric = _to_float(value)
    if numeric is None:
        return value, True, None  # non-numeric — pass through unchanged

    key_lower = key.lower()

    # Temperature: unit determined by magnitude
    if _is_temperature_key(key_lower):
        if numeric > 50:
            lo, hi = _TEMP_F
            unit = "°F"
        else:
            lo, hi = _TEMP_C
            unit = "°C"
        if not (lo <= numeric <= hi):
            warn = (
                f"Rejected {key}={value} {unit}: "
                f"outside valid range [{lo}, {hi}]"
            )
            logger.warning(f"VitalValidator: {warn}")
            return None, False, warn
        return value, True, None

    # All other vitals — resolve alias → canonical → range
    canonical = _ALIASES.get(key_lower, key_lower)
    if canonical in VITAL_RANGES:
        lo, hi = VITAL_RANGES[canonical]
        if not (lo <= numeric <= hi):
            warn = (
                f"Rejected {key}={value}: "
                f"outside valid range [{lo:.3g}, {hi:.3g}] for {canonical}"
            )
            logger.warning(f"VitalValidator: {warn}")
            return None, False, warn

    return value, True, None


def validate_vitals_dict(vitals: dict) -> tuple[dict, list[str]]:
    """
    Validate all entries in a vitals dictionary.

    Returns (valid_vitals, warning_messages).
    Keys with invalid values are dropped; unrecognised keys pass through.
    """
    valid: dict = {}
    warnings: list[str] = []

    for key, val in vitals.items():
        validated, is_valid, warning = validate_single(key, val)
        if warning:
            warnings.append(warning)
        if is_valid and validated is not None:
            valid[key] = validated

    return valid, warnings


def validate_clinical_slots(slots: dict) -> tuple[dict, list[str]]:
    """
    Validate temperature-related and numeric clinical slots.
    Non-vital slots (strings, booleans, etc.) pass through unchanged.

    Returns (valid_slots, warning_messages).
    """
    valid: dict = {}
    warnings: list[str] = []

    for key, val in slots.items():
        key_lower = key.lower()
        if _is_temperature_key(key_lower) or key_lower in _ALIASES:
            validated, is_valid, warning = validate_single(key, val)
            if warning:
                warnings.append(warning)
            if is_valid and validated is not None:
                valid[key] = validated
        else:
            valid[key] = val  # non-vital slot — pass through

    return valid, warnings
