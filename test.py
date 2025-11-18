# inspect_ccs_levels.py
#
# Quick script to inspect CCS levels using pyhealth.medcode.CrossMap.
# It:
#   - Loads raw diagnosis tables from MIMIC-III and MIMIC-IV
#   - Extracts unique ICD codes
#   - For each CCS level, maps ICD9CM -> CCSCM and ICD10CM -> CCSCM
#   - Prints coverage and number of unique CCS codes per level
#
# Run with:
#   uv run inspect_ccs_levels.py

import os
from collections import defaultdict

import pandas as pd
from pyhealth.medcode import CrossMap


# ----- CONFIG -----
MIMIC3_DIAG_PATH = "data/mimiciii-1.4/DIAGNOSES_ICD.csv"
MIMIC4_DIAG_PATH = "data/mimiciv-3.1/diagnoses_icd.csv"

# CCS levels you want to test
CCS_LEVELS = [1, 2, 3, 4]


def load_mimic3_icd9_codes(path: str) -> set[str]:
    """Load unique ICD-9-CM diagnosis codes from MIMIC-III DIAGNOSES_ICD."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"MIMIC-III diagnoses file not found at {path}")

    df = pd.read_csv(path, usecols=["ICD9_CODE"])
    codes = set(df["ICD9_CODE"].dropna().astype(str).str.strip())
    print(f"MIMIC-III: loaded {len(codes)} unique ICD9 codes")
    return codes


def load_mimic4_icd10_codes(path: str) -> set[str]:
    """Load unique ICD-10-CM diagnosis codes from MIMIC-IV diagnoses_icd."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"MIMIC-IV diagnoses file not found at {path}")

    df = pd.read_csv(path, usecols=["icd_code", "icd_version"])
    # In MIMIC-IV, icd_version == 10 => ICD-10-CM
    df = df[df["icd_version"] == 10]
    codes = set(df["icd_code"].dropna().astype(str).str.strip())
    print(f"MIMIC-IV: loaded {len(codes)} unique ICD10 codes")
    return codes


def inspect_level_for_codes(
    level: int,
    icd9_codes: set[str],
    icd10_codes: set[str],
    icd9_to_ccs: CrossMap,
    icd10_to_ccs: CrossMap,
):
    """For a given CCS level, map ICD codes and report stats."""

    # Containers
    icd9_mapped = 0
    icd9_total = len(icd9_codes)
    icd9_ccs_set = set()

    icd10_mapped = 0
    icd10_total = len(icd10_codes)
    icd10_ccs_set = set()

    # Map ICD-9 -> CCS
    for code in icd9_codes:
        try:
            ccs_list = icd9_to_ccs.map(code, target_kwargs={"level": level})
        except Exception:
            ccs_list = []
        if ccs_list:
            icd9_mapped += 1
            icd9_ccs_set.update(ccs_list)

    # Map ICD-10 -> CCS
    for code in icd10_codes:
        try:
            ccs_list = icd10_to_ccs.map(code, target_kwargs={"level": level})
        except Exception:
            ccs_list = []
        if ccs_list:
            icd10_mapped += 1
            icd10_ccs_set.update(ccs_list)

    # Compute coverage
    icd9_cov = icd9_mapped / icd9_total * 100 if icd9_total > 0 else 0.0
    icd10_cov = icd10_mapped / icd10_total * 100 if icd10_total > 0 else 0.0

    print("\n" + "=" * 70)
    print(f"CCS LEVEL {level}")
    print("-" * 70)
    print(f"MIMIC-III (ICD9CM):")
    print(f"  Unique ICD9 codes:          {icd9_total}")
    print(f"  Mapped ICD9 codes:          {icd9_mapped}")
    print(f"  Coverage:                   {icd9_cov:.2f}%")
    print(f"  Unique CCS codes (level {level}): {len(icd9_ccs_set)}")

    print(f"\nMIMIC-IV (ICD10CM):")
    print(f"  Unique ICD10 codes:         {icd10_total}")
    print(f"  Mapped ICD10 codes:         {icd10_mapped}")
    print(f"  Coverage:                   {icd10_cov:.2f}%")
    print(f"  Unique CCS codes (level {level}): {len(icd10_ccs_set)}")


def main():
    # ---- 1. Load unique ICD codes from raw CSVs ----
    icd9_codes = load_mimic3_icd9_codes(MIMIC3_DIAG_PATH)
    icd10_codes = load_mimic4_icd10_codes(MIMIC4_DIAG_PATH)

    # ---- 2. Initialize CrossMap objects once ----
    print("\nInitializing CrossMap objects...")
    icd9_to_ccs = CrossMap.load("ICD9CM", "CCSCM")
    icd10_to_ccs = CrossMap.load("ICD10CM", "CCSCM")

    # ---- 3. Inspect each CCS level ----
    for level in CCS_LEVELS:
        inspect_level_for_codes(
            level=level,
            icd9_codes=icd9_codes,
            icd10_codes=icd10_codes,
            icd9_to_ccs=icd9_to_ccs,
            icd10_to_ccs=icd10_to_ccs,
        )


if __name__ == "__main__":
    main()
