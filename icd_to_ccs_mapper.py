"""
ICD to CCS Mapping Module
Handles mapping of ICD-9-CM and ICD-10-CM codes to CCS categories
using the 2015 multi-level CCS tools + ICD-10→ICD-9 GEM.
"""

import csv
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Set


class ICDtoCCSMapper:
    """Maps ICD-9-CM and ICD-10-CM codes to CCS (Clinical Classifications Software) categories"""

    def __init__(self, data_path: str = "data/mappings"):
        # data_path should contain:
        #   Multi_Level_CCS_2015/ccs_multi_dx_tool_2015.csv
        #   Multi_Level_CCS_2015/dxmlabel-13.csv
        #   icd10cm_to_icd9cm_gem.csv
        self.data_path = Path(data_path)

        self.icd9_to_ccs: Dict[str, Dict[str, str]] = {}
        self.icd10_to_icd9: Dict[str, List[str]] = {}
        self.icd10_to_ccs: Dict[str, List[Tuple[str, str]]] = {}
        self.ccs_labels: Dict[str, str] = {}

    # ---------- ICD-9 → CCS from multi-level dx tool ----------

    def load_icd9_to_ccs(self, filename: str = "Multi_Level_CCS_2015/ccs_multi_dx_tool_2015.csv"):
        """Load ICD-9-CM to CCS mapping from multi-level diagnosis tool"""
        filepath = self.data_path / filename
        print(f"Loading ICD-9-CM to CCS mappings from {filepath}")

        codes_by_level: Dict[int, Set[str]] = {1: set(), 2: set(), 3: set(), 4: set()}

        with filepath.open("r", encoding="utf-8") as f:
            reader = csv.reader(f, quotechar="'")
            header = next(reader)  # skip header

            count = 0
            for row in reader:
                if len(row) < 9:
                    continue

                icd9_code = row[0].strip().strip("'")

                ccs_lvl1 = row[1].strip().strip("'")
                ccs_lvl1_label = row[2].strip().strip('"').strip()

                ccs_lvl2 = row[3].strip().strip("'")
                ccs_lvl2_label = row[4].strip().strip('"').strip()

                ccs_lvl3 = row[5].strip().strip("'")
                ccs_lvl3_label = row[6].strip().strip('"').strip()

                ccs_lvl4 = row[7].strip().strip("'")
                ccs_lvl4_label = row[8].strip().strip('"').strip()

                self.icd9_to_ccs[icd9_code] = {
                    "level_1": ccs_lvl1,
                    "level_1_label": ccs_lvl1_label,
                    "level_2": ccs_lvl2,
                    "level_2_label": ccs_lvl2_label,
                    "level_3": ccs_lvl3,
                    "level_3_label": ccs_lvl3_label,
                    "level_4": ccs_lvl4,
                    "level_4_label": ccs_lvl4_label,
                }

                # Track codes per level
                if ccs_lvl1:
                    codes_by_level[1].add(ccs_lvl1)
                    self.ccs_labels.setdefault(ccs_lvl1, ccs_lvl1_label)
                if ccs_lvl2:
                    codes_by_level[2].add(ccs_lvl2)
                    self.ccs_labels.setdefault(ccs_lvl2, ccs_lvl2_label)
                if ccs_lvl3:
                    codes_by_level[3].add(ccs_lvl3)
                    self.ccs_labels.setdefault(ccs_lvl3, ccs_lvl3_label)
                if ccs_lvl4:
                    codes_by_level[4].add(ccs_lvl4)
                    self.ccs_labels.setdefault(ccs_lvl4, ccs_lvl4_label)

                count += 1

        print(f"Loaded {count} ICD-9-CM to CCS mappings")
        for lvl in sorted(codes_by_level):
            print(f"  Distinct CCS codes at level {lvl}: {len(codes_by_level[lvl])}")
        all_codes = set().union(*codes_by_level.values())
        print(f"Unique CCS codes across levels 1–4 from ICD-9: {len(all_codes)}")

        return self.icd9_to_ccs

    # ---------- ICD-10 → ICD-9 GEM ----------

    def load_icd10_to_icd9_gem(self, filename: str = "icd10cm_to_icd9cm_gem.csv"):
        """Load ICD-10-CM → ICD-9-CM GEM (General Equivalence Mappings)"""
        filepath = self.data_path / filename
        print(f"Loading ICD-10-CM to ICD-9-CM GEM from {filepath}")

        df = pd.read_csv(filepath)

        count = 0
        for _, row in df.iterrows():
            icd10 = str(row["icd10cm"]).strip().upper()
            icd9 = str(row["icd9cm"]).strip()

            if not icd10 or not icd9:
                continue

            self.icd10_to_icd9.setdefault(icd10, []).append(icd9)
            count += 1

        print(f"Loaded {count} ICD-10-CM to ICD-9-CM GEM rows")
        print(f"Unique ICD-10 codes in GEM: {len(self.icd10_to_icd9)}")
        return self.icd10_to_icd9

    # ---------- ICD-10 → CCS via ICD-9 ----------

    def build_icd10_to_ccs(self):
        """Build ICD-10-CM → CCS mapping using ICD-9 multi-level CCS"""
        print("Building ICD-10-CM to CCS mappings...")

        count = 0
        unmapped = 0

        for icd10, icd9_list in self.icd10_to_icd9.items():
            ccs_mappings: Set[Tuple[str, str]] = set()

            for icd9 in icd9_list:
                icd9_variants = [icd9, icd9.lstrip("0")]

                for variant in icd9_variants:
                    mapping = self.icd9_to_ccs.get(variant)
                    if not mapping:
                        continue

                    for level_name in ["level_1", "level_2", "level_3", "level_4"]:
                        code = mapping.get(level_name)
                        if code and code.strip() and code != " ":
                            ccs_mappings.add((level_name, code))
                    break  # stop after first variant that hits

            if ccs_mappings:
                self.icd10_to_ccs[icd10] = list(ccs_mappings)
                count += 1
            else:
                unmapped += 1

        print(f"Built {count} ICD-10-CM to CCS mappings ({unmapped} ICD-10 codes unmapped)")
        return self.icd10_to_ccs

    # ---------- Optional: load full CCS labels from dxmlabel ----------

    def load_ccs_labels_from_dxmlabel(self, filename: str = "Multi_Level_CCS_2015/dxmlabel-13.csv"):
        """Load full list of multi-level CCS diagnosis categories / labels"""
        filepath = self.data_path / filename
        if not filepath.exists():
            print(f"{filepath} not found, skipping extra CCS label load")
            return

        print(f"Loading CCS category labels from {filepath}")
        with filepath.open("r", encoding="utf-8") as f:
            reader = csv.reader(f, quotechar="'")
            header = next(reader)  # first row is header
            for row in reader:
                if not row:
                    continue
                code = row[0].strip().strip("'").strip()
                if not code:
                    continue
                # label is second column
                if len(row) > 1:
                    label = row[1].strip().strip('"').strip()
                else:
                    label = ""
                if label:
                    self.ccs_labels.setdefault(code, label)

        print(f"Total CCS categories with labels in mapper: {len(self.ccs_labels)}")

    # ---------- Main mapping method used in ETL ----------

    def map_code(self, code: str, code_type: str = "auto", level: Optional[int] = None) -> List[str]:
        """
        Map an ICD code to CCS categories.

        Args:
            code: ICD code to map (ICD-9-CM or ICD-10-CM)
            code_type: 'ICD9', 'ICD10', or 'auto' (auto-detect)
            level: CCS level (1, 2, 3, or 4) or None for all levels

        Returns:
            List of CCS codes (strings like '1', '1.1', '1.1.1', '1.1.2.1', ...)
        """
        code = str(code).strip().upper()
        if not code:
            return []

        # Auto-detect ICD type
        if code_type == "auto":
            # ICD-10 codes usually start with a letter
            if code[0].isalpha():
                code_type = "ICD10"
            else:
                code_type = "ICD9"

        ccs_codes: List[str] = []

        if code_type == "ICD9":
            mapping = self.icd9_to_ccs.get(code)
            if not mapping:
                mapping = self.icd9_to_ccs.get(code.lstrip("0"))
            if not mapping:
                return []

            if level is None:
                for lvl in [1, 2, 3, 4]:
                    key = f"level_{lvl}"
                    val = mapping.get(key)
                    if val and val.strip() and val != " ":
                        ccs_codes.append(val)
            else:
                key = f"level_{level}"
                val = mapping.get(key)
                if val and val.strip() and val != " ":
                    ccs_codes.append(val)

        elif code_type == "ICD10":
            mappings = self.icd10_to_ccs.get(code)
            if not mappings:
                return []

            for lvl_str, ccs_code in mappings:
                if not ccs_code or ccs_code == " ":
                    continue
                if level is None:
                    ccs_codes.append(ccs_code)
                else:
                    lvl_num = int(lvl_str.split("_")[1])
                    if lvl_num == level:
                        ccs_codes.append(ccs_code)

        return ccs_codes

    # ---------- Convenience init ----------

    def initialize(self):
        """Initialize all mappings and labels"""
        self.load_icd9_to_ccs()
        self.load_icd10_to_icd9_gem()
        self.build_icd10_to_ccs()
        # Optional: bring in full label list (includes categories with no ICD codes)
        self.load_ccs_labels_from_dxmlabel()
        print("ICD→CCS mapper initialization complete!")
        return self


# Simple quick test when you run this file directly
if __name__ == "__main__":
    mapper = ICDtoCCSMapper()
    mapper.initialize()

    print("\n=== Testing ICD to CCS Mapping ===")

    test_icd9_codes = ["01000", "4280", "2500", "486", "V3001"]
    print("\nTesting ICD-9-CM codes:")
    for code in test_icd9_codes:
        ccs_codes = mapper.map_code(code, code_type="ICD9", level=None)
        print(f"  {code} -> {ccs_codes}")
        for ccs in ccs_codes:
            label = mapper.ccs_labels.get(ccs, "Unknown")
            print(f"    {ccs}: {label}")

    test_icd10_codes = ["A000", "I500", "E119", "J189", "Z3800"]
    print("\nTesting ICD-10-CM codes:")
    for code in test_icd10_codes:
        ccs_codes = mapper.map_code(code, code_type="ICD10", level=None)
        print(f"  {code} -> {ccs_codes}")
        for ccs in ccs_codes:
            label = mapper.ccs_labels.get(ccs, "Unknown")
            print(f"    {ccs}: {label}")

    print("\nMapper test complete.")
