#!/usr/bin/env python3
"""
Demonstration script for ICD to CCS mapping
Shows how to use the mapper with your MIMIC data
"""

from icd_to_ccs_mapper import ICDtoCCSMapper

def demonstrate_mapper():
    """Demonstrate the ICD to CCS mapper functionality"""
    
    print("=" * 60)
    print("ICD to CCS Mapper Demonstration")
    print("=" * 60)
    
    # Initialize mapper
    print("\n1. Initializing the mapper...")
    mapper = ICDtoCCSMapper()
    mapper.initialize()
    
    print("\n2. Mapping Statistics:")
    print(f"   - ICD-9-CM to CCS mappings: {len(mapper.icd9_to_ccs)}")
    print(f"   - ICD-10-CM to ICD-9-CM mappings: {len(mapper.icd10_to_icd9)}")
    print(f"   - ICD-10-CM to CCS mappings: {len(mapper.icd10_to_ccs)}")
    print(f"   - Unique CCS labels: {len(mapper.ccs_labels)}")
    
    # Demonstrate different mapping options
    print("\n3. Mapping Examples:")
    
    # Example 1: ICD-9 with all levels
    print("\n   a) ICD-9-CM '4280' (Congestive heart failure):")
    ccs_all = mapper.map_code("4280", code_type="ICD9", level=None)
    print(f"      All levels: {ccs_all}")
    for ccs in ccs_all:
        print(f"        - {ccs}: {mapper.ccs_labels.get(ccs, 'Unknown')}")
    
    # Example 2: ICD-9 with specific level
    print("\n   b) ICD-9-CM '486' (Pneumonia) - Level 3 only:")
    ccs_lvl3 = mapper.map_code("486", code_type="ICD9", level=3)
    print(f"      Level 3: {ccs_lvl3}")
    for ccs in ccs_lvl3:
        print(f"        - {ccs}: {mapper.ccs_labels.get(ccs, 'Unknown')}")
    
    # Example 3: ICD-10 mapping
    print("\n   c) ICD-10-CM 'J189' (Pneumonia, unspecified organism):")
    ccs_all = mapper.map_code("J189", code_type="ICD10", level=None)
    print(f"      All levels: {ccs_all}")
    for ccs in ccs_all:
        print(f"        - {ccs}: {mapper.ccs_labels.get(ccs, 'Unknown')}")
    
    # Example 4: Auto-detection
    print("\n   d) Auto-detecting code type:")
    test_codes = ["2500", "E119", "4019", "I10"]
    for code in test_codes:
        ccs = mapper.map_code(code, code_type="auto", level=1)
        detected_type = "ICD-10" if code[0].isalpha() else "ICD-9"
        if ccs:
            print(f"      {code} ({detected_type}) -> Level 1: {ccs[0]} - {mapper.ccs_labels.get(ccs[0], 'Unknown')}")
        else:
            print(f"      {code} ({detected_type}) -> No mapping found")
    
    # Show how to use with PyHealth
    print("\n4. Integration with PyHealth datasets:")
    print("""
    from pyhealth.datasets import MIMIC3Dataset
    
    # Load dataset
    dataset = MIMIC3Dataset(root="path/to/mimic", tables=["DIAGNOSES_ICD"])
    
    # For each visit, map ICD codes to CCS
    for patient_id, patient in dataset.patients.items():
        for visit_id, visit in patient.visits.items():
            icd_codes = visit.get_code_list(table='DIAGNOSES_ICD')
            
            ccs_codes = set()
            for icd_code in icd_codes:
                # Map to CCS (choose your level)
                mapped = mapper.map_code(icd_code, level=3)  # Level 3 for most granular
                ccs_codes.update(mapped)
            
            print(f"Visit {visit_id}: {len(icd_codes)} ICD -> {len(ccs_codes)} CCS codes")
    """)
    
    print("\n5. Key Points for ICENode Reproduction:")
    print("""
    - The mapper handles both ICD-9-CM and ICD-10-CM codes
    - ICD-10 codes are mapped through ICD-9 using the GEM crosswalk
    - You can choose CCS level (1=broad, 2=medium, 3=specific)
    - The modified main.py uses auto-detection for code type
    - Set CCS_LEVEL variable in main_modified.py to control granularity:
      * None = all levels
      * 1 = broad categories only
      * 2 = medium granularity  
      * 3 = most specific categories
    """)
    
    print("\n" + "=" * 60)
    print("Ready to use! Run main_modified.py to process your MIMIC data.")
    print("=" * 60)

if __name__ == "__main__":
    demonstrate_mapper()