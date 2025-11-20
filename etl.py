from pyhealth.datasets import MIMIC3Dataset, MIMIC4Dataset
import pickle
import json
import random
from pathlib import Path
from icd_to_ccs_mapper import ICDtoCCSMapper

mimic3 = MIMIC3Dataset(root="data/mimiciii-1.4", tables=["DIAGNOSES_ICD"], dev=False)
print("MIMIC-III dataset loaded:", len(mimic3.patients), "patients")

mimic4 = MIMIC4Dataset(root="data/mimiciv-3.1", tables=["diagnoses_icd"], dev=False)
print("MIMIC-IV dataset loaded:", len(mimic4.patients), "patients")
print("\nInitializing ICD to CCS mapper...")
mapper = ICDtoCCSMapper()
mapper.initialize()
def len_of_stay_in_days(visit):
    start = getattr(visit, "encounter_time", None)
    end = getattr(visit, "discharge_time", None)
    if start is None or end is None:
        return None
    return (end - start).total_seconds() / 86400.0


def filter_patient_ids(dataset):
    ids = []
    for pid, patient in dataset.patients.items():
        if len(patient.visits) < 2:
            continue

        ok = True

       
        for visit in patient.visits.values():
            days = len_of_stay_in_days(visit)
            if days is None or days > 14.0:
                ok = False
                break
        if not ok:
            continue

        
        timed_visits = []
        for visit_id, visit in patient.visits.items():
            encounter_time = getattr(visit, "encounter_time", None)
            discharge_time = getattr(visit, "discharge_time", None)
            if encounter_time is None or discharge_time is None:
                ok = False
                break
            timed_visits.append((encounter_time, discharge_time, visit_id))
        if not ok:
            continue

        timed_visits.sort(key=lambda x: x[0])
        for i in range(len(timed_visits) - 1):
            current_discharge = timed_visits[i][1]
            next_encounter = timed_visits[i + 1][0]
            if current_discharge > next_encounter:
                ok = False
                break

        if ok:
            ids.append(pid)

    return ids

mimic3_filtered_ids = filter_patient_ids(mimic3)
print("\nMIMIC-III filtered patient count:", len(mimic3_filtered_ids))

mimic4_filtered_ids = filter_patient_ids(mimic4)
print("MIMIC-IV filtered patient count:", len(mimic4_filtered_ids))

CCS_LEVEL = None

def map_visit_to_ccs_codes(visit):
    codes = set()
    icd_codes = []

    if hasattr(visit, "get_code_list"):
        try:
            icd_codes.extend(visit.get_code_list(table="DIAGNOSES_ICD"))
        except Exception:
            pass
        try:
            icd_codes.extend(visit.get_code_list(table="diagnoses_icd"))
        except Exception:
            pass

    for code in icd_codes:
        code_str = str(code).strip()
        if not code_str:
            continue
        mapped_ccs = mapper.map_code(code_str, code_type="auto", level=CCS_LEVEL)
        for c in mapped_ccs:
            codes.add(c)

    return sorted(codes)

def build_patient_visit_ccs_index(dataset, filtered_patient_ids):
    visit_ccs = {}
    stats = {"total_visits": 0, "visits_with_ccs": 0}

    for pid in filtered_patient_ids:
        patient = dataset.patients[pid]
        for vid, visit in patient.visits.items():
            ccs_codes = map_visit_to_ccs_codes(visit)
            visit_ccs[(pid, vid)] = ccs_codes

            stats["total_visits"] += 1
            if ccs_codes:
                stats["visits_with_ccs"] += 1

    print(f"  Total visits: {stats['total_visits']}")
    print(f"  Visits with CCS codes: {stats['visits_with_ccs']}")
    rate = 100.0 * stats["visits_with_ccs"] / max(1, stats["total_visits"])
    print(f"  Mapping success rate (per-visit): {rate:.1f}%")

    return visit_ccs

print("\nMapping MIMIC-III visits to CCS codes...")
mimic3_visit_ccs = build_patient_visit_ccs_index(mimic3, mimic3_filtered_ids)

print("\nMapping MIMIC-IV visits to CCS codes...")
mimic4_visit_ccs = build_patient_visit_ccs_index(mimic4, mimic4_filtered_ids)

unique_ccs_codes = set()
for codes in mimic3_visit_ccs.values():
    unique_ccs_codes.update(codes)
for codes in mimic4_visit_ccs.values():
    unique_ccs_codes.update(codes)

sorted_ccs_codes = sorted(unique_ccs_codes)
ccs_code_to_index = {code: idx for idx, code in enumerate(sorted_ccs_codes)}

ccs_level_str = "multi-level" if CCS_LEVEL is None else f"level {CCS_LEVEL}"
print(f"\nCCS categories in vocabulary ({ccs_level_str}): {len(sorted_ccs_codes)}")

print("\nSample CCS codes and labels:")
for code in sorted_ccs_codes[:10]:
    label = mapper.ccs_labels.get(code, "Unknown")
    print(f"  {code}: {label}")

def build_patient_timeseries(dataset, filtered_patient_ids, visit_ccs, ccs_code_to_index):
    patient_timeseries = {}

    for patient_id in filtered_patient_ids:
        patient = dataset.patients[patient_id]

        timed_visits = []
        for visit_id, visit in patient.visits.items():
            encounter_time = getattr(visit, "encounter_time", None)
            if encounter_time is None:
                continue

            timed_visits.append((encounter_time, visit_id, visit))

        if len(timed_visits) < 2:
            continue

        timed_visits.sort(key=lambda x: x[0])
        first_admission_time = timed_visits[0][0]

        visit_sequence = []
        for encounter_time, visit_id, visit in timed_visits:
            days_since_first_admission = (encounter_time - first_admission_time).total_seconds() / 86400.0

            ccs_codes = visit_ccs.get((patient_id, visit_id), [])
            if not ccs_codes:
                continue

            x = [0] * len(ccs_code_to_index)
            for code in ccs_codes:
                idx = ccs_code_to_index.get(code)
                if idx is not None:
                    x[idx] = 1

            visit_sequence.append((days_since_first_admission, x))

        if len(visit_sequence) >= 2:
            patient_timeseries[patient_id] = {
                "visits": visit_sequence,
                "n_visits": len(visit_sequence),
            }

    return patient_timeseries

mimic3_timeseries = build_patient_timeseries(mimic3, mimic3_filtered_ids, mimic3_visit_ccs, ccs_code_to_index)
mimic4_timeseries = build_patient_timeseries(mimic4, mimic4_filtered_ids, mimic4_visit_ccs, ccs_code_to_index)

print(f"\nMIMIC-III patients with valid timeseries: {len(mimic3_timeseries)}")
print(f"MIMIC-IV patients with valid timeseries: {len(mimic4_timeseries)}")

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(exist_ok=True, parents=True)

ccs_vocab_with_labels = {
    "codes": sorted_ccs_codes,
    "labels": {code: mapper.ccs_labels.get(code, "Unknown") for code in sorted_ccs_codes},
}
with (PROCESSED_DIR / "ccs_vocab.json").open("w") as f:
    json.dump(ccs_vocab_with_labels, f, indent=2)

with (PROCESSED_DIR / "mimic3_timeseries.pkl").open("wb") as f:
    pickle.dump(mimic3_timeseries, f)

with (PROCESSED_DIR / "mimic4_timeseries.pkl").open("wb") as f:
    pickle.dump(mimic4_timeseries, f)

print(f"\nProcessed data saved to {PROCESSED_DIR}")

def split_patients(patient_ids, seed=42, train_ratio=0.7, val_ratio=0.15):
    random.seed(seed)
    ids = list(patient_ids)
    random.shuffle(ids)
    n = len(ids)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val :]
    return train_ids, val_ids, test_ids

m3_ids = list(mimic3_timeseries.keys())
m4_ids = list(mimic4_timeseries.keys())

m3_train_ids, m3_val_ids, m3_test_ids = split_patients(m3_ids)
m4_train_ids, m4_val_ids, m4_test_ids = split_patients(m4_ids)

print(f"\nMIMIC-III train/val/test split: {len(m3_train_ids)}/{len(m3_val_ids)}/{len(m3_test_ids)}")
print(f"MIMIC-IV train/val/test split: {len(m4_train_ids)}/{len(m4_val_ids)}/{len(m4_test_ids)}")

m3_splits = {"train": m3_train_ids, "val": m3_val_ids, "test": m3_test_ids}
with (PROCESSED_DIR / "mimic3_splits.pkl").open("wb") as f:
    pickle.dump(m3_splits, f)

m4_splits = {"train": m4_train_ids, "val": m4_val_ids, "test": m4_test_ids}
with (PROCESSED_DIR / "mimic4_splits.pkl").open("wb") as f:
    pickle.dump(m4_splits, f)

print("\nData preparation complete!")
