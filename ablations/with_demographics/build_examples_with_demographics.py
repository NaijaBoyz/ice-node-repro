import pickle
from pathlib import Path


def build_examples_from_timeseries_with_demographics(dataset_name, processed_dir=None):
    if processed_dir is None:
        processed_dir = Path("ablations/with_demographics/data/processed")

    print(f"Generating {dataset_name.upper()} examples with demographics")
    with open(processed_dir / f"{dataset_name}_timeseries.pkl", "rb") as f:
        timeseries = pickle.load(f)

    with open(processed_dir / f"{dataset_name}_splits.pkl", "rb") as f:
        splits = pickle.load(f)

    examples = {"train": [], "val": [], "test": []}

    for split_name, patient_ids in splits.items():
        print(f"{dataset_name} split = {split_name}")

        for patient_id in patient_ids:
            patient_seq = timeseries.get(patient_id)
            if not patient_seq:
                continue

            visits = patient_seq["visits"]
            if len(visits) < 2:
                continue

            raw_times = [v[0] for v in visits]
            codes = [v[1] for v in visits]
            
            demographics = patient_seq.get("demographics", {})
            gender = demographics.get("gender", 0.5)
            normalized_age = demographics.get("normalized_age", 0.5)
            relative_duration = demographics.get("relative_duration", 0.5)

            example = {
                "times": raw_times,
                "codes": codes,
                "patient_id": patient_id,
                "n_visits": len(raw_times),
                "demographics": [gender, normalized_age, relative_duration],
            }

            examples[split_name].append(example)

        print(f"  {len(examples[split_name])} patients")

    for split_name in ["train", "val", "test"]:
        output = processed_dir / f"{dataset_name}_{split_name}_examples.pkl"
        with open(output, "wb") as f:
            pickle.dump(examples[split_name], f)

    return examples


if __name__ == "__main__":
    PROCESSED_DIR = Path("ablations/with_demographics/data/processed")

    print("\n" + "="*80)
    print("BUILDING EXAMPLES WITH DEMOGRAPHIC FEATURES")
    print("="*80)
    
    print("\n=== Building MIMIC-III examples ===")
    build_examples_from_timeseries_with_demographics("mimic3", PROCESSED_DIR)

    print("\n=== Building MIMIC-IV examples ===")
    build_examples_from_timeseries_with_demographics("mimic4", PROCESSED_DIR)

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
    print("\nDemographic features (3D vector):")
    print("  [0] gender: 0.0 (female), 1.0 (male), 0.5 (unknown)")
    print("  [1] normalized_age: age at first admission / 100")
    print("  [2] relative_duration: total trajectory duration / 365 days")
