import pickle
from pathlib import Path


def build_examples_from_timeseries(dataset_name, processed_dir=None):

    if processed_dir is None:
        processed_dir = Path("data/processed")

    print(f"Generating {dataset_name.upper()} examples (ICE-NODE normalization)")
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

            times = raw_times

            example = {
                "times": times,
                "codes": codes,
                "patient_id": patient_id,
                "n_visits": len(times),
            }

            examples[split_name].append(example)

        print(f"  {len(examples[split_name])} patients")

    for split_name in ["train", "val", "test"]:
        output = processed_dir / f"{dataset_name}_{split_name}_examples.pkl"
        with open(output, "wb") as f:
            pickle.dump(examples[split_name], f)

    return examples


if __name__ == "__main__":
    PROCESSED_DIR = Path("data/processed")

    print("\n=== Building MIMIC-III examples ===")
    build_examples_from_timeseries("mimic3", PROCESSED_DIR)

    print("\n=== Building MIMIC-IV examples ===")
    build_examples_from_timeseries("mimic4", PROCESSED_DIR)

    print("\nDone.")
