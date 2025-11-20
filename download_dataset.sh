#!/usr/bin/env bash
set -euo pipefail


USER_NAME="username"
USER_PASS="password"

M3_BASE_URL="https://physionet.org/files/mimiciii/1.4"
M3_DEST="./data/mimiciii-1.4"
M3_FILES=(
  "ADMISSIONS.csv.gz"
  "DIAGNOSES_ICD.csv.gz"
  "PATIENTS.csv.gz"
  "D_ICD_DIAGNOSES.csv.gz"
)

M4_BASE_URL="https://physionet.org/files/mimiciv/3.1/hosp"
M4_DEST="./data/mimiciv-3.1"
M4_FILES=(
  "patients.csv.gz"
  "admissions.csv.gz"
  "diagnoses_icd.csv.gz"
  "d_icd_diagnoses.csv.gz"
)

MAPPINGS_DEST="./data/mappings"
MULTI_CCS_DEST="$MAPPINGS_DEST/Multi_Level_CCS_2015"
MULTI_CCS_URL="https://hcup-us.ahrq.gov/toolssoftware/ccs/Multi_Level_CCS_2015.zip"

mkdir -p "$M3_DEST"
mkdir -p "$M4_DEST"
mkdir -p "$MAPPINGS_DEST"
mkdir -p "$MULTI_CCS_DEST"

echo "Downloading mapping files..."
wget -O "$MAPPINGS_DEST/CCS_ICD9_MultiLevelDx.txt" \
     "https://hcup-us.ahrq.gov/toolssoftware/ccs/AppendixCMultiDX.txt" # TODO: Update this and one below 
wget -O "$MAPPINGS_DEST/CCS_CategoryNames_FullLabels.pdf" \
     "https://hcup-us.ahrq.gov/toolssoftware/ccs/CCSCategoryNames_FullLabels.pdf" # TODO: Update this and one above
wget -O "$MAPPINGS_DEST/icd10cm_to_icd9cm_gem.csv" \
     "https://data.nber.org/gem/icd10cmtoicd9gem.csv"

echo "Downloading Multi-Level CCS ZIP..."
wget --continue --no-parent --directory-prefix="$MULTI_CCS_DEST" "$MULTI_CCS_URL"

echo "Unzipping Multi-Level CCS..."
unzip -o "$MULTI_CCS_DEST/Multi_Level_CCS_2015.zip" -d "$MULTI_CCS_DEST"

echo "Downloading MIMIC-III files..."
for f in "${M3_FILES[@]}"; do
  echo " -> $f"
  wget --user="$USER_NAME" --password="$USER_PASS" --continue --no-parent --directory-prefix="$M3_DEST" "${M3_BASE_URL}/${f}"
done

echo "Downloading MIMIC-IV files..."
for f in "${M4_FILES[@]}"; do
  echo " -> $f"
  wget --user="$USER_NAME" --password="$USER_PASS" --continue --no-parent --directory-prefix="$M4_DEST" "${M4_BASE_URL}/${f}"
done

echo "Decompressing files..."
cd "$M3_DEST"; gunzip *.csv.gz; cd - >/dev/null
cd "$M4_DEST"; gunzip *.csv.gz; cd - >/dev/null

echo "Done. Files are ready in $M3_DEST and $M4_DEST; mappings in $MAPPINGS_DEST"
