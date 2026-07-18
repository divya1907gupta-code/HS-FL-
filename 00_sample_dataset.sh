#!/bin/bash
# ============================================================================
# 00_sample_dataset.sh
#
# Streams the raw NF-BoT-IoT-V2 CSV directly out of the UQ-provided zip
# archive (37,763,498 rows) and writes a class-stratified sample to disk,
# WITHOUT ever extracting the full 6GB CSV or loading it fully into memory.
#
# Sampling rates (chosen to keep the working set tractable while preserving
# every rare-class example):
#   Benign          -> kept 100%   (135,037 rows)
#   Theft           -> kept 100%   (2,431 rows)
#   Reconnaissance  -> kept 5%     (~131,000 rows)
#   DoS             -> kept 2%     (~333,000 rows)
#   DDoS            -> kept 2%     (~367,000 rows)
#
# Result: ~967,000 rows written to bot_iot_sample_raw.csv
#
# Usage:
#   ./00_sample_dataset.sh /path/to/NF-BoT-IoT-V2.zip /path/to/output_dir
# ============================================================================
set -e

ZIP_PATH="${1:?Usage: $0 <path-to-NF-BoT-IoT-V2-zip> <output-dir>}"
OUT_DIR="${2:?Usage: $0 <path-to-NF-BoT-IoT-V2-zip> <output-dir>}"
mkdir -p "$OUT_DIR"

echo "Extracting feature-description file..."
unzip -j "$ZIP_PATH" "*/data/NetFlow_v2_Features.csv" -d "$OUT_DIR"

echo "Streaming + stratified-sampling the main CSV (this reads ~6GB via a pipe, no full extraction)..."
unzip -p "$ZIP_PATH" "*/data/NF-BoT-IoT-v2.csv" | awk -F',' '
BEGIN{srand(42)}
NR==1{print; next}
{
  atk=$NF; gsub(/\r/,"",atk);
  keep=0
  if (atk=="Benign") keep=1
  else if (atk=="Theft") keep=1
  else if (atk=="Reconnaissance") { if (rand()<0.05) keep=1 }
  else if (atk=="DoS") { if (rand()<0.02) keep=1 }
  else if (atk=="DDoS") { if (rand()<0.02) keep=1 }
  if (keep==1) print
}
' > "$OUT_DIR/bot_iot_sample_raw.csv"

echo "Done. Sample written to: $OUT_DIR/bot_iot_sample_raw.csv"
wc -l "$OUT_DIR/bot_iot_sample_raw.csv"
du -h "$OUT_DIR/bot_iot_sample_raw.csv"

echo ""
echo "Sanity check - attack class distribution in the FULL population (streamed count, not the sample):"
unzip -p "$ZIP_PATH" "*/data/NF-BoT-IoT-v2.csv" | awk -F',' 'NR==1{next} {atk[$NF]++} END{for (a in atk) print a, atk[a]}'
