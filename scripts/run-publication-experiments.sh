#!/usr/bin/env bash
# Run all publication experiments for the paper.
# Expected total runtime: ~40-70 min depending on hardware.
#
# Prerequisites:
#   1. python3 -m venv .venv && source .venv/bin/activate
#   2. pip install -r requirements.txt
#   3. bash scripts/download-intel-lab.sh   (Intel Lab dataset)
#   4. Place Freiburg 079 dataset at data/freiburg-079/fr079-sm.log
#
# Results are written to results/ and then analyzed by:
#   python scripts/analyze-publication.py
#   python scripts/generate-paper-figures.py
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Publication Experiments ==="
echo ""

echo "--- H-002: Single-agent (15 runs, ~10 min) ---"
time python src/experiments/single_agent_tbm/run.py \
    --config configs/single-agent-tbm/publication.yaml
echo ""

echo "--- H-003 Condition 1: Dynamic baseline (15 runs, ~15 min) ---"
time python src/experiments/hybrid_pgo_ds/run.py \
    --config configs/hybrid-pgo-ds/pub-dynamic-baseline.yaml
echo ""

echo "--- H-003 Condition 2: Noisy sensor (15 runs, ~15 min) ---"
time python src/experiments/hybrid_pgo_ds/run.py \
    --config configs/hybrid-pgo-ds/pub-noisy-sensor.yaml
echo ""

echo "--- Mechanism test (boundary sharpness vs robot count, ~5 min) ---"
time python src/experiments/mechanism_test/run.py \
    --config configs/mechanism-test/h002-matched.yaml
echo ""

echo "--- Intel Lab validation (~2 min) ---"
time python src/experiments/intel_lab/run.py \
    --config configs/intel-lab/default.yaml
echo ""

echo "--- Freiburg 079 validation (~2 min) ---"
time python src/experiments/intel_lab/run.py \
    --config configs/freiburg-079/default.yaml
echo ""

echo "=== All experiments complete. Results written to results/ ==="
echo ""
echo "Next steps:"
echo "  python scripts/analyze-publication.py"
echo "  python scripts/generate-paper-figures.py"
echo "  python scripts/generate-ral-figures.py"
