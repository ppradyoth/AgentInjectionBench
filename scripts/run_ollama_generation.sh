#!/usr/bin/env bash
# Full generation pipeline using Ollama (local, free).
# Targets 2500+ curated samples from 120 seeds.
#
# Usage:
#   ./scripts/run_ollama_generation.sh              # uses qwen2.5:7b (recommended, ~4.4GB)
#   ./scripts/run_ollama_generation.sh gemma3:4b    # lighter model (~3GB)
#   ./scripts/run_ollama_generation.sh llama3.2:3b  # lightest (~2GB)

set -e

MODEL="${1:-qwen2.5:7b}"
VARIATIONS=30          # 120 seeds × 30 = 3600 raw → ~2500+ after curation
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "========================================"
echo " AgentInjectionBench — Ollama Generation"
echo " Model:      $MODEL"
echo " Variations: $VARIATIONS per seed"
echo " Expected:   ~$((120 * VARIATIONS)) raw samples"
echo "========================================"

# 1. Pull model if not already present
echo ""
echo "[1/4] Pulling model $MODEL..."
ollama pull "$MODEL"

# 2. Make sure Ollama is serving (start if not)
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
  echo "Starting Ollama server..."
  ollama serve &
  sleep 3
fi

# 3. Generate
echo ""
echo "[2/4] Generating variations..."
cd "$REPO_ROOT"
python3 -m generation.generate \
  --provider ollama \
  --model "$MODEL" \
  --variations "$VARIATIONS" \
  --temperature 0.85 \
  --output data/agent_injection_bench_raw.jsonl

RAW_COUNT=$(wc -l < data/agent_injection_bench_raw.jsonl | tr -d ' ')
echo "  → $RAW_COUNT raw samples generated"

# 4. Curate + split
echo ""
echo "[3/4] Curating and splitting..."
python3 -m generation.curate \
  --input data/agent_injection_bench_raw.jsonl \
  --output data/agent_injection_bench.jsonl \
  --split

FINAL_COUNT=$(wc -l < data/agent_injection_bench.jsonl | tr -d ' ')
echo "  → $FINAL_COUNT curated samples"

# 5. Stats
echo ""
echo "[4/4] Dataset statistics:"
python3 -m generation.stats --input data/agent_injection_bench.jsonl

echo ""
echo "========================================"
echo " Done! $FINAL_COUNT samples in data/agent_injection_bench.jsonl"
echo " Run: git add data/ && git commit -m 'Expand dataset to $FINAL_COUNT samples'"
echo "========================================"
