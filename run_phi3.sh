#!/usr/bin/env bash
# Full Phi3V-Med Table 1 pipeline for one dataset.
#   bash run_phi3.sh vqa_rad            (vqa_rad | slake | pathvqa)
# Env overrides: DATA (prepared data root), WORK (outputs), PREC (fp16|bf16|4bit), SKIP_TRAIN=1, LIMIT=N
set -euo pipefail
DS=${1:?dataset: vqa_rad|slake|pathvqa}
DATA=${DATA:-data}
WORK=${WORK:-work}
PREC=${PREC:-fp16}
LIM=${LIMIT:+--limit $LIMIT}
D=$DATA/$DS
O=$WORK/$DS
mkdir -p "$O"

# 0) data (once, all three datasets: KB uses every train split)
[ -f "$DATA/data_paths.json" ] || python -m phi3cfg.download_data --output "$DATA"

# 1) downstream fine-tuning of Phi3V-Med (Sec. 4.3; Table 1 "Phi3V-Med 4.2B" row)
ADAPTER=$WORK/lora_$DS
if [ "${SKIP_TRAIN:-0}" = "1" ]; then ADAPTER=ecoxial2007/Phi-3V-Med
elif [ ! -f "$ADAPTER/adapter_model.safetensors" ]; then
  python -m phi3cfg.train --dataset-dir "$D" --output "$ADAPTER" --precision "$PREC"
fi
COMMON="--dataset-dir $D --adapter $ADAPTER --precision $PREC $LIM"

# 2) baseline + entropy (Eq. 4)
python -m phi3cfg.infer $COMMON --mode baseline --output "$O/baseline.jsonl"

# 3) knowledge base (once) + retrieval + simulated expert annotations
[ -f "$WORK/kb/kb_features.pt" ] || python -m phi3cfg.retrieval build --data-root "$DATA" --output "$WORK/kb"
python -m phi3cfg.retrieval search --kb "$WORK/kb" --dataset-dir "$D" --output "$O/annotations.jsonl"

# 4) Table 1 variants
A="--annotations $O/annotations.jsonl"
S="--select-from $O/baseline.jsonl --top-percent 5"
python -m phi3cfg.infer $COMMON $A --mode rag                   --output "$O/rag.jsonl"
python -m phi3cfg.infer $COMMON $A $S --mode rag                --output "$O/rag5.jsonl"
python -m phi3cfg.infer $COMMON $A $S --mode expert_rag         --output "$O/expert_rag5.jsonl"
python -m phi3cfg.infer $COMMON $A $S --mode expert_cfg         --output "$O/expert_cfg5.jsonl"

# 5) report
python -m phi3cfg.evaluate --dataset-dir "$D" --baseline "$O/baseline.jsonl" \
  --runs rag="$O/rag.jsonl" rag5="$O/rag5.jsonl" expert_rag5="$O/expert_rag5.jsonl" expert_cfg5="$O/expert_cfg5.jsonl" \
  --output "$O/report.json" ${LIMIT:+--partial}
