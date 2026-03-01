#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="/home/s447658/project/fiper/data"

WM_ROOT="/home/s447658/projects/dreamer_fiper_offline/wm_all5_seeds"
FEATS_ROOT="/home/s447658/projects/dreamer_fiper_feats_all5/feats_all5_seeds"
OUT_ROOT="/home/s447658/projects/dreamer_fiper_results"

TASKS=("pretzel" "push_chair" "push_t" "sorting" "stacking")
SEEDS=(0 1 2 3 4)

mkdir -p "$OUT_ROOT"

# 1) Train WMs (seeded)
for s in "${SEEDS[@]}"; do
  python train_wm_offline_success_rgb_all5_tasks_seeded.py \
    --data_root "$DATA_ROOT" \
    --out_root "$WM_ROOT" \
    --tasks "${TASKS[@]}" \
    --seed "$s" \
    --train_steps 20000 --seq_len 16 --batch_size 4
done

# 2) Extract feats (seeded)
for s in "${SEEDS[@]}"; do
  python extract_wm_feats_all5_tasks_seeded.py \
    --data_root "$DATA_ROOT" \
    --wm_root "$WM_ROOT" \
    --out_root "$FEATS_ROOT" \
    --tasks "${TASKS[@]}" \
    --seed "$s"
done

# 3) Score with v3 + aggregate across seeds
python wm_progress_monitor_per_task_threshold_episode_calib_v3_seeds.py \
  --data_root "$FEATS_ROOT" \
  --out_root "$OUT_ROOT" \
  --tasks "${TASKS[@]}" \
  --seeds "${SEEDS[@]}" \
  --calib_success_source calib_only \
  --bins auto --bins_min 4 --bins_max 10 --min_pts_per_bin 20 \
  --alpha 0.10 --var_floor 1e-4 \
  --window_mode adaptive --window_frac 0.2 --min_window 5 \
  --score_agg topk_mean --topk 7 \
  --persist_mode adaptive --persist_frac 0.25 --persist_max 8 \
  --calib_score_mode topk_mean \
  --debug 1 \
  --out_json wm_progress_monitor_episode_calib_v3_seeds.json

echo "DONE. Results in: $OUT_ROOT/wm_progress_monitor_episode_calib_v3_seeds.json"