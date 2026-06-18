#!/usr/bin/env bash
# Auto-copy new checkpoints from the S1-A (expert-only) headwrist run to the
# shared inference checkpoint directory. Polls every 60s.
#
#   src: /mnt/gyc_ckp/wjx/ctrlworld/s1_A_expert_only_headwrist_20260605_004728
#   dst: /mnt/public_ckp/cscsx_projects/ctrl_world_infer/checkpoints/5tasks_experts_only_epoch
#
# Robustness: skips ckpts younger than STABLE_SECS (still being written), and
# copies to a hidden .partial temp then atomically renames into place.

set -uo pipefail

SRC="${SRC:-/mnt/gyc_ckp/wjx/ctrlworld/s1_A_expert_only_headwrist_20260605_004728}"
DST="${DST:-/mnt/public_ckp/cscsx_projects/ctrl_world_infer/checkpoints/5tasks_experts_only_epoch}"
POLL_SECS="${POLL_SECS:-60}"
STABLE_SECS="${STABLE_SECS:-120}"

mkdir -p "${DST}"

echo "[INFO] Watching for new checkpoints (poll ${POLL_SECS}s, stable ${STABLE_SECS}s)..."
echo "[INFO] SRC: ${SRC}"
echo "[INFO] DST: ${DST}"

while true; do
    shopt -s nullglob
    for ckpt in "${SRC}"/checkpoint-*.pt; do
        fname="$(basename "${ckpt}")"
        dst_file="${DST}/${fname}"
        [[ -f "${dst_file}" ]] && continue

        age=$(( $(date +%s) - $(stat -c %Y "${ckpt}") ))
        if (( age < STABLE_SECS )); then
            echo "[$(date '+%m-%d %H:%M:%S')] Skip ${fname} (age ${age}s < ${STABLE_SECS}s, still writing?)"
            continue
        fi

        tmp="${DST}/.${fname}.partial"
        echo "[$(date '+%m-%d %H:%M:%S')] Copying ${fname} ($(du -h "${ckpt}" | cut -f1))..."
        if cp -f "${ckpt}" "${tmp}" && mv -f "${tmp}" "${dst_file}"; then
            echo "[$(date '+%m-%d %H:%M:%S')] Done -> ${dst_file}"
        else
            echo "[$(date '+%m-%d %H:%M:%S')] FAILED copying ${fname}, cleaning temp"
            rm -f "${tmp}"
        fi
    done
    shopt -u nullglob
    sleep "${POLL_SECS}"
done
