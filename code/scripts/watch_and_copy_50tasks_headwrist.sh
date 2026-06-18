#!/usr/bin/env bash
# Auto-copy new checkpoints from the all50 headwrist training run to the
# shared inference checkpoint directory. Polls every 60s.
#
#   src: /mnt/gyc_ckp/wjx/ctrlworld/ctrlworld_delta_ee_all50_8gpu_nf16_60k_headwrist_20260604_222617
#   dst: /mnt/public_ckp/cscsx_projects/ctrl_world_infer/checkpoints/50tasks_headwrist
#
# Robustness:
#   - skips a ckpt whose mtime is younger than STABLE_SECS (still being written;
#     an 8.6GB torch.save takes several seconds and we must not copy a partial file)
#   - copies to a hidden .partial temp then atomically renames into place, so any
#     reader downstream only ever sees a complete file.

set -uo pipefail

SRC="${SRC:-/mnt/gyc_ckp/wjx/ctrlworld/ctrlworld_delta_ee_all50_8gpu_nf16_60k_headwrist_20260604_222617}"
DST="${DST:-/mnt/public_ckp/cscsx_projects/ctrl_world_infer/checkpoints/50tasks_headwrist}"
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

        # already copied?
        [[ -f "${dst_file}" ]] && continue

        # skip if still being written (mtime too fresh)
        age=$(( $(date +%s) - $(stat -c %Y "${ckpt}") ))
        if (( age < STABLE_SECS )); then
            echo "[$(date '+%m-%d %H:%M:%S')] Skip ${fname} (age ${age}s < ${STABLE_SECS}s, still writing?)"
            continue
        fi

        # atomic copy: temp then rename (rename is atomic within DST filesystem)
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
