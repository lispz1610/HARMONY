#!/usr/bin/env bash
# HARMONY Multi-Rank Hardware Telemetry Probe
# Queries local GPU telemetry on compute nodes and prefixes output with host and rank.

set -uo pipefail

host="$(hostname -s 2>/dev/null || hostname)"
rank="${OMPI_COMM_WORLD_RANK:-${PMIX_RANK:-${MPI_LOCALRANKID:-${PBS_VNODENUM:-0}}}}"
now="$(date -u +"%Y-%m-%d %H:%M:%S" 2>/dev/null || date +"%Y-%m-%d %H:%M:%S")"

if command -v nvidia-smi >/dev/null 2>&1; then
    smi_output="$(nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,clocks.current.graphics,clocks.current.memory,power.draw,clocks_throttle_reasons.active --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ -n "$smi_output" ]]; then
        while IFS= read -r line; do
            [[ -n "$line" ]] && echo "${host}, ${rank}, ${line}"
        done <<< "$smi_output"
    else
        echo "${host}, ${rank}, ${now}, placeholder, N/A, N/A, N/A, N/A, N/A, [N/A] (nvidia-smi query failed)"
    fi
else
    echo "${host}, ${rank}, ${now}, placeholder, N/A, N/A, N/A, N/A, N/A, [N/A] (nvidia-smi not available)"
fi
