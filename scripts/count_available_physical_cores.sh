#!/bin/bash
set -euo pipefail

allowed="$(awk '/^Cpus_allowed_list:/ { print $2 }' /proc/self/status)"
if [[ -z "$allowed" ]]; then
    nproc
    exit 0
fi

expand_cpu_list() {
    local part first last cpu
    IFS=',' read -ra parts <<< "$1"
    for part in "${parts[@]}"; do
        if [[ "$part" == *-* ]]; then
            first="${part%-*}"
            last="${part#*-}"
            for ((cpu = first; cpu <= last; ++cpu)); do
                echo "$cpu"
            done
        else
            echo "$part"
        fi
    done
}

while read -r cpu; do
    topology="/sys/devices/system/cpu/cpu${cpu}/topology"
    if [[ -r "$topology/core_id" && -r "$topology/physical_package_id" ]]; then
        printf '%s:%s\n' \
            "$(<"$topology/physical_package_id")" \
            "$(<"$topology/core_id")"
    else
        echo "cpu:$cpu"
    fi
done < <(expand_cpu_list "$allowed") | sort -u | wc -l
