#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
system_mode="auto"
args=()

usage() {
    cat <<'EOF'
Usage: scripts/install_system_prereqs.sh [--system auto|ubuntu|euleros] [--dry-run]

Dispatches to the Ubuntu/Debian apt helper or the EulerOS yum/dnf helper.
EOF
}

detect_system() {
    if [[ ! -r /etc/os-release ]]; then
        echo "unknown"
        return
    fi

    # shellcheck disable=SC1091
    . /etc/os-release
    case "${ID:-}" in
        ubuntu|debian)
            echo "ubuntu"
            ;;
        euleros|hce|openEuler|openeuler)
            echo "euleros"
            ;;
        *)
            case "${PRETTY_NAME:-}" in
                *Euler*|*openEuler*)
                    echo "euleros"
                    ;;
                *Ubuntu*|*Debian*)
                    echo "ubuntu"
                    ;;
                *)
                    echo "unknown"
                    ;;
            esac
            ;;
    esac
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --system)
            if [[ "$#" -lt 2 ]]; then
                echo "[system-prereq] --system requires one of: auto, ubuntu, euleros" >&2
                exit 2
            fi
            system_mode="$2"
            shift 2
            ;;
        --system=*)
            system_mode="${1#--system=}"
            shift
            ;;
        --dry-run)
            args+=("$1")
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[system-prereq] unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "${system_mode}" == "auto" ]]; then
    system_mode="$(detect_system)"
fi

case "${system_mode}" in
    ubuntu)
        exec "${SCRIPT_DIR}/install_ubuntu_prereqs.sh" "${args[@]}"
        ;;
    euleros)
        exec "${SCRIPT_DIR}/install_euleros_prereqs.sh" "${args[@]}"
        ;;
    *)
        echo "[system-prereq] unsupported system mode: ${system_mode}" >&2
        echo "[system-prereq] use --system ubuntu or --system euleros explicitly if auto detection is wrong" >&2
        exit 1
        ;;
esac
