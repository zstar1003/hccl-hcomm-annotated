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

install_system_packages=0
dry_run=0
system_mode="auto"
third_party_args=()

while [[ "$#" -gt 0 ]]; do
    arg="$1"
    case "${arg}" in
        --install-system-packages)
            install_system_packages=1
            shift
            ;;
        --system)
            if [[ "$#" -lt 2 ]]; then
                echo "install.sh: --system requires one of: auto, ubuntu, euleros" >&2
                exit 2
            fi
            system_mode="$2"
            shift 2
            ;;
        --system=*)
            system_mode="${arg#--system=}"
            shift
            ;;
        --dry-run)
            dry_run=1
            third_party_args+=("${arg}")
            shift
            ;;
        -h|--help)
            cat <<'EOF'
Usage:
  ./install.sh [--install-system-packages] [--system auto|ubuntu|euleros] [third-party-options]

Examples:
  sudo ./install.sh --install-system-packages --system ubuntu
  sudo ./install.sh --install-system-packages --system euleros
  sudo ./install.sh --prefix /usr/local/third_lib

Notes:
  --dry-run prints planned actions without changing the system.
EOF
            exit 0
            ;;
        *)
            third_party_args+=("${arg}")
            shift
            ;;
    esac
done

if [[ "${install_system_packages}" -eq 1 ]]; then
    system_args=()
    system_args+=(--system "${system_mode}")
    if [[ "${dry_run}" -eq 1 ]]; then
        system_args+=(--dry-run)
    fi
    "${SCRIPT_DIR}/scripts/install_system_prereqs.sh" "${system_args[@]}"

    run_third_party=0
    for arg in "${third_party_args[@]}"; do
        if [[ "${arg}" != "--dry-run" ]]; then
            run_third_party=1
            break
        fi
    done
    if [[ "${run_third_party}" -eq 0 ]]; then
        exit 0
    fi
fi

exec "${SCRIPT_DIR}/scripts/setup_third_party.sh" "${third_party_args[@]}"
