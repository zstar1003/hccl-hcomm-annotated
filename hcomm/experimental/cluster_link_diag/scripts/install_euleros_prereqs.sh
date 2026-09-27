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

MIN_CMAKE_VERSION="3.16"
MIN_PYTHON_VERSION="3.9"
dry_run=0

usage() {
    cat <<'EOF'
Usage: scripts/install_euleros_prereqs.sh [--dry-run]

Checks the current EulerOS/openEuler machine and installs only missing system
packages required by disp_probe.
EOF
}

for arg in "$@"; do
    case "${arg}" in
        --dry-run)
            dry_run=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[system-prereq] unknown argument: ${arg}" >&2
            usage >&2
            exit 2
            ;;
    esac
done

log() {
    echo "[system-prereq] $*"
}

require_euler_host() {
    if [[ ! -r /etc/os-release ]]; then
        echo "[system-prereq] cannot detect OS: /etc/os-release is missing" >&2
        exit 1
    fi

    # shellcheck disable=SC1091
    . /etc/os-release
    case "${ID:-}:${PRETTY_NAME:-}" in
        euleros:*|hce:*|openEuler:*|openeuler:*|*:Euler*|*:openEuler*)
            ;;
        *)
            echo "[system-prereq] unsupported OS: ${PRETTY_NAME:-unknown}. This helper supports EulerOS/openEuler hosts only." >&2
            exit 1
            ;;
    esac

    if command -v dnf >/dev/null 2>&1; then
        PKG_MANAGER="dnf"
    elif command -v yum >/dev/null 2>&1; then
        PKG_MANAGER="yum"
    else
        echo "[system-prereq] required command is missing: yum or dnf" >&2
        exit 1
    fi

    if ! command -v rpm >/dev/null 2>&1; then
        echo "[system-prereq] required command is missing: rpm" >&2
        exit 1
    fi
}

version_ge() {
    local newer
    newer="$(printf '%s\n%s\n' "$2" "$1" | sort -V | tail -n 1)"
    [[ "${newer}" == "$1" ]]
}

python_version() {
    python3 - <<'PY' 2>/dev/null
import sys
print(".".join(str(x) for x in sys.version_info[:3]))
PY
}

cmake_version() {
    cmake --version 2>/dev/null | awk 'NR == 1 {print $3}'
}

package_installed() {
    rpm -q "$1" >/dev/null 2>&1
}

packages=()
declare -A package_reasons=()

add_package() {
    local package="$1"
    local reason="$2"

    if package_installed "${package}"; then
        log "skip installed package: ${package}"
        return
    fi

    for existing in "${packages[@]:-}"; do
        if [[ "${existing}" == "${package}" ]]; then
            return
        fi
    done

    packages+=("${package}")
    package_reasons["${package}"]="${reason}"
}

check_command_package() {
    local command_name="$1"
    local package="$2"
    local reason="$3"

    if command -v "${command_name}" >/dev/null 2>&1; then
        log "ok: ${command_name}"
    else
        add_package "${package}" "${reason}"
    fi
}

check_compiler_toolchain() {
    check_command_package gcc gcc "C compiler"
    check_command_package g++ gcc-c++ "C++ compiler"
    check_command_package make make "make build tool"
}

check_cmake() {
    if command -v cmake >/dev/null 2>&1; then
        local version
        version="$(cmake_version)"
        if version_ge "${version}" "${MIN_CMAKE_VERSION}"; then
            log "ok: cmake ${version}"
            return
        fi
        log "cmake ${version} is older than required ${MIN_CMAKE_VERSION}"
    fi
    add_package cmake "CMake >= ${MIN_CMAKE_VERSION}"
}

check_python3() {
    if command -v python3 >/dev/null 2>&1; then
        local version
        version="$(python_version)"
        if [[ -n "${version}" ]] && version_ge "${version}" "${MIN_PYTHON_VERSION}"; then
            log "ok: python3 ${version}"
        else
            log "python3 ${version:-unknown} is older than required ${MIN_PYTHON_VERSION}"
            add_package python3 "Python >= ${MIN_PYTHON_VERSION}"
        fi
    else
        add_package python3 "Python >= ${MIN_PYTHON_VERSION}"
    fi

    if python3 -m venv --help >/dev/null 2>&1; then
        log "ok: python3 venv"
    else
        add_package python3-virtualenv "Python venv support"
    fi

    if python3 -m pip --version >/dev/null 2>&1; then
        log "ok: python3 pip"
    else
        add_package python3-pip "Python pip"
    fi
}

verify_after_install() {
    local failed=0

    if ! command -v cmake >/dev/null 2>&1 || ! version_ge "$(cmake_version)" "${MIN_CMAKE_VERSION}"; then
        echo "[system-prereq] verify failed: cmake >= ${MIN_CMAKE_VERSION}" >&2
        failed=1
    fi
    if ! command -v python3 >/dev/null 2>&1 || ! version_ge "$(python_version)" "${MIN_PYTHON_VERSION}"; then
        echo "[system-prereq] verify failed: python3 >= ${MIN_PYTHON_VERSION}" >&2
        failed=1
    fi
    if ! python3 -m venv --help >/dev/null 2>&1; then
        echo "[system-prereq] verify failed: python3 venv module" >&2
        failed=1
    fi
    if ! python3 -m pip --version >/dev/null 2>&1; then
        echo "[system-prereq] verify failed: python3 pip" >&2
        failed=1
    fi
    if [[ "${failed}" -ne 0 ]]; then
        echo "[system-prereq] packages were installed, but this host still does not satisfy all requirements." >&2
        exit 1
    fi
}

require_euler_host

check_compiler_toolchain
check_cmake
check_command_package git git "git source fetch tool"
check_python3
check_command_package ssh openssh-clients "SSH client"
check_command_package rsync rsync "rsync file copy tool"

if [[ "${#packages[@]}" -eq 0 ]]; then
    log "all EulerOS system prerequisites are already satisfied"
    exit 0
fi

log "packages to install:"
for package in "${packages[@]}"; do
    log "  ${package}: ${package_reasons[${package}]}"
done

if [[ "${dry_run}" -eq 1 ]]; then
    log "dry run; no system packages were installed"
    exit 0
fi

if [[ "${EUID}" -ne 0 ]]; then
    echo "[system-prereq] installing system packages requires root. Re-run with sudo." >&2
    exit 1
fi

"${PKG_MANAGER}" install -y "${packages[@]}"
verify_after_install

log "EulerOS system prerequisites are ready"
