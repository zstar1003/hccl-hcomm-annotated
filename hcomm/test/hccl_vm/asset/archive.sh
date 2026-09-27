#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -e

MAX_ARCHIVE_COUNT=10

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

error_exit() {
    echo -e "${RED}[ERROR] $1${NC}" >&2
    exit 1
}

success_msg() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

info_msg() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

warn_msg() {
    echo -e "${YELLOW}[WARN] $1${NC}"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "$(basename "${SCRIPT_DIR}")" in
    bin|script)
        INSTALL_DIR="$(dirname "${SCRIPT_DIR}")"
        ;;
    *)
        INSTALL_DIR="${SCRIPT_DIR}"
        ;;
esac

ARCHIVE_DIR="${INSTALL_DIR}/archive"
LOGS_DIR="${INSTALL_DIR}/logs"
DATA_DIR="${INSTALL_DIR}/data"

mkdir -p "${ARCHIVE_DIR}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ARCHIVE_SUBDIR="${ARCHIVE_DIR}/${TIMESTAMP}"
mkdir -p "${ARCHIVE_SUBDIR}/logs"
mkdir -p "${ARCHIVE_SUBDIR}/data"

info_msg "Archive directory created: ${ARCHIVE_SUBDIR}"

ARCHIVE_DIRS=($(ls -1dt "${ARCHIVE_DIR}"/*/  2>/dev/null))
ARCHIVE_COUNT=${#ARCHIVE_DIRS[@]}

if [[ ${ARCHIVE_COUNT} -gt ${MAX_ARCHIVE_COUNT} ]]; then
    DELETE_COUNT=$((ARCHIVE_COUNT - MAX_ARCHIVE_COUNT))
    info_msg "Archive count (${ARCHIVE_COUNT}) exceeds limit (${MAX_ARCHIVE_COUNT}), removing ${DELETE_COUNT} oldest"
    for ((i=0; i<DELETE_COUNT; i++)); do
        OLDEST="${ARCHIVE_DIRS[$((ARCHIVE_COUNT - 1 - i))]}"
        warn_msg "Removing old archive: ${OLDEST}"
        rm -rf "${OLDEST}"
    done
fi

if [[ -d "${LOGS_DIR}" ]]; then
    cp -a "${LOGS_DIR}/." "${ARCHIVE_SUBDIR}/logs/"
 	info_msg "Copied logs to ${ARCHIVE_SUBDIR}/logs/"
else
    warn_msg "Logs directory not found: ${LOGS_DIR}"
fi

if [[ -d "${DATA_DIR}" ]]; then
    cp -a "${DATA_DIR}/." "${ARCHIVE_SUBDIR}/data/"
 	info_msg "Copied data to ${ARCHIVE_SUBDIR}/data/"
else
    warn_msg "Data directory not found: ${DATA_DIR}"
fi

success_msg "Archive completed: ${ARCHIVE_SUBDIR}"
