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

export THIRDLIB_ROOT="${THIRDLIB_ROOT:-/usr/local/third_lib}"
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend}"
export ASCEND_CANN_PATH="${ASCEND_CANN_PATH:-${ASCEND_HOME_PATH}/ascend-toolkit/latest/aarch64-linux}"

_disp_probe_prepend_path() {
    case ":${!1:-}:" in
        *":$2:"*) ;;
        *) export "$1=$2${!1:+:${!1}}" ;;
    esac
}

_disp_probe_prepend_path PATH "${THIRDLIB_ROOT}/python/venv/bin"
_disp_probe_prepend_path CPATH "${THIRDLIB_ROOT}/include"
_disp_probe_prepend_path CPATH "${THIRDLIB_ROOT}/include/eigen3"
_disp_probe_prepend_path LIBRARY_PATH "${THIRDLIB_ROOT}/lib"
_disp_probe_prepend_path LD_LIBRARY_PATH "${THIRDLIB_ROOT}/lib"
_disp_probe_prepend_path LD_LIBRARY_PATH "${ASCEND_CANN_PATH}/lib64"
_disp_probe_prepend_path CMAKE_PREFIX_PATH "${THIRDLIB_ROOT}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/disp_probe_matplotlib}"
mkdir -p "${MPLCONFIGDIR}" 2>/dev/null || true

unset -f _disp_probe_prepend_path
