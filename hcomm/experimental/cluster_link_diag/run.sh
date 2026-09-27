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

CONFIG="${CONFIG:-./control_json/910b2_info.json}"
RPC_HOST_BIN="${RPC_HOST_BIN:-./build/rpc_host}"
REMOTE_RPC_HOST="${REMOTE_RPC_HOST:-./bin/rpc_host}"
REMOTE_CONFIG="${REMOTE_CONFIG:-./control_json/910b2_info.json}"

usage() {
    echo "Usage: $0 {deploy [--pingpong-log [DIR]]|topo [--plot]|probe|controller|migrate-config [input] [output]}"
}

require_runtime_env() {
    local missing=()
    local var
    for var in THIRDLIB_ROOT ASCEND_HOME_PATH ASCEND_CANN_PATH; do
        if [[ -z "${!var:-}" ]]; then
            missing+=("${var}")
        fi
    done

    if (( ${#missing[@]} > 0 )); then
        echo "[run.sh][error] missing runtime env: ${missing[*]}" >&2
        echo '[run.sh][hint] please set the variables in README "环境与构建" and then run: source "$THIRDLIB_ROOT/share/disp_probe/third_party/env.sh"' >&2
        exit 2
    fi

    local env_script="${THIRDLIB_ROOT}/share/disp_probe/third_party/env.sh"
    if [[ ! -f "${env_script}" ]]; then
        echo "[run.sh][error] env script not found: ${env_script}" >&2
        echo "[run.sh][hint] please install third-party dependencies and source env.sh first" >&2
        exit 2
    fi

    if [[ ":${PATH}:" != *":${THIRDLIB_ROOT}/python/venv/bin:"* ]] \
        || [[ ":${LD_LIBRARY_PATH:-}:" != *":${THIRDLIB_ROOT}/lib:"* ]] \
        || [[ ":${LD_LIBRARY_PATH:-}:" != *":${ASCEND_CANN_PATH}/lib64:"* ]]; then
        echo "[run.sh][error] runtime env does not look sourced: PATH/LD_LIBRARY_PATH are incomplete" >&2
        echo "[run.sh][hint] run: source \"${env_script}\"" >&2
        exit 2
    fi
}

deploy() {
    local enable_pingpong_log=0
    local pingpong_log_dir="/root/output"
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --pingpong-log)
                enable_pingpong_log=1
                if [[ $# -gt 1 && "${2}" != --* ]]; then
                    pingpong_log_dir="$2"
                    shift
                fi
                ;;
            *)
                echo "[run.sh][error] unknown deploy option: $1" >&2
                usage >&2
                exit 2
                ;;
        esac
        shift
    done

    chmod 600 "${CONFIG}"
    python3 ./dispatcher/exec_realtime_cmd.py "pkill -f '${REMOTE_RPC_HOST}' || true"
    python3 ./dispatcher/disp_file_scp.py "${RPC_HOST_BIN}" "${REMOTE_RPC_HOST}"
    python3 ./dispatcher/disp_file_scp.py "${CONFIG}" "${REMOTE_CONFIG}"
    local remote_rpc_host_arg
    local remote_config_arg
    printf -v remote_rpc_host_arg "%q" "${REMOTE_RPC_HOST}"
    printf -v remote_config_arg "%q" "${REMOTE_CONFIG}"
    local remote_cmd="${remote_rpc_host_arg} -f ${remote_config_arg}"
    if [[ "${enable_pingpong_log}" == "1" ]]; then
        local pingpong_log_dir_arg
        printf -v pingpong_log_dir_arg "%q" "${pingpong_log_dir}"
        remote_cmd+=" --pingpong-local-log --pingpong-log-dir ${pingpong_log_dir_arg}"
    fi
    python3 ./dispatcher/exec_realtime_cmd.py -l "${remote_cmd}"
}

topo() {
    require_runtime_env
    ./build/probe_topo -f "${CONFIG}"
    if [[ "${PLOT_TOPO:-0}" == "1" || "${1:-}" == "--plot" ]]; then
        python3 ./plot/topo_plot.py
    fi
}

controller() {
    require_runtime_env
    ./build/probe_controller -f "${CONFIG}"
}

migrate_config() {
    if [[ $# -ge 2 ]]; then
        python3 ./scripts/migrate_config.py "$1" "$2"
    else
        python3 ./scripts/migrate_config.py "${1:-${CONFIG}}"
    fi
}

case "${1:-}" in
    deploy)
        shift
        deploy "$@"
        ;;
    probe)
        controller
        ;;
    topo)
        shift
        topo "$@"
        ;;
    controller)
        controller
        ;;
    migrate-config)
        shift
        migrate_config "$@"
        ;;
    *)
        usage
        exit 2
        ;;
esac
