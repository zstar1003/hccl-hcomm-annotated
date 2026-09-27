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

# hccl_config.sh - HCCL 环境变量配置

remove_files_by_prefix() {
  if [ "$#" -ne 1 ]; then
    echo "Usage: remove_files_by_prefix <prefix>" >&2
    return 2
  fi

  local prefix="$1"
  if [ -z "$prefix" ]; then
    return 0
  fi

  shopt -s nullglob
  local any_deleted=0
  for f in "${prefix}"*; do
    if [ -f "$f" ]; then
      rm -f -- "$f" && any_deleted=1
    fi
  done
  shopt -u nullglob

  # 无论是否删除了文件，均返回 0，确保脚本继续执行
  return 0
}

# 清理执行目录的冗余文件
remove_files_by_prefix "sqe_info_rank_"
remove_files_by_prefix "mc_instr_info_rank_"
rm -f "all_rank_input_output.txt"

# 设置CANN环境变量
source /home/teamserver/workspace/Ascend/cann/set_env.sh

# 设置 HCCL-VM 安装路径，基于脚本自身位置推断（兼容 bin/ 与 script/ 子目录）
_INSTALL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "$(basename "${_INSTALL_SCRIPT_DIR}")" in
    bin|script)
        export HCCL_VM_INSTALL_DIR="$(dirname "${_INSTALL_SCRIPT_DIR}")"
        ;;
    *)
        export HCCL_VM_INSTALL_DIR="${_INSTALL_SCRIPT_DIR}"
        ;;
esac
unset _INSTALL_SCRIPT_DIR

# 设置 ranktable.json文件路径 (与 mock-comm 生成路径保持一致)
export RANK_TABLE_FILE=${HCCL_VM_INSTALL_DIR}/data/ranktable.json

# 设置日志级别
export ASCEND_GLOBAL_LOG_LEVEL=1

# 设置日志打屏输出
export ASCEND_SLOG_PRINT_TO_STDOUT=1

# 设置 HCCL 运行模式（CCU、AI_CPU、AIV等展开模式）
export HCCL_OP_EXPANSION_MODE="CCU_SCHED"
# 或设置 HCCL 运行参数(AI_CPU展开模式) AI_CPU模式环境变量与其他模式不可同时设置
# export HCCL_OP_EXPANSION_MODE="AI_CPU"
# 或设置 HCCL 运行参数(AIV展开模式) AIV模式环境变量与其他模式不可同时设置
# export HCCL_OP_EXPANSION_MODE="AIV"

echo "HCCL-VM environment configured successfully!"
echo "HCCL_VM_INSTALL_DIR: ${HCCL_VM_INSTALL_DIR}"
