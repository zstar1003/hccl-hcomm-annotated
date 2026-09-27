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

set +e

echo "start run test case, please wait ..."
cd ${WORKSPACE}

log() {
  local dt
  dt=$(date '+%Y%m%d.%H%M%S')
  echo "===================================================================="
  echo "$dt : $*"
  echo "===================================================================="
}

log "init test case, please wait ..."
rm -rf /root/ascend/log

# ==============================
# 确定要测试的 ops 列表
# ==============================
declare -a ops
ops=("is_finite")
echo $ops

# ==============================
# 运行测试主循环
# ==============================
log "start run test case, please wait ..."

export ASCEND_GLOBAL_LOG_LEVEL=2
export ASCEND_SLOG_PRINT_TO_STDOUT=0

for op in "${ops[@]}"; do
  echo "Processing: $op"
  mode="eager"
  [ "$op" = "crop_and_resize" ] && mode="graph"
  source /usr/local/Ascend/cann/set_env.sh
  arm_package="cann-hcomm_linux-aarch64.run"
  RUN_PACKAGE_URL="https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/${obs_path}/${arm_package}"
  RUN_PACKAGE_NAME="cann-hccl-hcomm-aarch64.run"
  if ! wget -q -O "${RUN_PACKAGE_NAME}" "${RUN_PACKAGE_URL}"; then
    echo "Package download failed!URL:${RUN_PACKAGE_URL}"
  else
    echo "Package downloaded successfully: ./${RUN_PACKAGE_NAME}"
  fi
  # Add execute permission to the downloaded package
  echo "Adding execute permission: chmod +x ${RUN_PACKAGE_NAME}"
  chmod +x "${RUN_PACKAGE_NAME}" || echo "Failed to add execute permission to the package"
  yes "y" | bash "${RUN_PACKAGE_NAME}" --full --install-path=/usr/local/Ascend --quiet
  source /usr/local/Ascend/cann/set_env.sh
  bash build.sh --cb_test_verify 2>&1 | tee -a ./run_test.log
done

# ==============================
# 打包log
# ==============================
mkdir -p /root/ascend
slog_name="slog.tar.gz"
tar -zcf "${slog_name}" -C /root/ascend log

# ==============================
# 检查 NPU 状态
# ==============================
log "checking NPU status ..."
mkdir -p ./npu_log
npu-smi info  2>&1 | tee ./npu_log/npu_info.log

# ==============================
# 检查测试结果
# ==============================
log "checking test results ..."

date_time=`date +%Y%m%d`"."`date +%H%M%S`
if grep -E '\b(FAIL|errors|fail|failed|error|ERROR:|Error|error:)\b' "./run_test.log" | grep -v "error)"; then
    echo "$date_time : run test case failed"
    exit 1
else
    echo "$date_time : run test case success"
    exit 0
fi
