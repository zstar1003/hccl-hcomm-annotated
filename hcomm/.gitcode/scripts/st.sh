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

REPOSITORY_NAME="hcomm"
sudo update-alternatives --set gcc /usr/bin/gcc-14
export PATH=/opt/buildtools/python-3.10.2/bin:$PATH
gcc --version

if [ -z "${ASCEND_3RD_LIB_PATH}" ]; then
    export ASCEND_3RD_LIB_PATH=/home/jenkins/opensource
fi

LOG_HEAD()
{
    echo "========================================"
    echo "  $1"
    echo "========================================"
}

LOG_DO()
{
    echo "[LOG_DO] $*"
    "$@"
}

DP_ASSERT_EQUAL()
{
    local actual="$1"
    local expected="$2"
    local msg="$3"
    if [ "${actual}" != "${expected}" ]; then
        echo "::error::ASSERT FAILED: ${msg} (expected=${expected}, actual=${actual})"
        exit 1
    fi
}

cd "${WORKSPACE}/" || exit 1
wget -nv https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/${obs_path}/cann-hcomm_linux-x86_64.run
chmod u+x cann-hcomm_linux-x86_64.run
sudo chmod 777 /home/jenkins/Ascend
yes "y" | bash cann-hcomm_linux-x86_64.run --full --install-path=/home/jenkins/Ascend
install_exit_code=$?
DP_ASSERT_EQUAL "$install_exit_code" "0" "bash hcomm.run"

export ASCEND_HOME_PATH=/home/jenkins/Ascend/cann
source /home/jenkins/Ascend/cann/bin/setenv.bash

LOG_HEAD "Start run c++ testcase"
bash build.sh --st -j32
DP_ASSERT_EQUAL "$?" "0" "Run ST TESTCASE"