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
export ASCEND_HOME_PATH=/home/jenkins/Ascend/cann
source /home/jenkins/Ascend/cann/bin/setenv.bash

LOG_HEAD "Start run c++ testcase"
if [ "${GIT_TARGET_BRANCH}" == "competition/campus-2026" ]; then
    timeout 30m bash build.sh --ut --cov -j32
else
    bash build.sh --ut --cov -j32
fi
DP_ASSERT_EQUAL "$?" "0" "Run UT TESTCASE"

if [ "${GIT_TARGET_BRANCH}" == "master" ]; then
    echo "ut_process=ut_cov" >> "${ATOMGIT_OUTPUT}"
else
    echo "ut_process=continue" >> "${ATOMGIT_OUTPUT}"
fi