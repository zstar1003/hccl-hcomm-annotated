# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

target_include_directories(hccl_v2_utest PRIVATE
        ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/common/src/topo
        ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/common/src
        ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/common/src/config
        ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/common/src/mgr
        ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/inc
        ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/communicator/impl
        ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/op_base/src
        ${HCCL_CODE_ROOT}/src/legacy/ascend910/pub_inc
        ${HCCL_CODE_ROOT}/src/legacy/ascend910/pub_inc/inner
        ${HCCL_CODE_ROOT}/src/legacy/ascend910/pub_inc/new
        ${HCCL_CODE_ROOT}/src/legacy/ascend910/common
        ${HCCL_CODE_ROOT}/src/legacy/ascend910/common/error_manager
        ${HCCL_CODE_ROOT}/src/legacy/ascend910/common/debug/profiling/inc
        ${HCCL_CODE_ROOT}/src/legacy/ascend910/common/debug/profiling/inc/host
        ${HCCL_CODE_ROOT}/src/legacy/ascend910/common/debug/config
        ${HCCL_CODE_ROOT}/src/legacy/ascend910/common/stream
        ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/pub_inc
        ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/base/inc
        ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/base/alg_template
        ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/base/communicator
        ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/base/communicator/legacy
        ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl
        ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/resource_manager
        ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/task
        ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/legacy
        ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/coll_executor
        ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/inc
        ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/inc/adapter
        ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/common
        ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/resource/transport
        ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/resource/notify
        ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/resource/dispatcher_ctx
        ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/resource/socket
        ${HCCL_CODE_ROOT}/pkg_inc/hcomm/ccu
)

target_sources(hccl_v2_utest
        PUBLIC
        rank_table.cc
        ut_topoinfo_ranktable_partition.cc

)

target_sources(hccl_v2_utest PRIVATE
        ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/common/src/topo/topoinfo_ranktable_partition.cc
)
