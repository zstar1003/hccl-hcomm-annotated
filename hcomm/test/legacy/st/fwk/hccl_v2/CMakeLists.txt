# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 自动找到项目内所有头文件
FILE(GLOB_RECURSE sources_list LIST_DIRECTORIES TRUE ${HCCL_V2_CODE_ROOT}/*)
foreach (dir ${sources_list})
    IF ((IS_DIRECTORY ${dir} ) AND (${dir} MATCHES "hccl/orion") AND (NOT ${dir} MATCHES "hccl/orion/mdpi"))
        target_include_directories(hccl_st_fwk PUBLIC ${dir})
    endif ()
endforeach ()

FILE(GLOB_RECURSE hccl_next_dir_list LIST_DIRECTORIES TRUE ${HCCL_NEXT_CODE_ROOT})
foreach (dir ${hccl_next_dir_list})
    IF ((IS_DIRECTORY ${dir}) AND (${dir} MATCHES "whole/hccl_next"))
        target_include_directories(hccl_st_fwk PUBLIC ${dir})
    endif ()
endforeach ()

# 添加项目外部头文件
target_include_directories(hccl_st_fwk PUBLIC
        ${TOP_DIR}/inc
        ${TOP_DIR}/inc/external
        ${TOP_DIR}/inc/aicpu
        ${TOP_DIR}/inc/driver
        ${TOP_DIR}/abl/qos
        ${TOP_DIR}/abl/msprof/inc
        ${TOP_DIR}/inc/aicpu/tsd
        ${TOP_DIR}/drivers/network/inc/network/hccp/rdma_agent/client
        ${TOP_DIR}/drivers/network/inc/network/
        ${TOP_DIR}/abl/libc_sec/include
        ${TOP_DIR}/abl/slog/inc/toolchain
        ${TOP_DIR}/runtime/include/external
        ${TOP_DIR}/runtime/pkg_inc/runtime
        ${TOP_DIR}/runtime/pkg_inc/
        ${TOP_DIR}/runtime/pkg_inc/profiling
        ${TOP_DIR}/ace/npuruntime/inc
        ${TOP_DIR}/ace/npuruntime/inc/runtime
        ${TOP_DIR}/ace/npuruntime/inc/external/runtime
        ${TOP_DIR}/ace/npuruntime/acl/inc/external/
        ${TOP_DIR}/ace/npuruntime/acl/inc/external/acl
        ${TOP_DIR}/drivers/network/inc/network/
        ${TOP_DIR}/drivers/network/inc/network/hccp/rdma_agent/client
        ${TOP_DIR}/open_source/json/include/
        ${TOP_DIR}/abl/atrace/inc/utrace
        ${TOP_DIR}/inc/common/util/error_manager/
        ${TOP_DIR}/metadef/inc/common/util/error_manager/
        ${TOP_DIR}/runtime/src/dfx/adump/inc/adump
        ${TOP_DIR}/metadef/inc/external
        ${TOP_DIR}/metadef/pkg_inc
        ${TOP_DIR}/ace/comop/inc/hccl/
)

FILE (GLOB_RECURSE cppfile
    ${HCCL_V2_CODE_ROOT}/*.cc
    ${HCCL_V2_CODE_ROOT}/*.cpp
    ${HCCL_NEXT_CODE_ROOT}/*.cc
    ${HCCL_NEXT_CODE_ROOT}/*.cpp
)
# set(exclude_dir ${HCCL_V2_CODE_ROOT}/framework/entrance/)
# LIST (FILTER cppfile EXCLUDE REGEX "^${exclude_dir}/.*")
FILE (GLOB_RECURSE excludecppfile
    ${HCCL_V2_CODE_ROOT}/framework/entrance/*.cc
    ${HCCL_V2_CODE_ROOT}/service/collective/coll_operator.cc
    ${HCCL_V2_CODE_ROOT}/service/collective/coll_operator_check.cc
    ${HCCL_V2_CODE_ROOT}/service/collective/alg/coll_alg_factory/alg_template/template_utils.cc
    ${HCCL_V2_CODE_ROOT}/service/collective/alg/coll_alg_factory/alg_executor/executor_utils.cc
    ${HCCL_V2_CODE_ROOT}/service/collective/alg/coll_alg_factory/utils/host/host_utils.cc
    ${HCCL_V2_CODE_ROOT}/service/collective/alg/coll_alg_factory/utils/device/device_utils.cc
    ${HCCL_NEXT_CODE_ROOT}/service/collective/alg/coll_alg_factory/utils/host/host_utils.cc
    ${HCCL_NEXT_CODE_ROOT}/framework/topo/virtual_topo_detect/*.cc

    ${HCCL_V2_CODE_ROOT}/mdpi/*.cc
)
LIST (REMOVE_ITEM cppfile ${excludecppfile})

target_sources(hccl_st_fwk PRIVATE
    ${cppfile}
)

# 编译时使用的宏定义
target_compile_definitions(hccl_st_fwk PRIVATE
        ORION_AICPU_DEPLOY
        ORION_HOST_DEPLOY
        CCL_FWK_LLT
)

# 编译选项
target_compile_options(hccl_st_fwk PRIVATE
        -std=c++17
        -Werror
        -fno-common
        -fno-strict-aliasing
        -pipe
        -O0
)

