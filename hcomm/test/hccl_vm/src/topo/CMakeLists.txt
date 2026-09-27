# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_library(topo STATIC "")

target_sources(topo PRIVATE
    topo_cluster_ir.cc
    topo_cluster_parser.cc
    topo_cluster_dumper.cc
    topo_ascend_cluster_parser.cc
)

target_include_directories(topo PRIVATE
    ${THRID_PARTY_DIR}/nlohmann_json
    ${HCOMM_VM_ROOT_PATH}/include
    ${HCOMM_VM_ROOT_PATH}/include/runnerdb
    ${HCOMM_VM_ROOT_PATH}/include/topo
    ${HCOMM_VM_ROOT_PATH}/src/cmd
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/profiling
)

hcclvm_apply_common_flags(topo)

target_link_libraries(topo PRIVATE log)

# 日志染色: 模块 tag
target_compile_definitions(topo PRIVATE HCCL_VM_MODULE="TOPO")
