# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_subdirectory(modeldb)

add_library(runnerdb STATIC)

target_sources(runnerdb PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tracedb/db_sim_runner_db.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/tracedb/db_sim_runner_common.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/tracedb/db_sim_sqlite_db.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/tracedb/db_sim_runner_ops.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/tracedb/db_sim_op_db_ops.cc
)

target_include_directories(runnerdb PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/profiling
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/runnerdb
    ${HCOMM_VM_ROOT_PATH}/include
    ${HCOMM_VM_ROOT_PATH}/include/runnerdb
    ${HCOMM_VM_ROOT_PATH}/src/store
    ${HCOMM_VM_ROOT_PATH}/src/runnerdb
    ${HCOMM_VM_ROOT_PATH}/src/runnerdb/modeldb
)

target_link_libraries(runnerdb PRIVATE log modeldb sqlite3)

hcclvm_apply_common_flags(runnerdb)

# 日志染色: 模块 tag
target_compile_definitions(runnerdb PRIVATE HCCL_VM_MODULE="RUNNERDB")
