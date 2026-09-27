# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_library(modeldb STATIC "")

target_sources(modeldb PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/db_hccl_db_sqlite.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/db_hccl_db_factory.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/db_hccl_op_db_ops.cc
)

target_include_directories(modeldb PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/include/runnerdb
)

target_link_libraries(modeldb
    PRIVATE log
    PUBLIC sqlite3
)

hcclvm_apply_common_flags(modeldb)

# 日志染色: 模块 tag
target_compile_definitions(modeldb PRIVATE HCCL_VM_MODULE="RUNNERDB")
