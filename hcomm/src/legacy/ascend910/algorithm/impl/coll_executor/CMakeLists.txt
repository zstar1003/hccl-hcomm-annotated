# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_executor_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_native_executor_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_comm_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/alg_profiling.cc
)

if(TARGET hccl_alg)
    target_sources(hccl_alg PRIVATE
        ${src_list}
    )
endif()

if(TARGET ccl_kernel)
    target_sources(ccl_kernel PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/coll_executor_base.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/coll_native_executor_base.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/coll_comm_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/alg_profiling.cc
    )
    target_include_directories(ccl_kernel PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()

add_subdirectory(coll_all_gather)
add_subdirectory(coll_all_gather_v)
add_subdirectory(coll_all_reduce)
add_subdirectory(coll_reduce_scatter)
add_subdirectory(coll_reduce_scatter_v)
add_subdirectory(coll_reduce)
add_subdirectory(coll_send_receive)
add_subdirectory(coll_all_to_all)
add_subdirectory(coll_scatter)
add_subdirectory(coll_broadcast)
add_subdirectory(registry)
