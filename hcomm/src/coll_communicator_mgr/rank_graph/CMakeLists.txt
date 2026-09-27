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
    ${CMAKE_CURRENT_SOURCE_DIR}/rank_graph.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/rank_graph_v2.cc
)

set(v2_rank_graph_include_dirs
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/phy_topo
    ${CMAKE_CURRENT_SOURCE_DIR}/phy_topo_builder
    ${CMAKE_CURRENT_SOURCE_DIR}/rank_graph_builder
    ${CMAKE_CURRENT_SOURCE_DIR}/rank_table_info
    ${CMAKE_CURRENT_SOURCE_DIR}/topo_info
    ${HCOMM_DIR}/src/legacy/ascend950/common/types
    ${HCOMM_DIR}/src/legacy/ascend950/common/utils
    ${HCOMM_DIR}/src/legacy/ascend950/framework/topo/topo_addr_info/include
)

if(TARGET hcomm)
    target_sources(hcomm PRIVATE
        ${src_list}
    )
    target_include_directories(hcomm BEFORE PRIVATE ${v2_rank_graph_include_dirs})
endif()

if(TARGET ccl_kernel)
    target_include_directories(ccl_kernel PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()

if(TARGET hccl_v2)
    target_include_directories(hccl_v2 BEFORE PRIVATE ${v2_rank_graph_include_dirs})
endif()

add_subdirectory(common)
add_subdirectory(phy_topo)
add_subdirectory(phy_topo_builder)
add_subdirectory(rank_graph)
add_subdirectory(rank_graph_builder)
add_subdirectory(rank_table_info)
add_subdirectory(topo_info)
