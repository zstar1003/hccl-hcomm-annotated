# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(rootinfo_detect_sources
    rank_info_detect_service.cc
    rank_info_detect_client.cc
    rank_info_detect.cc
    bootstrap_ip.cc
    rank_info_dispatcher.cc
    socket_agent.cc
    whitelist.cc
)

set(rootinfo_detect_include_dirs
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HCOMM_DIR}/src/coll_communicator_mgr/rank_graph
    ${HCOMM_DIR}/src/coll_communicator_mgr/rank_graph/common
    ${HCOMM_DIR}/src/coll_communicator_mgr/rank_graph/phy_topo
    ${HCOMM_DIR}/src/coll_communicator_mgr/rank_graph/phy_topo_builder
    ${HCOMM_DIR}/src/coll_communicator_mgr/rank_graph/rank_graph
    ${HCOMM_DIR}/src/coll_communicator_mgr/rank_graph/rank_graph_builder
    ${HCOMM_DIR}/src/coll_communicator_mgr/rank_graph/rank_table_info
    ${HCOMM_DIR}/src/coll_communicator_mgr/rank_graph/topo_info
    ${HCOMM_DIR}/src/legacy/ascend950/framework/entrance
    ${HCOMM_DIR}/src/legacy/ascend950/framework/topo/rank_info_detect
    ${HCOMM_DIR}/src/legacy/ascend950/framework/topo/topo_addr_info/include
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/external_system
)

if(TARGET hcomm)
    target_sources(hcomm PRIVATE
        ${rootinfo_detect_sources}
        root_info_detect_provider.cc
    )
    target_include_directories(hcomm PRIVATE ${rootinfo_detect_include_dirs})
endif()

# The bridge slot and shared port-preemption singleton are called by legacy code and must be owned by hccl_v2.
if(TARGET hccl_v2)
    target_sources(hccl_v2 PRIVATE
        ${HCOMM_DIR}/src/legacy/ascend950/framework/topo/rank_info_detect/preempt_port_manager.cc
        root_info_detect_bridge.cc
    )
    target_include_directories(hccl_v2 PRIVATE ${rootinfo_detect_include_dirs})
endif()
