# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set (include_list
    ${checker_include_list}
    ${HCOMM_CODE_LEGACY}/common
    ${HCOMM_DIR}/src/base_comm/common/debug

    ${UTILS_DIR}/adapter_v1/aiv

    ${legacy_service_include_list}

    ${legacy_common_include_list}

    ${legacy_framework_include_list}

    ${legacy_platform_include_list}

    ${HCOMM_ROOT_PATH}/src/legacy/ascend910/pub_inc
    ${HCOMM_DIR}/src/legacy/ascend910/pub_inc/new
    ${HCOMM_DIR}/src/legacy/ascend910/common/error_manager    
    ${HCOMM_PLATFORM_INCLUDE_LIST}

    ${ASCEND_CANN_INCLUDE_LIST}

    ${CMAKE_SOURCE_DIR}/test/legacy/depends/toolchain/
    ${CMAKE_BINARY_DIR}/proto/alg_analyzer_proto/proto
)

set (adapter_v2_src_list
    ${UTILS_DIR}/adapter_v2/coll_service_stub.cc
    ${UTILS_DIR}/adapter_v2/host_task_transform.cc
    ${UTILS_DIR}/adapter_v2/type_conversion.cc
    ${UTILS_DIR}/adapter_v2/sal_stub.cc
    ${UTILS_DIR}/adapter_v2/log_stub.cc
    ${UTILS_DIR}/adapter_v2/slog_stub.cc

    ${UTILS_DIR}/adapter_v2/task_ccu.cc
    ${UTILS_DIR}/adapter_v2/ccu_all_rank_param_recorder.cc
    # ${UTILS_DIR}/adapter_v2/ccu_loopGroup_param.cc
    ${UTILS_DIR}/adapter_v2/ccu_task_common.cc
    ${UTILS_DIR}/adapter_v2/ccu_task_transform.cc
    ${UTILS_DIR}/adapter_v2/task_graph_revamp_parallel.cc
    ${UTILS_DIR}/adapter_v2/task_graph_revamp_bilateral_ccu.cc

    ${UTILS_DIR}/adapter_v2/adapter_rts/adapter_rts_stub.cc
    ${UTILS_DIR}/adapter_v2/coll_gen_json.cc
    ${UTILS_DIR}/adapter_v2/sal_stub.cc
    ${UTILS_DIR}/adapter_v2/alg_adapt_v2_interface.cc
    ${UTILS_DIR}/adapter_v2/buffer/local_ub_rma_buffer_stub.cc
    ${UTILS_DIR}/adapter_v2/adapter_hccp/orion_adapter_hccp_stub.cc
    ${UTILS_DIR}/adapter_v2/transport/ccu_jetty_stub.cpp
    ${UTILS_DIR}/adapter_v2/transport/ccu_connection_stub.cpp
    ${UTILS_DIR}/adapter_v2/transport/ccu_transport_group_stub.cpp
    ${UTILS_DIR}/adapter_v2/transport/ccu_transport_stub.cpp
    ${UTILS_DIR}/adapter_v2/transport/ccu_jetty_mgr_stub.cpp
    ${UTILS_DIR}/adapter_v2/transport/ccu_transport_group_manager_stub.cpp
    ${UTILS_DIR}/adapter_v2/transport/ccu_transport_manager_stub.cpp
    ${UTILS_DIR}/adapter_v2/transport/transport_utils.cpp
    ${UTILS_DIR}/adapter_v2/common/rdma_handle_manager_stub.cc
    ${UTILS_DIR}/adapter_v2/common/tp_manager_stub.cc
    ${UTILS_DIR}/adapter_v2/communicator/hccl_communicator_stub.cc
    ${UTILS_DIR}/adapter_v2/communicator/communicator_impl_stub.cc
)

file(GLOB_RECURSE CCU_PLATFORM_SRC "${HCOMM_LEGACY_PLATFORM}/ccu/*.cpp")
file(GLOB_RECURSE ENV_CONFIG_SRC "${HCOMM_LEGACY_FRAMEWORK}/env_config/*.cc")

add_library(hccl_alg_adaption_v2 SHARED
    # 2.0 源码
    ${HCOMM_LEGACY_PLATFORM}/common/dev_capability.cc
    ${HCOMM_LEGACY_PLATFORM}/common/tokenInfo_manager.cc
    ${HCOMM_LEGACY_PLATFORM}/resource/buffer/buffer.cc
    ${HCOMM_LEGACY_PLATFORM}/resource/buffer/dev_buffer.cc
    ${HCOMM_LEGACY_PLATFORM}/common/hccp_tlv_hdc_manager.cc

    ${HCOMM_CODE_LEGACY}/common/utils/hash_utils.cc
    ${HCOMM_CODE_LEGACY}/common/log.cc
    ${HCOMM_CODE_LEGACY}/common/exception/hccl_exception.cc
    ${HCOMM_CODE_LEGACY}/common/exception/exception_definition.cc

    ${HCOMM_LEGACY_FRAMEWORK}/communicator/hccl_params_pub.cc

    ${HCOMM_LEGACY_FRAMEWORK}/ccu/ccu_ins/ccu_ins.cc
    ${HCOMM_LEGACY_FRAMEWORK}/aiv/aiv_ins/aiv_ins.cc
    ${HCOMM_LEGACY_FRAMEWORK}/ccu/ccu_ins/ccu_ins_group.cpp
    ${HCOMM_LEGACY_FRAMEWORK}/ccu/ccu_manager/ccu_ctx_creator_registry.cpp
    ${HCOMM_LEGACY_FRAMEWORK}/ccu/ccu_manager/ccu_respack_mgr.cpp
    ${HCOMM_LEGACY_FRAMEWORK}/ccu/ccu_communicator.cpp
    ${HCOMM_LEGACY_FRAMEWORK}/ccu/ccu_ins_preprocessor.cpp
    ${HCOMM_LEGACY_FRAMEWORK}/ccu/ccu_manager/ccu_registered_ctx_mgr.cpp

    ${HCOMM_LEGACY_FRAMEWORK}/resource_manager/transport/aicpu/connected_link_mgr.cc

    ${HCOMM_LEGACY_FRAMEWORK}/topo/virtual_topo.cc
    ${HCOMM_RANK_GRAPH}/common/conn_interface.cc
    ${HCOMM_RANK_GRAPH}/rank_table_info/rank_table_info.cc
    ${HCOMM_RANK_GRAPH}/rank_table_info/new_rank_info.cc
    ${HCOMM_RANK_GRAPH}/rank_table_info/rank_level_info.cc
    ${HCOMM_RANK_GRAPH}/rank_table_info/address_info.cc
    ${HCOMM_RANK_GRAPH}/rank_table_info/control_plane.cc
    ${HCOMM_RANK_GRAPH}/phy_topo/phy_topo.cc
    ${HCOMM_RANK_GRAPH}/phy_topo_builder/phy_topo_builder.cc
    ${HCOMM_RANK_GRAPH}/topo_info/topo_info.cc
    ${HCOMM_RANK_GRAPH}/topo_info/edge_info.cc
    ${HCOMM_RANK_GRAPH}/topo_info/peer_info.cc
    ${HCOMM_RANK_GRAPH}/rank_graph/rank_graph.cc
    ${HCOMM_RANK_GRAPH}/rank_graph/net_instance.cc
    ${HCOMM_RANK_GRAPH}/rank_graph_builder/rank_graph_builder.cc
    ${HCOMM_RANK_GRAPH}/rank_graph_builder/detour_service.cc
    ${HCOMM_RANK_GRAPH}/rank_graph_builder/detour_rules.cc
    ${HCOMM_RANK_GRAPH}/rank_graph_builder/updater_for_64_plus_1.cc
    ${HCOMM_LEGACY_FRAMEWORK}/env_config/env_config.cc
    ${HCOMM_LEGACY_FRAMEWORK}/env_config/env_func.cc
    ${HCOMM_LEGACY_FRAMEWORK}/env_config/base_config_legacy.cc
    ${HCOMM_DIR}/src/base_comm/common/debug/plf_debug_config.cc

    ${HCOMM_LEGACY_FRAMEWORK}/misc/json_parser/json_parser.cc

    ${adapter_v2_src_list}
    ${CCU_PLATFORM_SRC}
)

#头文件搜索路径
target_include_directories(hccl_alg_adaption_v2 PRIVATE
    ${include_list}
    ${HCOMM_DIR}/src/legacy/ascend910/common
    ${HCOMM_DIR}/src/legacy/ascend910/common/error_manager
    ${HCOMM_DIR}/src/legacy/ascend950
    ${HCOMM_DIR}/test/stub/depends/pkg_inc/base
)

target_compile_definitions(hccl_alg_adaption_v2 PRIVATE
    google=ascend_private
    -DOPEN_HCCL_TESTU
    -DHCCL_ALG_ANALYZER_DAVID
    _GLIBCXX_USE_CXX11_ABI=0
    CCL_FWK_LLT
)

target_compile_options(hccl_alg_adaption_v2 PRIVATE
    -U_FORTIFY_SOURCE
    -O0
    -pipe
)

target_link_libraries(hccl_alg_adaption_v2 PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    $<BUILD_INTERFACE:rdma_core_headers>
    hccl_orion_alg_test
    ${ASCEND_CANN_PACKAGE_PATH}/lib64/libc_sec.so
    $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address>
    -pthread
    ascend_protobuf
    -lrt
    -ldl
)
