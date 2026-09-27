# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

get_filename_component(ALG_ANALYZER_UI_PROTO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ui/proto ABSOLUTE)

add_custom_target(alg_analyzer_proto
    DEPENDS host_protoc # 依赖 host_protoc
)
add_custom_command(TARGET alg_analyzer_proto
    COMMAND mkdir -p ${CMAKE_BINARY_DIR}/proto/alg_analyzer_proto/proto
    COMMAND cd ${ALG_ANALYZER_UI_PROTO_DIR} && $<TARGET_FILE:host_protoc> --cpp_out=${CMAKE_BINARY_DIR}/proto/alg_analyzer_proto/proto analysis_result.proto
)

add_dependencies(hccl_alg_analyzer alg_analyzer_proto)

target_include_directories(hccl_alg_analyzer PRIVATE
    ${UTILS_PUBLIC_INCLUDE_LIST}
    ${UTILS_INNER_INCLUDE_LIST}

    ${HCOMM_PUBLIC_INCLUDE_LIST}
    ${HCOMM_PLATFORM_INCLUDE_LIST}
    ${HCOMM_ALGORITHM_INCLUDE_LIST}
    ${HCOMM_COMMON_INCLUDE_LIST}
    ${HCOMM_FRAMEWORK_INCLUDE_LIST}
    ${HCOMM_LEGACY_INCLUDE_LIST}

    ${CMAKE_BINARY_DIR}/proto/alg_analyzer_proto/proto

    ${ASCEND_CANN_INCLUDE_LIST}
)

target_sources(hccl_alg_analyzer
        PRIVATE
        check_utils.cc
        checker.cc
        task_stub.cc
        task_graph_generator.cc
        topo_meta.cc
        ${CMAKE_BINARY_DIR}/proto/alg_analyzer_proto/proto/analysis_result.pb.cc
        aiv_task_stub.cc
)

add_subdirectory(task_collector)
add_subdirectory(rank_info_recorder)
add_subdirectory(mem_layout)
add_subdirectory(link_type_recorder)
add_subdirectory(semantics_check)
add_subdirectory(ui)
add_subdirectory(param_trans)
add_subdirectory(singletask_check)
add_subdirectory(mem_conflict_check)