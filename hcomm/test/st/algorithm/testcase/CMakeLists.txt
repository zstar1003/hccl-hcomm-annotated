# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB_RECURSE ALL_REDUCE_EXECUTOR "${HCOMM_CODE_ALGORITHM}/impl/coll_executor/coll_all_reduce/*.cc")
file(GLOB_RECURSE ALL_REDUCE_EXECUTOR_310P "${HCOMM_CODE_ALGORITHM}/impl/coll_executor/coll_all_reduce/310P/*.cc")

file(GLOB_RECURSE ALL_GATHER_EXECUTOR "${HCOMM_CODE_ALGORITHM}/impl/coll_executor/coll_all_gather/*.cc")
file(GLOB_RECURSE ALL_GATHER_EXECUTOR_310P "${HCOMM_CODE_ALGORITHM}/impl/coll_executor/coll_all_gather/310P/*.cc")

file(GLOB_RECURSE ALL_GATHER_V_EXECUTOR "${HCOMM_CODE_ALGORITHM}/impl/coll_executor/coll_all_gather_v/*.cc")

file(GLOB_RECURSE REDUCE_SCATTER_EXECUTOR "${HCOMM_CODE_ALGORITHM}/impl/coll_executor/coll_reduce_scatter/*.cc")
file(GLOB_RECURSE REDUCE_SCATTER_EXECUTOR_310P "${HCOMM_CODE_ALGORITHM}/impl/coll_executor/coll_reduce_scatter/310P/*.cc")

file(GLOB_RECURSE REDUCE_SCATTER_V_EXECUTOR "${HCOMM_CODE_ALGORITHM}/impl/coll_executor/coll_reduce_scatter_v/*.cc")
file(GLOB_RECURSE REDUCE_SCATTER_EXECUTOR_310P "${HCOMM_CODE_ALGORITHM}/impl/coll_executor/coll_reduce_scatter_v/310P/*.cc")

file(GLOB_RECURSE REDUCE_EXECUTOR "${HCOMM_CODE_ALGORITHM}/impl/coll_executor/coll_reduce/*.cc")

file(GLOB_RECURSE SEND_RECV_EXECUTOR "${HCOMM_CODE_ALGORITHM}/impl/coll_executor/coll_send_receive/*.cc")

file(GLOB_RECURSE BROADCAST_EXECUTOR "${HCOMM_CODE_ALGORITHM}/impl/coll_executor/coll_broadcast/*.cc")
file(GLOB_RECURSE BROADCAST_EXECUTOR_310P "${HCOMM_CODE_ALGORITHM}/impl/coll_executor/coll_broadcast/310P/*.cc")

file(GLOB_RECURSE ALL_TO_ALL_EXECUTOR "${HCOMM_CODE_ALGORITHM}/impl/coll_executor/coll_all_to_all/*.cc")
list(REMOVE_ITEM ALL_TO_ALL_EXECUTOR "${HCOMM_CODE_ALGORITHM}/impl/coll_executor/coll_all_to_all/coll_all_to_all_executor_aicpu.cc")

file(GLOB_RECURSE SCATTER_EXECUTOR "${HCOMM_CODE_ALGORITHM}/impl/coll_executor/coll_scatter/*.cc")

file(GLOB_RECURSE TEMPLATE_BASE "${HCOMM_CODE_ALGORITHM}/base/alg_template/*.cc")
file(GLOB_RECURSE TEMPLATE_BASE_COMPONENT "${HCOMM_CODE_ALGORITHM}/base/alg_template/component/*.cc")

set(src_list_alg
    ${HCOMM_CODE_ALGORITHM}/impl/alg_env_config.cc
    ${HCOMM_CODE_ALGORITHM}/impl/topo_matcher.cc
    ${HCOMM_CODE_ALGORITHM}/impl/alg_configurator.cc
    ${HCOMM_CODE_ALGORITHM}/impl/coll_alg_utils.cc
    ${HCOMM_CODE_ALGORITHM}/impl/hccl_alg.cc
    ${HCOMM_CODE_ALGORITHM}/impl/hccl_aiv.cc
    ${HCOMM_CODE_ALGORITHM}/impl/hccl_aiv_host.cc
    ${HCOMM_CODE_ALGORITHM}/base/communicator/topo_info_extractor.cc
    ${HCOMM_CODE_ALGORITHM}/base/communicator/calc_hd_transport_req.cc
    ${HCOMM_CODE_ALGORITHM}/base/communicator/calc_mesh_transport_req.cc
    ${HCOMM_CODE_ALGORITHM}/base/communicator/calc_p2p_transport_req.cc
    ${HCOMM_CODE_ALGORITHM}/base/communicator/calc_partial_mesh_transport_req.cc
    ${HCOMM_CODE_ALGORITHM}/base/communicator/calc_ring_transport_req.cc
    ${HCOMM_CODE_ALGORITHM}/base/communicator/calc_transport_req_base.cc
    ${HCOMM_CODE_ALGORITHM}/base/communicator/calc_hccs_plus_sio_transport_req.cc
    ${HCOMM_CODE_ALGORITHM}/base/communicator/calc_nb_transport_req.cc
    ${HCOMM_CODE_ALGORITHM}/base/communicator/calc_nhr_transport_req.cc
    ${HCOMM_CODE_ALGORITHM}/base/communicator/calc_nhr_v1_transport_req.cc
    ${HCOMM_CODE_ALGORITHM}/base/communicator/calc_ahc_broke_transport_req.cc
    ${HCOMM_CODE_ALGORITHM}/base/communicator/calc_ahc_transport_req_base.cc
    ${HCOMM_CODE_ALGORITHM}/base/communicator/calc_ahc_transport_req.cc

    ${HCOMM_CODE_ALGORITHM}/base/mc2_handler/mc2_handler.cc

    ${HCOMM_CODE_ALGORITHM}/impl/task/threadManage.cc
    ${HCOMM_CODE_ALGORITHM}/impl/task/parallel_task_loader.cc
    ${HCOMM_CODE_ALGORITHM}/impl/task/task_loader.cc
    ${HCOMM_CODE_ALGORITHM}/impl/operator/registry/coll_alg_op_registry.cc
    ${HCOMM_CODE_ALGORITHM}/impl/operator/coll_alg_operator.cc
    ${HCOMM_CODE_ALGORITHM}/impl/operator/all_reduce_operator.cc
    ${HCOMM_CODE_ALGORITHM}/impl/operator/all_gather_operator.cc
    ${HCOMM_CODE_ALGORITHM}/impl/operator/all_gather_v_operator.cc
    ${HCOMM_CODE_ALGORITHM}/impl/operator/reduce_scatter_operator.cc
    ${HCOMM_CODE_ALGORITHM}/impl/operator/reduce_scatter_v_operator.cc
    ${HCOMM_CODE_ALGORITHM}/impl/operator/send_operator.cc
    ${HCOMM_CODE_ALGORITHM}/impl/operator/receive_operator.cc
    ${HCOMM_CODE_ALGORITHM}/impl/operator/batch_write_operator.cc
    ${HCOMM_CODE_ALGORITHM}/impl/operator/broadcast_operator.cc
    ${HCOMM_CODE_ALGORITHM}/impl/operator/reduce_operator.cc
    ${HCOMM_CODE_ALGORITHM}/impl/operator/alltoall_operator.cc
    ${HCOMM_CODE_ALGORITHM}/impl/operator/scatter_operator.cc
    ${HCOMM_CODE_ALGORITHM}/impl/operator/batchsendrecv_operator.cc
    ${HCOMM_CODE_ALGORITHM}/impl/coll_executor/registry/coll_alg_exec_registry.cc
    ${HCOMM_CODE_ALGORITHM}/impl/coll_executor/coll_executor_base.cc
    ${HCOMM_CODE_ALGORITHM}/impl/coll_executor/coll_native_executor_base.cc
    ${HCOMM_CODE_ALGORITHM}/impl/coll_executor/coll_comm_executor.cc
    ${HCOMM_CODE_ALGORITHM}/impl/coll_executor/alg_profiling.cc

    # ${HCOMM_CODE_ALGORITHM}/base/alg_aiv_template/aiv_communication.h

    ${ALL_REDUCE_EXECUTOR}
    ${ALL_REDUCE_EXECUTOR_310P}

    ${ALL_GATHER_EXECUTOR}
    ${ALL_GATHER_EXECUTOR_310P}

    ${ALL_GATHER_V_EXECUTOR}

    ${REDUCE_SCATTER_EXECUTOR}
    ${REDUCE_SCATTER_EXECUTOR_310P}

    ${REDUCE_SCATTER_V_EXECUTOR}

    ${SEND_RECV_EXECUTOR}

    ${BROADCAST_EXECUTOR}
    ${BROADCAST_EXECUTOR_310P}

    ${REDUCE_EXECUTOR}

    ${SCATTER_EXECUTOR}

    ${ALL_TO_ALL_EXECUTOR}

    ${TEMPLATE_BASE}
    ${TEMPLATE_BASE_COMPONENT}
)

add_library(hccl_alg_test SHARED ${src_list_alg})

# set_source_files_properties(
#     ${HCOMM_CODE_ALGORITHM}/base/alg_aiv_template/aiv_communication.h PROPERTIES
#     LANGUAGE CXX
#     COMPILE_OPTIONS "-std=c++17"
# )

set(UTILS_INNER_INCLUDE_LIST
    ${HCOMM_ALG_UTILS_ROOT}/checker
    ${HCOMM_ALG_UTILS_ROOT}/checker/common/types
    ${HCOMM_ALG_UTILS_ROOT}/checker/rank_info_recorder
    ${HCOMM_ALG_UTILS_ROOT}/checker/mem_layout
    ${HCOMM_ALG_UTILS_ROOT}/checker/link_type_recorder
    ${HCOMM_ALG_UTILS_ROOT}/checker/task_collector
    ${HCOMM_ALG_UTILS_ROOT}/checker/semantics_check
    ${HCOMM_ALG_UTILS_ROOT}/checker/singletask_check
    ${HCOMM_ALG_UTILS_ROOT}/checker/mem_conflict_check
    ${HCOMM_ALG_UTILS_ROOT}/checker/ui/data_dumper
    ${HCOMM_ALG_UTILS_ROOT}/checker/param_trans
    ${HCOMM_ALG_UTILS_ROOT}/adapter_v1
    ${HCOMM_ALG_UTILS_ROOT}/adapter_v1/runtime
)

target_include_directories(hccl_alg_test PRIVATE
    ${UTILS_PUBLIC_INCLUDE_LIST}
    ${UTILS_INNER_INCLUDE_LIST}
    ${HCOMM_CODE_FRAMEWORK}/common/src/config
    ${HCOMM_PUBLIC_INCLUDE_LIST}
    ${HCOMM_FRAMEWORK_INCLUDE_LIST}
    ${HCOMM_ALGORITHM_INCLUDE_LIST}
    ${HCOMM_COMMON_INCLUDE_LIST}
    ${HCOMM_LEGACY_INCLUDE_LIST}
    ${ASCEND_CANN_INCLUDE_LIST}
    ${HCOMM_PLATFORM_INCLUDE_LIST}
)

target_compile_options(hccl_alg_test PRIVATE
    -Werror
    -fno-common
    -fno-strict-aliasing
    -pipe
    -std=c++17
    -Os
    -O2
    -x c++
    -fstack-protector-all
    -Wno-attributes
    $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address -fsanitize-recover=address,all -fno-omit-frame-pointer -g>
)

target_compile_definitions(hccl_alg_test PRIVATE
    -DOPEN_HCCL_TEST
    _GLIBCXX_USE_CXX11_ABI=0
)

target_link_libraries(hccl_alg_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    $<BUILD_INTERFACE:rdma_core_headers>
    -Wl,--no-as-needed
    -Wl,--no-as-needed
    pthread
    ${ASCEND_CANN_PACKAGE_PATH}/lib64/libc_sec.so
)

if("all" IN_LIST ST_TASKS OR "open_hccl_test" IN_LIST ST_TASKS)
    add_subdirectory(testcase)
endif()

if("all" IN_LIST ST_TASKS OR "executor_hccl_test" IN_LIST ST_TASKS)
    add_subdirectory(executor_testcase_generalization)
endif()

if("all" IN_LIST ST_TASKS OR "executor_reduce_hccl_test" IN_LIST ST_TASKS)
    add_subdirectory(executor_reduce_testcase_generalization)
endif()

if("all" IN_LIST ST_TASKS OR "executor_pipeline_hccl_test" IN_LIST ST_TASKS)
    add_subdirectory(executor_alltoall_A3_pipeline_testcase)
endif()
