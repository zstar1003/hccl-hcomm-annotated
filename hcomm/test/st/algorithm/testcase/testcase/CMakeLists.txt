# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_executable(open_hccl_test ${src_to_test_list})

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

#头文件搜索路径
target_include_directories(open_hccl_test PRIVATE
    ${UTILS_PUBLIC_INCLUDE_LIST}
    ${UTILS_INNER_INCLUDE_LIST}
    ${HCOMM_PUBLIC_INCLUDE_LIST}
    ${HCOMM_PLATFORM_INCLUDE_LIST}
    ${HCOMM_ALGORITHM_INCLUDE_LIST}
    ${HCOMM_COMMON_INCLUDE_LIST}
    ${HCOMM_FRAMEWORK_INCLUDE_LIST}
    ${HCOMM_LEGACY_INCLUDE_LIST}

    ${ASCEND_CANN_INCLUDE_LIST}
)

target_compile_definitions(open_hccl_test PRIVATE
    google=ascend_private
    _GLIBCXX_USE_CXX11_ABI=0
    -DOPEN_HCCL_TEST
)

target_compile_options(open_hccl_test PRIVATE
    -U_FORTIFY_SOURCE
    -Os
    -O0 -g
    $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address -fsanitize-recover=address,all -fno-omit-frame-pointer>
)

add_dependencies(open_hccl_test hccl_alg_analyzer hccl_alg_adaption hccl_alg_test)

target_link_libraries(open_hccl_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    $<BUILD_INTERFACE:rdma_core_headers>
    hccl_alg_analyzer
    hccl_alg_adaption
    hccl_alg_test
    $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address>
    -pthread
    ascend_protobuf
    -lrt
    -ldl
)

target_sources(open_hccl_test
        PUBLIC
        main.cc
        testcase_utils.cc
        testcase_ahc.cc
        testcase_aiv.cc
        testcase_alg_env_config.cc
        testcase_all_reduce.cc
        testcase_temp_register.cc
        testcase_all_gather.cc
        testcase_all_gather_v.cc
        testcase_reduce_scatter.cc
        testcase_reduce_scatter_v.cc
        testcase_all_to_all.cc
        testcase_all_to_allvc.cc
        testcase_all_to_allv.cc
        testcase_send_recv.cc
        testcase_broadcast.cc
        testcase_reduce.cc
        testcase_scatter.cc
        testcase_batchsendrecv.cc
        testcase_all_gather_v_for_310p.cc
        testcase_reduce_scatter_v_for_310p.cc
        testcase_template_base.cc
        testcase_misc.cc
        testcase_ars_alg.cc
)

enable_testing()
add_test(NAME open_hccl_test COMMAND open_hccl_test)
