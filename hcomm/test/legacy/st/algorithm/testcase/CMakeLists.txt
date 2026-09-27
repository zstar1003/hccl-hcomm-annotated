# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(include_list
    ${UTILS_DIR}/adapter_v1/aiv

    ${legacy_common_include_list}
    ${legacy_framework_include_list}
    ${legacy_platform_include_list}
    ${legacy_service_include_list}
    ${checker_include_list}

    ${HCOMM_CODE_LEGACY}/common

    ${HCOMM_PLATFORM_INCLUDE_LIST}

    ${ASCEND_CANN_INCLUDE_LIST}

    ${CMAKE_BINARY_DIR}/proto/alg_analyzer_proto/proto
    ${THIRD_PARTY_PATH}/protobuf/include

    ${HCOMM_DIR}/src/legacy/ascend910/common
    ${HCOMM_DIR}/src/legacy/ascend910/common/error_manager
)

file(GLOB_RECURSE ALG_CCU_CONTEXT_SRC "${HCOMM_LEGACY_SERVICE}/alg/coll_alg_factory/alg_ccu_context/*.cc")
file(GLOB_RECURSE ALG_CCU_TEMPLATE_SRC "${HCOMM_LEGACY_SERVICE}/alg/coll_alg_factory/alg_template/ccu_alg_template/*.cc")
file(GLOB_RECURSE ALG_AIV_TEMPLATE_SRC "${HCOMM_LEGACY_SERVICE}/alg/coll_alg_factory/alg_template/aiv_alg_template/*.cc")
file(GLOB_RECURSE ALG_PRIM_SRC "${HCOMM_LEGACY_SERVICE}/primitive/*.cc")
file(GLOB_RECURSE ALG_PRIM_EXECUTOR_SRC "${HCOMM_LEGACY_SERVICE}/alg/coll_alg_factory/alg_executor/prim_alg_executor/*.cc")
file(GLOB_RECURSE ALG_PRIM_TEMPLATE_SRC "${HCOMM_LEGACY_SERVICE}/alg/coll_alg_factory/alg_template/prim_alg_template/*.cc")

file(GLOB_RECURSE ALG_NEXT_SRC "${HCOMM_LEGACY_SERVICE}/alg/*.cc")

# 排除不想编译的文件
file(GLOB_RECURSE EXCLUDE_BATCH_SEND_RECV_SRC "${HCOMM_LEGACY_SERVICE}/*/ins_batchSendRecv_executor.cc")
file(GLOB_RECURSE EXCLUDE_DEVICE_SRC "${HCOMM_LEGACY_SERVICE}/*/device_utils.cc")
list(REMOVE_ITEM ALG_NEXT_SRC ${EXCLUDE_BATCH_SEND_RECV_SRC})
list(REMOVE_ITEM ALG_NEXT_SRC ${EXCLUDE_DEVICE_SRC})
file(GLOB_RECURSE PRIMITIVE_SRC "${HCOMM_LEGACY_SERVICE}/primitive/*.cc")

# 被测代码编译
set(david_to_test_list
    #被测代码
    ${HCOMM_LEGACY_SERVICE}/instruction/instruction.cc
    ${HCOMM_LEGACY_SERVICE}/coll_operator.cc
    ${ALG_CCU_CONTEXT_SRC}
    ${ALG_CCU_TEMPLATE_SRC}
    ${ALG_AIV_TEMPLATE_SRC}
    ${ALG_PRIM_SRC}
    ${ALG_PRIM_EXECUTOR_SRC}
    ${ALG_PRIM_TEMPLATE_SRC}
    ${ALG_NEXT_SRC}
    ${PRIMITIVE_SRC}
    ${UTILS_DIR}/common/acl_stub/acl_stub.cc
)

add_library(hccl_orion_alg_test SHARED ${david_to_test_list})

#头文件搜索路径
target_include_directories(hccl_orion_alg_test PRIVATE
    ${include_list}
    ${HCOMM_DIR}/src/legacy/ascend910/common
    ${HCOMM_DIR}/src/legacy/ascend910/common/error_manager
)

target_compile_definitions(hccl_orion_alg_test PRIVATE
    google=ascend_private
    -DOPEN_HCCL_TEST
    _GLIBCXX_USE_CXX11_ABI=0
)

target_compile_options(hccl_orion_alg_test PRIVATE
    -U_FORTIFY_SOURCE
    -O0
    -g
    -pipe
)

target_link_libraries(hccl_orion_alg_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    $<BUILD_INTERFACE:rdma_core_headers>
    $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address>
    -pthread
    ascend_protobuf
    -lrt
    -ldl
)

set(test_case_list
)

# add_subdirectory(aiv_1d_testcase)

if("all" IN_LIST ST_TASKS OR "legacy_all_testcase" IN_LIST ST_TASKS OR "legacy_alg_testcase" IN_LIST ST_TASKS)
    add_subdirectory(legacy_alg_testcase)
endif()

if("all" IN_LIST ST_TASKS OR "legacy_all_testcase" IN_LIST ST_TASKS OR "legacy_aicpu_2d_testcase" IN_LIST ST_TASKS)
    add_subdirectory(aicpu_2d_testcase)
endif()

if("all" IN_LIST ST_TASKS OR "legacy_all_testcase" IN_LIST ST_TASKS OR "legacy_ccu_1d_testcase_part1" IN_LIST ST_TASKS)
    add_subdirectory(ccu_1d_testcase_part1)
endif()

if("all" IN_LIST ST_TASKS OR "legacy_all_testcase" IN_LIST ST_TASKS OR "legacy_ccu_1d_testcase_part2" IN_LIST ST_TASKS)
    add_subdirectory(ccu_1d_testcase_part2)
endif()

if("all" IN_LIST ST_TASKS OR "legacy_all_testcase" IN_LIST ST_TASKS OR "legacy_alg_ccu_reduce" IN_LIST ST_TASKS)
    add_subdirectory(ccu_reduce_testcase)
endif()

if("all" IN_LIST ST_TASKS OR "legacy_all_testcase" IN_LIST ST_TASKS OR "legacy_ccu_1d_hf16p_testcase" IN_LIST ST_TASKS)
    add_subdirectory(ccu_1d_hf16p_testcase)
endif()

if("all" IN_LIST ST_TASKS OR "legacy_all_testcase" IN_LIST ST_TASKS OR "legacy_ccu_2d_testcase" IN_LIST ST_TASKS)
    add_subdirectory(ccu_2d_testcase)
endif()

if("all" IN_LIST ST_TASKS OR "legacy_all_testcase" IN_LIST ST_TASKS OR "legacy_function_ut_testcase" IN_LIST ST_TASKS)
    add_subdirectory(function_ut_testcase)
endif()
