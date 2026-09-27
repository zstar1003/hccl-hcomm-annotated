# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -D_GLIBCXX_USE_CXX11_ABI=0")

include_directories(
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/reduce_executor
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/runner_utils
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/external
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/store
    ${CMAKE_SOURCE_DIR}/include/runnerdb
    ${YAMLCPP_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/third_party/nlohmann_json
    ${HCOMM_CODE_ROOT}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/include
)

link_libraries(

    gtest
)

add_executable(test_reduce_add_executor reduce_add_executor_test.cc main.cc)
target_sources(test_reduce_add_executor PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/reduce_executor/reduce_add_executor.cc
)
set_target_properties(test_reduce_add_executor PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_link_libraries(test_reduce_add_executor ccu_executor_common_objs ${CCU_EXECUTOR_COMMON_LIBS})
ccu_executor_register_test(test_reduce_add_executor)

add_executable(test_reduce_max_executor reduce_max_executor_test.cc main.cc)
target_sources(test_reduce_max_executor PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/reduce_executor/reduce_max_executor.cc
)
set_target_properties(test_reduce_max_executor PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_link_libraries(test_reduce_max_executor ccu_executor_common_objs ${CCU_EXECUTOR_COMMON_LIBS})
ccu_executor_register_test(test_reduce_max_executor)

add_executable(test_reduce_min_executor reduce_min_executor_test.cc main.cc)
target_sources(test_reduce_min_executor PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/reduce_executor/reduce_min_executor.cc
)
set_target_properties(test_reduce_min_executor PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_link_libraries(test_reduce_min_executor ccu_executor_common_objs ${CCU_EXECUTOR_COMMON_LIBS})
ccu_executor_register_test(test_reduce_min_executor)

add_executable(test_ccu_reduce_operator ccu_reduce_operator_test.cc main.cc)
set_target_properties(test_ccu_reduce_operator PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_link_libraries(test_ccu_reduce_operator ccu_executor_common_objs ${CCU_EXECUTOR_COMMON_LIBS})
ccu_executor_register_test(test_ccu_reduce_operator)
