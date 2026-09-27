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

set(SUBCMD_LIBS
    gtest_main gtest
    hccl_plugin_objs
    cmd common store runnerdb topo log yaml-cpp sqlite3 dl rt
    z
)

set(SUBCMD_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/common
    ${CMAKE_SOURCE_DIR}/src/plugin
    ${CMAKE_SOURCE_DIR}/src/cmd
    ${CMAKE_SOURCE_DIR}/src/cmd/subcmd
    ${CMAKE_SOURCE_DIR}/src/utils
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/external
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/store
    ${CMAKE_SOURCE_DIR}/include/runnerdb
    ${CMAKE_SOURCE_DIR}/third_party/CLI11
    ${YAMLCPP_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/third_party/nlohmann_json
)

# test_subcmd_start
add_executable(test_subcmd_start subcmd_start_test.cc)
add_dependencies(test_subcmd_start hccl-vm)
target_include_directories(test_subcmd_start PRIVATE ${SUBCMD_INCLUDE_DIRS})
target_link_libraries(test_subcmd_start PRIVATE ${SUBCMD_LIBS})
set_target_properties(test_subcmd_start PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin")

# test_subcmd_run
add_executable(test_subcmd_run subcmd_run_test.cc)
add_dependencies(test_subcmd_run hccl-vm)
target_include_directories(test_subcmd_run PRIVATE ${SUBCMD_INCLUDE_DIRS})
target_link_libraries(test_subcmd_run PRIVATE ${SUBCMD_LIBS})
set_target_properties(test_subcmd_run PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin")

# test_subcmd_model
add_executable(test_subcmd_model subcmd_model_test.cc)
add_dependencies(test_subcmd_model hccl-vm)
target_include_directories(test_subcmd_model PRIVATE ${SUBCMD_INCLUDE_DIRS})
target_link_libraries(test_subcmd_model PRIVATE ${SUBCMD_LIBS})
set_target_properties(test_subcmd_model PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin")

# test_subcmd_log
add_executable(test_subcmd_log subcmd_log_test.cc)
add_dependencies(test_subcmd_log hccl-vm)
target_include_directories(test_subcmd_log PRIVATE ${SUBCMD_INCLUDE_DIRS})
target_link_libraries(test_subcmd_log PRIVATE ${SUBCMD_LIBS})
set_target_properties(test_subcmd_log PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin")

# test_subcmd_plugin
add_executable(test_subcmd_plugin subcmd_plugin_test.cc)
add_dependencies(test_subcmd_plugin hccl-vm)
target_include_directories(test_subcmd_plugin PRIVATE ${SUBCMD_INCLUDE_DIRS})
target_link_libraries(test_subcmd_plugin PRIVATE ${SUBCMD_LIBS})
set_target_properties(test_subcmd_plugin PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin")

# test_subcmd_table
add_executable(test_subcmd_table subcmd_table_test.cc)
add_dependencies(test_subcmd_table hccl-vm)
target_include_directories(test_subcmd_table PRIVATE ${SUBCMD_INCLUDE_DIRS})
target_link_libraries(test_subcmd_table PRIVATE ${SUBCMD_LIBS})
set_target_properties(test_subcmd_table PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin")
