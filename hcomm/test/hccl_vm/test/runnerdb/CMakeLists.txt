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
    ${CMAKE_SOURCE_DIR}/src/runnerdb
    ${CMAKE_SOURCE_DIR}/include/runnerdb
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/store
    ${YAMLCPP_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/third_party
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/include
)

# test_sqlite_table
add_executable(test_sqlite_table sqlite_table_test.cc)
target_link_libraries(test_sqlite_table PRIVATE
    sqlite3
    gtest_main
    gtest
)
set_target_properties(test_sqlite_table PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

# test_sim_models
add_executable(test_sim_models sim_models_test.cc)
target_link_libraries(test_sim_models PRIVATE
    sqlite3
    gtest_main
    gtest
)
set_target_properties(test_sim_models PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

# test_runner_db
add_executable(test_runner_db runner_db_test.cc)
target_link_libraries(test_runner_db PRIVATE
    sqlite3
    gtest_main
    gtest
)
set_target_properties(test_runner_db PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

# test_sim_runner_common
add_executable(test_sim_runner_common sim_runner_common_test.cc)
target_link_libraries(test_sim_runner_common PRIVATE
    pthread
    gmock_main
    gmock
    gtest
    runnerdb
    sqlite3
)
set_target_properties(test_sim_runner_common PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

# test_sim_runner_db
add_executable(test_sim_runner_db_test sim_runner_db_test.cc)
target_link_libraries(test_sim_runner_db_test PRIVATE
    pthread
    gmock_main
    gmock
    gtest
    runnerdb
    sqlite3
)
set_target_properties(test_sim_runner_db_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

# test_sim_runner_ops
add_executable(test_sim_runner_ops sim_runner_ops_test.cc)
target_link_libraries(test_sim_runner_ops PRIVATE
    pthread
    gmock_main
    gmock
    gtest
    runnerdb
    sqlite3
)
set_target_properties(test_sim_runner_ops PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

# test_sim_sqlite_db
add_executable(test_sim_sqlite_db_test sim_sqlite_db_test.cc)
target_link_libraries(test_sim_sqlite_db_test PRIVATE
    pthread
    gmock_main
    gmock
    gtest
    runnerdb
    sqlite3
)
set_target_properties(test_sim_sqlite_db_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
