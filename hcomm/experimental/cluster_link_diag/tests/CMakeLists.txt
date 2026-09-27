# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(DISP_PROBE_UTS
    t1_config_test
    t2_topology_test
    t3_probe_plan_test
    t4_metric_reduction_test
    t5_link_solver_test
    t6_metrics_collector_test)

foreach(test_name IN LISTS DISP_PROBE_UTS)
    add_executable(${test_name} ${test_name}.cpp)
    set_target_properties(${test_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${DISP_PROBE_INTERNAL_RUNTIME_DIR}/tests")
    target_include_directories(${test_name} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${PROJECT_SOURCE_DIR}/src/util")
    target_link_libraries(${test_name} PRIVATE
        Threads::Threads
        helper
        topo
        tool
        nlohmann_json)
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
    set_tests_properties(${test_name} PROPERTIES LABELS "UT")
endforeach()
