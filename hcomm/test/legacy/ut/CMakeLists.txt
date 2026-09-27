# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(HCCL_LLT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../)
set(HCCL_CODE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
get_filename_component(HCCL_CODE_ROOT "${HCCL_CODE_ROOT}" REALPATH)
get_filename_component(HCCL_LLT_ROOT "${HCCL_LLT_ROOT}" REALPATH)
set(HCCL_V2_CODE_ROOT ${HCCL_CODE_ROOT}/src/legacy/ascend950)

file(GLOB_RECURSE HCCL_V2_UTEST_FILES RELATIVE ${CMAKE_CURRENT_LIST_DIR}
    main.cc
)

add_executable(hccl_v2_utest ${HCCL_V2_UTEST_FILES})

FILE(GLOB_RECURSE ut_sources_list LIST_DIRECTORIES TRUE ${HCCL_LLT_ROOT}/ut)

foreach (dir ${ut_sources_list})
    IF ((IS_DIRECTORY ${dir} ) AND (${dir} MATCHES "test/legacy") AND (NOT ${dir} MATCHES "legacy/depends") AND (NOT ${dir} MATCHES "legacy/st/algorithm"))
        target_include_directories(hccl_v2_utest PUBLIC ${dir})
    endif ()
endforeach ()

FILE(GLOB_RECURSE sources_list LIST_DIRECTORIES TRUE ${HCCL_CODE_ROOT}/src/legacy/ascend950)

message(STATUS "************************=====${sources_list}")
foreach (dir ${sources_list})
    IF ((IS_DIRECTORY ${dir} ) AND (${dir} MATCHES "src/legacy/ascend950") AND (NOT ${dir} MATCHES "src/legacy/ascend950/mdpi"))
        target_include_directories(hccl_v2_utest PUBLIC ${dir})
    endif ()
endforeach ()

# 自动找到项目内所有头文件
FILE(GLOB_RECURSE hccl_v2_dir_list LIST_DIRECTORIES TRUE ${HCCL_V2_CODE_ROOT})
foreach (dir ${hccl_v2_dir_list})
    IF ((IS_DIRECTORY ${dir}) AND (${dir} MATCHES "hcomm/src/legacy/ascend950"))
        target_include_directories(hccl_v2_utest PUBLIC ${dir})
    endif ()
endforeach ()

# 新拓扑目录中存在与 legacy 同名的头文件，仅前置这些目录，避免改变其他头文件的搜索顺序。
target_include_directories(hccl_v2_utest BEFORE PRIVATE
        ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_info_detect/
        ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/
        ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/common/
        ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/phy_topo/
        ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/phy_topo_builder/
        ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/rank_graph/
        ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/rank_graph_builder/
        ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/rank_table_info/
        ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/topo_info/
    )

# 添加项目外部头文件
target_include_directories(hccl_v2_utest PRIVATE
        ${HCCL_CODE_ROOT}/include/
        ${HCCL_CODE_ROOT}/include/hccl
        ${HCCL_CODE_ROOT}/pkg_inc
        ${HCCL_CODE_ROOT}/pkg_inc/hccl
        ${HCCL_CODE_ROOT}/pkg_inc/legacy
        ${HCCL_CODE_ROOT}/pkg_inc/legacy/hccl

        ${HCCL_CODE_ROOT}/test/stub/depends/include/
        ${HCCL_CODE_ROOT}/test/stub/depends/include/driver/
        ${HCCL_CODE_ROOT}/test/stub/depends/include/mmpa/
        ${HCCL_CODE_ROOT}/test/stub/depends/pkg_inc/
        ${HCCL_CODE_ROOT}/test/stub/depends/pkg_inc/base
        ${HCCL_CODE_ROOT}/test/stub/depends/pkg_inc/profiling/
        ${HCCL_CODE_ROOT}/test/stub/depends/pkg_inc/trace/
        ${HCCL_CODE_ROOT}/test/stub/depends/toolchain/
        ${HCCL_CODE_ROOT}/test/stub/depends/pkg_inc/runtime/

        ${HCCL_CODE_ROOT}/src/legacy/ascend910/common/
        ${HCCL_CODE_ROOT}/src/legacy/ascend910/common/debug/profiling/inc
        ${HCCL_CODE_ROOT}/src/legacy/ascend910/pub_inc/
        ${HCCL_CODE_ROOT}/src/base_comm/
        ${HCCL_CODE_ROOT}/src/base_comm/resources/hccp/inc/network/
        ${HCCL_CODE_ROOT}/src/base_comm/resources/hccp/inc/
        ${HCCL_CODE_ROOT}/src/base_comm/resources/
        ${HCCL_CODE_ROOT}/src/base_comm/resources/endpoints/
        ${HCCL_CODE_ROOT}/src/base_comm/resources/reged_mems/
        ${HCCL_CODE_ROOT}/src/base_comm/common/
        ${HCCL_CODE_ROOT}/src/base_comm/common/debug/
        ${HCCL_CODE_ROOT}/src/base_comm/primitives/aicpu/task_cache
        ${HCCL_CODE_ROOT}/src/base_comm/primitives/api_c_adpt/
        ${HCCL_CODE_ROOT}/src/base_comm/resources/comm_engine_res/threads
        ${HCCL_CODE_ROOT}/src/base_comm/dfx/
        ${HCCL_CODE_ROOT}/src/base_comm/resources/comm_engine_res/threads/
        ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/dfx/profiling/aicpu/
        ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/dfx/profiling/aicpu/common/

        ${HCCL_LLT_ROOT}/common/virtual_topo_stub/
        ${HCCL_LLT_ROOT}/ut/framework/topo/ranktable/
        ${HCCL_CODE_ROOT}/external_depends/tsch

        ${HCOMM_DIR}/src/legacy/ascend910/pub_inc/new
        ${HCOMM_DIR}/src/legacy/ascend910/common/error_manager
        ${HCOMM_DIR}/src/legacy/ascend910/framework/common/src/exception
    )

target_link_libraries(hccl_v2_utest PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    $<BUILD_INTERFACE:rdma_core_headers>
    -pthread
    # hccl_llt
    -Wl,--whole-archive
    -Wl,--no-whole-archi
    -lgcov
    -lrt
    -ldl
    -pthread
)


# 编译选项
target_compile_options(hccl_v2_utest PRIVATE
        -std=c++17
        -Werror
        -fno-common
        -fno-strict-aliasing
        -pipe
        -U_FORTIFY_SOURCE
        -fno-var-tracking-assignments
        -Os
        -O0 -g --coverage -fprofile-arcs -ftest-coverage -fno-access-control
        $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address -fsanitize-recover=address,all -fno-omit-frame-pointer>
)

message(STATUS "ORION_MODE=${ORION_MODE}")
target_compile_definitions(hccl_v2_utest PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
    # private=public
    # protected=public
    HCOMM_CODE_ROOT_DIR="${HCCL_CODE_ROOT}"
    # $<$<STREQUAL:${ORION_MODE},true>:ORION_MODE>
    # ORION_MODE

    ORION_AICPU_DEPLOY
    ORION_HOST_DEPLOY
    OPEN_HCCL_TEST
    HCCL_V2
)

add_subdirectory(all_source_code)
add_subdirectory(common)
add_subdirectory(service)
add_subdirectory(framework)
add_subdirectory(aicpu)
add_subdirectory(unified_platform)

enable_testing()
add_test(NAME hccl_v2_utest COMMAND hccl_v2_utest)
