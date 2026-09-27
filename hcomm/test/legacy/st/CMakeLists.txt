# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(HCCL_V2_CODE_ROOT ${TOP_DIR}/ace/comop/hccl/orion)
set(HCCL_NEXT_CODE_ROOT ${TOP_DIR}/ace/comop/hccl/whole/hccl_next)
set(HCCL_LLT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O0 -fprofile-arcs -ftest-coverage")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -O0 -fprofile-arcs -ftest-coverage")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fprofile-arcs -ftest-coverage")

file(WRITE ${TOP_DIR}/ace/npuruntime/inc/external/runtime/rt_error_codes.h "")

add_executable(hccl_v2_stest main.cc)

FILE(GLOB_RECURSE sources_list LIST_DIRECTORIES TRUE ${HCCL_V2_CODE_ROOT}/*)

target_sources(hccl_v2_stest 
    PUBLIC
    ${HCCL_LLT_ROOT}/common/virtual_topo_stub/virtual_topo_stub.cc
    ${HCCL_LLT_ROOT}/common/common_interface.cc
)

foreach (dir ${sources_list})
    IF ((IS_DIRECTORY ${dir} ) AND (${dir} MATCHES "hccl/orion") AND (NOT ${dir} MATCHES "hccl/orion/mdpi"))
        target_include_directories(hccl_v2_stest PUBLIC ${dir})
    endif ()
endforeach ()

FILE(GLOB_RECURSE open_dir_list LIST_DIRECTORIES TRUE ${HCCL_NEXT_CODE_ROOT})
foreach (dir ${open_dir_list})
    IF ((IS_DIRECTORY ${dir}) AND (${dir} MATCHES "whole/hccl_next"))
        target_include_directories(hccl_v2_stest PUBLIC ${dir})
    endif ()
endforeach ()


target_include_directories(hccl_v2_stest PUBLIC
    ${HCCL_LLT_ROOT}/st
    ${HCCL_LLT_ROOT}/common
    ${HCCL_LLT_ROOT}/common/virtual_topo_stub
    ${TOP_DIR}/runtime/include/external
    ${TOP_DIR}/runtime/pkg_inc/runtime
    ${TOP_DIR}/runtime/pkg_inc/
    ${TOP_DIR}/runtime/pkg_inc/profiling
    ${TOP_DIR}/ace/comop/inc/
    ${TOP_DIR}/ace/comop/inc/hccl/
    ${TOP_DIR}/ace/comop/inc/external/
    ${TOP_DIR}/ace/npuruntime/inc/runtime/
    ${TOP_DIR}/ace/npuruntime/acl/inc/external/
    ${TOP_DIR}/inc/external/
    ${TOP_DIR}/runtime/src/dfx/adump/inc/adump
    ${TOP_DIR}/metadef/inc/external
    ${TOP_DIR}/metadef/pkg_inc
    ${TOP_DIR}/abl/mmpa/inc/mmpa
    ${TOP_DIR}/abl/mmpa/inc/mmpa/sub_inc
)


target_compile_options(hccl_v2_stest PRIVATE
        -std=c++17
        -Werror
        -fno-common
        -fno-strict-aliasing
        -pipe
        -g
        -O0
)

add_subdirectory(fwk)
add_subdirectory(test_case)

add_subdirectory(service/collective/alg)
add_subdirectory(otherFiles)

target_link_libraries(hccl_v2_stest PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub>
        $<BUILD_INTERFACE:hccl_headers>
        -lrt
        hccl_st_fwk
)
