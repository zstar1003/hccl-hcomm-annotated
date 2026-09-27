# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.9)
project(topoaddrinfo VERSION 1.0.0 DESCRIPTION "topo addr info generator")

add_library(topoaddrinfo SHARED
    src/hal.c
    src/topo.c
    src/eid_util.c
    src/rank_info_types.c
    src/topo_addr_info.c
    src/product_card.c
    src/product_server.c
    src/product_pod.c
    src/npu_nic_affinity.c
    src/xml_parser.c
    src/topo_addr_info_log.c
)

target_compile_options(topoaddrinfo PRIVATE
    -Werror
    -fno-common
    -fno-strict-aliasing
    $<$<CONFIG:Debug>:-Og -g>
    $<$<CONFIG:Release>:-O3>
)

# 符号隐藏: 仅导出白名单符号; UT/ST 打桩依赖动态符号, 不挂载
if(NOT ENABLE_TEST)
    target_link_options(topoaddrinfo PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_LIST_DIR}/topoaddrinfo.map")
    set_property(TARGET topoaddrinfo APPEND PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/topoaddrinfo.map")
endif()

target_include_directories(topoaddrinfo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${HCOMM_DIR}/src/legacy/ascend910/pub_inc
    ${HCOMM_DIR}/src/legacy/ascend910/pub_inc/new
    ${HCOMM_DIR}/src/legacy/ascend910/common/error_manager
    ${HCOMM_DIR}/include
)

target_link_libraries(topoaddrinfo PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<BUILD_INTERFACE:ascend_hal_headers>
    $<BUILD_INTERFACE:runtime_headers>
    dl
    c_sec
)

if(TARGET hcomm)
    if(POLICY CMP0079)
        cmake_policy(PUSH)
        cmake_policy(SET CMP0079 NEW)
    endif()
    target_link_libraries(hcomm topoaddrinfo)
    if(POLICY CMP0079)
        cmake_policy(POP)
    endif()
endif()

install(TARGETS topoaddrinfo
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} ${INSTALL_OPTIONAL}
    COMPONENT hcomm
)
