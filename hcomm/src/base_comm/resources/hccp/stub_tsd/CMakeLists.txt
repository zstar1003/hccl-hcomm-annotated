# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(stub_tsd_list
    stub_tsd.c
)
# 创建共享库
add_library(tsd_eventclient SHARED ${stub_tsd_list})

target_compile_options(tsd_eventclient PRIVATE
    -Werror
    -fno-common
    -fno-strict-aliasing
    $<$<CONFIG:Debug>:-Og -g>
    $<$<CONFIG:Release>:-O3>
)

# 符号隐藏: 仅导出白名单符号; UT/ST 打桩依赖动态符号, 不挂载
if(NOT ENABLE_TEST)
    target_link_options(tsd_eventclient PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_LIST_DIR}/tsd_eventclient.map")
    set_property(TARGET tsd_eventclient APPEND PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/tsd_eventclient.map")
endif()

target_link_libraries(tsd_eventclient PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<BUILD_INTERFACE:runtime_headers>
)
