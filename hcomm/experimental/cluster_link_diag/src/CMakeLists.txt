# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_subdirectory(util)

set(DISP_PROBE_PUBLIC_EXECUTABLES
    probe_controller
    probe_topo
    rpc_host)

# 获取当前目录下的所有.cpp和.c文件（不包含子目录）
file(GLOB EXEC_SOURCES 
    "*.cpp"
    "*.c"
)
# 为每个源文件创建一个可执行文件
foreach(source ${EXEC_SOURCES})
    get_filename_component(executable_name ${source} NAME_WE)
    add_executable(${executable_name} ${source})
    if(NOT executable_name IN_LIST DISP_PROBE_PUBLIC_EXECUTABLES)
        set_target_properties(${executable_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${DISP_PROBE_INTERNAL_RUNTIME_DIR}")
    endif()
    target_include_directories(${executable_name} PRIVATE
        "${DISP_PROBE_THIRDLIB_INCLUDE_DIR}"
        "${DISP_PROBE_THIRDLIB_INCLUDE_DIR}/eigen3")
    target_link_libraries(${executable_name} PRIVATE Threads::Threads)
    target_link_libraries(${executable_name} PRIVATE rpc)
    target_link_libraries(${executable_name} PRIVATE CLI)
    target_link_libraries(${executable_name} PRIVATE util)
    target_link_libraries(${executable_name} PRIVATE nlohmann_json)
    target_link_libraries(${executable_name} PRIVATE hccn_pingpong_rpc)
endforeach()
