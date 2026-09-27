# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16.0)
option(BUILD_OPEN_PROJECT   "Build open hcomm project."     ON)
option(ENABLE_BUILD_DEVICE  "Build hcomm device project."   OFF)
option(ENABLE_BUILD_AARCH   "Build hcomm aarch project."    OFF)
option(ENABLE_EXPERIMENTAL  "Build experimental components." OFF)

# 默认只编译 host 侧
if(NOT DEFINED PRODUCT_SIDE)
    set(PRODUCT_SIDE "host")
endif()

# 构建 Device 侧需设置 TOOLCHAIN_DIR 环境变量
if(DEFINED ENV{TOOLCHAIN_DIR} AND ENABLE_BUILD_DEVICE)
    set(ENABLE_DEVICE ON)
else()
    set(ENABLE_DEVICE OFF)
endif()

message(STATUS "ASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}")
message(STATUS "CANN_3RD_LIB_PATH=${CANN_3RD_LIB_PATH}")
message(STATUS "PRODUCT_SIDE=${PRODUCT_SIDE}")
message(STATUS "BUILD_OPEN_PROJECT=${BUILD_OPEN_PROJECT}")
message(STATUS "ENABLE_BUILD_DEVICE=${ENABLE_BUILD_DEVICE}")
message(STATUS "ENABLE_BUILD_AARCH=${ENABLE_BUILD_AARCH}")
message(STATUS "ENABLE_EXPERIMENTAL=${ENABLE_EXPERIMENTAL}")
message(STATUS "LAUNCH_COMPILE_TOOL=${LAUNCH_COMPILE_TOOL}")
message(STATUS "LAUNCH_LINK_TOOL=${LAUNCH_LINK_TOOL}")

set(HCOMM_DIR         "${CMAKE_CURRENT_SOURCE_DIR}")
set(HCC_TOOLCHAIN_DIR "${ASCEND_CANN_PACKAGE_PATH}/toolkit/toolchain/hcc")
set(HCC_CXX_COMPILER  "${HCC_TOOLCHAIN_DIR}/bin/aarch64-target-linux-gnu-g++")
set(HCC_C_COMPILER    "${HCC_TOOLCHAIN_DIR}/bin/aarch64-target-linux-gnu-gcc")

# 交叉编译场景（x86机器上编译arm包）
if(ENABLE_BUILD_AARCH)
    message(STATUS "TOOLCHAIN_DIR=${HCC_TOOLCHAIN_DIR}")
    set(CMAKE_SYSTEM_NAME Linux)
    set(CMAKE_SYSTEM_PROCESSOR aarch64)
    set(TOOLCHAIN_DIR      ${HCC_TOOLCHAIN_DIR})
    set(CMAKE_CXX_COMPILER ${HCC_CXX_COMPILER})
    set(CMAKE_C_COMPILER   ${HCC_C_COMPILER})
endif()

# 引入CANN/cmake库
include(cmake/fetch_cann_cmake.cmake)
project(hcomm)

# 初始化 CANN 项目配置
init_cann_project(PREPEND_MODULE_PATH)
add_cann_target_options()

# 采用 C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_FLAGS_RELEASE "")
set(CMAKE_CXX_FLAGS_RELEASE "")

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "12.0.0" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "13.0.0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=stringop-overread")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-error=stringop-overread")
endif()

# 获取 CPU 核心数
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    execute_process(
        COMMAND nproc
        OUTPUT_VARIABLE CPU_CORES
        OUTPUT_STRIP_TRAILING_WHITESPACE
    ) 
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    execute_process(
        COMMAND wmic cpu get NumberOfCores /value
        OUTPUT_VARIABLE CPU_CORES
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    # 解析输出
    string(REGEX REPLACE "NumberOfCores=([0-9]+)" "\\1" CPU_CORES "${CPU_CORES}")
else()
    # 默认设置为 1
    set(CPU_CORES 1)
endif()

math(EXPR JOBS "${CPU_CORES} * 2")
message(STATUS "Detected JOBS: ${JOBS}")
set(CMAKE_JOB_POOL_COMPILE "${JOBS}")
set(CMAKE_JOB_POOL_LINK "${JOBS}")

if(ENABLE_ASAN)
    # GCC 13+ 编译器开启 ASAN 时，<regex> 系统库头文件中存在误报，若定义 -Werror 则会导致编译失败
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=maybe-uninitialized")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-error=maybe-uninitialized")
endif()

if(ENABLE_TEST)
    enable_testing()
    include(cmake/config.cmake)
    include(cmake/intf_pub_gccnative.cmake)
    # 三方件
    include(cmake/third_party/mockcpp.cmake)
    add_cann_third_party(json)
    add_cann_third_party(gtest)
    add_cann_third_party(rdma-core)
    # 添加子目录
    add_subdirectory(test)

elseif(BUILD_OPEN_PROJECT)
    include(cmake/config.cmake)
    include(cmake/package.cmake)
    include(cmake/func.cmake)
    include(version.cmake)
    # 三方件
    include(cmake/hcomm_utils.cmake)
    include(cmake/third_party/ubengine.cmake)
    include(cmake/third_party/ibv_extend.cmake)
    add_cann_third_party(json)
    add_cann_third_party(openssl)
    add_cann_third_party(rdma-core)
    # 查找依赖包
    find_cann_package(acl_rt MODULE REQUIRED)
    find_cann_package(asc_devkit MODULE REQUIRED)
    find_cann_package(ascend_hal MODULE REQUIRED)
    find_cann_package(atrace MODULE REQUIRED)
    find_cann_package(error_manager MODULE REQUIRED)
    find_cann_package(mmpa MODULE REQUIRED)
    find_cann_package(msprof MODULE REQUIRED)
    find_cann_package(runtime MODULE REQUIRED)
    find_cann_package(securec MODULE REQUIRED)
    find_cann_package(tsdclient MODULE REQUIRED)
    find_cann_package(unified_dlog MODULE REQUIRED)
    # 检查构建依赖
    check_cann_pkg_build_deps("hcomm")
    add_version_info_targets()
    # 配置打包
    pack_built_in()
    # 添加子目录
    add_subdirectory(src)
    add_subdirectory(python)

    # Device 侧编译构建
    if(ENABLE_DEVICE)
        add_cann_device_project(hcomm
            EXT_ARGS -DLAUNCH_COMPILE_TOOL=${LAUNCH_COMPILE_TOOL}
        )
    endif()

    set(HCCL_HEAD
        include/hccl/hccl_comm.h
        include/hccl/hccl_types.h
        src/legacy/ascend910/pub_inc/hccn_rping.h
        include/hccl/hccl_rank_graph.h
        include/hccl/hccl_ccu_res.h
        include/hccl/hccl_res.h
        include/hccl/hccl_channel.h
        include/hccl/hccl_sym_win.h
        include/hccl/hccl_launch.h
        include/hccl/hccl_team.h
    )

    set(HCOMM_HEAD
        include/hcomm_primitives.h
    )

    install(FILES
        ${HCCL_HEAD}
        DESTINATION ${INSTALL_INCLUDE_DIR}/hccl/ ${INSTALL_OPTIONAL}
        COMPONENT hcomm
    )
    if(NOT BUILD_OPEN_PROJECT)
        install(FILES 
            ${HCOMM_HEAD} 
            DESTINATION ${INSTALL_INCLUDE_DIR}/ ${INSTALL_OPTIONAL}
            COMPONENT hcomm 
        )
    endif()
    install(FILES
        include/hcomm_primitives.h
        DESTINATION ${INSTALL_INCLUDE_DIR}/hcomm/ ${INSTALL_OPTIONAL}
        COMPONENT hcomm
    )
    install(FILES
        include/hcomm_res.h
        include/hcomm_res_defs.h
        include/hcomm_channel.h
        include/hcomm_team_defs.h
        DESTINATION ${INSTALL_INCLUDE_DIR}/hcomm/ ${INSTALL_OPTIONAL}
        COMPONENT hcomm
    )

    set(HCCL_PKG_HEAD
        pkg_inc/hccl/hccl_res_expt.h
        pkg_inc/hccl/hccl_diag.h
        pkg_inc/hccl/hcomm_diag.h
        pkg_inc/legacy/hccl/base.h
        pkg_inc/legacy/hccl/dtype_common.h
        pkg_inc/legacy/hccl/hccl_ctrl_plane.h
        pkg_inc/legacy/hccl/hccl_ex.h
        pkg_inc/legacy/hccl/hccl_inner.h
        pkg_inc/legacy/hccl/hccl_one_sided_services.h
        pkg_inc/legacy/hccl/hcom.h
        pkg_inc/legacy/hccl/workflow.h
    )
    install(FILES
        ${HCCL_PKG_HEAD}
        DESTINATION ${INSTALL_PKG_INCLUDE_DIR}/hccl/ ${INSTALL_OPTIONAL}
        COMPONENT hcomm
    )
    set(HCCL_INC_CCU_HEAD
        include/ccu/ccu_address.hpp
        include/ccu/ccu_array.hpp
        include/ccu/ccu_buffer.hpp
        include/ccu/ccu_control_flow_macro.h
        include/ccu/ccu_event.hpp
        include/ccu/ccu_func.hpp
        include/ccu/ccu_launch.h
        include/ccu/ccu_local_addr.hpp
        include/ccu/ccu_loop.hpp
        include/ccu/ccu_primitives.hpp
        include/ccu/ccu_remote_addr.hpp
        include/ccu/ccu_res.h
        include/ccu/ccu_res_defs.h
        include/ccu/ccu_types.h
        include/ccu/ccu_utils.hpp
        include/ccu/ccu_variable.hpp
    )
    install(FILES
        ${HCCL_INC_CCU_HEAD}
        DESTINATION ${INSTALL_INCLUDE_DIR}/hcomm/ccu/ ${INSTALL_OPTIONAL}
        COMPONENT hcomm
    )

    set(HCCL_PKG_INC_CCU_HEAD
        pkg_inc/hcomm/ccu/ccu_primitives_impl.h
    )
    install(FILES
        ${HCCL_PKG_INC_CCU_HEAD}
        DESTINATION ${INSTALL_PKG_INCLUDE_DIR}/hcomm/ccu/ ${INSTALL_OPTIONAL}
        COMPONENT hcomm
    )

    set(HCOMM_PKG_HEAD
        pkg_inc/hcomm/hcomm_primitives_expt.h
        pkg_inc/hcomm/hcomm_res_entity_defs.h
        pkg_inc/hcomm/hcomm_team_entity_defs.h
        pkg_inc/hcomm/hcomm_exception.h
    )
    install(FILES
        ${HCOMM_PKG_HEAD}
        DESTINATION ${INSTALL_PKG_INCLUDE_DIR}/hcomm/ ${INSTALL_OPTIONAL}
        COMPONENT hcomm
    )

    if(ENABLE_EXPERIMENTAL AND EXISTS "${HCOMM_DIR}/experimental/base_comm/nic_plugin/CMakeLists.txt")
        add_subdirectory(experimental/base_comm/nic_plugin)
    endif()
endif()
