# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(DEFAULT_BUILD_TYPE "Release")

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the build type: Release/Debug" FORCE)
endif()

if(CUSTOM_ASCEND_CANN_PACKAGE_PATH)
    set(ASCEND_CANN_PACKAGE_PATH  ${CUSTOM_ASCEND_CANN_PACKAGE_PATH})
elseif(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CANN_PACKAGE_PATH  $ENV{ASCEND_HOME_PATH})
elseif(DEFINED ENV{ASCEND_OPP_PATH})
    get_filename_component(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_OPP_PATH}/.." ABSOLUTE)
else()
    set(ASCEND_CANN_PACKAGE_PATH  "/usr/local/Ascend/ascend-toolkit/latest")
endif()

set(ASCEND_MOCKCPP_PACKAGE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

if (CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
    set(CMAKE_INSTALL_PREFIX     "${CMAKE_CURRENT_SOURCE_DIR}/output"  CACHE STRING "path for install()" FORCE)
endif ()

set(HI_PYTHON                     "python3"                       CACHE   STRING   "python executor")

message(STATUS "config.cmake BUILD_OPEN_PROJECT=${BUILD_OPEN_PROJECT}")
message(STATUS "config.cmake PRODUCT=${PRODUCT} PRODUCT_SIDE=${PRODUCT_SIDE}")

set(INSTALL_LIBRARY_DIR        ${CMAKE_SYSTEM_PROCESSOR}-linux/lib64)
set(INSTALL_INCLUDE_DIR        ${CMAKE_SYSTEM_PROCESSOR}-linux/include)
set(INSTALL_PKG_INCLUDE_DIR    ${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc)
set(INSTALL_CCL_KERNEL_JSON_DIR opp/built-in/op_impl/aicpu)
set(INSTALL_DPU_KERNEL_JSON_DIR opp/built-in/op_impl/dpu)
set(INSTALL_DEVICE_TAR_DIR compat)

if(ENABLE_BUILD_AARCH)
    set(INSTALL_DEVICE_LIBRARY_DIR ${CMAKE_SYSTEM_PROCESSOR}-linux/devlib/device)
else()
    set(INSTALL_DEVICE_LIBRARY_DIR ${CMAKE_HOST_SYSTEM_PROCESSOR}-linux/devlib/device)
endif()

set(CMAKE_SKIP_RPATH TRUE)
