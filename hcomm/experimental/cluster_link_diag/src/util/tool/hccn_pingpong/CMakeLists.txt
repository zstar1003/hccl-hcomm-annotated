# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_library(hccn_pingpong_rpc STATIC
    rpc_func.cpp
    hccn_pingpong.cpp)

set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
if(NOT ASCEND_HOME_PATH)
    set(ASCEND_HOME_PATH "/usr/local/Ascend")
endif()

set(ASCEND_CANN_PATH "$ENV{ASCEND_CANN_PATH}")
if(NOT ASCEND_CANN_PATH)
    set(ASCEND_CANN_PATH "${ASCEND_HOME_PATH}")
endif()

set(ASCEND_CANN_CANDIDATES
    "${ASCEND_CANN_PATH}"
    "${ASCEND_HOME_PATH}"
    "${ASCEND_HOME_PATH}/cann"
    "${ASCEND_HOME_PATH}/aarch64-linux"
    "${ASCEND_HOME_PATH}/ascend-toolkit/latest/aarch64-linux")
file(GLOB ASCEND_TOOLKIT_CANDIDATES LIST_DIRECTORIES true
    "${ASCEND_HOME_PATH}/ascend-toolkit/*/aarch64-linux")
list(APPEND ASCEND_CANN_CANDIDATES ${ASCEND_TOOLKIT_CANDIDATES})

foreach(candidate ${ASCEND_CANN_CANDIDATES})
    if(EXISTS "${candidate}/include/hccl/hccn_rping.h" AND
       EXISTS "${candidate}/include/acl/acl.h" AND
       EXISTS "${candidate}/lib64/libhccl_plf.so" AND
       EXISTS "${candidate}/lib64/libascendcl.so")
        set(ASCEND_CANN_PATH "${candidate}")
        break()
    endif()
endforeach()

if(NOT EXISTS "${ASCEND_CANN_PATH}/include/hccl/hccn_rping.h")
    message(FATAL_ERROR
        "Cannot find Ascend CANN/HCCL/ACL under ASCEND_CANN_PATH=${ASCEND_CANN_PATH} or ASCEND_HOME_PATH=${ASCEND_HOME_PATH}")
endif()

set(ASCEND_HCCL_LIB_CANDIDATES
    "${ASCEND_CANN_PATH}/lib64"
    "${ASCEND_HOME_PATH}/ascend-toolkit/7.0.1.5/fwkacllib/lib64"
    "${ASCEND_HOME_PATH}/ascend-toolkit/7.0.1.5/atc/lib64")
set(ASCEND_HCCL_LIB_DIR "")
foreach(candidate ${ASCEND_HCCL_LIB_CANDIDATES})
    if(EXISTS "${candidate}/libhccl.so")
        set(ASCEND_HCCL_LIB_DIR "${candidate}")
        break()
    endif()
endforeach()

if(NOT ASCEND_HCCL_LIB_DIR)
    message(FATAL_ERROR
        "Cannot find libhccl.so under ASCEND_CANN_PATH=${ASCEND_CANN_PATH} or ASCEND_HOME_PATH=${ASCEND_HOME_PATH}")
endif()

target_include_directories(hccn_pingpong_rpc PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ASCEND_CANN_PATH}/include)

target_link_directories(hccn_pingpong_rpc PUBLIC
    ${ASCEND_CANN_PATH}/lib64
    ${ASCEND_HCCL_LIB_DIR})

target_link_libraries(hccn_pingpong_rpc PUBLIC
    hccl
    hccl_plf
    ascendcl)
