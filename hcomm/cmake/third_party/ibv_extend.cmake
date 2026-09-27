# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

message(STATUS "Build third party library ibv_extend")

set(IBV_EXTEND_DRIVER_DIR "${HCOMM_DIR}/../driver/driver/src/custom/nda/ibv_extend")
set(IBV_EXTEND_FALLBACK_DIR "${HCOMM_DIR}/src/base_comm/resources/hccp/external_depends/ibv_extend")

if(IS_DIRECTORY "${IBV_EXTEND_DRIVER_DIR}")
    set(IBV_EXTEND_INCLUDE_DIR "${CMAKE_BINARY_DIR}/ibv_extend")
    file(GLOB IBV_EXTEND_DRIVER_CONTENTS "${IBV_EXTEND_DRIVER_DIR}/*")
    if(NOT IBV_EXTEND_DRIVER_CONTENTS)
        message(FATAL_ERROR "${IBV_EXTEND_DRIVER_DIR} have no ibv_extend headers")
    endif()
    file(COPY "${IBV_EXTEND_DRIVER_DIR}/" DESTINATION "${IBV_EXTEND_INCLUDE_DIR}")
    message(STATUS "Successfully copied ${IBV_EXTEND_DRIVER_DIR} to ${IBV_EXTEND_INCLUDE_DIR}.")
    message(STATUS "built ibv_extend headers.")
else()
    set(IBV_EXTEND_INCLUDE_DIR "${IBV_EXTEND_FALLBACK_DIR}")
    message(STATUS "IBV_EXTEND_INCLUDE_DIR IS ${IBV_EXTEND_INCLUDE_DIR}")
endif()