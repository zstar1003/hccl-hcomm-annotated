# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_custom_target(hccl-0.1.0-py3-none-any.whl ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/dist/hccl-0.1.0-py3-none-any.whl
)

# 读取sys_version.conf中版本号并打包到whl包中的hccl目录下
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/../version.cmake FILE_CONTENT)

string(REGEX MATCH "VERSION \"([^)]+)\"" MATCHED_VERSION "${FILE_CONTENT}")
if (NOT MATCHED_VERSION)
    message(FATAL_ERROR "can not find version num from version.cmake")
endif()

set(VERSION_NUMBER ${CMAKE_MATCH_1})

message(STATUS "Extracted version number: ${VERSION_NUMBER}")

set(SYS_VERSION ${VERSION_NUMBER})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/sys_version.py.in ${CMAKE_CURRENT_BINARY_DIR}/hccl/sys_version.py @ONLY)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/dist/hccl-0.1.0-py3-none-any.whl
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/setup.py ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/hccl ${CMAKE_CURRENT_BINARY_DIR}/hccl
	
    COMMAND ${HI_PYTHON} setup.py bdist_wheel
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/dist/hccl-0.1.0-py3-none-any.whl
    DESTINATION ${INSTALL_LIBRARY_DIR} ${INSTALL_OPTIONAL}
    COMPONENT hcomm
)
