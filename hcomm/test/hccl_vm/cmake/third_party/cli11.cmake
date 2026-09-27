# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

include_guard(GLOBAL)

set(CLI11_FILE "cli11-2.2.0.tar.gz")
set(CLI11_URL "https://raw.gitcode.com/src-openeuler/cli11/blobs/58c912141164a5c0f0139bfa91343fefe151d525/${CLI11_FILE}")
set(CLI11_PKG_PATH ${CMAKE_SOURCE_DIR}/third_party/${CLI11_FILE})
set(CLI11_TARGET_DIR ${CMAKE_SOURCE_DIR}/third_party/CLI11)
set(CLI11_HEADER ${CLI11_TARGET_DIR}/CLI11.hpp)
set(CLI11_INTERNAL_HEADER ${CLI11_TARGET_DIR}/CLI/CLI.hpp)

message(STATUS "[ThirdParty] CLI11_URL=${CLI11_URL}")
message(STATUS "[ThirdParty] CLI11_TARGET_DIR=${CLI11_TARGET_DIR}")

if(EXISTS "${CLI11_HEADER}" AND EXISTS "${CLI11_INTERNAL_HEADER}")
    message(STATUS "[ThirdParty] CLI11 already available in ${CLI11_TARGET_DIR}")
else()
    if(NOT EXISTS "${CLI11_PKG_PATH}")
        message(STATUS "[ThirdParty] Downloading CLI11 from ${CLI11_URL}")
        file(DOWNLOAD
            ${CLI11_URL}
            ${CLI11_PKG_PATH}
            STATUS _dl_status
            TIMEOUT 600
        )
        list(GET _dl_status 0 _dl_code)
        if(NOT _dl_code EQUAL 0)
            message(FATAL_ERROR
                "[ThirdParty] Failed to download CLI11 archive (${_dl_status}).\n"
                "URL: ${CLI11_URL}\n"
                "You can manually download and place the tarball at: ${CLI11_PKG_PATH}")
        endif()
        message(STATUS "[ThirdParty] CLI11 archive saved to ${CLI11_PKG_PATH}")
    else()
        message(STATUS "[ThirdParty] Found local CLI11 package: ${CLI11_PKG_PATH}")
    endif()

    set(_cli11_extract_dir "${CMAKE_BINARY_DIR}/_cli11_tmp")
    file(MAKE_DIRECTORY ${_cli11_extract_dir})
    file(ARCHIVE_EXTRACT
        INPUT ${CLI11_PKG_PATH}
        DESTINATION ${_cli11_extract_dir}
    )

    set(_cli11_src_include "${_cli11_extract_dir}/CLI11-2.2.0/include")
    if(NOT EXISTS "${_cli11_src_include}/CLI/CLI.hpp")
        message(FATAL_ERROR
            "[ThirdParty] CLI11 extraction failed. "
            "Expected CLI/CLI.hpp in ${_cli11_src_include}")
    endif()

    if(EXISTS "${CLI11_TARGET_DIR}/CLI")
        file(REMOVE_RECURSE ${CLI11_TARGET_DIR}/CLI)
    endif()
    file(MAKE_DIRECTORY ${CLI11_TARGET_DIR})
    file(RENAME ${_cli11_src_include}/CLI ${CLI11_TARGET_DIR}/CLI)

    file(WRITE ${CLI11_HEADER}
        "#pragma once\n"
        "// Forwarding header for backward compatibility\n"
        "#include \"CLI/CLI.hpp\"\n"
    )

    file(REMOVE_RECURSE ${_cli11_extract_dir})

    if(NOT EXISTS "${CLI11_HEADER}")
        message(FATAL_ERROR
            "[ThirdParty] CLI11 setup failed. "
            "Missing CLI11.hpp in ${CLI11_TARGET_DIR}")
    endif()
    message(STATUS "[ThirdParty] CLI11 extracted to ${CLI11_TARGET_DIR}")
endif()

add_library(cli11_lib INTERFACE)
target_include_directories(cli11_lib INTERFACE ${CLI11_TARGET_DIR})
add_library(CLI11::CLI11 ALIAS cli11_lib)
