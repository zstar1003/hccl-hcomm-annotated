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

set(SPDLOG_FILE "spdlog-v1.11.0.tar.gz")
set(SPDLOG_URL "https://raw.gitcode.com/src-openeuler/spdlog/blobs/c2dfb1aca26c607393665c836155613ff283de66/v1.11.0.tar.gz")
set(SPDLOG_PKG_PATH ${CMAKE_SOURCE_DIR}/third_party/${SPDLOG_FILE})
set(SPDLOG_TARGET_DIR ${CMAKE_SOURCE_DIR}/third_party/spdlog)
set(SPDLOG_HEADER ${SPDLOG_TARGET_DIR}/spdlog.h)

message(STATUS "[ThirdParty] SPDLOG_URL=${SPDLOG_URL}")
message(STATUS "[ThirdParty] SPDLOG_TARGET_DIR=${SPDLOG_TARGET_DIR}")

if(EXISTS "${SPDLOG_HEADER}" AND EXISTS "${SPDLOG_TARGET_DIR}/sinks/stdout_color_sinks.h")
    message(STATUS "[ThirdParty] spdlog already available in ${SPDLOG_TARGET_DIR}")
    return()
endif()

if(NOT EXISTS "${SPDLOG_PKG_PATH}")
    message(STATUS "[ThirdParty] Downloading spdlog from ${SPDLOG_URL}")
    file(DOWNLOAD
        ${SPDLOG_URL}
        ${SPDLOG_PKG_PATH}
        STATUS _dl_status
        TIMEOUT 600
    )
    list(GET _dl_status 0 _dl_code)
    if(NOT _dl_code EQUAL 0)
        message(FATAL_ERROR
            "[ThirdParty] Failed to download spdlog archive (${_dl_status}).\n"
            "URL: ${SPDLOG_URL}\n"
            "You can manually download and place the tarball at: ${SPDLOG_PKG_PATH}")
    endif()
    message(STATUS "[ThirdParty] spdlog archive saved to ${SPDLOG_PKG_PATH}")
else()
    message(STATUS "[ThirdParty] Found local spdlog package: ${SPDLOG_PKG_PATH}")
endif()

set(_spdlog_extract_dir "${CMAKE_BINARY_DIR}/_spdlog_tmp")
file(MAKE_DIRECTORY ${_spdlog_extract_dir})
file(ARCHIVE_EXTRACT
    INPUT ${SPDLOG_PKG_PATH}
    DESTINATION ${_spdlog_extract_dir}
)

set(_spdlog_src_include "${_spdlog_extract_dir}/spdlog-1.11.0/include/spdlog")
if(NOT EXISTS "${_spdlog_src_include}/spdlog.h")
    message(FATAL_ERROR
        "[ThirdParty] spdlog extraction failed. "
        "Expected spdlog.h in ${_spdlog_src_include}")
endif()

file(COPY ${_spdlog_src_include}/ DESTINATION ${SPDLOG_TARGET_DIR})

file(REMOVE_RECURSE ${_spdlog_extract_dir})

if(NOT EXISTS "${SPDLOG_HEADER}")
    message(FATAL_ERROR
        "[ThirdParty] spdlog setup failed. "
        "Missing spdlog.h in ${SPDLOG_TARGET_DIR}")
endif()
message(STATUS "[ThirdParty] spdlog extracted to ${SPDLOG_TARGET_DIR}")
