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

set(CANN_CMAKE_VERSION_TAG "master-044")
set(CANN_CMAKE_URL "https://raw.gitcode.com/cann/cmake/archive/refs/heads/${CANN_CMAKE_VERSION_TAG}.tar.gz")
set(CANN_CMAKE_ARCHIVE_FILE "cmake-${CANN_CMAKE_VERSION_TAG}.tar.gz")
set(CANN_CMAKE_PKG_PATH ${CMAKE_SOURCE_DIR}/third_party/${CANN_CMAKE_ARCHIVE_FILE})
set(CANN_CMAKE_TARGET_DIR ${CMAKE_SOURCE_DIR}/third_party/cann-cmake)
set(CANN_CMAKE_SCRIPT ${CANN_CMAKE_TARGET_DIR}/scripts/signtool/image_extract/ci_img_headler.py)

message(STATUS "[ThirdParty] CANN_CMAKE_VERSION_TAG=${CANN_CMAKE_VERSION_TAG}")
message(STATUS "[ThirdParty] CANN_CMAKE_URL=${CANN_CMAKE_URL}")
message(STATUS "[ThirdParty] CANN_CMAKE_TARGET_DIR=${CANN_CMAKE_TARGET_DIR}")

if(EXISTS "${CANN_CMAKE_SCRIPT}")
    message(STATUS "[ThirdParty] cann-cmake already available in ${CANN_CMAKE_TARGET_DIR}")
else()
    if(NOT EXISTS "${CANN_CMAKE_PKG_PATH}")
        message(STATUS "[ThirdParty] Downloading cann-cmake from ${CANN_CMAKE_URL}")
        file(DOWNLOAD
            ${CANN_CMAKE_URL}
            ${CANN_CMAKE_PKG_PATH}
            STATUS _dl_status
            TIMEOUT 600
        )
        list(GET _dl_status 0 _dl_code)
        if(NOT _dl_code EQUAL 0)
            file(REMOVE "${CANN_CMAKE_PKG_PATH}")
            message(FATAL_ERROR
                "[ThirdParty] Failed to download cann-cmake archive (${_dl_status}).\n"
                "URL: ${CANN_CMAKE_URL}\n"
                "You can manually download and place at: ${CANN_CMAKE_PKG_PATH}")
        endif()
        message(STATUS "[ThirdParty] cann-cmake archive saved to ${CANN_CMAKE_PKG_PATH}")
    else()
        message(STATUS "[ThirdParty] Found local cann-cmake package: ${CANN_CMAKE_PKG_PATH}")
    endif()

    set(_cann_cmake_extract_dir "${CMAKE_BINARY_DIR}/_cann_cmake_tmp")
    file(MAKE_DIRECTORY ${_cann_cmake_extract_dir})
    file(ARCHIVE_EXTRACT
        INPUT ${CANN_CMAKE_PKG_PATH}
        DESTINATION ${_cann_cmake_extract_dir}
    )
    file(GLOB _cann_cmake_extracted_entries "${_cann_cmake_extract_dir}/*")
    list(GET _cann_cmake_extracted_entries 0 _cann_cmake_extracted_dir)

    if(EXISTS "${CANN_CMAKE_TARGET_DIR}")
        file(REMOVE_RECURSE ${CANN_CMAKE_TARGET_DIR})
    endif()
    file(RENAME ${_cann_cmake_extracted_dir} ${CANN_CMAKE_TARGET_DIR})
    file(REMOVE_RECURSE ${_cann_cmake_extract_dir})

    if(NOT EXISTS "${CANN_CMAKE_SCRIPT}")
        message(FATAL_ERROR
            "[ThirdParty] cann-cmake extraction failed. "
            "Missing ${CANN_CMAKE_SCRIPT}")
    endif()
    message(STATUS "[ThirdParty] cann-cmake extracted to ${CANN_CMAKE_TARGET_DIR}")
endif()

set(CANN_CMAKE_DIR "${CANN_CMAKE_TARGET_DIR}" CACHE PATH "Path to cann-cmake")
