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

set(NLOHMANN_JSON_FILE "include.zip")
set(NLOHMANN_JSON_VERSION "3.11.3")
set(NLOHMANN_JSON_URL "https://gitcode.com/cann-src-third-party/json/releases/download/v${NLOHMANN_JSON_VERSION}/${NLOHMANN_JSON_FILE}")
set(NLOHMANN_JSON_PKG_PATH ${CMAKE_SOURCE_DIR}/third_party/${NLOHMANN_JSON_FILE})
set(NLOHMANN_JSON_TARGET_DIR ${CMAKE_SOURCE_DIR}/third_party/nlohmann_json)
set(NLOHMANN_JSON_HEADER ${NLOHMANN_JSON_TARGET_DIR}/json.hpp)

message(STATUS "[ThirdParty] NLOHMANN_JSON_VERSION=${NLOHMANN_JSON_VERSION}")
message(STATUS "[ThirdParty] NLOHMANN_JSON_URL=${NLOHMANN_JSON_URL}")
message(STATUS "[ThirdParty] NLOHMANN_JSON_TARGET_DIR=${NLOHMANN_JSON_TARGET_DIR}")

if(EXISTS "${NLOHMANN_JSON_HEADER}" AND EXISTS "${NLOHMANN_JSON_TARGET_DIR}/nlohmann/json.hpp")
    message(STATUS "[ThirdParty] nlohmann_json already available in ${NLOHMANN_JSON_TARGET_DIR}")
else()
    if(NOT EXISTS "${NLOHMANN_JSON_PKG_PATH}")
        message(STATUS "[ThirdParty] Downloading nlohmann_json from ${NLOHMANN_JSON_URL}")
        file(DOWNLOAD
            ${NLOHMANN_JSON_URL}
            ${NLOHMANN_JSON_PKG_PATH}
            STATUS _dl_status
            TIMEOUT 600
        )
        list(GET _dl_status 0 _dl_code)
        if(NOT _dl_code EQUAL 0)
            message(FATAL_ERROR
                "[ThirdParty] Failed to download nlohmann_json archive (${_dl_status}).\n"
                "URL: ${NLOHMANN_JSON_URL}\n"
                "You can manually download and place the zip at: ${NLOHMANN_JSON_PKG_PATH}")
        endif()
        message(STATUS "[ThirdParty] nlohmann_json archive saved to ${NLOHMANN_JSON_PKG_PATH}")
    else()
        message(STATUS "[ThirdParty] Found local nlohmann_json package: ${NLOHMANN_JSON_PKG_PATH}")
    endif()

    set(_nlohmann_extract_dir "${CMAKE_BINARY_DIR}/_nlohmann_json_tmp")
    file(MAKE_DIRECTORY ${_nlohmann_extract_dir})
    file(ARCHIVE_EXTRACT
        INPUT ${NLOHMANN_JSON_PKG_PATH}
        DESTINATION ${_nlohmann_extract_dir}
    )

    if(NOT EXISTS "${_nlohmann_extract_dir}/include/nlohmann/json.hpp")
        message(FATAL_ERROR
            "[ThirdParty] nlohmann_json extraction failed. "
            "Expected include/nlohmann/json.hpp in ${_nlohmann_extract_dir}")
    endif()

    file(MAKE_DIRECTORY ${NLOHMANN_JSON_TARGET_DIR})

    if(EXISTS "${NLOHMANN_JSON_TARGET_DIR}/nlohmann")
        file(REMOVE_RECURSE ${NLOHMANN_JSON_TARGET_DIR}/nlohmann)
    endif()
    file(RENAME ${_nlohmann_extract_dir}/include/nlohmann ${NLOHMANN_JSON_TARGET_DIR}/nlohmann)

    if(EXISTS "${NLOHMANN_JSON_HEADER}")
        file(REMOVE "${NLOHMANN_JSON_HEADER}")
    endif()
    file(COPY ${NLOHMANN_JSON_TARGET_DIR}/nlohmann/json.hpp DESTINATION ${NLOHMANN_JSON_TARGET_DIR})

    file(REMOVE_RECURSE ${_nlohmann_extract_dir})

    if(NOT EXISTS "${NLOHMANN_JSON_HEADER}")
        message(FATAL_ERROR
            "[ThirdParty] nlohmann_json setup failed. "
            "Missing json.hpp in ${NLOHMANN_JSON_TARGET_DIR}")
    endif()
    message(STATUS "[ThirdParty] nlohmann_json extracted to ${NLOHMANN_JSON_TARGET_DIR}")
endif()

if(NOT TARGET JSON::JSON)
    add_library(nlohmann_json_lib INTERFACE)
    target_include_directories(nlohmann_json_lib INTERFACE
        ${NLOHMANN_JSON_TARGET_DIR}
    )
    add_library(JSON::JSON ALIAS nlohmann_json_lib)
endif()
