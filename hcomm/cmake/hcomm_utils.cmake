# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

include_guard(GLOBAL)

unset(hcomm_utils_FOUND CACHE)
unset(TLS_ADP_LIBRARY CACHE)

if(NOT CANN_3RD_LIB_PATH)
    set(CANN_3RD_LIB_PATH ${HCOMM_DIR}/third_party)
endif()

if(PRODUCT_SIDE STREQUAL "device")
    set(HCOMM_UTILS_INSTALL_PATH ${CANN_3RD_LIB_PATH}/lib_cache/device/hcomm_utils)
    set(HCOMM_UTILS_PKG_PATH ${CANN_3RD_LIB_PATH}/pkg/device)
    set(HCOMM_UTILS_ARCH "${CMAKE_HOST_SYSTEM_PROCESSOR}")
else()
    set(HCOMM_UTILS_INSTALL_PATH ${CANN_3RD_LIB_PATH}/lib_cache/hcomm_utils)
    set(HCOMM_UTILS_PKG_PATH ${CANN_3RD_LIB_PATH}/pkg)
    set(HCOMM_UTILS_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
endif()

set(HCOMM_UTILS_VERSION "9.2.0")
set(HCOMM_UTILS_FILE "cann-hcomm-utils_${HCOMM_UTILS_VERSION}_linux-${HCOMM_UTILS_ARCH}.tar.gz")
set(HCOMM_UTILS_URL "https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260729_newest/${HCOMM_UTILS_FILE}")

find_library(TLS_ADP_LIBRARY
    NAMES libtls_adp.so
    PATH_SUFFIXES lib
    PATHS ${HCOMM_UTILS_INSTALL_PATH}/${PRODUCT_SIDE}
    NO_DEFAULT_PATH
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(hcomm_utils
    FOUND_VAR
        hcomm_utils_FOUND
    REQUIRED_VARS
        TLS_ADP_LIBRARY
)

if(hcomm_utils_FOUND AND NOT FORCE_REBUILD_CANN_3RD)
    message(STATUS "[ThirdParty] hcomm_utils found in ${TLS_ADP_LIBRARY}, and not force rebuild cann third_party")
else()
    file(GLOB HCOMM_UTILS_GLOB_PKG
        ${CANN_3RD_LIB_PATH}/cann-hcomm-utils_*_linux-${HCOMM_UTILS_ARCH}.tar.gz
    )
    if(EXISTS ${HCOMM_UTILS_GLOB_PKG})
        # 离线编译场景，优先使用已下载的包（忽略版本号）
        message(STATUS "[ThirdParty] Found local hcomm_utils package (ignore version): ${HCOMM_UTILS_GLOB_PKG}")
        set(HCOMM_UTILS_PROJECT_URL ${HCOMM_UTILS_GLOB_PKG})
        set(HCOMM_UTILS_FULLPATH ${HCOMM_UTILS_GLOB_PKG})
    elseif(EXISTS ${HCOMM_UTILS_PKG_PATH}/${HCOMM_UTILS_FILE})
        # 离线编译场景，优先使用已下载的包
        message(STATUS "[ThirdParty] Found local hcomm_utils package: ${HCOMM_UTILS_PKG_PATH}/${HCOMM_UTILS_FILE}")
        set(HCOMM_UTILS_PROJECT_URL ${HCOMM_UTILS_PKG_PATH}/${HCOMM_UTILS_FILE})
        set(HCOMM_UTILS_FULLPATH ${HCOMM_UTILS_PKG_PATH}/${HCOMM_UTILS_FILE})
    else()
        # 下载并解压
        message(STATUS "[ThirdParty] Downloading hcomm_utils from ${HCOMM_UTILS_URL}")
        set(HCOMM_UTILS_PROJECT_URL ${HCOMM_UTILS_URL})
        set(HCOMM_UTILS_FULLPATH ${HCOMM_UTILS_PKG_PATH}/${HCOMM_UTILS_FILE})
    endif()

    if(EXISTS ${HCOMM_UTILS_GLOB_PKG})
        set(HCOMM_UTILS_URL_HASH "")    # 忽略版本号，不校验哈希值
    elseif(HCOMM_UTILS_ARCH MATCHES "aarch64|ARM64|arm64")
        set(HCOMM_UTILS_URL_HASH "SHA256=f33f9bc5b6f6e2c74c8ac461a1f0e368219500ab9361cf63b73d1ac99567b3f3")
    else()
        set(HCOMM_UTILS_URL_HASH "SHA256=f2c6f5e8f4ba3561e39625f0773abb98b94923e31a7c20010959566aaacc29df")
    endif()

    include(ExternalProject)
    ExternalProject_Add(hcomm_utils
        URL ${HCOMM_UTILS_PROJECT_URL}
        URL_HASH ${HCOMM_UTILS_URL_HASH}
        DOWNLOAD_DIR ${HCOMM_UTILS_PKG_PATH}
        SOURCE_DIR ${HCOMM_UTILS_INSTALL_PATH}
        CONFIGURE_COMMAND tar -xf ${HCOMM_UTILS_FULLPATH} --overwrite --strip-components=2 -C <SOURCE_DIR>
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        DOWNLOAD_NO_EXTRACT TRUE
        DOWNLOAD_NO_PROGRESS TRUE
        TIMEOUT 300
    )
endif()

if(NOT EXISTS ${HCOMM_UTILS_INSTALL_PATH}/${PRODUCT_SIDE}/include)
    file(MAKE_DIRECTORY "${HCOMM_UTILS_INSTALL_PATH}/${PRODUCT_SIDE}/include/kms")
    file(MAKE_DIRECTORY "${HCOMM_UTILS_INSTALL_PATH}/${PRODUCT_SIDE}/include/legacy")
    file(MAKE_DIRECTORY "${HCOMM_UTILS_INSTALL_PATH}/${PRODUCT_SIDE}/include/tls_adp")
endif()

# 创建导入的目标
add_library(ascend_kms_headers INTERFACE IMPORTED)
add_dependencies(ascend_kms_headers hcomm_utils)
set_target_properties(ascend_kms_headers PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${HCOMM_UTILS_INSTALL_PATH}/${PRODUCT_SIDE}/include/kms"
)

add_library(ascend_kms SHARED IMPORTED)
add_dependencies(ascend_kms hcomm_utils)
set_target_properties(ascend_kms PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${HCOMM_UTILS_INSTALL_PATH}/${PRODUCT_SIDE}/include/kms"
    IMPORTED_LOCATION "${HCOMM_UTILS_INSTALL_PATH}/${PRODUCT_SIDE}/lib/libascend_kms.so"
)

add_library(tls_adp_headers INTERFACE IMPORTED)
add_dependencies(tls_adp_headers hcomm_utils)
set_target_properties(tls_adp_headers PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${HCOMM_UTILS_INSTALL_PATH}/${PRODUCT_SIDE}/include/tls_adp"
)

add_library(tls_adp SHARED IMPORTED)
add_dependencies(tls_adp hcomm_utils)
set_target_properties(tls_adp PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${HCOMM_UTILS_INSTALL_PATH}/${PRODUCT_SIDE}/include/tls_adp"
    IMPORTED_LOCATION "${HCOMM_UTILS_INSTALL_PATH}/${PRODUCT_SIDE}/lib/libtls_adp.so"
)

add_library(hccl_legacy_headers INTERFACE IMPORTED)
add_dependencies(hccl_legacy_headers hcomm_utils)
set_target_properties(hccl_legacy_headers PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${HCOMM_UTILS_INSTALL_PATH}/${PRODUCT_SIDE}/include/legacy"
)

if(PRODUCT_SIDE STREQUAL "host")
    add_library(hccl_legacy SHARED IMPORTED)
    add_dependencies(hccl_legacy hcomm_utils)
    set_target_properties(hccl_legacy PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${HCOMM_UTILS_INSTALL_PATH}/${PRODUCT_SIDE}/include/legacy"
        IMPORTED_LOCATION "${HCOMM_UTILS_INSTALL_PATH}/${PRODUCT_SIDE}/lib/libhccl_legacy.so"
    )

    # 安装 host 侧需要的库
    set(HCOMM_UTILS_HOST_LIB_FILES
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/libtls_adp.so
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/libhccl_legacy.so
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/MemSet_dynamic_AtomicAddrClean_1_ascend310p3.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/MemSet_dynamic_AtomicAddrClean_1_ascend910.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/MemSet_dynamic_AtomicAddrClean_1_ascend910b.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_add_float16_v51.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_add_float16_v80.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_add_float32_v51.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_add_int32_v51.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_add_int32_v80.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_add_int64_v80.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_add_int64_v81.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_add_int8_v51.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_add_int8_v80.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_maximum_float16_v51.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_maximum_float16_v80.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_maximum_float32_v51.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_maximum_float32_v80.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_maximum_int32_v51.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_maximum_int32_v80.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_maximum_int64_v80.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_maximum_int64_v81.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_maximum_int8_v51.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_maximum_int8_v80.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_minimum_float16_v51.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_minimum_float16_v80.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_minimum_float32_v51.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_minimum_float32_v80.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_minimum_int32_v51.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_minimum_int32_v80.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_minimum_int64_v80.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_minimum_int64_v81.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_minimum_int8_v51.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_minimum_int8_v80.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_mul_float16_v51.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_mul_float16_v80.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_mul_float16_v81_910B1.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_mul_float16_v81_910B2.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_mul_float16_v81_910B3.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_mul_float16_v81_910B4.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_mul_float32_v51.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_mul_float32_v80.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_mul_float32_v81_910B1.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_mul_float32_v81_910B2.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_mul_float32_v81_910B3.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_mul_float32_v81_910B4.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_mul_int32_v51.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_mul_int32_v80.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_mul_int32_v81_910B1.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_mul_int32_v81_910B2.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_mul_int32_v81_910B3.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_mul_int32_v81_910B4.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_mul_int64_v80.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_mul_int64_v81.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_mul_int8_v51.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_mul_int8_v80.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_mul_int8_v81_910B1.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_mul_int8_v81_910B2.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_mul_int8_v81_910B3.o
        ${HCOMM_UTILS_INSTALL_PATH}/host/lib/dynamic_mul_int8_v81_910B4.o
    )

    install(FILES ${HCOMM_UTILS_HOST_LIB_FILES}
        DESTINATION ${INSTALL_LIBRARY_DIR} ${INSTALL_OPTIONAL}
        COMPONENT hcomm
    )
endif()
