# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(orion_include_list
    #2.0 - checker
    ${HCCL_ALG_ROOT}/adapter_v2
    ${HCCL_ALG_ROOT}/adapter_v2/stub_ccu
    ${HCCL_ALG_ROOT}/adapter_v2/adapter_rts
    ${HCCL_ALG_V2_CODE_ROOT}/primitive
)

set(analyzer_test_list
    ${HCCL_ALG_ROOT}/checker/cmd/src/main.cc
    ${HCCL_ALG_ROOT}/checker/cmd/src/CheckerCmd.cc
    ${CMAKE_BINARY_DIR}/proto/alg_analyzer_proto/proto/cmd.pb.cc
)

add_executable(hccl_alg_analyzer_test
    ${analyzer_test_list}
)

target_compile_definitions(hccl_alg_analyzer_test PRIVATE
    google=ascend_private
    _GLIBCXX_USE_CXX11_ABI=0
)

#头文件搜索路径
target_include_directories(hccl_alg_analyzer_test PRIVATE
    ${UTILS_PUBLIC_INCLUDE_LIST}
    ${UTILS_INNER_INCLUDE_LIST}

    ${HCOMM_PUBLIC_INCLUDE_LIST}
    ${HCOMM_PLATFORM_INCLUDE_LIST}
    ${HCOMM_ALGORITHM_INCLUDE_LIST}
    ${HCOMM_COMMON_INCLUDE_LIST}
    ${HCOMM_FRAMEWORK_INCLUDE_LIST}
    ${HCOMM_LEGACY_INCLUDE_LIST}

    ${CMAKE_BINARY_DIR}/proto/alg_analyzer_proto/proto

    ${ASCEND_CANN_INCLUDE_LIST}
)

target_compile_options(hccl_alg_analyzer_test PRIVATE
    -U_FORTIFY_SOURCE
    -Os
    -O2
    -pipe
)

target_link_libraries(hccl_alg_analyzer_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    hccl_alg_adaption
    hccl_alg_analyzer_legacy
    -pthread
    ascend_protobuf
    ${ASCEND_CANN_PACKAGE_PATH}/lib64/libc_sec.so
    -lrt
    -ldl
)

enable_testing()
add_test(NAME hccl_alg_analyzer_test COMMAND hccl_alg_analyzer_test)
