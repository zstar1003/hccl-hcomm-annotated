# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(hccp_utest_list
    main.cc
    stub/tsd.c
    stub/mm.c
    stub/hccp_pub.c
    ${SRC_DIR}/test/ut/platform/hccp/stub/ut_dispatch.cc
    ${SRC_DIR}/src/base_comm/resources/hccp/hccp_service/param.c
    ${SRC_DIR}/src/base_comm/resources/hccp/common/dl_hal_function.c
    ${SRC_DIR}/src/base_comm/resources/hccp/common/ascend_hal_dl.c
    ${SRC_DIR}/src/base_comm/resources/hccp/hccp_service/main.c
    ${SRC_DIR}/src/base_comm/resources/hccp/inc/private/network/log/config_log.c
    st_test.cc
    tc_hccp.c
    )

add_executable(hccp_utest ${hccp_utest_list})

set_property(TARGET hccp_utest PROPERTY C_STANDARD 11)
target_include_directories(hccp_utest BEFORE PRIVATE
    ${SHADOW_OUT_DIR}
    stub
    ${SRC_DIR}/test/ut/platform/hccp/ut_include
    ${SRC_DIR}/test/stub/depends/include
    ${SRC_DIR}/test/stub/depends/pkg_inc/base
    ${SRC_DIR}/test/stub/depends/pkg_inc/aicpu/tsd
    ${SRC_DIR}/test/stub/depends/pkg_inc/aicpu/
    ${SRC_DIR}/test/ut/hccp/rdma_service_normal/rdma_service_normal/include
    ${SRC_DIR}/test/stub/depends/include
    ${SRC_DIR}/src/base_comm/resources/hccp/hccp_service
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/adapter
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/adapter/async
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/hdc
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/inc
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_service
    ${SRC_DIR}/src/base_comm/resources/hccp/common
    ${SRC_DIR}/src/base_comm/resources/hccp/inc/private/network
    ${SRC_DIR}/src/base_comm/resources/hccp/inc/private/network/log
    ${SRC_DIR}/src/base_comm/resources/hccp/inc/network
    ${SRC_DIR}/src/base_comm/resources/hccp/external_depends/inc
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/driver/
    ${ASCEND_CANN_PACKAGE_PATH}/include/driver/
    ${SRC_DIR}/src/base_comm/resources/hccp/external_depends/host_lite
    ${SRC_DIR}/src/base_comm/resources/hccp/external_depends/rdma-core/providers/hns
    ${SRC_DIR}/src/base_comm/resources/hccp/external_depends/rdma-core/include
)

target_compile_options(hccp_utest PRIVATE
    -U_FORTIFY_SOURCE
    -O0 -g --coverage -fprofile-arcs -ftest-coverage -fno-access-control
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>
    -Wall
    -Wextra
    -Werror
    -pedantic
    -Wshadow
    $<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address -fno-omit-frame-pointer -static-libasan -fsanitize=undefined -static-libubsan -fsanitize=leak -static-libtsan>
)

target_compile_definitions(hccp_utest PRIVATE
    HNS_ROCE_LLT
    USE_DISPATCH_EXPECT
    CONFIG_HCCP_LLT
)

target_link_options(hccp_utest PRIVATE
    $<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address -static-libasan -fsanitize=undefined -static-libubsan -fsanitize=leak -static-libtsan>
)

target_link_libraries(hccp_utest PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    $<BUILD_INTERFACE:rdma_core_headers>
    -lrt
    -pthread
    -lgcov
    -ldl
)

enable_testing()
add_test(NAME hccp_utest COMMAND hccp_utest)
