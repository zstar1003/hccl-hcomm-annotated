# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(rdma_agent_utest_list
    stub/hdc.c
    stub/rs.c
    stub/rs_ping.c
    stub/sec.c
    stub/devmm_api.c
    stub/devdrv_runtime_api.c
    stub/rs_tlv.c
    stub/rs_ctx.c
    stub/rs_nda.c
    tc_rdma_agent.c
    tc_ra_ping.c
    tc_ra_peer.c
    tc_ra_tlv.c
    tc_ra_ctx.c
    tc_ra_async.c
    tc_adp.c
    tc_ra_adp_tlv.c
    tc_hdc.c
    tc_host.c
    tc_ra_nda.c
    ${SRC_DIR}/test/ut/platform/hccp/rdma_agent/stub/dl.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/client/ra_host.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/client/ra_host_verbs.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/client/ra_common.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/client/ra_ping.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/client/ra_ctx.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/client/ra_rdma.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/client/ra_init.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/client/ra_socket.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/client/ra_tlv.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/client/ra_nda.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/client/async/ra_comm.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/client/async/ra_ctx.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/client/async/ra_socket.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/adapter/ra_adp.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/adapter/ra_adp_verbs.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/adapter/ra_adp_ping.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/adapter/ra_adp_socket.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/adapter/tlv/ra_adp_tlv.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/adapter/async/ra_adp_pool.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/adapter/async/ra_adp_async.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/adapter/ctx/ra_adp_ctx.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/hdc/ra_hdc.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/hdc/ra_hdc_ping.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/hdc/ra_hdc_ctx.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/hdc/ra_hdc_socket.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/hdc/ra_hdc_rdma.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/hdc/ra_hdc_tlv.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/hdc/async/ra_hdc_async.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/hdc/async/ra_hdc_async_ctx.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/hdc/async/ra_hdc_async_socket.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/peer/ra_peer.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/peer/ra_peer_socket.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/peer/ra_peer_ctx.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/peer/ra_peer_nda.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/comm/ra_comm.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/comm/ctx/ra_ctx_comm.c
    ${SRC_DIR}/src/base_comm/resources/hccp/common/dl_hal_function.c
    ${SRC_DIR}/test/ut/platform/hccp/stub/devdrv_runtime_api.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/hdc/ra_hdc_lite.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/hdc/ra_rdma_lite.c
    ${SRC_DIR}/src/base_comm/resources/hccp/inc/private/network/log/config_log.c
    main.cc
    ut_test.cc
    ${SRC_DIR}/test/ut/platform/hccp/stub/ut_dispatch.cc
    ${SRC_DIR}/test/ut/platform/hccp/stub/dlog_stub.cc)

add_executable(rdma_agent_utest ${rdma_agent_utest_list})

set_property(TARGET rdma_agent_utest PROPERTY C_STANDARD 11)
target_include_directories(rdma_agent_utest BEFORE PRIVATE
    ${SHADOW_OUT_DIR}
    ${SRC_DIR}/test/ut/platform/hccp/rdma_service_normal/include
    ${SRC_DIR}/test/ut/platform/hccp/rdma_agent/stub
    ${SRC_DIR}/test/ut/platform/hccp/ut_include
    ${SRC_DIR}/test/ut/platform/hccp/stub/
    ${SRC_DIR}/test/stub/depends/include
    ${SRC_DIR}/test/stub/depends/pkg_inc/base
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/client
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/adapter
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/adapter/tlv
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/adapter/async
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/adapter/ctx
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/hdc
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/hdc/async
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/peer
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/inc
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/comm
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_agent/comm/ctx
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_service
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_service/tlv
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_service/ctx
    ${SRC_DIR}/src/base_comm/resources/hccp/common
    ${SRC_DIR}/src/base_comm/resources/hccp/inc/network
    ${SRC_DIR}/src/base_comm/resources/hccp/inc/private/network
    ${SRC_DIR}/src/base_comm/resources/hccp/inc/private/network/log
    ${SRC_DIR}/src/base_comm/resources/hccp/external_depends/inc
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/driver/
    ${ASCEND_CANN_PACKAGE_PATH}/include/driver/
    ${SRC_DIR}/src/base_comm/resources/hccp/external_depends/host_lite
    ${SRC_DIR}/src/base_comm/resources/hccp/external_depends/ibv_extend
)

target_compile_options(rdma_agent_utest PRIVATE
    -U_FORTIFY_SOURCE -O0 -g
    $<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address -fno-omit-frame-pointer -static-libasan -fsanitize=undefined -static-libubsan -fsanitize=leak -static-libtsan>
)

target_compile_definitions(rdma_agent_utest PRIVATE
    HNS_ROCE_LLT
    USE_DISPATCH_EXPECT
    CONFIG_SSL
)

target_link_options(rdma_agent_utest PRIVATE
    $<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address -static-libasan -fsanitize=undefined -static-libubsan -fsanitize=leak -static-libtsan>
)

target_link_libraries(rdma_agent_utest PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    $<BUILD_INTERFACE:rdma_core_headers>
    -lrt
    -pthread
    -lgcov
    -ldl
)

enable_testing()
add_test(NAME rdma_agent_utest COMMAND rdma_agent_utest)
