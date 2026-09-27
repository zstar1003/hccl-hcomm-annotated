# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(CTX_DIR ${SRC_DIR}/src/base_comm/resources/hccp/rdma_service/ctx)

set(rs_ut_normal_list
    stub/cmd.c
    stub/init.c
    stub/device.c
    stub/mm.c
    stub/ssl.c
    stub/drv_usr_cfg.c
    stub/ccu_api.c
    stub/ub_api.c
    stub/net_api.c
    stub/net_co.c
    stub/nda_api.c
    tc_ut_rs.c
    tc_ut_rs_ping.c
    tc_ut_rs_ping_urma.c
    tc_ut_rs_tlv.c
    tc_ut_rs_ctx.c
    tc_ut_rs_ub.c
    tc_ut_rs_nda.c
    ${SRC_DIR}/test/ut/platform/hccp/stub/crypto/encrypt.c
    ${SRC_DIR}/test/ut/platform/hccp/stub/crypto/kmc_crypt.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_service/rs_drv_rdma.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_service/rs_ping.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_service/rs_ping_roce.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_service/rs_ping_inner.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_service/rs.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_service/rs_nda.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_service/rs_comm.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_service/rs_drv_sensor.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_service/rs_rdma.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_service/rs_nda.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_service/rs_epoll.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_service/rs_drv_socket.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_service/rs_socket.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_service/tlv/rs_tlv.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_service/tlv/rs_adp_nslb.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_service/dl_ibverbs_function.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_service/dl_ibv_extend_function.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_service/tlv/dl_netco_function.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_service/hccp_dl.c
    ${SRC_DIR}/test/ut/platform/hccp/stub/roce_user/stub_hns_roce_u_cmd.c
    ${SRC_DIR}/src/base_comm/resources/hccp/common/dl_hal_function.c
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_service/ctx/dl_net_function.c
    ${SRC_DIR}/test/ut/platform/hccp/stub/devdrv_runtime_api.c
    ${CTX_DIR}/dl_ccu_function.c
    ${CTX_DIR}/dl_urma_function.c
    ${CTX_DIR}/dl_net_function.c
    ${CTX_DIR}/dl_aubdfx_function.c
    ${CTX_DIR}/rs_ub_dfx.c
    ${CTX_DIR}/rs_ub_tp.c
    ${CTX_DIR}/rs_ub.c
    ${CTX_DIR}/rs_ctx.c
    ${CTX_DIR}/rs_esched.c
    ${CTX_DIR}/rs_ping_urma.c
    ${CTX_DIR}/rs_ub_jetty.c
    ${CTX_DIR}/rs_ub_jfc.c
    ${SRC_DIR}/src/base_comm/resources/hccp/inc/private/network/log/config_log.c
    main.cc
    ut_test.cc
    ${SRC_DIR}/test/ut/platform/hccp/stub/ut_dispatch.cc
    ${SRC_DIR}/test/ut/platform/hccp/stub/dlog_stub.cc
)

add_executable(rs_ut_normal ${rs_ut_normal_list})

set_property(TARGET rs_ut_normal PROPERTY C_STANDARD 11)
target_include_directories(rs_ut_normal BEFORE PRIVATE
    ${SHADOW_OUT_DIR}
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/driver/
    ${ASCEND_CANN_PACKAGE_PATH}/include/driver/
    ${SRC_DIR}/test/stub/depends/include
    ${SRC_DIR}/test/stub/depends/pkg_inc/base
    ${SRC_DIR}/test/ut/platform/hccp/stub/roce_user
    ${SRC_DIR}/test/ut/platform/hccp/stub/
    ${SRC_DIR}/test/ut/platform/hccp/ut_include
    ${SRC_DIR}/test/ut/platform/hccp/rdma_service_normal/stub
    ${SRC_DIR}/test/ut/platform/hccp/rdma_service_normal/include
    ${SRC_DIR}/src/base_comm/resources/hccp/common
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_service/
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_service/tlv
    ${SRC_DIR}/src/base_comm/resources/hccp/rdma_service/ctx
    ${SRC_DIR}/src/base_comm/resources/hccp/inc/private/network
    ${SRC_DIR}/src/base_comm/resources/hccp/inc/private/network/log
    ${SRC_DIR}/src/base_comm/resources/hccp/inc/network
    ${SRC_DIR}/src/base_comm/resources/hccp/external_depends/inc
    ${SRC_DIR}/src/base_comm/resources/hccp/external_depends/ubengine
    ${SRC_DIR}/src/base_comm/resources/hccp/external_depends/net_adapt
    ${SRC_DIR}/src/base_comm/resources/hccp/external_depends/aubdfx
    ${SRC_DIR}/src/base_comm/resources/hccp/external_depends/ccu
    ${SRC_DIR}/src/base_comm/resources/hccp/external_depends/ibv_extend
)

target_compile_options(rs_ut_normal PRIVATE
    -U_FORTIFY_SOURCE -O0 -g
    $<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address -fno-omit-frame-pointer -static-libasan -fsanitize=undefined -static-libubsan -fsanitize=leak -static-libtsan>
)

target_compile_definitions(rs_ut_normal PRIVATE
    HNS_ROCE_LLT USE_DISPATCH_EXPECT CA_CONFIG_LLT DEVICE_TLS_CONFIG CONFIG_SSL CONFIG_RDMA RS_DEVICE_CONFIG CUSTOM_INTERFACE RDMA_LITE_STUB_TEST CCU_CONFIG_LLT
)

target_link_options(rs_ut_normal PRIVATE
    $<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address -static-libasan -fsanitize=undefined -static-libubsan -fsanitize=leak -static-libtsan>
)

target_link_libraries(rs_ut_normal PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    $<BUILD_INTERFACE:rdma_core_headers>
    $<BUILD_INTERFACE:tls_adp_headers>
    -lrt
    -pthread
    -lgcov
    -ldl
)

enable_testing()
add_test(NAME rs_ut_normal COMMAND rs_ut_normal)
