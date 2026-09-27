# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(src_to_test_list
    main.cc
    hccl_api_base_test.cc
    ut_CcuComponent_test.cc
    ut_HcommCcuControlApi_test.cc
    ut_HcommCcuControlApi_ckenop_test.cc
    ut_HcommCcuResDescApi_test.cc
    ut_HcommCcuCascCntApi_test.cc
    ut_HcommCcuKernelQueryResReq_test.cc
    ut_CollectLoopGroupProfiling_test.cc

    mocks/ccu_device_mock_utils.cc
    mocks/ccu_drv_handle_mock.cc
    mocks/ccu_dev_mgr_mock.cc
    ut_CcuConn_test.cc
    ut_ccu_microcode.cc
    ut_ccu_microcode_loadx_storex.cc

    ccu_kernel_impl/ccu_groupcopy_alg.cc
    ccu_kernel_impl/ccu_groupcopy_kernel.cc
    # ut_ccu_mgr_test.cc  
    ./ccu_representation/interface/ut_ccu_interface_assist.cc
    ./ccu_device/ut_ccu_res_batch_allocator.cc
    ./ccu_device/ccu_comp/ccu_channel/ccu_pfe/ut_ccu_pfe_cfg_mgr.cc
    ./ccu_device/ccu_comp/ccu_channel/ccu_channel_ctx_v2/ut_ccu_wqebb_mgr_v2.cc
    ./ccu_device/ccu_comp/ccu_channel/ccu_channel_ctx_v2/ut_ccu_jetty_ctx_mgr_v2.cc
    ./ccu_device/ccu_comp/ccu_channel/ccu_channel_ctx_v2/ut_ccu_channel_ctx_mgr_v2.cc
    ./ccu_device/ccu_comp/ut_ccu_res_allocator.cc
    ./ccu_device/ut_ccu_res_specs.cc
    ./ut_ccu_transport_test.cc
    ./ut_ccu_kernel_declared_locxns_test.cc
)

add_executable(hccl_utest_comms_ccu_features ${src_to_test_list})

#头文件搜索路径
# base_comm 的 ccu_channel 等路径需置于 legacy 之前，避免同名头文件（ccu_wqebb_mgr.h/ccu_jetty_ctx_mgr.h）被 legacy 版本覆盖
target_include_directories(hccl_utest_comms_ccu_features PRIVATE
    BEFORE
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/pub_inc/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_device/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_device/ccu_comp/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_device/ccu_comp/ccu_channel/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_device/ccu_comp/ccu_channel/ccu_pfe/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_device/ccu_comp/ccu_channel/ccu_channel_ctx_v1/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_device/ccu_comp/ccu_channel/ccu_channel_ctx_v2/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_transport/
    ${UT_COMMON_INCLUDE_LIST}
    ${HCOMM_DIR}/src/base_comm/primitives/api_c_adpt/ccu
    ${HCOMM_DIR}/pkg_inc/hcomm/ccu
)

target_compile_definitions(hccl_utest_comms_ccu_features PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
    google=ascend_private
    -DOPEN_HCCL_TEST
    OPEN_BUILD_PROJECT
)

target_compile_options(hccl_utest_comms_ccu_features PRIVATE
    -U_FORTIFY_SOURCE
    -fno-var-tracking-assignments
    -Os
    -O0 -g --coverage -fprofile-arcs -ftest-coverage -fno-access-control
    $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address -fsanitize-recover=address,all -fno-omit-frame-pointer>
)

target_link_libraries(hccl_utest_comms_ccu_features PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    $<BUILD_INTERFACE:rdma_core_headers>
    $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address>
    -pthread
    hccl_llt
    -Wl,--whole-archive
    -Wl,--no-whole-archi
    -lgcov
    -lrt
    -ldl
)

add_dependencies(hccl_utest_comms_ccu_features hccl_llt)

enable_testing()
add_test(NAME hccl_utest_comms_ccu_features COMMAND hccl_utest_comms_ccu_features)

add_subdirectory(ccu_representation/reps/sync)
add_subdirectory(ccu_representation/reps/control)
add_subdirectory(ccu_representation/reps/common)
add_subdirectory(ccu_representation/reps/data)
add_subdirectory(ccu_representation/reps/arithmetic)
add_subdirectory(ccu_representation/reps/loop)
add_subdirectory(ccu_representation/reps/shift)
add_subdirectory(ccu_representation/reps/translator)
add_subdirectory(ccu_representation/interface)
add_subdirectory(ccu_microcode_opt)
