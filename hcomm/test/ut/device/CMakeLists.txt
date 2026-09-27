# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(hccl_utest_device_src_list
    main.cc
    # ut_aiv_device_utest.cc
    ut_stream_device_utest.cc
    ut_communicator_device_utest.cc
    ut_zero_copy_device_utest.cc
    ut_engine_aicpu_interface.cc
    ut_aicpu_one_side_service.cc
    ut_hccl_aicpu_utils.cc
    ut_ranktable_concise_device.cc
    ut_launch_aicpu.cc
    ut_hccl_aicpu_interface.cc
    ut_topoinfo_detect_device.cc

    ${HCCL_CODE_ROOT}/src/legacy/ascend910/common/stream/stream_utils_aicpu.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/common/debug/profiling/plugin_runner_device.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/common/src/config/env_config_aicpu.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/op_base/src/op_base_device.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/communicator/hccl_comm_device.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/communicator/impl/hccl_communicator_device.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/communicator/impl/hccl_communicator_attrs_device.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/communicator/impl/zero_copy/zero_copy_address_mgr_device.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/hccl_aiv_device.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/hccl_alg_device.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/common/src/topo/topo_device/topoinfo_ranktableConcise_device.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/common/src/topo/topo_device/topoinfo_detect_device.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/common/src/topo/topo_device/topoinfo_ranktableParser_device.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/common/src/topo/topo_device/topoinfo_ranktable_partition_device.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/common/src/topo/topo_device/topoinfo_ranktableStandard_device.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_to_all/coll_all_to_all_executor_aicpu.cc
    ${HCCL_CODE_ROOT}/src/base_comm/resources/comm_engine_res/threads/device/aicpu_thread_kernel.cc
    ${HCCL_CODE_ROOT}/src/base_comm/resources/comm_engine_res/threads/device/aicpu_thread_process.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/device/framework/aicpu_one_side_service.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/device/hccl_aicpu_interface.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/common/launch_aicpu/launch_aicpu.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/common/adapter/adapter_rts.cc
)

add_executable(hccl_utest_device ${hccl_utest_device_src_list})

target_include_directories(hccl_utest_device PRIVATE
    ${UT_COMMON_INCLUDE_LIST}
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/communicator/impl/independent_op/resource/device/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/communicator/impl/independent_op/resource/engine/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/communicator/impl/independent_op/device/
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/common/
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/communicator
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/communicator/device/
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/communicator/group_schedule_mgr
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/communicator/device/resource_mgr/local/my_rank/comm_engine_reses
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/communicator/device/resource_mgr/remote
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/communicator/device/c_adpt
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_engine/notify/
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_engine/
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_mems/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/comm_engine_res/launch/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/comm_engine_res/threads/device/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/comm_engine_res/threads/
)
target_compile_definitions(hccl_utest_device PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
    HALF_T=float
    google=ascend_private
    CCL_KERNEL
    CCL_LLT
    OPEN_HCCL_TEST
    OPEN_BUILD_PROJECT
)
target_compile_options(hccl_utest_device PRIVATE
    -U_FORTIFY_SOURCE
    -fno-var-tracking-assignments
    -Os
    -O0 -g --coverage -fprofile-arcs -ftest-coverage -fno-access-control
    $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address -fsanitize-recover=address,all -fno-omit-frame-pointer>
)

target_link_libraries(hccl_utest_device PRIVATE
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
add_dependencies(hccl_utest_device hccl_llt)

set(hccl_utest_launch_aicpu_fw_src_list
    main.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/common/launch_aicpu/launch_aicpu.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/common/adapter/adapter_rts.cc
    ut_launch_aicpu.cc
)
add_executable(hccl_utest_launch_aicpu_fw ${hccl_utest_launch_aicpu_fw_src_list})

target_include_directories(hccl_utest_launch_aicpu_fw PRIVATE
    ${UT_COMMON_INCLUDE_LIST}
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/communicator/impl/independent_op/resource/device/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/communicator/impl/independent_op/device/
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/common/
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/communicator
    ${HCCL_CODE_ROOT}/src/coll_communicator/communicator/device/
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/communicator/group_schedule_mgr
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_engine/notify/
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_engine/
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_mems/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/comm_engine_res/launch/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/comm_engine_res/threads/device/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/comm_engine_res/threads/
)
target_compile_definitions(hccl_utest_launch_aicpu_fw PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
    HALF_T=float
    google=ascend_private
    CCL_KERNEL
    CCL_LLT
    OPEN_HCCL_TEST
    OPEN_BUILD_PROJECT
)
target_compile_options(hccl_utest_launch_aicpu_fw PRIVATE
    -U_FORTIFY_SOURCE
    -fno-var-tracking-assignments
    -Os
    -O0 -g --coverage -fprofile-arcs -ftest-coverage -fno-access-control
    $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address -fsanitize-recover=address,all -fno-omit-frame-pointer>
)

target_link_libraries(hccl_utest_launch_aicpu_fw PRIVATE
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
add_dependencies(hccl_utest_launch_aicpu_fw hccl_llt)

enable_testing()
add_test(NAME hccl_utest_device COMMAND hccl_utest_device)
add_test(NAME hccl_utest_launch_aicpu_fw COMMAND hccl_utest_launch_aicpu_fw)
