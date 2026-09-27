# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 定义 ccl_kernel 动态链接库，在 device 侧使用
add_library(ccl_kernel SHARED)

# 预处理宏定义
target_compile_definitions(ccl_kernel PRIVATE
    HCCD
    CCL_KERNEL_AICPU
)

if(BUILD_OPEN_PROJECT)
    # 编译选项
    target_compile_options(ccl_kernel PRIVATE
        -Werror
        -fno-common
        -fno-strict-aliasing
        $<$<CONFIG:Debug>:-Og -g>
        $<$<CONFIG:Release>:-O3>
    )
else()
    # 编译选项
    target_compile_options(ccl_kernel PRIVATE
        -Werror
        -fno-common
        -fno-strict-aliasing
        -O3
    )

    # 链接选项
    target_link_options(ccl_kernel PRIVATE
        -s
    )
endif()

# 头文件搜索路径
target_include_directories(ccl_kernel PRIVATE
    ${HCOMM_DIR}/include
    ${HCOMM_DIR}/include/hccl
    ${HCOMM_DIR}/pkg_inc
    ${HCOMM_DIR}/pkg_inc/hccl
    ${HCOMM_DIR}/pkg_inc/hcomm
    ${HCOMM_DIR}/pkg_inc/legacy
    ${HCOMM_DIR}/pkg_inc/legacy/hccl
    ${HCOMM_DIR}/src/legacy/ascend910/pub_inc
    ${HCOMM_DIR}/src/legacy/ascend910/pub_inc/aicpu
    ${HCOMM_DIR}/src/legacy/ascend910/pub_inc/new
    ${HCOMM_DIR}/external_depends/tsch

    # src/common 头文件 (legacy/ascend910)
    ${HCOMM_DIR}/src/legacy/ascend910/common/stream
    ${HCOMM_DIR}/src/legacy/ascend910/common/debug/config
    ${HCOMM_DIR}/src/legacy/ascend910/common/debug/profiling/inc
    ${HCOMM_DIR}/src/legacy/ascend910/common/error_manager
    ${HCOMM_DIR}/src/legacy/ascend910/common/launch_aicpu
    ${HCOMM_DIR}/src/legacy/ascend910/common

    # src/legacy 头文件 (ascend950)
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/socket
    ${HCOMM_DIR}/src/legacy/ascend950/framework/env_config

    # src/framework 头文件 (legacy/ascend910)
    ${HCOMM_DIR}/src/legacy/ascend910/framework
    ${HCOMM_DIR}/src/legacy/ascend910/framework/inc
    ${HCOMM_DIR}/src/legacy/ascend910/framework/op_base/src
    ${HCOMM_DIR}/src/legacy/ascend910/framework/cluster_maintenance/health/heartbeat
    ${HCOMM_DIR}/src/legacy/ascend910/framework/cluster_maintenance/recovery/operator_retry
    ${HCOMM_DIR}/src/legacy/ascend910/framework/common/src/exception
    ${HCOMM_DIR}/src/legacy/ascend910/framework/communicator/impl
    ${HCOMM_DIR}/src/legacy/ascend910/framework/communicator/impl/resource_manager

    # framework/next 头文件 (拆分到 base_comm 和 coll_communicator_mgr)
    ${HCOMM_DIR}/src/base_comm/resources/endpoints
    ${HCOMM_DIR}/src/coll_communicator_mgr/resource_mgr/local/my_rank/endpoints
    ${HCOMM_DIR}/src/base_comm/resources/reged_mems
    ${HCOMM_DIR}/src/base_comm/resources/endpoint_pairs
    ${HCOMM_DIR}/src/base_comm/resources/endpoint_pairs/sockets
    ${HCOMM_DIR}/src/base_comm/resources/endpoint_pairs/channels
    ${HCOMM_DIR}/src/base_comm/common
    ${HCOMM_DIR}/src/base_comm/common/device
    ${HCOMM_DIR}/src/base_comm/resources/ccu/ccu_device
    ${HCOMM_DIR}/src/base_comm/primitives/api_c_adpt
    ${HCOMM_DIR}/src/coll_communicator_mgr
    ${HCOMM_DIR}/src/coll_communicator_mgr/communicator
    ${HCOMM_DIR}/src/coll_communicator_mgr/resource_mgr/local/my_rank
    ${HCOMM_DIR}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_engine/engine_ctxs
    ${HCOMM_DIR}/src/coll_communicator_mgr/resource_mgr/remote/rank_pairs
    ${HCOMM_DIR}/src/coll_communicator_mgr/dfx/profiling/aicpu
    ${HCOMM_DIR}/src/coll_communicator_mgr/dfx/profiling/aicpu/common
    ${HCOMM_DIR}/src/coll_communicator_mgr/dfx/profiling/host

    # src/platform 头文件 (legacy/ascend910)
    ${HCOMM_DIR}/src/legacy/ascend910/platform/inc
    ${HCOMM_DIR}/src/legacy/ascend910/platform/inc/adapter
    ${HCOMM_DIR}/src/legacy/ascend910/platform/common
    ${HCOMM_DIR}/src/legacy/ascend910/platform/common/buffer_manager
    ${HCOMM_DIR}/src/legacy/ascend910/platform/common
    ${HCOMM_DIR}/src/legacy/ascend910/platform/common/unique
    ${HCOMM_DIR}/src/legacy/ascend910/platform/common/unfold_cache
    ${HCOMM_DIR}/src/legacy/ascend910/platform/resource/transport
    ${HCOMM_DIR}/src/legacy/ascend910/platform/resource/transport/heterog
    ${HCOMM_DIR}/src/legacy/ascend910/platform/resource/notify
    ${HCOMM_DIR}/src/legacy/ascend910/platform/resource/dispatcher_ctx
    ${HCOMM_DIR}/src/legacy/ascend910/platform/resource/socket
    ${HCOMM_DIR}/src/legacy/ascend910/platform/task

    # hccp (base_comm/resources)
    ${HCOMM_DIR}/src/base_comm/resources/hccp/inc
    ${HCOMM_DIR}/src/base_comm/resources/hccp/inc/network

    # src/algorithm 头文件 (legacy/ascend910)
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/pub_inc
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/inc
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/alg_template
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/communicator
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/communicator/legacy
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/resource_manager
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/task
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/legacy
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/coll_executor

    # src/legacy 头文件
    ${HCOMM_DIR}/src/legacy/ascend950
    ${HCOMM_DIR}/src/legacy/ascend950/common
    ${HCOMM_DIR}/src/legacy/ascend950/common/exception
    ${HCOMM_DIR}/src/legacy/ascend950/common/types
    ${HCOMM_DIR}/src/legacy/ascend950/common/utils
    ${HCOMM_DIR}/src/legacy/ascend950/framework
    ${HCOMM_DIR}/src/legacy/ascend950/framework/aiv
    ${HCOMM_DIR}/src/legacy/ascend950/framework/aiv/aiv_ins
    ${HCOMM_DIR}/src/legacy/ascend950/framework/aiv/aiv_mc2
    ${HCOMM_DIR}/src/legacy/ascend950/framework/ccu
    ${HCOMM_DIR}/src/legacy/ascend950/framework/ccu/ccu_ins
    ${HCOMM_DIR}/src/legacy/ascend950/framework/ccu/ccu_manager
    ${HCOMM_DIR}/src/legacy/ascend950/framework/ccu/ccu_mc2
    ${HCOMM_DIR}/src/legacy/ascend950/framework/communicator
    ${HCOMM_DIR}/src/legacy/ascend950/framework/communicator/aicpu
    ${HCOMM_DIR}/src/legacy/ascend950/framework/communicator/aicpu/daemon
    ${HCOMM_DIR}/src/legacy/ascend950/framework/communicator/aicpu/inc
    ${HCOMM_DIR}/src/legacy/ascend950/framework/communicator/aicpu/one_sided_component
    ${HCOMM_DIR}/src/legacy/ascend950/framework/communicator/hostdpu
    ${HCOMM_DIR}/src/legacy/ascend950/framework/device_mode
    ${HCOMM_DIR}/src/legacy/ascend950/framework/dfx
    ${HCOMM_DIR}/src/legacy/ascend950/framework/dfx/aicpu/common
    ${HCOMM_DIR}/src/legacy/ascend950/framework/dfx/aicpu/profiling
    ${HCOMM_DIR}/src/legacy/ascend950/framework/dfx/aicpu/task_exception
    ${HCOMM_DIR}/src/legacy/ascend950/framework/dfx/common
    ${HCOMM_DIR}/src/legacy/ascend950/framework/dfx/profiling
    ${HCOMM_DIR}/src/legacy/ascend950/framework/dfx/task_exception
    ${HCOMM_DIR}/src/legacy/ascend950/framework/entrance
    ${HCOMM_DIR}/src/legacy/ascend950/framework/entrance/hcom
    ${HCOMM_DIR}/src/legacy/ascend950/framework/entrance/hcom_comm
    ${HCOMM_DIR}/src/legacy/ascend950/framework/entrance/one_sided_service
    ${HCOMM_DIR}/src/legacy/ascend950/framework/env_config
    ${HCOMM_DIR}/src/legacy/ascend950/framework/fault_recovery
    ${HCOMM_DIR}/src/legacy/ascend950/framework/misc/json_parser
    ${HCOMM_DIR}/src/legacy/ascend950/framework/misc/mask_event
    ${HCOMM_DIR}/src/legacy/ascend950/framework/misc/whitelist
    ${HCOMM_DIR}/src/legacy/ascend950/framework/resource_manager/buffer
    ${HCOMM_DIR}/src/legacy/ascend950/framework/resource_manager/buffer/aicpu
    ${HCOMM_DIR}/src/legacy/ascend950/framework/resource_manager/connection
    ${HCOMM_DIR}/src/legacy/ascend950/framework/resource_manager/connection/ub_ci_updater
    ${HCOMM_DIR}/src/legacy/ascend950/framework/resource_manager/notify
    ${HCOMM_DIR}/src/legacy/ascend950/framework/resource_manager/notify/aicpu
    ${HCOMM_DIR}/src/legacy/ascend950/framework/resource_manager/socket
    ${HCOMM_DIR}/src/legacy/ascend950/framework/resource_manager/stream
    ${HCOMM_DIR}/src/legacy/ascend950/framework/resource_manager/stream/aicpu
    ${HCOMM_DIR}/src/legacy/ascend950/framework/resource_manager/transport
    ${HCOMM_DIR}/src/legacy/ascend950/framework/resource_manager/transport/aicpu
    ${HCOMM_DIR}/src/legacy/ascend950/framework/service
    ${HCOMM_DIR}/src/legacy/ascend950/framework/service/aicpu
    ${HCOMM_DIR}/src/legacy/ascend950/framework/service/one_sided_service
    ${HCOMM_DIR}/src/legacy/ascend950/framework/topo
    ${HCOMM_DIR}/src/legacy/ascend950/framework/topo/new_topo_builder/common
    ${HCOMM_DIR}/src/legacy/ascend950/framework/topo/new_topo_builder/phy_topo
    ${HCOMM_DIR}/src/legacy/ascend950/framework/topo/new_topo_builder/phy_topo_builder
    ${HCOMM_DIR}/src/legacy/ascend950/framework/topo/new_topo_builder/rank_table_info
    ${HCOMM_DIR}/src/legacy/ascend950/framework/topo/new_topo_builder/topo_info
    ${HCOMM_DIR}/src/legacy/ascend950/framework/topo/rank_info_detect
    ${HCOMM_DIR}/src/legacy/ascend950/framework/topo/topo_addr_info/include
    ${HCOMM_DIR}/src/legacy/ascend950/framework/topo/topo_addr_info/src
    ${HCOMM_DIR}/src/legacy/ascend950/include
    ${HCOMM_DIR}/src/legacy/ascend950/interface
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/alg_registry
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/coll_alg_factory/alg_aiv_instruction
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/coll_alg_factory/alg_aiv_instruction/aiv_interface
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/coll_alg_factory/alg_ccu_context
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/coll_alg_factory/alg_ccu_context/all_gather
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/coll_alg_factory/alg_ccu_context/all_reduce
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/coll_alg_factory/alg_ccu_context/all_to_all
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/coll_alg_factory/alg_ccu_context/broadcast
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/coll_alg_factory/alg_ccu_context/reduce
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/coll_alg_factory/alg_ccu_context/reduce_scatter
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/coll_alg_factory/alg_ccu_context/scatter
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/coll_alg_factory/alg_executor
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/coll_alg_factory/alg_executor/ins_alg_executor
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/coll_alg_factory/alg_executor/ins_alg_executor/all_gather
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/coll_alg_factory/alg_executor/ins_alg_executor/all_reduce
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/coll_alg_factory/alg_executor/ins_alg_executor/all_to_all
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/coll_alg_factory/alg_executor/ins_alg_executor/broadcast
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/coll_alg_factory/alg_executor/ins_alg_executor/reduce
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/coll_alg_factory/alg_executor/ins_alg_executor/reduce_scatter
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/coll_alg_factory/alg_executor/ins_alg_executor/scatter
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/coll_alg_factory/alg_executor/ins_alg_executor/send_recv
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/coll_alg_factory/alg_executor/prim_alg_executor
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/coll_alg_factory/alg_template
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/coll_alg_factory/alg_template/aiv_alg_template
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/coll_alg_factory/alg_template/ccu_alg_template
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/coll_alg_factory/alg_template/ins_alg_template
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/coll_alg_factory/alg_template/prim_alg_template
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/coll_alg_factory/alg_topo_match
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/coll_alg_factory/utils/common
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/interface
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/interface/device
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/interface/host
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/selector
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/instruction
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/primitive
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/aiv
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/ccu/ccu_context
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/ccu/ccu_device
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/ccu/ccu_device/ccu_component
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/ccu/ccu_device/ccu_component/ccu_channel
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/ccu/ccu_device/ccu_component/ccu_channel/ccu_channel_v1
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/ccu/ccu_device/ccu_component/ccu_pfe
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/ccu/ccu_microcode
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/ccu/ccu_representation
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/ccu/ccu_representation/context
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/ccu/ccu_representation/interface
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/ccu/ccu_representation/reps
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/ccu/ccu_representation/reps/arithmetic
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/ccu/ccu_representation/reps/common
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/ccu/ccu_representation/reps/control
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/ccu/ccu_representation/reps/data
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/ccu/ccu_representation/reps/loop
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/ccu/ccu_representation/reps/sync
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/ccu/ccu_representation/reps/translator
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/ccu/dfx
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/common
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/external_system
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/pub_inc
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/pub_inc/ccu
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/pub_inc/resource/transport
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/buffer
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/buffer/aicpu
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/ccu_transport
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/connection
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/connection/aicpu
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/connection/ub_ci_updater
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/mem
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/net_dev
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/notify
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/notify/aicpu
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/socket
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/stream
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/stream/aicpu
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/task
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/transport
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/transport/aicpu
)

if(BUILD_OPEN_PROJECT)
    target_compile_definitions(ccl_kernel PRIVATE
        OPEN_BUILD_PROJECT
    )

    target_link_libraries(ccl_kernel PRIVATE
        $<BUILD_INTERFACE:intf_pub>
        $<BUILD_INTERFACE:acl_rt_headers>
        $<BUILD_INTERFACE:asc_host_headers>
        $<BUILD_INTERFACE:asc_kernel_headers>
        $<BUILD_INTERFACE:ascend_hal_headers>
        $<BUILD_INTERFACE:kernel_tiling_headers>
        $<BUILD_INTERFACE:slog_headers>
        $<BUILD_INTERFACE:atrace_headers>
        $<BUILD_INTERFACE:mmpa_headers>
        $<BUILD_INTERFACE:runtime_headers>
        $<BUILD_INTERFACE:rdma_core_headers>
        $<BUILD_INTERFACE:json>
        -Wl,--no-as-needed
        ascend_hal
        c_sec
        mmpa
        ccl_kernel_plf
        -Wl,--as-needed
        -lrt
        -ldl
        -lpthread
    )
else()
    target_include_directories(ccl_kernel PRIVATE
        ${TOP_DIR}/inc
        ${TOP_DIR}/inc/driver
        ${TOP_DIR}/metadef/inc/external
        ${TOP_DIR}/metadef/pkg_inc
        ${TOP_DIR}/metadef/pkg_inc
        ${TOP_DIR}/open_source/json/include
        ${TOP_DIR}/inc/aicpu/
        ${TOP_DIR}/atc/opcompiler/ascendc_compiler/api
        ${TOP_DIR}/atc/opcompiler/ascendc_compiler/api/impl
        ${TOP_DIR}/atc/opcompiler/ascendc_compiler/api/include
        ${TOP_DIR}/abl/atrace/inc/utrace
        ${TOP_DIR}/runtime/include/external
        ${TOP_DIR}/runtime/include/external/acl
        ${TOP_DIR}/runtime/pkg_inc
        ${TOP_DIR}/runtime/pkg_inc/runtime
        ${TOP_DIR}/runtime/pkg_inc/profiling
        ${TOP_DIR}/runtime/pkg_inc/trace
        ${TOP_DIR}/runtime/pkg_inc/base
        ${TOP_DIR}/runtime/pkg_inc/aicpu_sched
        ${TOP_DIR}/asc/asc-devkit
        ${TOP_DIR}/asc/asc-devkit/include/adv_api/hccl/internal
    )

    target_link_libraries(ccl_kernel PRIVATE
        $<BUILD_INTERFACE:intf_pub>
        $<BUILD_INTERFACE:mmpa_headers>
        $<BUILD_INTERFACE:msprof_headers>
        $<BUILD_INTERFACE:slog_headers>
        $<BUILD_INTERFACE:hccl_headers>
        $<BUILD_INTERFACE:npu_runtime_headers>
        $<BUILD_INTERFACE:atrace_headers>
        $<BUILD_INTERFACE:kernel_tiling_headers>
        -Wl,--no-as-needed
        c_sec
        ccl_kernel_plf
        mmpa
        -Wl,--as-needed
        -lrt
        -ldl
        -lpthread
        ofed_headers
    )

    install(TARGETS ccl_kernel
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} ${INSTALL_OPTIONAL}
        COMPONENT hcomm
    )
endif()

# 符号隐藏: 仅导出白名单符号; UT/ST 打桩依赖动态符号, 不挂载
if(NOT ENABLE_TEST)
    target_link_options(ccl_kernel PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_LIST_DIR}/ccl_kernel.map")
    set_property(TARGET ccl_kernel APPEND PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/ccl_kernel.map")
endif()

# 将 ccl_kernel.ini 转换为 json 格式
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ccl_kernel.json
    COMMAND ${HI_PYTHON} ${HCOMM_DIR}/cmake/scripts/parser_ini.py
                         ${CMAKE_CURRENT_LIST_DIR}/device/framework/ccl_kernel.ini
                         ${CMAKE_CURRENT_BINARY_DIR}/ccl_kernel.json
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    COMMENT "Generating ccl_kernel.json"
 	VERBATIM
)
add_custom_target(ccl_kernel_json ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/ccl_kernel.json
)

# 安装
install(TARGETS ccl_kernel
    LIBRARY DESTINATION ${INSTALL_DEVICE_LIBRARY_DIR} ${INSTALL_OPTIONAL}
    COMPONENT hcomm
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ccl_kernel.json
    DESTINATION ${INSTALL_CCL_KERNEL_JSON_DIR}/config ${INSTALL_OPTIONAL}
    COMPONENT hcomm
)
