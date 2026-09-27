# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 定义 ccl_kernel_plf 动态链接库，在 device 侧使用
add_library(ccl_kernel_plf SHARED)
# 定义 ccl_kernel_plf 静态链接库，在 device 侧使用
add_library(ccl_kernel_plf_a STATIC)

# 预处理宏定义
set(CCL_KERNEL_PLF_COMPILE_DEFINITIONS
    HCCD
    CCL_KERNEL
    CCL_KERNEL_AICPU
    USE_AICORE_REDUCESUM
    USE_AICORE_GATHERV2
    USE_AICORE_GATHERV2_INFER
)
target_compile_definitions(ccl_kernel_plf PRIVATE ${CCL_KERNEL_PLF_COMPILE_DEFINITIONS})
target_compile_definitions(ccl_kernel_plf_a PRIVATE ${CCL_KERNEL_PLF_COMPILE_DEFINITIONS})


if(BUILD_OPEN_PROJECT)
    # 编译选项
    set(CCL_KERNEL_PLF_COMPILE_OPTIONS
        -Werror
        -Wno-deprecated-declarations
        -fno-common
        -fno-strict-aliasing
        $<$<CONFIG:Debug>:-Og -g>
        $<$<CONFIG:Release>:-O2 -D_FORTIFY_SOURCE=2>
    )
else()
    # 编译选项
    set(CCL_KERNEL_PLF_COMPILE_OPTIONS
        -Werror
        -Wno-deprecated-declarations
        -fno-common
        -fno-strict-aliasing
        -O2
        -D_FORTIFY_SOURCE=2
    )
endif()
target_compile_options(ccl_kernel_plf PRIVATE
    ${CCL_KERNEL_PLF_COMPILE_OPTIONS}
)
# ccl_kernel_plf_a 额外编译选项
target_compile_options(ccl_kernel_plf_a PRIVATE
    ${CCL_KERNEL_PLF_COMPILE_OPTIONS}
    -ftrapv
)

if(BUILD_OPEN_PROJECT)
    # 链接选项
    set(CCL_KERNEL_PLF_LINK_OPTIONS
        -Wl,-Bsymbolic
        -Wl,--exclude-libs,ALL
    )
else()
    # 链接选项
    set(CCL_KERNEL_PLF_LINK_OPTIONS
        -Wl,-Bsymbolic
        -Wl,--exclude-libs,ALL
        -s
    )
endif()
target_link_options(ccl_kernel_plf PRIVATE
    ${CCL_KERNEL_PLF_LINK_OPTIONS}
)
target_link_options(ccl_kernel_plf_a PRIVATE
    ${CCL_KERNEL_PLF_LINK_OPTIONS}
)

# 头文件搜索路径
set(CCL_KERNEL_PLF_INCLUDE_LIST
    ${HCOMM_DIR}/include
    ${HCOMM_DIR}/include/hccl
    ${HCOMM_DIR}/pkg_inc
    ${HCOMM_DIR}/pkg_inc/hccl
    ${HCOMM_DIR}/pkg_inc/legacy
    ${HCOMM_DIR}/pkg_inc/legacy/hccl
    ${HCOMM_DIR}/src/base_comm/common

    # pub_inc (legacy/ascend910)
    ${HCOMM_DIR}/src/legacy/ascend910/pub_inc
    ${HCOMM_DIR}/src/legacy/ascend910/pub_inc/aicpu/
    ${HCOMM_DIR}/src/legacy/ascend910/pub_inc/inner
    ${HCOMM_DIR}/src/legacy/ascend910/pub_inc/new

    # algorithm (legacy/ascend910)
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/inc
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/pub_inc
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/alg_aiv_template
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/alg_template
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/alg_template/component
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/mc2_handler
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/communicator
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/inc
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/legacy
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/legacy/operator
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/communicator/legacy
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/resource_manager
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/task
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/operator
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/operator/registry
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/coll_executor
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/coll_executor/registry
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/coll_executor/coll_send_receive
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/310P
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/310P
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather_v
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/310P
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter_v
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter_v/310P
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_to_all
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/coll_executor/coll_scatter
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/alg_template/temp_all_gather
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/alg_template/temp_all_reduce
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/alg_template/temp_alltoall
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/alg_template/temp_alltoallv
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/alg_template/temp_broadcast
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/alg_template/temp_reduce
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/alg_template/temp_reduce_scatter
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/alg_template/temp_scatter

    # common (legacy/ascend910)
    ${HCOMM_DIR}/src/legacy/ascend910/common/health
    ${HCOMM_DIR}/src/legacy/ascend910/common/debug/profiling/inc/aicpu
    ${HCOMM_DIR}/src/legacy/ascend910/common/debug/profiling/inc
    ${HCOMM_DIR}/src/legacy/ascend910/common/debug/config/
    ${HCOMM_DIR}/src/legacy/ascend910/common/stream
    ${HCOMM_DIR}/src/legacy/ascend910/common/launch_device

    # framework (legacy/ascend910)
    ${HCOMM_DIR}/src/legacy/ascend910/framework/inc
    ${HCOMM_DIR}/src/legacy/ascend910/framework/inc/hccd
    ${HCOMM_DIR}/src/legacy/ascend910/framework/cluster_maintenance/health/heartbeat/
    ${HCOMM_DIR}/src/legacy/ascend910/framework/cluster_maintenance/detect/detect_connect_anomalies/
    ${HCOMM_DIR}/src/legacy/ascend910/framework/cluster_maintenance/recovery/operator_retry
    ${HCOMM_DIR}/src/legacy/ascend910/framework/
    ${HCOMM_DIR}/src/legacy/ascend910/framework/common/src/
    ${HCOMM_DIR}/src/legacy/ascend910/framework/common/src/exception
    ${HCOMM_DIR}/src/legacy/ascend910/framework/common/src/config
    ${HCOMM_DIR}/src/legacy/ascend910/framework/common/src/task
    ${HCOMM_DIR}/src/legacy/ascend910/framework/common/src/topo
    ${HCOMM_DIR}/src/legacy/ascend910/framework/common/src/mgr
    ${HCOMM_DIR}/src/legacy/ascend910/framework/common/src/aicpu
    ${HCOMM_DIR}/src/legacy/ascend910/framework/hcom
    ${HCOMM_DIR}/src/legacy/ascend910/framework/communicator/impl/
    ${HCOMM_DIR}/src/legacy/ascend910/framework/communicator/impl/one_sided_service/
    ${HCOMM_DIR}/src/legacy/ascend910/framework/communicator/impl/zero_copy
    ${HCOMM_DIR}/src/legacy/ascend910/framework/communicator/impl/resource_manager
    ${HCOMM_DIR}/src/coll_communicator_mgr/common
    ${HCOMM_DIR}/src/coll_communicator_mgr/rank_graph
    ${HCOMM_DIR}/src/legacy/ascend910/framework/op_base/src/

    # framework/next (拆分到 base_comm 和 coll_communicator_mgr)
    ${HCOMM_DIR}/src/coll_communicator_mgr/resource_mgr/local/my_rank
    ${HCOMM_DIR}/src/coll_communicator_mgr/communicator
    ${HCOMM_DIR}/src/coll_communicator_mgr/communicator/device
    ${HCOMM_DIR}/src/coll_communicator_mgr/resource_mgr/remote/rank_pairs
    ${HCOMM_DIR}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_engine/threads
    ${HCOMM_DIR}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_engine/notify
    ${HCOMM_DIR}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_engine
    ${HCOMM_DIR}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_mems
    ${HCOMM_DIR}/src/base_comm/resources/comm_engine_res/engine_ctxs
    ${HCOMM_DIR}/src/base_comm/resources/comm_engine_res/launch
    ${HCOMM_DIR}/src/base_comm/resources/comm_engine_res/threads
    ${HCOMM_DIR}/src/base_comm/resources/endpoint_pairs/sockets/
    ${HCOMM_DIR}/src/base_comm/resources/endpoints/
    ${HCOMM_DIR}/src/coll_communicator_mgr/resource_mgr/local/my_rank/endpoints
    ${HCOMM_DIR}/src/base_comm/resources/reged_mems
    ${HCOMM_DIR}/src/base_comm/resources/endpoint_pairs/

    # platform (legacy/ascend910)
    ${HCOMM_DIR}/src/legacy/ascend910/platform/
    ${HCOMM_DIR}/src/legacy/ascend910/platform/inc
    ${HCOMM_DIR}/src/legacy/ascend910/platform/inc/adapter
    ${HCOMM_DIR}/src/legacy/ascend910/platform/inc/adapter/hccd
    ${HCOMM_DIR}/src/legacy/ascend910/platform/common/
    ${HCOMM_DIR}/src/legacy/ascend910/platform/common/unique
    ${HCOMM_DIR}/src/legacy/ascend910/platform/common/buffer_manager
    ${HCOMM_DIR}/src/legacy/ascend910/platform/common/unfold_cache
    ${HCOMM_DIR}/src/legacy/ascend910/common/error_manager
    ${HCOMM_DIR}/src/legacy/ascend910/common/launch_aicpu
    ${HCOMM_DIR}/src/common

    # hccp (moved to base_comm/resources)
    ${HCOMM_DIR}/src/base_comm/resources/hccp
    ${HCOMM_DIR}/src/base_comm/resources/hccp/inc
    ${HCOMM_DIR}/src/base_comm/resources/hccp/inc/network
    ${HCOMM_DIR}/src/legacy/ascend910/platform/resource/dispatcher_ctx
    ${HCOMM_DIR}/src/legacy/ascend910/platform/resource/transport
    ${HCOMM_DIR}/src/legacy/ascend910/platform/resource/transport/device
    ${HCOMM_DIR}/src/legacy/ascend910/platform/resource/transport/host
    ${HCOMM_DIR}/src/legacy/ascend910/platform/resource/transport/heterog
    ${HCOMM_DIR}/src/legacy/ascend910/platform/resource/transport/onesided
    ${HCOMM_DIR}/src/legacy/ascend910/platform/resource/transport/onesided/device
    ${HCOMM_DIR}/src/legacy/ascend910/platform/resource/notify
    ${HCOMM_DIR}/src/legacy/ascend910/platform/resource/socket
    ${HCOMM_DIR}/src/legacy/ascend910/platform/ping_mesh
    ${HCOMM_DIR}/src/legacy/ascend910/platform/task
    ${HCOMM_DIR}/src/legacy/ascend910/platform/typical

    # legacy (ascend950)
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/transport
    ${HCOMM_DIR}/src/legacy/ascend950/common/utils/
    ${HCOMM_DIR}/src/legacy/ascend950/common/exception/
    ${HCOMM_DIR}/src/legacy/ascend950/common/types/
    ${HCOMM_DIR}/src/legacy/ascend950/common/
    ${HCOMM_DIR}/src/legacy/ascend950/framework/resource_manager/socket/
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/external_system/
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/common/
)
target_include_directories(ccl_kernel_plf PRIVATE
    ${CCL_KERNEL_PLF_INCLUDE_LIST}
    ${HCOMM_DIR}/src/legacy/ascend910/common
    ${HCOMM_DIR}/external_depends/tsch
)
target_include_directories(ccl_kernel_plf_a PRIVATE
    ${CCL_KERNEL_PLF_INCLUDE_LIST}
    ${HCOMM_DIR}/src/legacy/ascend910/common
)

if(BUILD_OPEN_PROJECT)
    target_compile_definitions(ccl_kernel_plf PRIVATE
        OPEN_BUILD_PROJECT
    )
    target_compile_definitions(ccl_kernel_plf_a PRIVATE 
        OPEN_BUILD_PROJECT
    )

    set(CCL_KERNEL_PLF_OPEN_INCLUDE_LIST
        # 临时依赖头文件，待删除
        ${HCOMM_DIR}/external_depends/tsch/
    )
    target_include_directories(ccl_kernel_plf PRIVATE
        ${CCL_KERNEL_PLF_OPEN_INCLUDE_LIST}
    )
    target_include_directories(ccl_kernel_plf_a PRIVATE
        ${CCL_KERNEL_PLF_OPEN_INCLUDE_LIST}
    )

    # 链接库
    set(CCL_KERNEL_PLF_LINK_LIBS
        $<BUILD_INTERFACE:intf_pub>
        $<BUILD_INTERFACE:ascend_hal_headers>
        $<BUILD_INTERFACE:atrace_headers>
        $<BUILD_INTERFACE:mmpa_headers>
        $<BUILD_INTERFACE:runtime_headers>
        $<BUILD_INTERFACE:slog_headers>
        $<BUILD_INTERFACE:rdma_core_headers>
        $<BUILD_INTERFACE:hccl_legacy_headers>
        $<BUILD_INTERFACE:json>
        -Wl,--no-as-needed
        c_sec
        aicpu_sharder
        mmpa
        -Wl,--as-needed
        -lrt
        -ldl
        -lpthread
    )
    target_link_libraries(ccl_kernel_plf
        ${CCL_KERNEL_PLF_LINK_LIBS}
    )
    target_link_libraries(ccl_kernel_plf_a PRIVATE
        ${CCL_KERNEL_PLF_LINK_LIBS}
    )
else()
    set(CCL_KERNEL_PLF_NON_OPEN_INCLUDE_LIST
        ${TOP_DIR}/inc
        ${TOP_DIR}/inc/aicpu/tsd
        ${TOP_DIR}/inc/driver
        ${TOP_DIR}/runtime/include/external
        ${TOP_DIR}/runtime/include/external/acl
        ${TOP_DIR}/runtime/include/driver
        ${TOP_DIR}/runtime/pkg_inc
        ${TOP_DIR}/runtime/pkg_inc/runtime
        ${TOP_DIR}/runtime/pkg_inc/profiling
        ${TOP_DIR}/runtime/pkg_inc/trace
        ${TOP_DIR}/runtime/pkg_inc/base
        ${TOP_DIR}/runtime/pkg_inc/aicpu_sched
        ${TOP_DIR}/inc/external
        ${TOP_DIR}/inc/aicpu/
        ${TOP_DIR}/inc/aicpu/aicpu_schedule/aicpu_sharder
        ${TOP_DIR}/abl/atrace/inc/utrace
        ${TOP_DIR}/abl/msprof/inc
        ${TOP_DIR}/abl/qos/qosmng
        ${TOP_DIR}/runtime/src/dfx/adump/inc/adump
        ${TOP_DIR}/inc/aicpu/common
        ${TOP_DIR}/ace/csruntime/inc/tsch
        ${TOP_DIR}/metadef
        ${TOP_DIR}/metadef/inc
        ${TOP_DIR}/metadef/pkg_inc
        ${TOP_DIR}/metadef/inc/common
        ${TOP_DIR}/metadef/inc/external
        ${TOP_DIR}/open_source/json/include
        ${TOP_DIR}/asc/asc-devkit/include/adv_api/hccl/internal
        ${TOP_DIR}/asc/asc-devkit/include/adv_api/hccl/internal/hcomm
    )
    target_include_directories(ccl_kernel_plf PRIVATE
        ${CCL_KERNEL_PLF_NON_OPEN_INCLUDE_LIST}
    )
    target_include_directories(ccl_kernel_plf_a PRIVATE
        ${CCL_KERNEL_PLF_NON_OPEN_INCLUDE_LIST}
    )

    # 链接库
    set(CCL_KERNEL_PLF_LINK_LIBS
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
        aicpu_sharder
        mmpa
        -Wl,--as-needed
        -lrt
        -ldl
        -lpthread
        ofed_headers
    )
    target_link_libraries(ccl_kernel_plf
        ${CCL_KERNEL_PLF_LINK_LIBS}
    )
    target_link_libraries(ccl_kernel_plf_a
        ${CCL_KERNEL_PLF_LINK_LIBS}
    )

    # 安装
    install(TARGETS ccl_kernel_plf
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} ${INSTALL_OPTIONAL}
        COMPONENT hcomm
    )
    install(TARGETS ccl_kernel_plf_a
        ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR} ${INSTALL_OPTIONAL}
        COMPONENT hcomm
    )
endif()

# 符号隐藏: 仅导出白名单符号; UT/ST 打桩依赖动态符号, 不挂载
if(NOT ENABLE_TEST)
    target_link_options(ccl_kernel_plf PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_LIST_DIR}/ccl_kernel_plf.map")
    set_property(TARGET ccl_kernel_plf APPEND PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/ccl_kernel_plf.map")
endif()

# 设置 ccl_kernel_plf_a 输出文件名
set_target_properties(ccl_kernel_plf_a
    PROPERTIES
    OUTPUT_NAME ccl_kernel_plf
)
