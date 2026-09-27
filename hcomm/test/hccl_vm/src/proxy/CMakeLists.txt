# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

message(STATUS "TOP_DIR: ${TOP_DIR}")
message(STATUS "ASCEND_CANN_PACKAGE_PATH: ${ASCEND_CANN_PACKAGE_PATH}")

set(PROXY_INC_DIRS
        # hccl_proxy依赖的HCCL业务代码头文件, 这里的头文件有要求: 开源用户可获取
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/profiling
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
        ${HCOMM_CODE_ROOT}/src/pub_inc
        ${HCOMM_CODE_ROOT}/src/framework/communicator/impl/independent_op/rank_graph
        ${HCOMM_CODE_ROOT}/src/framework/inc
        ${HCOMM_CODE_ROOT}/src/platform/hccp/external_depends/inc/log
        # hccl_proxy依赖的Checker L2其他模块的头文件
        ${HCOMM_VM_ROOT_PATH}/include
        ${HCOMM_VM_ROOT_PATH}/src/store
        ${HCOMM_VM_ROOT_PATH}/src/runnerdb
        # cann包头文件
        ${ASCEND_CANN_PACKAGE_PATH}/include/driver
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hccl
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/trace
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime/runtime/rts
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/aicpu
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/aicpu/tsd
        ${HCOMM_CODE_ROOT}/src/base_comm/resources/ccu/ccu_microcode
        ${HCOMM_CODE_ROOT}/src/base_comm/resources/ccu/pub_inc
        # hccl_vm头文件
        ${HCOMM_VM_ROOT_PATH}/include/runnerdb
        ${HCOMM_VM_ROOT_PATH}/include/store
        ${HCOMM_VM_ROOT_PATH}/include/topo
        ${HCOMM_VM_ROOT_PATH}/src/cmd
        ${HCOMM_VM_ROOT_PATH}/src/common
        ${HCCL_VM_SRC}
        ${HCCL_VM_SRC}/store
        # hcomm头文件
        ${HCOMM_CODE_PLATFORM}/hccp
        ${HCOMM_CODE_PLATFORM}/hccp/inc
        ${HCOMM_CODE_PLATFORM}/hccp/inc/network
        ${HCOMM_CODE_PLATFORM}/hccp/external_depends/ccu
        # hcomm头文件(适配hcomm PR#2167)
        ${HCOMM_CODE_ROOT}/src/base_comm/resources/hccp
        ${HCOMM_CODE_ROOT}/src/base_comm/resources/hccp/inc
        ${HCOMM_CODE_ROOT}/src/base_comm/resources/hccp/inc/network
        ${HCOMM_CODE_ROOT}/src/base_comm/resources/hccp/external_depends/ccu
        ${HCOMM_CODE_ROOT}/src/legacy/ascend910/pub_inc
        # hccl_proxy模块内部头文件
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/topomodel
        ${CMAKE_CURRENT_SOURCE_DIR}/proxy
)

set(PROXY_COMPILE_OPTS
        -Werror
        -fno-common
        -fno-strict-aliasing
        -pipe
        -O0
        -g
        -D_GLIBCXX_USE_CXX11_ABI=0
)

set(PROXY_LINK_LIBS
        -Wl,--whole-archive
        -Wl,--start-group
        log
        common
        runnerdb
        modeldb
        store
        -Wl,--end-group
        -Wl,--no-whole-archive
        dl
        rt
        sqlite3
        JSON::JSON
        ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/lib64/libc_sec.a
)

add_library(hccl_proxy_level0 SHARED "")
target_include_directories(hccl_proxy_level0 PRIVATE ${PROXY_INC_DIRS})
target_sources(hccl_proxy_level0 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/hccl_op_stub.cc
)
target_compile_options(hccl_proxy_level0 PRIVATE ${PROXY_COMPILE_OPTS})

target_link_options(hccl_proxy_level0 PRIVATE
        -Wl,--unresolved-symbols=ignore-in-object-files)
target_link_libraries(hccl_proxy_level0 PRIVATE
        dl
        rt
        ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/lib64/libc_sec.a)
install(TARGETS hccl_proxy_level0
    DESTINATION "lib/${CMAKE_SYSTEM_PROCESSOR}"
    PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE
)

add_library(hccl_proxy_level2 SHARED "")
target_include_directories(hccl_proxy_level2 PRIVATE ${PROXY_INC_DIRS})
target_sources(hccl_proxy_level2 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/aclrt_device_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aclrt_context_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aclrt_memory_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aclrt_notify_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aclrt_event_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aclrt_stream_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/scatter_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/log_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hccl_inner_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hccp_ccu_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hccp_ra_socket_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aclrt_new_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aclrt_exception_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aclrt_exec_control.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aclrt_kernel_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hccp_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ascend_hal_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/adapter_rts_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/tsd_client_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aprofiling_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aclrt_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hccl_proxy_common.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hccl_comm_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aclrt_runtime_config.cc
)
target_compile_options(hccl_proxy_level2 PRIVATE ${PROXY_COMPILE_OPTS})
target_link_libraries(hccl_proxy_level2 PRIVATE ${PROXY_LINK_LIBS})
install(TARGETS hccl_proxy_level2
    DESTINATION "lib/${CMAKE_SYSTEM_PROCESSOR}"
    # 文件的权限 (755)
    PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE
)

# AIV构建
option(BUILD_AIV "Enable aiv kernel build" OFF)
MESSAGE(STATUS "BUILD_AIV: " ${BUILD_AIV})
if(BUILD_AIV)
    add_subdirectory(aiv_kernel)
endif()
