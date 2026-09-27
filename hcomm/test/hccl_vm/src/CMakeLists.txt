# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_subdirectory(utils)
add_subdirectory(common)
add_subdirectory(proxy)
add_subdirectory(runnerdb)
add_subdirectory(store)
add_subdirectory(plugin)
add_subdirectory(cmd)
add_subdirectory(topo)

add_library(hccl_plugin_objs OBJECT
    plugin/hccl_plugin_manager.cc
    plugin/hccl_plugin.cc
)
target_include_directories(hccl_plugin_objs PRIVATE
    ${HCOMM_VM_ROOT_PATH}/include
    ${HCOMM_VM_ROOT_PATH}/include/runnerdb
    ${HCOMM_VM_ROOT_PATH}/include/store
    ${HCOMM_VM_ROOT_PATH}/src/cmd
    ${HCOMM_VM_ROOT_PATH}/src/common
    ${HCOMM_VM_ROOT_PATH}/src/plugin
    ${HCOMM_VM_ROOT_PATH}/src/proxy
    ${HCOMM_VM_ROOT_PATH}/src/store
    ${HCOMM_VM_ROOT_PATH}/src/utils
    ${HCCL_VM_SRC}/store
    ${THRID_PARTY_DIR}/nlohmann_json
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/ccu/ccu_microcode
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/plugin
)
hcclvm_apply_common_flags(hccl_plugin_objs)

add_executable(hccl-vm
    main.cc
    $<TARGET_OBJECTS:hccl_plugin_objs>
)

message(STATUS "ASCEND_CANN_PACKAGE_PATH = ${ASCEND_CANN_PACKAGE_PATH}")
target_include_directories(hccl-vm PRIVATE
    # hccl vm头文件
    ${HCOMM_VM_ROOT_PATH}/include
    ${HCOMM_VM_ROOT_PATH}/include/runnerdb
    ${HCOMM_VM_ROOT_PATH}/src/store
    ${HCOMM_VM_ROOT_PATH}/src/runnerdb
    ${HCOMM_VM_ROOT_PATH}/src/cmd

    ${HCOMM_CODE_ROOT}/src/base_comm/resources/ccu/ccu_microcode

    ${HCCL_VM_SRC}/store

    ## cann包头文件
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/include/toolchain

    # cann包头文件
    ${ASCEND_CANN_PACKAGE_PATH}/include/driver
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/trace
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime/runtime/rts

    # hccl_vm头文件
    ${HCOMM_VM_ROOT_PATH}/include/store

    # hccl_proxy模块内部头文件
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/proxy
    ${CMAKE_CURRENT_SOURCE_DIR}/plugin
)

target_link_libraries(hccl-vm
    PRIVATE
    -Wl,--whole-archive
    -Wl,--start-group
    cmd
    common
    store
    topo
    loader
    runnerdb
    modeldb
    log
    -Wl,--end-group
    -Wl,--no-whole-archive
    dl
    rt
    sqlite3
    yaml-cpp::yaml-cpp
    CLI11::CLI11
    JSON::JSON
)

hcclvm_apply_common_flags(hccl-vm)

if(TARGET third_party_yaml_cpp)
    add_dependencies(hccl-vm third_party_yaml_cpp)
endif()

install(TARGETS hccl-vm
    DESTINATION "bin"
    # 文件的权限 (755)
    PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE
)
