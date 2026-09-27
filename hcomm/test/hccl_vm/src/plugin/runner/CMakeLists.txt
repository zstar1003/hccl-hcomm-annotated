# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_subdirectory(ccu_executor)
add_subdirectory(aiv_executor)

file(GLOB_RECURSE VR_LIB_SOURCES CONFIGURE_DEPENDS
    "./runner_utils/*.cc"
    "./runner_worker/*.cc"
    ${CMAKE_CURRENT_SOURCE_DIR}/../hccl_virtual_runtime_main.cc
)
add_executable(runner ${VR_LIB_SOURCES})

hcclvm_apply_common_flags(runner)
set_target_properties(runner PROPERTIES
    INSTALL_RPATH "$ORIGIN/../../lib/${CMAKE_SYSTEM_PROCESSOR}"
    BUILD_RPATH "$ORIGIN")

target_include_directories(runner PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/include
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc/hccl

    ${HCOMM_CODE_ROOT}/src/base_comm/resources/ccu/ccu_microcode

    ${HCCL_VM_SRC}
    ${HCCL_VM_SRC}/store
    ${HCCL_VM_INCLUDE}
    ${HCCL_VM_INCLUDE}/runnerdb
    ${HCCL_VM_INCLUDE}/store

    ${THRID_PARTY_DIR}
    ./ccu_executor
    ./runner_utils
    ./runner_worker
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
)

target_link_libraries(runner PRIVATE
    -Wl,--whole-archive
    -Wl,--start-group
    aiv_executor
    store
    loader
    runnerdb
    modeldb
    ccu_executor
    log
    -Wl,--end-group
    -Wl,--no-whole-archive
    rt
    JSON::JSON
    yaml-cpp::yaml-cpp
    sqlite3
)
install(TARGETS runner
    DESTINATION "plugin/runner"
    # 文件的权限 (755)
    PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE
 )

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/../manifest.json"
    DESTINATION "plugin/runner"
    PERMISSIONS
        OWNER_READ OWNER_WRITE
        GROUP_READ
        WORLD_READ
)

# 日志染色: 模块 tag
target_compile_definitions(runner PRIVATE HCCL_VM_MODULE="RUNNER")
