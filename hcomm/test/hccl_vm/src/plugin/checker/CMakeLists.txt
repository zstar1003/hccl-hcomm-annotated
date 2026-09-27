# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 创建checker.bin
add_executable(checker
    main.cc
)

add_subdirectory(src)

# 如果需要暴露头文件目录
target_include_directories(checker PUBLIC
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/include
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc/hccl

    ${HCOMM_CODE_ROOT}/src/base_comm/resources/ccu/ccu_microcode

    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/checker
    ${CMAKE_CURRENT_SOURCE_DIR}/src/framework
    ${CMAKE_CURRENT_SOURCE_DIR}/src/framework/mem_conflict_check
    ${CMAKE_CURRENT_SOURCE_DIR}/src/framework/semantics_check
    ${CMAKE_CURRENT_SOURCE_DIR}/src/framework/singletask_check
    ${CMAKE_CURRENT_SOURCE_DIR}/src/framework/task_graph_generator
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dfx

    ${CMAKE_CURRENT_SOURCE_DIR}/header/external
    ${CMAKE_CURRENT_SOURCE_DIR}/header/internal

    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/runnerdb
    ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_SOURCE_DIR}/third_party/nlohmann_json
)

target_sources(checker PRIVATE
    ${HCCL_VM_INCLUDE}/sim_log.h
)

target_link_libraries(checker PRIVATE
    -Wl,--start-group
    log
    loader
    runnerdb
    modeldb
    store
    common
    -Wl,--end-group
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/lib64/libc_sec.a
    pthread
    yaml-cpp::yaml-cpp
    sqlite3
)

set_target_properties(checker PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(checker PROPERTIES
    INSTALL_RPATH "$ORIGIN/../../lib/${CMAKE_SYSTEM_PROCESSOR}"
    BUILD_RPATH "$ORIGIN")

hcclvm_apply_common_flags(checker)

install(TARGETS checker
    DESTINATION "plugin/checker"
    # 文件的权限 (755)
    PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE
 )

install(FILES "manifest.json"
    DESTINATION "plugin/checker"
    PERMISSIONS
        OWNER_READ OWNER_WRITE
        GROUP_READ
        WORLD_READ
)

# 日志染色: 模块 tag
target_compile_definitions(checker PRIVATE HCCL_VM_MODULE="CHECKER")
