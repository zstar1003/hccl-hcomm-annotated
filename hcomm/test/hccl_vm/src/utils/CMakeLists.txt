# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# --- log library (sim_log.cc + sim_yaml_config.cc + sim_common.cpp) ---
add_library(log STATIC sim_log.cc sim_yaml_config.cc sim_common_api.cc)

target_include_directories(log PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
        ${CMAKE_SOURCE_DIR}/third_party
        ${CMAKE_SOURCE_DIR}/third_party/nlohmann_json
        ${CMAKE_SOURCE_DIR}/third_party/yaml-cpp/include
        ${CMAKE_SOURCE_DIR}/third_party/spdlog
)

target_include_directories(log PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/profiling
        ${YAMLCPP_INCLUDE_DIR}
)

target_compile_definitions(log PRIVATE HAVE_YAML_CPP)

hcclvm_apply_common_flags(log)

target_link_libraries(log PRIVATE yaml-cpp::yaml-cpp)

if(TARGET third_party_yaml_cpp)
    add_dependencies(log third_party_yaml_cpp)
endif()

# --- loader library (from sim_loader.cc, migrated from src/loader) ---
add_library(loader STATIC sim_loader.cc)

hcclvm_apply_common_flags(loader)

target_include_directories(loader PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/runnerdb
    ${PROJECT_SOURCE_DIR}/include/store

    ${HCCL_VM_SRC}
    ${THRID_PARTY_DIR}/nlohmann_json

    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/include/toolchain

    ${ASCEND_CANN_PACKAGE_PATH}/include/driver
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/trace
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/profiling
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime/runtime/rts

    ${HCOMM_CODE_ROOT}/src/base_comm/resources/ccu/ccu_microcode
)

target_link_libraries(loader PRIVATE
    log
    runnerdb
    rt
)

# 日志染色: 模块 tag
target_compile_definitions(log PRIVATE HCCL_VM_MODULE="UTILS")

# 日志染色: 模块 tag
target_compile_definitions(loader PRIVATE HCCL_VM_MODULE="UTILS")
