# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/topoinfo_ranktableParser.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/topoinfo_ranktableStandard.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/topoinfo_ranktableConcise.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/topoinfo_ranktableHeterog.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/topoinfo_roletableParser.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/topoinfo_detect.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/topoinfo_detect_scalable.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/topoinfo_exchange_agent.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/topoinfo_exchange_server.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/topoinfo_exchange_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/topoinfo_exchange_dispatcher.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/topoinfo_parse.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/topoinfo_ranktable_partition.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_whitelist.cc
)

set(device_src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/topo_device/topoinfo_detect_device.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/topo_device/topoinfo_ranktable_partition_device.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/topo_device/topoinfo_ranktableParser_device.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/topo_device/topoinfo_parse_device.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/topo_device/topoinfo_ranktableConcise_device.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/topo_device/topoinfo_ranktableHeterog_device.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/topo_device/topoinfo_ranktableStandard_device.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/topo_device/topoinfo_roletableParser_device.cc
)

if(TARGET hcomm)
    target_sources(hcomm PRIVATE
        ${src_list}
        ${CMAKE_CURRENT_SOURCE_DIR}/topoinfo_ranktableOffline.cc
    )
    target_include_directories(hcomm PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()

if(TARGET ccl_kernel)
    target_sources(ccl_kernel PRIVATE
        ${device_src_list}
    )
    target_include_directories(ccl_kernel PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
