# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

#net_adapt
aux_source_directory(net_adapt/init srcs)
aux_source_directory(net_adapt/tbl_in srcs)
aux_source_directory(net_adapt/tbl_out srcs)
aux_source_directory(net_adapt/util srcs)
aux_source_directory(net_adapt/plugin/ch_letcp/client srcs)
aux_source_directory(net_adapt/plugin/ch_letcp/server srcs)

#cncoi
aux_source_directory(cncoi/src/comm srcs)
aux_source_directory(cncoi/src/pub srcs)
aux_source_directory(cncoi/src/sub srcs)

#idl
aux_source_directory(idl/src/bdep/src srcs)
aux_source_directory(idl/src/bdep/avl_dopra/src/v_avl3/src srcs)
aux_source_directory(idl/src/bdep/avl_dopra/src/v_avlbase/src srcs)
aux_source_directory(idl/src/design_pattern srcs)
aux_source_directory(idl/src/libbackfin/system_base srcs)
aux_source_directory(idl/src/libbackfin/debug_tools srcs)
aux_source_directory(idl/src/libbackfin/system_base/channel/comm srcs)
aux_source_directory(idl/src/libbackfin/system_base/channel/server srcs)
aux_source_directory(idl/src/libbackfin/system_base/channel/client srcs)
aux_source_directory(idl/src/libbackfin/utils srcs)
aux_source_directory(idl/src/plugin/mux_linux srcs)
aux_source_directory(idl/src/plugin/mux_job srcs)
aux_source_directory(idl/src/data_sync/dc srcs)
aux_source_directory(idl/src/data_sync/processor/pub_sub/pub/conn srcs)
aux_source_directory(idl/src/data_sync/processor/pub_sub/pub/puber srcs)
aux_source_directory(idl/src/data_sync/processor/pub_sub/pub/sess srcs)
aux_source_directory(idl/src/data_sync/processor/pub_sub/pub/table_type srcs)
aux_source_directory(idl/src/data_sync/processor/pub_sub/sub/conn srcs)
aux_source_directory(idl/src/data_sync/processor/pub_sub/sub/data srcs)
aux_source_directory(idl/src/data_sync/processor/pub_sub/sub/sess srcs)
aux_source_directory(idl/src/data_sync/processor/pub_sub/sub/suber srcs)
aux_source_directory(idl/src/data_sync/protocol/msgcodec/base srcs)
aux_source_directory(idl/src/plugin/twl_tmr/src srcs)
aux_source_directory(util/. srcs)

add_library(net_co SHARED)

target_sources(net_co PRIVATE ${srcs})
set(NETCO_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../netco/)

set(bkf_dep_include
    ${NETCO_PATH}/idl/src/bdep/include
    ${NETCO_PATH}/idl/src/bdep/include/avl_dopra
    ${NETCO_PATH}/idl/src/bdep/avl_dopra/src/v_avlbase/include
)

set(includes
    ${bkf_dep_include}
    ${NETCO_PATH}/idl/interface/include/dc
    ${NETCO_PATH}/idl/interface/include/debug_tools
    ${NETCO_PATH}/idl/interface/include/design_pattern
    ${NETCO_PATH}/idl/interface/include/mux
    ${NETCO_PATH}/idl/interface/include/mux_job
    ${NETCO_PATH}/idl/interface/include/pub_sub
    ${NETCO_PATH}/idl/interface/include/system_base
    ${NETCO_PATH}/idl/interface/include/twl_linux_tmr
    ${NETCO_PATH}/idl/interface/include/utils
    ${NETCO_PATH}/idl/interface/include/init_factory
    ${NETCO_PATH}/cncoi/include
    ${NETCO_PATH}/cncoi/src/include
    ${NETCO_PATH}/inc
    ${NETCO_PATH}/net_adapt/inc_frd
    ${NETCO_PATH}/idl/interface/include/system_base
    ${NETCO_PATH}/idl/interface/include/debug_tools
    ${NETCO_PATH}/net_adapt/plugin/ch_letcp/inc
    ${NETCO_PATH}/idl/src/libbackfin/include
    ${NETCO_PATH}/idl/src/data_sync/processor/pub_sub/pub/include
    ${NETCO_PATH}/idl/src/data_sync/processor/pub_sub/sub/include
    ${NETCO_PATH}/idl/src/plugin/twl_tmr/include
    ${NETCO_PATH}/idl/src/plugin/twl_tmr/inner
)

target_include_directories(net_co PRIVATE
    ${includes}
)

target_compile_options(net_co PRIVATE
	-Werror
    -Wno-cast-function-type
    -fno-common
	-fno-strict-aliasing
    $<$<CONFIG:Debug>:-Og -g>
    $<$<CONFIG:Release>:-O3>
)

target_link_options(net_co PRIVATE
    -Wl,-Bsymbolic
    -rdynamic
)

# 符号隐藏: 仅导出白名单符号; UT/ST 打桩依赖动态符号, 不挂载
if(NOT ENABLE_TEST)
    target_link_options(net_co PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_LIST_DIR}/net_co.map")
    set_property(TARGET net_co APPEND PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/net_co.map")
endif()

#链接动态库
target_link_libraries(net_co PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<BUILD_INTERFACE:ascend_hal_headers>
    $<BUILD_INTERFACE:msprof_headers>
    $<BUILD_INTERFACE:slog_headers>
    -Wl,--no-as-needed
    c_sec
    $<$<STREQUAL:${PRODUCT_SIDE},host>:unified_dlog>
    -Wl,--as-needed
    -lrt
    -ldl
)
