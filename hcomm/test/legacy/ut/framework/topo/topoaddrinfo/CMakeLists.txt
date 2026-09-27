# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

target_sources(hccl_v2_utest
        PUBLIC
        ut_eid.cc
        ut_npu_nic_affinity.cc
        ut_topoaddrinfo.cc
)

# 编译生产代码到测试二进制，代替 #include .c，避免 geninfo 行号错乱
target_sources(hccl_v2_utest PRIVATE
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/topo/topo_addr_info/src/npu_nic_affinity.c
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/topo/topo_addr_info/src/xml_parser.c
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/topo/topo_addr_info/src/topo_addr_info_log.c
)

# 覆盖路径宏，使测试在 /tmp 下构造假文件系统
target_compile_definitions(hccl_v2_utest PRIVATE
    XML_PATH="/tmp/ut_virtualTopology.xml"
    HCA_NET_PATH_TEMPLATE="/tmp/ut_hca/%s/device/net"
)

# 拦截 getifaddrs/freeifaddrs，返回假网络接口数据
target_link_options(hccl_v2_utest PRIVATE
    -Wl,--wrap=getifaddrs,--wrap=freeifaddrs
)
