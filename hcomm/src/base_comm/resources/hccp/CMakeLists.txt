# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(BUILD_OPEN_PROJECT AND (PRODUCT_SIDE STREQUAL "device"))
    # 添加子目录
    add_subdirectory(stub_tsd)
    add_subdirectory(netco)
    # 定义 hccp_service.bin 可执行文件，在 Device 侧使用
    include(hccp_service.cmake)
endif()

# 添加子目录，Host和Device都涉及
add_subdirectory(rdma_agent)
add_subdirectory(rdma_service)
