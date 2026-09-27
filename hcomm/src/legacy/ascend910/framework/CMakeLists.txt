# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(PRODUCT_SIDE STREQUAL "host")
    # 定义 hcomm 库，在 host 侧使用
    include(hcomm.cmake)
else()
    # 定义 ccl_kernel 库，在 device 侧使用
    include(ccl_kernel.cmake)
    # 添加 device 子目录
    add_subdirectory(device)
endif()

# 添加子目录
add_subdirectory(common)
add_subdirectory(communicator)
add_subdirectory(hcom)
add_subdirectory(op_base)
add_subdirectory(group)
add_subdirectory(cluster_maintenance)
add_subdirectory(nslbdp)
