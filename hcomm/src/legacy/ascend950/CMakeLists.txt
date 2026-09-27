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
    # 添加 hccl_v2 链接库，host 侧使用
    include(hccl_v2.cmake)
    # 添加 ccl_dpu 链接库，host 侧使用
    include(ccl_dpu.cmake)
endif()

# 添加子目录
add_subdirectory(common)
add_subdirectory(framework)
add_subdirectory(service)
add_subdirectory(unified_platform)
add_subdirectory(interface)
