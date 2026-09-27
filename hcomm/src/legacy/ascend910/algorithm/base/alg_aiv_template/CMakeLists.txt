# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_subdirectory(all_gather)
add_subdirectory(all_gather_v)
add_subdirectory(all_reduce)
add_subdirectory(all_to_all)
add_subdirectory(broadcast)
add_subdirectory(reduce_scatter)
add_subdirectory(reduce_scatter_v)

if(TARGET ccl_kernel)
    target_include_directories(ccl_kernel PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/inc
    )
endif()
