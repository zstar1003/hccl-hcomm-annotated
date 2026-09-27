# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 此处不包含alg文件夹， 算法平台单独打包为独立so文件
# 基础框架只引入部分头文件

add_subdirectory(instruction)
add_subdirectory(primitive)
add_subdirectory(alg)

if(TARGET hccl_v2)
    target_sources(hccl_v2
            PRIVATE
            coll_operator.cc
            coll_operator_check.cc
    )
endif()

if(TARGET ccl_kernel)
    target_sources(ccl_kernel PRIVATE
            coll_operator.cc
            coll_operator_check.cc
    )
endif()