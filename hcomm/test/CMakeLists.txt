# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(CMAKE_SKIP_RPATH FALSE)

# UT
if(ENABLE_UT)
    enable_testing()
    add_subdirectory(ut)
    add_subdirectory(legacy/ut)
endif()

# ST
if(ENABLE_ST)
    add_cann_third_party(protobuf)
    enable_testing()
    add_subdirectory(st/algorithm)
    add_subdirectory(legacy/st/algorithm)
endif()
