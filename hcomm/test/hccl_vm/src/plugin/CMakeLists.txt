# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_subdirectory(runner)
add_subdirectory(checker)

#install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/checker"
#    DESTINATION "plugin/validate"
#    # 文件的权限 (755)
#    FILE_PERMISSIONS
#        OWNER_READ OWNER_WRITE OWNER_EXECUTE
#        GROUP_READ GROUP_EXECUTE
#        WORLD_READ WORLD_EXECUTE
#    # 目录的权限 (755)
#    DIRECTORY_PERMISSIONS
#        OWNER_READ OWNER_WRITE OWNER_EXECUTE
#        GROUP_READ GROUP_EXECUTE
#        WORLD_READ WORLD_EXECUTE
#)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/insight"
    DESTINATION "plugin/visualization"
    # 文件的权限 (755)
    FILE_PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE
    # 目录的权限 (755)
    DIRECTORY_PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE
    PATTERN "frontend" EXCLUDE
    PATTERN "frontend_v3" EXCLUDE
)
