# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(TARGET hccl_v2)
    target_sources(hccl_v2
            PRIVATE
            conn_local_notify_manager.cc
            queue_notify_manager.cc
            queue_wait_group_cnt_notify_manager.cc
            queue_bcast_post_cnt_notify_manager.cc
            conn_local_cnt_notify_manager.cc
            host_device_sync_notify_manager.cc
            ccu_stream_sync_notify_manager.cc
    )
endif()

add_subdirectory(aicpu)