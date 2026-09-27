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
        ut_queue_notify_manager_test.cc
        ut_conn_local_notify_manager_test.cc
        ut_queue_cnt_notify_manager.cc
        ut_conn_local_cnt_notify_manager.cc
        ut_queue_bcast_post_cnt_notify_manager.cc
        ut_host_device_sync_notify_manager.cc
        ut_host_device_sync_notify_lite_mgr.cc
        ut_queue_notify_lite_mgr.cc
        ut_cnt_1ton_notify_lite_mgr.cc
        ut_cnt_nto1_notify_lite_mgr.cc
)