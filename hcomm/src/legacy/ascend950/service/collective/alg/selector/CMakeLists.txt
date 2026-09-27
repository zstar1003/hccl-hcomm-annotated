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
    target_sources(hccl_v2 PRIVATE
            execute_selector.cc
            base_selector.cc
            mc2_selector.cc
            selector_registry.cc
            auto_selector_base.cc
            all_gather_auto_selector.cc
            all_gather_v_auto_selector.cc
            all_reduce_auto_selector.cc
            reduce_scatter_auto_selector.cc
            alltoall_auto_selector.cc
            alltoallv_auto_selector.cc
            alltoallvc_auto_selector.cc
            batch_send_recv_auto_selector.cc
            broadcast_auto_selector.cc
            reduce_auto_selector.cc
            scatter_auto_selector.cc
            send_recv_auto_selector.cc
            all_gather_v_auto_selector.cc
            reduce_scatter_v_auto_selector.cc
    )
endif()