# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_subdirectory(ccu)
add_subdirectory(common)
add_subdirectory(resource)
target_sources(hccl_v2_stest
        PUBLIC
        ##st_adapter_rts.cc
        st_adapter_trace.cc
        st_buffer.cc
        st_cnt_notify.cc
        st_dev_ub_connection.cc
        st_p2p_connection.cc
        st_p2p_transport.cc
        st_trace.cc
        st_ub_ci_updater.cc
        st_ub_local_notify.cc
        st_ub_mem_transport.cc
        st_ub_transport_lite_impl.cc

        st_stream.cc
        st_stream_lite.cc
        st_rtsq_a5.cc
        st_notify_lite.cc
        st_cnt_1ton_notify_lite.cc
        st_cnt_nto1_notify_lite.cc
        st_rts_1ton_cnt_notify.cc
        st_rts_cnt_notify.cc

        st_tokenInfo_manager.cc
)
