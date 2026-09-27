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
        ut_ub_mem_transport.cc
        # ut_p2p_transport.cc
        ut_ub_transport_lite_impl.cc
        ut_ub_transport_lite_impl_wait_with_timeout.cc
        ut_p2p_transport_lite_impl_wait_with_timeout.cc
        ut_roce_transport_lite_impl.cc
)