/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../ut_hcomm_base.h"
#include "hccl_net_dev.h"
#include "hccl_network.h"
#include "hccl_socket.h"
#include "network_manager_pub.h"
#include "reged_mem_mgr.h"
#include "mem_name_repository_pub.h"
#include "global_net_dev_manager.h"
#include "transport_p2p_pub.h"
#include "transport_device_p2p_pub.h"
#include "channel_process.h"
#include "aicpu_ts_urma_channel_kernel.h"
#include "launch_aicpu.h"

using namespace hcomm;
using namespace hccl;

#include "../hccs_endpoint_test_common.h"
// namespace
