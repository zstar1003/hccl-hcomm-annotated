/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRANSPORT_CNT_NOTIFY_H
#define TRANSPORT_CNT_NOTIFY_H

#include <vector>
#include <unordered_map>
#include "base_mem_transport.h"
#include "local_cnt_notify.h"
namespace Hccl {

class CntNotifyResHelper {
public:
    BaseMemTransport::LocCntNotifyRes
    GetCntNotifyRes(const unordered_map<u32, vector<LocalCntNotify*>>& topicIdCntNotifyVecMap) const;

    u32 GetIndex(vector<char>& desc, u32 topicId, u32 pos) const;
};
} // namespace Hccl

#endif
