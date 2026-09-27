/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV1_DEVICE_INFO_RECORDER_H
#define HCCLV1_DEVICE_INFO_RECORDER_H

#include "llt_common.h"
#include "topo_meta.h"
#include "checker_def.h"
using namespace checker;

namespace hccl {

class DeviceInfoRecorder {
public:
    static DeviceInfoRecorder* Global();
    void Reset();

    void InitDeviceInfo(TopoMeta topoMeta, RankTable_t& rankTable, CheckerDevType uniDevType);
    std::map<u32, CheckerDevType> rankId2devType;
    std::map<u32, u32> rankId2superdeviceId;
};

} // namespace hccl

#endif
