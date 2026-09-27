/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_DEV_CAPABILITY_H
#define HCCLV2_DEV_CAPABILITY_H

#include "dev_capability.h"
#include <map>
#include "dev_type.h"
#include "reduce_op.h"
#include "data_type.h"

namespace Hccl {

class DevCapability {
public:
    DevCapability(const DevCapability& that) = delete;

    DevCapability& operator=(const DevCapability& that) = delete;

    static DevCapability& GetInstance();

    void Init(DevType givenDevType);

    void Reset(); // LLT 使用

    const std::map<DataType, bool>& GetInlineReduceDataTypeMap() const { return inlineReduceDataTypeMap; }

    const std::map<ReduceOp, bool>& GetInlineReduceOpMap() const { return inlineReduceOpMap; }

    u32 GetSdmaInlineReduceAlignBytes() const { return sdmaInlineReduceAlignBytes; }

    u32 GetNotifySize() const { return notifySize; }

    u64 GetSdmaSendMaxSize() const { return sdmaSendMaxSize; }

    u64 GetRdmaSendMaxSize() const { return rdmaSendMaxSize; }

    bool IsSupportDevNetInlineReduce() const { return isSupportDevNetInlineReduce; }

    bool IsSupportWriteWithNotify() const { return isSupportWriteWithNotify; }

    bool IsSupportStarsPollNetCq() const { return isSupportStarsPollNetCq; }

private:
    DevCapability();

    void Load910ACap();
    void Load910A3Cap();
    void LoadV82Cap();

    void Load910A910A3CommonCap();

    DevType devType;
    bool isInit{false};

    std::map<DataType, bool> inlineReduceDataTypeMap;
    std::map<ReduceOp, bool> inlineReduceOpMap;

    u32 sdmaInlineReduceAlignBytes{0};
    u32 notifySize{0};

    u64 sdmaSendMaxSize{0}; // 节点内单个SDMA任务发送数据支持的最大数据量
    u64 rdmaSendMaxSize{0}; // 节点间RDMA发送数据单个WQE支持的最大数据量

    bool isSupportDevNetInlineReduce{false};
    bool isSupportWriteWithNotify{false};
    bool isSupportStarsPollNetCq{false};
};

} // namespace Hccl
#endif
