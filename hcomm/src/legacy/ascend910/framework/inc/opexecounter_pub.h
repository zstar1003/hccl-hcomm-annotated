/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLOPEXECOUNTER_PUB_H
#define HCCLOPEXECOUNTER_PUB_H

#include <map>
#include <string>
#include <memory>
#include "dispatcher.h"
#include "hccl/hccl_types.h"
#include "aicpu_operator_pub.h"

namespace hccl {

constexpr int HEAD = 0;
constexpr int TAIL = 1;

HcclResult StarsCounter(
    const HcclDispatcher& dispatcher, Stream& stream, int flag, bool isAicpuMode, bool isRetry, bool isAivMode);
HcclResult FftsHeadCounter(const HcclDispatcher& dispatcher, Stream& stream);
HcclResult FftsTailCounter(const HcclDispatcher& dispatcher, Stream& stream);
HcclResult GetOpCountInfo(OpCounterInfo& opCounterInfo);
HcclResult ClearOpCounterMem();

class OpExeCounter {
public:
    static OpExeCounter& GetInstance(s32 deviceLogicID);
    HcclResult AddCounter(const HcclDispatcher& dispatcher, Stream& stream, int flag);
    HcclResult GetCounter(std::pair<int32_t, int32_t>& counter);
    HcclResult InitCounter();
    HcclResult DeInitCounter();
    HcclResult GetOpCountInfo(OpCounterInfo& opCounterInfo);
    HcclResult ClearOpCounterMem();

private:
    OpExeCounter() = default;
    ~OpExeCounter();
    void ReleaseMemHandles();

    void* headCountMem_{};
    void* tailCountMem_{};
    void* addOneMem_{};
    int refCount_ = 0;
    uint32_t memSize_ = 0;
    bool isNeedOpCounter_ = false;
};
} // namespace hccl

#endif // HCCLOPEXECOUNTER_PUB_H
