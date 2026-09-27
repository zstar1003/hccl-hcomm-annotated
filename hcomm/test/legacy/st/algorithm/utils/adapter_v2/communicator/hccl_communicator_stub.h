/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ADAPTER_V2_COMMUNICATOR_PUB_H
#define ADAPTER_V2_COMMUNICATOR_PUB_H

#include <memory>
#include <mutex>
#include "hccl_params_pub.h"
#include "hccl_types.h"
#include "ccu_ins_preprocessor.h"

namespace Hccl {
class CommunicatorImpl;
class HcclCommunicator {
public:
    explicit HcclCommunicator(const CommParams& commParams);
    HcclCommunicator(const CommParams& commParams, const HcclCommConfig* config);
    ~HcclCommunicator();

    HcclResult Init(const std::string& ranktableM, std::string& topoPath);
    void DeInit() const;

    HcclResult LoadOpbasedCollOp(CollOpParams& opParams, std::string& algName);
    HcclResult LoadOffloadCollOp(std::string& opTag, CollOpParams& opParams);

    // MC2 流程专用
    HcclResult GetRankSize(uint32_t* rankSize);
    HcclResult GetRankId(uint32_t& rankId);
    CcuInsPreprocessor* GetCcuInsPreprocessor();
    void TransformTask();

private:
    CommParams commParams;
    HcclCommConfig config{};
    std::unique_ptr<CommunicatorImpl> pimpl;
    CommunicatorImpl* GetCommImpl();
    std::mutex serialMutex;
};

} // namespace Hccl

#endif
