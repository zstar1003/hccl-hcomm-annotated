/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_ACLGRAPH_CALLBACK_H
#define HCCL_ACLGRAPH_CALLBACK_H

#include <mutex>
#include "hccl_communicator.h"
#include "hccl_common.h"

namespace hccl {
struct AclgraphDestroyCallbackParam {
    u64 modelId;
};

class AclgraphCallback {
public:
    static AclgraphCallback& GetInstance();
    HcclResult CleanCaptureRes(u64 modelId);
    void CleanCaptureRes(HcclCommunicator* communicator);
    HcclResult
    InsertNewTagToCaptureResMap(HcclCommunicator* communicator, const std::string& newTag, const OpParam& opParam);
    HcclResult RegisterModelId(HcclCommunicator* communicator, aclmdlRI rtModel, u64 modelId);

private:
    AclgraphCallback() = default;
    ~AclgraphCallback();
    HcclResult RegisterDestroyCallbackLocked(aclmdlRI rtModel, u64 modelId);
    std::mutex resMutex_;
    std::unordered_map<u64, std::unordered_map<HcclCommunicator*, std::unordered_set<std::string>>> captureResMap_;
    std::unordered_map<u64, AclgraphDestroyCallbackParam> captureCallbackParamMap_;
};

} // namespace hccl
#endif // HCCL_ACLGRAPH_CALLBACK_H
