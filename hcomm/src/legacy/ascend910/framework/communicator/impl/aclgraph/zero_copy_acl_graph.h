/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ZERO_COPY_ACL_GRAPH_H
#define ZERO_COPY_ACL_GRAPH_H
#include <atomic>
#include <memory>
#include <hccl/hccl_types.h>
#include <set>
#include "hccl_communicator_attrs.h"
#include "hccl/base.h"
#include "hccl_impl_pub.h"
#include "zero_copy/zero_copy_memory_agent.h"
#include "coll_alg_operator.h"
#include "hccl_alg.h"
namespace hccl {
class ZeroCopyAclGraph {
public:
    ZeroCopyAclGraph();
    ~ZeroCopyAclGraph() = default;
    bool
    SetAclGraphZeroCopyMode(DevType deviceType, HcclCMDType opType, OpParam& opParam, HcclAlg* impl, u64 bufferSize);
    std::string GetTagPrefix();
    void SetRetryEnable(bool retryEnable);

private:
    bool IsAlgoSupportAclGraphZeroCopyMode(HcclCMDType opType, OpParam& opParam, HcclAlg* impl, u64 bufferSize);
    bool IsScratchMemorySupportAclGraphZeroCopyMode(const OpParam& opParam, u64 bufferSize, u64 scratchMemSize);
    bool SetGraphMode(HcclCMDType opType, OpParam& opParam, HcclAlg* impl, u64 bufferSize);
    bool AlgoCheck(OpParam& opParam, std::unique_ptr<CollAlgOperator>& algo, u64 bufferSize);
    bool IsAclGraphZeroCopyAlgAvailable(HcclCMDType opType, OpParam& opParam);

private:
    std::atomic<u32> tagResourceIndex_;
    std::set<HcclCMDType> algoSet_;
    bool retryEnable_;
};
} // namespace hccl
#endif // end of ZERO_COPY_ACL_GRAPH_H
