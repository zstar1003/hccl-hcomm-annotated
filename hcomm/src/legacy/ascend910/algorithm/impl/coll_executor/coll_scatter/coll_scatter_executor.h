/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_SCATTER_EXECUTOR_H
#define COLL_SCATTER_EXECUTOR_H

#include "coll_comm_executor.h"
#include "coll_alg_exec_registry.h"

namespace hccl {

class CollScatterExecutor : public CollCommExecutor {
public:
    explicit CollScatterExecutor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher);
    ~CollScatterExecutor() override = default;

    HcclResult Orchestrate(OpParam& param, AlgResourceResponse& algRes) override;

protected:
    /* *************** 资源计算 *************** */
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;
    virtual HcclResult CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType);

    /* *************** 算法编排 *************** */
    // 按Level0、Level1、Level2可继续进行拆分。
    virtual HcclResult KernelRunLevel1(
        DeviceMem& inputMem, u64 count, HcclDataType dataType, u32& commIndex, u32 root, u32& subRoot,
        CommPlane commLevel, Stream& stream);
    // 用于需要Loop的Executor
    virtual HcclResult RunLoop(OpParam& param, AlgResourceResponse& algRes);
    virtual HcclResult RunLoopInner(OpParam& param, ExecMem& execMem, AlgResourceResponse& algRes);

    virtual bool IsHugeData(u64 curSize);
    /* *************** 通用工具 *************** */
    virtual HcclResult PrepareDataSlice(u64 dataCount, u32 unitSize, u32 sliceNum, std::vector<Slice>& dataSlice);
    virtual HcclResult ReorderSlice(std::vector<Slice>& dataSlice, std::vector<u32>& order);

    bool DMAReduceFlag_{false}; // 是否DMA消减的标志
private:
};

} // namespace hccl

#endif
