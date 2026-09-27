/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_ALLREDUCE_MID_COUNT_FOR_910_93_EXECUTOR_H
#define COLL_ALLREDUCE_MID_COUNT_FOR_910_93_EXECUTOR_H
#include "coll_all_reduce_executor.h"
namespace hccl {
class CollAllReduceMidCountFor91093Executor : public CollAllReduceExecutor {
public:
    explicit CollAllReduceMidCountFor91093Executor(
        const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher);
    ~CollAllReduceMidCountFor91093Executor() override = default;

private:
    /* *************** 资源计算 *************** */
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType) const;
    HcclResult CalcLevel1CommInfo(
        TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel2CommInfo(
        TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;

    /* *************** 算法编排 *************** */
    u64 CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize) override;
    HcclResult KernelRun(const OpParam& param, ExecMem& execMem) override;
};

} // namespace hccl

#endif
