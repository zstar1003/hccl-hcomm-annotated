/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_ALLREDUCE_ARS_FOR_910_93_EXECUTOR_H
#define COLL_ALLREDUCE_ARS_FOR_910_93_EXECUTOR_H

#include "coll_all_reduce_ring_for_910_93_executor.h"

namespace hccl {
class CollAllReduceARSFor91093Executor : public CollAllReduceRingFor91093Executor {
public:
    explicit CollAllReduceARSFor91093Executor(
        const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher);
    ~CollAllReduceARSFor91093Executor() override = default;

private:
    /* *************** 资源计算 *************** */
    u32 intraRingSize_ = 0;
    HcclResult CalcOptimalIntraRing(const OpParam& param) override;
    HcclResult CalcLevel0CommInfo(
        TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel1CommInfo(
        TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcStreamNum(u32& streamNum) override;
    HcclResult GetLevelCommInfo() override;
};

} // namespace hccl

#endif
