/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_ALG_TEMPLATE_BASE
#define AIV_ALG_TEMPLATE_BASE

#include <memory>
#include <map>
#include <vector>
#include <queue>
#include <string>
#include "const_val.h"
#include "template_utils.h"
#include "instruction.h"
#include "ins_queue.h"
#include "coll_alg_base.h"

namespace Hccl {
using InsQuePtr = std::shared_ptr<InsQueue>;
constexpr uint64_t TOPO_LEN_Y_OFFSET = 8;
constexpr uint64_t TOPO_LEN_Z_OFFSET = 16;
constexpr uint64_t MAX_DIM_NUM = 3;

class AivAlgTemplateBase {
public:
    explicit AivAlgTemplateBase(
        const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
        const std::map<RankId, u32>& tempVirtRankMap);
    virtual ~AivAlgTemplateBase();

    void SetCollOp(const CollAlgOperator& op);
    void SetDmaMode(const DmaMode dmaMode);
    void SetDataType(const DataType& dataType);
    void SetRoot(const u32 root);
    void InitReduceInfo(const ReduceOp& redOp, const DataType& dataType);

    virtual std::string Describe() const = 0;

    virtual HcclResult CalcRes(AlgTempResReq& tempResReq);
    virtual HcclResult CalcResDetour(const RankGraph* rankGraph, AlgTempResReq& tempResReq);
    virtual HcclResult CalcResDetour(ConnectedLinkMgr* linkMgr, AlgTempResReq& tempResReq);
    virtual u32 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType);
    virtual HcclResult CalNumBlocks(u32& numBlocks, u64 dataSize, u32 numBlocksLimit);

    virtual HcclResult GenExtIns(
        const TempFuncs& tempFuncs, const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
        std::vector<InsQuePtr>& tempInsQues);
    HcclResult setPathNumMap(const std::map<u32, u32>& rank2PathNumMap) const;

protected:
    void IncSliceId();

    CollAlgOperator op_;
    OpMode opMode_{OpMode::OPBASE};
    u32 root_{0};
    ReduceOp reduceOp_{ReduceOp::SUM};
    DataType dataType_{DataType::INT32};
    DmaMode dmaMode_ = DmaMode::DEFAULT;

    RankId myRank_{INVALID_RANKID};
    u32 tempRankSize_{0};
    std::vector<std::vector<RankId>> tempVTopo_;
    std::map<RankId, u32> tempVirtRankMap_;
    u32 linkNumBtwPeers_ = 1;

    u32 sliceId_{0}; // 用于组装aivTag
};

} // namespace Hccl

#endif // AIV_ALG_TEMPLATE_BASE
