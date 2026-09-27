/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_SCATTER_PLANT_LOCAL_REDUCE_H
#define REDUCE_SCATTER_PLANT_LOCAL_REDUCE_H

#include "alg_template_base_pub.h"

namespace hccl {
class ReduceScatterPlantLocalReduce : public AlgTemplateBase {
public:
    explicit ReduceScatterPlantLocalReduce(const HcclDispatcher dispatcher);
    ~ReduceScatterPlantLocalReduce() override;

    HcclResult Prepare(
        void* inputMemPtr, DeviceMem& cclInMem, DeviceMem& outputMem, const Stream& stream,
        std::vector<Stream>& subStreams, std::vector<std::shared_ptr<LocalNotify>>& meshSignal,
        std::vector<std::shared_ptr<LocalNotify>>& meshSignalAux, GroupSlicesInfo& grouSlicesInfo,
        const HcclReduceOp reductionOp, u32 all2allOffset, const HcclDataType dataType, bool isNeedSpaceBorrow,
        bool reverseMemUsage = false, bool isA3CrossNode = false) override;
    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links) override;

protected:
private:
    HcclResult MainRecordSub(Stream& mainStream, u32 firstSubStreamIndex, u32 totalTask); // 主流通知从流
    HcclResult SubWaitMain(u32 firstSubStreamIndex, u32 totalTask);                       // 从流等待主流
    HcclResult MainWaitSub(Stream& mainStream, u32 firstSubStreamIndex, u32 totalTask);   // 主流等待从流
    HcclResult SubRecordMain(u32 firstSubStreamIndex, u32 totalTask);                     // 从流等待主流
    HcclResult MainRecordLocalReduceWait(u32 lRMainStreamIndex);
    u32 CalcOutputIndex(const u32 round);
    bool isLastGroup(const u32 groupId);
    bool isLastRank(const u32 rankId) const;
    bool isLastBlockData(const u32 outputIndex) const;
    HcclResult LocalCopy(u32 groupId, const MemBlockInfo& memBlockInfo);
    HcclResult RunAlltoAll(const std::vector<LINK>& links, u32 groupId, const MemBlockInfo& memBlockInfo);
    HcclResult RunGroupAlltoAll(const std::vector<LINK>& links, u32 groupId, const MemBlockInfo& memBlockInfo);
    HcclResult RunLocalReduce(u32 groupId, const MemBlockInfo& memBlockInfo);
    u32 rankSize_{0};
    u32 localRank_{0};           // Level0当前rank对应的localRAnk
    u32 all2allSubStreamNum_{0}; // All2All需要的从流个数（用于主从流同步）
    u32 lRMainStreamId_{0};
    u32 all2allOffset_{0};           // 单机场景 or Level1场景下为0，仅多机Level0为1
    std::vector<Stream> subStreams_; // 从流
    const std::vector<std::shared_ptr<LocalNotify>>* meshSignalPtr_{nullptr};
    const std::vector<std::shared_ptr<LocalNotify>>* meshSignalAuxPtr_{nullptr};
    GroupSlicesInfo groupSlicesInfo_;
    void* inputMemPtr_{nullptr};
    bool isNeedSpaceBorrow_{false}; // 是否需要借用CCLIN空间完成通信(算子维度)
    bool isA3CrossNode_{false};
    UserMemType scratchMemType_{UserMemType::INPUT_MEM};
    UserMemType outputMemType_{UserMemType::OUTPUT_MEM};
};
} // namespace hccl
#endif /* REDUCE_SCATTER_PLANT_LOCAL_REDUCE_H */
