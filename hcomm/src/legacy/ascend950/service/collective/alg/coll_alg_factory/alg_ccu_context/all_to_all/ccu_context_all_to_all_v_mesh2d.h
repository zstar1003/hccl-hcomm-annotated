/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_CONTEXT_ALL_TO_ALL_V_MESH_2D_H_
#define HCCLV2_CCU_CONTEXT_ALL_TO_ALL_V_MESH_2D_H_

#include <vector>

#include <ios>
#include "log.h"
#include "ccu_context_alg_base.h"
#include "ccu_datatype.h"
#include "ccu_instruction_all_to_all_v_mesh2d.h"

namespace Hccl {
// a2a 对每个对端的发送接收信息
struct A2AVsingleSendRecvInfo {
    CcuRep::Variable sendOffset;
    CcuRep::Variable recvOffset;
    CcuRep::Variable sendTailSizeA; // 本rank给其他rank要发的尾块
    CcuRep::Variable sendTailSizeB; // 本rank给其他rank要发的尾块
    CcuRep::Variable sendTailSize;
    CcuRep::Variable recvTailSizeA; // 本rank从其他所有rank要收的数据
    CcuRep::Variable recvTailSizeB; // 本rank从其他所有rank要收的数据
    CcuRep::Variable sendLoopNum;   // 本rank给其他所有rank要发的轮数
    CcuRep::Variable recvLoopNum;   // 本rank从其他所有rank要收的数据
};

class CcuContextAllToAllVMesh2D : public CcuContextAlgBase {
public:
    CcuContextAllToAllVMesh2D(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group);
    ~CcuContextAllToAllVMesh2D() override {}

    void Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) override;

protected:
    // a2a 对每个对端的发送接收信息
    struct A2AVsingleSendRecvInfo {
        CcuRep::Variable sendOffset;
        CcuRep::Variable recvOffset;
        CcuRep::Variable sendTailSizeA; // 本rank给其他所有rank要发的数据
        CcuRep::Variable sendTailSizeB; // 本rank给其他所有rank要发的数据
        GroupOpSize sendTailGoSizeA;
        GroupOpSize sendTailGoSizeB;
        CcuRep::Variable sendTailSize;
        CcuRep::Variable recvTailSizeA; // 本rank从其他所有rank要收的数据
        CcuRep::Variable recvTailSizeB; // 本rank从其他所有rank要收的数据
        CcuRep::Variable sendLoopNum;   // 本rank给其他所有rank要发的轮数
        CcuRep::Variable recvLoopNum;   // 本rank从其他所有rank要收的数据
    };
    void GenAddrVariables(
        std::vector<CcuRep::Variable>& input, std::vector<CcuRep::Variable>& output,
        std::vector<CcuRep::Variable>& token);

    void CalcGroupSrcDst(std::vector<CcuRep::Memory>& src, std::vector<CcuRep::Memory>& dst);
    void LoadAll2allSendRecvInfo(A2AVsingleSendRecvInfo sendRecvInfo);

private:
    void InitResources();
    void LoadArgs();
    void ExchangeInfoAndSync();
    void RankSync(uint32_t signalIndex);
    void PostSync();
    void AxisSync(uint32_t signalIndex);
    void FirstStep();
    void SecondStep();
    void CalculateArgs();
    void DoAll2AllVMultiLoop();
    void UpdateLoopRecorder(uint16_t flag);
    uint32_t CalcDstRank(uint32_t sliceId, uint32_t peerId) const;
    uint32_t CalcTransIdx(uint32_t peerId) const;
    void GroupCopyToDstOutput(uint16_t sliceId, uint16_t peerId);
    void WriteToDstOutput(uint16_t sliceId, uint16_t peerId);
    void WriteToDstScratch(uint16_t sliceId, uint16_t peerId);
    void ReadFromSrc(uint16_t sliceId, uint16_t peerId);
    void CopyLoopNumRecorder();

    CcuRep::Variable input_;
    std::vector<CcuRep::Variable> output_;
    std::vector<CcuRep::Variable> scratch_;
    std::vector<CcuRep::Variable> token_;
    std::vector<std::vector<CcuRep::Variable>> sendLoopNumRecorder_; // 记录同轴的卡a要给另一个轴的卡b发多少轮
    std::vector<std::vector<CcuRep::Variable>> recvLoopNumRecorder_; // 记录同轴的卡a要从另一个轴的卡b收多少轮
    std::vector<std::vector<CcuRep::Variable>> LocSendLoopNumRecorder_; // 记录同轴的卡a要给另一个轴的卡b发多少轮
    std::vector<std::vector<CcuRep::Variable>> LocRecvLoopNumRecorder_; // 记录同轴的卡a要从另一个轴的卡b收多少轮
    std::vector<CcuRep::Variable> sendRecorder_;

    std::vector<CcuRep::Variable> isPostFlag_;
    CcuRep::Variable xnConst1_;
    CcuRep::Variable completedRankCount_;
    CcuRep::Variable xnHalfTransportSize_;
    CcuRep::Variable xnMaxTransportSize_;
    CcuRep::Variable curSendTailSize_;
    GroupOpSize xnHalfTransportGoSize_;
    GroupOpSize curSendTailGoSize_;
    std::vector<A2AVsingleSendRecvInfo> sendRecvInfo_;
    // srcOffset_，dstOffset_用于记录已经操作的数据量，主要影响在input和output上的偏移
    CcuRep::Variable srcOffset_;
    CcuRep::Variable dstOffset_;
    uint32_t axisId_{0};
    std::vector<uint32_t> dimSize_; // 每个维度的大小
    std::vector<uint32_t> dimId_;   // 本rank所在行或列的编号
    uint32_t localId_{0};           // 本rank所在行或列的编号
    uint32_t localSize_{0};         // 本rank所在行或列的总rank数
    uint32_t anotherId_{0};         // 本rank在另一个轴上的Id
    uint32_t anotherSize_{0};       // 本rank所在另一个轴上的总rank数

    // 中间步骤用的地址寄存器
    std::vector<CcuRep::Memory> inputAddrs_;
    std::vector<CcuRep::Memory> bufferAddrs_;
    std::vector<CcuRep::Memory> outputAddrs_;

    // firstScratchBaseOffset_，secondScratchBaseOffset_，大小为scratchmem的一半
    CcuRep::Variable firstScratchBaseOffset_;
    CcuRep::Variable secondScratchBaseOffset_;
    // localRank要 往/从 remoteRank的scratchmem上 写/读
    // offset指向localRank第一次在remoteRank上操作的地址
    // step指的是localRank在remoteRank上的每次操作的步长
    CcuRep::Variable firstScratchSliceOffset_;
    CcuRep::Variable firstScratchSliceStep_;
    CcuRep::Variable secondScratchSliceOffset_;
    CcuRep::Variable secondScratchSliceStep_;

    // 计算参数用
    uint64_t scratchSliceBias{0}; // scratchmem一半的大小
    uint64_t scratchSliceSize{0}; // scratchmem上每一格的大小
    uint64_t firstScratchBaseOffset{0};
    uint64_t secondScratchBaseOffset{0};
    uint64_t firstScratchSliceOffset{0};
    uint64_t firstScratchSliceStep{0};
    uint64_t secondScratchSliceOffset{0};
    uint64_t secondScratchSliceStep{0};

    // 在本地的搬运完成标记
    std::vector<CcuRep::MaskSignal> firstSignal_;
    std::vector<CcuRep::MaskSignal> secondSignal_;

    // 跨轴同步信号
    std::string localAxisSignalName_;
    std::string anotherAxisSignalName_;
    CcuRep::MaskSignal localAxisSignal_;
    CcuRep::MaskSignal anotherAxisSignal_;
};
} // namespace Hccl

#endif // HCCLV2_CCU_CONTEXT_ALL_TO_ALL_V_MESH_2D_H_
