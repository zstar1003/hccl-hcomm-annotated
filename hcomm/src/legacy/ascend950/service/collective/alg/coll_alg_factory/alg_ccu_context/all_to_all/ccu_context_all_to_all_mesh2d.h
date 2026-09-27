/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_CONTEXT_ALL_TO_ALL_MESH_2D_H
#define HCCLV2_CCU_CONTEXT_ALL_TO_ALL_MESH_2D_H

#include <vector>
#include <ios>
#include "log.h"
#include "ccu_context_alg_base.h"
#include "ccu_assist.h"
#include "ccu_datatype.h"
#include "ccu_instruction_all_to_all_mesh2d.h"

namespace Hccl {

class CcuContextAlltoAllMesh2D : public CcuContextAlgBase {
public:
    CcuContextAlltoAllMesh2D(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group);
    ~CcuContextAlltoAllMesh2D() override {}

    void Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) override;

private:
    void CalculateArgs(const CcuTaskArgAlltoAllMesh2D* taskArg);
    void InitResources();
    void LoadArgs();
    void ExchangeInfoAndSync();
    void RankSync(uint32_t signalIndex);
    void AxisSync(uint32_t signalIndex);
    void FirstStep();
    void FirstStepOneSlice(uint16_t sliceId);
    void SecondStep();
    void CreateLocalCopyLoop();
    void LocalCopyByLoopGroup(CcuRep::Memory dst, CcuRep::Memory src, GroupOpSize& goPara);

    std::vector<uint32_t> dimSize;
    uint32_t axisId{0};

    std::vector<uint32_t> dimId; // 本rank所在行或列的编号
    uint32_t localId{0};         // 本chip所在行或列的编号
    uint32_t localSize{0};       // 本rank所在行或列的总rank数
    uint32_t anotherId{0};       // 本rank在另一个轴上的Id
    uint32_t anotherSize{0};

    GroupOpSize goSize_;

    // 从外部获取的参数
    CcuRep::Variable input;
    std::vector<CcuRep::Variable> bufferA; // 第一轮的目的地，需要交换
    CcuRep::Variable bufferB;              // 第二轮的起始，不需要交换
    std::vector<CcuRep::Variable> output;  // 第二轮的目的地，需要交换
    std::vector<CcuRep::Variable> token;

    // mem资源准备，记录实际访问的带偏移地址
    std::vector<CcuRep::Memory> inputAddrs;
    std::vector<CcuRep::Memory> bufferAddrs;
    std::vector<CcuRep::Memory> outputAddrs;

    // 在本地的搬运完成标记
    std::vector<CcuRep::MaskSignal> firstSignal;
    std::vector<CcuRep::MaskSignal> secondSignal;

    CcuRep::Variable sliceSize_;
    CcuRep::Variable baseOffset; // 多轮搬运时的每轮基础偏移
    // 地址计算：srcStride=sendRecvSize+sendStride，dstStride=sendRecvSize+recvStride
    CcuRep::Variable firstTransportSize; // a/b块大小
    CcuRep::Variable firstChunkOffset;   // 0/a块大小的偏移
    // 第一轮中，从inputMem，每次循环向每个对端发自己的一片，多次循环中的偏移+步进：
    // (die0--baseOffset+srcStride+D0*srcStride, die1--baseOffset+D0*srcStride+srcStride)
    CcuRep::Variable firstInputStrideLocal;
    CcuRep::Variable firstInputStrideAnother;
    // 第一轮中，写到对端buffer，die0写到对端bufferY，die1写到对端bufferX；共localSize-1个对端，每个对端写anotherSize-1片
    // anotherSize-1次循环的偏移+步进：(localId*sliceSize+localSize*sliceSize)
    CcuRep::Variable firstBufferOffset;
    CcuRep::Variable firstBufferStride;
    // 第一轮中，写到对端output，自身rankId对应到输出偏移：(baseOffset+rankId*dstStride)
    CcuRep::Variable firstOutputOffset;

    CcuRep::Variable secondTransportSize; // b/a块大小
    CcuRep::Variable secondChunkOffset;   // a/0块大小的偏移
    // 第二轮中，从inputMem，在某一次循环中，给每个对端发送自己的一片，共localSize片；给多个对端的偏移+步进：
    // (die0--baseOffset+yId*D0*srcStride+srcStride, die1--baseOffset+xId*srcStride+D0*srcStride)
    CcuRep::Variable secondInputOffset;
    CcuRep::Variable secondInputStride;
    // 第二轮中，从buffer读取，die0读bufferX，die1读bufferY，共localSize-1个对端，为每个对端读anotherSize-1片；偏移+步进：
    // 沿本方向每个rank（dst）步进，die0--D1*sliceSize，die1--D0*sliceSize；
    // 沿另一方向每个rank（src）步进，die0/die1--sliceSize；每次循环中向每个对端发一片
    CcuRep::Variable secondBufferStrideLocal;
    CcuRep::Variable secondBufferStrideAnother;
    // 第二轮中，写到对端output，地址与分片来源相对应，偏移+步进：
    // (die0--baseOffset+xId*dstStride+D0*dstStride, die1--baseOffset+yId*D0*dstStride+dstStride)
    CcuRep::Variable secondOutputOffset;
    CcuRep::Variable secondOutputStride;

    // 跨轴同步信号
    std::string localAxisSignalName;
    std::string anotherAxisSignalName;
    CcuRep::MaskSignal localAxisSignal;
    CcuRep::MaskSignal anotherAxisSignal;

    // 在geneArgs中使用
    uint64_t firstTransportSizeValue{0};
    uint64_t firstChunkOffsetValue{0};
    uint64_t firstInputStrideLocalValue{0};
    uint64_t firstInputStrideAnotherValue{0};
    uint64_t firstBufferOffsetValue{0};
    uint64_t firstBufferStrideValue{0};
    uint64_t firstOutputOffsetValue{0};
    uint64_t secondTransportSizeValue{0};
    uint64_t secondChunkOffsetValue{0};
    uint64_t secondInputOffsetValue{0};
    uint64_t secondInputStrideValue{0};
    uint64_t secondBufferStrideLocalValue{0};
    uint64_t secondBufferStrideAnotherValue{0};
    uint64_t secondOutputOffsetValue{0};
    uint64_t secondOutputStrideValue{0};
};
} // namespace Hccl

#endif // HCCLV2_CCU_CONTEXT_ALL_TO_ALL_MESH_2D_H
