/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_CONTEXT_HALF_ALL_TO_ALL_V_MESH_1D_H_
#define HCCLV2_CCU_CONTEXT_HALF_ALL_TO_ALL_V_MESH_1D_H_

#include <vector>

#include <ios>
#include "log.h"
#include "ccu_context_alg_base.h"
#include "ccu_assist.h"
#include "ccu_datatype.h"
#include "ccu_instruction_half_alltoallv_mesh1d.h"

namespace Hccl {

/*
 当前只考虑mc2场景，且
 由于：
 1. 预留扩展为64P双die通过unions全互联的算法（需要解决通道保序的RC模式不支持unions的两个端口做bonding的问题）；
 2. Fast model当前不支持双die均连接每个对端，只有1D Fullmesh结构支持连接每个对端；
 故：
 拆分为两个context实现，每个context搬运一半数据；
 可实现：
 1. 两个context可以放在一个die或两个die上执行；
 2. 可以通过调整sendSize & sendOffset实现单context搬运全部数据，等同于常规典型1D Fullmesh算法；
 3. 通信域规模最大支持到64P（需要框架提供相应的64P连接）；
 */
/*
 * 此外，考虑到通道保序方案尚不明确，故后同步采用源端保序方式，当前最大支持8P通信域；
 */

class CcuContextHalfAllToAllVMesh1D : public CcuContextAlgBase {
public:
    CcuContextHalfAllToAllVMesh1D(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group);
    ~CcuContextHalfAllToAllVMesh1D() override {}

    void Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) override;

private:
    void ExchangeCtxResource();
    void LoadArgs();
    void LoadArgsFromMem();
    void MissionSync(uint32_t signalIndex);
    void PostSync();
    void CreateLocalCopyLoop();
    void LocalCopyByLoopGroup(CcuRep::Memory dst, CcuRep::Memory src, GroupOpSize& goPara);

    std::string ctxName_;
    uint32_t missionId_{
        0}; // 分别对应两个context，指示搬运前一半或后一半数据，两个mission可以放在同一个或不同的die上执行
    uint32_t signalNum_{0}; // 需要使用的signal数量
    uint64_t myCclBufferAddr_{0};

    std::vector<CcuRep::Variable> token_;

    // mc2的传入参数
    CcuRep::Variable userInAddr_;   // mc2传入的输入地址
    CcuRep::Variable sendSizeAddr_; // 切分后的分片大小，默认前rankSize个是sizeA，后rankSize个是sizeB
    CcuRep::Variable sendOffsetAddr_; // 切分后的分片相对userInAddr的偏移
    CcuRep::Variable recvOffset_; // 本端给任意对端写时相对对端cclBuffer首地址的偏移，status*winSize/2+rankId*GRID_SIZE
    GroupOpSize goSize_; // mc2传入的给自己的分片大小

    // 本地资源
    CcuRep::Memory curSrc_;
    std::vector<CcuRep::Memory> curDst_;
    std::vector<CcuRep::Variable> sendSizeA_;   // 最多64个连续的，对应数据片的前半部分
    std::vector<CcuRep::Variable> sendSizeB_;   // 最多64个连续的，对应数据片的后半部分
    std::vector<CcuRep::Variable> sendOffsetA_; // 最多64个连续的，对应数据片的前半部分
    std::vector<CcuRep::Variable> sendOffsetB_; // 最多64个连续的，对应数据片的后半部分
    CcuRep::MaskSignal ccuStartSignal_;
    CcuRep::MaskSignal ccuEndSignal_;
    std::vector<CcuRep::MaskSignal> writeDoneSignal_; // 写完成信号

    // 跨mission同步信号
    CcuRep::MaskSignal locMiSignal0_; // 本die的
    CcuRep::MaskSignal locMiSignal1_;
    CcuRep::MaskSignal anoMiSignal0_; // 映射另一个die的
    CcuRep::MaskSignal anoMiSignal1_;
    CcuRep::Variable anoUserInAddr_;
    CcuRep::Variable anoSendSizeAddr_;
    CcuRep::Variable anoSendOffsetAddr_;
    CcuRep::Variable anoRecvOffset_;
};
} // namespace Hccl

#endif // HCCLV2_CCU_CONTEXT_HALF_ALL_TO_ALL_V_MESH_1D_H_
