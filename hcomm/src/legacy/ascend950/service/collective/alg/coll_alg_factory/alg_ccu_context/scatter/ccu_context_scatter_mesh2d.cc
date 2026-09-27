/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_context_scatter_mesh2d.h"
#include "ccu_instruction_scatter_mesh2d.h"

namespace Hccl {

constexpr int VAR_IDX_0 = 0; // transport远端变量，一个transport当前最多只能有3个Var
constexpr int VAR_IDX_1 = 1;
constexpr int VAR_IDX_2 = 2;
constexpr int CKE_IDX_0 = 0; // 专门给后同步使用
constexpr int CKE_IDX_1 = 1; // 前同步使用
constexpr int CKE_IDX_2 = 2; // 前同步使用
constexpr int CKE_IDX_3 = 3; // 前同步使用
constexpr int CKE_AXIS = 4;  // 给die间同步使用
constexpr int DIM_NUM = 2;
constexpr int ZERO = 0;
constexpr int DIM_X = 0;
constexpr int DIM_Y = 1;

CcuContextScatterMesh2D::CcuContextScatterMesh2D(
    const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group)
    : CcuContextAlgBase(arg, transports, group)
{
    const CcuCtxArgScatterMesh2D* ctxArg = dynamic_cast<const CcuCtxArgScatterMesh2D*>(&arg);
    if (ctxArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextScatterMesh2D::ctxArg ptr is null"));
    }
    rankId_ = ctxArg->rankId_;
    dimSize_ = ctxArg->dimSize_; // vector, dimSize_[0]表示X轴的rank数，dimSize_[1]表示Y轴的rank数
    axisId_ = ctxArg->axisId_;   // 由外部传入，指明当前在X轴或者Y轴的CCU上
    rankSize_ = ctxArg->rankSize_;
    root_ = ctxArg->root_;
    // 参数校验
    if (transports.size() == 0) {
        THROW<NullPtrException>(StringFormat("CcuContextScatterMesh2D transports is empty"));
    }
    if (dimSize_.size() != DIM_NUM || dimSize_[0] == ZERO || dimSize_[1] == ZERO || rankSize_ == ZERO
        || axisId_ >= DIM_NUM) {
        THROW<NullPtrException>(StringFormat("[CcuContextScatterMesh2D]ctxArg params is invalid"));
    }

    // 分解出当前Rank的行列坐标
    dimId_.emplace_back(rankId_ % dimSize_[0]); // dimId_[0]表示在X轴1Dmesh拓扑中的localId
    dimId_.emplace_back(rankId_ / dimSize_[0]); // dimId_[1]表示在Y轴1Dmesh拓扑中的localId

    // 分解出Root的行列坐标
    rootDimId_.emplace_back(root_ % dimSize_[0]);
    rootDimId_.emplace_back(root_ / dimSize_[0]);

    localId_ = dimId_[axisId_];
    localSize_ = dimSize_[axisId_];

    localAxisSignal_ = CreateMaskSignal();

    localAxisSignalName_ = "CcuContextScatter2DAxisSync_" + std::to_string(axisId_);
    anotherAxisSignalName_ = "CcuContextScatter2DAxisSync_" + std::to_string(1 - axisId_);

    HCCL_INFO(
        "[ContextScatter2DMesh.init] rankId_[%llu], dimSize_[0][%llu], dimSize_[1][%llu], axisId_[%llu], "
        "root_[%llu], localId_[%llu], localSize_[%llu] ",
        rankId_, dimSize_[0], dimSize_[1], axisId_, root_, localId_, localSize_);
}

bool CcuContextScatterMesh2D::SameRowWithRoot()
{
    bool directConnected = false;
    if (dimId_[DIM_Y] == rootDimId_[DIM_Y]) {
        directConnected = true;
    }
    return directConnected;
}

bool CcuContextScatterMesh2D::SameColumnWithRoot()
{
    bool directConnected = false;
    if (dimId_[DIM_X] == rootDimId_[DIM_X]) {
        directConnected = true;
    }
    return directConnected;
}

void CcuContextScatterMesh2D::PrepareVariables()
{
    u32 transportId = 0;
    CHK_PRT_RET(
        transports.size() < localSize_,
        HCCL_ERROR("[CcuContextScatterMesh2D] transports size is less than localSize"), );
    input_ = CreateVariable();
    sliceSize_ = CreateVariable();
    stride_ = CreateVariable();
    for (u64 id = 0; id < localSize_; id++) {
        if (id == localId_) {
            scratch_.push_back(CreateVariable());
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else { // 非本地，使用远端Variable
            CHK_PRT_RET(
                transports[transportId] == nullptr,
                HCCL_ERROR("[CcuContextScatterMesh2D] Algorithm transport ptr is null"), );
            scratch_.push_back(CreateVariable((*transports[transportId]), VAR_IDX_0));
            output_.push_back(CreateVariable((*transports[transportId]), VAR_IDX_1));
            token_.push_back(CreateVariable((*transports[transportId]), VAR_IDX_2));
            transportId++;
        }
    }
    axisSliceSize_.push_back(CreateVariable());
    axisSliceSize_.push_back(CreateVariable());

    ExportMaskSignal(localAxisSignal_, localAxisSignalName_);      // 将本地的信号export出去
    anotherAxisSignal_ = ImportMaskSignal(anotherAxisSignalName_); // 导入另一个die的mask信号
    curGoSize_ = CreateGroupOpSize();
    return;
}

void CcuContextScatterMesh2D::LoadArgs()
{
    // 模板中的可变入参
    // 地址相关参数：input_,output_,scratch_, token_
    // 数据相关参数：sliceSize_, stride_,  axisSliceSize_ (axisSliceSize[DIM_X]为slice中通过x轴先传输的部分)
    // 顺序：inputAddr, outputAddr, scratchAddr, tokenInfo, sliceSize, stride, xSliceSize, ySliceSize
    Load(input_);
    Load(output_[localId_]);
    Load(token_[localId_]);
    Load(scratch_[localId_]);
    Load(sliceSize_);
    Load(stride_);
    Load(axisSliceSize_[DIM_X]);
    Load(axisSliceSize_[DIM_Y]);
    Load(curGoSize_);
    return;
}

void CcuContextScatterMesh2D::PreSync()
{
    uint16_t selfBit = 1 << localId_; // 本rank的mask
    uint16_t allBit = ((1 << localSize_) - 1) & (~(1 << localId_));

    for (auto t : transports) {
        if (t == nullptr) {
            THROW<NullPtrException>(StringFormat("CcuContextScatterMesh2D::PreSync, transport ptr is null"));
        }
        WriteVariableWithSignal(
            *t, scratch_[localId_], VAR_IDX_0, CKE_IDX_1,
            selfBit); // 传递CCLBuf信息, 把自己的CCLbuf给所有对端
        WriteVariableWithSignal(*t, output_[localId_], VAR_IDX_1, CKE_IDX_2, selfBit); // 传递output信息
        WriteVariableWithSignal(*t, token_[localId_], VAR_IDX_2, CKE_IDX_3, selfBit);  // 传递token信息
    }

    GroupWait(*transportGroup, CKE_IDX_1, allBit); // 等齐所有对端的信息
    GroupWait(*transportGroup, CKE_IDX_2, allBit);
    GroupWait(*transportGroup, CKE_IDX_3, allBit);
    return;
}

void CcuContextScatterMesh2D::Sync(uint32_t ckeId)
{
    uint16_t selfBit = 1 << localId_; // 本rank的mask
    uint16_t allBit = ((1 << localSize_) - 1) & (~(1 << localId_));

    for (auto t : transports) {
        if (t == nullptr) {
            THROW<NullPtrException>(StringFormat("CcuContextScatterMesh2D::PostSync, transport ptr is null"));
        }
        RemotePost(*t, ckeId, selfBit);
    }

    GroupWait(*transportGroup, ckeId, allBit);
    return;
}

void CcuContextScatterMesh2D::AxisSync(uint32_t signalIndex)
{
    if (signalIndex > 1) {
        THROW<InvalidParamsException>(
            StringFormat("[CcuContextScatterMesh2D] Unexpected SignalIndex[%u]", signalIndex));
    }
    LocalCtxPost(anotherAxisSignal_, 1 << (axisId_ + signalIndex * DIM_NUM));
    LocalWait(localAxisSignal_, 1 << (1 - axisId_ + signalIndex * DIM_NUM));
    return;
}

// 每次调用，准备好1D范围内，所有需要传递的src和dst，size固定为（localSize-1）
void CcuContextScatterMesh2D::CcuWrite1DMesh(
    std::vector<CcuRep::Memory>& src, std::vector<CcuRep::Memory>& dst, CcuRep::Variable& size)
{
    CcuRep::MaskSignal locMask = CreateMaskSignal();
    uint16_t allBitWithoutLocal = ((1 << localSize_) - 1) & (~(1 << src.size()));
    uint64_t transportId = 0;
    for (uint16_t r = 0; r < src.size(); r++) {
        CCU_IF(size == 0) { LocalPost(locMask, 1 << r); }
        CCU_IF(size != 0) { Write(*transports[transportId], dst[r], src[r], size, locMask, 1 << r); }
        transportId++;
    }
    // 等写完所有对端
    LocalWait(locMask, allBitWithoutLocal);
    return;
}

uint64_t CcuContextScatterMesh2D::CoordinateToGlobalId(uint32_t x, uint32_t y)
{
    uint64_t id = 0;
    if (axisId_ == DIM_X) {
        id = x + y * dimSize_[0];
    } else {
        id = y + x * dimSize_[0];
    }
    return id;
}

// a = a + i * b
void CcuContextScatterMesh2D::CcuMultiply(CcuRep::Memory& a, CcuRep::Variable& b, uint64_t i) const
{
    for (uint64_t j = 0; j < i; j++) {
        a.addr += b;
    }
    return;
}

void CcuContextScatterMesh2D::RelaySendFor1D(
    std::vector<CcuRep::Memory>& relaySrc, std::vector<CcuRep::Memory>& relayDst, uint64_t j)
{
    uint64_t globalId = 0;
    relaySrc.clear();
    relayDst.clear();
    for (uint64_t i = 0; i < dimSize_[axisId_]; i++) {
        if (i != dimId_[axisId_]) {
            // 准备中转数据
            CcuRep::Memory src = CreateMemory();
            CcuRep::Memory dst = CreateMemory();
            src.token = token_[i];
            dst.token = token_[i];
            // i,j的顺序为先本轴(axisId), 再另外一个轴(anotherAxisId)
            globalId = CoordinateToGlobalId(i, j);
            HCCL_INFO(
                "[CcuContextScatterMesh2D][PrepareRootSendInfo] src globalId[%llu], curRank[%u], j[%llu], i[%llu] ",
                globalId, rankId_, j, i);
            src.addr = input_;
            CcuMultiply(src, stride_, globalId); // src偏移：src +=  stride_ * globalId
            dst.addr = scratch_[i];
            CcuMultiply(
                dst, sliceSize_,
                globalId); // dst偏移, dst为scratch，没有stride相关，dst += sliceSize_ * globalId
            if (axisId_ == DIM_Y) {
                src.addr += axisSliceSize_[DIM_X];
                dst.addr += axisSliceSize_[DIM_X];
            }
            relaySrc.emplace_back(src);
            relayDst.emplace_back(dst);
        }
    }
    HCCL_INFO(
        "[CcuContextScatterMesh2D][PrepareRootSendInfo] relaySrcSize[%zu], relayDstSize_[%zu] ", relaySrc.size(),
        relayDst.size());
    CcuWrite1DMesh(relaySrc, relayDst, axisSliceSize_[axisId_]);
    return;
}

// ********************************************
// 准备Root需要发送的“中转”地址,并发送
// relaySrc: “中转”数据，从Root的inputAddr发送到同轴其他卡的scratchAddr
// relayDst: 对端的scratchAddr
// ********************************************
void CcuContextScatterMesh2D::PrepareAndTransferRootRelayInfo(
    std::vector<CcuRep::Memory>& relaySrc, std::vector<CcuRep::Memory>& relayDst)
{
    HCCL_INFO("[CcuContextScatterMesh2D][PrepareRootSendInfo] start axisId_[%llu]", axisId_);
    // 准备中转数据： i 为所有对端，负责中转数据，所有数据均发至i上； j为中转目的地
    HCCL_INFO(
        "[CcuContextScatterMesh2D][PrepareRootSendInfo] axisId[%llu], dimSize_[%llu] ", axisId_, dimSize_[1 - axisId_]);
    for (uint64_t j = 0; j < dimSize_[1 - axisId_]; j++) {
        if (j != dimId_[1 - axisId_]) {
            RelaySendFor1D(relaySrc, relayDst, j);
        }
    }
    HCCL_INFO("[CcuContextScatterMesh2D][PrepareRootSendInfo] Done axisId_[%llu]", axisId_);
    return;
}

// directSrc：“直达”数据，从root的inputAddr发送到同轴其他卡的outputAddr
// directDst; 对端的outputAddr
void CcuContextScatterMesh2D::PrepareAndTransferRootDirectInfo(
    std::vector<CcuRep::Memory>& directSrc, std::vector<CcuRep::Memory>& directDst)
{
    uint64_t globalId = 0;
    // 准备直达数据,直达数据只在自己的axisId上做1D的发送
    // i为要发送直达数据的对端, 不包含本地
    directSrc.clear();
    directDst.clear();
    for (uint64_t i = 0; i < dimSize_[axisId_]; i++) {
        if (i != dimId_[axisId_]) {
            CcuRep::Memory src = CreateMemory();
            CcuRep::Memory dst = CreateMemory();
            src.token = token_[i];
            dst.token = token_[i];
            globalId = CoordinateToGlobalId(i, dimId_[1 - axisId_]);
            HCCL_INFO(
                "[CcuContextScatterMesh2D][PrepareRootSendInfo] src globalId[%u], curRank[%u]", globalId, rankId_);
            src.addr = input_;
            CcuMultiply(src, stride_, globalId);
            dst.addr = output_[i]; // output只有一片，不需要偏移
            directSrc.emplace_back(src);
            directDst.emplace_back(dst);
        }
    }
    CcuWrite1DMesh(directSrc, directDst, sliceSize_);
    return;
}

void CcuContextScatterMesh2D::LocalTransfer()
{
    CcuRep::MaskSignal locMask = CreateMaskSignal();
    CcuRep::Memory src = CreateMemory();
    CcuRep::Memory dst = CreateMemory();
    src.token = token_[localId_];
    dst.token = token_[localId_];
    src.addr = input_;
    if (axisId_ == DIM_Y) {
        src.addr += axisSliceSize_[DIM_X];
    }

    CcuMultiply(src, stride_, root_);
    dst.addr = output_[localId_];

    if (axisId_ == DIM_Y) {
        dst.addr += axisSliceSize_[DIM_X];
    }
    HCCL_DEBUG("[CcuContextScatterMesh2D] use GroupCopy");
    GroupCopy(dst, src, curGoSize_);

    return;
}

// 准备转发节点，需要的转发数据地址
void CcuContextScatterMesh2D::RelaySend(std::vector<CcuRep::Memory>& relaySrc, std::vector<CcuRep::Memory>& relayDst)
{
    uint64_t globalId;
    relaySrc.clear();
    relayDst.clear();
    for (uint64_t i = 0; i < dimSize_[axisId_]; i++) {
        if (i != dimId_[axisId_]) {
            CcuRep::Memory src = CreateMemory();
            CcuRep::Memory dst = CreateMemory();
            src.token = token_[i];
            dst.token = token_[i];
            globalId = CoordinateToGlobalId(i, dimId_[1 - axisId_]);
            HCCL_INFO(
                "[CcuContextScatterMesh2D][PrepareRelaySendInfo] src globalId[%llu], curRank[%u], axisId[%llu], "
                "i:[%llu], localId[%llu]",
                globalId, rankId_, axisId_, i, localId_);
            src.addr = scratch_[localId_];
            CcuMultiply(src, sliceSize_, globalId);
            dst.addr = output_[i]; // output只有一片，不需要偏移
            if (axisId_ == DIM_X) {
                src.addr += axisSliceSize_[DIM_X];
                dst.addr += axisSliceSize_[DIM_X];
            }
            relaySrc.emplace_back(src);
            relayDst.emplace_back(dst);
        }
    }
    CcuWrite1DMesh(relaySrc, relayDst, axisSliceSize_[1 - axisId_]);
    return;
}

// *************************************************
// root的行为模式说明： X轴和Y轴行为一致；都是给1DMesh的其他卡发数据, 任一轴的行为：
// 1) 前同步
// 2）给与本轴mesh直连卡发“中转”数据
// 3）后同步
// 4）轴同步
// 5）前同步
// 6）给与本轴mesh直连卡发“直达”数据
// 7）后同步
// 8）轴同步，与4)中成对使用，保证正确性
// *************************************************
void CcuContextScatterMesh2D::RootSendAlgorithm()
{
    HCCL_INFO("[CcuContextScatterMesh2D][RootSendAlgorithm] Start");
    PrepareVariables();
    LoadArgs();
    // step1
    PreSync();

    // 准备"直达"&“中转”传输地址
    std::vector<CcuRep::Memory> directSrc;
    std::vector<CcuRep::Memory> directDst;
    std::vector<CcuRep::Memory> relaySrc;
    std::vector<CcuRep::Memory> relayDst;
    PrepareAndTransferRootRelayInfo(relaySrc, relayDst);

    Sync(CKE_IDX_0); // 后同步
    AxisSync(0);
    // step2
    Sync(CKE_IDX_1); // 前同步的功能

    PrepareAndTransferRootDirectInfo(directSrc, directDst);
    LocalTransfer();

    Sync(CKE_IDX_0); // 后同步
    AxisSync(1);     // 轴同步
    HCCL_INFO("[CcuContextScatterMesh2D][RootSendAlgorithm] Step2 AxisSync Done");
    return;
}

// *****************************************
// 与Root同行或同列的rank
// step1： 与Root同行的只有ccuX有，与Root同列的只有ccuY有；（目前先搞所有卡都有）
// 1) 前同步
// 2）后同步
// step2：收直达数据(同行的ccuX有，同列的ccuY有)；发step1收到的中转数据（同行的ccuY有，同列的ccuX有）
// 3）轴同步
// 4）前同步
// 5）发中转数据
// 6）后同步
// 7）轴同步
// *****************************************
void CcuContextScatterMesh2D::RelaySendAlgorithm()
{
    HCCL_INFO("[CcuContextScatterMesh2D][RelaySendAlgorithm] Start");
    PrepareVariables();
    LoadArgs();

    // step1:
    PreSync();       // 前同步
    Sync(CKE_IDX_0); // 后同步
    AxisSync(0);

    // step2:
    // 与Root同行的ccuX，或者 与Root同列的ccuY；才有step2的收直达数据
    if ((SameRowWithRoot() && axisId_ == DIM_X) or (SameColumnWithRoot() && axisId_ == DIM_Y)) {
        HCCL_INFO(
            "[CcuContextScatterMesh2D][RelaySendAlgorithm][1 actual Relay] into Relay action, axisId[%llu], "
            "isSameRowWithRoot[%d], isSameColwithRoot[%d], myRank[%llu], root[%llu]",
            axisId_, SameRowWithRoot(), SameColumnWithRoot(), rankId_, root_);
        Sync(CKE_IDX_1); // 前同步
        Sync(CKE_IDX_0); // 后同步
    }
    // 与Root同行的ccuY，或者 与Root同列的ccuX；才有step2的发中转数据
    if ((SameRowWithRoot() && axisId_ == DIM_Y) or (SameColumnWithRoot() && axisId_ == DIM_X)) {
        HCCL_INFO(
            "[CcuContextScatterMesh2D][RelaySendAlgorithm][2 actual Relay] into Relay action, axisId[%llu], "
            "isSameRowWithRoot[%d], isSameColwithRoot[%d], myRank[%llu], root[%llu]",
            axisId_, SameRowWithRoot(), SameColumnWithRoot(), rankId_, root_);
        Sync(CKE_IDX_1); // 前同步

        // 准备"直达"&“中转”传输地址
        std::vector<CcuRep::Memory> relaySrc;
        std::vector<CcuRep::Memory> relayDst;
        RelaySend(relaySrc, relayDst);

        Sync(CKE_IDX_0); // 后同步
    }
    AxisSync(1);
    HCCL_INFO("[CcuContextScatterMesh2D][RelaySendAlgorithm] step2 Done");
    return;
}

void CcuContextScatterMesh2D::NonDirectRecvAlgorithm()
{
    HCCL_INFO(
        "[CcuContextScatterMesh2D][NonDirectRecvAlgorithm] start, dimIdX_[%llu], dimIdY_[%llu]", dimId_[0], dimId_[1]);
    PrepareVariables();
    LoadArgs();
    PreSync();       // 前同步
    Sync(CKE_IDX_0); // 后同步
    AxisSync(0);
    Sync(CKE_IDX_1); // 前同步
    Sync(CKE_IDX_0); // 后同步
    AxisSync(1);
    HCCL_INFO(
        "[CcuContextScatterMesh2D][NonDirectRecvAlgorithm] Done, dimIdX_[%llu], dimIdY_[%llu]", dimId_[0], dimId_[1]);
    return;
}

void CcuContextScatterMesh2D::Algorithm()
{
    HCCL_INFO("[ccuScatterMesh2D_context] ScatterMesh2D run");
    // 分3种角色讨论，1）root； 2）与root同行同列的； 3）非直连的
    if (rankId_ == root_) {
        // root节点，X轴与Y轴的行为一致
        RootSendAlgorithm();
        return;
    } else if (SameRowWithRoot() or SameColumnWithRoot()) {
        RelaySendAlgorithm();
        return;
    } else {
        // 非直连
        NonDirectRecvAlgorithm();
        return;
    }
    HCCL_INFO("[ccuScatterMesh2D_context] ScatterMesh2D end");
    return;
}

std::vector<uint64_t> CcuContextScatterMesh2D::GeneArgs(const CcuTaskArg& arg)
{
    const CcuTaskArgScatterMesh2D* taskArg = dynamic_cast<const CcuTaskArgScatterMesh2D*>(&arg);
    if (taskArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextScatterMesh2D::taskArg ptr is null"));
    }

    uint64_t inputAddr = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t tokenInfo = taskArg->token_;
    uint64_t scratchAddr = taskArg->scratchAddr_;

    uint64_t sliceSize = taskArg->sliceSize_;
    uint64_t stride = taskArg->stride_;
    uint64_t xSliceSize = taskArg->xSliceSize_;
    uint64_t ySliceSize = taskArg->ySliceSize_;

    auto xSliceGoSize = CalGoSize(xSliceSize);
    auto ySliceGoSize = CalGoSize(ySliceSize);
    auto curGosize = (axisId_ == DIM_X) ? xSliceGoSize : ySliceGoSize;

    HCCL_INFO(
        "[CcuContextScatterMesh2DAlgo] inputAddr[%llu], outputAddr[%llu], scratchAddr[%llu], sliceSize[%llu], "
        "stride[%llu], xSliceSize[%llu], ySliceSize[%llu] ",
        inputAddr, outputAddr, scratchAddr, sliceSize, stride, xSliceSize, ySliceSize);
    // 8个参数
    return {inputAddr,  outputAddr, tokenInfo,    scratchAddr,  sliceSize,    stride,
            xSliceSize, ySliceSize, curGosize[0], curGosize[1], curGosize[2], curGosize[3]};
}
} // namespace Hccl
