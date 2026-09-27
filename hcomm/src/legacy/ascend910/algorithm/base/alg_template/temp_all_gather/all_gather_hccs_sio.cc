/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_hccs_sio.h"
#include "alg_template_register.h"

namespace hccl {
AllGatherHccsSio::AllGatherHccsSio(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher) {}

AllGatherHccsSio::~AllGatherHccsSio() {}

HcclResult AllGatherHccsSio::Prepare(
    SubCommInfo& outerCommInfoHccs, SubCommInfo& outerCommInfoSio, DeviceMem& usrInMem, DeviceMem& usrOutMem, u64 count,
    const HcclDataType dataType, const Stream& mainStream, std::vector<Stream>& meshStreams,
    std::vector<std::shared_ptr<LocalNotify>>& meshSignal, std::vector<std::shared_ptr<LocalNotify>>& meshSignalAux,
    u32 userRank, HcomCollOpInfo* opInfo)
{
    inputMem_ = usrInMem;
    outputMem_ = usrOutMem;
    stream_ = mainStream;
    meshStreams_ = meshStreams;
    meshSignal_ = meshSignal;
    meshSignalAux_ = meshSignalAux;
    userRank_ = userRank;
    dataType_ = dataType;
    dataBytes_ = count * SIZE_TABLE[dataType];
    count_ = count;
    outerCommInfoHccs_ = outerCommInfoHccs;
    outerCommInfoSio_ = outerCommInfoSio;
    opInfo_ = opInfo;
    totalDataBytes_ = opInfo->count * SIZE_TABLE[dataType_];
    return HCCL_SUCCESS;
}

// 主流所有从流
HcclResult AllGatherHccsSio::NotifySubStreamStart()
{
    for (u32 streamIndex = 0; streamIndex < meshStreams_.size(); streamIndex++) {
        CHK_RET(LocalNotify::Post(stream_, dispatcher_, meshSignalAux_[streamIndex], INVALID_VALUE_STAGE));
        CHK_RET(LocalNotify::Wait(
            meshStreams_[streamIndex], dispatcher_, meshSignalAux_[streamIndex], INVALID_VALUE_STAGE));
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherHccsSio::WaitSubStreamFinish()
{
    for (u32 streamIndex = 0; streamIndex < meshStreams_.size(); streamIndex++) {
        CHK_RET(
            LocalNotify::Post(meshStreams_[streamIndex], dispatcher_, meshSignal_[streamIndex], INVALID_VALUE_STAGE));
        CHK_RET(LocalNotify::Wait(stream_, dispatcher_, meshSignal_[streamIndex], INVALID_VALUE_STAGE));
    }
    return HCCL_SUCCESS;
}

HcclResult
AllGatherHccsSio::RunInterDie(const u32 dieRankId, const std::vector<LINK>& links, const u32 srcDMAMemSliceId)
{
    // 检查链接是否为空
    CHK_SMART_PTR_NULL(links[dieRankId]);

    // 获取远程内存指针
    void* remDMAMemPtr = nullptr;
    CHK_RET(links[dieRankId]->GetRemoteMem(UserMemType::INPUT_MEM, &remDMAMemPtr));

    // 确定需要传输的数据部分（上半部分或下半部分）
    u64 dataPartOffset = dieRankId * dataBytes_;
    u64 dataPartSize = count_ / 2 * SIZE_TABLE[dataType_];

    DeviceMem locDieDst;
    DeviceMem srcDieMem;

    // 定义本地目标内存和远程源内存
    if (srcDMAMemSliceId == 0) {
        locDieDst = dmaMem_[1].range(dataPartOffset, dataPartSize);
        srcDieMem = DeviceMem::create(static_cast<u8*>(remDMAMemPtr), dataPartSize);
    } else {
        locDieDst = dmaMem_[1].range(dataPartOffset + dataPartSize, dataBytes_ - dataPartSize);
        srcDieMem = DeviceMem::create(static_cast<u8*>(remDMAMemPtr) + dataPartSize, dataBytes_ - dataPartSize);
    }

    HCCL_INFO(
        "RunInterDie: dieRankId[%d], locDieDst ptr[%p], locDieDst size[%ld], remDMAMemPtr[%p]", dieRankId,
        locDieDst.ptr(), locDieDst.size(), remDMAMemPtr);
    // 执行异步内存复制
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, locDieDst, srcDieMem, meshStreams_[srcDMAMemSliceId]));

    return HCCL_SUCCESS;
}

HcclResult
AllGatherHccsSio::RunInterDieOpBase(const u32 dieRankId, const std::vector<LINK>& links, const u32 srcDMAMemSliceId)
{
    // 检查链接是否为空
    CHK_SMART_PTR_NULL(links[dieRankId]);

    // 获取远程CCLin内存指针
    void* remCCLMemPtr = nullptr;
    CHK_RET(links[dieRankId]->GetRemoteMem(UserMemType::INPUT_MEM, &remCCLMemPtr));

    // 确定需要传输的数据部分（上半部分或下半部分）
    u64 dataPartSize = count_ / 2 * SIZE_TABLE[dataType_];

    DeviceMem locDieDst;
    DeviceMem srcDieMem;
    DeviceMem usroutMem
        = DeviceMem::create(static_cast<u8*>(opInfo_->outputAddr) + totalDataBytes_ * dieRankId, dataBytes_);
    ;

    // 定义本地目标内存和远程源内存
    if (srcDMAMemSliceId == 0) {
        locDieDst = usroutMem.range(0, dataPartSize);
        srcDieMem = DeviceMem::create(static_cast<u8*>(remCCLMemPtr), dataPartSize);
    } else {
        locDieDst = usroutMem.range(dataPartSize, dataBytes_ - dataPartSize);
        srcDieMem = DeviceMem::create(static_cast<u8*>(remCCLMemPtr) + dataPartSize, dataBytes_ - dataPartSize);
    }
    u32 linkType = static_cast<u32>(links[dieRankId]->GetLinkType());
    HCCL_DEBUG("[AllGatherHccsSio][RunInterDieOpbase] dstRankId[%u], linkType[%u]", dieRankId, linkType);
    HCCL_INFO(
        "RunInterDieOpbase: dieRankId[%d], locDieDst ptr[%p], locDieDst size[%ld], remCCLMemPtr[%p]", dieRankId,
        locDieDst.ptr(), locDieDst.size(), remCCLMemPtr);
    // 执行异步内存复制
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, locDieDst, srcDieMem, meshStreams_[srcDMAMemSliceId]));

    return HCCL_SUCCESS;
}

// allgather的入口函数
HcclResult
AllGatherHccsSio::RunAsync(const u32 rank, const u32 rankSize, [[maybe_unused]] const std::vector<LINK>& links)
{
    /*rank0:
    从userin拷贝到userout上半部分
    userout下半部分的上半部分通过sio读取rank1的userin上半部分
    userout下半部分的下半部分通过hccs读取rank1的userin下半部分
    */

    /*rank1:
    从userin拷贝到userout下半部分
    userout上半部分的上半部分通过sio读取rank0的userin上半部分
    userout上半部分的下半部分通过hccs读取rank0的userin下半部分
    */
    intraRankSize_ = rankSize;
    u32 dieRankId = (rank + 1) % rankSize;
    // 数据切分为2
    static u32 HCCL_ALLGATHER_SPLIT_FACTOR = 2;

    // dmaMem0部分userin，dmaMem1部分userout
    DeviceMem dmaMem0 = DeviceMem::create(inputMem_.ptr(), dataBytes_);
    DeviceMem dmaMem1 = DeviceMem::create(outputMem_.ptr(), dataBytes_ * intraRankSize_);
    DeviceMem locDieDst = dmaMem1.range(dataBytes_ * rank, dataBytes_);

    HCCL_INFO(
        "RunAsync: dmaMem0 ptr[%p], dmaMem0 size[%ld]; dmaMem1 ptr[%p], dmaMem1 size[%ld]; locDieDst ptr[%p], "
        "locDieDst size[%ld]",
        inputMem_.ptr(), dataBytes_, outputMem_.ptr(), dataBytes_ * intraRankSize_, locDieDst.ptr(), dataBytes_);

    dmaMem_.push_back(dmaMem0); // userin
    dmaMem_.push_back(dmaMem1); // userout

    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        // usrin 到 cclin
        DeviceMem locDieUsrin = DeviceMem::create(static_cast<u8*>(opInfo_->inputAddr), dataBytes_);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dmaMem0, locDieUsrin, stream_));
    } else {
        // step 0操作 : 所有卡本地数据从userIn-->userout
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, locDieDst, dmaMem0, stream_));
    }

    // 主流启动从流
    CHK_RET(NotifySubStreamStart());

    // step 1 : die间 && device间并行收发

    // 数据搬运及后同步
    u32 srcDMAMemSliceId = 0;

    CHK_RET(outerCommInfoHccs_.links[dieRankId]->TxAck(meshStreams_[srcDMAMemSliceId]));    // AckRecord
    CHK_RET(outerCommInfoHccs_.links[dieRankId]->RxAck(meshStreams_[srcDMAMemSliceId]));    // AckWait
    CHK_RET(outerCommInfoSio_.links[dieRankId]->TxAck(meshStreams_[srcDMAMemSliceId + 1])); // AckRecord
    CHK_RET(outerCommInfoSio_.links[dieRankId]->RxAck(meshStreams_[srcDMAMemSliceId + 1])); // AckWait

    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        // 本地userout 读取die间 cclin by sio
        CHK_RET(RunInterDieOpBase(dieRankId, outerCommInfoHccs_.links, srcDMAMemSliceId));
        notifyIdx_++;

        // 本地userout 读取die间 userin by hccs
        // srcDMAMemSliceId++;
        CHK_RET(RunInterDieOpBase(dieRankId, outerCommInfoSio_.links, srcDMAMemSliceId + 1));

        // 本地usrout读取本地usrin
        DeviceMem locDieSrc = DeviceMem::create(
            static_cast<u8*>(opInfo_->inputAddr), count_ / HCCL_ALLGATHER_SPLIT_FACTOR * SIZE_TABLE[dataType_]);
        locDieDst = DeviceMem::create(
            static_cast<u8*>(opInfo_->outputAddr) + totalDataBytes_ * rank,
            count_ / HCCL_ALLGATHER_SPLIT_FACTOR * SIZE_TABLE[dataType_]);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, locDieDst, locDieSrc, meshStreams_[srcDMAMemSliceId + 2]));

        locDieSrc = DeviceMem::create(
            static_cast<u8*>(opInfo_->inputAddr) + count_ / HCCL_ALLGATHER_SPLIT_FACTOR * SIZE_TABLE[dataType_],
            dataBytes_ - count_ / HCCL_ALLGATHER_SPLIT_FACTOR * SIZE_TABLE[dataType_]);
        locDieDst = DeviceMem::create(
            static_cast<u8*>(opInfo_->outputAddr) + totalDataBytes_ * rank
                + count_ / HCCL_ALLGATHER_SPLIT_FACTOR * SIZE_TABLE[dataType_],
            dataBytes_ - count_ / HCCL_ALLGATHER_SPLIT_FACTOR * SIZE_TABLE[dataType_]);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, locDieDst, locDieSrc, meshStreams_[srcDMAMemSliceId + 3]));
    } else {
        // 本地userout 读取die间 userin by sio
        CHK_RET(RunInterDie(dieRankId, outerCommInfoHccs_.links, srcDMAMemSliceId));
        notifyIdx_++;

        // 本地userout 读取die间 userin by hccs
        // srcDMAMemSliceId++;
        CHK_RET(RunInterDie(dieRankId, outerCommInfoSio_.links, srcDMAMemSliceId + 1));
    }

    CHK_RET(outerCommInfoHccs_.links[dieRankId]->TxDataSignal(meshStreams_[srcDMAMemSliceId]));    // DataRecord
    CHK_RET(outerCommInfoHccs_.links[dieRankId]->RxDataSignal(meshStreams_[srcDMAMemSliceId]));    // Datawait
    CHK_RET(outerCommInfoSio_.links[dieRankId]->TxDataSignal(meshStreams_[srcDMAMemSliceId + 1])); // DataRecord
    CHK_RET(outerCommInfoSio_.links[dieRankId]->RxDataSignal(meshStreams_[srcDMAMemSliceId + 1])); // Datawait

    CHK_RET(WaitSubStreamFinish());

    HCCL_INFO("[AllGatherHccsSio][RunAsync]AllGatherHccsSio finished groupRankId[%u] ", userRank_);
    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_GATHER_HCCS_SIO, AllGatherHccsSio);
} // namespace hccl
