/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_scatter_comm_executor.h"

namespace hccl {
CollScatterCommExecutor::CollScatterCommExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollScatterExecutor(dispatcher, topoMatcher)
{}

HcclResult CollScatterCommExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollScatterCommExecutor] scatter starts.");
    DeviceMem& inputMem = execMem.inputMem;
    DeviceMem& outputMem = execMem.outputMem;
    u64 count = execMem.count;
    auto root = param.root;
    auto dataType = param.DataDes.dataType;
    Stream& stream = const_cast<Stream&>(param.stream);
    u32 userRank = topoAttr_.userRank;

    u32 commIndex = COMM_INDEX_0;
    // 统一走server间
    CommPlane commPlane = COMM_COMBINE;
    if (topoAttr_.deviceType == DevType::DEV_TYPE_910_93) {
        commPlane = COMM_COMBINE_ORDER;
    }

    CHK_RET(CheckCommSize(commPlane, COMM_INDEX_0 + 1));
    SubCommInfo combinedCommInfo = GetSubCommInfo(commPlane, COMM_INDEX_0);

    CHK_RET(KernelRunLevel1(inputMem, count, dataType, commIndex, root, userRank, commPlane, stream));

    // 将scratchMem赋值给outputMem
    u8* inputMemPtr = static_cast<u8*>(inputMem.ptr());
    CHK_PTR_NULL(inputMemPtr);
    DeviceMem resultMem(inputMemPtr + outputMem.size() * combinedCommInfo.localRank, outputMem.size());
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outputMem, resultMem, stream));
    return HCCL_SUCCESS;
}

HcclResult CollScatterCommExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcCombinedCommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollScatterCommExecutor::CalcCombinedCommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommPlane commPlane = COMM_COMBINE;
    if (topoAttr_.deviceType == DevType::DEV_TYPE_910_93) {
        commPlane = COMM_COMBINE_ORDER;
    }

    CommParaInfo commParaInfo(commPlane, CommType::COMM_TAG_MAX);
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
        commParaInfo.commType = CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING;
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
        commParaInfo.commType = CommType::COMM_TAG_NONUNIFORM_BRUCK;
    } else {
        commParaInfo.commType = CommType::COMM_TAG_RING_INNER;
    }
    CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[commPlane], inputType, outputType));
    return HCCL_SUCCESS;
}

REGISTER_EXEC("ScatterCommExecutor", ScatterComm, CollScatterCommExecutor);

} // namespace hccl
