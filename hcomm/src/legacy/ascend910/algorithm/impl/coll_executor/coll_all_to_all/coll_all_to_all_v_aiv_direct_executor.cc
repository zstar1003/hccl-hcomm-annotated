/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_to_all_v_aiv_direct_executor.h"

namespace hccl {

CollRunAlltoAllAivDirect::CollRunAlltoAllAivDirect(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAlltoAllExecutor(dispatcher, topoMatcher)
{}

// 超节点内建SDMA链路；超节点间建RDMA链路
HcclResult CollRunAlltoAllAivDirect::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commCombinePara(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commCombinePara, opTransport[COMM_LEVEL0], inputType, outputType));

    LevelNSubCommTransport& commTransportLevel0 = opTransport[COMM_LEVEL0];
    for (u32 subCommIndex = 0; subCommIndex < commTransportLevel0.size(); subCommIndex++) {
        for (auto& transportRequest : commTransportLevel0[subCommIndex].transportRequests) {
            transportRequest.isUsedRdma = topoAttr_.isUsedRdmaMap.at(transportRequest.remoteUserRank);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllAivDirect::CalcLevel1CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commCombinePara(COMM_MESH_L1, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commCombinePara, opTransport[COMM_MESH_L1], inputType, outputType));

    LevelNSubCommTransport& commTransportLevel0 = opTransport[COMM_MESH_L1];
    for (u32 subCommIndex = 0; subCommIndex < commTransportLevel0.size(); subCommIndex++) {
        for (auto& transportRequest : commTransportLevel0[subCommIndex].transportRequests) {
            transportRequest.isUsedRdma = true;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult
CollRunAlltoAllAivDirect::CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType) const
{
    inputType = TransportMemType::CCL_INPUT;
    outputType = TransportMemType::CCL_OUTPUT;

    HCCL_INFO(
        "[CollRunAlltoAllAivDirect][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]", tag_.c_str(),
        inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllAivDirect::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;

    CHK_RET(CalcTransportMemType(inputType, outputType));
    // level0 hccs 全连接通信域
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    // level1 roce deviceid通信域
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));

    return HCCL_SUCCESS;
}

REGISTER_EXEC("RunAlltoAllAivDirect", AlltoAllVAivDirect, CollRunAlltoAllAivDirect);
} // namespace hccl
