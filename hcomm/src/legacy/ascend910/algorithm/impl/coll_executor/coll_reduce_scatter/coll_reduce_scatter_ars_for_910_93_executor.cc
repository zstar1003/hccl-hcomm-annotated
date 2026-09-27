/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_reduce_scatter_ars_for_910_93_executor.h"

namespace hccl {
CollReduceScatterARSFor91093Executor::CollReduceScatterARSFor91093Executor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollReduceScatterRingFor91093Executor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = false;
}

HcclResult CollReduceScatterARSFor91093Executor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = LEVEL0_PLANE_NUM_IN_NPRING_SINGLE;
    if ((intraRingSize_ > FACTOR_TWO) && topoAttr_.isARSDoubleRing) {
        totalStreamNum = LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE;
    }
    streamNum = totalStreamNum - 1;
    HCCL_INFO("[CalcStreamNum] tag[%s] streamNum[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterARSFor91093Executor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CHK_RET(SetCommInfoForARS(intraRingSize_));
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_RING_INNER);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    CommParaInfo commARSIntra(COMM_LEVEL0_LOGICAL, CommType::COMM_TAG_RING_INNER);
    CHK_RET(CalcCommPlaneInfo(tag_, commARSIntra, opTransport[COMM_LEVEL0_LOGICAL], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterARSFor91093Executor::CalcLevel1CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
        HCCL_DEBUG("[CalcARSInterCommInfo] use ring comm type");
        CommParaInfo commARSInter(COMM_LEVEL1_LOGICAL, CommType::COMM_TAG_RING_INNER);
        CHK_RET(CalcCommPlaneInfo(tag_, commARSInter, opTransport[COMM_LEVEL1_LOGICAL], inputType, outputType));
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
        HCCL_DEBUG("[CalcARSInterCommInfo] use NB comm type");
        CommParaInfo commARSInter(COMM_LEVEL1_LOGICAL, CommType::COMM_TAG_NONUNIFORM_BRUCK);
        CHK_RET(CalcCommPlaneInfo(tag_, commARSInter, opTransport[COMM_LEVEL1_LOGICAL], inputType, outputType));
    } else {
        HCCL_DEBUG("[CalcARSInterCommInfo] use NHR comm type");
        CommParaInfo commARSInter(COMM_LEVEL1_LOGICAL, CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING);
        CHK_RET(CalcCommPlaneInfo(tag_, commARSInter, opTransport[COMM_LEVEL1_LOGICAL], inputType, outputType));
    }
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterARSFor91093Executor::GetLevelCommInfo()
{
    logicalLevel0plane_ = COMM_LEVEL0_LOGICAL;
    CHK_RET(CheckCommSize(logicalLevel0plane_, COMM_INDEX_0 + 1));
    logicalLevel0CommInfo_ = GetSubCommInfo(logicalLevel0plane_, COMM_INDEX_0);
    logicalLevel1plane_ = COMM_LEVEL1_LOGICAL;
    CHK_RET(CheckCommSize(logicalLevel1plane_, logicalLevel0CommInfo_.localRank + 1));
    logicalLevel1CommInfo_ = GetSubCommInfo(logicalLevel1plane_, logicalLevel0CommInfo_.localRank);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterARSFor91093Executor::CalcOptimalIntraRing(const OpParam& param)
{
    intraRingSize_
        = CalcOptimalIntraRingsize(param.DataDes.count, param.DataDes.dataType, HcclCMDType::HCCL_CMD_REDUCE_SCATTER);
    HCCL_INFO("intraRingSize_[%u]", intraRingSize_);
    return HCCL_SUCCESS;
}
REGISTER_EXEC("ReduceScatterARSFor91093Executor", ReduceScatterARSFor91093, CollReduceScatterARSFor91093Executor);
} // namespace hccl
