/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_operator_check.h"
#include "exception_util.h"
#include "adapter_error_manager_pub.h"
#include "hccl_log_keywords.h"

namespace Hccl {

void ReportOpCheckFailed(
    const std::string& paraName, const std::string& localPara, const std::string& remotePara, const OpType& optype,
    const std::string& optag)
{
    std::string opInfo = "Unknown";
    for (const auto& pair : HCOM_OP_TYPE_STR_MAP_V2) {
        if (pair.second == optype) {
            opInfo = std::string(pair.first);
            break;
        }
    }
    // 上报故障码EI0005
    RPT_INPUT_ERR(
        true, "EI0005", std::vector<std::string>({"ccl_op", "para_name", "local_para", "remote_para"}),
        std::vector<std::string>({opInfo, paraName, localPara, remotePara}));
    HCCL_ERROR(
        "[%s][%s] CMD information op[%s] tag[%s] %s check fail. local[%s], remote[%s]",
        LOG_KEYWORDS_INIT_CHANNEL.c_str(), LOG_KEYWORDS_PARAMETER_CONFLICT.c_str(), opInfo.c_str(), optag.c_str(),
        paraName.c_str(), localPara.c_str(), remotePara.c_str());
    THROW<InvalidParamsException>(StringFormat(
        "[RankConsistentImpl][CompareFrame][%s]op information%s group%s %s check fail. "
        "local[%s], remote[%s]",
        __func__, opInfo.c_str(), optag.c_str(), paraName.c_str(), localPara.c_str(), remotePara.c_str()));
}

void ReportOpCheckFailed(
    const std::string& paraName, uint32_t localPara, uint32_t remotePara, const OpType& optype,
    const std::string& optag)
{
    std::string opInfo = "Unknown";
    for (const auto& pair : HCOM_OP_TYPE_STR_MAP_V2) {
        if (pair.second == optype) {
            opInfo = std::string(pair.first);
            break;
        }
    }
    // 上报故障码EI0005
    RPT_INPUT_ERR(
        true, "EI0005", std::vector<std::string>({"ccl_op", "para_name", "local_para", "remote_para"}),
        std::vector<std::string>({opInfo, paraName, std::to_string(localPara), std::to_string(remotePara)}));
    HCCL_ERROR(
        "[%s][%s] CMD information op[%s] tag[%s] %s check fail. local[%u], remote[%u]",
        LOG_KEYWORDS_INIT_CHANNEL.c_str(), LOG_KEYWORDS_PARAMETER_CONFLICT.c_str(), opInfo.c_str(), optag.c_str(),
        paraName.c_str(), localPara, remotePara);
    THROW<InvalidParamsException>(StringFormat(
        "[RankConsistentImpl][CompareFrame][%s]op information%s group%s %s check fail. "
        "local[%u], remote[%u]",
        __func__, opInfo.c_str(), optag.c_str(), paraName.c_str(), localPara, remotePara));
}

void CompareDataDesOp(const CollOperator& localOpData, const CollOperator& remoteOpData)
{
    if (localOpData.dataDes.dataCount != remoteOpData.dataDes.dataCount) {
        ReportOpCheckFailed(
            "dataDes.dataCount", localOpData.dataDes.dataCount, remoteOpData.dataDes.dataCount, localOpData.opType,
            localOpData.opTag);
    }

    if (localOpData.dataDes.dataType != remoteOpData.dataDes.dataType) {
        ReportOpCheckFailed(
            "dataDes.dataType", localOpData.dataDes.dataType.Describe(), remoteOpData.dataDes.dataType.Describe(),
            localOpData.opType, localOpData.opTag);
    }

    if (localOpData.dataDes.strideCount != remoteOpData.dataDes.strideCount) {
        ReportOpCheckFailed(
            "dataDes.strideCount", localOpData.dataDes.strideCount, remoteOpData.dataDes.strideCount,
            localOpData.opType, localOpData.opTag);
    }
}

void CompareVDataDesOp(const CollOperator& localOpData, const CollOperator& remoteOpData)
{
    if (localOpData.vDataDes.dataType != remoteOpData.vDataDes.dataType) {
        ReportOpCheckFailed(
            "vDataDes.dataType", localOpData.vDataDes.dataType.Describe(), remoteOpData.vDataDes.dataType.Describe(),
            localOpData.opType, localOpData.opTag);
    }
}

void CompareAlltoAllOp(const CollOperator& localOpData, const CollOperator& remoteOpData)
{
    if (localOpData.all2AllDataDes.sendType != remoteOpData.all2AllDataDes.recvType) {
        ReportOpCheckFailed(
            "all2AllDataDes.sendType", localOpData.all2AllDataDes.sendType.Describe(),
            remoteOpData.all2AllDataDes.recvType.Describe(), localOpData.opType, localOpData.opTag);
    }

    if (localOpData.all2AllDataDes.recvType != remoteOpData.all2AllDataDes.sendType) {
        ReportOpCheckFailed(
            "all2AllDataDes.recvType", localOpData.all2AllDataDes.recvType.Describe(),
            remoteOpData.all2AllDataDes.sendType.Describe(), localOpData.opType, localOpData.opTag);
    }

    if (localOpData.all2AllDataDes.sendCount != remoteOpData.all2AllDataDes.recvCount) {
        ReportOpCheckFailed(
            "all2AllDataDes.sendCount", localOpData.all2AllDataDes.sendCount, remoteOpData.all2AllDataDes.recvCount,
            localOpData.opType, localOpData.opTag);
    }

    if (localOpData.all2AllDataDes.recvCount != remoteOpData.all2AllDataDes.sendCount) {
        ReportOpCheckFailed(
            "all2AllDataDes.recvCount", localOpData.all2AllDataDes.recvCount, remoteOpData.all2AllDataDes.sendCount,
            localOpData.opType, localOpData.opTag);
    }
}

void CompareAlltoAllVOp(const CollOperator& localOpData, const CollOperator& remoteOpData)
{
    if (localOpData.all2AllVDataDes.sendType != remoteOpData.all2AllVDataDes.recvType) {
        ReportOpCheckFailed(
            "all2AllVDataDes.sendType", localOpData.all2AllVDataDes.sendType.Describe(),
            remoteOpData.all2AllVDataDes.recvType.Describe(), localOpData.opType, localOpData.opTag);
    }

    if (localOpData.all2AllVDataDes.recvType != remoteOpData.all2AllVDataDes.sendType) {
        ReportOpCheckFailed(
            "all2AllVDataDes.recvType", localOpData.all2AllVDataDes.recvType.Describe(),
            remoteOpData.all2AllVDataDes.sendType.Describe(), localOpData.opType, localOpData.opTag);
    }
}

void CompareAlltoAllVCOp(const CollOperator& localOpData, const CollOperator& remoteOpData)
{
    if (localOpData.all2AllVCDataDes.sendType != remoteOpData.all2AllVCDataDes.recvType) {
        ReportOpCheckFailed(
            "all2AllVCDataDes.sendType", localOpData.all2AllVCDataDes.sendType.Describe(),
            remoteOpData.all2AllVCDataDes.recvType.Describe(), localOpData.opType, localOpData.opTag);
    }

    if (localOpData.all2AllVCDataDes.recvType != remoteOpData.all2AllVCDataDes.sendType) {
        ReportOpCheckFailed(
            "all2AllVCDataDes.recvType", localOpData.all2AllVCDataDes.recvType.Describe(),
            remoteOpData.all2AllVCDataDes.sendType.Describe(), localOpData.opType, localOpData.opTag);
    }
}

void CompareNormalOp(const CollOperator& localOpData, const CollOperator& remoteOpData)
{
    if (localOpData.opMode != remoteOpData.opMode) {
        ReportOpCheckFailed(
            "opMode", localOpData.opMode.Describe(), remoteOpData.opMode.Describe(), localOpData.opType,
            localOpData.opTag);
    }

    if (localOpData.opType == OpType::SEND) {
        if (remoteOpData.opType != OpType::RECV) {
            ReportOpCheckFailed(
                "opType", localOpData.opType.Describe(), remoteOpData.opType.Describe(), localOpData.opType,
                localOpData.opTag);
        }
    } else if (localOpData.opType == OpType::RECV) {
        if (remoteOpData.opType != OpType::SEND) {
            ReportOpCheckFailed(
                "opType", localOpData.opType.Describe(), remoteOpData.opType.Describe(), localOpData.opType,
                localOpData.opTag);
        }
    } else if (localOpData.opType != remoteOpData.opType) {
        ReportOpCheckFailed(
            "opType", localOpData.opType.Describe(), remoteOpData.opType.Describe(), localOpData.opType,
            localOpData.opTag);
    }

    if (localOpData.reduceOp != remoteOpData.reduceOp) {
        ReportOpCheckFailed(
            "reduceOp", localOpData.reduceOp.Describe(), remoteOpData.reduceOp.Describe(), localOpData.opType,
            localOpData.opTag);
    }

    if (localOpData.dataType != remoteOpData.dataType) {
        ReportOpCheckFailed(
            "dataType", localOpData.dataType.Describe(), remoteOpData.dataType.Describe(), localOpData.opType,
            localOpData.opTag);
    }

    if (localOpData.opType != OpType::ALLGATHERV && localOpData.opType != OpType::REDUCESCATTERV) {
        if (localOpData.dataCount != remoteOpData.dataCount) {
            ReportOpCheckFailed(
                "dataCount", localOpData.dataCount, remoteOpData.dataCount, localOpData.opType, localOpData.opTag);
        }
    }

    if (localOpData.root != remoteOpData.root) {
        ReportOpCheckFailed("root", localOpData.root, remoteOpData.root, localOpData.opType, localOpData.opTag);
    }

    if (localOpData.opType == OpType::SEND || localOpData.opType == OpType::RECV) {
        if (localOpData.myRank != remoteOpData.sendRecvRemoteRank) {
            ReportOpCheckFailed(
                "sendRecvRemoteRank", localOpData.myRank, remoteOpData.sendRecvRemoteRank, localOpData.opType,
                localOpData.opTag);
        }
    }

    if (localOpData.opTag != remoteOpData.opTag) {
        ReportOpCheckFailed("opTag", localOpData.opTag, remoteOpData.opTag, localOpData.opType, localOpData.opTag);
    }

    if (localOpData.staticAddr != remoteOpData.staticAddr) {
        ReportOpCheckFailed(
            "staticAddr", static_cast<uint32_t>(localOpData.staticAddr), static_cast<uint32_t>(remoteOpData.staticAddr),
            localOpData.opType, localOpData.opTag);
    }

    if (localOpData.staticShape != remoteOpData.staticShape) {
        ReportOpCheckFailed(
            "staticShape", static_cast<uint32_t>(localOpData.staticShape),
            static_cast<uint32_t>(remoteOpData.staticShape), localOpData.opType, localOpData.opTag);
    }

    if (localOpData.outputDataType != remoteOpData.outputDataType) {
        ReportOpCheckFailed(
            "outputDataType", localOpData.outputDataType.Describe(), remoteOpData.outputDataType.Describe(),
            localOpData.opType, localOpData.opTag);
    }
}

/*
当前校验类型不支持vDataDes和all2AllVCDataDes相关内容；batchSendRecvDataDes中的itemNum字段没有校验的必要，双端该值可能不相等
*/
void CheckCollOperator(const CollOperator& localOpData, const CollOperator& remoteOpData)
{
    CompareNormalOp(localOpData, remoteOpData);

    if (localOpData.opType == OpType::BATCHSENDRECV) {
        return;
    }

    if (localOpData.opType == OpType::ALLTOALL) {
        CompareAlltoAllOp(localOpData, remoteOpData);
        return;
    }

    if (localOpData.opType == OpType::ALLTOALLV) {
        CompareAlltoAllVOp(localOpData, remoteOpData);
        return;
    }

    if (localOpData.opType == OpType::ALLTOALLVC) {
        CompareAlltoAllVCOp(localOpData, remoteOpData);
        return;
    }

    if (localOpData.opType == OpType::ALLGATHERV || localOpData.opType == OpType::REDUCESCATTERV) {
        CompareVDataDesOp(localOpData, remoteOpData);
        return;
    }

    CompareDataDesOp(localOpData, remoteOpData);
    return;
}
} // namespace Hccl
