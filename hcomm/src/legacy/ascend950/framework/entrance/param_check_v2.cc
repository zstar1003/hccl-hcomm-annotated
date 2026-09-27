/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "param_check_v2.h"
#include <linux/limits.h>
#include <unordered_set>
#include <map>
#include <vector>
#include <fstream>
#include <linux/limits.h>
#include <adapter_error_manager_pub.h>
#include "log.h"
#include "exception_util.h"
#include "data_type.h"
#include "rank_table_crc_bridge.h"

using namespace std;
using namespace Hccl;
struct EnumHashV2 {
    template <typename T>
    std::size_t operator()(T t) const
    {
        return static_cast<std::size_t>(t);
    }
};

constexpr u64 SYS_MAX_COUNT = 0x7FFFFFFFF; // 系统当前支持的最大count数

const std::unordered_set<HcclDataType, EnumHashV2> HCCL_SUPPORT_DATA_TYPE_V2
    = {HCCL_DATA_TYPE_INT8,   HCCL_DATA_TYPE_INT16,   HCCL_DATA_TYPE_INT32,   HCCL_DATA_TYPE_FP16,
       HCCL_DATA_TYPE_FP32,   HCCL_DATA_TYPE_INT64,   HCCL_DATA_TYPE_UINT64,  HCCL_DATA_TYPE_UINT8,
       HCCL_DATA_TYPE_UINT16, HCCL_DATA_TYPE_UINT32,  HCCL_DATA_TYPE_FP64,    HCCL_DATA_TYPE_BFP16,
       HCCL_DATA_TYPE_HIF8,   HCCL_DATA_TYPE_FP8E4M3, HCCL_DATA_TYPE_FP8E5M2, HCCL_DATA_TYPE_FP8E8M0};

const std::unordered_set<HcclReduceOp, EnumHashV2> HCCL_SUPPORT_REDUCE_OP_V2
    = {HCCL_REDUCE_SUM, HCCL_REDUCE_MAX, HCCL_REDUCE_MIN, HCCL_REDUCE_PROD};

const std::unordered_set<HcclDataType, EnumHashV2> HCCL_SUPPORT_PROD_DATA_TYPE_V2
    = {HCCL_DATA_TYPE_INT64, HCCL_DATA_TYPE_UINT64, HCCL_DATA_TYPE_FP64};

const std::map<HcclReduceOp, std::string> HCOM_REDUCE_OP_STR_MAP_V2{
    {HcclReduceOp::HCCL_REDUCE_SUM, "sum"},
    {HcclReduceOp::HCCL_REDUCE_MAX, "max"},
    {HcclReduceOp::HCCL_REDUCE_MIN, "min"},
    {HcclReduceOp::HCCL_REDUCE_RESERVED, "reserved"}};

const std::map<HcclDataType, std::string> HCOM_DATA_TYPE_STR_MAP_V2{
    {HcclDataType::HCCL_DATA_TYPE_INT8, "int8"},       {HcclDataType::HCCL_DATA_TYPE_INT16, "int16"},
    {HcclDataType::HCCL_DATA_TYPE_INT32, "int32"},     {HcclDataType::HCCL_DATA_TYPE_INT64, "int64"},
    {HcclDataType::HCCL_DATA_TYPE_UINT64, "uint64"},   {HcclDataType::HCCL_DATA_TYPE_FP16, "float16"},
    {HcclDataType::HCCL_DATA_TYPE_FP32, "float32"},    {HcclDataType::HCCL_DATA_TYPE_UINT8, "uint8"},
    {HcclDataType::HCCL_DATA_TYPE_UINT16, "uint16"},   {HcclDataType::HCCL_DATA_TYPE_UINT32, "uint32"},
    {HcclDataType::HCCL_DATA_TYPE_FP64, "float64"},    {HcclDataType::HCCL_DATA_TYPE_BFP16, "bfloat16"},
    {HcclDataType::HCCL_DATA_TYPE_INT128, "int128"},   {HcclDataType::HCCL_DATA_TYPE_HIF8, "hif8"},
    {HcclDataType::HCCL_DATA_TYPE_FP8E4M3, "fp8e4m3"}, {HcclDataType::HCCL_DATA_TYPE_FP8E5M2, "fp8e5m2"},
    {HcclDataType::HCCL_DATA_TYPE_FP8E8M0, "fp8e8m0"}, {HcclDataType::HCCL_DATA_TYPE_RESERVED, "reserved"}};

std::string GetDataTypeEnumStrV2(HcclDataType dataType)
{
    auto iter = HCOM_DATA_TYPE_STR_MAP_V2.find(dataType);
    if (iter == HCOM_DATA_TYPE_STR_MAP_V2.end()) {
        return "HcclDataType(" + std::to_string(dataType) + ")";
    } else {
        return iter->second;
    }
}

HcclResult HcomCheckTagV2(const char* tag)
{
    CHK_PTR_NULL(tag);

    u32 tagLen = strnlen(tag, TAG_MAX_LEN + 1);
    if (tagLen == (TAG_MAX_LEN + 1) || tagLen == 0) {
        string errReason = "please check tagLen that is out of range, range[1," + std::to_string(TAG_MAX_LEN) + "]";
        RPT_INPUT_ERR(
            true, "EI0003", vector<string>({"ccl_op", "value", "parameter", "expect"}),
            vector<string>({"HcomCheckTagV2", std::to_string(tagLen), "tag", errReason}));
        if (tagLen == 0) {
            HCCL_ERROR(
                "[%s][%s] errNo[0x%016llx] tag is empty", LOG_KEYWORDS_TASK_EXEC.c_str(),
                LOG_KEYWORDS_INVALID_ARGUMENT.c_str(), HCCL_ERROR_CODE(HCCL_E_PARA));
        } else {
            HCCL_ERROR(
                "[%s][%s] errNo[0x%016llx] tag is too long, range[1,%u]", LOG_KEYWORDS_TASK_EXEC.c_str(),
                LOG_KEYWORDS_INVALID_ARGUMENT.c_str(), HCCL_ERROR_CODE(HCCL_E_PARA), TAG_MAX_LEN);
        }
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult HcomCheckGroupNameV2(const char* group)
{
    if (group != nullptr) {
        u32 groupLen = strnlen(group, GROUP_NAME_MAX_LEN + 1);
        if (groupLen == (GROUP_NAME_MAX_LEN + 1) || groupLen == 0) {
            string errReason
                = "please check groupLen that is out of range, range[1," + std::to_string(GROUP_NAME_MAX_LEN) + "]";
            RPT_INPUT_ERR(
                true, "EI0003", vector<string>({"ccl_op", "value", "parameter", "expect"}),
                vector<string>({"HcomCheckGroupNameV2", std::to_string(groupLen), "group name", errReason}));
            HCCL_ERROR(
                "[%s][%s] errNo[0x%016llx] group name[%s] length[%lu] is invalid", LOG_KEYWORDS_TASK_EXEC.c_str(),
                LOG_KEYWORDS_INVALID_ARGUMENT.c_str(), HCCL_ERROR_CODE(HCCL_E_PARA), group, groupLen);
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcomCheckCountV2(const u64 count)
{
    if (count > SYS_MAX_COUNT) {
        string errReason = "please check count that is bigger than MAX[" + std::to_string(SYS_MAX_COUNT) + "]";
        RPT_INPUT_ERR(
            true, "EI0003", vector<string>({"ccl_op", "value", "parameter", "expect"}),
            vector<string>({"HcomCheckCountV2", std::to_string(count), "count", errReason}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] count[%llu] is invalid(bigger than MAX count[%llu])",
            LOG_KEYWORDS_TASK_EXEC.c_str(), LOG_KEYWORDS_INVALID_ARGUMENT.c_str(), HCCL_ERROR_CODE(HCCL_E_PARA), count,
            SYS_MAX_COUNT);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

std::string GetReduceOpEnumStrV2(HcclReduceOp reduceOp)
{
    auto iter = HCOM_REDUCE_OP_STR_MAP_V2.find(reduceOp);
    if (iter == HCOM_REDUCE_OP_STR_MAP_V2.end()) {
        return "HcclReduceOp(" + std::to_string(reduceOp) + ")";
    } else {
        return iter->second;
    }
}

HcclResult HcomCheckDataTypeV2(const HcclDataType dataType)
{
    if (HCCL_SUPPORT_DATA_TYPE_V2.find(dataType) == HCCL_SUPPORT_DATA_TYPE_V2.end()) {
        std::string expect;
        bool isFirst = true;
        for (auto type : HCCL_SUPPORT_DATA_TYPE_V2) {
            auto it = DATA_TYPE_TO_STRING_MAP.find(type);
            if (it != DATA_TYPE_TO_STRING_MAP.end()) {
                if (!isFirst) {
                    expect += ",";
                }
                expect += it->second;
                isFirst = false;
            }
        }
        RPT_INPUT_ERR(
            true, "EI0003", vector<string>({"ccl_op", "value", "parameter", "expect"}),
            vector<string>({"HcomCheckDataTypeV2", GetDataTypeEnumStrV2(dataType).c_str(), "dataType", expect}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] data type[%s] not supported", LOG_KEYWORDS_TASK_EXEC.c_str(),
            LOG_KEYWORDS_INVALID_ARGUMENT.c_str(), HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT),
            GetDataTypeEnumStrV2(dataType).c_str());
        return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}

HcclResult
HcomCheckOpParamV2(const char* tag, const u64 count, const HcclDataType dataType, const char* group, const void* stream)
{
    HcclResult ret = HcomCheckGroupNameV2(group);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[Check][OpParam]errNo[0x%016llx] group name is invalid", HCCL_ERROR_CODE(ret)),
        ret);

    CHK_RET(HcomCheckOpParamV2(tag, count, dataType, stream));

    return HCCL_SUCCESS;
}

HcclResult HcomCheckOpParamV2(const u64 count, const HcclDataType dataType, const char* group)
{
    HcclResult ret = HcomCheckGroupNameV2(group);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[Check][OpParam]errNo[0x%016llx] group name is invalid", HCCL_ERROR_CODE(ret)),
        ret);

    CHK_RET(HcomCheckOpParamV2(count, dataType));

    return HCCL_SUCCESS;
}

HcclResult HcomCheckOpParamV2(
    const char* tag, const u64 count, const HcclDataType dataType, const void* stream) // 校验opParam失败都上报EI0003
{
    CHK_RET(HcomCheckOpParamV2(tag, count, dataType));

    if (stream == nullptr) {
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] stream is NULL.", LOG_KEYWORDS_TASK_EXEC.c_str(),
            LOG_KEYWORDS_INVALID_ARGUMENT.c_str(), HCCL_ERROR_CODE(HCCL_E_PTR));
        RPT_INPUT_ERR(
            true, "EI0003", vector<string>({"ccl_op", "value", "parameter", "expect"}),
            vector<string>({"HcomCheckOpParamV2", "NULL", "stream", "please check stream that should not be nullptr"}));
        return HCCL_E_PTR;
    }

    return HCCL_SUCCESS;
}

HcclResult HcomCheckOpParamV2(const char* tag, const u64 count, const HcclDataType dataType)
{
    HcclResult ret = HcomCheckTagV2(tag);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[Check][OpParam]errNo[0x%016llx] tag is invalid", HCCL_ERROR_CODE(ret)), ret);

    ret = HcomCheckCountV2(count);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[Check][OpParam]errNo[0x%016llx] count is out of range", HCCL_ERROR_CODE(ret)),
        ret);

    ret = HcomCheckDataTypeV2(dataType);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[Check][OpParam]errNo[0x%016llx] dataType is invalid", HCCL_ERROR_CODE(ret)),
        ret);

    return HCCL_SUCCESS;
}

HcclResult HcomCheckOpParamV2(const u64 count, const HcclDataType dataType)
{
    HcclResult ret = HcomCheckCountV2(count);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[Check][OpParam]errNo[0x%016llx] count is out of range", HCCL_ERROR_CODE(ret)),
        ret);

    ret = HcomCheckDataTypeV2(dataType);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[Check][OpParam]errNo[0x%016llx] dataType is invalid", HCCL_ERROR_CODE(ret)),
        ret);

    return HCCL_SUCCESS;
}

HcclResult HcomCheckReductionOpV2(const HcclReduceOp op)
{
    if (HCCL_SUPPORT_REDUCE_OP_V2.find(op) == HCCL_SUPPORT_REDUCE_OP_V2.end()) {
        std::string supportedList;
        bool first = true;
        for (const auto& validOp : HCCL_SUPPORT_REDUCE_OP_V2) {
            if (!first) {
                supportedList += ", ";
            }
            supportedList += GetReduceOpEnumStrV2(validOp);
            first = false;
        }
        RPT_INPUT_ERR(
            true, "EI0003", vector<string>({"ccl_op", "value", "parameter", "expect"}),
            vector<string>({"HcomCheckReductionOpV2", GetReduceOpEnumStrV2(op), "op", "one of " + supportedList}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] Op:[%s] not supported", LOG_KEYWORDS_TASK_EXEC.c_str(),
            LOG_KEYWORDS_INVALID_ARGUMENT.c_str(), HCCL_ERROR_CODE(HCCL_E_PARA), GetReduceOpEnumStrV2(op).c_str());
        return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}

HcclResult HcomCheckProdDataTypeV2(const HcclDataType dataType)
{
    if (HCCL_SUPPORT_PROD_DATA_TYPE_V2.find(dataType) == HCCL_SUPPORT_PROD_DATA_TYPE_V2.end()) {
        std::string supportedList;
        bool first = true;
        for (const auto& validType : HCCL_SUPPORT_PROD_DATA_TYPE_V2) {
            if (!first) {
                supportedList += ", ";
            }
            supportedList += GetDataTypeEnumStrV2(validType);
            first = false;
        }
        RPT_INPUT_ERR(
            true, "EI0003", vector<string>({"ccl_op", "value", "parameter", "expect"}),
            vector<string>(
                {"HcomCheckProdDataTypeV2", GetDataTypeEnumStrV2(dataType), "dataType", "one of " + supportedList}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] DataType:[%s] not supported PROD", LOG_KEYWORDS_TASK_EXEC.c_str(),
            LOG_KEYWORDS_INVALID_ARGUMENT.c_str(), HCCL_ERROR_CODE(HCCL_E_PARA),
            GetDataTypeEnumStrV2(dataType).c_str());
        return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}

HcclResult HcomCheckReduceDataTypeV2(const HcclDataType dataType, const HcclReduceOp op)
{
    switch (op) {
        case HCCL_REDUCE_SUM:
        case HCCL_REDUCE_MAX:
        case HCCL_REDUCE_MIN:
            break;
        case HCCL_REDUCE_PROD:
            CHK_RET(HcomCheckProdDataTypeV2(dataType));
            break;
        default:
            break;
    }
    return HCCL_SUCCESS;
}

HcclResult HcomCheckAlltoAllVExternalMemV2(
    const void* sendBuf, const void* sendCounts, const void* recvBuf, const void* recvCounts, u32 rankSize)
{
    CHK_PRT_RET(
        sendBuf != nullptr && recvBuf != nullptr && sendBuf == recvBuf,
        HCCL_ERROR("[HcomCheckAlltoAllVExternalMem] sendBuf and recvBuf cannot be same."), HCCL_E_PARA);
    CHECK_NULLPTR(sendCounts, "[HcomCheckAlltoAllVExternalMemV2] sendCounts is nullptr!");
    CHECK_NULLPTR(recvCounts, "[HcomCheckAlltoAllVExternalMemV2] recvCounts is nullptr!");
    u64* sendCountsPtr = const_cast<u64*>(static_cast<const u64*>(sendCounts));
    u64* recvCountsPtr = const_cast<u64*>(static_cast<const u64*>(recvCounts));
    bool hasSend = false;
    bool hasRecv = false;
    bool invalidSendCount = false;
    bool invalidRecvCount = false;
    for (u32 i = 0; i < rankSize; i++) {
        if (*(sendCountsPtr + i) != 0) {
            invalidSendCount = invalidSendCount || (*(sendCountsPtr + i) > SYS_MAX_COUNT);
            hasSend = true;
        }
        if (*(recvCountsPtr + i) != 0) {
            invalidRecvCount = invalidRecvCount || (*(recvCountsPtr + i) > SYS_MAX_COUNT);
            hasRecv = true;
        }
    }

    if (invalidSendCount || invalidRecvCount || HcclCheckLogLevel(DLOG_DEBUG)) {
        std::string sendCountStr = "sendCounts:";
        std::string recvCountStr = "recvCounts:";
        for (u32 i = 0; i < rankSize; i++) {
            sendCountStr += ' ' + std::to_string(*(sendCountsPtr + i));
            recvCountStr += ' ' + std::to_string(*(recvCountsPtr + i));
        }

        CHK_PRT_RET(
            invalidSendCount,
            HCCL_ERROR(
                "HcomCheckAlltoAllVExternalMem sendCounts[%s] is invalid.(bigger than MAX count[%llu])",
                sendCountStr.c_str(), SYS_MAX_COUNT),
            HCCL_E_PARA);
        CHK_PRT_RET(
            invalidRecvCount,
            HCCL_ERROR(
                "HcomCheckAlltoAllVExternalMem recvCounts[%s] is invalid.(bigger than MAX count[%llu])",
                recvCountStr.c_str(), SYS_MAX_COUNT),
            HCCL_E_PARA);

        HCCL_INFO("[HcomCheckAlltoAllVExternalMem] sendCounts: %s", sendCountStr.c_str());
        HCCL_INFO("[HcomCheckAlltoAllVExternalMem] recvCounts: %s", recvCountStr.c_str());
    }

    if (hasSend) {
        CHK_PTR_NULL(sendBuf);
    }
    if (hasRecv) {
        CHK_PTR_NULL(recvBuf);
    }
    return HCCL_SUCCESS;
}

HcclResult HcomCheckAlltoAllVCExternalMemV2(
    const void* sendBuf, const void* sendCountMatrix, const void* recvBuf, u32 rankSize, u32 rank)
{
    CHK_PRT_RET(
        sendBuf != nullptr && recvBuf != nullptr && sendBuf == recvBuf,
        HCCL_ERROR("[HcomCheckAlltoAllVCExternalMemV2] sendBuf and recvBuf addr cannot be same."), HCCL_E_PARA);

    CHECK_NULLPTR(sendCountMatrix, "[HcomCheckAlltoAllVCExternalMemV2] sendCountMatrix is nullptr!");
    u64* sendCountMatrixPtr = const_cast<u64*>(static_cast<const u64*>(sendCountMatrix));
    bool hasSend = false;
    bool hasRecv = false;

    for (u32 i = 0; i < rankSize; i++) {
        u64 sendCount = *(sendCountMatrixPtr + rank * rankSize + i);
        CHK_RET(HcomCheckCountV2(sendCount));
        if (hasSend == false && sendCount != 0) {
            hasSend = true;
        }
        u64 recvCount = *(sendCountMatrixPtr + i * rankSize + rank);
        if (hasRecv == false && recvCount != 0) {
            hasRecv = true;
        }
        HCCL_DEBUG(
            "[HcomCheckAlltoAllVCExternalMemV2] myrank[%u] rmtrank[%u] sendCount[%llu] recvCount[%llu]", rank, i,
            sendCount, recvCount);
    }
    if (hasSend) {
        CHK_PTR_NULL(sendBuf);
    }
    if (hasRecv) {
        CHK_PTR_NULL(recvBuf);
    }
    return HCCL_SUCCESS;
}

HcclResult HcomCheckAlltoAllVCEmptyV2(
    const void* sendBuf, const void* sendCountMatrix, const void* recvBuf, u32 rankSize, bool& isEmpty)
{
    CHK_PRT_RET(
        sendBuf != nullptr && recvBuf != nullptr && sendBuf == recvBuf,
        HCCL_ERROR("[HcomCheckAlltoAllVCEmptyV2] sendBuf and recvBuf addr cannot be same."), HCCL_E_PARA);

    CHECK_NULLPTR(sendCountMatrix, "[HcomCheckAlltoAllVCEmptyV2] sendCountMatrix is nullptr!");
    u64* sendCountMatrixPtr = const_cast<u64*>(static_cast<const u64*>(sendCountMatrix));
    bool hasSend = false;
    bool hasRecv = false;

    for (u32 i = 0; i < rankSize; i++) {
        for (u32 j = 0; j < rankSize; j++) {
            u64 sendCount = *(sendCountMatrixPtr + i * rankSize + j);
            CHK_RET(HcomCheckCountV2(sendCount));
            if (hasSend == false && sendCount != 0) {
                hasSend = true;
            }
            u64 recvCount = *(sendCountMatrixPtr + j * rankSize + i);
            CHK_RET(HcomCheckCountV2(recvCount));
            if (hasRecv == false && recvCount != 0) {
                hasRecv = true;
            }
            HCCL_DEBUG(
                "[HcomCheckAlltoAllVCEmptyV2] myrank[%u] rmtrank[%u] sendCount[%llu] recvCount[%llu]", i, j, sendCount,
                recvCount);
        }
    }
    isEmpty = !(hasSend || hasRecv);
    return HCCL_SUCCESS;
}

HcclResult HcomCheckUserRankV2(const u32 totalRanks, const u32 userRank)
{
    if (userRank >= totalRanks) {
        HCCL_ERROR(
            "[Check][UserRank]errNo[0x%016llx] userRank:[%u] is out of range[0 ~ %u]", HCCL_ERROR_CODE(HCCL_E_PARA),
            userRank, totalRanks - 1);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult HcomLoadRankTableFileV2(const char* clusterInfo, std::string& rankTableM)
{
    CHK_PTR_NULL(clusterInfo);
    CHK_PRT_RET(
        strlen(clusterInfo) >= PATH_MAX,
        HCCL_ERROR("[HcomLoadRankTableFileV2]clusterInfo exceeds PATH_MAX[%u]", PATH_MAX), HCCL_E_PARA);

    // 校验文件是否存在
    char resolvedPath[PATH_MAX] = {0};
    if (realpath(clusterInfo, resolvedPath) == nullptr) {
        std::string rankTablePath(clusterInfo);
        RPT_INPUT_ERR(
            true, "EI0004", std::vector<std::string>({"error_reason", "ranktable_path"}),
            std::vector<std::string>(
                {rankTablePath, "The rankTable file path does not exist, the permission is insufficient, or the JSON "
                                "format is incorrect."}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] RanktableRealPath: %s is not a valid real path", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CONFIG.c_str(), HCCL_ERROR_CODE(HCCL_E_PARA), clusterInfo);
        return HCCL_E_PARA;
    }

    HCCL_INFO("[RankTable]waiting for json file load complete");
    std::ifstream infoFile(resolvedPath, std::ifstream::in | std::ifstream::ate); // ate模式打开文件,方便获取size
    if (!infoFile) {
        HCCL_ERROR("[RankTable]open file %s failed", resolvedPath);
        return HCCL_E_INTERNAL;
    }

    uint64_t fileSize = infoFile.tellg();
    if (fileSize > RANKTABLE_MAX_SIZE) {
        HCCL_ERROR("[RankTable]load ranktable failed, file size = %llu is too large", fileSize);
        return HCCL_E_PARA;
    }

    infoFile.seekg(0, std::ifstream::beg); // 重置文件指针到开头,准备读取文件内容
    std::stringstream rankTableStr;
    rankTableStr << infoFile.rdbuf();
    rankTableM = rankTableStr.str();
    if (rankTableM.empty()) {
        HCCL_ERROR("[RankTable]load ranktable failed, file is empty");
        return HCCL_E_PARA;
    }
    s32 deviceLogicId = 0;
    aclrtGetDevice(&deviceLogicId);
    RankTableCrcBridge::GetInstance().RecordRankTableJsonCrc(deviceLogicId, rankTableM);
    return HCCL_SUCCESS;
}

HcclResult HcomCheckVOpParamV2(u32 rankId, u32 rankSize, u64 count, void* inCounts)
{
    u64* counts = static_cast<u64*>(inCounts);
    CHK_PRT_RET(
        rankId > rankSize,
        HCCL_ERROR("[%s] rankId[%u] is invalid(larger than rankSize[%u])", __func__, rankId, rankSize), HCCL_E_PARA);
    CHK_PRT_RET(
        count != counts[rankId],
        HCCL_ERROR("[%s] sendCount[%llu] is invalid(not equal to Counts[%llu])", __func__, count, counts[rankId]),
        HCCL_E_PARA);
    return HCCL_SUCCESS;
}

void HcomGetHashFromSendCountMatrixV2(
    u64& sendCountMatrixHash, const void* sendCountMatrix, u64 rankSize, const std::string& tag)
{
    std::string sendCountMatrixStr;
    std::hash<std::string> hashString;
    for (u32 i = 0; i < rankSize; i++) {
        for (u32 j = 0; j < rankSize; j++) {
            std::string curSendCountStr
                = std::to_string(*(static_cast<const u64*>(sendCountMatrix) + i * rankSize + j));
            sendCountMatrixStr += curSendCountStr + '_';
        }
    }
    sendCountMatrixHash = hashString(sendCountMatrixStr.c_str());
    HCCL_DEBUG("[HcomGetHashFromSendCountMatrix] tag[%s], sendCountMatrixHash[%llu]", tag.c_str(), sendCountMatrixHash);
}
