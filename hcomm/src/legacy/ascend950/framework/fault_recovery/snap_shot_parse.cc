/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "snap_shot_parse.h"
#include "checkcrc.h"
#include "ip_address.h"
#include "comm_manager.h"

using namespace Hccl;

SnapShotParser& SnapShotParser::GetInstance()
{
    static SnapShotParser instance;
    return instance;
}

// 保持快照，需要确保快照保持的是二进制流，需要确保换节点后能够继续使用，做save是使用，读出来dump，之后清空
BinaryStream& SnapShotParser::GetSnapShotBuf() { return snapShotStream_; }

// 恢复通信域，recover调用，把传入的备份流流反序列化到本地结构
HcclResult SnapShotParser::ParseSnapshotToLocalBuff(void* snapshotBuf, uint32_t snapshotBufSize, SnapShotBuf& localBuff)
{
    try {
        CHK_PTR_NULL(snapshotBuf);
        if (snapshotBufSize == 0) {
            HCCL_ERROR("[%s] snapshotBufSize is 0.", __func__);
            return HcclResult::HCCL_E_PARA;
        }

        uint32_t dataLen = *static_cast<uint32_t*>(snapshotBuf); // 获取存储的 size
        HCCL_INFO("[%s] dataLen[%u], snapshotBufSize[%u]", __func__, dataLen, snapshotBufSize);
        if (dataLen + sizeof(dataLen) + sizeof(u32) != snapshotBufSize) {
            HCCL_ERROR("[%s] The storage size does not match the input size.", __func__);
            return HcclResult::HCCL_E_PARA;
        }

        char* pCur = static_cast<char*>(snapshotBuf);
        pCur = pCur + sizeof(dataLen); // 跳过头4个字节，指向crc值

        // 获取crc值
        u32 crcValue = *reinterpret_cast<u32*>(pCur);
        pCur = pCur + sizeof(u32); // 跳过crc 指向真正的数据段
        HCCL_INFO("[%s] crcValue[%u]", __func__, crcValue);

        std::vector<char> bufVec(pCur, pCur + dataLen);
        BinaryStream buf(bufVec);

        // 校验crc
        CHK_RET(CheckBufCrc32(buf, crcValue));

        HCCL_INFO("snapshotBuf start");
        CHK_RET(DeAllSnapShotStaticBuf(buf, localBuff));
        CHK_RET(DeAllSnapShotDynamicBuf(buf, localBuff));
        DeserializeCcuStatusBuf(buf, localBuff);
    } catch (std::exception& e) {
        HCCL_ERROR("[%s]Failed, exception caught:%s, please check input snapshot!", __func__, e.what());
        return HCCL_E_INTERNAL;
    } catch (...) {
        HCCL_ERROR("parse snapshot fail, please check input snapshot!");
        return HCCL_E_INTERNAL;
    };
    HCCL_INFO(
        "[%s] end, snapshotBuf is:subGroup num[%u], comm groupName[%s]", __func__, localBuff.groupNum,
        localBuff.snapshot.groupName);
    return HCCL_SUCCESS;
}

HcclResult SnapShotParser::DeAllSnapShotStaticBuf(BinaryStream& buf, SnapShotBuf& localBuff)
{
    HCCL_INFO("[%s] start", __func__);
    // 解析 静态流公共数据 SnapShotPub
    DeserializeCommVersionInfo(buf, localBuff.snapShotPub);
    // 解析  全局通信域
    string tmpGroupName;
    buf >> tmpGroupName;
    DeserializeCommInfo(buf, localBuff.snapshot);
    s32 ret = memcpy_s(
        localBuff.snapshot.groupName, sizeof(localBuff.snapshot.groupName), tmpGroupName.c_str(), tmpGroupName.size());
    if (ret != 0) {
        THROW<InternalException>(StringFormat("[%s] memcpy_s failed, ret=%d", __func__, ret));
    }
    // 获取 子通信域 静态buf数量
    size_t groupNum{0};
    buf >> groupNum;
    localBuff.groupNum = static_cast<uint32_t>(groupNum);
    HCCL_INFO("subGroup num[%u], groupNum[%u]", groupNum, localBuff.groupNum);
    if (localBuff.groupNum == 0) {
        HCCL_INFO("snapShot static part is empty.");
    } else {
        localBuff.subSnapshot.resize(localBuff.groupNum);
        for (auto& subSnapshot : localBuff.subSnapshot) {
            // 解析  子通信域
            string subComGroupName;
            buf >> subComGroupName;
            DeserializeSubCommInfo(buf, subSnapshot);
            s32 tmpRet = memcpy_s(
                subSnapshot.groupName, sizeof(subSnapshot.groupName), subComGroupName.c_str(), subComGroupName.size());
            if (tmpRet != 0) {
                THROW<InternalException>(StringFormat("[%s] memcpy_s failed, ret=%d", __func__, tmpRet));
            }
        }
    }
    HCCL_INFO("[%s] end", __func__);

    return HCCL_SUCCESS;
}

HcclResult SnapShotParser::DeAllSnapShotDynamicBuf(BinaryStream& buf, SnapShotBuf& localBuff)
{
    HCCL_INFO("[%s] start", __func__);

    buf >> localBuff.snapShotPub.step;

    // 解析 全局通信域 动态短流
    SnapShotDynamic commDynamicInfo;
    CHK_RET(DeSnapShotDynamicBuf(buf, commDynamicInfo));
    localBuff.snapshot.snapShotComm.levelRankPairs = commDynamicInfo.levelRankPairs;
    localBuff.snapshot.snapShotComm.submittedOpCnt = commDynamicInfo.submittedOpCnt;
    localBuff.snapshot.snapShotComm.opMode = commDynamicInfo.opMode;
    localBuff.snapshot.snapShotComm.linkGroupPair = commDynamicInfo.linkGroupPair;
    localBuff.snapshot.snapShotComm.opExecuteConfig = commDynamicInfo.opExecuteConfig;
    localBuff.snapshot.snapShotComm.commExecuteConfig = commDynamicInfo.commExecuteConfig;
    localBuff.snapshot.snapShotComm.isLoadOp = commDynamicInfo.isLoadOp;

    // 获取 子通信域 动态buf 数量
    std::size_t groupNum{0};
    buf >> groupNum;
    std::size_t subSnapshotSize = localBuff.subSnapshot.size();
    HCCL_INFO(
        "submittedOpCnt[%u] step[%u] subGroup num[%u]", localBuff.snapshot.snapShotComm.submittedOpCnt,
        localBuff.snapShotPub.step, groupNum);
    if (subSnapshotSize != groupNum) {
        HCCL_ERROR(
            "[%s], subSnapshot size is wrong ,subSnapshot size is %u, "
            "subCommDynamicInfo Size is %u",
            __func__, subSnapshotSize, groupNum);
        return HcclResult::HCCL_E_INTERNAL;
    }

    for (auto& subSnapshot : localBuff.subSnapshot) {
        SnapShotDynamic subCommDynamicInfo;
        CHK_RET(DeSnapShotDynamicBuf(buf, subCommDynamicInfo));
        subSnapshot.snapShotSubComm.levelRankPairs = subCommDynamicInfo.levelRankPairs;
        subSnapshot.snapShotSubComm.submittedOpCnt = subCommDynamicInfo.submittedOpCnt;
        subSnapshot.snapShotSubComm.opMode = subCommDynamicInfo.opMode;
        subSnapshot.snapShotSubComm.linkGroupPair = subCommDynamicInfo.linkGroupPair;
        subSnapshot.snapShotSubComm.opExecuteConfig = subCommDynamicInfo.opExecuteConfig;
        subSnapshot.snapShotSubComm.commExecuteConfig = subCommDynamicInfo.commExecuteConfig;
        subSnapshot.snapShotSubComm.isLoadOp = subCommDynamicInfo.isLoadOp;
    }
    HCCL_INFO("[%s] end", __func__);

    return HCCL_SUCCESS;
}

void SnapShotParser::DeserializeCcuStatusBuf(BinaryStream& buf, SnapShotBuf& localBuff) const
{
    HCCL_INFO("[%s] start", __func__);

    size_t useMsCommIdsSize{0};
    buf >> useMsCommIdsSize;
    if (useMsCommIdsSize > MAX_NUM_COMM_USING_MS) {
        THROW<InternalException>(StringFormat(
            "[%s] useMsCommIdsSize[%zu] > MAX_NUM_COMM_USING_MS[%u]", __func__, useMsCommIdsSize,
            MAX_NUM_COMM_USING_MS));
    }
    HCCL_INFO("[SnapShotParser][%s] useMsCommIdsSize = [%u]", __func__, useMsCommIdsSize);
    localBuff.ccuStatusSnapshot.useMsCommIds.resize(useMsCommIdsSize);
    for (auto& useMsCommIdCharArr : localBuff.ccuStatusSnapshot.useMsCommIds) {
        string useMsCommId;
        buf >> useMsCommId;
        HCCL_INFO("[SnapShotParser][%s] useMsCommId is %s", __func__, useMsCommId.c_str());
        s32 ret
            = memcpy_s(useMsCommIdCharArr.data(), sizeof(useMsCommIdCharArr), useMsCommId.c_str(), useMsCommId.size());
        if (ret != 0) {
            THROW<InternalException>(StringFormat("[%s] memcpy_s failed, ret=%d", __func__, ret));
        }
    }

    size_t useSchedCommIdsSize{0};
    buf >> useSchedCommIdsSize;
    HCCL_INFO("[SnapShotParser][%s] useSchedCommIdsSize = [%u]", __func__, useSchedCommIdsSize);
    localBuff.ccuStatusSnapshot.useSchedCommIds.resize(useSchedCommIdsSize);
    for (auto& useSchedCommIdCharArr : localBuff.ccuStatusSnapshot.useSchedCommIds) {
        string useSchedCommId;
        buf >> useSchedCommId;
        HCCL_INFO("[SnapShotParser][%s] useSchedCommId is %s", __func__, useSchedCommId.c_str());
        s32 ret = memcpy_s(
            useSchedCommIdCharArr.data(), sizeof(useSchedCommIdCharArr), useSchedCommId.c_str(), useSchedCommId.size());
        if (ret != 0) {
            THROW<InternalException>(StringFormat("[%s] memcpy_s failed, ret=%d", __func__, ret));
        }
    }
    HCCL_INFO("[%s] end", __func__);
}

// 生成 单个通信域 动态短流
HcclResult SnapShotParser::SerializeDynamicInfo(
    const std::vector<std::pair<u32, RankId>>& levelRankPairs, u32 submittedOpCnt, BinaryStream& binStream) const
{
    size_t count = levelRankPairs.size();
    binStream << count;
    HCCL_INFO("[%s], levelRankPairs Size[%u]", __func__, count);
    for (auto& pair : levelRankPairs) {
        binStream << pair.first << pair.second;
    }
    binStream << submittedOpCnt;
    HCCL_INFO("submittedOpCnt[%u]", submittedOpCnt);
    return HcclResult::HCCL_SUCCESS;
}

// 解析 单个通信域 动态短流
HcclResult SnapShotParser::DeSnapShotDynamicBuf(BinaryStream& buf, SnapShotDynamic& dynamicInfo) const
{
    try {
        u32 opAccState{0};
        buf >> opAccState;
        HCCL_DEBUG("[%s], opAccState[%u]", __func__, opAccState);
        dynamicInfo.opExecuteConfig.accState = static_cast<AcceleratorState::Value>(opAccState);
        u32 commAccState{0};
        buf >> commAccState;
        HCCL_DEBUG("[%s], commAccState[%u]", __func__, commAccState);
        dynamicInfo.commExecuteConfig.accState = static_cast<AcceleratorState::Value>(commAccState);
        buf >> dynamicInfo.isLoadOp;
        HCCL_DEBUG("[%s], isLoadOp[%d]", __func__, dynamicInfo.isLoadOp);

        buf >> dynamicInfo.submittedOpCnt;
        HCCL_INFO("[%s], submittedOpCnt[%u]", __func__, dynamicInfo.submittedOpCnt);
        if (dynamicInfo.submittedOpCnt == 0) {
            return HcclResult::HCCL_SUCCESS;
        }

        buf >> dynamicInfo.opMode;
        HCCL_DEBUG("[%s], opMode[%d]", __func__, static_cast<s32>(dynamicInfo.opMode));

        std::size_t count{0};
        buf >> count;
        HCCL_INFO("levelRankPairs Size[%u]", count);
        if (count == 0) {
            HCCL_ERROR("[%s], levelRankPairs is empty.", __func__);
            return HcclResult::HCCL_E_INTERNAL;
        }
        dynamicInfo.levelRankPairs.resize(count);
        for (auto& pair : dynamicInfo.levelRankPairs) {
            buf >> pair.first >> pair.second; // 反序列化vector中的每个pair
        }

        size_t linkGroupPairCount{0};
        buf >> linkGroupPairCount;
        HCCL_INFO("[%s], linkGroupPairCount[%u]", __func__, linkGroupPairCount);
        dynamicInfo.linkGroupPair.resize(linkGroupPairCount);
        for (auto& linkGroupPair : dynamicInfo.linkGroupPair) {
            LinkGroup& linkGroup = linkGroupPair.first;
            u32& cntCkeNum = linkGroupPair.second;
            size_t linkSize{0};
            buf >> linkSize;
            for (size_t i = 0; i < linkSize; ++i) {
                RankId rankId;
                u32 dieId;
                buf >> rankId >> dieId;
                IpAddress localAddr(buf);
                IpAddress remoteAddr(buf);
                HCCL_INFO(
                    "[%s], rankId[%d], dieId[%u], localAddr[%s], remoteAddr[%s]", __func__, rankId, dieId,
                    localAddr.Describe().c_str(), remoteAddr.Describe().c_str());
                LinkInfo linkInfo{rankId, dieId, localAddr, remoteAddr};
                linkGroup.AddLink(linkInfo);
            }
            buf >> cntCkeNum;
            HCCL_INFO("[%s], cntCkeNum[%u], linkSize[%u]", __func__, cntCkeNum, linkSize);
        }
    } catch (HcclException& e) {
        HCCL_ERROR(e.what());
        return e.GetErrorCode();
    } catch (exception& e) {
        HCCL_ERROR(e.what());
        return HcclResult::HCCL_E_INTERNAL;
    } catch (...) {
        HCCL_ERROR("Unknown error occurs!");
        return HcclResult::HCCL_E_INTERNAL;
    }
    return HcclResult::HCCL_SUCCESS;
}
/* 全局通信域静态信息序列化 */
void SnapShotParser::SerializeCommonInfo(
    const CommParams& commParams, const HcclCommConfig& config, std::unique_ptr<RankTableInfo> ranktableInfo,
    std::shared_ptr<TopoInfo>& topoInfo, BinaryStream& binStream) const
{
    HCCL_INFO("Snapshot saving: Start to serialize static info.");
    // 1. 配置信息序列化
    SerializeCommConfigInfo(config, binStream);
    // 2. 参数信息序列化
    SerializeParamsInfo(commParams, binStream);
    // 3. rankTableInfo及crc信息序列化
    SerializeRankTableInfo(std::move(ranktableInfo), binStream);
    // 4. topoInfo及crc信息序列化
    SerializeTopoInfo(topoInfo, binStream);
}

void SnapShotParser::SerializeCommVersionInfo(BinaryStream& binStream) const
{
    HCCL_INFO("[%s]Snapshot saving: Start to serialize comm version info.", __func__);
    char snapshotVersion[SNAPSHOT_VERSION_SIZE] = "snapshotVersionIsFixNow";
    char cannVersion[SNAPSHOT_VERSION_SIZE] = "cannVersionIsFixNow";
    char hcclVersion[SNAPSHOT_VERSION_SIZE] = "hcclVersionIsFixNow";
    binStream << snapshotVersion << cannVersion << hcclVersion;
}

void SnapShotParser::SerializeCommConfigInfo(const HcclCommConfig& config, BinaryStream& binStream) const
{
    HCCL_INFO(
        "[%s]Snapshot saving: Start to serial commConfig info, hcclBufferSize[%u] MB", __func__, config.hcclBufferSize);
    binStream << config.reserved << config.hcclBufferSize << config.hcclDeterministic << config.hcclCommName
              << config.hcclUdi;
}

void SnapShotParser::SerializeParamsInfo(const CommParams& commParams, BinaryStream& binStream) const
{
    HCCL_INFO(
        "[%s]Snapshot saving: Start to serialize commParams: commId[%s], myRank[%d], "
        "rankSize[%u],rankInParentComm[%d], devType[%u], devUsed[%u]",
        __func__, commParams.commId.c_str(), commParams.myRank, commParams.rankSize, commParams.rankInParentComm,
        static_cast<u32>(commParams.devType), commParams.devUsed);
    binStream << commParams.commId << commParams.myRank << commParams.rankSize << commParams.rankInParentComm
              << static_cast<u32>(commParams.devType) << commParams.devUsed;
}

void SnapShotParser::SerializeRankTableInfo(std::unique_ptr<RankTableInfo> ranktableInfo, BinaryStream& binStream) const
{
    HCCL_INFO("[%s]Snapshot saving: Start to serialize rankTableInfo.", __func__);
    if (ranktableInfo == nullptr) {
        HCCL_WARNING("ranktableInfo is NULL.");
        return;
    }
    ranktableInfo->GetBinStream(true, binStream);
}

void SnapShotParser::SerializeTopoInfo(const std::shared_ptr<TopoInfo>& topoInfo, BinaryStream& binStream) const
{
    HCCL_INFO("[%s]Snapshot saving: Start to serialize topoInfo.", __func__);
    if (topoInfo == nullptr) {
        HCCL_WARNING("topoInfo is NULL.");
        return;
    }
    topoInfo->GetBinStream(binStream);
}

/* 全局通信域静态信息的反序列化 */
HcclResult SnapShotParser::DeserializeCommInfo(BinaryStream& binaryStream, Snapshot& snapShot)
{
    HCCL_INFO("[%s]Snapshot recovering: Start to deserialize static info.", __func__);
    CHK_RET(DeserializeCommConfigInfo(binaryStream, snapShot.snapShotComm.config));
    CHK_RET(DeserializeParamsInfo(binaryStream, snapShot.snapShotComm.commParams));
    CHK_RET(DeserializeRankTableInfo(binaryStream, snapShot.snapShotComm.rankTableInfo));
    CHK_RET(DeserializeTopoInfo(binaryStream, snapShot.snapShotComm.topoInfo));
    return HCCL_SUCCESS;
}

HcclResult SnapShotParser::DeserializeCommVersionInfo(BinaryStream& binaryStream, SnapShotPub& snapshotPub) const
{
    HCCL_INFO("[%s]Snapshot recovering: Start to deserialize commConfig info.", __func__);
    binaryStream >> snapshotPub.snapshotVersion >> snapshotPub.cannVersion >> snapshotPub.hcclVersion;
    return HCCL_SUCCESS;
}

HcclResult SnapShotParser::DeserializeCommConfigInfo(BinaryStream& binaryStream, HcclCommConfig& config) const
{
    HCCL_INFO("[%s]Snapshot recovering: Start to deserialize commConfig info.", __func__);
    binaryStream >> config.reserved >> config.hcclBufferSize >> config.hcclDeterministic >> config.hcclCommName
        >> config.hcclUdi;
    config.hcclBufferSize = 0;
    HCCL_INFO(
        "Snapshot recovering: hcclBufferSize[%u] MB, hcclDeterministic[%u], hcclCommName[%s], "
        "hcclUdi[%s]",
        config.hcclBufferSize, config.hcclDeterministic, config.hcclCommName, config.hcclUdi);
    return HCCL_SUCCESS;
}

HcclResult SnapShotParser::DeserializeParamsInfo(BinaryStream& binaryStream, Hccl::CommParams& commParams) const
{
    HCCL_INFO("[%s]Snapshot recovering: Start to deserialize params info.", __func__);
    binaryStream >> commParams.commId;
    binaryStream >> commParams.myRank;
    binaryStream >> commParams.rankSize;
    binaryStream >> commParams.rankInParentComm;
    u32 dev = 0;
    binaryStream >> dev;
    commParams.devType = static_cast<DevType::Value>(dev);
    binaryStream >> commParams.devUsed;
    commParams.isWorldGroup = true;
    commParams.commDepth = 0;
    HCCL_INFO(
        "Snapshot recovering: commId[%s], myRank[%d], rankSize[%u],rankInParentComm[%d], devType[%u], devUsed[%u]",
        commParams.commId.c_str(), commParams.myRank, commParams.rankSize, commParams.rankInParentComm,
        static_cast<u32>(commParams.devType), commParams.devUsed);
    return HCCL_SUCCESS;
}

HcclResult SnapShotParser::DeserializeRankTableInfo(BinaryStream& binaryStream, RankTableInfo& rankTableInfo) const
{
    HCCL_INFO("[%s]Snapshot recovering: Start to deserialize rankTable info.", __func__);
    rankTableInfo = RankTableInfo(binaryStream);
    return HCCL_SUCCESS;
}

HcclResult SnapShotParser::DeserializeTopoInfo(BinaryStream& binaryStream, TopoInfo& topoInfo) const
{
    HCCL_INFO("[%s]Snapshot recovering: Start to deserialize topo info.", __func__);
    topoInfo = TopoInfo(binaryStream);
    return HCCL_SUCCESS;
}
/* 子通信域静态信息的序列化 */
void SnapShotParser::SerializeSubCommInfo(
    const CommParams& commParams, const HcclCommConfig& subConfig, const std::vector<u32>& rankId,
    BinaryStream& binStream) const
{
    HCCL_INFO("[%s]Snapshot saving: Start to serialize subComm static info.", __func__);
    SerializeSubCommParamsInfo(commParams, binStream);
    SerializeSubCommConfigInfo(subConfig, binStream);
    SerializeRankIds(rankId, binStream);
}

void SnapShotParser::SerializeSubCommParamsInfo(const CommParams& commParams, BinaryStream& binStream) const
{
    HCCL_INFO(
        "[%s]Snapshot saving: Start to serialize sub commParams: commId[%s], myRank[%d], "
        "rankSize[%u],rankInParentComm[%d], devType[%u], devUsed[%u]",
        __func__, commParams.commId.c_str(), commParams.myRank, commParams.rankSize, commParams.rankInParentComm,
        static_cast<u32>(commParams.devType), commParams.devUsed);
    binStream << commParams.commId << commParams.myRank << commParams.rankSize << commParams.rankInParentComm
              << static_cast<u32>(commParams.devType) << commParams.devUsed;
}

void SnapShotParser::SerializeSubCommConfigInfo(const HcclCommConfig& subConfig, BinaryStream& binStream) const
{
    HCCL_INFO(
        "Snapshot recovering: hcclBufferSize[%u] MB, hcclDeterministic[%u], hcclCommName[%s], hcclUdi[%s]",
        subConfig.hcclBufferSize, subConfig.hcclDeterministic, subConfig.hcclCommName, subConfig.hcclUdi);
    binStream << subConfig.reserved << subConfig.hcclBufferSize << subConfig.hcclDeterministic << subConfig.hcclCommName
              << subConfig.hcclUdi;
}

void SnapShotParser::SerializeRankIds(const std::vector<u32>& rankIds, BinaryStream& binStream) const
{
    HCCL_INFO("[%s]Snapshot saving: Start to serialize rankIds.", __func__);
    binStream << rankIds.size();
    HCCL_INFO("rankIdsSize[%u]", rankIds.size());
    for (auto rankId : rankIds) {
        binStream << rankId;
    }
}

/* 子通信域静态信息的反序列化 */
HcclResult SnapShotParser::DeserializeSubCommInfo(BinaryStream& stream, SubSnapshot& subSnapShot)
{
    // 参数信息反序列化
    CHK_RET(DeserializeSubCommParamsInfo(stream, subSnapShot.snapShotSubComm.commParams));
    // 配置信息反序列化
    CHK_RET(DeserializeSubCommConfigInfo(stream, subSnapShot.snapShotSubComm.config));
    // rankIds反序列化
    CHK_RET(DeserializeRankIds(stream, subSnapShot.snapShotSubComm.rankIds));
    return HCCL_SUCCESS;
}

HcclResult SnapShotParser::DeserializeSubCommConfigInfo(BinaryStream& binaryStream, HcclCommConfig& subConfig) const
{
    HCCL_INFO("[%s]Snapshot recovering: Start to deserialize sub commConfig info.", __func__);
    binaryStream >> subConfig.reserved >> subConfig.hcclBufferSize >> subConfig.hcclDeterministic
        >> subConfig.hcclCommName >> subConfig.hcclUdi;
    HCCL_INFO(
        "Snapshot recovering: hcclReserved[%s], hcclBufferSize[%u] MB, hcclDeterministic[%u], hcclCommName[%s], "
        "hcclUdi[%s]",
        subConfig.reserved, subConfig.hcclBufferSize, subConfig.hcclDeterministic, subConfig.hcclCommName,
        subConfig.hcclUdi);
    return HCCL_SUCCESS;
}

HcclResult
SnapShotParser::DeserializeSubCommParamsInfo(BinaryStream& binaryStream, Hccl::CommParams& subCommParam) const
{
    u32 dev = 0;
    HCCL_INFO("[%s]Snapshot recovering: Start to deserialize sub params info.", __func__);
    binaryStream >> subCommParam.commId >> subCommParam.myRank >> subCommParam.rankSize >> subCommParam.rankInParentComm
        >> dev >> subCommParam.devUsed;
    subCommParam.devType = static_cast<DevType::Value>(dev);
    subCommParam.commDepth = 1;
    HCCL_INFO(
        "Snapshot recovering: commId[%s], myRank[%d], rankSize[%u],rankInParentComm[%d], devType[%u], devUsed[%u]",
        subCommParam.commId.c_str(), subCommParam.myRank, subCommParam.rankSize, subCommParam.rankInParentComm,
        static_cast<u32>(subCommParam.devType), subCommParam.devUsed);
    return HCCL_SUCCESS;
}

HcclResult SnapShotParser::DeserializeRankIds(BinaryStream& binaryStream, vector<RankId>& rankIds) const
{
    HCCL_INFO("[%s]Snapshot recovering: Start to deserialize rankIds.", __func__);
    size_t rankIdsSize;
    binaryStream >> rankIdsSize;
    HCCL_INFO("rankIdsSize[%u]", rankIdsSize);
    rankIds.resize(rankIdsSize);
    for (auto& id : rankIds) {
        binaryStream >> id;
    }
    return HCCL_SUCCESS;
}

HcclResult SnapShotParser::CalcBufCrc32(BinaryStream& buf, u32& crcValue) const
{
    CheckCrc crc;
    auto ret = crc.Calc32Crc(buf.GetString().c_str(), buf.GetSize(), &crcValue);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[CalcCrc32]calculate crc failed, ret[%d]", ret), ret);
    return HCCL_SUCCESS;
}

HcclResult SnapShotParser::CheckBufCrc32(BinaryStream& buf, const u32 otherCrcValue) const
{
    CheckCrc crc;
    u32 myCrcValue = 0;
    auto ret = crc.Calc32Crc(buf.GetString().c_str(), buf.GetSize(), &myCrcValue);
    HCCL_INFO("[CheckCrc32]myCrcValue[%u] otherCrcValue[%u]", myCrcValue, otherCrcValue);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[CheckCrc32]calculate crc failed, ret[%d]", ret), ret);
    CHK_PRT_RET(myCrcValue != otherCrcValue, HCCL_ERROR("[CalcCrc32]check crc failed, ret[%d]", ret), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

void SnapShotParser::SetIsNeedLoadOp(bool status) { isNeedLoadOp = status; }
bool SnapShotParser::GetIsNeedLoadOp() const { return isNeedLoadOp; }
