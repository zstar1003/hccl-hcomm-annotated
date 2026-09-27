/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_TASKEXCEPTION_H
#define CCU_TASKEXCEPTION_H

#include <array>
#include "global_mirror_tasks.h"
#include "ccu_error_info.h"
#include "rank_pair.h"
#include "ccu_error_info_v1.h"
#include "ccu_error_info_v2.h"
#include "ccu_rep_base_v1.h"
#include "ccu_rep_context_v1.h"
#include "ccu_device_pub.h"
#include "ccu_jetty_.h"
#include "coll_comm.h"

namespace hcomm {
class CcuUrmaChannel;
using RdmaHandle = void*;

using GetCcuCqeErrInfoCallBackHcomm = void (*)(
    u32 RemoteLocalId, u32 locDeviceId, uint16_t status, std::string LocalEid, std::string RemoteEid,
    std::string RemoteInsId);                                                 // 获取远端rankId的回调函数类型
void RegisterGetCcuCqeErrInfoCallBackHcomm(GetCcuCqeErrInfoCallBackHcomm p1); // 注册获取远端rankId的回调函数

using CcuGetErrStatusVecCallBack = std::vector<std::string> (*)(s32 deviceLogicID);
void RegisterCcuGetErrStatusVecCallBack(CcuGetErrStatusVecCallBack callback);

class CcuTaskException {
public:
    CcuTaskException() = default;
    ~CcuTaskException() = default;
    static void ProcessCcuException(const rtExceptionInfo_t* exceptionInfo, const Hccl::TaskInfo& taskInfo);

private:
    static HcclResult InitChannelMap(s32 deviceId, u64 ccuKernelHandle);
    static std::string GetGroupRankInfo(const Hccl::TaskInfo& taskInfo);

    static HcclResult PrintUbRegisters(s32 devLogicId, RdmaHandle rdmaHandle);
    static HcclResult
    PrintCcuUbRegisters(const std::vector<CcuErrorInfo>& errorInfos, s32 devLogicId, const Hccl::TaskInfo& taskInfo);
    static HcclResult
    GetCcuJettys(const CcuErrorInfo& errorInfo, std::pair<CcuChannelInfo, std::vector<CcuJetty*>>& ctx);
    static uint16_t GetChannleIdByCcuErrorInfo(const CcuErrorInfo& errorInfo);

    static void PrintCcuErrorInfo(uint32_t deviceId, uint16_t status, const Hccl::TaskInfo& taskInfo);
    static void
    PrintCcuErrorLog(const std::vector<CcuErrorInfo>& errorInfos, const Hccl::TaskInfo& taskInfo, u32 deviceId);

    // 获取信息的公用接口
    static HcclResult GetChannelIdByHandle(const ChannelHandle& channel, uint32_t& channelId);
    static HcclResult
    GetSignalIdByHandle(const ChannelHandle& channel, uint16_t semIdx, bool isRmtSig, uint32_t& signalId);
    static HcclResult GetVariableIdByHandle(const ChannelHandle& channel, uint16_t varIdx, uint32_t& varId);
    static HcclResult GetCcuUrmaChannel(const ChannelHandle& channel, CcuUrmaChannel*& channelImpl);

    // 获取ErrorMsg
    static std::string
    GetCcuErrorMsgByType(const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId);
    static std::string
    GetCcuErrorMsgLoop(const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId);
    static std::string GetCcuErrorMsgMission(const CcuErrorInfo& ccuErrorInfo);
    static std::string GetCcuErrorMsgDefault(const CcuErrorInfo& ccuErrorInfo);
    static std::string
    GetCcuErrorMsgLoopGroup(const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId);
    static std::string
    GetCcuErrorMsgLocPostSem(const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId);
    static std::string
    GetCcuErrorMsgLocWaitEvent(const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId);
    static std::string
    GetCcuErrorMsgLocWaitNotify(const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId);
    static std::string
    GetCcuErrorMsgRemPostSem(const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId);
    static std::string
    GetCcuErrorMsgRemWaitSem(const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId);
    static std::string
    GetCcuErrorMsgRemPostVar(const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId);
    static std::string
    GetCcuErrorMsgPostSharedSem(const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId);
    static std::string
    GetCcuErrorMsgRead(const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId);
    static std::string
    GetCcuErrorMsgWrite(const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId);
    static std::string
    GetCcuErrorMsgLocalCpy(const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId);
    static std::string
    GetCcuErrorMsgLocalReduce(const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId);
    static std::string
    GetCcuErrorMsgBufRead(const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId);
    static std::string
    GetCcuErrorMsgBufWrite(const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId);
    static std::string
    GetCcuErrorMsgBufLocRead(const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId);
    static std::string
    GetCcuErrorMsgBufLocWrite(const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId);
    static std::string
    GetCcuErrorMsgBufReduce(const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId);
    static std::string
    GetCcuErrorMsgWriteVarAtomic(const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId);
    static std::string
    GetCcuErrorMsgWriteWithCntInc(const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId);
    static std::string
    GetCcuErrorMsgCascCntWait(const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId);
    static std::string
    GetCcuErrorMsgCascCntClear(const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId);
    static std::string
    GetCcuErrorMsgLoadAddImm(const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId);
    static std::string
    GetCcuErrorMsgStoreAddImm(const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId);

    static HcclResult GetCcuChannelHandleById(u16 channelId, u64& channelHandle);
    static RankId GetRankIdByChannelId(uint16_t channelId, const Hccl::TaskInfo& taskInfo, u32 deviceId);
    static std::pair<Hccl::IpAddress, Hccl::IpAddress>
    GetAddrPairByChannelId(uint16_t channelId, const Hccl::TaskInfo& taskInfo, u32 deviceId);
    static std::string GetCcuLenErrorMsg(const uint64_t len);
    static HcclResult GenErrorInfoLoopGroup(
        const ErrorInfoBase& baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase, CcuRep::CcuRepContext& ctx,
        std::vector<CcuErrorInfo>& errorInfo);

    // 生成Error Info
    static void GenErrorInfoByRepType(
        const ErrorInfoBase& baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
        std::vector<CcuErrorInfo>& errorInfo);
    static void GenErrorInfoLocRecordEvent(
        const ErrorInfoBase& baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
        std::vector<CcuErrorInfo>& errorInfo);
    static void GenErrorInfoLocWaitNotify(
        const ErrorInfoBase& baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
        std::vector<CcuErrorInfo>& errorInfo);
    static void GenErrorInfoLocWaitEvent(
        const ErrorInfoBase& baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
        std::vector<CcuErrorInfo>& errorInfo);
    static void GenErrorInfoRemPostSem(
        const ErrorInfoBase& baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
        std::vector<CcuErrorInfo>& errorInfo);
    static void GenErrorInfoRemWaitSem(
        const ErrorInfoBase& baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
        std::vector<CcuErrorInfo>& errorInfo);
    static void GenErrorInfoRemPostVar(
        const ErrorInfoBase& baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
        std::vector<CcuErrorInfo>& errorInfo);
    static void GenErrorInfoPostSharedSem(
        const ErrorInfoBase& baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
        std::vector<CcuErrorInfo>& errorInfo);
    static void GenErrorInfoRead(
        const ErrorInfoBase& baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
        std::vector<CcuErrorInfo>& errorInfo);
    static void GenErrorInfoWrite(
        const ErrorInfoBase& baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
        std::vector<CcuErrorInfo>& errorInfo);
    static void GenErrorInfoLocalCpy(
        const ErrorInfoBase& baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
        std::vector<CcuErrorInfo>& errorInfo);
    static void GenErrorInfoLocalReduce(
        const ErrorInfoBase& baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
        std::vector<CcuErrorInfo>& errorInfo);
    static void GenErrorInfoBufRead(
        const ErrorInfoBase& baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
        std::vector<CcuErrorInfo>& errorInfo);
    static void GenErrorInfoBufWrite(
        const ErrorInfoBase& baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
        std::vector<CcuErrorInfo>& errorInfo);
    static void GenErrorInfoBufLocRead(
        const ErrorInfoBase& baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
        std::vector<CcuErrorInfo>& errorInfo);
    static void GenErrorInfoBufLocWrite(
        const ErrorInfoBase& baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
        std::vector<CcuErrorInfo>& errorInfo);
    static void GenErrorInfoBufReduce(
        const ErrorInfoBase& baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
        std::vector<CcuErrorInfo>& errorInfo);
    static void GenErrorInfoDefault(
        const ErrorInfoBase& baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
        std::vector<CcuErrorInfo>& errorInfo);
    static void GenErrorInfoWriteVarAtomic(
        const ErrorInfoBase& baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
        std::vector<CcuErrorInfo>& errorInfo);
    static void GenErrorInfoWriteWithCntInc(
        const ErrorInfoBase& baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
        std::vector<CcuErrorInfo>& errorInfo);
    static void GenErrorInfoCascCntWait(
        const ErrorInfoBase& baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
        std::vector<CcuErrorInfo>& errorInfo);
    static void GenErrorInfoCascCntClear(
        const ErrorInfoBase& baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
        std::vector<CcuErrorInfo>& errorInfo);
    static void GenErrorInfoLoadAddImm(
        const ErrorInfoBase& baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
        std::vector<CcuErrorInfo>& errorInfo);
    static void GenErrorInfoStoreAddImm(
        const ErrorInfoBase& baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
        std::vector<CcuErrorInfo>& errorInfo);

    static uint64_t GetCcuXnValue(int32_t deviceId, uint32_t dieId, uint32_t xnId);
    static HcclResult
    GenErrorInfoLoop(const ErrorInfoBase& baseInfo, CcuRep::CcuRepContext& ctx, std::vector<CcuErrorInfo>& errorInfo);
    static void GenStatusInfo(const ErrorInfoBase& baseInfo, std::vector<CcuErrorInfo>& errorInfo);
    static HcclResult GetCcuMissionContextRaw(
        int32_t deviceId, uint32_t dieId, uint32_t missionId, uint8_t* buf, size_t bufLen, size_t& copiedLen);
    static HcclResult GetCcuLoopContextRaw(
        int32_t deviceId, uint32_t dieId, uint32_t loopCtxId, uint8_t* buf, size_t bufLen, size_t& copiedLen);
    static HcclResult GetCcuErrorMsg(
        int32_t deviceId, uint16_t missionStatus, const Hccl::ParaCcu& ccuTaskParam,
        std::vector<CcuErrorInfo>& errorInfo);
    static void PrintPanicLogInfo(const uint8_t* panicLog);
    static uint16_t GetCcuCKEValue(int32_t deviceId, uint32_t dieId, uint32_t ckeId);

    static uint64_t GetCcuGSAValue(int32_t deviceId, uint32_t dieId, uint32_t gsaId);
    static uint16_t GetMSIdPerDie(uint16_t msId) { return msId & 0x7fff; }
    static void GetCcuCqeErrorInfo(
        const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 locDeviceId, uint8_t missionStatus);
    static void GetCcuCqeErrRemoteLocalIdByRankId(hccl::CollComm* collComm, uint32_t rankid, u32& remoteLocalId);
    static void GetCcuCqeErrNetInstanceByRankId(hccl::CollComm* collComm, uint32_t rankid, std::string& netInstanceId);
    static void ClusterMoniterGetCcuCqeErrInfo(
        u32 RemoteDeviceId, u32 locDeviceId, uint16_t status, std::string LocalEid, std::string RemoteEid,
        std::string RemoteInsId);
    static void NotifyControlPlaneOnUbError(
        const std::vector<CcuErrorInfo>& errorInfos, const Hccl::TaskInfo& taskInfo, u32 deviceId,
        uint8_t missionStatus);
};
} // namespace hcomm

#endif
