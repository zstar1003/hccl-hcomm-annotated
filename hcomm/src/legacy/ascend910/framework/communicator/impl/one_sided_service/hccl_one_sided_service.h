/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_HCCL_ONE_SIDED_SERVICE_H
#define HCOMM_HCCL_ONE_SIDED_SERVICE_H

#include <array>
#include <set>
#include <shared_mutex>

#include "i_hccl_one_sided_service.h"
#include "hccl_one_sided_conn.h"
#include "hccl_common.h"
#include "common.h"
#include "externalinput_pub.h"
#include "hccl_mem.h"
#include "global_mem_record.h"
#include "aicpu_operator_pub.h"
#include "comm_config_pub.h"

using HcclBatchData = struct HcclBatchDataDef {
    HcclComm comm;
    HcclCMDType cmdType;
    u32 remoteRank;
    HcclOneSideOpDesc* desc;
    u32 descNum;
    rtStream_t stream;
};

namespace hccl {
struct AicpuOneSideCommTiling {
    HcclCMDType cmdType;
    std::string tag;
    rtStream_t stream;
    u8 floatOverflowMode = ACL_RT_OVERFLOW_MODE_UNDEF;
    u8 dumpDebug = false;
    bool useRdma = false;
};

enum class AicpuLocalNotify : u32 { HOST_TO_AICPU_POST = 0, HOST_TO_AICPU_WAIT, LOCAL_NOTIFY_NUM };
constexpr u32 LOCAL_NOTIFY_NUM = static_cast<u32>(AicpuLocalNotify::LOCAL_NOTIFY_NUM);
static_assert(LOCAL_NOTIFY_NUM == AICPU_OP_NOTIFY_MAX_NUM, "AICPU notify max count not match");

constexpr size_t HCCL_MEM_DESC_STR_LEN = HCCL_MEM_DESC_LENGTH + 1 - (sizeof(u32) * 2);

class HcclOneSidedService : public IHcclOneSidedService {
public:
    using RankId = u32;
    using ProcessInfo = HcclOneSidedConn::ProcessInfo;

    struct HcclMemDescData {
        u32 localRankId;
        u32 remoteRankId;
        char memDesc[HCCL_MEM_DESC_STR_LEN];
    };

    HcclOneSidedService(
        std::unique_ptr<HcclSocketManager>& socketManager, std::unique_ptr<NotifyPool>& notifyPool,
        const CommConfig& commConfig);

    // 父类Config()等已经完成必要参数的配置
    HcclOneSidedService() = default;
    ~HcclOneSidedService() override;

    HcclResult ReMapMem(HcclMem* memInfoArray, u64 arraySize);
    HcclResult RegMem(void* addr, u64 size, HcclMemType type, RankId remoteRankId, HcclMemDesc& localMemDesc);
    HcclResult DeregMem(const HcclMemDesc& localMemDesc);
    // 可能返回超时
    HcclResult ExchangeMemDesc(
        RankId remoteRankId, const HcclMemDescs& localMemDescs, HcclMemDescs& remoteMemDescs, u32& actualNumOfRemote,
        const std::string& commIdentifier, s32 timeoutSec);

    void EnableMemAccess(const HcclMemDesc& remoteMemDesc, HcclMem& remoteMem);
    void DisableMemAccess(const HcclMemDesc& remoteMemDesc);

    void BatchPut(RankId remoteRankId, const HcclOneSideOpDesc* desc, u32 descNum, const rtStream_t& stream);
    void BatchGet(RankId remoteRankId, const HcclOneSideOpDesc* desc, u32 descNum, const rtStream_t& stream);

    HcclResult GetIsUsedRdma(RankId remoteRankId, bool& useRdma);

    // 主要完成通信域粒度的数据面建链和已绑定MR的交换、使能
    HcclResult Prepare(const std::string& commIdentifier, const HcclPrepareConfig* prepareConfig, s32 timeoutSec);
    HcclResult InitIsUsedRdmaMap(bool& needInitNic, bool& needInitVnic);
    HcclResult DeInit() override;

    HcclResult BindMem(void* memRecordHandle, const std::string& commIdentifier);   // 绑定一块全局内存
    HcclResult UnbindMem(void* memRecordHandle, const std::string& commIdentifier); // 解绑一块全局内存

    inline bool HasBoundMem() const // 判断有没有绑定着的内存
    {
        return !boundMemPtrSet_.empty();
    }

private:
    u32 registedMemCnt_{0};
    HcclResult IsUsedRdma(RankId remoteRankId, bool& useRdma);

    HcclResult SetupRemoteRankInfo(RankId remoteRankId, HcclRankLinkInfo& remoteRankInfo);
    HcclResult CreateConnection(
        RankId remoteRankId, const HcclRankLinkInfo& remoteRankInfo, std::shared_ptr<HcclOneSidedConn>& tempConn);
    HcclResult Grant(const HcclMemDesc& localMemDesc, const ProcessInfo& remoteProcess);
    HcclBuf* GetHcclBufByDesc(std::string& descStr, bool useRdma);

    // Prepare新增函数
    void ConnectByThread(
        std::shared_ptr<HcclOneSidedConn>& conn, const std::string& commIdentifier, s32 timeoutSec, HcclResult& retOut);
    HcclResult CreateLinkFullmesh(const std::string& commIdentifier, s32 timeoutSec);
    HcclResult RegBoundMem(HcclNetDevCtx netDevCtx, const HcclMem& localMem, HcclMemDesc& localMemDesc, HcclBuf& buf);
    HcclResult RegisterBoundMems();
    HcclResult ExchangeMemDescFullMesh();
    HcclResult ExchangeMemDescByThread(std::shared_ptr<HcclOneSidedConn>& conn, bool isUseRdma);
    HcclResult EnableMemAccessByThread();
    HcclResult EnableMemAccess();
    HcclResult DisableMemAccess();
    HcclResult Grant(HcclBuf& buf);
    HcclResult PrepareFullMesh(const std::string& commIdentifier, s32 timeoutSec);
    HcclResult RunFuncWithTimeout(
        std::function<HcclResult()> func, const std::string& commIdentifier, s32 timeoutSec, std::string functionName);

    static HcclResult CreateLaunchStream();
    HcclResult InitAicpuUnfoldMode();
    HcclResult LoadAICPUKernel(void);
    void UnloadAICPUKernel(void);
    HcclResult AicpuResourceInit();
    HcclResult ReportProfilingCommInfo(const Stream& kfcStream, const Stream& aicpuStream);
    HcclResult AicpuInitKernelLaunch();
    HcclResult CreateAicpuNotify(std::shared_ptr<LocalNotify>& localNotify, HcclSignalInfo& notifyInfo);
    HcclResult OrchestrateAicpu(
        RankId remoteRankId, HcclCMDType cmdType, const std::shared_ptr<HcclOneSidedConn>& conn,
        const HcclOneSideOpDesc* desc, u32 descNum, rtStream_t stream);
    u64 CalcTilingDynamicDataSize(HcclCMDType cmdType, u32 descNum);
    HcclResult InitAicpuTilingDataBuf(
        const AicpuOneSideCommTiling& tilingInfo, u32 remoteRankId, const std::shared_ptr<HcclOneSidedConn>& conn,
        const HcclOneSideOpDesc* desc, u32 descNum, u64 dynamicDataSize);
    HcclResult AicpuKernelLaunch(
        const std::shared_ptr<HcclOneSidedConn>& conn, const std::string& kernelName,
        const AicpuOneSideCommTiling& tilingInfo, u64 tilingDataSize);
    HcclResult AicpuUnfoldKernelLaunchV2(
        const std::string& kernelName, void* tilingDataPtr, u64 tilingDataSize, const rtStream_t stream);

    std::unordered_map<RankId, std::shared_ptr<HcclOneSidedConn>> oneSidedConns_{};
    mutable std::shared_mutex oneSidedConnsMutex_;

    std::unordered_map<RankId, bool> isUsedRdmaMap_;
    std::unordered_map<std::string, HcclBuf> desc2HcclBufMapIpc_{};
    std::unordered_map<std::string, HcclBuf> desc2HcclBufMapRoce_{};

    std::set<GlobalMemRecord*> boundMemPtrSet_{};
    std::shared_mutex boundMemPtrSetMtx_;
    s32 deviceLogicId_{HOST_DEVICE_ID};
    std::vector<HcclMemDesc> localMemIpcDescs_;
    std::vector<HcclMemDesc> localMemRoceDescs_;

    bool prepared_{false};                         // 表示是否prepare过
    std::atomic<bool> hasErrorFlag_{false};        // 用于表示多线程操作是否出错
    std::atomic<bool> hasTimeoutErrorFlag_{false}; // 用于表示多线程操作是否超时出错
    bool needRegRoceMem_{false};                   // 是否需要注册roce内存
    bool needRegIpcMem_{false};                    // 是否需要注册ipc内存
    ProcessInfo localProcess_{};
    static std::mutex regMutex_;
    std::map<u32, std::vector<HcclMemDesc>> localMemDescs_{};
    std::mutex descMtx_;

    bool isAicpuModeInited_{false};
    bool aicpuUnfoldMode_{false};

    bool isContextLaunched_{false};
    std::array<std::shared_ptr<LocalNotify>, LOCAL_NOTIFY_NUM> localAicpuNotify_;

    Stream execStream_;

    DeviceMem initApiDeviceMem_;

    HostMem tilingDataMem_;
    HostMem apiTilingDataMem_;

    HcclOneSideCommResParam commResPara_;
    DeviceMem commResParaDevice_;
    DeviceMem execStreamContext_;
    aclrtBinHandle binHandle_ = nullptr;
    CommConfig commConfig_;
};
} // namespace hccl

#endif
