/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LINK_DIRECT_NPU_PUB_H
#define LINK_DIRECT_NPU_PUB_H

#include <functional>

#include "../host/transport_net_pub.h"
#include "network/hccp_common.h"
#include "workflow_pub.h"
#include "private_types.h"
#include "adapter_hccp.h"
#include "universal_concurrent_map.h"
#include "transport_ibverbs_pub.h"

namespace hccl {

constexpr s64 AICPU_FLAG_AREA = 24; // aicpu需要的flag区大小

// BatchSendRecv建链数超过1k时，notify资源不足时使用；正常情况尽量避免使用
class TransportDirectNpu : public TransportNet {
public:
    explicit TransportDirectNpu(
        DispatcherPub* dispatcher, const std::unique_ptr<NotifyPool>& notifyPool, MachinePara& machinePara,
        std::chrono::milliseconds timeout);
    ~TransportDirectNpu() override;

    HcclResult Init() override;

    HcclResult DeInit() override;

    HcclResult TxAsync(UserMemType dstMemType, u64 dstOffset, const void* src, u64 len, Stream& stream) override;
    HcclResult TxAsync(std::vector<TxMemoryInfo>& txMems, Stream& stream) override;

    HcclResult RxAsync(UserMemType srcMemType, u64 srcOffset, void* dst, u64 len, Stream& stream) override;
    HcclResult RxAsync(std::vector<RxMemoryInfo>& rxMems, Stream& stream) override;

    HcclResult DataReceivedAck(Stream& stream) override;

    HcclResult TxAck(Stream& stream) override;
    HcclResult RxAck(Stream& stream) override;

    HcclResult TxDataSignal(Stream& stream) override;
    HcclResult RxDataSignal(Stream& stream) override;

    HcclResult CreateAicpuMem();
    HcclResult DestroyAicpuMem();

    HcclResult TxWaitDone(Stream& stream) override;

    HcclResult TxPrepare(Stream& stream) override;
    HcclResult RxPrepare(Stream& stream) override;

    HcclResult TxData(UserMemType dstMemType, u64 dstOffset, const void* src, u64 len, Stream& stream) override;
    HcclResult RxData(UserMemType srcMemType, u64 srcOffset, void* dst, u64 len, Stream& stream) override;

    HcclResult TxDone(Stream& stream) override;
    HcclResult RxDone(Stream& stream) override;

    HcclResult PostFin(Stream& stream) override;
    HcclResult WaitFin(Stream& stream) override;

    HcclResult PostFinAck(Stream& stream) override;
    HcclResult WaitFinAck(Stream& stream) override;

    HcclResult GetTransportErrorCqe(
        const HcclNetDevCtx netDevCtx, std::vector<std::pair<TransportBase*, CqeInfo>>& infos, u32& num);
    HcclIpAddress& GetRemoteIp();
    HcclResult GetTransportId(u32& id) override;
    HcclResult GetRemoteMem(UserMemType memType, void** remotePtr) override;
    HcclResult GetRemoteMemSize(UserMemType memType, u64& size) override;
    HcclResult GetLocalNotifyValueAddrKey(std::vector<AddrKey>& notifyValue) override;
    HcclResult GetLocalRdmaNotify(std::vector<HcclSignalInfo>& rdmaNotify) override;
    HcclResult GetRemoteRdmaNotifyAddrKey(std::vector<AddrKey>& rdmaNotifyAddr) override;

protected:
    HcclResult GetRemoteAddr(MemType memType, u8*& exchangeDataPtr, u64& exchangeDataBlankSize);
    HcclResult FillExchangeDataTotalSize() override;
    HcclResult ConstructExchangeForSend() override;
    HcclResult ParseReceivedExchangeData() override;

    HcclResult DestroyQpVct(std::vector<QpHandle>& qpHandles);
    HcclResult DestroyQP();
    HcclResult DeRegMR();
    void DeRegMRForQPhandles(MemMsg& memMsg);
    HcclResult DeRegOneMR(QpHandle& qpHandle, MemMsg& memMsg);
    // 初始化
    s32 GetQpMode();
    HcclResult CreateQp();                 // 创建QP
    HcclResult CreateSingleQp(s32 qpMode); // 两个rank之间仅创建一个QP
    HcclResult ConnectSingleQp(std::function<bool()> needStop = []() {
        return false;
    });
    HcclResult ConnectQp();
    HcclResult InitQpConnect();
    HcclResult GetQpAttr();

    HcclResult IsUseQpCreateWithAttrs(bool& isUseQpCreateWithAttrs, s32 qpMode);
    HcclResult GetNicHandle();
    u32 GetQpsPerConnection();
    virtual HcclResult RegUserMem(MemType memType, u8*& exchangeDataPtr, u64& exchangeDataBlankSize);
    HcclResult CreateOneQp(
        s32 qpMode, u32 qpsPerConnection, QpHandle& qpHandle, AiQpInfo& aiQpInfo, bool useAicpu = false,
        u32 udpSport = 0);
    virtual HcclResult GetMemInfo(UserMemType memType, void** dstMemPtr, u64* dstMemSize);

    HcclResult GetRemoteMemKey(UserMemType memType, uint32_t* remoteMemKey) override;
    HcclResult GetLocalMemDetails(UserMemType memType, MemDetails& memDetails) override;
    HcclResult GetAiQpInfo(std::vector<HcclQpInfoV2>& aiQpInfo) override;

    std::vector<QpHandle> qpHandles_;
    struct AiQpInfo aiQpInfo_ = {};
    std::vector<AiQpInfo> aiQpInfos_;
    u32 qpsPerConnection_;

    s32 access_;

    std::array<MemMsg, static_cast<u32>(MemType::MEM_TYPE_RESERVED)> memMsg_;
    std::array<MemMsg, static_cast<u32>(MemType::MEM_TYPE_RESERVED)> remoteMemMsg_;

    HcclWorkflowMode workFlowMode_; // 工作模式
    u32 currentQP_;
    RdmaHandle nicRdmaHandle_{nullptr};
    DevType localDeviceType{DevType::DEV_TYPE_COUNT};
    QPMode qpMode_{QPMode::INVALID}; // 是否为普通QP模式

private:
    HcclResult LoadAICPUKernel(void);
    void UnloadAICPUKernel(void);
    HcclResult LoadBinaryFromFile(
        const char* binPath, aclrtBinaryLoadOptionType optionType, uint32_t cpuKernelMode, aclrtBinHandle& binHandle);
    static void ProcessCqeInfo(
        const s32 deviceId, const struct CqeErrInfo* infolist, const u32 cqeNum,
        std::vector<std::pair<TransportBase*, CqeInfo>>& infos);
    // bit[63:32] devicePhyId, bit[31:0] qpn
    static UniversalConcurrentMap<u64, TransportDirectNpu*> g_qpn2IbversLinkMap_;
    static bool g_flag;
    static bool g_isSupCqeErrInfoListConfig;
    static u32 cqeErrQpn_;
    aclrtBinHandle binHandle_ = nullptr;
    DeviceMem aicpuMem_;
};
} // namespace hccl

#endif /* LINK_IBV_EXP_PUB_H */
