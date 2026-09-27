/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "transport_remote_access.h"
#include "externalinput_pub.h"

namespace hccl {
using namespace std;
std::array<DeviceMem, MAX_MODULE_DEVICE_NUM> TransportRemoteAccess::notifyValueMem_;
std::array<std::mutex, MAX_MODULE_DEVICE_NUM> TransportRemoteAccess::notifyValueMutex_;
std::array<Referenced, MAX_MODULE_DEVICE_NUM> TransportRemoteAccess::instanceRef_; // 实例计数，用于释放静态资源
TransportRemoteAccess::TransportRemoteAccess(
    const std::string tag, const HcclDispatcher dispatcher, const std::unique_ptr<NotifyPool>& notifyPool,
    const RemoteAccessPara& remoteAccessPara, const std::vector<MemRegisterAddr>& memRegistInfos, s32 deviceLogicId)
    : dispatcher_(dispatcher),
      notifyPool_(notifyPool),
      MemRegistInfos_(memRegistInfos),
      RemoteAccessPara_(remoteAccessPara),
      handle_(nullptr),
      ackNotify_(nullptr),
      access_(RA_ACCESS_LOCAL_WRITE | RA_ACCESS_REMOTE_WRITE | RA_ACCESS_REMOTE_READ),
      notifySize_(NOTIFY_BUFFER_SIZE),
      tag_(tag),
      timeout_(HCCL_LINK_TIME_OUT_S),
      deviceLogicId_(deviceLogicId)
{
    instanceRef_[deviceLogicId_].Ref();
}

TransportRemoteAccess::~TransportRemoteAccess()
{
    HCCL_DEBUG("~TransportRemoteAccess Enter!");
    HcclResult ret = HCCL_SUCCESS;
    struct MrInfoT mrInfo = {};
    /* 销毁本端mr */
    for (u32 idx = 0; idx < localRegMem_.size(); idx++) {
        mrInfo.addr = localRegMem_[idx];
        ret = HrtRaMrDereg(handle_, &mrInfo);
        if (ret != HCCL_SUCCESS) {
            HCCL_WARNING(
                "errNo[0x%016llx] in TransportRemoteAccess deconstruct, mr dereg failed. ", HCCL_ERROR_CODE(ret));
        }
    }

    ackNotify_ = nullptr;
    if (handle_ != nullptr) {
        ret = HrtRaQpDestroy(handle_);
        if (ret != HCCL_SUCCESS) {
            HCCL_WARNING(
                "errNo[0x%016llx] in TransportRemoteAccess deconstruct, qp destroy failed. ", HCCL_ERROR_CODE(ret));
        }
    }
    if (instanceRef_[deviceLogicId_].Unref() == 0) {
        std::unique_lock<std::mutex> lock(notifyValueMutex_[deviceLogicId_]);
        notifyValueMem_[deviceLogicId_].free();
    }
    HCCL_DEBUG("~TransportRemoteAccess Success!");
}
HcclResult TransportRemoteAccess::Init()
{
    HCCL_DEBUG("TransportRemoteAccess Init start");
    // 创建QP操作句柄
    CHK_RET(CreateQp());
    // 本端host/device内存地址注册
    CHK_RET(MrRegister());
    // notify注册，用于rdma消息同步
    CHK_RET(NotifyRegister());
    CHK_RET(ConnectQp());

    HCCL_DEBUG("TransportRemoteAccess Init end");
    return HCCL_SUCCESS;
}

HcclResult TransportRemoteAccess::CreateQp()
{
    // 创建qp handle, mode:普通qp
    HcclResult ret = HrtRaQpCreate(RemoteAccessPara_.nicRdmaHandle, QP_FLAG_RC, NORMAL_QP_MODE, handle_);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Create][Qp]create qp mode failed, handle is null, "
            "localRank[%u], qpMode[%d]",
            RemoteAccessPara_.localRank, NORMAL_QP_MODE),
        HCCL_E_ROCE_CONNECT);
    CHK_RET(SetQpAttrQos(handle_));
    // 配置RDMA Timeout时间
    CHK_RET(SetQpAttrTimeOut(handle_));
    // 配置RDMA Retry Cnt重传次数
    CHK_RET(SetQpAttrRetryCnt(handle_));
    return HCCL_SUCCESS;
}

HcclResult TransportRemoteAccess::NotifyRegister()
{
    // notify内存注册
    CHK_RET(SetLocalNotify());
    // 获取对端notify
    CHK_RET(GetRemoteNotifyInfo());
    return HCCL_SUCCESS;
}

HcclResult TransportRemoteAccess::MrRegister()
{
    void* memPtr = nullptr;
    if (MemRegistInfos_.size() == 0) {
        HCCL_ERROR("[Register][Mr]local mem info to register is empty!");
        return HCCL_E_PARA;
    }
    struct MrInfoT mrInfo = {};
    mrInfo.access = access_;
    for (size_t idx = 0; idx < MemRegistInfos_.size(); idx++) {
        memPtr = reinterpret_cast<void*>(static_cast<uintptr_t>(MemRegistInfos_[idx].addr));
        mrInfo.addr = memPtr;
        mrInfo.size = MemRegistInfos_[idx].length;

        CHK_RET(HrtRaMrReg(handle_, &mrInfo));
        localRegMem_.push_back(memPtr);
    }
    return HCCL_SUCCESS;
}

HcclResult TransportRemoteAccess::SetLocalNotify()
{
    // 申请ack notify，并发送至对端
    CHK_RET(CreateNotify());
    // 注册notify 内存信息
    CHK_RET(CreateNotifyValueBuffer());
    return HCCL_SUCCESS;
}

HcclResult TransportRemoteAccess::CreateNotify()
{
    u64 offset = 0;
    u64 notifyBaseVa = 0; // notify寄存器虚拟地址
    u64 notifyTotalSize = 0;

    /* 申请Notify Group ID */
    RemoteRankInfo info(deviceLogicId_, RemoteAccessPara_.remoteRank);
    CHK_RET(SalGetBareTgid(&info.remotePid)); // 当前进程id
    CHK_RET(notifyPool_->Alloc(tag_, info, ackNotify_));
    // 设置remote id
    s64 recvId = 0xFFFFFFFF00000000 | (static_cast<s64>(info.remotePid) & 0xFFFFFFFF);
    CHK_RET(ackNotify_->Grant(recvId));

    /* 获取notify寄存器虚拟基地址、大小, 物理地址回传值为空 */
    CHK_RET(HrtRaGetNotifyBaseAddr(RemoteAccessPara_.nicRdmaHandle, &notifyBaseVa, &notifyTotalSize));

    /* 获取notify虚拟地址 */
    CHK_RET(ackNotify_->GetNotifyOffset(offset));

    // notify寄存器的虚拟地址与物理地址偏移相同，所以虚拟地址为虚拟基地址加偏移
    u64 notifyVa = notifyBaseVa + offset;

    HCCL_INFO(
        "notifyBaseVa=0x%llx, notifyTotalSize=0x%x, offset=0x%llx, notifyVa=0x%llx ", notifyBaseVa, notifyTotalSize,
        offset, notifyVa);

    /* notify地址注册为mr, 在roce驱动中注册 */
    ackNotifyMsg_.mrRegFlag = 0;                                                    // mem注册给网卡标志位
    ackNotifyMsg_.addr = reinterpret_cast<void*>(static_cast<uintptr_t>(notifyVa)); // 本端notify地址交换给对端
    ackNotifyMsg_.len = notifySize_;
    ackNotifyMsg_.offset = offset;

    return HCCL_SUCCESS;
}

HcclResult TransportRemoteAccess::GetRemoteNotifyInfo()
{
    NotifyMsg mrMsg;
    s32 sRet = memset_s(&mrMsg, sizeof(NotifyMsg), 0, sizeof(NotifyMsg));
    CHK_PRT_RET(
        sRet != EOK,
        HCCL_ERROR(
            "[Get][NotifyInfo]errNo[0x%016llx]get remote addr, memory set 0 failed. "
            "params: destMaxSize[%zu], count[%zu]",
            HCCL_ERROR_CODE(HCCL_E_MEMORY), sizeof(NotifyMsg), sizeof(NotifyMsg)),
        HCCL_E_MEMORY);

    CHK_RET(hrtRaSocketBlockRecv(RemoteAccessPara_.socketFdhandle, &mrMsg, sizeof(NotifyMsg)));
    sRet = memcpy_s(&remoteNotifyDataMsg_, sizeof(NotifyMsg), &mrMsg, sizeof(NotifyMsg));
    CHK_PRT_RET(
        sRet != EOK,
        HCCL_ERROR(
            "[Get][NotifyInfo]errNo[0x%016llx] In TransportRemoteAccess get remote addr, memcpy failed. "
            "errorno[%d], params:destMaxSize[%zu],count[%zu]",
            HCCL_ERROR_CODE(HCCL_E_MEMORY), sRet, sizeof(NotifyMsg), sizeof(NotifyMsg)),
        HCCL_E_MEMORY);
    HCCL_INFO("recv success:len=%llu", mrMsg.len);

    return HCCL_SUCCESS;
}

HcclResult TransportRemoteAccess::CreateNotifyValueBuffer()
{
    std::unique_lock<std::mutex> lock(notifyValueMutex_[deviceLogicId_]);
    if (notifyValueMem_[deviceLogicId_].ptr() == nullptr) {
        u64 notifyVaule = 1; // notify值写1表示record
        CHK_RET(DeviceMem::alloc(notifyValueMem_[deviceLogicId_], notifyValueSize_));
        HCCL_DEBUG("create notify value size[%u]", notifySize_);

        CHK_RET(hrtMemSyncCopy(
            notifyValueMem_[deviceLogicId_].ptr(), notifyValueMem_[deviceLogicId_].size(), &notifyVaule, notifySize_,
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    }
    lock.unlock();
    struct MrInfoT mrInfo = {};
    mrInfo.addr = notifyValueMem_[deviceLogicId_].ptr();
    mrInfo.size = notifySize_;
    mrInfo.access = access_;
    CHK_RET(HrtRaMrReg(handle_, &mrInfo));
    // 将notify buffer地址保存
    localRegMem_.push_back(notifyValueMem_[deviceLogicId_].ptr());
    NotifyMsg msg = {};
    msg.mrRegFlag = REG_VALID;
    msg.addr = notifyValueMem_[deviceLogicId_].ptr();
    msg.len = notifySize_;

    /* 发送mem消息给对端 */
    HcclResult ret = hrtRaSocketBlockSend(RemoteAccessPara_.socketFdhandle, &msg, sizeof(NotifyMsg));
    if (ret != HCCL_SUCCESS) { // 发送成功字节数与发送字节数不等，发送失败
        HCCL_ERROR("[Create][NotifyValueBuffer]send=%zu", sizeof(NotifyMsg));
        return HCCL_E_INTERNAL;
    }

    return HCCL_SUCCESS;
}
HcclResult TransportRemoteAccess::RemoteRead(const std::vector<HcomRemoteAccessAddrInfo>& addrInfos, Stream& stream)
{
    CHK_SMART_PTR_NULL(stream);
    HCCL_INFO("TransportRemoteAccess RemoteRead begin");
    CHK_RET(RdmaDataTransport(addrInfos, RDMA_OP_READ));

    // 读取远端notify buffer，用于notify同步
    CHK_RET(ReadRemoteNotifyBuffer());

    // 等待TS把任务处理完成
    CHK_RET(LocalIpcNotify::Wait(
        stream, const_cast<HcclDispatcher>(dispatcher_), ackNotify_, INVALID_VALUE_STAGE, NOTIFY_INVALID_WAIT_TIME,
        RemoteAccessPara_.localRank, RemoteAccessPara_.remoteRank));

    HCCL_INFO("TransportRemoteAccess RemoteRead end");
    return HCCL_SUCCESS;
}

HcclResult TransportRemoteAccess::RemoteWrite(const std::vector<HcomRemoteAccessAddrInfo>& addrInfos, Stream& stream)
{
    CHK_SMART_PTR_NULL(stream);
    HCCL_RUN_INFO("TransportRemoteAccess RemoteWrite begin, addressNum[%u]", addrInfos.size());
    CHK_RET(RdmaDataTransport(addrInfos, RDMA_OP_WRITE));
    // 读取远端notify buffer，用于notify同步
    CHK_RET(ReadRemoteNotifyBuffer());
    // 等待TS把任务处理完成
    CHK_RET(LocalIpcNotify::Wait(
        stream, const_cast<HcclDispatcher>(dispatcher_), ackNotify_, INVALID_VALUE_STAGE, NOTIFY_INVALID_WAIT_TIME,
        RemoteAccessPara_.localRank, RemoteAccessPara_.remoteRank));

    return HCCL_SUCCESS;
}

HcclResult TransportRemoteAccess::RdmaDataTransport(const std::vector<HcomRemoteAccessAddrInfo>& addrInfos, s32 rdmaOp)
{
    if ((rdmaOp != RDMA_OP_WRITE) && (rdmaOp != RDMA_OP_READ)) {
        HCCL_ERROR("[Transport][RdmaData]invalid rdma op type, op:[%d]", rdmaOp);
        return HCCL_E_PARA;
    }
    CHK_PRT_RET(addrInfos.empty(), HCCL_ERROR("[Transport][RdmaData]addrInfos is empty!"), HCCL_E_PARA);
    u32 addressNum = addrInfos.size();
    // 构造wr信息
    std::vector<struct SendWrlistDataExt> wrVec(addressNum);
    std::vector<struct SendWrRsp> opRspVec(addressNum);
    struct SendWrlistDataExt* wr = wrVec.data();
    struct SendWrRsp* opRsp = opRspVec.data();
    struct SgList list = {};
    u64 length = addrInfos[0].length;

    HCCL_RUN_INFO("RdmaDataTransport begin, addressNum[%u], length[%u]", addressNum, length);
    for (size_t idx = 0; idx < addrInfos.size(); idx++) {
        list.addr = static_cast<u64>(static_cast<uintptr_t>(addrInfos[idx].localAddr));
        list.len = addrInfos[idx].length;

        wr[idx].memList = list;
        wr[idx].dstAddr = static_cast<u64>(static_cast<uintptr_t>(addrInfos[idx].remoteAddr));
        wr[idx].op = rdmaOp; /* RDMA_WRITE: 0  RDMA_READ: 4 */
        wr[idx].sendFlags = RA_SEND_SIGNALED;
    }
    u32 nowIdex = 0;
    u32 singleCompleteNum = 0;
    u32 sendNum = 0;
    HcclResult ret = HCCL_SUCCESS;
    u32 tryCount = 0;
    while (nowIdex < addressNum) {
        sendNum = addressNum - nowIdex;
        ret = HrtRaSendWrlistExt(handle_, wr, opRsp, sendNum, &singleCompleteNum);
        if (ret == HCCL_SUCCESS) {
            HCCL_INFO("dlRaSendWrlist success singleCompleteNum[%u], addressNum[%u]", singleCompleteNum, addressNum);
            return HCCL_SUCCESS;
        } else if (ret == ENOENT) { // 未完成发送，需重试
            nowIdex += singleCompleteNum;
            wr += singleCompleteNum;
            opRsp += singleCompleteNum;
            tryCount++;
            CHK_PRT_RET(
                tryCount > SEND_WRLIST_MAX_COUNT,
                HCCL_ERROR(
                    "[Transport][RdmaData]dlRaSendWrlist count beyond maxnum[%u], completenum[%u]",
                    SEND_WRLIST_MAX_COUNT, nowIdex),
                HCCL_E_NETWORK);
            continue;
        } else {
            HCCL_ERROR(
                "[Transport][RdmaData]In RdmaDataTransport, hrtRaSendWrlist failed. op[%d], ret[%d]", rdmaOp, ret);
            return HCCL_E_NETWORK;
        }
    }

    return HCCL_SUCCESS;
}

HcclResult TransportRemoteAccess::ReadRemoteNotifyBuffer()
{
    HCCL_INFO("In TransportRemoteAccess ReadRemoteNotifyBuffer begin");
    struct SgList list = {};
    struct SendWr wr = {};

    // 构造wr信息
    list.addr = static_cast<u64>(reinterpret_cast<uintptr_t>(ackNotifyMsg_.addr));
    list.len = remoteNotifyDataMsg_.len;

    wr.bufList = &list;
    wr.bufNum = 1; /* 此处list只有一个，设置为1 */
    wr.dstAddr = static_cast<u64>(reinterpret_cast<uintptr_t>(remoteNotifyDataMsg_.addr));
    wr.op = RDMA_OP_READ; /* RDMA_WRITE: 0 */
    wr.sendFlag = RA_SEND_SIGNALED;

    struct SendWrRsp opRsp = {};
    CHK_RET(HrtRaSendWr(handle_, &wr, &opRsp));
    HCCL_INFO("In TransportRemoteAccess ReadRemoteNotifyBuffer end");
    return HCCL_SUCCESS;
}

HcclResult TransportRemoteAccess::ConnectQp()
{
    // QP建链
    CHK_RET(HrtRaQpConnectAsync(handle_, RemoteAccessPara_.socketFdhandle));

    HCCL_INFO(
        "TransportRemoteAccess ConnectQp LocalRank[%u] "
        "RemoteRank[%u] LocalIp[%s]",
        RemoteAccessPara_.localRank, RemoteAccessPara_.remoteRank, RemoteAccessPara_.localIp.GetReadableAddress());

    // 查询QP建链是否成功
    s32 qpStatus = 0;
    auto startTime = std::chrono::steady_clock::now();
    HCCL_INFO("In link ibv, waiting for qp status ready...");
    while (true) {
        if ((std::chrono::steady_clock::now() - startTime) >= timeout_) {
            HCCL_ERROR("[Connect][Qp]get qp status timeout_=%lld s, qp_status=%d", timeout_, qpStatus);
            return HCCL_E_TIMEOUT;
        }

        s32 raRet = hrtGetRaQpStatus(handle_, &qpStatus);
        if ((!raRet) && (qpStatus == 1)) { // 为1时，qp 建链成功
            HCCL_INFO("GetRaQpStatus Success!");
            break;
        } else {
            // qp建链需要时间，获取qp状态直至超时
            SaluSleep(WAIT_US_COUNT);
        }
    }
    return HCCL_SUCCESS;
}
} // namespace hccl
