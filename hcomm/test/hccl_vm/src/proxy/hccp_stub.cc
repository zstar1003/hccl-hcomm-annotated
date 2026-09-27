/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 日志染色: 模块 tag (须在 include sim_log.h 之前)
#define HCCL_VM_MODULE "HCCP_STUB"

#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "atrace_types.h"
#include "dtype_common.h"
#include "sim_log.h"
#include "hccp_common.h"
#include "hccp_ctx.h"
#include "hccp_tlv.h"
#include "sim_ip_address.h"
#include "mem.h"
#include "rts_device.h"
#include "runtime/base.h"
#include "store_sim_device_memory_manager.h"
#include "store_sim_memory_manager.h"
#include "db_sim_op_db_ops.h"
#include "db_sim_runner_common.h"
#include "db_sim_runner_ops.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

int SimRaCustomChannel(void* tlvHandle, struct TlvMsg* sendMsg, struct TlvMsg* recvMsg);

//////////////////////RDMA/////////////////////////////

int RaSocketGetVnicIpInfos(
    unsigned int phyId, enum IdType type, unsigned int ids[], unsigned int num, struct IpInfo infos[])
{
    sim::Device device{};
    if (GetDeviceByPhysicalId(phyId, device) != ACL_SUCCESS) {
        HCCL_VM_ERROR("get device by phy id {} failed.", phyId);
        return -1;
    }

    uint64_t deviceIdx = device.id;
    auto endPoints = RunnerDB::GetByPred<sim::EndPoint>([deviceIdx](const sim::EndPoint& ep) {
        return ep.device_id == deviceIdx;
    });

    if (endPoints.empty()) {
        HCCL_VM_ERROR("no endpoints found for device {:d}", deviceIdx);
        return -1;
    }

    unsigned int count = std::min((unsigned int)endPoints.size(), num);
    for (unsigned int i = 0; i < count; i++) {
        infos[i].family = AF_INET6;
        std::string ipaddr = std::string(endPoints[i].ip_addr);
        if (ipaddr.find(':') == std::string::npos) {
            ipaddr = "::" + ipaddr;
        }
        IpAddress addr(ipaddr, AF_INET6);
        infos[i].ip.addr6 = addr.GetBinaryAddress().addr6;
        HCCL_VM_INFO("phyId:{:d} type:{:d} infos[{:d}]={}", phyId, (int)type, i, endPoints[i].ip_addr);
    }
    return 0;
}

int RaGetInterfaceVersion(unsigned int phyId, unsigned int interfaceOpcode, unsigned int* interfaceVersion)
{
    *interfaceVersion = 2; // GET_UBOE_FLAG_ENABLE_VERSION
    return 0;
}

int RaGetTsqpDepth(void* rdevHandle, unsigned int* tempDepth, unsigned int* qpNum)
{
    HCCL_VM_WARN("is empty");
    return 0;
}
int RaSetTsqpDepth(void* rdevHandle, unsigned int tempDepth, unsigned int* qpNum)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaRdevGetSupportLite(void* rdmaHandle, int* supportLite)
{
    if (rdmaHandle == nullptr || supportLite == nullptr) {
        HCCL_VM_ERROR("invalid params, rdmaHandle: {}, supportLite: {}", rdmaHandle, (void*)supportLite);
        return -1;
    }
    *supportLite = 1;
    uint64_t handleVal = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(rdmaHandle));
    HCCL_VM_INFO("rdmaHandle:{:d} supportLite:1", handleVal);
    return 0;
}

int RaSocketSetWhiteListStatus(unsigned int enable)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaSocketGetWhiteListStatus(unsigned int* enable)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaNormalQpCreate(void* rdevHandle, struct ibv_qp_init_attr* qpInitAttr, void** qpHandle, void** qp)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaNormalQpDestroy(void* qpHandle)
{
    if (qpHandle == nullptr) {
        HCCL_VM_ERROR("qpHandle is null");
        return HCCL_E_PTR;
    }
    return 0;
}

int RaMrReg(void* qpHandle, struct MrInfoT* info)
{
    if (qpHandle == nullptr || info == nullptr) {
        HCCL_VM_ERROR("invalid params, qpHandle: {}, info: {}", qpHandle, (void*)info);
        return -1;
    }

    uint64_t qpId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(qpHandle));
    auto qpOpt = RunnerDB::GetById<sim::RaQP>(qpId);
    if (!qpOpt.has_value()) {
        HCCL_VM_ERROR("QP {:d} not found", qpId);
        return -1;
    }

    sim::RaMR mr{};
    mr.addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(info->addr));
    mr.length = info->size;
    mr.local_key = (qpId << 32) | (rand() & 0xFFFFFFFF);
    mr.remote_key = mr.local_key | 0x1;

    auto mrId = RunnerDB::Add<sim::RaMR>(mr);
    info->lkey = mr.local_key;
    info->rkey = mr.remote_key;

    HCCL_VM_INFO(
        "QP {:d} reg MR id:{:d}, addr:{:x}, len:{:d}, lkey:{:x}, rkey:{:x}", qpId, mrId, mr.addr, mr.length,
        mr.local_key, mr.remote_key);
    return 0;
}

int RaMrDereg(void* qpHandle, struct MrInfoT* info)
{
    if (qpHandle == nullptr || info == nullptr) {
        HCCL_VM_ERROR("invalid params, qpHandle: {}, info: {}", qpHandle, (void*)info);
        return -1;
    }

    auto mrList = RunnerDB::GetByPred<sim::RaMR>([info](const sim::RaMR& mr) {
        return mr.local_key == info->lkey;
    });

    if (mrList.empty()) {
        HCCL_VM_WARN("MR with lkey {:x} not found", info->lkey);
        return 0;
    }

    for (const auto& mr : mrList) {
        RunnerDB::Delete<sim::RaMR>(mr.id);
        HCCL_VM_INFO("dereg MR id:{:d}, lkey:{:x}", mr.id, mr.local_key);
    }

    return 0;
}

int RaRegisterMr(const void* rdmaHandle, struct MrInfoT* info, void** mrHandle)
{
    if (rdmaHandle == nullptr || info == nullptr || mrHandle == nullptr) {
        HCCL_VM_ERROR(
            "invalid params, rdmaHandle: {}, info: {}, mrHandle: {}", rdmaHandle, (void*)info, (void*)mrHandle);
        return -1;
    }

    uint64_t raDevId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(rdmaHandle));

    sim::RaMR mr{};
    mr.addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(info->addr));
    mr.length = info->size;
    mr.local_key = (raDevId << 32) | (rand() & 0xFFFFFFFF);
    mr.remote_key = mr.local_key | 0x1;

    auto mrId = RunnerDB::Add<sim::RaMR>(mr);
    *mrHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(mrId));
    info->lkey = mr.local_key;
    info->rkey = mr.remote_key;

    HCCL_VM_INFO("RaDev {:d} reg MR id:{:d}, addr:{:x}, len:{:d}", raDevId, mrId, mr.addr, mr.length);
    return 0;
}

int RaRemapMr(const void* rdmaHandle, struct MemRemapInfo info[], unsigned int num)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaDeregisterMr(const void* rdmaHandle, void* mrHandle)
{
    if (rdmaHandle == nullptr || mrHandle == nullptr) {
        HCCL_VM_ERROR("invalid params, rdmaHandle: {}, mrHandle: {}", rdmaHandle, mrHandle);
        return -1;
    }

    uint64_t mrId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(mrHandle));
    auto mrOpt = RunnerDB::GetById<sim::RaMR>(mrId);
    if (!mrOpt.has_value()) {
        HCCL_VM_WARN("MR {:d} not found", mrId);
        return 0;
    }

    RunnerDB::Delete<sim::RaMR>(mrId);
    HCCL_VM_INFO("dereg MR id:{:d}", mrId);
    return 0;
}

int RaSendWr(void* qpHandle, struct SendWr* wr, struct SendWrRsp* opRsp)
{
    if (qpHandle == nullptr || wr == nullptr) {
        HCCL_VM_ERROR("invalid params, qpHandle: {}, wr: {}", qpHandle, (void*)wr);
        return -1;
    }

    uint64_t qpId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(qpHandle));
    auto qpOpt = RunnerDB::GetById<sim::RaQP>(qpId);
    if (!qpOpt.has_value()) {
        HCCL_VM_ERROR("QP {:d} not found", qpId);
        return -1;
    }

    if (qpOpt->state != 3) {
        HCCL_VM_ERROR("QP {:d} not in RTS state, current state:{:d}", qpId, qpOpt->state);
        return -1;
    }

    HCCL_VM_INFO("QP {:d} send op:{:d}, bufNum:{:d}, dstAddr:{:x}", qpId, wr->op, wr->bufNum, wr->dstAddr);
    return 0;
}

int RaSetQpAttrQos(void* qpHandle, struct QosAttr* attr)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaSetQpAttrTimeout(void* qpHandle, unsigned int* timeout)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaSetQpAttrRetryCnt(void* qpHandle, unsigned int* retryCnt)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaGetCqeErrInfo(unsigned int phyId, struct CqeErrInfo* info)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaCreateSrq(const void* rdmaHandle, struct SrqAttr* attr)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaDestroySrq(const void* rdmaHandle, struct SrqAttr* attr)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaCreateEventHandle(int* eventHandle)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaCtlEventHandle(int eventHandle, const void* fdHandle, int opcode, enum RaEpollEvent event)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaWaitEventHandle(
    int eventHandle, struct SocketEventInfoT* eventInfos, int timeout, unsigned int maxevents, unsigned int* eventsNum)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaDestroyEventHandle(int* eventHandle)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaCreateCompChannel(const void* rdmaHandle, void** compChannel)
{
    *compChannel = reinterpret_cast<void*>(static_cast<uintptr_t>(0xabcdU));
    return ((rdmaHandle == nullptr) || (compChannel == nullptr)) ? -1 : 0;
}

int RaDestroyCompChannel(const void* rdmaHandle, void* compChannel)
{
    return ((rdmaHandle == nullptr) || (compChannel == nullptr)) ? -1 : 0;
}

int RaLoopbackQpCreate(void* rdevHandle, struct LoopbackQpPair* qpPair, void** qpHandle)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaQpConnectAsync(void* qpHandle, const void* fdHandle)
{
    if (qpHandle == nullptr) {
        HCCL_VM_ERROR("qpHandle is null");
        return -1;
    }

    uint64_t qpId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(qpHandle));
    auto qpOpt = RunnerDB::GetById<sim::RaQP>(qpId);
    if (!qpOpt.has_value()) {
        HCCL_VM_ERROR("QP {:d} not found", qpId);
        return -1;
    }

    RunnerDB::Update<sim::RaQP>(qpId, [](sim::RaQP& qp) {
        qp.state = 3;
    });

    HCCL_VM_INFO("QP {:d} connected, state -> RTS", qpId);
    return 0;
}

int RaSendWrV2(void* qpHandle, struct SendWrV2* wr, struct SendWrRsp* opRsp)
{
    if (qpHandle == nullptr || wr == nullptr) {
        HCCL_VM_ERROR("invalid params, qpHandle: {}, wr: {}", qpHandle, (void*)wr);
        return -1;
    }

    uint64_t qpId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(qpHandle));
    auto qpOpt = RunnerDB::GetById<sim::RaQP>(qpId);
    if (!qpOpt.has_value()) {
        HCCL_VM_ERROR("QP {:d} not found", qpId);
        return -1;
    }

    if (qpOpt->state != 3) {
        HCCL_VM_ERROR("QP {:d} not in RTS state", qpId);
        return -1;
    }

    HCCL_VM_INFO("QP {:d} send V2", qpId);
    return 0;
}

int RaPollCq(void* qpHandle, bool isSendCq, unsigned int numEntries, void* wc)
{
    if (qpHandle == nullptr) {
        HCCL_VM_ERROR("qpHandle is null");
        return -1;
    }

    uint64_t qpId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(qpHandle));
    auto qpOpt = RunnerDB::GetById<sim::RaQP>(qpId);
    if (!qpOpt.has_value()) {
        HCCL_VM_ERROR("QP {:d} not found", qpId);
        return -1;
    }

    uint64_t cqId = isSendCq ? qpOpt->send_cq_handle : qpOpt->recv_cq_handle;

    auto cqOpt = RunnerDB::GetById<sim::RaCQ>(cqId);
    if (!cqOpt.has_value()) {
        HCCL_VM_WARN("CQ {:d} not found for QP {:d}", cqId, qpId);
        return 0;
    }

    auto cqeList = RunnerDB::GetByPred<sim::RaCQE>([cqId](const sim::RaCQE& cqe) {
        return cqe.cq_handle == cqId && cqe.status == 0;
    });

    unsigned int polled = 0;
    for (const auto& cqe : cqeList) {
        if (polled >= numEntries) {
            break;
        }
        RunnerDB::Delete<sim::RaCQE>(cqe.id);
        polled++;
    }

    HCCL_VM_INFO("QP {:d} CQ {:d} polled {:d} entries (isSendCq:{:d})", qpId, cqId, polled, isSendCq);
    return polled;
}

int RaRecvWrlist(void* qpHandle, struct RecvWrlistData* wr, unsigned int recvNum, unsigned int* completeNum)
{
    if (qpHandle == nullptr || wr == nullptr) {
        HCCL_VM_ERROR("invalid params, qpHandle: {}, wr: {}", qpHandle, (void*)wr);
        return -1;
    }

    uint64_t qpId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(qpHandle));
    auto qpOpt = RunnerDB::GetById<sim::RaQP>(qpId);
    if (!qpOpt.has_value()) {
        HCCL_VM_ERROR("QP {:d} not found", qpId);
        return -1;
    }

    if (qpOpt->state != 3) {
        HCCL_VM_ERROR("QP {:d} not in RTS state", qpId);
        return -1;
    }

    if (completeNum != nullptr) {
        *completeNum = recvNum;
    }

    HCCL_VM_INFO("QP {:d} post {:d} recv WRs", qpId, recvNum);
    return 0;
}

int RaGetQpContext(void* qpHandle, void** qp, void** sendCq, void** recvCq)
{
    if (qpHandle == nullptr) {
        HCCL_VM_ERROR("qpHandle is null");
        return -1;
    }

    uint64_t qpId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(qpHandle));
    auto qpOpt = RunnerDB::GetById<sim::RaQP>(qpId);
    if (!qpOpt.has_value()) {
        HCCL_VM_ERROR("QP {:d} not found", qpId);
        return -1;
    }

    if (qp != nullptr) {
        *qp = qpHandle;
    }
    if (sendCq != nullptr) {
        *sendCq = reinterpret_cast<void*>(static_cast<uintptr_t>(qpOpt->send_cq_handle));
    }
    if (recvCq != nullptr) {
        *recvCq = reinterpret_cast<void*>(static_cast<uintptr_t>(qpOpt->recv_cq_handle));
    }

    HCCL_VM_INFO("QP {:d} sendCq:{:d} recvCq:{:d}", qpId, qpOpt->send_cq_handle, qpOpt->recv_cq_handle);
    return 0;
}

int RaQpBatchModify(void* rdmaHandle, void* qpHandle[], unsigned int num, int expectStatus)
{
    if (qpHandle == nullptr || num == 0) {
        HCCL_VM_ERROR("invalid params, rdmaHandle: {}", rdmaHandle);
        return -1;
    }

    unsigned int modified = 0;
    for (unsigned int i = 0; i < num; i++) {
        if (qpHandle[i] == nullptr) {
            continue;
        }
        uint64_t qpId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(qpHandle[i]));
        auto qpOpt = RunnerDB::GetById<sim::RaQP>(qpId);
        if (!qpOpt.has_value()) {
            HCCL_VM_WARN("QP {:d} not found", qpId);
            continue;
        }

        RunnerDB::Update<sim::RaQP>(qpId, [expectStatus](sim::RaQP& qp) {
            qp.state = expectStatus;
        });
        modified++;
    }

    HCCL_VM_INFO("batch modify {:d} QPs to state {:d}", modified, expectStatus);
    return 0;
}

int RaRdevGetCqeErrInfoList(void* rdmaHandle, struct CqeErrInfo* infoList, unsigned int* num)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaRdevGetHandle(unsigned int phyId, void** rdmaHandle)
{
    if (rdmaHandle == nullptr) {
        HCCL_VM_ERROR("rdmaHandle is null");
        return -1;
    }

    sim::Device device{};
    if (GetDeviceByPhysicalId(phyId, device) != ACL_SUCCESS) {
        HCCL_VM_ERROR("get device by phy id {} failed.", phyId);
        return -1;
    }

    auto raDevRes = RunnerDB::GetOneByPred<sim::RaDevice>([device](const sim::RaDevice& dev) {
        return dev.device_id == device.id;
    });

    if (raDevRes.second) {
        *rdmaHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(raDevRes.first.id));
        HCCL_VM_INFO("found RaDevice id:{:d} for phyId:{:d}", raDevRes.first.id, phyId);
        return 0;
    }

    HCCL_VM_ERROR("no RaDevice found for phyId:{:d}", phyId);
    return -1;
}

int RaSaveSnapshot(struct RaInfo* info, enum SaveSnapshotAction action)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaRestoreSnapshot(struct RaInfo* info)
{
    HCCL_VM_WARN("is empty");
    return 0;
}
int RaRdevInitWithBackup(
    struct RdevInitInfo* initInfo, struct rdev* rdevInfo, struct rdev* backupRdevInfo, void** rdmaHandle)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaGetQpStatus(void* qpHandle, int* status)
{
    if (qpHandle == nullptr || status == nullptr) {
        HCCL_VM_ERROR("invalid params, qpHandle: {}, status: {}", qpHandle, (void*)status);
        return -1;
    }

    uint64_t qpId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(qpHandle));
    auto qpOpt = RunnerDB::GetById<sim::RaQP>(qpId);
    if (!qpOpt.has_value()) {
        HCCL_VM_ERROR("QP {:d} not found", qpId);
        return -1;
    }

    *status = qpOpt->state;
    HCCL_VM_INFO("QP {:d} status:{:d}", qpId, *status);
    return 0;
}

int RaSendWrlist(
    void* qpHandle, struct SendWrlistData wr[], struct SendWrRsp opRsp[], unsigned int sendNum,
    unsigned int* completeNum)
{
    if (qpHandle == nullptr || wr == nullptr) {
        HCCL_VM_ERROR("invalid params, qpHandle: {}, wr: {}", qpHandle, (void*)wr);
        return -1;
    }

    uint64_t qpId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(qpHandle));
    auto qpOpt = RunnerDB::GetById<sim::RaQP>(qpId);
    if (!qpOpt.has_value()) {
        HCCL_VM_ERROR("QP {:d} not found", qpId);
        return -1;
    }

    if (qpOpt->state != 3) {
        HCCL_VM_ERROR("QP {:d} not in RTS state", qpId);
        return -1;
    }

    unsigned int completed = sendNum;

    if (completeNum != nullptr) {
        *completeNum = completed;
    }

    HCCL_VM_INFO("QP {:d} send {:d} WRs, completed:{:d}", qpId, sendNum, completed);
    return 0;
}

int RaSendWrlistExt(
    void* qpHandle, struct SendWrlistDataExt wr[], struct SendWrRsp opRsp[], unsigned int sendNum,
    unsigned int* completeNum)
{
    if (qpHandle == nullptr || wr == nullptr) {
        HCCL_VM_ERROR("invalid params, qpHandle: {}, wr: {}", qpHandle, (void*)wr);
        return -1;
    }

    uint64_t qpId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(qpHandle));
    auto qpOpt = RunnerDB::GetById<sim::RaQP>(qpId);
    if (!qpOpt.has_value()) {
        HCCL_VM_ERROR("QP {:d} not found", qpId);
        return -1;
    }

    if (qpOpt->state != 3) {
        HCCL_VM_ERROR("QP {:d} not in RTS state", qpId);
        return -1;
    }

    unsigned int completed = sendNum;

    if (completeNum != nullptr) {
        *completeNum = completed;
    }

    HCCL_VM_INFO("QP {:d} send {:d} WRs (Ext), completed:{:d}", qpId, sendNum, completed);
    return 0;
}

int RaSendNormalWrlist(
    void* qpHandle, struct WrInfo wr[], struct SendWrRsp opRsp[], unsigned int sendNum, unsigned int* completeNum)
{
    if (qpHandle == nullptr || wr == nullptr) {
        HCCL_VM_ERROR("invalid params, qpHandle: {}, wr: {}", qpHandle, (void*)wr);
        return -1;
    }

    uint64_t qpId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(qpHandle));
    auto qpOpt = RunnerDB::GetById<sim::RaQP>(qpId);
    if (!qpOpt.has_value()) {
        HCCL_VM_ERROR("QP {:d} not found", qpId);
        return -1;
    }

    if (qpOpt->state != 3) {
        HCCL_VM_ERROR("QP {:d} not in RTS state", qpId);
        return -1;
    }

    unsigned int completed = sendNum;

    if (completeNum != nullptr) {
        *completeNum = completed;
    }

    HCCL_VM_INFO("QP {:d} send {:d} normal WRs, completed:{:d}", qpId, sendNum, completed);
    return 0;
}

int RaGetNotifyBaseAddr(void* rdevHandle, unsigned long long* va, unsigned long long* size)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaGetNotifyMrInfo(void* rdevHandle, struct MrInfoT* info)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaInit(struct RaInitConfig* config)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaDeinit(struct RaInitConfig* config)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaGetTlsEnable(struct RaInfo* info, bool* tlsEnable)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaGetHccnCfg(struct RaInfo* info, enum HccnCfgKey key, char* value, unsigned int* valueLen)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaRdevInitV2(struct RdevInitInfo initInfo, struct rdev rdevInfo, void** rdmaHandle)
{
    auto serverId = sim::GetCurServerId();
    if (serverId == 0) {
        HCCL_VM_ERROR("GetCurServerId failed");
        return -1;
    }

    sim::Runner runner;
    if (!sim::GetCurrRunnerTls(serverId, runner)) {
        HCCL_VM_ERROR("GetCurrRunnerTls failed");
        return -1;
    }
    auto currCtx = RunnerDB::GetById<sim::Context>(runner.current_ctx_id);
    if (!currCtx.has_value()) {
        HCCL_VM_ERROR("can not get CurrContext: {:d}", runner.current_ctx_id);
        return -1;
    }

    sim::RaDevice dev{};
    dev.device_id = currCtx->device_id;
    auto id = RunnerDB::Add<sim::RaDevice>(dev);

    *rdmaHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(id));
    HCCL_VM_INFO("add radev id {:d}", id);
    return 0;
}

int RaRdevInit(int mode, unsigned int notifyType, struct rdev rdevInfo, void** rdmaHandle)
{
    struct RdevInitInfo initInfo;
    RaRdevInitV2(initInfo, rdevInfo, rdmaHandle);
    return 0;
}

int RaRdevDeinit(void* rdmaHandle, unsigned int notifyType)
{
    uint64_t id = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(rdmaHandle));
    RunnerDB::Delete<sim::RaDevice>(id);
    HCCL_VM_INFO("delete id {:d}", id);
    return 0;
}

int RaCqCreate(void* rdevHandle, struct CqAttr* attr)
{
    uint64_t raDevId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(rdevHandle));
    sim::RaCQ cq{};
    cq.ra_dev_id = raDevId;
    auto id = RunnerDB::Add<sim::RaCQ>(cq);

    *(attr->qpContext) = reinterpret_cast<void*>(static_cast<uintptr_t>(id));

    HCCL_VM_WARN("RaDev {:d} add RaCQ id:{:d}", raDevId, id);
    return 0;
}

int RaCqDestroy(void* rdevHandle, struct CqAttr* attr)
{
    uint64_t raDevId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(rdevHandle));
    uint64_t cqId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(attr->qpContext));
    RunnerDB::Delete<sim::RaCQ>(cqId);

    HCCL_VM_WARN("RaDev {:d} delete RaCQ id:{:d}", raDevId, cqId);
    return 0;
}

int RaQpCreate(void* rdevHandle, int flag, int qpMode, void** qpHandle)
{
    if (rdevHandle == nullptr || qpHandle == nullptr) {
        HCCL_VM_ERROR("invalid params, rdevHandle: {}, qpHandle: {}", rdevHandle, (void*)qpHandle);
        return -1;
    }

    uint64_t raDevId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(rdevHandle));

    auto raDevOpt = RunnerDB::GetById<sim::RaDevice>(raDevId);
    if (!raDevOpt.has_value()) {
        HCCL_VM_ERROR("RaDevice {:d} not found", raDevId);
        return -1;
    }

    static std::atomic<uint32_t> s_qpNumCounter{1};

    sim::RaQP qp{};
    qp.state = 0;
    qp.ra_dev_id = raDevId;
    qp.qp_num = s_qpNumCounter.fetch_add(1);
    qp.type = qpMode;
    qp.send_cq_handle = 0;
    qp.recv_cq_handle = 0;
    qp.peer_qpn = 0;
    qp.perr_lid = 0;
    qp.taJettyId = 0;

    auto id = RunnerDB::Add<sim::RaQP>(qp);
    *qpHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(id));

    HCCL_VM_INFO("RaDev {:d} create QP id:{:d}, qp_num:{:d}, type:{:d}", raDevId, id, qp.qp_num, qp.type);
    return 0;
}

int RaQpCreateWithAttrs(void* rdevHandle, struct QpExtAttrs* extAttrs, void** qpHandle)
{
    if (rdevHandle == nullptr || qpHandle == nullptr) {
        HCCL_VM_ERROR("invalid params, rdevHandle: {}, qpHandle: {}", rdevHandle, (void*)qpHandle);
        return -1;
    }

    uint64_t raDevId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(rdevHandle));

    auto raDevOpt = RunnerDB::GetById<sim::RaDevice>(raDevId);
    if (!raDevOpt.has_value()) {
        HCCL_VM_ERROR("RaDevice {:d} not found", raDevId);
        return -1;
    }

    static std::atomic<uint32_t> s_qpNumCounter{1000};

    sim::RaQP qp{};
    qp.state = 0;
    qp.ra_dev_id = raDevId;
    qp.qp_num = s_qpNumCounter.fetch_add(1);
    qp.send_cq_handle = 0;
    qp.recv_cq_handle = 0;
    qp.peer_qpn = 0;
    qp.perr_lid = 0;
    qp.taJettyId = 0;

    if (extAttrs != nullptr) {
        qp.type = extAttrs->qpMode;
    } else {
        qp.type = 0;
    }

    auto id = RunnerDB::Add<sim::RaQP>(qp);
    *qpHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(id));

    HCCL_VM_INFO("RaDev {:d} create QP id:{:d}, qp_num:{:d}, qpMode:{:d}", raDevId, id, qp.qp_num, qp.type);
    return 0;
}

int RaAiQpCreate(void* rdevHandle, struct QpExtAttrs* attrs, struct AiQpInfo* info, void** qpHandle)
{
    int ret = RaQpCreate(rdevHandle, 0, 0, qpHandle);
    return ret;
}

int RaTypicalQpCreate(void* rdevHandle, int flag, int qpMode, struct TypicalQp* qpInfo, void** qpHandle)
{
    int ret = RaQpCreate(rdevHandle, 0, 0, qpHandle);
    return ret;
}

int RaCtxInit(struct CtxInitCfg* cfg, struct CtxInitAttr* attr, void** ctxHandle)
{
    Eid simEid{};
    memcpy(simEid.raw, attr->ub.eid.raw, sizeof(simEid));
    auto ipAddr = IpAddress(simEid).GetIpStr().substr(2);

    if (ipAddr == "0.0.0.0") {
        HCCL_VM_ERROR("can not support addr 0");
        return -1;
    }

    sim::EndPoint endPoint{};
    if (GetEndPointByIpAddr(ipAddr, endPoint) != 0) {
        HCCL_VM_ERROR("cannot find endpoint addr:{}", ipAddr);
        return -1;
    }

    sim::RaContext devRaCtx{};
    devRaCtx.device_id = endPoint.device_id;
    devRaCtx.endpoint_id = endPoint.id;
    devRaCtx.eidIndex = attr->ub.eidIndex;
    devRaCtx.mode = cfg->mode;
    auto ctxId = RunnerDB::Add<sim::RaContext>(devRaCtx);

    sim::RaTp tp{};
    tp.ctx_handle = ctxId;
    RunnerDB::Add<sim::RaTp>(tp);

    HCCL_VM_INFO("add ctx id {:d}, addr {}", ctxId, ipAddr);

    *ctxHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(ctxId));
    return 0;
}

int RaCtxDeinit(void* ctxHandle)
{
    uint64_t id = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ctxHandle));
    HCCL_VM_INFO("soft delete ctx id {:d}, for checker use after", id);
    return 0;
}

int RaGetDevBaseAttr(void* ctxHandle, struct DevBaseAttr* attr)
{
    uint64_t id = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ctxHandle));
    auto devCtx = RunnerDB::GetById<sim::RaContext>(id);
    if (!devCtx.has_value()) {
        HCCL_VM_ERROR("can not find dev by id:{:d}", id);
        return -1;
    }

    auto endPoint = RunnerDB::GetById<sim::EndPoint>(devCtx->endpoint_id);
    if (!endPoint.has_value()) {
        HCCL_VM_ERROR("can not find endpoint:{:d}", devCtx->endpoint_id);
        return -1;
    }

    attr->ub.dieId = static_cast<uint32_t>(endPoint->die_id);
    attr->ub.funcId = endPoint->func_id;

    for (int i = 0; i < MAX_PRIORITY_CNT; i++) {
        CtxSlInfo& priorityInfo = attr->ub.priorityInfo[i];
        priorityInfo.tpType.bs.rtp = 1;
    }

    attr->sqMaxDepth = 8192;                // max_jfs_depth
    attr->rqMaxDepth = 32768;               // max_jfr_depth
    attr->sqMaxSge = 13;                    // max_jfs_sge
    attr->rqMaxSge = 4;                     // max_jfr_sge
    attr->maxReadSize = 256 * 1024 * 1024;  // max_read_size
    attr->maxWriteSize = 256 * 1024 * 1024; // max_write_size
    attr->maxMsgSize = 65536;               // max_msg_size

    return 0;
}

int RaQpDestroy(void* qpHandle)
{
    if (qpHandle == nullptr) {
        HCCL_VM_ERROR("qpHandle is null");
        return -1;
    }

    uint64_t qpId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(qpHandle));
    auto qpOpt = RunnerDB::GetById<sim::RaQP>(qpId);
    if (!qpOpt.has_value()) {
        HCCL_VM_WARN("QP {:d} not found", qpId);
        return 0;
    }

    auto mrList = RunnerDB::GetByPred<sim::RaMR>([qpId](const sim::RaMR& mr) {
        return mr.vptr_id == qpId;
    });
    for (const auto& mr : mrList) {
        RunnerDB::Delete<sim::RaMR>(mr.id);
        HCCL_VM_INFO("cleanup MR {:d} for QP {:d}", mr.id, qpId);
    }

    auto cqeList = RunnerDB::GetByPred<sim::RaCQE>([qpOpt](const sim::RaCQE& cqe) {
        return cqe.cq_handle == qpOpt->send_cq_handle || cqe.cq_handle == qpOpt->recv_cq_handle;
    });
    for (const auto& cqe : cqeList) {
        RunnerDB::Delete<sim::RaCQE>(cqe.id);
    }

    HCCL_VM_INFO("destroy QP {:d}, qp_num:{:d}", qpId, qpOpt->qp_num);
    return 0;
}

int RaCtxQpCreate(void* ctxHandle, struct QpCreateAttr* attr, struct QpCreateInfo* info, void** qpHandle)
{
    if (ctxHandle == nullptr || attr == nullptr || info == nullptr || qpHandle == nullptr) {
        HCCL_VM_ERROR(
            "invalid params, ctxHandle: {}, attr: {}, info: {}, qpHandle: {}", ctxHandle, (void*)attr, (void*)info,
            (void*)qpHandle);
        return -1;
    }

    uint64_t sqBuffer = 0;
    uint32_t jettyId = 0;
    bool aicpuMode = (attr->ub.mode == JettyMode::JETTY_MODE_USER_CTL_NORMAL);
    if (aicpuMode) {
        sqBuffer = attr->ub.extMode.sq.buffVa;
        if (sqBuffer == 0) { // 判断是否需要由桩函数分配sqBuffer
            void* wqeBuf = nullptr;
            aclrtMalloc(&wqeBuf, attr->ub.extMode.sq.buffSize, aclrtMemMallocPolicy::ACL_MEM_MALLOC_NORMAL_ONLY);
            if (wqeBuf == nullptr) {
                HCCL_VM_ERROR("malloc wqeBuf failed.");
                return -1;
            }

            sqBuffer = reinterpret_cast<uint64_t>(wqeBuf);
        }
    }

    uint64_t ctxId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ctxHandle));
    static std::atomic<uint32_t> s_qpNumCounter{10000};

    auto devCtx = RunnerDB::GetById<sim::RaContext>(ctxId);
    if (!devCtx.has_value()) {
        HCCL_VM_ERROR("can not find Context:{:d}", ctxId);
        return -1;
    }

    auto endPoint = RunnerDB::GetById<sim::EndPoint>(devCtx->endpoint_id);
    if (!endPoint.has_value()) {
        HCCL_VM_ERROR("can not find endpoint:{:d}", devCtx->endpoint_id);
        return -1;
    }

    sim::RaJetty jty{};
    jty.ctx_handle = ctxId;
    jty.jetty_id = attr->ub.jettyId;
    jty.dieId = endPoint->die_id;
    jty.type = (uint8_t)attr->transportMode;
    jty.send_cq_handle = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(attr->scqHandle));
    jty.recv_cq_handle = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(attr->rcqHandle));
    jty.mode = attr->ub.mode;
    jty.pid = getpid();

    auto id = RunnerDB::Add<sim::RaJetty>(jty);
    *qpHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(id));

    // aicpu模式更新jetty_id和sqBuffer
    if (aicpuMode) {
        RunnerDB::Update<sim::RaJetty>(id, [id, sqBuffer](sim::RaJetty& jty) {
            jty.jetty_id = id;
            jty.sqBuffer = sqBuffer;
        });
        HCCL_VM_INFO("jettyId[{}], sqBuffer[{}].", id, sqBuffer);
    }

    if (info != nullptr) {
        // info->rdma.qpn = qp.qp_num;
        *(uint64_t*)(info->key.value) = id;
        info->key.size = sizeof(uint64_t);
        info->ub.sqBuffVa = sqBuffer;
        info->ub.id = aicpuMode ? id : attr->ub.jettyId;
    }

    HCCL_VM_INFO(
        "Ctx:{:d} create QP id:{:d}, scqHandle:{:d}, rcqHandle:{:d}, JettyId:{:d}, mode:{:d}", ctxId, id,
        jty.send_cq_handle, jty.recv_cq_handle, jty.jetty_id, jty.mode);
    return 0;
}

int RaCtxQpImport(void* ctxHandle, struct QpImportInfoT* qpInfo, void** remQpHandle)
{
    uint64_t remoteQpId = *(uint64_t*)(qpInfo->in.key.value);
    auto remoteQpOpt = RunnerDB::GetById<sim::RaJetty>(remoteQpId);
    if (!remoteQpOpt.has_value()) {
        HCCL_VM_ERROR("remote QP {:d} not found", remoteQpId);
        return -1;
    }

    auto rmtRaCtxOpt = RunnerDB::GetById<sim::RaContext>(remoteQpOpt->ctx_handle);
    if (!rmtRaCtxOpt.has_value()) {
        HCCL_VM_ERROR("remote RaContext {:d} not found", remoteQpOpt->ctx_handle);
        return -1;
    }

    uint64_t localRaCtxId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ctxHandle));
    auto localRaCtxOpt = RunnerDB::GetById<sim::RaContext>(localRaCtxId);
    if (!localRaCtxOpt.has_value()) {
        HCCL_VM_ERROR("local RaContext {:d} not found", localRaCtxId);
        return -1;
    }
    uint64_t localEndpointId = localRaCtxOpt->endpoint_id;
    uint64_t remoteEndpointId = rmtRaCtxOpt->endpoint_id;
    auto pairOpt
        = RunnerDB::GetOneByPred<sim::EndPointPair>([localEndpointId, remoteEndpointId](const sim::EndPointPair& pair) {
              return ((pair.local_enpoint_id == localEndpointId) && (pair.remote_enpoint_id == remoteEndpointId));
          });

    if (!pairOpt.second) {
        sim::EndPointPair endpointPair{};
        endpointPair.local_enpoint_id = localEndpointId;
        endpointPair.remote_enpoint_id = remoteEndpointId;
        endpointPair.tp_type = qpInfo->in.ub.tpType;
        auto id = RunnerDB::Add<sim::EndPointPair>(endpointPair);
    }
#if 0
    sim::JettyMapTab jettyMapInfo {};
    jettyMapInfo.id = 0;
    jettyMapInfo.opDetailId = 0;
    jettyMapInfo.srcDieId = localRaCtxOpt->eidIndex;
    jettyMapInfo.dstDieId = rmtRaCtxOpt->eidIndex;
    jettyMapInfo.srcRankId = sim::GetRankIdByDeviceId(localRaCtxOpt->device_id);
    jettyMapInfo.dstRankId = sim::GetRankIdByDeviceId(rmtRaCtxOpt->device_id);
    memcpy(jettyMapInfo.leid, &localEndpointId, sizeof(localEndpointId));
    memcpy(jettyMapInfo.reid, &remoteEndpointId, sizeof(remoteEndpointId));
    jettyMapInfo.protocol = 0;
    auto ret = sim::InsertJettyMap(jettyMapInfo);
    if (ret != 0) {
        HCCL_VM_ERROR("insert jetty map failed");
        return -1;
    }
#endif

    *remQpHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(remoteQpId));
    HCCL_VM_INFO(
        "import remote QpId:{:d}, pair l:{:d}, r:{:d}, tp_type:{:d}", remoteQpId, localEndpointId, remoteEndpointId,
        qpInfo->in.ub.tpType);
    return 0;
}

int RaCtxQpImportAsync(void* ctxHandle, struct QpImportInfoT* info, void** remQpHandle, void** reqHandle)
{
    if (ctxHandle == nullptr || info == nullptr || remQpHandle == nullptr || reqHandle == nullptr) {
        HCCL_VM_ERROR(
            "invalid params, ctxHandle: {}, info: {}, remQpHandle: {}, reqHandle: {}", ctxHandle, (void*)info,
            (void*)remQpHandle, (void*)reqHandle);
        return -1;
    }

    auto ret = RaCtxQpImport(ctxHandle, info, remQpHandle);
    *reqHandle = reinterpret_cast<void*>(0x12345678);

    HCCL_VM_INFO(
        "ctx {:d} import QP {:d}", static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ctxHandle)),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(*remQpHandle)));
    return ret;
}

int RaCtxQpBind(void* qpHandle, void* remQpHandle)
{
    if (qpHandle == nullptr || remQpHandle == nullptr) {
        HCCL_VM_ERROR("invalid params, qpHandle: {}, remQpHandle: {}", qpHandle, remQpHandle);
        return -1;
    }

    uint64_t localQpId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(qpHandle));
    uint64_t remoteQpId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(remQpHandle));

    auto localQpOpt = RunnerDB::GetById<sim::RaJetty>(localQpId);
    if (!localQpOpt.has_value()) {
        HCCL_VM_ERROR("local QP {:d} not found", localQpId);
        return -1;
    }

    auto remoteQpOpt = RunnerDB::GetById<sim::RaJetty>(remoteQpId);
    if (!remoteQpOpt.has_value()) {
        HCCL_VM_ERROR("remote QP {:d} not found", remoteQpId);
        return -1;
    }

    RunnerDB::Update<sim::RaJetty>(localQpId, [remoteQpId](sim::RaJetty& jty) {
        jty.peer_jetty_handle = remoteQpId;
        jty.state = 3;
    });

    HCCL_VM_INFO("bind local QP {:d} remote QP {:d}", localQpId, remoteQpId);
    return 0;
}

int RaCtxQpUnimport(void* ctxHandle, void* remQpHandle)
{
    uint64_t localRaDevId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ctxHandle));
    uint64_t remoteQpId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(remQpHandle));
    auto remoteQpOpt = RunnerDB::GetById<sim::RaJetty>(remoteQpId);
    if (!remoteQpOpt.has_value()) {
        HCCL_VM_ERROR("remote QP {:d} not found", remoteQpId);
        return -1;
    }

    auto rmtRaCtxOpt = RunnerDB::GetById<sim::RaContext>(remoteQpOpt->ctx_handle);
    if (!rmtRaCtxOpt.has_value()) {
        HCCL_VM_ERROR("remote RaContext {:d} not found", remoteQpOpt->ctx_handle);
        return -1;
    }

    HCCL_VM_INFO("unimport rmtQp {:d}  pair l:{:d}, r:{:d}", remoteQpId, localRaDevId, rmtRaCtxOpt->id);
    return 0;
}

int RaTypicalQpModify(void* qpHandle, struct TypicalQp* localQpInfo, struct TypicalQp* remoteQpInfo)
{
    if (qpHandle == nullptr) {
        HCCL_VM_ERROR("qpHandle is null");
        return -1;
    }

    uint64_t qpId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(qpHandle));
    auto qpOpt = RunnerDB::GetById<sim::RaQP>(qpId);
    if (!qpOpt.has_value()) {
        HCCL_VM_ERROR("QP {:d} not found", qpId);
        return -1;
    }

    if (remoteQpInfo != nullptr) {
        RunnerDB::Update<sim::RaQP>(qpId, [remoteQpInfo](sim::RaQP& qp) {
            qp.state = 3;
            qp.peer_qpn = remoteQpInfo->qpn;
            qp.perr_lid = remoteQpInfo->psn;
        });
        HCCL_VM_INFO("QP {:d} modify to RTS, peer_qpn:{:d}, psn:{:d}", qpId, remoteQpInfo->qpn, remoteQpInfo->psn);
    } else {
        RunnerDB::Update<sim::RaQP>(qpId, [](sim::RaQP& qp) {
            qp.state = 3;
        });
        HCCL_VM_INFO("QP {:d} modify to RTS", qpId);
    }

    return 0;
}

int RaTypicalSendWr(void* qpHandle, struct SendWr* wr, struct SendWrRsp* opRsp)
{
    if (qpHandle == nullptr || wr == nullptr) {
        HCCL_VM_ERROR("invalid params, qpHandle: {}, wr: {}", qpHandle, (void*)wr);
        return -1;
    }

    uint64_t qpId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(qpHandle));
    auto qpOpt = RunnerDB::GetById<sim::RaQP>(qpId);
    if (!qpOpt.has_value()) {
        HCCL_VM_ERROR("QP {:d} not found", qpId);
        return -1;
    }

    if (qpOpt->state != 3) {
        HCCL_VM_ERROR("QP {:d} not in RTS state", qpId);
        return -1;
    }

    HCCL_VM_INFO("QP {:d} typical send op:{:d}, bufNum:{:d}", qpId, wr->op, wr->bufNum);
    return 0;
}

int RaRdevGetPortStatus(void* rdmaHandle, enum PortStatus* status)
{
    if (rdmaHandle == nullptr || status == nullptr) {
        HCCL_VM_ERROR("invalid params, rdmaHandle: {}, status: {}", rdmaHandle, (void*)status);
        return -1;
    }

    *status = PORT_STATUS_ACTIVE;
    HCCL_VM_INFO("rdmaHandle:{:d} port status:UP", static_cast<uint64_t>(reinterpret_cast<uintptr_t>(rdmaHandle)));
    return 0;
}

int RaGetQpAttr(void* qpHandle, struct QpAttr* attr)
{
    if (qpHandle == nullptr || attr == nullptr) {
        HCCL_VM_ERROR("invalid params, qpHandle: {}, attr: {}", qpHandle, (void*)attr);
        return -1;
    }

    uint64_t qpId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(qpHandle));
    auto qpOpt = RunnerDB::GetById<sim::RaQP>(qpId);
    if (!qpOpt.has_value()) {
        HCCL_VM_ERROR("QP {:d} not found", qpId);
        return -1;
    }

    attr->qpn = qpOpt->qp_num;
    attr->udpSport = 0;
    attr->psn = 0;
    attr->gidIdx = 0;
    memset(attr->gid, 0, sizeof(attr->gid));

    HCCL_VM_INFO("QP {:d} qpn:{:d}", qpId, attr->qpn);
    return 0;
}

int RaSocketWhiteListAdd(void* socketHandle, struct SocketWlistInfoT whiteList[], unsigned int num)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaSocketWhiteListDel(void* socketHandle, struct SocketWlistInfoT whiteList[], unsigned int num)
{
    HCCL_VM_WARN("is empty");
    return 0;
}
int RaSocketAcceptCreditAdd(struct SocketListenInfoT conn[], unsigned int num, unsigned int creditLimit)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaGetIfnum(struct RaGetIfattr* config, unsigned int* num)
{
    sim::Device device{};
    if (GetDeviceByPhysicalId(config->phyId, device) != ACL_SUCCESS) {
        HCCL_VM_ERROR("get device by logic id {} failed.", config->phyId);
        return -1;
    }

    uint64_t deviceIdx = device.id;
    auto endPoints = RunnerDB::GetByPred<sim::EndPoint>([deviceIdx](const sim::EndPoint& ep) {
        return ep.device_id == deviceIdx;
    });

    *num = endPoints.size();
    HCCL_VM_WARN("Get num:{:d}", *num);
    return 0;
}

int RaGetIfaddrs(struct RaGetIfattr* config, struct InterfaceInfo interfaceInfos[], unsigned int* num)
{
    sim::Device device{};
    if (GetDeviceByPhysicalId(config->phyId, device) != ACL_SUCCESS) {
        HCCL_VM_ERROR("get device by logic id {} failed.", config->phyId);
        return -1;
    }

    uint64_t deviceIdx = device.id;
    auto endPoints = RunnerDB::GetByPred<sim::EndPoint>([deviceIdx](const sim::EndPoint& ep) {
        return ep.device_id == deviceIdx;
    });

    *num = endPoints.size();
    for (unsigned int i = 0; i < *num; i++) {
        interfaceInfos[i].family = AF_INET6;
        sprintf(interfaceInfos[i].ifname, "%s", endPoints.at(i).ip_addr);
        interfaceInfos[i].scopeId = 0;
        std::string ipaddr = std::string(endPoints.at(i).ip_addr);
        if (ipaddr.find(':') == std::string::npos) {
            ipaddr = "::" + ipaddr;
        }
        IpAddress addr(ipaddr, AF_INET6);
        interfaceInfos[i].ifaddr.ip.addr6 = addr.GetBinaryAddress().addr6;

        HCCL_VM_INFO("get Ip addr is ipaddr[{:d}]={}", i, endPoints.at(i).ip_addr);
    }
    return 0;
}

int RaTlvInit(struct TlvInitInfo* initInfo, unsigned int* bufferSize, void** tlvHandle)
{
    auto phyId = initInfo->phyId;
    sim::RaTlv raTlv{};
    raTlv.physical_id = phyId;
    auto tlvId = RunnerDB::Add<sim::RaTlv>(raTlv);
    *tlvHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(tlvId));
    return 0;
}

int RaTlvDeinit(void* tlvHandle)
{
    auto tlvId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tlvHandle));
    RunnerDB::Delete<sim::RaTlv>(tlvId);
    return 0;
}

struct ccu_mem_info {
    unsigned int long long mem_va;
    unsigned int mem_size;
    unsigned int resv[1];
};

struct ccu_mem_rsp {
    unsigned int die_id;
    unsigned int num;
    struct ccu_mem_info list[64U];
};

int GetCcuMemInfo(struct TlvMsg* sendMsg, struct TlvMsg* recvMsg)
{
    auto sendData = (CcuMemReq*)sendMsg->data;
    auto dieId = sendData->udieIdx;

    struct ccu_mem_rsp rsp {};
    rsp.die_id = dieId;
    rsp.num = 0;

    uint64_t ccu_die0_base_addr = 0x123123123;
    uint64_t ccu_die1_base_addr = 0x456456456;

    // ccu指令空间： 1MByte
    ccu_mem_info ccu_inst_mem = {0, 0, {0}};
    if (dieId == 0) {
        ccu_inst_mem.mem_va = ccu_die0_base_addr;
    } else if (dieId == 1) {
        ccu_inst_mem.mem_va = ccu_die1_base_addr;
    }
    ccu_inst_mem.mem_size = 0x100000;
    rsp.list[rsp.num++] = ccu_inst_mem;

    uint64_t ccu_die0_gsa_offset = ccu_die0_base_addr + 0x100000;
    uint64_t ccu_die1_gsa_offset = ccu_die1_base_addr + 0x100000;
    // ccu gsa寄存器： size = 24KByte, 实际预留32KByte
    ccu_mem_info ccu_gsa_mem = {0, 0, {0}};
    if (dieId == 0) {
        ccu_gsa_mem.mem_va = ccu_die0_gsa_offset;
    } else if (dieId == 1) {
        ccu_gsa_mem.mem_va = ccu_die1_gsa_offset;
    }
    ccu_gsa_mem.mem_size = 0x6000;
    rsp.list[rsp.num++] = ccu_gsa_mem;

    // ccu Xn寄存器： size = 24KByte, 实际预留32KByte
    uint64_t ccu_die0_xn_offset = ccu_die0_gsa_offset + 0x8000;
    uint64_t ccu_die1_xn_offset = ccu_die1_gsa_offset + 0x8000;
    ccu_mem_info ccu_xn_mem = {0, 0, {0}};
    if (dieId == 0) {
        ccu_xn_mem.mem_va = ccu_die0_xn_offset;
    } else if (dieId == 1) {
        ccu_xn_mem.mem_va = ccu_die1_xn_offset;
    }
    ccu_xn_mem.mem_size = 0x6000;
    rsp.list[rsp.num++] = ccu_xn_mem;

    // ccu CKB寄存器： size = 8KByte, 实际预留32KByte
    uint64_t ccu_die0_ckb_offset = ccu_die0_xn_offset + 0x8000;
    uint64_t ccu_die1_ckb_offset = ccu_die1_xn_offset + 0x8000;
    ccu_mem_info ccu_ckb_mem = {0, 0, {0}};
    if (dieId == 0) {
        ccu_ckb_mem.mem_va = ccu_die0_ckb_offset;
    } else if (dieId == 1) {
        ccu_ckb_mem.mem_va = ccu_die1_ckb_offset;
    }
    ccu_ckb_mem.mem_size = 0x2000;
    rsp.list[rsp.num++] = ccu_ckb_mem;

    // ccu PFE表： size = 256Byte, 实际预留32KByte
    uint64_t ccu_die0_pfe_offset = ccu_die0_ckb_offset + 0x8000;
    uint64_t ccu_die1_pfe_offset = ccu_die1_ckb_offset + 0x8000;
    ccu_mem_info ccu_pfe_mem = {0, 0, {0}};
    if (dieId == 0) {
        ccu_pfe_mem.mem_va = ccu_die0_pfe_offset;
    } else if (dieId == 1) {
        ccu_pfe_mem.mem_va = ccu_die1_pfe_offset;
    }
    ccu_pfe_mem.mem_size = 0x100;
    rsp.list[rsp.num++] = ccu_pfe_mem;

    // ccu channel映射表： size = 8KByte, 实际预留32KByte
    uint64_t ccu_die0_channel_offset = ccu_die0_pfe_offset + 0x8000;
    uint64_t ccu_die1_channel_offset = ccu_die1_pfe_offset + 0x8000;
    ccu_mem_info ccu_channel_mem = {0, 0, {0}};
    if (dieId == 0) {
        ccu_channel_mem.mem_va = ccu_die0_channel_offset;
    } else if (dieId == 1) {
        ccu_channel_mem.mem_va = ccu_die1_channel_offset;
    }
    ccu_channel_mem.mem_size = 0x2000;
    rsp.list[rsp.num++] = ccu_channel_mem;

    // ccu jetty context表： size = 4KByte, 实际预留32KByte
    uint64_t ccu_die0_ctx_offset = ccu_die0_channel_offset + 0x8000;
    uint64_t ccu_die1_ctx_offset = ccu_die1_channel_offset + 0x8000;
    ccu_mem_info ccu_ctx_mem = {0, 0, {0}};
    if (dieId == 0) {
        ccu_ctx_mem.mem_va = ccu_die0_ctx_offset;
    } else if (dieId == 1) {
        ccu_ctx_mem.mem_va = ccu_die1_ctx_offset;
    }
    ccu_ctx_mem.mem_size = 0x1000;
    rsp.list[rsp.num++] = ccu_ctx_mem;

    // ccu mission context表： size = 1KByte, 实际预留32KByte
    uint64_t ccu_die0_mission_offset = ccu_die0_ctx_offset + 0x8000;
    uint64_t ccu_die1_mission_offset = ccu_die1_ctx_offset + 0x8000;
    ccu_mem_info ccu_mission_mem = {0, 0, {0}};
    if (dieId == 0) {
        ccu_mission_mem.mem_va = ccu_die0_mission_offset;
    } else if (dieId == 1) {
        ccu_mission_mem.mem_va = ccu_die1_mission_offset;
    }
    ccu_mission_mem.mem_size = 0x1000;
    rsp.list[rsp.num++] = ccu_mission_mem;

    // ccu loop context表： size = 12.5KB, 实际预留32KByte
    uint64_t ccu_die0_loop_offset = ccu_die0_mission_offset + 0x8000;
    uint64_t ccu_die1_loop_offset = ccu_die1_mission_offset + 0x8000;
    ccu_mem_info ccu_loop_mem = {0, 0, {0}};
    if (dieId == 0) {
        ccu_loop_mem.mem_va = ccu_die0_loop_offset;
    } else if (dieId == 1) {
        ccu_loop_mem.mem_va = ccu_die1_loop_offset;
    }
    ccu_loop_mem.mem_size = 0x3000;
    rsp.list[rsp.num++] = ccu_loop_mem;

    memcpy(recvMsg->data, &rsp, sizeof(rsp));
    return 0;
}

int RaTlvRequest(void* tlvHandle, unsigned int moduleType, struct TlvMsg* sendMsg, struct TlvMsg* recvMsg)
{
    HCCL_VM_INFO("Enter into tlv request...module: {}, msg:{}", moduleType, static_cast<int>(sendMsg->type));
    switch (sendMsg->type) {
        case TlvCcuMsgType::MSG_TYPE_CCU_INIT:
            HCCL_VM_WARN("not support CCU_INIT");
            return 0;
        case TlvCcuMsgType::MSG_TYPE_CCU_UNINIT:
            HCCL_VM_WARN("not support CCU_UNINIT");
            return 0;
        case TlvCcuMsgType::MSG_TYPE_CCU_GET_MEM_INFO:
            GetCcuMemInfo(sendMsg, recvMsg);
            break;
        case TlvCcuMsgType::MSG_TYPE_CCU_DISPATCH_CMD:
            SimRaCustomChannel(tlvHandle, sendMsg, recvMsg);
            break;
        default:
            break;
    }
    return 0;
}

int RaPingInit(struct PingInitAttr* initAttr, struct PingInitInfo* initInfo, void** pingHandle)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaPingDeinit(void* pingHandle)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaPingTargetAdd(void* pingHandle, struct PingTargetInfo target[], uint32_t num)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaPingTaskStart(void* pingHandle, struct PingTaskAttr* attr)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaPingGetResults(void* pingHandle, struct PingTargetResult target[], uint32_t* num)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaPingTargetDel(void* pingHandle, struct PingTargetCommInfo target[], uint32_t num)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaPingTaskStop(void* pingHandle)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaCtxTokenIdAlloc(void* ctxHandle, struct HccpTokenId* info, void** tokenIdHandle)
{
    if (ctxHandle == nullptr || info == nullptr || tokenIdHandle == nullptr) {
        HCCL_VM_ERROR(
            "invalid params, ctxHandle: {}, info: {}, tokenIdHandle: {}", ctxHandle, (void*)info, (void*)tokenIdHandle);
        return -1;
    }
    sim::RaTokenId tokenId{};
    tokenId.ctx_handle = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ctxHandle));
    tokenId.token_id = ((uint32_t)(tokenId.ctx_handle >> 32)) ^ ((uint32_t)rand());
    info->tokenId = tokenId.token_id;
    auto id = RunnerDB::Add<sim::RaTokenId>(tokenId);
    *tokenIdHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(id));
    return 0;
}

int RaGetSecRandom(struct RaInfo* info, uint32_t* value)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaCtxLmemUnregister(void* ctxHandle, void* lmemHandle)
{
    if (ctxHandle == nullptr || lmemHandle == nullptr) {
        HCCL_VM_ERROR("invalid params, ctxHandle: {}, lmemHandle: {}", ctxHandle, lmemHandle);
        return -1;
    }
    uint64_t id = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(lmemHandle));
    RunnerDB::Delete<sim::RaLmem>(id);
    HCCL_VM_INFO("delete lmem id {:d}, for checker use after", id);

    return 0;
}

int RaCtxRmemUnimport(void* ctxHandle, void* rmemHandle)
{
    if (ctxHandle == nullptr || rmemHandle == nullptr) {
        HCCL_VM_ERROR("invalid params, ctxHandle: {}, rmemHandle: {}", ctxHandle, rmemHandle);
        return -1;
    }
    uint64_t id = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(rmemHandle));
    RunnerDB::Delete<sim::RaRmem>(id);
    HCCL_VM_INFO("delete rmem id {:d}, for checker use after", id);
    return 0;
}

int RaCtxQpUnimportAsync(void* remQpHandle, void** reqHandle)
{
    *reqHandle = reinterpret_cast<void*>(0x12345678);
    return 0;
}

int RaCtxLmemUnregisterAsync(void* ctxHandle, void* lmemHandle, void** reqHandle)
{
    *reqHandle = reinterpret_cast<void*>(0x12345678);
    return 0;
}

int RaCtxQpDestroyAsync(void* qpHandle, void** reqHandle)
{
    if (qpHandle == nullptr || reqHandle == nullptr) {
        HCCL_VM_ERROR("invalid params, qpHandle: {}, reqHandle: {}", qpHandle, (void*)reqHandle);
        return -1;
    }

    int ret = RaCtxQpDestroy(qpHandle);
    *reqHandle = reinterpret_cast<void*>(0x12345678);

    HCCL_VM_INFO(
        "destroy QP {:d}, reqHandle:{:x}", static_cast<uint64_t>(reinterpret_cast<uintptr_t>(qpHandle)),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(*reqHandle)));
    return ret;
}

int RaCtxQpDestroyBatchAsync(void* ctxHandle, void* qpHandle[], unsigned int* num, void** reqHandle)
{
    HCCL_VM_INFO("enter num {:d}", *num);
    int ret = 0;
    for (uint32_t i = 0; i < *num; i++) {
        ret |= RaCtxQpDestroyAsync(qpHandle[i], reqHandle);
    }
    return ret;
}

int RaCtxRmemImport(void* ctxHandle, struct MrImportInfoT* rmemInfo, void** rmemHandle)
{
    uint64_t remoteMemId = *(uint64_t*)(rmemInfo->in.key.value);
    auto remoteMemOpt = RunnerDB::GetById<sim::RaLmem>(remoteMemId);
    if (!remoteMemOpt.has_value()) {
        HCCL_VM_ERROR("remote Mem {:d} not found", remoteMemId);
        return -1;
    }
    *rmemHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(remoteMemId));

    auto rmtRaCtxOpt = RunnerDB::GetById<sim::RaContext>(remoteMemOpt->ctx_handle);
    if (!rmtRaCtxOpt.has_value()) {
        HCCL_VM_ERROR("remote RaContext {:d} not found", remoteMemOpt->ctx_handle);
        return -1;
    }

    sim::RaRmem mem{};
    mem.ctx_handle = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ctxHandle));
    mem.remote_key = remoteMemOpt->mem_key;
    mem.target_seg_handle = remoteMemId;
    mem.remote_eid = rmtRaCtxOpt->endpoint_id;
    auto id = RunnerDB::Add<sim::RaRmem>(mem);
    return 0;
}

int RaCtxChanCreate(void* ctxHandle, struct ChanInfoT* chanInfo, void** chanHandle)
{
    uint64_t raCtxId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ctxHandle));
    sim::RaChan ch{};
    ch.ctx_handle = raCtxId;
    auto id = RunnerDB::Add<sim::RaChan>(ch);
    *(chanHandle) = reinterpret_cast<void*>(static_cast<uintptr_t>(id));

    HCCL_VM_WARN("RaChan {:d} add RaChan id:{:d}", raCtxId, id);
    return 0;
}

int RaCtxChanDestroy(void* ctxHandle, void* chanHandle)
{
    uint64_t id = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ctxHandle));
    HCCL_VM_INFO("soft delete chan id {:d}, for checker use after", id);
    return 0;
}

int RaCtxQpDestroy(void* qpHandle)
{
    if (qpHandle == nullptr) {
        HCCL_VM_ERROR("qpHandle is null");
        return -1;
    }

    uint64_t id = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(qpHandle));
    HCCL_VM_INFO("soft delete jetty id {:d}, for checker use after", id);
    return 0;
}

int RaCtxTokenIdFree(void* ctxHandle, void* tokenIdHandle)
{
    if (ctxHandle == nullptr || tokenIdHandle == nullptr) {
        HCCL_VM_ERROR("invalid params, ctxHandle: {}, tokenIdHandle: {}", ctxHandle, tokenIdHandle);
        return -1;
    }
    uint64_t id = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tokenIdHandle));
    HCCL_VM_INFO("soft delete token id {:d}, for checker use after", id);
    return 0;
}

int RaCtxLmemRegister(void* ctxHandle, struct MrRegInfoT* lmemInfo, void** lmemHandle)
{
    if (ctxHandle == nullptr || lmemInfo == nullptr || lmemHandle == nullptr) {
        HCCL_VM_ERROR(
            "invalid params, ctxHandle: {}, lmemInfo: {}, lmemHandle: {}", ctxHandle, (void*)lmemInfo,
            (void*)lmemHandle);
        return -1;
    }
    sim::RaLmem memInfo{};
    memInfo.ctx_handle = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ctxHandle));
    memInfo.addr = lmemInfo->in.mem.addr;
    memInfo.size = lmemInfo->in.mem.size;
    uint64_t token = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(lmemInfo->in.ub.tokenIdHandle));
    auto tokenInfo = RunnerDB::GetById<sim::RaTokenId>(token);
    if (!tokenInfo.has_value()) {
        HCCL_VM_ERROR("tokenHandle {:d} not found", token);
        return -1;
    }
    memInfo.token_id = tokenInfo->token_id;
    memInfo.mem_key = (token << 32) | (rand() & 0xFFFFFFFF);
    auto id = RunnerDB::Add<sim::RaLmem>(memInfo);
    *reinterpret_cast<uint64_t*>(lmemInfo->out.key.value) = id;
    lmemInfo->out.key.size = sizeof(uint64_t);
    lmemInfo->out.ub.tokenId = memInfo.token_id;
    lmemInfo->out.ub.targetSegHandle = id;

    *lmemHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(id));
    return 0;
}

int RaCtxLmemRegisterAsync(void* ctxHandle, struct MrRegInfoT* lmemInfo, void** lmemHandle, void** reqHandle)
{
    HCCL_VM_ERROR("not support");
    return -1;
}

int RaGetTpInfoListAsync(
    void* ctxHandle, struct GetTpCfg* cfg, struct HccpTpInfo infoList[], unsigned int* num, void** reqHandle)
{
    *reqHandle = reinterpret_cast<void*>(0x12345678);
    *num = 1;
    infoList[0].tpHandle = 0x12345678;
    return 0;
}

int RaGetEidByIpAsync(void* ctxHandle, struct IpInfo ip[], union HccpEid eid[], unsigned int* num, void** reqHandle)
{
    HCCL_VM_ERROR("not support");
    return -1;
}

int RaGetTpAttrAsync(void* ctxHandle, uint64_t tpHandle, uint32_t* attrBitmap, struct TpAttr* attr, void** reqHandle)
{
    *reqHandle = reinterpret_cast<void*>(0x12345678);
    attr->slBitmap = 0xe;
    return 0;
}

int RaCtxQpQueryBatch(void* qpHandle[], struct JettyAttr attr[], unsigned int* num)
{
    HCCL_VM_ERROR("not support");
    return -1;
}

int RaCtxQpUnbind(void* qpHandle)
{
    if (qpHandle == nullptr) {
        HCCL_VM_ERROR("qpHandle is null");
        return -1;
    }

    uint64_t jtyId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(qpHandle));
    auto jtyOpt = RunnerDB::GetById<sim::RaJetty>(jtyId);
    if (!jtyOpt.has_value()) {
        HCCL_VM_WARN("QP {:d} not found", jtyId);
        return 0;
    }

    RunnerDB::Update<sim::RaJetty>(jtyId, [](sim::RaJetty& jty) {
        jty.peer_jetty_handle = 0;
        jty.state = 0;
    });

    HCCL_VM_INFO("unbind QP {:d}, reset to RESET state", jtyId);
    return 0;
}

int RaBatchSendWr(
    void* qpHandle, struct SendWrData wrList[], struct SendWrResp opResp[], unsigned int num, unsigned int* completeNum)
{
    HCCL_VM_ERROR("not support");
    return -1;
}

int RaCtxCqCreate(void* ctxHandle, struct CqInfoT* info, void** cqHandle)
{
    *cqHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(0xabcdU));
    return 0;
}

int RaCtxCqDestroy(void* ctxHandle, void* cqHandle)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaCtxUpdateCi(void* qpHandle, uint16_t ci)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int ra_get_async_req_result(void* req_handle, int* req_result)
{
    if (req_handle == nullptr || req_result == nullptr) {
        HCCL_VM_ERROR("invalid params, req_handle: {}, req_result: {}", req_handle, (void*)req_result);
        return -1;
    }

    *req_result = 1;
    HCCL_VM_INFO("req_handle:{:x} result:completed", static_cast<uint64_t>(reinterpret_cast<uintptr_t>(req_handle)));
    return 0;
}

int ra_get_qp_context(void* qpHandle, void** qp, void** sendCq, void** recvCq)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int ra_get_tsqp_depth(void* rdev_handle, unsigned int* temp_depth, unsigned int* qp_num)
{
    *temp_depth = 1;
    *qp_num = 1;
    return 0;
}

int ra_set_tsqp_depth(void* rdev_handle, unsigned int temp_depth, unsigned int* qp_num)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int ra_get_notify_mr_info(void* handle, struct mr_info* mrInfo)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int ra_send_wrlist_ext(
    void* qp_handle, struct send_wrlist_data_ext wr[], struct send_wr_rsp op_rsp[], unsigned int send_num,
    unsigned int* complete_num)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int ra_register_mr(const void* handle, struct mr_info* mrInfo, void** mrHandle)
{
    *mrHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(0xabcdU));
    return ((handle == nullptr) || (mrInfo == nullptr)) ? -1 : 0;
}

int ra_deregister_mr(const void* handle, void* mrHandle)
{
    return ((handle == nullptr) || (mrHandle == nullptr)) ? -1 : 0;
}

int ra_is_first_used(int ins_id)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int ra_epoll_ctl_add(const void* fd_handle, RaEpollEvent event)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int ra_epoll_ctl_mod(const void* fd_handle, RaEpollEvent event)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int ra_epoll_ctl_del(const void* fd_handle)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaCtxQpCreateAsync(
    void* ctxHandle, struct QpCreateAttr* attr, struct QpCreateInfo* info, void** qpHandle, void** reqHandle)
{
    if (ctxHandle == nullptr || attr == nullptr || qpHandle == nullptr || reqHandle == nullptr) {
        HCCL_VM_ERROR(
            "invalid params, ctxHandle: {}, attr: {}, qpHandle: {}, reqHandle: {}", ctxHandle, (void*)attr,
            (void*)qpHandle, (void*)reqHandle);
        return -1;
    }

    int ret = RaCtxQpCreate(ctxHandle, attr, info, qpHandle);
    if (ret != 0) {
        HCCL_VM_ERROR("RaCtxQpCreate failed");
        return ret;
    }

    *reqHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(0x12345678ULL));
    HCCL_VM_INFO(
        "ctx {:d} async create QP {:d}, reqHandle:{:x}, jettyId:{:d}, jettyMode:{:d}",
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ctxHandle)),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(*qpHandle)),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(*reqHandle)), attr->ub.jettyId,
        static_cast<uint32_t>(attr->ub.mode));

    return 0;
}

int RaSetTpAttrAsync(void* ctxHandle, uint64_t tpHandle, uint32_t attrBitmap, struct TpAttr* attr, void** reqHandle)
{
    *reqHandle = reinterpret_cast<void*>(0x12345678);
    return 0;
}

int RaCtxGetAuxInfo(void* ctxHandle, struct HccpAuxInfoIn* in, struct HccpAuxInfoOut* out)
{
    HCCL_VM_ERROR("Not support yet");
    return -1;
}

int RaCtxGetCrErrInfoList(void* ctxHandle, struct CrErrInfo* infoList, unsigned int* num)
{
    HCCL_VM_ERROR("Not support yet");
    return -1;
}

TraStatus AtraceSubmit(TraHandle handle, const void* buffer, uint32_t bufSize)
{
    if (handle == 0) {
        return 0;
    } else if (handle == 1) {
        HCCL_VM_ERROR("invalid params, handle: {}", handle);
        return -1;
    }
    return 0;
}

rtError_t rtPointerGetAttributes(rtPointerAttributes_t* attributes, const void* ptr)
{
    HCCL_VM_WARN("is empty");
    return 0;
}

rtError_t rtStreamGetCqid(const rtStream_t stm, uint32_t* cqId, uint32_t* logicCqId)
{
    static uint32_t i = 0U;
    *logicCqId = i++;
    return 0;
}

int RaGetEidByIp(void* ctxHandle, struct IpInfo ip[], union HccpEid eid[], unsigned int* num)
{
    if (ctxHandle == nullptr || ip == nullptr || eid == nullptr || num == nullptr) {
        HCCL_VM_ERROR(
            "invalid params, ctxHandle: {:p}, ip: {:p}, eid: {:p}, num: {}", ctxHandle, (void*)ip, (void*)eid,
            (void*)num);
        return -1;
    }

    if (*num == 0 || *num > HCCP_EID_IP_QUERY_MAX_NUM) {
        HCCL_VM_ERROR("num({}) must be in range [1, {}]", *num, HCCP_EID_IP_QUERY_MAX_NUM);
        return -1;
    }

    for (uint32_t i = 0; i < *num; i++) {
        BinaryAddr ipAddr;
        ipAddr.addr = ip[i].ip.addr;
        ipAddr.addr6 = ip[i].ip.addr6;

        IpAddress addr(ipAddr, ip[i].family);
        auto simEid = addr.GetEid();
        for (uint32_t j = 0; j < URMA_EID_LEN; j++) {
            eid[i].raw[j] = simEid.raw[j];
        }
    }

    return 0;
}

int RaGetIpByEid(void* ctxHandle, union HccpEid eid[], struct IpInfo ip[], unsigned int* num)
{
    if (ctxHandle == nullptr || ip == nullptr || eid == nullptr || num == nullptr) {
        HCCL_VM_ERROR(
            "invalid params, ctxHandle: {:p}, ip: {:p}, eid: {:p}, num: {}", ctxHandle, (void*)ip, (void*)eid,
            (void*)num);
        return -1;
    }

    if (*num == 0 || *num > HCCP_EID_IP_QUERY_MAX_NUM) {
        HCCL_VM_ERROR("num({}) must be in range [1, {}]", *num, HCCP_EID_IP_QUERY_MAX_NUM);
        return -1;
    }

    for (uint32_t i = 0; i < *num; i++) {
        Eid simEid;
        for (uint32_t j = 0; j < URMA_EID_LEN; j++) {
            simEid.raw[j] = eid[i].raw[j];
        }
        auto ipAddr = IpAddress(simEid);
        auto ipInfo = ipAddr.GetBinaryAddress();
        ip[i].ip.addr = ipInfo.addr;
        ip[i].ip.addr6 = ipInfo.addr6;
        ip[i].family = ipAddr.GetFamily();
    }

    return 0;
}

int RaGetIpByEidAsync(void* ctxHandle, union HccpEid eid[], struct IpInfo ip[], unsigned int* num, void** reqHandle)
{
    HCCL_VM_WARN("Enter RaGetIpByEidAsync");
    if (reqHandle == nullptr) {
        HCCL_VM_ERROR("invalid params, reqHandle: {:p}", (void*)reqHandle);
        return -1;
    }
    return RaGetIpByEid(ctxHandle, eid, ip, num);
}

#ifdef __cplusplus
}
#endif // __cplusplus
