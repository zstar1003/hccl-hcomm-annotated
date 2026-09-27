/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <adapter_hccp.h>
#include <securec.h>
#include <unordered_map>
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <ifaddrs.h>
#include <adapter_rts.h>
#include <mutex>
#include <memory>
#include <unordered_set>

#include "network/hccp_common.h"
#include "externalinput.h"
#include "dlra_function.h"
#include "log.h"
#include "../host/transport_ibverbs_pub.h"
#include "config_plf_log.h"

using namespace hccl;
using namespace std;

/* 检查函数返回值是否为ROCE_ENOMEM_RET, 记录指定日志, 并返回HCCL_E_OOM, 内存大小取决于qp深度配置 */
#define CHK_OOM_RET(ret, qpInfo)                                                                                   \
    do {                                                                                                           \
        if ((ret) == ROCE_ENOMEM_RET) {                                                                            \
            RPT_ENV_ERR(                                                                                           \
                true, "EI0011", std::vector<std::string>({"memory_size"}),                                         \
                std::vector<std::string>({"262144~3145728"}));                                                     \
            HCCL_ERROR(                                                                                            \
                "[%s] ra qp create fail, reason: out of memory. qpInfo:[%s], return: ret[%d]", __func__, (qpInfo), \
                (ret));                                                                                            \
            return HCCL_E_OOM;                                                                                     \
        }                                                                                                          \
    } while (0)

constexpr u32 MAX_NUM_OF_BATCH_CONN = 16;
constexpr u32 MAX_CQ_DEPTH = 65535;
constexpr u32 MAX_INLINE_DATA = 128;
constexpr u32 MAX_WR_NUM = 1024;
constexpr u32 MAX_RECV_SGE_NUM = 1;
constexpr u32 REPEAT_RAINIT_ERROR_CODE = 328002;
constexpr u32 REPEAT_LISTEN_ERROR_CODE = 128205;

// network 获取版本信息参数
constexpr u32 SOCKET_BATCH_CLOSE_INTERFACE = 1;
constexpr u32 SOCKET_BATCH_CLOSE_SUP_VER = 2;
constexpr u32 QP_ATTR_QOS_INTERFACE = 29;       // RA_RS_SET_QP_ATTR_QOS 的 opcode为29
constexpr u32 QP_ATTR_TIMEOUT_INTERFACE = 30;   // RA_RS_SET_QP_ATTR_TIMEOUT 的 opcode为30
constexpr u32 QP_ATTR_RETRY_CNT_INTERFACE = 31; // RA_RS_SET_QP_ATTR_RETRY_CNT 的 opcode为31
constexpr u32 QP_ATTR_QOS_SUP_VER = 1;          // 当前支持的版本号为1

constexpr u32 IFNUM_INTERFACE = 33;        // RA_RS_GET_IFNUM的opcode为33
constexpr u32 IFNUM_INTERFACE_VERSION = 1; // 支持的RA_RS_GET_IFNUM_VERSION为1

constexpr u32 IFADDRS_V2_INTERFACE = 38;         // RA_RS_GET_IFADDRS_V2的opcode为38
constexpr u32 IFADDRS_V2_INTERFACE_VERSTOIN = 3; // 支持获取chip上所有ip addr的IFADDRS_V2_INTERFACE_VERSTOIN为3

constexpr u32 RDEV_INIT_WITH_BACKUP = 81;        // RA_RS_RDEV_INIT_WITH_BACKUP的opcode为81
constexpr u32 RDEV_INIT_WITH_BACKUP_SUP_VER = 1; // 当前支持的版本号为1

constexpr u32 ALL_NIC_NUM_910_93 = 2;               // 910_93 上最大网卡数量
constexpr u32 ALL_NIC_NUM_910_A2 = 1;               // 910 A2 上最大网卡数量
constexpr u32 MAX_ALL_NIC_NUM = ALL_NIC_NUM_910_93; // 最大可能的网卡数量

constexpr u32 QP_ATTR_TIMEOUT_SUPPORT_VER = 1;   // 当前支持配置RDMA TimeOut的版本号为1
constexpr u32 QP_ATTR_RETRY_CNT_SUPPORT_VER = 1; // 当前支持配置RDMA RetryCnt的版本号为1

constexpr u32 CQE_ERR_INFO_INTERFACE = 32;      // RA_RS_GET_CQE_ERR_INFO 的 opcode为32
constexpr u32 CQE_ERR_INFO_LIST_INTERFACE = 80; // RA_RS_GET_CQE_ERR_INFO_LIST 的 opcode为80
constexpr u32 CQE_ERR_INFO_SUP_VER = 1;         // 当前支持的版本号为1

constexpr u32 QP_CREATE_WITH_ATTRS_INTERFACE = 39; // RA_RS_QP_CREATE_WITH_ATTRS 的 opcode为39
constexpr u32 QP_CREATE_WITH_ATTRS_SUP_VER = 1;    // 当前支持的版本号为1

constexpr u32 SOCKET_VNIC_IP_INFOS_INTERFACE = 55; // RA_RS_GET_VNIC_IP_INFOS  的 opcode为55
constexpr u32 SOCKET_VNIC_IP_INFOS_SUP_VER = 1;    // 当前支持的版本号为1

constexpr u32 GET_NOTIFY_BA = 14;        // RA_RS_GET_NOTIFY_BA 的 opcode为14
constexpr u32 GET_NOTIFY_BA_VERSION = 2; // 当前支持的版本号为2

constexpr u32 SEND_NORMAL_WRLIST = 83;
constexpr u32 SEND_NORMAL_WRLIST_VERSION = 1;

constexpr u32 TLV_INIT = 87;
constexpr u32 TLV_DEINIT = 88;
constexpr u32 TLV_REQUEST = 89;
constexpr u32 TLV_VERSION = 1;

constexpr u32 GET_TLS_ENABLE = 95;
constexpr u32 TLS_ENABLE_VERSION = 1;
// handle ref
constexpr u32 FIRST_HANDLE_REF = 1;

constexpr s32 HCCL_SEND_CQ_DEPTH_DEFAULT = (8 * 1024); // HCCL 默认的scq深度

constexpr u32 TYPICAL_QP_MODIFY = 46;        // opcode: RA_RS_TYPICAL_QP_MODIFY
constexpr u32 TYPICAL_QP_MODIFY_VERSION = 2; // 支持QP解耦socket建链版本号

constexpr u32 SOCKET_ABORT = 97;        // opcode: RA_RS_SOCKET_ABORT
constexpr u32 SOCKET_ABORT_VERSION = 1; // 支持socket abort的版本号

constexpr u32 RS_INIT = 15;                      // opcode: RA_RS_INIT
constexpr u32 RS_INIT_SUPPORT_ASYNC_VERSION = 2; // 支持socket async的版本号

constexpr u32 ROCE_ENOMEM_RET = 328100; // 创建qp时由于内存不足的错误返回值

template <typename T>
struct HandleInfo {
    std::mutex handleMutex;
    std::unordered_map<u32, T> handleMap;
    std::unordered_map<T, u32> handleRef;
};

HandleInfo<SocketHandle> g_socketHandleInfo;
HandleInfo<RdmaHandle> g_rdmaHandleInfo;

#if T_DESC("RDMA异步", true)
HcclResult HrtRaQpCreate(RdmaHandle rdmaHandle, int flag, int qpMode, QpHandle& qpHandle)
{
    string qpInfo = string("rdmaHandle:") + to_string(reinterpret_cast<intptr_t>(rdmaHandle)) + string("qpHandle:")
                    + to_string(reinterpret_cast<intptr_t>(&qpHandle)) + string("flag:") + to_string(flag)
                    + string("qpMode:") + to_string(qpMode);

    s32 ret = DlRaFunction::GetInstance().dlRaQpCreate(rdmaHandle, flag, qpMode, &qpHandle);

    CHK_OOM_RET(ret, qpInfo.c_str());

    CHK_PRT_RET(
        ret != 0 || (qpHandle == nullptr),
        HCCL_ERROR(
            "[Create][RaQp]errNo[0x%016llx] ra qp create fail. qpInfo:[%s], return: ret[%d]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), qpInfo.c_str(), ret),
        HCCL_E_NETWORK);

    struct QpAttr attr {};
    CHK_RET(hrtRaGetQpAttr(qpHandle, &attr));
    s32 deviceId = 0;
    if (hrtGetDevice(&deviceId) != HCCL_SUCCESS) {
        deviceId = -1;
    }
    PLF_CONFIG_DEBUG(PLF_RES, "Create Qp para: deviceId[%d] qpn[%u] qpInfo[%s]", deviceId, attr.qpn, qpInfo.c_str());
    return HCCL_SUCCESS;
}

HcclResult
hrtRaTypicalQpCreate(RdmaHandle rdmaHandle, int flag, int qpMode, struct TypicalQp* qpInfo, QpHandle& qpHandle)
{
    std::string qpInfoStr = std::string("rdmaHandle:") + std::to_string(reinterpret_cast<intptr_t>(rdmaHandle))
                            + std::string("flag:") + std::to_string(flag) + std::string("qpMode:")
                            + std::to_string(qpMode) + std::to_string(reinterpret_cast<intptr_t>(&qpHandle));

    s32 ret = DlRaFunction::GetInstance().dlRaTypicalQpCreate(rdmaHandle, flag, qpMode, qpInfo, &qpHandle);

    CHK_OOM_RET(ret, qpInfoStr.c_str());

    RPT_ENV_ERR(
        ret != 0 || (qpHandle == nullptr), "EI0007", std::vector<std::string>({"resource_type", "resource_info"}),
        std::vector<std::string>({"qp", "CreateQp"}));

    CHK_PRT_RET(
        ret != 0 || (qpHandle == nullptr),
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] ra qp create fail. "
            "params: flag[%d], qpMode[%d]. return: ret[%d]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RESOURCE.c_str(), HCCL_ERROR_CODE(HCCL_E_NETWORK), flag,
            qpMode, ret),
        HCCL_E_NETWORK);

    s32 deviceId = 0;
    if (hrtGetDevice(&deviceId) != HCCL_SUCCESS) {
        deviceId = -1;
    }
    PLF_CONFIG_DEBUG(
        PLF_RES, "Create Qp para: deviceId[%d] qpn[%u] qpInfo[%s]", deviceId, qpInfo->qpn, qpInfoStr.c_str());
    return HCCL_SUCCESS;
}

HcclResult CreateTypicalCq(RdmaHandle rdmaHandle, u32 cqDepth, u32& cqn, void** cqHandle)
{
    HCCL_DEBUG("CreateTypicalCq cqDepth[%u]", cqDepth);

    s32 ret = DlRaFunction::GetInstance().dlRaTypicalCqCreate(rdmaHandle, cqDepth, &cqn, cqHandle);
    CHK_PRT_RET(
        ret != 0 || (*cqHandle == nullptr), HCCL_ERROR("[CreateTypicalCq]create typical cq failed. ret[%d]", ret),
        HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult DestroyTypicalCq(RdmaHandle rdmaHandle, u32 cqn, void* cqHandle)
{
    HCCL_DEBUG("DestroyTypicalCq cqn[%u]", cqn);

    s32 ret = DlRaFunction::GetInstance().dlRaTypicalCqDestroy(rdmaHandle, cqn, cqHandle);
    CHK_PRT_RET(ret != 0, HCCL_ERROR("[DestroyTypicalCq]destroy typical cq failed. ret[%d]", ret), HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult HrtRaQpDestroyWithoutCQ(QpHandle handle)
{
    s32 ret = DlRaFunction::GetInstance().dlRaQpDestroyWithoutCQ(handle);
    CHK_PRT_RET(
        ret != 0, HCCL_ERROR("[HrtRaQpDestroyWithoutCQ]destroy qp without cq failed. ret[%d]", ret), HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult hrtRaTypicalQpModify(QpHandle qpHandle, struct TypicalQp* localQpInfo, struct TypicalQp* remoteQpInfo)
{
    std::string qpInfo = std::string("qpHandle:") + std::to_string(reinterpret_cast<intptr_t>(qpHandle))
                         + std::string("localQpInfo:") + std::to_string(reinterpret_cast<intptr_t>(&localQpInfo))
                         + std::string("remoteQpInfo:") + std::to_string(reinterpret_cast<intptr_t>(&remoteQpInfo));

    s32 ret = DlRaFunction::GetInstance().dlRaTypicalQpModify(qpHandle, localQpInfo, remoteQpInfo);
    RPT_ENV_ERR(
        ret != 0, "EI0007", std::vector<std::string>({"resource_type", "resource_info"}),
        std::vector<std::string>({"qp", "ModifyQp"}));

    CHK_PRT_RET(
        ret == ROCE_EOPENSRC,
        HCCL_RUN_WARNING(
            "[%s][%s]ra qp modify need retry.", LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RESOURCE.c_str()),
        HCCL_E_AGAIN);
    CHK_PRT_RET(
        ret != 0,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] ra qp modify fail. return: ret[%d]", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RESOURCE.c_str(), HCCL_ERROR_CODE(HCCL_E_NETWORK), ret),
        HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult hrtRaTypicalSendWr(QpHandle handle, struct SendWr* wr, struct SendWrRsp* opRsp)
{
    s32 ret = 0;
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(GetExternalInputHcclLinkTimeOut());

    HCCL_DEBUG("ra send wr");
    while (true) {
        ret = DlRaFunction::GetInstance().dlRaTypicalSendWr(handle, wr, opRsp);
        if (ret == 0) {
            break; // 成功跳出
        } else if (
            (ret == SOCK_ENOENT) || (ret == SOCK_EAGAIN)
            || (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && ret == ROCE_ENOMEM)) {
            bool bTimeout = ((std::chrono::steady_clock::now() - startTime) >= timeout);
            CHK_PRT_RET(
                bTimeout,
                HCCL_ERROR(
                    "[Send][RaWr]errNo[0x%016llx] ra get send async timeout[%d s]. "
                    "return[%d], params: send_wrAddr[%p], opRspAddr[%p]",
                    HCCL_ERROR_CODE(HCCL_E_ROCE_TRANSFER), timeout, ret, wr, opRsp),
                HCCL_E_ROCE_TRANSFER);
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
        } else {
            HCCL_ERROR(
                "[Send][RaWr]ra send async fail. return[%d], para: send_wrAddr[%p], "
                "opRspAddr[%p].",
                ret, wr, opRsp);
            return HCCL_E_ROCE_TRANSFER; // 非-2/-11场景错误，不轮询，直接退出
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HrtRaQpDestroy(QpHandle handle)
{
    struct QpAttr attr {};
    CHK_RET(hrtRaGetQpAttr(handle, &attr));
    s32 deviceId = 0;
    if (hrtGetDevice(&deviceId) != HCCL_SUCCESS) {
        deviceId = -1;
    }
    PLF_CONFIG_DEBUG(PLF_RES, "Destroy Qp para: deviceId[%d] qpn[%u]", deviceId, attr.qpn);

    s32 ret = 0;
    auto startTime = chrono::steady_clock::now();
    auto timeout = chrono::seconds(GetExternalInputHcclLinkTimeOut());
    while (true) {
        ret = DlRaFunction::GetInstance().dlRaQpDestroy(handle);
        if (ret == 0) {
            break; // 成功跳出
        } else if (ret == ROCE_EAGAIN) {
            bool bTimeout = ((chrono::steady_clock::now() - startTime) >= timeout);
            CHK_PRT_RET(
                bTimeout,
                HCCL_ERROR(
                    "[Destroy][RaQp]errNo[0x%016llx] ra qp destroy timeout[%d s]. "
                    "return[%d].",
                    HCCL_ERROR_CODE(HCCL_E_NETWORK), timeout, ret),
                HCCL_E_NETWORK);
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
        } else {
            HCCL_ERROR(
                "[Destroy][RaQp]errNo[0x%016llx] ra qp destroy fail. return[%d].", HCCL_ERROR_CODE(HCCL_E_NETWORK),
                ret);
            return HCCL_E_NETWORK; // 非ra限速场景错误，不轮询，直接退出
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HrtRaGetQpDepth(RdmaHandle rdmaHandle, unsigned int* tempDepth, unsigned int* qpNum)
{
    CHK_PTR_NULL(rdmaHandle);

    s32 ret = DlRaFunction::GetInstance().dlRaGetQpDepth(rdmaHandle, tempDepth, qpNum);
    CHK_PRT_RET(
        ret != 0,
        HCCL_ERROR(
            "[HrtRaGetQpDepth]errNo[0x%016llx] ra get qp depth fail. return[%d]", HCCL_ERROR_CODE(HCCL_E_NETWORK), ret),
        HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult HrtRaSetQpDepth(RdmaHandle rdmaHandle, unsigned int tempDepth, unsigned int* qpNum)
{
    CHK_PTR_NULL(rdmaHandle);

    s32 ret = DlRaFunction::GetInstance().dlRaSetQpDepth(rdmaHandle, tempDepth, qpNum);
    CHK_PRT_RET(
        ret != 0,
        HCCL_ERROR(
            "[dlRaSetQpDepth]errNo[0x%016llx] ra set qp depth fail. return[%d]", HCCL_ERROR_CODE(HCCL_E_NETWORK), ret),
        HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult HrtRaQpNonBlockConnectAsync(QpHandle handle, const SocketHandle sockHandle)
{
    s32 ret = DlRaFunction::GetInstance().dlRaQpConnectAsync(handle, sockHandle);
    if (ret == 0) {
        return HCCL_SUCCESS;
    } else if (ret == ROCE_EAGAIN) {
        return HCCL_E_AGAIN;
    } else {
        HCCL_ERROR(
            "[HrtRaQpNonBlockConnectAsync]errNo[0x%016llx] ra qp connect async fail. return[%d].",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), ret);
        return HCCL_E_NETWORK;
    }

    return HCCL_SUCCESS;
}

HcclResult
HrtRaQpConnectAsync(QpHandle handle, const SocketHandle sockHandle, std::function<bool()> needStop, u32 timeout)
{
    s32 ret = 0;
    auto startTime = chrono::steady_clock::now();
    const chrono::seconds timeoutSec = chrono::seconds(timeout > 0 ? timeout : GetExternalInputHcclLinkTimeOut());
    while (true) {
        CHK_PRT_RET(needStop(), HCCL_ERROR("Terminating operation due to external request"), HCCL_E_INTERNAL);

        ret = DlRaFunction::GetInstance().dlRaQpConnectAsync(handle, sockHandle);
        if (ret == 0) {
            break; // 成功跳出
        } else if (ret == SOCK_EAGAIN) {
            bool bTimeout = ((chrono::steady_clock::now() - startTime) >= timeoutSec);
            CHK_PRT_RET(
                bTimeout,
                HCCL_ERROR(
                    "[ConnectAsync][RaQp]errNo[0x%016llx] ra qp connect async "
                    "timeout[%lld s]. return[%d].",
                    HCCL_ERROR_CODE(HCCL_E_NETWORK), timeoutSec, ret),
                HCCL_E_NETWORK);
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
        } else {
            HCCL_ERROR(
                "[ConnectAsync][RaQp]errNo[0x%016llx] ra qp connect async fail. return[%d]",
                HCCL_ERROR_CODE(HCCL_E_NETWORK), ret);
            return HCCL_E_NETWORK; // 非ra限速场景错误，不轮询，直接退出
        }
    }
    return HCCL_SUCCESS;
}

s32 hrtGetRaQpStatus(QpHandle handle, int* status)
{
    return DlRaFunction::GetInstance().dlRaGetQpStatus(handle, status);
}

HcclResult HrtRaMrReg(QpHandle handle, struct MrInfoT* mrInfo)
{
    CHK_PTR_NULL(mrInfo);
    HCCL_DEBUG("ra mr reg: addr[%p], size[%llu], access[%d].", mrInfo->addr, mrInfo->size, mrInfo->access);
    s32 ret = DlRaFunction::GetInstance().dlRaMrReg(handle, mrInfo);
    CHK_PRT_RET(
        ret != 0,
        HCCL_ERROR(
            "[Reg][RaMr]errNo[0x%016llx] ra mr reg fail. return[%d], params: "
            "addr[%p], size[%llu], access[%d]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), ret, mrInfo->addr, mrInfo->size, mrInfo->access),
        HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult HrtRaMrDereg(QpHandle handle, struct MrInfoT* mrInfo)
{
    CHK_PTR_NULL(mrInfo);
    HCCL_INFO(
        "ra mr dereg: qphandle[%p], addr[%p], size[%llu Byte], access[%d].", handle, mrInfo->addr, mrInfo->size,
        mrInfo->access);
    s32 ret = DlRaFunction::GetInstance().dlRaMrDereg(handle, mrInfo);
    CHK_PRT_RET(
        ret != 0,
        HCCL_ERROR(
            "[Dereg][RaMr]errNo[0x%016llx] ra mr dereg fail. return[%d], params: "
            "addr[%p], size[%llu Byte], access[%d]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), ret, mrInfo->addr, mrInfo->size, mrInfo->access),
        HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult hrtRaRegGlobalMr(const RdmaHandle rdmaHandle, struct MrInfoT& mrInfo, MrHandle& mrHandle)
{
    CHK_PTR_NULL(rdmaHandle);
    CHK_PTR_NULL(mrInfo.addr);
    CHK_PRT_RET(
        (mrInfo.size <= 0),
        HCCL_ERROR("[hrtRaRegGlobalMr]memory size[%llu Byte] should be greater than 0.", mrInfo.size), HCCL_E_PARA);

    s32 ret = DlRaFunction::GetInstance().dlRaRegGlobalMr(rdmaHandle, &mrInfo, &mrHandle);
    CHK_PRT_RET(
        ret != 0,
        HCCL_ERROR(
            "[hrtRaRegGlobalMr]errNo[0x%016llx] ra reg global mr fail. return[%d], params: "
            "addr[%p], size[%llu Byte], access[%d]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), ret, mrInfo.addr, mrInfo.size, mrInfo.access),
        HCCL_E_NETWORK);
    HCCL_DEBUG(
        "[hrtRaRegGlobalMr]ra reg global mr: addr[%p], size[%llu Byte], access[%d]", mrInfo.addr, mrInfo.size,
        mrInfo.access);
    return HCCL_SUCCESS;
}

HcclResult hrtRaDeRegGlobalMr(const RdmaHandle rdmaHandle, MrHandle mrHandle)
{
    CHK_PTR_NULL(rdmaHandle);
    CHK_PTR_NULL(mrHandle);

    HCCL_DEBUG("[hrtRaDeRegGlobalMr]ra dereg global.");
    s32 ret = DlRaFunction::GetInstance().dlRaDeRegGlobalMr(rdmaHandle, mrHandle);
    CHK_PRT_RET(
        ret != 0,
        HCCL_ERROR(
            "[hrtRaDeRegGlobalMr]errNo[0x%016llx] ra dereg global mr fail. return[%d]", HCCL_ERROR_CODE(HCCL_E_NETWORK),
            ret),
        HCCL_E_NETWORK);

    return HCCL_SUCCESS;
}

HcclResult HrtRaSendWr(QpHandle handle, struct SendWr* wr, struct SendWrRsp* opRsp)
{
    s32 ret = 0;
    auto startTime = chrono::steady_clock::now();
    auto timeout = chrono::seconds(GetExternalInputHcclLinkTimeOut());

    HCCL_DEBUG("ra send wr.");
    while (true) {
        ret = DlRaFunction::GetInstance().dlRaSendWr(handle, wr, opRsp);
        if (ret == 0) {
            break; // 成功跳出
        } else if (
            (ret == SOCK_ENOENT) || (ret == ROCE_EAGAIN)
            || (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && ret == ROCE_ENOMEM)) {
            bool bTimeout = ((chrono::steady_clock::now() - startTime) >= timeout);
            CHK_PRT_RET(
                bTimeout,
                HCCL_ERROR(
                    "[Send][RaWr]errNo[0x%016llx] ra get send async timeout[%d s]. "
                    "return[%d], params: send_wrAddr[%p], opRspAddr[%p]",
                    HCCL_ERROR_CODE(HCCL_E_ROCE_TRANSFER), timeout, ret, wr, opRsp),
                HCCL_E_ROCE_TRANSFER);
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
        } else {
            HCCL_ERROR(
                "[Send][RaWr]ra send async fail. return[%d], para: send_wrAddr[%p], "
                "opRspAddr[%p].",
                ret, wr, opRsp);
            return HCCL_E_ROCE_TRANSFER; // 非-2/-11场景错误，不轮询，直接退出
        }
    }

    return HCCL_SUCCESS;
}

HcclResult HrtRaSendWrV2(QpHandle handle, struct SendWrV2* wr, struct SendWrRsp* opRsp, HcclWorkflowMode workflowMode)
{
    s32 ret = 0;
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(GetExternalInputHcclLinkTimeOut());

    HCCL_DEBUG("ra send wr.");
    while (true) {
        ret = DlRaFunction::GetInstance().dlRaSendWrV2(handle, wr, opRsp);
        if (ret == 0) {
            break; // 成功跳出
        } else if (
            (ret == SOCK_ENOENT) || (ret == ROCE_EAGAIN)
            || (workflowMode == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && ret == ROCE_ENOMEM)) {
            HCCL_WARNING("after 1ms sendwr, ret=%d", ret);
            bool bTimeout = ((std::chrono::steady_clock::now() - startTime) >= timeout);
            CHK_PRT_RET(
                bTimeout,
                HCCL_ERROR(
                    "[Send][RaWr]errNo[0x%016llx] ra get send async timeout[%d s]. "
                    "return[%d], params: send_wrAddr[%p], opRspAddr[%p]",
                    HCCL_ERROR_CODE(HCCL_E_ROCE_TRANSFER), timeout, ret, wr, opRsp),
                HCCL_E_ROCE_TRANSFER);
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
        } else {
            HCCL_ERROR(
                "[Send][RaWr]ra send async fail. return[%d], para: send_wrAddr[%p], "
                "opRspAddr[%p].",
                ret, wr, opRsp);
            return HCCL_E_ROCE_TRANSFER; // 非-2/-11场景错误，不轮询，直接退出
        }
    }

    return HCCL_SUCCESS;
}

HcclResult HrtRaSendWrVerbs(QpHandle handle, struct SendWrVerbs* wr, struct SendWrRsp* opRsp)
{
    s32 ret = 0;
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(GetExternalInputHcclLinkTimeOut());

    HCCL_DEBUG("ra send wr verbs.");
    while (true) {
        ret = DlRaFunction::GetInstance().dlRaSendWrVerbs(handle, wr, opRsp);
        if (ret == 0) {
            break;
        } else if (
            (ret == SOCK_ENOENT) || (ret == ROCE_EAGAIN)
            || (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && ret == ROCE_ENOMEM)) {
            bool bTimeOut = ((std::chrono::steady_clock::now() - startTime) >= timeout);
            CHK_PRT_RET(
                bTimeOut,
                HCCL_ERROR(
                    "[HrtRaSendWrVerbs][RaWr]errNo[0x%016llx] ra get send async timeout[%d s]. "
                    "return[%d], params: send_wrAddr[%p], opRspAddr[%p]",
                    HCCL_ERROR_CODE(HCCL_E_ROCE_TRANSFER), timeout, ret, wr, opRsp),
                HCCL_E_ROCE_TRANSFER);
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
        } else {
            HCCL_ERROR(
                "[HrtRaSendWrVerbs][RaWr]ra send async fail. return[%d], para: send_wrAddr[%p], "
                "opRspAddr[%p].",
                ret, wr, opRsp);
            return HCCL_E_ROCE_TRANSFER;
        }
    }

    return HCCL_SUCCESS;
}

HcclResult HrtRaRecvWrVerbs(QpHandle handle, struct RecvWrVerbs* wr)
{
    s32 ret = 0;
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(GetExternalInputHcclLinkTimeOut());

    HCCL_DEBUG("ra recv wr verbs.");
    while (true) {
        ret = DlRaFunction::GetInstance().dlRaRecvWrVerbs(handle, wr);
        if (ret == 0) {
            break;
        } else if (
            (ret == SOCK_ENOENT) || (ret == ROCE_EAGAIN)
            || (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && ret == ROCE_ENOMEM)) {
            bool bTimeout = ((std::chrono::steady_clock::now() - startTime) >= timeout);
            CHK_PRT_RET(
                bTimeout,
                HCCL_ERROR(
                    "[Recv][RaWr]errNo[0x%016llx] ra get recv async timeout[%d s]. "
                    "return[%d], params: recv_wrAddr[%p]",
                    HCCL_ERROR_CODE(HCCL_E_ROCE_TRANSFER), timeout, ret, wr),
                HCCL_E_ROCE_TRANSFER);
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
        } else {
            HCCL_ERROR("[Recv][RaWr]ra recv async fail. return[%d], para: recv_wrAddr[%p].", ret, wr);
            return HCCL_E_ROCE_TRANSFER;
        }
    }

    return HCCL_SUCCESS;
}

s32 hrtRaPollCq(QpHandle handle, bool is_send_cq, unsigned int num, void* wc)
{
    CHK_PTR_NULL(handle);
    CHK_PTR_NULL(wc);

    u32 ret = DlRaFunction::GetInstance().dlRaPollCq(handle, is_send_cq, num, wc);
    CHK_PRT_RET(static_cast<u32>(ret) > num, HCCL_ERROR("[hrtRaPollCq] PollCq fail. return[%d]", ret), ret);
    return ret;
}

s32 HrtRaPollTypicalCq(void* cqHandle, u32 num, void* wc)
{
    CHK_PTR_NULL(cqHandle);
    CHK_PTR_NULL(wc);
    u32 ret = DlRaFunction::GetInstance().dlRaPollTypicalCq(cqHandle, num, wc);
    CHK_PRT_RET(static_cast<u32>(ret) > num, HCCL_ERROR("[HrtRaPollTypicalCq] PollCq fail. return[%d]", ret), ret);
    return ret;
}

HcclResult hrtRaQpBatchModify(RdmaHandle rdmaHandle, QpHandle qpHandle[], unsigned int num, int expectStatus)
{
    if (DlRaFunction::GetInstance().dlRaQpBatchModify == nullptr) {
        HCCL_ERROR("[Send][RaQpBatchModify]driver package does not support ra_qp_batch_modify interface, "
                   "please change new one");
        return HCCL_E_NOT_SUPPORT;
    }
    s32 ret = DlRaFunction::GetInstance().dlRaQpBatchModify(rdmaHandle, &qpHandle[0], num, expectStatus);
    CHK_PRT_RET(
        ret != 0 || (qpHandle[0] == nullptr),
        HCCL_ERROR(
            "[BatchModify][RaQp]errNo[0x%016llx] ra qp batch modify fail. "
            "params: num[%u], expectStatus[%d]. return: ret[%d]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), num, expectStatus),
        HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult HrtRaSendWrlist(
    QpHandle handle, struct SendWrlistData wr[], struct SendWrRsp opRsp[], unsigned int sendNum,
    unsigned int* completeNum)
{
    if (DlRaFunction::GetInstance().dlRaSendWrlist == nullptr) {
        HCCL_ERROR("[Send][RaWrlist]driver package does not support hrtRaSendWrlist interface, "
                   "please change new one");
        return HCCL_E_NOT_SUPPORT;
    }
    s32 ret = 0;
    auto startTime = chrono::steady_clock::now();
    auto timeout = chrono::seconds(GetExternalInputHcclLinkTimeOut());
    u32 remainNum = sendNum;
    unsigned int completeNumLocal = 0;
    *completeNum = 0;
    while (true) {
        if (remainNum > sendNum) {
            HCCL_ERROR(
                "[Send][RaWr]ra wr list send async fail. return[%d], remainNum[%u], "
                "sendNum[%u].",
                HCCL_E_ROCE_TRANSFER, remainNum, sendNum);
            return HCCL_E_ROCE_TRANSFER; // 非-2/-11场景错误，不轮询，直接退出
        }
        if (remainNum == 0) {
            break;
        }
        ret = DlRaFunction::GetInstance().dlRaSendWrlist(
            handle, wr + (sendNum - remainNum), opRsp + (sendNum - remainNum), remainNum, &completeNumLocal);
        *completeNum += completeNumLocal;
        if (ret == 0) {
            break; // 成功跳出
        } else if (
            (ret == SOCK_ENOENT) || (ret == ROCE_EAGAIN)
            || (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && ret == ROCE_ENOMEM)) {
            remainNum -= completeNumLocal;
            bool bTimeout = ((chrono::steady_clock::now() - startTime) >= timeout);
            CHK_PRT_RET(
                bTimeout,
                HCCL_ERROR(
                    "[Send][RaWrList]errNo[0x%016llx] ra send wrlist async timeout[%d s]. "
                    "return[%d], params: send_wrAddr[%p], opRspAddr[%p]",
                    HCCL_ERROR_CODE(HCCL_E_ROCE_TRANSFER), timeout, ret, wr, opRsp),
                HCCL_E_ROCE_TRANSFER);
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
        } else {
            HCCL_ERROR(
                "[Send][RaWr]ra wr list send async fail. return[%d], para: send_wrAddr[%p], dst_addr[%p],"
                " bufAddr[%p], bufLen[%u], opRspAddr[%p].",
                ret, wr, wr->dstAddr, wr->memList.addr, wr->memList.len, opRsp);
            return HCCL_E_ROCE_TRANSFER; // 非-2/-11场景错误，不轮询，直接退出
        }
    }

    return HCCL_SUCCESS;
}

HcclResult HrtRaSendWrlistExt(
    QpHandle handle, struct SendWrlistDataExt wr[], struct SendWrRsp opRsp[], unsigned int sendNum,
    unsigned int* completeNum)
{
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    if (deviceType != DevType::DEV_TYPE_910B && deviceType != DevType::DEV_TYPE_910_93) {
        vector<SendWrlistData> wqeList(sendNum);
        struct SendWrlistData* data = wqeList.data();
        for (unsigned int i = 0; i < sendNum; i++) {
            s32 sret = memcpy_s(&data[i], sizeof(SendWrlistData), &wr[i], sizeof(SendWrlistData));
            CHK_PRT_RET(
                sret != EOK, HCCL_ERROR("[WqeList][Add]add wqe list, memcpy wqe failed. errorno[%d]", sret),
                HCCL_E_MEMORY);
        }
        CHK_RET(HrtRaSendWrlist(handle, data, opRsp, sendNum, completeNum));
    } else {
        static bool flag = false;
        if (UNLIKELY(flag == false)) {
            if (UNLIKELY(DlRaFunction::GetInstance().dlRaSendWrlistExt == nullptr)) {
                HCCL_ERROR("[Send][RaWrlistExt]driver package does not support hrtRaSendWrlist interface, "
                           "please change new one");
                return HCCL_E_NOT_SUPPORT;
            }
            flag = true;
        }

        s32 ret = 0;
        auto startTime = chrono::steady_clock::now();
        auto timeout = chrono::seconds(GetExternalInputHcclLinkTimeOut());
        u32 remainNum = sendNum;
        unsigned int completeNumLocal = 0;
        *completeNum = 0;
        while (true) {
            if (remainNum > sendNum) {
                HCCL_ERROR(
                    "[Send][RaWr]ra wr list send async fail. return[%d], remainNum[%u], "
                    "sendNum[%u].",
                    HCCL_E_ROCE_TRANSFER, remainNum, sendNum);
                return HCCL_E_ROCE_TRANSFER;
            }
            if (remainNum == 0) {
                break;
            }
            ret = DlRaFunction::GetInstance().dlRaSendWrlistExt(
                handle, wr + (sendNum - remainNum), opRsp + (sendNum - remainNum), remainNum, &completeNumLocal);
            *completeNum += completeNumLocal;
            if (ret == 0) {
                break; // 成功跳出
            } else if (
                (ret == SOCK_ENOENT) || (ret == ROCE_EAGAIN)
                || (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && ret == ROCE_ENOMEM)) {
                remainNum -= completeNumLocal;
                bool bTimeout = ((chrono::steady_clock::now() - startTime) >= timeout);
                CHK_PRT_RET(
                    bTimeout,
                    HCCL_ERROR(
                        "[Send][RaWr]errNo[0x%016llx] ra wrlist send async timeout[%d s]. "
                        "return[%d], params: send_wrAddr[%p], opRspAddr[%p]",
                        HCCL_ERROR_CODE(HCCL_E_ROCE_TRANSFER), timeout, ret, wr, opRsp),
                    HCCL_E_ROCE_TRANSFER);
                SaluSleep(ONE_MILLISECOND_OF_USLEEP);
            } else {
                HCCL_ERROR(
                    "[Send][RaWr]ra wrlist send async fail. return[%d], para: send_wrAddr[%p], "
                    "opRspAddr[%p].",
                    ret, wr, opRsp);
                return HCCL_E_ROCE_TRANSFER; // 非-2/-11场景错误，不轮询，直接退出
            }
        }
    }

    return HCCL_SUCCESS;
}

HcclResult HrtRaSendNormalWrlist(
    QpHandle handle, struct WrInfo wr[], struct SendWrRsp opRsp[], unsigned int sendNum, unsigned int* completeNum)
{
    if (UNLIKELY(DlRaFunction::GetInstance().dlRaSendWrlist == nullptr)) {
        HCCL_ERROR("[Send][RaWrlist]driver package does not support hrtRaSendWrlist interface, "
                   "please change new one");
        return HCCL_E_NOT_SUPPORT;
    }
    s32 ret = 0;
    auto startTime = chrono::steady_clock::now();
    auto timeout = chrono::seconds(GetExternalInputHcclLinkTimeOut());
    u32 remainNum = sendNum;
    unsigned int completeNumLocal = 0;
    *completeNum = 0;
    while (true) {
        if (UNLIKELY(remainNum > sendNum)) {
            HCCL_ERROR(
                "[Send][RaWr]ra wr list send async fail. return[%d], remainNum[%u], "
                "sendNum[%u].",
                HCCL_E_ROCE_TRANSFER, remainNum, sendNum);
            return HCCL_E_ROCE_TRANSFER; // 非-2/-11场景错误，不轮询，直接退出
        }
        if (remainNum == 0) {
            break;
        }
        ret = DlRaFunction::GetInstance().dlRaSendNormalWrlist(
            handle, wr + (sendNum - remainNum), opRsp + (sendNum - remainNum), remainNum, &completeNumLocal);
        *completeNum += completeNumLocal;
        if (ret == 0) {
            break; // 成功跳出
        }
        if ((ret == ROCE_ENOENT) || (ret == ROCE_EAGAIN) || ret == ROCE_ENOMEM) {
            remainNum -= completeNumLocal;
            bool bTimeout = ((chrono::steady_clock::now() - startTime) >= timeout);
            CHK_PRT_RET(
                bTimeout,
                HCCL_ERROR(
                    "[Send][HrtRaSendNormalWrlist]errNo[0x%016llx] ra send wrlist async timeout[%d s]. "
                    "return[%d], params: send_wrAddr[%p], opRspAddr[%p]",
                    HCCL_ERROR_CODE(HCCL_E_ROCE_TRANSFER), timeout, ret, wr, opRsp),
                HCCL_E_ROCE_TRANSFER);
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
        } else {
            HCCL_ERROR(
                "[Send][RaWr]ra wr list send async fail. return[%d], para: send_wrAddr[%p], dst_addr[%p],"
                " bufAddr[%p], bufLen[%u], opRspAddr[%p].",
                ret, wr, wr->dstAddr, wr->memList.addr, wr->memList.len, opRsp);
            return HCCL_E_ROCE_TRANSFER; // 非-2/-11场景错误，不轮询，直接退出
        }
    }

    return HCCL_SUCCESS;
}

HcclResult HrtRaGetNotifyBaseAddr(RdmaHandle handle, u64* va, u64* size, std::function<bool()> needStop)
{
    s32 ret = 0;
    auto startTime = chrono::steady_clock::now();
    auto timeout = chrono::seconds(GetExternalInputHcclLinkTimeOut());
    while (true) {
        CHK_PRT_RET(needStop(), HCCL_ERROR("Terminating operation due to external request"), HCCL_E_INTERNAL);

        ret = DlRaFunction::GetInstance().dlRaGetNotifyBaseAddr(handle, va, size);
        if (ret == 0) {
            break; // 成功跳出
        } else if (ret == ROCE_EAGAIN) {
            bool bTimeout = ((chrono::steady_clock::now() - startTime) >= timeout);
            CHK_PRT_RET(
                bTimeout,
                HCCL_ERROR(
                    "[Get][RaNotifyBaseAddr]errNo[0x%016llx] ra get notify base addr "
                    "timeout[%d s]. return[%d], params: va[0x%llx], size[%llu Byte]",
                    HCCL_ERROR_CODE(HCCL_E_NETWORK), timeout, ret, *va, *size),
                HCCL_E_NETWORK);
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
        } else {
            HCCL_ERROR(
                "[Get][RaNotifyBaseAddr]errNo[0x%016llx] ra get notify base addr fail. return[%d], params: "
                "va[0x%llx], size[%llu]",
                HCCL_ERROR_CODE(HCCL_E_NETWORK), ret, *va, *size);
            return HCCL_E_NETWORK; // 非ra限速场景错误，不轮询，直接退出
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HrtRaGetNotifyMrInfo(u32 phyId, RdmaHandle handle, struct MrInfoT* mrInfo)
{
    s32 ret = 0;
    u32 getNotifyBaVersion = 0;
    HcclResult vRet = hrtRaGetInterfaceVersion(phyId, GET_NOTIFY_BA, &getNotifyBaVersion);
    if (vRet != HCCL_SUCCESS || getNotifyBaVersion < GET_NOTIFY_BA_VERSION) {
        HCCL_ERROR("this package does not support HrtRaGetNotifyMrInfo for device, please change new package");
        return HCCL_E_NOT_SUPPORT;
    }
    auto startTime = chrono::steady_clock::now();
    auto timeout = chrono::seconds(GetExternalInputHcclLinkTimeOut());
    while (true) {
        ret = DlRaFunction::GetInstance().dlRaGetNotifyMrInfo(handle, mrInfo);
        if (ret == 0) {
            break; // 成功跳出
        } else if (ret == ROCE_EAGAIN) {
            bool bTimeout = ((chrono::steady_clock::now() - startTime) >= timeout);
            CHK_PRT_RET(
                bTimeout,
                HCCL_ERROR(
                    "[Get][RaGetNotifyMrInfo]errNo[0x%016llx] ra get notify mr info "
                    "timeout[%d s]. return[%d]",
                    HCCL_ERROR_CODE(HCCL_E_NETWORK), timeout, ret),
                HCCL_E_NETWORK);
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
        } else {
            HCCL_ERROR(
                "[Get][RaGetNotifyMrInfo]errNo[0x%016llx] ra get notify mr info fail. return[%d]",
                HCCL_ERROR_CODE(HCCL_E_NETWORK), ret);
            return HCCL_E_NETWORK;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HrtRaInit(struct RaInitConfig* config)
{
    CHK_RET(DlRaFunction::GetInstance().DlRaFunctionInit());
    s32 ret = 0;
    auto startTime = chrono::steady_clock::now();
    auto timeout = chrono::seconds(GetExternalInputHcclLinkTimeOut());
    while (true) {
        ret = DlRaFunction::GetInstance().dlRaInit(config);
        if (ret == 0) {
            break; // 成功跳出
        } else if (ret == HCCP_EAGAIN) {
            bool bTimeout = ((chrono::steady_clock::now() - startTime) >= timeout);
            CHK_PRT_RET(
                bTimeout,
                HCCL_ERROR(
                    "[Init][Ra]errNo[0x%016llx] ra init timeout[%lld s]. return[%d], "
                    "phyId[%u], nicPosition[%u], hdcType[%d]",
                    HCCL_ERROR_CODE(HCCL_E_TIMEOUT), timeout, ret, config->phyId, config->nicPosition, config->hdcType),
                HCCL_E_TIMEOUT);
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
        } else {
            if (ret == REPEAT_RAINIT_ERROR_CODE) {
                HCCL_RUN_WARNING(
                    "ra init repeatedly, return. phyId[%u] nicPosition[%u] hdcType[%d]", config->phyId,
                    config->nicPosition, config->hdcType);
                return HCCL_E_PARA;
            }
            HCCL_ERROR(
                "[Init][Ra]errNo[0x%016llx] ra init fail ret[%d] phyId[%u] nicPosition[%u] hdcType[%d]",
                HCCL_ERROR_CODE(HCCL_E_NETWORK), ret, config->phyId, config->nicPosition, config->hdcType);
            return HCCL_E_NETWORK; // 非ra限速场景错误，不轮询。直接退出
        }
    }
    HCCL_INFO("init ra success.");
    return HCCL_SUCCESS;
}

HcclResult HrtRaRdmaInit(int mode, u32 notifyType, struct rdev rdevInfo, RdmaHandle& rdmaHandle)
{
    s32 ret = DlRaFunction::GetInstance().dlRaRdmaInit(mode, notifyType, rdevInfo, &rdmaHandle);
    RPT_INPUT_ERR(
        ret == HCCP_ELINKDOWN, "EI0009", vector<string>({"device_id", "reason"}),
        vector<string>({std::to_string(rdevInfo.phyId), "The network port is down"}));
#ifndef HCCD
    vector<HcclIpAddress> deviceIp;
    CHK_RET(hrtRaGetDeviceIP(rdevInfo.phyId, deviceIp));
    CHK_PRT_RET(deviceIp.size() < 1, HCCL_ERROR("Get ip address failed, phyId[%u]", rdevInfo.phyId), HCCL_E_INTERNAL);
    RPT_INPUT_ERR(
        ret == HCCP_EINVALIDIPS, "EI0014", vector<string>({"value", "variable", "expect"}),
        vector<string>(
            {string(HcclIpAddress(rdevInfo.localIp.addr.s_addr).GetReadableIP()), "IP",
             string(deviceIp[0].GetReadableIP())}));
#endif
    CHK_PRT_CONT(
        ret == HCCP_EINVALIDIPS,
        HCCL_ERROR(
            "[%s][%s]the IP address in the ranktable is inconsistent with the IP address of the network adapter.",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CHECK.c_str()));

    CHK_PRT_RET(ret == HCCP_ELINKDOWN, HCCL_RUN_WARNING("ra rdma init need retry."), HCCL_E_AGAIN);
    CHK_PRT_RET(
        ret != 0 || (rdmaHandle == nullptr),
        HCCL_ERROR(
            "[Init][RaRdma]errNo[0x%016llx] rdma init fail. "
            "params: mode[%d]. notifyType[%u] phyId[%u] family[%d] s_addr[%u] ret[%d]",
            HCCL_ERROR_CODE(HCCL_E_INTERNAL), mode, notifyType, rdevInfo.phyId, rdevInfo.family,
            rdevInfo.localIp.addr.s_addr, ret),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult HrtRaRdmaInitWithAttr(struct RdevInitInfo& init_info, const struct rdev& rdevInfo, RdmaHandle& rdmaHandle)
{
    HCCL_INFO(
        "mode:[%d], NotifyTypeT:[%u], enabled910aLite:[%d], disabledLiteThread:[%d], enabled2mbLite:[%d]",
        init_info.mode, init_info.notifyType, init_info.enabled910aLite, init_info.disabledLiteThread,
        init_info.enabled2mbLite);

    s32 ret = DlRaFunction::GetInstance().dlRaRdmaInitWithAttr(init_info, rdevInfo, &rdmaHandle);
    RPT_INPUT_ERR(
        ret == HCCP_ELINKDOWN, "EI0009", vector<string>({"device_id", "reason"}),
        vector<string>({std::to_string(rdevInfo.phyId), "The network port is down"}));
    CHK_PRT_CONT(
        ret == HCCP_ELINKDOWN, HCCL_ERROR(
                                   "[%s][%s]rdma init failed because RoCE link status is down, please check the "
                                   "network adapter configuration.",
                                   LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RESOURCE.c_str()));
#ifndef HCCD
    if (init_info.mode != NETWORK_PEER_ONLINE) {
        vector<HcclIpAddress> deviceIp;
        CHK_RET(hrtRaGetDeviceIP(rdevInfo.phyId, deviceIp));
        CHK_PRT_RET(
            deviceIp.size() < 1, HCCL_ERROR("Get ip address failed, phyId[%u]", rdevInfo.phyId), HCCL_E_INTERNAL);
        RPT_INPUT_ERR(
            ret == HCCP_EINVALIDIPS, "EI0014", vector<string>({"value", "variable", "expect"}),
            vector<string>(
                {string(HcclIpAddress(rdevInfo.localIp.addr.s_addr).GetReadableIP()), "IP",
                 string(deviceIp[0].GetReadableIP())}));
    }
#endif
    CHK_PRT_CONT(
        ret == HCCP_EINVALIDIPS,
        HCCL_ERROR(
            "[%s][%s]the IP address in the ranktable is inconsistent with the IP address of the network adapter.",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CHECK.c_str()));

    CHK_PRT_RET(
        ret != 0 || (rdmaHandle == nullptr),
        HCCL_ERROR(
            "[Init][RaRdma]errNo[0x%016llx] rdma init fail. "
            "return: ret[%d]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), ret),
        HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult HrtRdmaInitWithBackupAttr(
    struct RdevInitInfo& init_info, struct rdev& rdevInfo, struct rdev& backupRdevInfo, RdmaHandle& rdmaHandle)
{
    HCCL_INFO(
        "[%s]mode:[%d], NotifyTypeT:[%u], enabled910aLite:[%d], disabledLiteThread:[%d], "
        "enabled2mbLite:[%d]",
        __func__, init_info.mode, init_info.notifyType, init_info.enabled910aLite, init_info.disabledLiteThread,
        init_info.enabled2mbLite);

    // 获取版本号查看是否兼容
    u32 rdmainitBackupVersion = 0;
    HcclResult vRet = hrtRaGetInterfaceVersion(rdevInfo.phyId, RDEV_INIT_WITH_BACKUP, &rdmainitBackupVersion);
    if (vRet != HCCL_SUCCESS || rdmainitBackupVersion < RDEV_INIT_WITH_BACKUP_SUP_VER) {
        HCCL_WARNING("this package does not support HrtRdmaInitWithBackupAttr, please change new package.");
        return HCCL_E_NOT_SUPPORT;
    }

    s32 ret
        = DlRaFunction::GetInstance().dlRaRdmaInitWithBackupAttr(&init_info, &rdevInfo, &backupRdevInfo, &rdmaHandle);
    RPT_INPUT_ERR(
        ret == HCCP_ELINKDOWN, "EI0009", vector<string>({"device_id", "reason"}),
        vector<string>({std::to_string(rdevInfo.phyId), "The network port is down"}));
    CHK_PRT_CONT(
        ret == HCCP_ELINKDOWN, HCCL_ERROR(
                                   "[%s][%s]rdma init failed because RoCE link status is down, please check the "
                                   "network adapter configuration.",
                                   LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RESOURCE.c_str()));
#ifndef HCCD
    vector<HcclIpAddress> deviceIp;
    CHK_RET(hrtRaGetDeviceIP(rdevInfo.phyId, deviceIp));
    CHK_PRT_RET(deviceIp.size() < 1, HCCL_ERROR("Get ip address failed, phyId[%u]", rdevInfo.phyId), HCCL_E_INTERNAL);
    RPT_INPUT_ERR(
        ret == HCCP_EINVALIDIPS, "EI0014", vector<string>({"value", "variable", "expect"}),
        vector<string>(
            {string(HcclIpAddress(rdevInfo.localIp.addr.s_addr).GetReadableIP()), "IP",
             string(deviceIp[0].GetReadableIP())}));
#endif
    CHK_PRT_CONT(
        ret == HCCP_EINVALIDIPS,
        HCCL_ERROR(
            "[%s][%s]the IP address in the ranktable is inconsistent with the IP address of the network adapter.",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CHECK.c_str()));

    CHK_PRT_RET(
        ret != 0 || (rdmaHandle == nullptr),
        HCCL_ERROR(
            "[Init][RaRdma]errNo[0x%016llx] rdma init fail. "
            "return: ret[%d]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), ret),
        HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult HrtRaRdmaInitRef(int mode, u32 notifyType, const struct rdev& rdevInfo, RdmaHandle& rdmaHandle)
{
    lock_guard<mutex> lock(g_rdmaHandleInfo.handleMutex);
    if (g_rdmaHandleInfo.handleMap.find(rdevInfo.localIp.addr.s_addr) != g_rdmaHandleInfo.handleMap.end()) {
        HCCL_DEBUG(
            "The rdmaHandle[%p] corresponding to the ipAddr[%u] has been initialized.", rdmaHandle,
            rdevInfo.localIp.addr.s_addr);

        rdmaHandle = g_rdmaHandleInfo.handleMap[rdevInfo.localIp.addr.s_addr];
        g_rdmaHandleInfo.handleRef[rdmaHandle]++;
        return HCCL_SUCCESS;
    }

    CHK_RET(HrtRaRdmaInit(mode, notifyType, rdevInfo, rdmaHandle));
    g_rdmaHandleInfo.handleMap[rdevInfo.localIp.addr.s_addr] = rdmaHandle;
    g_rdmaHandleInfo.handleRef[rdmaHandle] = FIRST_HANDLE_REF;
    return HCCL_SUCCESS;
}

HcclResult HrtRaRdmaGetHandle(unsigned int phyId, RdmaHandle& rdmaHandle)
{
    CHK_SMART_PTR_NULL(DlRaFunction::GetInstance().dlRaRdmaGetHandle);
    s32 ret = DlRaFunction::GetInstance().dlRaRdmaGetHandle(phyId, &rdmaHandle);

    CHK_PRT_RET(
        ret != 0 || (rdmaHandle == nullptr),
        HCCL_ERROR(
            "[Get][RdmaHandle]errNo[0x%016llx] "
            "get rdma handle fail. return: ret[%d]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), ret),
        HCCL_E_NETWORK);

    HCCL_DEBUG("get rdma handle success.");
    return HCCL_SUCCESS;
}

HcclResult HrtGetRdmaLiteStatus(RdmaHandle rdmaHandle, int* supportLite)
{
    if (rdmaHandle == nullptr) {
        HCCL_ERROR("[Get][RdmaLiteStatus]rdmaHandle is nullptr, please input the correct rdmaHandle");
        return HCCL_E_PTR;
    }
    s32 ret = DlRaFunction::GetInstance().dlRaGetRdmaLiteStatus(rdmaHandle, supportLite);
    CHK_PRT_RET(
        ret != 0,
        HCCL_ERROR(
            "[Get][RdmaLiteStatus]errNo[0x%016llx] get rdma lite status fail. "
            "return: ret[%d]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), ret),
        HCCL_E_NETWORK);

    return HCCL_SUCCESS;
}

HcclResult HrtRaDeInit(struct RaInitConfig* config)
{
    s32 ret = 0;
    auto startTime = chrono::steady_clock::now();
    auto timeout = chrono::seconds(GetExternalInputHcclLinkTimeOut());
    while (true) {
        ret = DlRaFunction::GetInstance().dlRaDeInit(config);
        if (ret == 0) {
            break; // 成功跳出
        } else if (ret == HCCP_EAGAIN) {
            bool bTimeout = ((chrono::steady_clock::now() - startTime) >= timeout);
            CHK_PRT_RET(
                bTimeout,
                HCCL_ERROR(
                    "[DeInit][Ra]errNo[0x%016llx] ra deinit timeout[%lld s]. return[%d], "
                    "phyId[%u] nicPosition[%u] hdcType[%d]",
                    HCCL_ERROR_CODE(HCCL_E_TIMEOUT), timeout, ret, config->phyId, config->nicPosition, config->hdcType),
                HCCL_E_TIMEOUT);
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
        } else {
            HCCL_ERROR(
                "[DeInit][Ra]errNo[0x%016llx] ra deinit fail. ret[%d] phyId[%u] nicPosition[%u] hdcType[%d]",
                HCCL_ERROR_CODE(HCCL_E_NETWORK), ret, config->phyId, config->nicPosition, config->hdcType);
            return HCCL_E_NETWORK; // 非ra限速场景错误，不轮询。直接退出
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HrtRaRdmaDeInit(RdmaHandle& rdmaHandle, u32 notifyType)
{
    CHK_PTR_NULL(rdmaHandle);
    s32 ret = DlRaFunction::GetInstance().dlRaRdmaDeInit(rdmaHandle, notifyType);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[DeInit][RaRdma] rdmaHandle[%p]", rdmaHandle);
        rdmaHandle = nullptr;
    }
    CHK_PRT_RET(
        ret != 0,
        HCCL_ERROR(
            "[DeInit][RaRdma]errNo[0x%016llx] rt rdev deinit fail. return[%d]."
            "notifyType[%u]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), ret, notifyType),
        HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult HrtRaRdmaDeInitRef(RdmaHandle& rdmaHandle, u32 notifyType)
{
    lock_guard<mutex> lock(g_rdmaHandleInfo.handleMutex);
    g_rdmaHandleInfo.handleRef[rdmaHandle]--;
    if (g_rdmaHandleInfo.handleRef[rdmaHandle] == 0) {
        HCCL_DEBUG("This rdmaHandle[%p] is about to be deinitialized.", rdmaHandle);
        CHK_RET(HrtRaRdmaDeInit(rdmaHandle, notifyType));
        auto it = g_rdmaHandleInfo.handleMap.begin();
        while (it != g_rdmaHandleInfo.handleMap.end()) {
            if (it->second == rdmaHandle) {
                it = g_rdmaHandleInfo.handleMap.erase(it);
            } else {
                ++it;
            }
        }

        g_rdmaHandleInfo.handleRef.erase(rdmaHandle);
    }

    return HCCL_SUCCESS;
}

HcclResult hrtRaSocketInit(int mode, struct rdev rdevInfo, SocketHandle& socketHandle)
{
    s32 ret = DlRaFunction::GetInstance().dlRaSocketInit(mode, rdevInfo, &socketHandle);

    CHK_PRT_RET(
        ret != 0 || (socketHandle == nullptr),
        HCCL_ERROR(
            "[Init][RaSock]errNo[0x%016llx] "
            "ra socket init fail. params: mode[%d]. return: ret[%d] phyId[%u] family[%d] s_addr[%u]",
            HCCL_ERROR_CODE(HCCL_E_INTERNAL), mode, ret, rdevInfo.phyId, rdevInfo.family, rdevInfo.localIp.addr.s_addr),
        HCCL_E_INTERNAL);

    HCCL_INFO(
        "socket init success, ip[%u] device id[%u], socketHandle[%p]", rdevInfo.localIp.addr.s_addr, rdevInfo.phyId,
        socketHandle);
    return HCCL_SUCCESS;
}

HcclResult hrtRaSocketInitV1(int mode, struct SocketInitInfoT socket_init, SocketHandle& socketHandle)
{
    s32 ret = DlRaFunction::GetInstance().dlRaSocketInitV1(mode, socket_init, &socketHandle);

    CHK_PRT_RET(
        ret != 0 || (socketHandle == nullptr),
        HCCL_ERROR(
            "[Init][RaSockV1]errNo[0x%016llx] ra socket v1 init fail. params: mode[%d]. return: ret[%d]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), mode, ret),
        HCCL_E_NETWORK);
    HCCL_INFO("socket init v1 success, socketHandle[%p]", socketHandle);
    return HCCL_SUCCESS;
}

HcclResult hrtRaSocketInitRef(int mode, const struct rdev& rdevInfo, SocketHandle& socketHandle)
{
    lock_guard<mutex> lock(g_socketHandleInfo.handleMutex);
    if (g_socketHandleInfo.handleMap.find(rdevInfo.localIp.addr.s_addr) != g_socketHandleInfo.handleMap.end()) {
        HCCL_DEBUG(
            "The socketHandle[%p] corresponding to the ipAddr[%u] has been initialized.", socketHandle,
            rdevInfo.localIp.addr.s_addr);

        socketHandle = g_socketHandleInfo.handleMap[rdevInfo.localIp.addr.s_addr];
        g_socketHandleInfo.handleRef[socketHandle]++;
        return HCCL_SUCCESS;
    }

    CHK_RET(hrtRaSocketInit(mode, rdevInfo, socketHandle));
    g_socketHandleInfo.handleMap[rdevInfo.localIp.addr.s_addr] = socketHandle;
    g_socketHandleInfo.handleRef[socketHandle] = FIRST_HANDLE_REF;
    return HCCL_SUCCESS;
}

HcclResult hrtRaSocketDeInit(SocketHandle& socketHandle)
{
    CHK_PTR_NULL(socketHandle);
    s32 ret = DlRaFunction::GetInstance().dlRaSocketDeInit(socketHandle);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[DeInit][RaSocket] socketHandle[%p]", socketHandle);
        socketHandle = nullptr;
    }
    CHK_PRT_RET(
        ret != 0,
        HCCL_ERROR(
            "[DeInit][RaSocket]errNo[0x%016llx] rt socket deinit fail. return[%d]", HCCL_ERROR_CODE(HCCL_E_NETWORK),
            ret),
        HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult hrtRaSocketDeInitRef(SocketHandle& socketHandle)
{
    lock_guard<mutex> lock(g_socketHandleInfo.handleMutex);
    g_socketHandleInfo.handleRef[socketHandle]--;
    if (g_socketHandleInfo.handleRef[socketHandle] == 0) {
        HCCL_DEBUG("This socketHandle[%p] is about to be deinitialized.", socketHandle);
        CHK_RET(hrtRaSocketDeInit(socketHandle));
        auto it = g_socketHandleInfo.handleMap.begin();
        while (it != g_socketHandleInfo.handleMap.end()) {
            if (it->second == socketHandle) {
                it = g_socketHandleInfo.handleMap.erase(it);
            } else {
                ++it;
            }
        }

        g_socketHandleInfo.handleRef.erase(socketHandle);
    }

    return HCCL_SUCCESS;
}

HcclResult hrtRaSocketNonBlockListenStart(struct SocketListenInfoT conn[], u32 num)
{
    CheckConnPort(conn, num);
    s32 ret = DlRaFunction::GetInstance().dlRaSocketListenStart(conn, num);
    if (ret == SOCK_EAGAIN) {
        return HCCL_E_AGAIN;
    } else if (ret == SOCK_EADDRINUSE) {
        HCCL_INFO(
            "ra socket listen could not start, due to the port[%u] has already been bound. "
            "please try another port or check the port status",
            (num > 0 ? conn[0].port : HCCL_INVALID_PORT));
        return HCCL_E_UNAVAIL;
    } else if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "errNo[0x%016llx] ra socket listen start fail. return[%d], num[%u]", HCCL_ERROR_CODE(HCCL_E_TCP_CONNECT),
            ret, num);
        for (u32 idx = 0; idx < num; idx++) {
            HCCL_ERROR("cur idx[%u] port[%u] phase[%u] err[%u]", idx, conn[idx].port, conn[idx].phase, conn[idx].err);
        }
        return HCCL_E_TCP_CONNECT;
    }

    return HCCL_SUCCESS;
}

HcclResult hrtRaSocketAcceptCreditAdd(struct SocketListenInfoT conn[], u32 num, u32 creditLimit)
{
    s32 ret = 0;
    ret = DlRaFunction::GetInstance().dlRaSocketAcceptCreditAdd(conn, num, creditLimit);
    CHK_PRT_RET(
        ret != 0,
        HCCL_ERROR(
            "socket accept credit add failed, ret[%d], port[%u], creditLimit[%d]", ret, conn[0].port, creditLimit),
        HCCL_E_TCP_CONNECT);
    return HCCL_SUCCESS;
}

HcclResult hrtRaSocketListenStart(struct SocketListenInfoT conn[], u32 num)
{
    s32 ret = 0;
    auto startTime = chrono::steady_clock::now();
    auto timeout = chrono::seconds(GetExternalInputHcclLinkTimeOut());
    CHK_PRT_RET(num == 0, HCCL_ERROR("[ListenStart][RaSocket] num is zero"), HCCL_E_PARA);
    while (true) {
        ret = hrtRaSocketNonBlockListenStart(conn, num);
        if (ret == 0) {
            break; // 成功跳出
        } else if (ret == HCCL_E_AGAIN) {
            bool bTimeout = ((chrono::steady_clock::now() - startTime) >= timeout);
            RPT_CALL_ERR(
                bTimeout, "ra socket listen failed. timeout[%d s], return[%d], num[%u]",
                GetExternalInputHcclLinkTimeOut(), ret, num);

            CHK_PRT_RET(
                bTimeout,
                HCCL_ERROR(
                    "[ListenStart][RaSocket]errNo[0x%016llx] ra socket listen start "
                    "timeout[%d s]. return[%d]",
                    HCCL_ERROR_CODE(HCCL_E_TIMEOUT), GetExternalInputHcclLinkTimeOut(), ret),
                HCCL_E_TIMEOUT);
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
        } else if (ret == HCCL_E_UNAVAIL) {
            return HCCL_E_UNAVAIL;
        } else {
            HCCL_ERROR("[hrtRaSocketListenStart]ra socket listen start fail, ret[%d]", ret);
            return HCCL_E_TCP_CONNECT;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult hrtRaSocketListenStop(struct SocketListenInfoT conn[], u32 num)
{
    s32 ret = 0;
    auto startTime = chrono::steady_clock::now();
    auto timeout = chrono::seconds(GetExternalInputHcclLinkTimeOut());
    CheckConnPort(conn, num);
    while (true) {
        ret = DlRaFunction::GetInstance().dlRaSocketListenStop(conn, num);
        if (ret == 0 || ret == SOCK_ENODEV) {
            break; // 成功跳出
        } else if (ret == SOCK_EAGAIN) {
            bool bTimeout = ((chrono::steady_clock::now() - startTime) >= timeout);
            if (!bTimeout) {
                SaluSleep(ONE_MILLISECOND_OF_USLEEP);
                continue;
            }
            HCCL_ERROR(
                "[ListenStop][RaSocket]errNo[0x%016llx] ra socket listen stop fail timeout[%d]s, ret[%d], num[%u]",
                HCCL_ERROR_CODE(HCCL_E_TIMEOUT), timeout, ret, num);
            for (u32 idx = 0; idx < num; idx++) {
                HCCL_ERROR(
                    "cur idx[%u] port[%u] phase[%u] err[%u]", idx, conn[idx].port, conn[idx].phase, conn[idx].err);
            }
            return HCCL_E_TIMEOUT;
        } else {
            HCCL_ERROR(
                "[ListenStop][RaSocket]errNo[0x%016llx] ra socket listen stop fail. return[%d], num[%u]",
                HCCL_ERROR_CODE(HCCL_E_TCP_CONNECT), ret, num);
            for (u32 idx = 0; idx < num; idx++) {
                HCCL_ERROR(
                    "cur idx[%u] port[%u] phase[%u] err[%u]", idx, conn[idx].port, conn[idx].phase, conn[idx].err);
            }
            return HCCL_E_TCP_CONNECT; // 非ra限速场景错误，不轮询，直接退出
        }
    }
    return HCCL_SUCCESS;
}

HcclResult hrtRaSocketNonBlockBatchAbort(SocketConnectInfoT conn[], u32 num)
{
    CheckConnPort(conn, num);
    s32 ret = DlRaFunction::GetInstance().dlRaSocketBatchAbort(conn, num);
    if (ret == 0) {
        return HCCL_SUCCESS;
    } else if (ret == SOCK_EAGAIN) {
        return HCCL_E_AGAIN;
    } else {
        HCCL_ERROR(
            "[hrtRaSocketNonBlockBatchAbort]errNo[0x%016llx] ra socket batch abort fail. "
            "return[%d], num[%u]",
            HCCL_ERROR_CODE(HCCL_E_TCP_CONNECT), ret, num);
        for (u32 idx = 0; idx < num; idx++) {
            HCCL_ERROR(
                "cur idx[%u] remoteIp[%u] port[%u] tag[%s]", idx, conn[idx].remoteIp.addr.s_addr, conn[idx].port,
                conn[idx].tag);
        }
        return HCCL_E_TCP_CONNECT;
    }

    return HCCL_SUCCESS;
}

HcclResult IsSupportRaSocketAbort(bool& isSupportRaSocketAbort)
{
    isSupportRaSocketAbort = false;
    s32 deviceLogicID = -1;
    u32 devicePhyId = 0;
    CHK_RET(hrtGetDevice(&deviceLogicID));
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(deviceLogicID), devicePhyId));
    u32 configVersion = 0;

    // 获取版本号查看是否兼容
    HcclResult ret = hrtRaGetInterfaceVersion(devicePhyId, SOCKET_ABORT, &configVersion);
    CHK_PRT_RET(
        ret == HCCL_E_NETWORK,
        HCCL_ERROR(
            "[IsSupportRaSendNormalWrlist]hrtRaGetInterfaceVersion "
            "failed, interface[%u]",
            SOCKET_ABORT),
        ret);
    if (ret == HCCL_E_NOT_SUPPORT) {
        HCCL_WARNING("this package does not support hrtRaGetInterfaceVersion, please change new package");
        return HCCL_SUCCESS;
    }

    if (configVersion >= SOCKET_ABORT_VERSION) {
        isSupportRaSocketAbort = true;
    }
    HCCL_INFO("isSupportRaSocketAbort support:%d, configVersion:%d", isSupportRaSocketAbort, configVersion);
    return HCCL_SUCCESS;
}

HcclResult hrtRaSocketNonBlockBatchConnect(SocketConnectInfoT conn[], u32 num)
{
    CheckConnPort(conn, num);
    s32 ret = DlRaFunction::GetInstance().dlRaSocketBatchConnect(conn, num);
    if (ret == 0) {
        return HCCL_SUCCESS;
    } else if (ret == SOCK_EAGAIN) {
        return HCCL_E_AGAIN;
    } else {
        HCCL_ERROR(
            "[HrtRaQpNonBlockConnectAsync]errNo[0x%016llx] ra socket batch connect fail. "
            "return[%d], num[%u]",
            HCCL_ERROR_CODE(HCCL_E_TCP_CONNECT), ret, num);
        for (u32 idx = 0; idx < num; idx++) {
            HCCL_ERROR(
                "cur idx[%u] remoteIp[%u] port[%u] tag[%s]", idx, conn[idx].remoteIp.addr.s_addr, conn[idx].port,
                conn[idx].tag);
        }
        return HCCL_E_TCP_CONNECT;
    }

    return HCCL_SUCCESS;
}

HcclResult SocketBatchConnect(SocketConnectInfoT conn[], u32 num, std::function<bool()> needStop)
{
    s32 ret = 0;
    auto startTime = chrono::steady_clock::now();
    auto timeout = chrono::seconds(GetExternalInputHcclLinkTimeOut());
    CheckConnPort(conn, num);
    while (true) {
        CHK_PRT_RET(needStop(), HCCL_ERROR("Terminating operation due to external request"), HCCL_E_INTERNAL);

        ret = DlRaFunction::GetInstance().dlRaSocketBatchConnect(conn, num);
        if (ret == 0) {
            break; // 成功跳出
        } else if (ret == SOCK_EAGAIN) {
            bool bTimeout = ((chrono::steady_clock::now() - startTime) >= timeout);
            RPT_CALL_ERR(
                bTimeout, "ra socket batch connect failed. timeout[%d s], return[%d]",
                GetExternalInputHcclLinkTimeOut(), ret);
            CHK_PRT_RET(
                bTimeout,
                HCCL_ERROR(
                    "[BatchConnect][RaSocket]errNo[0x%016llx] ra socket batch connect "
                    "timeout[%lld s]. return[%d]",
                    HCCL_ERROR_CODE(HCCL_E_TIMEOUT), GetExternalInputHcclLinkTimeOut(), ret),
                HCCL_E_TIMEOUT);
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
        } else {
            RPT_CALL_ERR_PRT("ra socket batch connect failed. return[%d]", ret);
            HCCL_ERROR(
                "[BatchConnect][RaSocket]errNo[0x%016llx] ra socket batch connect fail. return[%d], params: ",
                HCCL_ERROR_CODE(HCCL_E_TCP_CONNECT), ret);
            return HCCL_E_TCP_CONNECT; // 非ra限速场景错误，不轮询，直接退出
        }
    }
    return HCCL_SUCCESS;
}

HcclResult
hrtRaSocketBatchConnect(struct SocketConnectInfoT conn[], u32 num, u32 maxLen, std::function<bool()> needStop)
{
    CHK_PTR_NULL(conn);
    CHK_PRT_RET(
        (num > maxLen) || (num == 0),
        HCCL_ERROR(
            "[hrtRaSocketBatchConnect][RaSocket]ra socket batch connect "
            "para error, num[%u], maxLen[%u]",
            num, maxLen),
        HCCL_E_PARA);

    HCCL_INFO("batch connect, port[%u], remoteip[%x]", conn[0].port, conn[0].remoteIp);
    // batchConnect函数指针。底层接口一次最多建链16条，超过16条调用多次batch connect
    u32 exeNum = 0;
    SocketConnectInfoT* connBase = conn;
    while (num > 0) {
        exeNum = num > MAX_NUM_OF_BATCH_CONN ? MAX_NUM_OF_BATCH_CONN : num;
        CHK_RET(SocketBatchConnect(connBase, exeNum, needStop));
        connBase += exeNum;
        num -= exeNum;
    }

    return HCCL_SUCCESS;
}

HcclResult hrtRaSocketBatchClose(struct SocketCloseInfoT conn[], u32 num, u32 maxLen)
{
    CHK_PTR_NULL(conn);
    HCCL_INFO("ra socket batch close fdhandle[%p]", conn->fdHandle);
    CHK_PRT_RET(
        (num > maxLen) || (num == 0),
        HCCL_ERROR(
            "[BatchClose][RaSocket]ra socket batch connect para error "
            "num[%u], maxLen[%u]",
            num, maxLen),
        HCCL_E_PARA);
    s32 ret = 0;
    auto startTime = chrono::steady_clock::now();
    auto timeout = chrono::seconds(GetExternalInputHcclLinkTimeOut());
    while (true) {
        ret = DlRaFunction::GetInstance().dlRaSocketBatchClose(conn, num);
        if (ret == 0) {
            break; // 成功跳出
        } else if (ret == SOCK_EAGAIN) {
            bool bTimeout = ((chrono::steady_clock::now() - startTime) >= timeout);
            if (!bTimeout) {
                SaluSleep(ONE_MILLISECOND_OF_USLEEP);
                continue;
            }
            HCCL_ERROR(
                "[BatchClose][RaSocket]errNo[0x%016llx] ra socket batch close timeout[%d s], ret[%d], num[%u]",
                HCCL_ERROR_CODE(HCCL_E_TIMEOUT), timeout, ret, num);
            for (u32 idx = 0; idx < num; idx++) {
                HCCL_ERROR("cur idx[%u] disuseLinger[%d]", idx, conn[idx].disuseLinger);
            }
            return HCCL_E_TIMEOUT;
        } else {
            HCCL_ERROR(
                "[BatchClose][RaSocket]errNo[0x%016llx] ra socket batch close fail. return[%d], num[%u]",
                HCCL_ERROR_CODE(HCCL_E_TCP_CONNECT), ret, num);
            for (u32 idx = 0; idx < num; idx++) {
                HCCL_ERROR("cur idx[%u] disuseLinger[%d]", idx, conn[idx].disuseLinger);
            }
            return HCCL_E_TCP_CONNECT; // 非ra限速场景错误，不轮询，直接退出
        }
    }
    HCCL_INFO(
        "ra socket batch close success,take time [%lld]us",
        std::chrono::duration_cast<std::chrono::microseconds>(chrono::steady_clock::now() - startTime));
    return HCCL_SUCCESS;
}

s32 hrtRaGetSockets(u32 role, struct SocketInfoT conn[], u32 num, u32* connectedNum)
{
    return DlRaFunction::GetInstance().dlRaGetSockets(role, conn, num, connectedNum);
}

HcclResult hrtRaNonBlockGetSockets(u32 role, struct SocketInfoT conn[], u32 num, u32* connectedNum)
{
    CHK_PTR_NULL(conn);
    CHK_PRT_RET(num == 0, HCCL_ERROR("[hrtRaBlockGetSockets]ra get rasocket para error, num[%d]", num), HCCL_E_PARA);
    s32 ret = DlRaFunction::GetInstance().dlRaGetSockets(role, conn, num, connectedNum);
    if (ret == 0) {
        return HCCL_SUCCESS;
    } else if (ret == SOCK_EAGAIN) {
        return HCCL_E_AGAIN;
    } else {
        HCCL_ERROR(
            "[hrtRaNonBlockGetSockets]get ra socket error. role[%u], num[%u], ret[%d], connected num[%u]", role, num,
            ret, *connectedNum);
        for (u32 idx = 0; idx < num; idx++) {
            HCCL_ERROR(
                "cur idx[%u] socketHandle[%u] s_addr[%u] tag[%s]", idx, conn[idx].socketHandle,
                conn[idx].remoteIp.addr.s_addr, conn[idx].tag);
        }
        return HCCL_E_TCP_CONNECT;
    }

    return HCCL_SUCCESS;
}

HcclResult hrtRaBlockGetSockets(u32 role, struct SocketInfoT conn[], u32 num)
{
    CHK_PTR_NULL(conn);
    CHK_PRT_RET(num == 0, HCCL_ERROR("[hrtRaBlockGetSockets]ra get rasocket para error"), HCCL_E_PARA);
    s32 sockRet = 0;
    u32 gotSocketsCnt = 0;
    auto startTime = chrono::steady_clock::now();
    auto timeout = chrono::seconds(GetExternalInputHcclLinkTimeOut());
    while (true) {
        if ((chrono::steady_clock::now() - startTime) >= timeout) {
            HCCL_ERROR(
                "[hrtRaBlockGetSockets] get rasocket timeout role[%u], num[%u], goten[%u], "
                "timeout[%lld s], the HCCL_CONNECT_TIMEOUT may be insufficient.",
                role, num, gotSocketsCnt, timeout);
            return HCCL_E_TIMEOUT;
        }
        u32 connectedNum = 0;
        sockRet = hrtRaGetSockets(role, conn, num, &connectedNum);
        if ((connectedNum == 0 && sockRet == 0) || (sockRet == SOCK_EAGAIN)) {
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
        } else if (sockRet != 0) {
            HCCL_ERROR(
                "[Get][RaSocket]get rasocket error. role[%u], num[%u], sockRet[%d], connectednum[%u]", role, num,
                sockRet, connectedNum);
            return HCCL_E_TCP_CONNECT;
        } else {
            gotSocketsCnt += connectedNum;
            if (gotSocketsCnt == num) {
                HCCL_INFO("block get sockets success, socket num[%u]", gotSocketsCnt);
                break;
            } else if (gotSocketsCnt > num) {
                HCCL_ERROR("[Get][RaSocket]total Sockets[%u], more than needed num[%u]!", gotSocketsCnt, num);
                return HCCL_E_TCP_CONNECT;
            } else {
                SaluSleep(ONE_MILLISECOND_OF_USLEEP);
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult hrtRaSocketNonBlockSendHeterog(const FdHandle fdHandle, const void* data, u64 size, u64* sentSize)
{
    if (size > SOCKET_SEND_MAX_SIZE) {
        HCCL_ERROR(
            "[hrtRaSocketNonBlockSend]errNo[0x%016llx] ra socket send size is too large, "
            "data[%p], size[%llu Byte]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), data, size);
        return HCCL_E_PARA;
    }
    s32 ret = DlRaFunction::GetInstance().dlRaSocketSend(fdHandle, data, size, sentSize);
    if (ret == 0) {
        return HCCL_SUCCESS;
    } else if (ret == SOCK_EAGAIN) {
        return HCCL_E_AGAIN;
    } else {
        HCCL_RUN_WARNING(
            "[hrtRaSocketNonBlockSend]ra socket send failed, data[%p], size[%llu Byte], "
            "sent[%llu Byte], ret[%d]",
            data, size, *sentSize, ret);
        return HCCL_E_NETWORK;
    }

    return HCCL_SUCCESS;
}

s32 hrtRaSocketNonBlockSend(const FdHandle fdHandle, const void* data, u64 size, u64* sentSize)
{
    return DlRaFunction::GetInstance().dlRaSocketSend(fdHandle, data, size, sentSize);
}

HcclResult hrtRaSocketNonBlockSendHeart(const FdHandle fdHandle, const void* data, u64 size, u64* sentSize)
{
    if (size > SOCKET_SEND_MAX_SIZE) {
        HCCL_ERROR(
            "[hrtRaSocketNonBlockSend]errNo[0x%016llx] ra socket send size is too large, "
            "data[%p], size[%llu]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), data, size);
        return HCCL_E_PARA;
    }
    s32 ret = DlRaFunction::GetInstance().dlRaSocketSend(fdHandle, data, size, sentSize);
    if (ret == 0) {
        return HCCL_SUCCESS;
    } else if (ret == SOCK_EAGAIN) {
        return HCCL_E_AGAIN;
    } else if (ret == SOCK_CLOSE) {
        return HCCL_E_INTERNAL; // 暂时用这个错误表示hccp进程异常退出
    } else {
        HCCL_WARNING(
            "[hrtRaSocketNonBlockSend]ra socket send failed, fdHandle[%p], data[%p], size[%llu], "
            "sent[%llu], ret[%d], errno[%d][%s]",
            fdHandle, data, size, *sentSize, ret, errno, strerror(errno));
        return HCCL_E_NETWORK;
    }

    return HCCL_SUCCESS;
}

HcclResult hrtRaSocketBlockSend(const FdHandle fdHandle, const void* data, u64 sendSize, std::function<bool()> needStop)
{
    CHK_PTR_NULL(data);
    if (sendSize > SOCKET_SEND_MAX_SIZE) {
        HCCL_ERROR(
            "[Send][RaSocket]errNo[0x%016llx] ra socket send size is too large, "
            "data[%p], size[%llu Byte]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), data, sendSize);
        return HCCL_E_PARA;
    }
    s64 ret = 0;
    void* sendData = const_cast<void*>(data);
    const chrono::seconds timeout = chrono::seconds(GetExternalInputHcclLinkTimeOut());
    const auto start = chrono::steady_clock::now();
    u64 totalSentSize = 0;
    u64 sentSize = 0;

    HCCL_DEBUG("before ra socket send, para: data[%p], size[%llu Byte]", sendData, sendSize);

    while (true) {
        CHK_PRT_RET(needStop(), HCCL_ERROR("Terminating operation due to external request"), HCCL_E_INTERNAL);

        // 底层ra_socket_send host网卡无限制，device网卡由于HDC通道限制的限制有大小限制(目前大小为64KB)
        ret = DlRaFunction::GetInstance().dlRaSocketSend(
            fdHandle, reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(sendData) + totalSentSize),
            sendSize - totalSentSize, &sentSize);
        HCCL_DEBUG("ra socket send, data[%p], size[%llu Byte] send size[%llu Byte]", sendData, sendSize, totalSentSize);
        if (ret == 0) {
            totalSentSize += sentSize;
            if (totalSentSize == sendSize) { // 只有完全发送完才返回成功
                break;
            }

            CHK_PRT_RET(
                (totalSentSize > sendSize),
                HCCL_ERROR(
                    "[Send][RaSocket]errNo[0x%016llx] ra socket send failed, "
                    "data[%p], size[%llu Byte], retSize[%llu Byte]",
                    HCCL_ERROR_CODE(HCCL_E_NETWORK), data, sendSize, sentSize),
                HCCL_E_NETWORK);
            SaluSleep(ONE_HUNDRED_MICROSECOND_OF_USLEEP);
        } else if (ret == SOCK_EAGAIN) {
            /* ra速率限制 retry */
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
        } else {
            HCCL_ERROR(
                "[Send][RaSocket]errNo[0x%016llx] ra socket send failed, data[%p], size[%llu], "
                "sent[%llu Byte], ret[%d]",
                HCCL_ERROR_CODE(HCCL_E_NETWORK), data, sendSize, sentSize, ret);
            return HCCL_E_NETWORK;
        }

        /* 获取当前时间，如果耗时超过timeout，则返回错误 */
        const auto elapsed = chrono::duration_cast<chrono::seconds>(chrono::steady_clock::now() - start);
        if (elapsed > timeout) {
            HCCL_ERROR(
                "[Send][RaSocket]errNo[0x%016llx] Wait timeout for sockets send, data[%p], "
                "size[%llu Byte], sentsize[%llu Byte]",
                HCCL_ERROR_CODE(HCCL_E_NETWORK), data, sendSize, sentSize);
            return HCCL_E_TIMEOUT;
        }
    }
    HCCL_DEBUG("ra socket send finished.");
    return HCCL_SUCCESS;
}

s32 hrtRaSocketRecv(const FdHandle fdHandle, void* data, u64 size, u64* recvSize)
{
    return DlRaFunction::GetInstance().dlRaSocketRecv(fdHandle, data, size, recvSize);
}

HcclResult hrtRaSocketNonBlockRecvHeterog(const FdHandle fdHandle, void* data, u64 size, u64* recvSize)
{
    s32 ret = DlRaFunction::GetInstance().dlRaSocketRecv(fdHandle, data, size, recvSize);
    if (ret == 0) {
        return HCCL_SUCCESS;
    } else if (ret == SOCK_EAGAIN) {
        return HCCL_E_AGAIN;
    } else {
        HCCL_RUN_WARNING(
            "[hrtRaSocketNonBlockRecv]ra socket recv failed, data[%p], size[%llu Byte], "
            "recv[%llu Byte], ret[%d], errno[%d][%s]",
            data, size, recvSize, ret, errno, strerror(errno));
        return HCCL_E_TCP_TRANSFER;
    }

    return HCCL_SUCCESS;
}

s32 hrtRaSocketNonBlockRecv(const FdHandle fdHandle, void* data, u64 size, u64* recvSize)
{
    return DlRaFunction::GetInstance().dlRaSocketRecv(fdHandle, data, size, recvSize);
    ;
}

HcclResult hrtRaSocketNonBlockRecvHeart(const FdHandle fdHandle, void* data, u64 size, u64* recvSize)
{
    s32 ret = DlRaFunction::GetInstance().dlRaSocketRecv(fdHandle, data, size, recvSize);
    if (ret == 0) {
        return HCCL_SUCCESS;
    } else if (ret == SOCK_EAGAIN) {
        return HCCL_E_AGAIN;
    } else if (ret == SOCK_CLOSE) {
        return HCCL_E_INTERNAL; // 暂时用这个错误码表示hccp进程异常退出
    } else {
        HCCL_WARNING(
            "[hrtRaSocketNonBlockRecvHeart]ra socket recv failed, data[%p], size[%llu], "
            "recv[%llu], ret[%d], errno[%d][%s]",
            data, size, recvSize, ret, errno, strerror(errno));
        return HCCL_E_TCP_TRANSFER;
    }
    return HCCL_SUCCESS;
}

HcclResult
hrtRaSocketBlockRecv(const FdHandle fdHandle, void* data, u64 size, std::function<bool()> needStop, u32 timeout)
{
    auto startTime = chrono::steady_clock::now();
    void* recvData = const_cast<void*>(data);
    u64 recvSize = 0;
    s32 rtRet = 0;
    u64 getedLen = 0;
    const chrono::seconds timeoutSec = chrono::seconds(timeout > 0 ? timeout : GetExternalInputHcclLinkTimeOut());

    HCCL_DEBUG("before ra socket recv, para: data[%p], size[%llu]", recvData, size);
    while (true) {
        CHK_PRT_RET(needStop(), HCCL_ERROR("Terminating operation due to external request"), HCCL_E_INTERNAL);

        if ((chrono::steady_clock::now() - startTime) >= timeoutSec) {
            HCCL_ERROR(
                "[Recv][RaSocket]errNo[0x%016llx] Wait timeout for sockets recv, data[%p], "
                "size[%llu Byte], recvSize[%llu Byte] timeout[%lld s]. Peerrank did not send the data in time. "
                "Check whether the peerrank is abnormal.",
                HCCL_ERROR_CODE(HCCL_E_NETWORK), data, size, recvSize, timeoutSec);
            return HCCL_E_TIMEOUT;
        }
        rtRet = DlRaFunction::GetInstance().dlRaSocketRecv(
            fdHandle, reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(recvData) + getedLen), size - getedLen,
            &recvSize);
        if ((rtRet == 0) && (recvSize > 0)) { // 接收完成，也有可能要多次接收
            getedLen += recvSize;
            CHK_PRT_RET(
                getedLen > size,
                HCCL_ERROR(
                    "[Recv][RaSocket]errNo[0x%016llx] socket receive "
                    "rtSize[%llu Byte] bigger size[%zu Byte]",
                    HCCL_ERROR_CODE(HCCL_E_TCP_TRANSFER), getedLen, size),
                HCCL_E_TCP_TRANSFER);
            if (getedLen == size) {
                break;
            }
        } else if ((rtRet == 0) && (recvSize == 0)) {
            HCCL_ERROR("[Recv][RaSocket]recv fail, bufLen[%llu], recLen[%llu]", size, recvSize);
            return HCCL_E_TCP_TRANSFER;
        } else if (rtRet == SOCK_EAGAIN) {
            /* 尚未接收到数据,延时1ms */
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
            continue;
        } else if (rtRet != 0) { // 等于0为连接关闭，小于0的其他场景为出错
            HCCL_ERROR(
                "[Recv][RaSocket]errNo[0x%016llx] recv fail, data[%p], size[%llu], rtRet[%d]",
                HCCL_ERROR_CODE(HCCL_E_TCP_TRANSFER), data, size, rtRet);
            return HCCL_E_TCP_TRANSFER;
        }
    }
    HCCL_DEBUG("ra socket receive finished");
    return HCCL_SUCCESS;
}

HcclResult IsSupportHdcAsync(bool& isSupportHdcAsync)
{
    isSupportHdcAsync = false;
    s32 deviceLogicID = -1;
    u32 devicePhyId = 0;
    CHK_RET(hrtGetDevice(&deviceLogicID));
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(deviceLogicID), devicePhyId));
    u32 version = 0;

    // 获取版本号查看是否兼容
    HcclResult ret = hrtRaGetInterfaceVersion(devicePhyId, RS_INIT, &version);
    CHK_PRT_RET(
        ret == HCCL_E_NETWORK,
        HCCL_ERROR(
            "[IsSupportHdcAsync]hrtRaGetInterfaceVersion "
            "failed, interface[%u]",
            RS_INIT),
        ret);
    if (ret == HCCL_E_NOT_SUPPORT) {
        HCCL_WARNING("this package does not support hrtRaGetInterfaceVersion, please change new package");
        return HCCL_SUCCESS;
    }

    if (version >= RS_INIT_SUPPORT_ASYNC_VERSION) {
        isSupportHdcAsync = true;
    }

    HCCL_INFO("[IsSupportHdcAsync] isSupportHdcAsync[%d], version[%d]", isSupportHdcAsync, version);
    return HCCL_SUCCESS;
}

s32 hrtRaSocketSendAsync(const FdHandle fdHandle, const void* data, u64 size, u64* sentSize, void** reqHandle)
{
    if (DlRaFunction::GetInstance().dlRaSocketSendAsync == nullptr) {
        HCCL_WARNING("this package does not support hrtRaSocketSendAsync, please change new package");
        return OTHERS_ENOTSUPP;
    }
    return DlRaFunction::GetInstance().dlRaSocketSendAsync(fdHandle, data, size, sentSize, reqHandle);
}

s32 hrtRaSocketRecvAsync(const FdHandle fdHandle, void* data, u64 size, u64* receivedSize, void** reqHandle)
{
    if (DlRaFunction::GetInstance().dlRaSocketRecvAsync == nullptr) {
        HCCL_WARNING("this package does not support hrtRaSocketRecvAsync, please change new package");
        return OTHERS_ENOTSUPP;
    }
    return DlRaFunction::GetInstance().dlRaSocketRecvAsync(fdHandle, data, size, receivedSize, reqHandle);
}

s32 hrtRaSocketGetAsyncReqResult(void* reqHandle, s32* reqResult)
{
    if (DlRaFunction::GetInstance().dlRaGetAsyncReqResult == nullptr) {
        HCCL_WARNING("this package does not support hrtRaSocketGetAsyncReqResult, please change new package");
        return OTHERS_ENOTSUPP;
    }
    return DlRaFunction::GetInstance().dlRaGetAsyncReqResult(reqHandle, reqResult);
}

HcclResult hrtGetHostIf(vector<pair<string, HcclIpAddress>>& hostIfs, u32 devPhyId)
{
    struct RaGetIfattr config = {};
    config.phyId = devPhyId;
    config.nicPosition = static_cast<u32>(NICDeployment::NIC_DEPLOYMENT_HOST);
    config.isAll = false;

    u32 ifAddrNum = 0;
    CHK_RET(hrtGetIfNum(config, ifAddrNum));
    HCCL_RUN_INFO("[Get][HostIf]hrtGetIfNum success. ifAddrNum[%u].", ifAddrNum);
    if (ifAddrNum == 0) {
        HCCL_WARNING("[Get][HostIf]there is no valid host interface, ifAddrNum[%u].", ifAddrNum);
        return HCCL_SUCCESS;
    }

    struct InterfaceInfo* ifAddrInfos;
    NEW_NOTHROW(ifAddrInfos, struct InterfaceInfo[ifAddrNum], return HCCL_E_MEMORY);
    shared_ptr<struct InterfaceInfo> ifAddrInfoPtrs(ifAddrInfos, default_delete<struct InterfaceInfo[]>());

    s32 sRet = memset_s(ifAddrInfos, ifAddrNum * sizeof(InterfaceInfo), 0, ifAddrNum * sizeof(InterfaceInfo));
    if (sRet != EOK) {
        HCCL_ERROR(
            "[Get][HostIf]errNo[0x%016llx] memoryset ifAddrInfos to 0 failed. params: "
            "dest[%p], dest_size[%zu Byte], count[%zu]",
            HCCL_ERROR_CODE(HCCL_E_SYSCALL), ifAddrInfos, ifAddrNum * sizeof(InterfaceInfo),
            ifAddrNum * sizeof(InterfaceInfo));
        return HCCL_E_SYSCALL;
    }
    CHK_RET(hrtGetIfAddress(config, ifAddrInfos, ifAddrNum));

    for (u32 i = 0; i < ifAddrNum; i++) {
        HcclInAddr temp;
        temp.addr = ifAddrInfos[i].ifaddr.ip.addr;
        temp.addr6 = ifAddrInfos[i].ifaddr.ip.addr6;
        HcclIpAddress ipInfo(ifAddrInfos[i].family, temp);
        CHK_PRT_RET(ipInfo.IsInvalid(), HCCL_ERROR("ip is invalid."), HCCL_E_PARA);
        CHK_RET(ipInfo.SetIfName(ifAddrInfos[i].ifname));
        CHK_RET(ipInfo.SetScopeID(ifAddrInfos[i].scopeId));
        hostIfs.push_back({ifAddrInfos[i].ifname, ipInfo});
        HCCL_INFO(
            "[Get][HostIf]hrtGetIfAddress: idx[%u] ifname[%s] ip[%s]", i, ifAddrInfos[i].ifname,
            ipInfo.GetReadableAddress());
    }

    return HCCL_SUCCESS;
}

HcclResult hrtEpollCtlAdd(const FdHandle fdHandle, RaEpollEvent event)
{
    s32 ret = DlRaFunction::GetInstance().dlRaEpollCtlAdd(fdHandle, event);
    CHK_PRT_RET(ret != 0, HCCL_ERROR("[Add][EpollCtl] failed"), HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult hrtEpollCtlMod(const FdHandle fdHandle, RaEpollEvent event)
{
    s32 ret = DlRaFunction::GetInstance().dlRaEpollCtlMod(fdHandle, event);
    CHK_PRT_RET(ret != 0, HCCL_ERROR("[Mod][EpollCtl] failed"), HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult hrtEpollCtlDel(const FdHandle fdHandle)
{
    s32 ret = DlRaFunction::GetInstance().dlRaEpollCtlDel(fdHandle);
    CHK_PRT_RET(ret != 0, HCCL_ERROR("[Del][EpollCtl] failed"), HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult hrtSetRecvDataCallback(const SocketHandle socketHandle, const void* callback)
{
    s32 ret = DlRaFunction::GetInstance().dlRaSetRecvDataCallback(socketHandle, callback);
    CHK_PRT_RET(ret != 0, HCCL_ERROR("[Set][RecvDataCallback] failed"), HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}
#endif

#if T_DESC("WhiteList", true)

HcclResult hrtRaSocketSetWhiteListStatus(u32 enable)
{
    s32 ret = DlRaFunction::GetInstance().dlRaSocketSetWhiteListStatus(enable);
    CHK_PRT_RET(
        ret != 0,
        HCCL_ERROR(
            "[Set][WhiteListStatus]errNo[0x%016llx] ra socket set white list fail, return[%d]."
            " para: enable[%u]",
            HCCL_ERROR_CODE(HCCL_E_TCP_CONNECT), ret, enable),
        HCCL_E_TCP_CONNECT);
    HCCL_INFO("set host socket whitelist status[%u] success.", enable);
    return HCCL_SUCCESS;
}

HcclResult hrtRaSocketGetWhiteListStatus(u32& enable)
{
    s32 ret = DlRaFunction::GetInstance().dlRaSocketGetWhiteListStatus(&enable);
    CHK_PRT_RET(
        ret != 0,
        HCCL_ERROR(
            "[Get][WhiteListStatus]errNo[0x%016llx] ra socket get white list fail, return[%d].",
            HCCL_ERROR_CODE(HCCL_E_TCP_CONNECT), ret),
        HCCL_E_TCP_CONNECT);
    return HCCL_SUCCESS;
}

HcclResult hrtRaSocketWhiteListAdd(SocketHandle socketHandle, struct SocketWlistInfoT whiteList[], u32 num)
{
    HCCL_INFO("add white list: num[%u].", num);
    for (u32 i = 0; i < num; i++) {
        HCCL_DEBUG(
            "add white list: idx[%u], remoteIp[%u], tag[%s].", i, whiteList[i].remoteIp.addr.s_addr, whiteList[i].tag);
        s32 ret = DlRaFunction::GetInstance().dlRaSocketWhiteListAdd(socketHandle, whiteList + i, 1);
        CHK_PRT_RET(
            ret != 0,
            HCCL_ERROR(
                "[Add][RaSocketWhiteList]errNo[0x%016llx] ra white list add fail, return[%d].",
                HCCL_ERROR_CODE(HCCL_E_TCP_CONNECT), ret),
            HCCL_E_TCP_CONNECT);
    }

    return HCCL_SUCCESS;
}

HcclResult hrtRaSocketWhiteListDel(SocketHandle socketHandle, struct SocketWlistInfoT whiteList[], u32 num)
{
    HCCL_DEBUG("delete white list: num[%u].", num);
    for (u32 i = 0; i < num; i++) {
        HCCL_DEBUG(
            "del white list: idx[%u], remoteIp[%u], tag[%s].", i, whiteList[i].remoteIp.addr.s_addr, whiteList[i].tag);
        s32 ret = DlRaFunction::GetInstance().dlRaSocketWhiteListDel(socketHandle, whiteList + i, 1);
        CHK_PRT_RET(
            ret != 0,
            HCCL_ERROR(
                "[Del][RaSocketWhiteList]errNo[0x%016llx] ra white list del fail, return[%d].",
                HCCL_ERROR_CODE(HCCL_E_TCP_CONNECT), ret),
            HCCL_E_TCP_CONNECT);
    }

    return HCCL_SUCCESS;
}

#endif

HcclResult hrtGetIfNum([[maybe_unused]] struct RaGetIfattr& config, [[maybe_unused]] u32& num)
{
#ifndef HCCD
    if (DlRaFunction::GetInstance().dlRaGetIfNum == nullptr) {
        HCCL_WARNING("this package does not support hrtGetIfNum, please change new package");
        return HCCL_SUCCESS;
    }

    s32 ret = DlRaFunction::GetInstance().dlRaGetIfNum(&config, &num);
    constexpr s32 MAX_SUPPORT_IFNUM = 65536;
    CHK_PRT_RET(
        (ret != 0 || num > MAX_SUPPORT_IFNUM),
        HCCL_ERROR(
            "[Get][IfNum]errNo[0x%016llx] ra get if num fail."
            " ret[%d], num[%u] should be less than [%u]",
            HCCL_ERROR_CODE(HCCL_E_TCP_CONNECT), ret, num, MAX_SUPPORT_IFNUM),
        HCCL_E_TCP_CONNECT);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtGetIfNum]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtGetIfAddress(
    [[maybe_unused]] struct RaGetIfattr& config, [[maybe_unused]] struct InterfaceInfo ifaddrInfos[],
    [[maybe_unused]] u32& num)
{
#ifndef HCCD
    CHK_PRT_RET(
        num == 0,
        HCCL_ERROR(
            "[Get][IfAddress]errNo[0x%016llx] ra get if address fail. input param num[%u] "
            "is invalid.",
            HCCL_ERROR_CODE(HCCL_E_INTERNAL), num),
        HCCL_E_INTERNAL);
    s32 ret = DlRaFunction::GetInstance().dlRaGetIfAddress(&config, ifaddrInfos, &num);
    CHK_PRT_RET(
        ret != 0,
        HCCL_ERROR(
            "[Get][IfAddress]errNo[0x%016llx] ra get if address fail. ret[%d], num[%u]",
            HCCL_ERROR_CODE(HCCL_E_TCP_CONNECT), ret, num),
        HCCL_E_TCP_CONNECT);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtGetIfAddress]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtRaGetDeviceIP(u32 devicePhyId, vector<HcclIpAddress>& ipAddr)
{
    struct RaGetIfattr config = {};
    config.phyId = devicePhyId;
    config.nicPosition = static_cast<u32>(NICDeployment::NIC_DEPLOYMENT_DEVICE);
    config.isAll = false;

    u32 ifAddrNum = HCCL_DEVICE_NIC_NUM;
    CHK_RET(hrtGetIfNum(config, ifAddrNum));
    ifAddrNum = ifAddrNum > HCCL_DEVICE_NIC_NUM ? HCCL_DEVICE_NIC_NUM : ifAddrNum;
    HCCL_RUN_INFO("[Get][DeviceIP]hrtGetIfNum success. ifAddrNum[%u].", ifAddrNum);

    if (ifAddrNum == 0) {
        HCCL_WARNING("[Get][DeviceIP]device has no ip information, phyId[%u]", devicePhyId);
        return HCCL_SUCCESS;
    }

    struct InterfaceInfo ifAddrInfos[HCCL_DEVICE_NIC_NUM];
    s32 sRet = memset_s(
        ifAddrInfos, sizeof(InterfaceInfo) * HCCL_DEVICE_NIC_NUM, 0, sizeof(InterfaceInfo) * HCCL_DEVICE_NIC_NUM);
    CHK_PRT_RET(
        sRet != EOK,
        HCCL_ERROR(
            "[Get][DeviceIP]errNo[0x%016llx] memoryset ifAddrInfos to 0 failed. params: "
            "dest[%p], dest_size[%zu Byte], count[%zu]",
            HCCL_ERROR_CODE(HCCL_E_SYSCALL), ifAddrInfos, sizeof(InterfaceInfo) * HCCL_DEVICE_NIC_NUM,
            sizeof(InterfaceInfo) * HCCL_DEVICE_NIC_NUM),
        HCCL_E_SYSCALL);

    CHK_RET(hrtGetIfAddress(config, ifAddrInfos, ifAddrNum));

    CHK_PRT_RET(
        ifAddrNum > HCCL_DEVICE_NIC_NUM,
        HCCL_ERROR(
            "[Get][DeviceIP]hrtGetIfAddress fail. ifAddrNum[%u] should be below %u", ifAddrNum, HCCL_DEVICE_NIC_NUM),
        HCCL_E_TCP_CONNECT);

    for (u32 i = 0; i < ifAddrNum; i++) {
        HcclInAddr temp;
        temp.addr = ifAddrInfos[i].ifaddr.ip.addr;
        temp.addr6 = ifAddrInfos[i].ifaddr.ip.addr6;
        HcclIpAddress ipInfo(ifAddrInfos[i].family, temp);
        CHK_PRT_RET(ipInfo.IsInvalid(), HCCL_ERROR("ip is invalid."), HCCL_E_PARA);
        CHK_RET(ipInfo.SetIfName(ifAddrInfos[i].ifname));
        CHK_RET(ipInfo.SetScopeID(ifAddrInfos[i].scopeId));
        ipAddr.push_back(ipInfo);
        HCCL_RUN_INFO(
            "[Get][DeviceIP]hrtGetIfAddress: idx[%u] ifname[%s] ip[%s]", i, ifAddrInfos[i].ifname,
            ipInfo.GetReadableAddress());
    }

    return HCCL_SUCCESS;
}

HcclResult hrtRaGetDeviceAllNicIP(vector<vector<HcclIpAddress>>& ipAddr)
{
    s32 deviceLogicID = -1;
    u32 devicePhyId = 0;
    CHK_RET(hrtGetDevice(&deviceLogicID));
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(deviceLogicID), devicePhyId));
    // 获取版本号查看是否兼容
    u32 ifnumVersion = 0;
    HcclResult vRet = hrtRaGetInterfaceVersion(devicePhyId, IFADDRS_V2_INTERFACE, &ifnumVersion);
    if (vRet != HCCL_SUCCESS || ifnumVersion < IFADDRS_V2_INTERFACE_VERSTOIN) {
        HCCL_WARNING("this package does not support hrtRaGetDeviceAllNicIP, please change new package.");
        return HCCL_SUCCESS;
    }
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    CHK_PRT_RET(
        deviceType != DevType::DEV_TYPE_910_93 && deviceType != DevType::DEV_TYPE_910B,
        HCCL_ERROR("[Get][DeviceAllNicIP] is not supported on device type[%d]. Please check device type.", deviceType),
        HCCL_E_NOT_SUPPORT);

    struct RaGetIfattr config = {};
    config.phyId = devicePhyId;
    config.nicPosition = static_cast<u32>(NICDeployment::NIC_DEPLOYMENT_DEVICE);
    config.isAll = true;

    u32 nicNum = deviceType == DevType::DEV_TYPE_910_93 ? ALL_NIC_NUM_910_93 : ALL_NIC_NUM_910_A2;
    u32 maxNicIpNum = HCCL_DEVICE_NIC_NUM * nicNum;

    u32 ifAddrNum = maxNicIpNum;
    CHK_RET(hrtGetIfNum(config, ifAddrNum));
    ifAddrNum = ifAddrNum > maxNicIpNum ? maxNicIpNum : ifAddrNum;
    HCCL_RUN_INFO("[Get][DeviceAllNicIP]hrtGetIfNum success. ifAddrNum[%u].", ifAddrNum);

    if (ifAddrNum == 0) {
        HCCL_WARNING("[Get][DeviceAllNicIP]device has no ip information, phyId[%u]", devicePhyId);
        return HCCL_SUCCESS;
    }

    struct InterfaceInfo ifAddrInfos[HCCL_DEVICE_NIC_NUM * MAX_ALL_NIC_NUM] = {};
    CHK_RET(hrtGetIfAddress(config, ifAddrInfos, ifAddrNum));
    CHK_PRT_RET(
        ifAddrNum > maxNicIpNum,
        HCCL_ERROR(
            "[Get][DeviceAllNicIP]hrtGetIfAddress fail. ifAddrNum[%u] should be below %u", ifAddrNum, maxNicIpNum),
        HCCL_E_TCP_CONNECT);

    unordered_map<string, size_t> ifname2Index;
    for (u32 i = 0; i < ifAddrNum; i++) {
        HcclInAddr temp;
        temp.addr = ifAddrInfos[i].ifaddr.ip.addr;
        temp.addr6 = ifAddrInfos[i].ifaddr.ip.addr6;
        HcclIpAddress ipInfo(ifAddrInfos[i].family, temp);
        CHK_PRT_RET(ipInfo.IsInvalid(), HCCL_ERROR("ip is invalid."), HCCL_E_PARA);
        CHK_RET(ipInfo.SetIfName(ifAddrInfos[i].ifname));
        CHK_RET(ipInfo.SetScopeID(ifAddrInfos[i].scopeId));
        if (ifname2Index.find(ifAddrInfos[i].ifname) == ifname2Index.end()) {
            ifname2Index.emplace(ifAddrInfos[i].ifname, ipAddr.size());
            ipAddr.emplace_back(vector<HcclIpAddress>());
        }
        ipAddr[ifname2Index[ifAddrInfos[i].ifname]].push_back(ipInfo);
        HCCL_RUN_INFO(
            "[Get][DeviceAllNicIP]hrtGetIfAddress: idx[%u] ifname[%s] ip[%s]", i, ifAddrInfos[i].ifname,
            ipInfo.GetReadableAddress());
    }

    return HCCL_SUCCESS;
}

HcclResult hrtRaGetInterfaceVersion(unsigned int phyId, unsigned int interfaceOpcode, unsigned int* interfaceVersion)
{
    HCCL_DEBUG("hrtRaGetInterfaceVersion phyId[%u], opCode[%u]", phyId, interfaceOpcode);
    if (DlRaFunction::GetInstance().dlRaGetInterfaceVersion == nullptr) {
        HCCL_WARNING("driver package does not support hrtRaGetInterfaceVersion, please change new package");
        return HCCL_E_NOT_SUPPORT;
    }
    s32 ret = DlRaFunction::GetInstance().dlRaGetInterfaceVersion(phyId, interfaceOpcode, interfaceVersion);
    CHK_PRT_RET(
        ret != 0,
        HCCL_ERROR(
            "[Get][InterfaceVersion]errNo[0x%016llx] ra get interface version fail. ret[%d]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), ret),
        HCCL_E_NETWORK);
    HCCL_INFO("hrtRaGetInterfaceVersion phyId[%u], opCode[%u], version[%u]", phyId, interfaceOpcode, *interfaceVersion);
    return HCCL_SUCCESS;
}

HcclResult GetIsSupSockBatchCloseImmed(u32 phyId, bool& isSupportBatchClose)
{
    u32 batchCloseVersion = 0;
    isSupportBatchClose = false;
    // 获取版本号看是否兼容
    HcclResult ret = hrtRaGetInterfaceVersion(phyId, SOCKET_BATCH_CLOSE_INTERFACE, &batchCloseVersion);
    CHK_PRT_RET(
        ret == HCCL_E_NETWORK,
        HCCL_ERROR(
            "[Get][IsSupSockBatchCloseImmed]comm base hrtRaGetInterfaceVersion "
            "failed, interface[%u]",
            SOCKET_BATCH_CLOSE_INTERFACE),
        ret);
    if (ret == HCCL_E_NOT_SUPPORT) {
        HCCL_WARNING("this package does not support hrtRaGetInterfaceVersion, please change new package");
        return HCCL_SUCCESS;
    }
    if (batchCloseVersion >= SOCKET_BATCH_CLOSE_SUP_VER) {
        isSupportBatchClose = true;
    }
    return HCCL_SUCCESS;
}

HcclResult hrtRaCreateCq(RdmaHandle handle, struct CqAttr* attr)
{
    CHK_PTR_NULL(handle);
    CHK_PTR_NULL(attr);
    CHK_PTR_NULL(attr->ibSendCq);
    CHK_PTR_NULL(attr->ibRecvCq);
    CHK_PTR_NULL(attr->qpContext);
    HCCL_DEBUG(
        "ra create cq: sendCqDepth[%d], recvCqDepth[%d], sendCqEventId[%d], recvCqEventId[%d]", attr->sendCqDepth,
        attr->recvCqDepth, attr->sendCqEventId, attr->recvCqEventId);
    s32 ret = DlRaFunction::GetInstance().dlRaCreateCq(handle, attr);
    CHK_PRT_RET(
        ret != 0,
        HCCL_ERROR(
            "[Create][RaCq]errNo[0x%016llx] ra create cq fail. return[%d] "
            "sendCqDepth[%d], recvCqDepth[%d], sendCqEventId[%d], recvCqEventId[%d]",
            HCCL_ERROR_CODE(HCCL_E_INTERNAL), ret, attr->sendCqDepth, attr->recvCqDepth, attr->sendCqEventId,
            attr->recvCqEventId),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

map<string, vector<CqInfo>> g_qpRecords;
mutex g_qpRecordsMutex;
HcclResult CreateCq(RdmaHandle rdmaHandle, CqInfo& cq)
{
    struct CqAttr attr = {};
    attr.qpContext = &cq.context;
    attr.ibSendCq = &cq.sq;
    attr.ibRecvCq = &cq.rq;
    attr.sendCqDepth = cq.depth;
    attr.recvCqDepth = cq.depth;

    attr.sendCqEventId = cq.sqEvent;
    attr.recvCqEventId = cq.rqEvent;
    attr.sendChannel = cq.sendChannel;
    attr.recvChannel = cq.recvChannel;
    attr.srqContext = cq.srqContext;
    CHK_RET(hrtRaCreateCq(rdmaHandle, &attr));
    return HCCL_SUCCESS;
}

HcclResult hrtRaDestroyCq(RdmaHandle handle, struct CqAttr* attr)
{
    CHK_PTR_NULL(handle);
    CHK_PTR_NULL(attr);
    s32 ret = DlRaFunction::GetInstance().dlRaDestroyCq(handle, attr);
    CHK_PRT_RET(
        ret != 0,
        HCCL_ERROR("[Destroy][RaCq]errNo[0x%016llx] ra destroy cq fail. ret[%d]", HCCL_ERROR_CODE(HCCL_E_NETWORK), ret),
        HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult
hrtRaNormalQpCreate(RdmaHandle handle, struct ibv_qp_init_attr* initAttr, QpHandle& qpHandle, struct ibv_qp*& qp)
{
    CHK_PTR_NULL(handle);
    CHK_PTR_NULL(initAttr);
    HCCL_DEBUG("ra normal qp create: initAttr[%p]", initAttr);
    s32 ret
        = DlRaFunction::GetInstance().dlRaNormalQpCreate(handle, initAttr, &qpHandle, reinterpret_cast<void**>(&qp));

    std::string qpInfo = std::string("qp_type[") + std::to_string(initAttr->qp_type) + std::string("] ")
                         + std::string("max_inline_data[") + std::to_string(initAttr->cap.max_inline_data)
                         + std::string("] ") + std::string("max_send_wr[") + std::to_string(initAttr->cap.max_send_wr)
                         + std::string("] ") + std::string("max_send_sge[") + std::to_string(initAttr->cap.max_send_sge)
                         + std::string("] ") + std::string("max_recv_wr[") + std::to_string(initAttr->cap.max_recv_wr)
                         + std::string("] ") + std::string("max_recv_sge[") + std::to_string(initAttr->cap.max_recv_sge)
                         + std::string("]");

    CHK_OOM_RET(ret, qpInfo.c_str());

    CHK_PRT_RET(
        ret != 0,
        HCCL_ERROR(
            "[Create][NormalQp]errNo[0x%016llx] ra create normal qp fail.ret[%d]"
            "qp_type[%u] max_inline_data[%u] max_send_wr[%u] max_send_sge[%u] max_recv_wr[%u] max_recv_sge[%u]",
            HCCL_ERROR_CODE(HCCL_E_INTERNAL), ret, initAttr->qp_type, initAttr->cap.max_inline_data,
            initAttr->cap.max_send_wr, initAttr->cap.max_send_sge, initAttr->cap.max_recv_wr,
            initAttr->cap.max_recv_sge),
        HCCL_E_INTERNAL);

    struct QpAttr attr {};
    CHK_RET(hrtRaGetQpAttr(qpHandle, &attr));
    s32 deviceId = 0;
    if (hrtGetDevice(&deviceId) != HCCL_SUCCESS) {
        deviceId = -1;
    }
    PLF_CONFIG_DEBUG(
        PLF_RES,
        "Create Qp para: deviceId[%d] qpn[%u] qp_type[%u] max_inline_data[%u] max_send_wr[%u] max_send_sge[%u] "
        "max_recv_wr[%u] max_recv_sge[%u]",
        deviceId, attr.qpn, initAttr->qp_type, initAttr->cap.max_inline_data, initAttr->cap.max_send_wr,
        initAttr->cap.max_send_sge, initAttr->cap.max_recv_wr, initAttr->cap.max_recv_sge);
    return HCCL_SUCCESS;
}

HcclResult hrtRaNormalQpDestroy(QpHandle qpHandle)
{
    struct QpAttr attr {};
    CHK_RET(hrtRaGetQpAttr(qpHandle, &attr));
    s32 deviceId = 0;
    if (hrtGetDevice(&deviceId) != HCCL_SUCCESS) {
        deviceId = -1;
    }
    PLF_CONFIG_DEBUG(PLF_RES, "Destroy Qp para: deviceId[%d] qpn[%u]", deviceId, attr.qpn);

    CHK_PTR_NULL(qpHandle);
    s32 ret = DlRaFunction::GetInstance().dlRaNormalQpDestroy(qpHandle);
    CHK_PRT_RET(
        ret != 0,
        HCCL_ERROR(
            "[Destroy][NormalQp]errNo[0x%016llx] ra destroy normal qp fail. ret[%d] qpHandle[%p]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), ret, qpHandle),
        HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult DestroyCq(RdmaHandle rdmaHandle, CqInfo& cq)
{
    struct CqAttr attr = {};
    attr.qpContext = &cq.context;
    attr.ibSendCq = &cq.sq;
    attr.ibRecvCq = &cq.rq;
    CHK_RET(hrtRaDestroyCq(rdmaHandle, &attr));
    return HCCL_SUCCESS;
}

HcclResult ConstructQpAttrs(s32 qpMode, struct QpExtAttrs& attrs, const QueueDepthAttr& qpDepth, bool isWorkFlowLib)
{
    HCCL_INFO(
        "[ConstructQpAttrs][qpDepth]sendCqDepth[%u], recvCqDepth[%u], sqDepth[%u], rqDepth[%u]", qpDepth.sendCqDepth,
        qpDepth.recvCqDepth, qpDepth.sqDepth, qpDepth.rqDepth);
    CHK_PRT_RET(CheckQpDepth(qpDepth.sendCqDepth) != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CheckQpDepth]sendCqDepth[%u] is invalid, sendCqDepth should be power of 2 and in [%u, %u]",
                    qpDepth.sendCqDepth, QP_DEPTH_MIN, QP_DEPTH_MAX);
                , HCCL_E_PARA);
    CHK_PRT_RET(CheckQpDepth(qpDepth.recvCqDepth) != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CheckQpDepth]recvCqDepth[%u] is invalid, recvCqDepth should be power of 2 and in [%u, %u]",
                    qpDepth.recvCqDepth, QP_DEPTH_MIN, QP_DEPTH_MAX);
                , HCCL_E_PARA);
    CHK_PRT_RET(CheckQpDepth(qpDepth.sqDepth) != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CheckQpDepth]sqDepth[%u] is invalid, sqDepth should be power of 2 and in [%u, %u]",
                    qpDepth.sqDepth, QP_DEPTH_MIN, QP_DEPTH_MAX);
                , HCCL_E_PARA);
    CHK_PRT_RET(CheckQpDepth(qpDepth.rqDepth) != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CheckQpDepth]rqDepth[%u] is invalid, rqDepth should be power of 2 and in [%u, %u]",
                    qpDepth.rqDepth, QP_DEPTH_MIN, QP_DEPTH_MAX);
                , HCCL_E_PARA);

    attrs.qpMode = qpMode;
    attrs.version = QP_CREATE_WITH_ATTR_VERSION;
    attrs.cqAttr.recvCqDepth = (qpDepth.recvCqDepth == INVALID_UINT) ? DEFAULT_MAX_RECV_CQ_DEPTH : qpDepth.recvCqDepth;
    attrs.qpAttr.cap.max_inline_data = DEFAULT_MAX_INLINE_DATA;
    attrs.qpAttr.cap.max_send_sge = DEFAULT_MAX_SEND_SGE;
    attrs.qpAttr.cap.max_recv_wr = (qpDepth.rqDepth == INVALID_UINT) ? DEFAULT_MAX_RECV_WR : qpDepth.rqDepth;
    attrs.qpAttr.cap.max_recv_sge = DEFAULT_MAX_RECV_SGE;
    attrs.qpAttr.qp_type = IBV_QPT_RC;

    if (qpDepth.sqDepth == INVALID_UINT) {
        if (qpMode == OFFLINE_QP_MODE_EXT || isWorkFlowLib) {
            attrs.qpAttr.cap.max_send_wr = DEFAULT_OFFLINE_MAX_SEND_WR;
        } else {
            attrs.qpAttr.cap.max_send_wr = DEFAULT_OPBASE_MAX_SEND_WR;
        }
    } else {
        attrs.qpAttr.cap.max_send_wr = qpDepth.sqDepth;
    }
    if (qpDepth.sendCqDepth == INVALID_UINT) {
        attrs.cqAttr.sendCqDepth = DEFAULT_MAX_SEND_CQ_DEPTH;
        if (qpMode == OFFLINE_QP_MODE_EXT || qpMode == OFFLINE_QP_MODE || isWorkFlowLib) {
            attrs.cqAttr.sendCqDepth = HCCL_SEND_CQ_DEPTH_DEFAULT;
        }
    } else {
        attrs.cqAttr.sendCqDepth = qpDepth.sendCqDepth;
    }
    HCCL_INFO(
        "[ConstructQpAttrs][attr]sendCqDepth[%d], recvCqDepth[%d], max_send_wr[%u], max_recv_wr[%u]",
        attrs.cqAttr.sendCqDepth, attrs.cqAttr.recvCqDepth, attrs.qpAttr.cap.max_send_wr, attrs.qpAttr.cap.max_recv_wr);
    return HCCL_SUCCESS;
}

HcclResult CreateQp(RdmaHandle rdmaHandle, int& flag, s32& qpMode, QpInfo& qp, bool isESMode)
{
    HCCL_INFO("CreateQp qpMode[%d], isESMode[%d].", qpMode, isESMode);
    if (isESMode && (qpMode == OFFLINE_QP_MODE_EXT || qpMode == OPBASE_QP_MODE_EXT)) {
        struct QpExtAttrs attrs {};
        QueueDepthAttr qpDepth{};
        CHK_RET(ConstructQpAttrs(qpMode, attrs, qpDepth));
        attrs.udpSport = 0x0;
        attrs.qpAttr.cap.max_send_wr = HETEROG_OFFLINE_EXT_MAX_SEND_WR;
        attrs.cqAttr.sendCqDepth = DEFAULT_MAX_ONE_SIDED_SEND_CQ_DEPTH;
        CHK_RET(hrtRaQpCreateWithAttrs(rdmaHandle, &attrs, qp.qpHandle));
    } else {
        CHK_RET(HrtRaQpCreate(rdmaHandle, flag, qpMode, qp.qpHandle));
    }

    // Hdc模式下HCCP不支持hrtRaGetQpContext接口
    HcclResult ret = SetQpAttrQos(qp.qpHandle, qp.trafficClass, qp.serviceLevel);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[CreateQp] SetQpAttrQos fail, ret[%d], destroy QP", ret);
        HrtRaQpDestroy(qp.qpHandle);
        return ret;
    }
    // 配置RDMA Timeout时间
    ret = SetQpAttrTimeOut(qp.qpHandle);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[CreateQp] SetQpAttrTimeOut fail, ret[%d], destroy QP", ret);
        HrtRaQpDestroy(qp.qpHandle);
        return ret;
    }
    // 配置RDMA Retry Cnt重传次数
    ret = SetQpAttrRetryCnt(qp.qpHandle);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[CreateQp] SetQpAttrRetryCnt fail, ret[%d], destroy QP", ret);
        HrtRaQpDestroy(qp.qpHandle);
        return ret;
    }

    return HCCL_SUCCESS;
}

HcclResult CreateNormalQp(RdmaHandle rdmaHandle, QpInfo& qp)
{
    struct ibv_qp_init_attr ibQpAttr;
    CHK_SAFETY_FUNC_RET(memset_s(&ibQpAttr, sizeof(ibv_qp_init_attr), 0, sizeof(ibv_qp_init_attr)));
    ibQpAttr.qp_context = qp.context;
    ibQpAttr.send_cq = qp.sendCq;
    ibQpAttr.recv_cq = qp.recvCq;
    ibQpAttr.srq = qp.srq;
    ibQpAttr.qp_type = IBV_QPT_RC;
    ibQpAttr.cap.max_inline_data = MAX_INLINE_DATA;
    ibQpAttr.cap.max_send_wr = qp.attr.maxWr;
    ibQpAttr.cap.max_send_sge = qp.attr.maxSendSge;
    ibQpAttr.cap.max_recv_wr = (qp.srq == nullptr ? qp.attr.maxWr : 0);
    ibQpAttr.cap.max_recv_sge = (qp.srq == nullptr ? qp.attr.maxRecvSge : 0);
    CHK_RET(hrtRaNormalQpCreate(rdmaHandle, &ibQpAttr, qp.qpHandle, qp.qp));
    HcclResult ret = SetQpAttrQos(qp.qpHandle, qp.trafficClass, qp.serviceLevel);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[CreateNormalQp] SetQpAttrQos fail, ret[%d], destroy QP", ret);
        HrtRaQpDestroy(qp.qpHandle);
        return ret;
    }
    // 配置RDMA Timeout时间
    ret = SetQpAttrTimeOut(qp.qpHandle);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[CreateNormalQp] SetQpAttrTimeOut fail, ret[%d], destroy QP", ret);
        HrtRaQpDestroy(qp.qpHandle);
        return ret;
    }
    // 配置RDMA Retry Cnt重传次数
    ret = SetQpAttrRetryCnt(qp.qpHandle);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[CreateNormalQp] SetQpAttrRetryCnt fail, ret[%d], destroy QP", ret);
        HrtRaQpDestroy(qp.qpHandle);
        return ret;
    }

    return HCCL_SUCCESS;
}

HcclResult CreateCqAndQp(RdmaHandle& rdmaHandle, string& label, QpConfig& config, QpInfo& info)
{
    unique_lock<mutex> lock(g_qpRecordsMutex);
    bool createCq = false;
    if (g_qpRecords[label].empty()) {
        HCCL_INFO("create cq: label[%s] is empty, need create cq.", label.c_str());
        createCq = true;
    } else if ((g_qpRecords[label].back().depth - g_qpRecords[label].back().used) < config.maxWr) {
        HCCL_INFO(
            "create cq: label[%s] has %u qp, last cq used %u, need create cq.", label.c_str(),
            g_qpRecords[label].size(), g_qpRecords[label].back().used);
        createCq = true;
    } else {
        HCCL_INFO(
            "create cq: label[%s] has %u qp, last cq used %u, not need create cq.", label.c_str(),
            g_qpRecords[label].size(), g_qpRecords[label].back().used);
    }

    if (createCq) {
        CqInfo cq(nullptr, info.srqCq, nullptr, MAX_CQ_DEPTH, config.sqEvent, config.rqEvent, info.srqContext);
        CHK_RET(CreateCq(rdmaHandle, cq));
        QpInfo qp(
            config, rdmaHandle, nullptr, nullptr, cq.context, cq.sq, cq.rq, info.srq, info.srqCq, info.srqContext);
        HcclResult ret = CreateNormalQp(rdmaHandle, qp);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[CreateCqAndQp] CreateNormalQp fail, ret[%d], destroy CQ", ret);
            DestroyCq(rdmaHandle, cq);
            return ret;
        }

        cq.used += qp.attr.maxWr;
        cq.qps.push_back(qp);
        g_qpRecords[label].push_back(cq);
        info = qp;
    } else {
        QpInfo qp(
            config, rdmaHandle, nullptr, nullptr, g_qpRecords[label].back().context, g_qpRecords[label].back().sq,
            g_qpRecords[label].back().rq, info.srq, info.srqCq, info.srqContext);
        CHK_RET(CreateNormalQp(rdmaHandle, qp));

        g_qpRecords[label].back().used += config.maxWr;
        g_qpRecords[label].back().qps.push_back(qp);
        info = qp;
    }
    return HCCL_SUCCESS;
}

HcclResult CreateQpWithSharedCq(
    RdmaHandle rdmaHandle, HcclIpAddress& selfIp, HcclIpAddress& peerIp, s32 sqEvent, s32 rqEvent, QpInfo& info,
    s32 qpAppend, u32 maxSegNum)
{
    QpConfig config(selfIp, peerIp, MAX_WR_NUM, maxSegNum, MAX_RECV_SGE_NUM, sqEvent, rqEvent);

    string label = string(selfIp.GetReadableIP()) + "_" + string(peerIp.GetReadableIP()) + "_"
                   + to_string(config.sqEvent) + "_" + to_string(config.rqEvent) + "_" + to_string(qpAppend);

    HCCL_RUN_INFO(
        "CreateQpWithSharedCq selfIp[%s] peerIp[%s] maxWr[%u] maxSendSge[%u] maxRecvSge[%u]"
        "sqEvent[%d] rqEvent[%d]",
        selfIp.GetReadableIP(), peerIp.GetReadableIP(), config.maxWr, config.maxSendSge, config.maxRecvSge,
        config.sqEvent, config.rqEvent);
    CHK_RET(CreateCqAndQp(rdmaHandle, label, config, info));
    return HCCL_SUCCESS;
}

HcclResult DestroyQpWithSharedCq(const QpInfo& info, s32 qpAppend)
{
    if (info.qpHandle == nullptr) {
        return HCCL_SUCCESS;
    }

    string label = string(info.attr.selfIp.GetReadableIP()) + "_" + string(info.attr.peerIp.GetReadableIP()) + "_"
                   + to_string(info.attr.sqEvent) + "_" + to_string(info.attr.rqEvent) + "_" + to_string(qpAppend);

    unique_lock<mutex> lock(g_qpRecordsMutex);
    if (g_qpRecords[label].empty()) {
        HCCL_ERROR("qp label[%s] does not exist.", label.c_str());
        return HCCL_E_PARA;
    } else {
        for (auto itCq = g_qpRecords[label].begin(); itCq != g_qpRecords[label].end(); itCq++) {
            if ((*itCq).context == info.context) {
                for (auto itQp = (*itCq).qps.begin(); itQp != (*itCq).qps.end(); itQp++) {
                    if ((*itQp).qpHandle == info.qpHandle) {
                        HCCL_INFO("destroy qpHandle");
                        CHK_RET(hrtRaNormalQpDestroy(info.qpHandle));
                        (*itCq).qps.erase(itQp);
                        if ((*itCq).used > info.attr.maxWr) {
                            (*itCq).used -= info.attr.maxWr;
                        } else if ((*itCq).used == info.attr.maxWr) {
                            HCCL_INFO("destroy cq:%p", (*itCq).context);
                            CHK_RET(DestroyCq(info.rdmaHandle, *itCq));
                            g_qpRecords[label].erase(itCq);
                        } else {
                            HCCL_ERROR(
                                "DestroyQp: cq used[%u] should be greater than the qp maxwr[%u]", (*itCq).used,
                                info.attr.maxWr);
                            return HCCL_E_PARA;
                        }
                        return HCCL_SUCCESS;
                    }
                }
                HCCL_ERROR("DestroyQp: the qp does not exist");
                return HCCL_E_PARA;
            }
        }
        HCCL_ERROR("DestroyQp: the cq does not exist");
        return HCCL_E_PARA;
    }
}

HcclResult CreateQpWithCq(
    RdmaHandle rdmaHandle, s32 sqEvent, s32 rqEvent, void* sendChannel, void* recvChannel, QpInfo& info, bool isHdcMode,
    bool isESMode)
{
    struct ibv_comp_channel* sChannel = reinterpret_cast<struct ibv_comp_channel*>(sendChannel);
    struct ibv_comp_channel* rChannel = reinterpret_cast<struct ibv_comp_channel*>(recvChannel);

    QpConfig config(MAX_WR_NUM, MAX_SEND_SGE_NUM, MAX_RECV_SGE_NUM, sqEvent, rqEvent);
    CqInfo cq(
        nullptr, nullptr, nullptr, MAX_CQ_DEPTH, config.sqEvent, config.rqEvent, info.srqContext, sChannel, rChannel);
    if (!isHdcMode) {
        // hdc模式下hccp没有对外提供创建CQ的接口
        CHK_RET(CreateCq(rdmaHandle, cq));
    }
    QpInfo qp(
        config, rdmaHandle, nullptr, nullptr, cq.context, cq.sq, cq.rq, info.srq, info.srqCq, info.srqContext, sChannel,
        rChannel, info.trafficClass, info.serviceLevel);

    if (isHdcMode) {
        CHK_RET(CreateQp(rdmaHandle, info.flag, info.qpMode, qp, isESMode));
        info.qpHandle = qp.qpHandle;
        info.qp = qp.qp;
        info.sendCq = qp.sendCq;
        info.recvCq = qp.recvCq;
    } else {
        HcclResult ret = CreateNormalQp(rdmaHandle, qp);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[CreateQpWithCq] CreateNormalQp fail, ret[%d], destroy CQ", ret);
            DestroyCq(rdmaHandle, cq);
            return ret;
        }
        info = qp;
    }
    return HCCL_SUCCESS;
}

HcclResult DestroyQpWithCq(const QpInfo& info, bool isHdcMode)
{
    if (info.qpHandle == nullptr) {
        return HCCL_SUCCESS;
    }

    if (isHdcMode) {
        CHK_RET(HrtRaQpDestroy(info.qpHandle));
    } else {
        CHK_RET(hrtRaNormalQpDestroy(info.qpHandle));
    }

    CqInfo cq;
    cq.context = info.context;
    cq.rq = info.recvCq;
    cq.sq = info.sendCq;
    if (!isHdcMode) {
        CHK_RET(DestroyCq(info.rdmaHandle, cq));
    }

    return HCCL_SUCCESS;
}

HcclResult CreateAiQp(RdmaHandle rdmaHandle, struct AiQpInfo& aiQpInfo, QpInfo& info, u32 devicePhyId)
{
    struct QpExtAttrs attrs {};
    QueueDepthAttr qpDepth{};
    CHK_RET(ConstructQpAttrs(info.qpMode, attrs, qpDepth, false));
    attrs.qpAttr.cap.max_send_wr = HETEROG_OFFLINE_EXT_MAX_SEND_WR;
    attrs.cqAttr.sendCqDepth = DEFAULT_MAX_ONE_SIDED_SEND_CQ_DEPTH;
    attrs.udpSport = 0;

    CHK_RET(hrtRaAiQpCreate(devicePhyId, rdmaHandle, &attrs, &aiQpInfo, info.qpHandle));

    HcclResult ret = SetQpAttrQos(info.qpHandle, info.trafficClass, info.serviceLevel);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[CreateAiQp] SetQpAttrQos fail, ret[%d], destroy qpHandle", ret);
        HrtRaQpDestroy(info.qpHandle);
        return ret;
    }
    ret = SetQpAttrTimeOut(info.qpHandle);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[CreateAiQp] SetQpAttrTimeOut fail, ret[%d], destroy qpHandle", ret);
        HrtRaQpDestroy(info.qpHandle);
        return ret;
    }
    ret = SetQpAttrRetryCnt(info.qpHandle);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[CreateAiQp] SetQpAttrRetryCnt fail, ret[%d], destroy qpHandle", ret);
        HrtRaQpDestroy(info.qpHandle);
        return ret;
    }

    info.qp = reinterpret_cast<struct ibv_qp*>(aiQpInfo.aiQpAddr);
    if (info.qp == nullptr) {
        HCCL_ERROR("info.qp is nullptr.");
        HrtRaQpDestroy(info.qpHandle);
        return HCCL_E_PARA;
    }

    info.sendCq = reinterpret_cast<struct ibv_cq*>(aiQpInfo.aiScqAddr);
    info.recvCq = reinterpret_cast<struct ibv_cq*>(aiQpInfo.aiRcqAddr);

    return HCCL_SUCCESS;
}

HcclResult DestroyAiQp(const QpInfo& info)
{
    if (info.qpHandle == nullptr) {
        return HCCL_SUCCESS;
    }

    CHK_RET(HrtRaQpDestroy(info.qpHandle));

    return HCCL_SUCCESS;
}

HcclResult hrtRaSetQpAttrQos(QpHandle qpHandle, struct QosAttr& attr)
{
    s32 ret = DlRaFunction::GetInstance().dlRaSetQpAttrQos(qpHandle, &attr);
    CHK_PRT_RET(
        ret != 0, HCCL_ERROR("[Set][SqAttr]set qp attr qos failed tc[%u] sl[%u] ret[%d]", attr.tc, attr.sl, ret),
        HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult hrtRaSetQpAttrTimeOut(QpHandle qpHandle, u32& timeOut)
{
    s32 ret = DlRaFunction::GetInstance().dlRaSetQpAttrTimeOut(qpHandle, &timeOut);
    CHK_PRT_RET(
        ret != 0, HCCL_ERROR("[Set][SqAttr]set qp attr timeout[%u s] failed ret[%d]", timeOut, ret), HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult hrtRaSetQpAttrRetryCnt(QpHandle qpHandle, u32& retryCnt)
{
    s32 ret = DlRaFunction::GetInstance().dlRaSetQpAttrRetryCnt(qpHandle, &retryCnt);
    CHK_PRT_RET(
        ret != 0, HCCL_ERROR("[Set][SqAttr]set qp attr retrycnt[%u] failed ret[%d]", retryCnt, ret), HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult SetQpAttrQos(QpHandle qpHandle, u32 tc, u32 sl)
{
    struct QosAttr qosAttr = {};
    if (tc == HCCL_COMM_TRAFFIC_CLASS_CONFIG_NOT_SET && sl == HCCL_COMM_SERVICE_LEVEL_CONFIG_NOT_SET) {
        qosAttr.tc = GetExternalInputRdmaTrafficClass();
        qosAttr.sl = GetExternalInputRdmaServerLevel();
        HCCL_INFO(
            "[%s]set qp qos success by environment variable or default value, TC[%u] SL[%u]", __func__, qosAttr.tc,
            qosAttr.sl);
    } else {
        qosAttr.tc = tc;
        qosAttr.sl = sl;
        HCCL_INFO("[%s]set qp qos success by config, TC[%u] SL[%u]", __func__, qosAttr.tc, qosAttr.sl);
    }

    CHK_RET(hrtRaSetQpAttrQos(qpHandle, qosAttr));
    HCCL_INFO("[%s]rdmaTrafficClass[%u], rdmaServerLevel[%u].", __func__, qosAttr.tc, qosAttr.sl);

    return HCCL_SUCCESS;
}

HcclResult SetQpAttrTimeOut(QpHandle qpHandle, u32 timeout)
{
    u32 rdmaTimeOut = (timeout == INVALID_UINT) ? GetExternalInputRdmaTimeOut() : timeout;
    CHK_RET(hrtRaSetQpAttrTimeOut(qpHandle, rdmaTimeOut));
    HCCL_INFO("[SetQpAttrTimeOut]rdmaTimeOut[%u].", rdmaTimeOut);

    return HCCL_SUCCESS;
}

HcclResult SetQpAttrRetryCnt(QpHandle qpHandle, u32 retryCnt)
{
    u32 rdmaRetryCnt = (retryCnt == INVALID_UINT) ? GetExternalInputRdmaRetryCnt() : retryCnt;
    CHK_RET(hrtRaSetQpAttrRetryCnt(qpHandle, rdmaRetryCnt));
    HCCL_INFO("[SetQpAttrRetryCnt]rdmaRetryCnt[%u].", rdmaRetryCnt);

    return HCCL_SUCCESS;
}

HcclResult hrtRaCreateCompChannel(RdmaHandle rdmaHandle, void** compChannel)
{
    s32 ret = DlRaFunction::GetInstance().dlRaCreateCompChannel(rdmaHandle, compChannel);
    CHK_PRT_RET(
        ret != 0,
        HCCL_ERROR(
            "[Create][CompChannel]errNo[0x%016llx] ra create comp channel fail. "
            "return[%d], params: rdmaHandle[%p], compChannel[%p]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), ret, rdmaHandle, compChannel),
        HCCL_E_NETWORK);

    return HCCL_SUCCESS;
}

HcclResult hrtRaDestroyCompChannel(RdmaHandle rdmaHandle, void* compChannel)
{
    s32 ret = DlRaFunction::GetInstance().dlRaDestroyCompChannel(rdmaHandle, compChannel);
    CHK_PRT_RET(
        ret != 0,
        HCCL_ERROR(
            "[Destroy][CompChannel]errNo[0x%016llx] ra destroy comp channel fail. "
            "return[%d], params: rdmaHandle[%p], compChannel[%p]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), ret, rdmaHandle, compChannel),
        HCCL_E_NETWORK);

    return HCCL_SUCCESS;
}

HcclResult hrtRaGetCqeErrInfo(unsigned int phyId, struct CqeErrInfo* info)
{
    s32 ret = DlRaFunction::GetInstance().dlRaGetCqeErrInfo(phyId, info);
    CHK_PRT_RET(ret != 0, HCCL_ERROR("[hrtRaGetCqeErrInfo]Get Cqe err info failed"), HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}
HcclResult hrtRaGetCqeErrInfoList(RdmaHandle rdmaHandle, struct CqeErrInfo* infolist, u32* num)
{
    CHK_PTR_NULL(rdmaHandle);
    CHK_PTR_NULL(DlRaFunction::GetInstance().dlRaGetCqeErrInfoList);
    s32 ret = DlRaFunction::GetInstance().dlRaGetCqeErrInfoList(rdmaHandle, infolist, num);
    CHK_PRT_RET(ret != 0, HCCL_ERROR("[dlRaGetCqeErrInfoList]Get Cqe err info list failed"), HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult IsSuppCqeErrInfoListConfig(bool& supCqeErrInfoListConfig)
{
    u32 phyId = 0; // phyId无实际意义，这里直接传入0
    u32 configVersion = 0;
    supCqeErrInfoListConfig = false;

    // 获取版本号查看是否兼容
    HcclResult ret = hrtRaGetInterfaceVersion(phyId, CQE_ERR_INFO_LIST_INTERFACE, &configVersion);
    CHK_PRT_RET(
        ret == HCCL_E_NETWORK,
        HCCL_ERROR(
            "[IsSuppportCqeErrInfoListConfig]hrtRaGetInterfaceVersion "
            "failed, interface[%u]",
            CQE_ERR_INFO_INTERFACE),
        ret);
    if (ret == HCCL_E_NOT_SUPPORT) {
        HCCL_WARNING("this package does not support hrtRaGetInterfaceVersion, please change new package");
        return HCCL_SUCCESS;
    }

    if (configVersion >= CQE_ERR_INFO_SUP_VER) {
        supCqeErrInfoListConfig = true;
    }
    HCCL_INFO("IsSuppportCqeErrInfoListConfig support:%d", supCqeErrInfoListConfig);
    return HCCL_SUCCESS;
}

HcclResult IsSupportRaSendNormalWrlist(bool& isSupportRaSendNormalWrlist)
{
    s32 deviceLogicID = -1;
    u32 devicePhyId = 0;
    CHK_RET(hrtGetDevice(&deviceLogicID));
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(deviceLogicID), devicePhyId));
    u32 configVersion = 0;
    isSupportRaSendNormalWrlist = false;

    // 获取版本号查看是否兼容
    HcclResult ret = hrtRaGetInterfaceVersion(devicePhyId, SEND_NORMAL_WRLIST, &configVersion);
    CHK_PRT_RET(
        ret == HCCL_E_NETWORK,
        HCCL_ERROR(
            "[IsSupportRaSendNormalWrlist]hrtRaGetInterfaceVersion "
            "failed, interface[%u]",
            CQE_ERR_INFO_INTERFACE),
        ret);
    if (ret == HCCL_E_NOT_SUPPORT) {
        HCCL_WARNING("this package does not support hrtRaGetInterfaceVersion, please change new package");
        return HCCL_SUCCESS;
    }

    if (configVersion >= SEND_NORMAL_WRLIST_VERSION) {
        isSupportRaSendNormalWrlist = true;
    }
    HCCL_INFO("IsSupportRaSendNormalWrlist support:%d", isSupportRaSendNormalWrlist);
    return HCCL_SUCCESS;
}

HcclResult hrtRaGetQpAttr(QpHandle qpHandle, struct QpAttr* attr)
{
    s32 ret = DlRaFunction::GetInstance().dlRaGetQpAttr(qpHandle, attr);
    CHK_PRT_RET(ret != 0, HCCL_ERROR("Get qpn info failed"), HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult hrtRaCreateSrq(RdmaHandle rdmaHandle, SrqInfo& srqInfo)
{
    struct SrqAttr attr = {};
    attr.ibSrq = &srqInfo.srq;
    attr.ibRecvCq = &srqInfo.srqCq;
    attr.maxSge = MAX_RECV_SGE_NUM;
    attr.context = &srqInfo.context;
    attr.srqEventId = srqInfo.srqEvent;
    attr.srqDepth = srqInfo.srqDepth;
    attr.cqDepth = MAX_CQ_DEPTH;
    s32 ret = DlRaFunction::GetInstance().dlRaCreateSrq(rdmaHandle, &attr);
    CHK_PRT_RET(
        ret != 0,
        HCCL_ERROR(
            "[Create][Srq]errNo[0x%016llx] ra create srq fail. "
            "return[%d], params: rdmaHandle[%p]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), ret, rdmaHandle),
        HCCL_E_NETWORK);

    return HCCL_SUCCESS;
}

HcclResult hrtRaDestroySrq(RdmaHandle rdmaHandle, SrqInfo& srqInfo)
{
    struct SrqAttr attr = {};
    attr.context = &srqInfo.context;
    attr.ibSrq = &srqInfo.srq;
    s32 ret = DlRaFunction::GetInstance().dlRaDestroyeSrq(rdmaHandle, &attr);
    CHK_PRT_RET(
        ret != 0,
        HCCL_ERROR(
            "[Destroy][Srq]errNo[0x%016llx] ra destroy srq fail. "
            "return[%d], params: rdmaHandle[%p]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), ret, rdmaHandle),
        HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult hrtRaCreateEventHandle(s32& eventHandle)
{
    if (DlRaFunction::GetInstance().dlRaCreateEventHandle == nullptr) {
        HCCL_ERROR("driver package does not support hrtRaCreateEventHandle, please change new package");
        return HCCL_E_NOT_SUPPORT;
    }
    s32 ret = DlRaFunction::GetInstance().dlRaCreateEventHandle(&eventHandle);
    CHK_PRT_RET(ret != 0, HCCL_ERROR("Create event handle failed, ret is [%d]", ret), HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult hrtRaCtlEventHandle(s32 eventHandle, const FdHandle fdHandle, int opCode, HcclEpollEvent event)
{
    if (DlRaFunction::GetInstance().dlRaCtlEventHandle == nullptr) {
        HCCL_ERROR("driver package does not support hrtRaCtlEventHandle, please change new package");
        return HCCL_E_NOT_SUPPORT;
    }
    RaEpollEvent epollEvent = static_cast<RaEpollEvent>(event);
    CHK_PRT_RET(
        (epollEvent < RA_EPOLLIN) && (epollEvent >= RA_EPOLLINVALD),
        HCCL_ERROR("epoll event[%d] is invalid", epollEvent), HCCL_E_NETWORK);
    s32 ret = DlRaFunction::GetInstance().dlRaCtlEventHandle(eventHandle, fdHandle, opCode, epollEvent);
    CHK_PRT_RET(ret != 0, HCCL_ERROR("Control event handle failed, ret is [%d]", ret), HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult hrtRaWaitEventHandle(
    s32 eventHandle, std::vector<SocketEventInfo>& eventInfos, s32 timeOut, u32 maxEvents, u32& eventsNum)
{
    if (DlRaFunction::GetInstance().dlRaWaitEventHandle == nullptr) {
        HCCL_ERROR("driver package does not support hrtRaWaitEventHandle, please change new package");
        return HCCL_E_NOT_SUPPORT;
    }
    std::vector<struct SocketEventInfoT> raEventInfos(maxEvents);
    s32 ret = DlRaFunction::GetInstance().dlRaWaitEventHandle(
        eventHandle, raEventInfos.data(), timeOut, maxEvents, &eventsNum);
    CHK_PRT_RET(ret != 0, HCCL_ERROR("Wait event handle failed, ret is [%d]", ret), HCCL_E_NETWORK);
    for (u32 i = 0; i < eventsNum; i++) {
        eventInfos[i].fdHandle = raEventInfos[i].fdHandle;
    }
    return HCCL_SUCCESS;
}

HcclResult hrtRaDestroyEventHandle(s32& eventHandle)
{
    if (DlRaFunction::GetInstance().dlRaDestroyEventHandle == nullptr) {
        HCCL_ERROR("driver package does not support hrtRaDestroyEventHandle, please change new package");
        return HCCL_E_NOT_SUPPORT;
    }
    s32 ret = DlRaFunction::GetInstance().dlRaDestroyEventHandle(&eventHandle);
    CHK_PRT_RET(ret != 0, HCCL_ERROR("Destroy event handle failed, ret is [%d]", ret), HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult hrtRaQpCreateWithAttrs(RdmaHandle rdmaHandle, struct QpExtAttrs* attrs, QpHandle& qpHandle)
{
    string qpInfo = string("rdmaHandle:[") + to_string(reinterpret_cast<intptr_t>(rdmaHandle)) + string("],qpHandle:[")
                    + to_string(reinterpret_cast<intptr_t>(&qpHandle)) + string("]; qp attr:[qpMode:")
                    + to_string(attrs->qpMode) + string(",udpSport:") + to_string(attrs->udpSport) + string(",version:")
                    + to_string(attrs->version) + string(",memAlign:") + to_string(attrs->memAlign)
                    + string("]; cq attr: [sendCqDepth:") + to_string(attrs->cqAttr.sendCqDepth)
                    + string(",recvCqDepth:") + to_string(attrs->cqAttr.recvCqDepth) + string(",sendCqCompVector:")
                    + to_string(attrs->cqAttr.sendCqCompVector) + string(",recvCqCompVector:")
                    + to_string(attrs->cqAttr.recvCqCompVector) + string(",cap.max_send_wr:")
                    + to_string(attrs->qpAttr.cap.max_send_wr) + string(",cap.max_recv_wr:")
                    + to_string(attrs->qpAttr.cap.max_recv_wr) + "]";

    s32 ret = DlRaFunction::GetInstance().dlRaQpCreateWithAttrs(rdmaHandle, attrs, &qpHandle);
    if (ret == ROCE_ENOMEM_RET && GetExternalInputRdmaFastPost()) {
        HCCL_ERROR(
            "[%s]create qp failed because of memory error, you can try to unset HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT "
            "and execute again",
            __func__);
    }

    CHK_OOM_RET(ret, qpInfo.c_str());

    CHK_PRT_RET(
        ret != 0 || (qpHandle == nullptr),
        HCCL_ERROR(
            "[Create][RaQp]errNo[0x%016llx] ra qp create with attrs fail. qpInfo:[%s], return: ret[%d]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), qpInfo.c_str(), ret),
        HCCL_E_NETWORK);

    struct QpAttr attr {};
    CHK_RET(hrtRaGetQpAttr(qpHandle, &attr));
    s32 deviceId = 0;
    if (hrtGetDevice(&deviceId) != HCCL_SUCCESS) {
        deviceId = -1;
    }
    PLF_CONFIG_DEBUG(PLF_RES, "Create Qp para: deviceId[%d] qpn[%u] qpInfo[%s]", deviceId, attr.qpn, qpInfo.c_str());
    return HCCL_SUCCESS;
}

HcclResult hrtRaQpCreateWithCQWithAttrs(
    RdmaHandle rdmaHandle, struct QpExtAttrs* attrs, unsigned int sendCqn, unsigned int recvCqn, QpHandle& qpHandle)
{
    s32 ret = DlRaFunction::GetInstance().dlRaQpCreateWithCQWithAttrs(rdmaHandle, attrs, sendCqn, recvCqn, &qpHandle);
    if (ret != 0 || qpHandle == nullptr) {
        HCCL_ERROR("[Create][RaQpWithCQ] ra qp create with cq with attrs fail. ret[%d]", ret);
        return HCCL_E_NETWORK;
    }

    struct QpAttr attr {};
    CHK_RET(hrtRaGetQpAttr(qpHandle, &attr));
    s32 deviceId = 0;
    if (hrtGetDevice(&deviceId) != HCCL_SUCCESS) {
        deviceId = -1;
    }
    PLF_CONFIG_DEBUG(PLF_RES, "Create QpWithCQ para: deviceId[%d] qpn[%u]", deviceId, attr.qpn);
    return HCCL_SUCCESS;
}

HcclResult
hrtRaAiQpCreate(u32 phyId, RdmaHandle rdmaHandle, struct QpExtAttrs* attrs, struct AiQpInfo* info, QpHandle& qpHandle)
{
    u32 aiQpCreateVersion = 0;
    HcclResult vRet = hrtRaGetInterfaceVersion(phyId, AI_QP_CREATE, &aiQpCreateVersion);
    if (vRet != HCCL_SUCCESS || aiQpCreateVersion < AI_QP_CREATE_VERSION) {
        HCCL_ERROR("this package does not support hrtRaAiQpCreate for device, please change new package");
        return HCCL_E_NOT_SUPPORT;
    }
    s32 ret = DlRaFunction::GetInstance().dlRaAiQpCreate(rdmaHandle, attrs, info, &qpHandle);

    string qpInfo = string("qp attr:[qpMode:") + to_string(attrs->qpMode) + string(",udpSport:")
                    + to_string(attrs->udpSport) + string(",version:") + to_string(attrs->version)
                    + string(",memAlign:") + to_string(attrs->memAlign) + string("]; cq attr: [sendCqDepth:")
                    + to_string(attrs->cqAttr.sendCqDepth) + string(",recvCqDepth:")
                    + to_string(attrs->cqAttr.recvCqDepth) + string(",sendCqCompVector:")
                    + to_string(attrs->cqAttr.sendCqCompVector) + string(",recvCqCompVector:")
                    + to_string(attrs->cqAttr.recvCqCompVector) + string(",cap.max_send_wr:")
                    + to_string(attrs->qpAttr.cap.max_send_wr) + string(",cap.max_recv_wr:")
                    + to_string(attrs->qpAttr.cap.max_recv_wr) + "]";

    CHK_OOM_RET(ret, qpInfo.c_str());

    CHK_PRT_RET(
        ret != 0 || (qpHandle == nullptr),
        HCCL_ERROR(
            "[Create][RaAiQp]errNo[0x%016llx] ra ai qp create fail. "
            "return: ret[%d]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), ret),
        HCCL_E_NETWORK);

    struct QpAttr attr {};
    CHK_RET(hrtRaGetQpAttr(qpHandle, &attr));
    s32 deviceId = 0;
    if (hrtGetDevice(&deviceId) != HCCL_SUCCESS) {
        deviceId = -1;
    }
    PLF_CONFIG_DEBUG(
        PLF_RES, "Create Qp para: deviceId[%d] qpn[%u] sq_depth[%u] rq_depth[%u] scq_depth[%u] rcq_depth[%u]", deviceId,
        attr.qpn, attrs->qpAttr.cap.max_send_wr, attrs->qpAttr.cap.max_recv_wr, attrs->cqAttr.sendCqDepth,
        attrs->cqAttr.recvCqDepth);
    return HCCL_SUCCESS;
}

HcclResult hrtRaRecvWrlist(QpHandle handle, struct RecvWrlistData* wr, unsigned int recvNum, unsigned int* completeNum)
{
    if (DlRaFunction::GetInstance().dlRaRecvWrlist == nullptr) {
        HCCL_ERROR("[Recv][RaWrlist]driver package does not support hrtRaRecvWrlist interface, "
                   "please change new one");
        return HCCL_E_NOT_SUPPORT;
    }
    s32 ret = 0;
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(GetExternalInputHcclLinkTimeOut());
    u32 remainNum = 0;
    unsigned int completeNumLocal = 0;
    *completeNum = 0;
    while (true) {
        if (remainNum == recvNum) {
            break;
        }

        ret = DlRaFunction::GetInstance().dlRaRecvWrlist(handle, wr + remainNum, recvNum, &completeNumLocal);
        *completeNum += completeNumLocal;

        if (ret == 0) {
            break; // 成功跳出
        } else if (
            (ret == SOCK_ENOENT) || (ret == SOCK_EAGAIN)
            || (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && ret == ROCE_ENOMEM)) {
            remainNum += completeNumLocal;
            bool bTimeout = ((std::chrono::steady_clock::now() - startTime) >= timeout);
            CHK_PRT_RET(
                bTimeout,
                HCCL_ERROR(
                    "[Recv][RaWrList]errNo[0x%016llx] ra Recv wrlist async timeout[%d s]. "
                    "return[%d], params: send_wrAddr[%p]",
                    HCCL_ERROR_CODE(HCCL_E_ROCE_TRANSFER), timeout, ret, wr),
                HCCL_E_ROCE_TRANSFER);
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
        } else {
            HCCL_ERROR("[Recv][RaWr]ra wr list Recv async fail. return[%d], para: Recv_wrAddr[%p]", ret, wr);
            return HCCL_E_ROCE_TRANSFER; // 非-2/-11场景错误，不轮询，直接退出
        }
    }
    return HCCL_SUCCESS;
}

std::mutex g_deviceVnicIpMutex;
map<u32, HcclIpAddress>
    g_deviceIdVnicInfoMap; // 记录deviceid和vnic ip的关系，用于非超节点模式server内查询，避免重复查询
map<u32, HcclIpAddress> g_sdidVnicInfoMap; // 记录sdid和vnic ip的关系，用于超节点模式，避免重复查询
HcclResult IsSuppportRaGetSocketVnicIps(bool& supportGetSocketVnicIp)
{
    u32 phyId = 0; // phyId无实际意义，这里直接传入0
    u32 supportGetSocketVnicIpVersion = 0;
    supportGetSocketVnicIp = false;
    // 获取版本号查看是否兼容
    HcclResult ret = hrtRaGetInterfaceVersion(phyId, SOCKET_VNIC_IP_INFOS_INTERFACE, &supportGetSocketVnicIpVersion);
    CHK_PRT_RET(
        ret == HCCL_E_NETWORK,
        HCCL_ERROR(
            "[IsSuppportRaGetSocketVnicIps]hrtRaGetInterfaceVersion "
            "failed, interface[%u]",
            SOCKET_VNIC_IP_INFOS_INTERFACE),
        ret);
    if (ret == HCCL_E_NOT_SUPPORT) {
        HCCL_WARNING("this package does not support hrtRaGetInterfaceVersion, please change new package");
        return HCCL_SUCCESS;
    }

    if (supportGetSocketVnicIpVersion >= SOCKET_VNIC_IP_INFOS_SUP_VER) {
        supportGetSocketVnicIp = true;
    }

    return HCCL_SUCCESS;
}

HcclResult
hrtRaGetSocketVnicIpInfos(u32 phyId, enum IdType type, const vector<u32>& deviceIds, vector<HcclIpAddress>& vnicIps)
{
    u32 vnicIpNum = deviceIds.size();
    CHK_PRT_RET(
        vnicIpNum == 0, HCCL_ERROR("[hrtRaGetSocketVnicIpInfos]ra get VnicIp para error, num[%u]", vnicIpNum),
        HCCL_E_PARA);
    unique_lock<mutex> lock(g_deviceVnicIpMutex);
    std::map<u32, HcclIpAddress>& vnicInfoMap = (type == PHY_ID_VNIC_IP) ? g_deviceIdVnicInfoMap : g_sdidVnicInfoMap;
    for (u32 i = 0; i < vnicIpNum; i++) {
        HcclIpAddress vnicIP;
        auto iter = vnicInfoMap.find(deviceIds[i]);
        // 缓存查找到，直接从缓存获取
        if (iter != vnicInfoMap.end()) {
            vnicIP = iter->second;
            HCCL_INFO(
                "[hrtRaGetSocketVnicIpInfos] vnicInfoMap deviceIds[%u] found, Ip[%s]", deviceIds[i],
                vnicIP.GetReadableAddress());
        } else {
            struct IpInfo vnicIpInfo = {};
            s32 sRet = memset_s(&vnicIpInfo, sizeof(IpInfo), 0, sizeof(IpInfo));
            CHK_PRT_RET(
                sRet != EOK,
                HCCL_ERROR(
                    "[hrtRaGetSocketVnicIpInfos]errNo[0x%016llx] memset vnicIpInfo to 0 failed."
                    "params: dest[%p], dest_size[%zu], count[%zu]",
                    HCCL_ERROR_CODE(HCCL_E_SYSCALL), &vnicIpInfo, sizeof(IpInfo), sizeof(IpInfo)),
                HCCL_E_SYSCALL);
            u32 devId = deviceIds[i];
            s32 ret = DlRaFunction::GetInstance().dlRaGetSocketVnicIpInfos(phyId, type, &devId, 1, &vnicIpInfo);
            CHK_PRT_RET(
                ret != 0,
                HCCL_ERROR(
                    "[hrtRaGetSocketVnicIpInfo]errNo[0x%016llx] ra get VnicIp fail. ret[%d]",
                    HCCL_ERROR_CODE(HCCL_E_TCP_CONNECT), ret),
                HCCL_E_TCP_CONNECT);

            HcclInAddr temp;
            temp.addr = vnicIpInfo.ip.addr;
            temp.addr6 = vnicIpInfo.ip.addr6;
            HcclIpAddress ipInfo(vnicIpInfo.family, temp);
            CHK_PRT_RET(ipInfo.IsInvalid(), HCCL_ERROR("ip is invalid."), HCCL_E_PARA);
            vnicInfoMap.insert({deviceIds[i], ipInfo});
            vnicIP = ipInfo;
            HCCL_INFO(
                "[hrtRaGetSocketVnicIpInfos] add vnicInfoMap, deviceIds[%u], Ip[%s]", deviceIds[i],
                vnicIP.GetReadableAddress());
        }
        vnicIps.push_back(vnicIP);
    }
    return HCCL_SUCCESS;
}

HcclResult H2DTlvInit(struct TlvInitInfo* init_info, uint32_t* buffer_size, void** tlv_handle)
{
    u32 tlvVersion = 0;
    u32 phyId = 0; // phyId无实际意义，这里直接传入0
    HcclResult vRet = hrtRaGetInterfaceVersion(phyId, TLV_INIT, &tlvVersion);
    if (vRet != HCCL_SUCCESS || tlvVersion < TLV_VERSION) {
        HCCL_WARNING("this package does not support H2DTlvInit for device, please change new package");
        return HCCL_E_NOT_SUPPORT;
    }

    s32 ret = DlRaFunction::GetInstance().dlH2DTlvInit(init_info, buffer_size, tlv_handle);
    CHK_PRT_RET(
        ret != 0,
        HCCL_WARNING(
            "[H2DTlvInit]errNo[0x%016llx] dlH2DTlvInit fail. "
            "return: ret[%d]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), ret),
        HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult H2DTlvRequest(void* tlv_handle, unsigned int module_type, struct TlvMsg* send_msg, struct TlvMsg* recv_msg)
{
    u32 tlvVersion = 0;
    u32 phyId = 0; // phyId无实际意义，这里直接传入0
    HcclResult vRet = hrtRaGetInterfaceVersion(phyId, TLV_REQUEST, &tlvVersion);
    if (vRet != HCCL_SUCCESS || tlvVersion < TLV_VERSION) {
        HCCL_WARNING("this package does not support H2DTlvRequest for device, please change new package");
        return HCCL_E_NOT_SUPPORT;
    }

    if (DlRaFunction::GetInstance().dlH2DTlvRequest == nullptr) {
        HCCL_WARNING("driver package does not support H2DTlvRequest, please change new package");
        return HCCL_E_NOT_SUPPORT;
    }

    s32 ret = DlRaFunction::GetInstance().dlH2DTlvRequest(tlv_handle, module_type, send_msg, recv_msg);
    CHK_PRT_RET(
        ret != 0,
        HCCL_WARNING(
            "[H2DTlvRequest]errNo[0x%016llx] dlH2DTlvRequest fail. module_type[%u]"
            "return: ret[%d]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), module_type, ret),
        HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult H2DTlvDeinit(void* tlv_handle)
{
    u32 tlvVersion = 0;
    u32 phyId = 0; // phyId无实际意义，这里直接传入0
    HcclResult vRet = hrtRaGetInterfaceVersion(phyId, TLV_DEINIT, &tlvVersion);
    if (vRet != HCCL_SUCCESS || tlvVersion < TLV_VERSION) {
        HCCL_WARNING("this package does not support H2DTlvDeinit for device, please change new package");
        return HCCL_E_NOT_SUPPORT;
    }

    s32 ret = DlRaFunction::GetInstance().dlH2DTlvDeinit(tlv_handle);
    CHK_PRT_RET(
        ret != 0,
        HCCL_WARNING(
            "[H2DTlvDeinit]errNo[0x%016llx] ra tlv deinit fail. "
            "return: ret[%d]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), ret),
        HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult hrtRaGetSingleSocketVnicIpInfo(u32 phyId, DeviceIdType deviceIdType, u32 deviceId, HcclIpAddress& vnicIP)
{
    bool supportGetSocketVnicIp = false;
    IsSuppportRaGetSocketVnicIps(supportGetSocketVnicIp);
    if (!supportGetSocketVnicIp) {
        // 非超节点场景，如果不支持查询vnicip，返回成功,继续使用phyid作为vnicip; 超节点如不支持，返错退出
        return (deviceIdType == DeviceIdType::DEVICE_ID_TYPE_PHY_ID) ? (HCCL_SUCCESS) : (HCCL_E_NOT_SUPPORT);
    }
    std::vector<u32> deviceIds;
    vector<HcclIpAddress> vnicIPs;
    IdType idType = static_cast<IdType>(deviceIdType);
    deviceIds.push_back(deviceId);
    CHK_RET(hrtRaGetSocketVnicIpInfos(phyId, idType, deviceIds, vnicIPs));
    vnicIP = vnicIPs[0];
    HCCL_INFO(
        "Get available Vnic info success, phyId[%u], deviceIdType[%d], deviceId[0x%x], Vnic ip[%s]", phyId, idType,
        deviceId, vnicIP.GetReadableAddress());
    return HCCL_SUCCESS;
}

HcclResult hrtRaPingInit(struct PingInitAttr* initAttr, struct PingInitInfo* initInfo, void** pingHandle)
{
    if (DlRaFunction::GetInstance().dlRaPingInit == nullptr) {
        HCCL_ERROR("driver package does not support hrtRaPingInit, please change new package");
        return HCCL_E_NOT_SUPPORT;
    }
    s32 ret = DlRaFunction::GetInstance().dlRaPingInit(initAttr, initInfo, pingHandle);
    CHK_PRT_RET(ret != 0, HCCL_ERROR("Rping init failed, ret is [%d]", ret), HCCL_E_NOT_SUPPORT);
    return HCCL_SUCCESS;
}

HcclResult hrtRaPingDeinit(void* pingHandle)
{
    if (DlRaFunction::GetInstance().dlRaPingDeinit == nullptr) {
        HCCL_ERROR("driver package does not support hrtRaPingDeinit, please change new package");
        return HCCL_E_NOT_SUPPORT;
    }
    s32 ret = DlRaFunction::GetInstance().dlRaPingDeinit(pingHandle);
    CHK_PRT_RET(ret != 0, HCCL_ERROR("Rping deinit failed, ret is [%d]", ret), HCCL_E_NOT_SUPPORT);
    return HCCL_SUCCESS;
}

HcclResult hrtRaPingTargetAdd(void* pingHandle, struct PingTargetInfo target[], uint32_t num)
{
    if (DlRaFunction::GetInstance().dlRaPingTargetAdd == nullptr) {
        HCCL_ERROR("driver package does not support hrtRaPingTargetAdd, please change new package");
        return HCCL_E_NOT_SUPPORT;
    }
    s32 ret = DlRaFunction::GetInstance().dlRaPingTargetAdd(pingHandle, target, num);
    CHK_PRT_RET(ret != 0, HCCL_ERROR("Rping add target failed, ret is [%d], num[%u]", ret, num), HCCL_E_NOT_SUPPORT);
    return HCCL_SUCCESS;
}

HcclResult hrtRaPingTargetDel(void* pingHandle, struct PingTargetCommInfo target[], uint32_t num)
{
    if (DlRaFunction::GetInstance().dlRaPingTargetDel == nullptr) {
        HCCL_ERROR("driver package does not support hrtRaPingTargetDel, please change new package");
        return HCCL_E_NOT_SUPPORT;
    }
    s32 ret = DlRaFunction::GetInstance().dlRaPingTargetDel(pingHandle, target, num);
    CHK_PRT_RET(ret != 0, HCCL_ERROR("Rping delete target failed, ret is [%d], num[%u]", ret, num), HCCL_E_NOT_SUPPORT);
    return HCCL_SUCCESS;
}

HcclResult hrtRaPingTaskStart(void* pingHandle, struct PingTaskAttr* attr)
{
    if (DlRaFunction::GetInstance().dlRaPingTaskStart == nullptr) {
        HCCL_ERROR("driver package does not support hrtRaPingTaskStart, please change new package");
        return HCCL_E_NOT_SUPPORT;
    }
    s32 ret = DlRaFunction::GetInstance().dlRaPingTaskStart(pingHandle, attr);
    CHK_PRT_RET(ret != 0, HCCL_ERROR("Rping start task failed, ret is [%d]", ret), HCCL_E_NOT_SUPPORT);
    return HCCL_SUCCESS;
}

HcclResult hrtRaPingTaskStop(void* pingHandle)
{
    if (DlRaFunction::GetInstance().dlRaPingTaskStop == nullptr) {
        HCCL_ERROR("driver package does not support hrtRaPingTaskStop, please change new package");
        return HCCL_E_NOT_SUPPORT;
    }
    s32 ret = DlRaFunction::GetInstance().dlRaPingTaskStop(pingHandle);
    CHK_PRT_RET(ret != 0, HCCL_ERROR("Rping stop task failed, ret is [%d]", ret), HCCL_E_NOT_SUPPORT);
    return HCCL_SUCCESS;
}

HcclResult hrtRaPingGetResults(void* pingHandle, struct PingTargetResult target[], uint32_t* num)
{
    if (DlRaFunction::GetInstance().dlRaPingGetResults == nullptr) {
        HCCL_ERROR("driver package does not support hrtRaPingGetResults, please change new package");
        return HCCL_E_NOT_SUPPORT;
    }
    s32 ret = DlRaFunction::GetInstance().dlRaPingGetResults(pingHandle, target, num);
    CHK_PRT_RET(ret == ROCE_EAGAIN, HCCL_WARNING("Rping get results busy, try again", ret), HCCL_E_AGAIN);
    CHK_PRT_RET(ret != 0, HCCL_ERROR("Rping get results failed, ret is [%d]", ret), HCCL_E_NOT_SUPPORT);
    return HCCL_SUCCESS;
}

HcclResult hrtRaIsFirstUsed(s32 insId, bool& used)
{
    CHK_SMART_PTR_NULL(DlRaFunction::GetInstance().dlRaIsFirstUsed);
    s32 ret = DlRaFunction::GetInstance().dlRaIsFirstUsed(insId);

    CHK_PRT_RET(
        ret != 0 && (ret != static_cast<s32>(true)),
        HCCL_ERROR(
            "[hrtRaIsFirstUsed]errNo[0x%016llx] "
            "failed ret[%d]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), ret),
        HCCL_E_NETWORK);

    used = ret == 0 ? false : true;

    HCCL_DEBUG("hrtRaIsFirstUsed insId[%d] success.", insId);
    return HCCL_SUCCESS;
}

HcclResult hrtRaIsLastUsed(s32 insId, bool& used)
{
    CHK_SMART_PTR_NULL(DlRaFunction::GetInstance().dlRaIsLastUsed);
    s32 ret = DlRaFunction::GetInstance().dlRaIsLastUsed(insId);

    CHK_PRT_RET(
        ret != 0 && (ret != static_cast<s32>(true)),
        HCCL_ERROR(
            "[hrtRaIsLastUsed]errNo[0x%016llx] "
            "failed ret[%d]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), ret),
        HCCL_E_NETWORK);

    used = ret == 0 ? false : true;

    HCCL_DEBUG("hrtRaIsLastUsed insId[%d] success.", insId);
    return HCCL_SUCCESS;
}

HcclResult hrtRaRdevGetPortStatus(RdmaHandle rdmaHandle, enum PortStatus* status)
{
    CHK_PTR_NULL(rdmaHandle);
    CHK_SMART_PTR_NULL(DlRaFunction::GetInstance().dlRaRdevGetPortStatus);
    s32 ret = DlRaFunction::GetInstance().dlRaRdevGetPortStatus(rdmaHandle, status);

    CHK_PRT_RET(
        ret != 0,
        HCCL_ERROR(
            "[hrtRaRdevGetPortStatus]errNo[0x%016llx] "
            "failed ret[%d]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), ret),
        HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult HrtRaRemapMr(RdmaHandle rdmaHandle, struct MemRemapInfo info[], unsigned int num)
{
    CHK_PTR_NULL(rdmaHandle);
    if (UNLIKELY(DlRaFunction::GetInstance().dlRaRemapMr == nullptr)) {
        HCCL_ERROR("driver package does not support HrtRaRemapMr, please change new package");
        return HCCL_E_NETWORK;
    };
    s32 ret = DlRaFunction::GetInstance().dlRaRemapMr(rdmaHandle, info, num);

    CHK_PRT_RET(
        ret != 0,
        HCCL_ERROR(
            "[HrtRaRemapMr]errNo[0x%016llx] "
            "failed ret[%d]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), ret),
        HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult CreateQpWithDepthConfig(
    RdmaHandle rdmaHandle, s32 qpMode, const QpConfigInfo& qpConfig, QpHandle& qpHandle, struct TypicalQp& qpInfo)
{
    HCCL_DEBUG(
        "CreateQp qpMode[%d], sq_depth[%u], rq_depth[%u], scq_depth[%u], rcq_depth[%u], TC[%u], SL[%u], "
        "rdmaRetryCnt[%u], rdmaTimeOut[%u]",
        qpMode, qpConfig.sq_depth, qpConfig.rq_depth, qpConfig.scq_depth, qpConfig.rcq_depth, qpInfo.tc, qpInfo.sl,
        qpInfo.retryCnt, qpInfo.retryTime);

    struct QpExtAttrs ext_attrs {};
    ext_attrs.qpMode = qpMode;
    ext_attrs.cqAttr.sendCqDepth = qpConfig.scq_depth;
    ext_attrs.cqAttr.recvCqDepth = qpConfig.rcq_depth;
    ext_attrs.qpAttr.cap.max_send_wr = qpConfig.sq_depth;
    ext_attrs.qpAttr.cap.max_recv_wr = qpConfig.rq_depth;
    ext_attrs.version = QP_CREATE_WITH_ATTR_VERSION;
    ext_attrs.qpAttr.cap.max_inline_data = DEFAULT_MAX_INLINE_DATA;
    ext_attrs.qpAttr.cap.max_send_sge = DEFAULT_MAX_SEND_SGE;
    ext_attrs.qpAttr.cap.max_recv_sge = DEFAULT_MAX_RECV_SGE;
    ext_attrs.qpAttr.qp_type = IBV_QPT_RC;
    ext_attrs.udpSport = 0x0;
    ext_attrs.cstmFlag.bs.useResvMem = qpConfig.use_resv_mem;
    ext_attrs.resvMemPoolId = qpConfig.resv_mem_pool_id;
    s32 deviceLogicID = -1;
    u32 devicePhyId = 0;
    CHK_RET(hrtGetDevice(&deviceLogicID));
    u32 typicalQpModifyVersion = 0;
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(deviceLogicID), devicePhyId));
    // ra_qp_create_with_attrs创建的QP, 后续要使用ra_typical_qp_modify
    // 需要判断ra_typical_qp_modify对应opcode:RA_RS_TYPICAL_QP_MODIFY是否支持支持QP解耦socket建链
    HcclResult vRet = hrtRaGetInterfaceVersion(devicePhyId, TYPICAL_QP_MODIFY, &typicalQpModifyVersion);
    if (vRet != HCCL_SUCCESS || typicalQpModifyVersion < TYPICAL_QP_MODIFY_VERSION) {
        HCCL_ERROR("this package does not support CreateQpWithDepthConfig for device, please change new package");
        return HCCL_E_NOT_SUPPORT;
    }

    CHK_RET(hrtRaQpCreateWithAttrs(rdmaHandle, &ext_attrs, qpHandle));

    struct QpAttr attr {};
    HcclResult ret = hrtRaGetQpAttr(qpHandle, &attr);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[CreateQpWithDepthConfig] hrtRaGetQpAttr failed, ret[%d].", ret);
        HrtRaQpDestroy(qpHandle);
        return ret;
    }
    qpInfo.qpn = attr.qpn;
    qpInfo.gidIdx = attr.gidIdx;
    for (uint32_t i = 0; i < HCCP_GID_RAW_LEN; i++) {
        qpInfo.gid[i] = attr.gid[i];
    }
    qpInfo.psn = attr.psn;
    HCCL_DEBUG("CreateQpWithDepthConfig qpn[%u], gidIdx[%u], psn[%u]", qpInfo.qpn, qpInfo.gidIdx, qpInfo.psn);
    return HCCL_SUCCESS;
}

HcclResult CreateQpWithCQConfig(
    RdmaHandle rdmaHandle, s32 qpMode, const QpConfigWithCQInfo& qpConfig, QpHandle& qpHandle, struct TypicalQp& qpInfo)
{
    HCCL_INFO(
        "CreateQpWithCQ qpMode[%d], sq_depth[%u], rq_depth[%u], scq_depth[%u], rcq_depth[%u], "
        "sendCqn[%u], recvCqn[%u], use_resv_mem[%u], resv_mem_pool_id[%u], "
        "sq_sig_all[%d], max_send_sge[%u], max_recv_sge[%u], max_inline_data[%u]",
        qpMode, qpConfig.sq_depth, qpConfig.rq_depth, qpConfig.scq_depth, qpConfig.rcq_depth, qpConfig.sendCqn,
        qpConfig.recvCqn, qpConfig.use_resv_mem, qpConfig.resv_mem_pool_id, qpConfig.sq_sig_all, qpConfig.max_send_sge,
        qpConfig.max_recv_sge, qpConfig.max_inline_data);

    struct QpExtAttrs ext_attrs {};
    ext_attrs.qpMode = qpMode;
    ext_attrs.cqAttr.sendCqDepth = qpConfig.scq_depth;
    ext_attrs.cqAttr.recvCqDepth = qpConfig.rcq_depth;
    ext_attrs.qpAttr.cap.max_send_wr = qpConfig.sq_depth;
    ext_attrs.qpAttr.cap.max_recv_wr = qpConfig.rq_depth;
    ext_attrs.version = QP_CREATE_WITH_ATTR_VERSION;
    ext_attrs.qpAttr.cap.max_inline_data = qpConfig.max_inline_data;
    ext_attrs.qpAttr.cap.max_send_sge = qpConfig.max_send_sge;
    ext_attrs.qpAttr.cap.max_recv_sge = qpConfig.max_recv_sge;
    ext_attrs.qpAttr.qp_type = IBV_QPT_RC;
    ext_attrs.qpAttr.sq_sig_all = qpConfig.sq_sig_all;
    ext_attrs.udpSport = 0x0;
    ext_attrs.cstmFlag.bs.useResvMem = qpConfig.use_resv_mem;
    ext_attrs.resvMemPoolId = qpConfig.resv_mem_pool_id;

    CHK_RET(hrtRaQpCreateWithCQWithAttrs(rdmaHandle, &ext_attrs, qpConfig.sendCqn, qpConfig.recvCqn, qpHandle));

    struct QpAttr attr {};
    HcclResult ret = hrtRaGetQpAttr(qpHandle, &attr);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[CreateQpWithCQConfig] hrtRaGetQpAttr failed, ret[%d].", ret);
        HrtRaQpDestroy(qpHandle);
        return ret;
    }
    qpInfo.qpn = attr.qpn;
    qpInfo.gidIdx = attr.gidIdx;
    for (uint32_t i = 0; i < HCCP_GID_RAW_LEN; i++) {
        qpInfo.gid[i] = attr.gid[i];
    }
    qpInfo.psn = attr.psn;
    HCCL_DEBUG("CreateQpWithCQConfig qpn[%u], gidIdx[%u], psn[%u]", qpInfo.qpn, qpInfo.gidIdx, qpInfo.psn);
    return HCCL_SUCCESS;
}

HcclResult HrtRaGetTlsEnable(struct RaInfo* info, bool* tlsEnable)
{
    u32 tlsVersion = 0;
    u32 phyId = 0; // phyId无实际意义，这里直接传入0
    HcclResult vRet = hrtRaGetInterfaceVersion(phyId, GET_TLS_ENABLE, &tlsVersion);
    if (vRet != HCCL_SUCCESS || tlsVersion < TLS_ENABLE_VERSION) {
        HCCL_WARNING("this package does not support HrtRaGetTlsEnable for device, please change new package");
        return HCCL_E_NOT_SUPPORT;
    }
    HCCL_DEBUG("HrtRaGetTlsEnable tlsVersion[%u]", tlsVersion);
    s32 ret = DlRaFunction::GetInstance().dlRaRaGetTlsEnable(info, tlsEnable);
    CHK_PRT_RET(
        ret != 0,
        HCCL_ERROR(
            "[HrtRaGetTlsEnable]errNo[0x%016llx] "
            "failed ret[%d]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), ret),
        HCCL_E_NETWORK);
    HCCL_INFO("HrtRaGetTlsEnable phyId[%u], tlsEnable[%d]", info->phyId, *tlsEnable);
    return HCCL_SUCCESS;
}

HcclResult SnapShotSaveAction(s32 networkMode, u32 devicePhyId, HcclSaveSnapShotAction action)
{
    HCCL_INFO("%s networkMode[%d], devicePhyId[%u], action[%d]", __func__, networkMode, devicePhyId, action);
    struct RaInfo raInfo = {};
    raInfo.mode = networkMode;
    raInfo.phyId = devicePhyId;
    s32 ret = DlRaFunction::GetInstance().dlRaSaveSnapShot(&raInfo, static_cast<enum SaveSnapshotAction>(action));
    CHK_PRT_RET(
        ret != 0,
        HCCL_ERROR(
            "%s errNo[0x%016llx] failed ret[%d], networkMode[%d], phyId[%u], action[%d]", __func__,
            HCCL_ERROR_CODE(HCCL_E_NETWORK), ret, networkMode, devicePhyId, action),
        HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult SnapShotRestoreAction(s32 networkMode, u32 devicePhyId)
{
    HCCL_INFO("%s networkMode[%d], devicePhyId[%u]", __func__, networkMode, devicePhyId);
    struct RaInfo raInfo = {};
    raInfo.mode = networkMode;
    raInfo.phyId = devicePhyId;
    s32 ret = DlRaFunction::GetInstance().dlRaRestoreSnapShot(&raInfo);
    CHK_PRT_RET(
        ret != 0,
        HCCL_ERROR(
            "%s errNo[0x%016llx] failed ret[%d], networkMode[%d], phyId[%u]", __func__, HCCL_ERROR_CODE(HCCL_E_NETWORK),
            ret, networkMode, devicePhyId),
        HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult HrtRaGetHccnCfg(s32 networkMode, u32 devicePhyId, enum HccnCfgKeyT key, std::string& value)
{
    u32 raGetHccnCfg = 0;
    HcclResult vRet = hrtRaGetInterfaceVersion(devicePhyId, GET_HCCH_CFG, &raGetHccnCfg);
    static bool isPrintWarning = false;
    if (vRet != HCCL_SUCCESS || raGetHccnCfg < GET_HCCH_CFG_VERSION
        || UNLIKELY(DlRaFunction::GetInstance().dlRaGetHccnCfg == nullptr)) {
        if (!isPrintWarning) {
            HCCL_WARNING(
                "[HrtRaGetHccnCfg] this package does not support HrtRaGetHccnCfg for device, "
                "please change new package ret[%d], version[%lu]",
                static_cast<int>(vRet), raGetHccnCfg);
            isPrintWarning = true;
        }
        return HCCL_SUCCESS;
    }

    if ((key == HccnCfgKeyT::HCCN_RESV_MEM_INFO) && (raGetHccnCfg <= GET_HCCH_CFG_VERSION)) {
        HCCL_WARNING(
            "[HrtRaGetHccnCfg] this package does not support resvMem for device, "
            "please change new package ret[%d], version[%lu]",
            static_cast<int>(vRet), raGetHccnCfg);
        return HCCL_SUCCESS;
    }

    struct RaInfo raInfo = {};
    raInfo.mode = networkMode;
    raInfo.phyId = devicePhyId;

    HccnCfgKey hccnKey{HccnCfgKey::HCCN_CFG_UDP_PORT_MODE};
    switch (key) {
        case HccnCfgKeyT::HCCN_UDP_PORT_MODE:
            hccnKey = HccnCfgKey::HCCN_CFG_UDP_PORT_MODE;
            break;
        case HccnCfgKeyT::HCCN_MULTI_QP_COUNT:
            hccnKey = HccnCfgKey::HCCN_CFG_MULTI_QP_COUNT;
            break;
        case HccnCfgKeyT::HCCN_MULTI_QP_UDP_PORTS:
            hccnKey = HccnCfgKey::HCCN_CFG_MULTI_QP_UDP_PORTS;
            break;
        case HccnCfgKeyT::HCCN_RESV_MEM_INFO:
            hccnKey = HccnCfgKey::HCCN_CFG_RESV_MEM_INFO;
            break;
        default:
            HCCL_ERROR("[HrtRaGetHccnCfg]not support key[%d]", key);
            return HCCL_E_PARA;
    }

    constexpr std::uint32_t READ_MAX_LEN = 1024 * 2;
    std::vector<char> buffer(READ_MAX_LEN);
    int actualLen = static_cast<int>(buffer.size());
    s32 ret = DlRaFunction::GetInstance().dlRaGetHccnCfg(&raInfo, hccnKey, buffer.data(), &actualLen);
    if (ret == 0 && actualLen == 0) { // 文件不存在的话 HCCP长度返回0,且ret为0
        HCCL_WARNING(
            "[HrtRaGetHccnCfg] device networkMode[%d] with phyId[%u], "
            "get hccn config key[%d] info is empty. Possible reasons: "
            "1. Device not need to use multi_qp/nslb-dp settings. "
            "2. In this package, hccn_tool not support multi_qp/nslb-dp settings. "
            "3. The right key not exist in device's config file or key's value is empty.",
            networkMode, devicePhyId, key);
        value.assign(buffer.data(), actualLen);
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(
        ret != 0, // 其他
        HCCL_ERROR(
            "[HrtRaGetHccnCfg]errNo[0x%016llx] error occurred."
            " networkMode[%d], devicePhyId[%u], key[%d], return: ret[%d]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), networkMode, devicePhyId, key, ret),
        HCCL_E_NETWORK);
    value.assign(buffer.data(), actualLen != 0 && buffer[actualLen - 1] == '\0' ? actualLen - 1 : actualLen);
    HCCL_DEBUG(
        "[HrtRaGetHccnCfg]devicePhyId[%u] key[%d], value[%s], value len[%d]", devicePhyId, key, value.c_str(),
        actualLen);
    return HCCL_SUCCESS;
}

HcclResult hrtRaGetSecRandom(struct RaInfo* info, unsigned int* token)
{
    if (DlRaFunction::GetInstance().dlRaGetSecRandom == nullptr) {
        HCCL_ERROR("driver package does not support dlRaGetSecRandom, please change new package");
        return HCCL_E_NOT_SUPPORT;
    }
    s32 ret = DlRaFunction::GetInstance().dlRaGetSecRandom(info, token);
    if (ret != 0) {
        HCCL_ERROR("[HrtRaGetSecRandom] RaGetSecRandom failed, call interface, ret[%d]", ret);
        return HCCL_E_NETWORK;
    }
    return HCCL_SUCCESS;
}

HcclResult hrtRaGetDevEidInfoNum(RaInfo info, unsigned int* num)
{
    if (DlRaFunction::GetInstance().dlRaGetDevEidInfoNum == nullptr) {
        HCCL_ERROR("driver package does not support dlRaGetDevEidInfoNum, please change new package");
        return HCCL_E_NOT_SUPPORT;
    }
    s32 ret = DlRaFunction::GetInstance().dlRaGetDevEidInfoNum(info, num);
    if (ret != 0) {
        HCCL_ERROR("[HrtRaGetSecRandom] RaGetDevEidInfoNum failed, call interface, ret[%d]", ret);
        return HCCL_E_NETWORK;
    }
    return HCCL_SUCCESS;
}

HcclResult hrtRaGetDevEidInfoList(RaInfo info, struct HccpDevEidInfo* eid_info, unsigned int* num)
{
    if (DlRaFunction::GetInstance().dlRaGetDevEidInfoList == nullptr) {
        HCCL_ERROR("driver package does not support dlRaGetDevEidInfoNum, please change new package");
        return HCCL_E_NOT_SUPPORT;
    }
    s32 ret = DlRaFunction::GetInstance().dlRaGetDevEidInfoList(info, eid_info, num);
    if (ret != 0) {
        HCCL_ERROR("[HrtRaGetSecRandom] RaGetDevEidInfoList failed, call interface, ret[%d]", ret);
        return HCCL_E_NETWORK;
    }
    return HCCL_SUCCESS;
}
