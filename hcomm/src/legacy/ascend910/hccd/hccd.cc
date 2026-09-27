/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_types.h"
#include "hccl/base.h"
#include "hccl/hccl_ex.h"
#include "dlra_function.h"
#include "externalinput_pub.h"
#include "mr_manager.h"
#include "rank_consistentcy_checker.h"
#include "transport_heterog_def.h"
#include "transport_heterog.h"
#include "hccd_private.h"
// ltm指定config路径
#include "common/src/config.h"
#include "hccd_comm.h"
#include "hccd_pub.h"
#include "adapter_hal.h"
#include "dlhal_function.h"
#include <dlog_pub.h>

using namespace std;
using namespace hccl;
constexpr u32 TIME_THREE_TIMEGAP = 3;

HcclResult HccdGenerateCommId(hccl::HcclCommParams& params)
{
    s32 sRet = memset_s(params.id.internal, HCCL_ROOT_INFO_BYTES, 0, sizeof(params.id.internal));
    CHK_PRT_RET(sRet != EOK, HCCL_ERROR("[GenerateCommId]memory set error. return[%d].", sRet), HCCL_E_PARA);

    HcclRootInfo uniqueId;
    std::string group;
    CHK_RET(HccdComm::GetUniqueId(&uniqueId));

    group = "hccl_heterog_group";

    sRet = snprintf_s(
        params.id.internal, HCCL_ROOT_INFO_BYTES, HCCL_ROOT_INFO_BYTES - 1, "%s%s%s", uniqueId.internal, "-",
        group.c_str());
    CHK_PRT_RET(
        sRet == -1,
        HCCL_ERROR("[GenerateCommId]errNo[0x%016llx] sal snprintf_s error", HCCL_ERROR_CODE(HCCL_E_INTERNAL)),
        HCCL_E_INTERNAL);
    HCCL_INFO("params.id.internal [%s]", params.id.internal);
    return HCCL_SUCCESS;
}
int32_t DlogSetAttr(LogAttr logAttrInfo) __attribute((weak));
HcclResult HcclInitComm(const char* rankTableM, uint32_t rank, const CommAttr* attr, HcclComm* comm, HccdInfo& rankInfo)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclUs startut = TIME_NOW();
    auto timeGap = TIME_NOW() - startut;

    // 入参合法性检查
    CHK_PTR_NULL(rankTableM);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(attr);

    // 为了解决云助端device 日志不上传的问题，需要调用DlogSetAttr
    CHK_RET(DlHalFunction::GetInstance().DlHalFunctionInit());
    unsigned int chipId = 0;
    unsigned int vfid = 0;
    unsigned int hostPid = 0;
    unsigned int cpType = 0;
    int curPid = SalGetPid();
    CHK_RET(HrtHalDrvQueryProcessHostPid(curPid, &chipId, &vfid, &hostPid, &cpType));
    LogAttr logattr{};
    logattr.type = APPLICATION;
    logattr.pid = hostPid;
    logattr.deviceId = attr->deviceId;
    if (DlogSetAttr != nullptr && DlogSetAttr(logattr) != 0) {
        HCCL_ERROR("DlogSetAttr failed");
        return HCCL_E_SYSCALL;
    }

    std::string rankTableStr = rankTableM;

    /* 接口交互信息日志 */
    HCCL_RUN_INFO("Entry-HcclInitComm:clusterInfo, rank[%u]", rank);
    if (attr->mode != WorkMode::HCCL_MODE_AI_CPU) {
        CHK_RET(DlRaFunction::GetInstance().DlRaFunctionInit());
    }

    /* --------------初始化------------------------- */
    bool errorFlag = false;
    hccl::HccdComm* pComm = nullptr;
    do {
        // 初始化外部参数
        ret = InitExternalInput();
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[Init][HccdComm]errNo[0x%016llx] init external input error", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        ret = InitExternalInputHeterog();
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[Init][HccdComm]errNo[0x%016llx] init external input error", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        // 解析rankTable_json对象，将解析的信息保存在rankinfo中，ranktableCRC计算
        ret = HcclParseRanktable(rankTableStr, to_string(rank), rankInfo.params, rankInfo.rankTable);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Init][HccdComm]errNo[0x%016llx] hccl analysis ranktable "
                "info error:rank [%u]",
                HCCL_ERROR_CODE(ret), rank),
            errorFlag = true);

        // 生成通信域标识符
        ret = HccdGenerateCommId(rankInfo.params);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[Init][OtherInfo]errNo[0x%016llx] generate CommId error", HCCL_ERROR_CODE(HCCL_E_INTERNAL)),
            HCCL_E_INTERNAL);

        for (auto& rankIter : rankInfo.rankTable.rankList) {
            if (rankIter.rankId == rank) {
                if (attr->mode != WorkMode::HCCL_MODE_PS && attr->mode != WorkMode::HCCL_MODE_AI_CPU) {
                    rankIter.deviceInfo.devicePhyId = attr->deviceId;
                    break;
                }
            }
        }

        // new新对象
        pComm = new (std::nothrow) hccl::HccdComm(rankInfo.rankTable.collectiveId);
        CHK_PTR_NULL(pComm);

        // 根据rankinfo结构体的内容,初始化通信域
        rankInfo.params.commHandle = pComm;
        rankInfo.params.attr = *attr;
        ret = pComm->init(rankInfo.params, rankInfo.rankTable);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[Init][HccdComm]errNo[0x%016llx] HeterogComm init error", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        TransportHeterog::RecordRankTableCrc(rankInfo.params.ranktableCrc);

        // 打印rankTable信息
        ret = ShowRanktableConfigInfo(rankInfo.cloudFlag, rankInfo.params, rankInfo.rankTable);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[Init][HccdComm]errNo[0x%016llx] put ranktable info error", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        *comm = pComm;
    } while (0);

    if (errorFlag) {
        HCCL_ERROR("[Init][CommClusterInfo]HeterogCommClusterInfto failed, return[0x%016llx]", HCCL_ERROR_CODE(ret));
        (void)HcclFinalizeComm(pComm);
        return ret;
    }

    HCCL_RUN_INFO(
        "HcclInitComm success,take time [%lld]us, collectiveId[%s], rankSize[%u], rank[%u]",
        TAKE_TIME_US((TIME_NOW() - startut), (TIME_THREE_TIMEGAP * timeGap)), pComm->GetIdentifier().c_str(),
        rankInfo.rankTable.rankNum, rank);
    return HCCL_SUCCESS;
}

HcclResult HcclInitComm(const char* rankTableM, uint32_t rank, const CommAttr* attr, HcclComm* comm)
{
    HccdInfo rankInfo;
    return HcclInitComm(rankTableM, rank, attr, comm, rankInfo);
}

HcclResult HcclFinalizeComm(HcclComm comm)
{
    HcclUs startut = TIME_NOW();
    auto timeGap = TIME_NOW() - startut;
    u32 rankSize = 0;
    u32 rank = 0;
    // 入参检查
    CHK_PRT_RET(
        comm == nullptr, HCCL_WARNING("[Destroy][HcclHeterogComm]An empty comm given, skip destroy."), HCCL_SUCCESS);
    // 指针类型转换hcclComm*<-void*
    hccl::HccdComm* hccdComm = static_cast<hccl::HccdComm*>(comm);
    HCCL_RUN_INFO("Entry-HcclFinalizeComm: comm[%p].", comm);

    HcclResult ret = HCCL_SUCCESS;
    bool errorFlag = false;
    do {
        // 记录打印信息
        ret = hccdComm->GetUserRank(rank);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[Finalize][HccdComm]errNo[0x%016llx] get user rank error.", HCCL_ERROR_CODE(ret)),
            errorFlag = true);
        ret = hccdComm->GetRankSize(rankSize);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[Finalize][HccdComm]errNo[0x%016llx] get rank size error.", HCCL_ERROR_CODE(ret)),
            errorFlag = true);
    } while (0);

    std::string collectiveId = hccdComm->GetIdentifier();
    // 模型运行结束后hcom destroy时，将记录的rank table crc置为0
    TransportHeterog::RecordRankTableCrc(0);

    // 释放资源
    delete hccdComm;
    hccdComm = nullptr;

    if (errorFlag) {
        HCCL_ERROR("[Finalize][HccdComm]HcclFinalizeComm failed, return[0x%016llx].", HCCL_ERROR_CODE(ret));
        return ret;
    }

    HcclUs endut = TIME_NOW();
    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "HcclFinalizeComm success, take time [%lld]us, collectiveId[%s], rankSize[%u], rank[%u].",
        TAKE_TIME_US((endut - startut), (TIME_THREE_TIMEGAP * timeGap)), collectiveId.c_str(), rankSize, rank);
    return HCCL_SUCCESS;
}

// 注册全局内存，全进程共享，以优化RDMA性能
HcclResult HcclRegisterMemory(HcclComm comm, void* buffer, uint64_t size)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(buffer);
    HcclUs startut = TIME_NOW();

    hccl::HccdComm* hccdComm = static_cast<hccl::HccdComm*>(comm);
    u32 localRank = INVALID_VALUE_RANKID;
    CHK_RET(hccdComm->GetUserRank(localRank));
    HCCL_RUN_INFO(
        "Entry-HcclRegisterMemory: comm[%s], addr[%p], size[%llu] localRank[%u].", hccdComm->GetIdentifier().c_str(),
        buffer, size, localRank);

    CHK_RET(hccdComm->RegisterMemory(buffer, size));

    HCCL_RUN_INFO(
        "Hccl Register Memory success, take time [%lld]us, addr[%p], size[%llu].", DURATION_US(TIME_NOW() - startut),
        buffer, size);
    return HCCL_SUCCESS;
}

// 解注册全局内存，全进程共享，以优化RDMA性能
HcclResult HcclUnregisterMemory(HcclComm comm, void* buffer)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(buffer);
    HcclUs startut = TIME_NOW();

    hccl::HccdComm* hccdComm = static_cast<hccl::HccdComm*>(comm);
    u32 localRank = INVALID_VALUE_RANKID;
    CHK_RET(hccdComm->GetUserRank(localRank));
    HCCL_RUN_INFO(
        "Entry-HcclUnregisterMemory: comm[%s], addr[%p], localRank[%u].", hccdComm->GetIdentifier().c_str(), buffer,
        localRank);

    CHK_RET(hccdComm->UnregisterMemory(buffer));

    HCCL_RUN_INFO(
        "Hccl Unregister Memory success, take time [%lld]us, addr[%p].", DURATION_US(TIME_NOW() - startut), buffer);
    return HCCL_SUCCESS;
}

// 以进程粒度注册全局内存，多Server场景使用
HcclResult HcclRegisterGlobalMemory(void* addr, u64 size)
{
    CHK_PTR_NULL(addr);
    HcclUs startut = TIME_NOW();
    HcclResult ret = MrManager::GetInstance().RegGlobalMr(addr, size);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[HcclRegisterGlobalMemory]errNo[0x%016llx] Hccl Register Global Memory failed, size[%llu].",
            HCCL_ERROR_CODE(ret), size),
        ret);
    HCCL_INFO("Hccl Register Global success, take time [%lld]us.", DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

// 以进程粒度注销全局内存，多Server场景使用
HcclResult HcclUnregisterGlobalMemory(void* addr)
{
    CHK_PTR_NULL(addr);
    HcclUs startut = TIME_NOW();
    HcclResult ret = MrManager::GetInstance().DeRegGlobalMr(addr);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[HcclUnregisterGlobalMemory]errNo[0x%016llx] Hccl Unregister Global Memory failed.", HCCL_ERROR_CODE(ret)),
        ret);
    HCCL_INFO("Hccl Unregister Global success, take time [%lld]us.", DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

// 异步发送数据，发送完成后会通过event上报完成状态。根据{commHandle, dstRank, tag}为粒度，顺序发送。不同的颗粒间可并行。
int HcclIsend(void* buffer, int count, HcclDataType dataType, int dstRank, int tag, HcclComm comm, HcclRequest* request)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(request);
    CHK_PRT_RET(
        buffer == nullptr && count != 0, HCCL_ERROR("HcclIsend failed, buffer is nullptr, count is %d", count),
        HCCL_E_PARA);

    hccl::HccdComm* hccdComm = static_cast<hccl::HccdComm*>(comm);
    u32 localRank = INVALID_VALUE_RANKID;
    u32 userRequire = 0;
    CHK_RET(hccdComm->GetUserRank(localRank));

    /* 接口交互信息日志 */
    HCCL_INFO(
        "Entry-HcclIsend: comm[%s] localRank[%u] dstRank[%d] tag[%d]: addr[%p] count[%d] dtype[%s] "
        "request[%p]",
        hccdComm->GetIdentifier().c_str(), localRank, dstRank, tag, buffer, count, GetDataTypeEnumStr(dataType).c_str(),
        request);

    TIME_PRINT(
        CHK_RET(hccdComm->Isend(buffer, count, dataType, static_cast<u32>(dstRank), tag, *request, userRequire)));

    /* 关键状态记录 */
    HCCL_INFO(
        "HcclIsend success. comm[%s] localRank[%u] dstRank[%d] tag[%d]: addr[%p] count[%d] dtype[%s] "
        "*request[%p].",
        hccdComm->GetIdentifier().c_str(), localRank, dstRank, tag, buffer, count, GetDataTypeEnumStr(dataType).c_str(),
        *request);
    return HCCL_SUCCESS;
}

// 类mpi 查询已捕获的recv request操作信息，根据{commHandle, peerRank, tag}为粒度，按照recv
// request接收顺序依次返回已捕获的
int HcclImprobe(int srcRank, int tag, HcclComm comm, int* flag, HcclMessage* msg, HcclStatus* status)
{
    // 入参校验
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(flag);
    CHK_PTR_NULL(msg);
    CHK_PTR_NULL(status);

    // 关键日志记录
    hccl::HccdComm* hccdComm = static_cast<hccl::HccdComm*>(comm);
    u32 localRank = INVALID_VALUE_RANKID;
    CHK_RET(hccdComm->GetUserRank(localRank));
    HCCL_INFO(
        "Entry-HcclImprobe: comm[%s] srcRank[%d] localRank[%u] tag[%d]", hccdComm->GetIdentifier().c_str(), srcRank,
        localRank, tag);
    // 获取 recv request entry
    CHK_RET(hccdComm->Improbe(static_cast<u32>(srcRank), tag, *flag, *msg, *status));
    HcclMessageInfo* hcclMsg = static_cast<HcclMessageInfo*>(*msg);
    if (*flag == HCCL_IMPROBE_COMPLETED) {
        hcclMsg->commHandle = comm;
    }
    // 关键日志记录
    HCCL_INFO(
        "HcclImprobe success. comm[%s] srcRank[%d] localRank[%u] tag[%d]: flag[%d] status[%d] count[%d] msg[%p].",
        hccdComm->GetIdentifier().c_str(), srcRank, localRank, tag, *flag, status->error, status->count, *msg);
    return HCCL_SUCCESS;
}

int HcclGetCount(const HcclStatus* status, HcclDataType dataType, int* count)
{
    // 入参校验
    CHK_PTR_NULL(status);
    CHK_PTR_NULL(count);

    if (status->error != 0) {
        HCCL_WARNING("GetCount::Failed to obtain the count status[%d].", status->error);
        return HCCL_E_PARA;
    }

    *count = status->count;

    // 关键日志记录
    HCCL_INFO(
        "HcclGetCount success. peerRank[%d] tag[%d] status[%d] dataType[%s] count[%d].", status->srcRank, status->tag,
        status->error, GetDataTypeEnumStr(dataType).c_str(), *count);
    return HCCL_SUCCESS;
}

int HcclImrecv(void* buffer, int count, HcclDataType dataType, HcclMessage* msg, HcclRequest* request)
{
    CHK_PTR_NULL(msg);
    CHK_PTR_NULL(*msg);
    CHK_PTR_NULL(request);

    HcclMessageInfo* hcclMsg = static_cast<HcclMessageInfo*>(*msg);
    hccl::HccdComm* hccdComm = static_cast<hccl::HccdComm*>(hcclMsg->commHandle);
    CHK_PTR_NULL(hccdComm);
    uint32_t peerRank = hcclMsg->envelope.envelope.epParam.src.rank;
    uint32_t tag = hcclMsg->envelope.envelope.epParam.src.tag;
    /* 接口交互信息日志 */
    HCCL_INFO(
        "Entry-HcclImrecv: comm[%s] peerRank[%u] tag[%u]: addr[%p] count[%d] dtype[%s] msg[%p]",
        hccdComm->GetIdentifier().c_str(), peerRank, tag, buffer, count, GetDataTypeEnumStr(dataType).c_str(), *msg);

    TIME_PRINT(CHK_RET(hccdComm->Imrecv(buffer, count, dataType, *msg, *request)));

    HcclRequestInfo* hcclReq = static_cast<HcclRequestInfo*>(*request);
    hcclReq->commHandle = hccdComm;
    /* 关键状态记录 */
    HCCL_INFO(
        "HcclImrecv success. comm[%s] peerRank[%u] tag[%u]: addr[%p] count[%d] dtype[%s] msg[%p] request[%p].",
        hccdComm->GetIdentifier().c_str(), peerRank, tag, buffer, count, GetDataTypeEnumStr(dataType).c_str(), *msg,
        *request);
    return HCCL_SUCCESS;
}

int HcclTestSome(int count, HcclRequest requestArray[], int* compCount, int compIndices[], HcclStatus compStatus[])
{
    // 入参校验
    CHK_PTR_NULL(compCount);
    CHK_PTR_NULL(requestArray);
    CHK_PTR_NULL(compIndices);
    CHK_PTR_NULL(compStatus);

    *compCount = 0;
    bool errorFlag = false;
    HcclResult ret = HCCL_SUCCESS;
    for (int i = 0; i < count; ++i) {
        HcclRequestInfo* hcclReq = reinterpret_cast<HcclRequestInfo*>(requestArray[i]);
        if (hcclReq == nullptr) {
            HCCL_WARNING("[%d]th hcclRequest is nullptr, no need to testSome", i);
            continue;
        }

        hccl::HccdComm* hccdComm = reinterpret_cast<hccl::HccdComm*>(hcclReq->commHandle);
        CHK_PTR_NULL(hccdComm);
        s32 comp = HCCL_TEST_INCOMPLETED;
        ret = hccdComm->HcclTest(requestArray[i], comp, compStatus[*compCount]);
        if (ret != HCCL_SUCCESS) {
            compStatus[*compCount].error = GetExternalInputHcclIsTcpMode() ? HCCL_E_TCP_TRANSFER : HCCL_E_ROCE_TRANSFER;
            compIndices[*compCount] = i;
            errorFlag = true;
            (*compCount)++;
        } else if (comp == HCCL_TEST_COMPLETED) {
            requestArray[i] = nullptr;
            compIndices[*compCount] = i;
            compStatus[*compCount].error = HCCL_SUCCESS;
            (*compCount)++;
        }
        HCCL_INFO(
            "HcclTestSome: array[%d/%d] request[%p] comm[%s] peerRank[%u] tag[%d] type[%u] flag[%d] "
            "compCount[%d] status[%d]",
            i + 1, count, hcclReq, hccdComm->GetIdentifier().c_str(), hcclReq->transportRequest.epParam.src.rank,
            hcclReq->transportRequest.epParam.src.tag, hcclReq->transportRequest.requestType, comp, *compCount,
            hcclReq->transportRequest.status);
    }
    if (errorFlag) {
        HCCL_ERROR("HcclTestSome: some request link is exception");
        return HCCL_E_IN_STATUS;
    }
    return HCCL_SUCCESS;
}
