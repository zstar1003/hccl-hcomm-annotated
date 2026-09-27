/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl/hccl_ex.h"
#include "hccl_types.h"
#include "hccl_comm_conn.h"
#include "hccl_comm_conn_mgr.h"

using namespace std;
using namespace hccl;

HcclResult HcclRawOpen(HcclConn* conn)
{
    CHK_PTR_NULL(conn);

    HCCL_RUN_INFO("Entry %s start conn[%llu]", __func__, hash<HcclConn*>{}(conn));
    HcclCommConn** comm = reinterpret_cast<HcclCommConn**>(conn);
    CHK_RET(HcclCommConnMgr::GetInstance().AddAndGetCommConn(*comm));
    HCCL_RUN_INFO("%s success conn[%llu]", __func__, hash<void*>{}(*comm));

    return HCCL_SUCCESS;
}

HcclResult HcclRawClose(HcclConn conn)
{
    CHK_PTR_NULL(conn);

    HCCL_RUN_INFO("Entry %s start conn[%llu]", __func__, hash<void*>{}(conn));
    HcclCommConn* comm = static_cast<HcclCommConn*>(conn);
    CHK_RET(HcclCommConnMgr::GetInstance().DelCommConn(comm));
    HCCL_RUN_INFO("%s success conn[%llu]", __func__, hash<void*>{}(conn));

    return HCCL_SUCCESS;
}

HcclResult HcclRawForceClose(HcclConn conn)
{
    CHK_PTR_NULL(conn);

    HCCL_RUN_INFO("Entry %s start conn[%llu]", __func__, hash<void*>{}(conn));
    HcclCommConn* comm = static_cast<HcclCommConn*>(conn);
    comm->SetForceClose();
    CHK_RET(HcclCommConnMgr::GetInstance().DelCommConn(comm));
    HCCL_RUN_INFO("%s success conn[%llu]", __func__, hash<void*>{}(conn));

    return HCCL_SUCCESS;
}

HcclResult HcclRawConnect(HcclConn conn, HcclAddr* connectAddr)
{
    CHK_PTR_NULL(conn);
    CHK_PTR_NULL(connectAddr);

    HCCL_DEBUG("Entry %s start", __func__);
    if (HcclCommConnMgr::GetInstance().IsExistCommConn(*connectAddr)) {
        HCCL_ERROR(
            "cur client to remote ip[%s] port[%u] comm conn is exist.",
            HcclIpAddress((*connectAddr).info.tcp.ipv4Addr).GetReadableIP(), (*connectAddr).info.tcp.port);
        return HCCL_E_UNAVAIL;
    }

    if (HcclCommConnMgr::GetInstance().IsExceedMaxLinkNum(CLIENT_ROLE_SOCKET)) {
        HCCL_ERROR("The maximum number of communication connections that can be created is %u.", MAX_CONN_LINK_NUM);
        return HCCL_E_UNAVAIL;
    }
    HcclCommConn* comm = static_cast<HcclCommConn*>(conn);
    CHK_RET(comm->Connect(*connectAddr));
    HcclCommConnMgr::GetInstance().InsertConnectCommMap(*connectAddr, conn);
    HCCL_RUN_INFO(
        "%s success conn[%llu] connectAddr[%llu]", __func__, hash<void*>{}(conn), hash<HcclAddr*>{}(connectAddr));

    return HCCL_SUCCESS;
}

HcclResult HcclRawBind(HcclConn conn, HcclAddr* bindAddr)
{
    CHK_PTR_NULL(conn);
    CHK_PTR_NULL(bindAddr);

    HCCL_RUN_INFO("Entry %s start", __func__);
    HcclCommConn* comm = static_cast<HcclCommConn*>(conn);
    CHK_RET(comm->Bind(*bindAddr));
    HCCL_RUN_INFO("%s success conn[%llu] bindAddr[%llu]", __func__, hash<void*>{}(conn), hash<HcclAddr*>{}(bindAddr));

    return HCCL_SUCCESS;
}

HcclResult HcclRawListen(HcclConn conn, int backLog)
{
    CHK_PTR_NULL(conn);

    HCCL_RUN_INFO("Entry %s start conn[%llu]", __func__, hash<void*>{}(conn));
    HcclCommConn* comm = static_cast<HcclCommConn*>(conn);
    CHK_RET(comm->Listen(backLog));
    HCCL_RUN_INFO("%s success", __func__);

    return HCCL_SUCCESS;
}

HcclResult HcclRawAccept(HcclConn conn, HcclAddr* acceptAddr, HcclConn* acceptConn)
{
    CHK_PTR_NULL(conn);
    CHK_PTR_NULL(acceptAddr);
    CHK_PTR_NULL(acceptConn);

    HCCL_DEBUG("Entry %s start", __func__);
    HcclCommConn* comm = static_cast<HcclCommConn*>(conn);
    HcclCommConn** newConn = reinterpret_cast<HcclCommConn**>(acceptConn);

    CHK_RET(comm->Accept(*acceptAddr, *newConn));
    CHK_RET(HcclCommConnMgr::GetInstance().AddCommConn(*newConn));
    HCCL_RUN_INFO(
        "%s success conn[%llu] acceptAddr[%llu] acceptConn[%llu]", __func__, hash<void*>{}(conn),
        hash<HcclAddr*>{}(acceptAddr), hash<void*>{}(*newConn));

    return HCCL_SUCCESS;
}

HcclResult HcclRawIsend(const void* buf, int count, HcclDataType dataType, HcclConn conn, HcclRequest* request)
{
    CHK_PTR_NULL(conn);
    CHK_PTR_NULL(request);

    HCCL_DEBUG("Entry %s start", __func__);
    HcclCommConn* comm = static_cast<HcclCommConn*>(conn);
    CHK_RET(comm->Isend(buf, count, dataType, *request));
    HcclRequestInfo* hcclReq = static_cast<HcclRequestInfo*>(*request);
    hcclReq->commHandle = comm;
    HCCL_DEBUG("%s success", __func__);

    return HCCL_SUCCESS;
}

HcclResult HcclRawImprobe(HcclConn conn, int* flag, HcclMessage* msg, HcclStatus* status)
{
    CHK_PTR_NULL(conn);
    CHK_PTR_NULL(flag);
    CHK_PTR_NULL(msg);
    CHK_PTR_NULL(status);

    HCCL_DEBUG("Entry %s start", __func__);
    HcclCommConn* comm = static_cast<HcclCommConn*>(conn);
    CHK_RET(comm->Improbe(*flag, *msg, *status));
    HcclMessageInfo* hcclMsg = static_cast<HcclMessageInfo*>(*msg);
    if (*flag == HCCL_IMPROBE_COMPLETED) {
        hcclMsg->commHandle = comm;
    }
    HCCL_DEBUG("%s success", __func__);

    return HCCL_SUCCESS;
}
HcclResult HcclRawImrecv(void* buf, int count, HcclDataType datatype, HcclMessage* msg, HcclRequest* request)
{
    CHK_PTR_NULL(msg);
    CHK_PTR_NULL(*msg);
    CHK_PTR_NULL(request);

    HCCL_DEBUG("Entry %s start", __func__);
    HcclMessageInfo* hcclMsg = static_cast<HcclMessageInfo*>(*msg);
    HcclCommConn* comm = static_cast<HcclCommConn*>(hcclMsg->commHandle);
    CHK_RET(comm->Imrecv(buf, count, datatype, *msg, *request));
    HcclRequestInfo* hcclReq = static_cast<HcclRequestInfo*>(*request);
    hcclReq->commHandle = comm;
    *msg = nullptr;
    HCCL_DEBUG("%s success", __func__);

    return HCCL_SUCCESS;
}

HcclResult HcclRawImrecvScatter(
    void* buf[], int count[], int bufCount, HcclDataType datatype, HcclMessage* msg, HcclRequest* request)
{
    CHK_PTR_NULL(buf);
    CHK_PTR_NULL(count);
    CHK_PTR_NULL(msg);
    CHK_PTR_NULL(*msg);
    CHK_PTR_NULL(request);

    if (bufCount > MAX_SCATTER_BUF_NUM) {
        HCCL_ERROR("bufCount[%d] should be less than %d", bufCount, MAX_SCATTER_BUF_NUM);
        return HCCL_E_PARA;
    }

    HCCL_DEBUG("Entry %s start", __func__);
    HcclMessageInfo* hcclMsg = static_cast<HcclMessageInfo*>(*msg);
    HcclCommConn* comm = static_cast<HcclCommConn*>(hcclMsg->commHandle);
    CHK_RET(comm->ImrecvScatter(buf, count, bufCount, datatype, *msg, *request));
    HcclRequestInfo* hcclReq = static_cast<HcclRequestInfo*>(*request);
    hcclReq->commHandle = comm;
    *msg = nullptr;

    return HCCL_SUCCESS;
}

HcclResult HcclRawGetCount(const HcclStatus* status, HcclDataType dataType, int* count)
{
    // 入参校验
    CHK_PTR_NULL(status);
    CHK_PTR_NULL(count);

    HCCL_DEBUG("Entry %s start", __func__);
    if (status->error != 0) {
        HCCL_WARNING("Failed to obtain the count status[%d].", status->error);
        return HCCL_E_PARA;
    }

    *count = status->count;
    HCCL_DEBUG(
        "%s success. peerRank[%d] tag[%d] status[%d] dataType[%s] count[%d].", __func__, status->srcRank, status->tag,
        status->error, GetDataTypeEnumStr(dataType).c_str(), *count);

    return HCCL_SUCCESS;
}

HcclResult
HcclRawTestSome(int count, HcclRequest requestArray[], int* compCount, int compIndices[], HcclStatus compStatus[])
{
    // 入参校验
    CHK_PTR_NULL(compCount);
    CHK_PTR_NULL(requestArray);
    CHK_PTR_NULL(compIndices);
    CHK_PTR_NULL(compStatus);

    HCCL_DEBUG("Entry %s start", __func__);
    *compCount = 0;
    bool errorFlag = false;
    HcclResult ret = HCCL_SUCCESS;
    for (int i = 0; i < count; ++i) {
        HcclRequestInfo* hcclReq = reinterpret_cast<HcclRequestInfo*>(requestArray[i]);
        if (hcclReq == nullptr) {
            HCCL_INFO("[%d]th hcclRequest is nullptr, no need to testSome", i);
            continue;
        }

        HcclCommConn* comm = reinterpret_cast<hccl::HcclCommConn*>(hcclReq->commHandle);
        CHK_PTR_NULL(comm);

        s32 comp = HCCL_TEST_INCOMPLETED;
        ret = comm->Test(requestArray[i], comp, compStatus[*compCount]);
        if (ret != HCCL_SUCCESS) {
            compStatus[*compCount].error = HCCL_E_ROCE_TRANSFER;
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
            "HcclRawTestSome: array[%d/%d] type[%u] flag[%d] compCount[%d] status[%d]", i + 1, count,
            hcclReq->transportRequest.requestType, comp, *compCount, hcclReq->transportRequest.status);
    }

    if (errorFlag) {
        HCCL_ERROR("HcclRawTestSome: some request link is exception");
        return HCCL_E_INTERNAL;
    }

    HCCL_DEBUG("%s success", __func__);

    return HCCL_SUCCESS;
}
