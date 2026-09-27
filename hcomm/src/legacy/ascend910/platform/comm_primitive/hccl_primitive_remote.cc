/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "transport.h"
#include "dispatcher_pub.h"
#include "hccl_primitive_remote.h"
#include "hccl_primitive_local.h"

using namespace hccl;
extern HcclResult GetPubDispatcher(hccl::DispatcherPub** dispatcherPtr);
HcclResult HcclRemoteWrite(StreamHandle streamHandle, HcclMemTransport memTransport, HcclBuf* rmtBuf, HcclBuf* locBuf)
{
    CHK_PTR_NULL(streamHandle);
    CHK_PTR_NULL(memTransport);
    CHK_PTR_NULL(rmtBuf);
    CHK_PTR_NULL(locBuf);
    HCCL_DEBUG(
        "[HcclRemoteWrite]streamHandle[%p], memTransport[%p], locBuf addr[%p], rmtBuf addr[%p], len[%llu].",
        streamHandle, memTransport, locBuf->addr, rmtBuf->addr, rmtBuf->len);
    Stream* stream = reinterpret_cast<Stream*>(streamHandle);
    struct Transport::Buffer localBuf(locBuf->addr, locBuf->len);
    struct Transport::Buffer remoteBuf(rmtBuf->addr, rmtBuf->len);

    return reinterpret_cast<Transport*>(memTransport)->WriteAsync(remoteBuf, localBuf, *stream);
}

HcclResult HcclRemoteRead(StreamHandle streamHandle, HcclMemTransport memTransport, HcclBuf* locBuf, HcclBuf* rmtBuf)
{
    CHK_PTR_NULL(streamHandle);
    CHK_PTR_NULL(memTransport);
    CHK_PTR_NULL(locBuf);
    CHK_PTR_NULL(rmtBuf);
    HCCL_DEBUG(
        "[HcclRemoteRead]streamHandle[%p], memTransport[%p], locBuf addr[%p], rmtBuf addr[%p], len[%llu].",
        streamHandle, memTransport, locBuf->addr, rmtBuf->addr, rmtBuf->len);
    Stream* stream = reinterpret_cast<Stream*>(streamHandle);

    struct Transport::Buffer localBuf(locBuf->addr, locBuf->len);
    struct Transport::Buffer remoteBuf(rmtBuf->addr, rmtBuf->len);
    return reinterpret_cast<Transport*>(memTransport)->ReadAsync(localBuf, remoteBuf, *stream);
}

HcclResult HcclRemoteWriteReduce(
    StreamHandle streamHandle, HcclMemTransport memTransport, HcclBuf* rmtBuf, HcclBuf* locBuf,
    HcclReduceInfo reduceInfo)
{
    CHK_PTR_NULL(streamHandle);
    CHK_PTR_NULL(memTransport);
    CHK_PTR_NULL(rmtBuf);
    CHK_PTR_NULL(locBuf);
    HCCL_DEBUG(
        "[HcclRemoteWriteReduce]streamHandle[%p], memTransport[%p], locBuf addr[%p], rmtBuf addr[%p], len[%llu],"
        " dataType[%d], reduceOp[%d].",
        streamHandle, memTransport, locBuf->addr, rmtBuf->addr, rmtBuf->len, reduceInfo.dataType, reduceInfo.reduceOp);
    Stream* stream = reinterpret_cast<Stream*>(streamHandle);
    struct Transport::Buffer localBuf(locBuf->addr, locBuf->len);
    struct Transport::Buffer remoteBuf(rmtBuf->addr, rmtBuf->len);
    return reinterpret_cast<Transport*>(memTransport)
        ->WriteReduceAsync(remoteBuf, localBuf, reduceInfo.dataType, reduceInfo.reduceOp, *stream);
}

HcclResult HcclRemoteReadReduce(
    StreamHandle streamHandle, HcclMemTransport memTransport, HcclBuf* locBuf, HcclBuf* rmtBuf,
    HcclReduceInfo reduceInfo)
{
    CHK_PTR_NULL(streamHandle);
    CHK_PTR_NULL(memTransport);
    CHK_PTR_NULL(locBuf);
    CHK_PTR_NULL(rmtBuf);
    HCCL_DEBUG(
        "[HcclRemoteReadReduce]streamHandle[%p], memTransport[%p], locBuf addr[%p], rmtBuf addr[%p], len[%llu],"
        " dataType[%d], reduceOp[%d].",
        streamHandle, memTransport, locBuf->addr, rmtBuf->addr, rmtBuf->len, reduceInfo.dataType, reduceInfo.reduceOp);

    // 后续使用transport，p2p支持，rdma不支持
    if (reinterpret_cast<Transport*>(memTransport)->GetLinkType() == LinkType::LINK_ROCE) {
        HCCL_ERROR("[HcclRemoteReadReduce]ROCE is not supported.");
        return HCCL_E_NOT_SUPPORT;
    }

    DispatcherPub* dispatcherPtr = nullptr;
    CHK_RET(GetPubDispatcher(&dispatcherPtr));

    Stream* stream = reinterpret_cast<Stream*>(streamHandle);
    return dispatcherPtr->InlineReduceAsync(
        rmtBuf->addr, rmtBuf->len / SIZE_TABLE[reduceInfo.dataType], reduceInfo.dataType, reduceInfo.reduceOp, *stream,
        locBuf->addr, INVALID_VALUE_RANKID, LinkType::LINK_ONCHIP);
}

HcclResult HcclRemoteNotifyRecord(StreamHandle streamHandle, HcclMemTransport memTransport, uint32_t notifyIndex)
{
    CHK_PTR_NULL(streamHandle);
    CHK_PTR_NULL(memTransport);
    HCCL_DEBUG(
        "[HcclRemoteNotifyRecord]streamHandle[%p], memTransport[%p], notifyIndex[%d].", streamHandle, memTransport,
        notifyIndex);
    Stream* stream = reinterpret_cast<Stream*>(streamHandle);
    return reinterpret_cast<Transport*>(memTransport)->Post(notifyIndex, *stream);
}

HcclResult HcclRemoteNotifyWait(
    StreamHandle streamHandle, HcclMemTransport memTransport, uint32_t notifyIndex, const uint32_t timeOut)
{
    CHK_PTR_NULL(streamHandle);
    CHK_PTR_NULL(memTransport);
    HCCL_DEBUG(
        "[HcclRemoteNotifyWait]streamHandle[%p], memTransport[%p], notifyIndex[%u], timeOut[%u s].", streamHandle,
        memTransport, notifyIndex, timeOut);
    Stream* stream = reinterpret_cast<Stream*>(streamHandle);
    return reinterpret_cast<Transport*>(memTransport)->Wait(notifyIndex, *stream, timeOut);
}

HcclResult HcclRemoteWriteWithNotify(
    StreamHandle streamHandle, HcclMemTransport memTransport, HcclBuf* rmtBuf, HcclBuf* locBuf, uint32_t notifyIndex)
{
    CHK_PTR_NULL(streamHandle);
    CHK_PTR_NULL(memTransport);
    CHK_PTR_NULL(locBuf);
    CHK_PTR_NULL(rmtBuf);
    HCCL_DEBUG(
        "[HcclRemoteWriteWithNotify]streamHandle[%p], memTransport[%p], locBuf addr[%p], rmtBuf addr[%p]"
        ", len[%llu], notifyIndex[%u].",
        streamHandle, memTransport, locBuf->addr, rmtBuf->addr, rmtBuf->len, notifyIndex);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HcclRemoteWriteReduceWithNotify(
    StreamHandle streamHandle, HcclMemTransport memTransport, HcclBuf* rmtBuf, HcclBuf* locBuf,
    HcclReduceInfo reduceInfo, uint32_t notifyIndex)
{
    CHK_PTR_NULL(streamHandle);
    CHK_PTR_NULL(memTransport);
    CHK_PTR_NULL(locBuf);
    CHK_PTR_NULL(rmtBuf);
    HCCL_DEBUG(
        "[HcclRemoteWriteReduceWithNotify]streamHandle[%p], memTransport[%p], locBuf addr[%p], rmtBuf addr[%p],"
        " len[%llu], dataType[%d], reduceOp[%d], notifyIndex[%u].",
        streamHandle, memTransport, locBuf->addr, rmtBuf->addr, rmtBuf->len, reduceInfo.dataType, reduceInfo.reduceOp,
        notifyIndex);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HcclRemoteFence(StreamHandle streamHandle, HcclMemTransport memTransport, uint32_t orderFlag)
{
    CHK_PTR_NULL(streamHandle);
    CHK_PTR_NULL(memTransport);
    HCCL_DEBUG(
        "[HcclRemoteFence]streamHandle[%p], memTransport[%p], orderFlag[%u].", streamHandle, memTransport, orderFlag);
    return reinterpret_cast<Transport*>(memTransport)->Fence();
}

HcclResult HcclRemoteBatchWrite(
    StreamHandle streamHandle, HcclMemTransport memTransport, HcclBufPair* bufPairs, uint32_t bufPairNum)
{
    CHK_PTR_NULL(streamHandle);
    CHK_PTR_NULL(memTransport);
    CHK_PTR_NULL(bufPairs);
    HCCL_DEBUG(
        "[HcclRemoteBatchWrite]streamHandle[%p], memTransport[%p], bufPairNum[%u].", streamHandle, memTransport,
        bufPairNum);
    Stream* stream = reinterpret_cast<Stream*>(streamHandle);
    Transport* transport = reinterpret_cast<Transport*>(memTransport);
    for (uint32_t i = 0; i < bufPairNum; i++) {
        CHK_PTR_NULL(bufPairs[i].loc.addr);
        CHK_PTR_NULL(bufPairs[i].rmt.addr);
        struct Transport::Buffer localBuf(bufPairs[i].loc.addr, bufPairs[i].loc.len);
        struct Transport::Buffer remoteBuf(bufPairs[i].rmt.addr, bufPairs[i].rmt.len);
        CHK_RET(transport->WriteAsync(remoteBuf, localBuf, *stream));
    }

    return HCCL_SUCCESS;
}

HcclResult HcclRemoteBatchRead(
    StreamHandle streamHandle, HcclMemTransport memTransport, HcclBufPair* bufPairs, uint32_t bufPairNum)
{
    CHK_PTR_NULL(streamHandle);
    CHK_PTR_NULL(memTransport);
    CHK_PTR_NULL(bufPairs);
    CHK_PRT_RET(bufPairNum == 0, HCCL_ERROR("[HcclRemoteBatchRead]bufPairsNum is 0."), HCCL_E_PARA);
    HCCL_DEBUG(
        "[HcclRemoteBatchRead]streamHandle[%p], memTransport[%p], bufPairNum[%u].", streamHandle, memTransport,
        bufPairNum);
    Stream* stream = reinterpret_cast<Stream*>(streamHandle);
    Transport* transport = reinterpret_cast<Transport*>(memTransport);

    for (uint32_t i = 0; i < bufPairNum; i++) {
        CHK_PTR_NULL(bufPairs[i].loc.addr);
        CHK_PTR_NULL(bufPairs[i].rmt.addr);
        struct Transport::Buffer localBuf(bufPairs[i].loc.addr, bufPairs[i].loc.len);
        struct Transport::Buffer remoteBuf(bufPairs[i].rmt.addr, bufPairs[i].rmt.len);
        CHK_RET(transport->ReadAsync(localBuf, remoteBuf, *stream));
    }

    return HCCL_SUCCESS;
}

HcclResult HcclRemoteBatchTransfer(
    StreamHandle streamHandle, HcclMemTransport memTransport, const HcclBatchTransferInfo* transferInfo,
    [[maybe_unused]] uint32_t bufPairNum)
{
    CHK_PTR_NULL(streamHandle);
    CHK_PTR_NULL(memTransport);
    CHK_PTR_NULL(transferInfo);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HcclRemoteDrain(StreamHandle streamHandle, HcclMemTransport memTransport)
{
    CHK_PTR_NULL(streamHandle);
    CHK_PTR_NULL(memTransport);
    Stream* stream = reinterpret_cast<Stream*>(streamHandle);
    Transport* transport = reinterpret_cast<Transport*>(memTransport);
    return transport->Drain(*stream);
}
