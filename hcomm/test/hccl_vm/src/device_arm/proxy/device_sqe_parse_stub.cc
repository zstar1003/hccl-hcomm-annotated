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
#define HCCL_VM_MODULE "DEVICE"

#include "device_sqe_parse_stub.h"

#include <cstdint>
#include <cstring>
#include <map>
#include <set>
#include <unordered_map>

#include "hccl_task_collection.h" // 临时新增的，后续删除
#include "sim_log.h"
#include "sqe_v82_stub.h"
#include "udma_data_struct_stub.h"

constexpr int SHIFT_BIT32 = 32;
uint32_t curRankId = 0;

void ParseA5SqeFromSqBuffer(uint32_t devId, struct halSqCqConfigInfo* info)
{
    // devId会在不同Server内重复，此处使用Device进程维护的rankId
    GetCurRankId(&curRankId);
    uint32_t streamId = info->sqId;
    int tail = info->value[0];
    int head = GetSqTail(streamId);
    UpdateSqTail(streamId, tail);

    uint8_t* sqBuffer = nullptr;
    GetSqBufferAddr(&sqBuffer);

    // 计算本轮下发的SQE的数量
    int sqeCnt = (HCCL_SQE_MAX_CNT + tail - head) % HCCL_SQE_MAX_CNT;
    //  临时缓冲区，用于存放从sqBuffer拷贝的数据
    uint8_t tempBuffer[HCCL_SQE_SIZE * HCCL_SQE_MAX_CNT];
    if (tail >= head) {
        // 数据未绕圈，直接拷贝
        memcpy(tempBuffer, sqBuffer + head * HCCL_SQE_SIZE, sqeCnt * HCCL_SQE_SIZE);
    } else {
        // 数据绕圈，分两次拷贝
        int firstPart = HCCL_SQE_MAX_CNT - head; // 从head到缓冲区  末尾的拷贝数量
        int secondPart = tail;                   // 从缓冲区开头到tail的拷贝数量
        memcpy(tempBuffer, sqBuffer + head * HCCL_SQE_SIZE, firstPart * HCCL_SQE_SIZE);
        memcpy(tempBuffer + firstPart * HCCL_SQE_SIZE, sqBuffer, secondPart * HCCL_SQE_SIZE);
    }

    for (int i = 0; i < sqeCnt; i++) {
        int sqOffIndex = i;
        void* sqeBuf = static_cast<uint8_t*>(tempBuffer) + sqOffIndex * HCCL_SQE_SIZE;
        Rt91095StarsSqeHeader* header = reinterpret_cast<Rt91095StarsSqeHeader*>(sqeBuf);
        switch (header->type) {
            case static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA): {
                ParseDavidSDMASqe(streamId, sqeBuf);
                break;
            }
            case static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT): {
                ParseDavidNotifySqe(streamId, sqeBuf, false);
                break;
            }
            case static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD): {
                ParseDavidNotifySqe(streamId, sqeBuf, true);
                break;
            }
            case static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_UBDMA): {
                ParseDavidUDMASqe(streamId, sqeBuf);
                break;
            }
            default: {
                HCCL_VM_ERROR("not support sqe type[{}].", static_cast<uint32_t>(header->type));
                break;
            }
        }
    }
}

void ParseDavidSDMASqe(uint32_t streamId, void* sqeBuf)
{
    Rt91095StarsMemcpySqe* sqe = reinterpret_cast<Rt91095StarsMemcpySqe*>(sqeBuf);
    HcclTaskMetaData taskMeta;
    taskMeta.streamId = streamId;
    taskMeta.rankId = curRankId;
    taskMeta.jettyId = UINT32_MAX;

    uint64_t length = sqe->u.strideMode0.lengthMove;
    uint64_t srcOffset
        = TransLocalAddrToVirtual(GetFull64BitAddr(sqe->u.strideMode0.srcAddrLow, sqe->u.strideMode0.srcAddrHigh));
    uint64_t dstOffset
        = TransLocalAddrToVirtual(GetFull64BitAddr(sqe->u.strideMode0.dstAddrLow, sqe->u.strideMode0.dstAddrHigh));
    uint64_t srcRankId = GetRankIdByDevAddr(srcOffset);
    uint64_t dstRankId = GetRankIdByDevAddr(dstOffset);

    if (sqe->opcode == 0) { // 表示是memcpy
        taskMeta.taskType = HccLTaskMetaType::MEM_CPY;
        taskMeta.taskData.transMem.srcOffset = srcOffset;
        taskMeta.taskData.transMem.dstOffset = dstOffset;
        taskMeta.taskData.transMem.dstRankId = dstRankId;
        taskMeta.taskData.transMem.srcRankId = srcRankId;
        taskMeta.taskData.transMem.len = length;
        InsertTaskToCollectionDev(&taskMeta);
        PrintTaskMetaData(taskMeta);
    } else { // reduce
        taskMeta.taskType = HccLTaskMetaType::REDUCE;
        taskMeta.taskData.reduce.reduceOp = ParseReduceTypeDavid(sqe->opcode);
        taskMeta.taskData.reduce.dataType = ParseDataTypeDavid(sqe->opcode);
        taskMeta.taskData.reduce.srcOffset = srcOffset;
        taskMeta.taskData.reduce.dstOffset = dstOffset;
        taskMeta.taskData.reduce.dstRankId = dstRankId;
        taskMeta.taskData.reduce.srcRankId = srcRankId;
        taskMeta.taskData.reduce.dataCount = length;
        InsertTaskToCollectionDev(&taskMeta);
        PrintTaskMetaData(taskMeta);
    }
}

void ParseDavidNotifySqe(uint32_t streamId, void* sqeBuf, bool isPost)
{
    Rt91095StarsNotifySqe* sqe = reinterpret_cast<Rt91095StarsNotifySqe*>(sqeBuf);
    HcclTaskMetaData taskMeta;
    taskMeta.streamId = streamId;
    taskMeta.rankId = curRankId;
    taskMeta.jettyId = UINT32_MAX;
    taskMeta.taskType = isPost ? HccLTaskMetaType::NOTIFY_RECORD : HccLTaskMetaType::NOTIFY_WAIT;
    taskMeta.taskData.notify.notifyId = sqe->notifyId;
    taskMeta.taskData.notify.srcRankId = curRankId;
    taskMeta.taskData.notify.dstRankId = 0; // 目前暂不区分notify的目的rank，默认为0
    PrintTaskMetaData(taskMeta);
    InsertTaskToCollectionDev(&taskMeta);
}

uint32_t CalculateCiValue(uint64_t wqeBuffer, uint32_t piValue)
{
    uint32_t cqeCnt = 0;
    uint32_t prePiVal = (piValue >= 2) ? piValue - 2 : 0; // 先向前查找两个WQE
    uint64_t wqeAddr = wqeBuffer + prePiVal * HCCL_WQE_SIZE;
    UdmaSqeCommon* ubCommon = reinterpret_cast<UdmaSqeCommon*>(wqeAddr);
    while (true) {
        if (ubCommon->opcode != UdmaSqOpcode::UDMA_OPC_WRITE_WITH_NOTIFY) {
            prePiVal = piValue - 1;
            wqeAddr = wqeBuffer + prePiVal * HCCL_WQE_SIZE;
            ubCommon = reinterpret_cast<UdmaSqeCommon*>(wqeAddr);
        }

        piValue = prePiVal;

        // 得到前一个真实的WQE(64字节或128字节)
        if (ubCommon->cqe == 1) {
            cqeCnt++;
        }

        // 找到了前一次dolBell的结束下标
        if (cqeCnt == 2) {
            if (ubCommon->opcode != UdmaSqOpcode::UDMA_OPC_WRITE_WITH_NOTIFY) {
                return prePiVal + 1;
            }
            return prePiVal + 2;
        }

        // 找到wqeBuffer的最头部了，直接返回
        if (prePiVal == 0) {
            return prePiVal;
        }

        // 继续向前查找两个WQE
        prePiVal = (piValue >= 2) ? piValue - 2 : 0;
        wqeAddr = wqeBuffer + prePiVal * HCCL_WQE_SIZE;
        ubCommon = reinterpret_cast<UdmaSqeCommon*>(wqeAddr);
    }

    HCCL_VM_ERROR("calculate ciValue failed.");
    return 0;
}

void ParseDavidUDMASqe(uint32_t streamId, void* sqeBuf)
{
    Rt91095StarsUbdmaDBmodeSqe* ubSqe = reinterpret_cast<Rt91095StarsUbdmaDBmodeSqe*>(sqeBuf);
    uint32_t jettyId = ubSqe->jettyId1;
    uint64_t wqeBuffer = 0;
    if (!GetWqebufferByJettyId(jettyId, wqeBuffer)) {
        HCCL_VM_ERROR("GetWqebufferByJettyId failed, jettyId[{}].", jettyId);
        return;
    }

    uint32_t piValue = ubSqe->piValue1;
    uint32_t ciValue = CalculateCiValue(wqeBuffer, ubSqe->piValue1);
    if (ciValue >= piValue) {
        HCCL_VM_ERROR("jettyId[{}] ciVal[{}] piValue[{}] streamId[{}].", jettyId, ciValue, piValue, streamId);
        return;
    }

    HCCL_VM_INFO("jettyId[{}] ciVal[{}] piValue[{}] streamId[{}].", jettyId, ciValue, piValue, streamId);
    for (auto index = ciValue; index < piValue; index++) {
        uint64_t wqeAddr = wqeBuffer + index * HCCL_WQE_SIZE;
        UdmaSqeCommon* ubCommon = reinterpret_cast<UdmaSqeCommon*>(wqeAddr);
        switch (ubCommon->opcode) {
            case static_cast<int>(UdmaSqOpcode::UDMA_OPC_WRITE): {
                ParseDavidUBReadWriteSqe(wqeAddr, streamId, jettyId, false);
                break;
            }
            case static_cast<int>(UdmaSqOpcode::UDMA_OPC_WRITE_WITH_NOTIFY): {
                ParseDavidUBWriteWithNotifySqe(wqeAddr, streamId, jettyId);
                index++; // 占128字节两个WQE
                break;
            }
            case static_cast<int>(UdmaSqOpcode::UDMA_OPC_READ): {
                ParseDavidUBReadWriteSqe(wqeAddr, streamId, jettyId, true);
                break;
            }
            default: {
                HCCL_VM_ERROR("not support opcode[{}].", static_cast<uint32_t>(ubCommon->opcode));
                return;
            }
        }
    }
}

void ParseDavidUBReadWriteSqe(uint64_t wqeAddr, uint16_t streamId, uint32_t jettyId, bool isRead)
{
    UdmaSqeWrite* ubWqe = reinterpret_cast<UdmaSqeWrite*>(wqeAddr);
    HcclTaskMetaData taskMeta;
    taskMeta.streamId = streamId;
    taskMeta.rankId = curRankId;
    taskMeta.jettyId = jettyId;
    memcpy(taskMeta.rmEid, ubWqe->comm.rmtEid, 16);

    uint32_t rmtRankId = GetRmtRankIdByEid(ubWqe->comm.rmtEid[0]);
    // case1:UbConnLite::InlineWrite 写Notify
    if (ubWqe->comm.inlineEn == 1) {
        uint64_t notifyAddr = GetFull64BitAddr(ubWqe->comm.rmtAddrLow, ubWqe->comm.rmtAddrHigh);
        taskMeta.taskType = HccLTaskMetaType::NOTIFY_RECORD;
        taskMeta.taskData.notify.notifyId = notifyAddr;
        taskMeta.taskData.notify.srcRankId = curRankId;
        taskMeta.taskData.notify.dstRankId = rmtRankId;
        PrintTaskMetaData(taskMeta);
        InsertTaskToCollectionDev(&taskMeta);
        return;
    }

    // case2:UbConnLite::Write Copy
    uint64_t length = static_cast<uint64_t>(ubWqe->u.sge.length);
    uint64_t locAddr = GetFull64BitAddr(ubWqe->u.sge.dataAddrLow, ubWqe->u.sge.dataAddrHigh);
    uint64_t rmtAddr = GetFull64BitAddr(ubWqe->comm.rmtAddrLow, ubWqe->comm.rmtAddrHigh);
    locAddr = TransLocalAddrToVirtual(locAddr);
    rmtAddr = TransRemoteAddrToVirtualByRank(rmtAddr, rmtRankId);
    uint64_t srcOffset = isRead ? rmtAddr : locAddr;
    uint64_t dstOffset = isRead ? locAddr : rmtAddr;
    uint32_t srcRankId = isRead ? rmtRankId : curRankId;
    uint32_t dstRankId = isRead ? curRankId : rmtRankId;
    taskMeta.taskType = HccLTaskMetaType::MEM_CPY;
    taskMeta.taskData.transMem.srcOffset = srcOffset;
    taskMeta.taskData.transMem.dstOffset = dstOffset;
    taskMeta.taskData.transMem.dstRankId = dstRankId;
    taskMeta.taskData.transMem.srcRankId = srcRankId;
    taskMeta.taskData.transMem.len = length;

    // case3:UbConnLite::WriteReduce Reduce
    if (ubWqe->comm.udfFlag == 1) {
        taskMeta.taskType = HccLTaskMetaType::REDUCE;
        taskMeta.taskData.reduce.reduceOp = ParseUbReduceTypeDavid(ubWqe->comm.inlinedata.udfData.reduceOp);
        taskMeta.taskData.reduce.dataType = ParseUbDataTypeDavid(ubWqe->comm.inlinedata.udfData.reduceType);
        taskMeta.taskData.reduce.srcOffset = srcOffset;
        taskMeta.taskData.reduce.dstOffset = dstOffset;
        taskMeta.taskData.reduce.dstRankId = dstRankId;
        taskMeta.taskData.reduce.srcRankId = srcRankId;
        taskMeta.taskData.reduce.dataCount = length;
    }

    PrintTaskMetaData(taskMeta);
    InsertTaskToCollectionDev(&taskMeta);
}

void ParseDavidUBWriteWithNotifySqe(uint64_t wqeAddr, uint16_t streamId, uint32_t jettyId)
{
    UdmaSqeWriteWithNotify* ubWqe = reinterpret_cast<UdmaSqeWriteWithNotify*>(wqeAddr);
    HcclTaskMetaData taskMeta1;
    taskMeta1.streamId = streamId;
    taskMeta1.rankId = curRankId;
    taskMeta1.jettyId = jettyId;
    memcpy(taskMeta1.rmEid, ubWqe->comm.rmtEid, 16);
    uint32_t rmtRankId = GetRmtRankIdByEid(ubWqe->comm.rmtEid[0]);

    // 1.先构造MEM_CPY(或SDMA_REDUCE)所需参数
    uint64_t dstOffset
        = TransRemoteAddrToVirtualByRank(GetFull64BitAddr(ubWqe->comm.rmtAddrLow, ubWqe->comm.rmtAddrHigh), rmtRankId);
    uint64_t srcOffset
        = TransLocalAddrToVirtual(GetFull64BitAddr(ubWqe->localU.sge.dataAddrLow, ubWqe->localU.sge.dataAddrHigh));
    uint64_t length = static_cast<uint64_t>(ubWqe->localU.sge.length);
    uint32_t srcRankId = GetRankIdByDevAddr(srcOffset);
    uint32_t dstRankId = GetRankIdByDevAddr(dstOffset);
    taskMeta1.taskType = HccLTaskMetaType::MEM_CPY;
    taskMeta1.taskData.transMem.srcOffset = srcOffset;
    taskMeta1.taskData.transMem.dstOffset = dstOffset;
    taskMeta1.taskData.transMem.dstRankId = dstRankId;
    taskMeta1.taskData.transMem.srcRankId = srcRankId;
    taskMeta1.taskData.transMem.len = length;
    if (ubWqe->comm.udfFlag == 1) { // Reduce操作标识
        taskMeta1.taskType = HccLTaskMetaType::REDUCE;
        taskMeta1.taskData.reduce.reduceOp = ParseUbReduceTypeDavid(ubWqe->comm.inlinedata.udfData.reduceOp);
        taskMeta1.taskData.reduce.dataType = ParseUbDataTypeDavid(ubWqe->comm.inlinedata.udfData.reduceType);
        taskMeta1.taskData.reduce.srcOffset = srcOffset;
        taskMeta1.taskData.reduce.dstOffset = dstOffset;
        taskMeta1.taskData.reduce.dstRankId = dstRankId;
        taskMeta1.taskData.reduce.srcRankId = srcRankId;
        taskMeta1.taskData.reduce.dataCount = length;
    }
    PrintTaskMetaData(taskMeta1);
    InsertTaskToCollectionDev(&taskMeta1);

    // 2.再构造NOTIFY_RECORD所需参数
    HcclTaskMetaData taskMeta2;
    taskMeta2.streamId = streamId;
    taskMeta2.rankId = curRankId;
    taskMeta2.jettyId = jettyId;
    memcpy(taskMeta2.rmEid, ubWqe->comm.rmtEid, 16);

    uint64_t notifyAddr = GetFull64BitAddr(ubWqe->notify.notifyAddrLow, ubWqe->notify.notifyAddrHigh);
    taskMeta2.taskType = HccLTaskMetaType::NOTIFY_RECORD;
    taskMeta2.taskData.notify.notifyId = notifyAddr;
    taskMeta2.taskData.notify.srcRankId = curRankId;
    taskMeta2.taskData.notify.dstRankId = rmtRankId;
    PrintTaskMetaData(taskMeta2);
    InsertTaskToCollectionDev(&taskMeta2);
}

uint64_t GetFull64BitAddr(uint32_t lowAddr, uint32_t highAddr)
{
    return (static_cast<uint64_t>(highAddr) << SHIFT_BIT32) | static_cast<uint64_t>(lowAddr);
}

std::map<uint32_t, HcclReduceOp> DavidReduceOpMap
    = {{0x01, HcclReduceOp::HCCL_REDUCE_SUM},
       {0x02, HcclReduceOp::HCCL_REDUCE_MAX},
       {0x03, HcclReduceOp::HCCL_REDUCE_MIN}};

HcclReduceOp ParseReduceTypeDavid(uint8_t result)
{
    uint8_t reduceType = result & 0x0F; // 直接与 0x0F 按位与，保留低 4 位
    if (DavidReduceOpMap.find(static_cast<uint32_t>(reduceType)) != DavidReduceOpMap.end()) {
        return DavidReduceOpMap[reduceType];
    }

    HCCL_VM_ERROR("not support reduceType[{}].", reduceType);
    return HcclReduceOp::HCCL_REDUCE_RESERVED;
}

std::map<uint32_t, HcclDataType> DavidDataTypeMap
    = {{0x00, HcclDataType::HCCL_DATA_TYPE_INT8},   {0x10, HcclDataType::HCCL_DATA_TYPE_INT16},
       {0x20, HcclDataType::HCCL_DATA_TYPE_INT32},  {0x30, HcclDataType::HCCL_DATA_TYPE_UINT8},
       {0x40, HcclDataType::HCCL_DATA_TYPE_UINT16}, {0x50, HcclDataType::HCCL_DATA_TYPE_UINT32},
       {0x60, HcclDataType::HCCL_DATA_TYPE_FP16},   {0x70, HcclDataType::HCCL_DATA_TYPE_FP32},
       {0x80, HcclDataType::HCCL_DATA_TYPE_BFP16}};

HcclDataType ParseDataTypeDavid(uint8_t result)
{
    uint8_t dataType = static_cast<uint32_t>(result) & 0xF0; // 提取高4位
    if (DavidDataTypeMap.find(static_cast<uint32_t>(dataType)) != DavidDataTypeMap.end()) {
        return DavidDataTypeMap[dataType];
    }

    HCCL_VM_ERROR("not support dataType[{}].", dataType);
    return HcclDataType::HCCL_DATA_TYPE_RESERVED;
}

std::map<uint32_t, HcclReduceOp> DavidUbReduceOpMap = {
    {0xA, HcclReduceOp::HCCL_REDUCE_SUM}, {0x8, HcclReduceOp::HCCL_REDUCE_MAX}, {0x9, HcclReduceOp::HCCL_REDUCE_MIN}};

HcclReduceOp ParseUbReduceTypeDavid(uint32_t type)
{
    if (DavidUbReduceOpMap.find(type) != DavidUbReduceOpMap.end()) {
        return DavidUbReduceOpMap[type];
    }

    HCCL_VM_ERROR("not support type[{}].", type);
    return HcclReduceOp::HCCL_REDUCE_RESERVED;
}

std::map<uint32_t, HcclDataType> DavidUbDataTypeMap
    = {{0x0, HcclDataType::HCCL_DATA_TYPE_INT8},   {0x1, HcclDataType::HCCL_DATA_TYPE_INT16},
       {0x2, HcclDataType::HCCL_DATA_TYPE_INT32},  {0x3, HcclDataType::HCCL_DATA_TYPE_UINT8},
       {0x4, HcclDataType::HCCL_DATA_TYPE_UINT16}, {0x5, HcclDataType::HCCL_DATA_TYPE_UINT32},
       {0x6, HcclDataType::HCCL_DATA_TYPE_FP16},   {0x7, HcclDataType::HCCL_DATA_TYPE_FP32},
       {0x8, HcclDataType::HCCL_DATA_TYPE_BFP16}};

HcclDataType ParseUbDataTypeDavid(uint32_t type)
{
    if (DavidUbDataTypeMap.find(type) != DavidUbDataTypeMap.end()) {
        return DavidUbDataTypeMap[type];
    }

    HCCL_VM_ERROR("not support type[{}].", type);
    return HcclDataType::HCCL_DATA_TYPE_RESERVED;
}

void PrintTaskMetaData(const HcclTaskMetaData& taskMeta)
{
    pid_t pid = getpid();
    switch (taskMeta.taskType) {
        case HccLTaskMetaType::MEM_CPY:
            HCCL_VM_INFO(
                "pid{}: rankId[{}] streamId[{}] taskType[MEM_CPY], srcOffset[{}] dstOffset[{}], len[{}], "
                "srcRankId[{}], dstRankId[{}]",
                pid, taskMeta.rankId, taskMeta.streamId, taskMeta.taskData.transMem.srcOffset,
                taskMeta.taskData.transMem.dstOffset, taskMeta.taskData.transMem.len,
                taskMeta.taskData.transMem.srcRankId, taskMeta.taskData.transMem.dstRankId);
            break;
        case HccLTaskMetaType::REDUCE:
            HCCL_VM_INFO(
                "pid{}: rankId[{}] streamId[{}] taskType[REDUCE], srcOffset[{}] dstOffset[{}], len[{}], srcRankId[{}], "
                "dstRankId[{}], reduceOp[%{}], dataType[%{}]",
                pid, taskMeta.rankId, taskMeta.streamId, taskMeta.taskData.reduce.srcOffset,
                taskMeta.taskData.reduce.dstOffset, taskMeta.taskData.reduce.dataCount,
                taskMeta.taskData.reduce.srcRankId, taskMeta.taskData.reduce.dstRankId,
                taskMeta.taskData.reduce.reduceOp, taskMeta.taskData.reduce.dataType);
            break;
        case HccLTaskMetaType::NOTIFY_WAIT:
            HCCL_VM_INFO(
                "pid{}: rankId[{}] streamId[{}] taskType[NOTIFY_WAIT], notifyId[{}], srcRankId[{}], dstRankId[{}]", pid,
                taskMeta.rankId, taskMeta.streamId, taskMeta.taskData.notify.notifyId,
                taskMeta.taskData.notify.srcRankId, taskMeta.taskData.notify.dstRankId);
            break;
        case HccLTaskMetaType::NOTIFY_RECORD:
            HCCL_VM_INFO(
                "pid{}: rankId[{}] streamId[{}] taskType[NOTIFY_RECORD], notifyId[{}], srcRankId[{}], dstRankId[{}]",
                pid, taskMeta.rankId, taskMeta.streamId, taskMeta.taskData.notify.notifyId,
                taskMeta.taskData.notify.srcRankId, taskMeta.taskData.notify.dstRankId);
            break;
        default:
            break;
    }
}

uint32_t GetRmtRankIdByEid(uint32_t eid)
{
    std::string ip = std::to_string((eid >> 24) & 0xFF) + "." + std::to_string((eid >> 16) & 0xFF) + "."
                     + std::to_string((eid >> 8) & 0xFF) + "." + std::to_string(eid & 0xFF);
    HCCL_VM_INFO("eid[{}] convert to ip[{}].", eid, ip.c_str());
    return GetRankIdByIpAddr(ip);
}
