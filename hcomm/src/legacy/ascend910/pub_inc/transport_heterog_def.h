/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRANSPORT_HETEROG_DEF_H
#define TRANSPORT_HETEROG_DEF_H

#include <hccl/hccl_types.h>
#include "hccl/base.h"
#include "sal_pub.h"
#include "mr_manager.h"
#include "memory_alloc_ring.h"
#include "heterog_mem_blocks_manager_pub.h"
#include <memory>

namespace hccl {
constexpr s64 DEFAULT_GLOBAL_STEP_VALUE = -1;
constexpr u32 DEFAULT_TABLE_ID_VALUE = 0;

constexpr u32 HCCL_TEST_INCOMPLETED = 0;
constexpr u32 HCCL_TEST_COMPLETED = 1;
constexpr u32 HCCL_IMPROBE_INCOMPLETED = 0;
constexpr u32 HCCL_IMPROBE_COMPLETED = 1;

using CommHandle = void*;
using TransportHandle = void*;

enum class HcclHeterogCommType : s32 { INVALID = 0, PCIE = 1, RDMA = 2 };

using MemType = enum TagMemType {
    USER_INPUT_MEM,
    USER_OUTPUT_MEM,
    DATA_NOTIFY_MEM,
    ACK_NOTIFY_MEM,
    DATA_ACK_NOTIFY_MEM,
    MULTI_QP_DATA_NOTIFY_MEM,
    NOTIFY_SRC_MEM,
    ENVELOPE_SHM_MEM,
    SEND_NOTIFY_MEM,
    RECV_NOTIFY_MEM,
    NOTIFY_VALUE_MEM,
    RESPONCE_MEM,
    RESPONCE_VALUE_MEM,
    RESPONCE_CANCEL_VALUE_MEM,
    MUILT_NOTIFY_MEM,
    AICPU_SYNC_MEM,
    MEM_TYPE_RESERVED
};

enum class HcclRequestType { HCCL_REQUEST_SEND, HCCL_REQUEST_RECV, HCCL_REQUEST_INVAIL };

using TransData = struct TransDataDef {
    u64 srcBuf;
    u64 dstBuf;
    u64 count;
    u32 dataType;
    bool errorFlag;
    u32 tableId;
    s64 globalStep;

    TransDataDef()
        : srcBuf(0),
          dstBuf(0),
          count(0),
          dataType(HCCL_DATA_TYPE_RESERVED),
          errorFlag(false),
          tableId(DEFAULT_TABLE_ID_VALUE),
          globalStep(DEFAULT_GLOBAL_STEP_VALUE)
    {}
    TransDataDef(
        u64 srcBuf, u64 dstBuf, u64 count, HcclDataType dataType, bool errorFlag = false,
        u32 tableId = DEFAULT_TABLE_ID_VALUE, s64 globalStep = DEFAULT_GLOBAL_STEP_VALUE)
        : srcBuf(srcBuf),
          dstBuf(dstBuf),
          count(count),
          dataType(dataType),
          errorFlag(errorFlag),
          tableId(tableId),
          globalStep(globalStep)
    {}
};

using TransportEndPointInfo = struct TransportEndPointInfoDef {
    u32 commId; // 该rank所在通信域的通信域ID
    u32 rank;   // 该rank所在通信域内的user rank
    s32 tag;    // 通信使用的user tag

    TransportEndPointInfoDef() : commId(0), rank(INVALID_VALUE_RANKID), tag(-1) {}
    TransportEndPointInfoDef(u32 commId, u32 rank, s32 tag) : commId(commId), rank(rank), tag(tag) {}
    bool operator==(const TransportEndPointInfoDef& that) const
    {
        return ((this->commId == that.commId) && (this->rank == that.rank) && (this->tag == that.tag));
    }
};

using TransportEndPointParam = struct TransportEndPointParamDef {
    TransportEndPointInfo src;
    TransportEndPointInfo dst;

    TransportEndPointParamDef() : src(), dst() {}
    TransportEndPointParamDef(TransportEndPointInfo& src, TransportEndPointInfo& dst) : src(src), dst(dst) {}
};

using TransportRequestInfo = struct TransportRequestInfoDef {
    TransData transData;
    TransportEndPointParam epParam;
    HcclRequestType requestType;
    u8 protocol; // rendezvous:0; eager:1
    u64 msn;
    s32 status;
    u64 envoffset;
    u64 tranoffset;

    TransportRequestInfoDef()
        : requestType(HcclRequestType::HCCL_REQUEST_INVAIL),
          protocol(0),
          msn(0),
          status(-1),
          envoffset(0),
          tranoffset(0)
    {}
};

struct HcclRequestInfo {
    s32 tag;
    CommHandle commHandle;
    TransportHandle transportHandle;
    TransportRequestInfo transportRequest;
    HcclRequestInfo* next;
    HcclRequestInfo() : tag(INVALID_INT), commHandle(nullptr), transportHandle(nullptr) {}
};

using HcclUserRequire = struct HcclUserRequireDef {
    u32 tableId;
    s64 globalStep;
    HcclUserRequireDef() : tableId(DEFAULT_TABLE_ID_VALUE), globalStep(DEFAULT_GLOBAL_STEP_VALUE) {}
    HcclUserRequireDef(u32 tableId, s64 globalStep = DEFAULT_GLOBAL_STEP_VALUE)
        : tableId(tableId),
          globalStep(globalStep)
    {}
};

using HcclEnvelope = struct HcclEnvelopeDef {
    u8 protocol; // rendezvous:0; eager:1
    TransData transData;
    TransportEndPointParam epParam;
    u32 key;    // RDMA Read用的Key
    u64 msn;    // 消息序列号
    u64 rsv[4]; // 临时驱动问题，保证128字节对齐，待驱动问题上线后，删除此代码。
    HcclEnvelopeDef() : protocol(0), key(0), msn(0) {}
    HcclEnvelopeDef(u8 protocol, TransData& transData, TransportEndPointParam& epParam, u32 key, u64 msn)
        : protocol(protocol),
          transData(transData),
          epParam(epParam),
          key(key),
          msn(msn)
    {}
};

struct HcclEsRdmaInfoForLookup {
    s32 errorStatus{};
    HcclHeterogCommType commType{};
    HcclEnvelope envelope{};
};

struct HcclEsRdmaInfoForUpdate : public HcclEsRdmaInfoForLookup {
    HcclEnvelope envelopeValue{};
};

using HcclEsRdmaInfo = HcclEsRdmaInfoForUpdate;

using HcclEnvelopePcie = struct HcclEnvelopePcieDef {
    MemType memType;
    u64 offset;
    u64 count;
    u32 dataType;
    bool updateEndFlag; // embedding service update flag
    u32 tableId;
    s64 globalStep;

    HcclEnvelopePcieDef()
        : memType(USER_INPUT_MEM),
          offset(0),
          count(0),
          dataType(HCCL_DATA_TYPE_RESERVED),
          updateEndFlag(true),
          tableId(DEFAULT_TABLE_ID_VALUE),
          globalStep(DEFAULT_GLOBAL_STEP_VALUE)
    {}
    HcclEnvelopePcieDef(
        MemType memType, u64 offset, u64 count, u32 dataType, bool updateEndFlag = false,
        u32 tableId = DEFAULT_TABLE_ID_VALUE, s64 globalStep = DEFAULT_GLOBAL_STEP_VALUE)
        : memType(memType),
          offset(offset),
          count(count),
          dataType(dataType),
          updateEndFlag(updateEndFlag),
          tableId(tableId),
          globalStep(globalStep)
    {}
};

using HcclEnvelopeSummary = struct HcclEnvelopeSummaryDef {
    HcclEnvelope envelope;
    HcclEnvelopePcie pcieEnvelope;
    s32 status;
    HcclEnvelopeSummaryDef() : envelope(), status(0) {}
    HcclEnvelopeSummaryDef(HcclEnvelope& envelope, s32 status) : envelope(envelope), status(status) {}
};

using HcclMessageInfo = struct HcclMessageInfoDef {
    CommHandle commHandle;
    TransportHandle transportHandle;
    HcclEnvelopeSummary envelope;
    HcclMessageInfoDef() : commHandle(nullptr), transportHandle(nullptr) {}
};

struct RecvWrInfo {
    TransportHandle transportHandle = nullptr;
    void* buf = nullptr;
};
} // namespace hccl
#endif
