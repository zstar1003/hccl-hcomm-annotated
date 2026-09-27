/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRANSPORT_MEM_H
#define TRANSPORT_MEM_H

#include <hccl/hccl_types.h>
#include <atomic>
#include "dispatcher.h"
#include "notify_pool.h"
#include "hccl_socket.h"
#include "hccl_network_pub.h"
#include "hccl_common.h"
#include "hccl_mem.h"
#include "transport_pub.h"

namespace hccl {

enum class RmaMemType : int {
    DEVICE = 0, // device侧内存
    HOST = 1,   // host侧内存
    TYPE_NUM
};

constexpr size_t TRANSPORT_EMD_ESC_SIZE = 512U - (sizeof(u32) * 2);

class TransportMem {
public:
    enum class TpType : int { IPC = 0, ROCE = 1, ROCE_DEVICE, TYPE_NUM };

    struct AttrInfo {
        u32 localRankId{INVALID_VALUE_RANKID};
        u32 remoteRankId{INVALID_VALUE_RANKID};
        u32 sdid{INVALID_UINT};     // 本端所属超节点
        u32 serverId{INVALID_UINT}; // 本端所属server
        u32 trafficClass{HCCL_COMM_TRAFFIC_CLASS_CONFIG_NOT_SET};
        u32 serviceLevel{HCCL_COMM_SERVICE_LEVEL_CONFIG_NOT_SET};
        u32 timeout{INVALID_UINT}; // 传输超时时间
    };

    struct RmaMemDesc {
        u32 localRankId;
        u32 remoteRankId;
        char memDesc[TRANSPORT_EMD_ESC_SIZE];
    };

    struct RmaMemDescs {
        RmaMemDesc* array;
        u32 arrayLength;
    };

    struct RmaOpMem {
        void* addr;
        u64 size;
    };

    struct RmaMem {
        RmaMemType type; // segment的内存类型
        void* addr;      // segment的虚拟地址
        u64 size;        // segment的size
    };

    static std::shared_ptr<TransportMem> Create(
        TpType tpType, const std::unique_ptr<NotifyPool>& notifyPool, const HcclNetDevCtx& netDevCtx,
        const HcclDispatcher& dispatcher, AttrInfo& attrInfo);
    static std::shared_ptr<TransportMem> Create(
        TpType tpType, const std::unique_ptr<NotifyPool>& notifyPool, const HcclNetDevCtx& netDevCtx,
        const HcclDispatcher& dispatcher, AttrInfo& attrInfo, bool aicpuUnfoldMode);
    // AICPU侧创建Transport
    static std::shared_ptr<TransportMem>
    Create(TpType tpType, const HcclQpInfoV2& qpInfo, const HcclDispatcher& dispatcher, AttrInfo& attrInfo);

    explicit TransportMem(
        const std::unique_ptr<NotifyPool>& notifyPool, const HcclNetDevCtx& netDevCtx, const HcclDispatcher& dispatcher,
        AttrInfo& attrInfo);
    TransportMem(
        const std::unique_ptr<NotifyPool>& notifyPool, const HcclNetDevCtx& netDevCtx, const HcclDispatcher& dispatcher,
        AttrInfo& attrInfo, bool aicpuUnfoldMode);
    virtual ~TransportMem();
    virtual HcclResult
    ExchangeMemDesc(const RmaMemDescs& localMemDescs, RmaMemDescs& remoteMemDescs, u32& actualNumOfRemote)
        = 0;
    virtual HcclResult EnableMemAccess(const RmaMemDesc& remoteMemDesc, RmaMem& remoteMem) = 0;
    virtual HcclResult DisableMemAccess(const RmaMemDesc& remoteMemDesc) = 0;
    virtual HcclResult SetDataSocket(const std::shared_ptr<HcclSocket>& socket);

    virtual HcclResult SetSocket(const std::shared_ptr<HcclSocket>& socket) = 0;
    virtual HcclResult Connect(s32 timeoutSec) = 0;
    virtual HcclResult Write(const HcclBuf& remoteMem, const HcclBuf& localMem, const rtStream_t& stream) = 0;
    virtual HcclResult Read(const HcclBuf& localMem, const HcclBuf& remoteMem, const rtStream_t& stream) = 0;
    /**
     * @brief 旧版Write
     * @deprecated 参数优化，改用 `Write(const HcclBuf &remoteMem, const HcclBuf &localMem, const rtStream_t &stream)`。
     */
    virtual HcclResult Write(const RmaOpMem& remoteMem, const RmaOpMem& localMem, const rtStream_t& stream) = 0;
    /**
     * @brief 旧版Read
     * @deprecated 参数优化，改用 `Read(const HcclBuf &localMem, const HcclBuf &remoteMem, const rtStream_t &stream)`。
     */
    virtual HcclResult Read(const RmaOpMem& localMem, const RmaOpMem& remoteMem, const rtStream_t& stream) = 0;
    virtual HcclResult AddOpFence(const rtStream_t& stream) = 0;

    virtual HcclResult
    GetTransInfo(HcclQpInfoV2& qpInfo, u32* lkey, u32* rkey, HcclBuf* localMem, HcclBuf* remoteMem, u32 num)
        = 0;
    virtual HcclResult WaitOpFence(const rtStream_t& stream) = 0;

    // AICPU侧批量下发读、写操作，下发wr后敲Doorbell
    virtual HcclResult
    BatchWrite(const std::vector<MemDetails>& remoteMems, const std::vector<MemDetails>& localMems, Stream& stream)
        = 0;
    virtual HcclResult
    BatchRead(const std::vector<MemDetails>& localMems, const std::vector<MemDetails>& remoteMems, Stream& stream)
        = 0;
    virtual HcclResult AddOpFence(const MemDetails& localFenceMem, const MemDetails& remoteFenceMem, Stream& stream)
        = 0;

protected:
    // 从 string 拷贝到 memDesc
    HcclResult RmaMemDescCopyFromStr(RmaMemDesc& rmaMemDesc, const std::string& memDescStr) const
    {
        if (memcpy_s(rmaMemDesc.memDesc, TRANSPORT_EMD_ESC_SIZE, memDescStr.c_str(), memDescStr.size() + 1) != EOK) {
            return HCCL_E_INTERNAL;
        }
        return HCCL_SUCCESS;
    }

    // 从 memDesc 转换为 string
    std::string RmaMemDescCopyToStr(const RmaMemDesc& rmaMemDesc) const
    {
        return std::string(rmaMemDesc.memDesc, TRANSPORT_EMD_ESC_SIZE);
    }

    HcclResult DoExchangeMemDesc(const RmaMemDescs& localMemDescs, RmaMemDescs& remoteMemDescs, u32& actualNumOfRemote);
    HcclResult SendLocalMemDesc(const RmaMemDescs& localMemDescs);
    HcclResult ReceiveRemoteMemDesc(RmaMemDescs& remoteMemDescs, u32& actualNumOfRemote);

    const std::unique_ptr<NotifyPool>& notifyPool_;
    HcclNetDevCtx netDevCtx_{nullptr};
    HcclDispatcher dispatcher_{nullptr};

    u32 localRankId_{0};
    u32 remoteRankId_{0};
    std::shared_ptr<HcclSocket> socket_{nullptr};

    std::shared_ptr<HcclSocket> dataSocket_{nullptr};

    bool aicpuUnfoldMode_{false};
};
} // namespace hccl
#endif
