/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRANSPORT_VIRTURAL_PUB_H
#define TRANSPORT_VIRTURAL_PUB_H

#include "transport_base_pub.h"
namespace hccl {
class TransportVirtural : public TransportBase {
public:
    explicit TransportVirtural(
        DispatcherPub* dispatcher, const std::unique_ptr<NotifyPool>& notifyPool, MachinePara& machinePara,
        std::chrono::milliseconds timeout, u32 index);
    ~TransportVirtural() override;

    HcclResult TxAck(Stream& stream) override;
    HcclResult RxAck(Stream& stream) override;
    HcclResult TxAsync(std::vector<TxMemoryInfo>& txMems, Stream& stream) override;
    HcclResult RxAsync(std::vector<RxMemoryInfo>& rxMems, Stream& stream) override;
    HcclResult TxDataSignal(Stream& stream) override;
    HcclResult RxDataSignal(Stream& stream) override;

    HcclResult TxPrepare(Stream& stream) override;
    HcclResult RxPrepare(Stream& stream) override;

    HcclResult TxData(UserMemType dstMemType, u64 dstOffset, const void* src, u64 len, Stream& stream) override;
    HcclResult RxData(UserMemType srcMemType, u64 srcOffset, void* dst, u64 len, Stream& stream) override;

    HcclResult TxDone(Stream& stream) override;
    HcclResult RxDone(Stream& stream) override;

protected:
    u32 currentIndex_; /* vTransport维护的index */
};
} // namespace hccl
#endif /* TRANSPORT_VIRTURAL_PUB_H */
