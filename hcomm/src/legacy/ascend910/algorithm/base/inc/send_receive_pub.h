/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SEND_RECEIVE_PUB_H
#define SEND_RECEIVE_PUB_H

#include "alg_template_base_pub.h"

namespace hccl {
class SendReceive : public AlgTemplateBase {
public:
    explicit SendReceive(
        const HcclDispatcher dispatcher, const std::shared_ptr<Transport>& link,
        const u32 peerRank = INVALID_VALUE_RANKID, const u64 chunkNum = HCCL_CHUNK_SIZE, bool retryEnable = false);

    ~SendReceive() override;

    HcclResult SendPrepare(
        const DeviceMem& inputMem,
        //  const u64 count,
        //  const HcclDataType dataType,
        const u32 destRank,
        //  const u32 tag,
        const Stream& stream);

    HcclResult ReceivePrepare(
        const DeviceMem& outputMem,
        // const u64 count,
        // const HcclDataType dataType,
        const u32 srcRank,
        // const u32 tag,
        const Stream& stream);

    HcclResult SendRunAsync();

    HcclResult ReceiveRunAsync();

    HcclResult BatchSendRunAsync();

    HcclResult BatchReceiveRunAsync();

protected:
private:
    const std::shared_ptr<Transport>& transLink_; /* 本rank到对端rank的link */

    u32 peerRank_; /** 对端的user rank */

    // DeviceMem scratchMem_;     /** 临时Devicememory */

    const u64 chunkSize_; /** 单次操作的最大buffer size(字节) */

    // u32 srTag_;               /** send receive使用的标签 */

    bool retryEnable_; /** 判断重执行场景是否开启 */
};
} // namespace hccl

#endif /* SEND_RECEIVE_PUB_H */
