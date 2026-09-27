/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mem_transport_lite.h"
#include "ub_transport_lite_impl.h"
#include "p2p_transport_lite_impl.h"
#include "binary_stream.h"
#include "not_support_exception.h"
#include "internal_exception.h"
#include "exception_util.h"
namespace Hccl {

MemTransportLite::MemTransportLite(
    std::vector<char>& uniqueId, std::function<void(u32 streamId, u32 taskId, const TaskParam& taskParam)> callback)
{
    BinaryStream binaryStream(uniqueId);
    // 反序列化，得到type，基于type创建不同的 impl
    u32 theType;
    binaryStream >> theType;
    if (theType >= static_cast<u32>(TransportType::INVALID)) {
        THROW<InternalException>(StringFormat("type %u is error", theType));
    }

    type = static_cast<TransportType::Value>(theType);

    if (type == TransportType::UB) {
        impl = std::make_unique<UbTransportLiteImpl>(uniqueId, callback);
    } else if (type == TransportType::P2P) {
        impl = std::make_unique<P2PTransportLiteImpl>(uniqueId, callback);
    } else {
        THROW<NotSupportException>(StringFormat("%s doesnot support now", type.Describe().c_str()));
    }
}

std::string MemTransportLite::Describe() const
{
    return StringFormat("MemTransportLite[type=%s]", type.Describe().c_str());
}

} // namespace Hccl
