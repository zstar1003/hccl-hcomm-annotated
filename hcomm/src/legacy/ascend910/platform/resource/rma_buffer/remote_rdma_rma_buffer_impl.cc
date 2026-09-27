/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "remote_rdma_rma_buffer_impl.h"

namespace hccl {
RemoteRdmaRmaBufferImpl::RemoteRdmaRmaBufferImpl()
    : RmaBuffer(nullptr, nullptr, 0, RmaMemType::TYPE_NUM, RmaType::RDMA_RMA)
{}

RemoteRdmaRmaBufferImpl::~RemoteRdmaRmaBufferImpl() {}

HcclResult RemoteRdmaRmaBufferImpl::Deserialize(const std::string& msg)
{
    std::istringstream iss(msg);
    iss.read(reinterpret_cast<char_t*>(&rkey), sizeof(rkey));
    HCCL_DEBUG("[RemoteRdmaRmaBufferImpl][Deserialize]rkey[%u]", rkey);
    return HCCL_SUCCESS;
}
} // namespace hccl
