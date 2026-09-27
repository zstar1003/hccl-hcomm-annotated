/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "remote_access.h"
#include <algorithm>
#include "remote_access_impl.h"

namespace hccl {
using namespace std;

RemoteAccess::RemoteAccess() {}

RemoteAccess::~RemoteAccess() { impl_ = nullptr; }

HcclResult RemoteAccess::Init(u32 rank, const vector<MemRegisterAddr>& addrInfos, const RmaRankTable& rankTable)
{
    impl_.reset(new (std::nothrow) RemoteAccessImpl());
    CHK_PTR_NULL(impl_);
    CHK_RET(impl_->Init(rank, addrInfos, rankTable));
    return HCCL_SUCCESS;
}

HcclResult RemoteAccess::RemoteRead(const vector<HcomRemoteAccessAddrInfo>& addrInfos, HcclRtStream stream)
{
    CHK_PTR_NULL(impl_);
    CHK_PTR_NULL(stream);
    CHK_RET(impl_->RemoteRead(addrInfos, stream));
    return HCCL_SUCCESS;
}

HcclResult RemoteAccess::RemoteWrite(const vector<HcomRemoteAccessAddrInfo>& addrInfos, HcclRtStream stream)
{
    CHK_PTR_NULL(impl_);
    CHK_PTR_NULL(stream);
    CHK_RET(impl_->RemoteWrite(addrInfos, stream));
    return HCCL_SUCCESS;
}
} // namespace hccl
