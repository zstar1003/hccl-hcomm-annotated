/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRANSPORT_NET_PUB_H
#define TRANSPORT_NET_PUB_H

#include "transport_base_pub.h"

namespace hccl {
class TransportNet : public TransportBase {
public:
    explicit TransportNet(
        DispatcherPub* dispatcher, const std::unique_ptr<NotifyPool>& notifyPool, MachinePara& machinePara,
        std::chrono::milliseconds timeout);
    ~TransportNet() override;

protected:
};
} // namespace hccl
#endif /* * TRANSPORT_NET_PUB_H */
