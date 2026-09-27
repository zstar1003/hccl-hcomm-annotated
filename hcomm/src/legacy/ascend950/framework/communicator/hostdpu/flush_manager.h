/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_FLUSH_MANAGER_H
#define HCCL_FLUSH_MANAGER_H

#include <memory.h>
#include <unordered_map>
#include "flush_handle.h"
#include "infiniband/verbs.h"

namespace Hccl {

#define ROCE_WR_FLUSH (static_cast<ibv_wr_opcode>(0x40))

class FlushManager {
public:
    static FlushManager& GetInstance();
    FlushManager(const FlushManager&) = delete;
    FlushManager& operator=(const FlushManager&) = delete;

    HcclResult initFlushHandle(IpAddress ip, u32 devPhyId);
    HcclResult Flush();

private:
    FlushManager();
    ~FlushManager();
    std::unordered_map<IpAddress, std::shared_ptr<FlushHandle>> flushHandleMap_;

    // flush实现
    HcclResult FlushParamPrepare(std::shared_ptr<FlushHandle> flushHandlePtr, ibv_send_wr* swr) const;
    HcclResult ExecuteRdmaRead(ibv_qp* loopbackqp0, ibv_cq* cq, ibv_send_wr& swr, int timeoutSec) const;

    // flush销毁
    HcclResult DestroyAll();
    std::mutex mutex_;
    int FlushPostSend(struct ibv_qp* qp, struct ibv_send_wr* wr, struct ibv_send_wr** bad_wr) const
    {
        return ibv_post_send(qp, wr, bad_wr);
    }
    int FlushPollCq(struct ibv_cq* cq, int num_entries, struct ibv_wc* wc) const
    {
        return ibv_poll_cq(cq, num_entries, wc);
    }
};

} // namespace Hccl

#endif
