/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_RTS_NOTIFY_H
#define HCCLV2_RTS_NOTIFY_H

#include <string>
#include "orion_adapter_rts.h"
#include "orion_adapter_hccp.h"
#include "stream.h"
#include "notify_lite.h"

namespace Hccl {

class RtsNotify {
public:
    RtsNotify(const RtsNotify& that) = delete;

    RtsNotify& operator=(const RtsNotify& that) = delete;

    explicit RtsNotify(bool devUsed = false);

    ~RtsNotify();

    string Describe() const;

    void Wait(const Stream& stream, u32 timeout) const;

    void Post(const Stream& stream) const;

    std::string SetIpcName() const;

    void SetIpcPid(s32 pid) const;

    u32 GetId() const;

    u64 GetOffset() const;

    u64 GetHandleAddr() const;

    u32 GetSize() const;

    bool IsDevUsed() const;

    u32 GetDevPhyId() const;

    std::vector<char> GetUniqueId() const;

private:
    u32 devPhyId;
    bool devUsed;
    RtNotify_t handle{};
    u32 id{0};
};

struct IpcNotifyExchangeData {
    char_t name[RTS_IPC_MEM_NAME_LEN]{0};
    u64 handleAddr{0}; // 两rank处于相同Server, 相同进程下, 携带指针 RtNotify 的值
    u32 id{0};
    u32 pid{0};
    u32 devPhyId{0};
    bool devUsed{false};
};

struct RdmaNotifyExchangeData {
    u64 addr{0};
    u32 id{0};
    u32 size{0};
    u8 key[RDMA_MEM_KEY_MAX_LEN]{0};
};

struct UbNotifyExchangeData {
    u64 addr{0};
    u32 id{0};
    u32 size{0};
    u8 key[HRT_UB_MEM_KEY_MAX_LEN]{0};
    u32 tokenValue{0};
    u32 tokenId{0};
    u32 keySize{0};
};

} // namespace Hccl

#endif // !HCCLV2_RTS_NOTIFY_H
