/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_IPC_REMOTE_NOTIFY_H
#define HCCLV2_IPC_REMOTE_NOTIFY_H

#include "rts_notify_v2.h"
#include "stream.h"
#include "rma_type.h"
#include "serializable.h"

namespace Hccl {

class BaseRemoteNotify {
public:
    explicit BaseRemoteNotify(const RmaType& type) : type(type) {}

    virtual ~BaseRemoteNotify() = default;

    virtual string Describe() const = 0;

    virtual void Post(const Stream& stream) const = 0; // 待修改: will move this method to synchronizer

protected:
    RmaType type;
};

class IpcRemoteNotify : public BaseRemoteNotify {
public:
    IpcRemoteNotify();

    explicit IpcRemoteNotify(const Serializable& rmtDto);

    IpcRemoteNotify(const IpcRemoteNotify& that) = delete;

    IpcRemoteNotify& operator=(const IpcRemoteNotify& that) = delete;

    void Post(const Stream& stream) const override;

    string Describe() const override;

    inline u32 GetId() const { return id; }

    inline u64 GetAddr() const { return addr; }

    inline u32 GetSize() const { return size; }

private:
    char_t name[RTS_IPC_MEM_NAME_LEN]{0};
    u64 handleAddr{0}; // 两rank处于相同Server, 相同进程下, 携带指针 RtNotify 的值
    u32 id{0};
    u32 rmtPid{0};
    u32 rmtDevPhyId{0};
    bool devUsed{false};
    RtNotify_t handle{};
    u64 addr{0};
    u32 size{0};
};

} // namespace Hccl

#endif // !HCCLV2_IPC_REMOTE_NOTIFY_H
