/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_EXCHANGE_IPC_NOTIFY_DTO_H
#define HCCLV2_EXCHANGE_IPC_NOTIFY_DTO_H

#include <string>
#include "serializable.h"
#include "binary_stream.h"
#include "string_util.h"
#include "types.h"
#include "binary_stream.h"
#include "orion_adapter_rts.h"
namespace Hccl {

class ExchangeIpcNotifyDto : public Serializable { // Ipc Notify  需要交换的DTO
public:
    ExchangeIpcNotifyDto() {}

    ExchangeIpcNotifyDto(u64 handleAddr, u32 id, u32 pid, u32 devPhyId, bool devUsed)
        : handleAddr(handleAddr),
          id(id),
          pid(pid),
          devPhyId(devPhyId),
          devUsed(devUsed)
    {}

    void Serialize(Hccl::BinaryStream& stream) override
    {
        stream << handleAddr << id << pid << devPhyId << devUsed << name;
    }

    void Deserialize(Hccl::BinaryStream& stream) override
    {
        stream >> handleAddr >> id >> pid >> devPhyId >> devUsed >> name;
    }

    std::string Describe() const override
    {
        std::string strName(name, name + RTS_IPC_MEM_NAME_LEN);
        return StringFormat(
            "ExchangeIpcNotifyDto[handleAddr=0x%llx, id=%u, pid=%u, devPhyId=%u, devUsed=%d, name=%s]", handleAddr, id,
            pid, devPhyId, devUsed, strName.c_str());
    }

    u64 handleAddr{0}; // 两rank处于相同Server, 相同进程下, 携带指针 RtNotify 的值
    u32 id{0};
    u32 pid{0};
    u32 devPhyId{0};
    bool devUsed{false};
    char_t name[RTS_IPC_MEM_NAME_LEN]{0};
};

} // namespace Hccl

#endif
