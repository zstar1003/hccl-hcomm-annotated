/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_EXCHANGE_IPC_BUFFER_DTO_H
#define HCCLV2_EXCHANGE_IPC_BUFFER_DTO_H

#include <string>
#include "types.h"
#include "binary_stream.h"
#include "serializable.h"
#include "orion_adapter_rts.h"
#include "string_util.h"
namespace Hccl {

class ExchangeIpcBufferDto : public Serializable { // Ipc Rma Buffer 需要交换的DTO
public:
    ExchangeIpcBufferDto() {}

    ExchangeIpcBufferDto(u64 addr, u64 size, u64 offset, u32 pid, const char* memInfo)
        : addr(addr),
          size(size),
          offset(offset),
          pid(pid),
          memInfo(memInfo)
    {}

    void Serialize(Hccl::BinaryStream& stream) override { stream << addr << size << offset << pid << name << memInfo; }

    void Deserialize(Hccl::BinaryStream& stream) override
    {
        stream >> addr >> size >> offset >> pid >> name >> memInfo;
    }

    std::string Describe() const override
    {
        std::string strName(name, name + RTS_IPC_MEM_NAME_LEN);
        return StringFormat(
            "ExchangeIpcBufferDto[addr=0x%llx, size=0x%llx, offset=0x%llx, pid=%u, name=%s, memInfo=%s]", addr, size,
            offset, pid, strName.c_str(), memInfo.c_str());
    }

    u64 addr{0};
    u64 size{0};
    u64 offset{0};
    u32 pid{0};
    char_t name[RTS_IPC_MEM_NAME_LEN]{0};
    std::string memInfo{""};
};

} // namespace Hccl

#endif
