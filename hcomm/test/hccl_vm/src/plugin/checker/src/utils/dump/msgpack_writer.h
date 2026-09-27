/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_VM_MSGPACK_WRITER_H
#define HCCL_VM_MSGPACK_WRITER_H

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace HcclSim {
class MsgpackWriter {
public:
    explicit MsgpackWriter(std::vector<uint8_t>& buffer);
    explicit MsgpackWriter(std::ostream& out);

    void WriteUInt(uint64_t value);
    void WriteString(const std::string& value);
    void WriteArrayHeader(size_t count);
    void WriteMapHeader(size_t count);
    void WriteRawBytes(const uint8_t* data, size_t size);
    void WriteRawBytes(const std::vector<uint8_t>& bytes);

private:
    void WriteByte(uint8_t value);
    void WriteBytes(const uint8_t* data, size_t size);
    void WriteUInt16(uint16_t value);
    void WriteUInt32(uint32_t value);
    void WriteUInt64(uint64_t value);

    std::vector<uint8_t>* m_buffer = nullptr;
    std::ostream* m_out = nullptr;
};
} // namespace HcclSim

#endif // HCCL_VM_MSGPACK_WRITER_H
