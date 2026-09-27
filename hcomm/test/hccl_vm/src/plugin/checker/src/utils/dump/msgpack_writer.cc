/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <limits>
#include <ostream>

#include "dump/msgpack_writer.h"

namespace HcclSim {
MsgpackWriter::MsgpackWriter(std::vector<uint8_t>& buffer) : m_buffer(&buffer) {}

MsgpackWriter::MsgpackWriter(std::ostream& out) : m_out(&out) {}

void MsgpackWriter::WriteUInt(uint64_t value)
{
    if (value <= 0x7F) {
        WriteByte(static_cast<uint8_t>(value));
        return;
    }
    if (value <= std::numeric_limits<uint8_t>::max()) {
        WriteByte(0xCC);
        WriteByte(static_cast<uint8_t>(value));
        return;
    }
    if (value <= std::numeric_limits<uint16_t>::max()) {
        WriteByte(0xCD);
        WriteUInt16(static_cast<uint16_t>(value));
        return;
    }
    if (value <= std::numeric_limits<uint32_t>::max()) {
        WriteByte(0xCE);
        WriteUInt32(static_cast<uint32_t>(value));
        return;
    }
    WriteByte(0xCF);
    WriteUInt64(value);
}

void MsgpackWriter::WriteString(const std::string& value)
{
    const uint64_t length = static_cast<uint64_t>(value.size());
    if (length <= 31) {
        WriteByte(static_cast<uint8_t>(0xA0 | length));
    } else if (length <= std::numeric_limits<uint8_t>::max()) {
        WriteByte(0xD9);
        WriteByte(static_cast<uint8_t>(length));
    } else if (length <= std::numeric_limits<uint16_t>::max()) {
        WriteByte(0xDA);
        WriteUInt16(static_cast<uint16_t>(length));
    } else {
        WriteByte(0xDB);
        WriteUInt32(static_cast<uint32_t>(length));
    }
    WriteBytes(reinterpret_cast<const uint8_t*>(value.data()), value.size());
}

void MsgpackWriter::WriteArrayHeader(size_t count)
{
    if (count <= 15) {
        WriteByte(static_cast<uint8_t>(0x90 | count));
        return;
    }
    if (count <= std::numeric_limits<uint16_t>::max()) {
        WriteByte(0xDC);
        WriteUInt16(static_cast<uint16_t>(count));
        return;
    }
    WriteByte(0xDD);
    WriteUInt32(static_cast<uint32_t>(count));
}

void MsgpackWriter::WriteMapHeader(size_t count)
{
    if (count <= 15) {
        WriteByte(static_cast<uint8_t>(0x80 | count));
        return;
    }
    if (count <= std::numeric_limits<uint16_t>::max()) {
        WriteByte(0xDE);
        WriteUInt16(static_cast<uint16_t>(count));
        return;
    }
    WriteByte(0xDF);
    WriteUInt32(static_cast<uint32_t>(count));
}

void MsgpackWriter::WriteRawBytes(const uint8_t* data, size_t size) { WriteBytes(data, size); }

void MsgpackWriter::WriteRawBytes(const std::vector<uint8_t>& bytes) { WriteBytes(bytes.data(), bytes.size()); }

void MsgpackWriter::WriteByte(uint8_t value)
{
    if (m_buffer != nullptr) {
        m_buffer->push_back(value);
        return;
    }
    m_out->put(static_cast<char>(value));
}

void MsgpackWriter::WriteBytes(const uint8_t* data, size_t size)
{
    if (size == 0) {
        return;
    }
    if (m_buffer != nullptr) {
        m_buffer->insert(m_buffer->end(), data, data + size);
        return;
    }
    m_out->write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
}

void MsgpackWriter::WriteUInt16(uint16_t value)
{
    WriteByte(static_cast<uint8_t>((value >> 8) & 0xFF));
    WriteByte(static_cast<uint8_t>(value & 0xFF));
}

void MsgpackWriter::WriteUInt32(uint32_t value)
{
    WriteByte(static_cast<uint8_t>((value >> 24) & 0xFF));
    WriteByte(static_cast<uint8_t>((value >> 16) & 0xFF));
    WriteByte(static_cast<uint8_t>((value >> 8) & 0xFF));
    WriteByte(static_cast<uint8_t>(value & 0xFF));
}

void MsgpackWriter::WriteUInt64(uint64_t value)
{
    WriteByte(static_cast<uint8_t>((value >> 56) & 0xFF));
    WriteByte(static_cast<uint8_t>((value >> 48) & 0xFF));
    WriteByte(static_cast<uint8_t>((value >> 40) & 0xFF));
    WriteByte(static_cast<uint8_t>((value >> 32) & 0xFF));
    WriteByte(static_cast<uint8_t>((value >> 24) & 0xFF));
    WriteByte(static_cast<uint8_t>((value >> 16) & 0xFF));
    WriteByte(static_cast<uint8_t>((value >> 8) & 0xFF));
    WriteByte(static_cast<uint8_t>(value & 0xFF));
}
} // namespace HcclSim
