/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_EXCHANGE_UB_CONN_DTO_H
#define HCCLV2_EXCHANGE_UB_CONN_DTO_H

#include <string>
#include "binary_stream.h"
#include "serializable.h"
#include "orion_adapter_hccp.h"
namespace Hccl {

class ExchangeUbConnDto : public Serializable { // UB建链需要交换的信息
public:
    ExchangeUbConnDto() {}

    ExchangeUbConnDto(u32 tokenValue, u32 qpKeySize, u64 tpHandle, u32 psn)
        : tokenValue(tokenValue),
          qpKeySize(qpKeySize),
          tpHandle(tpHandle),
          psn(psn)
    {}

    void Serialize(Hccl::BinaryStream& stream) override
    {
        stream << tokenValue << qpKeySize << qpKey << eid << tpHandle << psn;
    }

    void Deserialize(Hccl::BinaryStream& stream) override
    {
        stream >> tokenValue >> qpKeySize >> qpKey >> eid >> tpHandle >> psn;
    }

    std::string Describe() const override
    {
        return StringFormat("ExchangeUbConnDto=[qpKeySize=%u, tpHandle=0x%llx, psn=%u]", qpKeySize, tpHandle, psn);
    }

    u32 tokenValue{0};
    u32 qpKeySize{HRT_UB_QP_KEY_MAX_LEN};
    u8 qpKey[HRT_UB_QP_KEY_MAX_LEN]{0};
    u8 eid[16]{0};
    u64 tpHandle{0};
    u32 psn{0};
};

} // namespace Hccl

#endif
