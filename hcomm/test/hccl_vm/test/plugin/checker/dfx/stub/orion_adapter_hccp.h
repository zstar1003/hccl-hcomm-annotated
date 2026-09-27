/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ORION_ADAPTER_HCCP_H_
#define ORION_ADAPTER_HCCP_H_

#include <cstddef>
#include <cstdint>
#include <cstring>

#define MAKE_ENUM(enumClass, ...) enum class enumClass { __VA_ARGS__ }

namespace Hccl {
MAKE_ENUM(HrtNetworkMode, PEER, HDC);

struct HRaInfo {
    HrtNetworkMode mode;
    uint32_t phyId;
    HRaInfo(HrtNetworkMode m, uint32_t p) : mode(m), phyId(p) {}
};

void HrtRaCustomChannel(const HRaInfo& raInfo, void* customIn, void* customOut);
} // namespace Hccl

namespace hcomm {
namespace CcuRep {
    enum class CcuOpcodeType {
        CCU_U_OP_GET_VERSION = 0,
        CCU_U_OP_GET_MISSION_CTX = 208,
        CCU_U_OP_GET_LOOP_CTX = 209,
        CCU_U_OP_GET_GSA = 202,
        CCU_U_OP_GET_XN = 203,
        CCU_U_OP_GET_CKE = 204,
    };

    union CcuDataTypeUnion {
        uint64_t u64;
        uint32_t u32;
        uint8_t u8;
    };

    struct CcuData {
        uint32_t udieIdx;
        uint32_t dataArraySize;
        uint32_t dataLen;
        CcuDataTypeUnion dataArray[8];
    };

    union CcuDataUnion {
        char raw[2048];
        struct CcuData dataInfo;
    };

    struct CustomChannelInfoIn {
        CcuDataUnion data;
        uint32_t offsetStartIdx;
        CcuOpcodeType op;

        CustomChannelInfoIn() : offsetStartIdx(0), op(CcuOpcodeType::CCU_U_OP_GET_VERSION)
        {
            memset(&data, 0, sizeof(data));
        }
    };

    struct CustomChannelInfoOut {
        CcuDataUnion data;
        uint32_t offsetNextIdx;
        int opRet;

        CustomChannelInfoOut() : offsetNextIdx(0), opRet(0) { memset(&data, 0, sizeof(data)); }
    };
} // namespace CcuRep
} // namespace hcomm

#endif
