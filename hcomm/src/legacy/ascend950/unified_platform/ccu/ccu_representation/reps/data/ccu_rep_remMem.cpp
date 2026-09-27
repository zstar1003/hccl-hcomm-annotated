/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_rep.h"
#include "ccu_assist.h"

#include "string_util.h"

namespace Hccl {
namespace CcuRep {

    CcuRepRemMem::CcuRepRemMem(const CcuTransport& transport, Memory rem) : transport(transport), rem(rem)
    {
        type = CcuRepType::REM_MEM;
        instrCount = 2; // 指令数为2个
    }

    bool CcuRepRemMem::Translate(CcuInstr*& instr, uint16_t& instrId, [[maybe_unused]] const TransDep& dep)
    {
        this->instrId = instrId;
        translated = true;

        CcuTransport::CclBufferInfo cclBufferInfo;
        uint32_t index = 0;
        transport.GetRmtBuffer(cclBufferInfo, index);
        auto addr = cclBufferInfo.addr;
        auto tokenId = cclBufferInfo.tokenId;
        auto tokenValue = cclBufferInfo.tokenValue;

        auto tokenInfo = GetToken(tokenId, tokenValue, 1);

        LoadImdToGSAInstr(instr++, rem.addr.Id(), addr);
        LoadImdToXnInstr(instr++, rem.token.Id(), tokenInfo, CCU_LOAD_TO_XN_SEC_INFO);

        instrId += instrCount;

        return translated;
    }

    std::string CcuRepRemMem::Describe() { return StringFormat("Get Remote Buffer Addr and TokenInfo By Transport"); }

}; // namespace CcuRep
}; // namespace Hccl
