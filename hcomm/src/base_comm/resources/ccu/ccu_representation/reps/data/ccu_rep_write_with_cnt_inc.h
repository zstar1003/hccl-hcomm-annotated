/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_REPRESENTATION_WRITE_WITH_CNT_INC_H
#define HCOMM_CCU_REPRESENTATION_WRITE_WITH_CNT_INC_H

#include "ccu_rep_base_v1.h"
#include "ccu_datatype_v1.h"

namespace hcomm {
namespace CcuRep {

    class CcuRepWriteWithCntInc : public CcuRepBase {
    public:
        CcuRepWriteWithCntInc(
            CcuInsGeneratorBase* insGenPtr, Variable& channel, RemoteAddr& rem, LocalAddr& loc, Variable& len,
            RemoteAddr& incCnt);

        bool Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep) override;
        std::string Describe() override;

        uint16_t GetChannelVarId() { return channel.Id(); }
        uint16_t GetLocAddrId() { return loc.addr.Id(); }
        uint16_t GetLocTokenId() { return loc.token.Id(); }
        uint16_t GetRemAddrId() { return rem.addr.Id(); }
        uint16_t GetRemTokenId() { return rem.token.Id(); }
        uint16_t GetLenId() { return len.Id(); }
        uint16_t GetIncCntAddrId() { return incCnt.addr.Id(); }
        uint16_t GetIncCntTokenId() { return incCnt.token.Id(); }

        const Variable& GetChannelVar() const { return channel; }
        const RemoteAddr& GetRem() const { return rem; }
        const LocalAddr& GetLoc() const { return loc; }
        const Variable& GetLen() const { return len; }
        const RemoteAddr& GetIncCnt() const { return incCnt; }

    private:
        CcuInsGeneratorBase* insGenPtr{nullptr};
        Variable channel;
        RemoteAddr rem;
        LocalAddr loc;
        Variable len;
        RemoteAddr incCnt;
    };

}; // namespace CcuRep
}; // namespace hcomm
#endif // HCOMM_CCU_REPRESENTATION_WRITE_WITH_CNT_INC_H
