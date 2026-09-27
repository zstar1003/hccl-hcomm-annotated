/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_REPRESENTATION_READ_H
#define HCOMM_CCU_REPRESENTATION_READ_H

#include "ccu_rep_base_v1.h"
#include "ccu_datatype_v1.h"

namespace hcomm {
namespace CcuRep {

    class CcuRepRead : public CcuRepBase {
    public:
        CcuRepRead(
            CcuInsGeneratorBase* insGenPtr, const ChannelHandle channel, LocalAddr loc, RemoteAddr rem, Variable len,
            CompletedEvent sem, uint16_t mask);
        CcuRepRead(
            CcuInsGeneratorBase* insGenPtr, const ChannelHandle channel, LocalAddr loc, RemoteAddr rem, Variable len,
            uint16_t dataType, uint16_t opType, CompletedEvent sem, uint16_t mask);
        bool Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep) override;
        std::string Describe() override;
        uint16_t GetLocAddrId() { return loc.addr.Id(); }
        uint16_t GetLocTokenId() { return loc.token.Id(); }
        uint16_t GetRemAddrId() { return rem.addr.Id(); }
        uint16_t GetRemTokenId() { return rem.token.Id(); }
        uint16_t GetLenId() { return len.Id(); }
        uint16_t GetSemId() { return sem.Id(); }
        uint32_t GetChannelId() const { return channelId; }

        ChannelHandle GetChannel() const { return channel; }
        LocalAddr GetLoc() { return loc; }
        RemoteAddr GetRem() { return rem; }
        Variable GetLen() { return len; }
        CompletedEvent GetSem() { return sem; }
        uint16_t GetMask() const { return mask; }
        uint16_t GetDataType() const { return dataType; }
        uint16_t GetOpType() const { return opType; }
        uint16_t GetReduceFlag() const { return reduceFlag; }

    private:
        CcuInsGeneratorBase* insGenPtr{nullptr};
        ChannelHandle channel;
        uint32_t channelId{0};

        LocalAddr loc;
        RemoteAddr rem;
        Variable len;

        CompletedEvent sem;
        uint16_t mask{0};

        uint16_t dataType{0};
        uint16_t opType{0};
        uint16_t reduceFlag{0};
    };

}; // namespace CcuRep
}; // namespace hcomm
#endif // HCOMM_CCU_REPRESENTATION_READ_H
