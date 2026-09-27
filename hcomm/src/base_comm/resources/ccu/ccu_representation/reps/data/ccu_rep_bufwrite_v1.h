/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_REPRESENTATION_BUFWRITE_H
#define HCOMM_CCU_REPRESENTATION_BUFWRITE_H

#include "ccu_rep_base_v1.h"
#include "ccu_datatype_v1.h"

namespace hcomm {
namespace CcuRep {

    class CcuRepBufWrite : public CcuRepBase {
    public:
        CcuRepBufWrite(
            CcuInsGeneratorBase* insGenPtr, const ChannelHandle channel, CcuBuf src, RemoteAddr dst, Variable len,
            CompletedEvent sem, uint16_t mask);
        bool Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep) override;
        std::string Describe() override;

        uint16_t GetSrcId() { return src.Id(); }
        uint16_t GetDstAddrId() { return dst.addr.Id(); }
        uint16_t GetDstTokenId() { return dst.token.Id(); }
        uint16_t GetLenId() { return len.Id(); }
        uint16_t GetSemId() { return sem.Id(); }
        uint32_t GetChannelId() const { return channelId; }
        CcuBuf GetSrc() { return src; }
        RemoteAddr GetDst() { return dst; }
        Variable GetLen() { return len; }
        CompletedEvent GetSem() { return sem; }
        uint16_t GetMask() const { return mask; }
        ChannelHandle GetChannel() const { return channel; }

    private:
        CcuInsGeneratorBase* insGenPtr{nullptr};
        ChannelHandle channel;
        uint32_t channelId{0};

        CcuBuf src;
        RemoteAddr dst;
        Variable len;

        CompletedEvent sem;
        uint16_t mask{0};
    };

}; // namespace CcuRep
}; // namespace hcomm
#endif // HCOMM_CCU_REPRESENTATION_BUFWRITE_H
