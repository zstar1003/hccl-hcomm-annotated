/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_REPRESENTATION_REMPOSTSEM_H
#define HCOMM_CCU_REPRESENTATION_REMPOSTSEM_H

#include "ccu_rep_base_v1.h"

namespace hcomm {
namespace CcuRep {

    class CcuRepRemPostSem : public CcuRepBase {
    public:
        explicit CcuRepRemPostSem(
            CcuInsGeneratorBase* insGenPtr, const ChannelHandle channel, uint16_t semIndex, uint16_t mask);
        bool Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep) override;
        std::string Describe() override;
        uint32_t GetId() override { return signalId; }
        uint32_t GetChannelId() const { return channelId; }

        ChannelHandle GetChannel() const { return channel; }
        uint16_t GetSemIndex() const { return semIndex; }
        uint16_t GetMask() const { return mask; }

    private:
        CcuInsGeneratorBase* insGenPtr{nullptr};
        ChannelHandle channel;
        uint32_t channelId{0};
        uint16_t semIndex{0};
        uint16_t mask{0};
        uint32_t signalId{0};
    };

}; // namespace CcuRep
}; // namespace hcomm
#endif // HCOMM_CCU_REPRESENTATION_REMPOSTSEM_H
