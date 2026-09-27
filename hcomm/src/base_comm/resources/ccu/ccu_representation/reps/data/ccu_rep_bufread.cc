/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exception_util.h"
#include "ccu_api_exception.h"
#include "hcomm_c_adpt.h"
#include "../../../../endpoint_pairs/channels/ccu/ccu_urma_channel.h"
#include "ccu_ins_generator_base.h"

namespace hcomm {
namespace CcuRep {

    CcuRepBufRead::CcuRepBufRead(
        CcuInsGeneratorBase* insGenPtr, const ChannelHandle channel, RemoteAddr src, CcuBuf dst, Variable len,
        CompletedEvent sem, uint32_t mask)
        : insGenPtr(insGenPtr),
          channel(channel),
          src(src),
          dst(dst),
          len(len),
          sem(sem),
          mask(mask)
    {
        type = CcuRepType::BUF_READ;
        instrCount = insGenPtr->GetInstrCount(type);
    }

    bool CcuRepBufRead::Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep)
    {
        this->instrId = instrId;
        translated = true;

        CHK_PRT_THROW(
            insGenPtr->CcuRepBufReadTranslate(ccuKernel, instr, this, dep) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[CcuRepBufRead][Translate] failed to translate for instrId[%u]", instrId),
            Hccl::CcuApiException, "CcuRepBufRead translate failed");
        instrId += instrCount;

        return translated;
    }

    std::string CcuRepBufRead::Describe()
    {
        void* channelPtr{nullptr};
        auto ret = HcommChannelGet(channel, &channelPtr);
        if (ret != HcclResult::HCCL_SUCCESS) {
            Hccl::THROW<Hccl::CcuApiException>("failed to get ccu channel, type[%d]", type);
        }

        auto* channelImpl = dynamic_cast<CcuUrmaChannel*>(static_cast<Channel*>(channelPtr));
        if (channelImpl == nullptr) {
            Hccl::THROW<Hccl::CcuApiException>(
                "[%s] failed to cast channel[0x%llx] to CcuUrmaChannel", __func__, channel);
        }
        return Hccl::StringFormat(
            "Read Rmt Mem[%u] To CcuBuf[%u], len[%u], ChannalId[%u], sem[%u], mask[%04x]", src.addr.Id(), dst.Id(),
            len.Id(), channelImpl->GetChannelId(), sem.Id(), mask);
    }

}; // namespace CcuRep
}; // namespace hcomm
