/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_ins_generator_v1.h"
#include "hcomm_c_adpt.h"
#include "ccu_rep_base_v1.h"
#include "../../../../endpoint_pairs/channels/ccu/ccu_urma_channel.h"
#include "ccu_api_exception.h"
#include "ccu_assist_v1.h"
#include "ccu_log.h"
#include "../../../ccu_device/ccu_res_specs.h"

namespace hcomm {
namespace CcuRep {

#define UNUSED(x) (void)(x)

    namespace {
        template <typename T>
        void LoadAddrArg(CcuInstr*& instr, const T& dst, const T& src, const TransDep& dep)
        {
            LoadGSAGSAInstr(instr++, dst.addr.Id(), src.addr.Id(), dep.reserveGsaId);
            LoadXXInstr(instr++, dst.token.Id(), src.token.Id(), dep.reserveXnId);
        }

        template <typename T>
        HcclResult
        LoadAddrListArg(CcuInstr*& instr, const std::vector<T>& dst, const std::vector<T>& src, const TransDep& dep)
        {
            if (src.size() != dst.size()) {
                HCCL_ERROR("Mismatched Arg Size: srcSize[%u], dstSize[%u]", src.size(), dst.size());
                return HCCL_E_PARA;
            }
            for (uint32_t j = 0; j < src.size(); j++) {
                LoadAddrArg(instr, dst[j], src[j], dep);
            }
            return HcclResult::HCCL_SUCCESS;
        }
    } // namespace

    HcclResult CcuInsGeneratorV1::CcuRepBufLocReadTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepBufLocRead* repBufLocRead, const TransDep& dep)
    {
        UNUSED(ccuKernel);
        CHK_PTR_NULL(repBufLocRead);
        TransLocMemToLocMSInstr(
            instr++, repBufLocRead->GetDst().Id(), repBufLocRead->GetSrc().addr.Id(),
            repBufLocRead->GetSrc().token.Id(), repBufLocRead->GetLen().Id(), dep.reserveChannalId[0],
            repBufLocRead->GetSem().Id(), repBufLocRead->GetMask(), 0, 0, 1, 1);

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV1::CcuRepBufLocWriteTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepBufLocWrite* repBufLocWrite, const TransDep& dep)
    {
        UNUSED(ccuKernel);
        CHK_PTR_NULL(repBufLocWrite);
        TransLocMSToLocMemInstr(
            instr++, repBufLocWrite->GetDst().addr.Id(), repBufLocWrite->GetDst().token.Id(),
            repBufLocWrite->GetSrc().Id(), repBufLocWrite->GetLen().Id(), dep.reserveChannalId[0],
            repBufLocWrite->GetSem().Id(), repBufLocWrite->GetMask(), 0, 0, 1, 1);

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV1::CcuRepBufReadTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepBufRead* repBufRead, const TransDep& dep)
    {
        UNUSED(ccuKernel);
        UNUSED(dep);
        CHK_PTR_NULL(repBufRead);
        void* channelPtr{nullptr};
        CHK_PRT_RET(
            static_cast<HcclResult>(HcommChannelGet(repBufRead->GetChannel(), &channelPtr)) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("failed to get ccu channel, type[%d]", repBufRead->Type()), HCCL_E_INTERNAL);

        auto* channelImpl = dynamic_cast<CcuUrmaChannel*>(static_cast<Channel*>(channelPtr));
        CHK_PTR_NULL(channelImpl);

        TransRmtMemToLocMSInstr(
            instr++, repBufRead->GetDst().Id(), repBufRead->GetSrc().addr.Id(), repBufRead->GetSrc().token.Id(),
            repBufRead->GetLen().Id(), channelImpl->GetChannelId(), repBufRead->GetSem().Id(), repBufRead->GetMask(), 0,
            0, 1, 1);

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV1::CcuRepWriteTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepWrite* repWrite)
    {
        UNUSED(ccuKernel);
        CHK_PTR_NULL(repWrite);
        void* channelPtr{nullptr};
        CHK_PRT_RET(
            static_cast<HcclResult>(HcommChannelGet(repWrite->GetChannel(), &channelPtr)) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("failed to get ccu channel, type[%d]", repWrite->Type()), HCCL_E_INTERNAL);

        auto* channelImpl = dynamic_cast<CcuUrmaChannel*>(static_cast<Channel*>(channelPtr));
        CHK_PTR_NULL(channelImpl);
        TransLocMemToRmtMemInstr(
            instr++, repWrite->GetRem().addr.Id(), repWrite->GetRem().token.Id(), repWrite->GetLoc().addr.Id(),
            repWrite->GetLoc().token.Id(), repWrite->GetLen().Id(), channelImpl->GetChannelId(),
            repWrite->GetDataType(), repWrite->GetOpType(), repWrite->GetSem().Id(), repWrite->GetMask(), 0, 0, 1, 1,
            repWrite->GetReduceFlag());

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV1::CcuRepReadTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepRead* repRemMem)
    {
        UNUSED(ccuKernel);
        CHK_PTR_NULL(repRemMem);
        void* channelPtr{nullptr};
        CHK_PRT_RET(
            static_cast<HcclResult>(HcommChannelGet(repRemMem->GetChannel(), &channelPtr)) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("failed to get ccu channel, type[%d]", repRemMem->Type()), HCCL_E_INTERNAL);

        auto* channelImpl = dynamic_cast<CcuUrmaChannel*>(static_cast<Channel*>(channelPtr));
        CHK_PTR_NULL(channelImpl);
        TransRmtMemToLocMemInstr(
            instr++, repRemMem->GetLoc().addr.Id(), repRemMem->GetLoc().token.Id(), repRemMem->GetRem().addr.Id(),
            repRemMem->GetRem().token.Id(), repRemMem->GetLen().Id(), channelImpl->GetChannelId(),
            repRemMem->GetDataType(), repRemMem->GetOpType(), repRemMem->GetSem().Id(), repRemMem->GetMask(), 0, 0, 1,
            1, repRemMem->GetReduceFlag());

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV1::CcuRepRemMemTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepRemMem* repRemMem)
    {
        UNUSED(ccuKernel);
        CHK_PTR_NULL(repRemMem);
        void* channelPtr{nullptr};
        CHK_PRT_RET(
            static_cast<HcclResult>(HcommChannelGet(repRemMem->GetChannel(), &channelPtr)) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("failed to get ccu channel, type[%d]", repRemMem->Type()), HCCL_E_INTERNAL);

        auto* channelImpl = dynamic_cast<CcuUrmaChannel*>(static_cast<Channel*>(channelPtr));
        uint32_t size{0}, tokenId{0}, tokenValue{0};
        uint64_t addr{0};
        CHK_PTR_NULL(channelImpl);
        CHK_PRT_RET(
            channelImpl->GetRmtBuffer(addr, size, tokenId, tokenValue) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR(
                "[CcuRepRemMem][%s] failed to get remote buffer, channelHandle[0x%llx].", __func__,
                repRemMem->GetChannel()),
            HCCL_E_UNAVAIL); // 当前认为channel只持有一个buffer

        auto tokenInfo = GetToken(tokenId, tokenValue, 1);

        LoadImdToGSAInstr(instr++, repRemMem->GetRem().addr.Id(), addr);
        LoadImdToXnInstr(instr++, repRemMem->GetRem().token.Id(), tokenInfo, CCU_LOAD_TO_XN_SEC_INFO);

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV1::CcuRepLocCpyTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepLocCpy* ccuRepLocCpy, const TransDep& dep)
    {
        UNUSED(ccuKernel);
        CHK_PTR_NULL(ccuRepLocCpy);
        if (ccuRepLocCpy->GetReduceFlag() == 0) {
            TransLocMemToLocMemInstr(
                instr++, ccuRepLocCpy->GetDst().addr.Id(), ccuRepLocCpy->GetDst().token.Id(),
                ccuRepLocCpy->GetSrc().addr.Id(), ccuRepLocCpy->GetSrc().token.Id(), ccuRepLocCpy->GetLen().Id(),
                dep.reserveChannalId[0], ccuRepLocCpy->GetSem().Id(), ccuRepLocCpy->GetMask(), 0, 0, 1, 1);
        } else {
            // 这个翻译需要验证
            TransLocMemToRmtMemInstr(
                instr++, ccuRepLocCpy->GetDst().addr.Id(), ccuRepLocCpy->GetDst().token.Id(),
                ccuRepLocCpy->GetSrc().addr.Id(), ccuRepLocCpy->GetSrc().token.Id(), ccuRepLocCpy->GetLen().Id(),
                dep.reserveChannalId[0], ccuRepLocCpy->GetDataType(), ccuRepLocCpy->GetOpType(),
                ccuRepLocCpy->GetSem().Id(), ccuRepLocCpy->GetMask(), 0, 0, 1, 1, ccuRepLocCpy->GetReduceFlag());
        }

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV1::CcuRepBufWriteTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepBufWrite* ccuRepBufWrite, const TransDep& dep)
    {
        UNUSED(ccuKernel);
        UNUSED(dep);
        CHK_PTR_NULL(ccuRepBufWrite);
        void* channelPtr{nullptr};
        CHK_PRT_RET(
            static_cast<HcclResult>(HcommChannelGet(ccuRepBufWrite->GetChannel(), &channelPtr))
                != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("failed to get ccu channel, type[%d]", ccuRepBufWrite->Type()), HCCL_E_INTERNAL);

        auto* channelImpl = dynamic_cast<CcuUrmaChannel*>(static_cast<Channel*>(channelPtr));
        CHK_PTR_NULL(channelImpl);
        TransLocMSToRmtMemInstr(
            instr++, ccuRepBufWrite->GetDst().addr.Id(), ccuRepBufWrite->GetDst().token.Id(),
            ccuRepBufWrite->GetSrc().Id(), ccuRepBufWrite->GetLen().Id(), channelImpl->GetChannelId(),
            ccuRepBufWrite->GetSem().Id(), ccuRepBufWrite->GetMask(), 0, 0, 1, 1);

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV1::CcuRepBufReduceTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepBufReduce* ccuRepBufReduce)
    {
        UNUSED(ccuKernel);
        CHK_PTR_NULL(ccuRepBufReduce);
        if (ccuRepBufReduce->GetCount() < CCU_REDUCE_MIN_MS) {
            HCCL_ERROR("count[%u] must be at least %u", ccuRepBufReduce->GetCount(), CCU_REDUCE_MIN_MS);
            return HCCL_E_PARA;
        }
        if (ccuRepBufReduce->GetCount() > CCU_REDUCE_MAX_MS || ccuRepBufReduce->GetMem().size() > CCU_REDUCE_MAX_MS) {
            HCCL_ERROR(
                "count[%u] and mem size[%zu] must be less than %u", ccuRepBufReduce->GetCount(),
                ccuRepBufReduce->GetMem().size(), CCU_REDUCE_MAX_MS);
            return HCCL_E_PARA;
        }

        // 这里需要注意，在数据格式膨胀的情况下，需要传入用来存放输出的MSId
        // 特别是2P场景，输入MS的数目为2，但是在8bit进，32bit出的场景，输出MS的数目为4
        // 传入的MS中已经包含了需要使用的输入输出的最大量，因此，这里应该直接去MS的size
        auto mem = ccuRepBufReduce->GetMem();
        uint16_t msId[CCU_REDUCE_MAX_MS] = {0};
        for (uint16_t i = 0; i < mem.size(); i++) {
            msId[i] = mem[i].Id();
        }

        if (ccuRepBufReduce->GetOpType() == CCU_REDUCE_SUM) {
            if (ccuRepBufReduce->GetOutputDataType() == 1) { // 1是fp16
                AddInstr(
                    instr++, msId, ccuRepBufReduce->GetCount(), ccuRepBufReduce->GetOutputDataType(),
                    ccuRepBufReduce->GetDataType(), ccuRepBufReduce->GetSem().Id(), ccuRepBufReduce->GetMask(), 0, 0, 1,
                    ccuRepBufReduce->GetXnIdLength().Id());
            } else if (ccuRepBufReduce->GetOutputDataType() == 2) { // 2是bf16
                AddInstr(
                    instr++, msId, ccuRepBufReduce->GetCount(), ccuRepBufReduce->GetOutputDataType(),
                    ccuRepBufReduce->GetDataType(), ccuRepBufReduce->GetSem().Id(), ccuRepBufReduce->GetMask(), 0, 0, 1,
                    ccuRepBufReduce->GetXnIdLength().Id());
            } else {
                AddInstr(
                    instr++, msId, ccuRepBufReduce->GetCount(), 0, ccuRepBufReduce->GetDataType(),
                    ccuRepBufReduce->GetSem().Id(), ccuRepBufReduce->GetMask(), 0, 0, 1,
                    ccuRepBufReduce->GetXnIdLength().Id());
            }
        } else if (ccuRepBufReduce->GetOpType() == CCU_REDUCE_MAX) {
            MaxInstr(
                instr++, msId, ccuRepBufReduce->GetCount(), ccuRepBufReduce->GetDataType(),
                ccuRepBufReduce->GetSem().Id(), ccuRepBufReduce->GetMask(), 0, 0, 1,
                ccuRepBufReduce->GetXnIdLength().Id());
        } else if (ccuRepBufReduce->GetOpType() == CCU_REDUCE_MIN) {
            MinInstr(
                instr++, msId, ccuRepBufReduce->GetCount(), ccuRepBufReduce->GetDataType(),
                ccuRepBufReduce->GetSem().Id(), ccuRepBufReduce->GetMask(), 0, 0, 1,
                ccuRepBufReduce->GetXnIdLength().Id());
        }

        return HcclResult::HCCL_SUCCESS;
    }

    uint32_t CcuInsGeneratorV1::GetInstrCount(CcuRepType repType)
    {
        if (repTypeInstrCount.find(repType) == repTypeInstrCount.end()) {
            Hccl::THROW<Hccl::CcuApiException>("[%s] Unsupported repType[%d]", __func__, repType);
        }
        return repTypeInstrCount[repType];
    }

    HcclResult CcuInsGeneratorV1::CcuRepLocRecordEventTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepLocRecordEvent* ccuRepLocRecordEvent)
    {
        UNUSED(ccuKernel);
        CHK_PTR_NULL(ccuRepLocRecordEvent);
        SetCKEInstr(instr++, ccuRepLocRecordEvent->GetEvent().Id(), ccuRepLocRecordEvent->GetMask(), 0, 0, 1);

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV1::CcuRepLocWaitEventTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepLocWaitEvent* ccuRepLocWaitEvent)
    {
        UNUSED(ccuKernel);
        CHK_PTR_NULL(ccuRepLocWaitEvent);
        // SetCKEInstr支持硬件profiling功能
        if (ccuRepLocWaitEvent->GetIsProfiling()) {
            SetCKEInstr(instr++, 0, 0, ccuRepLocWaitEvent->GetEvent().Id(), ccuRepLocWaitEvent->GetMask(), 1);
        } else {
            ClearCKEInstr(instr++, 0, 0, ccuRepLocWaitEvent->GetEvent().Id(), ccuRepLocWaitEvent->GetMask(), 1);
        }

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV1::CcuRepLocWaitNotifyTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepLocWaitNotify* ccuRepLocWaitNotify)
    {
        UNUSED(ccuKernel);
        CHK_PTR_NULL(ccuRepLocWaitNotify);
        // SetCKEInstr支持硬件profiling功能
        if (ccuRepLocWaitNotify->GetIsProfiling()) {
            SetCKEInstr(instr++, 0, 0, ccuRepLocWaitNotify->GetNotify().Id(), ccuRepLocWaitNotify->GetMask(), 1);
        } else {
            ClearCKEInstr(instr++, 0, 0, ccuRepLocWaitNotify->GetNotify().Id(), ccuRepLocWaitNotify->GetMask(), 1);
        }

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV1::CcuRepRemWaitSemTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepRemWaitSem* ccuRepRemWaitSem)
    {
        UNUSED(ccuKernel);
        CHK_PTR_NULL(ccuRepRemWaitSem);
        void* channelPtr{nullptr};
        CHK_PRT_RET(
            static_cast<HcclResult>(HcommChannelGet(ccuRepRemWaitSem->GetChannel(), &channelPtr))
                != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("failed to get ccu channel, type[%d]", ccuRepRemWaitSem->Type()), HCCL_E_INTERNAL);

        auto* channelImpl = dynamic_cast<CcuUrmaChannel*>(static_cast<Channel*>(channelPtr));
        CHK_PTR_NULL(channelImpl);
        uint32_t locCkeId{0};
        CHK_PRT_RET(
            channelImpl->GetLocCkeByIndex(ccuRepRemWaitSem->GetSemIndex(), locCkeId) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR(
                "[CcuRepRemWaitSem][%s] failed to get loc cke id, channelHandle[0x%llx], semIndex[%u].", __func__,
                ccuRepRemWaitSem->GetChannel(), ccuRepRemWaitSem->GetSemIndex()),
            HCCL_E_UNAVAIL);

        // 需要profiling的使用SetCKEInstr, 否则使用ClearCKEInstr
        if (ccuRepRemWaitSem->GetIsProfiling()) {
            SetCKEInstr(instr++, 0, 0, locCkeId, ccuRepRemWaitSem->GetMask(), 1);
        } else {
            ClearCKEInstr(instr++, 0, 0, locCkeId, ccuRepRemWaitSem->GetMask(), 1);
        }

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV1::CcuRepRemPostVarTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepRemPostVar* ccuRepRemPostVar)
    {
        UNUSED(ccuKernel);
        CHK_PTR_NULL(ccuRepRemPostVar);
        void* channelPtr{nullptr};
        CHK_PRT_RET(
            static_cast<HcclResult>(HcommChannelGet(ccuRepRemPostVar->GetChannel(), &channelPtr))
                != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("failed to get ccu channel, type[%d]", ccuRepRemPostVar->Type()), HCCL_E_INTERNAL);

        auto* channelImpl = dynamic_cast<CcuUrmaChannel*>(static_cast<Channel*>(channelPtr));
        CHK_PTR_NULL(channelImpl);
        uint32_t rmtXnId{0};
        CHK_PRT_RET(
            channelImpl->GetRmtXnByIndex(ccuRepRemPostVar->GetParamIndex(), rmtXnId) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR(
                "[CcuRepRemPostSem][%s] failed to get remote xn id, channelHandle[0x%llx].", __func__,
                ccuRepRemPostVar->GetChannel()),
            HCCL_E_UNAVAIL);

        uint32_t rmtCkeId{0};
        CHK_PRT_RET(
            channelImpl->GetRmtCkeByIndex(ccuRepRemPostVar->GetSemIndex(), rmtCkeId) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR(
                "[CcuRepRemPostSem][%s] failed to get remote cke id, channelHandle[0x%llx].", __func__,
                ccuRepRemPostVar->GetChannel()),
            HCCL_E_UNAVAIL);

        SyncXnInstr(
            instr++, rmtXnId, ccuRepRemPostVar->GetParam().Id(), channelImpl->GetChannelId(), rmtCkeId,
            ccuRepRemPostVar->GetMask(), 0, 0, 0, 0, 1);

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV1::CcuRepRemPostSemTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepRemPostSem* ccuRepRemPostSem, const TransDep& dep)
    {
        UNUSED(ccuKernel);
        CHK_PTR_NULL(ccuRepRemPostSem);
        void* channelPtr{nullptr};
        CHK_PRT_RET(
            static_cast<HcclResult>(HcommChannelGet(ccuRepRemPostSem->GetChannel(), &channelPtr))
                != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("failed to get ccu channel, type[%d]", ccuRepRemPostSem->Type()), HCCL_E_INTERNAL);

        auto* channelImpl = dynamic_cast<CcuUrmaChannel*>(static_cast<Channel*>(channelPtr));
        CHK_PTR_NULL(channelImpl);
        uint32_t rmtCkeId{0};
        CHK_PRT_RET(
            channelImpl->GetRmtCkeByIndex(ccuRepRemPostSem->GetSemIndex(), rmtCkeId) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR(
                "[CcuRepRemPostSem][%s] failed to get remote cke id, channelHandle[0x%llx].", __func__,
                ccuRepRemPostSem->GetChannel()),
            HCCL_E_UNAVAIL);

        SyncCKEInstr(
            instr++, rmtCkeId, dep.reserveCkeId, ccuRepRemPostSem->GetMask(), channelImpl->GetChannelId(), 0, 0, 0, 0,
            1);

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV1::CcuRepRecordSharedNotifyTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepRecordSharedNotify* ccuRepRecordSharedNotify, const TransDep& dep)
    {
        UNUSED(ccuKernel);
        CHK_PTR_NULL(ccuRepRecordSharedNotify);
        // 非本die时利用环回访问
        if (ccuRepRecordSharedNotify->GetNotify().DieId() != dep.dieId) {
            SyncCKEInstr(
                instr++, ccuRepRecordSharedNotify->GetNotify().Id(), dep.reserveCkeId,
                ccuRepRecordSharedNotify->GetMask(), dep.reserveChannalId[1], 0, 0, 0, 0, 1);
        } else {
            SetCKEInstr(
                instr++, ccuRepRecordSharedNotify->GetNotify().Id(), ccuRepRecordSharedNotify->GetMask(), 0, 0, 1);
        }

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV1::CcuRepAddTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepAdd* ccuRepAdd, [[maybe_unused]] const TransDep& dep)
    {
        UNUSED(ccuKernel);
        CHK_PTR_NULL(ccuRepAdd);
        switch (ccuRepAdd->GetSubType()) {
            case AddSubType::ADDR_PLUS_VAR_TO_ADDR: {
                LoadGSAXnInstr(
                    instr++, ccuRepAdd->GetAddrC().Id(), ccuRepAdd->GetAddrA().Id(), ccuRepAdd->GetVarB().Id());
                break;
            }
            case AddSubType::ADDR_PLUS_ADDR_TO_ADDR: {
                LoadGSAGSAInstr(
                    instr++, ccuRepAdd->GetAddrC().Id(), ccuRepAdd->GetAddrA().Id(), ccuRepAdd->GetAddrB().Id());
                break;
            }
            case AddSubType::VAR_PLUS_VAR_TO_VAR: {
                LoadXXInstr(instr++, ccuRepAdd->GetVarC().Id(), ccuRepAdd->GetVarA().Id(), ccuRepAdd->GetVarB().Id());
                break;
            }
            case AddSubType::SELF_ADD_ADDRESS: {
                LoadGSAXnInstr(
                    instr++, ccuRepAdd->GetAddrA().Id(), ccuRepAdd->GetAddrA().Id(), ccuRepAdd->GetVarB().Id());
                break;
            }
            case AddSubType::SELF_ADD_VARIABLE: {
                LoadXXInstr(instr++, ccuRepAdd->GetVarA().Id(), ccuRepAdd->GetVarA().Id(), ccuRepAdd->GetVarB().Id());
                break;
            }
            default: {
                HCCL_ERROR("Invalid Add, subType[%d]", static_cast<int>(ccuRepAdd->GetSubType()));
                return HCCL_E_PARA;
            }
        }

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV1::CcuRepAssignTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepAssign* ccuRepAssign, const TransDep& dep)
    {
        UNUSED(ccuKernel);
        CHK_PTR_NULL(ccuRepAssign);
        switch (ccuRepAssign->GetSubType()) {
            case AssignSubType::IMD_TO_VARIABLE: {
                LoadImdToXnInstr(instr++, ccuRepAssign->GetVarA().Id(), ccuRepAssign->GetImmed());
                break;
            }
            case AssignSubType::IMD_TO_ADDR: {
                LoadImdToGSAInstr(instr++, ccuRepAssign->GetAddrA().Id(), ccuRepAssign->GetImmed());
                break;
            }
            case AssignSubType::VAR_TO_ADDR: {
                LoadGSAXnInstr(instr++, ccuRepAssign->GetAddrA().Id(), dep.reserveGsaId, ccuRepAssign->GetVarA().Id());
                break;
            }
            case AssignSubType::ADDR_TO_ADDR: {
                LoadGSAGSAInstr(
                    instr++, ccuRepAssign->GetAddrB().Id(), ccuRepAssign->GetAddrA().Id(), dep.reserveGsaId);
                break;
            }
            case AssignSubType::VAR_TO_VAR: {
                LoadXXInstr(instr++, ccuRepAssign->GetVarB().Id(), ccuRepAssign->GetVarA().Id(), dep.reserveXnId);
                break;
            }
            default: {
                HCCL_ERROR("Invalid Assign, subType[%d]", static_cast<int>(ccuRepAssign->GetSubType()));
                return HCCL_E_PARA;
            }
        }

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV1::CcuRepMulTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepMul* ccuRepMul)
    {
        UNUSED(ccuKernel);
        UNUSED(instr);
        UNUSED(ccuRepMul);
        return HCCL_E_NOT_SUPPORT;
    }

    HcclResult CcuInsGeneratorV1::CcuRepSubTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepSub* ccuRepSub)
    {
        UNUSED(ccuKernel);
        UNUSED(instr);
        UNUSED(ccuRepSub);
        return HCCL_E_NOT_SUPPORT;
    }

    HcclResult CcuInsGeneratorV1::CcuRepAndTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepAnd* ccuRepAnd, const TransDep& dep)
    {
        UNUSED(ccuKernel);
        UNUSED(instr);
        UNUSED(ccuRepAnd);
        UNUSED(dep);
        return HCCL_E_NOT_SUPPORT;
    }

    HcclResult CcuInsGeneratorV1::CcuRepNotTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepNot* ccuRepNot, const TransDep& dep)
    {
        UNUSED(ccuKernel);
        UNUSED(instr);
        UNUSED(ccuRepNot);
        UNUSED(dep);
        return HCCL_E_NOT_SUPPORT;
    }

    HcclResult CcuInsGeneratorV1::CcuRepOrTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepOr* ccuRepOr, const TransDep& dep)
    {
        UNUSED(ccuKernel);
        UNUSED(instr);
        UNUSED(ccuRepOr);
        UNUSED(dep);
        return HCCL_E_NOT_SUPPORT;
    }

    HcclResult CcuInsGeneratorV1::CcuRepXorTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepXor* ccuRepXor, const TransDep& dep)
    {
        UNUSED(ccuKernel);
        UNUSED(instr);
        UNUSED(ccuRepXor);
        UNUSED(dep);
        return HCCL_E_NOT_SUPPORT;
    }

    HcclResult CcuInsGeneratorV1::CcuRepShLTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepShL* ccuRepShL, const TransDep& dep)
    {
        UNUSED(ccuKernel);
        UNUSED(instr);
        UNUSED(ccuRepShL);
        UNUSED(dep);
        return HCCL_E_NOT_SUPPORT;
    }

    HcclResult CcuInsGeneratorV1::CcuRepShRTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepShR* ccuRepShR, const TransDep& dep)
    {
        UNUSED(ccuKernel);
        UNUSED(instr);
        UNUSED(ccuRepShR);
        UNUSED(dep);
        return HCCL_E_NOT_SUPPORT;
    }

    HcclResult CcuInsGeneratorV1::CcuRepFuncBlockTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepFuncBlock* funcBlockPtr,
        const TransDep& dep, uint32_t step)
    {
        UNUSED(ccuKernel);
        CHK_PTR_NULL(funcBlockPtr);
        std::vector<CcuRepArg>& outArgs = funcBlockPtr->GetOutArgs();
        CcuRepReferenceManager* funcManager = funcBlockPtr->GetFuncManager();
        CHK_PTR_NULL(funcManager);
        if (step == 0) {
            // 函数入口为nop
            LoadImdToXnInstr(instr++, dep.reserveXnId, 0); // 向记录常量0的Xn再次赋值0作为nop操作
            curInstrId++;
        } else if (step == 1) {
            // 处理输出的参数
            uint32_t iOutArg = 0;
            for (uint32_t i = 0; i < outArgs.size(); i++) {
                if (outArgs[i].type == CcuArgType::VARIABLE) {
                    LoadXXInstr(
                        instr++, funcManager->GetFuncOut()[iOutArg++].Id(), outArgs[i].var.Id(), dep.reserveXnId);
                    curInstrId++;
                } else if (outArgs[i].type == CcuArgType::VARIABLE_LIST) {
                    for (uint32_t j = 0; j < outArgs[i].varList.size(); j++) {
                        LoadXXInstr(
                            instr++, funcManager->GetFuncOut()[iOutArg++].Id(), outArgs[i].varList[j].Id(),
                            dep.reserveXnId);
                        curInstrId++;
                    }
                }
            }

            // 返回调用处
            JumpInstr(instr++, funcManager->GetFuncRet(funcBlockPtr->GetCallLayer()).Id(), dep.reserveXnId, 1);
            curInstrId++;
        } else {
            HCCL_ERROR("Unsupported step[%d] for CcuRepFuncBlockTranslate", step);
            return HCCL_E_PARA;
        }
        return HcclResult::HCCL_SUCCESS;
    }

    void CcuInsGeneratorV1::LoadFuncCallInArgs(
        CcuInstr* instr, std::vector<CcuRepArg>& inArgs, std::vector<Variable>& formalIns, uint16_t reserveXnId) const
    {
        uint32_t idx = 0;
        for (uint32_t i = 0; i < inArgs.size(); i++) {
            if (inArgs[i].type == CcuArgType::VARIABLE) {
                LoadXXInstr(instr + idx, formalIns[idx].Id(), inArgs[i].var.Id(), reserveXnId);
                idx++;
            } else if (inArgs[i].type == CcuArgType::VARIABLE_LIST) {
                for (uint32_t j = 0; j < inArgs[i].varList.size(); j++) {
                    LoadXXInstr(instr + idx, formalIns[idx].Id(), inArgs[i].varList[j].Id(), reserveXnId);
                    idx++;
                }
            }
        }
    }

    void CcuInsGeneratorV1::LoadFuncCallOutArgs(
        CcuInstr* instr, uint32_t offset, std::vector<CcuRepArg>& outArgs, CcuRepReferenceManager* funcManager,
        uint16_t reserveXnId) const
    {
        uint32_t idx = 0;
        for (uint32_t i = 0; i < outArgs.size(); i++) {
            if (outArgs[i].type == CcuArgType::VARIABLE) {
                LoadXXInstr(
                    instr + offset + idx, outArgs[i].var.Id(), funcManager->GetFuncOut()[idx].Id(), reserveXnId);
                idx++;
            } else if (outArgs[i].type == CcuArgType::VARIABLE_LIST) {
                for (uint32_t j = 0; j < outArgs[i].varList.size(); j++) {
                    LoadXXInstr(
                        instr + offset + idx, outArgs[i].varList[j].Id(), funcManager->GetFuncOut()[idx].Id(),
                        reserveXnId);
                    idx++;
                }
            }
        }
    }

    HcclResult CcuInsGeneratorV1::CcuRepFuncCallTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepFuncCall* funcCallPtr, const TransDep& dep)
    {
        UNUSED(ccuKernel);
        (void)curInstrId;
        (void)instr;

        FuncCallContext ctx;
        CHK_RET(PrepareFuncCallContext(funcCallPtr, ctx));

        std::vector<CcuRepArg>& outArgs = funcCallPtr->GetOutArgs();
        std::vector<CcuRepArg>& inArgs = funcCallPtr->GetInArgs();
        uint32_t inArgCount = ctx.inArgCount;
        CcuInstr* funcInstr = ctx.instr;
        CcuRepReferenceManager* funcManager = ctx.funcManager;
        std::vector<Variable>& formalIns = ctx.formalIns;
        std::shared_ptr<CcuRepFuncBlock>& funcBlock = ctx.funcBlock;
        LoadFuncCallInArgs(funcInstr, inArgs, formalIns, dep.reserveXnId);

        uint32_t locId = 0;
        if (funcBlock != nullptr) {
            LoadImdToXnInstr(
                funcInstr + inArgCount + locId++, funcManager->GetFuncCall().Id(), funcBlock->StartInstrId());
        } else {
            LoadXXInstr(
                funcInstr + inArgCount + locId++, funcManager->GetFuncCall().Id(), funcCallPtr->GetFuncAddrVar().Id(),
                dep.reserveXnId);
        }

        LoadImdToXnInstr(
            funcInstr + inArgCount + locId++, funcManager->GetFuncRet(funcCallPtr->GetCallLayer()).Id(),
            funcCallPtr->StartInstrId() + inArgCount + 3); // 需要指向函数返回位置，为输入指令Id + 3
        JumpInstr(funcInstr + inArgCount + locId++, funcManager->GetFuncCall().Id(), dep.reserveXnId, 1);
        LoadImdToXnInstr(funcInstr + inArgCount + locId++, dep.reserveXnId, 0);

        uint32_t extraInstrNum = GetInstrCount(funcCallPtr->Type());
        LoadFuncCallOutArgs(funcInstr, inArgCount + extraInstrNum, outArgs, funcManager, dep.reserveXnId);
        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV1::CcuRepJumpTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepJump* jumpPtr, const TransDep& dep)
    {
        UNUSED(ccuKernel);
        (void)instr;
        (void)curInstrId;

        // 翻译直接跳转指令
        CHK_PTR_NULL(jumpPtr);
        std::shared_ptr<CcuRepJumpLabel> jumpLabel = jumpPtr->GetJumpLabel();
        CHK_PTR_NULL(jumpLabel);
        LoadImdToXnInstr(jumpPtr->GetInstr() + 0, jumpPtr->GetTargetInstrId().Id(), jumpLabel->StartInstrId());
        JumpInstr(jumpPtr->GetInstr() + 1, jumpPtr->GetTargetInstrId().Id(), dep.reserveXnId, 1);

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV1::CcuRepJumpNETranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepJumpNE* jumpNEPtr,
        [[maybe_unused]] const TransDep& dep)
    {
        UNUSED(ccuKernel);
        UNUSED(instr);
        UNUSED(curInstrId);
        CHK_PTR_NULL(jumpNEPtr);
        std::shared_ptr<CcuRepJumpLabel> jumpLabel = jumpNEPtr->GetJumpLabel();
        CHK_PTR_NULL(jumpLabel);
        LoadImdToXnInstr(jumpNEPtr->GetInstr() + 0, jumpNEPtr->GetTargetInstrId().Id(), jumpLabel->StartInstrId());
        JumpInstr(
            jumpNEPtr->GetInstr() + 1, jumpNEPtr->GetTargetInstrId().Id(), jumpNEPtr->GetCondition().Id(),
            jumpNEPtr->GetExpectedNum());

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV1::CcuRepJumpEQTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepJumpEQ* jumpEQPtr, const TransDep& dep)
    {
        UNUSED(ccuKernel);
        UNUSED(instr);
        UNUSED(curInstrId);

        CHK_PTR_NULL(jumpEQPtr);
        std::shared_ptr<CcuRepJumpLabel> jumpLabel = jumpEQPtr->GetJumpLabel();
        CHK_PTR_NULL(jumpLabel);
        uint32_t localInstrIndex = 0;
        CcuInstr* startInstr = jumpEQPtr->GetInstr();
        Variable& targetInstrId = jumpEQPtr->GetTargetInstrId();
        LoadImdToXnInstr(
            startInstr + localInstrIndex++, targetInstrId.Id(),
            jumpEQPtr->StartInstrId() + 4); // 需要指向NOP位置，为输入指令Id + 4
        JumpInstr(
            startInstr + localInstrIndex++, targetInstrId.Id(), jumpEQPtr->GetCondition().Id(),
            jumpEQPtr->GetExpectedNum());
        LoadImdToXnInstr(startInstr + localInstrIndex++, targetInstrId.Id(), jumpLabel->StartInstrId());
        JumpInstr(startInstr + localInstrIndex++, targetInstrId.Id(), dep.reserveXnId, 1);
        LoadImdToXnInstr(startInstr + localInstrIndex++, dep.reserveXnId, 0);

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV1::CcuRepJumpLETranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepJumpLE* jumpLEPtr,
        [[maybe_unused]] const TransDep& dep)
    {
        UNUSED(ccuKernel);
        (void)instr;
        (void)curInstrId;
        CHK_PTR_NULL(jumpLEPtr);
        HCCL_ERROR("Unsupported Jump type for CcuV1: %s", jumpLEPtr->Describe().c_str());
        return HCCL_E_NOT_SUPPORT;
    }

    HcclResult CcuInsGeneratorV1::CcuRepJumpGETranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepJumpGE* jumpGEPtr,
        [[maybe_unused]] const TransDep& dep)
    {
        UNUSED(ccuKernel);
        (void)instr;
        (void)curInstrId;
        CHK_PTR_NULL(jumpGEPtr);
        HCCL_ERROR("Unsupported Jump type for CcuV1: %s", jumpGEPtr->Describe().c_str());
        return HCCL_E_NOT_SUPPORT;
    }

    HcclResult CcuInsGeneratorV1::CcuRepJumpGTTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepJumpGT* jumpGTPtr,
        [[maybe_unused]] const TransDep& dep)
    {
        UNUSED(ccuKernel);
        (void)instr;
        (void)curInstrId;
        CHK_PTR_NULL(jumpGTPtr);
        HCCL_ERROR("Unsupported Jump type for CcuV1: %s", jumpGTPtr->Describe().c_str());
        return HCCL_E_NOT_SUPPORT;
    }

    HcclResult CcuInsGeneratorV1::CcuRepJumpLTTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepJumpLT* jumpLTPtr,
        [[maybe_unused]] const TransDep& dep)
    {
        UNUSED(ccuKernel);
        (void)instr;
        (void)curInstrId;
        CHK_PTR_NULL(jumpLTPtr);
        HCCL_ERROR("Unsupported Jump type for CcuV1: %s", jumpLTPtr->Describe().c_str());
        return HCCL_E_NOT_SUPPORT;
    }

    HcclResult CcuInsGeneratorV1::CcuRepLoopTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepLoop* loopPtr)
    {
        UNUSED(ccuKernel);
        UNUSED(curInstrId);
        CHK_PTR_NULL(loopPtr);
        auto loopBlock = loopPtr->GetLoopBlock();
        CHK_PTR_NULL(loopBlock);

        LoopInstr(
            instr++, loopBlock->StartInstrId(), loopBlock->StartInstrId() + loopBlock->InstrCount() - 1,
            loopPtr->GetLoopParam()->Id());
        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV1::LoadLoopCallArg(
        CcuInstr*& instr, const CcuRepArg& inArg, const CcuRepArg& blkArg, const TransDep& dep) const
    {
        switch (inArg.type) {
            case CcuArgType::VARIABLE:
                LoadXXInstr(instr++, blkArg.var.Id(), inArg.var.Id(), dep.reserveXnId);
                break;
            case CcuArgType::VARIABLE_LIST:
                if (inArg.varList.size() != blkArg.varList.size()) {
                    HCCL_ERROR(
                        "Mismatched Arg Size, inArg.varList.size[%zu], blkArg.varList.size[%zu]", inArg.varList.size(),
                        blkArg.varList.size());
                    return HCCL_E_PARA;
                }
                for (uint32_t j = 0; j < inArg.varList.size(); j++) {
                    LoadXXInstr(instr++, blkArg.varList[j].Id(), inArg.varList[j].Id(), dep.reserveXnId);
                }
                break;
            case CcuArgType::MEMORY:
                LoadAddrArg(instr, blkArg.mem, inArg.mem, dep);
                break;
            case CcuArgType::LOCAL_ADDR:
                LoadAddrArg(instr, blkArg.localAddr, inArg.localAddr, dep);
                break;
            case CcuArgType::REMOTE_ADDR:
                LoadAddrArg(instr, blkArg.remoteAddr, inArg.remoteAddr, dep);
                break;
            case CcuArgType::MEMORY_LIST:
                CHK_RET(LoadAddrListArg(instr, blkArg.memList, inArg.memList, dep));
                break;
            case CcuArgType::LOCAL_ADDR_LIST:
                CHK_RET(LoadAddrListArg(instr, blkArg.localAddrList, inArg.localAddrList, dep));
                break;
            case CcuArgType::REMOTE_ADDR_LIST:
                CHK_RET(LoadAddrListArg(instr, blkArg.remoteAddrList, inArg.remoteAddrList, dep));
                break;
            default:
                HCCL_ERROR("Mismatched Arg Type, inArg.type[%d]", static_cast<int>(inArg.type));
                return HCCL_E_PARA;
        }
        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV1::CcuRepLoopCallTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepLoopCall* loopCallPtr, const TransDep& dep)
    {
        UNUSED(ccuKernel);
        UNUSED(curInstrId);
        CHK_PTR_NULL(loopCallPtr);
        auto loopBlock = loopCallPtr->GetLoopBlock();
        CHK_PTR_NULL(loopBlock);
        std::vector<CcuRepArg>& inArgs = loopCallPtr->GetInArgs();

        for (uint32_t i = 0; i < inArgs.size(); i++) {
            const CcuRepArg& blkArg = loopBlock->GetArg(i);
            const CcuRepArg& inArg = inArgs[i];
            if (inArg.type != blkArg.type) {
                HCCL_ERROR(
                    "Mismatched Arg Type, inArg.type[%d], blkArg.type[%d]", static_cast<int>(inArg.type),
                    static_cast<int>(blkArg.type));
                return HCCL_E_PARA;
            }
            CHK_RET(LoadLoopCallArg(instr, inArg, blkArg, dep));
        }

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV1::CcuRepSetLoopTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepSetLoop* setLoopPtr)
    {
        UNUSED(ccuKernel);
        (void)curInstrId;
        CHK_PTR_NULL(setLoopPtr);
        LoadImdToXnInstr(instr++, setLoopPtr->loopParam.Id(), GetLoopParam(setLoopPtr->executor.Id(), 0, 0));
        LoadXXInstr(instr++, setLoopPtr->loopParam.Id(), setLoopPtr->loopParam.Id(), setLoopPtr->var.Id());
        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV1::CcuRepLoadTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepLoad* loadPtr, const TransDep& dep)
    {
        UNUSED(ccuKernel);
        (void)curInstrId;
        CHK_PTR_NULL(loadPtr);
        uint64_t varAddr = dep.xnBaseAddr[dep.dieId] + CCU_RESOURCE_XN_PER_SIZE * loadPtr->GetVar().Id();

        LoadImdToGSAInstr(instr++, dep.commGsa[0], varAddr);
        LoadImdToGSAInstr(instr++, dep.commGsa[1], loadPtr->GetAddr());
        LoadImdToXnInstr(instr++, dep.commXn[0], dep.ccuResSpaceTokenInfo, CCU_LOAD_TO_XN_SEC_INFO);
        LoadImdToXnInstr(instr++, dep.commXn[1], dep.memTokenInfo, CCU_LOAD_TO_XN_SEC_INFO);
        LoadImdToXnInstr(instr++, dep.commXn[2], CCU_RESOURCE_XN_PER_SIZE * loadPtr->GetNum());
        TransLocMemToLocMemInstr(
            instr++, dep.commGsa[0], dep.commXn[0], dep.commGsa[1], dep.commXn[1], dep.commXn[2],
            dep.reserveChannalId[0], dep.commSignal, loadPtr->GetMask(), 0, 0, 1, 1);
        SetCKEInstr(instr++, 0, 0, dep.commSignal, loadPtr->GetMask(), 1);

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV1::CcuRepLoadVarTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepLoadVar* loadVarPtr, const TransDep& dep)
    {
        UNUSED(ccuKernel);
        (void)curInstrId;
        CHK_PTR_NULL(loadVarPtr);
        uint64_t varAddr = dep.xnBaseAddr[dep.dieId] + CCU_RESOURCE_XN_PER_SIZE * loadVarPtr->GetVar().Id();
        LoadImdToGSAInstr(instr++, dep.commGsa[0], varAddr);
        LoadGSAXnInstr(instr++, dep.commGsa[1], dep.reserveGsaId, loadVarPtr->GetSrc().Id());
        LoadImdToXnInstr(instr++, dep.commXn[0], dep.ccuResSpaceTokenInfo, CCU_LOAD_TO_XN_SEC_INFO);
        LoadImdToXnInstr(instr++, dep.commXn[1], dep.memTokenInfo, CCU_LOAD_TO_XN_SEC_INFO);
        LoadImdToXnInstr(instr++, dep.commXn[2], CCU_RESOURCE_XN_PER_SIZE * loadVarPtr->GetNum());
        TransLocMemToLocMemInstr(
            instr++, dep.commGsa[0], dep.commXn[0], dep.commGsa[1], dep.commXn[1], dep.commXn[2],
            dep.reserveChannalId[0], dep.commSignal, loadVarPtr->GetMask(), 0, 0, 1, 1);
        SetCKEInstr(instr++, 0, 0, dep.commSignal, loadVarPtr->GetMask(), 1);

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV1::CcuRepLoadArgTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepLoadArg* loadArgPtr, const TransDep& dep)
    {
        UNUSED(ccuKernel);
        (void)curInstrId;
        CHK_PTR_NULL(loadArgPtr);
        if (dep.isFuncBlock) {
            // Xn(var) = Xn(loadXnId) + 0
            LoadXXInstr(instr++, loadArgPtr->GetVar().Id(), dep.loadXnId, dep.reserveXnId);
        } else {
            LoadSqeArgsToXnInstr(instr++, loadArgPtr->GetVar().Id(), loadArgPtr->GetArgId());
        }

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV1::CcuRepNopTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepNop* nopPtr, const TransDep& dep)
    {
        UNUSED(ccuKernel);
        (void)curInstrId;
        (void)nopPtr;
        LoadImdToXnInstr(instr++, dep.reserveXnId, 0);

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV1::CcuRepStoreTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepStore* storePtr, const TransDep& dep)
    {
        UNUSED(ccuKernel);
        (void)curInstrId;
        CHK_PTR_NULL(storePtr);
        uint64_t varAddr = dep.xnBaseAddr[dep.dieId] + CCU_RESOURCE_XN_PER_SIZE * storePtr->GetVar().Id();

        LoadImdToGSAInstr(instr++, dep.commGsa[0], storePtr->GetAddr());
        LoadImdToGSAInstr(instr++, dep.commGsa[1], varAddr);
        LoadImdToXnInstr(instr++, dep.commXn[0], dep.memTokenInfo, CCU_LOAD_TO_XN_SEC_INFO);
        LoadImdToXnInstr(instr++, dep.commXn[1], dep.ccuResSpaceTokenInfo, CCU_LOAD_TO_XN_SEC_INFO);
        LoadImdToXnInstr(instr++, dep.commXn[2], CCU_RESOURCE_XN_PER_SIZE * storePtr->GetNum());
        TransLocMemToLocMemInstr(
            instr++, dep.commGsa[0], dep.commXn[0], dep.commGsa[1], dep.commXn[1], dep.commXn[2],
            dep.reserveChannalId[0], dep.commSignal, storePtr->GetMask(), 0, 0, 1, 1);
        SetCKEInstr(instr++, 0, 0, dep.commSignal, storePtr->GetMask(), 1);

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV1::CcuRepStoreVarTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepStoreVar* storeVarPtr, const TransDep& dep)
    {
        UNUSED(ccuKernel);
        (void)curInstrId;
        CHK_PTR_NULL(storeVarPtr);
        uint64_t varAddr = dep.xnBaseAddr[dep.dieId] + CCU_RESOURCE_XN_PER_SIZE * storeVarPtr->GetVar().Id();

        LoadImdToGSAInstr(instr++, dep.commGsa[0], varAddr);
        LoadGSAXnInstr(instr++, dep.commGsa[1], dep.reserveGsaId, storeVarPtr->GetDst().Id());
        LoadImdToXnInstr(instr++, dep.commXn[0], dep.memTokenInfo, CCU_LOAD_TO_XN_SEC_INFO);
        LoadImdToXnInstr(instr++, dep.commXn[1], dep.ccuResSpaceTokenInfo, CCU_LOAD_TO_XN_SEC_INFO);
        LoadImdToXnInstr(instr++, dep.commXn[2], CCU_RESOURCE_XN_PER_SIZE * storeVarPtr->GetNum());
        TransLocMemToLocMemInstr(
            instr++, dep.commGsa[1], dep.commXn[0], dep.commGsa[0], dep.commXn[1], dep.commXn[2],
            dep.reserveChannalId[0], dep.commSignal, storeVarPtr->GetMask(), 0, 0, 1, 1);
        SetCKEInstr(instr++, 0, 0, dep.commSignal, storeVarPtr->GetMask(), 1);

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV1::CcuRepLoopGroupBundleTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepLoopGroupBundle* bundlePtr,
        const TransDep& dep)
    {
        UNUSED(ccuKernel);
        CHK_PTR_NULL(bundlePtr);
        const auto& loops = bundlePtr->GetLoops();

        LoadLoopGroupParams(instr, curInstrId, bundlePtr, dep);
        LoadLoopGroupBundleConfig(instr, curInstrId, bundlePtr);

        constexpr uint16_t kLoopEntryInstrOffset = 3;
        LoopGroupInstr(
            instr++, curInstrId + kLoopEntryInstrOffset, bundlePtr->GetParallelVar().Id(),
            bundlePtr->GetOffsetParam().Id(), 0); // 向后3条为loop指令
        curInstrId++;

        uint16_t loopCount = static_cast<uint16_t>(loops.size());
        uint16_t jumpTargetInstrId = curInstrId + 2 + loopCount + 1; // 跳转目标为向后2条+loop指令条数+额外1条
        LoadImdToXnInstr(instr++, dep.reserveXnId, jumpTargetInstrId);
        curInstrId++;
        JumpInstr(instr++, dep.reserveXnId, dep.reserveXnId, 1);
        curInstrId++;

        for (const auto& loop : loops) {
            const auto& block = loop.repLoopBlock;
            CHK_PTR_NULL(block);
            LoopInstr(
                instr++, block->StartInstrId(), block->StartInstrId() + block->InstrCount() - 1,
                loop.loopParamVar.Id());
            curInstrId++;
        }

        LoadImdToXnInstr(instr++, dep.reserveXnId, 0);
        curInstrId++;

        return HcclResult::HCCL_SUCCESS;
    }

    void CcuInsGeneratorV1::LoadLoopGroupParams(
        CcuInstr*& instr, uint16_t& curInstrId, const CcuRepLoopGroupBundle* bundlePtr, const TransDep& dep) const
    {
        const auto& loops = bundlePtr->GetLoops();
        for (const auto& loop : loops) {
            if (loop.layout == CcuRepLoopGroupBundle::Layout::Config) {
                uint64_t lpImm = GetLoopParam(loop.executor.Id(), loop.config.addrOffset, loop.config.iterNum);
                LoadImdToXnInstr(instr++, loop.loopParamVar.Id(), lpImm);
                curInstrId++;
            } else {
                uint64_t ctxImm = static_cast<uint64_t>(loop.executor.Id()) << 45; // 左移45位到对应字段然后相加
                LoadImdToXnInstr(instr++, dep.reserveXnId, ctxImm);
                curInstrId++;
                LoadXXInstr(instr++, loop.loopParamVar.Id(), loop.loopParamVar.Id(), dep.reserveXnId);
                curInstrId++;
            }
        }
    }

    void CcuInsGeneratorV1::LoadLoopGroupBundleConfig(
        CcuInstr*& instr, uint16_t& curInstrId, const CcuRepLoopGroupBundle* bundlePtr) const
    {
        if (bundlePtr->GetLayout() != CcuRepLoopGroupBundle::Layout::Config) {
            return;
        }
        uint64_t parallelImm = GetParallelParam(
            bundlePtr->GetConfig().cloneNum, bundlePtr->GetRepeatLoopIdx(), bundlePtr->GetTotalLoopNum());
        LoadImdToXnInstr(instr++, bundlePtr->GetParallelVar().Id(), parallelImm);
        curInstrId++;

        uint64_t offsetImm = ::hcomm::CcuRep::GetOffsetParam(
            bundlePtr->GetConfig().addrOffset, bundlePtr->GetConfig().ccuBufferOffset,
            bundlePtr->GetConfig().eventOffset);
        LoadImdToXnInstr(instr++, bundlePtr->GetOffsetParam().Id(), offsetImm);
        curInstrId++;
    }

    uint16_t CcuInsGeneratorV1::CcuRepLoopGroupBundleInstrCount(const CcuRepLoopGroupBundle* bundlePtr) const
    {
        if (bundlePtr == nullptr) {
            Hccl::THROW<Hccl::CcuApiException>("[%s] bundlePtr is nullptr", __func__);
        }
        const auto& loops = bundlePtr->GetLoops();
        const uint16_t loopCount = static_cast<uint16_t>(loops.size());
        uint16_t varBasedLoopCount = 0;
        for (const auto& loop : loops) {
            if (loop.layout != CcuRepLoopGroupBundle::Layout::Config) {
                varBasedLoopCount++;
            }
        }
        // 每 loop：config 1 条载入 / var 2 条；config bundle 额外 2 条(parallel+offset)；+loopgroup1 +跳过2 +每loop
        // Loop1 +收尾1
        const uint16_t groupOffset = (loopCount - varBasedLoopCount) + (varBasedLoopCount * 2)
                                     + (bundlePtr->GetLayout() == CcuRepLoopGroupBundle::Layout::Config ? 2 : 0);
        return groupOffset + 1 + 2 + loopCount + 1; // 跳过2条
    }

} // namespace CcuRep
} // namespace hcomm
