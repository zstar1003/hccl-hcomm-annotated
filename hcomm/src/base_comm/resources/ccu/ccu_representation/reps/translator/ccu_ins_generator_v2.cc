/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_ins_generator_v2.h"
#include "hcomm_c_adpt.h"
#include "ccu_rep_base_v1.h"
#include "../../../../endpoint_pairs/channels/ccu/ccu_urma_channel.h"
#include "ccu_api_exception.h"
#include "ccu_assist_v1.h"
#include "hcom_common.h"
#include "../../../ccu_device/ccu_res_specs.h"

#include <iostream>

namespace hcomm {

namespace CcuRep {
#define UNUSED(x) static_cast<void>(x)
    // V2中所有variable的Id()需要换成IdWithFlag()
    namespace {
        constexpr uint8_t URMA_DMA_OP_READ = 0x6;  // UB URMA WQEBB opcode: read
        constexpr uint8_t URMA_DMA_OP_WRITE = 0x3; // UB URMA WQEBB opcode: write
        constexpr uint8_t URMA_DMA_OP_WRITE_WITH_CNT_INC
            = 0x1A;                                   // UB URMA WQEBB opcode: write with cascade counter increase
        constexpr uint8_t SPLIT_MODE_PACKET = 1;      // TransMem按包为单位切分
        constexpr uint32_t REL_JMP_INSTR_NUM = 9;     // RelJmp生成的指令数
        constexpr uint32_t FUNC_CALL_JMP_OFFSET = 11; // RelJmp+Jump+Nop
        constexpr uint32_t FUNC_CALL_RET_OFFSET = 12; // FUNC_CALL_JMP_OFFSET + 1
        // loopParamVar 布局：iterNum[12:0] gsaOffset[44:13] ctxId[52:45]
        constexpr uint64_t LOOP_ITER_NUM_MASK = 0x1FFFULL;
        constexpr uint32_t LOOP_GSA_OFFSET_SHIFT = 13;
        constexpr uint64_t LOOP_GSA_OFFSET_MASK = 0xFFFFFFFFULL;
        constexpr uint64_t LOOP_FIELD_MASK = 0x7FULL;

        template <typename T>
        void LoadAddrArg(CcuInstr*& instr, const T& dst, const T& src)
        {
            CcuV2::Assign(instr++, dst.addr.Id(), src.addr.Id());
            CcuV2::Assign(instr++, dst.token.Id(), src.token.Id());
        }

        template <typename T>
        HcclResult LoadAddrListArg(CcuInstr*& instr, const std::vector<T>& dst, const std::vector<T>& src)
        {
            if (src.size() != dst.size()) {
                HCCL_ERROR("Mismatched Arg Size: srcSize[%u], dstSize[%u]", src.size(), dst.size());
                return HCCL_E_PARA;
            }
            for (uint32_t j = 0; j < src.size(); j++) {
                LoadAddrArg(instr, dst[j], src[j]);
            }
            return HcclResult::HCCL_SUCCESS;
        }
        HcclResult GetUrmaChannel(ChannelHandle channelHandle, CcuUrmaChannel*& channelImpl)
        {
            void* channelPtr{nullptr};
            CHK_RET(static_cast<HcclResult>(HcommChannelGet(channelHandle, &channelPtr)));
            channelImpl = dynamic_cast<CcuUrmaChannel*>(static_cast<Channel*>(channelPtr));
            CHK_PTR_NULL(channelImpl);
            return HcclResult::HCCL_SUCCESS;
        }

        HcclResult GetRmtToken(const CcuUrmaChannel* channelImpl, uint64_t& rmtToken)
        {
            uint32_t rmtTokenId{0};
            uint32_t rmtTokenValue{0};
            CHK_PTR_NULL(channelImpl);
            CHK_RET(channelImpl->GetRmtCcuBufferTokenInfo(rmtTokenId, rmtTokenValue));
            rmtToken = CcuRep::GetToken(rmtTokenId, rmtTokenValue, 1);
            return HcclResult::HCCL_SUCCESS;
        }
    } // namespace

    uint32_t CcuInsGeneratorV2::GetInstrCount(CcuRepType repType)
    {
        if (repTypeInstrCount.find(repType) == repTypeInstrCount.end()) {
            Hccl::THROW<Hccl::CcuApiException>("[%s] Unsupported repType[%d]", __func__, repType);
        }
        return repTypeInstrCount[repType];
    }

    HcclResult CcuInsGeneratorV2::CcuRepBufLocReadTranslate(
        [[maybe_unused]] CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepBufLocRead* repBufLocRead, const TransDep& dep)
    {
        CHK_PTR_NULL(repBufLocRead);
        CcuV2::CacheConfig config = {};

        TransLocMemToLocMS(
            instr++, repBufLocRead->GetDst().Id(), repBufLocRead->GetSrc().addr.Id(),
            repBufLocRead->GetSrc().token.Id(), repBufLocRead->GetLen().Id(), dep.reserveXnId,
            repBufLocRead->GetSem().Id(), repBufLocRead->GetMask(), config);

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV2::CcuRepBufLocWriteTranslate(
        [[maybe_unused]] CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepBufLocWrite* repBufLocWrite, const TransDep& dep)
    {
        CHK_PTR_NULL(repBufLocWrite);
        CcuV2::CacheConfig config = {};

        TransLocMSToLocMem(
            instr++, repBufLocWrite->GetDst().addr.Id(), repBufLocWrite->GetDst().token.Id(),
            repBufLocWrite->GetSrc().Id(), repBufLocWrite->GetLen().Id(), dep.reserveXnId,
            repBufLocWrite->GetSem().Id(), repBufLocWrite->GetMask(), config);

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV2::CcuRepBufReadTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepBufRead* repBufRead, const TransDep& dep)
    {
        CHK_PTR_NULL(repBufRead);
        CHK_PTR_NULL(ccuKernel);
        CcuUrmaChannel* channelImpl{nullptr};
        CHK_RET(GetUrmaChannel(repBufRead->GetChannel(), channelImpl));

        CcuV2::TransMemNotifyInfo notify = {};
        CcuV2::TransMemReduceInfo reduce = {};
        CcuV2::TransMemConfig config = {};
        config.dmaOpCode = URMA_DMA_OP_READ; // UB URMA WQEBB opcode: read
        config.src_mode = 1;
        config.dst_mode = 0;
        config.msIdMode = 1;
        uint32_t channelId = channelImpl->GetChannelId();
        const auto& constValue2VarMap = ccuKernel->GetConstValue2VarMap();
        TransMem(
            instr++, repBufRead->GetDst().Id(), constValue2VarMap.at(dep.ccuResSpaceTokenInfo).Id(),
            repBufRead->GetSrc().addr.Id(), repBufRead->GetSrc().token.Id(), repBufRead->GetLen().Id(),
            constValue2VarMap.at(channelId).Id(), notify, reduce, config, repBufRead->GetSem().Id(),
            repBufRead->GetMask());

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV2::CcuRepWriteTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepWrite* repWrite)
    {
        CHK_PTR_NULL(repWrite);
        CHK_PTR_NULL(ccuKernel);
        CcuUrmaChannel* channelImpl{nullptr};
        CHK_RET(GetUrmaChannel(repWrite->GetChannel(), channelImpl));
        CcuV2::TransMemNotifyInfo notify = {};
        CcuV2::TransMemReduceInfo reduce = {};
        CcuV2::TransMemConfig config = {};
        config.dmaOpCode = URMA_DMA_OP_WRITE; // UB URMA WQEBB opcode: write
        config.src_mode = 0;
        config.dst_mode = 1;

        if (repWrite->GetReduceFlag() == 1) {
            config.udfEnable = 1;
            reduce.udfType = 0;
            reduce.reduceDataType = repWrite->GetDataType();
            reduce.reduceOpCode = repWrite->GetOpType();
        }
        uint32_t channelId = channelImpl->GetChannelId();
        const auto& constValue2VarMap = ccuKernel->GetConstValue2VarMap();

        uint64_t rmtCcuResToken{};
        CHK_RET(GetRmtToken(channelImpl, rmtCcuResToken));
        notify.xntId = constValue2VarMap.at(rmtCcuResToken).Id();
        TransMem(
            instr++, repWrite->GetRem().addr.Id(), repWrite->GetRem().token.Id(), repWrite->GetLoc().addr.Id(),
            repWrite->GetLoc().token.Id(), repWrite->GetLen().Id(), constValue2VarMap.at(channelId).Id(), notify,
            reduce, config, repWrite->GetSem().Id(), repWrite->GetMask());

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV2::CcuRepReadTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepRead* repRemMem)
    {
        CHK_PTR_NULL(repRemMem);
        CHK_PTR_NULL(ccuKernel);
        CcuUrmaChannel* channelImpl{nullptr};
        CHK_RET(GetUrmaChannel(repRemMem->GetChannel(), channelImpl));

        CcuV2::TransMemNotifyInfo notify = {};
        CcuV2::TransMemReduceInfo reduce = {};
        CcuV2::TransMemConfig config = {};
        config.dmaOpCode = URMA_DMA_OP_READ; // UB URMA WQEBB opcode: read

        if (repRemMem->GetReduceFlag() == 1) {
            config.udfEnable = 1;
            reduce.udfType = 0;
            reduce.reduceDataType = repRemMem->GetDataType();
            reduce.reduceOpCode = repRemMem->GetOpType();
        }
        uint32_t channelId = channelImpl->GetChannelId();
        const auto& constValue2VarMap = ccuKernel->GetConstValue2VarMap();
        TransMem(
            instr++, repRemMem->GetLoc().addr.Id(), repRemMem->GetLoc().token.Id(), repRemMem->GetRem().addr.Id(),
            repRemMem->GetRem().token.Id(), repRemMem->GetLen().Id(), constValue2VarMap.at(channelId).Id(), notify,
            reduce, config, repRemMem->GetSem().Id(), repRemMem->GetMask());

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV2::CcuRepRemMemTranslate(
        [[maybe_unused]] CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepRemMem* repRemMem)
    {
        CHK_PTR_NULL(repRemMem);
        CcuUrmaChannel* channelImpl{nullptr};
        CHK_RET(GetUrmaChannel(repRemMem->GetChannel(), channelImpl));

        uint64_t addr{0};
        uint32_t size{0}, tokenId{0}, tokenValue{0};
        CHK_PRT_RET(
            channelImpl->GetRmtBuffer(addr, size, tokenId, tokenValue) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR(
                "[CcuRepRemMem][%s] failed to get remote buffer, channelHandle[0x%llx].", __func__,
                repRemMem->GetChannel()),
            HCCL_E_INTERNAL); // 当前认为channel只持有一个buffer

        auto tokenInfo = GetToken(tokenId, tokenValue, 1);

        CcuV2::LoadImdToXn(instr++, repRemMem->GetRem().addr.Id(), addr);
        CcuV2::LoadImdToXn(instr++, repRemMem->GetRem().token.Id(), tokenInfo);

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV2::CcuRepLocCpyTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepLocCpy* ccuRepLocCpy, const TransDep& dep)
    {
        CHK_PTR_NULL(ccuRepLocCpy);
        if (ccuRepLocCpy->GetReduceFlag() == 0 && ccuRepLocCpy->GetUseCcuBuffer() == true) {
            CcuV2::CacheConfig cacheConfig{0x0, 0x0};
            CcuV2::TransLocMemToLocMem(
                instr++, ccuRepLocCpy->GetDstAddrId(), ccuRepLocCpy->GetDstTokenId(), ccuRepLocCpy->GetSrcAddrId(),
                ccuRepLocCpy->GetSrcTokenId(), ccuRepLocCpy->GetLenId(), ccuRepLocCpy->GetFirstBufId(),
                ccuRepLocCpy->GetUsedBufNum(), ccuRepLocCpy->GetSemId(), ccuRepLocCpy->GetMask(), cacheConfig,
                cacheConfig);
        } else {
            // 使用旧接口或带规约场景，都走环回
            CcuV2::TransMemNotifyInfo notify = {};
            CcuV2::TransMemReduceInfo reduce = {};
            CcuV2::TransMemConfig config = {};
            config.dmaOpCode = URMA_DMA_OP_WRITE; // UB URMA WQEBB opcode: write
            if (ccuRepLocCpy->GetReduceFlag() == 1) {
                config.udfEnable = 1;
                reduce.udfType = 0;
                reduce.reduceDataType = ccuRepLocCpy->GetDataType();
                reduce.reduceOpCode = ccuRepLocCpy->GetOpType();
            }
            const auto& constValue2VarMap = ccuKernel->GetConstValue2VarMap();
            CcuV2::TransMem(
                instr++, ccuRepLocCpy->GetDstAddrId(), ccuRepLocCpy->GetDstTokenId(), ccuRepLocCpy->GetSrcAddrId(),
                ccuRepLocCpy->GetSrcTokenId(), ccuRepLocCpy->GetLenId(),
                constValue2VarMap.at(dep.reserveChannalId[0]).Id(), notify, reduce, config, ccuRepLocCpy->GetSemId(),
                ccuRepLocCpy->GetMask());
        }

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV2::CcuRepBufWriteTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepBufWrite* ccuRepBufWrite, const TransDep& dep)
    {
        CHK_PTR_NULL(ccuRepBufWrite);
        CHK_PTR_NULL(ccuKernel);
        CcuV2::TransMemNotifyInfo notify = {};
        CcuV2::TransMemReduceInfo reduce = {};
        CcuV2::TransMemConfig config = {};
        config.dmaOpCode = URMA_DMA_OP_WRITE; // UB URMA WQEBB opcode: write
        config.src_mode = 0;                  // 不偏移
        config.dst_mode = 1;                  // 偏移 256K
        config.msIdMode = 1;                  // 使用msId模式
        CcuUrmaChannel* channelImpl{nullptr};
        CHK_RET(GetUrmaChannel(ccuRepBufWrite->GetChannel(), channelImpl));

        uint32_t channelId = channelImpl->GetChannelId();
        const auto& constValue2VarMap = ccuKernel->GetConstValue2VarMap();

        uint64_t rmtCcuResToken{};
        CHK_RET(GetRmtToken(channelImpl, rmtCcuResToken));
        notify.xntId = constValue2VarMap.at(rmtCcuResToken).Id();
        HCCL_INFO(
            "DstTokenXnId[%u], NotifyTokenXnId[%u]", constValue2VarMap.at(dep.ccuResSpaceTokenInfo).Id(), notify.xntId);
        TransMem(
            instr++, ccuRepBufWrite->GetDst().addr.Id(), ccuRepBufWrite->GetDst().token.Id(),
            ccuRepBufWrite->GetSrc().Id(), constValue2VarMap.at(dep.ccuResSpaceTokenInfo).Id(),
            ccuRepBufWrite->GetLen().Id(), constValue2VarMap.at(channelId).Id(), notify, reduce, config,
            ccuRepBufWrite->GetSem().Id(), ccuRepBufWrite->GetMask());

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV2::CcuRepBufReduceTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepBufReduce* ccuRepBufReduce)
    {
        UNUSED(ccuKernel);
        CHK_PTR_NULL(ccuRepBufReduce);
        if (ccuRepBufReduce->GetCount() > CCU_REDUCE_MAX_MS || ccuRepBufReduce->GetMem().size() > CCU_REDUCE_MAX_MS) {
            HCCL_ERROR(
                "count[%u] and mem size[%zu] must be less than %u", ccuRepBufReduce->GetCount(),
                ccuRepBufReduce->GetMem().size(), CCU_REDUCE_MAX_MS);
            return HCCL_E_PARA;
        }
        if (ccuRepBufReduce->GetCount() < CCU_REDUCE_MIN_MS) {
            HCCL_ERROR("count[%u] must be at least %u", ccuRepBufReduce->GetCount(), CCU_REDUCE_MIN_MS);
            return HCCL_E_PARA;
        }

        uint16_t msId[CCU_REDUCE_MAX_MS] = {0};
        const auto& mem = ccuRepBufReduce->GetMem();
        for (uint16_t i = 0; i < mem.size(); i++) {
            msId[i] = mem[i].Id();
        }

        if (ccuRepBufReduce->GetOpType() == CCU_REDUCE_SUM) {
            if (ccuRepBufReduce->GetOutputDataType() == 1) { // 1是fp16
                CcuV2::ReduceAdd(
                    instr++, msId, ccuRepBufReduce->GetCount(), ccuRepBufReduce->GetOutputDataType(),
                    ccuRepBufReduce->GetDataType(), ccuRepBufReduce->GetSem().Id(), ccuRepBufReduce->GetMask(),
                    ccuRepBufReduce->GetXnIdLength().Id());
            } else if (ccuRepBufReduce->GetOutputDataType() == 2) { // 2是bf16
                CcuV2::ReduceAdd(
                    instr++, msId, ccuRepBufReduce->GetCount(), ccuRepBufReduce->GetOutputDataType(),
                    ccuRepBufReduce->GetDataType(), ccuRepBufReduce->GetSem().Id(), ccuRepBufReduce->GetMask(),
                    ccuRepBufReduce->GetXnIdLength().Id());
            } else {
                CcuV2::ReduceAdd(
                    instr++, msId, ccuRepBufReduce->GetCount(), 0, ccuRepBufReduce->GetDataType(),
                    ccuRepBufReduce->GetSem().Id(), ccuRepBufReduce->GetMask(), ccuRepBufReduce->GetXnIdLength().Id());
            }
        } else if (ccuRepBufReduce->GetOpType() == CCU_REDUCE_MAX) {
            CcuV2::ReduceMax(
                instr++, msId, ccuRepBufReduce->GetCount(), ccuRepBufReduce->GetDataType(),
                ccuRepBufReduce->GetSem().Id(), ccuRepBufReduce->GetMask(), ccuRepBufReduce->GetXnIdLength().Id());
        } else if (ccuRepBufReduce->GetOpType() == CCU_REDUCE_MIN) {
            CcuV2::ReduceMin(
                instr++, msId, ccuRepBufReduce->GetCount(), ccuRepBufReduce->GetDataType(),
                ccuRepBufReduce->GetSem().Id(), ccuRepBufReduce->GetMask(), ccuRepBufReduce->GetXnIdLength().Id());
        } else {
            HCCL_ERROR("[CcuRepBufReduceTranslate] Unsupported opType[%d]", ccuRepBufReduce->GetOpType());
            return HCCL_E_PARA;
        }

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV2::CcuRepLocRecordEventTranslate(
        [[maybe_unused]] CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepLocRecordEvent* ccuRepLocRecordEvent)
    {
        CHK_PTR_NULL(ccuRepLocRecordEvent);
        CcuV2::SetCKE(instr++, ccuRepLocRecordEvent->GetEvent().Id(), ccuRepLocRecordEvent->GetMask(), 0, 0, 1);

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV2::CcuRepLocWaitEventTranslate(
        [[maybe_unused]] CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepLocWaitEvent* ccuRepLocWaitEvent)
    {
        CHK_PTR_NULL(ccuRepLocWaitEvent);
        // 需要profiling的使用SetCKEInstr, 否则使用ClearCKEInstr
        if (ccuRepLocWaitEvent->GetIsProfiling()) {
            CcuV2::SetCKE(instr++, 0, 0, ccuRepLocWaitEvent->GetEvent().Id(), ccuRepLocWaitEvent->GetMask(), 1);
        } else {
            CcuV2::ClearCKE(instr++, 0, 0, ccuRepLocWaitEvent->GetEvent().Id(), ccuRepLocWaitEvent->GetMask(), 1);
        }

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV2::CcuRepLocWaitNotifyTranslate(
        [[maybe_unused]] CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepLocWaitNotify* ccuRepLocWaitNotify)
    {
        CHK_PTR_NULL(ccuRepLocWaitNotify);
        if (ccuRepLocWaitNotify->GetIsProfiling()) {
            CcuV2::SetCKE(instr++, 0, 0, ccuRepLocWaitNotify->GetNotify().Id(), ccuRepLocWaitNotify->GetMask(), 1);
        } else {
            CcuV2::ClearCKE(instr++, 0, 0, ccuRepLocWaitNotify->GetNotify().Id(), ccuRepLocWaitNotify->GetMask(), 1);
        }

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV2::CcuRepRemWaitSemTranslate(
        [[maybe_unused]] CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepRemWaitSem* ccuRepRemWaitSem)
    {
        CHK_PTR_NULL(ccuRepRemWaitSem);
        CcuUrmaChannel* channelImpl{nullptr};
        CHK_RET(GetUrmaChannel(ccuRepRemWaitSem->GetChannel(), channelImpl));

        uint32_t locCkeId{0};
        CHK_PRT_RET(
            channelImpl->GetLocCkeByIndex(ccuRepRemWaitSem->GetSemIndex(), locCkeId) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR(
                "[CcuRepRemWaitSem][%s] failed to get loc cke id, channelHandle[0x%llx], semIndex[%u].", __func__,
                ccuRepRemWaitSem->GetChannel(), ccuRepRemWaitSem->GetSemIndex()),
            HCCL_E_INTERNAL);

        // 需要profiling的使用SetCKEInstr, 否则使用ClearCKEInstr
        if (ccuRepRemWaitSem->GetIsProfiling()) {
            CcuV2::SetCKE(instr++, 0, 0, locCkeId, ccuRepRemWaitSem->GetMask(), 1);
        } else {
            CcuV2::ClearCKE(instr++, 0, 0, locCkeId, ccuRepRemWaitSem->GetMask(), 1);
        }

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV2::CcuRepRemPostVarTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepRemPostVar* ccuRepRemPostVar)
    {
        CHK_PTR_NULL(ccuRepRemPostVar);
        CHK_PTR_NULL(ccuKernel);
        CcuUrmaChannel* channelImpl{nullptr};
        CHK_RET(GetUrmaChannel(ccuRepRemPostVar->GetChannel(), channelImpl));

        uint32_t channelId = channelImpl->GetChannelId();
        uint64_t rmtSignalAddr{0};
        uint64_t rmtVarAddr{0};
        CHK_PRT_RET(
            channelImpl->GetRmtSignalAddrByIndex(ccuRepRemPostVar->GetSemIndex(), rmtSignalAddr)
                != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR(
                "[CcuInsGeneratorV2][%s] failed to get remote signal addr, channelHandle[0x%llx].", __func__,
                ccuRepRemPostVar->GetChannel()),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(
            channelImpl->GetRmtVarAddrByIndex(ccuRepRemPostVar->GetParamIndex(), rmtVarAddr)
                != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR(
                "[CcuInsGeneratorV2][%s] failed to get remote var addr, channelHandle[0x%llx].", __func__,
                ccuRepRemPostVar->GetChannel()),
            HCCL_E_INTERNAL);
        uint64_t rmtToken{};
        CHK_RET(GetRmtToken(channelImpl, rmtToken));

        const auto& constValue2VarMap = ccuKernel->GetConstValue2VarMap();
        CcuV2::TransMemNotifyInfo notify = {};
        notify.xnId = constValue2VarMap.at(rmtSignalAddr).Id();
        notify.xntId = constValue2VarMap.at(rmtToken).Id();
        notify.value = ccuRepRemPostVar->GetMask();

        SyncWtX(
            instr++, constValue2VarMap.at(rmtVarAddr).Id(), constValue2VarMap.at(rmtToken).Id(),
            ccuRepRemPostVar->GetParam().Id(), constValue2VarMap.at(channelId).Id(), notify, 0, 0);

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV2::CcuRepRemPostSemTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepRemPostSem* ccuRepRemPostSem,
        [[maybe_unused]] const TransDep& dep)
    {
        CHK_PTR_NULL(ccuRepRemPostSem);
        CHK_PTR_NULL(ccuKernel);
        CcuUrmaChannel* channelImpl{nullptr};
        CHK_RET(GetUrmaChannel(ccuRepRemPostSem->GetChannel(), channelImpl));

        uint64_t rmtSignalAddr{0};
        uint32_t channelId = channelImpl->GetChannelId();
        CHK_PRT_RET(
            channelImpl->GetRmtSignalAddrByIndex(ccuRepRemPostSem->GetSemIndex(), rmtSignalAddr)
                != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR(
                "[CcuInsGeneratorV2][%s] failed to get remote signal addr, channelHandle[0x%llx].", __func__,
                ccuRepRemPostSem->GetChannel()),
            HCCL_E_INTERNAL);
        uint64_t rmtToken{};
        CHK_RET(GetRmtToken(channelImpl, rmtToken));

        const auto& constValue2VarMap = ccuKernel->GetConstValue2VarMap();
        CcuV2::TransMemNotifyInfo notify = {};
        notify.xnId = constValue2VarMap.at(rmtSignalAddr).Id();
        notify.xntId = constValue2VarMap.at(rmtToken).Id();
        notify.value = ccuRepRemPostSem->GetMask();

        CcuV2::SyncWtX(instr++, notify, constValue2VarMap.at(channelId).Id(), 0, 0);

        return HcclResult::HCCL_SUCCESS;
    }

    constexpr uint16_t CCU_RESOURCE_CKE_MASK_LENGTH = 2;
    HcclResult CcuInsGeneratorV2::CcuRepRecordSharedNotifyTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepRecordSharedNotify* ccuRepRecordSharedNotify, const TransDep& dep)
    {
        CHK_PTR_NULL(ccuRepRecordSharedNotify);
        CHK_PTR_NULL(ccuKernel);
        if (ccuRepRecordSharedNotify->GetNotify().DieId() != dep.dieId) {
            // 计算对端cke地址，对端Xn基地址 + 256K + (ccuRepRecordSharedNotify->GetNotify().Id() * 8)
            uint64_t ckeAddr = dep.xnBaseAddr[ccuRepRecordSharedNotify->GetNotify().DieId()]
                               + CCU_RESOURCE_XN_V2_RESERVE_SIZE
                               + (ccuRepRecordSharedNotify->GetNotify().Id() * CCU_RESOURCE_CKE_PER_SIZE);
            const auto& constValue2VarMap = ccuKernel->GetConstValue2VarMap();
            CcuV2::CacheConfig cacheConfig{0x0, 0x0};
            CcuV2::StoreXToMem(
                instr++,
                constValue2VarMap.at(ckeAddr).Id(),                             // 对端cke xn
                constValue2VarMap.at(dep.memTokenInfo).Id(),                    // 对端cke xn token
                constValue2VarMap.at(ccuRepRecordSharedNotify->GetMask()).Id(), // cke mask
                constValue2VarMap.at(CCU_RESOURCE_CKE_MASK_LENGTH).Id(),        // cke mask，2字节
                cacheConfig,
                dep.commSignal, // store完成后同步的cke
                1);             // store完成后同步的cke mask
            CcuV2::SetCKE(instr++, 0, 0, dep.commSignal, 1, 1);
        } else {
            CcuV2::SetCKE(
                instr++, ccuRepRecordSharedNotify->GetNotify().Id(), ccuRepRecordSharedNotify->GetMask(), 0, 0, 1);
        }

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV2::CcuRepAddTranslate(
        [[maybe_unused]] CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepAdd* ccuRepAdd, const TransDep& dep)
    {
        HCCL_INFO("Use table-driven translate for add");
        UNUSED(dep);
        CHK_PTR_NULL(ccuRepAdd);

        // {isImmed, isSelf, dstVar, srcAVar, srcBAddr} — 按 AddSubType 枚举顺序
        static constexpr struct {
            bool isImmed;
            bool isSelf;
            bool dstVar;
            bool srcAVar;
            bool srcBAddr;
        } table[] = {
            {false, false, false, false, false}, // INVALID
            {false, false, false, false, false}, // ADDR_PLUS_VAR_TO_ADDR
            {false, false, false, false, true},  // ADDR_PLUS_ADDR_TO_ADDR
            {false, false, true, true, false},   // VAR_PLUS_VAR_TO_VAR
            {false, true, false, false, false},  // SELF_ADD_ADDRESS
            {false, true, true, true, false},    // SELF_ADD_VARIABLE
            {true, false, true, true, false},    // VAR_PLUS_IMMED_TO_VAR
            {true, false, false, false, false},  // ADDR_PLUS_IMMED_TO_ADDR
            {false, false, false, true, false},  // VAR_PLUS_VAR_TO_ADDR
            {true, true, false, false, false},   // SELF_ADD_IMMED_ADDRESS
            {true, true, true, true, false},     // SELF_ADD_IMMED_VARIABLE
            {true, false, false, true, false},   // VAR_PLUS_IMMED_TO_ADDR
            {true, false, true, false, false},   // ADDR_PLUS_IMMED_TO_VAR
            {false, false, true, false, true},   // ADDR_PLUS_ADDR_TO_VAR
        };

        auto idx = static_cast<size_t>(ccuRepAdd->GetSubType());
        if (idx == 0 || idx >= sizeof(table) / sizeof(table[0])) {
            HCCL_ERROR("Invalid AddSubType[%d]", idx);
            return HCCL_E_PARA;
        }
        const auto& info = table[idx];

        uint16_t srcAId = info.srcAVar ? ccuRepAdd->GetVarA().Id() : ccuRepAdd->GetAddrA().Id();
        uint16_t dstId = info.isSelf ? srcAId : (info.dstVar ? ccuRepAdd->GetVarC().Id() : ccuRepAdd->GetAddrC().Id());

        if (info.isImmed) {
            CcuV2::AddI(instr++, dstId, srcAId, ccuRepAdd->GetImmedB(), 0, 0);
        } else {
            uint16_t srcBId = info.srcBAddr ? ccuRepAdd->GetAddrB().Id() : ccuRepAdd->GetVarB().Id();
            CcuV2::Add(instr++, dstId, srcAId, srcBId, 0, 0);
        }

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV2::CcuRepAssignTranslate(
        [[maybe_unused]] CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepAssign* ccuRepAssign, const TransDep& dep)
    {
        UNUSED(dep);
        CHK_PTR_NULL(ccuRepAssign);
        switch (ccuRepAssign->GetSubType()) {
            case AssignSubType::IMD_TO_VARIABLE: {
                CcuV2::AssignI(instr++, ccuRepAssign->GetVarA().Id(), ccuRepAssign->GetImmed());
                break;
            }
            case AssignSubType::IMD_TO_ADDR: {
                CcuV2::AssignI(instr++, ccuRepAssign->GetAddrA().Id(), ccuRepAssign->GetImmed());
                break;
            }
            case AssignSubType::VAR_TO_ADDR: {
                CcuV2::Assign(instr++, ccuRepAssign->GetAddrA().Id(), ccuRepAssign->GetVarA().Id());
                break;
            }
            case AssignSubType::ADDR_TO_ADDR: {
                CcuV2::Assign(instr++, ccuRepAssign->GetAddrB().Id(), ccuRepAssign->GetAddrA().Id());
                break;
            }
            case AssignSubType::VAR_TO_VAR: {
                CcuV2::Assign(instr++, ccuRepAssign->GetVarB().Id(), ccuRepAssign->GetVarA().Id());
                break;
            }
            default: {
                HCCL_ERROR("Invalid Assign, subType[%d]", static_cast<int>(ccuRepAssign->GetSubType()));
                return HCCL_E_PARA;
            }
        }

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult
    CcuInsGeneratorV2::CcuRepMulTranslate([[maybe_unused]] CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepMul* ccuRepMul)
    {
        CHK_PTR_NULL(ccuRepMul);

        // {isImmed, isSelf, dstVar, srcAVar, srcBAddr} — 按 MulSubType 枚举顺序
        static constexpr struct {
            bool isImmed;
            bool isSelf;
            bool dstVar;
            bool srcAVar;
            bool srcBAddr;
        } table[] = {
            {false, false, false, false, false}, // INVALID
            {false, false, true, true, false},   // VAR_MUL_VAR_TO_VAR
            {true, false, true, true, false},    // VAR_MUL_IMMED_TO_VAR
            {false, true, true, true, false},    // SELF_MUL_VAR_VARIABLE
            {true, true, true, true, false},     // SELF_MUL_IMMED_VARIABLE
            {false, false, false, true, false},  // VAR_MUL_VAR_TO_ADDR
            {false, false, false, true, true},   // VAR_MUL_ADDR_TO_ADDR
            {true, false, false, true, false},   // VAR_MUL_IMMED_TO_ADDR
            {true, false, false, false, false},  // ADDR_MUL_IMMED_TO_ADDR
            {false, true, false, false, false},  // SELF_MUL_VAR_ADDRESS
            {true, true, false, false, false},   // SELF_MUL_IMMED_ADDRESS
            {true, false, true, false, false},   // ADDR_MUL_IMMED_TO_VAR
        };

        auto idx = static_cast<size_t>(ccuRepMul->GetSubType());
        if (idx == 0 || idx >= sizeof(table) / sizeof(table[0])) {
            HCCL_ERROR("Invalid Mul, subType[%d]", static_cast<int>(ccuRepMul->GetSubType()));
            return HCCL_E_PARA;
        }
        const auto& info = table[idx];

        uint16_t srcAId = info.srcAVar ? ccuRepMul->GetVarA().Id() : ccuRepMul->GetAddrA().Id();
        uint16_t dstId = info.isSelf ? srcAId : (info.dstVar ? ccuRepMul->GetVarC().Id() : ccuRepMul->GetAddrC().Id());

        if (info.isImmed) {
            CcuV2::MulI(instr++, dstId, srcAId, ccuRepMul->GetImmedB(), 0, 0);
        } else {
            uint16_t srcBId = info.srcBAddr ? ccuRepMul->GetAddrB().Id() : ccuRepMul->GetVarB().Id();
            CcuV2::Mul(instr++, dstId, srcAId, srcBId, 0, 0);
        }

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult
    CcuInsGeneratorV2::CcuRepSubTranslate([[maybe_unused]] CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepSub* ccuRepSub)
    {
        CHK_PTR_NULL(ccuRepSub);

        // {isImmed, isSelf, dstVar, srcAVar, srcBAddr} — 按 MinusSubType 枚举顺序
        static constexpr struct {
            bool isImmed;
            bool isSelf;
            bool dstVar;
            bool srcAVar;
            bool srcBAddr;
        } table[] = {
            {false, false, false, false, false}, // INVALID
            {false, false, true, true, false},   // VAR_MINUS_VAR_TO_VAR
            {true, false, true, true, false},    // VAR_MINUS_IMMED_TO_VAR
            {false, true, true, true, false},    // SELF_SUB_VAR_VARIABLE
            {true, true, true, true, false},     // SELF_SUB_IMMED_VARIABLE
            {false, false, false, false, false}, // ADDR_MINUS_VAR_TO_ADDR
            {true, false, false, false, false},  // ADDR_MINUS_IMMED_TO_ADDR
            {false, true, false, false, false},  // SELF_SUB_VAR_ADDRESS
            {true, true, false, false, false},   // SELF_SUB_IMMED_ADDRESS
            {true, false, false, true, false},   // VAR_MINUS_IMMED_TO_ADDR
            {true, false, true, false, false},   // ADDR_MINUS_IMMED_TO_VAR
        };

        auto idx = static_cast<size_t>(ccuRepSub->GetSubType());
        if (idx == 0 || idx >= sizeof(table) / sizeof(table[0])) {
            HCCL_ERROR("Invalid Sub, subType[%d]", static_cast<int>(ccuRepSub->GetSubType()));
            return HCCL_E_PARA;
        }
        const auto& info = table[idx];

        uint16_t srcAId = info.srcAVar ? ccuRepSub->GetVarA().Id() : ccuRepSub->GetAddrA().Id();
        uint16_t dstId = info.isSelf ? srcAId : (info.dstVar ? ccuRepSub->GetVarC().Id() : ccuRepSub->GetAddrC().Id());

        if (info.isImmed) {
            CcuV2::SubI(instr++, dstId, srcAId, ccuRepSub->GetImmedB(), 0, 0);
        } else {
            uint16_t srcBId = info.srcBAddr ? ccuRepSub->GetAddrB().Id() : ccuRepSub->GetVarB().Id();
            CcuV2::Sub(instr++, dstId, srcAId, srcBId, 0, 0);
        }

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV2::CcuRepAndTranslate(
        [[maybe_unused]] CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepAnd* ccuRepAnd,
        [[maybe_unused]] const TransDep& dep)
    {
        CHK_PTR_NULL(ccuRepAnd);
        switch (ccuRepAnd->GetSubType()) {
            case AndSubType::VAR_AND_VAR_TO_VAR: {
                CcuV2::And(
                    instr++, ccuRepAnd->GetVarC().Id(), ccuRepAnd->GetVarA().Id(), ccuRepAnd->GetVarB().Id(), 0, 0);
                break;
            }
            case AndSubType::SELF_AND_VAR_VARIABLE: {
                CcuV2::And(
                    instr++, ccuRepAnd->GetVarC().Id(), ccuRepAnd->GetVarC().Id(), ccuRepAnd->GetVarB().Id(), 0, 0);
                break;
            }
            default: {
                HCCL_ERROR("Invalid And, subType[%d]", static_cast<int>(ccuRepAnd->GetSubType()));
                return HCCL_E_PARA;
            }
        }
        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV2::CcuRepNotTranslate(
        [[maybe_unused]] CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepNot* ccuRepNot,
        [[maybe_unused]] const TransDep& dep)
    {
        CHK_PTR_NULL(ccuRepNot);
        switch (ccuRepNot->GetSubType()) {
            case NotSubType::VAR_EQUALS_NOT_VAR: {
                CcuV2::Not(instr++, ccuRepNot->GetVarC().Id(), ccuRepNot->GetVarB().Id(), 0, 0);
                break;
            }
            default: {
                HCCL_ERROR("Invalid Not, subType[%d]", static_cast<int>(ccuRepNot->GetSubType()));
                return HCCL_E_PARA;
            }
        }
        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV2::CcuRepOrTranslate(
        [[maybe_unused]] CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepOr* ccuRepOr,
        [[maybe_unused]] const TransDep& dep)
    {
        CHK_PTR_NULL(ccuRepOr);
        switch (ccuRepOr->GetSubType()) {
            case OrSubType::VAR_OR_VAR_TO_VAR: {
                CcuV2::Or(instr++, ccuRepOr->GetVarC().Id(), ccuRepOr->GetVarA().Id(), ccuRepOr->GetVarB().Id(), 0, 0);
                break;
            }
            case OrSubType::SELF_OR_VAR_VARIABLE: {
                CcuV2::Or(instr++, ccuRepOr->GetVarC().Id(), ccuRepOr->GetVarC().Id(), ccuRepOr->GetVarB().Id(), 0, 0);
                break;
            }
            default: {
                HCCL_ERROR("Invalid Or, subType[%d]", static_cast<int>(ccuRepOr->GetSubType()));
                return HCCL_E_PARA;
            }
        }
        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV2::CcuRepXorTranslate(
        [[maybe_unused]] CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepXor* ccuRepXor,
        [[maybe_unused]] const TransDep& dep)
    {
        CHK_PTR_NULL(ccuRepXor);
        switch (ccuRepXor->GetSubType()) {
            case XorSubType::VAR_XOR_VAR_TO_VAR: {
                CcuV2::Xor(
                    instr++, ccuRepXor->GetVarC().Id(), ccuRepXor->GetVarA().Id(), ccuRepXor->GetVarB().Id(), 0, 0);
                break;
            }
            case XorSubType::SELF_XOR_VAR_VARIABLE: {
                CcuV2::Xor(
                    instr++, ccuRepXor->GetVarC().Id(), ccuRepXor->GetVarC().Id(), ccuRepXor->GetVarB().Id(), 0, 0);
                break;
            }
            default: {
                HCCL_ERROR("Invalid Xor, subType[%d]", static_cast<int>(ccuRepXor->GetSubType()));
                return HCCL_E_PARA;
            }
        }
        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV2::CcuRepShLTranslate(
        [[maybe_unused]] CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepShL* ccuRepShL,
        [[maybe_unused]] const TransDep& dep)
    {
        CHK_PTR_NULL(ccuRepShL);
        if (ccuRepShL->GetShiftType() == ShiftType::LOGICAL_SHIFT) {
            switch (ccuRepShL->GetShiftSubType()) {
                case ShiftSubType::VAR_EQUALS_VAR_SHIFT_VAR: {
                    CcuV2::SLL(
                        instr++, ccuRepShL->GetVarD().Id(), ccuRepShL->GetVarN().Id(), ccuRepShL->GetVarM().Id(), 0, 0);
                    break;
                }
                case ShiftSubType::VAR_SHIFT_ASSIGN_VAR: {
                    CcuV2::SLL(
                        instr++, ccuRepShL->GetVarD().Id(), ccuRepShL->GetVarD().Id(), ccuRepShL->GetVarM().Id(), 0, 0);
                    break;
                }
                case ShiftSubType::ADDR_EQUALS_VAR_SHIFT_VAR: {
                    CcuV2::SLL(
                        instr++, ccuRepShL->GetAddressD().Id(), ccuRepShL->GetVarN().Id(), ccuRepShL->GetVarM().Id(), 0,
                        0);
                    break;
                }
                case ShiftSubType::ADDR_SHIFT_ASSIGN_VAR: {
                    CcuV2::SLL(
                        instr++, ccuRepShL->GetAddressD().Id(), ccuRepShL->GetAddressD().Id(),
                        ccuRepShL->GetVarM().Id(), 0, 0);
                    break;
                }
                default: {
                    HCCL_ERROR("Invalid Shift left, shiftSubType[%d]", static_cast<int>(ccuRepShL->GetShiftSubType()));
                    return HCCL_E_PARA;
                }
            }
        } else {
            HCCL_ERROR("Invalid Shift left, shiftType[%d]", static_cast<int>(ccuRepShL->GetShiftType()));
            return HCCL_E_PARA;
        }
        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV2::CcuRepShRTranslate(
        [[maybe_unused]] CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepShR* ccuRepShR,
        [[maybe_unused]] const TransDep& dep)
    {
        CHK_PTR_NULL(ccuRepShR);
        if (ccuRepShR->GetShiftType() == ShiftType::LOGICAL_SHIFT) {
            switch (ccuRepShR->GetShiftSubType()) {
                case ShiftSubType::VAR_EQUALS_VAR_SHIFT_VAR: {
                    CcuV2::SRL(
                        instr++, ccuRepShR->GetVarD().Id(), ccuRepShR->GetVarN().Id(), ccuRepShR->GetVarM().Id(), 0, 0);
                    break;
                }
                case ShiftSubType::VAR_SHIFT_ASSIGN_VAR: {
                    CcuV2::SRL(
                        instr++, ccuRepShR->GetVarD().Id(), ccuRepShR->GetVarD().Id(), ccuRepShR->GetVarM().Id(), 0, 0);
                    break;
                }
                case ShiftSubType::ADDR_EQUALS_VAR_SHIFT_VAR: {
                    CcuV2::SRL(
                        instr++, ccuRepShR->GetAddressD().Id(), ccuRepShR->GetVarN().Id(), ccuRepShR->GetVarM().Id(), 0,
                        0);
                    break;
                }
                case ShiftSubType::ADDR_SHIFT_ASSIGN_VAR: {
                    CcuV2::SRL(
                        instr++, ccuRepShR->GetAddressD().Id(), ccuRepShR->GetAddressD().Id(),
                        ccuRepShR->GetVarM().Id(), 0, 0);
                    break;
                }
                default: {
                    HCCL_ERROR("Invalid Shift right, shiftSubType[%d]", static_cast<int>(ccuRepShR->GetShiftSubType()));
                    return HCCL_E_PARA;
                }
            }
        } else {
            HCCL_ERROR("Invalid Shift right, shiftType[%d]", static_cast<int>(ccuRepShR->GetShiftType()));
            return HCCL_E_PARA;
        }
        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV2::CcuRepFuncBlockTranslate(
        [[maybe_unused]] CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepFuncBlock* funcBlockPtr,
        const TransDep& dep, uint32_t step)
    {
        CHK_PTR_NULL(funcBlockPtr);
        std::vector<CcuRepArg>& inArgs = funcBlockPtr->GetInArgs();
        (void)inArgs;
        std::vector<CcuRepArg>& outArgs = funcBlockPtr->GetOutArgs();
        CcuRepReferenceManager* funcManager = funcBlockPtr->GetFuncManager();
        CHK_PTR_NULL(funcManager);
        uint16_t callLayer = funcBlockPtr->GetCallLayer();
        if (step == 0) {
            // 函数入口为nop
            CcuV2::Nop(instr++);
            curInstrId++;
        } else if (step == 1) {
            // 处理输出的参数
            uint32_t iOutArg = 0;
            const auto& funcOut = funcManager->GetFuncOut();
            if (iOutArg >= funcOut.size()) {
                HCCL_ERROR("[FuncBlock] out arg index %u >= funcOut size %zu", iOutArg, funcOut.size());
                return HCCL_E_PARA;
            }
            for (uint32_t i = 0; i < outArgs.size(); i++) {
                if (outArgs[i].type == CcuArgType::VARIABLE) {
                    CcuV2::Add(instr++, funcOut[iOutArg++].Id(), outArgs[i].var.Id(), dep.reserveXnId);
                    curInstrId++;
                } else if (outArgs[i].type == CcuArgType::VARIABLE_LIST) {
                    for (uint32_t j = 0; j < outArgs[i].varList.size(); j++) {
                        CcuV2::Add(instr++, funcOut[iOutArg++].Id(), outArgs[i].varList[j].Id(), dep.reserveXnId);
                        curInstrId++;
                    }
                }
            }

            // 返回调用处
            uint32_t relJmpInstrNum = REL_JMP_INSTR_NUM; // relJmp需要9条指令
            CcuV2::RelJmp(
                instr, funcManager->GetFuncRet(callLayer).Id(), curInstrId + relJmpInstrNum, dep.commXn[0],
                dep.commXn[1]);
            instr += relJmpInstrNum;
            curInstrId += relJmpInstrNum;
            CcuV2::Jump(instr++, funcManager->GetFuncRet(callLayer).Id(), dep.reserveXnId, dep.reserveXnId, 0);
            curInstrId++;
        } else {
            HCCL_ERROR("Unsupported step[%d] for CcuRepFuncBlockTranslate", step);
            return HCCL_E_PARA;
        }
        return HcclResult::HCCL_SUCCESS;
    }

    void CcuInsGeneratorV2::LoadFuncCallInArgs(
        CcuInstr* instr, std::vector<CcuRepArg>& inArgs, std::vector<Variable>& formalIns, uint16_t reserveXnId) const
    {
        uint32_t idx = 0;
        for (uint32_t i = 0; i < inArgs.size(); i++) {
            if (inArgs[i].type == CcuArgType::VARIABLE) {
                CcuV2::Add(instr + idx, formalIns[idx].Id(), inArgs[i].var.Id(), reserveXnId);
                idx++;
            } else if (inArgs[i].type == CcuArgType::VARIABLE_LIST) {
                for (uint32_t j = 0; j < inArgs[i].varList.size(); j++) {
                    CcuV2::Add(instr + idx, formalIns[idx].Id(), inArgs[i].varList[j].Id(), reserveXnId);
                    idx++;
                }
            }
        }
    }

    void CcuInsGeneratorV2::LoadFuncCallOutArgs(
        CcuInstr* instr, uint32_t offset, std::vector<CcuRepArg>& outArgs, CcuRepReferenceManager* funcManager,
        uint16_t reserveXnId) const
    {
        uint32_t idx = 0;
        for (uint32_t i = 0; i < outArgs.size(); i++) {
            if (outArgs[i].type == CcuArgType::VARIABLE) {
                CcuV2::Add(instr + offset + idx, outArgs[i].var.Id(), funcManager->GetFuncOut()[idx].Id(), reserveXnId);
                idx++;
            } else if (outArgs[i].type == CcuArgType::VARIABLE_LIST) {
                for (uint32_t j = 0; j < outArgs[i].varList.size(); j++) {
                    CcuV2::Add(
                        instr + offset + idx, outArgs[i].varList[j].Id(), funcManager->GetFuncOut()[idx].Id(),
                        reserveXnId);
                    idx++;
                }
            }
        }
    }

    HcclResult CcuInsGeneratorV2::CcuRepFuncCallTranslate(
        [[maybe_unused]] CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepFuncCall* funcCallPtr,
        const TransDep& dep)
    {
        (void)instr;
        (void)curInstrId;

        FuncCallContext ctx;
        CHK_RET(PrepareFuncCallContext(funcCallPtr, ctx));

        std::vector<CcuRepArg>& inArgs = funcCallPtr->GetInArgs();
        std::vector<CcuRepArg>& outArgs = funcCallPtr->GetOutArgs();
        uint32_t inArgCount = ctx.inArgCount;
        CcuRepReferenceManager* funcManager = ctx.funcManager;
        std::shared_ptr<CcuRepFuncBlock>& funcBlock = ctx.funcBlock;
        CcuInstr* funcInstr = ctx.instr;
        std::vector<Variable>& formalIns = ctx.formalIns;
        Variable funcAddrVar = funcCallPtr->GetFuncAddrVar();
        int32_t callLayer = funcCallPtr->GetCallLayer();
        uint16_t instrId = funcCallPtr->StartInstrId();
        LoadFuncCallInArgs(funcInstr, inArgs, formalIns, dep.reserveXnId);

        uint32_t locId = 0;
        if (funcBlock != nullptr) {
            CcuV2::LoadImdToXn(
                funcInstr + inArgCount + locId++, funcManager->GetFuncCall().Id(), funcBlock->StartInstrId());
        } else {
            CcuV2::Add(
                funcInstr + inArgCount + locId++, funcManager->GetFuncCall().Id(), funcAddrVar.Id(), dep.reserveXnId);
        }
        CcuV2::LoadImdToXn(
            funcInstr + inArgCount + locId++, funcManager->GetFuncRet(callLayer).Id(),
            instrId + inArgCount + FUNC_CALL_RET_OFFSET); // 需要指向函数返回位置
        CcuV2::RelJmp(
            funcInstr + inArgCount + locId, funcManager->GetFuncCall().Id(),
            instrId + inArgCount + FUNC_CALL_JMP_OFFSET, // Jmp的目标指令为其后11条指令
            dep.commXn[0], dep.commXn[1]);
        locId += REL_JMP_INSTR_NUM; // relJmp需要9条指令
        CcuV2::Jump(
            funcInstr + inArgCount + locId++, funcManager->GetFuncCall().Id(), dep.reserveXnId, dep.reserveXnId, 0);
        CcuV2::Nop(funcInstr + inArgCount + locId++);

        uint32_t extraInstrNum = GetInstrCount(funcCallPtr->Type());
        LoadFuncCallOutArgs(funcInstr, inArgCount + extraInstrNum, outArgs, funcManager, dep.reserveXnId);
        return HcclResult::HCCL_SUCCESS;
    }

    uint32_t GetRelativeInstrId(uint32_t currentInstrId, uint32_t targetInstrId)
    {
        static constexpr uint32_t CCU_INSTR_ID_SPACE_SIZE = 0x10000;
        if (targetInstrId > currentInstrId) {
            return targetInstrId - currentInstrId;
        } else {
            uint32_t diff = currentInstrId - targetInstrId;
            if (diff >= CCU_INSTR_ID_SPACE_SIZE) {
                HCCL_ERROR("Jump distance %u exceeds instr id space %u", diff, CCU_INSTR_ID_SPACE_SIZE);
                return 0;
            }
            return CCU_INSTR_ID_SPACE_SIZE - diff;
        }
    }

    HcclResult CcuInsGeneratorV2::CcuRepJumpTranslate(
        [[maybe_unused]] CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepJump* jumpPtr,
        const TransDep& dep)
    {
        (void)instr;
        (void)curInstrId;
        (void)dep;
        // 翻译直接跳转指令
        CHK_PTR_NULL(jumpPtr);
        std::shared_ptr<CcuRepJumpLabel> jumpLabel = jumpPtr->GetJumpLabel();
        CHK_PTR_NULL(jumpLabel);
        uint16_t instrIdOffset = GetRelativeInstrId(jumpPtr->StartInstrId() + 1, jumpLabel->StartInstrId());
        CcuV2::LoadImdToXn(jumpPtr->GetInstr() + 0, jumpPtr->GetTargetInstrId().Id(), instrIdOffset, 0, 0);
        //  无条件跳转
        CcuV2::Jump(
            jumpPtr->GetInstr() + 1, jumpPtr->GetTargetInstrId().Id(), 0, 0, static_cast<int>(ConditionType::DEFAULT));

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV2::CcuRepJumpTranslateV2Base(
        CcuInstr* const& curInstr, uint16_t& curInstrId, CcuRepJumpBase* jumpBasePtr, uint64_t expected,
        const Variable& condition, const Variable& expectedVar, ConditionType condType) const
    {
        (void)curInstr;
        (void)curInstrId;
        CHK_PTR_NULL(jumpBasePtr);
        CcuInstr* instr = jumpBasePtr->GetInstr();
        uint16_t instrId = jumpBasePtr->StartInstrId();
        Variable& targetInstrId = jumpBasePtr->GetTargetInstrId();
        std::shared_ptr<CcuRepJumpLabel> jumpLabel = jumpBasePtr->GetJumpLabel();
        CHK_PTR_NULL(jumpLabel);

        if (jumpBasePtr->IsComparedWithImmd()) {
            CcuV2::LoadImdToXn(instr + 0, expectedVar.Id(), expected, 0, 0);
            uint16_t instrIdOffset = GetRelativeInstrId(instrId + 2, jumpLabel->StartInstrId()); // 2: jump指令偏移
            CcuV2::LoadImdToXn(instr + 1, targetInstrId.Id(), instrIdOffset, 0, 0);
            CcuV2::Jump(
                instr + 2, targetInstrId.Id(), condition.Id(), expectedVar.Id(), // 2: jump指令偏移
                static_cast<int>(condType));
        } else {
            uint16_t instrIdOffset = GetRelativeInstrId(instrId + 1, jumpLabel->StartInstrId());
            CcuV2::LoadImdToXn(instr + 0, targetInstrId.Id(), instrIdOffset, 0, 0);
            CcuV2::Jump(instr + 1, targetInstrId.Id(), condition.Id(), expectedVar.Id(), static_cast<int>(condType));
        }

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV2::CcuRepJumpNETranslate(
        [[maybe_unused]] CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepJumpNE* jumpNEPtr,
        const TransDep& dep)
    {
        (void)dep;
        CHK_PTR_NULL(jumpNEPtr);
        return CcuRepJumpTranslateV2Base(
            instr, curInstrId, jumpNEPtr, jumpNEPtr->GetExpectedNum(), jumpNEPtr->GetCondition(),
            jumpNEPtr->GetExpectedVar(), ConditionType::NOT_EQUAL);
    }

    HcclResult CcuInsGeneratorV2::CcuRepJumpEQTranslate(
        [[maybe_unused]] CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepJumpEQ* jumpEQPtr,
        const TransDep& dep)
    {
        (void)dep;
        CHK_PTR_NULL(jumpEQPtr);
        return CcuRepJumpTranslateV2Base(
            instr, curInstrId, jumpEQPtr, jumpEQPtr->GetExpectedNum(), jumpEQPtr->GetCondition(),
            jumpEQPtr->GetExpectedVar(), ConditionType::EQUAL);
    }

    HcclResult CcuInsGeneratorV2::CcuRepJumpLETranslate(
        [[maybe_unused]] CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepJumpLE* jumpLEPtr,
        const TransDep& dep)
    {
        (void)dep;
        CHK_PTR_NULL(jumpLEPtr);
        return CcuRepJumpTranslateV2Base(
            instr, curInstrId, jumpLEPtr, jumpLEPtr->GetExpectedNum(), jumpLEPtr->GetCondition(),
            jumpLEPtr->GetExpectedVar(), ConditionType::LESS_EQUAL);
    }

    HcclResult CcuInsGeneratorV2::CcuRepJumpGETranslate(
        [[maybe_unused]] CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepJumpGE* jumpGEPtr,
        const TransDep& dep)
    {
        (void)dep;
        CHK_PTR_NULL(jumpGEPtr);
        return CcuRepJumpTranslateV2Base(
            instr, curInstrId, jumpGEPtr, jumpGEPtr->GetExpectedNum(), jumpGEPtr->GetCondition(),
            jumpGEPtr->GetExpectedVar(), ConditionType::GREATER_EQUAL);
    }

    HcclResult CcuInsGeneratorV2::CcuRepJumpGTTranslate(
        [[maybe_unused]] CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepJumpGT* jumpGTPtr,
        const TransDep& dep)
    {
        (void)dep;
        CHK_PTR_NULL(jumpGTPtr);
        return CcuRepJumpTranslateV2Base(
            instr, curInstrId, jumpGTPtr, jumpGTPtr->GetExpectedNum(), jumpGTPtr->GetCondition(),
            jumpGTPtr->GetExpectedVar(), ConditionType::GREATER_THAN);
    }

    HcclResult CcuInsGeneratorV2::CcuRepJumpLTTranslate(
        [[maybe_unused]] CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepJumpLT* jumpLTPtr,
        const TransDep& dep)
    {
        (void)dep;
        CHK_PTR_NULL(jumpLTPtr);
        return CcuRepJumpTranslateV2Base(
            instr, curInstrId, jumpLTPtr, jumpLTPtr->GetExpectedNum(), jumpLTPtr->GetCondition(),
            jumpLTPtr->GetExpectedVar(), ConditionType::LESS_THAN);
    }

    HcclResult CcuInsGeneratorV2::CcuRepLoopTranslate(
        [[maybe_unused]] CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepLoop* loopPtr)
    {
        (void)curInstrId;
        CHK_PTR_NULL(loopPtr);
        auto loopBlock = loopPtr->GetLoopBlock();
        CHK_PTR_NULL(loopBlock);
        CcuV2::Loop(
            instr++, loopBlock->StartInstrId(), loopBlock->StartInstrId() + loopBlock->InstrCount() - 1,
            loopPtr->GetLoopIterNum().Id(), loopPtr->GetLoopGsaOffset().Id(), loopPtr->GetLoopParam()->Id());
        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult
    CcuInsGeneratorV2::LoadLoopCallArg(CcuInstr*& instr, const CcuRepArg& inArg, const CcuRepArg& blkArg) const
    {
        switch (inArg.type) {
            case CcuArgType::VARIABLE:
                CcuV2::Assign(instr++, blkArg.var.Id(), inArg.var.Id());
                break;
            case CcuArgType::VARIABLE_LIST:
                if (inArg.varList.size() != blkArg.varList.size()) {
                    HCCL_ERROR(
                        "Mismatched Arg Size, inArg.varList.size[%zu], blkArg.varList.size[%zu]", inArg.varList.size(),
                        blkArg.varList.size());
                    return HCCL_E_PARA;
                }
                for (uint32_t j = 0; j < inArg.varList.size(); j++) {
                    CcuV2::Assign(instr++, blkArg.varList[j].Id(), inArg.varList[j].Id());
                }
                break;
            case CcuArgType::MEMORY:
                LoadAddrArg(instr, blkArg.mem, inArg.mem);
                break;
            case CcuArgType::LOCAL_ADDR:
                LoadAddrArg(instr, blkArg.localAddr, inArg.localAddr);
                break;
            case CcuArgType::REMOTE_ADDR:
                LoadAddrArg(instr, blkArg.remoteAddr, inArg.remoteAddr);
                break;
            case CcuArgType::MEMORY_LIST:
                CHK_RET(LoadAddrListArg(instr, blkArg.memList, inArg.memList));
                break;
            case CcuArgType::LOCAL_ADDR_LIST:
                CHK_RET(LoadAddrListArg(instr, blkArg.localAddrList, inArg.localAddrList));
                break;
            case CcuArgType::REMOTE_ADDR_LIST:
                CHK_RET(LoadAddrListArg(instr, blkArg.remoteAddrList, inArg.remoteAddrList));
                break;
            default:
                HCCL_ERROR("Mismatched Arg Type, inArg.type[%d]", static_cast<int>(inArg.type));
                return HCCL_E_PARA;
        }
        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV2::CcuRepLoopCallTranslate(
        [[maybe_unused]] CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepLoopCall* loopCallPtr,
        const TransDep& dep)
    {
        (void)curInstrId;
        (void)dep;
        CHK_PTR_NULL(loopCallPtr);
        std::vector<CcuRepArg>& inArgs = loopCallPtr->GetInArgs();
        auto loopBlock = loopCallPtr->GetLoopBlock();
        CHK_PTR_NULL(loopBlock);

        for (uint32_t i = 0; i < inArgs.size(); i++) {
            const CcuRepArg& inArg = inArgs[i];
            const CcuRepArg& blkArg = loopBlock->GetArg(i);
            if (inArg.type != blkArg.type) {
                HCCL_ERROR(
                    "Mismatched Arg Type, inArg.type[%d], blkArg.type[%d]", static_cast<int>(inArg.type),
                    static_cast<int>(blkArg.type));
                return HCCL_E_PARA;
            }
            CHK_RET(LoadLoopCallArg(instr, inArg, blkArg));
        }

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV2::CcuRepSetLoopTranslate(
        [[maybe_unused]] CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepSetLoop* setLoopPtr)
    {
        (void)curInstrId;
        CHK_PTR_NULL(setLoopPtr);
        // CCU V121 将executorId赋值给loop指令的Xp，上层计算的loopContextId实际上没有用
        CcuV2::LoadImdToXn(instr++, setLoopPtr->loopParam.Id(), setLoopPtr->executor.Id());
        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV2::CcuRepLoadTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepLoad* loadPtr, const TransDep& dep)
    {
        (void)curInstrId;
        CHK_PTR_NULL(loadPtr);
        CHK_PTR_NULL(ccuKernel);
        const auto& constValue2VarMap = ccuKernel->GetConstValue2VarMap();
        CcuV2::CacheConfig cacheConfig{0x0, 0x0};

        // var寄存器的真实id在常量准备阶段（Register）无法获取，故不记录在常量表中，单独赋值处理
        CcuV2::LoadImdToXn(instr++, dep.commXn[0], loadPtr->GetVar().Id()); // dst xn id
        CcuV2::LoadXFromMem(
            instr++, dep.commXn[0], constValue2VarMap.at(loadPtr->GetAddr()).Id(),
            constValue2VarMap.at(dep.memTokenInfo).Id(),
            constValue2VarMap.at(CCU_RESOURCE_XN_PER_SIZE * loadPtr->GetNum()).Id(), cacheConfig, dep.commSignal,
            loadPtr->GetMask());
        CcuV2::SetCKE(instr++, 0, 0, dep.commSignal, loadPtr->GetMask(), 1);

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV2::CcuRepLoadVarTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepLoadVar* loadVarPtr, const TransDep& dep)
    {
        (void)curInstrId;
        CHK_PTR_NULL(loadVarPtr);
        CHK_PTR_NULL(ccuKernel);
        const auto& constValue2VarMap = ccuKernel->GetConstValue2VarMap();
        CcuV2::CacheConfig cacheConfig{0x0, 0x0};

        // var寄存器的真实id在常量准备阶段（Register）无法获取，故不记录在常量表中，单独赋值处理
        CcuV2::LoadImdToXn(instr++, dep.commXn[0], loadVarPtr->GetVar().Id()); // dst xn id
        CcuV2::LoadXFromMem(
            instr++, dep.commXn[0], loadVarPtr->GetSrc().Id(), constValue2VarMap.at(dep.memTokenInfo).Id(),
            constValue2VarMap.at(CCU_RESOURCE_XN_PER_SIZE * loadVarPtr->GetNum()).Id(), cacheConfig, dep.commSignal,
            loadVarPtr->GetMask());
        CcuV2::SetCKE(instr++, 0, 0, dep.commSignal, loadVarPtr->GetMask(), 1);
        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV2::CcuRepLoadArgTranslate(
        [[maybe_unused]] CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepLoadArg* loadArgPtr,
        const TransDep& dep)
    {
        (void)curInstrId;
        CHK_PTR_NULL(loadArgPtr);
        if (dep.isFuncBlock) {
            // Xn(var) = Xn(loadXnId) + 0
            CcuV2::Add(instr++, loadArgPtr->GetVar().Id(), dep.loadXnId, dep.reserveXnId);
        } else {
            CcuV2::LoadSqeArgsToX(instr++, loadArgPtr->GetVar().Id(), loadArgPtr->GetArgId());
        }

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV2::CcuRepNopTranslate(
        [[maybe_unused]] CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepNop* nopPtr,
        const TransDep& dep)
    {
        (void)curInstrId;
        (void)nopPtr;
        (void)dep;
        CcuV2::Nop(instr++);

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV2::CcuRepStoreTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepStore* storePtr, const TransDep& dep)
    {
        (void)curInstrId;
        CHK_PTR_NULL(storePtr);
        CHK_PTR_NULL(ccuKernel);
        const auto& constValue2VarMap = ccuKernel->GetConstValue2VarMap();
        CcuV2::CacheConfig cacheConfig{0x0, 0x0};

        // var寄存器的真实id在常量准备阶段（Register）无法获取，故不记录在常量表中，单独赋值处理
        CcuV2::LoadImdToXn(instr++, dep.commXn[0], storePtr->GetVar().Id()); // src xn id
        CcuV2::StoreXToMem(
            instr++, constValue2VarMap.at(storePtr->GetAddr()).Id(), constValue2VarMap.at(dep.memTokenInfo).Id(),
            dep.commXn[0], constValue2VarMap.at(CCU_RESOURCE_XN_PER_SIZE * storePtr->GetNum()).Id(), cacheConfig,
            dep.commSignal, storePtr->GetMask());
        CcuV2::SetCKE(instr++, 0, 0, dep.commSignal, storePtr->GetMask(), 1);

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV2::CcuRepStoreVarTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepStoreVar* storeVarPtr, const TransDep& dep)
    {
        CHK_PTR_NULL(storeVarPtr);
        CHK_PTR_NULL(ccuKernel);
        (void)curInstrId;
        CcuV2::CacheConfig cacheConfig{0x0, 0x0};

        // var寄存器的真实id在常量准备阶段（Register）无法获取，故不记录在常量表中，单独赋值处理
        CcuV2::LoadImdToXn(instr++, dep.commXn[0], storeVarPtr->GetVar().Id());
        const auto& constValue2VarMap = ccuKernel->GetConstValue2VarMap();
        if (storeVarPtr->GetHscbFlag()) {
            CcuV2::HSCBStoreXToMem(
                instr++, storeVarPtr->GetDst().Id(), dep.commXn[0],
                constValue2VarMap.at(CCU_RESOURCE_XN_PER_SIZE * storeVarPtr->GetNum()).Id(), cacheConfig,
                dep.commSignal, storeVarPtr->GetMask());
        } else {
            CcuV2::StoreXToMem(
                instr++, storeVarPtr->GetDst().Id(), constValue2VarMap.at(dep.memTokenInfo).Id(), dep.commXn[0],
                constValue2VarMap.at(CCU_RESOURCE_XN_PER_SIZE * storeVarPtr->GetNum()).Id(), cacheConfig,
                dep.commSignal, storeVarPtr->GetMask());
        }
        CcuV2::SetCKE(instr++, 0, 0, dep.commSignal, storeVarPtr->GetMask(), 1);

        return HcclResult::HCCL_SUCCESS;
    }

    // halfRtt
    HcclResult CcuInsGeneratorV2::CcuRepWriteVarAtomicTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepWriteVarAtomic* writeVarAtomPtr,
        const TransDep& dep)
    {
        CHK_PTR_NULL(writeVarAtomPtr);
        (void)ccuKernel;
        (void)curInstrId;
        (void)dep;

        CcuV2::SyncAtX(
            instr++, writeVarAtomPtr->GetVarAddrId(), writeVarAtomPtr->GetVarTokenId(), writeVarAtomPtr->GetTargetId(),
            writeVarAtomPtr->GetChannelVarId(), writeVarAtomPtr->GetSemId(), writeVarAtomPtr->GetMask());
        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV2::CcuRepWriteWithCntIncTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepWriteWithCntInc* writeCntIncPtr,
        const TransDep& dep)
    {
        CHK_PTR_NULL(writeCntIncPtr);
        (void)ccuKernel;
        (void)curInstrId;
        (void)dep;

        CcuV2::TransMemNotifyInfo notify{};
        CcuV2::TransMemReduceInfo reduce{};
        CcuV2::TransMemConfig config{};
        config.dmaOpCode = URMA_DMA_OP_WRITE_WITH_CNT_INC;
        config.splitMode = SPLIT_MODE_PACKET;
        notify.xnId = writeCntIncPtr->GetIncCntAddrId();
        notify.xntId = writeCntIncPtr->GetIncCntTokenId();
        notify.value = 1;
        CcuV2::TransMem(
            instr++, writeCntIncPtr->GetRemAddrId(), writeCntIncPtr->GetRemTokenId(), writeCntIncPtr->GetLocAddrId(),
            writeCntIncPtr->GetLocTokenId(), writeCntIncPtr->GetLenId(), writeCntIncPtr->GetChannelVarId(), notify,
            reduce, config, 0, 0);
        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV2::CcuRepCascCntWaitTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepCascCntWait* cascCntWaitPtr,
        const TransDep& dep)
    {
        CHK_PTR_NULL(cascCntWaitPtr);
        (void)curInstrId;
        (void)dep;

        CntXnBlock cascCntBlock;
        CHK_RET(ccuKernel->GetCascCntBlock(cascCntWaitPtr->GetCntHandle(), cascCntBlock));
        cascCntWaitPtr->SetCascCntBlock(cascCntBlock);

        CcuV2::LoadImdToXn(
            instr++, static_cast<uint16_t>(cascCntBlock.expectedCntXn), cascCntWaitPtr->GetOutCntTarget(), 0, 0);
        CcuV2::Wait(
            instr++, static_cast<uint16_t>(cascCntBlock.totalCntXn), static_cast<uint16_t>(cascCntBlock.expectedCntXn),
            0);
        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV2::CcuRepCascCntClearTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepCascCntClear* cascCntClearPtr,
        const TransDep& dep)
    {
        CHK_PTR_NULL(cascCntClearPtr);
        (void)curInstrId;
        (void)dep;

        CntXnBlock cascCntBlock;
        CHK_RET(ccuKernel->GetCascCntBlock(cascCntClearPtr->GetCntHandle(), cascCntBlock));
        cascCntClearPtr->SetCascCntBlock(cascCntBlock);

        CcuV2::ClearX(
            instr++, static_cast<uint16_t>(cascCntBlock.wishCntXns.first),
            static_cast<uint16_t>(cascCntBlock.expectedCntXn), 0, 0, cascCntClearPtr->GetSemId(),
            cascCntClearPtr->GetMask());
        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV2::CcuRepLoadAddImmTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepLoadAddImm* loadAddImmPtr,
        const TransDep& dep)
    {
        CHK_PTR_NULL(loadAddImmPtr);
        (void)ccuKernel;
        (void)curInstrId;
        (void)dep;

        CcuV2::LoadX(
            instr++, loadAddImmPtr->GetDstId(), loadAddImmPtr->GetSrcId(), loadAddImmPtr->GetSrcOffsetId(),
            loadAddImmPtr->GetImmAddValue(), 0, 0, 0);
        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV2::CcuRepStoreAddImmTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepStoreAddImm* storeAddImmPtr,
        const TransDep& dep)
    {
        CHK_PTR_NULL(storeAddImmPtr);
        (void)ccuKernel;
        (void)curInstrId;
        (void)dep;

        CcuV2::StoreX(
            instr++, storeAddImmPtr->GetDstId(), storeAddImmPtr->GetSrcId(), storeAddImmPtr->GetImmAddValue(),
            storeAddImmPtr->GetDstOffsetId(), 0, 0, 0);
        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV2::PrepareConstValue(CcuRepBase* repPtr, const TransDep& dep, CcuKernel* ccuKernel)
    {
        CHK_PTR_NULL(repPtr);
        CHK_PTR_NULL(ccuKernel);
        HCCL_INFO("CcuInsGeneratorV2::PrepareConstValue Current RepType[%d]", repPtr->Type());
        switch (repPtr->Type()) {
            case CcuRepType::LOAD:
                return PrepareLoadConstValue(repPtr, dep, ccuKernel);
            case CcuRepType::LOAD_VAR:
                return PrepareLoadVarConstValue(repPtr, dep, ccuKernel);
            case CcuRepType::STORE:
                return PrepareStoreConstValue(repPtr, dep, ccuKernel);
            case CcuRepType::STORE_VAR:
                return PrepareStoreVarConstValue(repPtr, dep, ccuKernel);
            case CcuRepType::REM_POST_SEM:
                return PrepareRemPostSemConstValue(repPtr, dep, ccuKernel);
            case CcuRepType::REM_POST_VAR:
                return PrepareRemPostVarConstValue(repPtr, dep, ccuKernel);
            case CcuRepType::WRITE:
                return PrepareWriteConstValue(repPtr, dep, ccuKernel);
            case CcuRepType::READ:
                return PrepareReadConstValue(repPtr, dep, ccuKernel);
            case CcuRepType::BUF_WRITE:
                return PrepareBufWriteConstValue(repPtr, dep, ccuKernel);
            case CcuRepType::BUF_READ:
                return PrepareBufReadConstValue(repPtr, dep, ccuKernel);
            case CcuRepType::LOCAL_CPY:
                return PrepareLocCpyConstValue(repPtr, dep, ccuKernel);
            case CcuRepType::RECORD_SHARED_NOTIFY:
                return PrepareRecordSharedNotifyConstValue(repPtr, dep, ccuKernel);
            default:
                break;
        }
        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult
    CcuInsGeneratorV2::PrepareLoadConstValue(CcuRepBase* repPtr, const TransDep& dep, CcuKernel* ccuKernel) const
    {
        CcuRepLoad* loadPtr = dynamic_cast<CcuRepLoad*>(repPtr);
        CHK_PTR_NULL(loadPtr);
        std::vector<uint64_t> values
            = {loadPtr->GetAddr(), dep.memTokenInfo, CCU_RESOURCE_XN_PER_SIZE * loadPtr->GetNum()};
        return ccuKernel->Add2ConstValue2VarMap(values);
    }

    HcclResult
    CcuInsGeneratorV2::PrepareLoadVarConstValue(CcuRepBase* repPtr, const TransDep& dep, CcuKernel* ccuKernel) const
    {
        CcuRepLoadVar* loadVarPtr = dynamic_cast<CcuRepLoadVar*>(repPtr);
        CHK_PTR_NULL(loadVarPtr);
        std::vector<uint64_t> values = {dep.memTokenInfo, CCU_RESOURCE_XN_PER_SIZE * loadVarPtr->GetNum()};
        return ccuKernel->Add2ConstValue2VarMap(values);
    }

    HcclResult
    CcuInsGeneratorV2::PrepareStoreConstValue(CcuRepBase* repPtr, const TransDep& dep, CcuKernel* ccuKernel) const
    {
        CcuRepStore* storePtr = dynamic_cast<CcuRepStore*>(repPtr);
        CHK_PTR_NULL(storePtr);
        std::vector<uint64_t> values
            = {storePtr->GetAddr(), dep.memTokenInfo, CCU_RESOURCE_XN_PER_SIZE * storePtr->GetNum()};
        return ccuKernel->Add2ConstValue2VarMap(values);
    }

    HcclResult
    CcuInsGeneratorV2::PrepareStoreVarConstValue(CcuRepBase* repPtr, const TransDep& dep, CcuKernel* ccuKernel) const
    {
        CcuRepStoreVar* storeVarPtr = dynamic_cast<CcuRepStoreVar*>(repPtr);
        CHK_PTR_NULL(storeVarPtr);
        std::vector<uint64_t> values = {CCU_RESOURCE_XN_PER_SIZE * storeVarPtr->GetNum()};
        if (!storeVarPtr->GetHscbFlag()) {
            values.push_back(dep.memTokenInfo);
        }
        return ccuKernel->Add2ConstValue2VarMap(values);
    }

    HcclResult CcuInsGeneratorV2::PrepareRemPostSemConstValue(
        CcuRepBase* repPtr, [[maybe_unused]] const TransDep& dep, CcuKernel* ccuKernel) const
    {
        CcuRepRemPostSem* repRemPostSemPtr = dynamic_cast<CcuRepRemPostSem*>(repPtr);
        CHK_PTR_NULL(repRemPostSemPtr);
        CcuUrmaChannel* channelImpl{};
        CHK_RET(GetUrmaChannel(repRemPostSemPtr->GetChannel(), channelImpl));
        uint64_t rmtSignalAddr{0};
        CHK_PRT_RET(
            channelImpl->GetRmtSignalAddrByIndex(repRemPostSemPtr->GetSemIndex(), rmtSignalAddr)
                != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR(
                "[CcuInsGeneratorV2][%s] failed to get remote signal addr, channelHandle[0x%llx].", __func__,
                repRemPostSemPtr->GetChannel()),
            HCCL_E_INTERNAL);
        uint64_t rmtToken{};
        CHK_RET(GetRmtToken(channelImpl, rmtToken));
        std::vector<uint64_t> values = {channelImpl->GetChannelId(), rmtSignalAddr, rmtToken};
        return ccuKernel->Add2ConstValue2VarMap(values);
    }

    HcclResult CcuInsGeneratorV2::PrepareRemPostVarConstValue(
        CcuRepBase* repPtr, [[maybe_unused]] const TransDep& dep, CcuKernel* ccuKernel) const
    {
        CcuRepRemPostVar* repRemPostVarPtr = dynamic_cast<CcuRepRemPostVar*>(repPtr);
        CHK_PTR_NULL(repRemPostVarPtr);
        CcuUrmaChannel* channelImpl{};
        CHK_RET(GetUrmaChannel(repRemPostVarPtr->GetChannel(), channelImpl));
        uint64_t rmtSignalAddr{0};
        uint64_t rmtVarAddr{0};
        CHK_PRT_RET(
            channelImpl->GetRmtSignalAddrByIndex(repRemPostVarPtr->GetSemIndex(), rmtSignalAddr)
                != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR(
                "[CcuInsGeneratorV2][%s] failed to get remote signal addr, channelHandle[0x%llx].", __func__,
                repRemPostVarPtr->GetChannel()),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(
            channelImpl->GetRmtVarAddrByIndex(repRemPostVarPtr->GetParamIndex(), rmtVarAddr)
                != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR(
                "[CcuInsGeneratorV2][%s] failed to get remote var addr, channelHandle[0x%llx].", __func__,
                repRemPostVarPtr->GetChannel()),
            HCCL_E_INTERNAL);
        uint64_t rmtToken{};
        CHK_RET(GetRmtToken(channelImpl, rmtToken));
        std::vector<uint64_t> values = {channelImpl->GetChannelId(), rmtSignalAddr, rmtToken, rmtVarAddr};
        return ccuKernel->Add2ConstValue2VarMap(values);
    }

    HcclResult CcuInsGeneratorV2::PrepareWriteConstValue(
        CcuRepBase* repPtr, [[maybe_unused]] const TransDep& dep, CcuKernel* ccuKernel) const
    {
        CcuRepWrite* repWritePtr = dynamic_cast<CcuRepWrite*>(repPtr);
        CHK_PTR_NULL(repWritePtr);
        CcuUrmaChannel* channelImpl{};
        CHK_RET(GetUrmaChannel(repWritePtr->GetChannel(), channelImpl));
        uint64_t rmtToken{};
        CHK_RET(GetRmtToken(channelImpl, rmtToken));
        std::vector<uint64_t> values = {channelImpl->GetChannelId(), rmtToken};
        return ccuKernel->Add2ConstValue2VarMap(values);
    }

    HcclResult CcuInsGeneratorV2::PrepareReadConstValue(
        CcuRepBase* repPtr, [[maybe_unused]] const TransDep& dep, CcuKernel* ccuKernel) const
    {
        CcuRepRead* repReadPtr = dynamic_cast<CcuRepRead*>(repPtr);
        CHK_PTR_NULL(repReadPtr);
        CcuUrmaChannel* channelImpl{};
        CHK_RET(GetUrmaChannel(repReadPtr->GetChannel(), channelImpl));
        std::vector<uint64_t> values = {channelImpl->GetChannelId()};
        return ccuKernel->Add2ConstValue2VarMap(values);
    }

    HcclResult
    CcuInsGeneratorV2::PrepareBufWriteConstValue(CcuRepBase* repPtr, const TransDep& dep, CcuKernel* ccuKernel) const
    {
        CcuRepBufWrite* repBufWritePtr = dynamic_cast<CcuRepBufWrite*>(repPtr);
        CHK_PTR_NULL(repBufWritePtr);
        CcuUrmaChannel* channelImpl{};
        CHK_RET(GetUrmaChannel(repBufWritePtr->GetChannel(), channelImpl));
        uint64_t rmtToken{};
        CHK_RET(GetRmtToken(channelImpl, rmtToken));
        std::vector<uint64_t> values = {channelImpl->GetChannelId(), rmtToken, dep.ccuResSpaceTokenInfo};
        return ccuKernel->Add2ConstValue2VarMap(values);
    }

    HcclResult
    CcuInsGeneratorV2::PrepareBufReadConstValue(CcuRepBase* repPtr, const TransDep& dep, CcuKernel* ccuKernel) const
    {
        CcuRepBufRead* repBufReadPtr = dynamic_cast<CcuRepBufRead*>(repPtr);
        CHK_PTR_NULL(repBufReadPtr);
        CcuUrmaChannel* channelImpl{};
        CHK_RET(GetUrmaChannel(repBufReadPtr->GetChannel(), channelImpl));
        std::vector<uint64_t> values = {channelImpl->GetChannelId(), dep.ccuResSpaceTokenInfo};
        return ccuKernel->Add2ConstValue2VarMap(values);
    }

    HcclResult
    CcuInsGeneratorV2::PrepareLocCpyConstValue(CcuRepBase* repPtr, const TransDep& dep, CcuKernel* ccuKernel) const
    {
        CcuRepLocCpy* repLocCpyPtr = dynamic_cast<CcuRepLocCpy*>(repPtr);
        CHK_PTR_NULL(repLocCpyPtr);
        std::vector<uint64_t> values = {dep.reserveChannalId[0]};
        return ccuKernel->Add2ConstValue2VarMap(values);
    }

    HcclResult CcuInsGeneratorV2::PrepareRecordSharedNotifyConstValue(
        CcuRepBase* repPtr, const TransDep& dep, CcuKernel* ccuKernel) const
    {
        CcuRepRecordSharedNotify* sharedNotifyPtr = dynamic_cast<CcuRepRecordSharedNotify*>(repPtr);
        CHK_PTR_NULL(sharedNotifyPtr);
        uint64_t ckeAddr = dep.xnBaseAddr[sharedNotifyPtr->GetNotify().DieId()] + CCU_RESOURCE_XN_V2_RESERVE_SIZE
                           + (sharedNotifyPtr->GetNotify().Id() * CCU_RESOURCE_CKE_PER_SIZE);
        std::vector<uint64_t> values
            = {ckeAddr, dep.memTokenInfo, CCU_RESOURCE_CKE_MASK_LENGTH, sharedNotifyPtr->GetMask()};
        return ccuKernel->Add2ConstValue2VarMap(values);
    }

    HcclResult CcuInsGeneratorV2::LoopConfigTranslate(
        CcuInstr*& instr, uint16_t& curInstrId, const CcuRepLoopGroupBundle* bundlePtr, const TransDep& dep) const
    {
        for (const auto& loop : bundlePtr->GetLoops()) {
            CcuV2::LoadImdToXn(instr++, loop.ctxIdVar.Id(), loop.executor.Id());
            curInstrId++;
            if (loop.layout == CcuRepLoopGroupBundle::Layout::Config) {
                CcuV2::LoadImdToXn(instr++, loop.iterNumVar.Id(), loop.config.iterNum);
                curInstrId++;
                CcuV2::LoadImdToXn(instr++, loop.addrOffsetVar.Id(), loop.config.addrOffset);
                curInstrId++;
            } else if (loop.layout == CcuRepLoopGroupBundle::Layout::PackedVar) {
                // loopParamVar 布局：iterNum[12:0] gsaOffset[44:13] ctxId[52:45]
                CcuV2::LoadImdToXn(instr++, dep.reserveXnId, LOOP_ITER_NUM_MASK);
                curInstrId++;
                CcuV2::And(instr++, loop.iterNumVar.Id(), loop.loopParamVar.Id(), dep.reserveXnId);
                curInstrId++;
                CcuV2::LoadImdToXn(instr++, dep.reserveXnId, LOOP_GSA_OFFSET_SHIFT);
                curInstrId++;
                CcuV2::SRL(instr++, loop.addrOffsetVar.Id(), loop.loopParamVar.Id(), dep.reserveXnId);
                curInstrId++;
                CcuV2::LoadImdToXn(instr++, dep.reserveXnId, LOOP_GSA_OFFSET_MASK);
                curInstrId++;
                CcuV2::And(instr++, loop.addrOffsetVar.Id(), loop.addrOffsetVar.Id(), dep.reserveXnId);
                curInstrId++;
            }
        }
        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV2::LoopGroupConfigTranslate(
        CcuInstr*& instr, uint16_t& curInstrId, const CcuRepLoopGroupBundle* bundlePtr, const TransDep& dep,
        bool isConfig, bool isCompat, uint16_t& loopGroupConfigId) const
    {
        loopGroupConfigId = bundlePtr->GetParallelVar().Id();
        if (isConfig) {
            const auto& cfg = bundlePtr->GetConfig();
            uint64_t parallelImm
                = GetParallelParamV2(cfg.cloneNum, bundlePtr->GetRepeatLoopIdx(), bundlePtr->GetTotalLoopNum());
            CcuV2::LoadImdToXn(instr++, bundlePtr->GetParallelVar().Id(), parallelImm);
            curInstrId++;
            uint64_t offsetImm = GetOffsetParam(cfg.addrOffset, cfg.ccuBufferOffset, cfg.eventOffset);
            CcuV2::LoadImdToXn(instr++, bundlePtr->GetOffsetParam().Id(), offsetImm);
            curInstrId++;
            CcuV2::LoadImdToXn(instr++, bundlePtr->GetXnOffsetVar().Id(), cfg.varOffset);
            curInstrId++;
        } else if (isCompat) {
            const uint16_t newXm = bundlePtr->GetNewParallelVar().Id();
            const uint16_t scratch = bundlePtr->GetScratchVar().Id();
            const uint16_t src = bundlePtr->GetParallelVar().Id();
            struct FieldMap {
                uint16_t srcShift;
                uint16_t dstShift;
            };
            const FieldMap fields[] = {{41, 0}, {48, 10}, {55, 19}};
            for (size_t i = 0; i < sizeof(fields) / sizeof(fields[0]); i++) {
                const uint16_t dst = (i == 0) ? newXm : scratch;
                CcuV2::LoadImdToXn(instr++, dep.reserveXnId, fields[i].srcShift);
                curInstrId++;
                CcuV2::SRL(instr++, dst, src, dep.reserveXnId);
                curInstrId++;
                CcuV2::LoadImdToXn(instr++, dep.reserveXnId, LOOP_FIELD_MASK);
                curInstrId++;
                CcuV2::And(instr++, dst, dst, dep.reserveXnId);
                curInstrId++;
                if (fields[i].dstShift != 0) {
                    CcuV2::LoadImdToXn(instr++, dep.reserveXnId, fields[i].dstShift);
                    curInstrId++;
                    CcuV2::SLL(instr++, dst, dst, dep.reserveXnId);
                    curInstrId++;
                    CcuV2::Or(instr++, newXm, newXm, scratch);
                    curInstrId++;
                }
            }
            CcuV2::LoadImdToXn(instr++, dep.reserveXnId, 0);
            curInstrId++;
            loopGroupConfigId = newXm;
        }
        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult CcuInsGeneratorV2::CcuRepLoopGroupBundleTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepLoopGroupBundle* bundlePtr,
        const TransDep& dep)
    {
        UNUSED(ccuKernel);
        CHK_PTR_NULL(bundlePtr);
        const auto& loops = bundlePtr->GetLoops();
        const auto layout = bundlePtr->GetLayout();
        // 兼容路径：V2 下需插入位重排指令
        const bool isCompat = (layout == CcuRepLoopGroupBundle::Layout::PackedVar);
        const bool isConfig = (layout == CcuRepLoopGroupBundle::Layout::Config);

        // 每 loop 寄存器：按各 loop 自身构造方式载入，与 group 的构造方式相互独立
        CHK_RET(LoopConfigTranslate(instr, curInstrId, bundlePtr, dep));

        // loopGroupConfig：config 按 960 位分布打包立即数；兼容路径把旧 parallelVar 重排到新分布
        // 新分布：LoopNum[9:0] RepeatLoopIndex[18:10] ExtendNum[27:19]
        // 旧分布：totalLoopNum<<41 repeatLoopIndex<<48 repeatNum<<55
        uint16_t loopGroupConfigId{};
        CHK_RET(LoopGroupConfigTranslate(instr, curInstrId, bundlePtr, dep, isConfig, isCompat, loopGroupConfigId));

        // LoopGroup：loops 定义在其后第 3 条起（跳过 2 条无条件跳转）
        // xnOffset(Xp)：config/version960 有来源，兼容路径无来源填 0
        constexpr uint16_t kLoopEntryInstrOffset = 3;
        const uint16_t xnOffsetId = isCompat ? dep.reserveXnId : bundlePtr->GetXnOffsetVar().Id();
        CcuV2::LoopGroup(
            instr++, curInstrId + kLoopEntryInstrOffset, loopGroupConfigId, bundlePtr->GetOffsetParam().Id(),
            xnOffsetId); // 向后3条为loop入口
        curInstrId++;

        // 无条件跳转，跳过后续 loop 定义
        const uint16_t loopCount = static_cast<uint16_t>(loops.size());
        const uint16_t jumpInstrId = curInstrId + 1;
        const uint16_t targetInstrId = curInstrId + 2 + loopCount; // 跳转目的为loop指令后额外2条
        CcuV2::LoadImdToXn(instr++, dep.reserveXnId, GetRelativeInstrId(jumpInstrId, targetInstrId));
        curInstrId++;
        CcuV2::Jump(instr++, dep.reserveXnId, 0, 0, static_cast<int>(ConditionType::DEFAULT));
        curInstrId++;

        for (const auto& loop : loops) {
            const auto& block = loop.repLoopBlock;
            CHK_PTR_NULL(block);
            CcuV2::Loop(
                instr++, block->StartInstrId(), block->StartInstrId() + block->InstrCount() - 1, loop.iterNumVar.Id(),
                loop.addrOffsetVar.Id(), loop.ctxIdVar.Id());
            curInstrId++;
        }

        CcuV2::LoadImdToXn(instr++, dep.reserveXnId, 0);
        curInstrId++;

        return HcclResult::HCCL_SUCCESS;
    }

    uint16_t CcuInsGeneratorV2::CcuRepLoopGroupBundleInstrCount(const CcuRepLoopGroupBundle* bundlePtr) const
    {
        if (bundlePtr == nullptr) {
            Hccl::THROW<Hccl::CcuApiException>("[%s] bundlePtr is nullptr", __func__);
        }
        uint16_t total = 0;
        // 每 loop 按自身构造方式计条数
        constexpr uint16_t kConfigLoopInstrCount = 4;       // 3 条载入(ctxId/iterNum/gsaOffset) + 1 条 Loop
        constexpr uint16_t kVersionV2LoopInstrCount = 2;    // 1 条 ctxId 载入 + 1 条 Loop
        constexpr uint16_t kPackedVarLoopInstrCount = 8;    // 7 条(ctxId1 + 位重排6) + 1 条 Loop
        constexpr uint16_t kConfigGroupHeaderCount = 7;     // 打包3 + loopgroup1 + 跳过2 + 收尾1
        constexpr uint16_t kVersionV2GroupHeaderCount = 4;  // loopgroup1 + 跳过2 + 收尾1
        constexpr uint16_t kPackedVarGroupHeaderCount = 23; // 位重排19 + loopgroup1 + 跳过2 + 收尾1
        for (const auto& loop : bundlePtr->GetLoops()) {
            switch (loop.layout) {
                case CcuRepLoopGroupBundle::Layout::Config:
                    total += kConfigLoopInstrCount;
                    break;
                case CcuRepLoopGroupBundle::Layout::VersionV2:
                    total += kVersionV2LoopInstrCount;
                    break;
                case CcuRepLoopGroupBundle::Layout::PackedVar:
                default:
                    total += kPackedVarLoopInstrCount;
                    break;
            }
        }
        // group 头按 group 自身构造方式计条数
        switch (bundlePtr->GetLayout()) {
            case CcuRepLoopGroupBundle::Layout::Config:
                total += kConfigGroupHeaderCount;
                break;
            case CcuRepLoopGroupBundle::Layout::VersionV2:
                total += kVersionV2GroupHeaderCount;
                break;
            case CcuRepLoopGroupBundle::Layout::PackedVar:
            default:
                total += kPackedVarGroupHeaderCount;
                break;
        }
        return total;
    }

} // namespace CcuRep
} // namespace hcomm
