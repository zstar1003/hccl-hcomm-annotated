/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_task_transform_v3.h"
#include "type_conversion.h"
#include "ccu_all_rank_param_recorder_v3.h"
#include "ccu_task_common_v3.h"
#include "storage_manager.h"
#include "sim_log.h"
#include "ccu_loop_merge_v3.h"
#include "ccu_task_transform_instruct_common_v3.h"
#include "utils/error_codes.h"

namespace {

constexpr uint16_t LOADSQEARGSTOGSA_CODE = 0x0;
constexpr uint16_t LOADSQEARGSTOXN_CODE = 0x1;
constexpr uint16_t LOADIMDTOGSA_CODE = 0x2;
constexpr uint16_t LOADIMDTOXN_CODE = 0x3;
constexpr uint16_t LOADGSAXN_CODE = 0x4;
constexpr uint16_t LOADGSAGSA_CODE = 0x5;
constexpr uint16_t LOADXX_CODE = 0x6;

constexpr uint16_t LOOP_CODE = 0x0;
constexpr uint16_t LOOPGROUP_CODE = 0x1;
constexpr uint16_t SETCKE_CODE = 0x2;
constexpr uint16_t CLEARCKE_CODE = 0x4;
constexpr uint16_t JMP_CODE = 0x5;

constexpr uint16_t TRANSLOCMEMTOLOCMS_CODE = 0x0;
constexpr uint16_t TRANSRMTMEMTOLOCMS_CODE = 0x1;
constexpr uint16_t TRANSLOCMSTOLOCMEM_CODE = 0x2;
constexpr uint16_t TRANSLOCMSTORMTMEM_CODE = 0x3;
constexpr uint16_t TRANSRMTMSTOLOCMEM_CODE = 0x4;
constexpr uint16_t TRANSLOCMSTOLOCMS_CODE = 0x5;
constexpr uint16_t TRANSRMTMSTOLOCMS_CODE = 0x6;
constexpr uint16_t TRANSLOCMSTORMTMS_CODE = 0x7;
constexpr uint16_t TRANSRMTMEMTOLOCMEM_CODE = 0x8;
constexpr uint16_t TRANSLOCMEMTORMTMEM_CODE = 0x9;
constexpr uint16_t TRANSLOCMEMTOLOCMEM_CODE = 0xa;
constexpr uint16_t SYNCCKE_CODE = 0xb;
constexpr uint16_t SYNCGSA_CODE = 0xc;
constexpr uint16_t SYNCXN_CODE = 0xd;

constexpr uint16_t ADD_CODE = 0x0;
constexpr uint16_t MAX_CODE = 0x1;
constexpr uint16_t MIN_CODE = 0x2;

constexpr uint64_t UB_MAX_SIZE = 256 * 1024 * 1024;

#define CHK_GET_XN_V3(curCcuTask, queId, xnId, xnValue)                                                        \
    do {                                                                                                       \
        uint32_t _checkerDieId = INVALID_DIE_ID;                                                               \
        (curCcuTask)->GetDieId((queId), _checkerDieId);                                                        \
        const uint32_t _checkerInstrId                                                                         \
            = (curCcuTask)->microCodePosInQue[(queId)] + (curCcuTask)->startInstrIdInQue[(queId)];             \
        if (AllRankParamRecorder::Global()->GetXn((curCcuTask)->GetRankId(), _checkerDieId, (xnId), (xnValue)) \
            != HCCL_SUCCESS) {                                                                                 \
            HCCL_VM_ERROR(                                                                                     \
                "{} Failed to read XN register before it was initialized, rankId={}, "                         \
                "dieId={}, instrId={}, xnId={}",                                                               \
                MakeErrorCodeText(ErrorCode::GRAPH_REGISTER_UNINITIALIZED).c_str(),                            \
                static_cast<uint32_t>((curCcuTask)->GetRankId()), _checkerDieId, _checkerInstrId,              \
                static_cast<uint16_t>(xnId));                                                                  \
            return HCCL_E_PARA;                                                                                \
        }                                                                                                      \
    } while (0)

#define CHK_GET_GSA_V3(curCcuTask, queId, gsaId, gsaValue)                                                        \
    do {                                                                                                          \
        uint32_t _checkerDieId = INVALID_DIE_ID;                                                                  \
        (curCcuTask)->GetDieId((queId), _checkerDieId);                                                           \
        const uint32_t _checkerInstrId                                                                            \
            = (curCcuTask)->microCodePosInQue[(queId)] + (curCcuTask)->startInstrIdInQue[(queId)];                \
        if (AllRankParamRecorder::Global()->GetGSA((curCcuTask)->GetRankId(), _checkerDieId, (gsaId), (gsaValue)) \
            != HCCL_SUCCESS) {                                                                                    \
            HCCL_VM_ERROR(                                                                                        \
                "{} Failed to read GSA register before it was initialized, rankId={}, "                           \
                "dieId={}, instrId={}, gsaId={}",                                                                 \
                MakeErrorCodeText(ErrorCode::GRAPH_REGISTER_UNINITIALIZED).c_str(),                               \
                static_cast<uint32_t>((curCcuTask)->GetRankId()), _checkerDieId, _checkerInstrId,                 \
                static_cast<uint16_t>(gsaId));                                                                    \
            return HCCL_E_PARA;                                                                                   \
        }                                                                                                         \
    } while (0)

} // namespace

namespace HcclSim {
namespace TaskGraphGeneratorV3 {

    static std::map<uint16_t, uint16_t> ccuReduceTypeMap
        = {{10, CcuRep::CCU_REDUCE_SUM}, {9, CcuRep::CCU_REDUCE_MIN}, {8, CcuRep::CCU_REDUCE_MAX}};

    union LoopXm {
        uint64_t value;
        struct {
            uint64_t loopCnt : 13;
            uint64_t gsaStride : 32;
            uint64_t loopCtxId : 8;
            uint64_t reserved : 11;
        };
    };

    union LoopGroupXn {
        uint64_t value;
        struct {
            uint64_t reservedLow : 41;
            uint64_t loopInsCnt : 7;
            uint64_t expandOffset : 7;
            uint64_t expandCnt : 7;
            uint64_t reservedHigh : 2;
        };
    };

    union LoopGroupXm {
        uint64_t value;
        struct {
            uint64_t ckOffset : 10;
            uint64_t msOffset : 11;
            uint64_t gsaOffset : 32;
            uint64_t reserved : 11;
        };
    };

    struct ErrorInfoBase {
        int32_t deviceId;
        uint8_t dieId;
        uint8_t missionId;
        uint16_t currentInsId;
        uint16_t status;
    };

    struct LoopGroupParam {
        std::vector<LoopXm> loopXms;
        LoopGroupXn loopGroupXn;
        LoopGroupXm loopGroupXm;
        u32 curLoopIdx = 0;   // 表示当前处理第几个loop
        u32 curExpandCnt = 0; // 该loop被第几次展开
        u32 curLoopCnt = 0;   // 表示当前Loop第几次循环
    };

    HcclResult IsLoopGroupParamNull(LoopGroupParam* loopGroupParam)
    {
        if (loopGroupParam != nullptr) {
            HCCL_VM_ERROR(
                "{} This A5 instruction cannot be used inside a loop expansion.",
                MakeErrorCodeText(ErrorCode::GRAPH_UNSUPPORTED).c_str());
            return HCCL_E_INTERNAL;
        }
        return HCCL_SUCCESS;
    }

    HcclResult TransformLoadSqeArgsToGSAInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
        LoopGroupParam* loopGroupParam)
    {
        CHK_RET(IsLoopGroupParamNull(loopGroupParam));
        uint16_t sqeArgsId = instr->v1.loadSqeArgsToGSA.sqeArgsId;
        uint16_t gsaId = instr->v1.loadSqeArgsToGSA.gsaId;
        uint64_t argVal = 0;
        curCcuTask->GetSqe(queId, sqeArgsId, argVal);

        RankId rankId = curCcuTask->GetRankId();
        uint32_t dieId;
        curCcuTask->GetDieId(queId, dieId);

        CHK_RET(AllRankParamRecorder::Global()->SetGSA(rankId, dieId, gsaId, argVal));
        HCCL_VM_DEBUG("Load SqeArg[{}]({}) to GSA[{}]", sqeArgsId, argVal, gsaId);
        return HCCL_SUCCESS;
    }

    HcclResult TransformLoadSqeArgsToXnInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
        LoopGroupParam* loopGroupParam)
    {
        CHK_RET(IsLoopGroupParamNull(loopGroupParam));
        uint16_t sqeArgsId = instr->v1.loadSqeArgsToXn.sqeArgsId;
        uint16_t xnId = instr->v1.loadSqeArgsToXn.xnId;
        uint64_t argVal = 0;
        curCcuTask->GetSqe(queId, sqeArgsId, argVal);

        RankId rankId = curCcuTask->GetRankId();
        uint32_t dieId;
        curCcuTask->GetDieId(queId, dieId);

        CHK_RET(AllRankParamRecorder::Global()->SetXn(rankId, dieId, xnId, argVal));
        HCCL_VM_DEBUG("Load SqeArg[{}]({}) to Xn[{}]", sqeArgsId, argVal, xnId);
        return HCCL_SUCCESS;
    }

    HcclResult TransformLoadImdToGSAInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
        LoopGroupParam* loopGroupParam)
    {
        CHK_RET(IsLoopGroupParamNull(loopGroupParam));
        uint16_t gsaId = instr->v1.loadImdToGSA.gsaId;
        uint64_t immediate = instr->v1.loadImdToGSA.immediate;

        RankId rankId = curCcuTask->GetRankId();
        uint32_t dieId;
        curCcuTask->GetDieId(queId, dieId);

        CHK_RET(AllRankParamRecorder::Global()->SetGSA(rankId, dieId, gsaId, immediate));
        HCCL_VM_DEBUG("Load immediate[{}] to GSA[{}]", immediate, gsaId);
        return HCCL_SUCCESS;
    }

    HcclResult TransformLoadImdToXnInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
        LoopGroupParam* loopGroupParam)
    {
        CHK_RET(IsLoopGroupParamNull(loopGroupParam));
        uint16_t xnId = instr->v1.loadImdToXn.xnId;
        uint64_t immediate = instr->v1.loadImdToXn.immediate;

        RankId rankId = curCcuTask->GetRankId();
        uint32_t dieId;
        curCcuTask->GetDieId(queId, dieId);

        CHK_RET(AllRankParamRecorder::Global()->SetXn(rankId, dieId, xnId, immediate));
        HCCL_VM_DEBUG("Load immediate[{}] to Xn[{}]", immediate, xnId);
        return HCCL_SUCCESS;
    }

    HcclResult TransformLoadGSAXnInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
        LoopGroupParam* loopGroupParam)
    {
        CHK_RET(IsLoopGroupParamNull(loopGroupParam));
        uint16_t gsAmId = instr->v1.loadGSAXn.gsAmId;
        uint16_t xnId = instr->v1.loadGSAXn.xnId;
        uint16_t gsAdId = instr->v1.loadGSAXn.gsAdId;

        RankId rankId = curCcuTask->GetRankId();
        uint32_t dieId;
        curCcuTask->GetDieId(queId, dieId);

        uint64_t argVal1 = 0;
        uint64_t argVal2 = 0;
        CHK_GET_GSA_V3(curCcuTask, queId, gsAmId, argVal1);
        CHK_GET_XN_V3(curCcuTask, queId, xnId, argVal2);
        CHK_RET(AllRankParamRecorder::Global()->SetGSA(rankId, dieId, gsAdId, argVal1 + argVal2));
        HCCL_VM_DEBUG(
            "Load GSA[{}]({}) + Xn[{}]({}) to GSA[{}]({})", gsAmId, argVal1, xnId, argVal2, gsAdId, argVal1 + argVal2);
        return HCCL_SUCCESS;
    }

    HcclResult TransformLoadGSAGSAInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
        LoopGroupParam* loopGroupParam)
    {
        CHK_RET(IsLoopGroupParamNull(loopGroupParam));
        uint16_t gsAmId = instr->v1.loadGSAGSA.gsAmId;
        uint16_t gsAnId = instr->v1.loadGSAGSA.gsAnId;
        uint16_t gsAdId = instr->v1.loadGSAGSA.gsAdId;

        RankId rankId = curCcuTask->GetRankId();
        uint32_t dieId;
        curCcuTask->GetDieId(queId, dieId);

        uint64_t argVal1 = 0;
        uint64_t argVal2 = 0;
        CHK_GET_GSA_V3(curCcuTask, queId, gsAmId, argVal1);
        CHK_GET_GSA_V3(curCcuTask, queId, gsAnId, argVal2);
        CHK_RET(AllRankParamRecorder::Global()->SetGSA(rankId, dieId, gsAdId, argVal1 + argVal2));
        HCCL_VM_DEBUG(
            "Load GSA[{}]({}) + GSA[{}]({}) to GSA[{}]({})", gsAmId, argVal1, gsAnId, argVal2, gsAdId,
            argVal1 + argVal2);
        return HCCL_SUCCESS;
    }

    HcclResult TransformLoadXXInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
        LoopGroupParam* loopGroupParam)
    {
        CHK_RET(IsLoopGroupParamNull(loopGroupParam));
        uint16_t xmId = instr->v1.loadXX.xmId;
        uint16_t xnId = instr->v1.loadXX.xnId;
        uint16_t xdId = instr->v1.loadXX.xdId;
        uint64_t argVal1 = 0;
        uint64_t argVal2 = 0;

        RankId rankId = curCcuTask->GetRankId();
        uint32_t dieId;
        curCcuTask->GetDieId(queId, dieId);

        CHK_GET_XN_V3(curCcuTask, queId, xmId, argVal1);
        CHK_GET_XN_V3(curCcuTask, queId, xnId, argVal2);
        CHK_RET(AllRankParamRecorder::Global()->SetXn(rankId, dieId, xdId, argVal1 + argVal2));
        HCCL_VM_DEBUG(
            "Load Xn[{}]({}) + Xn[{}]({}) to Xn[{}]({})", xmId, argVal1, xnId, argVal2, xdId, argVal1 + argVal2);
        return HCCL_SUCCESS;
    }

    HcclResult TransformSetCKEInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
        LoopGroupParam* loopGroupParam)
    {
        CHK_RET(IsLoopGroupParamNull(loopGroupParam));
        uint16_t clearType = instr->v1.setCKE.clearType;
        uint16_t setCKEId = instr->v1.setCKE.setCKEId;
        uint16_t setCKEMask = instr->v1.setCKE.setCKEMask;
        uint16_t waitCKEId = instr->v1.setCKE.waitCKEId;
        uint16_t waitCKEMask = instr->v1.setCKE.waitCKEMask;
        // 当前只支持clearType为1的场景
        if (clearType != 0x0001) {
            HCCL_VM_ERROR(
                "{} CheckerV3 only supports clearType=1 for this instruction, "
                "instruction=SetCKE, actualClearType={}",
                MakeErrorCodeText(ErrorCode::GRAPH_UNSUPPORTED).c_str(), clearType);
            return HCCL_E_INTERNAL;
        }

        RankId rankId = curCcuTask->GetRankId();
        uint32_t dieId;
        curCcuTask->GetDieId(queId, dieId);

        CHK_RET(ProcessWaitMask(rankId, dieId, curCcuTask, queId, waitCKEId, waitCKEMask, isContinue));
        if (!isContinue) {
            return HCCL_SUCCESS;
        }
        CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask, true));
        CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));
        HCCL_VM_DEBUG(
            "Wait CKE[{}:0x{:04x}], Set CKE[{}:0x{:04x}], clearType[{}]", waitCKEId, waitCKEMask, setCKEId, setCKEMask,
            clearType);
        return HCCL_SUCCESS;
    }

    // 循环中CKE需要进行自动偏移
    uint16_t UpdateCKEId(uint16_t CKEId, LoopGroupParam* loopGroupParam)
    {
        // 不在循环中，不需要刷新
        if (loopGroupParam == nullptr) {
            return CKEId;
        }

        HCCL_VM_DEBUG(
            "Update CKEId[{}], loopIdx[{}], ckOffset[{}]", CKEId, loopGroupParam->curLoopIdx,
            static_cast<int>(loopGroupParam->loopGroupXm.ckOffset));
        return UpdateId(
            CKEId, loopGroupParam->curLoopIdx, loopGroupParam->loopGroupXn.expandOffset,
            loopGroupParam->loopGroupXm.ckOffset, loopGroupParam->curExpandCnt);
    }

    // 循环中MS需要进行自动偏移
    uint16_t UpdateMSId(uint16_t MSId, LoopGroupParam* loopGroupParam)
    {
        // 不在循环中，不需要刷新
        if (loopGroupParam == nullptr) {
            return MSId;
        }

        return UpdateId(
            MSId, loopGroupParam->curLoopIdx, loopGroupParam->loopGroupXn.expandOffset,
            loopGroupParam->loopGroupXm.msOffset, loopGroupParam->curExpandCnt);
    }

    // 更新GSA的地址
    uint64_t UpdateGSAValue(uint64_t gsaAddr, LoopGroupParam* loopGroupParam)
    {
        // 不在循环中，不需要刷新
        if (loopGroupParam == nullptr) {
            return gsaAddr;
        }

        LoopXm& xm = loopGroupParam->loopXms[loopGroupParam->curLoopIdx];

        if (loopGroupParam->curLoopIdx < loopGroupParam->loopGroupXn.expandOffset) {
            return gsaAddr + loopGroupParam->curLoopCnt * xm.gsaStride;
        }

        return gsaAddr + loopGroupParam->curExpandCnt * loopGroupParam->loopGroupXm.gsaOffset
               + loopGroupParam->curLoopCnt * xm.gsaStride;
    }

    // todo: 这边可能支持reduce，需要处理一下
    HcclResult TransformTransLocMemToRmtMemInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
        LoopGroupParam* loopGroupParam)
    {
        CHK_RET(IsLoopGroupParamNull(loopGroupParam));
        uint16_t rmtGSAId = instr->v1.transLocMemToRmtMem.rmtGSAId;
        uint16_t rmtXnId = instr->v1.transLocMemToRmtMem.rmtXnId;
        uint16_t locGSAId = instr->v1.transLocMemToRmtMem.locGSAId;
        uint16_t locXnId = instr->v1.transLocMemToRmtMem.locXnId;
        uint16_t lengthXnId = instr->v1.transLocMemToRmtMem.lengthXnId;
        uint16_t channelId = instr->v1.transLocMemToRmtMem.channelId;
        uint16_t reduceDataType = instr->v1.transLocMemToRmtMem.reduceDataType;
        uint16_t reduceOpCode = instr->v1.transLocMemToRmtMem.reduceOpCode;
        uint16_t clearType = instr->v1.transLocMemToRmtMem.clearType;
        uint16_t lengthEn = instr->v1.transLocMemToRmtMem.lengthEn;
        uint16_t reduceEn = instr->v1.transLocMemToRmtMem.reduceEn;
        uint16_t setCKEId = instr->v1.transLocMemToRmtMem.setCKEId;
        uint16_t setCKEMask = instr->v1.transLocMemToRmtMem.setCKEMask;
        uint16_t waitCKEId = instr->v1.transLocMemToRmtMem.waitCKEId;
        uint16_t waitCKEMask = instr->v1.transLocMemToRmtMem.waitCKEMask;

        RankId rankId = curCcuTask->GetRankId();
        uint32_t dieId;
        curCcuTask->GetDieId(queId, dieId);

        CHK_RET(ProcessWaitMask(rankId, dieId, curCcuTask, queId, waitCKEId, waitCKEMask, isContinue));
        if (!isContinue) {
            return HCCL_SUCCESS;
        }

        // 获取远端rankId与dieId
        // todo: 从共享内存获取channel映射表
        if (g_allRankChannelInfo[rankId][dieId].count(channelId) == 0) {
            HCCL_VM_ERROR(
                "{} Cannot find channel mapping for this instruction, instruction=TransLocMemToRmtMem, "
                "localRankId={}, localDieId={}, channelId={}",
                MakeErrorCodeText(ErrorCode::GRAPH_RESOURCE_NOT_FOUND).c_str(), static_cast<uint32_t>(rankId), dieId,
                channelId);
            return HCCL_E_INTERNAL;
        }

        uint64_t locAddr = 0x0;
        uint64_t rmtAddr = 0x0;
        CHK_GET_GSA_V3(curCcuTask, queId, locGSAId, locAddr);
        CHK_GET_GSA_V3(curCcuTask, queId, rmtGSAId, rmtAddr);
        uint64_t len = 0;
        CHK_GET_XN_V3(curCcuTask, queId, lengthXnId, len);
        const uint32_t instrId = curCcuTask->microCodePosInQue[queId] + curCcuTask->startInstrIdInQue[queId];
        if (len == 0) {
            HCCL_VM_ERROR(
                "{} Transfer size is 0, instruction=TransLocMemToRmtMem:\n"
                "  ctx: rankId={}, dieId={}, queueId={}, instrId={}\n"
                "  size: transferSize={}",
                MakeErrorCodeText(ErrorCode::GRAPH_STRUCTURE_INVALID).c_str(), static_cast<uint32_t>(rankId), dieId,
                queId, instrId, static_cast<unsigned long long>(len));
            return HCCL_E_INTERNAL;
        }
        if (len > UB_MAX_SIZE) {
            HCCL_VM_ERROR(
                "{} Transfer size exceeds the UB upper limit, instruction=TransLocMemToRmtMem:\n"
                "  size: transferSize={}, ubLimit={}\n"
                "  ctx: rankId={}, dieId={}, queueId={}, instrId={}",
                MakeErrorCodeText(ErrorCode::GRAPH_STRUCTURE_INVALID).c_str(), static_cast<unsigned long long>(len),
                static_cast<unsigned long long>(UB_MAX_SIZE), static_cast<uint32_t>(rankId), dieId, queId, instrId);
            return HCCL_E_INTERNAL;
        }
        DataSlice srcSlice;
        DataSlice dstSlice;
        RankId rId;
        // todo: 后续插件通过读取vm的输出文件，重建memlayout的地址类型信息
        // 南向接口：input/output buffer同北向一样劫持算子入口函数；CCL buffer需要劫持北向接口HcclGetHcclBuffer
        CHK_RET(curCcuTask->GetStorageManager().GetSlice(locAddr, len, srcSlice));
        CHK_RET(curCcuTask->GetStorageManager().GetSlice(rmtAddr, len, dstSlice, &rId));

        RankId rmtRankId = g_allRankChannelInfo[rankId][dieId][channelId].dstRank;
        HCCL_VM_DEBUG(
            "Preparing local-memory to remote-memory transfer, localRankId={}, remoteRankId={}", rankId, rmtRankId);
        // 说明transportId写错了
        if (rId != rmtRankId) {
            HCCL_VM_ERROR(
                "{} Remote address resolves to a different rank than the selected channel, "
                "instruction=TransLocMemToRmtMem:\n"
                "  ctx: rankId={}, dieId={}, queueId={}, instrId={}, channelId={}\n"
                "  rank: expectedRemoteRankId={}, actualRemoteRankId={}\n"
                "  addr: remoteAddr={}",
                MakeErrorCodeText(ErrorCode::GRAPH_REMOTE_RANK_MISMATCH).c_str(), static_cast<uint32_t>(rankId), dieId,
                queId, instrId, channelId, static_cast<uint32_t>(rmtRankId), static_cast<uint32_t>(rId),
                static_cast<unsigned long long>(rmtAddr));
            return HCCL_E_INTERNAL;
        }
        HcclReduceOp checkerReduceOp;
        HcclDataType checkerDataType;
        if (reduceEn) {
            checkerReduceOp = g_ReduceOp2CheckerReduceOp_ccu[reduceOpCode];
            DataType hcclDataType;
            CHK_RET(GetHcclDataTypeFromCCUDataType(reduceDataType, ccuReduceTypeMap[reduceOpCode], hcclDataType));
            checkerDataType = g_DataType2CheckerDataType_aicpu[hcclDataType];
        }

        if (rmtRankId != rankId) {
            if (reduceEn) {
                AddWriteReduce(
                    rankId, rmtRankId, queId, curCcuTask, srcSlice, dstSlice, checkerDataType, checkerReduceOp);
            } else {
                AddWrite(rankId, rmtRankId, queId, curCcuTask, srcSlice, dstSlice);
            }
        } else {
            if (reduceEn) {
                AddLocalReduce(rankId, queId, curCcuTask, srcSlice, dstSlice, checkerDataType, checkerReduceOp);
            } else {
                AddLocalCopy(rankId, queId, curCcuTask, srcSlice, dstSlice);
            }
        }

        CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask));
        CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

        HCCL_VM_DEBUG(
            "Completed local-memory to remote-memory transfer:\n"
            "  wait: waitCkeId={}, waitMask=0x{:04x}\n"
            "  src: srcGsaId={}, srcOffsetXnId={}\n"
            "  dst: dstGsaId={}, dstOffsetXnId={}\n"
            "  xfer: lengthXnId={}, channelId={}\n"
            "  set: setCkeId={}, setMask=0x{:04x}\n"
            "  ctrl: clearType={}, lengthEnabled={}, reduceEnabled={}, reduceDataType={}, reduceOpCode={}",
            waitCKEId, waitCKEMask, locGSAId, locXnId, rmtGSAId, rmtXnId, lengthXnId, channelId, setCKEId, setCKEMask,
            clearType, lengthEn, reduceEn, reduceDataType, reduceOpCode);
        return HCCL_SUCCESS;
    }

    HcclResult TransformTransLocMemToLocMemInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
        LoopGroupParam* loopGroupParam)
    {
        CHK_RET(IsLoopGroupParamNull(loopGroupParam));
        uint16_t dstGSAId = instr->v1.transLocMemToLocMem.dstGSAId;
        uint16_t dstXnId = instr->v1.transLocMemToLocMem.dstXnId;
        uint16_t srcGSAId = instr->v1.transLocMemToLocMem.srcGSAId;
        uint16_t srcXnId = instr->v1.transLocMemToLocMem.srcXnId;
        uint16_t lengthXnId = instr->v1.transLocMemToLocMem.lengthXnId;
        uint16_t channelId = instr->v1.transLocMemToLocMem.channelId;
        uint16_t clearType = instr->v1.transLocMemToLocMem.clearType;
        uint16_t lengthEn = instr->v1.transLocMemToLocMem.lengthEn;
        uint16_t setCKEId = instr->v1.transLocMemToLocMem.setCKEId;
        uint16_t setCKEMask = instr->v1.transLocMemToLocMem.setCKEMask;
        uint16_t waitCKEId = instr->v1.transLocMemToLocMem.waitCKEId;
        uint16_t waitCKEMask = instr->v1.transLocMemToLocMem.waitCKEMask;

        RankId rankId = curCcuTask->GetRankId();
        uint32_t dieId;
        curCcuTask->GetDieId(queId, dieId);

        CHK_RET(ProcessWaitMask(rankId, dieId, curCcuTask, queId, waitCKEId, waitCKEMask, isContinue));
        if (!isContinue) {
            return HCCL_SUCCESS;
        }

        uint64_t srcAddr = 0x0;
        uint64_t dstAddr = 0x0;
        CHK_GET_GSA_V3(curCcuTask, queId, srcGSAId, srcAddr);
        CHK_GET_GSA_V3(curCcuTask, queId, dstGSAId, dstAddr);
        uint64_t len = 0;
        CHK_GET_XN_V3(curCcuTask, queId, lengthXnId, len);
        const uint32_t instrId = curCcuTask->microCodePosInQue[queId] + curCcuTask->startInstrIdInQue[queId];
        if (len == 0) {
            HCCL_VM_ERROR(
                "{} Transfer size is 0, instruction=TransLocMemToLocMem:\n"
                "  ctx: rankId={}, dieId={}, queueId={}, instrId={}\n"
                "  size: transferSize={}",
                MakeErrorCodeText(ErrorCode::GRAPH_STRUCTURE_INVALID).c_str(), static_cast<uint32_t>(rankId), dieId,
                queId, instrId, static_cast<unsigned long long>(len));
            return HCCL_E_INTERNAL;
        }
        if (len > UB_MAX_SIZE) {
            HCCL_VM_ERROR(
                "{} Transfer size exceeds the UB upper limit, instruction=TransLocMemToLocMem:\n"
                "  size: transferSize={}, ubLimit={}\n"
                "  ctx: rankId={}, dieId={}, queueId={}, instrId={}",
                MakeErrorCodeText(ErrorCode::GRAPH_STRUCTURE_INVALID).c_str(), static_cast<unsigned long long>(len),
                static_cast<unsigned long long>(UB_MAX_SIZE), static_cast<uint32_t>(rankId), dieId, queId, instrId);
            return HCCL_E_INTERNAL;
        }

        DataSlice srcSlice;
        DataSlice dstSlice;
        // todo: 后续插件通过读取vm的输出文件，重建memlayout的地址类型信息
        CHK_RET(curCcuTask->GetStorageManager().GetSlice(srcAddr, len, srcSlice));
        CHK_RET(curCcuTask->GetStorageManager().GetSlice(dstAddr, len, dstSlice));

        HCCL_VM_DEBUG(
            "Built local-memory to local-memory copy, rankId={}, srcSlice={}, dstSlice={}, "
            "transferSize={}",
            rankId, srcSlice.Describe(), dstSlice.Describe(), len);

        AddLocalCopy(rankId, queId, curCcuTask, srcSlice, dstSlice);

        CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask));

        CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

        HCCL_VM_DEBUG(
            "Completed local-memory to local-memory transfer:\n"
            "  wait: waitCkeId={}, waitMask=0x{:04x}\n"
            "  src: srcGsaId={}, srcOffsetXnId={}\n"
            "  dst: dstGsaId={}, dstOffsetXnId={}\n"
            "  xfer: lengthXnId={}, channelId={}\n"
            "  set: setCkeId={}, setMask=0x{:04x}\n"
            "  ctrl: clearType={}, lengthEnabled={}",
            waitCKEId, waitCKEMask, srcGSAId, srcXnId, dstGSAId, dstXnId, lengthXnId, channelId, setCKEId, setCKEMask,
            clearType, lengthEn);
        return HCCL_SUCCESS;
    }

    HcclResult TransformSyncCKEInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
        LoopGroupParam* loopGroupParam)
    {
        CHK_RET(IsLoopGroupParamNull(loopGroupParam));
        uint16_t rmtCKEId = instr->v1.syncCKE.rmtCKEId;
        uint16_t locCKEId = instr->v1.syncCKE.locCKEId;
        uint16_t locCKEMask = instr->v1.syncCKE.locCKEMask;
        uint16_t channelId = instr->v1.syncCKE.channelId;
        uint16_t clearType = instr->v1.syncCKE.clearType;
        uint16_t setCKEId = instr->v1.syncCKE.setCKEId;
        uint16_t setCKEMask = instr->v1.syncCKE.setCKEMask;
        uint16_t waitCKEId = instr->v1.syncCKE.waitCKEId;
        uint16_t waitCKEMask = instr->v1.syncCKE.waitCKEMask;

        RankId rankId = curCcuTask->GetRankId();
        uint32_t dieId;
        curCcuTask->GetDieId(queId, dieId);

        CHK_RET(ProcessWaitMask(rankId, dieId, curCcuTask, queId, waitCKEId, waitCKEMask, isContinue));
        if (!isContinue) {
            return HCCL_SUCCESS;
        }

        // 获取远端rankId与dieId
        if (g_allRankChannelInfo[rankId][dieId].count(channelId) == 0) {
            HCCL_VM_ERROR(
                "{} Cannot find channel mapping for this instruction, instruction=SyncCKE, "
                "localRankId={}, localDieId={}, channelId={}",
                MakeErrorCodeText(ErrorCode::GRAPH_RESOURCE_NOT_FOUND).c_str(), static_cast<uint32_t>(rankId), dieId,
                channelId);
            return HCCL_E_INTERNAL;
        }

        RankId rmtRankId = g_allRankChannelInfo[rankId][dieId][channelId].dstRank;
        uint32_t rmtDieId = g_allRankChannelInfo[rankId][dieId][channelId].remoteDieId;

        if (locCKEMask != 0x0000) {
            AddPost(rankId, rmtRankId, queId, curCcuTask, rmtDieId, rmtCKEId, locCKEMask);

            // 将对应的bit位置一
            uint16_t ckeValue = 0;
            uint16_t localCkeValue = 0;
            CHK_RET(AllRankParamRecorder::Global()->GetCKE(rankId, dieId, locCKEId, localCkeValue));
            const uint16_t maskedLocalCkeValue = static_cast<uint16_t>(localCkeValue & locCKEMask);
            HCCL_VM_DEBUG(
                "CKE[{}]({}) & mask[0x{:04x}] -> {}", locCKEId, localCkeValue, locCKEMask, maskedLocalCkeValue);
            CHK_RET(AllRankParamRecorder::Global()->GetCKE(rmtRankId, rmtDieId, rmtCKEId, ckeValue));
            const uint16_t mergedCkeValue = static_cast<uint16_t>(ckeValue | maskedLocalCkeValue);
            CHK_RET(AllRankParamRecorder::Global()->SetCKE(rmtRankId, rmtDieId, rmtCKEId, mergedCkeValue));
            HCCL_VM_DEBUG(
                "CKE[{}]({}) | {} -> CKE[{}]({})", rmtCKEId, ckeValue, maskedLocalCkeValue, rmtCKEId, mergedCkeValue);
        }

        CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask, true));

        CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

        HCCL_VM_DEBUG(
            "Wait CKE[{}:0x{:04x}]:\n"
            "  sync: Sync LocCKE[{}:{:04x}] To rmtCKE[{}:{:04x}] Use Channel[{}]\n"
            "  set: Set CKE[{}:0x{:04x}]\n"
            "  ctrl: clearType[{}]",
            waitCKEId, waitCKEMask, locCKEId, locCKEMask, rmtCKEId, locCKEMask, channelId, setCKEId, setCKEMask,
            clearType);
        return HCCL_SUCCESS;
    }

    HcclResult TransformSyncXnInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
        LoopGroupParam* loopGroupParam)
    {
        CHK_RET(IsLoopGroupParamNull(loopGroupParam));
        uint16_t rmtXnId = instr->v1.syncXn.rmtXnId;
        uint16_t locXnId = instr->v1.syncXn.locXnId;
        uint16_t channelId = instr->v1.syncXn.channelId;
        uint16_t setRmtCKEId = instr->v1.syncXn.setRmtCKEId;
        uint16_t setRmtCKEMask = instr->v1.syncXn.setRmtCKEMask;
        uint16_t clearType = instr->v1.syncXn.clearType;
        uint16_t setCKEId = instr->v1.syncXn.setCKEId;
        uint16_t setCKEMask = instr->v1.syncXn.setCKEMask;
        uint16_t waitCKEId = instr->v1.syncXn.waitCKEId;
        uint16_t waitCKEMask = instr->v1.syncXn.waitCKEMask;

        RankId rankId = curCcuTask->GetRankId();
        uint32_t dieId;
        curCcuTask->GetDieId(queId, dieId);

        CHK_RET(ProcessWaitMask(rankId, dieId, curCcuTask, queId, waitCKEId, waitCKEMask, isContinue));
        if (!isContinue) {
            return HCCL_SUCCESS;
        }

        // 获取远端rankId与dieId
        if (g_allRankChannelInfo[rankId][dieId].count(channelId) == 0) {
            HCCL_VM_ERROR(
                "{} Cannot find channel mapping for this instruction, instruction=SyncXn, "
                "localRankId={}, localDieId={}, channelId={}",
                MakeErrorCodeText(ErrorCode::GRAPH_RESOURCE_NOT_FOUND).c_str(), static_cast<uint32_t>(rankId), dieId,
                channelId);
            return HCCL_E_INTERNAL;
        }

        RankId rmtRankId = g_allRankChannelInfo[rankId][dieId][channelId].dstRank;
        uint32_t rmtDieId = g_allRankChannelInfo[rankId][dieId][channelId].remoteDieId;

        uint64_t localXnValue = 0;
        CHK_GET_XN_V3(curCcuTask, queId, locXnId, localXnValue);
        CHK_RET(AllRankParamRecorder::Global()->SetXn(rmtRankId, rmtDieId, rmtXnId, localXnValue));

        if (setRmtCKEMask != 0x0000) {
            AddPost(rankId, rmtRankId, queId, curCcuTask, rmtDieId, setRmtCKEId, setRmtCKEMask);

            // 将对应的bit位置一
            uint16_t ckeValue = 0;
            CHK_RET(AllRankParamRecorder::Global()->GetCKE(rmtRankId, rmtDieId, setRmtCKEId, ckeValue));
            CHK_RET(AllRankParamRecorder::Global()->SetCKE(rmtRankId, rmtDieId, setRmtCKEId, ckeValue | setRmtCKEMask));
        }

        CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask, true));

        CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

        HCCL_VM_DEBUG(
            "Wait CKE[{}:0x{:04x}]:\n"
            "  sync: Sync Xn[{}]({}) To rmtXn[{}] Use Channel[{}]\n"
            "  set: Set rmtCKE[{}:0x{:04x}], Set CKE[{}:0x{:04x}]\n"
            "  ctrl: clearType[{}]",
            waitCKEId, waitCKEMask, locXnId, localXnValue, rmtXnId, channelId, setRmtCKEId, setRmtCKEMask, setCKEId,
            setCKEMask, clearType);
        return HCCL_SUCCESS;
    }

    HcclResult TransformJumpInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
        LoopGroupParam* loopGroupParam)
    {
        CHK_RET(IsLoopGroupParamNull(loopGroupParam));
        uint16_t conditionXnId = instr->v1.jmp.conditionXnId;
        uint64_t expectData = instr->v1.jmp.expectData;
        uint16_t dstInstrXnId = instr->v1.jmp.dstInstrXnId;

        RankId rankId = curCcuTask->GetRankId();
        uint32_t dieId;
        curCcuTask->GetDieId(queId, dieId);

        uint64_t conditionValue = 0;
        CHK_GET_XN_V3(curCcuTask, queId, conditionXnId, conditionValue);
        if (conditionValue != expectData) {
            uint64_t nextInsIdx = 0;
            CHK_GET_XN_V3(curCcuTask, queId, dstInstrXnId, nextInsIdx);
            curCcuTask->microCodePosInQue[queId] = nextInsIdx - curCcuTask->startInstrIdInQue[queId];
        }

        HCCL_VM_DEBUG(
            "Jump if Xn[{}]({}) != immediate[{}], targetXn[{}]", conditionXnId, conditionValue, expectData,
            dstInstrXnId);
        return HCCL_SUCCESS;
    }

    HcclResult TransformTransLocMemToLocMSInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
        LoopGroupParam* loopGroupParam)
    {
        uint16_t locMSId = UpdateMSId(instr->v1.transLocMemToLocMS.locMSId, loopGroupParam);
        uint16_t locGSAId = instr->v1.transLocMemToLocMS.locGSAId;
        uint16_t locXnId = instr->v1.transLocMemToLocMS.locXnId;
        uint16_t lengthXnId = instr->v1.transLocMemToLocMS.lengthXnId;
        uint16_t channelId = instr->v1.transLocMemToLocMS.channelId;
        uint16_t clearType = instr->v1.transLocMemToLocMS.clearType;
        uint16_t lengthEn = instr->v1.transLocMemToLocMS.lengthEn;
        uint16_t setCKEId = UpdateCKEId(instr->v1.transLocMemToLocMS.setCKEId, loopGroupParam);
        uint16_t setCKEMask = instr->v1.transLocMemToLocMS.setCKEMask;
        uint16_t waitCKEId = UpdateCKEId(instr->v1.transLocMemToLocMS.waitCKEId, loopGroupParam);
        uint16_t waitCKEMask = instr->v1.transLocMemToLocMS.waitCKEMask;

        RankId rankId = curCcuTask->GetRankId();
        uint32_t dieId;
        curCcuTask->GetDieId(queId, dieId);

        CHK_RET(ProcessWaitMask(rankId, dieId, curCcuTask, queId, waitCKEId, waitCKEMask, isContinue));
        if (!isContinue) {
            return HCCL_SUCCESS;
        }

        uint64_t locMemAddr = 0x0;
        CHK_GET_GSA_V3(curCcuTask, queId, locGSAId, locMemAddr);
        locMemAddr = UpdateGSAValue(locMemAddr, loopGroupParam);
        uint64_t len = 0;
        CHK_GET_XN_V3(curCcuTask, queId, lengthXnId, len);
        const uint32_t instrId = curCcuTask->microCodePosInQue[queId] + curCcuTask->startInstrIdInQue[queId];
        if (len == 0) {
            HCCL_VM_ERROR(
                "{} Transfer size is 0, instruction=TransLocMemToLocMS:\n"
                "  ctx: rankId={}, dieId={}, queueId={}, instrId={}\n"
                "  size: transferSize={}",
                MakeErrorCodeText(ErrorCode::GRAPH_STRUCTURE_INVALID).c_str(), static_cast<uint32_t>(rankId), dieId,
                queId, instrId, static_cast<unsigned long long>(len));
            return HCCL_E_INTERNAL;
        }

        DataSlice srcSlice;
        DataSlice dstSlice;
        // todo: 后续插件通过读取vm的输出文件，重建memlayout的地址类型信息
        CHK_RET(curCcuTask->GetStorageManager().GetSlice(locMemAddr, len, srcSlice));
        CHK_RET(GenSliceFromMs(locMSId, len, dstSlice));

        AddLocalCopy(rankId, queId, curCcuTask, srcSlice, dstSlice);

        CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask));

        CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

        HCCL_VM_DEBUG(
            "Completed local-memory to local-MS transfer:\n"
            "  wait: waitCkeId={}, waitMask=0x{:04x}\n"
            "  src: srcGsaId={}, srcOffsetXnId={}\n"
            "  dst: dstMsPage={}, dstMsOffset={}\n"
            "  xfer: lengthXnId={}, channelId={}\n"
            "  set: setCkeId={}, setMask=0x{:04x}\n"
            "  ctrl: clearType={}, lengthEnabled={}",
            waitCKEId, waitCKEMask, locGSAId, locXnId, locMSId / 0x8000, locMSId % 0x8000, lengthXnId, channelId,
            setCKEId, setCKEMask, clearType, lengthEn);

        return HCCL_SUCCESS;
    }

    HcclResult TransformTransLocMSToLocMemInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
        LoopGroupParam* loopGroupParam)
    {
        uint16_t locGSAId = instr->v1.transLocMSToLocMem.locGSAId;
        uint16_t locXnId = instr->v1.transLocMSToLocMem.locXnId;
        uint16_t locMSId = UpdateMSId(instr->v1.transLocMSToLocMem.locMSId, loopGroupParam);
        uint16_t lengthXnId = instr->v1.transLocMSToLocMem.lengthXnId;
        uint16_t channelId = instr->v1.transLocMSToLocMem.channelId;
        uint16_t clearType = instr->v1.transLocMSToLocMem.clearType;
        uint16_t lengthEn = instr->v1.transLocMSToLocMem.lengthEn;
        uint16_t setCKEId = UpdateCKEId(instr->v1.transLocMSToLocMem.setCKEId, loopGroupParam);
        uint16_t setCKEMask = instr->v1.transLocMSToLocMem.setCKEMask;
        uint16_t waitCKEId = UpdateCKEId(instr->v1.transLocMSToLocMem.waitCKEId, loopGroupParam);
        uint16_t waitCKEMask = instr->v1.transLocMSToLocMem.waitCKEMask;

        RankId rankId = curCcuTask->GetRankId();
        uint32_t dieId;
        curCcuTask->GetDieId(queId, dieId);

        CHK_RET(ProcessWaitMask(rankId, dieId, curCcuTask, queId, waitCKEId, waitCKEMask, isContinue));
        if (!isContinue) {
            return HCCL_SUCCESS;
        }

        uint64_t len = 0;
        CHK_GET_XN_V3(curCcuTask, queId, lengthXnId, len);
        const uint32_t instrId = curCcuTask->microCodePosInQue[queId] + curCcuTask->startInstrIdInQue[queId];
        if (len == 0) {
            HCCL_VM_ERROR(
                "{} Transfer size is 0, instruction=TransLocMSToLocMem:\n"
                "  ctx: rankId={}, dieId={}, queueId={}, instrId={}\n"
                "  size: transferSize={}",
                MakeErrorCodeText(ErrorCode::GRAPH_STRUCTURE_INVALID).c_str(), static_cast<uint32_t>(rankId), dieId,
                queId, instrId, static_cast<unsigned long long>(len));
            return HCCL_E_INTERNAL;
        }
        DataSlice srcSlice;
        CHK_RET(GenSliceFromMs(locMSId, len, srcSlice));

        uint64_t locMemAddr = 0x0;
        CHK_GET_GSA_V3(curCcuTask, queId, locGSAId, locMemAddr);
        locMemAddr = UpdateGSAValue(locMemAddr, loopGroupParam);
        DataSlice dstSlice;
        CHK_RET(curCcuTask->GetStorageManager().GetSlice(locMemAddr, len, dstSlice));

        AddLocalCopy(rankId, queId, curCcuTask, srcSlice, dstSlice);

        CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask));

        CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

        HCCL_VM_DEBUG(
            "Completed local-MS to local-memory transfer:\n"
            "  wait: waitCkeId={}, waitMask=0x{:04x}\n"
            "  src: srcMsPage={}, srcMsOffset={}\n"
            "  dst: dstGsaId={}, dstOffsetXnId={}\n"
            "  xfer: lengthXnId={}, channelId={}\n"
            "  set: setCkeId={}, setMask=0x{:04x}\n"
            "  ctrl: clearType={}, lengthEnabled={}",
            waitCKEId, waitCKEMask, locMSId / 0x8000, locMSId % 0x8000, locGSAId, locXnId, lengthXnId, channelId,
            setCKEId, setCKEMask, clearType, lengthEn);

        return HCCL_SUCCESS;
    }

    HcclResult TransformTransLocMSToLocMSInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
        LoopGroupParam* loopGroupParam)
    {
        uint16_t dstMSId = UpdateMSId(instr->v1.transLocMSToLocMS.dstMSId, loopGroupParam);
        uint16_t srcMSId = UpdateMSId(instr->v1.transLocMSToLocMS.srcMSId, loopGroupParam);
        uint16_t lengthXnId = instr->v1.transLocMSToLocMS.lengthXnId;
        uint16_t channelId = instr->v1.transLocMSToLocMS.channelId;
        uint16_t clearType = instr->v1.transLocMSToLocMS.clearType;
        uint16_t lengthEn = instr->v1.transLocMSToLocMS.lengthEn;
        uint16_t setCKEId = UpdateCKEId(instr->v1.transLocMSToLocMS.setCKEId, loopGroupParam);
        uint16_t setCKEMask = instr->v1.transLocMSToLocMS.setCKEMask;
        uint16_t waitCKEId = UpdateCKEId(instr->v1.transLocMSToLocMS.waitCKEId, loopGroupParam);
        uint16_t waitCKEMask = instr->v1.transLocMSToLocMS.waitCKEMask;

        RankId rankId = curCcuTask->GetRankId();
        uint32_t dieId;
        curCcuTask->GetDieId(queId, dieId);

        CHK_RET(ProcessWaitMask(rankId, dieId, curCcuTask, queId, waitCKEId, waitCKEMask, isContinue));
        if (!isContinue) {
            return HCCL_SUCCESS;
        }

        uint64_t len = 0;
        CHK_GET_XN_V3(curCcuTask, queId, lengthXnId, len);
        const uint32_t instrId = curCcuTask->microCodePosInQue[queId] + curCcuTask->startInstrIdInQue[queId];
        if (len == 0) {
            HCCL_VM_ERROR(
                "{} Transfer size is 0, instruction=TransLocMSToLocMS:\n"
                "  ctx: rankId={}, dieId={}, queueId={}, instrId={}\n"
                "  size: transferSize={}",
                MakeErrorCodeText(ErrorCode::GRAPH_STRUCTURE_INVALID).c_str(), static_cast<uint32_t>(rankId), dieId,
                queId, instrId, static_cast<unsigned long long>(len));
            return HCCL_E_INTERNAL;
        }
        DataSlice srcSlice;
        CHK_RET(GenSliceFromMs(srcMSId, len, srcSlice));
        DataSlice dstSlice;
        CHK_RET(GenSliceFromMs(dstMSId, len, dstSlice));

        AddLocalCopy(rankId, queId, curCcuTask, srcSlice, dstSlice);

        CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask));

        CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

        HCCL_VM_DEBUG(
            "Completed local-MS to local-MS transfer:\n"
            "  wait: waitCkeId={}, waitMask=0x{:04x}\n"
            "  src: srcMsPage={}, srcMsOffset={}\n"
            "  dst: dstMsPage={}, dstMsOffset={}\n"
            "  xfer: lengthXnId={}, channelId={}\n"
            "  set: setCkeId={}, setMask=0x{:04x}\n"
            "  ctrl: clearType={}, lengthEnabled={}",
            waitCKEId, waitCKEMask, srcMSId / 0x8000, srcMSId % 0x8000, dstMSId / 0x8000, dstMSId % 0x8000, lengthXnId,
            channelId, setCKEId, setCKEMask, clearType, lengthEn);

        return HCCL_SUCCESS;
    }

    HcclResult TransformTransLocMSToRmtMemInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
        LoopGroupParam* loopGroupParam)
    {
        uint16_t rmtGSAId = instr->v1.transLocMSToRmtMem.rmtGSAId;
        uint16_t rmtXnId = instr->v1.transLocMSToRmtMem.rmtXnId;
        uint16_t locMSId = UpdateMSId(instr->v1.transLocMSToRmtMem.locMSId, loopGroupParam);
        uint16_t lengthXnId = instr->v1.transLocMSToRmtMem.lengthXnId;
        uint16_t channelId = instr->v1.transLocMSToRmtMem.channelId;
        uint16_t clearType = instr->v1.transLocMSToRmtMem.clearType;
        uint16_t lengthEn = instr->v1.transLocMSToRmtMem.lengthEn;
        uint16_t setCKEId = UpdateCKEId(instr->v1.transLocMSToRmtMem.setCKEId, loopGroupParam);
        uint16_t setCKEMask = instr->v1.transLocMSToRmtMem.setCKEMask;
        uint16_t waitCKEId = UpdateCKEId(instr->v1.transLocMSToRmtMem.waitCKEId, loopGroupParam);
        uint16_t waitCKEMask = instr->v1.transLocMSToRmtMem.waitCKEMask;

        RankId rankId = curCcuTask->GetRankId();
        uint32_t dieId;
        curCcuTask->GetDieId(queId, dieId);

        CHK_RET(ProcessWaitMask(rankId, dieId, curCcuTask, queId, waitCKEId, waitCKEMask, isContinue));
        if (!isContinue) {
            return HCCL_SUCCESS;
        }

        uint64_t len = 0;
        CHK_GET_XN_V3(curCcuTask, queId, lengthXnId, len);
        const uint32_t instrId = curCcuTask->microCodePosInQue[queId] + curCcuTask->startInstrIdInQue[queId];
        if (len == 0) {
            HCCL_VM_ERROR(
                "{} Transfer size is 0, instruction=TransLocMSToRmtMem:\n"
                "  ctx: rankId={}, dieId={}, queueId={}, instrId={}\n"
                "  size: transferSize={}",
                MakeErrorCodeText(ErrorCode::GRAPH_STRUCTURE_INVALID).c_str(), static_cast<uint32_t>(rankId), dieId,
                queId, instrId, static_cast<unsigned long long>(len));
            return HCCL_E_INTERNAL;
        }
        DataSlice srcSlice;
        CHK_RET(GenSliceFromMs(locMSId, len, srcSlice));

        uint64_t rmtMemAddr = 0x0;
        CHK_GET_GSA_V3(curCcuTask, queId, rmtGSAId, rmtMemAddr);
        rmtMemAddr = UpdateGSAValue(rmtMemAddr, loopGroupParam);
        DataSlice dstSlice;
        // todo: 后续插件通过读取vm的输出文件，重建memlayout的地址类型信息
        CHK_RET(curCcuTask->GetStorageManager().GetSlice(rmtMemAddr, len, dstSlice));

        RankId rmtRankId = g_allRankChannelInfo[rankId][dieId][channelId].dstRank;
        AddWrite(rankId, rmtRankId, queId, curCcuTask, srcSlice, dstSlice);

        CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask));

        CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

        HCCL_VM_DEBUG(
            "Completed local-MS to remote-memory transfer:\n"
            "  wait: waitCkeId={}, waitMask=0x{:04x}\n"
            "  src: srcMsPage={}, srcMsOffset={}\n"
            "  dst: dstGsaId={}, dstOffsetXnId={}\n"
            "  xfer: lengthXnId={}, channelId={}\n"
            "  set: setCkeId={}, setMask=0x{:04x}\n"
            "  ctrl: clearType={}, lengthEnabled={}",
            waitCKEId, waitCKEMask, locMSId / 0x8000, locMSId % 0x8000, rmtGSAId, rmtXnId, lengthXnId, channelId,
            setCKEId, setCKEMask, clearType, lengthEn);

        return HCCL_SUCCESS;
    }

    HcclResult TransformTransLocMSToRmtMSInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
        LoopGroupParam* loopGroupParam)
    {
        uint16_t rmtMSId = UpdateMSId(instr->v1.transLocMSToRmtMS.rmtMSId, loopGroupParam);
        uint16_t locMSId = UpdateMSId(instr->v1.transLocMSToRmtMS.locMSId, loopGroupParam);
        uint16_t lengthXnId = instr->v1.transLocMSToRmtMS.lengthXnId;
        uint16_t channelId = instr->v1.transLocMSToRmtMS.channelId;
        uint16_t setRmtCKEId = UpdateCKEId(instr->v1.transLocMSToRmtMS.setRmtCKEId, loopGroupParam);
        uint16_t setRmtCKEMask = instr->v1.transLocMSToRmtMS.setRmtCKEMask;

        uint16_t clearType = instr->v1.transLocMSToRmtMS.clearType;
        uint16_t lengthEn = instr->v1.transLocMSToRmtMS.lengthEn;
        uint16_t setCKEId = UpdateCKEId(instr->v1.transLocMSToRmtMS.setCKEId, loopGroupParam);
        uint16_t setCKEMask = instr->v1.transLocMSToRmtMS.setCKEMask;
        uint16_t waitCKEId = UpdateCKEId(instr->v1.transLocMSToRmtMS.waitCKEId, loopGroupParam);
        uint16_t waitCKEMask = instr->v1.transLocMSToRmtMS.waitCKEMask;

        RankId rankId = curCcuTask->GetRankId();
        uint32_t dieId;
        curCcuTask->GetDieId(queId, dieId);

        CHK_RET(ProcessWaitMask(rankId, dieId, curCcuTask, queId, waitCKEId, waitCKEMask, isContinue));
        if (!isContinue) {
            return HCCL_SUCCESS;
        }

        uint64_t len = 0;
        CHK_GET_XN_V3(curCcuTask, queId, lengthXnId, len);
        const uint32_t instrId = curCcuTask->microCodePosInQue[queId] + curCcuTask->startInstrIdInQue[queId];
        if (len == 0) {
            HCCL_VM_ERROR(
                "{} Transfer size is 0, instruction=TransLocMSToRmtMS:\n"
                "  ctx: rankId={}, dieId={}, queueId={}, instrId={}\n"
                "  size: transferSize={}",
                MakeErrorCodeText(ErrorCode::GRAPH_STRUCTURE_INVALID).c_str(), static_cast<uint32_t>(rankId), dieId,
                queId, instrId, static_cast<unsigned long long>(len));
            return HCCL_E_INTERNAL;
        }

        DataSlice srcSlice;
        CHK_RET(GenSliceFromMs(locMSId, len, srcSlice));
        DataSlice dstSlice;
        CHK_RET(GenSliceFromMs(rmtMSId, len, dstSlice));

        RankId rmtRankId = g_allRankChannelInfo[rankId][dieId][channelId].dstRank;
        uint32_t rmtDieId = g_allRankChannelInfo[rankId][dieId][channelId].remoteDieId;

        AddWrite(rankId, rmtRankId, queId, curCcuTask, srcSlice, dstSlice);

        if (setRmtCKEMask != 0x0000) {
            AddPost(rankId, rmtRankId, queId, curCcuTask, rmtDieId, setRmtCKEId, setRmtCKEMask);

            // 将对应的bit位置一
            uint16_t ckeValue = 0;
            CHK_RET(AllRankParamRecorder::Global()->GetCKE(rmtRankId, rmtDieId, setRmtCKEId, ckeValue));
            CHK_RET(AllRankParamRecorder::Global()->SetCKE(rmtRankId, rmtDieId, setRmtCKEId, ckeValue | setRmtCKEMask));
        }

        CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask));

        CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

        HCCL_VM_DEBUG(
            "Completed local-MS to remote-MS transfer:\n"
            "  wait: waitCkeId={}, waitMask=0x{:04x}\n"
            "  src: srcMsPage={}, srcMsOffset={}\n"
            "  dst: dstMsPage={}, dstMsOffset={}\n"
            "  xfer: lengthXnId={}, channelId={}\n"
            "  set: remoteSetCkeId={}, remoteSetMask=0x{:04x}, localSetCkeId={}, localSetMask=0x{:04x}\n"
            "  ctrl: clearType={}, lengthEnabled={}",
            waitCKEId, waitCKEMask, locMSId / 0x8000, locMSId % 0x8000, rmtMSId / 0x8000, rmtMSId % 0x8000, lengthXnId,
            channelId, setRmtCKEId, setRmtCKEMask, setCKEId, setCKEMask, clearType, lengthEn);
        return HCCL_SUCCESS;
    }

    HcclResult TransformTransRmtMemToLocMSInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
        LoopGroupParam* loopGroupParam)
    {
        uint16_t locMSId = UpdateMSId(instr->v1.transRmtMemToLocMS.locMSId, loopGroupParam);
        uint16_t rmtGSAId = instr->v1.transRmtMemToLocMS.rmtGSAId;
        uint16_t rmtXnId = instr->v1.transRmtMemToLocMS.rmtXnId;
        uint16_t lengthXnId = instr->v1.transRmtMemToLocMS.lengthXnId;
        uint16_t channelId = instr->v1.transRmtMemToLocMS.channelId;
        uint16_t clearType = instr->v1.transRmtMemToLocMS.clearType;
        uint16_t lengthEn = instr->v1.transRmtMemToLocMS.lengthEn;
        uint16_t setCKEId = UpdateCKEId(instr->v1.transRmtMemToLocMS.setCKEId, loopGroupParam);
        uint16_t setCKEMask = instr->v1.transRmtMemToLocMS.setCKEMask;
        uint16_t waitCKEId = UpdateCKEId(instr->v1.transRmtMemToLocMS.waitCKEId, loopGroupParam);
        uint16_t waitCKEMask = instr->v1.transRmtMemToLocMS.waitCKEMask;

        RankId rankId = curCcuTask->GetRankId();
        uint32_t dieId;
        curCcuTask->GetDieId(queId, dieId);

        CHK_RET(ProcessWaitMask(rankId, dieId, curCcuTask, queId, waitCKEId, waitCKEMask, isContinue));
        if (!isContinue) {
            return HCCL_SUCCESS;
        }

        uint64_t rmtAddr = 0x0;
        CHK_GET_GSA_V3(curCcuTask, queId, rmtGSAId, rmtAddr);
        rmtAddr = UpdateGSAValue(rmtAddr, loopGroupParam);
        uint64_t len = 0;
        CHK_GET_XN_V3(curCcuTask, queId, lengthXnId, len);
        const uint32_t instrId = curCcuTask->microCodePosInQue[queId] + curCcuTask->startInstrIdInQue[queId];
        if (len == 0) {
            HCCL_VM_ERROR(
                "{} Transfer size is 0, instruction=TransRmtMemToLocMS:\n"
                "  ctx: rankId={}, dieId={}, queueId={}, instrId={}\n"
                "  size: transferSize={}",
                MakeErrorCodeText(ErrorCode::GRAPH_STRUCTURE_INVALID).c_str(), static_cast<uint32_t>(rankId), dieId,
                queId, instrId, static_cast<unsigned long long>(len));
            return HCCL_E_INTERNAL;
        }

        DataSlice srcSlice;
        DataSlice dstSlice;
        // todo: 后续插件通过读取vm的输出文件，重建memlayout的地址类型信息
        CHK_RET(curCcuTask->GetStorageManager().GetSlice(rmtAddr, len, srcSlice));
        CHK_RET(GenSliceFromMs(locMSId, len, dstSlice));

        // 获取远端rankId与dieId
        if (g_allRankChannelInfo[rankId][dieId].count(channelId) == 0) {
            HCCL_VM_ERROR(
                "{} Cannot find channel mapping for this instruction, instruction=TransRmtMemToLocMS, "
                "localRankId={}, localDieId={}, channelId={}",
                MakeErrorCodeText(ErrorCode::GRAPH_RESOURCE_NOT_FOUND).c_str(), static_cast<uint32_t>(rankId), dieId,
                channelId);
            return HCCL_E_INTERNAL;
        }

        RankId rmtRankId = g_allRankChannelInfo[rankId][dieId][channelId].dstRank;
        AddRead(rankId, rmtRankId, queId, curCcuTask, srcSlice, dstSlice);

        CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask));

        CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

        HCCL_VM_DEBUG(
            "Completed remote-memory to local-MS transfer:\n"
            "  wait: waitCkeId={}, waitMask=0x{:04x}\n"
            "  src: srcGsaId={}, srcOffsetXnId={}\n"
            "  dst: dstMsPage={}, dstMsOffset={}\n"
            "  xfer: lengthXnId={}, channelId={}\n"
            "  set: setCkeId={}, setMask=0x{:04x}\n"
            "  ctrl: clearType={}, lengthEnabled={}",
            waitCKEId, waitCKEMask, rmtGSAId, rmtXnId, locMSId / 0x8000, locMSId % 0x8000, lengthXnId, channelId,
            setCKEId, setCKEMask, clearType, lengthEn);
        return HCCL_SUCCESS;
    }

    HcclResult TransformTransRmtMSToLocMemInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
        LoopGroupParam* loopGroupParam)
    {
        uint16_t locGSAId = instr->v1.transRmtMSToLocMem.locGSAId;
        uint16_t locXnId = instr->v1.transRmtMSToLocMem.locXnId;
        uint16_t rmtMSId = UpdateMSId(instr->v1.transRmtMSToLocMem.rmtMSId, loopGroupParam);
        uint16_t lengthXnId = instr->v1.transRmtMSToLocMem.lengthXnId;
        uint16_t channelId = instr->v1.transRmtMSToLocMem.channelId;
        uint16_t clearType = instr->v1.transRmtMSToLocMem.clearType;
        uint16_t lengthEn = instr->v1.transRmtMSToLocMem.lengthEn;
        uint16_t setCKEId = UpdateCKEId(instr->v1.transRmtMSToLocMem.setCKEId, loopGroupParam);
        uint16_t setCKEMask = instr->v1.transRmtMSToLocMem.setCKEMask;
        uint16_t waitCKEId = UpdateCKEId(instr->v1.transRmtMSToLocMem.waitCKEId, loopGroupParam);
        uint16_t waitCKEMask = instr->v1.transRmtMSToLocMem.waitCKEMask;

        RankId rankId = curCcuTask->GetRankId();
        uint32_t dieId;
        curCcuTask->GetDieId(queId, dieId);

        CHK_RET(ProcessWaitMask(rankId, dieId, curCcuTask, queId, waitCKEId, waitCKEMask, isContinue));
        if (!isContinue) {
            return HCCL_SUCCESS;
        }

        uint64_t len = 0;
        CHK_GET_XN_V3(curCcuTask, queId, lengthXnId, len);
        const uint32_t instrId = curCcuTask->microCodePosInQue[queId] + curCcuTask->startInstrIdInQue[queId];
        if (len == 0) {
            HCCL_VM_ERROR(
                "{} Transfer size is 0, instruction=TransRmtMSToLocMem:\n"
                "  ctx: rankId={}, dieId={}, queueId={}, instrId={}\n"
                "  size: transferSize={}",
                MakeErrorCodeText(ErrorCode::GRAPH_STRUCTURE_INVALID).c_str(), static_cast<uint32_t>(rankId), dieId,
                queId, instrId, static_cast<unsigned long long>(len));
            return HCCL_E_INTERNAL;
        }

        DataSlice srcSlice;
        CHK_RET(GenSliceFromMs(rmtMSId, len, srcSlice));

        uint64_t localAddr = 0x0;
        CHK_GET_GSA_V3(curCcuTask, queId, locGSAId, localAddr);
        localAddr = UpdateGSAValue(localAddr, loopGroupParam);
        DataSlice dstSlice;
        CHK_RET(curCcuTask->GetStorageManager().GetSlice(localAddr, len, dstSlice));

        RankId rmtRankId = g_allRankChannelInfo[rankId][dieId][channelId].dstRank;
        AddRead(rankId, rmtRankId, queId, curCcuTask, srcSlice, dstSlice);

        CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask));

        CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

        HCCL_VM_DEBUG(
            "Completed remote-MS to local-memory transfer:\n"
            "  wait: waitCkeId={}, waitMask=0x{:04x}\n"
            "  src: srcMsPage={}, srcMsOffset={}\n"
            "  dst: dstGsaId={}, dstOffsetXnId={}\n"
            "  xfer: lengthXnId={}, channelId={}\n"
            "  set: setCkeId={}, setMask=0x{:04x}\n"
            "  ctrl: clearType={}, lengthEnabled={}",
            waitCKEId, waitCKEMask, rmtMSId / 0x8000, rmtMSId % 0x8000, locGSAId, locXnId, lengthXnId, channelId,
            setCKEId, setCKEMask, clearType, lengthEn);
        return HCCL_SUCCESS;
    }

    HcclResult TransformTransRmtMSToLocMSInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
        LoopGroupParam* loopGroupParam)
    {
        uint16_t locMSId = UpdateMSId(instr->v1.transRmtMSToLocMS.locMSId, loopGroupParam);
        uint16_t rmtMSId = UpdateMSId(instr->v1.transRmtMSToLocMS.rmtMSId, loopGroupParam);
        uint16_t lengthXnId = instr->v1.transRmtMSToLocMS.lengthXnId;
        uint16_t channelId = instr->v1.transRmtMSToLocMS.channelId;
        uint16_t clearType = instr->v1.transRmtMSToLocMS.clearType;
        uint16_t lengthEn = instr->v1.transRmtMSToLocMS.lengthEn;
        uint16_t setCKEId = UpdateCKEId(instr->v1.transRmtMSToLocMS.setCKEId, loopGroupParam);
        uint16_t setCKEMask = instr->v1.transRmtMSToLocMS.setCKEMask;
        uint16_t waitCKEId = UpdateCKEId(instr->v1.transRmtMSToLocMS.waitCKEId, loopGroupParam);
        uint16_t waitCKEMask = instr->v1.transRmtMSToLocMS.waitCKEMask;

        RankId rankId = curCcuTask->GetRankId();
        uint32_t dieId;
        curCcuTask->GetDieId(queId, dieId);

        CHK_RET(ProcessWaitMask(rankId, dieId, curCcuTask, queId, waitCKEId, waitCKEMask, isContinue));
        if (!isContinue) {
            return HCCL_SUCCESS;
        }

        uint64_t len = 0;
        CHK_GET_XN_V3(curCcuTask, queId, lengthXnId, len);
        const uint32_t instrId = curCcuTask->microCodePosInQue[queId] + curCcuTask->startInstrIdInQue[queId];
        if (len == 0) {
            HCCL_VM_ERROR(
                "{} Transfer size is 0, instruction=TransRmtMSToLocMS:\n"
                "  ctx: rankId={}, dieId={}, queueId={}, instrId={}\n"
                "  size: transferSize={}",
                MakeErrorCodeText(ErrorCode::GRAPH_STRUCTURE_INVALID).c_str(), static_cast<uint32_t>(rankId), dieId,
                queId, instrId, static_cast<unsigned long long>(len));
            return HCCL_E_INTERNAL;
        }

        DataSlice srcSlice;
        CHK_RET(GenSliceFromMs(rmtMSId, len, srcSlice));
        DataSlice dstSlice;
        CHK_RET(GenSliceFromMs(locMSId, len, dstSlice));

        RankId rmtRankId = g_allRankChannelInfo[rankId][dieId][channelId].dstRank;
        AddRead(rankId, rmtRankId, queId, curCcuTask, srcSlice, dstSlice);

        CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask));

        CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

        HCCL_VM_DEBUG(
            "Completed remote-MS to local-MS transfer:\n"
            "  wait: waitCkeId={}, waitMask=0x{:04x}\n"
            "  src: srcMsPage={}, srcMsOffset={}\n"
            "  dst: dstMsPage={}, dstMsOffset={}\n"
            "  xfer: lengthXnId={}, channelId={}\n"
            "  set: setCkeId={}, setMask=0x{:04x}\n"
            "  ctrl: clearType={}, lengthEnabled={}",
            waitCKEId, waitCKEMask, rmtMSId / 0x8000, rmtMSId % 0x8000, locMSId / 0x8000, locMSId % 0x8000, lengthXnId,
            channelId, setCKEId, setCKEMask, clearType, lengthEn);
        return HCCL_SUCCESS;
    }

    // todo：这边待处理reduce信息
    HcclResult TransformTransRmtMemToLocMemInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
        LoopGroupParam* loopGroupParam)
    {
        CHK_RET(IsLoopGroupParamNull(loopGroupParam));
        uint16_t locGSAId = instr->v1.transRmtMemToLocMem.locGSAId;
        uint16_t locXnId = instr->v1.transRmtMemToLocMem.locXnId;
        uint16_t rmtGSAId = instr->v1.transRmtMemToLocMem.rmtGSAId;
        uint16_t rmtXnId = instr->v1.transRmtMemToLocMem.rmtXnId;
        uint16_t lengthXnId = instr->v1.transRmtMemToLocMem.lengthXnId;
        uint16_t channelId = instr->v1.transRmtMemToLocMem.channelId;
        uint16_t reduceDataType = instr->v1.transRmtMemToLocMem.reduceDataType;
        uint16_t reduceOpCode = instr->v1.transRmtMemToLocMem.reduceOpCode;
        uint16_t clearType = instr->v1.transRmtMemToLocMem.clearType;
        uint16_t lengthEn = instr->v1.transRmtMemToLocMem.lengthEn;
        uint16_t reduceEn = instr->v1.transRmtMemToLocMem.reduceEn;
        uint16_t setCKEId = instr->v1.transRmtMemToLocMem.setCKEId;
        uint16_t setCKEMask = instr->v1.transRmtMemToLocMem.setCKEMask;
        uint16_t waitCKEId = instr->v1.transRmtMemToLocMem.waitCKEId;
        uint16_t waitCKEMask = instr->v1.transRmtMemToLocMem.waitCKEMask;

        RankId rankId = curCcuTask->GetRankId();
        uint32_t dieId;
        curCcuTask->GetDieId(queId, dieId);

        CHK_RET(ProcessWaitMask(rankId, dieId, curCcuTask, queId, waitCKEId, waitCKEMask, isContinue));
        if (!isContinue) {
            return HCCL_SUCCESS;
        }

        uint64_t len = 0;
        CHK_GET_XN_V3(curCcuTask, queId, lengthXnId, len);
        const uint32_t instrId = curCcuTask->microCodePosInQue[queId] + curCcuTask->startInstrIdInQue[queId];
        if (len == 0) {
            HCCL_VM_ERROR(
                "{} Transfer size is 0, instruction=TransRmtMemToLocMem:\n"
                "  ctx: rankId={}, dieId={}, queueId={}, instrId={}\n"
                "  size: transferSize={}",
                MakeErrorCodeText(ErrorCode::GRAPH_STRUCTURE_INVALID).c_str(), static_cast<uint32_t>(rankId), dieId,
                queId, instrId, static_cast<unsigned long long>(len));
            return HCCL_E_INTERNAL;
        }
        if (len > UB_MAX_SIZE) {
            HCCL_VM_ERROR(
                "{} Transfer size exceeds the UB upper limit, instruction=TransRmtMemToLocMem:\n"
                "  size: transferSize={}, ubLimit={}\n"
                "  ctx: rankId={}, dieId={}, queueId={}, instrId={}",
                MakeErrorCodeText(ErrorCode::GRAPH_STRUCTURE_INVALID).c_str(), static_cast<unsigned long long>(len),
                static_cast<unsigned long long>(UB_MAX_SIZE), static_cast<uint32_t>(rankId), dieId, queId, instrId);
            return HCCL_E_INTERNAL;
        }

        uint64_t localAddr = 0x0;
        uint64_t rmtAddr = 0x0;
        CHK_GET_GSA_V3(curCcuTask, queId, locGSAId, localAddr);
        CHK_GET_GSA_V3(curCcuTask, queId, rmtGSAId, rmtAddr);

        DataSlice srcSlice;
        RankId rId;
        CHK_RET(curCcuTask->GetStorageManager().GetSlice(rmtAddr, len, srcSlice, &rId));
        DataSlice dstSlice;
        CHK_RET(curCcuTask->GetStorageManager().GetSlice(localAddr, len, dstSlice));

        RankId rmtRankId = g_allRankChannelInfo[rankId][dieId][channelId].dstRank;
        HCCL_VM_DEBUG(
            "Preparing remote-memory to local-memory transfer, localRankId={}, remoteRankId={}", rankId, rmtRankId);
        // 说明transportId写错了
        if (rId != rmtRankId) {
            HCCL_VM_ERROR(
                "{} Remote address resolves to a different rank than the selected channel, "
                "instruction=TransRmtMemToLocMem:\n"
                "  ctx: rankId={}, dieId={}, queueId={}, instrId={}, channelId={}\n"
                "  rank: expectedRemoteRankId={}, actualRemoteRankId={}\n"
                "  addr: remoteAddr={}",
                MakeErrorCodeText(ErrorCode::GRAPH_REMOTE_RANK_MISMATCH).c_str(), static_cast<uint32_t>(rankId), dieId,
                queId, instrId, channelId, static_cast<uint32_t>(rmtRankId), static_cast<uint32_t>(rId),
                static_cast<unsigned long long>(rmtAddr));
            return HCCL_E_INTERNAL;
        }

        HcclReduceOp checkerReduceOp;
        HcclDataType checkerDataType;
        if (reduceEn) {
            checkerReduceOp = g_ReduceOp2CheckerReduceOp_ccu[reduceOpCode];
            DataType hcclDataType;
            CHK_RET(GetHcclDataTypeFromCCUDataType(reduceDataType, ccuReduceTypeMap[reduceOpCode], hcclDataType));
            checkerDataType = g_DataType2CheckerDataType_aicpu[hcclDataType];
        }

        if (rmtRankId != rankId) {
            if (reduceEn) {
                AddReadReduce(
                    rankId, rmtRankId, queId, curCcuTask, srcSlice, dstSlice, checkerDataType, checkerReduceOp);
            } else {
                AddRead(rankId, rmtRankId, queId, curCcuTask, srcSlice, dstSlice);
            }
        } else {
            if (reduceEn) {
                AddLocalReduce(rankId, queId, curCcuTask, srcSlice, dstSlice, checkerDataType, checkerReduceOp);
            } else {
                AddLocalCopy(rankId, queId, curCcuTask, srcSlice, dstSlice);
            }
        }

        CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask));
        CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

        HCCL_VM_DEBUG(
            "Completed remote-memory to local-memory transfer:\n"
            "  wait: waitCkeId={}, waitMask=0x{:04x}\n"
            "  src: srcGsaId={}, srcOffsetXnId={}\n"
            "  dst: dstGsaId={}, dstOffsetXnId={}\n"
            "  xfer: lengthXnId={}, channelId={}\n"
            "  set: setCkeId={}, setMask=0x{:04x}\n"
            "  ctrl: clearType={}, lengthEnabled={}, reduceEnabled={}, reduceDataType={}, reduceOpCode={}",
            waitCKEId, waitCKEMask, rmtGSAId, rmtXnId, locGSAId, locXnId, lengthXnId, channelId, setCKEId, setCKEMask,
            clearType, lengthEn, reduceEn, reduceDataType, reduceOpCode);

        return HCCL_SUCCESS;
    }

    static bool ConsumeProducedMask(std::map<uint16_t, uint16_t>& producedMasks, uint16_t ckeId, uint16_t waitMask)
    {
        if (waitMask == 0) {
            return true;
        }
        const uint16_t producedMask = producedMasks[ckeId];
        if ((producedMask & waitMask) != waitMask) {
            return false;
        }
        producedMasks[ckeId] = static_cast<uint16_t>(producedMask & ~waitMask);
        return true;
    }

    static void ProduceMask(std::map<uint16_t, uint16_t>& producedMasks, uint16_t ckeId, uint16_t setMask)
    {
        if (setMask == 0) {
            return;
        }
        producedMasks[ckeId] = static_cast<uint16_t>(producedMasks[ckeId] | setMask);
    }

    static HcclResult ValidateLoopLen(uint64_t len)
    {
        if (len == 0) {
            HCCL_VM_ERROR(
                "{} Loop-merged transfer size is 0, instruction=LoopMergedTransfer, "
                "transferSize={}",
                MakeErrorCodeText(ErrorCode::GRAPH_STRUCTURE_INVALID).c_str(), static_cast<unsigned long long>(len));
            return HCCL_E_INTERNAL;
        }
        return HCCL_SUCCESS;
    }

    static HcclResult
    GetRemoteRankByChannel(RankId rankId, uint32_t dieId, uint16_t channelId, RankId& rmtRankId, uint32_t& rmtDieId)
    {
        if (g_allRankChannelInfo[rankId][dieId].count(channelId) == 0) {
            HCCL_VM_ERROR(
                "{} Cannot find channel mapping for this instruction, instruction=LoopMergedTransfer, "
                "localRankId={}, localDieId={}, channelId={}",
                MakeErrorCodeText(ErrorCode::GRAPH_RESOURCE_NOT_FOUND).c_str(), static_cast<uint32_t>(rankId), dieId,
                channelId);
            return HCCL_E_INTERNAL;
        }
        rmtRankId = g_allRankChannelInfo[rankId][dieId][channelId].dstRank;
        rmtDieId = g_allRankChannelInfo[rankId][dieId][channelId].remoteDieId;
        return HCCL_SUCCESS;
    }

    static void SetLoopParamAtIteration(const LoopGroupParam& sampleParam, u32 curLoopCnt, LoopGroupParam& iterParam)
    {
        iterParam = sampleParam;
        iterParam.curLoopCnt = curLoopCnt;
    }

    static HcclResult CollectLoopCkeOpsA5(
        const CcuRep::CcuInstr* instr, const LoopGroupParam& sampleParam, CcuLoopCkeOpV3& waitOp, CcuLoopCkeOpV3& setOp)
    {
        switch (instr->header.type) {
            case CTRL_TYPE:
                if (instr->header.code == CLEARCKE_CODE) {
                    waitOp
                        = {UpdateCKEId(instr->v1.clearCKE.waitCKEId, const_cast<LoopGroupParam*>(&sampleParam)),
                           instr->v1.clearCKE.waitCKEMask};
                    return HCCL_SUCCESS;
                }
                return HCCL_E_NOT_SUPPORT;
            case TRANS_TYPE:
                switch (instr->header.code) {
                    case TRANSLOCMEMTOLOCMS_CODE:
                        waitOp
                            = {UpdateCKEId(
                                   instr->v1.transLocMemToLocMS.waitCKEId, const_cast<LoopGroupParam*>(&sampleParam)),
                               instr->v1.transLocMemToLocMS.waitCKEMask};
                        setOp
                            = {UpdateCKEId(
                                   instr->v1.transLocMemToLocMS.setCKEId, const_cast<LoopGroupParam*>(&sampleParam)),
                               instr->v1.transLocMemToLocMS.setCKEMask};
                        return HCCL_SUCCESS;
                    case TRANSRMTMEMTOLOCMS_CODE:
                        waitOp
                            = {UpdateCKEId(
                                   instr->v1.transRmtMemToLocMS.waitCKEId, const_cast<LoopGroupParam*>(&sampleParam)),
                               instr->v1.transRmtMemToLocMS.waitCKEMask};
                        setOp
                            = {UpdateCKEId(
                                   instr->v1.transRmtMemToLocMS.setCKEId, const_cast<LoopGroupParam*>(&sampleParam)),
                               instr->v1.transRmtMemToLocMS.setCKEMask};
                        return HCCL_SUCCESS;
                    case TRANSLOCMSTOLOCMEM_CODE:
                        waitOp
                            = {UpdateCKEId(
                                   instr->v1.transLocMSToLocMem.waitCKEId, const_cast<LoopGroupParam*>(&sampleParam)),
                               instr->v1.transLocMSToLocMem.waitCKEMask};
                        setOp
                            = {UpdateCKEId(
                                   instr->v1.transLocMSToLocMem.setCKEId, const_cast<LoopGroupParam*>(&sampleParam)),
                               instr->v1.transLocMSToLocMem.setCKEMask};
                        return HCCL_SUCCESS;
                    case TRANSLOCMSTORMTMEM_CODE:
                        waitOp
                            = {UpdateCKEId(
                                   instr->v1.transLocMSToRmtMem.waitCKEId, const_cast<LoopGroupParam*>(&sampleParam)),
                               instr->v1.transLocMSToRmtMem.waitCKEMask};
                        setOp
                            = {UpdateCKEId(
                                   instr->v1.transLocMSToRmtMem.setCKEId, const_cast<LoopGroupParam*>(&sampleParam)),
                               instr->v1.transLocMSToRmtMem.setCKEMask};
                        return HCCL_SUCCESS;
                    case TRANSRMTMSTOLOCMEM_CODE:
                        waitOp
                            = {UpdateCKEId(
                                   instr->v1.transRmtMSToLocMem.waitCKEId, const_cast<LoopGroupParam*>(&sampleParam)),
                               instr->v1.transRmtMSToLocMem.waitCKEMask};
                        setOp
                            = {UpdateCKEId(
                                   instr->v1.transRmtMSToLocMem.setCKEId, const_cast<LoopGroupParam*>(&sampleParam)),
                               instr->v1.transRmtMSToLocMem.setCKEMask};
                        return HCCL_SUCCESS;
                    case TRANSLOCMSTOLOCMS_CODE:
                        waitOp
                            = {UpdateCKEId(
                                   instr->v1.transLocMSToLocMS.waitCKEId, const_cast<LoopGroupParam*>(&sampleParam)),
                               instr->v1.transLocMSToLocMS.waitCKEMask};
                        setOp
                            = {UpdateCKEId(
                                   instr->v1.transLocMSToLocMS.setCKEId, const_cast<LoopGroupParam*>(&sampleParam)),
                               instr->v1.transLocMSToLocMS.setCKEMask};
                        return HCCL_SUCCESS;
                    case TRANSRMTMSTOLOCMS_CODE:
                        waitOp
                            = {UpdateCKEId(
                                   instr->v1.transRmtMSToLocMS.waitCKEId, const_cast<LoopGroupParam*>(&sampleParam)),
                               instr->v1.transRmtMSToLocMS.waitCKEMask};
                        setOp
                            = {UpdateCKEId(
                                   instr->v1.transRmtMSToLocMS.setCKEId, const_cast<LoopGroupParam*>(&sampleParam)),
                               instr->v1.transRmtMSToLocMS.setCKEMask};
                        return HCCL_SUCCESS;
                    case TRANSLOCMSTORMTMS_CODE:
                        if (instr->v1.transLocMSToRmtMS.setRmtCKEMask != 0) {
                            return HCCL_E_NOT_SUPPORT;
                        }
                        waitOp
                            = {UpdateCKEId(
                                   instr->v1.transLocMSToRmtMS.waitCKEId, const_cast<LoopGroupParam*>(&sampleParam)),
                               instr->v1.transLocMSToRmtMS.waitCKEMask};
                        setOp
                            = {UpdateCKEId(
                                   instr->v1.transLocMSToRmtMS.setCKEId, const_cast<LoopGroupParam*>(&sampleParam)),
                               instr->v1.transLocMSToRmtMS.setCKEMask};
                        return HCCL_SUCCESS;
                    default:
                        return HCCL_E_NOT_SUPPORT;
                }
            case REDUCE_TYPE:
                if (instr->header.code == ADD_CODE) {
                    waitOp
                        = {UpdateCKEId(instr->v1.add.waitCKEId, const_cast<LoopGroupParam*>(&sampleParam)),
                           instr->v1.add.waitCKEMask};
                    setOp
                        = {UpdateCKEId(instr->v1.add.setCKEId, const_cast<LoopGroupParam*>(&sampleParam)),
                           instr->v1.add.setCKEMask};
                    return HCCL_SUCCESS;
                }
                if (instr->header.code == MAX_CODE) {
                    waitOp
                        = {UpdateCKEId(instr->v1.max.waitCKEId, const_cast<LoopGroupParam*>(&sampleParam)),
                           instr->v1.max.waitCKEMask};
                    setOp
                        = {UpdateCKEId(instr->v1.max.setCKEId, const_cast<LoopGroupParam*>(&sampleParam)),
                           instr->v1.max.setCKEMask};
                    return HCCL_SUCCESS;
                }
                if (instr->header.code == MIN_CODE) {
                    waitOp
                        = {UpdateCKEId(instr->v1.min.waitCKEId, const_cast<LoopGroupParam*>(&sampleParam)),
                           instr->v1.min.waitCKEMask};
                    setOp
                        = {UpdateCKEId(instr->v1.min.setCKEId, const_cast<LoopGroupParam*>(&sampleParam)),
                           instr->v1.min.setCKEMask};
                    return HCCL_SUCCESS;
                }
                return HCCL_E_NOT_SUPPORT;
            default:
                return HCCL_E_NOT_SUPPORT;
        }
    }

    static HcclResult CollectTransLoopInstrA5(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, uint16_t instrId,
        const LoopGroupParam& sampleParam, uint64_t loopCnt, std::shared_ptr<CcuLoopInstrV3>& instrInLoop)
    {
        RankId rankId = curCcuTask->GetRankId();
        uint32_t dieId = INVALID_DIE_ID;
        curCcuTask->GetDieId(queId, dieId);
        auto transInstr = EnsureCcuLoopInstr<CcuLoopTransMemV3>(instrInLoop, rankId, dieId, instrId);
        if (transInstr == nullptr) {
            return HCCL_E_MEMORY;
        }
        CcuLoopCkeOpV3 waitOp;
        CcuLoopCkeOpV3 setOp;
        CHK_RET(CollectLoopCkeOpsA5(instr, sampleParam, waitOp, setOp));
        transInstr->AddWait(waitOp.ckeId, waitOp.mask);
        transInstr->AddSet(setOp.ckeId, setOp.mask);

        for (uint64_t iter = 0; iter < loopCnt; ++iter) {
            LoopGroupParam iterParam;
            SetLoopParamAtIteration(sampleParam, static_cast<u32>(iter), iterParam);
            uint64_t len = 0;
            DataSlice srcSlice;
            DataSlice dstSlice;
            RankId rmtRankId = INVALID_RANK_ID;
            uint32_t rmtDieId = INVALID_DIE_ID;
            switch (instr->header.code) {
                case TRANSLOCMEMTOLOCMS_CODE: {
                    const auto& op = instr->v1.transLocMemToLocMS;
                    uint64_t locMemAddr = 0;
                    CHK_GET_GSA_V3(curCcuTask, queId, op.locGSAId, locMemAddr);
                    CHK_GET_XN_V3(curCcuTask, queId, op.lengthXnId, len);
                    CHK_RET(ValidateLoopLen(len));
                    locMemAddr = UpdateGSAValue(locMemAddr, &iterParam);
                    CHK_RET(curCcuTask->GetStorageManager().GetSlice(locMemAddr, len, srcSlice));
                    const uint16_t locMSId = UpdateMSId(op.locMSId, &iterParam);
                    CHK_RET(GenSliceFromMs(locMSId, len, dstSlice));
                    transInstr->srcs.push_back(MakeCcuMemSlice(rankId, srcSlice));
                    transInstr->dsts.push_back(MakeCcuMemSlice(rankId, dstSlice));
                    transInstr->msIds.insert(locMSId);
                    break;
                }
                case TRANSRMTMEMTOLOCMS_CODE: {
                    const auto& op = instr->v1.transRmtMemToLocMS;
                    CHK_RET(GetRemoteRankByChannel(rankId, dieId, op.channelId, rmtRankId, rmtDieId));
                    uint64_t rmtAddr = 0;
                    CHK_GET_GSA_V3(curCcuTask, queId, op.rmtGSAId, rmtAddr);
                    CHK_GET_XN_V3(curCcuTask, queId, op.lengthXnId, len);
                    CHK_RET(ValidateLoopLen(len));
                    rmtAddr = UpdateGSAValue(rmtAddr, &iterParam);
                    CHK_RET(curCcuTask->GetStorageManager().GetSlice(rmtAddr, len, srcSlice));
                    const uint16_t locMSId = UpdateMSId(op.locMSId, &iterParam);
                    CHK_RET(GenSliceFromMs(locMSId, len, dstSlice));
                    transInstr->srcs.push_back(MakeCcuMemSlice(rmtRankId, srcSlice));
                    transInstr->dsts.push_back(MakeCcuMemSlice(rankId, dstSlice));
                    transInstr->msIds.insert(locMSId);
                    break;
                }
                case TRANSLOCMSTOLOCMEM_CODE: {
                    const auto& op = instr->v1.transLocMSToLocMem;
                    CHK_GET_XN_V3(curCcuTask, queId, op.lengthXnId, len);
                    CHK_RET(ValidateLoopLen(len));
                    const uint16_t locMSId = UpdateMSId(op.locMSId, &iterParam);
                    CHK_RET(GenSliceFromMs(locMSId, len, srcSlice));
                    uint64_t locMemAddr = 0;
                    CHK_GET_GSA_V3(curCcuTask, queId, op.locGSAId, locMemAddr);
                    locMemAddr = UpdateGSAValue(locMemAddr, &iterParam);
                    CHK_RET(curCcuTask->GetStorageManager().GetSlice(locMemAddr, len, dstSlice));
                    transInstr->srcs.push_back(MakeCcuMemSlice(rankId, srcSlice));
                    transInstr->dsts.push_back(MakeCcuMemSlice(rankId, dstSlice));
                    transInstr->msIds.insert(locMSId);
                    break;
                }
                case TRANSLOCMSTORMTMEM_CODE: {
                    const auto& op = instr->v1.transLocMSToRmtMem;
                    CHK_RET(GetRemoteRankByChannel(rankId, dieId, op.channelId, rmtRankId, rmtDieId));
                    CHK_GET_XN_V3(curCcuTask, queId, op.lengthXnId, len);
                    CHK_RET(ValidateLoopLen(len));
                    const uint16_t locMSId = UpdateMSId(op.locMSId, &iterParam);
                    CHK_RET(GenSliceFromMs(locMSId, len, srcSlice));
                    uint64_t rmtMemAddr = 0;
                    CHK_GET_GSA_V3(curCcuTask, queId, op.rmtGSAId, rmtMemAddr);
                    rmtMemAddr = UpdateGSAValue(rmtMemAddr, &iterParam);
                    CHK_RET(curCcuTask->GetStorageManager().GetSlice(rmtMemAddr, len, dstSlice));
                    transInstr->srcs.push_back(MakeCcuMemSlice(rankId, srcSlice));
                    transInstr->dsts.push_back(MakeCcuMemSlice(rmtRankId, dstSlice));
                    transInstr->msIds.insert(locMSId);
                    break;
                }
                case TRANSRMTMSTOLOCMEM_CODE: {
                    const auto& op = instr->v1.transRmtMSToLocMem;
                    CHK_RET(GetRemoteRankByChannel(rankId, dieId, op.channelId, rmtRankId, rmtDieId));
                    CHK_GET_XN_V3(curCcuTask, queId, op.lengthXnId, len);
                    CHK_RET(ValidateLoopLen(len));
                    const uint16_t rmtMSId = UpdateMSId(op.rmtMSId, &iterParam);
                    CHK_RET(GenSliceFromMs(rmtMSId, len, srcSlice));
                    uint64_t localAddr = 0;
                    CHK_GET_GSA_V3(curCcuTask, queId, op.locGSAId, localAddr);
                    localAddr = UpdateGSAValue(localAddr, &iterParam);
                    CHK_RET(curCcuTask->GetStorageManager().GetSlice(localAddr, len, dstSlice));
                    transInstr->srcs.push_back(MakeCcuMemSlice(rmtRankId, srcSlice));
                    transInstr->dsts.push_back(MakeCcuMemSlice(rankId, dstSlice));
                    transInstr->msIds.insert(rmtMSId);
                    break;
                }
                case TRANSLOCMSTOLOCMS_CODE: {
                    const auto& op = instr->v1.transLocMSToLocMS;
                    CHK_GET_XN_V3(curCcuTask, queId, op.lengthXnId, len);
                    CHK_RET(ValidateLoopLen(len));
                    const uint16_t srcMSId = UpdateMSId(op.srcMSId, &iterParam);
                    const uint16_t dstMSId = UpdateMSId(op.dstMSId, &iterParam);
                    CHK_RET(GenSliceFromMs(srcMSId, len, srcSlice));
                    CHK_RET(GenSliceFromMs(dstMSId, len, dstSlice));
                    transInstr->srcs.push_back(MakeCcuMemSlice(rankId, srcSlice));
                    transInstr->dsts.push_back(MakeCcuMemSlice(rankId, dstSlice));
                    transInstr->msIds.insert(srcMSId);
                    transInstr->msIds.insert(dstMSId);
                    break;
                }
                case TRANSRMTMSTOLOCMS_CODE: {
                    const auto& op = instr->v1.transRmtMSToLocMS;
                    CHK_RET(GetRemoteRankByChannel(rankId, dieId, op.channelId, rmtRankId, rmtDieId));
                    CHK_GET_XN_V3(curCcuTask, queId, op.lengthXnId, len);
                    CHK_RET(ValidateLoopLen(len));
                    const uint16_t rmtMSId = UpdateMSId(op.rmtMSId, &iterParam);
                    const uint16_t locMSId = UpdateMSId(op.locMSId, &iterParam);
                    CHK_RET(GenSliceFromMs(rmtMSId, len, srcSlice));
                    CHK_RET(GenSliceFromMs(locMSId, len, dstSlice));
                    transInstr->srcs.push_back(MakeCcuMemSlice(rmtRankId, srcSlice));
                    transInstr->dsts.push_back(MakeCcuMemSlice(rankId, dstSlice));
                    transInstr->msIds.insert(rmtMSId);
                    transInstr->msIds.insert(locMSId);
                    break;
                }
                case TRANSLOCMSTORMTMS_CODE: {
                    const auto& op = instr->v1.transLocMSToRmtMS;
                    if (op.setRmtCKEMask != 0) {
                        return HCCL_E_NOT_SUPPORT;
                    }
                    CHK_RET(GetRemoteRankByChannel(rankId, dieId, op.channelId, rmtRankId, rmtDieId));
                    CHK_GET_XN_V3(curCcuTask, queId, op.lengthXnId, len);
                    CHK_RET(ValidateLoopLen(len));
                    const uint16_t locMSId = UpdateMSId(op.locMSId, &iterParam);
                    const uint16_t rmtMSId = UpdateMSId(op.rmtMSId, &iterParam);
                    CHK_RET(GenSliceFromMs(locMSId, len, srcSlice));
                    CHK_RET(GenSliceFromMs(rmtMSId, len, dstSlice));
                    transInstr->srcs.push_back(MakeCcuMemSlice(rankId, srcSlice));
                    transInstr->dsts.push_back(MakeCcuMemSlice(rmtRankId, dstSlice));
                    transInstr->msIds.insert(locMSId);
                    transInstr->msIds.insert(rmtMSId);
                    break;
                }
                default:
                    return HCCL_E_NOT_SUPPORT;
            }
        }
        return HCCL_SUCCESS;
    }

    static HcclResult CollectReduceLoopInstrA5(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, uint16_t instrId,
        const LoopGroupParam& sampleParam, uint64_t loopCnt, std::shared_ptr<CcuLoopInstrV3>& instrInLoop)
    {
        RankId rankId = curCcuTask->GetRankId();
        uint32_t dieId = INVALID_DIE_ID;
        curCcuTask->GetDieId(queId, dieId);
        auto reduceInstr = EnsureCcuLoopInstr<CcuLoopReduceV3>(instrInLoop, rankId, dieId, instrId);
        if (reduceInstr == nullptr) {
            return HCCL_E_MEMORY;
        }

        CcuLoopCkeOpV3 waitOp;
        CcuLoopCkeOpV3 setOp;
        CHK_RET(CollectLoopCkeOpsA5(instr, sampleParam, waitOp, setOp));
        reduceInstr->AddWait(waitOp.ckeId, waitOp.mask);
        reduceInstr->AddSet(setOp.ckeId, setOp.mask);

        uint16_t count = 0;
        uint16_t dataType = 0;
        uint16_t lengthId = 0;
        const uint16_t* msIds = nullptr;
        uint16_t reduceType = CcuRep::CCU_REDUCE_SUM;
        HcclReduceOp checkerReduceOp = HCCL_REDUCE_SUM;
        if (instr->header.code == ADD_CODE) {
            count = instr->v1.add.count;
            dataType = instr->v1.add.dataType;
            lengthId = instr->v1.add.XnIdLength;
            msIds = instr->v1.add.msId;
            reduceType = CcuRep::CCU_REDUCE_SUM;
            checkerReduceOp = HCCL_REDUCE_SUM;
        } else if (instr->header.code == MAX_CODE) {
            count = instr->v1.max.count;
            dataType = instr->v1.max.dataType;
            lengthId = instr->v1.max.XnIdLength;
            msIds = instr->v1.max.msId;
            reduceType = CcuRep::CCU_REDUCE_MAX;
            checkerReduceOp = HCCL_REDUCE_MAX;
        } else if (instr->header.code == MIN_CODE) {
            count = instr->v1.min.count;
            dataType = instr->v1.min.dataType;
            lengthId = instr->v1.min.XnIdLength;
            msIds = instr->v1.min.msId;
            reduceType = CcuRep::CCU_REDUCE_MIN;
            checkerReduceOp = HCCL_REDUCE_MIN;
        } else {
            return HCCL_E_NOT_SUPPORT;
        }

        DataType hcclDataType;
        CHK_RET(GetHcclDataTypeFromCCUDataType(dataType, reduceType, hcclDataType));
        reduceInstr->dataType = g_DataType2CheckerDataType_aicpu[hcclDataType];
        reduceInstr->reduceOp = checkerReduceOp;

        uint64_t len = 0;
        CHK_GET_XN_V3(curCcuTask, queId, lengthId, len);
        CHK_RET(ValidateLoopLen(len));
        for (uint64_t iter = 0; iter < loopCnt; ++iter) {
            LoopGroupParam iterParam;
            SetLoopParamAtIteration(sampleParam, static_cast<u32>(iter), iterParam);
            std::vector<MemSlice> srcGroup;
            for (uint16_t i = 1; i < count + 2; ++i) {
                DataSlice srcSlice;
                const uint16_t srcMSId = UpdateMSId(msIds[i], &iterParam);
                CHK_RET(GenSliceFromMs(srcMSId, len, srcSlice));
                srcGroup.push_back(MakeCcuMemSlice(rankId, srcSlice));
                reduceInstr->msIds.insert(srcMSId);
            }
            DataSlice dstSlice;
            const uint16_t dstMSId = UpdateMSId(msIds[0], &iterParam);
            CHK_RET(GenSliceFromMs(dstMSId, len, dstSlice));
            reduceInstr->srcs.push_back(std::move(srcGroup));
            reduceInstr->dsts.push_back(MakeCcuMemSlice(rankId, dstSlice));
            reduceInstr->msIds.insert(dstMSId);
        }
        return HCCL_SUCCESS;
    }

    static HcclResult CollectClearCkeLoopInstrA5(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, uint16_t instrId,
        const LoopGroupParam& sampleParam, std::shared_ptr<CcuLoopInstrV3>& instrInLoop)
    {
        if (instr->header.code != CLEARCKE_CODE || instr->v1.clearCKE.clearMask != 0) {
            return HCCL_E_NOT_SUPPORT;
        }
        RankId rankId = curCcuTask->GetRankId();
        uint32_t dieId = INVALID_DIE_ID;
        curCcuTask->GetDieId(queId, dieId);
        auto clearInstr = EnsureCcuLoopInstr<CcuLoopClearCkeV3>(instrInLoop, rankId, dieId, instrId);
        if (clearInstr == nullptr) {
            return HCCL_E_MEMORY;
        }
        CcuLoopCkeOpV3 waitOp;
        CcuLoopCkeOpV3 setOp;
        CHK_RET(CollectLoopCkeOpsA5(instr, sampleParam, waitOp, setOp));
        clearInstr->AddWait(waitOp.ckeId, waitOp.mask);
        clearInstr->clearCKEId = UpdateCKEId(instr->v1.clearCKE.clearCKEId, const_cast<LoopGroupParam*>(&sampleParam));
        clearInstr->clearMask = instr->v1.clearCKE.clearMask;
        clearInstr->clearType = instr->v1.clearCKE.clearType;
        return HCCL_SUCCESS;
    }

    static HcclResult CollectMergedLoopInstrsA5(
        CcuGraphStateV3* curCcuTask, uint32_t queId, uint16_t startInstrId, uint16_t endInstrId,
        const LoopGroupParam& sampleParam, uint64_t loopCnt, CcuLoopInstrsV3& mergedInstrs)
    {
        mergedInstrs.startInstrId = startInstrId;
        mergedInstrs.endInstrId = endInstrId;
        mergedInstrs.loopCnt = loopCnt;
        mergedInstrs.instrs.clear();
        std::map<uint16_t, uint16_t> producedMasks;
        hcomm::CcuRep::CcuInstrInfo& microCodeQue = curCcuTask->instrInfo[queId];
        for (uint16_t insId = startInstrId; insId <= endInstrId; ++insId) {
            const CcuRep::CcuInstr* curInstr = &microCodeQue.instrVec[insId - curCcuTask->startInstrIdInQue[queId]];
            CcuLoopCkeOpV3 waitOp;
            CcuLoopCkeOpV3 setOp;
            HcclResult ret = CollectLoopCkeOpsA5(curInstr, sampleParam, waitOp, setOp);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            if (!ConsumeProducedMask(producedMasks, waitOp.ckeId, waitOp.mask)) {
                return HCCL_E_NOT_SUPPORT;
            }
            ProduceMask(producedMasks, setOp.ckeId, setOp.mask);

            std::shared_ptr<CcuLoopInstrV3> instrInLoop;
            if (curInstr->header.type == CTRL_TYPE) {
                ret = CollectClearCkeLoopInstrA5(curInstr, curCcuTask, queId, insId, sampleParam, instrInLoop);
            } else if (curInstr->header.type == TRANS_TYPE) {
                ret = CollectTransLoopInstrA5(curInstr, curCcuTask, queId, insId, sampleParam, loopCnt, instrInLoop);
            } else if (curInstr->header.type == REDUCE_TYPE) {
                ret = CollectReduceLoopInstrA5(curInstr, curCcuTask, queId, insId, sampleParam, loopCnt, instrInLoop);
            } else {
                ret = HCCL_E_NOT_SUPPORT;
            }
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            mergedInstrs.instrs.push_back(std::move(instrInLoop));
        }

        for (const auto& entry : producedMasks) {
            if (entry.second != 0) {
                return HCCL_E_NOT_SUPPORT;
            }
        }
        return HCCL_SUCCESS;
    }

    static HcclResult CollectLoopExpandA5(
        const CcuRep::CcuInstr* loopInstr, CcuGraphStateV3* curCcuTask, uint32_t queId, const LoopGroupParam& loopParam,
        CcuLoopInstrsV3& mergedInstrs, uint64_t& loopCnt)
    {
        if (loopInstr->header.type != CTRL_TYPE || loopInstr->header.code != LOOP_CODE) {
            return HCCL_E_NOT_SUPPORT;
        }
        RankId rankId = curCcuTask->GetRankId();
        uint32_t dieId = INVALID_DIE_ID;
        curCcuTask->GetDieId(queId, dieId);
        uint64_t value = 0;
        CHK_GET_XN_V3(curCcuTask, queId, loopInstr->v1.loop.xnId, value);
        LoopGroupParam sampleParam = loopParam;
        if (sampleParam.loopXms.size() <= sampleParam.curLoopIdx) {
            sampleParam.loopXms.resize(sampleParam.curLoopIdx + 1);
        }
        sampleParam.loopXms[sampleParam.curLoopIdx].value = value;
        sampleParam.curLoopCnt = 0;
        loopCnt = sampleParam.loopXms[sampleParam.curLoopIdx].loopCnt;
        if (loopCnt == 0) {
            return HCCL_E_NOT_SUPPORT;
        }
        return CollectMergedLoopInstrsA5(
            curCcuTask, queId, loopInstr->v1.loop.startInstrId, loopInstr->v1.loop.endInstrId, sampleParam, loopCnt,
            mergedInstrs);
    }

    static HcclResult RecordMergedLoopParamA5(
        const CcuRep::CcuInstr* loopInstr, CcuGraphStateV3* curCcuTask, uint32_t queId, LoopGroupParam& loopParam)
    {
        RankId rankId = curCcuTask->GetRankId();
        uint32_t dieId = INVALID_DIE_ID;
        curCcuTask->GetDieId(queId, dieId);
        uint64_t value = 0;
        CHK_GET_XN_V3(curCcuTask, queId, loopInstr->v1.loop.xnId, value);
        if (loopParam.loopXms.size() <= loopParam.curLoopIdx) {
            loopParam.loopXms.resize(loopParam.curLoopIdx + 1);
        }
        loopParam.loopXms[loopParam.curLoopIdx].value = value;
        return HCCL_SUCCESS;
    }

    static HcclResult TryProcessMergedLoopA5(
        const CcuRep::CcuInstr* loopInstr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
        LoopGroupParam& loopParam, uint32_t loopGroupIdx, uint32_t expandTimes, bool& merged)
    {
        merged = false;
        CcuLoopV3 mergedLoop;
        mergedLoop.expandCnt = expandTimes;
        mergedLoop.instrId = loopInstr->v1.loop.startInstrId;
        uint64_t mergedLoopCnt = 0;
        for (uint32_t expandCnt = 0; expandCnt < expandTimes; ++expandCnt) {
            LoopGroupParam expandParam = loopParam;
            expandParam.curExpandCnt = expandCnt;
            CcuLoopInstrsV3 expandInstrs;
            uint64_t loopCnt = 0;
            HcclResult ret = CollectLoopExpandA5(loopInstr, curCcuTask, queId, expandParam, expandInstrs, loopCnt);
            if (ret != HCCL_SUCCESS) {
                if (ret == HCCL_E_NOT_SUPPORT) {
                    return HCCL_SUCCESS;
                }
                return ret;
            }
            if (expandCnt == 0) {
                mergedLoopCnt = loopCnt;
            } else if (mergedLoopCnt != loopCnt) {
                return HCCL_SUCCESS;
            }
            mergedLoop.loopExpands.push_back(std::move(expandInstrs));
        }
        if (mergedLoop.loopExpands.empty() || !mergedLoop.LoopIterationMerge() || !mergedLoop.LoopExpandMerge()) {
            return HCCL_SUCCESS;
        }

        CHK_RET(RecordMergedLoopParamA5(loopInstr, curCcuTask, queId, loopParam));
        auto loopIdx = curCcuTask->loopIdx[loopGroupIdx]++;
        TaskNode* loopStart = AddLoopStartTask(
            queId, loopIdx, loopGroupIdx, curCcuTask, loopInstr->v1.loop.startInstrId, loopInstr->v1.loop.endInstrId,
            mergedLoopCnt, static_cast<uint64_t>(expandTimes));
        if (loopStart == nullptr) {
            return HCCL_E_INTERNAL;
        }
        CHK_RET(EmitMergedLoopInstrsV3(curCcuTask, queId, mergedLoop.loopExpands.front(), isContinue));
        if (!isContinue) {
            merged = true;
            return HCCL_SUCCESS;
        }
        TaskNode* loopEnd = AddLoopEndTask(queId, loopIdx, loopGroupIdx, curCcuTask);
        if (loopEnd == nullptr) {
            return HCCL_E_INTERNAL;
        }
        const uint16_t loopInstructionId = static_cast<uint16_t>(loopInstr->v1.loop.startInstrId);
        HCCL_VM_INFO(
            "Merged LoopGroup body successfully, loopInstructionId={}, loopCount={}, "
            "expandTimes={}, mergedInstructionCount={}",
            loopInstructionId, mergedLoopCnt, expandTimes, mergedLoop.loopExpands.front().instrs.size());
        merged = true;
        return HCCL_SUCCESS;
    }

    // 处理Loop指令
    HcclResult ProcessLoopIns(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
        LoopGroupParam* loopGroupParam, uint32_t loopGroupIdx);

    HcclResult TransformLoopGroupInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
        LoopGroupParam* loopInfo)
    {
        CHK_RET(IsLoopGroupParamNull(loopInfo));
        uint16_t startLoopInstrId = instr->v1.loopGroup.startLoopInstrId;
        uint16_t xnId = instr->v1.loopGroup.xnId;
        uint16_t xmId = instr->v1.loopGroup.xmId;
        uint16_t highPerfModeEn = instr->v1.loopGroup.highPerfModeEn;

        RankId rankId = curCcuTask->GetRankId();
        uint32_t dieId;
        curCcuTask->GetDieId(queId, dieId);

        LoopGroupParam loopGroupParam;
        CHK_GET_XN_V3(curCcuTask, queId, xnId, loopGroupParam.loopGroupXn.value);
        CHK_GET_XN_V3(curCcuTask, queId, xmId, loopGroupParam.loopGroupXm.value);

        hcomm::CcuRep::CcuInstrInfo& microCodeQue = curCcuTask->instrInfo[queId];
        auto loopGroupIdx = curCcuTask->loopGroupIdx++;
        uint64_t loopCnt = loopGroupParam.loopGroupXn.loopInsCnt;
        HCCL_VM_DEBUG(
            "Parsed LoopGroup configuration, loopCount={}, expandOffset={}, expandCount={}", loopCnt,
            static_cast<uint64_t>(loopGroupParam.loopGroupXn.expandOffset),
            static_cast<uint64_t>(loopGroupParam.loopGroupXn.expandCnt));

        for (u32 curLoopIdx = 0; curLoopIdx < loopCnt; curLoopIdx++) {
            loopGroupParam.curLoopIdx = curLoopIdx;
            // 计算当前Loop指令的位置
            u32 insPos = startLoopInstrId + curLoopIdx - curCcuTask->startInstrIdInQue[queId];
            const uint32_t expandTimes = (curLoopIdx < loopGroupParam.loopGroupXn.expandOffset) ?
                                             1U :
                                             static_cast<uint32_t>(loopGroupParam.loopGroupXn.expandCnt + 1U);
            bool merged = false;
            CHK_RET(TryProcessMergedLoopA5(
                &microCodeQue.instrVec[insPos], curCcuTask, queId, isContinue, loopGroupParam, loopGroupIdx,
                expandTimes, merged));
            if (!isContinue) {
                return HCCL_SUCCESS;
            }
            if (merged) {
                continue;
            }
            if (curLoopIdx < loopGroupParam.loopGroupXn.expandOffset) {
                // 不用进行Loop展开
                CHK_RET(ProcessLoopIns(
                    &microCodeQue.instrVec[insPos], curCcuTask, queId, isContinue, &loopGroupParam, loopGroupIdx));
            } else {
                // 需要进行loop展开
                for (u32 expandCnt = 0; expandCnt <= loopGroupParam.loopGroupXn.expandCnt; expandCnt++) {
                    loopGroupParam.curExpandCnt = expandCnt;
                    CHK_RET(ProcessLoopIns(
                        &microCodeQue.instrVec[insPos], curCcuTask, queId, isContinue, &loopGroupParam, loopGroupIdx));
                }
            }
        }

        HCCL_VM_DEBUG(
            "Finished LoopGroup expansion, startLoopInstructionId={}, loopGroupXnId={}, "
            "loopGroupXmId={}, highPerfModeEnabled={}",
            startLoopInstrId, xnId, xmId, highPerfModeEn);
        return HCCL_SUCCESS;
    }

    HcclResult TransformLoopInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
        LoopGroupParam* loopGroupParam)
    {
        uint16_t startInstrId = instr->v1.loop.startInstrId;
        uint16_t endInstrId = instr->v1.loop.endInstrId;
        uint16_t xnId = instr->v1.loop.xnId;
        HCCL_VM_ERROR(
            "{} A loop instruction cannot be executed by itself; it must be triggered from a "
            "LoopGroup, startInstructionId={}, endInstructionId={}, loopCountXnId={}",
            MakeErrorCodeText(ErrorCode::GRAPH_UNSUPPORTED).c_str(), startInstrId, endInstrId, xnId);
        // 当前Loop都需要通过LoopGoup来触发，暂不支持单独解析Loop命令
        return HCCL_E_INTERNAL;
    }

    HcclResult TransformClearCKEInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
        LoopGroupParam* loopGroupParam)
    {
        uint16_t clearType = instr->v1.clearCKE.clearType;
        uint16_t clearCKEId = UpdateCKEId(instr->v1.clearCKE.clearCKEId, loopGroupParam);
        uint16_t clearMask = instr->v1.clearCKE.clearMask;
        uint16_t waitCKEId = UpdateCKEId(instr->v1.clearCKE.waitCKEId, loopGroupParam);
        uint16_t waitCKEMask = instr->v1.clearCKE.waitCKEMask;

        RankId rankId = curCcuTask->GetRankId();
        uint32_t dieId;
        curCcuTask->GetDieId(queId, dieId);

        CHK_RET(ProcessWaitMask(rankId, dieId, curCcuTask, queId, waitCKEId, waitCKEMask, isContinue));
        if (!isContinue) {
            return HCCL_SUCCESS;
        }

        if (clearMask != 0x0000) {
            HCCL_VM_ERROR(
                "{} CheckerV3 only supports clearing an entire CKE register here; partial clear "
                "mask is not supported, clearCkeId={}, clearMask=0x{:04x}",
                MakeErrorCodeText(ErrorCode::GRAPH_UNSUPPORTED).c_str(), clearCKEId, clearMask);
            return HCCL_E_INTERNAL;
        }

        CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

        HCCL_VM_DEBUG(
            "Wait CKE[{}:0x{:04x}], Clear CKE[{}:0x{:04x}], clearType[{}]", waitCKEId, waitCKEMask, clearCKEId,
            clearMask, clearType);
        return HCCL_SUCCESS;
    }

    HcclResult TransformSyncGSAInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
        LoopGroupParam* loopGroupParam)
    {
        CHK_RET(IsLoopGroupParamNull(loopGroupParam));
        uint16_t rmtGSAId = instr->v1.syncGSA.rmtGSAId;
        uint16_t locGSAId = instr->v1.syncGSA.locGSAId;
        uint16_t channelId = instr->v1.syncGSA.channelId;
        uint16_t setRmtCKEId = instr->v1.syncGSA.setRmtCKEId;
        uint16_t setRmtCKEMask = instr->v1.syncGSA.setRmtCKEMask;
        uint16_t clearType = instr->v1.syncGSA.clearType;
        uint16_t setCKEId = instr->v1.syncGSA.setCKEId;
        uint16_t setCKEMask = instr->v1.syncGSA.setCKEMask;
        uint16_t waitCKEId = instr->v1.syncGSA.waitCKEId;
        uint16_t waitCKEMask = instr->v1.syncGSA.waitCKEMask;

        RankId rankId = curCcuTask->GetRankId();
        uint32_t dieId;
        curCcuTask->GetDieId(queId, dieId);

        CHK_RET(ProcessWaitMask(rankId, dieId, curCcuTask, queId, waitCKEId, waitCKEMask, isContinue));
        if (!isContinue) {
            return HCCL_SUCCESS;
        }

        // 获取远端rankId与dieId
        if (g_allRankChannelInfo[rankId][dieId].count(channelId) == 0) {
            HCCL_VM_ERROR(
                "{} Cannot find channel mapping for this instruction, instruction=SyncGSA, "
                "localRankId={}, localDieId={}, channelId={}",
                MakeErrorCodeText(ErrorCode::GRAPH_RESOURCE_NOT_FOUND).c_str(), static_cast<uint32_t>(rankId), dieId,
                channelId);
            return HCCL_E_INTERNAL;
        }

        RankId rmtRankId = g_allRankChannelInfo[rankId][dieId][channelId].dstRank;
        uint32_t rmtDieId = g_allRankChannelInfo[rankId][dieId][channelId].remoteDieId;

        uint64_t localGSAValue = 0;
        CHK_GET_GSA_V3(curCcuTask, queId, locGSAId, localGSAValue);
        CHK_RET(AllRankParamRecorder::Global()->SetGSA(rmtRankId, rmtDieId, rmtGSAId, localGSAValue));

        if (setRmtCKEMask != 0x0000) {
            AddPost(rankId, rmtRankId, queId, curCcuTask, rmtDieId, setRmtCKEId, setRmtCKEMask);

            // 将对应的bit位置一
            uint16_t ckeValue = 0;
            CHK_RET(AllRankParamRecorder::Global()->GetCKE(rmtRankId, rmtDieId, setRmtCKEId, ckeValue));
            CHK_RET(AllRankParamRecorder::Global()->SetCKE(rmtRankId, rmtDieId, setRmtCKEId, ckeValue | setRmtCKEMask));
        }

        CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask, true));

        CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

        HCCL_VM_DEBUG(
            "Wait CKE[{}:0x{:04x}]:\n"
            "  sync: Sync GSA[{}]({}) To rmtGSA[{}] Use Channel[{}]\n"
            "  set: Set rmtCKE[{}:0x{:04x}], Set CKE[{}:0x{:04x}]\n"
            "  ctrl: clearType[{}]",
            waitCKEId, waitCKEMask, locGSAId, localGSAValue, rmtGSAId, channelId, setRmtCKEId, setRmtCKEMask, setCKEId,
            setCKEMask, clearType);

        return HCCL_SUCCESS;
    }

    HcclResult GetHcclDataTypeFromCCUDataType(uint16_t ccuDataType, uint16_t ccuReduceType, DataType& dataType)
    {
        static std::map<uint16_t, DataType> ccuSumDataTypeMap = {
            {0, DataType::FP32},    {1, DataType::FP16},    {2, DataType::BFP16}, {3, DataType::HIF8},
            {4, DataType::FP8E4M3}, {5, DataType::FP8E5M2}, {6, DataType::INT8},  {7, DataType::UINT8},
            {8, DataType::INT16},   {9, DataType::INT32},
        };

        static std::map<uint16_t, DataType> ccuMaxMinDataTypeMap = {
            {0, DataType::FP32},  {1, DataType::FP16},  {2, DataType::BFP16}, {6, DataType::INT8},
            {7, DataType::UINT8}, {8, DataType::INT16}, {9, DataType::INT32},

        };

        if (ccuReduceType == CcuRep::CCU_REDUCE_SUM) {
            if (ccuSumDataTypeMap.find(ccuDataType) == ccuSumDataTypeMap.end()) {
                HCCL_VM_ERROR(
                    "{} This CCU reduce data type is not supported for SUM, ccuDataType={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_UNSUPPORTED).c_str(), ccuDataType);
                return HCCL_E_INTERNAL;
            }
            dataType = ccuSumDataTypeMap[ccuDataType];
        } else if (ccuReduceType == CcuRep::CCU_REDUCE_MAX || ccuReduceType == CcuRep::CCU_REDUCE_MIN) {
            if (ccuMaxMinDataTypeMap.find(ccuDataType) == ccuMaxMinDataTypeMap.end()) {
                HCCL_VM_ERROR(
                    "{} This CCU reduce data type is not supported for MAX or MIN, "
                    "ccuDataType={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_UNSUPPORTED).c_str(), ccuDataType);
                return HCCL_E_INTERNAL;
            }
            dataType = ccuMaxMinDataTypeMap[ccuDataType];
        } else {
            HCCL_VM_ERROR(
                "{} This CCU reduce opcode is not supported, ccuReduceType={}",
                MakeErrorCodeText(ErrorCode::GRAPH_UNSUPPORTED).c_str(), ccuReduceType);
            return HCCL_E_INTERNAL;
        }

        return HCCL_SUCCESS;
    }

    HcclResult TransformAddInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
        LoopGroupParam* loopGroupParam)
    {
        uint16_t count = instr->v1.add.count;
        uint16_t castEn = instr->v1.add.castEn;
        uint16_t dataType = instr->v1.add.dataType;
        uint16_t clearType = instr->v1.add.clearType;
        uint16_t setCKEId = UpdateCKEId(instr->v1.add.setCKEId, loopGroupParam);
        uint16_t setCKEMask = instr->v1.add.setCKEMask;
        uint16_t waitCKEId = UpdateCKEId(instr->v1.add.waitCKEId, loopGroupParam);
        uint16_t waitCKEMask = instr->v1.add.waitCKEMask;
        uint16_t lengthId = instr->v1.add.XnIdLength;

        uint16_t msId[CcuRep::CCU_REDUCE_MAX_MS];
        for (uint16_t index = 0; index < CcuRep::CCU_REDUCE_MAX_MS; index++) {
            msId[index] = UpdateMSId(instr->v1.add.msId[index], loopGroupParam);
        }
        // todo: 对于不支持的数据类型需要进行报错处理

        RankId rankId = curCcuTask->GetRankId();
        uint32_t dieId;
        curCcuTask->GetDieId(queId, dieId);

        CHK_RET(ProcessWaitMask(rankId, dieId, curCcuTask, queId, waitCKEId, waitCKEMask, isContinue));
        if (!isContinue) {
            return HCCL_SUCCESS;
        }

        uint64_t len = 0;
        CHK_GET_XN_V3(curCcuTask, queId, lengthId, len);
        DataSlice dstSlice;
        CHK_RET(GenSliceFromMs(msId[0], len, dstSlice));

        std::vector<DataSlice> srcSlices;
        for (uint16_t i = 1; i < count + 2; i++) {
            DataSlice tmp;
            CHK_RET(GenSliceFromMs(msId[i], len, tmp));
            srcSlices.push_back(tmp);
        }

        DataType hcclDataType;
        CHK_RET(GetHcclDataTypeFromCCUDataType(dataType, CcuRep::CCU_REDUCE_SUM, hcclDataType));
        AddLocalBatchReduce(rankId, queId, curCcuTask, srcSlices, dstSlice, hcclDataType);

        CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask));

        CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

        HCCL_VM_DEBUG(
            "Completed local SUM reduce:\n"
            "  wait: waitCkeId={}, waitMask=0x{:04x}\n"
            "  src: msList={}, sourceCount={}, dataType={}, castEnabled={}\n"
            "  set: setCkeId={}, setMask=0x{:04x}\n"
            "  ctrl: clearType={}",
            waitCKEId, waitCKEMask, ParseMSList(instr), count, dataType, castEn, setCKEId, setCKEMask, clearType);

        return HCCL_SUCCESS;
    }

    HcclResult TransformMaxInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
        LoopGroupParam* loopGroupParam)
    {
        uint16_t count = instr->v1.max.count;
        uint16_t dataType = instr->v1.max.dataType;
        uint16_t clearType = instr->v1.max.clearType;
        uint16_t setCKEId = UpdateCKEId(instr->v1.max.setCKEId, loopGroupParam);
        uint16_t setCKEMask = instr->v1.max.setCKEMask;
        uint16_t waitCKEId = UpdateCKEId(instr->v1.max.waitCKEId, loopGroupParam);
        uint16_t waitCKEMask = instr->v1.max.waitCKEMask;
        uint16_t lengthId = instr->v1.max.XnIdLength;

        uint16_t msId[CcuRep::CCU_REDUCE_MAX_MS];
        for (uint16_t index = 0; index < CcuRep::CCU_REDUCE_MAX_MS; index++) {
            msId[index] = UpdateMSId(instr->v1.max.msId[index], loopGroupParam);
        }
        // todo: 对于不支持的数据类型需要进行报错处理

        RankId rankId = curCcuTask->GetRankId();
        uint32_t dieId;
        curCcuTask->GetDieId(queId, dieId);

        CHK_RET(ProcessWaitMask(rankId, dieId, curCcuTask, queId, waitCKEId, waitCKEMask, isContinue));
        if (!isContinue) {
            return HCCL_SUCCESS;
        }

        uint64_t len = 0;
        CHK_GET_XN_V3(curCcuTask, queId, lengthId, len);
        DataSlice dstSlice;
        CHK_RET(GenSliceFromMs(msId[0], len, dstSlice));

        std::vector<DataSlice> srcSlices;
        for (uint16_t i = 1; i < count + 2; i++) {
            DataSlice tmp;
            CHK_RET(GenSliceFromMs(msId[i], len, tmp));
            srcSlices.push_back(tmp);
        }

        DataType hcclDataType;
        CHK_RET(GetHcclDataTypeFromCCUDataType(dataType, CcuRep::CCU_REDUCE_MAX, hcclDataType));
        AddLocalBatchReduce(rankId, queId, curCcuTask, srcSlices, dstSlice, hcclDataType);

        CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask));

        CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

        HCCL_VM_DEBUG(
            "Completed local MAX reduce:\n"
            "  wait: waitCkeId={}, waitMask=0x{:04x}\n"
            "  src: msList={}, sourceCount={}, dataType={}\n"
            "  set: setCkeId={}, setMask=0x{:04x}\n"
            "  ctrl: clearType={}",
            waitCKEId, waitCKEMask, ParseMSList(instr), count, dataType, setCKEId, setCKEMask, clearType);

        return HCCL_SUCCESS;
    }

    HcclResult TransformMinInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
        LoopGroupParam* loopGroupParam)
    {
        uint16_t count = instr->v1.min.count;
        uint16_t dataType = instr->v1.min.dataType;
        uint16_t clearType = instr->v1.min.clearType;
        uint16_t setCKEId = UpdateCKEId(instr->v1.min.setCKEId, loopGroupParam);
        uint16_t setCKEMask = instr->v1.min.setCKEMask;
        uint16_t waitCKEId = UpdateCKEId(instr->v1.min.waitCKEId, loopGroupParam);
        uint16_t waitCKEMask = instr->v1.min.waitCKEMask;
        uint16_t lengthId = instr->v1.min.XnIdLength;

        uint16_t msId[CcuRep::CCU_REDUCE_MAX_MS];
        for (uint16_t index = 0; index < CcuRep::CCU_REDUCE_MAX_MS; index++) {
            msId[index] = UpdateMSId(instr->v1.min.msId[index], loopGroupParam);
        }
        // todo: 对于不支持的数据类型需要进行报错处理

        RankId rankId = curCcuTask->GetRankId();
        uint32_t dieId;
        curCcuTask->GetDieId(queId, dieId);

        CHK_RET(ProcessWaitMask(rankId, dieId, curCcuTask, queId, waitCKEId, waitCKEMask, isContinue));
        if (!isContinue) {
            return HCCL_SUCCESS;
        }

        uint64_t len = 0;
        CHK_GET_XN_V3(curCcuTask, queId, lengthId, len);
        DataSlice dstSlice;
        CHK_RET(GenSliceFromMs(msId[0], len, dstSlice));

        std::vector<DataSlice> srcSlices;
        for (uint16_t i = 1; i < count + 2; i++) {
            DataSlice tmp;
            CHK_RET(GenSliceFromMs(msId[i], len, tmp));
            srcSlices.push_back(tmp);
        }

        DataType hcclDataType;
        CHK_RET(GetHcclDataTypeFromCCUDataType(dataType, CcuRep::CCU_REDUCE_MIN, hcclDataType));
        AddLocalBatchReduce(rankId, queId, curCcuTask, srcSlices, dstSlice, hcclDataType);

        CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask));

        CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

        HCCL_VM_DEBUG(
            "Completed local MIN reduce:\n"
            "  wait: waitCkeId={}, waitMask=0x{:04x}\n"
            "  src: msList={}, sourceCount={}, dataType={}\n"
            "  set: setCkeId={}, setMask=0x{:04x}\n"
            "  ctrl: clearType={}",
            waitCKEId, waitCKEMask, ParseMSList(instr), count, dataType, setCKEId, setCKEMask, clearType);
        return HCCL_SUCCESS;
    }

    // 以下是一些辅助函数，用于转换函数使用
    HcclResult TransformLoadSqeArgsToGSAInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
    {
        auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
        return TransformLoadSqeArgsToGSAInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
    }

    HcclResult TransformLoadSqeArgsToXnInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
    {
        auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
        return TransformLoadSqeArgsToXnInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
    }

    HcclResult TransformLoadImdToGSAInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
    {
        auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
        return TransformLoadImdToGSAInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
    }

    HcclResult TransformLoadImdToXnInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
    {
        auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
        return TransformLoadImdToXnInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
    }

    HcclResult TransformLoadGSAXnInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
    {
        auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
        return TransformLoadGSAXnInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
    }

    HcclResult TransformLoadGSAGSAInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
    {
        auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
        return TransformLoadGSAGSAInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
    }

    HcclResult TransformLoadXXInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
    {
        auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
        return TransformLoadXXInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
    }

    HcclResult TransformLoopGroupInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
    {
        auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
        return TransformLoopGroupInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
    }

    HcclResult TransformLoopInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
    {
        auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
        return TransformLoopInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
    }

    HcclResult TransformSetCKEInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
    {
        auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
        return TransformSetCKEInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
    }

    HcclResult TransformClearCKEInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
    {
        auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
        return TransformClearCKEInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
    }

    HcclResult TransformJumpInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
    {
        auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
        return TransformJumpInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
    }

    HcclResult TransformTransLocMemToLocMSInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
    {
        auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
        return TransformTransLocMemToLocMSInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
    }

    HcclResult TransformTransRmtMemToLocMSInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
    {
        auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
        return TransformTransRmtMemToLocMSInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
    }

    HcclResult TransformTransLocMSToLocMemInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
    {
        auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
        return TransformTransLocMSToLocMemInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
    }

    HcclResult TransformTransLocMSToRmtMemInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
    {
        auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
        return TransformTransLocMSToRmtMemInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
    }
    HcclResult TransformTransRmtMSToLocMemInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
    {
        auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
        return TransformTransRmtMSToLocMemInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
    }

    HcclResult TransformTransLocMSToLocMSInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
    {
        auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
        return TransformTransLocMSToLocMSInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
    }

    HcclResult TransformTransRmtMSToLocMSInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
    {
        auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
        return TransformTransRmtMSToLocMSInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
    }

    HcclResult TransformTransLocMSToRmtMSInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
    {
        auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
        return TransformTransLocMSToRmtMSInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
    }

    HcclResult TransformTransRmtMemToLocMemInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
    {
        auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
        return TransformTransRmtMemToLocMemInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
    }

    HcclResult TransformTransLocMemToRmtMemInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
    {
        auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
        return TransformTransLocMemToRmtMemInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
    }

    HcclResult TransformTransLocMemToLocMemInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
    {
        auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
        return TransformTransLocMemToLocMemInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
    }

    HcclResult TransformSyncCKEInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
    {
        auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
        return TransformSyncCKEInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
    }

    HcclResult TransformSyncGSAInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
    {
        auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
        return TransformSyncGSAInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
    }

    HcclResult TransformSyncXnInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
    {
        auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
        return TransformSyncXnInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
    }

    HcclResult TransformAddInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
    {
        auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
        return TransformAddInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
    }

    HcclResult TransformMaxInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
    {
        auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
        return TransformMaxInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
    }

    HcclResult TransformMinInstr(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
    {
        auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
        return TransformMinInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
    }

    std::unordered_map<uint16_t, TransformInstrFunc> transformA5InstrSqeMap{
        {CcuRep::InstrHeader(LOAD_TYPE, LOADSQEARGSTOGSA_CODE).header, &TransformLoadSqeArgsToGSAInstr},
        {CcuRep::InstrHeader(LOAD_TYPE, LOADSQEARGSTOXN_CODE).header, &TransformLoadSqeArgsToXnInstr},
        {CcuRep::InstrHeader(LOAD_TYPE, LOADIMDTOGSA_CODE).header, &TransformLoadImdToGSAInstr},
        {CcuRep::InstrHeader(LOAD_TYPE, LOADIMDTOXN_CODE).header, &TransformLoadImdToXnInstr},
        {CcuRep::InstrHeader(LOAD_TYPE, LOADGSAXN_CODE).header, &TransformLoadGSAXnInstr},
        {CcuRep::InstrHeader(LOAD_TYPE, LOADGSAGSA_CODE).header, &TransformLoadGSAGSAInstr},
        {CcuRep::InstrHeader(LOAD_TYPE, LOADXX_CODE).header, &TransformLoadXXInstr},
        {CcuRep::InstrHeader(CTRL_TYPE, LOOPGROUP_CODE).header, &TransformLoopGroupInstr},
        {CcuRep::InstrHeader(CTRL_TYPE, LOOP_CODE).header, &TransformLoopInstr},
        {CcuRep::InstrHeader(CTRL_TYPE, SETCKE_CODE).header, &TransformSetCKEInstr},
        {CcuRep::InstrHeader(CTRL_TYPE, CLEARCKE_CODE).header, &TransformClearCKEInstr},
        {CcuRep::InstrHeader(CTRL_TYPE, JMP_CODE).header, &TransformJumpInstr},
        {CcuRep::InstrHeader(TRANS_TYPE, TRANSLOCMEMTOLOCMS_CODE).header, &TransformTransLocMemToLocMSInstr},
        {CcuRep::InstrHeader(TRANS_TYPE, TRANSRMTMEMTOLOCMS_CODE).header, &TransformTransRmtMemToLocMSInstr},
        {CcuRep::InstrHeader(TRANS_TYPE, TRANSLOCMSTOLOCMEM_CODE).header, &TransformTransLocMSToLocMemInstr},
        {CcuRep::InstrHeader(TRANS_TYPE, TRANSLOCMSTORMTMEM_CODE).header, &TransformTransLocMSToRmtMemInstr},
        {CcuRep::InstrHeader(TRANS_TYPE, TRANSRMTMSTOLOCMEM_CODE).header, &TransformTransRmtMSToLocMemInstr},
        {CcuRep::InstrHeader(TRANS_TYPE, TRANSLOCMSTOLOCMS_CODE).header, &TransformTransLocMSToLocMSInstr},
        {CcuRep::InstrHeader(TRANS_TYPE, TRANSRMTMSTOLOCMS_CODE).header, &TransformTransRmtMSToLocMSInstr},
        {CcuRep::InstrHeader(TRANS_TYPE, TRANSLOCMSTORMTMS_CODE).header, &TransformTransLocMSToRmtMSInstr},
        {CcuRep::InstrHeader(TRANS_TYPE, TRANSRMTMEMTOLOCMEM_CODE).header, &TransformTransRmtMemToLocMemInstr},
        {CcuRep::InstrHeader(TRANS_TYPE, TRANSLOCMEMTORMTMEM_CODE).header, &TransformTransLocMemToRmtMemInstr},
        {CcuRep::InstrHeader(TRANS_TYPE, TRANSLOCMEMTOLOCMEM_CODE).header, &TransformTransLocMemToLocMemInstr},
        {CcuRep::InstrHeader(TRANS_TYPE, SYNCCKE_CODE).header, &TransformSyncCKEInstr},
        {CcuRep::InstrHeader(TRANS_TYPE, SYNCGSA_CODE).header, &TransformSyncGSAInstr},
        {CcuRep::InstrHeader(TRANS_TYPE, SYNCXN_CODE).header, &TransformSyncXnInstr},
        {CcuRep::InstrHeader(REDUCE_TYPE, ADD_CODE).header, &TransformAddInstr},
        {CcuRep::InstrHeader(REDUCE_TYPE, MAX_CODE).header, &TransformMaxInstr},
        {CcuRep::InstrHeader(REDUCE_TYPE, MIN_CODE).header, &TransformMinInstr}};

    HcclResult ProcessLoopIns(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
        LoopGroupParam* loopGroupParam, uint32_t loopGroupIdx)
    {
        uint16_t startInstrId = instr->v1.loop.startInstrId;
        uint16_t endInstrId = instr->v1.loop.endInstrId;
        uint16_t xnId = instr->v1.loop.xnId;

        RankId rankId = curCcuTask->GetRankId();
        uint32_t dieId;
        curCcuTask->GetDieId(queId, dieId);

        uint64_t value = 0;
        CHK_GET_XN_V3(curCcuTask, queId, xnId, value);
        LoopXm xm{};
        xm.value = value;
        loopGroupParam->loopXms.push_back(xm);

        auto loopIdx = curCcuTask->loopIdx[loopGroupIdx]++;
        auto loopStart = AddLoopStartTask(
            queId, loopIdx, loopGroupIdx, curCcuTask, startInstrId, endInstrId, static_cast<uint64_t>(xm.loopCnt),
            1ULL); // loop前新增loopStart标记节点

        HCCL_VM_DEBUG("Entered loop execution, loopCount={}", static_cast<uint64_t>(xm.loopCnt));
        hcomm::CcuRep::CcuInstrInfo& microCodeQue = curCcuTask->instrInfo[queId];
        u32& pos = curCcuTask->microCodePosInQue[queId];
        const u32 savedPos = pos;
        for (u32 curLoopCnt = 0; curLoopCnt < xm.loopCnt; curLoopCnt++) {
            loopGroupParam->curLoopCnt = curLoopCnt;
            HCCL_VM_DEBUG(
                "Running loop iteration, loopCount={}, currentIteration={}", static_cast<uint64_t>(xm.loopCnt),
                curLoopCnt);
            for (uint16_t insId = startInstrId; insId <= endInstrId; insId++) {
                pos = insId - curCcuTask->startInstrIdInQue[queId];
                // 获取当前要处理的指令
                const CcuRep::CcuInstr* instr = &microCodeQue.instrVec[pos];
                const uint16_t instructionHeader = static_cast<uint16_t>(instr->header.header);
                HCCL_VM_DEBUG("Expanding loop instruction, instructionId={}", insId);
                if (transformA5InstrSqeMap.count(instructionHeader) == 0) {
                    HCCL_VM_ERROR(
                        "{} This A5 CCU instruction type is not supported during loop expansion, "
                        "rankId={}, queueId={}, instructionId={}, instructionHeader=0x{:04x}",
                        MakeErrorCodeText(ErrorCode::GRAPH_UNSUPPORTED).c_str(), curCcuTask->GetRankId(), queId, insId,
                        instructionHeader);
                    pos = savedPos;
                    return HCCL_E_INTERNAL;
                }
                CHK_RET(
                    transformA5InstrSqeMap[instructionHeader](instr, curCcuTask, queId, isContinue, loopGroupParam));
            }
        }
        pos = savedPos;
        auto loopEnd = AddLoopEndTask(queId, loopIdx, loopGroupIdx, curCcuTask); // loop后新增loopEnd标记节点

        HCCL_VM_DEBUG(
            "Finished loop expansion, startInstructionId={}, endInstructionId={}, "
            "loopCountXnId={}",
            startInstrId, endInstrId, xnId);
        return HCCL_SUCCESS;
    }

    InstructMapA5::InstructMapA5() { transformInstrSqeMap = transformA5InstrSqeMap; }

    HcclResult InstructMapA5::Transform(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
    {
        const uint16_t instructionHeader = static_cast<uint16_t>(instr->header.header);
        auto it = transformInstrSqeMap.find(instructionHeader);
        if (it == transformInstrSqeMap.end()) {
            HCCL_VM_ERROR(
                "{} This A5 CCU instruction type is not supported by CheckerV3 graph expansion, "
                "rankId={}, queueId={}, instructionHeader=0x{:04x}",
                MakeErrorCodeText(ErrorCode::GRAPH_UNSUPPORTED).c_str(),
                curCcuTask == nullptr ? std::string("null") : std::to_string(curCcuTask->GetRankId()), queId,
                instructionHeader);
            return HCCL_E_INTERNAL;
        }
        auto* param = static_cast<LoopGroupParam*>(loopParam);
        return it->second(instr, curCcuTask, queId, isContinue, param);
    }

} // namespace TaskGraphGeneratorV3
} // namespace HcclSim
