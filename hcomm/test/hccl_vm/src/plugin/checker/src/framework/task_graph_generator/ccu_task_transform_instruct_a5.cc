/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include "ccu_all_rank_param_recorder.h"
#include "ccu_task_common.h"
#include "ccu_task_transform.h"
#include "ccu_task_transform_instruct_common.h"
#include "sim_log.h"
#include "storage_manager.h"
#include "task_graph_generator.h"
#include "type_conversion.h"

using namespace HcclSim;
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
} // namespace

// todo: 后续从共享内存/文件中获取channel映射表。
extern std::map<RankId, std::map<u32, ChannelsPerDie>> g_allRankChannelInfo;

static std::map<uint16_t, uint16_t> ccuReduceTypeMap
    = {{10, CcuRep::CCU_REDUCE_SUM}, {9, CcuRep::CCU_REDUCE_MIN}, {8, CcuRep::CCU_REDUCE_MAX}};

namespace HcclSim {
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
        HCCL_ERROR("this ins do not support loop");
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult TransformLoadSqeArgsToGSAInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
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
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
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
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
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
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
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
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
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
    CHK_RET(AllRankParamRecorder::Global()->GetGSA(rankId, dieId, gsAmId, argVal1));
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xnId, argVal2));
    CHK_RET(AllRankParamRecorder::Global()->SetGSA(rankId, dieId, gsAdId, argVal1 + argVal2));
    HCCL_VM_DEBUG(
        "Load GSA[{}]({}) + Xn[{}]({}) to GSA[{}]({})", gsAmId, argVal1, xnId, argVal2, gsAdId, argVal1 + argVal2);
    return HCCL_SUCCESS;
}

HcclResult TransformLoadGSAGSAInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
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
    CHK_RET(AllRankParamRecorder::Global()->GetGSA(rankId, dieId, gsAmId, argVal1));
    CHK_RET(AllRankParamRecorder::Global()->GetGSA(rankId, dieId, gsAnId, argVal2));
    CHK_RET(AllRankParamRecorder::Global()->SetGSA(rankId, dieId, gsAdId, argVal1 + argVal2));
    HCCL_VM_DEBUG(
        "Load GSA[{}]({}) + GSA[{}]({}) to GSA[{}]({})", gsAmId, argVal1, gsAnId, argVal2, gsAdId, argVal1 + argVal2);
    return HCCL_SUCCESS;
}

HcclResult TransformLoadXXInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
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

    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xmId, argVal1));
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xnId, argVal2));
    CHK_RET(AllRankParamRecorder::Global()->SetXn(rankId, dieId, xdId, argVal1 + argVal2));
    HCCL_VM_DEBUG("Load Xn[{}]({}) + Xn[{}]({}) to Xn[{}]({})", xmId, argVal1, xnId, argVal2, xdId, argVal1 + argVal2);
    return HCCL_SUCCESS;
}

HcclResult TransformSetCKEInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
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
        HCCL_ERROR("do not support clearType[%hu]", clearType);
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
        "Wait CKE[{}:{:04x}], Set CKE[{}:{:04x}], clearType[{}]", waitCKEId, waitCKEMask, setCKEId, setCKEMask,
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
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
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
        HCCL_ERROR("channel id[%hu] in rank[%u] is not found", channelId, rankId);
        return HCCL_E_INTERNAL;
    }

    uint64_t locAddr = 0x0;
    uint64_t rmtAddr = 0x0;
    AllRankParamRecorder::Global()->GetGSA(rankId, dieId, locGSAId, locAddr);
    AllRankParamRecorder::Global()->GetGSA(rankId, dieId, rmtGSAId, rmtAddr);
    uint64_t len = 0;
    AllRankParamRecorder::Global()->GetXn(rankId, dieId, lengthXnId, len);
    CHK_PRT_RET(len == 0, HCCL_ERROR("The size of data transfer is 0."), HCCL_E_INTERNAL);
    CHK_PRT_RET(
        len > UB_MAX_SIZE, HCCL_ERROR("The size of data transfer is more than max transport size of UB."),
        HCCL_E_INTERNAL);
    DataSlice srcSlice;
    DataSlice dstSlice;
    RankId rId;
    // todo: 后续插件通过读取vm的输出文件，重建memlayout的地址类型信息
    // 南向接口：input/output buffer同北向一样劫持算子入口函数；CCL buffer需要劫持北向接口HcclGetHcclBuffer
    CHK_RET(StorageManager::GetInstance().GetSlice(locAddr, len, srcSlice));
    CHK_RET(StorageManager::GetInstance().GetSlice(rmtAddr, len, dstSlice, &rId));

    RankId rmtRankId = g_allRankChannelInfo[rankId][dieId][channelId].dstRank;
    HCCL_VM_DEBUG("curRank= {}, dstRank= {}", rankId, rmtRankId);
    // 说明transportId写错了
    if (rId != rmtRankId) {
        HCCL_ERROR(
            "remoteRankId calculated by rmtAddr and channelId %u is not equal,"
            "remoteId is %d, but the addr is in rank %d",
            channelId, rmtRankId, rId);
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
            AddWriteReduce(rankId, rmtRankId, queId, curCcuTask, srcSlice, dstSlice, checkerDataType, checkerReduceOp);
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
        "Wait CKE[{}:{:04x}], Trans LocMem[{}:{}] To RmtMem[{}:{}] With LengthXn[{}] Use Channel[{}], Set "
        "CKE[{}:{:04x}], clearType[{}], lengthEn[{}], DataType[{}], ReduceType[{}] reduceEn[{}]",
        waitCKEId, waitCKEMask, locGSAId, locXnId, rmtGSAId, rmtXnId, lengthXnId, channelId, setCKEId, setCKEMask,
        clearType, lengthEn, reduceDataType, reduceOpCode, reduceEn);
    return HCCL_SUCCESS;
}

HcclResult TransformTransLocMemToLocMemInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
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
    AllRankParamRecorder::Global()->GetGSA(rankId, dieId, srcGSAId, srcAddr);
    AllRankParamRecorder::Global()->GetGSA(rankId, dieId, dstGSAId, dstAddr);
    uint64_t len = 0;
    AllRankParamRecorder::Global()->GetXn(rankId, dieId, lengthXnId, len);
    CHK_PRT_RET(len == 0, HCCL_ERROR("The size of data transfer is 0."), HCCL_E_INTERNAL);
    CHK_PRT_RET(
        len > UB_MAX_SIZE, HCCL_ERROR("The size of data transfer is more than max transport size of UB."),
        HCCL_E_INTERNAL);

    DataSlice srcSlice;
    DataSlice dstSlice;
    // todo: 后续插件通过读取vm的输出文件，重建memlayout的地址类型信息
    HCCL_VM_DEBUG(
        "TransformTransLocMemToLocMemInstr srcGSAId= {}, srcAddr= {}, dstGSAId={}, dstAddr= {}, len= {}", srcGSAId,
        srcAddr, dstGSAId, dstAddr, len);
    CHK_RET(StorageManager::GetInstance().GetSlice(srcAddr, len, srcSlice));
    CHK_RET(StorageManager::GetInstance().GetSlice(dstAddr, len, dstSlice));

    AddLocalCopy(rankId, queId, curCcuTask, srcSlice, dstSlice);

    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask));

    CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

    HCCL_VM_DEBUG(
        "Wait CKE[{}:{:04x}], Trans LocMem[{}:{}] To LocMem[{}:{}] With LengthXn[{}] Use Channel[{}], Set "
        "CKE[{}:{:04x}], clearType[{}], lengthEn[{}]",
        waitCKEId, waitCKEMask, srcGSAId, srcXnId, dstGSAId, dstXnId, lengthXnId, channelId, setCKEId, setCKEMask,
        clearType, lengthEn);
    return HCCL_SUCCESS;
}

HcclResult TransformSyncCKEInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
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
        HCCL_ERROR("channel id[%hu] in rank[%u] is not found", channelId, rankId);
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
        HCCL_VM_DEBUG(
            "Get loc rank: {:d}, die: {:d} CKE: {:d}, value: {:d}, mask: {:x}", rankId, static_cast<uint32_t>(dieId),
            locCKEId, localCkeValue, locCKEMask);
        localCkeValue = localCkeValue & locCKEMask;
        CHK_RET(AllRankParamRecorder::Global()->GetCKE(rmtRankId, rmtDieId, rmtCKEId, ckeValue));
        HCCL_VM_DEBUG(
            "Get rmt rank: {:d}, die: {:d} CKE: {:d}, value: {:d}, mask: {:x}", rmtRankId,
            static_cast<uint32_t>(rmtDieId), rmtCKEId, ckeValue, locCKEMask);
        CHK_RET(AllRankParamRecorder::Global()->SetCKE(rmtRankId, rmtDieId, rmtCKEId, ckeValue | localCkeValue));
        HCCL_VM_DEBUG(
            "Set rank: {:d}, die: {:d} CKE: {:d}, value: {:d}, mask: {:x}", rmtRankId, static_cast<uint32_t>(rmtDieId),
            rmtCKEId, localCkeValue, locCKEMask);
    }

    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask, true));

    CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

    HCCL_VM_DEBUG(
        "Wait CKE[{}:{:04x}], Sync LocCKE[{}:{:04x}] To rmtCKE[{}:{:04x}] Use Channel[{}], Set "
        "CKE[{}:{:04x}], clearType[{}]",
        waitCKEId, waitCKEMask, locCKEId, locCKEMask, rmtCKEId, locCKEMask, channelId, setCKEId, setCKEMask, clearType);
    return HCCL_SUCCESS;
}

HcclResult TransformSyncXnInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
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
        HCCL_ERROR("channel id[%hu] in rank[%u] die[%u] is not found", channelId, rankId, static_cast<uint32_t>(dieId));
        return HCCL_E_INTERNAL;
    }

    RankId rmtRankId = g_allRankChannelInfo[rankId][dieId][channelId].dstRank;
    uint32_t rmtDieId = g_allRankChannelInfo[rankId][dieId][channelId].remoteDieId;

    uint64_t localXnValue = 0;
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, locXnId, localXnValue));
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
        "Wait CKE[{}:{:04x}], Sync locXnId[{}] To rmtXnId[{}] Use Channel[{}], Set rmtCKE[{}:{:04x}], Set "
        "CKE[{}:{:04x}], clearType[{}]",
        waitCKEId, waitCKEMask, locXnId, rmtXnId, channelId, setRmtCKEId, setRmtCKEMask, setCKEId, setCKEMask,
        clearType);
    return HCCL_SUCCESS;
}

HcclResult TransformJumpInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
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
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, conditionXnId, conditionValue));
    if (conditionValue != expectData) {
        uint64_t nextInsIdx = 0;
        CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, dstInstrXnId, nextInsIdx));
        curCcuTask->microCodePosInQue[queId] = nextInsIdx - curCcuTask->startInstrIdInQue[queId];
    }

    HCCL_VM_DEBUG(
        "When conditionXn[{}][{}] not equal to expectData[{}], Jump To InstrIdXn[{}]", conditionXnId, conditionValue,
        expectData, dstInstrXnId);
    return HCCL_SUCCESS;
}

HcclResult TransformTransLocMemToLocMSInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
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
    AllRankParamRecorder::Global()->GetGSA(rankId, dieId, locGSAId, locMemAddr);
    locMemAddr = UpdateGSAValue(locMemAddr, loopGroupParam);
    uint64_t len = 0;
    AllRankParamRecorder::Global()->GetXn(rankId, dieId, lengthXnId, len);
    CHK_PRT_RET(len == 0, HCCL_ERROR("The size of data transfer is 0."), HCCL_E_INTERNAL);

    DataSlice srcSlice;
    DataSlice dstSlice;
    // todo: 后续插件通过读取vm的输出文件，重建memlayout的地址类型信息
    CHK_RET(StorageManager::GetInstance().GetSlice(locMemAddr, len, srcSlice));
    CHK_RET(GenSliceFromMs(locMSId, len, dstSlice));

    AddLocalCopy(rankId, queId, curCcuTask, srcSlice, dstSlice);

    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask));

    CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

    HCCL_VM_DEBUG(
        "Wait CKE[{}:{:04x}], Trans LocMem[{}:{}] To LocMS[{}:{}] With LengthXn[{}] Use Channel[{}], Set "
        "CKE[{}:{:04x}], clearType[{}], lengthEn[{}]",
        waitCKEId, waitCKEMask, locGSAId, locXnId, locMSId / 0x8000, locMSId % 0x8000, lengthXnId, channelId, setCKEId,
        setCKEMask, clearType, lengthEn);

    return HCCL_SUCCESS;
}

HcclResult TransformTransLocMSToLocMemInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
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
    AllRankParamRecorder::Global()->GetXn(rankId, dieId, lengthXnId, len);
    CHK_PRT_RET(len == 0, HCCL_ERROR("The size of data transfer is 0."), HCCL_E_INTERNAL);
    DataSlice srcSlice;
    CHK_RET(GenSliceFromMs(locMSId, len, srcSlice));

    uint64_t locMemAddr = 0x0;
    AllRankParamRecorder::Global()->GetGSA(rankId, dieId, locGSAId, locMemAddr);
    locMemAddr = UpdateGSAValue(locMemAddr, loopGroupParam);
    DataSlice dstSlice;
    CHK_RET(StorageManager::GetInstance().GetSlice(locMemAddr, len, dstSlice));

    AddLocalCopy(rankId, queId, curCcuTask, srcSlice, dstSlice);

    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask));

    CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

    HCCL_VM_DEBUG(
        "Wait CKE[{}:{:04x}], Trans LocMS[{}:{}] To LocMem[{}:{}] With LengthXn[{}] Use Channel[{}], Set "
        "CKE[{}:{:04x}], clearType[{}], lengthEn[{}]",
        waitCKEId, waitCKEMask, locMSId / 0x8000, locMSId % 0x8000, locGSAId, locXnId, lengthXnId, channelId, setCKEId,
        setCKEMask, clearType, lengthEn);

    return HCCL_SUCCESS;
}

HcclResult TransformTransLocMSToLocMSInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
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
    AllRankParamRecorder::Global()->GetXn(rankId, dieId, lengthXnId, len);
    CHK_PRT_RET(len == 0, HCCL_ERROR("The size of data transfer is 0."), HCCL_E_INTERNAL);
    DataSlice srcSlice;
    CHK_RET(GenSliceFromMs(srcMSId, len, srcSlice));
    DataSlice dstSlice;
    CHK_RET(GenSliceFromMs(dstMSId, len, srcSlice));

    AddLocalCopy(rankId, queId, curCcuTask, srcSlice, dstSlice);

    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask));

    CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

    HCCL_VM_DEBUG(
        "Wait CKE[{}:{:04x}], Trans LocMS[{}:{}] To LocMS[{}:{}] With LengthXn[{}] Use Channel[{}], "
        "Set CKE[{}:{:04x}], clearType[{}], lengthEn[{}]",
        waitCKEId, waitCKEMask, srcMSId / 0x8000, srcMSId % 0x8000, dstMSId / 0x8000, dstMSId % 0x8000, lengthXnId,
        channelId, setCKEId, setCKEMask, clearType, lengthEn);

    return HCCL_SUCCESS;
}

HcclResult TransformTransLocMSToRmtMemInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
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
    AllRankParamRecorder::Global()->GetXn(rankId, dieId, lengthXnId, len);
    CHK_PRT_RET(len == 0, HCCL_ERROR("The size of data transfer is 0."), HCCL_E_INTERNAL);
    DataSlice srcSlice;
    CHK_RET(GenSliceFromMs(locMSId, len, srcSlice));

    uint64_t rmtMemAddr = 0x0;
    AllRankParamRecorder::Global()->GetGSA(rankId, dieId, rmtGSAId, rmtMemAddr);
    rmtMemAddr = UpdateGSAValue(rmtMemAddr, loopGroupParam);
    DataSlice dstSlice;
    // todo: 后续插件通过读取vm的输出文件，重建memlayout的地址类型信息
    CHK_RET(StorageManager::GetInstance().GetSlice(rmtMemAddr, len, dstSlice));

    RankId rmtRankId = g_allRankChannelInfo[rankId][dieId][channelId].dstRank;
    AddWrite(rankId, rmtRankId, queId, curCcuTask, srcSlice, dstSlice);

    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask));

    CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

    HCCL_VM_DEBUG(
        "Wait CKE[{}:{:04x}], Trans LocMS[{}:{}] To RmtMem[{}:{}] With LengthXn[{}] Use Channel[{}], Set "
        "CKE[{}:{:04x}], clearType[{}], lengthEn[{}]",
        waitCKEId, waitCKEMask, locMSId / 0x8000, locMSId % 0x8000, rmtGSAId, rmtXnId, lengthXnId, channelId, setCKEId,
        setCKEMask, clearType, lengthEn);

    return HCCL_SUCCESS;
}

HcclResult TransformTransLocMSToRmtMSInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
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
    AllRankParamRecorder::Global()->GetXn(rankId, dieId, lengthXnId, len);
    CHK_PRT_RET(len == 0, HCCL_ERROR("The size of data transfer is 0."), HCCL_E_INTERNAL);

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
        "Wait CKE[{}:{:04x}], Trans LocMS[{}:{}] To RmtMS[{}:{}] With LengthXn[{}] Use Channel[{}], Set "
        "RmtCKE[{}:{:04x}], Set CKE[{}:{:04x}], clearType[{}], lengthEn[{}]",
        waitCKEId, waitCKEMask, locMSId / 0x8000, locMSId % 0x8000, rmtMSId / 0x8000, rmtMSId % 0x8000, lengthXnId,
        channelId, setRmtCKEId, setRmtCKEMask, setCKEId, setCKEMask, clearType, lengthEn);
    return HCCL_SUCCESS;
}

HcclResult TransformTransRmtMemToLocMSInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
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
    AllRankParamRecorder::Global()->GetGSA(rankId, dieId, rmtGSAId, rmtAddr);
    rmtAddr = UpdateGSAValue(rmtAddr, loopGroupParam);
    uint64_t len = 0;
    AllRankParamRecorder::Global()->GetXn(rankId, dieId, lengthXnId, len);
    CHK_PRT_RET(len == 0, HCCL_ERROR("The size of data transfer is 0."), HCCL_E_INTERNAL);

    DataSlice srcSlice;
    DataSlice dstSlice;
    // todo: 后续插件通过读取vm的输出文件，重建memlayout的地址类型信息
    CHK_RET(StorageManager::GetInstance().GetSlice(rmtAddr, len, srcSlice));
    CHK_RET(GenSliceFromMs(locMSId, len, dstSlice));

    // 获取远端rankId与dieId
    if (g_allRankChannelInfo[rankId][dieId].count(channelId) == 0) {
        HCCL_ERROR("channel id[%hu] in rank[%u] is not found", channelId, rankId);
        return HCCL_E_INTERNAL;
    }

    RankId rmtRankId = g_allRankChannelInfo[rankId][dieId][channelId].dstRank;
    AddRead(rankId, rmtRankId, queId, curCcuTask, srcSlice, dstSlice);

    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask));

    CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

    HCCL_VM_DEBUG(
        "Wait CKE[{}:{:04x}], Trans RmtMem[{}:{}] To LocMS[{}:{}] With LengthXn[{}] Use Channel[{}], Set "
        "CKE[{}:{:04x}], clearType[{}], lengthEn[{}]",
        waitCKEId, waitCKEMask, rmtGSAId, rmtXnId, locMSId / 0x8000, locMSId % 0x8000, lengthXnId, channelId, setCKEId,
        setCKEMask, clearType, lengthEn);
    return HCCL_SUCCESS;
}

HcclResult TransformTransRmtMSToLocMemInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
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
    AllRankParamRecorder::Global()->GetXn(rankId, dieId, lengthXnId, len);
    CHK_PRT_RET(len == 0, HCCL_ERROR("The size of data transfer is 0."), HCCL_E_INTERNAL);

    DataSlice srcSlice;
    CHK_RET(GenSliceFromMs(rmtMSId, len, srcSlice));

    uint64_t localAddr = 0x0;
    AllRankParamRecorder::Global()->GetGSA(rankId, dieId, locGSAId, localAddr);
    localAddr = UpdateGSAValue(localAddr, loopGroupParam);
    DataSlice dstSlice;
    CHK_RET(StorageManager::GetInstance().GetSlice(localAddr, len, dstSlice));

    RankId rmtRankId = g_allRankChannelInfo[rankId][dieId][channelId].dstRank;
    AddRead(rankId, rmtRankId, queId, curCcuTask, srcSlice, dstSlice);

    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask));

    CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

    HCCL_VM_DEBUG(
        "Wait CKE[{}:{:04x}], Trans RmtMS[{}:{}] To LocMem[{}:{}] With LengthXn[{}] Use Channel[{}], Set "
        "CKE[{}:{:04x}], clearType[{}], lengthEn[{}]",
        waitCKEId, waitCKEMask, rmtMSId / 0x8000, rmtMSId % 0x8000, locGSAId, locXnId, lengthXnId, channelId, setCKEId,
        setCKEMask, clearType, lengthEn);
    return HCCL_SUCCESS;
}

HcclResult TransformTransRmtMSToLocMSInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
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
    AllRankParamRecorder::Global()->GetXn(rankId, dieId, lengthXnId, len);
    CHK_PRT_RET(len == 0, HCCL_ERROR("The size of data transfer is 0."), HCCL_E_INTERNAL);

    DataSlice srcSlice;
    CHK_RET(GenSliceFromMs(rmtMSId, len, srcSlice));
    DataSlice dstSlice;
    CHK_RET(GenSliceFromMs(locMSId, len, dstSlice));

    RankId rmtRankId = g_allRankChannelInfo[rankId][dieId][channelId].dstRank;
    AddRead(rankId, rmtRankId, queId, curCcuTask, srcSlice, dstSlice);

    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask));

    CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

    HCCL_VM_DEBUG(
        "Wait CKE[{}:{:04x}], Trans RmtMS[{}:{}] To LocMS[{}:{}] With LengthXn[{}] Use Channel[{}], "
        "Set CKE[{}:{:04x}], clearType[{}], lengthEn[{}]",
        waitCKEId, waitCKEMask, rmtMSId / 0x8000, rmtMSId % 0x8000, locMSId / 0x8000, locMSId % 0x8000, lengthXnId,
        channelId, setCKEId, setCKEMask, clearType, lengthEn);
    return HCCL_SUCCESS;
}

// todo：这边待处理reduce信息
HcclResult TransformTransRmtMemToLocMemInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
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
    AllRankParamRecorder::Global()->GetXn(rankId, dieId, lengthXnId, len);
    CHK_PRT_RET(len == 0, HCCL_ERROR("The size of data transfer is 0."), HCCL_E_INTERNAL);
    CHK_PRT_RET(
        len > UB_MAX_SIZE, HCCL_ERROR("The size of data transfer is more than max transport size of UB."),
        HCCL_E_INTERNAL);

    uint64_t localAddr = 0x0;
    uint64_t rmtAddr = 0x0;
    AllRankParamRecorder::Global()->GetGSA(rankId, dieId, locGSAId, localAddr);
    AllRankParamRecorder::Global()->GetGSA(rankId, dieId, rmtGSAId, rmtAddr);

    DataSlice srcSlice;
    RankId rId;
    CHK_RET(StorageManager::GetInstance().GetSlice(rmtAddr, len, srcSlice, &rId));
    DataSlice dstSlice;
    CHK_RET(StorageManager::GetInstance().GetSlice(localAddr, len, dstSlice));

    RankId rmtRankId = g_allRankChannelInfo[rankId][dieId][channelId].dstRank;
    HCCL_VM_DEBUG("curRank= {}, dstRank= {}", rankId, rmtRankId);
    // 说明transportId写错了
    if (rId != rmtRankId) {
        HCCL_ERROR(
            "remoteRankId calculated by rmtAddr and channelId %u is not equal,"
            "remoteId is %d, but the addr is in rank %d",
            channelId, rmtRankId, rId);
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
            AddReadReduce(rankId, rmtRankId, queId, curCcuTask, srcSlice, dstSlice, checkerDataType, checkerReduceOp);
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
        "Wait CKE[{}:{:04x}], Trans RmtMem[{}:{}] To LocMem[{}:{}] With LengthXn[{}] Use Channel[{}], Set "
        "CKE[{}:{:04x}], clearType[{}], lengthEn[{}], DataType[{}], ReduceType[{}] reduceEn[{}]",
        waitCKEId, waitCKEMask, rmtGSAId, rmtXnId, locGSAId, locXnId, lengthXnId, channelId, setCKEId, setCKEMask,
        clearType, lengthEn, reduceDataType, reduceOpCode, reduceEn);

    return HCCL_SUCCESS;
}

// 处理Loop指令
HcclResult ProcessLoopIns(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
    LoopGroupParam* loopGroupParam, uint32_t loopGroupIdx);

HcclResult TransformLoopGroupInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
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
    AllRankParamRecorder::Global()->GetXn(rankId, dieId, xnId, loopGroupParam.loopGroupXn.value);
    AllRankParamRecorder::Global()->GetXn(rankId, dieId, xmId, loopGroupParam.loopGroupXm.value);

    hcomm::CcuRep::CcuInstrInfo& microCodeQue = curCcuTask->instrInfo[queId];
    auto loopGroupIdx = curCcuTask->loopGroupIdx++;
    uint64_t loopCnt = loopGroupParam.loopGroupXn.loopInsCnt;
    HCCL_VM_DEBUG(
        "loop cnt= {}, loop offset= {}, expand cnt= {}", loopCnt,
        static_cast<uint64_t>(loopGroupParam.loopGroupXn.expandOffset),
        static_cast<uint64_t>(loopGroupParam.loopGroupXn.expandCnt));

    for (u32 curLoopIdx = 0; curLoopIdx < loopCnt; curLoopIdx++) {
        loopGroupParam.curLoopIdx = curLoopIdx;
        // 计算当前Loop指令的位置
        u32 insPos = startLoopInstrId + curLoopIdx - curCcuTask->startInstrIdInQue[queId];
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
        "LoopGroup From startLoopInstrId[{}] with loopGroupXn[{}], offsetXn[{}] and highPerfModeEn[{}]",
        startLoopInstrId, xnId, xmId, highPerfModeEn);
    return HCCL_SUCCESS;
}

HcclResult TransformLoopInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
    LoopGroupParam* loopGroupParam)
{
    uint16_t startInstrId = instr->v1.loop.startInstrId;
    uint16_t endInstrId = instr->v1.loop.endInstrId;
    uint16_t xnId = instr->v1.loop.xnId;
    HCCL_ERROR("Loop From startInstrId[%u] to endInstrId[%u] with loopXn[%u]", startInstrId, endInstrId, xnId);
    // 当前Loop都需要通过LoopGoup来触发，暂不支持单独解析Loop命令
    return HCCL_E_INTERNAL;
}

HcclResult TransformClearCKEInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
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
        HCCL_ERROR("do not support Clear CKE[%u:%04x]", clearCKEId, clearMask);
        return HCCL_E_INTERNAL;
    }

    CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

    HCCL_VM_DEBUG(
        "Wait CKE[{}:{:04x}], Clear CKE[{}:{:04x}], clearType[{}]", waitCKEId, waitCKEMask, clearCKEId, clearMask,
        clearType);
    return HCCL_SUCCESS;
}

HcclResult TransformSyncGSAInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
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
        HCCL_ERROR("channel id[%hu] in rank[%u] is not found", channelId, rankId);
        return HCCL_E_INTERNAL;
    }

    RankId rmtRankId = g_allRankChannelInfo[rankId][dieId][channelId].dstRank;
    uint32_t rmtDieId = g_allRankChannelInfo[rankId][dieId][channelId].remoteDieId;

    uint64_t localGSAValue = 0;
    CHK_RET(AllRankParamRecorder::Global()->GetGSA(rankId, dieId, locGSAId, localGSAValue));
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
        "Wait CKE[{}:{:04x}], Sync locGSAId[{}] To rmtGSAId[{}] Use Channel[{}], Set rmtCKE[{}:{:04x}], Set "
        "CKE[{}:{:04x}], clearType[{}]",
        waitCKEId, waitCKEMask, locGSAId, rmtGSAId, channelId, setRmtCKEId, setRmtCKEMask, setCKEId, setCKEMask,
        clearType);

    return HCCL_SUCCESS;
}

HcclResult GetHcclDataTypeFromCCUDataType(uint16_t ccuDataType, uint16_t ccuReduceType, DataType& dataType)
{
    static std::map<uint16_t, DataType> ccuSumDataTypeMap = {
        {0, DataType::FP32},    {1, DataType::FP16}, {2, DataType::BFP16}, {3, DataType::HIF8},  {4, DataType::FP8E4M3},
        {5, DataType::FP8E5M2}, {6, DataType::INT8}, {7, DataType::UINT8}, {8, DataType::INT16}, {9, DataType::INT32},
    };

    static std::map<uint16_t, DataType> ccuMaxMinDataTypeMap = {
        {0, DataType::FP32},  {1, DataType::FP16},  {2, DataType::BFP16}, {6, DataType::INT8},
        {7, DataType::UINT8}, {8, DataType::INT16}, {9, DataType::INT32},
    };

    if (ccuReduceType == CcuRep::CCU_REDUCE_SUM) {
        if (ccuSumDataTypeMap.find(ccuDataType) == ccuSumDataTypeMap.end()) {
            HCCL_ERROR("Unsupported CCUDataType[%hu] for Ccu SUM", ccuDataType);
            return HCCL_E_INTERNAL;
        }
        dataType = ccuSumDataTypeMap[ccuDataType];
    } else if (ccuReduceType == CcuRep::CCU_REDUCE_MAX || ccuReduceType == CcuRep::CCU_REDUCE_MIN) {
        if (ccuMaxMinDataTypeMap.find(ccuDataType) == ccuMaxMinDataTypeMap.end()) {
            HCCL_ERROR("Unsupported CCUDataType[%hu] for Ccu MAX/MIN", ccuDataType);
            return HCCL_E_INTERNAL;
        }
        dataType = ccuMaxMinDataTypeMap[dataType];
    } else {
        HCCL_ERROR("Unsupported ccuReduceType[%hu]", ccuReduceType);
        return HCCL_E_INTERNAL;
    }

    return HCCL_SUCCESS;
}

HcclResult TransformAddInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
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
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, lengthId, len));
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
        "Wait CKE[{}:{:04x}], Add {} with Count[{}], DataType[{}] and CastEn[{}], Set CKE[{}:{:04x}], clearType[{}]",
        waitCKEId, waitCKEMask, ParseMSList(instr), count, dataType, castEn, setCKEId, setCKEMask, clearType);

    return HCCL_SUCCESS;
}

HcclResult TransformMaxInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
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
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, lengthId, len));
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
        "Wait CKE[{}:{:04x}], Max {} with Count[{}], DataType[{}] and CastEn[{}], Set CKE[{}:{:04x}], clearType[{}]",
        waitCKEId, waitCKEMask, ParseMSList(instr), count, dataType, "N/A", setCKEId, setCKEMask, clearType);

    return HCCL_SUCCESS;
}

HcclResult TransformMinInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
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
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, lengthId, len));
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
        "Wait CKE[{}:{:04x}], Min {} with Count[{}], DataType[{}] and CastEn[{}], Set CKE[{}:{:04x}], clearType[{}]",
        waitCKEId, waitCKEMask, HcclSim::ParseMSList(instr), count, dataType, "N/A", setCKEId, setCKEMask, clearType);
    return HCCL_SUCCESS;
}

// 以下是一些辅助函数，用于转换函数使用
HcclResult TransformLoadSqeArgsToGSAInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
    return TransformLoadSqeArgsToGSAInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformLoadSqeArgsToXnInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
    return TransformLoadSqeArgsToXnInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformLoadImdToGSAInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
    return TransformLoadImdToGSAInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformLoadImdToXnInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
    return TransformLoadImdToXnInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformLoadGSAXnInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
    return TransformLoadGSAXnInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformLoadGSAGSAInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
    return TransformLoadGSAGSAInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformLoadXXInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
    return TransformLoadXXInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformLoopGroupInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
    return TransformLoopGroupInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformLoopInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
    return TransformLoopInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformSetCKEInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
    return TransformSetCKEInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformClearCKEInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
    return TransformClearCKEInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformJumpInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
    return TransformJumpInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformTransLocMemToLocMSInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
    return TransformTransLocMemToLocMSInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformTransRmtMemToLocMSInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
    return TransformTransRmtMemToLocMSInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformTransLocMSToLocMemInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
    return TransformTransLocMSToLocMemInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformTransLocMSToRmtMemInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
    return TransformTransLocMSToRmtMemInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}
HcclResult TransformTransRmtMSToLocMemInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
    return TransformTransRmtMSToLocMemInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformTransLocMSToLocMSInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
    return TransformTransLocMSToLocMSInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformTransRmtMSToLocMSInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
    return TransformTransRmtMSToLocMSInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformTransLocMSToRmtMSInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
    return TransformTransLocMSToRmtMSInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformTransRmtMemToLocMemInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
    return TransformTransRmtMemToLocMemInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformTransLocMemToRmtMemInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
    return TransformTransLocMemToRmtMemInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformTransLocMemToLocMemInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
    return TransformTransLocMemToLocMemInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformSyncCKEInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
    return TransformSyncCKEInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformSyncGSAInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
    return TransformSyncGSAInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformSyncXnInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
    return TransformSyncXnInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformAddInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
    return TransformAddInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformMaxInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParam*>(loopParam);
    return TransformMaxInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformMinInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
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
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
    LoopGroupParam* loopGroupParam, uint32_t loopGroupIdx)
{
    uint16_t startInstrId = instr->v1.loop.startInstrId;
    uint16_t endInstrId = instr->v1.loop.endInstrId;
    uint16_t xnId = instr->v1.loop.xnId;

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId;
    curCcuTask->GetDieId(queId, dieId);

    uint64_t value = 0;
    AllRankParamRecorder::Global()->GetXn(rankId, dieId, xnId, value);
    LoopXm xm{};
    xm.value = value;
    loopGroupParam->loopXms.push_back(xm);

    HCCL_VM_DEBUG("debug...{}, {}", loopGroupIdx, curCcuTask->loopGroupInfo_.size());
    auto loopIdx = curCcuTask->loopIdx[loopGroupIdx]++;
    auto loopStart = AddLoopStartTask(queId, loopIdx, loopGroupIdx, curCcuTask); // loop前新增loopStart标记节点

    HCCL_VM_DEBUG("loop cnt = {}", static_cast<uint64_t>(xm.loopCnt));
    hcomm::CcuRep::CcuInstrInfo& microCodeQue = curCcuTask->instrInfo[queId];
    for (u32 curLoopCnt = 0; curLoopCnt < xm.loopCnt; curLoopCnt++) {
        loopGroupParam->curLoopCnt = curLoopCnt;
        HCCL_VM_DEBUG("loop cnt = {}, cur loop= {}", static_cast<uint64_t>(xm.loopCnt), curLoopCnt);
        for (uint16_t insId = startInstrId; insId <= endInstrId; insId++) {
            // 获取当前要处理的指令
            const CcuRep::CcuInstr* instr = &microCodeQue.instrVec[insId - curCcuTask->startInstrIdInQue[queId]];
            HCCL_VM_DEBUG("Current process instruction id = {}", insId);
            if (transformA5InstrSqeMap.count(instr->header.header) == 0) {
                HCCL_ERROR("ins type not supported %hu", instr->header.header);
                return HCCL_E_INTERNAL;
            }
            CHK_RET(transformA5InstrSqeMap[instr->header.header](instr, curCcuTask, queId, isContinue, loopGroupParam));
        }
    }
    auto loopEnd = AddLoopEndTask(queId, loopIdx, loopGroupIdx, curCcuTask); // loop后新增loopEnd标记节点

    HCCL_VM_DEBUG("Loop From startInstrId[{}] to endInstrId[{}] with loopXn[{}]", startInstrId, endInstrId, xnId);
    return HCCL_SUCCESS;
}

InstructMapA5::InstructMapA5() { transformInstrSqeMap = transformA5InstrSqeMap; }

HcclResult InstructMapA5::Transform(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto it = transformInstrSqeMap.find(instr->header.header);
    if (it == transformInstrSqeMap.end()) {
        HCCL_ERROR("[A5] Unsupported: 0x%04x", instr->header.header);
        return HCCL_E_INTERNAL;
    }
    auto* param = static_cast<LoopGroupParam*>(loopParam);
    return it->second(instr, curCcuTask, queId, isContinue, param);
}
} // namespace HcclSim
