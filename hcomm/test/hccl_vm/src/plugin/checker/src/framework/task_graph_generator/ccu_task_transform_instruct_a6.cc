/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ccu_all_rank_param_recorder.h"
#include "ccu_task_common.h"
#include "ccu_task_transform.h"
#include "ccu_task_transform_instruct_common.h"
#include "sim_log.h"
#include "storage_manager.h"
#include "task_graph_generator.h"
#include "type_conversion.h"

using namespace HcclSim;

extern std::map<RankId, std::map<u32, ChannelsPerDie>> g_allRankChannelInfo;
namespace {
// 18
constexpr uint16_t A6_LOADSQEARGSTOXN_CODE = 0x1;
constexpr uint16_t A6_LOADIMDTOX_CODE = 0x2;
constexpr uint16_t A6_LOADX_CODE = 0x6;
constexpr uint16_t A6_STOREX_CODE = 0x7;
constexpr uint16_t A6_CLEARX_CODE = 0x8;
constexpr uint16_t A6_NOP_CODE = 0x9;
constexpr uint16_t A6_LOAD_CODE = 0xA;
constexpr uint16_t A6_STORE_CODE = 0xB;
constexpr uint16_t A6_ADD_CODE = 0xD;
constexpr uint16_t A6_SUB_CODE = 0xE;
constexpr uint16_t A6_MUL_CODE = 0xF;
constexpr uint16_t A6_AND_CODE = 0x10;
constexpr uint16_t A6_OR_CODE = 0x11;
constexpr uint16_t A6_NOT_CODE = 0x12;
constexpr uint16_t A6_XOR_CODE = 0x13;
constexpr uint16_t A6_SHL_CODE = 0x14;
constexpr uint16_t A6_SHR_CODE = 0x15;
constexpr uint16_t A6_POPCNT_CODE = 0x16;

// 7
constexpr uint16_t A6_LOOP_CODE = 0x0;
constexpr uint16_t A6_LOOPGROUP_CODE = 0x1;
constexpr uint16_t A6_SETCKBIT_CODE = 0x2;
constexpr uint16_t A6_CLEARCKE_CODE = 0x4;
constexpr uint16_t A6_JMP_CODE = 0x5;
constexpr uint16_t A6_WAIT_CODE = 0x7;
constexpr uint16_t A6_FENCE_CODE = 0x8;

// 7
constexpr uint16_t A6_TRANSLOCMEMTOLOCMS_CODE = 0x0;
constexpr uint16_t A6_TRANSLOCMSTOLOCMEM_CODE = 0x2;
constexpr uint16_t A6_TRANSLOCMSTOLOCMS_CODE = 0x5;
constexpr uint16_t A6_TRANSLOCMEMTOLOCMEM_CODE = 0x6;
constexpr uint16_t A6_TRANSMEM_CODE = 0x10;
constexpr uint16_t A6_SYNCXNWTX_CODE = 0xd;
constexpr uint16_t A6_SYNCATX_CODE = 0xe;

// 3
constexpr uint16_t A6_REDUCEADD_CODE = 0x0;
constexpr uint16_t A6_REDUCEMAX_CODE = 0x1;
constexpr uint16_t A6_REDUCEMIN_CODE = 0x2;
constexpr uint16_t A6_LOOP_BOARD_ENTRY = 1024; // LOOP指令的起始地址
constexpr uint64_t UB_MAX_SIZE = 256 * 1024 * 1024;
constexpr uint16_t MAX_LOADX_STOREX_ID_NUM = 16383; // loadx/storeX 指令S使用，最大寄存器ID个数
constexpr uint32_t JUMP_INSTR_MAX = 0x10000;

static std::map<uint16_t, uint16_t> ccuReduceTypeMap
    = {{10, CcuRep::CCU_REDUCE_SUM}, {9, CcuRep::CCU_REDUCE_MIN}, {8, CcuRep::CCU_REDUCE_MAX}};

// loop指令使用的结构体，第n个通用寄存器，高32位保留，低32位表示搬运用到的GSA地址
union LoopXnA6 {
    uint64_t value{0};
    struct {
        uint64_t loopIntrGSA : 32;
        uint64_t reserved : 32;
    };
};

// loop指令使用的结构体，第m个通用寄存器，Loop需要执行循环的次数，高21位保留
union LoopXmA6 {
    uint64_t value{0};
    struct {
        uint64_t loopIterCnt : 13;
        uint64_t reserved : 51;
    };
};

// loop指令使用的结构体，第p个通用寄存器，该循环使用的contextId，高55位保留
union LoopXpA6 {
    uint64_t value{0};
    struct {
        uint64_t loopCtxId : 9;
        uint64_t reserved : 55;
    };
};

// loop结构体，包括了Xn、Xm、Xp三个通用寄存器
struct LoopA6 {
    LoopXnA6 loopXn{};
    LoopXmA6 loopXm{};
    LoopXpA6 loopXp{};
};

// loopGroup指令使用的结构体，第n个通用寄存器，loop指令个数，高10位表示展开次数，低9位表示循环次数，高35位保留
union LoopGroupXnA6 {
    uint64_t value{0};
    struct {
        uint64_t loopInsCnt : 10;
        uint64_t expandOffset : 9;
        uint64_t expandCnt : 9;
        uint64_t reservedHigh : 35;
    };
};

// loopGroup指令使用的结构体，第m个通用寄存器，存储CK的offset、MS的offset、GSA的offset，高11位保留
union LoopGroupXmA6 {
    uint64_t value{0};
    struct {
        uint64_t ckOffset : 10;
        uint64_t msOffset : 11;
        uint64_t gsaOffset : 32;
        uint64_t reserved : 11;
    };
};

// loopGroup指令使用的结构体，第p个通用寄存器，存储Xnid的偏移量，高32位保留
union LoopGroupXpA6 {
    uint64_t value{0};
    struct {
        uint64_t xnIdOffset : 32;
        uint64_t reserved : 32;
    };
};

// loopGroup指令使用的结构体，使用包括Xm、Xp、Xn三个通用寄存器以及loop循环的内容
struct LoopGroupParamA6 {
    std::vector<LoopA6> loopXms;
    LoopGroupXnA6 loopGroupXn;
    LoopGroupXmA6 loopGroupXm;
    LoopGroupXpA6 loopGroupXp;
    u32 curLoopIdx = 0;   // 表示当前处理第几个loop
    u32 curExpandCnt = 0; // 该loop被第几次展开
    u32 curLoopCnt = 0;   // 表示当前Loop第几次循环
};

HcclResult CheckLoopGroupNotSupport(LoopGroupParamA6* loopGroupParam)
{
    if (loopGroupParam != nullptr) {
        HCCL_ERROR("this ins do not support loop");
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

// 循环中CKE需要进行自动偏移
uint16_t UpdateCKEId(uint16_t CKEId, LoopGroupParamA6* loopGroupParam)
{
    // 不在循环中，不需要刷新
    if (loopGroupParam == nullptr) {
        return CKEId;
    }

    return UpdateId(
        CKEId, loopGroupParam->curLoopIdx, loopGroupParam->loopGroupXn.expandOffset,
        loopGroupParam->loopGroupXm.ckOffset, loopGroupParam->curExpandCnt);
}

// 循环中XN需要进行自动偏移
uint16_t UpdateXnId(uint16_t xnId, LoopGroupParamA6* loopGroupParam)
{
    // 不在循环中，不需要刷新
    if (loopGroupParam == nullptr) {
        return xnId;
    }

    return UpdateId(
        xnId, loopGroupParam->curLoopIdx, loopGroupParam->loopGroupXn.expandOffset,
        loopGroupParam->loopGroupXp.xnIdOffset, loopGroupParam->curExpandCnt);
}

// 循环中MS需要进行自动偏移
uint16_t UpdateMSId(uint16_t MSId, LoopGroupParamA6* loopGroupParam)
{
    // 不在循环中，不需要刷新
    if (loopGroupParam == nullptr) {
        return MSId;
    }

    return UpdateId(
        MSId, loopGroupParam->curLoopIdx, loopGroupParam->loopGroupXn.expandOffset,
        loopGroupParam->loopGroupXm.msOffset, loopGroupParam->curExpandCnt);
}

uint64_t UpdateAddressWithoutStride(uint64_t addr, LoopGroupParamA6* loopGroupParam)
{
    if (loopGroupParam == nullptr) {
        return addr;
    }
    LoopA6& xm = loopGroupParam->loopXms[loopGroupParam->curLoopIdx];
    return addr + loopGroupParam->curLoopCnt * xm.loopXn.loopIntrGSA;
}

uint64_t UpdateAddress(uint64_t addr, LoopGroupParamA6* loopGroupParam, uint16_t addrExpandCoef = 0)
{
    // 不在循环中，不需要刷新
    if (loopGroupParam == nullptr) {
        return addr;
    }
    LoopA6& xm = loopGroupParam->loopXms[loopGroupParam->curLoopIdx];

    if (loopGroupParam->curLoopIdx < loopGroupParam->loopGroupXn.expandOffset) {
        return (addr + loopGroupParam->curLoopCnt * xm.loopXn.loopIntrGSA) << addrExpandCoef;
    }

    return addr + ((loopGroupParam->curExpandCnt * loopGroupParam->loopGroupXm.gsaOffset) << addrExpandCoef)
           + ((loopGroupParam->curLoopCnt * xm.loopXn.loopIntrGSA) << addrExpandCoef);
}

inline uint16_t GetXnId(uint16_t xnIdField, LoopGroupParamA6* loopGroupParam)
{
    uint16_t xnIdMode = xnIdField & 0x8000;
    uint16_t xnIdOri = xnIdField & 0x7FFF;
    return (xnIdMode == 0) ? xnIdOri : UpdateXnId(xnIdOri, loopGroupParam);
}

// 判断g_allRankChannelInfo是否有数据
bool IsExistRemoteDieInfo(RankId rankId, u32 dieId, uint32_t channel)
{
    auto rankIt = g_allRankChannelInfo.find(rankId);
    if (rankIt == g_allRankChannelInfo.end()) {
        return false;
    }
    auto dieIt = rankIt->second.find(dieId);
    if (dieIt == rankIt->second.end()) {
        return false;
    }
    return dieIt->second.count(channel) > 0;
}

// 将mession Sqe终端额args信息写到第n个X寄存器中
HcclResult TransformLoadSqeArgsToXnInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
    LoopGroupParamA6* loopGroupParam)
{
    (void)isContinue;
    CHK_RET(CheckLoopGroupNotSupport(loopGroupParam));
    uint16_t sqeArgsId = instr->v2.loadSqeArgsToX.sqeArgsId;
    uint16_t xnId = instr->v2.loadSqeArgsToX.xnId;
    uint64_t argVal = 0;
    curCcuTask->GetSqe(queId, sqeArgsId, argVal);

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId{0};
    curCcuTask->GetDieId(queId, dieId);

    CHK_RET(AllRankParamRecorder::Global()->SetXn(rankId, dieId, xnId, argVal));
    uint16_t ckeId = UpdateCKEId(instr->v2.loadSqeArgsToX.setCKEId, loopGroupParam);
    uint16_t ckeMask = instr->v2.loadSqeArgsToX.setCKEMask;
    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, ckeId, ckeMask));
    HCCL_INFO("Load SqeArg[%u](%llu) to Xn[%u]", sqeArgsId, argVal, xnId);
    return HCCL_SUCCESS;
}

// 将立即数 写到第n个X寄存器中
HcclResult TransformLoadImdToXInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
    LoopGroupParamA6* loopGroupParam)
{
    (void)isContinue;
    uint16_t xnId = GetXnId(instr->v2.loadImdToX.xnId, loopGroupParam);
    uint64_t immediate = instr->v2.loadImdToX.immediate;

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId{0};
    curCcuTask->GetDieId(queId, dieId);

    CHK_RET(AllRankParamRecorder::Global()->SetXn(rankId, dieId, xnId, immediate));
    // cke设置
    uint16_t ckeId = UpdateCKEId(instr->v2.loadImdToX.setCKEId, loopGroupParam);
    uint16_t ckeMask = instr->v2.loadImdToX.setCKEMask;
    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, ckeId, ckeMask));
    HCCL_INFO("Load immediate[%llu] to Xn[%u]", immediate, xnId);
    return HCCL_SUCCESS;
}

// 判断XnId是否有效
HcclResult CheckXnValid(uint16_t xnId, uint16_t xnIdMin, uint16_t xnIdMax)
{
    if (xnId < xnIdMin || xnId > xnIdMax) {
        HCCL_ERROR("CheckXnValid ERROR,xnId = [%u], XnmIdMin = [%u], XnmIdMax = [%u]", xnId, xnIdMin, xnIdMax);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult GetXnValueAndCheck(RankId rankId, uint32_t dieId, uint16_t xnId, uint64_t& xnValue)
{
    CHK_RET(CheckXnValid(xnId, 0, MAX_LOADX_STOREX_ID_NUM));
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xnId, xnValue));
    return HCCL_SUCCESS;
}

HcclResult TransformLoadXInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
    LoopGroupParamA6* loopGroupParam)
{
    (void)isContinue;
    uint8_t mode = instr->v2.loadStoreX.oMode;

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId{0};
    curCcuTask->GetDieId(queId, dieId);

    uint16_t xsId = GetXnId(instr->v2.loadStoreX.xsId, loopGroupParam);
    uint64_t xsValue = 0;
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xsId, xsValue));

    uint64_t xdValue = 0;
    if (mode == 0) {
        // 计算公式：*Xd=*X(*Xs+立即数)+立即数（64位）
        uint16_t immedataSo = instr->v2.loadStoreX.xso;
        uint16_t immedataDo = instr->v2.loadStoreX.xdo;
        uint16_t xsTmpId = static_cast<uint16_t>(xsValue) + immedataSo;
        uint64_t xsTmpValue = 0;
        CHK_RET(GetXnValueAndCheck(rankId, dieId, xsTmpId, xsTmpValue));
        xdValue = xsTmpValue + immedataDo;
        HCCL_INFO("LoadX,Xd[%u] to Xs[%u],offsetSo[%u],offsetDo[%u]", xsTmpId, xsTmpId, immedataSo, immedataDo);
    } else if (mode == 1) {
        // 计算公式：*Xd=*X(*Xs+*Xso)+*Xdo
        uint16_t xsSoId = GetXnId(instr->v2.loadStoreX.xso, loopGroupParam);
        uint64_t xsSoValue = 0;
        CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xsSoId, xsSoValue));

        uint16_t xsDoId = GetXnId(instr->v2.loadStoreX.xdo, loopGroupParam);
        uint64_t xsDoValue = 0;
        CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xsDoId, xsDoValue));

        uint16_t xsTmpId = static_cast<uint16_t>(xsValue + xsSoValue);
        uint64_t xsTmpValue = 0;
        CHK_RET(GetXnValueAndCheck(rankId, dieId, xsTmpId, xsTmpValue));
        xdValue = xsTmpValue + xsDoValue;
    } else {
        HCCL_ERROR("TransformLoadXInstr ERROR,mode=[%u]", mode);
        return HCCL_E_PARA;
    }

    uint16_t xdId = GetXnId(instr->v2.loadStoreX.xdId, loopGroupParam);
    CHK_RET(AllRankParamRecorder::Global()->SetXn(rankId, dieId, xdId, xdValue));
    // cke设置
    uint16_t ckeId = UpdateCKEId(instr->v2.loadStoreX.setCKEId, loopGroupParam);
    uint16_t ckeMask = instr->v2.loadStoreX.setCKEMask;
    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, ckeId, ckeMask));
    return HCCL_SUCCESS;
}

// StoreX指令
HcclResult TransformStoreXInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
    LoopGroupParamA6* loopGroupParam)
{
    (void)isContinue;
    uint8_t mode = instr->v2.loadStoreX.oMode;

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId{0};
    curCcuTask->GetDieId(queId, dieId);

    uint16_t xsId = GetXnId(instr->v2.loadStoreX.xsId, loopGroupParam);
    uint64_t xsValue = 0;
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xsId, xsValue));

    uint16_t xdId = GetXnId(instr->v2.loadStoreX.xdId, loopGroupParam);
    uint64_t xdValue = 0;
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xdId, xdValue));

    uint16_t newId = 0;
    uint16_t newValue = 0;
    if (mode == 0) {
        // 计算公式：*X(*Xd + 立即数) =*Xs+*Xso
        uint16_t immedataSo = instr->v2.loadStoreX.xso;
        uint16_t immedataDo = instr->v2.loadStoreX.xdo;

        newId = static_cast<uint16_t>(xdValue) + immedataDo;
        newValue = xsValue + static_cast<uint64_t>(immedataSo);
    } else if (mode == 1) {
        // 计算公式：*X(*Xd+*Xdo) =*Xs+*Xso
        uint16_t xsDoId = GetXnId(instr->v2.loadStoreX.xdo, loopGroupParam);
        uint64_t xsDoValue = 0;
        CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xsDoId, xsDoValue));
        newId = static_cast<uint16_t>(xdValue + xsDoValue);

        uint16_t xsSoId = GetXnId(instr->v2.loadStoreX.xso, loopGroupParam);
        uint64_t xsSoValue = 0;
        CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xsSoId, xsSoValue));

        newValue = xsValue + xsSoValue;
    } else {
        HCCL_ERROR("TransformStoreXInstr ERROR,mode=[%u]", mode);
        return HCCL_E_PARA;
    }

    CHK_RET(CheckXnValid(newId, 0, MAX_LOADX_STOREX_ID_NUM));
    CHK_RET(AllRankParamRecorder::Global()->SetXn(rankId, dieId, newId, newValue));
    // cke设置
    uint16_t ckeId = UpdateCKEId(instr->v2.loadStoreX.setCKEId, loopGroupParam);
    uint16_t ckeMask = instr->v2.loadStoreX.setCKEMask;
    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, ckeId, ckeMask));
    return HCCL_SUCCESS;
}

// ClearX指令，对指定的X寄存器进行清零操作
HcclResult TransformClearXInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
    LoopGroupParamA6* loopGroupParam)
{
    (void)isContinue;
    // 清除范围为Xn~Xm  Xn的有效值？
    uint16_t xnOriId = instr->v2.clearX.xnId;
    uint16_t xnId = (instr->v2.clearX.xnIdMode == 0) ? xnOriId : UpdateXnId(xnOriId, loopGroupParam);
    uint16_t xmOriId = instr->v2.clearX.xmId;
    uint16_t xmId = (instr->v2.clearX.xmIdMode == 0) ? xmOriId : UpdateXnId(xmOriId, loopGroupParam);
    // 当xnId>xmId时，说明是清除范围为Xn~MAX_LOADX_STOREX_ID_NUM，xmId无效
    xmId = (xnId <= xmId) ? xmId : MAX_LOADX_STOREX_ID_NUM;
    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId{0};
    curCcuTask->GetDieId(queId, dieId);

    for (uint16_t i = xnId; i <= xmId; i++) {
        CHK_RET(AllRankParamRecorder::Global()->SetXn(rankId, dieId, i, 0));
    }
    // cke设置
    uint16_t ckeId = UpdateCKEId(instr->v2.clearX.setCKEId, loopGroupParam);
    uint16_t ckeMask = instr->v2.clearX.setCKEMask;
    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, ckeId, ckeMask));
    return HCCL_SUCCESS;
}

// Nop指令，空指令，用于排指令流水
HcclResult TransformNopInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
    LoopGroupParamA6* loopGroupParam)
{
    (void)isContinue;
    (void)instr;
    (void)curCcuTask;
    (void)queId;
    (void)loopGroupParam;
    return HCCL_SUCCESS;
}

// Load指令，将指定的内存地址的数据读到CCU内部的指定空间寄存器中,目前只支持Xn寄存器
HcclResult TransformLoadInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
    LoopGroupParamA6* loopGroupParam)
{
    (void)isContinue;
    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId{0};
    curCcuTask->GetDieId(queId, dieId);
    uint16_t dstType = instr->v2.load.dstType;
    if (dstType != 0) {
        // 目前只支持Xn寄存器
        HCCL_ERROR("TransformLoadInstr ERROR,dstType=[%u]", dstType);
        return HCCL_E_PARA;
    }
    uint16_t xdId = GetXnId(instr->v2.load.xdId, loopGroupParam);
    // 读取回来的数据放入空间的索引号
    uint64_t xdValue = 0;
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xdId, xdValue));
    uint16_t xsId = GetXnId(instr->v2.load.xsId, loopGroupParam);
    uint64_t xsStartAddr = 0;
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xsId, xsStartAddr));
    // 需要判断xsStartAddr是否8bye对齐
    if ((xsStartAddr & 0x7) != 0) {
        HCCL_ERROR("TransformLoadInstr ERROR,xsStartAddr = [%llu]", xsStartAddr);
        return HCCL_E_PARA;
    }
    uint16_t xlId = GetXnId(instr->v2.load.xlId, loopGroupParam);
    uint64_t length = 0;
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xlId, length));
    if ((length & 0x7) != 0) {
        HCCL_ERROR("TransformLoadInstr ERROR,length = [%llu]", length);
        return HCCL_E_PARA;
    }
    std::vector<uint64_t> entry;
    CHK_RET(AllRankParamRecorder::Global()->GetHBM(rankId, dieId, xsStartAddr, entry));
    if (length / 8 > entry.size()) {
        HCCL_ERROR("TransformLoadInstr ERROR,length = [%llu]", length);
        return HCCL_E_PARA;
    }
    for (size_t i = 0; i < length / 8; i++) {
        CHK_RET(AllRankParamRecorder::Global()->SetXn(rankId, dieId, xdValue++, entry[i]));
    }
    // cke设置
    uint16_t ckeId = UpdateCKEId(instr->v2.load.setCKEId, loopGroupParam);
    uint16_t ckeMask = instr->v2.load.setCKEMask;
    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, ckeId, ckeMask));
    return HCCL_SUCCESS;
}

// Store指令，将CCU内部的指定空间寄存器的数据写到指定的内存地址中
HcclResult TransformStoreInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
    LoopGroupParamA6* loopGroupParam)
{
    (void)isContinue;
    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId{0};
    curCcuTask->GetDieId(queId, dieId);
    uint16_t srcType = instr->v2.store.srcType;
    if (srcType != 0) {
        // 目前只支持Xn寄存器
        HCCL_ERROR("TransformStoreInstr ERROR,srcType=[%u]", srcType);
        return HCCL_E_PARA;
    }
    uint16_t xdId = GetXnId(instr->v2.store.xdId, loopGroupParam);
    // 读取回来的数据起始地址
    uint64_t xdValue = 0;
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xdId, xdValue));
    uint16_t xsId = GetXnId(instr->v2.store.xsId, loopGroupParam);
    uint64_t xsValue = 0;
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xsId, xsValue));
    // 需要判断xsStartAddr是否8bye对齐
    uint16_t xlId = GetXnId(instr->v2.store.xlId, loopGroupParam);
    uint64_t length = 0;
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xlId, length));
    if ((length & 0x7) != 0) {
        HCCL_ERROR("TransformStoreInstr ERROR,length = [%llu]", length);
        return HCCL_E_PARA;
    }
    std::vector<uint64_t> entry;
    for (size_t i = 0; i < length / 8; i++) {
        uint64_t xsValueTmp = 0;
        CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xsValue++, xsValueTmp));
        entry.push_back(xsValueTmp);
    }
    CHK_RET(AllRankParamRecorder::Global()->SetHBM(rankId, dieId, xdValue, entry));

    // cke设置
    uint16_t ckeId = UpdateCKEId(instr->v2.store.setCKEId, loopGroupParam);
    uint16_t ckeMask = instr->v2.store.setCKEMask;
    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, ckeId, ckeMask));

    return HCCL_SUCCESS;
}

// Add指令，实现算数“+”
HcclResult TransformAddInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
    LoopGroupParamA6* loopGroupParam)
{
    uint16_t xnId = GetXnId(instr->v2.operate.xnId, loopGroupParam);
    uint16_t xdId = GetXnId(instr->v2.operate.xdId, loopGroupParam);
    uint16_t parMode = instr->v2.operate.parMode;
    uint16_t ckeId = UpdateCKEId(instr->v2.operate.setCKEId, loopGroupParam);
    uint16_t ckeMask = instr->v2.operate.setCKEMask;

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId{0};
    curCcuTask->GetDieId(queId, dieId);

    uint64_t xnValue = 0;
    uint64_t xdValue = 0;
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xnId, xnValue));
    // Xd = Xn + immed
    if (parMode == 0) {
        xdValue = xnValue + instr->v2.operate.xmId;
        HCCL_INFO("Add Xn[%u](%llu) + immed[%u] to Xd[%u](%llu)", xnId, xnValue, instr->v2.operate.xmId, xdId, xdValue);
    } else {
        // Xd = Xn + Xm
        uint16_t xmId = GetXnId(instr->v2.operate.xmId, loopGroupParam);
        uint64_t xmValue = 0;
        CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xmId, xmValue));
        xdValue = xnValue + xmValue;
        HCCL_INFO("Add Xn[%u](%llu) + Xm[%u](%llu) to Xd[%u](%llu)", xnId, xnValue, xmId, xmValue, xdId, xdValue);
    }

    CHK_RET(AllRankParamRecorder::Global()->SetXn(rankId, dieId, xdId, xdValue));
    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, ckeId, ckeMask));
    return HCCL_SUCCESS;
}

// Sub指令，实现算数“-”
HcclResult TransformSubInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
    LoopGroupParamA6* loopGroupParam)
{
    uint16_t xnId = GetXnId(instr->v2.operate.xnId, loopGroupParam);
    uint16_t xdId = GetXnId(instr->v2.operate.xdId, loopGroupParam);

    uint16_t parMode = instr->v2.operate.parMode;
    uint16_t ckeId = UpdateCKEId(instr->v2.operate.setCKEId, loopGroupParam);
    uint16_t ckeMask = instr->v2.operate.setCKEMask;

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId{0};
    curCcuTask->GetDieId(queId, dieId);

    uint64_t xnValue = 0;
    uint64_t xdValue = 0;
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xnId, xnValue));
    if (parMode == 0) {
        xdValue = xnValue - instr->v2.operate.xmId;
        HCCL_INFO("Sub Xn[%u](%llu) - immed[%u] to Xd[%u](%llu)", xnId, xnValue, instr->v2.operate.xmId, xdId, xdValue);
    } else {
        uint16_t xmId = GetXnId(instr->v2.operate.xmId, loopGroupParam);
        uint64_t xmValue = 0;
        CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xmId, xmValue));
        xdValue = xnValue - xmValue;
        HCCL_INFO("Sub Xn[%u](%llu) - Xm[%u](%llu) to Xd[%u](%llu)", xnId, xnValue, xmId, xmValue, xdId, xdValue);
    }

    CHK_RET(AllRankParamRecorder::Global()->SetXn(rankId, dieId, xdId, xdValue));
    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, ckeId, ckeMask));
    return HCCL_SUCCESS;
}

// Mul指令，实现算数“*”
HcclResult TransformMulInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
    LoopGroupParamA6* loopGroupParam)
{
    uint16_t xnId = GetXnId(instr->v2.operate.xnId, loopGroupParam);
    uint16_t xdId = GetXnId(instr->v2.operate.xdId, loopGroupParam);
    uint16_t parMode = instr->v2.operate.parMode;
    uint16_t ckeId = UpdateCKEId(instr->v2.operate.setCKEId, loopGroupParam);
    uint16_t ckeMask = instr->v2.operate.setCKEMask;

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId{0};
    curCcuTask->GetDieId(queId, dieId);

    uint64_t xnValue = 0;
    uint64_t xdValue = 0;
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xnId, xnValue));
    if (parMode == 0) {
        xdValue = (xnValue & 0xFFFFFFFF) * instr->v2.operate.xmId;
        HCCL_INFO(
            "Mul Xn[%u](%llu) * immed[%u] to Xd[%u](%llu)", xnId, xnValue & 0xFFFFFFFF, instr->v2.operate.xmId, xdId,
            xdValue);
    } else {
        uint16_t xmId = GetXnId(instr->v2.operate.xmId, loopGroupParam);
        uint64_t xmValue = 0;
        CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xmId, xmValue));
        xdValue = (xnValue & 0xFFFFFFFF) * (xmValue & 0xFFFFFFFF);
        HCCL_INFO(
            "Mul Xn[%u](%llu) * Xm[%u](%llu) to Xd[%u](%llu)", xnId, xnValue & 0xFFFFFFFF, xmId, xmValue & 0xFFFFFFFF,
            xdId, xdValue);
    }

    CHK_RET(AllRankParamRecorder::Global()->SetXn(rankId, dieId, xdId, xdValue));
    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, ckeId, ckeMask));
    return HCCL_SUCCESS;
}

// And指令，实现逻辑“&”
HcclResult TransformANDInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
    LoopGroupParamA6* loopGroupParam)
{
    uint16_t xnId = GetXnId(instr->v2.operate.xnId, loopGroupParam);
    uint16_t xdId = GetXnId(instr->v2.operate.xdId, loopGroupParam);
    uint16_t xmId = GetXnId(instr->v2.operate.xmId, loopGroupParam);
    uint16_t ckeId = UpdateCKEId(instr->v2.operate.setCKEId, loopGroupParam);
    uint16_t ckeMask = instr->v2.operate.setCKEMask;

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId{0};
    curCcuTask->GetDieId(queId, dieId);

    uint64_t xnValue = 0;
    uint64_t xmValue = 0;
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xnId, xnValue));
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xmId, xmValue));
    uint64_t xdValue = xnValue & xmValue;
    HCCL_INFO("And Xn[%u](%llu) & Xm[%u](%llu) to Xd[%u](%llu)", xnId, xnValue, xmId, xmValue, xdId, xdValue);

    CHK_RET(AllRankParamRecorder::Global()->SetXn(rankId, dieId, xdId, xdValue));
    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, ckeId, ckeMask));
    return HCCL_SUCCESS;
}

// Or指令，实现逻辑“|”
HcclResult TransformORInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
    LoopGroupParamA6* loopGroupParam)
{
    uint16_t xnId = GetXnId(instr->v2.operate.xnId, loopGroupParam);
    uint16_t xdId = GetXnId(instr->v2.operate.xdId, loopGroupParam);
    uint16_t xmId = GetXnId(instr->v2.operate.xmId, loopGroupParam);
    uint16_t ckeId = UpdateCKEId(instr->v2.operate.setCKEId, loopGroupParam);
    uint16_t ckeMask = instr->v2.operate.setCKEMask;

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId{0};
    curCcuTask->GetDieId(queId, dieId);

    uint64_t xnValue = 0;
    uint64_t xmValue = 0;
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xnId, xnValue));
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xmId, xmValue));
    uint64_t xdValue = xnValue | xmValue;
    HCCL_INFO("Or Xn[%u](%llu) | Xm[%u](%llu) to Xd[%u](%llu)", xnId, xnValue, xmId, xmValue, xdId, xdValue);

    CHK_RET(AllRankParamRecorder::Global()->SetXn(rankId, dieId, xdId, xdValue));
    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, ckeId, ckeMask));
    return HCCL_SUCCESS;
}

// Not指令，实现逻辑“~”
HcclResult TransformNOTInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
    LoopGroupParamA6* loopGroupParam)
{
    uint16_t xnId = GetXnId(instr->v2.operate.xnId, loopGroupParam);
    uint16_t xdId = GetXnId(instr->v2.operate.xdId, loopGroupParam);
    uint16_t ckeId = UpdateCKEId(instr->v2.operate.setCKEId, loopGroupParam);
    uint16_t ckeMask = instr->v2.operate.setCKEMask;

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId{0};
    curCcuTask->GetDieId(queId, dieId);

    uint64_t xnValue = 0;
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xnId, xnValue));
    uint64_t xdValue = ~xnValue;
    HCCL_INFO("Not Xn[%u](%llu) to Xd[%u](%llu)", xnId, xnValue, xdId, xdValue);

    CHK_RET(AllRankParamRecorder::Global()->SetXn(rankId, dieId, xdId, xdValue));
    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, ckeId, ckeMask));
    return HCCL_SUCCESS;
}

// Xor指令，实现逻辑“^”
HcclResult TransformXORInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
    LoopGroupParamA6* loopGroupParam)
{
    uint16_t xnId = GetXnId(instr->v2.operate.xnId, loopGroupParam);
    uint16_t xdId = GetXnId(instr->v2.operate.xdId, loopGroupParam);
    uint16_t xmId = GetXnId(instr->v2.operate.xmId, loopGroupParam);
    uint16_t ckeId = UpdateCKEId(instr->v2.operate.setCKEId, loopGroupParam);
    uint16_t ckeMask = instr->v2.operate.setCKEMask;

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId{0};
    curCcuTask->GetDieId(queId, dieId);

    uint64_t xnValue = 0;
    uint64_t xmValue = 0;
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xnId, xnValue));
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xmId, xmValue));
    uint64_t xdValue = xnValue ^ xmValue;
    HCCL_INFO("Xor Xn[%u](%llu) ^ Xm[%u](%llu) to Xd[%u](%llu)", xnId, xnValue, xmId, xmValue, xdId, xdValue);

    CHK_RET(AllRankParamRecorder::Global()->SetXn(rankId, dieId, xdId, xdValue));
    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, ckeId, ckeMask));
    return HCCL_SUCCESS;
}

// SHL指令，实现逻辑“<<”
HcclResult TransformSHLInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
    LoopGroupParamA6* loopGroupParam)
{
    uint16_t xnId = GetXnId(instr->v2.operate.xnId, loopGroupParam);
    uint16_t xdId = GetXnId(instr->v2.operate.xdId, loopGroupParam);
    uint16_t xmId = GetXnId(instr->v2.operate.xmId, loopGroupParam);
    uint8_t shiftType = instr->v2.operate.shiftType;
    uint16_t ckeId = UpdateCKEId(instr->v2.operate.setCKEId, loopGroupParam);
    uint16_t ckeMask = instr->v2.operate.setCKEMask;

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId{0};
    curCcuTask->GetDieId(queId, dieId);

    uint64_t xnValue = 0;
    uint64_t xdValue = 0;
    uint64_t xmValue = 0;
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xnId, xnValue));
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xmId, xmValue));

    switch (shiftType) {
        case 0: // 逻辑移位
            xdValue = xnValue << xmValue;
            HCCL_INFO(
                "Shl logical Xn[%u](%llu) << Xm[%u](%llu) to Xd[%u](%llu)", xnId, xnValue, xmId, xmValue, xdId,
                xdValue);
            break;
        case 1: // 算术移位
            xdValue = static_cast<int64_t>(xnValue) << xmValue;
            HCCL_INFO(
                "Shl arithmetic Xn[%u](%llu) << Xm[%u](%llu) to Xd[%u](%llu)", xnId, xnValue, xmId, xmValue, xdId,
                xdValue);
            break;
        case 2: // 循环移位
            xdValue = (xnValue << (xmValue & 0x3F)) | (xnValue >> (64 - (xmValue & 0x3F)));
            HCCL_INFO(
                "Shl rotate Xn[%u](%llu) <<< Xm[%u](%llu) to Xd[%u](%llu)", xnId, xnValue, xmId, xmValue, xdId,
                xdValue);
            break;
        default:
            HCCL_ERROR("Unsupported shift type: [%u]", shiftType);
            return HCCL_E_INTERNAL;
    }

    CHK_RET(AllRankParamRecorder::Global()->SetXn(rankId, dieId, xdId, xdValue));
    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, ckeId, ckeMask));
    return HCCL_SUCCESS;
}

// SHR指令，实现逻辑“>>”
HcclResult TransformSHRInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
    LoopGroupParamA6* loopGroupParam)
{
    uint16_t xnId = GetXnId(instr->v2.operate.xnId, loopGroupParam);
    uint16_t xdId = GetXnId(instr->v2.operate.xdId, loopGroupParam);
    uint16_t xmId = GetXnId(instr->v2.operate.xmId, loopGroupParam);
    uint8_t shiftType = instr->v2.operate.shiftType;
    uint16_t ckeId = UpdateCKEId(instr->v2.operate.setCKEId, loopGroupParam);
    uint16_t ckeMask = instr->v2.operate.setCKEMask;

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId{0};
    curCcuTask->GetDieId(queId, dieId);

    uint64_t xnValue = 0;
    uint64_t xdValue = 0;
    uint64_t xmValue = 0;
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xnId, xnValue));
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xmId, xmValue));

    switch (shiftType) {
        case 0: // 逻辑移位
            xdValue = xnValue >> xmValue;
            HCCL_INFO(
                "Shr logical Xn[%u](%llu) >> Xm[%u](%llu) to Xd[%u](%llu)", xnId, xnValue, xmId, xmValue, xdId,
                xdValue);
            break;
        case 1: // 算术移位
            xdValue = static_cast<int64_t>(xnValue) >> xmValue;
            HCCL_INFO(
                "Shr arithmetic Xn[%u](%llu) >> Xm[%u](%llu) to Xd[%u](%llu)", xnId, xnValue, xmId, xmValue, xdId,
                xdValue);
            break;
        case 2: // 循环移位
            xdValue = (xnValue >> (xmValue & 0x3F)) | (xnValue << (64 - (xmValue & 0x3F)));
            HCCL_INFO(
                "Shr rotate Xn[%u](%llu) >>> Xm[%u](%llu) to Xd[%u](%llu)", xnId, xnValue, xmId, xmValue, xdId,
                xdValue);
            break;
        default:
            HCCL_ERROR("Unsupported shift type:[%u]", shiftType);
            return HCCL_E_INTERNAL;
    }

    CHK_RET(AllRankParamRecorder::Global()->SetXn(rankId, dieId, xdId, xdValue));
    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, ckeId, ckeMask));
    return HCCL_SUCCESS;
}

// POPCNT指令,统计指定64bit寄存器中的二进制1的个数
HcclResult TransformPopcntInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
    LoopGroupParamA6* loopGroupParam)
{
    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId{0};
    curCcuTask->GetDieId(queId, dieId);

    uint16_t xnId = GetXnId(instr->v2.operate.xnId, loopGroupParam);
    uint64_t xnValue = 0;
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xnId, xnValue));

    uint16_t xdId = GetXnId(instr->v2.operate.xdId, loopGroupParam);
    uint64_t xdValue = __builtin_popcountll(xnValue);

    CHK_RET(AllRankParamRecorder::Global()->SetXn(rankId, dieId, xdId, xdValue));
    uint16_t ckeId = UpdateCKEId(instr->v2.operate.setCKEId, loopGroupParam);
    uint16_t ckeMask = instr->v2.operate.setCKEMask;
    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, ckeId, ckeMask));
    return HCCL_SUCCESS;
}

// 处理Loop指令
HcclResult ProcessLoopIns(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
    LoopGroupParamA6* loopGroupParam, uint32_t loopGroupIdx);

HcclResult TransformLoopInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
    LoopGroupParamA6* loopGroupParam)
{
    uint16_t startInstrId = instr->v2.loop.startInstrId;
    uint16_t endInstrId = instr->v2.loop.endInstrId;
    uint16_t xnId = instr->v2.loop.xnId;
    HCCL_ERROR(
        "Loop instruction is not supported separately. startInstrId[%u] to endInstrId[%u] with loopXn[%u]",
        startInstrId, endInstrId, xnId);
    // 当前Loop都需要通过LoopGoup来触发，暂不支持单独解析Loop命令
    return HCCL_E_INTERNAL;
}

// LoopGroup指令
HcclResult TransformLoopGroupInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
    LoopGroupParamA6* loopInfo)
{
    CHK_RET(CheckLoopGroupNotSupport(loopInfo));
    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId{0};
    curCcuTask->GetDieId(queId, dieId);
    LoopGroupParamA6 loopGroupParam{};
    uint16_t startLoopInstrId = instr->v2.loopGroup.startLoopInstrId;
    uint16_t xnId = instr->v2.loopGroup.xnId & 0x7FFF;
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xnId, loopGroupParam.loopGroupXn.value));

    uint16_t xmId = instr->v2.loopGroup.xmId & 0x7FFF;
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xmId, loopGroupParam.loopGroupXm.value));

    uint16_t xpId = instr->v2.loopGroup.xpId & 0x7FFF;
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xpId, loopGroupParam.loopGroupXp.value));
    // 数值校验
    if (loopGroupParam.loopGroupXn.loopInsCnt > 512 || loopGroupParam.loopGroupXn.expandCnt > 511
        || loopGroupParam.loopGroupXn.expandOffset > 511
        || loopGroupParam.loopGroupXn.expandOffset > loopGroupParam.loopGroupXn.loopInsCnt) {
        HCCL_ERROR(
            "LoopGroup Xn[%u] loopInsCnt[%u] expandCnt[%u] expandOffset[%u] is invalid",
            loopGroupParam.loopGroupXn.value, loopGroupParam.loopGroupXn.loopInsCnt,
            loopGroupParam.loopGroupXn.expandCnt, loopGroupParam.loopGroupXn.expandOffset);
        return HCCL_E_INTERNAL;
    }

    hcomm::CcuRep::CcuInstrInfo& microCodeQue = curCcuTask->instrInfo[queId];
    auto loopGroupIdx = curCcuTask->loopGroupIdx++;
    uint64_t loopCnt = loopGroupParam.loopGroupXn.loopInsCnt;
    HCCL_VM_INFO(
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

    HCCL_INFO("LoopGroup From startLoopInstrId[%u] with loopGroupXn[%u], offsetXn[%u]", startLoopInstrId, xnId, xmId);
    return HCCL_SUCCESS;
}

// SetCKBit指令
HcclResult TransformSetCKBitInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
    LoopGroupParamA6* loopGroupParam)
{
    uint16_t clearType = instr->v2.setCKE.clearType;
    uint16_t setCKEId = UpdateCKEId(instr->v2.setCKE.setCKEId, loopGroupParam);
    uint16_t setCKEMask = instr->v2.setCKE.setCKEMask;
    uint16_t waitCKEId = UpdateCKEId(instr->v2.setCKE.waitCKEId, loopGroupParam);
    uint16_t waitCKEMask = instr->v2.setCKE.waitCKEMask;
    // 当前只支持clearType为1的场景
    if (clearType != 0x0001) {
        HCCL_ERROR("do not support clearType[%hu]", clearType);
        return HCCL_E_INTERNAL;
    }

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId{0};
    curCcuTask->GetDieId(queId, dieId);

    CHK_RET(ProcessWaitMask(rankId, dieId, curCcuTask, queId, waitCKEId, waitCKEMask, isContinue));
    if (!isContinue) {
        return HCCL_SUCCESS;
    }
    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask, true));
    CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));
    HCCL_INFO(
        "Wait CKE[%u:%04x], Set CKE[%u:%04x], clearType[%u]", waitCKEId, waitCKEMask, setCKEId, setCKEMask, clearType);
    return HCCL_SUCCESS;
}

// ClearCKE指令
HcclResult TransformClearCKEInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
    LoopGroupParamA6* loopGroupParam)
{
    uint16_t clearType = instr->v2.clearCKE.clearType;
    uint16_t clearCKEId = UpdateCKEId(instr->v2.clearCKE.clearCKEId, loopGroupParam);
    uint16_t clearMask = instr->v2.clearCKE.clearMask;
    uint16_t waitCKEId = UpdateCKEId(instr->v2.clearCKE.waitCKEId, loopGroupParam);
    uint16_t waitCKEMask = instr->v2.clearCKE.waitCKEMask;

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId{0};
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

    HCCL_INFO(
        "Wait CKE[%u:%04x], Clear CKE[%u:%04x], clearType[%u]", waitCKEId, waitCKEMask, clearCKEId, clearMask,
        clearType);
    return HCCL_SUCCESS;
}

HcclResult TransformJumpInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
    LoopGroupParamA6* loopGroupParam)
{
    uint16_t expectedXnId = GetXnId(instr->v2.jmp.expectedXnId, loopGroupParam);
    uint16_t conditionXnId = GetXnId(instr->v2.jmp.conditionXnId, loopGroupParam);
    uint16_t relTarInstrXnId = GetXnId(instr->v2.jmp.relTarInstrXnId, loopGroupParam);

    uint8_t conditionType = instr->v2.jmp.conditionType;
    uint8_t jumpMode = instr->v2.jmp.jumpMode;

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId{0};
    curCcuTask->GetDieId(queId, dieId);

    uint64_t expectedValue = 0;
    uint64_t conditionValue = 0;
    uint64_t relTarInstrValue = 0;
    if (conditionType != 6) {
        CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, expectedXnId, expectedValue));
        CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, conditionXnId, conditionValue));
    }
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, relTarInstrXnId, relTarInstrValue));

    uint64_t nextInsIdx = 0;
    auto curInstrId = curCcuTask->microCodePosInQue[queId];
    auto updateNextInsIdx
        = [jumpMode, curInstrId, relTarInstrValue, queId, curCcuTask, &nextInsIdx](bool isJump) -> void {
        if (isJump) {
            if (jumpMode == 0) {
                // 相对跳转
                nextInsIdx = curInstrId + relTarInstrValue;
                if (nextInsIdx >= JUMP_INSTR_MAX) {
                    nextInsIdx -= JUMP_INSTR_MAX;
                }
            } else if (jumpMode == 1) {
                // 绝对跳转
                nextInsIdx = relTarInstrValue;
            }
            curCcuTask->microCodePosInQue[queId] = nextInsIdx - curCcuTask->startInstrIdInQue[queId];
        }
    };

    switch (conditionType) {
        case 0: // 等于
            updateNextInsIdx(conditionValue == expectedValue);
            HCCL_INFO(
                "When conditionXn[%u][%llu] equal to expectData[%u][%llu], Jump Mode[%u],Jump instruction "
                "offset[%u][%llu]",
                conditionXnId, conditionValue, expectedXnId, expectedValue, jumpMode, relTarInstrXnId,
                relTarInstrValue);
            break;
        case 1: // 不等于
            updateNextInsIdx(conditionValue != expectedValue);
            HCCL_INFO(
                "When conditionXn[%u][%llu] not equal to expectData[%u][%llu], Jump Mode[%u], Jump instruction "
                "offset[%u][%llu]",
                conditionXnId, conditionValue, expectedXnId, expectedValue, jumpMode, relTarInstrXnId,
                relTarInstrValue);
            break;
        case 2: // 大于
            updateNextInsIdx(conditionValue > expectedValue);
            HCCL_INFO(
                "When conditionXn[%u][%llu] greater than expectData[%u][%llu], Jump Mode[%u], Jump instruction "
                "offset[%u][%llu]",
                conditionXnId, conditionValue, expectedXnId, expectedValue, jumpMode, relTarInstrXnId,
                relTarInstrValue);
            break;
        case 3: // 大于/等于
            updateNextInsIdx(conditionValue >= expectedValue);
            HCCL_INFO(
                "When conditionXn[%u][%llu] greater than or equal to expectData[%u][%llu], Jump Mode[%u], Jump "
                "instruction offset[%u][%llu]",
                conditionXnId, conditionValue, expectedXnId, expectedValue, jumpMode, relTarInstrXnId,
                relTarInstrValue);
            break;
        case 4: // 小于
            updateNextInsIdx(conditionValue < expectedValue);
            HCCL_INFO(
                "When conditionXn[%u][%llu] less than expectData[%u][%llu], Jump Mode[%u], Jump instruction "
                "offset[%u][%llu]",
                conditionXnId, conditionValue, expectedXnId, expectedValue, jumpMode, relTarInstrXnId,
                relTarInstrValue);
            break;
        case 5: // 小于/等于
            updateNextInsIdx(conditionValue <= expectedValue);
            HCCL_INFO(
                "When conditionXn[%u][%llu] less than or equal to expectData[%u][%llu], Jump Mode[%u], Jump "
                "instruction offset[%u][%llu]",
                conditionXnId, conditionValue, expectedXnId, expectedValue, jumpMode, relTarInstrXnId,
                relTarInstrValue);
            break;
        default: // 无条件跳转
            updateNextInsIdx(true);
            HCCL_INFO(
                "When conditionXn[%u][%llu] is true, Jump Mode[%u], Jump instruction offset[%u][%llu]", conditionXnId,
                conditionValue, jumpMode, relTarInstrXnId, relTarInstrValue);
            break;
    }
    return HCCL_SUCCESS;
}

// Wait指令
HcclResult TransformWaitInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
    LoopGroupParamA6* loopGroupParam)
{
    uint16_t expectedXnId = GetXnId(instr->v2.wait.expectedXnId, loopGroupParam);
    uint16_t conditionXnId = GetXnId(instr->v2.wait.conditionXnId, loopGroupParam);

    uint8_t conditionType = instr->v2.wait.conditionType;

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId{0};
    curCcuTask->GetDieId(queId, dieId);

    uint64_t expectedValue = 0;
    uint64_t conditionValue = 0;
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, expectedXnId, expectedValue));
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, conditionXnId, conditionValue));

    uint64_t nextInsIdx = 0;
    auto curInstrId = curCcuTask->microCodePosInQue[queId];
    auto updateNextInsIdx = [curInstrId, &nextInsIdx](bool notWait) -> void {
        if (notWait) {
            nextInsIdx = curInstrId + 1;
        } else {
            nextInsIdx = curInstrId;
        }
    };

    switch (conditionType) {
        case 0: // 等于
            updateNextInsIdx(conditionValue == expectedValue);
            HCCL_INFO(
                "When conditionXn[%u][%llu] equal to expectData[%u][%llu], WaitFlag[%u]", conditionXnId, conditionValue,
                expectedXnId, expectedValue, conditionValue == expectedValue);
            break;
        case 1: // 不等于
            updateNextInsIdx(conditionValue != expectedValue);
            HCCL_INFO(
                "When conditionXn[%u][%llu] not equal to expectData[%u][%llu], WaitFlag[%u]", conditionXnId,
                conditionValue, expectedXnId, expectedValue, conditionValue != expectedValue);
            break;
        case 2: // 大于
            updateNextInsIdx(conditionValue > expectedValue);
            HCCL_INFO(
                "When conditionXn[%u][%llu] greater than expectData[%u][%llu], WaitFlag[%u]", conditionXnId,
                conditionValue, expectedXnId, expectedValue, conditionValue > expectedValue);
            break;
        case 3: // 大于/等于
            updateNextInsIdx(conditionValue >= expectedValue);
            HCCL_INFO(
                "When conditionXn[%u][%llu] greater than or equal to expectData[%u][%llu], WaitFlag[%u]", conditionXnId,
                conditionValue, expectedXnId, expectedValue, conditionValue >= expectedValue);
            break;
        case 4: // 小于
            updateNextInsIdx(conditionValue < expectedValue);
            HCCL_INFO(
                "When conditionXn[%u][%llu] less than expectData[%u][%llu], WaitFlag[%u]", conditionXnId,
                conditionValue, expectedXnId, expectedValue, conditionValue < expectedValue);
            break;
        case 5: // 小于/等于
            updateNextInsIdx(conditionValue <= expectedValue);
            HCCL_INFO(
                "When conditionXn[%u][%llu] less than or equal to expectData[%u][%llu], WaitFlag[%u]", conditionXnId,
                conditionValue, expectedXnId, expectedValue, conditionValue <= expectedValue);
            break;
        default: // 参考等于
            updateNextInsIdx(conditionValue == expectedValue);
            HCCL_INFO(
                "When conditionXn[%u][%llu] equal to expectData[%u][%llu], WaitFlag[%u]", conditionXnId, conditionValue,
                expectedXnId, expectedValue, conditionValue == expectedValue);
            break;
    }
    curCcuTask->microCodePosInQue[queId] = nextInsIdx - curCcuTask->startInstrIdInQue[queId];
    return HCCL_SUCCESS;
}

// Fence指令
HcclResult TransformFenceInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
    LoopGroupParamA6* loopGroupParam)
{
    HCCL_ERROR("TransformFenceInstr is not supported.");
    return HCCL_E_PARA;
}

// TransLocMemToLocMS指令：将内存中的数据搬移到本端MS
HcclResult TransformTransLocMemToLocMSInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
    LoopGroupParamA6* loopGroupParam)
{
    uint16_t msId = instr->v2.transLocMemToLocMS.msId;
    uint16_t locMSId = UpdateMSId(msId, loopGroupParam);
    uint16_t xsId = GetXnId(instr->v2.transLocMemToLocMS.xsId, loopGroupParam);
    uint16_t xstId = GetXnId(instr->v2.transLocMemToLocMS.xstId, loopGroupParam);
    uint16_t xlId = GetXnId(instr->v2.transLocMemToLocMS.xlId, loopGroupParam);
    uint16_t xoId = GetXnId(instr->v2.transLocMemToLocMS.xoId, loopGroupParam);

    // allocHint和victimHint用于L2 Cache功能，暂不支持
    uint16_t setCkeId = UpdateCKEId(instr->v2.transLocMemToLocMS.setCKEId, loopGroupParam);
    uint16_t setCkeMask = instr->v2.transLocMemToLocMS.setCKEMask;

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId{0};
    curCcuTask->GetDieId(queId, dieId);

    uint64_t locMemAddr = 0x0;
    uint16_t addrExpandCoef = 0;
    uint64_t addrExpandInfo = 0;
    uint64_t msStartAddrOffset = 0; // MS地址从非4KB对齐位置开始
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xsId, locMemAddr));
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xstId, addrExpandInfo));
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xoId, msStartAddrOffset));
    // 取[54:53]bit位
    addrExpandCoef = (addrExpandInfo >> 53) & 0x3;
    locMemAddr = UpdateAddress(locMemAddr, loopGroupParam, addrExpandCoef);
    uint64_t len = 0;
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xlId, len));
    CHK_PRT_RET(len == 0, HCCL_ERROR("The size of data transfer is 0."), HCCL_E_INTERNAL);

    DataSlice srcSlice;
    DataSlice dstSlice;
    // todo: MS地址偏移计算
    CHK_RET(StorageManager::GetInstance().GetSlice(locMemAddr, len, srcSlice));
    CHK_RET(GenSliceFromMs(locMSId, len, dstSlice));

    AddLocalCopy(rankId, queId, curCcuTask, srcSlice, dstSlice);

    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCkeId, setCkeMask));

    return HCCL_SUCCESS;
}

// TransformTransLocMSToLocMemInstr指令：将本端MS数据搬运到内存中
HcclResult TransformTransLocMSToLocMemInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
    LoopGroupParamA6* loopGroupParam)
{
    uint16_t msId = instr->v2.transLocMSToLocMem.msId;
    uint16_t locMSId = UpdateMSId(msId, loopGroupParam);

    uint16_t xdId = GetXnId(instr->v2.transLocMSToLocMem.xdId, loopGroupParam);
    uint16_t xdtId = GetXnId(instr->v2.transLocMSToLocMem.xdtId, loopGroupParam);
    uint16_t xlId = GetXnId(instr->v2.transLocMSToLocMem.xlId, loopGroupParam);
    uint16_t xoId = GetXnId(instr->v2.transLocMSToLocMem.xoId, loopGroupParam);

    // allocHint和victimHint用于L2 Cache功能，暂不支持
    uint16_t setCkeId = UpdateCKEId(instr->v2.transLocMSToLocMem.setCKEId, loopGroupParam);
    uint16_t setCkeMask = instr->v2.transLocMSToLocMem.setCKEMask;

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId{0};
    curCcuTask->GetDieId(queId, dieId);

    uint64_t locMemAddr = 0x0;
    uint16_t addrExpandCoef = 0;
    uint64_t addrExpandInfo = 0;
    uint64_t msStartAddrOffset = 0; // MS地址从非4KB对齐位置开始
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xdId, locMemAddr));
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xdtId, addrExpandInfo));
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xoId, msStartAddrOffset));
    // 取[54:53]bit位
    addrExpandCoef = (addrExpandInfo >> 53) & 0x3;
    locMemAddr = UpdateAddress(locMemAddr, loopGroupParam, addrExpandCoef);
    uint64_t len = 0;
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xlId, len));
    CHK_PRT_RET(len == 0, HCCL_ERROR("The size of data transfer is 0."), HCCL_E_INTERNAL);

    DataSlice srcSlice;
    DataSlice dstSlice;
    // todo: MS地址偏移计算
    CHK_RET(GenSliceFromMs(locMSId, len, srcSlice));
    CHK_RET(StorageManager::GetInstance().GetSlice(locMemAddr, len, dstSlice));

    AddLocalCopy(rankId, queId, curCcuTask, srcSlice, dstSlice);

    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCkeId, setCkeMask));
    return HCCL_SUCCESS;
}

//   TransformTransLocMSToLocMS指令：将本端MS数据搬运到另一端MS中，用于die内部数据搬运
HcclResult TransformTransLocMSToLocMSInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
    LoopGroupParamA6* loopGroupParam)
{
    uint16_t msdId = instr->v2.transLocMSToLocMS.msdId;
    uint16_t locMsdId = UpdateMSId(msdId, loopGroupParam);
    uint16_t mssId = instr->v2.transLocMSToLocMS.mssId;
    uint16_t locMssId = UpdateMSId(mssId, loopGroupParam);

    uint16_t xlId = GetXnId(instr->v2.transLocMSToLocMS.xlId, loopGroupParam);
    uint16_t xoId = GetXnId(instr->v2.transLocMSToLocMS.xoId, loopGroupParam);
    // allocHint和victimHint用于L2 Cache功能，暂不支持
    uint16_t setCkeId = UpdateCKEId(instr->v2.transLocMSToLocMS.setCKEId, loopGroupParam);
    uint16_t setCkeMask = instr->v2.transLocMSToLocMS.setCKEMask;

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId{0};
    curCcuTask->GetDieId(queId, dieId);

    uint64_t msStartAddrOffset = 0; // MS地址从非4KB对齐位置开始
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xoId, msStartAddrOffset));
    uint64_t len = 0;
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xlId, len));
    CHK_PRT_RET(len == 0, HCCL_ERROR("The size of data transfer is 0."), HCCL_E_INTERNAL);

    DataSlice srcSlice;
    DataSlice dstSlice;
    // todo: MS地址偏移计算
    CHK_RET(GenSliceFromMs(locMssId, len, srcSlice));
    CHK_RET(GenSliceFromMs(locMsdId, len, dstSlice));

    AddLocalCopy(rankId, queId, curCcuTask, srcSlice, dstSlice);

    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCkeId, setCkeMask));
    return HCCL_SUCCESS;
}

// TransformTransLocMemToLocMem指令：将本端内存数据搬运到本端内存中
HcclResult TransformTransLocMemToLocMemInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
    LoopGroupParamA6* loopGroupParam)
{
    uint16_t xdId = GetXnId(instr->v2.transLocMemToLocMem.xdId, loopGroupParam);
    uint16_t xdtId = GetXnId(instr->v2.transLocMemToLocMem.xdtId, loopGroupParam);
    uint16_t xsId = GetXnId(instr->v2.transLocMemToLocMem.xsId, loopGroupParam);
    uint16_t xstId = GetXnId(instr->v2.transLocMemToLocMem.xstId, loopGroupParam);
    uint16_t xlId = GetXnId(instr->v2.transLocMemToLocMem.xlId, loopGroupParam);

    uint16_t msdTmpId = instr->v2.transLocMemToLocMem.usedMSId;
    uint16_t locMsdId = UpdateMSId(msdTmpId, loopGroupParam);
    uint16_t msNum = instr->v2.transLocMemToLocMem.msNum;
    // allocHint和victimHint用于L2 Cache功能，暂不支持
    uint16_t setCkeId = UpdateCKEId(instr->v2.transLocMemToLocMem.setCKEId, loopGroupParam);
    uint16_t setCkeMask = instr->v2.transLocMemToLocMem.setCKEMask;

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId{0};
    curCcuTask->GetDieId(queId, dieId);

    uint64_t srcMemAddr = 0x0;
    uint64_t dstMemAddr = 0x0;
    uint16_t addrExpandCoef = 0;
    uint64_t addrExpandInfo = 0;
    // todo: localMem到
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xsId, srcMemAddr));
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xdId, dstMemAddr));
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xdtId, addrExpandInfo));
    // 取[54:53]bit位
    addrExpandCoef = (addrExpandInfo >> 53) & 0x3;
    srcMemAddr = UpdateAddress(srcMemAddr, loopGroupParam, addrExpandCoef);
    dstMemAddr = UpdateAddress(dstMemAddr, loopGroupParam, addrExpandCoef);
    uint64_t len = 0;
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xlId, len));
    CHK_PRT_RET(len == 0, HCCL_ERROR("The size of data transfer is 0."), HCCL_E_INTERNAL);

    DataSlice srcSlice;
    DataSlice dstSlice;
    // todo: MS地址偏移计算
    CHK_RET(StorageManager::GetInstance().GetSlice(srcMemAddr, len, srcSlice));
    CHK_RET(StorageManager::GetInstance().GetSlice(dstMemAddr, len, dstSlice));

    AddLocalCopy(rankId, queId, curCcuTask, srcSlice, dstSlice);

    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCkeId, setCkeMask));
    return HCCL_SUCCESS;
}

uint16_t ChooseReduceType(uint16_t udfType)
{
    if (udfType == 8) {
        return CcuRep::CCU_REDUCE_SUM;
    } else if (udfType == 9) {
        return CcuRep::CCU_REDUCE_MAX;
    } else if (udfType == 10) {
        return CcuRep::CCU_REDUCE_MIN;
    } else {
        return CcuRep::CCU_REDUCE_MAX_MS;
    }
}

// TransformTransMem指令：使用UB进行数据搬移
HcclResult TransformTransMemInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
    LoopGroupParamA6* loopGroupParam)
{
    // 参数校验
    uint16_t udfType = instr->v2.transMem.udfType;
    if (udfType != 0) {
        HCCL_ERROR("TransformTransMemInstr is not supported. udfType:[%u]", udfType);
        return HCCL_E_PARA;
    }
    uint16_t dmaOpCode = instr->v2.transMem.dmaOpCode; // 用于判断是读还是写
    // 1. 优先执行指令搬移
    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId{0};
    curCcuTask->GetDieId(queId, dieId);

    uint16_t xdId = GetXnId(instr->v2.transMem.xdId, loopGroupParam);

    uint16_t xsId = GetXnId(instr->v2.transMem.xsId, loopGroupParam);
    uint16_t xlId = GetXnId(instr->v2.transMem.xlId, loopGroupParam);
    uint64_t len = 0;
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xlId, len));
    // 通过channal找对端信息
    uint16_t xcId = GetXnId(instr->v2.transMem.xcId, loopGroupParam);
    uint64_t xcValue = 0;
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xcId, xcValue));
    // 需要找对端的rank信息
    if (!IsExistRemoteDieInfo(rankId, dieId, xcValue)) {
        HCCL_ERROR(
            "TransformTransMemInstr RemoteDieInfo is not exist. rankId:[%u], dieId:[%u], xcValue:[%lu]", rankId, dieId,
            xcValue);
        return HCCL_E_PARA;
    }
    uint16_t srcMode = instr->v2.transMem.src_mode;
    uint16_t dstMode = instr->v2.transMem.dst_mode;
    RankId rmtRankId = g_allRankChannelInfo[rankId][dieId][xcValue].dstRank;
    uint32_t rmtDieId = g_allRankChannelInfo[rankId][dieId][xcValue].remoteDieId;
    uint16_t msMode = instr->v2.transMem.msIdMode;
    DataSlice srcSlice;
    DataSlice dstSlice;
    uint64_t xdValue = 0;
    uint64_t xsValue = 0;
    uint16_t reduceEn = instr->v2.transMem.udfEnable;

    HcclReduceOp checkerReduceOp;
    HcclDataType checkerDataType;
    if (reduceEn) {
        checkerReduceOp = g_ReduceOp2CheckerReduceOp_ccu[instr->v2.transMem.reduceOpCode];
        DataType hcclDataType;
        CHK_RET(GetHcclDataTypeFromCCUDataType(
            instr->v2.transMem.reduceDataType, ccuReduceTypeMap[instr->v2.transMem.reduceOpCode], hcclDataType));
        checkerDataType = g_DataType2CheckerDataType_aicpu[hcclDataType];
    }

    if (dmaOpCode == 3 || dmaOpCode == 5) {
        // 写数据 dst为对端 src为本端
        if (msMode == 1) {
            // XsId寄存器中存的就是MSId
            CHK_RET(GenSliceFromMs(UpdateMSId(instr->v2.transMem.xsId, loopGroupParam), len, srcSlice));
        } else {
            CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xsId, xsValue));
            xsValue = (srcMode == 1) ? UpdateAddress(xsValue, loopGroupParam) :
                                       UpdateAddressWithoutStride(xsValue, loopGroupParam);
            CHK_RET(StorageManager::GetInstance().GetSlice(xsValue, len, srcSlice));
        }
        CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xdId, xdValue));
        xdValue = (dstMode == 1) ? UpdateAddress(xdValue, loopGroupParam) :
                                   UpdateAddressWithoutStride(xdValue, loopGroupParam);
        CHK_RET(StorageManager::GetInstance().GetSlice(xdValue, len, dstSlice));
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
    } else if (dmaOpCode == 6) {
        // 读数据 src为对端 dst为本端
        CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xsId, xsValue));
        CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xsId, xsValue));
        xsValue = (srcMode == 1) ? UpdateAddress(xsValue, loopGroupParam) :
                                   UpdateAddressWithoutStride(xsValue, loopGroupParam);
        CHK_RET(StorageManager::GetInstance().GetSlice(xsValue, len, srcSlice));
        if (msMode == 1) {
            CHK_RET(GenSliceFromMs(UpdateMSId(instr->v2.transMem.xdId, loopGroupParam), len, dstSlice));
        } else {
            CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xdId, xdValue));
            xdValue = (dstMode == 1) ? UpdateAddress(xdValue, loopGroupParam) :
                                       UpdateAddressWithoutStride(xdValue, loopGroupParam);
            CHK_RET(StorageManager::GetInstance().GetSlice(xdValue, len, dstSlice));
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
    } else {
        HCCL_ERROR("TransformTransMemInstr is not supported. dmaOpCode:[%u]", dmaOpCode);
        return HCCL_E_PARA;
    }

    // 写notify信息
    if (dmaOpCode == 5) {
        uint16_t xnId = GetXnId(instr->v2.transMem.xsId, loopGroupParam);
        uint64_t xnAddr = 0;
        CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xnId, xnAddr));
        uint16_t xnIdRmt = 0;
        CHK_RET(AllRankParamRecorder::Global()->GetXnIdByAddr(dieId, CcuComponerntType::XN_A6, xnAddr, xnIdRmt));
        uint32_t notifyValue = instr->v2.transMem.value;
        CHK_RET(
            AllRankParamRecorder::Global()->SetXn(rmtRankId, rmtDieId, xnIdRmt, static_cast<uint64_t>(notifyValue)));
    }

    uint16_t setCkeId = UpdateCKEId(instr->v2.transMem.setCKEId, loopGroupParam);
    uint16_t setCkeMask = instr->v2.transMem.setCKEMask;
    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCkeId, setCkeMask));
    return HCCL_SUCCESS;
}

// TransformSyncXnWtx指令：同步指令，将本端寄存器的值以write操作同步到远端，用于同步XN和WTX
HcclResult TransformSyncXnWtxInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
    LoopGroupParamA6* loopGroupParam)
{
    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId{0};
    curCcuTask->GetDieId(queId, dieId);
    uint16_t xdId = GetXnId(instr->v2.syncWtX.xdId, loopGroupParam);
    uint64_t xdValue = 0;
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xdId, xdValue));

    uint16_t xcId = GetXnId(instr->v2.syncWtX.xcId, loopGroupParam);
    uint64_t xcValue = 0;
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xcId, xcValue));
    if (!IsExistRemoteDieInfo(rankId, dieId, xcValue)) {
        HCCL_ERROR(
            "TransformSyncXnWtxInstr RemoteDieInfo is not exist. rankId:[%u], dieId:[%u], xcValue:[%lu]", rankId, dieId,
            xcValue);
        return HCCL_E_PARA;
    }
    // 需要找对端的rank信息
    RankId rmtRankId = g_allRankChannelInfo[rankId][dieId][xcValue].dstRank;
    uint32_t rmtDieId = g_allRankChannelInfo[rankId][dieId][xcValue].remoteDieId;

    uint16_t parMode = instr->v2.syncWtX.parMode;
    uint64_t xsValue = instr->v2.syncWtX.xsId;
    if (parMode == 1) {
        uint16_t xsId = GetXnId(instr->v2.syncWtX.xsId, loopGroupParam);
        CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xsId, xsValue));
    }
    // 将Xs内容写到Xd中
    uint16_t xdIdRmt{0};
    CcuComponerntType ccuType{CcuComponerntType::UNKNOWN};
    CHK_RET(AllRankParamRecorder::Global()->GetXnAndTypeIdByAddr(dieId, xdValue, ccuType, xdIdRmt));
    if (ccuType == CcuComponerntType::XN_A6) {
        CHK_RET(AllRankParamRecorder::Global()->SetXn(rmtRankId, rmtDieId, xdIdRmt, xsValue));
    } else if (ccuType == CcuComponerntType::CKE_A6) {
        CHK_RET(ProcessSetMask(rmtRankId, rmtDieId, curCcuTask, queId, xdIdRmt, xsValue));
    } else {
        HCCL_ERROR("TransformSyncXnWtxInstr is not supported. xdId:[%u]", xdId);
        return HCCL_E_PARA;
    }

    uint16_t notifyValid = instr->v2.syncWtX.notifyValid;
    // 写notify信息
    if (notifyValid == 1) {
        uint16_t xnId = GetXnId(instr->v2.syncWtX.xnId, loopGroupParam);
        uint64_t xnAddr = 0;
        CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xnId, xnAddr));
        uint16_t cekIdRmtId = 0;
        CHK_RET(AllRankParamRecorder::Global()->GetXnIdByAddr(dieId, CcuComponerntType::CKE_A6, xnAddr, cekIdRmtId));
        uint32_t notifyValue = instr->v2.syncWtX.value;
        CHK_RET(ProcessSetMask(rmtRankId, rmtDieId, curCcuTask, queId, cekIdRmtId, static_cast<uint64_t>(notifyValue)));
    }
    return HCCL_SUCCESS;
}

// TransformSyncAtx指令：同步指令，将本端寄存器的值以atomic操作同步到远端，用于同步ATX
HcclResult TransformSyncAtxInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
    LoopGroupParamA6* loopGroupParam)
{
    HCCL_ERROR("TransformSyncAtxInstr is not supported.");
    return HCCL_E_PARA;
}

// ReduceAdd指令
HcclResult TransformReduceAddInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
    LoopGroupParamA6* loopGroupParam)
{
    uint16_t count = instr->v2.reduce.count;
    uint16_t castEn = instr->v2.reduce.castEn;
    uint16_t dataType = instr->v2.reduce.dataType;
    uint16_t setCKEId = UpdateCKEId(instr->v2.reduce.setCKEId, loopGroupParam);
    uint16_t setCKEMask = instr->v2.reduce.setCKEMask;
    uint16_t lengthId = instr->v2.reduce.XnIdLength;

    uint16_t msId[CcuRep::CCU_REDUCE_MAX_MS];
    for (uint16_t index = 0; index < CcuRep::CCU_REDUCE_MAX_MS; index++) {
        msId[index] = UpdateMSId(instr->v2.reduce.msId[index], loopGroupParam);
    }
    // todo: 对于不支持的数据类型需要进行报错处理

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId{0};
    curCcuTask->GetDieId(queId, dieId);

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

    HCCL_INFO(
        "Add [%s] with Count[%u], DataType[%u] and CastEn[%u], Set CKE[%u:%04x]", ParseMSList(instr).c_str(), count,
        dataType, castEn, setCKEId, setCKEMask);

    return HCCL_SUCCESS;
}

HcclResult TransformReduceMaxInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
    LoopGroupParamA6* loopGroupParam)
{
    uint16_t count = instr->v2.reduce.count;
    uint16_t dataType = instr->v2.reduce.dataType;
    uint16_t setCKEId = UpdateCKEId(instr->v2.reduce.setCKEId, loopGroupParam);
    uint16_t setCKEMask = instr->v2.reduce.setCKEMask;
    uint16_t lengthId = instr->v2.reduce.XnIdLength;

    uint16_t msId[CcuRep::CCU_REDUCE_MAX_MS];
    for (uint16_t index = 0; index < CcuRep::CCU_REDUCE_MAX_MS; index++) {
        msId[index] = UpdateMSId(instr->v2.reduce.msId[index], loopGroupParam);
    }
    // todo: 对于不支持的数据类型需要进行报错处理

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId{0};
    curCcuTask->GetDieId(queId, dieId);

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

    HCCL_INFO(
        "Max[%s] with Count[%u], DataType[%u], Set CKE[%u:%04x]", ParseMSList(instr).c_str(), count, dataType, setCKEId,
        setCKEMask);

    return HCCL_SUCCESS;
}

// ReduceMin指令
HcclResult TransformReduceMinInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
    LoopGroupParamA6* loopGroupParam)
{
    uint16_t count = instr->v2.reduce.count;
    uint16_t dataType = instr->v2.reduce.dataType;
    uint16_t setCKEId = UpdateCKEId(instr->v2.reduce.setCKEId, loopGroupParam);
    uint16_t setCKEMask = instr->v2.reduce.setCKEMask;
    uint16_t lengthId = instr->v2.reduce.XnIdLength;

    uint16_t msId[CcuRep::CCU_REDUCE_MAX_MS];
    for (uint16_t index = 0; index < CcuRep::CCU_REDUCE_MAX_MS; index++) {
        msId[index] = UpdateMSId(instr->v2.reduce.msId[index], loopGroupParam);
    }
    // todo: 对于不支持的数据类型需要进行报错处理
    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId{0};
    curCcuTask->GetDieId(queId, dieId);

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

    HCCL_INFO(
        "Min [%s] with Count[%u], DataType[%u], Set CKE[%u:%04x]", HcclSim::ParseMSList(instr).c_str(), count, dataType,
        setCKEId, setCKEMask);
    return HCCL_SUCCESS;
}

// 以下为数据转换函数，只起到转换数据作用
HcclResult TransformLoadSqeArgsToXnInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
    return TransformLoadSqeArgsToXnInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformLoadImdToXInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
    return TransformLoadImdToXInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformLoadXInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
    return TransformLoadXInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformStoreXInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
    return TransformStoreXInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformClearXInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
    return TransformClearXInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformNopInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
    return TransformNopInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformLoadInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
    return TransformLoadInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformStoreInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
    return TransformStoreInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformAddInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
    return TransformAddInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformSubInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
    return TransformSubInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformMulInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
    return TransformMulInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformANDInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
    return TransformANDInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformORInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
    return TransformORInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformNOTInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
    return TransformNOTInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformXORInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
    return TransformXORInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformSHLInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
    return TransformSHLInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformSHRInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
    return TransformSHRInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformPopcntInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
    return TransformPopcntInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformLoopInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
    return TransformLoopInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformLoopGroupInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
    return TransformLoopGroupInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformSetCKBitInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
    return TransformSetCKBitInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformClearCKEInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
    return TransformClearCKEInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformJumpInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
    return TransformJumpInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformWaitInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
    return TransformWaitInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformFenceInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
    return TransformFenceInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformTransLocMemToLocMSInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
    return TransformTransLocMemToLocMSInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformTransLocMSToLocMemInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
    return TransformTransLocMSToLocMemInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformTransLocMSToLocMSInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
    return TransformTransLocMSToLocMSInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformTransLocMemToLocMemInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
    return TransformTransLocMemToLocMemInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformTransMemInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
    return TransformTransMemInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformSyncXnWtxInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
    return TransformSyncXnWtxInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformSyncAtxInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
    return TransformSyncAtxInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformReduceAddInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
    return TransformReduceAddInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformReduceMaxInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
    return TransformReduceMaxInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

HcclResult TransformReduceMinInstr(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
    return TransformReduceMinInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
}

std::unordered_map<uint16_t, TransformInstrFunc> transformA6InstrSqeMap{
    {CcuRep::InstrHeader(LOAD_TYPE, A6_LOADSQEARGSTOXN_CODE).header, &TransformLoadSqeArgsToXnInstr},
    {CcuRep::InstrHeader(LOAD_TYPE, A6_LOADIMDTOX_CODE).header, &TransformLoadImdToXInstr},
    {CcuRep::InstrHeader(LOAD_TYPE, A6_LOADX_CODE).header, &TransformLoadXInstr},
    {CcuRep::InstrHeader(LOAD_TYPE, A6_STOREX_CODE).header, &TransformStoreXInstr},
    {CcuRep::InstrHeader(LOAD_TYPE, A6_CLEARX_CODE).header, &TransformClearXInstr},
    {CcuRep::InstrHeader(LOAD_TYPE, A6_NOP_CODE).header, &TransformNopInstr},
    {CcuRep::InstrHeader(LOAD_TYPE, A6_LOAD_CODE).header, &TransformLoadInstr},
    {CcuRep::InstrHeader(LOAD_TYPE, A6_STORE_CODE).header, &TransformStoreInstr},
    {CcuRep::InstrHeader(LOAD_TYPE, A6_ADD_CODE).header, &TransformAddInstr},
    {CcuRep::InstrHeader(LOAD_TYPE, A6_SUB_CODE).header, &TransformSubInstr},
    {CcuRep::InstrHeader(LOAD_TYPE, A6_MUL_CODE).header, &TransformMulInstr},
    {CcuRep::InstrHeader(LOAD_TYPE, A6_AND_CODE).header, &TransformANDInstr},
    {CcuRep::InstrHeader(LOAD_TYPE, A6_OR_CODE).header, &TransformORInstr},
    {CcuRep::InstrHeader(LOAD_TYPE, A6_NOT_CODE).header, &TransformNOTInstr},
    {CcuRep::InstrHeader(LOAD_TYPE, A6_XOR_CODE).header, &TransformXORInstr},
    {CcuRep::InstrHeader(LOAD_TYPE, A6_SHL_CODE).header, &TransformSHLInstr},
    {CcuRep::InstrHeader(LOAD_TYPE, A6_SHR_CODE).header, &TransformSHRInstr},
    {CcuRep::InstrHeader(LOAD_TYPE, A6_POPCNT_CODE).header, &TransformPopcntInstr},

    {CcuRep::InstrHeader(CTRL_TYPE, A6_LOOP_CODE).header, &TransformLoopInstr},
    {CcuRep::InstrHeader(CTRL_TYPE, A6_LOOPGROUP_CODE).header, &TransformLoopGroupInstr},
    {CcuRep::InstrHeader(CTRL_TYPE, A6_SETCKBIT_CODE).header, &TransformSetCKBitInstr},
    {CcuRep::InstrHeader(CTRL_TYPE, A6_CLEARCKE_CODE).header, &TransformClearCKEInstr},
    {CcuRep::InstrHeader(CTRL_TYPE, A6_JMP_CODE).header, &TransformJumpInstr},
    {CcuRep::InstrHeader(CTRL_TYPE, A6_WAIT_CODE).header, &TransformWaitInstr},
    {CcuRep::InstrHeader(CTRL_TYPE, A6_FENCE_CODE).header, &TransformFenceInstr},

    {CcuRep::InstrHeader(TRANS_TYPE, A6_TRANSLOCMEMTOLOCMS_CODE).header, &TransformTransLocMemToLocMSInstr},
    {CcuRep::InstrHeader(TRANS_TYPE, A6_TRANSLOCMSTOLOCMEM_CODE).header, &TransformTransLocMSToLocMemInstr},
    {CcuRep::InstrHeader(TRANS_TYPE, A6_TRANSLOCMSTOLOCMS_CODE).header, &TransformTransLocMSToLocMSInstr},
    {CcuRep::InstrHeader(TRANS_TYPE, A6_TRANSLOCMEMTOLOCMEM_CODE).header, &TransformTransLocMemToLocMemInstr},
    {CcuRep::InstrHeader(TRANS_TYPE, A6_TRANSMEM_CODE).header, &TransformTransMemInstr},
    {CcuRep::InstrHeader(TRANS_TYPE, A6_SYNCXNWTX_CODE).header, &TransformSyncXnWtxInstr},
    {CcuRep::InstrHeader(TRANS_TYPE, A6_SYNCATX_CODE).header, &TransformSyncAtxInstr},

    {CcuRep::InstrHeader(REDUCE_TYPE, A6_REDUCEADD_CODE).header, &TransformReduceAddInstr},
    {CcuRep::InstrHeader(REDUCE_TYPE, A6_REDUCEMAX_CODE).header, &TransformReduceMaxInstr},
    {CcuRep::InstrHeader(REDUCE_TYPE, A6_REDUCEMIN_CODE).header, &TransformReduceMinInstr}};

HcclResult ProcessLoopIns(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue,
    LoopGroupParamA6* loopGroupParam, uint32_t loopGroupIdx)
{
    uint16_t mode = instr->v2.loop.mode;
    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId{0};
    curCcuTask->GetDieId(queId, dieId);
    LoopA6 loop{};
    uint16_t xpId = GetXnId(instr->v2.loop.xpId, loopGroupParam);
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xpId, loop.loopXp.value));
    if (mode == 1) {
        // Xp中存储的是Loop Context ID，当mode为1时，在loop与自身预期值checkEntry相等时才进行展开。
        // 否则，直接跳过该loop。
        uint16_t wishCKEBit = instr->v2.loop.wishCKEBit;
        if (wishCKEBit != loop.loopXp.loopCtxId + A6_LOOP_BOARD_ENTRY) {
            isContinue = false;
            return HCCL_SUCCESS;
        }
    }
    uint16_t startInstrId = instr->v2.loop.startInstrId;
    uint16_t endInstrId = instr->v2.loop.endInstrId;
    uint16_t xnId = GetXnId(instr->v2.loop.xnId, loopGroupParam);
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xnId, loop.loopXn.value));

    uint16_t xmId = GetXnId(instr->v2.loop.xmId, loopGroupParam);
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xmId, loop.loopXm.value));

    loopGroupParam->loopXms.push_back(loop);

    HCCL_VM_INFO("debug...{}, {}", loopGroupIdx, curCcuTask->loopGroupInfo_.size());
    auto loopIdx = curCcuTask->loopIdx[loopGroupIdx]++;
    auto loopStart = AddLoopStartTask(queId, loopIdx, loopGroupIdx, curCcuTask); // loop前新增loopStart标记节点

    HCCL_VM_INFO("loop cnt = {}", static_cast<uint64_t>(loop.loopXm.loopIterCnt));
    hcomm::CcuRep::CcuInstrInfo& microCodeQue = curCcuTask->instrInfo[queId];
    for (u32 curLoopCnt = 0; curLoopCnt < loop.loopXm.loopIterCnt; curLoopCnt++) {
        loopGroupParam->curLoopCnt = curLoopCnt;
        HCCL_VM_INFO("loop cnt = {}, cur loop= {}", static_cast<uint64_t>(loop.loopXm.loopIterCnt), curLoopCnt);
        for (uint16_t insId = startInstrId; insId <= endInstrId; insId++) {
            // 获取当前要处理的指令
            const CcuRep::CcuInstr* instrVec = &microCodeQue.instrVec[insId - curCcuTask->startInstrIdInQue[queId]];
            HCCL_INFO("Current process instruction id = [%u]", insId);
            if (transformA6InstrSqeMap.count(instrVec->header.header) == 0) {
                HCCL_ERROR("ins type not supported [%hu]", instr->header.header);
                return HCCL_E_INTERNAL;
            }
            CHK_RET(transformA6InstrSqeMap[instrVec->header.header](
                instrVec, curCcuTask, queId, isContinue, loopGroupParam));
        }
    }
    auto loopEnd = AddLoopEndTask(queId, loopIdx, loopGroupIdx, curCcuTask); // loop后新增loopEnd标记节点

    HCCL_INFO("Loop From startInstrId[%u] to endInstrId[%u] with loopXn[%u]", startInstrId, endInstrId, xnId);
    return HCCL_SUCCESS;
}
} // namespace

namespace HcclSim {
InstructMapA6::InstructMapA6() { transformInstrSqeMap = transformA6InstrSqeMap; }

HcclResult InstructMapA6::Transform(
    const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
{
    auto it = transformInstrSqeMap.find(instr->header.header);
    if (it == transformInstrSqeMap.end()) {
        HCCL_ERROR("[A6] Unsupported: 0x%04x", instr->header.header);
        return HCCL_E_INTERNAL;
    }
    auto* param = static_cast<LoopGroupParamA6*>(loopParam);
    return it->second(instr, curCcuTask, queId, isContinue, param);
}
} // namespace HcclSim
