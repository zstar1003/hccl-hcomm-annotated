/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_microcode_v1.h"

#include <unordered_map>

#include "string_util.h"

namespace {
constexpr uint16_t LOAD_TYPE = 0x0;
constexpr uint16_t CTRL_TYPE = 0x1;
constexpr uint16_t TRANS_TYPE = 0x2;
constexpr uint16_t REDUCE_TYPE = 0x3;

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
} // namespace

namespace hcomm {
namespace CcuRep {

    // *GSAId = *sqeArgsId
    void LoadSqeArgsToGSAInstr(CcuInstr* instr, uint16_t gsaId, uint16_t sqeArgsId)
    {
        instr->header = InstrHeader(LOAD_TYPE, LOADSQEARGSTOGSA_CODE);
        instr->v1.loadSqeArgsToGSA.gsaId = gsaId;
        instr->v1.loadSqeArgsToGSA.sqeArgsId = sqeArgsId;
    }

    // *XnId = *sqeArgsId
    void LoadSqeArgsToXnInstr(CcuInstr* instr, uint16_t xnId, uint16_t sqeArgsId)
    {
        instr->header = InstrHeader(LOAD_TYPE, LOADSQEARGSTOXN_CODE);
        instr->v1.loadSqeArgsToXn.xnId = xnId;
        instr->v1.loadSqeArgsToXn.sqeArgsId = sqeArgsId;
    }

    // *GSAId = immediate
    void LoadImdToGSAInstr(CcuInstr* instr, uint16_t gsaId, uint64_t immediate)
    {
        instr->header = InstrHeader(LOAD_TYPE, LOADIMDTOGSA_CODE);
        instr->v1.loadImdToGSA.gsaId = gsaId;
        instr->v1.loadImdToGSA.immediate = immediate;
    }

    // *XnId = immediate，secFlag用于打印时判断immediate是否是敏感信息，如果是，请设置secFlag=CCU_LOAD_TO_XN_SEC_INFO
    void LoadImdToXnInstr(CcuInstr* instr, uint16_t xnId, uint64_t immediate, uint16_t secFlag)
    {
        instr->header = InstrHeader(LOAD_TYPE, LOADIMDTOXN_CODE);
        instr->v1.loadImdToXn.xnId = xnId;
        instr->v1.loadImdToXn.immediate = immediate;
        instr->v1.loadImdToXn.secFlag = secFlag;
    }

    // *GSAdId = *GSAmId + *XnId
    void LoadGSAXnInstr(CcuInstr* instr, uint16_t gsAdId, uint16_t gsAmId, uint16_t xnId)
    {
        instr->header = InstrHeader(LOAD_TYPE, LOADGSAXN_CODE);
        instr->v1.loadGSAXn.gsAdId = gsAdId;
        instr->v1.loadGSAXn.gsAmId = gsAmId;
        instr->v1.loadGSAXn.xnId = xnId;
    }

    // *GSAdId = *GSAmId + *GSAnId
    void LoadGSAGSAInstr(CcuInstr* instr, uint16_t gsAdId, uint16_t gsAmId, uint16_t gsAnId)
    {
        instr->header = InstrHeader(LOAD_TYPE, LOADGSAGSA_CODE);
        instr->v1.loadGSAGSA.gsAdId = gsAdId;
        instr->v1.loadGSAGSA.gsAmId = gsAmId;
        instr->v1.loadGSAGSA.gsAnId = gsAnId;
    }

    // *XdId = *XmId + *XnId
    void LoadXXInstr(CcuInstr* instr, uint16_t xdId, uint16_t xmId, uint16_t xnId)
    {
        instr->header = InstrHeader(LOAD_TYPE, LOADXX_CODE);
        instr->v1.loadXX.xdId = xdId;
        instr->v1.loadXX.xmId = xmId;
        instr->v1.loadXX.xnId = xnId;
    }

    // startInstrId ~ endInstrId之间的指令构成loop
    // Xn寄存器中的内容：LoopNum[61:55], RepeatNum[54:48], NoRepeatNum[47:41], LoopCtxId[40:33], Offset[32:13],
    // IterNum[12:0] loop执行IterNum次 loop每次执行, *GSA偏移为Offset loop在第LoopCtxId个LoopEngine上执行
    void LoopInstr(CcuInstr* instr, uint16_t startInstrId, uint16_t endInstrId, uint16_t xnId)
    {
        instr->header = InstrHeader(CTRL_TYPE, LOOP_CODE);
        instr->v1.loop.startInstrId = startInstrId;
        instr->v1.loop.endInstrId = endInstrId;
        instr->v1.loop.xnId = xnId;
    }

    // startLoopInstrId为LoopGroup所包含的Loop的起始地址
    // Xn寄存器中的内容：LoopNum[61:55], RepeatNum[54:48], NoRepeatNum[47:41], LoopCtxId[40:33], Offset[32:13],
    // IterNum[12:0] 不自动展开的Loop的个数：NoRepeatNum 自动展开的Loop的个数：RepeatNum 自动展开成LoopNum个Loop
    void
    LoopGroupInstr(CcuInstr* instr, uint16_t startLoopInstrId, uint16_t xnId, uint16_t xmId, uint16_t highPerfModeEn)
    {
        instr->header = InstrHeader(CTRL_TYPE, LOOPGROUP_CODE);
        instr->v1.loopGroup.startLoopInstrId = startLoopInstrId;
        instr->v1.loopGroup.xnId = xnId;
        instr->v1.loopGroup.xmId = xmId;
        instr->v1.loopGroup.highPerfModeEn = highPerfModeEn & 0x1;
    }

    // 后续函数中, 均需要wait到<waitCKEId, waitCKEMask>后, 再执行相关操作, 执行完之后再set<setCKEId, setCKEMask>
    // clearType = 1时, wait到之后需要对<waitCKEId, waitCKEMask>清零, 否则不清零

    // set<setCKEId, setCKEMask>
    void SetCKEInstr(
        CcuInstr* instr, uint16_t setCKEId, uint16_t setCKEMask, uint16_t waitCKEId, uint16_t waitCKEMask,
        uint16_t clearType)
    {
        instr->header = InstrHeader(CTRL_TYPE, SETCKE_CODE);
        instr->v1.setCKE.clearType = clearType & 0x1;
        instr->v1.setCKE.setCKEId = setCKEId;
        instr->v1.setCKE.setCKEMask = setCKEMask;
        instr->v1.setCKE.waitCKEId = waitCKEId;
        instr->v1.setCKE.waitCKEMask = waitCKEMask;
    }

    // clear<setCKEId, setCKEMask>
    void ClearCKEInstr(
        CcuInstr* instr, uint16_t clearCKEId, uint16_t clearMask, uint16_t waitCKEId, uint16_t waitCKEMask,
        uint16_t clearType)
    {
        instr->header = InstrHeader(CTRL_TYPE, CLEARCKE_CODE);
        instr->v1.clearCKE.clearType = clearType & 0x1;
        instr->v1.clearCKE.clearCKEId = clearCKEId;
        instr->v1.clearCKE.clearMask = clearMask;
        instr->v1.clearCKE.waitCKEId = waitCKEId;
        instr->v1.clearCKE.waitCKEMask = waitCKEMask;
    }

    void JumpInstr(CcuInstr* instr, uint16_t dstInstrXnId, uint16_t conditionXnId, uint64_t expectData)
    {
        instr->header = InstrHeader(CTRL_TYPE, JMP_CODE);
        instr->v1.jmp.dstInstrXnId = dstInstrXnId;
        instr->v1.jmp.conditionXnId = conditionXnId;
        instr->v1.jmp.expectData = expectData;
    }

    // 本端Memory传输到本端MS
    // locMSId: 本端MSId
    // <locGSAId, locXnId>: 本端Memory地址和Token
    // 数据长度: length
    void TransLocMemToLocMSInstr(
        CcuInstr* instr, uint16_t locMSId, uint16_t locGSAId, uint16_t locXnId, uint16_t lengthXnId, uint16_t channelId,
        uint16_t setCKEId, uint16_t setCKEMask, uint16_t waitCKEId, uint16_t waitCKEMask, uint16_t clearType,
        uint16_t lengthEn)
    {
        instr->header = InstrHeader(TRANS_TYPE, TRANSLOCMEMTOLOCMS_CODE);
        instr->v1.transLocMemToLocMS.locMSId = locMSId;
        instr->v1.transLocMemToLocMS.locGSAId = locGSAId;
        instr->v1.transLocMemToLocMS.locXnId = locXnId;
        instr->v1.transLocMemToLocMS.lengthXnId = lengthXnId;
        instr->v1.transLocMemToLocMS.channelId = channelId;
        instr->v1.transLocMemToLocMS.clearType = clearType & 0x1;
        instr->v1.transLocMemToLocMS.lengthEn = lengthEn & 0x1;
        instr->v1.transLocMemToLocMS.setCKEId = setCKEId;
        instr->v1.transLocMemToLocMS.setCKEMask = setCKEMask;
        instr->v1.transLocMemToLocMS.waitCKEId = waitCKEId;
        instr->v1.transLocMemToLocMS.waitCKEMask = waitCKEMask;
    }

    // 远端Memory传输到本端MS
    // locMSId: 本端MSId
    // <rmtGSAId, rmtXnId>: 远端Memory地址和Token
    // 数据长度: length
    // 路径: channelId
    void TransRmtMemToLocMSInstr(
        CcuInstr* instr, uint16_t locMSId, uint16_t rmtGSAId, uint16_t rmtXnId, uint16_t lengthXnId, uint16_t channelId,
        uint16_t setCKEId, uint16_t setCKEMask, uint16_t waitCKEId, uint16_t waitCKEMask, uint16_t clearType,
        uint16_t lengthEn)
    {
        instr->header = InstrHeader(TRANS_TYPE, TRANSRMTMEMTOLOCMS_CODE);
        instr->v1.transRmtMemToLocMS.locMSId = locMSId;
        instr->v1.transRmtMemToLocMS.rmtGSAId = rmtGSAId;
        instr->v1.transRmtMemToLocMS.rmtXnId = rmtXnId;
        instr->v1.transRmtMemToLocMS.lengthXnId = lengthXnId;
        instr->v1.transRmtMemToLocMS.channelId = channelId;
        instr->v1.transRmtMemToLocMS.clearType = clearType & 0x1;
        instr->v1.transRmtMemToLocMS.lengthEn = lengthEn & 0x1;
        instr->v1.transRmtMemToLocMS.setCKEId = setCKEId;
        instr->v1.transRmtMemToLocMS.setCKEMask = setCKEMask;
        instr->v1.transRmtMemToLocMS.waitCKEId = waitCKEId;
        instr->v1.transRmtMemToLocMS.waitCKEMask = waitCKEMask;
    }

    // 本端MS传输到本端Memory
    // <locGSAId, locXnId>: 本端Memory地址和Token
    // locMSId: 本端MSId
    // 数据长度: length
    void TransLocMSToLocMemInstr(
        CcuInstr* instr, uint16_t locGSAId, uint16_t locXnId, uint16_t locMSId, uint16_t lengthXnId, uint16_t channelId,
        uint16_t setCKEId, uint16_t setCKEMask, uint16_t waitCKEId, uint16_t waitCKEMask, uint16_t clearType,
        uint16_t lengthEn)
    {
        instr->header = InstrHeader(TRANS_TYPE, TRANSLOCMSTOLOCMEM_CODE);
        instr->v1.transLocMSToLocMem.locGSAId = locGSAId;
        instr->v1.transLocMSToLocMem.locXnId = locXnId;
        instr->v1.transLocMSToLocMem.locMSId = locMSId;
        instr->v1.transLocMSToLocMem.lengthXnId = lengthXnId;
        instr->v1.transLocMSToLocMem.channelId = channelId;
        instr->v1.transLocMSToLocMem.clearType = clearType & 0x1;
        instr->v1.transLocMSToLocMem.lengthEn = lengthEn & 0x1;
        instr->v1.transLocMSToLocMem.setCKEId = setCKEId;
        instr->v1.transLocMSToLocMem.setCKEMask = setCKEMask;
        instr->v1.transLocMSToLocMem.waitCKEId = waitCKEId;
        instr->v1.transLocMSToLocMem.waitCKEMask = waitCKEMask;
    }

    // 本端MS传输到远端Memory
    // locMSId: 本端MSId
    // <rmtGSAId, rmtXnId>: 远端Memory地址和Token
    // 数据长度: length
    // 路径: channelId
    void TransLocMSToRmtMemInstr(
        CcuInstr* instr, uint16_t rmtGSAId, uint16_t rmtXnId, uint16_t locMSId, uint16_t lengthXnId, uint16_t channelId,
        uint16_t setCKEId, uint16_t setCKEMask, uint16_t waitCKEId, uint16_t waitCKEMask, uint16_t clearType,
        uint16_t lengthEn)
    {
        instr->header = InstrHeader(TRANS_TYPE, TRANSLOCMSTORMTMEM_CODE);
        instr->v1.transLocMSToRmtMem.rmtGSAId = rmtGSAId;
        instr->v1.transLocMSToRmtMem.rmtXnId = rmtXnId;
        instr->v1.transLocMSToRmtMem.locMSId = locMSId;
        instr->v1.transLocMSToRmtMem.lengthXnId = lengthXnId;
        instr->v1.transLocMSToRmtMem.channelId = channelId;
        instr->v1.transLocMSToRmtMem.clearType = clearType & 0x1;
        instr->v1.transLocMSToRmtMem.lengthEn = lengthEn & 0x1;
        instr->v1.transLocMSToRmtMem.setCKEId = setCKEId;
        instr->v1.transLocMSToRmtMem.setCKEMask = setCKEMask;
        instr->v1.transLocMSToRmtMem.waitCKEId = waitCKEId;
        instr->v1.transLocMSToRmtMem.waitCKEMask = waitCKEMask;
    }

    // 远端MS传输到本端Memory
    // <locGSAId, locXnId>: 本端Memory地址和Token
    // rmtMSId: 远端MSId
    // 数据长度: length
    // 路径: channelId
    void TransRmtMSToLocMemInstr(
        CcuInstr* instr, uint16_t locGSAId, uint16_t locXnId, uint16_t rmtMSId, uint16_t lengthXnId, uint16_t channelId,
        uint16_t setCKEId, uint16_t setCKEMask, uint16_t waitCKEId, uint16_t waitCKEMask, uint16_t clearType,
        uint16_t lengthEn)
    {
        instr->header = InstrHeader(TRANS_TYPE, TRANSRMTMSTOLOCMEM_CODE);
        instr->v1.transRmtMSToLocMem.locGSAId = locGSAId;
        instr->v1.transRmtMSToLocMem.locXnId = locXnId;
        instr->v1.transRmtMSToLocMem.rmtMSId = rmtMSId;
        instr->v1.transRmtMSToLocMem.lengthXnId = lengthXnId;
        instr->v1.transRmtMSToLocMem.channelId = channelId;
        instr->v1.transRmtMSToLocMem.clearType = clearType & 0x1;
        instr->v1.transRmtMSToLocMem.lengthEn = lengthEn & 0x1;
        instr->v1.transRmtMSToLocMem.setCKEId = setCKEId;
        instr->v1.transRmtMSToLocMem.setCKEMask = setCKEMask;
        instr->v1.transRmtMSToLocMem.waitCKEId = waitCKEId;
        instr->v1.transRmtMSToLocMem.waitCKEMask = waitCKEMask;
    }

    // 本端MS传输到本端MS
    // dstMSId: 本端目的MSId
    // srcMSId: 本端源MSId
    // 数据长度: length
    void TransLocMSToLocMSInstr(
        CcuInstr* instr, uint16_t dstMSId, uint16_t srcMSId, uint16_t lengthXnId, uint16_t channelId, uint16_t setCKEId,
        uint16_t setCKEMask, uint16_t waitCKEId, uint16_t waitCKEMask, uint16_t clearType, uint16_t lengthEn)
    {
        instr->header = InstrHeader(TRANS_TYPE, TRANSLOCMSTOLOCMS_CODE);
        instr->v1.transLocMSToLocMS.dstMSId = dstMSId;
        instr->v1.transLocMSToLocMS.srcMSId = srcMSId;
        instr->v1.transLocMSToLocMS.lengthXnId = lengthXnId;
        instr->v1.transLocMSToLocMS.channelId = channelId;
        instr->v1.transLocMSToLocMS.clearType = clearType & 0x1;
        instr->v1.transLocMSToLocMS.lengthEn = lengthEn & 0x1;
        instr->v1.transLocMSToLocMS.setCKEId = setCKEId;
        instr->v1.transLocMSToLocMS.setCKEMask = setCKEMask;
        instr->v1.transLocMSToLocMS.waitCKEId = waitCKEId;
        instr->v1.transLocMSToLocMS.waitCKEMask = waitCKEMask;
    }

    // 远端MS传输到本端MS
    // locMSId: 本端MSId
    // rmtMSId: 远端MSId
    // 数据长度: length
    // 路径: channelId
    void TransRmtMSToLocMSInstr(
        CcuInstr* instr, uint16_t locMSId, uint16_t rmtMSId, uint16_t lengthXnId, uint16_t channelId, uint16_t setCKEId,
        uint16_t setCKEMask, uint16_t waitCKEId, uint16_t waitCKEMask, uint16_t clearType, uint16_t lengthEn)
    {
        instr->header = InstrHeader(TRANS_TYPE, TRANSRMTMSTOLOCMS_CODE);
        instr->v1.transRmtMSToLocMS.locMSId = locMSId;
        instr->v1.transRmtMSToLocMS.rmtMSId = rmtMSId;
        instr->v1.transRmtMSToLocMS.lengthXnId = lengthXnId;
        instr->v1.transRmtMSToLocMS.channelId = channelId;
        instr->v1.transRmtMSToLocMS.clearType = clearType & 0x1;
        instr->v1.transRmtMSToLocMS.lengthEn = lengthEn & 0x1;
        instr->v1.transRmtMSToLocMS.setCKEId = setCKEId;
        instr->v1.transRmtMSToLocMS.setCKEMask = setCKEMask;
        instr->v1.transRmtMSToLocMS.waitCKEId = waitCKEId;
        instr->v1.transRmtMSToLocMS.waitCKEMask = waitCKEMask;
    }

    void TransLocMSToRmtMSInstr(
        CcuInstr* instr, uint16_t rmtMSId, uint16_t locMSId, uint16_t lengthXnId, uint16_t channelId,
        uint16_t setRmtCKEId, uint16_t setRmtCKEMask, uint16_t setCKEId, uint16_t setCKEMask, uint16_t waitCKEId,
        uint16_t waitCKEMask, uint16_t clearType, uint16_t lengthEn)
    {
        instr->header = InstrHeader(TRANS_TYPE, TRANSLOCMSTORMTMS_CODE);
        instr->v1.transLocMSToRmtMS.rmtMSId = rmtMSId;
        instr->v1.transLocMSToRmtMS.locMSId = locMSId;
        instr->v1.transLocMSToRmtMS.lengthXnId = lengthXnId;
        instr->v1.transLocMSToRmtMS.channelId = channelId;
        instr->v1.transLocMSToRmtMS.setRmtCKEId = setRmtCKEId;
        instr->v1.transLocMSToRmtMS.setRmtCKEMask = setRmtCKEMask;

        instr->v1.transLocMSToRmtMS.clearType = clearType & 0x1;
        instr->v1.transLocMSToRmtMS.lengthEn = lengthEn & 0x1;
        instr->v1.transLocMSToRmtMS.setCKEId = setCKEId;
        instr->v1.transLocMSToRmtMS.setCKEMask = setCKEMask;
        instr->v1.transLocMSToRmtMS.waitCKEId = waitCKEId;
        instr->v1.transLocMSToRmtMS.waitCKEMask = waitCKEMask;
    }

    // 远端Memory传输到本端Memory
    // <locGSAId, locXnId>: 本端Memory地址和Token
    // <rmtGSAId, rmtXnId>: 远端Memory地址和Token
    // 数据长度: length
    // 路径: channelId
    void TransRmtMemToLocMemInstr(
        CcuInstr* instr, uint16_t locGSAId, uint16_t locXnId, uint16_t rmtGSAId, uint16_t rmtXnId, uint16_t lengthXnId,
        uint16_t channelId, uint16_t reduceDataType, uint16_t reduceOpCode, uint16_t setCKEId, uint16_t setCKEMask,
        uint16_t waitCKEId, uint16_t waitCKEMask, uint16_t clearType, uint16_t lengthEn, uint16_t reduceEn)
    {
        instr->header = InstrHeader(TRANS_TYPE, TRANSRMTMEMTOLOCMEM_CODE);
        instr->v1.transRmtMemToLocMem.locGSAId = locGSAId;
        instr->v1.transRmtMemToLocMem.locXnId = locXnId;
        instr->v1.transRmtMemToLocMem.rmtGSAId = rmtGSAId;
        instr->v1.transRmtMemToLocMem.rmtXnId = rmtXnId;
        instr->v1.transRmtMemToLocMem.lengthXnId = lengthXnId;
        instr->v1.transRmtMemToLocMem.channelId = channelId;
        instr->v1.transRmtMemToLocMem.udfType = 0;
        instr->v1.transRmtMemToLocMem.reduceDataType = reduceDataType & 0xf;
        instr->v1.transRmtMemToLocMem.reduceOpCode = reduceOpCode & 0xf;
        instr->v1.transRmtMemToLocMem.clearType = clearType & 0x1;
        instr->v1.transRmtMemToLocMem.lengthEn = lengthEn & 0x1;
        instr->v1.transRmtMemToLocMem.reduceEn = reduceEn & 0x1;
        instr->v1.transRmtMemToLocMem.setCKEId = setCKEId;
        instr->v1.transRmtMemToLocMem.setCKEMask = setCKEMask;
        instr->v1.transRmtMemToLocMem.waitCKEId = waitCKEId;
        instr->v1.transRmtMemToLocMem.waitCKEMask = waitCKEMask;
    }

    // 本端Memory传输到远端Memory
    // <rmtGSAId, rmtXnId>: 远端Memory地址和Token
    // <locGSAId, locXnId>: 本端Memory地址和Token
    // 数据长度: length
    // 路径: channelId
    void TransLocMemToRmtMemInstr(
        CcuInstr* instr, uint16_t rmtGSAId, uint16_t rmtXnId, uint16_t locGSAId, uint16_t locXnId, uint16_t lengthXnId,
        uint16_t channelId, uint16_t reduceDataType, uint16_t reduceOpCode, uint16_t setCKEId, uint16_t setCKEMask,
        uint16_t waitCKEId, uint16_t waitCKEMask, uint16_t clearType, uint16_t lengthEn, uint16_t reduceEn)
    {
        instr->header = InstrHeader(TRANS_TYPE, TRANSLOCMEMTORMTMEM_CODE);
        instr->v1.transLocMemToRmtMem.rmtGSAId = rmtGSAId;
        instr->v1.transLocMemToRmtMem.rmtXnId = rmtXnId;
        instr->v1.transLocMemToRmtMem.locGSAId = locGSAId;
        instr->v1.transLocMemToRmtMem.locXnId = locXnId;
        instr->v1.transLocMemToRmtMem.lengthXnId = lengthXnId;
        instr->v1.transLocMemToRmtMem.channelId = channelId;
        instr->v1.transLocMemToRmtMem.udfType = 0;
        instr->v1.transLocMemToRmtMem.reduceDataType = reduceDataType & 0xf;
        instr->v1.transLocMemToRmtMem.reduceOpCode = reduceOpCode & 0xf;
        instr->v1.transLocMemToRmtMem.clearType = clearType & 0x1;
        instr->v1.transLocMemToRmtMem.lengthEn = lengthEn & 0x1;
        instr->v1.transLocMemToRmtMem.reduceEn = reduceEn & 0x1;
        instr->v1.transLocMemToRmtMem.setCKEId = setCKEId;
        instr->v1.transLocMemToRmtMem.setCKEMask = setCKEMask;
        instr->v1.transLocMemToRmtMem.waitCKEId = waitCKEId;
        instr->v1.transLocMemToRmtMem.waitCKEMask = waitCKEMask;
    }

    void TransLocMemToLocMemInstr(
        CcuInstr* instr, uint16_t dstGSAId, uint16_t dstXnId, uint16_t srcGSAId, uint16_t srcXnId, uint16_t lengthXnId,
        uint16_t channelId, uint16_t setCKEId, uint16_t setCKEMask, uint16_t waitCKEId, uint16_t waitCKEMask,
        uint16_t clearType, uint16_t lengthEn)
    {
        instr->header = InstrHeader(TRANS_TYPE, TRANSLOCMEMTOLOCMEM_CODE);
        instr->v1.transLocMemToLocMem.dstGSAId = dstGSAId;
        instr->v1.transLocMemToLocMem.dstXnId = dstXnId;
        instr->v1.transLocMemToLocMem.srcGSAId = srcGSAId;
        instr->v1.transLocMemToLocMem.srcXnId = srcXnId;
        instr->v1.transLocMemToLocMem.lengthXnId = lengthXnId;
        instr->v1.transLocMemToLocMem.channelId = channelId;
        instr->v1.transLocMemToLocMem.clearType = clearType & 0x1;
        instr->v1.transLocMemToLocMem.lengthEn = lengthEn & 0x1;
        instr->v1.transLocMemToLocMem.setCKEId = setCKEId;
        instr->v1.transLocMemToLocMem.setCKEMask = setCKEMask;
        instr->v1.transLocMemToLocMem.waitCKEId = waitCKEId;
        instr->v1.transLocMemToLocMem.waitCKEMask = waitCKEMask;
    }

    // 本端CKE同步到远端CKE
    // rmtCKEId: 远端CKEId
    // locCKEId: 本端CKEId
    // locCKEMask: mask
    // 路径: channelId
    void SyncCKEInstr(
        CcuInstr* instr, uint16_t rmtCKEId, uint16_t locCKEId, uint16_t locCKEMask, uint16_t channelId,
        uint16_t setCKEId, uint16_t setCKEMask, uint16_t waitCKEId, uint16_t waitCKEMask, uint16_t clearType)
    {
        instr->header = InstrHeader(TRANS_TYPE, SYNCCKE_CODE);
        instr->v1.syncCKE.rmtCKEId = rmtCKEId;
        instr->v1.syncCKE.locCKEId = locCKEId;
        instr->v1.syncCKE.locCKEMask = locCKEMask;
        instr->v1.syncCKE.channelId = channelId;
        instr->v1.syncCKE.clearType = clearType & 0x1;
        instr->v1.syncCKE.setCKEId = setCKEId;
        instr->v1.syncCKE.setCKEMask = setCKEMask;
        instr->v1.syncCKE.waitCKEId = waitCKEId;
        instr->v1.syncCKE.waitCKEMask = waitCKEMask;
    }

    // 本端<locGSAId>同步到远端<rmtGSAId> 同步完成后, 置位远端的<setRmtCKEId, setRmtCKEMask>
    void SyncGSAInstr(
        CcuInstr* instr, uint16_t rmtGSAId, uint16_t locGSAId, uint16_t channelId, uint16_t setRmtCKEId,
        uint16_t setRmtCKEMask, uint16_t setCKEId, uint16_t setCKEMask, uint16_t waitCKEId, uint16_t waitCKEMask,
        uint16_t clearType)
    {
        instr->header = InstrHeader(TRANS_TYPE, SYNCGSA_CODE);
        instr->v1.syncGSA.rmtGSAId = rmtGSAId;
        instr->v1.syncGSA.locGSAId = locGSAId;
        instr->v1.syncGSA.channelId = channelId;
        instr->v1.syncGSA.setRmtCKEId = setRmtCKEId;
        instr->v1.syncGSA.setRmtCKEMask = setRmtCKEMask;
        instr->v1.syncGSA.clearType = clearType & 0x1;
        instr->v1.syncGSA.setCKEId = setCKEId;
        instr->v1.syncGSA.setCKEMask = setCKEMask;
        instr->v1.syncGSA.waitCKEId = waitCKEId;
        instr->v1.syncGSA.waitCKEMask = waitCKEMask;
    }

    // 本端<locInputXnId, locOutputXnId>同步到远端<rmtInputXnId, rmtOutputXnId> 同步完成后, 置位远端的<setRmtCKEId,
    // setRmtCKEMask>
    void SyncXnInstr(
        CcuInstr* instr, uint16_t rmtXnId, uint16_t locXnId, uint16_t channelId, uint16_t setRmtCKEId,
        uint16_t setRmtCKEMask, uint16_t setCKEId, uint16_t setCKEMask, uint16_t waitCKEId, uint16_t waitCKEMask,
        uint16_t clearType)
    {
        instr->header = InstrHeader(TRANS_TYPE, SYNCXN_CODE);
        instr->v1.syncXn.rmtXnId = rmtXnId;
        instr->v1.syncXn.locXnId = locXnId;
        instr->v1.syncXn.channelId = channelId;
        instr->v1.syncXn.setRmtCKEId = setRmtCKEId;
        instr->v1.syncXn.setRmtCKEMask = setRmtCKEMask;
        instr->v1.syncXn.clearType = clearType & 0x1;
        instr->v1.syncXn.setCKEId = setCKEId;
        instr->v1.syncXn.setCKEMask = setCKEMask;
        instr->v1.syncXn.waitCKEId = waitCKEId;
        instr->v1.syncXn.waitCKEMask = waitCKEMask;
    }

    // MSA~MSH Reduce到 MSA
    // count: 参与Reduce的MS数目
    // castEn: 输出是否截断
    // dataType: Reduce数据类型
    void AddInstr(
        CcuInstr* instr, uint16_t* msId, uint16_t count, uint16_t castEn, uint16_t dataType, uint16_t setCKEId,
        uint16_t setCKEMask, uint16_t waitCKEId, uint16_t waitCKEMask, uint16_t clearType, uint16_t XnIdLength)
    {
        count -= 2; // CCU指令中指定count数为实际参与运算的MS数减2
        instr->header = InstrHeader(REDUCE_TYPE, ADD_CODE);
        for (uint16_t index = 0; index < CCU_REDUCE_MAX_MS; index++) {
            instr->v1.add.msId[index] = msId[index];
        }
        instr->v1.add.XnIdLength = XnIdLength;
        instr->v1.add.count = count & 0x7;
        instr->v1.add.castEn = castEn & 0x3;
        instr->v1.add.dataType = dataType & 0x1f;
        instr->v1.add.clearType = clearType & 0x1;
        instr->v1.add.setCKEId = setCKEId;
        instr->v1.add.setCKEMask = setCKEMask;
        instr->v1.add.waitCKEId = waitCKEId;
        instr->v1.add.waitCKEMask = waitCKEMask;
    }

    void MaxInstr(
        CcuInstr* instr, uint16_t* msId, uint16_t count, uint16_t dataType, uint16_t setCKEId, uint16_t setCKEMask,
        uint16_t waitCKEId, uint16_t waitCKEMask, uint16_t clearType, uint16_t XnIdLength)
    {
        count -= 2; // CCU指令中指定count数为实际参与运算的MS数减2
        instr->header = InstrHeader(REDUCE_TYPE, MAX_CODE);
        for (uint16_t index = 0; index < CCU_REDUCE_MAX_MS; index++) {
            instr->v1.add.msId[index] = msId[index];
        }
        instr->v1.max.XnIdLength = XnIdLength;
        instr->v1.max.count = count & 0x7;
        instr->v1.max.dataType = dataType & 0x1f;
        instr->v1.max.clearType = clearType & 0x1;
        instr->v1.max.setCKEId = setCKEId;
        instr->v1.max.setCKEMask = setCKEMask;
        instr->v1.max.waitCKEId = waitCKEId;
        instr->v1.max.waitCKEMask = waitCKEMask;
    }

    void MinInstr(
        CcuInstr* instr, uint16_t* msId, uint16_t count, uint16_t dataType, uint16_t setCKEId, uint16_t setCKEMask,
        uint16_t waitCKEId, uint16_t waitCKEMask, uint16_t clearType, uint16_t XnIdLength)
    {
        count -= 2; // CCU指令中指定count数为实际参与运算的MS数减2
        instr->header = InstrHeader(REDUCE_TYPE, MIN_CODE);
        for (uint16_t index = 0; index < CCU_REDUCE_MAX_MS; index++) {
            instr->v1.add.msId[index] = msId[index];
        }
        instr->v1.max.XnIdLength = XnIdLength;
        instr->v1.min.count = count & 0x7;
        instr->v1.min.dataType = dataType & 0x1f;
        instr->v1.min.clearType = clearType & 0x1;
        instr->v1.min.setCKEId = setCKEId;
        instr->v1.min.setCKEMask = setCKEMask;
        instr->v1.min.waitCKEId = waitCKEId;
        instr->v1.min.waitCKEMask = waitCKEMask;
    }

    std::string ParseLoadSqeArgsToGSAInstr(const CcuInstr* instr)
    {
        uint16_t gsaId = instr->v1.loadSqeArgsToGSA.gsaId;
        uint16_t sqeArgsId = instr->v1.loadSqeArgsToGSA.sqeArgsId;

        return Hccl::StringFormat("Load SqeArg[%u] to GSA[%u]", sqeArgsId, gsaId);
    }

    static std::string ParseLoadSqeArgsToXnInstr(const CcuInstr* instr)
    {
        uint16_t xnId = instr->v1.loadSqeArgsToXn.xnId;
        uint16_t sqeArgsId = instr->v1.loadSqeArgsToXn.sqeArgsId;

        return Hccl::StringFormat("Load SqeArg[%u] to Xn[%u]", sqeArgsId, xnId);
    }

    static std::string ParseLoadImdToGSAInstr(const CcuInstr* instr)
    {
        uint16_t gsaId = instr->v1.loadImdToGSA.gsaId;
        uint64_t immediate = instr->v1.loadImdToGSA.immediate;

        return Hccl::StringFormat("Load immediate[%llu] to GSA[%u]", immediate, gsaId);
    }

    static std::string ParseLoadImdToXnInstr(const CcuInstr* instr)
    {
        uint16_t xnId = instr->v1.loadImdToXn.xnId;
        uint64_t immediate = instr->v1.loadImdToXn.immediate;

        if (instr->v1.loadImdToXn.secFlag == CCU_LOAD_TO_XN_SEC_INFO) {
            return Hccl::StringFormat("Load immediate[tokenInfo] to Xn[%u]", xnId);
        }

        return Hccl::StringFormat("Load immediate[%llu] to Xn[%u]", immediate, xnId);
    }

    static std::string ParseLoadGSAXnInstr(const CcuInstr* instr)
    {
        uint16_t gsAdId = instr->v1.loadGSAXn.gsAdId;
        uint16_t gsAmId = instr->v1.loadGSAXn.gsAmId;
        uint16_t xnId = instr->v1.loadGSAXn.xnId;

        return Hccl::StringFormat("Load GSA[%u] + Xn[%u] to GSA[%u]", gsAmId, xnId, gsAdId);
    }

    static std::string ParseLoadGSAGSAInstr(const CcuInstr* instr)
    {
        uint16_t gsAdId = instr->v1.loadGSAGSA.gsAdId;
        uint16_t gsAmId = instr->v1.loadGSAGSA.gsAmId;
        uint16_t gsAnId = instr->v1.loadGSAGSA.gsAnId;

        return Hccl::StringFormat("Load GSA[%u] + GSA[%u] to GSA[%u]", gsAmId, gsAnId, gsAdId);
    }

    static std::string ParseLoadXXInstr(const CcuInstr* instr)
    {
        uint16_t xdId = instr->v1.loadXX.xdId;
        uint16_t xmId = instr->v1.loadXX.xmId;
        uint16_t xnId = instr->v1.loadXX.xnId;

        return Hccl::StringFormat("Load Xn[%u] + Xn[%u] to Xn[%u]", xmId, xnId, xdId);
    }

    static std::string ParseLoopInstr(const CcuInstr* instr)
    {
        uint16_t startInstrId = instr->v1.loop.startInstrId;
        uint16_t endInstrId = instr->v1.loop.endInstrId;
        uint16_t xnId = instr->v1.loop.xnId;

        return Hccl::StringFormat(
            "Loop From startInstrId[%u] to endInstrId[%u] with loopXn[%u]", startInstrId, endInstrId, xnId);
    }

    static std::string ParseLoopGroupInstr(const CcuInstr* instr)
    {
        uint16_t startLoopInstrId = instr->v1.loopGroup.startLoopInstrId;
        uint16_t xnId = instr->v1.loopGroup.xnId;
        uint16_t xmId = instr->v1.loopGroup.xmId;
        uint16_t highPerfModeEn = instr->v1.loopGroup.highPerfModeEn;

        return Hccl::StringFormat(
            "LoopGroup From startLoopInstrId[%u] with loopGroupXn[%u], offsetXn[%u] and highPerfModeEn[%u]",
            startLoopInstrId, xnId, xmId, highPerfModeEn);
    }

    static std::string ParseSetCKEInstr(const CcuInstr* instr)
    {
        uint16_t clearType = instr->v1.setCKE.clearType;
        uint16_t setCKEId = instr->v1.setCKE.setCKEId;
        uint16_t setCKEMask = instr->v1.setCKE.setCKEMask;
        uint16_t waitCKEId = instr->v1.setCKE.waitCKEId;
        uint16_t waitCKEMask = instr->v1.setCKE.waitCKEMask;

        return Hccl::StringFormat(
            "Wait CKE[%u:%04x], Set CKE[%u:%04x], clearType[%u]", waitCKEId, waitCKEMask, setCKEId, setCKEMask,
            clearType);
    }

    static std::string ParseClearCKEInstr(const CcuInstr* instr)
    {
        uint16_t clearType = instr->v1.clearCKE.clearType;
        uint16_t clearCKEId = instr->v1.clearCKE.clearCKEId;
        uint16_t clearMask = instr->v1.clearCKE.clearMask;
        uint16_t waitCKEId = instr->v1.clearCKE.waitCKEId;
        uint16_t waitCKEMask = instr->v1.clearCKE.waitCKEMask;

        return Hccl::StringFormat(
            "Wait CKE[%u:%04x], Clear CKE[%u:%04x], clearType[%u]", waitCKEId, waitCKEMask, clearCKEId, clearMask,
            clearType);
    }

    static std::string ParseJumpInstr(const CcuInstr* instr)
    {
        uint16_t dstInstrXnId = instr->v1.jmp.dstInstrXnId;
        uint16_t conditionXnId = instr->v1.jmp.conditionXnId;
        uint64_t expectData = instr->v1.jmp.expectData;

        return Hccl::StringFormat(
            "When conditionXn[%u] not equal to expectData[%llu], Jump To InstrIdXn[%u]", conditionXnId, expectData,
            dstInstrXnId);
    }

    static std::string ParseTransLocMemToLocMSInstr(const CcuInstr* instr)
    {
        uint16_t locMSId = instr->v1.transLocMemToLocMS.locMSId;
        uint16_t locGSAId = instr->v1.transLocMemToLocMS.locGSAId;
        uint16_t locXnId = instr->v1.transLocMemToLocMS.locXnId;
        uint16_t lengthXnId = instr->v1.transLocMemToLocMS.lengthXnId;
        uint16_t channelId = instr->v1.transLocMemToLocMS.channelId;
        uint16_t clearType = instr->v1.transLocMemToLocMS.clearType;
        uint16_t lengthEn = instr->v1.transLocMemToLocMS.lengthEn;
        uint16_t setCKEId = instr->v1.transLocMemToLocMS.setCKEId;
        uint16_t setCKEMask = instr->v1.transLocMemToLocMS.setCKEMask;
        uint16_t waitCKEId = instr->v1.transLocMemToLocMS.waitCKEId;
        uint16_t waitCKEMask = instr->v1.transLocMemToLocMS.waitCKEMask;

        return Hccl::StringFormat(
            "Wait CKE[%u:%04x], Trans LocMem[%u:%u] To LocMS[%u:%u] With LengthXn[%u] Use Channel[%u], Set "
            "CKE[%u:%04x], clearType[%u], lengthEn[%u]",
            waitCKEId, waitCKEMask, locGSAId, locXnId, locMSId / 0x8000, locMSId % 0x8000, lengthXnId, channelId,
            setCKEId, setCKEMask, clearType, lengthEn);
    }

    static std::string ParseTransRmtMemToLocMSInstr(const CcuInstr* instr)
    {
        uint16_t locMSId = instr->v1.transRmtMemToLocMS.locMSId;
        uint16_t rmtGSAId = instr->v1.transRmtMemToLocMS.rmtGSAId;
        uint16_t rmtXnId = instr->v1.transRmtMemToLocMS.rmtXnId;
        uint16_t lengthXnId = instr->v1.transRmtMemToLocMS.lengthXnId;
        uint16_t channelId = instr->v1.transRmtMemToLocMS.channelId;
        uint16_t clearType = instr->v1.transRmtMemToLocMS.clearType;
        uint16_t lengthEn = instr->v1.transRmtMemToLocMS.lengthEn;
        uint16_t setCKEId = instr->v1.transRmtMemToLocMS.setCKEId;
        uint16_t setCKEMask = instr->v1.transRmtMemToLocMS.setCKEMask;
        uint16_t waitCKEId = instr->v1.transRmtMemToLocMS.waitCKEId;
        uint16_t waitCKEMask = instr->v1.transRmtMemToLocMS.waitCKEMask;

        return Hccl::StringFormat(
            "Wait CKE[%u:%04x], Trans RmtMem[%u:%u] To LocMS[%u:%u] With LengthXn[%u] Use Channel[%u], Set "
            "CKE[%u:%04x], clearType[%u], lengthEn[%u]",
            waitCKEId, waitCKEMask, rmtGSAId, rmtXnId, locMSId / 0x8000, locMSId % 0x8000, lengthXnId, channelId,
            setCKEId, setCKEMask, clearType, lengthEn);
    }

    static std::string ParseTransLocMSToLocMemInstr(const CcuInstr* instr)
    {
        uint16_t locGSAId = instr->v1.transLocMSToLocMem.locGSAId;
        uint16_t locXnId = instr->v1.transLocMSToLocMem.locXnId;
        uint16_t locMSId = instr->v1.transLocMSToLocMem.locMSId;
        uint16_t lengthXnId = instr->v1.transLocMSToLocMem.lengthXnId;
        uint16_t channelId = instr->v1.transLocMSToLocMem.channelId;
        uint16_t clearType = instr->v1.transLocMSToLocMem.clearType;
        uint16_t lengthEn = instr->v1.transLocMSToLocMem.lengthEn;
        uint16_t setCKEId = instr->v1.transLocMSToLocMem.setCKEId;
        uint16_t setCKEMask = instr->v1.transLocMSToLocMem.setCKEMask;
        uint16_t waitCKEId = instr->v1.transLocMSToLocMem.waitCKEId;
        uint16_t waitCKEMask = instr->v1.transLocMSToLocMem.waitCKEMask;

        return Hccl::StringFormat(
            "Wait CKE[%u:%04x], Trans LocMS[%u:%u] To LocMem[%u:%u] With LengthXn[%u] Use Channel[%u], Set "
            "CKE[%u:%04x], clearType[%u], lengthEn[%u]",
            waitCKEId, waitCKEMask, locMSId / 0x8000, locMSId % 0x8000, locGSAId, locXnId, lengthXnId, channelId,
            setCKEId, setCKEMask, clearType, lengthEn);
    }

    static std::string ParseTransLocMSToRmtMemInstr(const CcuInstr* instr)
    {
        uint16_t rmtGSAId = instr->v1.transLocMSToRmtMem.rmtGSAId;
        uint16_t rmtXnId = instr->v1.transLocMSToRmtMem.rmtXnId;
        uint16_t locMSId = instr->v1.transLocMSToRmtMem.locMSId;
        uint16_t lengthXnId = instr->v1.transLocMSToRmtMem.lengthXnId;
        uint16_t channelId = instr->v1.transLocMSToRmtMem.channelId;
        uint16_t clearType = instr->v1.transLocMSToRmtMem.clearType;
        uint16_t lengthEn = instr->v1.transLocMSToRmtMem.lengthEn;
        uint16_t setCKEId = instr->v1.transLocMSToRmtMem.setCKEId;
        uint16_t setCKEMask = instr->v1.transLocMSToRmtMem.setCKEMask;
        uint16_t waitCKEId = instr->v1.transLocMSToRmtMem.waitCKEId;
        uint16_t waitCKEMask = instr->v1.transLocMSToRmtMem.waitCKEMask;

        return Hccl::StringFormat(
            "Wait CKE[%u:%04x], Trans LocMS[%u:%u] To RmtMem[%u:%u] With LengthXn[%u] Use Channel[%u], Set "
            "CKE[%u:%04x], clearType[%u], lengthEn[%u]",
            waitCKEId, waitCKEMask, locMSId / 0x8000, locMSId % 0x8000, rmtGSAId, rmtXnId, lengthXnId, channelId,
            setCKEId, setCKEMask, clearType, lengthEn);
    }

    static std::string ParseTransRmtMSToLocMemInstr(const CcuInstr* instr)
    {
        uint16_t locGSAId = instr->v1.transRmtMSToLocMem.locGSAId;
        uint16_t locXnId = instr->v1.transRmtMSToLocMem.locXnId;
        uint16_t rmtMSId = instr->v1.transRmtMSToLocMem.rmtMSId;
        uint16_t lengthXnId = instr->v1.transRmtMSToLocMem.lengthXnId;
        uint16_t channelId = instr->v1.transRmtMSToLocMem.channelId;
        uint16_t clearType = instr->v1.transRmtMSToLocMem.clearType;
        uint16_t lengthEn = instr->v1.transRmtMSToLocMem.lengthEn;
        uint16_t setCKEId = instr->v1.transRmtMSToLocMem.setCKEId;
        uint16_t setCKEMask = instr->v1.transRmtMSToLocMem.setCKEMask;
        uint16_t waitCKEId = instr->v1.transRmtMSToLocMem.waitCKEId;
        uint16_t waitCKEMask = instr->v1.transRmtMSToLocMem.waitCKEMask;

        return Hccl::StringFormat(
            "Wait CKE[%u:%04x], Trans RmtMS[%u:%u] To LocMem[%u:%u] With LengthXn[%u] Use Channel[%u], Set "
            "CKE[%u:%04x], clearType[%u], lengthEn[%u]",
            waitCKEId, waitCKEMask, rmtMSId / 0x8000, rmtMSId % 0x8000, locGSAId, locXnId, lengthXnId, channelId,
            setCKEId, setCKEMask, clearType, lengthEn);
    }

    static std::string ParseTransLocMSToLocMSInstr(const CcuInstr* instr)
    {
        uint16_t dstMSId = instr->v1.transLocMSToLocMS.dstMSId;
        uint16_t srcMSId = instr->v1.transLocMSToLocMS.srcMSId;
        uint16_t lengthXnId = instr->v1.transLocMSToLocMS.lengthXnId;
        uint16_t channelId = instr->v1.transLocMSToLocMS.channelId;
        uint16_t clearType = instr->v1.transLocMSToLocMS.clearType;
        uint16_t lengthEn = instr->v1.transLocMSToLocMS.lengthEn;
        uint16_t setCKEId = instr->v1.transLocMSToLocMS.setCKEId;
        uint16_t setCKEMask = instr->v1.transLocMSToLocMS.setCKEMask;
        uint16_t waitCKEId = instr->v1.transLocMSToLocMS.waitCKEId;
        uint16_t waitCKEMask = instr->v1.transLocMSToLocMS.waitCKEMask;

        return Hccl::StringFormat(
            "Wait CKE[%u:%04x], Trans LocMS[%u:%u] To LocMS[%u:%u] With LengthXn[%u] Use Channel[%u], "
            "Set CKE[%u:%04x], "
            "clearType[%u], lengthEn[%u]",
            waitCKEId, waitCKEMask, srcMSId / 0x8000, srcMSId % 0x8000, dstMSId / 0x8000, dstMSId % 0x8000, lengthXnId,
            channelId, setCKEId, setCKEMask, clearType, lengthEn);
    }

    static std::string ParseTransRmtMSToLocMSInstr(const CcuInstr* instr)
    {
        uint16_t locMSId = instr->v1.transRmtMSToLocMS.locMSId;
        uint16_t rmtMSId = instr->v1.transRmtMSToLocMS.rmtMSId;
        uint16_t lengthXnId = instr->v1.transRmtMSToLocMS.lengthXnId;
        uint16_t channelId = instr->v1.transRmtMSToLocMS.channelId;
        uint16_t clearType = instr->v1.transRmtMSToLocMS.clearType;
        uint16_t lengthEn = instr->v1.transRmtMSToLocMS.lengthEn;
        uint16_t setCKEId = instr->v1.transRmtMSToLocMS.setCKEId;
        uint16_t setCKEMask = instr->v1.transRmtMSToLocMS.setCKEMask;
        uint16_t waitCKEId = instr->v1.transRmtMSToLocMS.waitCKEId;
        uint16_t waitCKEMask = instr->v1.transRmtMSToLocMS.waitCKEMask;

        return Hccl::StringFormat(
            "Wait CKE[%u:%04x], Trans RmtMS[%u:%u] To LocMS[%u:%u] With LengthXn[%u] Use Channel[%u], "
            "Set CKE[%u:%04x], "
            "clearType[%u], lengthEn[%u]",
            waitCKEId, waitCKEMask, rmtMSId / 0x8000, rmtMSId % 0x8000, locMSId / 0x8000, locMSId % 0x8000, lengthXnId,
            channelId, setCKEId, setCKEMask, clearType, lengthEn);
    }

    static std::string ParseTransLocMSToRmtMSInstr(const CcuInstr* instr)
    {
        uint16_t rmtMSId = instr->v1.transLocMSToRmtMS.rmtMSId;
        uint16_t locMSId = instr->v1.transLocMSToRmtMS.locMSId;
        uint16_t lengthXnId = instr->v1.transLocMSToRmtMS.lengthXnId;
        uint16_t channelId = instr->v1.transLocMSToRmtMS.channelId;
        uint16_t setRmtCKEId = instr->v1.transLocMSToRmtMS.setRmtCKEId;
        uint16_t setRmtCKEMask = instr->v1.transLocMSToRmtMS.setRmtCKEMask;

        uint16_t clearType = instr->v1.transLocMSToRmtMS.clearType;
        uint16_t lengthEn = instr->v1.transLocMSToRmtMS.lengthEn;
        uint16_t setCKEId = instr->v1.transLocMSToRmtMS.setCKEId;
        uint16_t setCKEMask = instr->v1.transLocMSToRmtMS.setCKEMask;
        uint16_t waitCKEId = instr->v1.transLocMSToRmtMS.waitCKEId;
        uint16_t waitCKEMask = instr->v1.transLocMSToRmtMS.waitCKEMask;

        return Hccl::StringFormat(
            "Wait CKE[%u:%04x], Trans LocMS[%u:%u] To RmtMS[%u:%u] With LengthXn[%u] Use Channel[%u], Set "
            "RmtCKE[%u:%04x], Set CKE[%u:%04x], clearType[%u], lengthEn[%u]",
            waitCKEId, waitCKEMask, locMSId / 0x8000, locMSId % 0x8000, rmtMSId / 0x8000, rmtMSId % 0x8000, lengthXnId,
            channelId, setRmtCKEId, setRmtCKEMask, setCKEId, setCKEMask, clearType, lengthEn);
    }

    static std::string ParseTransRmtMemToLocMemInstr(const CcuInstr* instr)
    {
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

        return Hccl::StringFormat(
            "Wait CKE[%u:%04x], Trans RmtMem[%u:%u] To LocMem[%u:%u] With LengthXn[%u] Use Channel[%u], Set "
            "CKE[%u:%04x], clearType[%u], lengthEn[%u], DataType[%u], ReduceType[%u] reduceEn[%u]",
            waitCKEId, waitCKEMask, rmtGSAId, rmtXnId, locGSAId, locXnId, lengthXnId, channelId, setCKEId, setCKEMask,
            clearType, lengthEn, reduceDataType, reduceOpCode, reduceEn);
    }

    static std::string ParseTransLocMemToRmtMemInstr(const CcuInstr* instr)
    {
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

        return Hccl::StringFormat(
            "Wait CKE[%u:%04x], Trans LocMem[%u:%u] To RmtMem[%u:%u] With LengthXn[%u] Use Channel[%u], Set "
            "CKE[%u:%04x], clearType[%u], lengthEn[%u], DataType[%u], ReduceType[%u] reduceEn[%u]",
            waitCKEId, waitCKEMask, locGSAId, locXnId, rmtGSAId, rmtXnId, lengthXnId, channelId, setCKEId, setCKEMask,
            clearType, lengthEn, reduceDataType, reduceOpCode, reduceEn);
    }

    static std::string ParseTransLocMemToLocMemInstr(const CcuInstr* instr)
    {
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

        return Hccl::StringFormat(
            "Wait CKE[%u:%04x], Trans LocMem[%u:%u] To LocMem[%u:%u] With LengthXn[%u] Use Channel[%u], Set "
            "CKE[%u:%04x], clearType[%u], lengthEn[%u]",
            waitCKEId, waitCKEMask, srcGSAId, srcXnId, dstGSAId, dstXnId, lengthXnId, channelId, setCKEId, setCKEMask,
            clearType, lengthEn);
    }

    static std::string ParseSyncCKEInstr(const CcuInstr* instr)
    {
        uint16_t rmtCKEId = instr->v1.syncCKE.rmtCKEId;
        uint16_t locCKEId = instr->v1.syncCKE.locCKEId;
        uint16_t locCKEMask = instr->v1.syncCKE.locCKEMask;
        uint16_t channelId = instr->v1.syncCKE.channelId;
        uint16_t clearType = instr->v1.syncCKE.clearType;
        uint16_t setCKEId = instr->v1.syncCKE.setCKEId;
        uint16_t setCKEMask = instr->v1.syncCKE.setCKEMask;
        uint16_t waitCKEId = instr->v1.syncCKE.waitCKEId;
        uint16_t waitCKEMask = instr->v1.syncCKE.waitCKEMask;

        return Hccl::StringFormat(
            "Wait CKE[%u:%04x], Sync LocCKE[%u:%04x] To rmtCKE[%u:%04x] Use Channel[%u], Set "
            "CKE[%u:%04x], clearType[%u]",
            waitCKEId, waitCKEMask, locCKEId, locCKEMask, rmtCKEId, locCKEMask, channelId, setCKEId, setCKEMask,
            clearType);
    }

    static std::string ParseSyncGSAInstr(const CcuInstr* instr)
    {
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

        return Hccl::StringFormat(
            "Wait CKE[%u:%04x], Sync locGSAId[%u] To rmtGSAId[%u] Use Channel[%u], Set rmtCKE[%u:%04x], Set "
            "CKE[%u:%04x], clearType[%u]",
            waitCKEId, waitCKEMask, locGSAId, rmtGSAId, channelId, setRmtCKEId, setRmtCKEMask, setCKEId, setCKEMask,
            clearType);
    }

    static std::string ParseSyncXnInstr(const CcuInstr* instr)
    {
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

        return Hccl::StringFormat(
            "Wait CKE[%u:%04x], Sync locXnId[%u] To rmtXnId[%u] Use Channel[%u], Set rmtCKE[%u:%04x], Set "
            "CKE[%u:%04x], clearType[%u]",
            waitCKEId, waitCKEMask, locXnId, rmtXnId, channelId, setRmtCKEId, setRmtCKEMask, setCKEId, setCKEMask,
            clearType);
    }

    static std::string ParseMSList(const CcuInstr* instr)
    {
        uint16_t msId[CCU_REDUCE_MAX_MS];
        uint16_t count = instr->v1.add.count;
        for (uint16_t index = 0; index < CCU_REDUCE_MAX_MS; index++) {
            msId[index] = instr->v1.add.msId[index];
        }

        // CCU指令中指定count数为实际参与运算的MS数减2，因此当count +
        // 2大于CCU_REDUCE_MAX_MS时，说明MS列表中的MS数量超过了CCU指令的最大支持数量，此时无法正确解析MS列表，直接返回"MS[]"
        if (count + 2 > CCU_REDUCE_MAX_MS) {
            return "MS[]";
        }

        std::string res = "MS[";
        for (uint16_t i = 0; i < count + 2; i++) { // 循环范围 0~count + 2
            if (i == count + 1) {
                res += std::to_string(msId[i] / 0x8000) + ":" + std::to_string(msId[i] % 0x8000) + "]";
            } else {
                res += std::to_string(msId[i] / 0x8000) + ":" + std::to_string(msId[i] % 0x8000) + ", ";
            }
        }
        return res;
    }

    static std::string ParseAddInstr(const CcuInstr* instr)
    {
        uint16_t count = instr->v1.add.count;
        uint16_t castEn = instr->v1.add.castEn;
        uint16_t dataType = instr->v1.add.dataType;
        uint16_t clearType = instr->v1.add.clearType;
        uint16_t setCKEId = instr->v1.add.setCKEId;
        uint16_t setCKEMask = instr->v1.add.setCKEMask;
        uint16_t waitCKEId = instr->v1.add.waitCKEId;
        uint16_t waitCKEMask = instr->v1.add.waitCKEMask;

        return Hccl::StringFormat(
            "Wait CKE[%u:%04x], Add %s with Count[%u], DataType[%u] and CastEn[%u], Set CKE[%u:%04x], clearType[%u]",
            waitCKEId, waitCKEMask, ParseMSList(instr).c_str(), count, dataType, castEn, setCKEId, setCKEMask,
            clearType);
    }

    static std::string ParseMaxInstr(const CcuInstr* instr)
    {
        uint16_t count = instr->v1.max.count;
        uint16_t dataType = instr->v1.max.dataType;
        uint16_t clearType = instr->v1.max.clearType;
        uint16_t setCKEId = instr->v1.max.setCKEId;
        uint16_t setCKEMask = instr->v1.max.setCKEMask;
        uint16_t waitCKEId = instr->v1.max.waitCKEId;
        uint16_t waitCKEMask = instr->v1.max.waitCKEMask;

        return Hccl::StringFormat(
            "Wait CKE[%u:%04x], Max %s with Count[%u], DataType[%u], Set CKE[%u:%04x], clearType[%u]", waitCKEId,
            waitCKEMask, ParseMSList(instr).c_str(), count, dataType, setCKEId, setCKEMask, clearType);
    }

    static std::string ParseMinInstr(const CcuInstr* instr)
    {
        uint16_t count = instr->v1.min.count;
        uint16_t dataType = instr->v1.min.dataType;
        uint16_t clearType = instr->v1.min.clearType;
        uint16_t setCKEId = instr->v1.min.setCKEId;
        uint16_t setCKEMask = instr->v1.min.setCKEMask;
        uint16_t waitCKEId = instr->v1.min.waitCKEId;
        uint16_t waitCKEMask = instr->v1.min.waitCKEMask;

        return Hccl::StringFormat(
            "Wait CKE[%u:%04x], Min %s with Count[%u], DataType[%u], Set CKE[%u:%04x], clearType[%u]", waitCKEId,
            waitCKEMask, ParseMSList(instr).c_str(), count, dataType, setCKEId, setCKEMask, clearType);
    }

    using ParseInstrFunc = std::string (*)(const CcuInstr*);

    static std::unordered_map<uint16_t, ParseInstrFunc> g_parseInstrSqeMap = {
        {InstrHeader(LOAD_TYPE, LOADSQEARGSTOGSA_CODE).header, &ParseLoadSqeArgsToGSAInstr},
        {InstrHeader(LOAD_TYPE, LOADSQEARGSTOXN_CODE).header, &ParseLoadSqeArgsToXnInstr},
        {InstrHeader(LOAD_TYPE, LOADIMDTOGSA_CODE).header, &ParseLoadImdToGSAInstr},
        {InstrHeader(LOAD_TYPE, LOADIMDTOXN_CODE).header, &ParseLoadImdToXnInstr},
        {InstrHeader(LOAD_TYPE, LOADGSAXN_CODE).header, &ParseLoadGSAXnInstr},
        {InstrHeader(LOAD_TYPE, LOADGSAGSA_CODE).header, &ParseLoadGSAGSAInstr},
        {InstrHeader(LOAD_TYPE, LOADXX_CODE).header, &ParseLoadXXInstr},
        {InstrHeader(CTRL_TYPE, LOOP_CODE).header, &ParseLoopInstr},
        {InstrHeader(CTRL_TYPE, LOOPGROUP_CODE).header, &ParseLoopGroupInstr},
        {InstrHeader(CTRL_TYPE, SETCKE_CODE).header, &ParseSetCKEInstr},
        {InstrHeader(CTRL_TYPE, CLEARCKE_CODE).header, &ParseClearCKEInstr},
        {InstrHeader(CTRL_TYPE, JMP_CODE).header, &ParseJumpInstr},
        {InstrHeader(TRANS_TYPE, TRANSLOCMEMTOLOCMS_CODE).header, &ParseTransLocMemToLocMSInstr},
        {InstrHeader(TRANS_TYPE, TRANSRMTMEMTOLOCMS_CODE).header, &ParseTransRmtMemToLocMSInstr},
        {InstrHeader(TRANS_TYPE, TRANSLOCMSTOLOCMEM_CODE).header, &ParseTransLocMSToLocMemInstr},
        {InstrHeader(TRANS_TYPE, TRANSLOCMSTORMTMEM_CODE).header, &ParseTransLocMSToRmtMemInstr},
        {InstrHeader(TRANS_TYPE, TRANSRMTMSTOLOCMEM_CODE).header, &ParseTransRmtMSToLocMemInstr},
        {InstrHeader(TRANS_TYPE, TRANSLOCMSTOLOCMS_CODE).header, &ParseTransLocMSToLocMSInstr},
        {InstrHeader(TRANS_TYPE, TRANSRMTMSTOLOCMS_CODE).header, &ParseTransRmtMSToLocMSInstr},
        {InstrHeader(TRANS_TYPE, TRANSLOCMSTORMTMS_CODE).header, &ParseTransLocMSToRmtMSInstr},
        {InstrHeader(TRANS_TYPE, TRANSRMTMEMTOLOCMEM_CODE).header, &ParseTransRmtMemToLocMemInstr},
        {InstrHeader(TRANS_TYPE, TRANSLOCMEMTORMTMEM_CODE).header, &ParseTransLocMemToRmtMemInstr},
        {InstrHeader(TRANS_TYPE, TRANSLOCMEMTOLOCMEM_CODE).header, &ParseTransLocMemToLocMemInstr},
        {InstrHeader(TRANS_TYPE, SYNCCKE_CODE).header, &ParseSyncCKEInstr},
        {InstrHeader(TRANS_TYPE, SYNCGSA_CODE).header, &ParseSyncGSAInstr},
        {InstrHeader(TRANS_TYPE, SYNCXN_CODE).header, &ParseSyncXnInstr},
        {InstrHeader(REDUCE_TYPE, ADD_CODE).header, &ParseAddInstr},
        {InstrHeader(REDUCE_TYPE, MAX_CODE).header, &ParseMaxInstr},
        {InstrHeader(REDUCE_TYPE, MIN_CODE).header, &ParseMinInstr},
    };

    std::string ParseInstr(const CcuInstr* instr)
    {
        if (g_parseInstrSqeMap.find(instr->header.header) == g_parseInstrSqeMap.end()) {
            return Hccl::StringFormat("Unsupported instruction with header: 0x%04x", instr->header.header);
        }
        return g_parseInstrSqeMap[instr->header.header](instr);
    }

}; // namespace CcuRep
}; // namespace hcomm
