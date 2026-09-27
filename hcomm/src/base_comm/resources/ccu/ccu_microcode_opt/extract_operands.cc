/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "extract_operands.h"

#include "config/barrier_config.h"

namespace hcomm {
namespace CcuOpt {

    namespace {

        // reduce.count 字段以 "实际 ms 数 - 2" 编码, 且仅占低 3 位:
        // 实际 ms 数 = (count_field & 0x7) + 2.
        constexpr uint16_t CCU_REDUCE_COUNT_MASK = 0x7;
        constexpr uint16_t CCU_REDUCE_COUNT_BIAS = 2;
        // operate.parMode == 1: 第二操作数为寄存器 (Xd = Xn @ Xm); == 0 时为 imm16.
        constexpr uint16_t PARMODE_REGISTER_PAIR = 1;
        // set/clearCKE.clearType == 1: 命中后自动清零 waitCKEId, 该 CKE 位既读又写.
        constexpr uint16_t CLEARTYPE_AUTO = 1;

        inline void AddRead(std::vector<RegOperand>& out, RegType type, uint16_t regId)
        {
            if (regId == 0 && type == RegType::CKE) {
                return;
            }
            out.push_back(RegOperand{type, regId, /*isDef=*/false});
        }

        inline void AddWrite(std::vector<RegOperand>& out, RegType type, uint16_t regId)
        {
            if (regId == 0 && type == RegType::CKE) {
                return;
            }
            out.push_back(RegOperand{type, regId, /*isDef=*/true});
        }

        // Load 类中的访存 / 清零指令 (非算子). 返回 true 表示已处理该 code.
        bool ExtractLoadMemOps(std::vector<RegOperand>& out, const CcuRep::CcuInstr& instr)
        {
            using namespace InstrCodeV2;
            switch (instr.header.code) {
                case LOADSQEARGSTOX_CODE:
                    AddWrite(out, RegType::XN, instr.v2.loadSqeArgsToX.xnId);
                    return true;
                case LOADIMDTOX_CODE:
                    AddWrite(out, RegType::XN, instr.v2.loadImdToX.xnId);
                    return true;
                case LOADX_CODE:
                    // LoadX 语义: xn[i] = array[0 + xn[offset]]. 字段落位以 ccu_microcode_v2.cc
                    // CcuV2::LoadX 为准 (串讲材料表格与代码不符, 以代码为准):
                    //   xdId = i        (LoadX 写的目标单 xn, 唯一 def, 后端受管写后读写者);
                    //   xso  = array 基址 (读);
                    //   xsId = offset    (读);
                    //   xdo  = 立即数     (非寄存器, 不提取).
                    AddWrite(out, RegType::XN, instr.v2.loadStoreX.xdId);
                    AddRead(out, RegType::XN, instr.v2.loadStoreX.xso);
                    AddRead(out, RegType::XN, instr.v2.loadStoreX.xsId);
                    return true;
                case STOREX_CODE:
                    // StoreX 语义: array[0 + xn[offset]] = xn[i]. 字段落位以 ccu_microcode_v2.cc
                    // CcuV2::StoreX 为准 (translate 传 dst=array, src=i, srcOffset=立即数, dstOffset=offset):
                    //   xdo  = array 基址 (StoreX 写的目标, 唯一 def, 后端受管写后读写者);
                    //   xsId = i         (被存的数据, 读);
                    //   xdId = offset    (读);
                    //   xso  = 立即数     (非寄存器, 不提取).
                    AddWrite(out, RegType::XN, instr.v2.loadStoreX.xdo);
                    AddRead(out, RegType::XN, instr.v2.loadStoreX.xsId);
                    AddRead(out, RegType::XN, instr.v2.loadStoreX.xdId);
                    return true;
                case CLEARX_CODE:
                    // ClearX 清空连续 xn 区间 [xnId, xmId], 整片既读又写: 同片 ClearX 之间构成写后读.
                    // xnIdMode/xmIdMode==0 表示端点为寄存器 id (translate 恒传 0), 直接按寄存器提取.
                    AddRead(out, RegType::XN, instr.v2.clearX.xnId);
                    AddRead(out, RegType::XN, instr.v2.clearX.xmId);
                    AddWrite(out, RegType::XN, instr.v2.clearX.xnId);
                    AddWrite(out, RegType::XN, instr.v2.clearX.xmId);
                    return true;
                case NOP_CODE:
                    // 无寄存器操作数.
                    return true;
                case LOAD_CODE:
                    AddRead(out, RegType::XN, instr.v2.load.xsId);
                    AddRead(out, RegType::XN, instr.v2.load.xstId);
                    AddRead(out, RegType::XN, instr.v2.load.xlId);
                    AddWrite(out, RegType::XN, instr.v2.load.xdId);
                    return true;
                case STORE_CODE:
                    AddRead(out, RegType::XN, instr.v2.store.xdId);
                    AddRead(out, RegType::XN, instr.v2.store.xdtId);
                    AddRead(out, RegType::XN, instr.v2.store.xsId);
                    AddRead(out, RegType::XN, instr.v2.store.xlId);
                    AddRead(out, RegType::XN, instr.v2.store.xhId);
                    return true;
                default:
                    return false;
            }
        }

        // Load 类中的算子指令 (Add / Sub / Mul / And / Or / Xor / Shl / Shr / Popcnt / Not).
        // Xd = Xn @ Xm (parMode == 1) 或 Xd = Xn @ imm16 (parMode == 0); Not / Popcnt 仅使用 Xn.
        void ExtractLoadOperator(std::vector<RegOperand>& out, const CcuRep::CcuInstr& instr)
        {
            using namespace InstrCodeV2;
            switch (instr.header.code) {
                case ADD_CODE:
                case SUB_CODE:
                case MUL_CODE:
                case AND_CODE:
                case OR_CODE:
                case XOR_CODE:
                case SHL_CODE:
                case SHR_CODE:
                case POPCNT_CODE:
                    AddWrite(out, RegType::XN, instr.v2.operate.xdId);
                    AddRead(out, RegType::XN, instr.v2.operate.xnId);
                    if (instr.v2.operate.parMode == PARMODE_REGISTER_PAIR) {
                        AddRead(out, RegType::XN, instr.v2.operate.xmId);
                    }
                    break;
                case NOT_CODE:
                    AddWrite(out, RegType::XN, instr.v2.operate.xdId);
                    AddRead(out, RegType::XN, instr.v2.operate.xnId);
                    break;
                default:
                    break;
            }
        }

        void ExtractLoadType(std::vector<RegOperand>& out, const CcuRep::CcuInstr& instr)
        {
            if (!ExtractLoadMemOps(out, instr)) {
                ExtractLoadOperator(out, instr);
            }
            // setCKEId 不作为 CKE def, 不提取 (见文件顶部 set 语义说明).
        }

        void ExtractCtrlType(std::vector<RegOperand>& out, const CcuRep::CcuInstr& instr)
        {
            using namespace InstrCodeV2;
            switch (instr.header.code) {
                case LOOP_CODE:
                    // xmId = IterNum, xnId = Offset, xpId = ContextId; 三者都是 Xn 寄存器.
                    AddRead(out, RegType::XN, instr.v2.loop.xmId);
                    AddRead(out, RegType::XN, instr.v2.loop.xnId);
                    AddRead(out, RegType::XN, instr.v2.loop.xpId);
                    break;
                case LOOPGROUP_CODE:
                    AddRead(out, RegType::XN, instr.v2.loopGroup.xnId);
                    AddRead(out, RegType::XN, instr.v2.loopGroup.xmId);
                    AddRead(out, RegType::XN, instr.v2.loopGroup.xpId);
                    break;
                case SETCKBIT_CODE:
                    AddRead(out, RegType::CKE, instr.v2.setCKE.waitCKEId);
                    if (instr.v2.setCKE.clearType == CLEARTYPE_AUTO) {
                        AddWrite(out, RegType::CKE, instr.v2.setCKE.waitCKEId);
                    }
                    break;
                case CLEARCKBIT_CODE:
                    // 与 setcke 对称: 只有 clearType=1 自动清零的 waitCKEId 才是 CKE 写者 (read + def);
                    // clearCKEId 只是主动清某位, 同样不作为触发写后读的 def.
                    AddRead(out, RegType::CKE, instr.v2.clearCKE.waitCKEId);
                    if (instr.v2.clearCKE.clearType == CLEARTYPE_AUTO) {
                        AddWrite(out, RegType::CKE, instr.v2.clearCKE.waitCKEId);
                    }
                    break;
                case JMP_CODE:
                    AddRead(out, RegType::XN, instr.v2.jmp.relTarInstrXnId);
                    AddRead(out, RegType::XN, instr.v2.jmp.conditionXnId);
                    AddRead(out, RegType::XN, instr.v2.jmp.expectedXnId);
                    break;
                case WAIT_CODE:
                    AddRead(out, RegType::XN, instr.v2.wait.conditionXnId);
                    AddRead(out, RegType::XN, instr.v2.wait.expectedXnId);
                    break;
                case FENCE_CODE:
                    break;
                default:
                    break;
            }
        }

        // Trans 类中的纯搬运指令 (Mem<->MS / Mem<->Mem / TransMem). 返回 true 表示已处理该 code.
        bool ExtractTransMoveOps(std::vector<RegOperand>& out, const CcuRep::CcuInstr& instr)
        {
            using namespace InstrCodeV2;
            switch (instr.header.code) {
                case TRANSLOCMEMTOLOCMS_CODE:
                    AddRead(out, RegType::XN, instr.v2.transLocMemToLocMS.xsId);
                    AddRead(out, RegType::XN, instr.v2.transLocMemToLocMS.xstId);
                    AddRead(out, RegType::XN, instr.v2.transLocMemToLocMS.xlId);
                    AddRead(out, RegType::XN, instr.v2.transLocMemToLocMS.xoId);
                    AddWrite(out, RegType::MS, instr.v2.transLocMemToLocMS.msId);
                    return true;
                case TRANSLOCMSTOLOCMEM_CODE:
                    AddRead(out, RegType::MS, instr.v2.transLocMSToLocMem.msId);
                    AddRead(out, RegType::XN, instr.v2.transLocMSToLocMem.xdId);
                    AddRead(out, RegType::XN, instr.v2.transLocMSToLocMem.xdtId);
                    AddRead(out, RegType::XN, instr.v2.transLocMSToLocMem.xlId);
                    AddRead(out, RegType::XN, instr.v2.transLocMSToLocMem.xoId);
                    return true;
                case TRANSLOCMSTOLOCMS_CODE:
                    AddRead(out, RegType::MS, instr.v2.transLocMSToLocMS.mssId);
                    AddRead(out, RegType::XN, instr.v2.transLocMSToLocMS.xlId);
                    AddRead(out, RegType::XN, instr.v2.transLocMSToLocMS.xoId);
                    AddWrite(out, RegType::MS, instr.v2.transLocMSToLocMS.msdId);
                    return true;
                case TRANSLOCMEMTOLOCMEM_CODE:
                    AddRead(out, RegType::XN, instr.v2.transLocMemToLocMem.xdId);
                    AddRead(out, RegType::XN, instr.v2.transLocMemToLocMem.xdtId);
                    AddRead(out, RegType::XN, instr.v2.transLocMemToLocMem.xsId);
                    AddRead(out, RegType::XN, instr.v2.transLocMemToLocMem.xstId);
                    AddRead(out, RegType::XN, instr.v2.transLocMemToLocMem.xlId);
                    // usedMSId / msNum 描述临时 MS 区段, 保守视作 read.
                    AddRead(out, RegType::MS, instr.v2.transLocMemToLocMem.usedMSId);
                    return true;
                case TRANSMEM_CODE:
                    AddRead(out, RegType::XN, instr.v2.transMem.xdId);
                    AddRead(out, RegType::XN, instr.v2.transMem.xdtId);
                    AddRead(out, RegType::XN, instr.v2.transMem.xsId);
                    AddRead(out, RegType::XN, instr.v2.transMem.xstId);
                    AddRead(out, RegType::XN, instr.v2.transMem.xlId);
                    AddRead(out, RegType::XN, instr.v2.transMem.xcId);
                    AddRead(out, RegType::XN, instr.v2.transMem.xnId);
                    AddRead(out, RegType::XN, instr.v2.transMem.xntId);
                    return true;
                default:
                    return false;
            }
        }

        // Trans 类中的同步指令 (SyncWtX / SyncAtX).
        void ExtractTransSyncOps(std::vector<RegOperand>& out, const CcuRep::CcuInstr& instr)
        {
            using namespace InstrCodeV2;
            switch (instr.header.code) {
                case SYNCWTX_CODE:
                    AddRead(out, RegType::XN, instr.v2.syncWtX.xdId);
                    AddRead(out, RegType::XN, instr.v2.syncWtX.xdtId);
                    AddRead(out, RegType::XN, instr.v2.syncWtX.xsId);
                    AddRead(out, RegType::XN, instr.v2.syncWtX.xcId);
                    if (instr.v2.syncWtX.notifyValid != 0) {
                        AddRead(out, RegType::XN, instr.v2.syncWtX.xnId);
                        AddRead(out, RegType::XN, instr.v2.syncWtX.xntId);
                    }
                    break;
                case SYNCATX_CODE:
                    AddRead(out, RegType::XN, instr.v2.syncAtX.xdId);
                    AddRead(out, RegType::XN, instr.v2.syncAtX.xdtId);
                    AddRead(out, RegType::XN, instr.v2.syncAtX.xsId);
                    AddRead(out, RegType::XN, instr.v2.syncAtX.xcId);
                    break;
                default:
                    break;
            }
        }

        void ExtractTransType(std::vector<RegOperand>& out, const CcuRep::CcuInstr& instr)
        {
            if (!ExtractTransMoveOps(out, instr)) {
                ExtractTransSyncOps(out, instr);
            }
            // setCKEId 不作为 CKE def, 不提取 (见文件顶部 set 语义说明).
        }

        void ExtractReduceType(std::vector<RegOperand>& out, const CcuRep::CcuInstr& instr)
        {
            // ReduceAdd / ReduceMax / ReduceMin: MSA~MSH reduce to MSA, msId[0] 既读又写.
            uint16_t countInInstr = instr.v2.reduce.count;
            // countInInstr 是 "count - 2" 后存进去的, 实际 ms 数 = count_field + 2.
            uint16_t realCount = (countInInstr & CCU_REDUCE_COUNT_MASK) + CCU_REDUCE_COUNT_BIAS;
            if (realCount > CcuRep::CCU_REDUCE_MAX_MS) {
                realCount = CcuRep::CCU_REDUCE_MAX_MS;
            }

            AddRead(out, RegType::MS, instr.v2.reduce.msId[0]);
            AddWrite(out, RegType::MS, instr.v2.reduce.msId[0]);
            for (uint16_t i = 1; i < realCount; ++i) {
                AddRead(out, RegType::MS, instr.v2.reduce.msId[i]);
            }
            AddRead(out, RegType::XN, instr.v2.reduce.XnIdLength);
            // setCKEId 不作为 CKE def, 不提取 (见文件顶部 set 语义说明).
        }

    } // namespace

    std::vector<RegOperand> ExtractOperandsV2(const CcuRep::CcuInstr& instr)
    {
        using namespace InstrCodeV2;
        std::vector<RegOperand> out;
        constexpr size_t MAX_INSTR_OPERAND_NUM = 8;
        out.reserve(MAX_INSTR_OPERAND_NUM);
        switch (instr.header.type) {
            case LOAD_TYPE:
                ExtractLoadType(out, instr);
                break;
            case CTRL_TYPE:
                ExtractCtrlType(out, instr);
                break;
            case TRANS_TYPE:
                ExtractTransType(out, instr);
                break;
            case REDUCE_TYPE:
                ExtractReduceType(out, instr);
                break;
            default:
                break;
        }
        return out;
    }

} // namespace CcuOpt
} // namespace hcomm
