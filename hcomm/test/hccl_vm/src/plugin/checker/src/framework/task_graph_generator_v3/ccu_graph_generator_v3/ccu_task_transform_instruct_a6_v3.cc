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

using namespace HcclSim;

namespace HcclSim {
namespace TaskGraphGeneratorV3 {

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
                HCCL_VM_ERROR(
                    "{} This A6 instruction cannot be used inside a loop expansion.",
                    MakeErrorCodeText(ErrorCode::GRAPH_UNSUPPORTED).c_str());
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

            const uint32_t expandOffset = static_cast<uint32_t>(loopGroupParam->loopGroupXn.expandOffset);
            const uint32_t ckOffset = static_cast<uint32_t>(loopGroupParam->loopGroupXm.ckOffset);
            return UpdateId(CKEId, loopGroupParam->curLoopIdx, expandOffset, ckOffset, loopGroupParam->curExpandCnt);
        }

        // 循环中XN需要进行自动偏移
        uint16_t UpdateXnId(uint16_t xnId, LoopGroupParamA6* loopGroupParam)
        {
            // 不在循环中，不需要刷新
            if (loopGroupParam == nullptr) {
                return xnId;
            }

            const uint32_t expandOffset = static_cast<uint32_t>(loopGroupParam->loopGroupXn.expandOffset);
            const uint32_t xnIdOffset = static_cast<uint32_t>(loopGroupParam->loopGroupXp.xnIdOffset);
            return UpdateId(xnId, loopGroupParam->curLoopIdx, expandOffset, xnIdOffset, loopGroupParam->curExpandCnt);
        }

        // 循环中MS需要进行自动偏移
        uint16_t UpdateMSId(uint16_t MSId, LoopGroupParamA6* loopGroupParam)
        {
            // 不在循环中，不需要刷新
            if (loopGroupParam == nullptr) {
                return MSId;
            }

            const uint32_t expandOffset = static_cast<uint32_t>(loopGroupParam->loopGroupXn.expandOffset);
            const uint32_t msOffset = static_cast<uint32_t>(loopGroupParam->loopGroupXm.msOffset);
            return UpdateId(MSId, loopGroupParam->curLoopIdx, expandOffset, msOffset, loopGroupParam->curExpandCnt);
        }

        uint64_t UpdateAddressWithoutStride(uint64_t addr, LoopGroupParamA6* loopGroupParam)
        {
            if (loopGroupParam == nullptr) {
                return addr;
            }
            LoopA6& xm = loopGroupParam->loopXms[loopGroupParam->curLoopIdx];
            const uint64_t loopIntrGsa = static_cast<uint64_t>(xm.loopXn.loopIntrGSA);
            return addr + loopGroupParam->curLoopCnt * loopIntrGsa;
        }

        uint64_t UpdateAddress(uint64_t addr, LoopGroupParamA6* loopGroupParam, uint16_t addrExpandCoef = 0)
        {
            // 不在循环中，不需要刷新
            if (loopGroupParam == nullptr) {
                return addr;
            }
            LoopA6& xm = loopGroupParam->loopXms[loopGroupParam->curLoopIdx];
            const uint32_t expandOffset = static_cast<uint32_t>(loopGroupParam->loopGroupXn.expandOffset);
            const uint64_t loopIntrGsa = static_cast<uint64_t>(xm.loopXn.loopIntrGSA);

            if (loopGroupParam->curLoopIdx < expandOffset) {
                return (addr + loopGroupParam->curLoopCnt * loopIntrGsa) << addrExpandCoef;
            }

            const uint64_t gsaOffset = static_cast<uint64_t>(loopGroupParam->loopGroupXm.gsaOffset);
            return addr + ((loopGroupParam->curExpandCnt * gsaOffset) << addrExpandCoef)
                   + ((loopGroupParam->curLoopCnt * loopIntrGsa) << addrExpandCoef);
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
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
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
            HCCL_VM_DEBUG("Load SqeArg[{}]({}) to Xn[{}]", sqeArgsId, argVal, xnId);
            return HCCL_SUCCESS;
        }

        // 将立即数 写到第n个X寄存器中
        HcclResult TransformLoadImdToXInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
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
            HCCL_VM_DEBUG("Load immediate[{}] to Xn[{}]", immediate, xnId);
            return HCCL_SUCCESS;
        }

        // 判断XnId是否有效
        HcclResult CheckXnValid(uint16_t xnId, uint16_t xnIdMin, uint16_t xnIdMax)
        {
            if (xnId < xnIdMin || xnId > xnIdMax) {
                HCCL_VM_ERROR(
                    "{} Xn register id is out of the valid range, xnId={}, validMin={}, validMax={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_OUT_OF_RANGE).c_str(), xnId, xnIdMin, xnIdMax);
                return HCCL_E_PARA;
            }
            return HCCL_SUCCESS;
        }

        HcclResult GetXnValueAndCheck(CcuGraphStateV3* curCcuTask, uint32_t queId, uint16_t xnId, uint64_t& xnValue)
        {
            CHK_RET(CheckXnValid(xnId, 0, MAX_LOADX_STOREX_ID_NUM));
            CHK_GET_XN_V3(curCcuTask, queId, xnId, xnValue);
            return HCCL_SUCCESS;
        }

        HcclResult TransformLoadXInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            LoopGroupParamA6* loopGroupParam)
        {
            (void)isContinue;
            uint8_t mode = instr->v2.loadStoreX.oMode;

            RankId rankId = curCcuTask->GetRankId();
            uint32_t dieId{0};
            curCcuTask->GetDieId(queId, dieId);

            uint16_t xsId = GetXnId(instr->v2.loadStoreX.xsId, loopGroupParam);
            uint64_t xsValue = 0;
            CHK_GET_XN_V3(curCcuTask, queId, xsId, xsValue);

            uint16_t xdId = GetXnId(instr->v2.loadStoreX.xdId, loopGroupParam);
            uint64_t xdValue = 0;
            if (mode == 0) {
                // 计算公式：*Xd=*X(*Xs+立即数)+立即数（64位）
                uint16_t immedataSo = instr->v2.loadStoreX.xso;
                uint16_t immedataDo = instr->v2.loadStoreX.xdo;
                uint16_t xsTmpId = static_cast<uint16_t>(xsValue) + immedataSo;
                uint64_t xsTmpValue = 0;
                CHK_RET(GetXnValueAndCheck(curCcuTask, queId, xsTmpId, xsTmpValue));
                xdValue = xsTmpValue + immedataDo;
                HCCL_VM_DEBUG(
                    "LoadX Xn[{}]({}) + immediate[{}] -> Xn[{}]({}), + immediate[{}] -> Xn[{}]({})", xsId, xsValue,
                    immedataSo, xsTmpId, xsTmpValue, immedataDo, xdId, xdValue);
            } else if (mode == 1) {
                // 计算公式：*Xd=*X(*Xs+*Xso)+*Xdo
                uint16_t xsSoId = GetXnId(instr->v2.loadStoreX.xso, loopGroupParam);
                uint64_t xsSoValue = 0;
                CHK_GET_XN_V3(curCcuTask, queId, xsSoId, xsSoValue);

                uint16_t xsDoId = GetXnId(instr->v2.loadStoreX.xdo, loopGroupParam);
                uint64_t xsDoValue = 0;
                CHK_GET_XN_V3(curCcuTask, queId, xsDoId, xsDoValue);

                uint16_t xsTmpId = static_cast<uint16_t>(xsValue + xsSoValue);
                uint64_t xsTmpValue = 0;
                CHK_RET(GetXnValueAndCheck(curCcuTask, queId, xsTmpId, xsTmpValue));
                xdValue = xsTmpValue + xsDoValue;
                HCCL_VM_DEBUG(
                    "LoadX Xn[{}]({}) + Xn[{}]({}) -> Xn[{}]({}), + Xn[{}]({}) -> Xn[{}]({})", xsId, xsValue, xsSoId,
                    xsSoValue, xsTmpId, xsTmpValue, xsDoId, xsDoValue, xdId, xdValue);
            } else {
                HCCL_VM_ERROR(
                    "{} This instruction mode is not supported by CheckerV3 graph expansion, "
                    "instruction=LoadX, mode={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_UNSUPPORTED).c_str(), mode);
                return HCCL_E_PARA;
            }

            CHK_RET(AllRankParamRecorder::Global()->SetXn(rankId, dieId, xdId, xdValue));
            // cke设置
            uint16_t ckeId = UpdateCKEId(instr->v2.loadStoreX.setCKEId, loopGroupParam);
            uint16_t ckeMask = instr->v2.loadStoreX.setCKEMask;
            CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, ckeId, ckeMask));
            return HCCL_SUCCESS;
        }

        // StoreX指令
        HcclResult TransformStoreXInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            LoopGroupParamA6* loopGroupParam)
        {
            (void)isContinue;
            uint8_t mode = instr->v2.loadStoreX.oMode;

            RankId rankId = curCcuTask->GetRankId();
            uint32_t dieId{0};
            curCcuTask->GetDieId(queId, dieId);

            uint16_t xsId = GetXnId(instr->v2.loadStoreX.xsId, loopGroupParam);
            uint64_t xsValue = 0;
            CHK_GET_XN_V3(curCcuTask, queId, xsId, xsValue);

            uint16_t xdId = GetXnId(instr->v2.loadStoreX.xdId, loopGroupParam);
            uint64_t xdValue = 0;
            CHK_GET_XN_V3(curCcuTask, queId, xdId, xdValue);

            uint16_t newId = 0;
            uint16_t newValue = 0;
            if (mode == 0) {
                // 计算公式：*X(*Xd + 立即数) =*Xs+*Xso
                uint16_t immedataSo = instr->v2.loadStoreX.xso;
                uint16_t immedataDo = instr->v2.loadStoreX.xdo;

                newId = static_cast<uint16_t>(xdValue) + immedataDo;
                newValue = xsValue + static_cast<uint64_t>(immedataSo);
                HCCL_VM_DEBUG(
                    "StoreX Xn[{}]({}) + immediate[{}] -> Xn[{}], Xn[{}]({}) + immediate[{}] -> Xn[{}]({})", xdId,
                    xdValue, immedataDo, newId, xsId, xsValue, immedataSo, newId, newValue);

            } else if (mode == 1) {
                // 计算公式：*X(*Xd+*Xdo) =*Xs+*Xso
                uint16_t xsDoId = GetXnId(instr->v2.loadStoreX.xdo, loopGroupParam);
                uint64_t xsDoValue = 0;
                CHK_GET_XN_V3(curCcuTask, queId, xsDoId, xsDoValue);
                newId = static_cast<uint16_t>(xdValue + xsDoValue);

                uint16_t xsSoId = GetXnId(instr->v2.loadStoreX.xso, loopGroupParam);
                uint64_t xsSoValue = 0;
                CHK_GET_XN_V3(curCcuTask, queId, xsSoId, xsSoValue);

                newValue = xsValue + xsSoValue;
                HCCL_VM_DEBUG(
                    "StoreX Xn[{}]({}) + Xn[{}]({}) -> Xn[{}], Xn[{}]({}) + Xn[{}]({}) -> Xn[{}]({})", xdId, xdValue,
                    xsDoId, xsDoValue, newId, xsId, xsValue, xsSoId, xsSoValue, newId, newValue);
            } else {
                HCCL_VM_ERROR(
                    "{} This instruction mode is not supported by CheckerV3 graph expansion, "
                    "instruction=StoreX, mode={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_UNSUPPORTED).c_str(), mode);
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
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
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
            HCCL_VM_DEBUG("Clear Xn[{}] ~ Xn[{}] to 0", xnId, xmId);
            return HCCL_SUCCESS;
        }

        // Nop指令，空指令，用于排指令流水
        HcclResult TransformNopInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
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
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            LoopGroupParamA6* loopGroupParam)
        {
            (void)isContinue;
            RankId rankId = curCcuTask->GetRankId();
            uint32_t dieId{0};
            curCcuTask->GetDieId(queId, dieId);
            uint16_t dstType = instr->v2.load.dstType;
            if (dstType != 0) {
                // 目前只支持Xn寄存器
                HCCL_VM_ERROR(
                    "{} Load only supports writing into Xn registers, actualDstType={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_UNSUPPORTED).c_str(), dstType);
                return HCCL_E_PARA;
            }
            uint16_t xdId = GetXnId(instr->v2.load.xdId, loopGroupParam);
            // 读取回来的数据放入空间的索引号
            uint64_t xdValue = 0;
            CHK_GET_XN_V3(curCcuTask, queId, xdId, xdValue);
            uint16_t xsId = GetXnId(instr->v2.load.xsId, loopGroupParam);
            uint64_t xsStartAddr = 0;
            CHK_GET_XN_V3(curCcuTask, queId, xsId, xsStartAddr);
            // 需要判断xsStartAddr是否8bye对齐
            if ((xsStartAddr & 0x7) != 0) {
                HCCL_VM_ERROR(
                    "{} Load source address must be 8-byte aligned, sourceAddress={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_ADDRESS_INVALID).c_str(), xsStartAddr);
                return HCCL_E_PARA;
            }
            uint16_t xlId = GetXnId(instr->v2.load.xlId, loopGroupParam);
            uint64_t length = 0;
            CHK_GET_XN_V3(curCcuTask, queId, xlId, length);
            if ((length & 0x7) != 0) {
                HCCL_VM_ERROR(
                    "{} Load length must be a multiple of 8 bytes, length={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_STRUCTURE_INVALID).c_str(), length);
                return HCCL_E_PARA;
            }
            std::vector<uint64_t> entry;
            if (AllRankParamRecorder::Global()->GetHBM(rankId, dieId, xsStartAddr, entry) != HCCL_SUCCESS) {
                const uint32_t instrId = curCcuTask->microCodePosInQue[queId] + curCcuTask->startInstrIdInQue[queId];
                HCCL_VM_ERROR(
                    "{} Failed to read HBM content before it was initialized, rankId={}, dieId={}, "
                    "instrId={}, hbmAddr={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_REGISTER_UNINITIALIZED).c_str(), static_cast<uint32_t>(rankId),
                    dieId, instrId, xsStartAddr);
                return HCCL_E_PARA;
            }
            if (length / 8 > entry.size()) {
                HCCL_VM_ERROR(
                    "{} Load length exceeds the number of 64-bit entries available in HBM, "
                    "requestedLength={}, availableEntryCount={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_STRUCTURE_INVALID).c_str(), length, entry.size());
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
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            LoopGroupParamA6* loopGroupParam)
        {
            (void)isContinue;
            RankId rankId = curCcuTask->GetRankId();
            uint32_t dieId{0};
            curCcuTask->GetDieId(queId, dieId);
            uint16_t srcType = instr->v2.store.srcType;
            if (srcType != 0) {
                // 目前只支持Xn寄存器
                HCCL_VM_ERROR(
                    "{} Store only supports reading from Xn registers, actualSrcType={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_UNSUPPORTED).c_str(), srcType);
                return HCCL_E_PARA;
            }
            uint16_t xdId = GetXnId(instr->v2.store.xdId, loopGroupParam);
            // 读取回来的数据起始地址
            uint64_t xdValue = 0;
            CHK_GET_XN_V3(curCcuTask, queId, xdId, xdValue);
            uint16_t xsId = GetXnId(instr->v2.store.xsId, loopGroupParam);
            uint64_t xsValue = 0;
            CHK_GET_XN_V3(curCcuTask, queId, xsId, xsValue);
            // 需要判断xsStartAddr是否8bye对齐
            uint16_t xlId = GetXnId(instr->v2.store.xlId, loopGroupParam);
            uint64_t length = 0;
            CHK_GET_XN_V3(curCcuTask, queId, xlId, length);
            if ((length & 0x7) != 0) {
                HCCL_VM_ERROR(
                    "{} Store length must be a multiple of 8 bytes, length={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_STRUCTURE_INVALID).c_str(), length);
                return HCCL_E_PARA;
            }
            std::vector<uint64_t> entry;
            for (size_t i = 0; i < length / 8; i++) {
                uint64_t xsValueTmp = 0;
                CHK_GET_XN_V3(curCcuTask, queId, xsValue++, xsValueTmp);
                entry.push_back(xsValueTmp);
            }
            if ((entry.size() % 8) != 0) {
                const uint32_t instrId = curCcuTask->microCodePosInQue[queId] + curCcuTask->startInstrIdInQue[queId];
                HCCL_VM_ERROR(
                    "{} Store data length must be a multiple of 64 bytes before writing HBM, "
                    "rankId={}, dieId={}, queueId={}, instrId={}, hbmAddr={}, dataEntryCount={}, dataLengthBytes={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_ADDRESS_INVALID).c_str(), static_cast<uint32_t>(rankId), dieId,
                    queId, instrId, xdValue, entry.size(), entry.size() * sizeof(uint64_t));
                return HCCL_E_PARA;
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
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
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
            CHK_GET_XN_V3(curCcuTask, queId, xnId, xnValue);
            // Xd = Xn + immed
            if (parMode == 0) {
                xdValue = xnValue + instr->v2.operate.xmId;
                HCCL_VM_DEBUG(
                    "Xn[{}]({}) + immediate[{}] -> Xn[{}]({})", xnId, xnValue, instr->v2.operate.xmId, xdId, xdValue);
            } else {
                // Xd = Xn + Xm
                uint16_t xmId = GetXnId(instr->v2.operate.xmId, loopGroupParam);
                uint64_t xmValue = 0;
                CHK_GET_XN_V3(curCcuTask, queId, xmId, xmValue);
                xdValue = xnValue + xmValue;
                HCCL_VM_DEBUG("Xn[{}]({}) + Xn[{}]({}) -> Xn[{}]({})", xnId, xnValue, xmId, xmValue, xdId, xdValue);
            }

            CHK_RET(AllRankParamRecorder::Global()->SetXn(rankId, dieId, xdId, xdValue));
            CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, ckeId, ckeMask));
            return HCCL_SUCCESS;
        }

        // Sub指令，实现算数“-”
        HcclResult TransformSubInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
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
            CHK_GET_XN_V3(curCcuTask, queId, xnId, xnValue);
            if (parMode == 0) {
                xdValue = xnValue - instr->v2.operate.xmId;
                HCCL_VM_DEBUG(
                    "Xn[{}]({}) - immediate[{}] -> Xn[{}]({})", xnId, xnValue, instr->v2.operate.xmId, xdId, xdValue);
            } else {
                uint16_t xmId = GetXnId(instr->v2.operate.xmId, loopGroupParam);
                uint64_t xmValue = 0;
                CHK_GET_XN_V3(curCcuTask, queId, xmId, xmValue);
                xdValue = xnValue - xmValue;
                HCCL_VM_DEBUG("Xn[{}]({}) - Xn[{}]({}) -> Xn[{}]({})", xnId, xnValue, xmId, xmValue, xdId, xdValue);
            }

            CHK_RET(AllRankParamRecorder::Global()->SetXn(rankId, dieId, xdId, xdValue));
            CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, ckeId, ckeMask));
            return HCCL_SUCCESS;
        }

        // Mul指令，实现算数“*”
        HcclResult TransformMulInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
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
            CHK_GET_XN_V3(curCcuTask, queId, xnId, xnValue);
            if (parMode == 0) {
                xdValue = (xnValue & 0xFFFFFFFF) * instr->v2.operate.xmId;
                HCCL_VM_DEBUG(
                    "low32(Xn[{}]({})) * immediate[{}] -> Xn[{}]({})", xnId, xnValue & 0xFFFFFFFF,
                    instr->v2.operate.xmId, xdId, xdValue);
            } else {
                uint16_t xmId = GetXnId(instr->v2.operate.xmId, loopGroupParam);
                uint64_t xmValue = 0;
                CHK_GET_XN_V3(curCcuTask, queId, xmId, xmValue);
                xdValue = (xnValue & 0xFFFFFFFF) * (xmValue & 0xFFFFFFFF);
                HCCL_VM_DEBUG(
                    "low32(Xn[{}]({})) * low32(Xn[{}]({})) -> Xn[{}]({})", xnId, xnValue & 0xFFFFFFFF, xmId,
                    xmValue & 0xFFFFFFFF, xdId, xdValue);
            }

            CHK_RET(AllRankParamRecorder::Global()->SetXn(rankId, dieId, xdId, xdValue));
            CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, ckeId, ckeMask));
            return HCCL_SUCCESS;
        }

        // And指令，实现逻辑“&”
        HcclResult TransformANDInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
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
            CHK_GET_XN_V3(curCcuTask, queId, xnId, xnValue);
            CHK_GET_XN_V3(curCcuTask, queId, xmId, xmValue);
            uint64_t xdValue = xnValue & xmValue;
            HCCL_VM_DEBUG("Xn[{}]({}) & Xn[{}]({}) -> Xn[{}]({})", xnId, xnValue, xmId, xmValue, xdId, xdValue);

            CHK_RET(AllRankParamRecorder::Global()->SetXn(rankId, dieId, xdId, xdValue));
            CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, ckeId, ckeMask));
            return HCCL_SUCCESS;
        }

        // Or指令，实现逻辑“|”
        HcclResult TransformORInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
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
            CHK_GET_XN_V3(curCcuTask, queId, xnId, xnValue);
            CHK_GET_XN_V3(curCcuTask, queId, xmId, xmValue);
            uint64_t xdValue = xnValue | xmValue;
            HCCL_VM_DEBUG("Xn[{}]({}) | Xn[{}]({}) -> Xn[{}]({})", xnId, xnValue, xmId, xmValue, xdId, xdValue);

            CHK_RET(AllRankParamRecorder::Global()->SetXn(rankId, dieId, xdId, xdValue));
            CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, ckeId, ckeMask));
            return HCCL_SUCCESS;
        }

        // Not指令，实现逻辑“~”
        HcclResult TransformNOTInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
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
            CHK_GET_XN_V3(curCcuTask, queId, xnId, xnValue);
            uint64_t xdValue = ~xnValue;
            HCCL_VM_DEBUG("~Xn[{}]({}) -> Xn[{}]({})", xnId, xnValue, xdId, xdValue);

            CHK_RET(AllRankParamRecorder::Global()->SetXn(rankId, dieId, xdId, xdValue));
            CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, ckeId, ckeMask));
            return HCCL_SUCCESS;
        }

        // Xor指令，实现逻辑“^”
        HcclResult TransformXORInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
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
            CHK_GET_XN_V3(curCcuTask, queId, xnId, xnValue);
            CHK_GET_XN_V3(curCcuTask, queId, xmId, xmValue);
            uint64_t xdValue = xnValue ^ xmValue;
            HCCL_VM_DEBUG("Xn[{}]({}) ^ Xn[{}]({}) -> Xn[{}]({})", xnId, xnValue, xmId, xmValue, xdId, xdValue);

            CHK_RET(AllRankParamRecorder::Global()->SetXn(rankId, dieId, xdId, xdValue));
            CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, ckeId, ckeMask));
            return HCCL_SUCCESS;
        }

        // SHL指令，实现逻辑“<<”
        HcclResult TransformSHLInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
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
            CHK_GET_XN_V3(curCcuTask, queId, xnId, xnValue);
            CHK_GET_XN_V3(curCcuTask, queId, xmId, xmValue);

            switch (shiftType) {
                case 0: // 逻辑移位
                    xdValue = xnValue << xmValue;
                    HCCL_VM_DEBUG(
                        "Xn[{}]({}) << Xn[{}]({}) [logical] -> Xn[{}]({})", xnId, xnValue, xmId, xmValue, xdId,
                        xdValue);
                    break;
                case 1: // 算术移位
                    xdValue = static_cast<int64_t>(xnValue) << xmValue;
                    HCCL_VM_DEBUG(
                        "Xn[{}]({}) << Xn[{}]({}) [arithmetic] -> Xn[{}]({})", xnId, xnValue, xmId, xmValue, xdId,
                        xdValue);
                    break;
                case 2: // 循环移位
                    xdValue = (xnValue << (xmValue & 0x3F)) | (xnValue >> (64 - (xmValue & 0x3F)));
                    HCCL_VM_DEBUG(
                        "Xn[{}]({}) << Xn[{}]({}) [rotate] -> Xn[{}]({})", xnId, xnValue, xmId, xmValue, xdId, xdValue);
                    break;
                default:
                    HCCL_VM_ERROR(
                        "{} Shift-left type is not supported, shiftType={}",
                        MakeErrorCodeText(ErrorCode::GRAPH_UNSUPPORTED).c_str(), shiftType);
                    return HCCL_E_INTERNAL;
            }

            CHK_RET(AllRankParamRecorder::Global()->SetXn(rankId, dieId, xdId, xdValue));
            CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, ckeId, ckeMask));
            return HCCL_SUCCESS;
        }

        // SHR指令，实现逻辑“>>”
        HcclResult TransformSHRInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
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
            CHK_GET_XN_V3(curCcuTask, queId, xnId, xnValue);
            CHK_GET_XN_V3(curCcuTask, queId, xmId, xmValue);

            switch (shiftType) {
                case 0: // 逻辑移位
                    xdValue = xnValue >> xmValue;
                    HCCL_VM_DEBUG(
                        "Xn[{}]({}) >> Xn[{}]({}) [logical] -> Xn[{}]({})", xnId, xnValue, xmId, xmValue, xdId,
                        xdValue);
                    break;
                case 1: // 算术移位
                    xdValue = static_cast<int64_t>(xnValue) >> xmValue;
                    HCCL_VM_DEBUG(
                        "Xn[{}]({}) >> Xn[{}]({}) [arithmetic] -> Xn[{}]({})", xnId, xnValue, xmId, xmValue, xdId,
                        xdValue);
                    break;
                case 2: // 循环移位
                    xdValue = (xnValue >> (xmValue & 0x3F)) | (xnValue << (64 - (xmValue & 0x3F)));
                    HCCL_VM_DEBUG(
                        "Xn[{}]({}) >> Xn[{}]({}) [rotate] -> Xn[{}]({})", xnId, xnValue, xmId, xmValue, xdId, xdValue);
                    break;
                default:
                    HCCL_VM_ERROR(
                        "{} Shift-right type is not supported, shiftType={}",
                        MakeErrorCodeText(ErrorCode::GRAPH_UNSUPPORTED).c_str(), shiftType);
                    return HCCL_E_INTERNAL;
            }

            CHK_RET(AllRankParamRecorder::Global()->SetXn(rankId, dieId, xdId, xdValue));
            CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, ckeId, ckeMask));
            return HCCL_SUCCESS;
        }

        // POPCNT指令,统计指定64bit寄存器中的二进制1的个数
        HcclResult TransformPopcntInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            LoopGroupParamA6* loopGroupParam)
        {
            RankId rankId = curCcuTask->GetRankId();
            uint32_t dieId{0};
            curCcuTask->GetDieId(queId, dieId);

            uint16_t xnId = GetXnId(instr->v2.operate.xnId, loopGroupParam);
            uint64_t xnValue = 0;
            CHK_GET_XN_V3(curCcuTask, queId, xnId, xnValue);

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
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            LoopGroupParamA6* loopGroupParam, uint32_t loopGroupIdx);

        HcclResult TransformLoopInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            LoopGroupParamA6* loopGroupParam)
        {
            uint16_t startInstrId = instr->v2.loop.startInstrId;
            uint16_t endInstrId = instr->v2.loop.endInstrId;
            uint16_t xnId = instr->v2.loop.xnId;
            HCCL_VM_ERROR(
                "{} A loop instruction cannot be executed by itself; it must be triggered from a "
                "LoopGroup, startInstructionId={}, endInstructionId={}, loopCountXnId={}",
                MakeErrorCodeText(ErrorCode::GRAPH_UNSUPPORTED).c_str(), startInstrId, endInstrId, xnId);
            // 当前Loop都需要通过LoopGoup来触发，暂不支持单独解析Loop命令
            return HCCL_E_INTERNAL;
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
                    "{} Loop-merged transfer size is 0, instruction=LoopMergedTransfer",
                    MakeErrorCodeText(ErrorCode::GRAPH_STRUCTURE_INVALID).c_str());
                return HCCL_E_INTERNAL;
            }
            return HCCL_SUCCESS;
        }

        static void
        SetLoopParamAtIteration(const LoopGroupParamA6& sampleParam, u32 curLoopCnt, LoopGroupParamA6& iterParam)
        {
            iterParam = sampleParam;
            iterParam.curLoopCnt = curLoopCnt;
        }

        static HcclResult CollectLoopCkeOpsA6(
            const CcuRep::CcuInstr* instr, const LoopGroupParamA6& sampleParam, CcuLoopCkeOpV3& waitOp,
            CcuLoopCkeOpV3& setOp)
        {
            switch (instr->header.type) {
                case CTRL_TYPE:
                    if (instr->header.code == A6_CLEARCKE_CODE) {
                        waitOp
                            = {UpdateCKEId(instr->v2.clearCKE.waitCKEId, const_cast<LoopGroupParamA6*>(&sampleParam)),
                               instr->v2.clearCKE.waitCKEMask};
                        return HCCL_SUCCESS;
                    }
                    return HCCL_E_NOT_SUPPORT;
                case TRANS_TYPE:
                    if (instr->header.code == A6_TRANSLOCMEMTOLOCMS_CODE) {
                        setOp
                            = {UpdateCKEId(
                                   instr->v2.transLocMemToLocMS.setCKEId, const_cast<LoopGroupParamA6*>(&sampleParam)),
                               instr->v2.transLocMemToLocMS.setCKEMask};
                        return HCCL_SUCCESS;
                    }
                    if (instr->header.code == A6_TRANSLOCMSTOLOCMEM_CODE) {
                        setOp
                            = {UpdateCKEId(
                                   instr->v2.transLocMSToLocMem.setCKEId, const_cast<LoopGroupParamA6*>(&sampleParam)),
                               instr->v2.transLocMSToLocMem.setCKEMask};
                        return HCCL_SUCCESS;
                    }
                    if (instr->header.code == A6_TRANSLOCMSTOLOCMS_CODE) {
                        setOp
                            = {UpdateCKEId(
                                   instr->v2.transLocMSToLocMS.setCKEId, const_cast<LoopGroupParamA6*>(&sampleParam)),
                               instr->v2.transLocMSToLocMS.setCKEMask};
                        return HCCL_SUCCESS;
                    }
                    if (instr->header.code == A6_TRANSLOCMEMTOLOCMEM_CODE) {
                        setOp
                            = {UpdateCKEId(
                                   instr->v2.transLocMemToLocMem.setCKEId, const_cast<LoopGroupParamA6*>(&sampleParam)),
                               instr->v2.transLocMemToLocMem.setCKEMask};
                        return HCCL_SUCCESS;
                    }
                    if (instr->header.code == A6_TRANSMEM_CODE) {
                        if (instr->v2.transMem.dmaOpCode == 5) {
                            return HCCL_E_NOT_SUPPORT;
                        }
                        setOp
                            = {UpdateCKEId(instr->v2.transMem.setCKEId, const_cast<LoopGroupParamA6*>(&sampleParam)),
                               instr->v2.transMem.setCKEMask};
                        return HCCL_SUCCESS;
                    }
                    return HCCL_E_NOT_SUPPORT;
                case REDUCE_TYPE:
                    if (instr->header.code == A6_REDUCEADD_CODE || instr->header.code == A6_REDUCEMAX_CODE
                        || instr->header.code == A6_REDUCEMIN_CODE) {
                        setOp
                            = {UpdateCKEId(instr->v2.reduce.setCKEId, const_cast<LoopGroupParamA6*>(&sampleParam)),
                               instr->v2.reduce.setCKEMask};
                        return HCCL_SUCCESS;
                    }
                    return HCCL_E_NOT_SUPPORT;
                default:
                    return HCCL_E_NOT_SUPPORT;
            }
        }

        static HcclResult CollectTransMemGenericLoopInstrA6(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, uint16_t instrId,
            const LoopGroupParamA6& sampleParam, uint64_t loopCnt, std::shared_ptr<CcuLoopInstrV3>& instrInLoop)
        {
            const auto& op = instr->v2.transMem;
            if (op.udfType != 0 || op.dmaOpCode == 5) {
                return HCCL_E_NOT_SUPPORT;
            }

            RankId rankId = curCcuTask->GetRankId();
            uint32_t dieId = INVALID_DIE_ID;
            curCcuTask->GetDieId(queId, dieId);
            const uint16_t xlId = GetXnId(op.xlId, const_cast<LoopGroupParamA6*>(&sampleParam));
            uint64_t len = 0;
            CHK_GET_XN_V3(curCcuTask, queId, xlId, len);
            CHK_RET(ValidateLoopLen(len));

            const uint16_t xcId = GetXnId(op.xcId, const_cast<LoopGroupParamA6*>(&sampleParam));
            uint64_t channelId = 0;
            CHK_GET_XN_V3(curCcuTask, queId, xcId, channelId);
            if (!IsExistRemoteDieInfo(rankId, dieId, channelId)) {
                return HCCL_E_NOT_SUPPORT;
            }
            RankId rmtRankId = g_allRankChannelInfo[rankId][dieId][channelId].dstRank;

            CcuLoopCkeOpV3 waitOp;
            CcuLoopCkeOpV3 setOp;
            CHK_RET(CollectLoopCkeOpsA6(instr, sampleParam, waitOp, setOp));
            const bool reduceEn = (op.udfEnable != 0);
            std::shared_ptr<CcuLoopTransMemV3> transInstr;
            std::shared_ptr<CcuLoopReduceV3> reduceInstr;
            if (reduceEn) {
                reduceInstr = EnsureCcuLoopInstr<CcuLoopReduceV3>(instrInLoop, rankId, dieId, instrId);
                if (reduceInstr == nullptr) {
                    return HCCL_E_MEMORY;
                }
                DataType hcclDataType;
                CHK_RET(
                    GetHcclDataTypeFromCCUDataType(op.reduceDataType, ccuReduceTypeMap[op.reduceOpCode], hcclDataType));
                reduceInstr->dataType = g_DataType2CheckerDataType_aicpu[hcclDataType];
                reduceInstr->reduceOp = g_ReduceOp2CheckerReduceOp_ccu[op.reduceOpCode];
                reduceInstr->AddWait(waitOp.ckeId, waitOp.mask);
                reduceInstr->AddSet(setOp.ckeId, setOp.mask);
            } else {
                transInstr = EnsureCcuLoopInstr<CcuLoopTransMemV3>(instrInLoop, rankId, dieId, instrId);
                if (transInstr == nullptr) {
                    return HCCL_E_MEMORY;
                }
                transInstr->AddWait(waitOp.ckeId, waitOp.mask);
                transInstr->AddSet(setOp.ckeId, setOp.mask);
            }

            for (uint64_t iter = 0; iter < loopCnt; ++iter) {
                LoopGroupParamA6 iterParam;
                SetLoopParamAtIteration(sampleParam, static_cast<u32>(iter), iterParam);
                const uint16_t xdId = GetXnId(op.xdId, &iterParam);
                const uint16_t xsId = GetXnId(op.xsId, &iterParam);
                DataSlice srcSlice;
                DataSlice dstSlice;
                uint64_t xdValue = 0;
                uint64_t xsValue = 0;

                if (op.dmaOpCode == 3) {
                    if (op.msIdMode == 1) {
                        const uint16_t srcMSId = UpdateMSId(op.xsId, &iterParam);
                        CHK_RET(GenSliceFromMs(srcMSId, len, srcSlice));
                        if (transInstr != nullptr) {
                            transInstr->msIds.insert(srcMSId);
                        }
                        if (reduceInstr != nullptr) {
                            reduceInstr->msIds.insert(srcMSId);
                        }
                    } else {
                        CHK_GET_XN_V3(curCcuTask, queId, xsId, xsValue);
                        xsValue = (op.src_mode == 1) ? UpdateAddress(xsValue, &iterParam) :
                                                       UpdateAddressWithoutStride(xsValue, &iterParam);
                        CHK_RET(curCcuTask->GetStorageManager().GetSlice(xsValue, len, srcSlice));
                    }
                    CHK_GET_XN_V3(curCcuTask, queId, xdId, xdValue);
                    xdValue = (op.dst_mode == 1) ? UpdateAddress(xdValue, &iterParam) :
                                                   UpdateAddressWithoutStride(xdValue, &iterParam);
                    CHK_RET(curCcuTask->GetStorageManager().GetSlice(xdValue, len, dstSlice));
                    if (reduceInstr != nullptr) {
                        reduceInstr->srcs.push_back({MakeCcuMemSlice(rankId, srcSlice)});
                        reduceInstr->dsts.push_back(MakeCcuMemSlice(rmtRankId, dstSlice));
                    } else {
                        transInstr->srcs.push_back(MakeCcuMemSlice(rankId, srcSlice));
                        transInstr->dsts.push_back(MakeCcuMemSlice(rmtRankId, dstSlice));
                    }
                } else if (op.dmaOpCode == 6) {
                    CHK_GET_XN_V3(curCcuTask, queId, xsId, xsValue);
                    xsValue = (op.src_mode == 1) ? UpdateAddress(xsValue, &iterParam) :
                                                   UpdateAddressWithoutStride(xsValue, &iterParam);
                    CHK_RET(curCcuTask->GetStorageManager().GetSlice(xsValue, len, srcSlice));
                    if (op.msIdMode == 1) {
                        const uint16_t dstMSId = UpdateMSId(op.xdId, &iterParam);
                        CHK_RET(GenSliceFromMs(dstMSId, len, dstSlice));
                        if (transInstr != nullptr) {
                            transInstr->msIds.insert(dstMSId);
                        }
                        if (reduceInstr != nullptr) {
                            reduceInstr->msIds.insert(dstMSId);
                        }
                    } else {
                        CHK_GET_XN_V3(curCcuTask, queId, xdId, xdValue);
                        xdValue = (op.dst_mode == 1) ? UpdateAddress(xdValue, &iterParam) :
                                                       UpdateAddressWithoutStride(xdValue, &iterParam);
                        CHK_RET(curCcuTask->GetStorageManager().GetSlice(xdValue, len, dstSlice));
                    }
                    if (reduceInstr != nullptr) {
                        reduceInstr->srcs.push_back({MakeCcuMemSlice(rmtRankId, srcSlice)});
                        reduceInstr->dsts.push_back(MakeCcuMemSlice(rankId, dstSlice));
                    } else {
                        transInstr->srcs.push_back(MakeCcuMemSlice(rmtRankId, srcSlice));
                        transInstr->dsts.push_back(MakeCcuMemSlice(rankId, dstSlice));
                    }
                } else {
                    return HCCL_E_NOT_SUPPORT;
                }
            }
            return HCCL_SUCCESS;
        }

        static HcclResult CollectTransLoopInstrA6(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, uint16_t instrId,
            const LoopGroupParamA6& sampleParam, uint64_t loopCnt, std::shared_ptr<CcuLoopInstrV3>& instrInLoop)
        {
            if (instr->header.code == A6_TRANSMEM_CODE) {
                return CollectTransMemGenericLoopInstrA6(
                    instr, curCcuTask, queId, instrId, sampleParam, loopCnt, instrInLoop);
            }

            RankId rankId = curCcuTask->GetRankId();
            uint32_t dieId = INVALID_DIE_ID;
            curCcuTask->GetDieId(queId, dieId);
            auto transInstr = EnsureCcuLoopInstr<CcuLoopTransMemV3>(instrInLoop, rankId, dieId, instrId);
            if (transInstr == nullptr) {
                return HCCL_E_MEMORY;
            }
            CcuLoopCkeOpV3 waitOp;
            CcuLoopCkeOpV3 setOp;
            CHK_RET(CollectLoopCkeOpsA6(instr, sampleParam, waitOp, setOp));
            transInstr->AddWait(waitOp.ckeId, waitOp.mask);
            transInstr->AddSet(setOp.ckeId, setOp.mask);

            for (uint64_t iter = 0; iter < loopCnt; ++iter) {
                LoopGroupParamA6 iterParam;
                SetLoopParamAtIteration(sampleParam, static_cast<u32>(iter), iterParam);
                DataSlice srcSlice;
                DataSlice dstSlice;
                uint64_t len = 0;
                switch (instr->header.code) {
                    case A6_TRANSLOCMEMTOLOCMS_CODE: {
                        const auto& op = instr->v2.transLocMemToLocMS;
                        uint64_t locMemAddr = 0;
                        uint64_t addrExpandInfo = 0;
                        uint64_t msStartAddrOffset = 0;
                        const uint16_t xsId = GetXnId(op.xsId, &iterParam);
                        const uint16_t xstId = GetXnId(op.xstId, &iterParam);
                        const uint16_t xlId = GetXnId(op.xlId, &iterParam);
                        const uint16_t xoId = GetXnId(op.xoId, &iterParam);
                        CHK_GET_XN_V3(curCcuTask, queId, xsId, locMemAddr);
                        CHK_GET_XN_V3(curCcuTask, queId, xstId, addrExpandInfo);
                        CHK_GET_XN_V3(curCcuTask, queId, xoId, msStartAddrOffset);
                        const uint16_t addrExpandCoef = (addrExpandInfo >> 53) & 0x3;
                        locMemAddr = UpdateAddress(locMemAddr, &iterParam, addrExpandCoef);
                        CHK_GET_XN_V3(curCcuTask, queId, xlId, len);
                        CHK_RET(ValidateLoopLen(len));
                        CHK_RET(curCcuTask->GetStorageManager().GetSlice(locMemAddr, len, srcSlice));
                        const uint16_t locMSId = UpdateMSId(op.msId, &iterParam);
                        CHK_RET(GenSliceFromMs(locMSId, len, dstSlice));
                        transInstr->srcs.push_back(MakeCcuMemSlice(rankId, srcSlice));
                        transInstr->dsts.push_back(MakeCcuMemSlice(rankId, dstSlice));
                        transInstr->msIds.insert(locMSId);
                        break;
                    }
                    case A6_TRANSLOCMSTOLOCMEM_CODE: {
                        const auto& op = instr->v2.transLocMSToLocMem;
                        uint64_t locMemAddr = 0;
                        uint64_t addrExpandInfo = 0;
                        uint64_t msStartAddrOffset = 0;
                        const uint16_t xdId = GetXnId(op.xdId, &iterParam);
                        const uint16_t xdtId = GetXnId(op.xdtId, &iterParam);
                        const uint16_t xlId = GetXnId(op.xlId, &iterParam);
                        const uint16_t xoId = GetXnId(op.xoId, &iterParam);
                        CHK_GET_XN_V3(curCcuTask, queId, xdId, locMemAddr);
                        CHK_GET_XN_V3(curCcuTask, queId, xdtId, addrExpandInfo);
                        CHK_GET_XN_V3(curCcuTask, queId, xoId, msStartAddrOffset);
                        const uint16_t addrExpandCoef = (addrExpandInfo >> 53) & 0x3;
                        locMemAddr = UpdateAddress(locMemAddr, &iterParam, addrExpandCoef);
                        CHK_GET_XN_V3(curCcuTask, queId, xlId, len);
                        CHK_RET(ValidateLoopLen(len));
                        const uint16_t locMSId = UpdateMSId(op.msId, &iterParam);
                        CHK_RET(GenSliceFromMs(locMSId, len, srcSlice));
                        CHK_RET(curCcuTask->GetStorageManager().GetSlice(locMemAddr, len, dstSlice));
                        transInstr->srcs.push_back(MakeCcuMemSlice(rankId, srcSlice));
                        transInstr->dsts.push_back(MakeCcuMemSlice(rankId, dstSlice));
                        transInstr->msIds.insert(locMSId);
                        break;
                    }
                    case A6_TRANSLOCMSTOLOCMS_CODE: {
                        const auto& op = instr->v2.transLocMSToLocMS;
                        const uint16_t xlId = GetXnId(op.xlId, &iterParam);
                        uint64_t msStartAddrOffset = 0;
                        const uint16_t xoId = GetXnId(op.xoId, &iterParam);
                        CHK_GET_XN_V3(curCcuTask, queId, xoId, msStartAddrOffset);
                        CHK_GET_XN_V3(curCcuTask, queId, xlId, len);
                        CHK_RET(ValidateLoopLen(len));
                        const uint16_t srcMSId = UpdateMSId(op.mssId, &iterParam);
                        const uint16_t dstMSId = UpdateMSId(op.msdId, &iterParam);
                        CHK_RET(GenSliceFromMs(srcMSId, len, srcSlice));
                        CHK_RET(GenSliceFromMs(dstMSId, len, dstSlice));
                        transInstr->srcs.push_back(MakeCcuMemSlice(rankId, srcSlice));
                        transInstr->dsts.push_back(MakeCcuMemSlice(rankId, dstSlice));
                        transInstr->msIds.insert(srcMSId);
                        transInstr->msIds.insert(dstMSId);
                        break;
                    }
                    case A6_TRANSLOCMEMTOLOCMEM_CODE: {
                        const auto& op = instr->v2.transLocMemToLocMem;
                        const uint16_t xdId = GetXnId(op.xdId, &iterParam);
                        const uint16_t xdtId = GetXnId(op.xdtId, &iterParam);
                        const uint16_t xsId = GetXnId(op.xsId, &iterParam);
                        const uint16_t xlId = GetXnId(op.xlId, &iterParam);
                        uint64_t srcMemAddr = 0;
                        uint64_t dstMemAddr = 0;
                        uint64_t addrExpandInfo = 0;
                        CHK_GET_XN_V3(curCcuTask, queId, xsId, srcMemAddr);
                        CHK_GET_XN_V3(curCcuTask, queId, xdId, dstMemAddr);
                        CHK_GET_XN_V3(curCcuTask, queId, xdtId, addrExpandInfo);
                        const uint16_t addrExpandCoef = (addrExpandInfo >> 53) & 0x3;
                        srcMemAddr = UpdateAddress(srcMemAddr, &iterParam, addrExpandCoef);
                        dstMemAddr = UpdateAddress(dstMemAddr, &iterParam, addrExpandCoef);
                        CHK_GET_XN_V3(curCcuTask, queId, xlId, len);
                        CHK_RET(ValidateLoopLen(len));
                        CHK_RET(curCcuTask->GetStorageManager().GetSlice(srcMemAddr, len, srcSlice));
                        CHK_RET(curCcuTask->GetStorageManager().GetSlice(dstMemAddr, len, dstSlice));
                        transInstr->srcs.push_back(MakeCcuMemSlice(rankId, srcSlice));
                        transInstr->dsts.push_back(MakeCcuMemSlice(rankId, dstSlice));
                        break;
                    }
                    default:
                        return HCCL_E_NOT_SUPPORT;
                }
            }
            return HCCL_SUCCESS;
        }

        static HcclResult CollectReduceLoopInstrA6(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, uint16_t instrId,
            const LoopGroupParamA6& sampleParam, uint64_t loopCnt, std::shared_ptr<CcuLoopInstrV3>& instrInLoop)
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
            CHK_RET(CollectLoopCkeOpsA6(instr, sampleParam, waitOp, setOp));
            reduceInstr->AddWait(waitOp.ckeId, waitOp.mask);
            reduceInstr->AddSet(setOp.ckeId, setOp.mask);

            HcclReduceOp checkerReduceOp = HCCL_REDUCE_SUM;
            uint16_t reduceType = CcuRep::CCU_REDUCE_SUM;
            if (instr->header.code == A6_REDUCEADD_CODE) {
                checkerReduceOp = HCCL_REDUCE_SUM;
                reduceType = CcuRep::CCU_REDUCE_SUM;
            } else if (instr->header.code == A6_REDUCEMAX_CODE) {
                checkerReduceOp = HCCL_REDUCE_MAX;
                reduceType = CcuRep::CCU_REDUCE_MAX;
            } else if (instr->header.code == A6_REDUCEMIN_CODE) {
                checkerReduceOp = HCCL_REDUCE_MIN;
                reduceType = CcuRep::CCU_REDUCE_MIN;
            } else {
                return HCCL_E_NOT_SUPPORT;
            }

            DataType hcclDataType;
            CHK_RET(GetHcclDataTypeFromCCUDataType(instr->v2.reduce.dataType, reduceType, hcclDataType));
            reduceInstr->dataType = g_DataType2CheckerDataType_aicpu[hcclDataType];
            reduceInstr->reduceOp = checkerReduceOp;

            const uint16_t lengthId = GetXnId(instr->v2.reduce.XnIdLength, const_cast<LoopGroupParamA6*>(&sampleParam));
            uint64_t len = 0;
            CHK_GET_XN_V3(curCcuTask, queId, lengthId, len);
            CHK_RET(ValidateLoopLen(len));
            for (uint64_t iter = 0; iter < loopCnt; ++iter) {
                LoopGroupParamA6 iterParam;
                SetLoopParamAtIteration(sampleParam, static_cast<u32>(iter), iterParam);
                std::vector<MemSlice> srcGroup;
                for (uint16_t i = 1; i < instr->v2.reduce.count + 2; ++i) {
                    DataSlice srcSlice;
                    const uint16_t srcMSId = UpdateMSId(instr->v2.reduce.msId[i], &iterParam);
                    CHK_RET(GenSliceFromMs(srcMSId, len, srcSlice));
                    srcGroup.push_back(MakeCcuMemSlice(rankId, srcSlice));
                    reduceInstr->msIds.insert(srcMSId);
                }
                DataSlice dstSlice;
                const uint16_t dstMSId = UpdateMSId(instr->v2.reduce.msId[0], &iterParam);
                CHK_RET(GenSliceFromMs(dstMSId, len, dstSlice));
                reduceInstr->srcs.push_back(std::move(srcGroup));
                reduceInstr->dsts.push_back(MakeCcuMemSlice(rankId, dstSlice));
                reduceInstr->msIds.insert(dstMSId);
            }
            return HCCL_SUCCESS;
        }

        static HcclResult CollectClearCkeLoopInstrA6(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, uint16_t instrId,
            const LoopGroupParamA6& sampleParam, std::shared_ptr<CcuLoopInstrV3>& instrInLoop)
        {
            if (instr->header.code != A6_CLEARCKE_CODE || instr->v2.clearCKE.clearMask != 0) {
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
            CHK_RET(CollectLoopCkeOpsA6(instr, sampleParam, waitOp, setOp));
            clearInstr->AddWait(waitOp.ckeId, waitOp.mask);
            clearInstr->clearCKEId
                = UpdateCKEId(instr->v2.clearCKE.clearCKEId, const_cast<LoopGroupParamA6*>(&sampleParam));
            clearInstr->clearMask = instr->v2.clearCKE.clearMask;
            clearInstr->clearType = instr->v2.clearCKE.clearType;
            return HCCL_SUCCESS;
        }

        static HcclResult CollectMergedLoopInstrsA6(
            CcuGraphStateV3* curCcuTask, uint32_t queId, uint16_t startInstrId, uint16_t endInstrId,
            const LoopGroupParamA6& sampleParam, uint64_t loopCnt, CcuLoopInstrsV3& mergedInstrs)
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
                HcclResult ret = CollectLoopCkeOpsA6(curInstr, sampleParam, waitOp, setOp);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                if (!ConsumeProducedMask(producedMasks, waitOp.ckeId, waitOp.mask)) {
                    return HCCL_E_NOT_SUPPORT;
                }
                ProduceMask(producedMasks, setOp.ckeId, setOp.mask);

                std::shared_ptr<CcuLoopInstrV3> instrInLoop;
                if (curInstr->header.type == CTRL_TYPE) {
                    ret = CollectClearCkeLoopInstrA6(curInstr, curCcuTask, queId, insId, sampleParam, instrInLoop);
                } else if (curInstr->header.type == TRANS_TYPE) {
                    ret = CollectTransLoopInstrA6(
                        curInstr, curCcuTask, queId, insId, sampleParam, loopCnt, instrInLoop);
                } else if (curInstr->header.type == REDUCE_TYPE) {
                    ret = CollectReduceLoopInstrA6(
                        curInstr, curCcuTask, queId, insId, sampleParam, loopCnt, instrInLoop);
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

        static HcclResult CollectLoopExpandA6(
            const CcuRep::CcuInstr* loopInstr, CcuGraphStateV3* curCcuTask, uint32_t queId,
            const LoopGroupParamA6& loopParam, CcuLoopInstrsV3& mergedInstrs, uint64_t& loopCnt)
        {
            if (loopInstr->header.type != CTRL_TYPE || loopInstr->header.code != A6_LOOP_CODE
                || loopInstr->v2.loop.mode == 1) {
                return HCCL_E_NOT_SUPPORT;
            }
            RankId rankId = curCcuTask->GetRankId();
            uint32_t dieId = INVALID_DIE_ID;
            curCcuTask->GetDieId(queId, dieId);
            LoopA6 loop;
            uint16_t xpId = GetXnId(loopInstr->v2.loop.xpId, const_cast<LoopGroupParamA6*>(&loopParam));
            CHK_GET_XN_V3(curCcuTask, queId, xpId, loop.loopXp.value);
            uint16_t xnId = GetXnId(loopInstr->v2.loop.xnId, const_cast<LoopGroupParamA6*>(&loopParam));
            CHK_GET_XN_V3(curCcuTask, queId, xnId, loop.loopXn.value);
            uint16_t xmId = GetXnId(loopInstr->v2.loop.xmId, const_cast<LoopGroupParamA6*>(&loopParam));
            CHK_GET_XN_V3(curCcuTask, queId, xmId, loop.loopXm.value);

            LoopGroupParamA6 sampleParam = loopParam;
            if (sampleParam.loopXms.size() <= sampleParam.curLoopIdx) {
                sampleParam.loopXms.resize(sampleParam.curLoopIdx + 1);
            }
            sampleParam.loopXms[sampleParam.curLoopIdx] = loop;
            sampleParam.curLoopCnt = 0;
            loopCnt = static_cast<uint64_t>(loop.loopXm.loopIterCnt);
            if (loopCnt == 0) {
                return HCCL_E_NOT_SUPPORT;
            }
            return CollectMergedLoopInstrsA6(
                curCcuTask, queId, loopInstr->v2.loop.startInstrId, loopInstr->v2.loop.endInstrId, sampleParam, loopCnt,
                mergedInstrs);
        }

        static HcclResult RecordMergedLoopParamA6(
            const CcuRep::CcuInstr* loopInstr, CcuGraphStateV3* curCcuTask, uint32_t queId, LoopGroupParamA6& loopParam)
        {
            RankId rankId = curCcuTask->GetRankId();
            uint32_t dieId = INVALID_DIE_ID;
            curCcuTask->GetDieId(queId, dieId);

            LoopA6 loop;
            uint16_t xpId = GetXnId(loopInstr->v2.loop.xpId, &loopParam);
            CHK_GET_XN_V3(curCcuTask, queId, xpId, loop.loopXp.value);
            uint16_t xnId = GetXnId(loopInstr->v2.loop.xnId, &loopParam);
            CHK_GET_XN_V3(curCcuTask, queId, xnId, loop.loopXn.value);
            uint16_t xmId = GetXnId(loopInstr->v2.loop.xmId, &loopParam);
            CHK_GET_XN_V3(curCcuTask, queId, xmId, loop.loopXm.value);

            if (loopParam.loopXms.size() <= loopParam.curLoopIdx) {
                loopParam.loopXms.resize(loopParam.curLoopIdx + 1);
            }
            loopParam.loopXms[loopParam.curLoopIdx] = loop;
            return HCCL_SUCCESS;
        }

        static HcclResult TryProcessMergedLoopA6(
            const CcuRep::CcuInstr* loopInstr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            LoopGroupParamA6& loopParam, uint32_t loopGroupIdx, uint32_t expandTimes, bool& merged)
        {
            merged = false;
            CcuLoopV3 mergedLoop;
            mergedLoop.expandCnt = expandTimes;
            mergedLoop.instrId = loopInstr->v2.loop.startInstrId;
            uint64_t mergedLoopCnt = 0;
            for (uint32_t expandCnt = 0; expandCnt < expandTimes; ++expandCnt) {
                LoopGroupParamA6 expandParam = loopParam;
                expandParam.curExpandCnt = expandCnt;
                CcuLoopInstrsV3 expandInstrs;
                uint64_t loopCnt = 0;
                HcclResult ret = CollectLoopExpandA6(loopInstr, curCcuTask, queId, expandParam, expandInstrs, loopCnt);
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

            CHK_RET(RecordMergedLoopParamA6(loopInstr, curCcuTask, queId, loopParam));
            auto loopIdx = curCcuTask->loopIdx[loopGroupIdx]++;
            TaskNode* loopStart = AddLoopStartTask(queId, loopIdx, loopGroupIdx, curCcuTask);
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
            HCCL_VM_INFO(
                "Merged LoopGroup body successfully, loopInstructionId={}, loopCount={}, "
                "expandTimes={}, mergedInstructionCount={}",
                loopInstr->v2.loop.startInstrId, mergedLoopCnt, expandTimes,
                mergedLoop.loopExpands.front().instrs.size());
            merged = true;
            return HCCL_SUCCESS;
        }

        // LoopGroup指令
        HcclResult TransformLoopGroupInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            LoopGroupParamA6* loopInfo)
        {
            CHK_RET(CheckLoopGroupNotSupport(loopInfo));
            RankId rankId = curCcuTask->GetRankId();
            uint32_t dieId{0};
            curCcuTask->GetDieId(queId, dieId);
            LoopGroupParamA6 loopGroupParam{};
            uint16_t startLoopInstrId = instr->v2.loopGroup.startLoopInstrId;
            uint16_t xnId = instr->v2.loopGroup.xnId & 0x7FFF;
            CHK_GET_XN_V3(curCcuTask, queId, xnId, loopGroupParam.loopGroupXn.value);

            uint16_t xmId = instr->v2.loopGroup.xmId & 0x7FFF;
            CHK_GET_XN_V3(curCcuTask, queId, xmId, loopGroupParam.loopGroupXm.value);

            uint16_t xpId = instr->v2.loopGroup.xpId & 0x7FFF;
            CHK_GET_XN_V3(curCcuTask, queId, xpId, loopGroupParam.loopGroupXp.value);
            const uint32_t loopInsCnt = static_cast<uint32_t>(loopGroupParam.loopGroupXn.loopInsCnt);
            const uint32_t expandCnt = static_cast<uint32_t>(loopGroupParam.loopGroupXn.expandCnt);
            const uint32_t expandOffset = static_cast<uint32_t>(loopGroupParam.loopGroupXn.expandOffset);
            // 数值校验
            if (loopInsCnt > 512 || expandCnt > 511 || expandOffset > 511 || expandOffset > loopInsCnt) {
                HCCL_VM_ERROR(
                    "{} LoopGroup configuration is invalid, loopGroupXnValue={}, loopCount={}, "
                    "expandCount={}, expandOffset={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_STRUCTURE_INVALID).c_str(), loopGroupParam.loopGroupXn.value,
                    loopInsCnt, expandCnt, expandOffset);
                return HCCL_E_INTERNAL;
            }

            hcomm::CcuRep::CcuInstrInfo& microCodeQue = curCcuTask->instrInfo[queId];
            auto loopGroupIdx = curCcuTask->loopGroupIdx++;
            const uint64_t loopCnt = loopInsCnt;
            HCCL_VM_INFO(
                "Parsed LoopGroup configuration, loopCount={}, expandOffset={}, expandCount={}", loopCnt,
                static_cast<uint64_t>(expandOffset), static_cast<uint64_t>(expandCnt));

            for (u32 curLoopIdx = 0; curLoopIdx < loopCnt; curLoopIdx++) {
                loopGroupParam.curLoopIdx = curLoopIdx;
                // 计算当前Loop指令的位置
                u32 insPos = startLoopInstrId + curLoopIdx - curCcuTask->startInstrIdInQue[queId];
                const uint32_t expandTimes = (curLoopIdx < expandOffset) ? 1U : (expandCnt + 1U);
                bool merged = false;
                CHK_RET(TryProcessMergedLoopA6(
                    &microCodeQue.instrVec[insPos], curCcuTask, queId, isContinue, loopGroupParam, loopGroupIdx,
                    expandTimes, merged));
                if (!isContinue) {
                    return HCCL_SUCCESS;
                }
                if (merged) {
                    continue;
                }
                if (curLoopIdx < expandOffset) {
                    // 不用进行Loop展开
                    CHK_RET(ProcessLoopIns(
                        &microCodeQue.instrVec[insPos], curCcuTask, queId, isContinue, &loopGroupParam, loopGroupIdx));
                } else {
                    // 需要进行loop展开
                    for (u32 expandIdx = 0; expandIdx <= expandCnt; expandIdx++) {
                        loopGroupParam.curExpandCnt = expandIdx;
                        CHK_RET(ProcessLoopIns(
                            &microCodeQue.instrVec[insPos], curCcuTask, queId, isContinue, &loopGroupParam,
                            loopGroupIdx));
                    }
                }
            }

            HCCL_VM_INFO(
                "Finished LoopGroup expansion, startLoopInstructionId={}, loopGroupXnId={}, "
                "loopGroupXmId={}",
                startLoopInstrId, xnId, xmId);
            return HCCL_SUCCESS;
        }

        // SetCKBit指令
        HcclResult TransformSetCKBitInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            LoopGroupParamA6* loopGroupParam)
        {
            uint16_t clearType = instr->v2.setCKE.clearType;
            uint16_t setCKEId = UpdateCKEId(instr->v2.setCKE.setCKEId, loopGroupParam);
            uint16_t setCKEMask = instr->v2.setCKE.setCKEMask;
            uint16_t waitCKEId = UpdateCKEId(instr->v2.setCKE.waitCKEId, loopGroupParam);
            uint16_t waitCKEMask = instr->v2.setCKE.waitCKEMask;
            // 当前只支持clearType为1的场景
            if (clearType != 0x0001) {
                HCCL_VM_ERROR(
                    "{} CheckerV3 only supports clearType=1 for this instruction, "
                    "instruction=SetCKBit, actualClearType={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_UNSUPPORTED).c_str(), clearType);
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
            HCCL_VM_DEBUG(
                "Wait CKE[{}:0x{:04x}], Set CKE[{}:0x{:04x}], clearType[{}]", waitCKEId, waitCKEMask, setCKEId,
                setCKEMask, clearType);
            return HCCL_SUCCESS;
        }

        // ClearCKE指令
        HcclResult TransformClearCKEInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
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

        HcclResult TransformJumpInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
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
                CHK_GET_XN_V3(curCcuTask, queId, expectedXnId, expectedValue);
                CHK_GET_XN_V3(curCcuTask, queId, conditionXnId, conditionValue);
            }
            CHK_GET_XN_V3(curCcuTask, queId, relTarInstrXnId, relTarInstrValue);

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
                    HCCL_VM_DEBUG(
                        "Jump if Xn[{}]({}) == Xn[{}]({}), jumpMode[{}], targetXn[{}]({})", conditionXnId,
                        conditionValue, expectedXnId, expectedValue, jumpMode, relTarInstrXnId, relTarInstrValue);
                    break;
                case 1: // 不等于
                    updateNextInsIdx(conditionValue != expectedValue);
                    HCCL_VM_DEBUG(
                        "Jump if Xn[{}]({}) != Xn[{}]({}), jumpMode[{}], targetXn[{}]({})", conditionXnId,
                        conditionValue, expectedXnId, expectedValue, jumpMode, relTarInstrXnId, relTarInstrValue);
                    break;
                case 2: // 大于
                    updateNextInsIdx(conditionValue > expectedValue);
                    HCCL_VM_DEBUG(
                        "Jump if Xn[{}]({}) > Xn[{}]({}), jumpMode[{}], targetXn[{}]({})", conditionXnId,
                        conditionValue, expectedXnId, expectedValue, jumpMode, relTarInstrXnId, relTarInstrValue);
                    break;
                case 3: // 大于/等于
                    updateNextInsIdx(conditionValue >= expectedValue);
                    HCCL_VM_DEBUG(
                        "Jump if Xn[{}]({}) >= Xn[{}]({}), jumpMode[{}], targetXn[{}]({})", conditionXnId,
                        conditionValue, expectedXnId, expectedValue, jumpMode, relTarInstrXnId, relTarInstrValue);
                    break;
                case 4: // 小于
                    updateNextInsIdx(conditionValue < expectedValue);
                    HCCL_VM_DEBUG(
                        "Jump if Xn[{}]({}) < Xn[{}]({}), jumpMode[{}], targetXn[{}]({})", conditionXnId,
                        conditionValue, expectedXnId, expectedValue, jumpMode, relTarInstrXnId, relTarInstrValue);
                    break;
                case 5: // 小于/等于
                    updateNextInsIdx(conditionValue <= expectedValue);
                    HCCL_VM_DEBUG(
                        "Jump if Xn[{}]({}) <= Xn[{}]({}), jumpMode[{}], targetXn[{}]({})", conditionXnId,
                        conditionValue, expectedXnId, expectedValue, jumpMode, relTarInstrXnId, relTarInstrValue);
                    break;
                default: // 无条件跳转
                    updateNextInsIdx(true);
                    HCCL_VM_DEBUG(
                        "Jump always, Xn[{}]({}), jumpMode[{}], targetXn[{}]({})", conditionXnId, conditionValue,
                        jumpMode, relTarInstrXnId, relTarInstrValue);
                    break;
            }
            return HCCL_SUCCESS;
        }

        // Wait指令
        HcclResult TransformWaitInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
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
            CHK_GET_XN_V3(curCcuTask, queId, expectedXnId, expectedValue);
            CHK_GET_XN_V3(curCcuTask, queId, conditionXnId, conditionValue);

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
                    HCCL_VM_DEBUG(
                        "Wait Xn[{}]({}) == Xn[{}]({}), satisfied[{}]", conditionXnId, conditionValue, expectedXnId,
                        expectedValue, conditionValue == expectedValue);
                    break;
                case 1: // 不等于
                    updateNextInsIdx(conditionValue != expectedValue);
                    HCCL_VM_DEBUG(
                        "Wait Xn[{}]({}) != Xn[{}]({}), satisfied[{}]", conditionXnId, conditionValue, expectedXnId,
                        expectedValue, conditionValue != expectedValue);
                    break;
                case 2: // 大于
                    updateNextInsIdx(conditionValue > expectedValue);
                    HCCL_VM_DEBUG(
                        "Wait Xn[{}]({}) > Xn[{}]({}), satisfied[{}]", conditionXnId, conditionValue, expectedXnId,
                        expectedValue, conditionValue > expectedValue);
                    break;
                case 3: // 大于/等于
                    updateNextInsIdx(conditionValue >= expectedValue);
                    HCCL_VM_DEBUG(
                        "Wait Xn[{}]({}) >= Xn[{}]({}), satisfied[{}]", conditionXnId, conditionValue, expectedXnId,
                        expectedValue, conditionValue >= expectedValue);
                    break;
                case 4: // 小于
                    updateNextInsIdx(conditionValue < expectedValue);
                    HCCL_VM_DEBUG(
                        "Wait Xn[{}]({}) < Xn[{}]({}), satisfied[{}]", conditionXnId, conditionValue, expectedXnId,
                        expectedValue, conditionValue < expectedValue);
                    break;
                case 5: // 小于/等于
                    updateNextInsIdx(conditionValue <= expectedValue);
                    HCCL_VM_DEBUG(
                        "Wait Xn[{}]({}) <= Xn[{}]({}), satisfied[{}]", conditionXnId, conditionValue, expectedXnId,
                        expectedValue, conditionValue <= expectedValue);
                    break;
                default: // 参考等于
                    updateNextInsIdx(conditionValue == expectedValue);
                    HCCL_VM_DEBUG(
                        "Wait Xn[{}]({}) == Xn[{}]({}), satisfied[{}]", conditionXnId, conditionValue, expectedXnId,
                        expectedValue, conditionValue == expectedValue);
                    break;
            }
            curCcuTask->microCodePosInQue[queId] = nextInsIdx - curCcuTask->startInstrIdInQue[queId];
            return HCCL_SUCCESS;
        }

        // Fence指令
        HcclResult TransformFenceInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            LoopGroupParamA6* loopGroupParam)
        {
            HCCL_VM_ERROR(
                "{} Fence instruction is not supported by CheckerV3 graph expansion",
                MakeErrorCodeText(ErrorCode::GRAPH_UNSUPPORTED).c_str());
            return HCCL_E_PARA;
        }

        // TransLocMemToLocMS指令：将内存中的数据搬移到本端MS
        HcclResult TransformTransLocMemToLocMSInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
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
            CHK_GET_XN_V3(curCcuTask, queId, xsId, locMemAddr);
            CHK_GET_XN_V3(curCcuTask, queId, xstId, addrExpandInfo);
            CHK_GET_XN_V3(curCcuTask, queId, xoId, msStartAddrOffset);
            // 取[54:53]bit位
            addrExpandCoef = (addrExpandInfo >> 53) & 0x3;
            locMemAddr = UpdateAddress(locMemAddr, loopGroupParam, addrExpandCoef);
            uint64_t len = 0;
            CHK_GET_XN_V3(curCcuTask, queId, xlId, len);
            if (len == 0) {
                const uint32_t instrId = curCcuTask->microCodePosInQue[queId] + curCcuTask->startInstrIdInQue[queId];
                HCCL_VM_ERROR(
                    "{} Transfer size is 0, instruction=TransLocMemToLocMS, rankId={}, dieId={}, "
                    "queueId={}, instrId={}, transferSize={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_STRUCTURE_INVALID).c_str(), static_cast<uint32_t>(rankId), dieId,
                    queId, instrId, len);
                return HCCL_E_INTERNAL;
            }

            DataSlice srcSlice;
            DataSlice dstSlice;
            // todo: MS地址偏移计算
            CHK_RET(curCcuTask->GetStorageManager().GetSlice(locMemAddr, len, srcSlice));
            CHK_RET(GenSliceFromMs(locMSId, len, dstSlice));

            AddLocalCopy(rankId, queId, curCcuTask, srcSlice, dstSlice);

            CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCkeId, setCkeMask));

            return HCCL_SUCCESS;
        }

        // TransformTransLocMSToLocMemInstr指令：将本端MS数据搬运到内存中
        HcclResult TransformTransLocMSToLocMemInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
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
            CHK_GET_XN_V3(curCcuTask, queId, xdId, locMemAddr);
            CHK_GET_XN_V3(curCcuTask, queId, xdtId, addrExpandInfo);
            CHK_GET_XN_V3(curCcuTask, queId, xoId, msStartAddrOffset);
            // 取[54:53]bit位
            addrExpandCoef = (addrExpandInfo >> 53) & 0x3;
            locMemAddr = UpdateAddress(locMemAddr, loopGroupParam, addrExpandCoef);
            uint64_t len = 0;
            CHK_GET_XN_V3(curCcuTask, queId, xlId, len);
            if (len == 0) {
                const uint32_t instrId = curCcuTask->microCodePosInQue[queId] + curCcuTask->startInstrIdInQue[queId];
                HCCL_VM_ERROR(
                    "{} Transfer size is 0, instruction=TransLocMSToLocMem, rankId={}, dieId={}, "
                    "queueId={}, instrId={}, transferSize={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_STRUCTURE_INVALID).c_str(), static_cast<uint32_t>(rankId), dieId,
                    queId, instrId, len);
                return HCCL_E_INTERNAL;
            }

            DataSlice srcSlice;
            DataSlice dstSlice;
            // todo: MS地址偏移计算
            CHK_RET(GenSliceFromMs(locMSId, len, srcSlice));
            CHK_RET(curCcuTask->GetStorageManager().GetSlice(locMemAddr, len, dstSlice));

            AddLocalCopy(rankId, queId, curCcuTask, srcSlice, dstSlice);

            CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCkeId, setCkeMask));
            return HCCL_SUCCESS;
        }

        //   TransformTransLocMSToLocMS指令：将本端MS数据搬运到另一端MS中，用于die内部数据搬运
        HcclResult TransformTransLocMSToLocMSInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
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
            CHK_GET_XN_V3(curCcuTask, queId, xoId, msStartAddrOffset);
            uint64_t len = 0;
            CHK_GET_XN_V3(curCcuTask, queId, xlId, len);
            if (len == 0) {
                const uint32_t instrId = curCcuTask->microCodePosInQue[queId] + curCcuTask->startInstrIdInQue[queId];
                HCCL_VM_ERROR(
                    "{} Transfer size is 0, instruction=TransLocMSToLocMS, rankId={}, dieId={}, "
                    "queueId={}, instrId={}, transferSize={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_STRUCTURE_INVALID).c_str(), static_cast<uint32_t>(rankId), dieId,
                    queId, instrId, len);
                return HCCL_E_INTERNAL;
            }

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
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
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
            CHK_GET_XN_V3(curCcuTask, queId, xsId, srcMemAddr);
            CHK_GET_XN_V3(curCcuTask, queId, xdId, dstMemAddr);
            CHK_GET_XN_V3(curCcuTask, queId, xdtId, addrExpandInfo);
            // 取[54:53]bit位
            addrExpandCoef = (addrExpandInfo >> 53) & 0x3;
            srcMemAddr = UpdateAddress(srcMemAddr, loopGroupParam, addrExpandCoef);
            dstMemAddr = UpdateAddress(dstMemAddr, loopGroupParam, addrExpandCoef);
            uint64_t len = 0;
            CHK_GET_XN_V3(curCcuTask, queId, xlId, len);
            if (len == 0) {
                const uint32_t instrId = curCcuTask->microCodePosInQue[queId] + curCcuTask->startInstrIdInQue[queId];
                HCCL_VM_ERROR(
                    "{} Transfer size is 0, instruction=TransLocMemToLocMem, rankId={}, dieId={}, "
                    "queueId={}, instrId={}, transferSize={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_STRUCTURE_INVALID).c_str(), static_cast<uint32_t>(rankId), dieId,
                    queId, instrId, len);
                return HCCL_E_INTERNAL;
            }

            DataSlice srcSlice;
            DataSlice dstSlice;
            // todo: MS地址偏移计算
            CHK_RET(curCcuTask->GetStorageManager().GetSlice(srcMemAddr, len, srcSlice));
            CHK_RET(curCcuTask->GetStorageManager().GetSlice(dstMemAddr, len, dstSlice));

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
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            LoopGroupParamA6* loopGroupParam)
        {
            // 参数校验
            uint16_t udfType = instr->v2.transMem.udfType;
            if (udfType != 0) {
                HCCL_VM_ERROR(
                    "{} This TransMem udfType is not supported by CheckerV3 graph expansion, "
                    "udfType={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_UNSUPPORTED).c_str(), udfType);
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
            CHK_GET_XN_V3(curCcuTask, queId, xlId, len);
            // 通过channal找对端信息
            uint16_t xcId = GetXnId(instr->v2.transMem.xcId, loopGroupParam);
            uint64_t xcValue = 0;
            CHK_GET_XN_V3(curCcuTask, queId, xcId, xcValue);
            // 需要找对端的rank信息
            if (!IsExistRemoteDieInfo(rankId, dieId, xcValue)) {
                HCCL_VM_ERROR(
                    "{} Cannot find remote rank/die mapping for this channel, instruction=TransMem, "
                    "localRankId={}, localDieId={}, channelValue={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_RESOURCE_NOT_FOUND).c_str(), static_cast<uint32_t>(rankId),
                    dieId, static_cast<unsigned long long>(xcValue));
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
                    instr->v2.transMem.reduceDataType, ccuReduceTypeMap[instr->v2.transMem.reduceOpCode],
                    hcclDataType));
                checkerDataType = g_DataType2CheckerDataType_aicpu[hcclDataType];
            }

            if (dmaOpCode == 3 || dmaOpCode == 5) {
                // 写数据 dst为对端 src为本端
                if (msMode == 1) {
                    // XsId寄存器中存的就是MSId
                    CHK_RET(GenSliceFromMs(UpdateMSId(instr->v2.transMem.xsId, loopGroupParam), len, srcSlice));
                } else {
                    CHK_GET_XN_V3(curCcuTask, queId, xsId, xsValue);
                    xsValue = (srcMode == 1) ? UpdateAddress(xsValue, loopGroupParam) :
                                               UpdateAddressWithoutStride(xsValue, loopGroupParam);
                    CHK_RET(curCcuTask->GetStorageManager().GetSlice(xsValue, len, srcSlice));
                }
                CHK_GET_XN_V3(curCcuTask, queId, xdId, xdValue);
                xdValue = (dstMode == 1) ? UpdateAddress(xdValue, loopGroupParam) :
                                           UpdateAddressWithoutStride(xdValue, loopGroupParam);
                CHK_RET(curCcuTask->GetStorageManager().GetSlice(xdValue, len, dstSlice));
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
                CHK_GET_XN_V3(curCcuTask, queId, xsId, xsValue);
                xsValue = (srcMode == 1) ? UpdateAddress(xsValue, loopGroupParam) :
                                           UpdateAddressWithoutStride(xsValue, loopGroupParam);
                CHK_RET(curCcuTask->GetStorageManager().GetSlice(xsValue, len, srcSlice));
                if (msMode == 1) {
                    CHK_RET(GenSliceFromMs(UpdateMSId(instr->v2.transMem.xdId, loopGroupParam), len, dstSlice));
                } else {
                    CHK_GET_XN_V3(curCcuTask, queId, xdId, xdValue);
                    xdValue = (dstMode == 1) ? UpdateAddress(xdValue, loopGroupParam) :
                                               UpdateAddressWithoutStride(xdValue, loopGroupParam);
                    CHK_RET(curCcuTask->GetStorageManager().GetSlice(xdValue, len, dstSlice));
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
                HCCL_VM_ERROR(
                    "{} This TransMem dmaOpCode is not supported by CheckerV3 graph expansion, "
                    "dmaOpCode={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_UNSUPPORTED).c_str(), dmaOpCode);
                return HCCL_E_PARA;
            }

            // 写notify信息
            if (dmaOpCode == 5) {
                uint16_t xnId = GetXnId(instr->v2.transMem.xsId, loopGroupParam);
                uint64_t xnAddr = 0;
                CHK_GET_XN_V3(curCcuTask, queId, xnId, xnAddr);
                uint16_t xnIdRmt = 0;
                CHK_RET(
                    AllRankParamRecorder::Global()->GetXnIdByAddr(dieId, CcuComponerntType::XN_A6, xnAddr, xnIdRmt));
                uint32_t notifyValue = instr->v2.transMem.value;
                CHK_RET(AllRankParamRecorder::Global()->SetXn(
                    rmtRankId, rmtDieId, xnIdRmt, static_cast<uint64_t>(notifyValue)));
            }

            uint16_t setCkeId = UpdateCKEId(instr->v2.transMem.setCKEId, loopGroupParam);
            uint16_t setCkeMask = instr->v2.transMem.setCKEMask;
            CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCkeId, setCkeMask));
            return HCCL_SUCCESS;
        }

        // TransformSyncXnWtx指令：同步指令，将本端寄存器的值以write操作同步到远端，用于同步XN和WTX
        HcclResult TransformSyncXnWtxInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            LoopGroupParamA6* loopGroupParam)
        {
            RankId rankId = curCcuTask->GetRankId();
            uint32_t dieId{0};
            curCcuTask->GetDieId(queId, dieId);
            uint16_t xdId = GetXnId(instr->v2.syncWtX.xdId, loopGroupParam);
            uint64_t xdValue = 0;
            CHK_GET_XN_V3(curCcuTask, queId, xdId, xdValue);

            uint16_t xcId = GetXnId(instr->v2.syncWtX.xcId, loopGroupParam);
            uint64_t xcValue = 0;
            CHK_GET_XN_V3(curCcuTask, queId, xcId, xcValue);
            if (!IsExistRemoteDieInfo(rankId, dieId, xcValue)) {
                HCCL_VM_ERROR(
                    "{} Cannot find remote rank/die mapping for this channel, instruction=SyncXnWtx, "
                    "localRankId={}, localDieId={}, channelValue={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_RESOURCE_NOT_FOUND).c_str(), static_cast<uint32_t>(rankId),
                    dieId, static_cast<unsigned long long>(xcValue));
                return HCCL_E_PARA;
            }
            // 需要找对端的rank信息
            RankId rmtRankId = g_allRankChannelInfo[rankId][dieId][xcValue].dstRank;
            uint32_t rmtDieId = g_allRankChannelInfo[rankId][dieId][xcValue].remoteDieId;

            uint16_t parMode = instr->v2.syncWtX.parMode;
            uint64_t xsValue = instr->v2.syncWtX.xsId;
            if (parMode == 1) {
                uint16_t xsId = GetXnId(instr->v2.syncWtX.xsId, loopGroupParam);
                CHK_GET_XN_V3(curCcuTask, queId, xsId, xsValue);
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
                HCCL_VM_ERROR(
                    "{} SyncXnWtx destination address does not map to a supported remote target; "
                    "expected XN_A6 or CKE_A6, xdId={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_UNSUPPORTED).c_str(), xdId);
                return HCCL_E_PARA;
            }

            uint16_t notifyValid = instr->v2.syncWtX.notifyValid;
            // 写notify信息
            if (notifyValid == 1) {
                uint16_t xnId = GetXnId(instr->v2.syncWtX.xnId, loopGroupParam);
                uint64_t xnAddr = 0;
                CHK_GET_XN_V3(curCcuTask, queId, xnId, xnAddr);
                uint16_t cekIdRmtId = 0;
                CHK_RET(AllRankParamRecorder::Global()->GetXnIdByAddr(
                    dieId, CcuComponerntType::CKE_A6, xnAddr, cekIdRmtId));
                uint32_t notifyValue = instr->v2.syncWtX.value;
                CHK_RET(ProcessSetMask(
                    rmtRankId, rmtDieId, curCcuTask, queId, cekIdRmtId, static_cast<uint64_t>(notifyValue)));
            }
            return HCCL_SUCCESS;
        }

        // TransformSyncAtx指令：同步指令，将本端寄存器的值以atomic操作同步到远端，用于同步ATX
        HcclResult TransformSyncAtxInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            LoopGroupParamA6* loopGroupParam)
        {
            HCCL_VM_ERROR(
                "{} SyncAtx instruction is not supported by CheckerV3 graph expansion",
                MakeErrorCodeText(ErrorCode::GRAPH_UNSUPPORTED).c_str());
            return HCCL_E_PARA;
        }

        // ReduceAdd指令
        HcclResult TransformReduceAddInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
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

            HCCL_VM_INFO(
                "Completed local SUM reduce, msList={}, sourceCount={}, dataType={}, "
                "castEnabled={}, setCkeId={}, setMask=0x{:04x}",
                ParseMSList(instr).c_str(), count, dataType, castEn, setCKEId, setCKEMask);

            return HCCL_SUCCESS;
        }

        HcclResult TransformReduceMaxInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
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

            HCCL_VM_INFO(
                "Completed local MAX reduce, msList={}, sourceCount={}, dataType={}, "
                "setCkeId={}, setMask=0x{:04x}",
                ParseMSList(instr).c_str(), count, dataType, setCKEId, setCKEMask);

            return HCCL_SUCCESS;
        }

        // ReduceMin指令
        HcclResult TransformReduceMinInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
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

            HCCL_VM_INFO(
                "Completed local MIN reduce, msList={}, sourceCount={}, dataType={}, "
                "setCkeId={}, setMask=0x{:04x}",
                HcclSim::TaskGraphGeneratorV3::ParseMSList(instr).c_str(), count, dataType, setCKEId, setCKEMask);
            return HCCL_SUCCESS;
        }

        // 以下为数据转换函数，只起到转换数据作用
        HcclResult TransformLoadSqeArgsToXnInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            void* loopParam)
        {
            auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
            return TransformLoadSqeArgsToXnInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
        }

        HcclResult TransformLoadImdToXInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            void* loopParam)
        {
            auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
            return TransformLoadImdToXInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
        }

        HcclResult TransformLoadXInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            void* loopParam)
        {
            auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
            return TransformLoadXInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
        }

        HcclResult TransformStoreXInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            void* loopParam)
        {
            auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
            return TransformStoreXInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
        }

        HcclResult TransformClearXInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            void* loopParam)
        {
            auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
            return TransformClearXInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
        }

        HcclResult TransformNopInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            void* loopParam)
        {
            auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
            return TransformNopInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
        }

        HcclResult TransformLoadInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            void* loopParam)
        {
            auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
            return TransformLoadInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
        }

        HcclResult TransformStoreInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            void* loopParam)
        {
            auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
            return TransformStoreInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
        }

        HcclResult TransformAddInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            void* loopParam)
        {
            auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
            return TransformAddInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
        }

        HcclResult TransformSubInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            void* loopParam)
        {
            auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
            return TransformSubInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
        }

        HcclResult TransformMulInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            void* loopParam)
        {
            auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
            return TransformMulInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
        }

        HcclResult TransformANDInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            void* loopParam)
        {
            auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
            return TransformANDInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
        }

        HcclResult TransformORInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            void* loopParam)
        {
            auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
            return TransformORInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
        }

        HcclResult TransformNOTInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            void* loopParam)
        {
            auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
            return TransformNOTInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
        }

        HcclResult TransformXORInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            void* loopParam)
        {
            auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
            return TransformXORInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
        }

        HcclResult TransformSHLInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            void* loopParam)
        {
            auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
            return TransformSHLInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
        }

        HcclResult TransformSHRInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            void* loopParam)
        {
            auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
            return TransformSHRInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
        }

        HcclResult TransformPopcntInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            void* loopParam)
        {
            auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
            return TransformPopcntInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
        }

        HcclResult TransformLoopInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            void* loopParam)
        {
            auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
            return TransformLoopInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
        }

        HcclResult TransformLoopGroupInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            void* loopParam)
        {
            auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
            return TransformLoopGroupInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
        }

        HcclResult TransformSetCKBitInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            void* loopParam)
        {
            auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
            return TransformSetCKBitInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
        }

        HcclResult TransformClearCKEInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            void* loopParam)
        {
            auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
            return TransformClearCKEInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
        }

        HcclResult TransformJumpInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            void* loopParam)
        {
            auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
            return TransformJumpInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
        }

        HcclResult TransformWaitInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            void* loopParam)
        {
            auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
            return TransformWaitInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
        }

        HcclResult TransformFenceInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            void* loopParam)
        {
            auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
            return TransformFenceInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
        }

        HcclResult TransformTransLocMemToLocMSInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            void* loopParam)
        {
            auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
            return TransformTransLocMemToLocMSInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
        }

        HcclResult TransformTransLocMSToLocMemInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            void* loopParam)
        {
            auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
            return TransformTransLocMSToLocMemInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
        }

        HcclResult TransformTransLocMSToLocMSInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            void* loopParam)
        {
            auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
            return TransformTransLocMSToLocMSInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
        }

        HcclResult TransformTransLocMemToLocMemInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            void* loopParam)
        {
            auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
            return TransformTransLocMemToLocMemInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
        }

        HcclResult TransformTransMemInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            void* loopParam)
        {
            auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
            return TransformTransMemInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
        }

        HcclResult TransformSyncXnWtxInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            void* loopParam)
        {
            auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
            return TransformSyncXnWtxInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
        }

        HcclResult TransformSyncAtxInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            void* loopParam)
        {
            auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
            return TransformSyncAtxInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
        }

        HcclResult TransformReduceAddInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            void* loopParam)
        {
            auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
            return TransformReduceAddInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
        }

        HcclResult TransformReduceMaxInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            void* loopParam)
        {
            auto* loopGroupParam = static_cast<LoopGroupParamA6*>(loopParam);
            return TransformReduceMaxInstr(instr, curCcuTask, queId, isContinue, loopGroupParam);
        }

        HcclResult TransformReduceMinInstr(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            void* loopParam)
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
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            LoopGroupParamA6* loopGroupParam, uint32_t loopGroupIdx)
        {
            uint16_t mode = instr->v2.loop.mode;
            RankId rankId = curCcuTask->GetRankId();
            uint32_t dieId{0};
            curCcuTask->GetDieId(queId, dieId);
            LoopA6 loop{};
            uint16_t xpId = GetXnId(instr->v2.loop.xpId, loopGroupParam);
            CHK_GET_XN_V3(curCcuTask, queId, xpId, loop.loopXp.value);
            if (mode == 1) {
                // Xp中存储的是Loop Context ID，当mode为1时，在loop与自身预期值checkEntry相等时才进行展开。
                // 否则，直接跳过该loop。
                uint16_t wishCKEBit = instr->v2.loop.wishCKEBit;
                const uint16_t loopCtxId = static_cast<uint16_t>(loop.loopXp.loopCtxId);
                if (wishCKEBit != loopCtxId + A6_LOOP_BOARD_ENTRY) {
                    isContinue = false;
                    return HCCL_SUCCESS;
                }
            }
            uint16_t startInstrId = instr->v2.loop.startInstrId;
            uint16_t endInstrId = instr->v2.loop.endInstrId;
            uint16_t xnId = GetXnId(instr->v2.loop.xnId, loopGroupParam);
            CHK_GET_XN_V3(curCcuTask, queId, xnId, loop.loopXn.value);

            uint16_t xmId = GetXnId(instr->v2.loop.xmId, loopGroupParam);
            CHK_GET_XN_V3(curCcuTask, queId, xmId, loop.loopXm.value);

            loopGroupParam->loopXms.push_back(loop);
            const u32 loopIterCnt = static_cast<u32>(loop.loopXm.loopIterCnt);

            auto loopIdx = curCcuTask->loopIdx[loopGroupIdx]++;
            auto loopStart = AddLoopStartTask(
                queId, loopIdx, loopGroupIdx, curCcuTask, startInstrId, endInstrId, static_cast<uint64_t>(loopIterCnt),
                1ULL); // loop前新增loopStart标记节点

            HCCL_VM_INFO("Entered loop execution, loopCount={}", static_cast<uint64_t>(loopIterCnt));
            hcomm::CcuRep::CcuInstrInfo& microCodeQue = curCcuTask->instrInfo[queId];
            u32& pos = curCcuTask->microCodePosInQue[queId];
            const u32 savedPos = pos;
            for (u32 curLoopCnt = 0; curLoopCnt < loopIterCnt; curLoopCnt++) {
                loopGroupParam->curLoopCnt = curLoopCnt;
                HCCL_VM_INFO(
                    "Running loop iteration, loopCount={}, currentIteration={}", static_cast<uint64_t>(loopIterCnt),
                    curLoopCnt);
                for (uint16_t insId = startInstrId; insId <= endInstrId; insId++) {
                    pos = insId - curCcuTask->startInstrIdInQue[queId];
                    // 获取当前要处理的指令
                    const CcuRep::CcuInstr* instrVec = &microCodeQue.instrVec[pos];
                    HCCL_VM_DEBUG(
                        "Expanding loop instruction, rankId={}, queueId={}, instructionId={}", curCcuTask->GetRankId(),
                        queId, insId);
                    if (transformA6InstrSqeMap.count(instrVec->header.header) == 0) {
                        HCCL_VM_ERROR(
                            "{} This A6 CCU instruction type is not supported during loop "
                            "expansion, rankId={}, queueId={}, instructionId={}, instructionHeader=0x{:04x}",
                            MakeErrorCodeText(ErrorCode::GRAPH_UNSUPPORTED).c_str(), curCcuTask->GetRankId(), queId,
                            insId, instrVec->header.header);
                        pos = savedPos;
                        return HCCL_E_INTERNAL;
                    }
                    CHK_RET(transformA6InstrSqeMap[instrVec->header.header](
                        instrVec, curCcuTask, queId, isContinue, loopGroupParam));
                }
            }
            pos = savedPos;
            auto loopEnd = AddLoopEndTask(queId, loopIdx, loopGroupIdx, curCcuTask); // loop后新增loopEnd标记节点

            HCCL_VM_DEBUG(
                "Finished loop expansion, startInstructionId={}, endInstructionId={}, "
                "loopCountRegisterId={}",
                startInstrId, endInstrId, xnId);
            return HCCL_SUCCESS;
        }

    } // namespace

    InstructMapA6::InstructMapA6() { transformInstrSqeMap = transformA6InstrSqeMap; }

    HcclResult InstructMapA6::Transform(
        const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam)
    {
        auto it = transformInstrSqeMap.find(instr->header.header);
        if (it == transformInstrSqeMap.end()) {
            HCCL_VM_ERROR(
                "{} This A6 CCU instruction type is not supported by CheckerV3 graph "
                "expansion, rankId={}, queueId={}, instructionHeader=0x{:04x}",
                MakeErrorCodeText(ErrorCode::GRAPH_UNSUPPORTED).c_str(),
                curCcuTask == nullptr ? std::string("null") : std::to_string(curCcuTask->GetRankId()), queId,
                instr->header.header);
            return HCCL_E_INTERNAL;
        }
        auto* param = static_cast<LoopGroupParamA6*>(loopParam);
        return it->second(instr, curCcuTask, queId, isContinue, param);
    }

} // namespace TaskGraphGeneratorV3
} // namespace HcclSim
