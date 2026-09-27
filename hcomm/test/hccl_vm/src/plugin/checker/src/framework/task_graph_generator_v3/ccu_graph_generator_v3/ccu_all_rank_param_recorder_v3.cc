/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_all_rank_param_recorder_v3.h"
#include <sys/types.h>
#include "sim_log.h"
#include "utils/error_codes.h"

namespace HcclSim {
namespace TaskGraphGeneratorV3 {

    AllRankParamRecorder* AllRankParamRecorder::Global()
    {
        static AllRankParamRecorder* allRankParamRecorder = new AllRankParamRecorder;
        return allRankParamRecorder;
    }

    void AllRankParamRecorder::Reset()
    {
        curXn.clear();
        curGSA.clear();
        curCKE.clear();
        curHBM.clear();
        seenPost.clear();
        postNodeMeta.clear();
        devType_ = DevType::DEV_TYPE_COUNT;
        ccu_resource_base_addr_.clear();
        return;
    }

    void AllRankParamRecorder::InitParam()
    {
        curXn.clear();
        curGSA.clear();
        curHBM.clear();
        seenPost.clear();
        postNodeMeta.clear();
    }

    void AllRankParamRecorder::RegisterPostNode(TaskNode* node, const CcuPostNodeMetaV3& meta)
    {
        if (node == nullptr) {
            return;
        }
        postNodeMeta[node] = meta;
    }

    uint32_t AllRankParamRecorder::GetPostRemainingCkeMask(const TaskNode* node) const
    {
        auto it = postNodeMeta.find(node);
        if (it == postNodeMeta.end()) {
            return 0;
        }
        return it->second.remainingCkeMask;
    }

    void AllRankParamRecorder::SetPostRemainingCkeMask(TaskNode* node, uint32_t remainingCkeMask)
    {
        auto it = postNodeMeta.find(node);
        if (it == postNodeMeta.end()) {
            return;
        }
        it->second.remainingCkeMask = static_cast<uint16_t>(remainingCkeMask);
    }

    const CcuPostNodeMetaV3* AllRankParamRecorder::GetPostNodeMeta(const TaskNode* node) const
    {
        auto it = postNodeMeta.find(node);
        if (it == postNodeMeta.end()) {
            return nullptr;
        }
        return &it->second;
    }

    HcclResult AllRankParamRecorder::SetXn(uint32_t rankId, uint32_t dieId, uint16_t xnId, uint64_t xnValue)
    {
        curXn[rankId][dieId][xnId] = xnValue;
        return HCCL_SUCCESS;
    }

    HcclResult AllRankParamRecorder::SetGSA(uint32_t rankId, uint32_t dieId, uint16_t gsaId, uint64_t gsaValue)
    {
        curGSA[rankId][dieId][gsaId] = gsaValue;
        return HCCL_SUCCESS;
    }

    HcclResult AllRankParamRecorder::SetCKE(uint32_t rankId, uint32_t dieId, uint16_t ckeId, uint16_t ckeValue)
    {
        curCKE[rankId][dieId][ckeId] = ckeValue;
        return HCCL_SUCCESS;
    }

    HcclResult AllRankParamRecorder::GetXn(uint32_t rankId, uint32_t dieId, uint16_t xnId, uint64_t& xnValue)
    {
        if (curXn.find(rankId) != curXn.end()) {
            if (curXn[rankId].find(dieId) != curXn[rankId].end()) {
                if (curXn[rankId][dieId].find(xnId) != curXn[rankId][dieId].end()) {
                    xnValue = curXn[rankId][dieId][xnId];
                    return HCCL_SUCCESS;
                }
            }
        }
        xnValue = 0;
        return HCCL_E_PARA;
    }

    HcclResult AllRankParamRecorder::GetGSA(uint32_t rankId, uint32_t dieId, uint16_t gsaId, uint64_t& gsaValue)
    {
        if (curGSA.find(rankId) != curGSA.end()) {
            if (curGSA[rankId].find(dieId) != curGSA[rankId].end()) {
                if (curGSA[rankId][dieId].find(gsaId) != curGSA[rankId][dieId].end()) {
                    gsaValue = curGSA[rankId][dieId][gsaId];
                    return HCCL_SUCCESS;
                }
            }
        }
        gsaValue = 0;
        return HCCL_E_PARA;
    }

    HcclResult AllRankParamRecorder::GetCKE(uint32_t rankId, uint32_t dieId, uint16_t ckeId, uint16_t& ckeValue)
    {
        if (curCKE.find(rankId) != curCKE.end()) {
            if (curCKE[rankId].find(dieId) != curCKE[rankId].end()) {
                if (curCKE[rankId][dieId].find(ckeId) != curCKE[rankId][dieId].end()) {
                    ckeValue = curCKE[rankId][dieId][ckeId];
                    return HCCL_SUCCESS;
                }
            }
        }

        ckeValue = 0;
        return HCCL_SUCCESS;
    }

    std::map<uint16_t, uint64_t> AllRankParamRecorder::GetXnSnapshot(uint32_t rankId, uint32_t dieId) const
    {
        auto rankIt = curXn.find(rankId);
        if (rankIt == curXn.end()) {
            return {};
        }
        auto dieIt = rankIt->second.find(dieId);
        if (dieIt == rankIt->second.end()) {
            return {};
        }
        return dieIt->second;
    }

    std::map<uint16_t, uint64_t> AllRankParamRecorder::GetGSASnapshot(uint32_t rankId, uint32_t dieId) const
    {
        auto rankIt = curGSA.find(rankId);
        if (rankIt == curGSA.end()) {
            return {};
        }
        auto dieIt = rankIt->second.find(dieId);
        if (dieIt == rankIt->second.end()) {
            return {};
        }
        return dieIt->second;
    }

    std::map<uint16_t, uint16_t> AllRankParamRecorder::GetCKESnapshot(uint32_t rankId, uint32_t dieId) const
    {
        auto rankIt = curCKE.find(rankId);
        if (rankIt == curCKE.end()) {
            return {};
        }
        auto dieIt = rankIt->second.find(dieId);
        if (dieIt == rankIt->second.end()) {
            return {};
        }
        return dieIt->second;
    }

    HcclResult AllRankParamRecorder::CheckAllPostMatch()
    {
        for (const auto& rankPair : seenPost) {
            for (const auto& diePair : rankPair.second) {
                for (const auto& regPair : diePair.second) {
                    for (const auto* post : regPair.second) {
                        const auto* meta = GetPostNodeMeta(post);
                        if (meta != nullptr) {
                            HCCL_VM_WARN(
                                "{} Found CCU post/local-post tasks that were never consumed by "
                                "any Wait task:\n  unconsumedPostCount={}, remainingCkeMask=0x{:x}, isLocal={}\n"
                                "  firstUnconsumedPostNode={}",
                                MakeErrorCodeText(ErrorCode::GRAPH_UNMATCHED).c_str(), regPair.second.size(),
                                meta->remainingCkeMask, meta->isLocal,
                                post == nullptr ? "node=null" : post->Describe().c_str());
                            continue;
                        }
                        HCCL_VM_WARN(
                            "{} Found CCU post/local-post tasks that were never consumed by any "
                            "Wait task:\n  unconsumedPostCount={}\n  firstUnconsumedPostNode={}",
                            MakeErrorCodeText(ErrorCode::GRAPH_UNMATCHED).c_str(), regPair.second.size(),
                            post == nullptr ? "node=null" : post->Describe().c_str());
                    }
                }
            }
        }

        return HCCL_SUCCESS;
    }

    HcclResult
    AllRankParamRecorder::SetHBM(uint32_t rankId, uint32_t dieId, uint64_t hbmAddr, const std::vector<uint64_t>& data)
    {
        if (data.size() % 8 != 0) {
            return HCCL_E_PARA;
        }
        curHBM[rankId][dieId][hbmAddr] = data;
        return HCCL_SUCCESS;
    }

    HcclResult
    AllRankParamRecorder::GetHBM(uint32_t rankId, uint32_t dieId, uint64_t hbmAddr, std::vector<uint64_t>& data)
    {
        if (curHBM.find(rankId) != curHBM.end()) {
            if (curHBM[rankId].find(dieId) != curHBM[rankId].end()) {
                if (curHBM[rankId][dieId].find(hbmAddr) != curHBM[rankId][dieId].end()) {
                    data = curHBM[rankId][dieId][hbmAddr];
                    return HCCL_SUCCESS;
                }
            }
        }
        data.clear();
        return HCCL_E_PARA;
    }

    const uint64_t XN_OFFSET_ADDR_A6 = 0x100000;
    const uint64_t XN_CKB_ADDR_A6 = 0x140000;
    const uint64_t XN_PFE_ADDR_A6 = 0x148000;
    const uint64_t XN_CHANNEL_ADDR_A6 = 0x150000;
    const uint64_t XN_JETTY_ADDR_A6 = 0x170000;
    const uint64_t XN_MISSION_ADDR_A6 = 0x178000;
    const uint64_t XN_LOOP_ADDR_A6 = 0x180000;
    const uint64_t XN_CCUA0_MS_ADDR_A6 = 0x4000000; // MS的起始地址CCU0为64M
    const uint64_t XN_CCUA1_MS_ADDR_A6 = 0x4200000; // MS的起始地址CCU1为66M
    const uint64_t XN_CCUA2_MS_ADDR_A6 = 0x4400000; // MS的起始地址CCU2为68M
    const uint64_t XN_CCUA3_MS_ADDR_A6 = 0x4600000; // MS的起始地址CCU3为70M
    const uint64_t XN_MS_SIZE_A6 = 4000;            // 每个MS占用4k
    const uint64_t MS_INTERLEAVE_NUM = 8;           // 每组8个MS
    const uint64_t CCUA_MS_ADDR_GAP = 0x200000;     // CCUA寄存器的MS地址间隔
    const uint64_t CCUA_NUM_A6 = 4;
    const uint64_t GROUP_TOTAL_MS = MS_INTERLEAVE_NUM * CCUA_NUM_A6; // 一组总共32个MS

    const uint64_t CCU_MS_START_ADDR[4]
        = {XN_CCUA0_MS_ADDR_A6, XN_CCUA1_MS_ADDR_A6, XN_CCUA2_MS_ADDR_A6, XN_CCUA3_MS_ADDR_A6};

    HcclResult AllRankParamRecorder::GetMSIdByAddr(uint32_t dieId, uint64_t addr, uint16_t& msId)
    {
        if (dieId >= ccu_resource_base_addr_.size()) {
            const std::string maxDieId = ccu_resource_base_addr_.empty() ?
                                             "null" :
                                             std::to_string(static_cast<uint32_t>(ccu_resource_base_addr_.size() - 1));
            HCCL_VM_ERROR(
                "{} dieId is out of range when converting address to MS id, dieId={}, "
                "maxDieId={}",
                MakeErrorCodeText(ErrorCode::GRAPH_OUT_OF_RANGE).c_str(), dieId, maxDieId);
            return HCCL_E_PARA;
        }
        uint64_t msAddr = addr - ccu_resource_base_addr_[dieId];
        uint64_t ccuIndex{UINT64_MAX};
        for (uint64_t i = 0; i < CCUA_NUM_A6; ++i) {
            uint64_t ccuEnd = CCU_MS_START_ADDR[i] + CCUA_MS_ADDR_GAP;
            if (msAddr >= CCU_MS_START_ADDR[i] && msAddr < ccuEnd) {
                ccuIndex = i;
                break;
            }
        }
        if (ccuIndex == UINT64_MAX) {
            HCCL_VM_ERROR(
                "{} Address does not fall into any known MS address range, localMsAddr={}, "
                "rawAddr={}",
                MakeErrorCodeText(ErrorCode::GRAPH_ADDRESS_INVALID).c_str(), msAddr, addr);
            return HCCL_E_PARA;
        }
        uint64_t localOffset = msAddr - CCU_MS_START_ADDR[ccuIndex];
        uint64_t localMsIndex = localOffset / XN_MS_SIZE_A6;
        uint64_t groupIndex = localMsIndex / MS_INTERLEAVE_NUM;
        uint64_t posInGroup = localMsIndex % MS_INTERLEAVE_NUM;
        msId = groupIndex * GROUP_TOTAL_MS + ccuIndex * MS_INTERLEAVE_NUM + posInGroup;
        return HCCL_SUCCESS;
    }

    // 根据type的类型找基础地址
    uint64_t findBaseAddr(CcuComponerntType type)
    {
        if (type == CcuComponerntType::XN_A6) {
            return XN_OFFSET_ADDR_A6;
        } else if (type == CcuComponerntType::CKE_A6) {
            return XN_CKB_ADDR_A6;
        } else if (type == CcuComponerntType::PFE_A6) {
            return XN_PFE_ADDR_A6;
        } else if (type == CcuComponerntType::CHANNEL_A6) {
            return XN_CHANNEL_ADDR_A6;
        } else if (type == CcuComponerntType::JETTY_A6) {
            return XN_JETTY_ADDR_A6;
        } else if (type == CcuComponerntType::MISSION_A6) {
            return XN_MISSION_ADDR_A6;
        } else if (type == CcuComponerntType::LOOP_A6) {
            return XN_LOOP_ADDR_A6;
        } else {
            return 0x000000000;
        }
    }

    // 根据地址值来找到类型
    CcuComponerntType findTypeByAddr(uint64_t addr)
    {
        if (addr >= XN_OFFSET_ADDR_A6 && addr < XN_CKB_ADDR_A6) {
            return CcuComponerntType::XN_A6;
        } else if (addr >= XN_CKB_ADDR_A6 && addr < XN_PFE_ADDR_A6) {
            return CcuComponerntType::CKE_A6;
        } else if (addr >= XN_PFE_ADDR_A6 && addr < XN_CHANNEL_ADDR_A6) {
            return CcuComponerntType::PFE_A6;
        } else if (addr >= XN_CHANNEL_ADDR_A6 && addr < XN_JETTY_ADDR_A6) {
            return CcuComponerntType::CHANNEL_A6;
        } else if (addr >= XN_JETTY_ADDR_A6 && addr < XN_MISSION_ADDR_A6) {
            return CcuComponerntType::JETTY_A6;
        } else if (addr >= XN_MISSION_ADDR_A6 && addr < XN_LOOP_ADDR_A6) {
            return CcuComponerntType::MISSION_A6;
        } else if (addr >= XN_LOOP_ADDR_A6 && addr < XN_LOOP_ADDR_A6 + 0x8000) {
            return CcuComponerntType::LOOP_A6;
        } else {
            return CcuComponerntType::UNKNOWN;
        }
    }

    // 根据type的类型返回寄存器每块的数量
    uint16_t findBaseSize(CcuComponerntType type)
    {
        if (type == CcuComponerntType::XN_A6 || type == CcuComponerntType::CKE_A6
            || type == CcuComponerntType::PFE_A6) {
            return 8;
        } else if (
            type == CcuComponerntType::CHANNEL_A6 || type == CcuComponerntType::JETTY_A6
            || type == CcuComponerntType::LOOP_A6) {
            return 32;
        } else if (type == CcuComponerntType::MISSION_A6) {
            return 64;
        } else {
            return 0;
        }
    }

    // 通过XnId所在的地址值来找到XnId
    HcclResult
    AllRankParamRecorder::GetXnAndTypeIdByAddr(uint32_t dieId, uint64_t xnAddr, CcuComponerntType& type, uint16_t& xnId)
    {
        if (dieId >= ccu_resource_base_addr_.size()) {
            const std::string maxDieId = ccu_resource_base_addr_.empty() ?
                                             "null" :
                                             std::to_string(static_cast<uint32_t>(ccu_resource_base_addr_.size() - 1));
            HCCL_VM_ERROR(
                "{} dieId is out of range when converting address to register id, dieId={}, "
                "maxDieId={}",
                MakeErrorCodeText(ErrorCode::GRAPH_OUT_OF_RANGE).c_str(), dieId, maxDieId);
            return HCCL_E_PARA;
        }
        type = findTypeByAddr(xnAddr - ccu_resource_base_addr_[dieId]);
        if (type == CcuComponerntType::UNKNOWN) {
            HCCL_VM_ERROR(
                "{} Address does not belong to any known CCU component range, addr={}, "
                "dieId={}, dieBaseAddr={}",
                MakeErrorCodeText(ErrorCode::GRAPH_ADDRESS_INVALID).c_str(), xnAddr, dieId,
                ccu_resource_base_addr_[dieId]);
            return HCCL_E_PARA;
        }
        uint64_t baseAddr = findBaseAddr(type) + ccu_resource_base_addr_[dieId];
        uint16_t sizeofXn = findBaseSize(type);
        xnId = static_cast<uint16_t>((xnAddr - baseAddr) / static_cast<uint64_t>(sizeofXn));
        return HCCL_SUCCESS;
    }

    // 通过XnId所在的地址值来找到XnId
    HcclResult
    AllRankParamRecorder::GetXnIdByAddr(uint32_t dieId, CcuComponerntType type, uint64_t xnAddr, uint16_t& xnId)
    {
        if (dieId >= ccu_resource_base_addr_.size()) {
            const std::string maxDieId = ccu_resource_base_addr_.empty() ?
                                             "null" :
                                             std::to_string(static_cast<uint32_t>(ccu_resource_base_addr_.size() - 1));
            HCCL_VM_ERROR(
                "{} dieId is out of range when converting address to register id, dieId={}, "
                "maxDieId={}",
                MakeErrorCodeText(ErrorCode::GRAPH_OUT_OF_RANGE).c_str(), dieId, maxDieId);
            return HCCL_E_PARA;
        }
        // 需要判断界限  todo
        uint64_t baseAddr = findBaseAddr(type) + ccu_resource_base_addr_[dieId];
        uint16_t sizeofXn = findBaseSize(type);
        xnId = static_cast<uint16_t>((xnAddr - baseAddr) / static_cast<uint64_t>(sizeofXn));
        return HCCL_SUCCESS;
    }

    // 通过XnId所在的地址值来找到XnId
    HcclResult
    AllRankParamRecorder::GetAddrByXnId(uint32_t dieId, CcuComponerntType type, uint16_t xnId, uint64_t& xnAddr)
    {
        // 需要判断界限  todo
        if (dieId >= ccu_resource_base_addr_.size()) {
            const std::string maxDieId = ccu_resource_base_addr_.empty() ?
                                             "null" :
                                             std::to_string(static_cast<uint32_t>(ccu_resource_base_addr_.size() - 1));
            HCCL_VM_ERROR(
                "{} dieId is out of range when converting register id to address, dieId={}, "
                "maxDieId={}",
                MakeErrorCodeText(ErrorCode::GRAPH_OUT_OF_RANGE).c_str(), dieId, maxDieId);
            return HCCL_E_PARA;
        }
        uint64_t baseAddr = findBaseAddr(type) + ccu_resource_base_addr_[dieId];
        uint16_t sizeofXn = findBaseSize(type);
        xnAddr = baseAddr + static_cast<uint64_t>(xnId) * sizeofXn;
        return HCCL_SUCCESS;
    }

} // namespace TaskGraphGeneratorV3
} // namespace HcclSim
