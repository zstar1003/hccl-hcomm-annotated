/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_resource_manager.h"

#include <cstdint>
#include <cstring>

#include "ccu_microcode_v1.h"
#include "sim_log.h"
#include "ccu_reduce_operator.h"
#include "sim_models.h"
#include "store_sim_shm_memory_common.h"

using namespace std;
using namespace hcomm::CcuRep;

void CcuResourceManager::Reset()
{
    HCCL_VM_DEBUG("Clearing all CCU resource state (KE/XN/MS/simulators)");
    for (size_t rankId = 0; rankId < ccuResData_.v1Res.size(); rankId++) {
        if (ccuResData_.v1Res[rankId] != nullptr) {
            ccuResData_.v1Res[rankId]->Reset();
        }
    }
    for (size_t rankId = 0; rankId < ccuResData_.v2Res.size(); rankId++) {
        if (ccuResData_.v2Res[rankId] != nullptr) {
            ccuResData_.v2Res[rankId]->Reset();
        }
    }
}

void CcuResourceManager::Init(
    int rankId, int rankSize, RunnerCcuVersion version, const std::vector<uint64_t>& ccuResourceBaseAddr)
{
    ccuResData_.version = version;
    if (ccuResData_.version == RunnerCcuVersion::CCU_V1) {
        if (static_cast<int>(ccuResData_.v1Res.size()) < rankSize) {
            ccuResData_.v1Res.resize(rankSize);
        }
        ccuResData_.v1Res[rankId] = std::make_unique<CcuResourceV1>(rankId, rankSize);
    } else if (ccuResData_.version == RunnerCcuVersion::CCU_V2) {
        if (static_cast<int>(ccuResData_.v2Res.size()) < rankSize) {
            ccuResData_.v2Res.resize(rankSize);
        }
        ccuResData_.v2Res[rankId] = std::make_unique<CcuResourceV2>(rankId, rankSize);
    } else {
        HCCL_VM_ERROR("ccu version {} not supported", static_cast<int>(ccuResData_.version));
    }
    ccuResourceBaseAddr_ = ccuResourceBaseAddr;
}

void CcuResourceManager::InitInstrInfo(int rankId, int dieId, const CcuInstrData& ccuInstrInfo)
{
    auto version = ccuResData_.version;
    if (version == RunnerCcuVersion::CCU_V1) {
        ccuResData_.v1Res[rankId]->instrSpace_[dieId] = ccuInstrInfo;
    } else if (ccuResData_.version == RunnerCcuVersion::CCU_V2) {
        ccuResData_.v2Res[rankId]->instrSpace_[dieId] = ccuInstrInfo;
    } else {
        HCCL_VM_ERROR("ccu version {} not supported", static_cast<int>(version));
    }
}

void CcuResourceManager::InitChannelInfo(int rankId, const RankChannelInfo& channelInfo)
{
    auto version = ccuResData_.version;
    if (version == RunnerCcuVersion::CCU_V1) {
        ccuResData_.v1Res[rankId]->channelId2RmtRankMap_ = channelInfo;
    } else if (ccuResData_.version == RunnerCcuVersion::CCU_V2) {
        ccuResData_.v2Res[rankId]->channelId2RmtRankMap_ = channelInfo;
    } else {
        HCCL_VM_ERROR("ccu version {} not supported", static_cast<int>(version));
    }
}

void CcuResourceManager::AddTaskInfo(int rankId, const HcclTaskMetaData& task)
{
    auto dieId = task.taskData.ccu.dieId;
    if (dieId >= HcclSim::DIE_NUM) {
        HCCL_VM_ERROR("invalid dieId[{}].", dieId);
        return;
    }
    auto version = ccuResData_.version;
    if (version == RunnerCcuVersion::CCU_V1) {
        ccuResData_.v1Res[rankId]->ccuTaskInfos_[dieId] = task.taskData.ccu;
    } else if (ccuResData_.version == RunnerCcuVersion::CCU_V2) {
        ccuResData_.v2Res[rankId]->ccuTaskInfos_[dieId] = task.taskData.ccu;
    } else {
        HCCL_VM_ERROR("ccu version {} not supported", static_cast<int>(version));
    }
}

uint64_t CcuResourceManager::GetSqeArgValue(int rankId, int dieId, uint16_t argId) const
{
    auto version = ccuResData_.version;
    if (version == RunnerCcuVersion::CCU_V1) {
        return ccuResData_.v1Res[rankId]->ccuTaskInfos_[dieId].args[argId];
    } else if (ccuResData_.version == RunnerCcuVersion::CCU_V2) {
        return ccuResData_.v2Res[rankId]->ccuTaskInfos_[dieId].args[argId];
    } else {
        HCCL_VM_ERROR("ccu version {} not supported", static_cast<int>(version));
        return U64_INVALID;
    }
}

uint64_t CcuResourceManager::GetXnValue(int rankId, int dieId, uint16_t xnId) const
{
    auto version = ccuResData_.version;
    if (version == RunnerCcuVersion::CCU_V1) {
        return ccuResData_.v1Res[rankId]->xn_[dieId][xnId];
    } else if (ccuResData_.version == RunnerCcuVersion::CCU_V2) {
        return ccuResData_.v2Res[rankId]->xn_[dieId][xnId];
    } else {
        HCCL_VM_ERROR("ccu version {} not supported", static_cast<int>(version));
        return U64_INVALID;
    }
}

uint64_t CcuResourceManager::GetGsaValue(int rankId, int dieId, uint16_t gsaId) const
{
    auto version = ccuResData_.version;
    if (version == RunnerCcuVersion::CCU_V1) {
        return ccuResData_.v1Res[rankId]->gsa_[dieId][gsaId];
    } else if (ccuResData_.version == RunnerCcuVersion::CCU_V2) {
        return ccuResData_.v2Res[rankId]->gsa_[dieId][gsaId];
    } else {
        HCCL_VM_ERROR("ccu version {} not supported", static_cast<int>(version));
        return U64_INVALID;
    }
}

uint16_t CcuResourceManager::GetCkeValue(int rankId, int dieId, uint16_t ckeId) const
{
    auto version = ccuResData_.version;
    if (version == RunnerCcuVersion::CCU_V1) {
        return ccuResData_.v1Res[rankId]->cke_[dieId][ckeId];
    } else if (ccuResData_.version == RunnerCcuVersion::CCU_V2) {
        return ccuResData_.v2Res[rankId]->cke_[dieId][ckeId];
    } else {
        HCCL_VM_ERROR("ccu version {} not supported", static_cast<int>(version));
        return U16_INVALID;
    }
}

std::pair<int, int> CcuResourceManager::GetRmtCcu(int rankId, int dieId, uint16_t channelId) const
{
    int rmtRank = S32_INVALID;
    int rmtDie = S32_INVALID;

    if (channelId >= SimCcuV1::MAX_CCU_CHANNEL_NUM) {
        HCCL_VM_ERROR("invalid channelId[{}], max={}", channelId, SimCcuV1::MAX_CCU_CHANNEL_NUM);
        return std::make_pair(S32_INVALID, S32_INVALID);
    }

    auto version = ccuResData_.version;
    if (version == RunnerCcuVersion::CCU_V1) {
        rmtRank = ccuResData_.v1Res[rankId]->channelId2RmtRankMap_[dieId][channelId].rankId;
        rmtDie = ccuResData_.v1Res[rankId]->channelId2RmtRankMap_[dieId][channelId].dieId;
    } else if (ccuResData_.version == RunnerCcuVersion::CCU_V2) {
        rmtRank = ccuResData_.v2Res[rankId]->channelId2RmtRankMap_[dieId][channelId].rankId;
        rmtDie = ccuResData_.v2Res[rankId]->channelId2RmtRankMap_[dieId][channelId].dieId;
    } else {
        HCCL_VM_ERROR("ccu version {} not supported", static_cast<int>(version));
        return std::make_pair(S32_INVALID, S32_INVALID);
    }

    if (dieId >= HcclSim::DIE_NUM || rmtRank == S32_INVALID || rmtDie == S32_INVALID) {
        HCCL_VM_ERROR("get invalid value, dieId{}, rmtRank{}, rmtDie{}", dieId, rmtRank, rmtDie);
        return std::make_pair(S32_INVALID, S32_INVALID);
    }

    return std::make_pair(rmtRank, rmtDie);
}

void CcuResourceManager::UpdateXnValue(int rankId, int dieId, uint16_t xnId, uint64_t value)
{
    auto version = ccuResData_.version;
    if (version == RunnerCcuVersion::CCU_V1) {
        ccuResData_.v1Res[rankId]->xn_[dieId][xnId] = value;
    } else if (ccuResData_.version == RunnerCcuVersion::CCU_V2) {
        ccuResData_.v2Res[rankId]->xn_[dieId][xnId] = value;
    } else {
        HCCL_VM_ERROR("ccu version {} not supported", static_cast<int>(version));
    }
}

void CcuResourceManager::UpdateGsaValue(int rankId, int dieId, uint16_t gsaId, uint64_t value)
{
    auto version = ccuResData_.version;
    if (version == RunnerCcuVersion::CCU_V1) {
        ccuResData_.v1Res[rankId]->gsa_[dieId][gsaId] = value;
    } else if (ccuResData_.version == RunnerCcuVersion::CCU_V2) {
        ccuResData_.v2Res[rankId]->gsa_[dieId][gsaId] = value;
    } else {
        HCCL_VM_ERROR("ccu version {} not supported", static_cast<int>(version));
    }
}

void CcuResourceManager::UpdateCkeValue(int rankId, int dieId, uint16_t ckeId, uint16_t value)
{
    auto version = ccuResData_.version;
    if (version == RunnerCcuVersion::CCU_V1) {
        ccuResData_.v1Res[rankId]->cke_[dieId][ckeId] = value;
    } else if (ccuResData_.version == RunnerCcuVersion::CCU_V2) {
        ccuResData_.v2Res[rankId]->cke_[dieId][ckeId] = value;
    } else {
        HCCL_VM_ERROR("ccu version {} not supported", static_cast<int>(version));
    }
}

void CcuResourceManager::TransMemToMem(
    void* srcBuf, void* dstBuf, uint64_t length, bool reduceEn, uint16_t reduceOp, uint16_t dataType)
{
    HCCL_VM_DEBUG("srcBuf={}, dstBuf={}, length={}", srcBuf, dstBuf, length);
    if (srcBuf == nullptr) {
        HCCL_VM_ERROR("srcBuf invalid...");
        return;
    }

    if (dstBuf == nullptr) {
        HCCL_VM_ERROR("dstBuf invalid...");
        return;
    }

    sim::PhyMemBlock srcPhyMem{};
    auto srcAddr = sim::AcquireDevPtrInNoHostProcess((void*)srcBuf, srcPhyMem);
    if (srcAddr == nullptr) {
        HCCL_VM_ERROR("无法获取srcBuf的设备地址(addr= {:x})！", reinterpret_cast<uintptr_t>(srcBuf));
        return;
    }

    sim::PhyMemBlock dstPhyMem{};
    auto dstAddr = sim::AcquireDevPtrInNoHostProcess((void*)dstBuf, dstPhyMem);
    if (dstAddr == nullptr) {
        HCCL_VM_ERROR("无法获取dstBuf的设备地址(addr= {:x})！", reinterpret_cast<uintptr_t>(dstBuf));
        return;
    }

    if (reduceEn) {
        ReduceProcess(srcAddr, dstAddr, length, dataType, reduceOp, ccuResData_.version);
    } else {
        memcpy(dstAddr, srcAddr, length);
    }

    sim::ReleaseInNoHostProcess(srcPhyMem);
    sim::ReleaseInNoHostProcess(dstPhyMem);
}

void CcuResourceManager::TransMSToMS(
    int srcRank, int srcDie, int dstRank, int dstDie, uint16_t srcMsId, uint16_t dstMsId, uint16_t length)
{
    memcpy(GetMsAddr(dstRank, dstDie, dstMsId), GetMsAddr(srcRank, srcDie, srcMsId), length);
}

bool CcuResourceManager::TransMSToMem(int rankId, int dieId, uint16_t msId, void* buf, uint16_t length)
{
    if (buf == nullptr) {
        HCCL_VM_ERROR("param invalid...");
        return false;
    }

    sim::PhyMemBlock dstPhyMem{};
    auto dstAddr = sim::AcquireDevPtrInNoHostProcess((void*)buf, dstPhyMem);
    if (dstAddr == nullptr) {
        HCCL_VM_ERROR("无法获取buf的设备地址(addr= {:x})！", reinterpret_cast<uintptr_t>(buf));
        return false;
    }

    memcpy(dstAddr, GetMsAddr(rankId, dieId, msId), length);

    sim::ReleaseInNoHostProcess(dstPhyMem);
    return true;
}

bool CcuResourceManager::TransMemToXn(int rankId, int dieId, uint16_t xnId, uint64_t buf, uint16_t length)
{
    sim::PhyMemBlock srcPhyMem{};
    auto srcAddr = sim::AcquireDevPtrInNoHostProcess((void*)buf, srcPhyMem);
    if (srcAddr == nullptr) {
        HCCL_VM_ERROR("无法获取buf的设备地址(addr= {:x})！", buf);
        return false;
    }
    memcpy(GetXnAddr(rankId, dieId, xnId), srcAddr, length);

    sim::ReleaseInNoHostProcess(srcPhyMem);
    return true;
}

bool CcuResourceManager::TransXnToMem(int rankId, int dieId, uint16_t xnId, uint64_t buf, uint16_t length)
{
    sim::PhyMemBlock dstPhyMem{};
    auto dstAddr = sim::AcquireDevPtrInNoHostProcess((void*)buf, dstPhyMem);
    if (dstAddr == nullptr) {
        HCCL_VM_ERROR("无法获取buf的设备地址(addr= {:x})！", buf);
        return false;
    }
    memcpy(dstAddr, GetXnAddr(rankId, dieId, xnId), length);

    sim::ReleaseInNoHostProcess(dstPhyMem);
    return true;
}

bool CcuResourceManager::TransMemToMS(int rankId, int dieId, uint16_t msId, void* buf, uint16_t length)
{
    if (buf == nullptr) {
        HCCL_VM_ERROR("param invalid...");
        return false;
    }

    sim::PhyMemBlock srcPhyMem{};
    auto srcAddr = sim::AcquireDevPtrInNoHostProcess((void*)buf, srcPhyMem);
    if (srcAddr == nullptr) {
        HCCL_VM_ERROR("无法获取buf的设备地址(addr= {:x})！", reinterpret_cast<uintptr_t>(buf));
        return false;
    }
    memcpy(GetMsAddr(rankId, dieId, msId), srcAddr, length);

    sim::ReleaseInNoHostProcess(srcPhyMem);
    return true;
}

char* CcuResourceManager::GetMsAddr(int rankId, int dieId, uint16_t msId) const
{
    auto version = ccuResData_.version;
    uint32_t offset = static_cast<uint32_t>(msId) * HcclSim::BYTE_NUM_4K; // 一个MS容量为4KB
    if (version == RunnerCcuVersion::CCU_V1) {
        return (ccuResData_.v1Res[rankId]->ms_[dieId].data() + offset);
    } else if (ccuResData_.version == RunnerCcuVersion::CCU_V2) {
        return (ccuResData_.v2Res[rankId]->ms_[dieId].data() + offset);
    } else {
        HCCL_VM_ERROR("ccu version {} not supported", static_cast<int>(version));
        return nullptr;
    }
}

uint64_t* CcuResourceManager::GetXnAddr(int rankId, int dieId, uint16_t xnId) const
{
    auto version = ccuResData_.version;
    uint32_t offset = static_cast<uint32_t>(xnId) * 64; // 一个MS容量为4KB
    if (version == RunnerCcuVersion::CCU_V1) {
        return (ccuResData_.v1Res[rankId]->xn_[dieId].data() + offset);
    } else if (ccuResData_.version == RunnerCcuVersion::CCU_V2) {
        return (ccuResData_.v2Res[rankId]->xn_[dieId].data() + offset);
    } else {
        HCCL_VM_ERROR("ccu version {} not supported", static_cast<int>(version));
        return nullptr;
    }
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

bool CcuResourceManager::GetMSIdByAddr(uint32_t dieId, uint64_t addr, uint16_t& msId)
{
    if (dieId >= ccuResourceBaseAddr_.size()) {
        HCCL_VM_ERROR("dieId is out of range, dieId=[{}]", dieId);
        return false;
    }
    uint64_t msAddr = addr - ccuResourceBaseAddr_[dieId];
    uint64_t ccuIndex{UINT64_MAX};
    for (uint64_t i = 0; i < CCUA_NUM_A6; ++i) {
        uint64_t ccuEnd = CCU_MS_START_ADDR[i] + CCUA_MS_ADDR_GAP;
        if (msAddr >= CCU_MS_START_ADDR[i] && msAddr < ccuEnd) {
            ccuIndex = i;
            break;
        }
    }
    if (ccuIndex == UINT64_MAX) {
        HCCL_VM_ERROR("msAddr is out of range, msAddr=[{}],addr=[{}]", msAddr, addr);
        return false;
    }
    uint64_t localOffset = msAddr - CCU_MS_START_ADDR[ccuIndex];
    uint64_t localMsIndex = localOffset / XN_MS_SIZE_A6;
    uint64_t groupIndex = localMsIndex / MS_INTERLEAVE_NUM;
    uint64_t posInGroup = localMsIndex % MS_INTERLEAVE_NUM;
    msId = groupIndex * GROUP_TOTAL_MS + ccuIndex * MS_INTERLEAVE_NUM + posInGroup;
    return true;
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
    if (type == CcuComponerntType::XN_A6 || type == CcuComponerntType::CKE_A6 || type == CcuComponerntType::PFE_A6) {
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
bool CcuResourceManager::GetXnAndTypeIdByAddr(uint32_t dieId, uint64_t xnAddr, CcuComponerntType& type, uint16_t& xnId)
{
    if (dieId >= ccuResourceBaseAddr_.size()) {
        HCCL_VM_ERROR("dieId is out of range, dieId=[{}]", dieId);
        return false;
    }
    type = findTypeByAddr(xnAddr - ccuResourceBaseAddr_[dieId]);
    if (type == CcuComponerntType::UNKNOWN) {
        HCCL_VM_ERROR(
            "unknown type, addr=[{}],dieId=[{}],ccuResourceBaseAddr_[dieId]=[{}]", xnAddr, dieId,
            ccuResourceBaseAddr_[dieId]);
        return false;
    }
    uint64_t baseAddr = findBaseAddr(type) + ccuResourceBaseAddr_[dieId];
    uint16_t sizeofXn = findBaseSize(type);
    xnId = static_cast<uint16_t>((xnAddr - baseAddr) / static_cast<uint64_t>(sizeofXn));
    return true;
}

// 通过XnId所在的地址值来找到XnId
bool CcuResourceManager::GetXnIdByAddr(uint32_t dieId, CcuComponerntType type, uint64_t xnAddr, uint16_t& xnId)
{
    if (dieId >= ccuResourceBaseAddr_.size()) {
        HCCL_VM_ERROR("dieId is out of range, dieId=[{}]", dieId);
        return false;
    }
    // 需要判断界限  todo
    uint64_t baseAddr = findBaseAddr(type) + ccuResourceBaseAddr_[dieId];
    uint16_t sizeofXn = findBaseSize(type);
    xnId = static_cast<uint16_t>((xnAddr - baseAddr) / static_cast<uint64_t>(sizeofXn));
    return true;
}

// 通过XnId所在的地址值来找到XnId
bool CcuResourceManager::GetAddrByXnId(uint32_t dieId, CcuComponerntType type, uint16_t xnId, uint64_t& xnAddr)
{
    // 需要判断界限  todo
    if (dieId >= ccuResourceBaseAddr_.size()) {
        HCCL_VM_ERROR("dieId is out of range, dieId=[{}]", dieId);
        return false;
    }
    uint64_t baseAddr = findBaseAddr(type) + ccuResourceBaseAddr_[dieId];
    uint16_t sizeofXn = findBaseSize(type);
    xnAddr = baseAddr + static_cast<uint64_t>(xnId) * sizeofXn;
    return true;
}

uint16_t CcuResourceManager::GetInstrCnt(int rankId, int dieId) const
{
    auto version = ccuResData_.version;
    if (version == RunnerCcuVersion::CCU_V1) {
        return ccuResData_.v1Res[rankId]->instrSpace_[dieId].instrCnt;
    } else if (ccuResData_.version == RunnerCcuVersion::CCU_V2) {
        return ccuResData_.v2Res[rankId]->instrSpace_[dieId].instrCnt;
    } else {
        HCCL_VM_ERROR("ccu version {} not supported", static_cast<int>(version));
        return U16_INVALID;
    }
}

std::vector<hcomm::CcuRep::CcuInstr> CcuResourceManager::GetInstrData(int rankId, int dieId) const
{
    auto version = ccuResData_.version;
    if (version == RunnerCcuVersion::CCU_V1) {
        return ccuResData_.v1Res[rankId]->instrSpace_[dieId].instrData;
    } else if (ccuResData_.version == RunnerCcuVersion::CCU_V2) {
        return ccuResData_.v2Res[rankId]->instrSpace_[dieId].instrData;
    } else {
        HCCL_VM_ERROR("ccu version {} not supported", static_cast<int>(version));
        return {};
    }
}

std::string CcuResourceManager::GetInstrDescribe(int rankId, int dieId, int instrId) const
{
#ifndef DEVICE_STUB
    auto version = ccuResData_.version;
    if (version == RunnerCcuVersion::CCU_V1) {
        // return hcomm::CcuRep::ParseInstr(&(ccuResData_.v1Res[rankId]->instrSpace_[dieId].instrData[instrId]));
    } else if (ccuResData_.version == RunnerCcuVersion::CCU_V2) {
        // return hcomm::CcuRep::ParseInstr(&(ccuResData_.v2Res[rankId]->instrSpace_[dieId].instrData[instrId]));
    } else {
        HCCL_VM_ERROR("ccu version {} not supported", static_cast<int>(version));
        return "";
    }
#endif
    return "";
}

std::shared_ptr<CcuSimulator>
CcuResourceManager::InitSimulator(int rankId, int dieId, uint16_t instrStartId, uint16_t endInstrId, uint16_t instCnt)
{
    auto version = ccuResData_.version;
    if (version == RunnerCcuVersion::CCU_V1) {
        if (ccuResData_.v1Res[rankId]->simulators_[dieId] == nullptr) {
            ccuResData_.v1Res[rankId]->simulators_[dieId]
                = std::make_shared<CcuSimulator>(rankId, dieId, instrStartId, endInstrId, instCnt, version);
            return ccuResData_.v1Res[rankId]->simulators_[dieId];
        } else {
            ccuResData_.v1Res[rankId]->simulators_[dieId]->Init(instrStartId, endInstrId, instCnt, version);
            return ccuResData_.v1Res[rankId]->simulators_[dieId];
        }
    } else if (version == RunnerCcuVersion::CCU_V2) {
        if (ccuResData_.v2Res[rankId]->simulators_[dieId] == nullptr) {
            ccuResData_.v2Res[rankId]->simulators_[dieId]
                = std::make_shared<CcuSimulator>(rankId, dieId, instrStartId, endInstrId, instCnt, version);
            return ccuResData_.v2Res[rankId]->simulators_[dieId];
        } else {
            ccuResData_.v2Res[rankId]->simulators_[dieId]->Init(instrStartId, endInstrId, instCnt, version);
            return ccuResData_.v2Res[rankId]->simulators_[dieId];
        }
    } else {
        HCCL_VM_ERROR("ccu version {} not supported", static_cast<int>(version));
        return nullptr;
    }
}

void CcuResourceManager::DumpCcuInstructions(int rankId) const
{
    if (!enableDump_) {
        return;
    }
#ifndef DEVICE_STUB
    for (uint32_t dieId = 0; dieId < HcclSim::DIE_NUM; dieId++) {
        HCCL_VM_DEBUG(
            "==========================print ccu instructions start: rank[{}], dieId[{}]==========================",
            rankId, dieId);
        auto instrInfoCnt = GetInstrCnt(rankId, dieId);
        HCCL_VM_DEBUG("*******************rankId=[{}], dieId=[{}], instrCnt=[{}]", rankId, dieId, instrInfoCnt);

        for (uint32_t i = 0; i < instrInfoCnt; i++) {
            HCCL_VM_TRACE("ccu instruction info: {}: {}", i, GetInstrDescribe(rankId, dieId, i));
        }
        HCCL_VM_DEBUG(
            "==========================print ccu instructions end: rank[{}], dieId[{}]==========================",
            rankId, dieId);
    }
#endif
}

void CcuResourceManager::DumpCcuXnResouceInfo(int rankId) const
{
    for (uint32_t dieId = 0; dieId < HcclSim::DIE_NUM; dieId++) {
        HCCL_VM_DEBUG(
            "==========================All CCU XN Resouce Info Start: rank[{}], dieId[{}]===========================",
            rankId, dieId);
        HCCL_VM_DEBUG("-------------------------DieId[{}] XN Resouce Info Start-------------------------", dieId);
        for (uint32_t xnId = 0; xnId < SimCcuV1::CCU_RESOURCE_XN_MAX; xnId++) {
            auto xnValue = GetXnValue(rankId, dieId, xnId);
            if (xnValue != 0) {
                HCCL_VM_TRACE("[XN: Id = {}, Value = {:x}]", xnId, xnValue);
            }
        }
        HCCL_VM_DEBUG("-------------------------DieId[{}] XN Resouce Info End-------------------------", dieId);
        HCCL_VM_DEBUG(
            "==========================All CCU XN Resouce Info End: rank[{}], dieId[{}]===========================",
            rankId, dieId);
    }
}

void CcuResourceManager::DumpCcuGsaResouceInfo(int rankId) const
{
    for (uint32_t dieId = 0; dieId < HcclSim::DIE_NUM; dieId++) {
        HCCL_VM_DEBUG(
            "==========================All CCU GSA Resouce Info Start: rank[{}], dieId[{}]===========================",
            rankId, dieId);
        HCCL_VM_DEBUG("-------------------------DieId[{}] GSA Resouce Info Start-------------------------", dieId);
        for (uint32_t gsaId = 0; gsaId < SimCcuV1::CCU_RESOURCE_GSA_MAX; gsaId++) {
            auto gsaValue = GetGsaValue(rankId, dieId, gsaId);
            if (gsaValue != 0) {
                HCCL_VM_TRACE("[GSA: Id = {}, Value = {:x}]", gsaId, gsaValue);
            }
        }
        HCCL_VM_DEBUG("-------------------------DieId[{}] GSA Resouce Info End-------------------------", dieId);
        HCCL_VM_DEBUG(
            "==========================All CCU GSA Resouce Info End: rank[{}], dieId[{}]===========================",
            rankId, dieId);
    }
}

void CcuResourceManager::DumpCcuCkeResouceInfo(int rankId) const
{
    for (uint32_t dieId = 0; dieId < HcclSim::DIE_NUM; dieId++) {
        HCCL_VM_DEBUG(
            "==========================All CCU CKE Resouce Info Start: rank[{}], dieId[{}]===========================",
            rankId, dieId);
        HCCL_VM_DEBUG("-------------------------DieId[{}] CKE Resouce Info Start-------------------------", dieId);
        for (uint32_t ckeId = 0; ckeId < SimCcuV1::CCU_RESOURCE_MS_NUM; ckeId++) {
            auto ckeValue = GetCkeValue(rankId, dieId, ckeId);
            if (ckeValue != 0) {
                HCCL_VM_TRACE("[CKE: Id = {}, Value = {:x}]", ckeId, ckeValue);
            }
        }
        HCCL_VM_DEBUG("-------------------------DieId[{}] CKE Resouce Info End-------------------------", dieId);
        HCCL_VM_DEBUG(
            "==========================All CCU CKE Resouce Info End: rank[{}], dieId[{}]===========================",
            rankId, dieId);
    }
}

void CcuResourceManager::DumpCcuChannelResouceInfo(int rankId) const
{
    for (int dieId = 0; dieId < HcclSim::DIE_NUM; dieId++) {
        HCCL_VM_DEBUG(
            "==========================All CCU CHANNEL Resouce Info Start: rank[{}], "
            "dieId[{}]===========================",
            rankId, dieId);
        HCCL_VM_DEBUG("-------------------------DieId[{}] CHANNEL Resouce Info Start-------------------------", dieId);
        for (uint32_t chId = 0; chId < SimCcuV1::MAX_CCU_CHANNEL_NUM; chId++) {
            auto rmtCcu = GetRmtCcu(rankId, dieId, chId);
            if (rmtCcu.first != S32_INVALID && rmtCcu.second != S32_INVALID) {
                HCCL_VM_TRACE(
                    "[CHANNEL: Id = {}, Local[{}:{}] -> Remote[{}:{}]]", chId, rankId, dieId, rmtCcu.first,
                    rmtCcu.second);
            }
        }
        HCCL_VM_DEBUG("-------------------------DieId[{}] CHANNEL Resouce Info End-------------------------", dieId);
        HCCL_VM_DEBUG(
            "==========================All CCU CHANNEL Resouce Info End: rank[{}], "
            "dieId[{}]===========================",
            rankId, dieId);
    }
}

void CcuResourceManager::DumpChannelId2RmtRank(int rankId, int dieId) const
{
    if (!enableDump_) {
        return;
    }

    auto version = ccuResData_.version;
    for (uint32_t i = 0; i < SimCcuV1::MAX_CCU_CHANNEL_NUM; i++) {
        auto rmtCcu = GetRmtCcu(rankId, dieId, i);
        if (rmtCcu.first == INT32_MAX || rmtCcu.second == INT32_MAX) {
            continue;
        }
        HCCL_VM_DEBUG("channelId[{}], ccu[{}:{} --> {}:{}]", i, rankId, dieId, rmtCcu.first, rmtCcu.second);
    }
}

void CcuResourceManager::DumpCcuAllResouceInfo(int rankId) const
{
    DumpCcuXnResouceInfo(rankId);
    DumpCcuGsaResouceInfo(rankId);
    DumpCcuCkeResouceInfo(rankId);
    DumpCcuChannelResouceInfo(rankId);
}
