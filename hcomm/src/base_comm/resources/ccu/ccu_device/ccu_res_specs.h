/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_RES_SPECS_H
#define CCU_RES_SPECS_H

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include "ccu_dev_mgr_imp.h"
#include "dev_type.h"

namespace hcomm {

constexpr uint32_t CCU_RESOURCE_SIZE = 72 * 1024 * 1024; // CCU资源空间大小
constexpr uint64_t CCU_V1_CCUM_OFFSET = 0x800000;        // V1 CCUM 偏移，位于CCUA之后
constexpr uint64_t CCU_V2_CCUM_OFFSET = 0x0;             // V2 CCUM 偏移，位于CCUA之前

constexpr uint64_t CCU_V1_WQE_BASIC_BLOCK_OFFSET = (CCU_V1_CCUM_OFFSET + 0x800000);

constexpr uint32_t CCU_ONE_WQE_SIZE = 64;   // Bytes
constexpr uint32_t CCU_WQE_NUM_PER_SQE = 4; // URMA 约束每个SQE包含4个WQEBB
constexpr uint32_t CCU_MIN_SQ_DEPTH = 16;
constexpr uint32_t CCU_MAX_SQ_DEPTH = 256;
constexpr uint16_t CCU_START_TA_JETTY_ID = 1024;    // IMP给系统预留给CCU的jetty Id起始编号
constexpr uint32_t CCU_SQ_BUFFER_SIZE = 256 * 1024; // ccu 每个jetty sq buffer size 固定为256k
constexpr uint32_t CCU_WQEBB_RESOURCE_NUM = 4096;
constexpr uint32_t CCU_V1_PER_DIE_PFE_RESERVED_NUM = 16; // ccu 每个IO die预留16个PFE表
constexpr uint32_t CCU_V2_PER_DIE_PFE_RESERVED_NUM = 20; // ccu v2每个IO die预留20个PFE表
constexpr uint8_t CCU_PER_DIE_JETTY_RESERVED_NUM = 128;  // ccu 每个IO die默认jetty数量

constexpr uint32_t CCU_V2_COUNT_XN_NUM = 4096; // 暂时采用宏定义

constexpr uint64_t CCU_RESOURCE_INS_RESERVE_SIZE = 0x100000;  // INS预留空间1M
constexpr uint64_t CCU_V1_RESOURCE_GSA_RESERVE_SIZE = 0x8000; // v1 GSA预留空间32K
constexpr uint64_t CCU_V2_RESOURCE_GSA_RESERVE_SIZE = 0x0;    // v2 GSA删除
constexpr uint32_t CCU_RESOURCE_XN_PER_SIZE = 8;
constexpr uint32_t CCU_RESOURCE_CKE_PER_SIZE = 8;
constexpr uint32_t CCU_RESOURCE_XN_V1_RESERVE_SIZE = 32 * 1024;  // xn预留空间32K
constexpr uint32_t CCU_RESOURCE_XN_V2_RESERVE_SIZE = 256 * 1024; // xn预留空间256K
constexpr uint16_t CCU_RESOURCE_INSTR_PER_SIZE = 32;
constexpr uint32_t CCU_V2_RESOURCE_TOTAL_CNT_XNS_NUM = 4; // 0.5RTT配置寄存器共4组

constexpr uint32_t MOVE_16_BITS = 16;
constexpr uint32_t MOVE_20_BITS = 20;
constexpr uint32_t MOVE_24_BITS = 24;

constexpr uint32_t INVALID_VALUE = 0;
constexpr uint64_t INVALID_ADDR = 0;

constexpr uint32_t CCU_V2_FIXED_SQ_SIZE = 32;

// CcuBlockResStrategy 定义了资源管理块资源类型的块大小
struct CcuBlockResStrategy {
    uint32_t loopNum{16};
    uint32_t ckeNum{16};
    uint32_t msNum{128};
    uint32_t xnNum{16};
    uint32_t gsaNum{16};
    uint32_t missionNum{2};
};

// ccu v2 专用，指示channel与jetty配比关系
struct CcuChannelJettyMap {
    uint32_t channelNum{1};
    uint32_t jettyNum{1};
    constexpr CcuChannelJettyMap(uint32_t channelNum, uint32_t jettyNum) : channelNum(channelNum), jettyNum(jettyNum) {}
};

// 预留后续用于模式选择
constexpr CcuChannelJettyMap CCU_CHANNEL_JETTY_MAP[]{
    {8, 1},
    {32, 4},
};

struct CcuResSpecInfo {
    // 基础信息
    uint32_t msId{0};
    uint32_t missionKey{0};
    uint64_t resourceAddr{0};
    // 通过能力寄存器获取
    uint32_t loopEngineNum{0};
    uint32_t missionNum{0};
    uint32_t instructionNum{0};
    uint32_t xnNum{0};
    uint32_t gsaNum{0};
    uint32_t msNum{0};
    uint32_t ckeNum{0};
    uint32_t jettyNum{0};
    uint32_t channelNum{0};
    uint32_t pfeNum{0};
    // 额外资源信息
    uint32_t countXnNum{0};
    uint32_t loopCkeNum{0};
    uint32_t wqeBBNum{CCU_WQEBB_RESOURCE_NUM};
    uint32_t dieNum{CCU_MAX_IODIE_NUM};
    CcuChannelJettyMap channelJettyMap{8, 1};
};

HcclResult CcuGetMainboardId(uint32_t userDevId, Hccl::HcclMainboardId& hcclMainboardId);

MAKE_ENUM(ServeMode, ARMX86, NORMAL);

class CcuResSpecifications {
public:
    static CcuResSpecifications& GetInstance(const int32_t userDevId);
    HcclResult Init();
    HcclResult Deinit();
    ServeMode GetServeMode() const;
    CcuVersion GetCcuVersion() const;
    HcclResult GetDieEnableFlag(const uint8_t dieId, bool& dieEnableFlag) const;

    HcclResult GetResourceAddr(const uint8_t dieId, uint64_t& resourceAddr) const;
    HcclResult GetXnBaseAddr(const uint8_t dieId, uint64_t& xnBaseAddr) const;
    HcclResult GetCkeBaseAddr(const uint8_t dieId, uint64_t& ckeBaseAddr) const;
    uint64_t GetXnOffsetCcumBaseAddr(const uint8_t dieId) const;
    uint64_t GetCkeOffsetCcumBaseAddr(const uint8_t dieId) const;
    HcclResult GetXnOffsetCcumAddrById(const uint8_t dieId, uint32_t id, uint64_t& xnAddr) const;
    HcclResult GetCkeOffsetCcumAddrById(const uint8_t dieId, uint32_t id, uint64_t& ckeAddr) const;

    HcclResult GetMsId(const uint8_t dieId, uint32_t& msId) const;
    HcclResult GetMissionKey(const uint8_t dieId, uint32_t& missionKey) const;

    // 寄存器资源
    HcclResult GetMissionNum(const uint8_t dieId, uint32_t& missionNum) const;

    HcclResult GetMsNum(const uint8_t dieId, uint32_t& msNum) const;
    HcclResult GetLoopEngineNum(const uint8_t dieId, uint32_t& loopNum) const;
    HcclResult GetCkeNum(const uint8_t dieId, uint32_t& ckeNum) const;
    HcclResult GetXnNum(const uint8_t dieId, uint32_t& xnNum) const;
    HcclResult GetInstructionNum(const uint8_t dieId, uint32_t& instrNum) const;
    // v1 独有资源
    HcclResult GetGsaNum(const uint8_t dieId, uint32_t& gsaNum) const;
    // v2 新增资源
    HcclResult GetLoopCkeNum(const uint8_t dieId, uint32_t& loopCkeNum) const;
    HcclResult GetCountXnNum(const uint8_t dieId, uint32_t& countXnNum) const;
    HcclResult GetCascCntNum(const uint8_t dieId, uint32_t& cascCntNum) const;

    // channel资源
    HcclResult GetChannelNum(const uint8_t dieId, uint32_t& channelNum) const;
    HcclResult GetJettyNum(const uint8_t dieId, uint32_t& jettyNum) const;
    HcclResult GetPfeReservedNum(const uint8_t dieId, uint32_t& pfeNum) const;
    HcclResult GetPfeNum(const uint8_t dieId, uint32_t& pfeNum) const;
    HcclResult GetWqeBBNum(const uint8_t dieId, uint32_t& wqeBBNum) const;
    // v2新增channel信息
    HcclResult GetChannelJettyMap(const uint8_t dieId, CcuChannelJettyMap& channelJettyMap) const;

    using GetResSpecFunc = HcclResult (CcuResSpecifications::*)(const uint8_t, uint32_t&) const;
    using ResSpecFuncPair = std::pair<ResType, GetResSpecFunc>;
    static constexpr std::array<ResSpecFuncPair, 8> GET_RES_SPEC_FUNC_ARRAY
        = {{{ResType::LOOP, &CcuResSpecifications::GetLoopEngineNum},
            {ResType::MS, &CcuResSpecifications::GetMsNum},
            {ResType::CKE, &CcuResSpecifications::GetCkeNum},
            {ResType::XN, &CcuResSpecifications::GetXnNum},
            {ResType::COUNT_XN, &CcuResSpecifications::GetCountXnNum},
            {ResType::GSA, &CcuResSpecifications::GetGsaNum},
            {ResType::INS, &CcuResSpecifications::GetInstructionNum},
            {ResType::MISSION, &CcuResSpecifications::GetMissionNum}}};

private:
    explicit CcuResSpecifications() = default;
    ~CcuResSpecifications() = default;
    CcuResSpecifications(const CcuResSpecifications& that) = delete;
    CcuResSpecifications& operator=(const CcuResSpecifications& that) = delete;

    HcclResult CheckDieValid(
        const std::string& funcName, const int32_t userDevId, const uint8_t dieId,
        const std::array<bool, CCU_MAX_IODIE_NUM>& dieEnableFlags) const;

private:
    bool initFlag_{false};
    int32_t userDevId_{0};
    uint32_t devPhyId_{0};
    ServeMode serveMode_{ServeMode::ARMX86}; // 默认按a+x模式，规避使用部分ccua
    CcuVersion ccuVersion_{CcuVersion::CCU_INVALID};
    std::array<bool, CCU_MAX_IODIE_NUM> dieEnableFlags_{};
    std::array<CcuResSpecInfo, CCU_MAX_IODIE_NUM> resSpecs_{};
}; // class CcuResSpecifications
} // namespace hcomm
#endif // CCU_RES_SPECS_H
