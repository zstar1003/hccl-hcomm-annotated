/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_DEVICE_MANAGER_H
#define CCU_DEVICE_MANAGER_H

#include <array>
#include "hccl/hccl_types.h"

#include "ip_address.h"
#include "ccu_dev_mgr.h"
#include "orion_adapter_hccp.h"
#include "local_ub_rma_buffer.h"

#include "../../../../../base_comm/resources/ccu/pub_inc/ccu_res_repo.h"

namespace Hccl {

using CcuResHandle = void*;

MAKE_ENUM(CcuVersion, CCU_V1, CCU_V2, CCU_INVALID);

MAKE_ENUM(ResType, LOOP, MS, CKE, XN, COUNT_XN, GSA, INS, MISSION, CASC_CNT);

// 使用对外头文件的数据结构
using MissionReqType = hcomm::MissionReqType;
using ResInfo = hcomm::ResInfo;
using MissionResInfo = hcomm::MissionResInfo;
using CcuResRepository = hcomm::CcuResRepository;
using MissionReq = hcomm::MissionReq;
using CcuResReq = hcomm::CcuResReq;

struct ChannelPara {
    uint32_t feId;
    uint32_t jettyNum;
    uint32_t sqSize;
};

using JettyInfo = CcuJettyInfo;
using ChannelInfo = CcuChannelInfo;

struct JettyCfg {
    uint16_t jettyCtxId;
    uint64_t dbVa;
    uint32_t dbTokenId;
    uint32_t dbTokenValue;
};

struct ChannelCfg {
    uint32_t channelId{0};
    // remote channel info
    Eid remoteEid{};
    uint32_t tpn{0};

    uint64_t remoteCcuVa{0};
    uint32_t memTokenId{0};
    uint32_t memTokenValue{0};
    std::vector<JettyCfg> jettyCfgs;
};

/* opcode definition */
enum class CcuOpcodeType {
    CCU_U_OP_GET_VERSION = 0, /* 获取CCU版本号 */

    CCU_U_OP_K_MIN = 10, /* 定义需要向内核发送请求的操作最小值 */

    CCU_U_OP_GET_BASIC_INFO = 11,  /* 获取基础信息 */
    CCU_U_OP_GET_DIE_WORKING = 15, /* 获取该dieId是否工作 */

    CCU_U_OP_SET_MSID_TOKEN = 53,       /* 设置连续MSID的配置值和Token相关值 */
    CCU_U_OP_SET_TASKKILL = 54,         /* 启动taskkill任务 */
    CCU_U_OP_CLEAN_TASKKILL_STATE = 55, /* 清除taskkill任务 */
    CCU_U_OP_CLEAN_TIF_TABLE = 56,      /* 清除TIF表项 */

    CCU_U_OP_K_MAX = 100,              /* 定义需要向内核发送请求的造作最大值 */
    CCU_U_OP_SET_TIF_SPLIT_SIZE = 127, /* 配置0.5RTT特性中的count计数单位 */
    CCU_U_OP_SET_XN_TOTAL_CNT = 128,   /* 配置比较寄存器 */

    /* 以下为操作CCU映射到用户态资源空间的操作码 */
    CCU_U_OP_IN_RS_MIN = 200,       /* 定义一个在RS空间操作的最小值 */
    CCU_U_OP_GET_INSTRUCTION = 201, /* 设置INS指令 */
    CCU_U_OP_GET_GSA = 202,         /* 获取GSA数据 */
    CCU_U_OP_GET_XN = 203,          /* 获取XN数据 */
    CCU_U_OP_GET_CKE = 204,         /* 获取CKE数据 */
    CCU_U_OP_GET_PFE = 205,         /* 获取PFE数据 */
    CCU_U_OP_GET_CHANNEL = 206,     /* 获取Channel数据 */
    CCU_U_OP_GET_JETTY_CTX = 207,   /* 获取Jetty_ctx数据 */
    CCU_U_OP_GET_MISSION_CTX = 208, /* 获取Mission_ctx数据 */
    CCU_U_OP_GET_LOOP_CTX = 209,    /* 获取Loop_ctx数据 */

    CCU_U_OP_SET_INSTRUCTION = 251, /* 设置INS指令 */
    CCU_U_OP_SET_GSA = 252,         /* 设置GSA数据 */
    CCU_U_OP_SET_XN = 253,          /* 设置XN数据 */
    CCU_U_OP_SET_CKE = 254,         /* 设置CKE数据 */
    CCU_U_OP_SET_PFE = 255,         /* 设置PFE数据 */
    CCU_U_OP_SET_CHANNEL = 256,     /* 设置Channel数据 */
    CCU_U_OP_SET_JETTY_CTX = 257,   /* 设置Jetty_ctx数据 */
    CCU_U_OP_SET_MISSION_CTX = 258, /* 设置Mission_ctx数据 */
    CCU_U_OP_SET_LOOP_CTX = 259,    /* 设置Loop_ctx数据 */

    CCU_U_OP_IN_RS_MAX = 300, /* 定义一个在RS空间操作的最大值 */
};

struct CcuDataByte8 {
    char raw[8];
};

struct CcuDataByte32 {
    char raw[32];
};

struct CcuDataByte64 {
    char raw[64];
};

struct CcuInstrInfo {
    uint64_t resourceAddr;
};

constexpr uint32_t CCU_ENABLE_FLAG = 1;

struct CcuDieInfo {
    uint32_t enableFlag;
};

enum class CcuVersionEnum {
    CCU_V1 = 0,
    CCU_V2 = 1,
};

struct CcuTifSplitSize {
    uint32_t splitPktUnit;
    uint32_t tpSplitSize;
    uint32_t ctpSplitSize;
};

struct CcuTotalCntXn {
    uint32_t cntIndex;
    uint32_t totalValue;
    uint32_t totalAddr;
    uint32_t flagFromAddr;
    uint32_t flagToAddr;
};

struct CcuDataCaps {
    uint32_t cap0;
    uint32_t cap1;
    uint32_t cap2;
    uint32_t cap3;
    uint32_t cap4;
};

struct CcuBaseInfoData {
    uint32_t msId;

    uint32_t tokenId;
    uint32_t tokenValue;
    uint32_t tokenValid;

    uint32_t missionKey;
    uint64_t resourceAddr;
    struct CcuDataCaps caps;
};

union CcuDataTypeUnion {
    struct CcuDataByte8 byte8;
    struct CcuDataByte32 byte32;
    struct CcuDataByte64 byte64;
    struct CcuBaseInfoData baseinfo;
    struct CcuInstrInfo insinfo;
    struct CcuDieInfo dieinfo;
    CcuVersionEnum ccuVersion;
    struct CcuTifSplitSize tifSplitSize;
    struct CcuTotalCntXn xnTotalCnt;
    // struct CcuHighPerfXn   highPerfXn;
};

struct CcuData {
    uint32_t udieIdx;
    uint32_t dataLen;                    /* 数据的总长度（sizeof(dataArray[xxx]) *  dataArraySize的值 */
    uint32_t dataArraySize;              /* dataArray数组的个数 */
    union CcuDataTypeUnion dataArray[8]; /* 不同类型的数据，通过联合体来存储 */
};

union CcuDataUnion {
    char raw[2048]; /* 对外呈现是一个字符数组，内部转换成对应类型CcuData */
    struct CcuData dataInfo;
};

struct CustomChannelInfoIn {
    CcuDataUnion data; /* 对外呈现是一个字符数组，内部转换成对应类型ccu_data */
    uint32_t offsetStartIdx; /* 对应需要操作的元素的idx位置，位置用正整数代替，使用者不需要关心元素的实际大小 */
    CcuOpcodeType op;

    CustomChannelInfoIn() : offsetStartIdx(0), op(CcuOpcodeType::CCU_U_OP_GET_VERSION)
    {
        (void)memset_s(&data, sizeof(data), 0, sizeof(data));
    }
};

struct CustomChannelInfoOut {
    CcuDataUnion data;      /* 对外呈现是一个字符数组，内部转换成对应类型CcuData */
    uint32_t offsetNextIdx; /* 操作后返回下一个元素的idx位置，位置用正整数代替，使用者不需要关心元素的实际大小 */
    int opRet;

    CustomChannelInfoOut() : offsetNextIdx(0), opRet(0) { (void)memset_s(&data, sizeof(data), 0, sizeof(data)); }
};

constexpr uint32_t SHIFT_2BITS = 2;
constexpr uint32_t SHIFT_4BITS = 4;
constexpr uint32_t SHIFT_8BITS = 8;
constexpr uint32_t SHIFT_12BITS = 12;
constexpr uint32_t SHIFT_16BITS = 16;
constexpr uint32_t SHIFT_20BITS = 20;
constexpr uint32_t SHIFT_24BITS = 24;
constexpr uint32_t SHIFT_40BITS = 40;

class CcuDeviceManager {
public:
    CcuDeviceManager() = delete;
    ~CcuDeviceManager() = delete;

    static HcclResult GetCcuVersion(const int32_t deviceLogicId, CcuVersion& ccuVersion);

    static HcclResult
    GetCcuResourceSpaceBufInfo(const int32_t deviceLogicId, const uint8_t dieId, uint64_t& addr, uint64_t& size);
    static HcclResult GetCcuResourceSpaceTokenInfo(
        const int32_t deviceLogicId, const uint8_t dieId, uint64_t& tokenId, uint64_t& tokenValue);
    static HcclResult GetCcuResourceSpaceTokenInfoForLocal(
        const int32_t deviceLogicId, const uint8_t dieId, uint64_t& tokenId, uint64_t& tokenValue);

    static HcclResult ConfigChannel(const int32_t deviceLogicId, const uint8_t dieId, ChannelCfg& cfg);
    static HcclResult
    GetLoopChannelId(const int32_t deviceLogicId, const uint8_t srcDieId, const uint8_t dstDieId, uint32_t& channIdx);

    static HcclResult GetResource(const int32_t deviceLogicId, const CcuResHandle handle, CcuResRepository& ccuResRepo);
    static HcclResult AllocResHandle(const int32_t deviceLogicId, const CcuResReq resReq, CcuResHandle& handle);
    static HcclResult ReleaseResHandle(const int32_t deviceLogicId, const CcuResHandle handle);

    static HcclResult AllocIns(const int32_t deviceLogicId, const uint8_t dieId, const uint32_t num, ResInfo& insInfo);
    static HcclResult ReleaseIns(const int32_t deviceLogicId, const uint8_t dieId, ResInfo& insInfo);
    static HcclResult
    AllocCke(const int32_t deviceLogicId, const uint8_t dieId, const uint32_t num, std::vector<ResInfo>& ckeInfos);
    static HcclResult ReleaseCke(const int32_t deviceLogicId, const uint8_t dieId, std::vector<ResInfo>& ckeInfos);
    static HcclResult
    AllocXn(const int32_t deviceLogicId, const uint8_t dieId, const uint32_t num, std::vector<ResInfo>& xnInfos);
    static HcclResult ReleaseXn(const int32_t deviceLogicId, const uint8_t dieId, std::vector<ResInfo>& xnInfos);

    static HcclResult GetMissionKey(const int32_t deviceLogicId, const uint8_t dieId, uint32_t& missionKey);
    static HcclResult GetInstructionNum(const int32_t deviceLogicId, const uint8_t dieId, uint32_t& instrNum);
    static HcclResult GetXnBaseAddr(const uint32_t devLogicId, const uint8_t dieId, uint64_t& xnBaseAddr);
};

HcclResult CheckDieValid(
    const char* funcName, const int32_t devLogicId, const uint8_t dieId,
    const std::array<bool, MAX_CCU_IODIE_NUM>& dieEnableFlags);

}; // namespace Hccl

#endif // CCU_DEVICE_MANAGER_H
