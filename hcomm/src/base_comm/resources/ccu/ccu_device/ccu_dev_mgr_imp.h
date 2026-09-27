/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_DEV_MGR_IMP_H
#define CCU_DEV_MGR_IMP_H

#include "ccu_device_pub.h"

#include <string>

#include "hcomm_adapter_hccp.h" // 暂时引入Eid

#include "unified_platform/pub_inc/ccu/ccu_dev_mgr.h"
#include "unified_platform/ccu/ccu_device/ccu_device_manager.h"

namespace hcomm {

using CcuVersion = Hccl::CcuVersion;
using ResType = Hccl::ResType;
using ChannelPara = Hccl::ChannelPara;

/* 开源自定义算子CCU设备管理实现，当前支持新老通信域混跑，
 * 暂时改用legacy数据结构，避免反向依赖
 * MAKE_ENUM(CcuVersion, CCU_V1, CCU_V2, CCU_INVALID);
 * MAKE_ENUM(ResType, LOOP, MS, CKE, XN, COUNT_XN, GSA, INS, MISSION, CASC_CNT);
 * struct ChannelPara {
 *     uint32_t feId;
 *     uint32_t jettyNum;
 *     uint32_t sqSize;
 * };
 */

using JettyInfo = CcuJettyInfo;
using ChannelInfo = CcuChannelInfo;

using JettyCfg = Hccl::JettyCfg;
/* 开源自定义算子CCU设备管理实现，当前支持新老通信域混跑，
 * 暂时改用legacy数据结构，避免反向依赖
 * struct JettyCfg {
 *     uint16_t jettyCtxId;
 *     uint64_t dbVa;
 *     uint32_t dbTokenId;
 *     uint32_t dbTokenValue;
 * };
 */

using ChannelCfg = Hccl::ChannelCfg;
/* 开源自定义算子CCU设备管理实现，当前支持新老通信域混跑，
 * 暂时改用legacy数据结构，避免反向依赖
 * struct ChannelCfg {
 *     uint32_t channelId{0};
 *     // remote channel info
 *     Eid remoteEid{};
 *     uint32_t tpn{0};

 *     uint64_t remoteCcuVa{0};
 *     uint32_t memTokenId{0};
 *     uint32_t memTokenValue{0};
 *     std::vector<JettyCfg> jettyCfgs;
 * };
 */

using CcuOpcodeType = Hccl::CcuOpcodeType;
/* 开源自定义算子CCU设备管理实现，当前支持新老通信域混跑，
 * 暂时改用legacy数据结构，避免反向依赖
 * // opcode definition
 * enum class CcuOpcodeType {
 *     CCU_U_OP_GET_VERSION =  0, // 获取CCU版本号

 *     CCU_U_OP_K_MIN             = 10, // 定义需要向内核发送请求的操作最小值
 *     CCU_U_OP_GET_BASIC_INFO    = 11, // 获取基础信息
 *     CCU_U_OP_GET_DIE_WORKING   = 15, // 获取该dieId是否工作
 *     CCU_U_OP_SET_MSID_TOKEN       = 53, // 设置连续MSID的配置值和Token相关值
 *     CCU_U_OP_SET_TASKKILL         = 54, // 启动taskkill任务
 *     CCU_U_OP_CLEAN_TASKKILL_STATE = 55, // 清除taskkill任务
 *     CCU_U_OP_CLEAN_TIF_TABLE      = 56, // 清除TIF表项
 *     CCU_U_OP_K_MAX = 100, // 定义需要向内核发送请求的造作最大值
 *
 *     CCU_U_OP_SET_TIF_SPLIT_SIZE = 127, // 配置0.5RTT特性中的count计数单位
 *     CCU_U_OP_SET_XN_TOTAL_CNT   = 128, // 配置比较寄存器
 *
 *     // 以下为操作CCU映射到用户态资源空间的操作码
 *     CCU_U_OP_IN_RS_MIN       = 200, // 定义一个在RS空间操作的最小值
 *     CCU_U_OP_GET_INSTRUCTION = 201, // 设置INS指令
 *     CCU_U_OP_GET_GSA         = 202, // 获取GSA数据
 *     CCU_U_OP_GET_XN          = 203, // 获取XN数据
 *     CCU_U_OP_GET_CKE         = 204, // 获取CKE数据
 *     CCU_U_OP_GET_PFE         = 205, // 获取PFE数据
 *     CCU_U_OP_GET_CHANNEL     = 206, // 获取Channel数据
 *     CCU_U_OP_GET_JETTY_CTX   = 207, // 获取Jetty_ctx数据
 *     CCU_U_OP_GET_MISSION_CTX = 208, // 获取Mission_ctx数据
 *     CCU_U_OP_GET_LOOP_CTX    = 209, // 获取Loop_ctx数据

 *     CCU_U_OP_SET_INSTRUCTION = 251, // 设置INS指令
 *     CCU_U_OP_SET_GSA         = 252, // 设置GSA数据
 *     CCU_U_OP_SET_XN          = 253, // 设置XN数据
 *     CCU_U_OP_SET_CKE         = 254, // 设置CKE数据
 *     CCU_U_OP_SET_PFE         = 255, // 设置PFE数据
 *     CCU_U_OP_SET_CHANNEL     = 256, // 设置Channel数据
 *     CCU_U_OP_SET_JETTY_CTX   = 257, // 设置Jetty_ctx数据
 *     CCU_U_OP_SET_MISSION_CTX = 258, // 设置Mission_ctx数据
 *     CCU_U_OP_SET_LOOP_CTX    = 259, // 设置Loop_ctx数据
 *     CCU_U_OP_IN_RS_MAX = 300, 定义一个在RS空间操作的最大值
 * };
 */

using CcuDataByte8 = Hccl::CcuDataByte8;
/* 开源自定义算子CCU设备管理实现，当前支持新老通信域混跑，
 * 暂时改用legacy数据结构，避免反向依赖
 * struct CcuDataByte8 {
 *     char raw[8];
 * };
 */

using CcuDataByte32 = Hccl::CcuDataByte32;
/* 开源自定义算子CCU设备管理实现，当前支持新老通信域混跑，
 * 暂时改用legacy数据结构，避免反向依赖
 * struct CcuDataByte32 {
 *     char raw[32];
 * };
 */

using CcuDataByte64 = Hccl::CcuDataByte64;
/* 开源自定义算子CCU设备管理实现，当前支持新老通信域混跑，
 * 暂时改用legacy数据结构，避免反向依赖
 * struct CcuDataByte64 {
 *      char raw[64];
 * };
 */

using CcuInstrInfo = Hccl::CcuInstrInfo;
/* 开源自定义算子CCU设备管理实现，当前支持新老通信域混跑，
 * 暂时改用legacy数据结构，避免反向依赖
 * struct CcuInstrInfo {
 *     uint64_t resourceAddr;
 * };
 */

/* 开源自定义算子CCU设备管理实现，当前支持新老通信域混跑，
 * 暂时改用legacy数据结构，避免反向依赖
 * constexpr uint32_t CCU_ENABLE_FLAG = 1;
 */

using CcuDieInfo = Hccl::CcuDieInfo;
/* 开源自定义算子CCU设备管理实现，当前支持新老通信域混跑，
 * 暂时改用legacy数据结构，避免反向依赖
 * struct CcuDieInfo {
 *     uint32_t enableFlag;
 * };
 */

using CcuVersionEnum = Hccl::CcuVersionEnum;
/* 开源自定义算子CCU设备管理实现，当前支持新老通信域混跑，
 * 暂时改用legacy数据结构，避免反向依赖
 * enum class CcuVersionEnum {
 *     CCU_V1 = 0,
 *     CCU_V2 = 1,
 * };
 */

using CcuDataCaps = Hccl::CcuDataCaps;
/* 开源自定义算子CCU设备管理实现，当前支持新老通信域混跑，
 * 暂时改用legacy数据结构，避免反向依赖
 * struct CcuDataCaps {
 *     uint32_t cap0;
 *     uint32_t cap1;
 *     uint32_t cap2;
 *     uint32_t cap3;
 *     uint32_t cap4;
 * };
 */

using CcuBaseInfoData = Hccl::CcuBaseInfoData;
/* 开源自定义算子CCU设备管理实现，当前支持新老通信域混跑，
 * 暂时改用legacy数据结构，避免反向依赖
 * struct CcuBaseInfoData {
 *     uint32_t           msId;

 *     uint32_t           tokenId;
 *     uint32_t           tokenValue;
 *     uint32_t           tokenValid;

 *     uint32_t           missionKey;
 *     uint64_t           resourceAddr;
 *     struct CcuDataCaps caps;
 * };
 */

using CcuDataTypeUnion = Hccl::CcuDataTypeUnion;
/* 开源自定义算子CCU设备管理实现，当前支持新老通信域混跑，
 * 暂时改用legacy数据结构，避免反向依赖
 * union CcuDataTypeUnion {
 *     struct CcuDataByte8    byte8;
 *     struct CcuDataByte32   byte32;
 *     struct CcuDataByte64   byte64;
 *     struct CcuBaseInfoData baseinfo;
 *     struct CcuInstrInfo    insinfo;
 *     struct CcuDieInfo      dieinfo;
 *     CcuVersionEnum         ccuVersion;
 *     struct CcuTifSplitSize tifSplitSize;
 *     struct CcuTotalCntXn   xnTotalCnt;
 *     // struct CcuHighPerfXn   highPerfXn;
 * };
 */

using CcuData = Hccl::CcuData;
/* 开源自定义算子CCU设备管理实现，当前支持新老通信域混跑，
 * 暂时改用legacy数据结构，避免反向依赖
 * struct CcuData {
 *     uint32_t               udieIdx;
 *     uint32_t               dataLen;       // 数据的总长度（sizeof(dataArray[xxx]) *  dataArraySize的值
 *     uint32_t               dataArraySize; // dataArray数组的个数
 *     union CcuDataTypeUnion dataArray[8];  // 不同类型的数据，通过联合体来存储
 * };
 */

using CcuDataUnion = Hccl::CcuDataUnion;
/* 开源自定义算子CCU设备管理实现，当前支持新老通信域混跑，
 * 暂时改用legacy数据结构，避免反向依赖
 * union CcuDataUnion {
 *     char           raw[2048]; // 对外呈现是一个字符数组，内部转换成对应类型CcuData
 *     struct CcuData dataInfo;
 * };
 */

using CustomChannelInfoIn = Hccl::CustomChannelInfoIn;
/* 开源自定义算子CCU设备管理实现，当前支持新老通信域混跑，
 * 暂时改用legacy数据结构，避免反向依赖
 * struct CustomChannelInfoIn {
 *     CcuDataUnion data; // 对外呈现是一个字符数组，内部转换成对应类型ccu_data
 *     uint32_t offsetStartIdx; // 对应需要操作的元素的idx位置，位置用正整数代替，使用者不需要关心元素的实际大小
 *     CcuOpcodeType op;

 *     CustomChannelInfoIn() : offsetStartIdx(0), op(CcuOpcodeType::CCU_U_OP_GET_VERSION) {
 *         (void)memset_s(&data, sizeof(data), 0, sizeof(data));
 *     }
 * };
 */

using CustomChannelInfoOut = Hccl::CustomChannelInfoOut;
/* 开源自定义算子CCU设备管理实现，当前支持新老通信域混跑，
 * 暂时改用legacy数据结构，避免反向依赖
 * struct CustomChannelInfoOut {
 *     CcuDataUnion data;      // 对外呈现是一个字符数组，内部转换成对应类型CcuData
 *     uint32_t offsetNextIdx; // 操作后返回下一个元素的idx位置，位置用正整数代替，使用者不需要关心元素的实际大小
 *     int opRet;

 *     CustomChannelInfoOut() : offsetNextIdx(0), opRet(0) {
 *         (void)memset_s(&data, sizeof(data), 0, sizeof(data));
 *     }
 * };
 */

/* 开源自定义算子CCU设备管理实现，当前支持新老通信域混跑，
 * 暂时改用legacy数据结构，避免反向依赖
 * constexpr uint32_t SHIFT_2BITS  = 2;
 * constexpr uint32_t SHIFT_4BITS  = 4;
 * constexpr uint32_t SHIFT_8BITS  = 8;
 * constexpr uint32_t SHIFT_12BITS = 12;
 * constexpr uint32_t SHIFT_16BITS = 16;
 * constexpr uint32_t SHIFT_20BITS = 20;
 * constexpr uint32_t SHIFT_24BITS = 24;
 * constexpr uint32_t SHIFT_40BITS = 40;
 */

class CcuDevMgrImp {
public:
    CcuDevMgrImp() = delete;
    ~CcuDevMgrImp() = delete;

    static HcclResult GetCcuVersion(const int32_t userDevId, CcuVersion& ccuVersion);

    static HcclResult
    GetCcuResourceSpaceBufInfo(const int32_t userDevId, const uint8_t dieId, uint64_t& addr, uint64_t& size);
    static HcclResult
    GetCcuResourceSpaceTokenInfo(const int32_t userDevId, const uint8_t dieId, uint64_t& tokenId, uint64_t& tokenValue);

    static HcclResult ConfigChannel(const int32_t userDevId, const uint8_t dieId, ChannelCfg& cfg);
    static HcclResult
    GetLoopChannelId(const int32_t userDevId, const uint8_t srcDieId, const uint8_t dstDieId, uint32_t& channIdx);

    static HcclResult GetResource(const int32_t userDevId, const CcuResHandle resHandle, CcuResRepository& ccuResRepo);
    static HcclResult AllocResHandle(const int32_t userDevId, const CcuResReq resReq, CcuResHandle& resHandle);
    static HcclResult ReleaseResHandle(const int32_t userDevId, const CcuResHandle resHandle);
    static HcclResult
    QueryRemainRes(const int32_t userDevId, const uint8_t dieId, const ResType& internalType, uint32_t& remainNum);

    static HcclResult AllocIns(const int32_t userDevId, const uint8_t dieId, const uint32_t num, ResInfo& insInfo);
    static HcclResult ReleaseIns(const int32_t userDevId, const uint8_t dieId, const ResInfo& insInfo);
    static uint32_t GetInsConsecutiveRemainSize(const int32_t userDevId, const uint8_t dieId);
    static uint32_t GetCascCntBlockRemainSize(const int32_t deviceuserDevIdLogicId, const uint8_t dieId);
    static HcclResult
    AllocCke(const int32_t userDevId, const uint8_t dieId, const uint32_t num, std::vector<ResInfo>& ckeInfos);
    static HcclResult ReleaseCke(const int32_t userDevId, const uint8_t dieId, const std::vector<ResInfo>& ckeInfos);
    static HcclResult
    AllocXn(const int32_t userDevId, const uint8_t dieId, const uint32_t num, std::vector<ResInfo>& xnInfos);
    static HcclResult ReleaseXn(const int32_t userDevId, const uint8_t dieId, const std::vector<ResInfo>& xnInfos);

    static HcclResult GetMissionKey(const int32_t userDevId, const uint8_t dieId, uint32_t& missionKey);
    static HcclResult GetResSpecsInstructionNum(const int32_t userDevId, const uint8_t dieId, uint32_t& instrNum);
    static HcclResult GetAllocatableMaxLoopEngineNum(const int32_t userDevId, const uint8_t dieId, uint32_t& loopNum);
    static HcclResult GetAllocatableMaxMsNum(const int32_t userDevId, const uint8_t dieId, uint32_t& msNum);
    static HcclResult GetAllocatableMaxCkeNum(const int32_t userDevId, const uint8_t dieId, uint32_t& ckeNum);
    static HcclResult GetAllocatableMaxXnNum(const int32_t userDevId, const uint8_t dieId, uint32_t& xnNum);
    static HcclResult GetAllocatableMaxGsaNum(const int32_t userDevId, const uint8_t dieId, uint32_t& gsaNum);
    static HcclResult GetResSpecsMissionNum(const int32_t userDevId, const uint8_t dieId, uint32_t& missionNum);
    static HcclResult GetResSpecsCascCntNum(const int32_t deviceLogicId, const uint8_t dieId, uint32_t& cascCntNum);
    static HcclResult GetXnBaseAddr(const int32_t userDevId, const uint8_t dieId, uint64_t& xnBaseAddr);
    static HcclResult GetCkeBaseAddr(const int32_t userDevId, const uint8_t dieId, uint64_t& ckeBaseAddr);
    static HcclResult
    GetXnOffsetCcumAddrById(const int32_t userDevId, const uint8_t dieId, uint16_t id, uint64_t& xnAddr);
    static HcclResult
    GetCkeOffsetCcumAddrById(const int32_t userDevId, const uint8_t dieId, uint16_t id, uint64_t& ckeAddr);
};

HcclResult CheckDieValid(
    const char* funcName, const int32_t userDevId, const uint8_t dieId,
    const std::array<bool, CCU_MAX_IODIE_NUM>& dieEnableFlags);

} // namespace hcomm

#endif // CCU_DEV_MGR_IMP_H
