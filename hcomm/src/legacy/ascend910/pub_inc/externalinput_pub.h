/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXTERNALINPUT_PUB_H
#define EXTERNALINPUT_PUB_H

#include <string>
#include <hccl/hccl_types.h>
#include "hccl/base.h"
#include "hccl_common.h"

constexpr u32 HCCL_WHITELIST_ON = 1;
constexpr s32 HCCL_LINK_TIME_OUT_S = 120;      // HCCL 默认的建链超时时间设置为120s
constexpr u32 HCCL_ALGO_LEVEL_0 = 0;           // HCCL 算法层级0
constexpr u32 HCCL_ALGO_LEVEL_1 = 1;           // HCCL 算法层级1
constexpr u32 HCCL_ALGO_LEVEL_2 = 2;           // HCCL 算法层级2
constexpr u32 HCCL_ALGO_LEVEL_3 = 3;           // HCCL 算法层级3
constexpr u32 HCCL_ALGO_LEVEL_NUM = 4;         // HCCL 算法层级最多4级
constexpr u32 HCCL_RETRY_ENABLE_LEVEL_0 = 0;   // HCCL 重执行层级0
constexpr u32 HCCL_RETRY_ENABLE_LEVEL_1 = 1;   // HCCL 重执行层级1
constexpr u32 HCCL_RETRY_ENABLE_LEVEL_2 = 2;   // HCCL 重执行层级2
constexpr u32 HCCL_RETRY_ENABLE_LEVEL_NUM = 3; // HCCL 重执行层级最多3级
constexpr u32 HCCL_INVALID_PORT = 65536;       // HCCL 默认无效端口号

constexpr u32 HCCL_QPS_PER_CONNECTION_DEFAULT = 1; // HCCL 默认的rank 间QP个数（仅单算子下生效）
constexpr u32 HCCL_MULTI_QP_THRESHOLD_DEFAULT
    = 512; // 单位kB, 当每个QP分担到的数据量小于512字节时，就不做多QP拆分payload
constexpr u32 HCCL_MULTI_QP_THRESHOLD_MAX = 8192; // 单位kB, qp散列数据门限是8MB

constexpr u32 HCCL_CCL_COMM_DEFAULT_BUFFER_SIZE = 200;
constexpr u32 HCCL_CCL_COMM_BUFFER_MIN = 1;
constexpr u64 HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE = (1 * 1024 * 1024);

typedef enum {
    DETERMINISTIC_DISABLE = 0, // 不支持确定性
    DETERMINISTIC_ENABLE,      // 支持确定性，不支持规约保序
    DETERMINISTIC_STRICT       // 支持确定性以及规约保序
} DeterministicEnableLevel;

enum class ProtocolType {
    TCP = 1, // 拉远TCP模式
    RDMA,    // 拉远RDMA模式
    RESERVED // 拉远未进行模式使能
};
struct MasterInfo {
    hccl::HcclIpAddress serverIp;
    u32 port = HCCL_INVALID_PORT;
    u32 serverDeviceId = INVALID_VALUE_RANKID;
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    hccl::HcclIpAddress agentIp;
};

struct HcclSocketIfName {
    std::vector<std::string> configIfNames; // 用户输入的网卡名列表
    bool searchNot;                         // 匹配还是不匹配，TRUE：不匹配，FALSE：匹配
    bool searchExact;                       // 精确匹配或前缀匹配，TRUE：精确匹配，FALSE：前缀匹配
};

HcclResult InitExternalInput();

HcclResult SetTcpMode(const bool isTcpMode);

HcclResult InitExternalInputHeterog();

HcclResult SetDeterministic(u8 deterministic);

HcclResult SetFftsSwitch(const bool switchStatus);

void SetIfProfile(bool ifProfile);

const bool& GetIfProfile();

void SetProfConfig(u64 profConfig);

const u32& GetExternalInputHcclIfBasePort();

const u32& GetExternalInputTaskExceptionSwitch();

const u32& GetExternalInputIntraRoceSwitch();

const u32& GetExternalInputHcclEnableWhitelist();

const bool& GetExternalInputHcclDumpDebug();

const std::string& GetExternalInputHcclWhiteListFile();

const std::string& GetExternalInputProfilingOption();

const std::string& GetExternalInputCannVersion();

const double& GetExternalInputHcclExecTimeOut();

const s32& GetExternalInputHcclLinkTimeOut();

const s32& GetExternalInputHcclSocketFamily();

const bool& GetExternalInputProfilingMode();

const bool& GetExternalInputHcclIsTcpMode();

const bool& GetExternalInputHcclEnableFfts();

const bool& GetExternalInputHcclDeterministic();

const u8& GetExternalInputHcclDeterministicV2();

const bool& GetExternalInputHcclEnablePipline();

const bool& GetExternalInputInterHccsDisable();

const u64& GetExternalInputCCLBuffSize();

const HcclExecTimeoutSet& GetExternalInputHcclExecTimeoutSet();

const ProtocolType& GetExternalInputProtocolType();

const MasterInfo& GetExternalInputMasterInfo();

const bool& GetExternalInputHcclAicpuUnfold();

const uint8_t& GetExternalInputAicpuCacheEnable();

const bool& GetExternalInputHcclAivMode();

u32 GetExternalInputQpsPerConnection();

u32 GetExternalInputMultiQpThreshold();

const bool& GetRemoteIsHdc();

const bool& GetExternalInputIntraServerRetryEnable();

const bool& GetExternalInputInterServerRetryEnable();

const bool& GetExternalInputInterSuperPodRetryEnable();

const u32& GetExternalInputRetryMaxCnt();

const bool& GetExternalInputOpCounter(); // deprecated

const u32& GetExternalInputRetryHoldTime();

const u32& GetExternalInputRetryIntervalTime();

const bool& GetExternalInputHcclDumpDebug();

const std::string& GetExternalInputLogicSuperPodId();

const bool& GetExternalInputRdmaFastPost();

const bool& GetExternalInputHcclEnableEntryLog();

const std::string& GetExternalInputQpSrcPortConfigPath();

void SetExternalInputDebugConfig(u64 value);

HcclResult SetHccLExecTimeOut(const char* execTimeOutStr, const HcclExecTimeoutSet execTimeOutSet);

HcclResult SetMasterInfo(
    const std::string& masterIp, const std::string& masterPort, const std::string& masterDeviceId,
    const std::string& rankSize, const std::string& rankIp);

void SetDumpDebugMode(const bool dumpDebug);

#endif //  EXTERNALINPUT_PUB_H
