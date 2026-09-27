/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_ENV_CONFIG_H
#define HCCL_ENV_CONFIG_H

#include <vector>
#include <hccl/hccl_types.h>
#include "hccl/base.h"
#include "alg_env_config.h"

/*************** Interfaces ***************/
using HcclSocketPortRange = struct HcclSocketPortRangeDef {
    u32 min;
    u32 max;
};

enum SocketLocation { SOCKET_HOST = 0, SOCKET_NPU = 1 };

enum class InconsistentCheckMode {
    OFF = -1,  // 完全不做校验
    FIRST = 0, // 仅首次算子下发校验
    ON = 1     // 每次算子下发都校验
};

// 定义结构体封装环境变量配置参数
struct EnvConfigParam {
    std::string envName; // 环境变量名
    u32 defaultValue;    // 默认值
    u32 minValue;        // 最小值
    u32 maxValue;        // 最大值
    u32 baseValue;       // 基数（可选，默认配置为0）
};
constexpr s32 HCCL_MIN_CONNECT_FAULT_DETECTION_TIME = 20; // HCCL探测最小超时时间设置为20s
HcclResult InitEnvConfig();

bool GetExternalInputHostPortSwitch();

bool GetExternalInputNpuPortSwitch();

const std::vector<HcclSocketPortRange>& GetExternalInputHostSocketPortRange();

const std::vector<HcclSocketPortRange>& GetExternalInputNpuSocketPortRange();

const bool& GetExternalInputHcclHeartBeatEnable();

const bool& GetExternalInputStuckDetect();

const InconsistentCheckMode& GetExternalInconsistentCheckSwitch();

s32& GetExternalInputDfsConnectionFaultDetectionTime();

u32& GetExternalInputDfsTaskMonitorInterval();

/*************** For Internal Use ***************/

struct EnvConfig {
    // 初始化标识
    bool initialized;

    // 环境变量参数
    bool hostSocketPortSwitch; // HCCL_HOST_SOCKET_PORT_RANGE 环境变量配置则开启；否则关闭
    bool npuSocketPortSwitch;  // HCCL_NPU_SOCKET_PORT_RANGE 环境变量配置则开启；否则关闭
    std::vector<HcclSocketPortRange> hostSocketPortRange;
    std::vector<HcclSocketPortRange> npuSocketPortRange;
    u32 rdmaTrafficClass;
    u32 rdmaServerLevel;
    u32 rdmaTimeOut;  // RDMA超时时间，配置范围5-24，默认值为20
    u32 rdmaRetryCnt; // RDMA重传次数，配置范围1-7，默认值为7
    bool enableClusterHeartBeat;
    bool opCounterEnable;
    InconsistentCheckMode inconsistentCheckSwitch;
    s32 dfsConnectionFaultDetectionTime;
    u32 dfsTaskMonitorInterval;

    // HCCL_ALGO环境变量参数
    bool specificAlgoMode;
    std::map<HcclCMDType, std::vector<HcclAlgoType>> hcclAlgoConfig;

    EnvConfig() { SetDefaultParams(); }
    void SetDefaultParams()
    {
        initialized = false;
        // 环境变量参数
        hostSocketPortSwitch = false;
        npuSocketPortSwitch = false;
        // 初始化 SocketPortRange 为默认值
        hostSocketPortRange.clear();
        npuSocketPortRange.clear();
        // 初始化 rdmaTrafficClass 为默认值
        rdmaTrafficClass = HCCL_RDMA_TC_DEFAULT;
        // 初始化 rdmaServerLevel 为默认值
        rdmaServerLevel = HCCL_RDMA_SL_DEFAULT;
        rdmaTimeOut = HCCL_RDMA_TIMEOUT_DEFAULT;
        rdmaRetryCnt = HCCL_RDMA_RETRY_CNT_DEFAULT;
        // 初始化 enableClusterHeartBeat 为默认值
        enableClusterHeartBeat = true;
        opCounterEnable = true;
        // 初始化 inconsistentCheckSwitch 为默认值
        inconsistentCheckSwitch = InconsistentCheckMode::FIRST;
        dfsConnectionFaultDetectionTime = HCCL_MIN_CONNECT_FAULT_DETECTION_TIME;
        dfsTaskMonitorInterval = 0;
        specificAlgoMode = false;
        for (u32 opType = 0; opType < static_cast<u32>(HcclCMDType::HCCL_CMD_MAX); opType++) {
            hcclAlgoConfig[static_cast<HcclCMDType>(opType)]
                = std::vector<HcclAlgoType>(HCCL_ALGO_LEVEL_NUM, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT);
        }
    }

    static const u32 MAX_LEN_OF_DIGIT_ENV = 10; // 数字环境变量最大长度

    static const u32 HCCL_RDMA_TC_DEFAULT = 132; // 默认的traffic class为132（33*4）
    static const u32 HCCL_RDMA_TC_MIN = 0;
    static const u32 HCCL_RDMA_TC_MAX = 255;
    static const u32 HCCL_RDMA_TC_BASE = 4; // RDMATrafficClass需要时4的整数倍

    static const u32 HCCL_RDMA_SL_DEFAULT = 4; // 默认的server level为4
    static const u32 HCCL_RDMA_SL_MIN = 0;
    static const u32 HCCL_RDMA_SL_MAX = 7;

    static const u32 HCCL_RDMA_TIMEOUT_DEFAULT = 20;    // 默认的TIMEOUT配置为20(对应时间4.096*2^20us)
    static const u32 HCCL_RDMA_TIMEOUT_MIN = 5;         // TIMEOUT最小值为5
    static const u32 HCCL_RDMA_TIMEOUT_MAX = 24;        // TIMEOUT最大值为24
    static const u32 HCCL_RDMA_TIMEOUT_MAX_910_93 = 20; // 910B和910_93 TIMEOUT最大值为20

    static const u32 HCCL_RDMA_RETRY_CNT_DEFAULT = 7; // 默认的Retry Cnt为7
    static const u32 HCCL_RDMA_RETRY_CNT_MIN = 1;     // Retry Cnt最小值为1
    static const u32 HCCL_RDMA_RETRY_CNT_MAX = 7;     // Retry Cnt最大值为7

    static const u32 HCCL_QOS_MIN = 0;
    static const u32 HCCL_QOS_MAX = 7;
    static const u32 HCCL_QOS_DEFAULT = 6;
    /// UBC CCU 路径：通信域 QoS 未配置或与 TP 策略分组对齐时的默认 QoS（0–7）
    static const u32 UB_QOS_DEFAULT = 4;
    // 解析RDMATrafficClass
    HcclResult ParseRDMATrafficClass();
    // 解析RDMAServerLevel
    HcclResult ParseRDMAServerLevel();

    HcclResult ParseRDMATimeOut(std::pair<u32, u32>& rdmaTimeOutRange);

    HcclResult ParseRDMARetryCnt();

    static const u32& GetExternalInputRdmaTrafficClass();
    static const u32& GetExternalInputRdmaServerLevel();
    static const u32& GetExternalInputRdmaTimeOut();
    static const u32& GetExternalInputRdmaRetryCnt();

    bool CheckEnvLen(const char* envStr, u32 envMaxLen);
};
static EnvConfig g_envConfig;

HcclResult ResetEnvConfigInitState();

HcclResult InitEnvParam();

HcclResult ParseHostSocketPortRange();

HcclResult ParseNpuSocketPortRange();

HcclResult CheckSocketPortRangeValid(const std::string& envName, const std::vector<HcclSocketPortRange>& portRanges);

HcclResult PortRangeSwitchOn(const SocketLocation& socketLoc);

HcclResult ParseDFSConfig();

HcclResult SetHcclAlgoConfig(const std::string& hcclAlgo);

HcclResult ParseHcclAlgo();

void PrintSocketPortRange(const std::string& envName, const std::vector<HcclSocketPortRange>& portRangeVec);

HcclResult ParseEnvConfig(const EnvConfigParam& param, std::string& envValue, u32& resultValue);

HcclResult
ParseSingleDFSConfigItem(const std::string& dfsConfigEnv, const std::string& configName, std::string& configResult);

HcclResult ParseMonitor(std::string& taskMonitorInterval, s32& monitorTime);

HcclResult GetKeyWordPath(const std::string& cannEnvStr, const std::string& keyStr, std::string& cannPath);

HcclResult ParseLibraryPath(std::string& cannPath);
#endif // HCCL_ENV_INPUT_H
