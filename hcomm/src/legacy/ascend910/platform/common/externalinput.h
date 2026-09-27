/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXTERNALINPUT_H
#define EXTERNALINPUT_H

#include <map>
#include <nlohmann/json.hpp>

#include "externalinput_pub.h"

constexpr s32 MAX_LENGTH_OF_U32 = 10;
constexpr u32 MAX_LEN_OF_DIGIT_ENV = 10; // 数字环境变量最大长度

constexpr u32 HCCL_WHITELIST_OFF = 0;

constexpr s32 HCCL_MIN_LINK_TIME_OUT_S = 120;        // HCCL 建链最小超时时间设置为120s
constexpr s32 HCCL_MAX_LINK_TIME_OUT_S = (120 * 60); // HCCL 最大建链超时时间设置为120*60s

constexpr s32 HCCL_EXEC_TIME_OUT_S
    = NOTIFY_MAX_WAIT_TIME; // 910B和910_93场景非HCCL默认的Notify wait超时时间设置为最大超时时间
constexpr s32 HCCL_EXEC_TIME_OUT_S_910_93
    = NOTIFY_MAX_WAIT_TIME_910_93; // 910B和910_93 HCCL默认的Notify wait超时时间设置为最大超时时间
constexpr s32 HCCL_INTEVAL_EXEC_TIME_OUT_S = 68; // notifywait的设置参数必须是68的整数倍

constexpr u32 HCCL_MEM_SAMPLER_ITER_U = 0x00000000; // 内存采样迭代轮次默认不启用值为0x00000000
constexpr s32 HCCL_DEAFULT_P2P_DISABLE = 0;         // HCCL 默认P2P使能

constexpr u32 HCCL_RDMA_TC_DEFAULT = 132; // 默认的traffic class为132(33*4)
constexpr u32 HCCL_RDMA_TC_MIN = 0;       // traffic class最小值为0
constexpr u32 HCCL_RDMA_TC_MAX = 255;     // traffic class最大值为255
constexpr u32 HCCL_RDMA_TC_BASE = 4;      // RDMATrafficClass需要是4的整数倍

constexpr u32 HCCL_RDMA_SL_DEFAULT = 4; // 默认的server level为4
constexpr u32 HCCL_RDMA_SL_MIN = 0;     // server level最小值为0
constexpr u32 HCCL_RDMA_SL_MAX = 7;     // server level最大值为7

constexpr u32 HCCL_RDMA_TIMEOUT_DEFAULT = 20;    // 默认的TIMEOUT配置为20(对应时间4.096*2^20us)
constexpr u32 HCCL_RDMA_TIMEOUT_MIN = 5;         // TIMEOUT最小值为5
constexpr u32 HCCL_RDMA_TIMEOUT_MAX = 24;        // TIMEOUT最大值为24
constexpr u32 HCCL_RDMA_TIMEOUT_MAX_910_93 = 20; // 910B和910_93 TIMEOUT最大值为20

constexpr u32 HCCL_RDMA_RETRY_CNT_DEFAULT = 7; // 默认的Retry Cnt为7
constexpr u32 HCCL_RDMA_RETRY_CNT_MIN = 1;     // Retry Cnt最小值为1
constexpr u32 HCCL_RDMA_RETRY_CNT_MAX = 7;     // Retry Cnt最大值为7
constexpr u32 HCCL_BASE_PORT_MIN = 1024;       // Base port最小值为1024

constexpr u32 HCCL_RETRY_MAXCNT_DEFAULT = 1;       // 最大重执行次数，默认配置1
constexpr u32 HCCL_RETRY_HOLD_TIME_DEFAULT = 5000; // 首次重执行等待时间，默认5s
constexpr u32 HCCL_RETRY_INTERVAL_DEFAULT = 1000;  // 重执行间隔，默认1s

constexpr u32 HCCL_RETRY_MAXCNT_MIN = 1;        // 最大重执行次数，最小值1次
constexpr u32 HCCL_RETRY_MAXCNT_MAX = 10;       // 最大重执行次数，最大值10次
constexpr u32 HCCL_RETRY_HLOD_TIME_MAX = 60000; // 首次重执行等待时间，最大值60s, 单位ms
constexpr u32 HCCL_RETRY_INTERVAL_MAX = 60000;  // 重执行间隔，最大值60s, 单位ms
constexpr int HCCL_RETRY_PARAM_NUM = 3; // 重执行参数个数(MaxCnt/HoldTime/IntervalTime)，sscanf_s解析成功的返回值

constexpr u32 MAX_LEN_OF_LOGIC_SUPER_ID = 128; // 逻辑超节点最大长度
//  外部输入参数
struct ExternalInput {
    //  初始化判断
    bool initialized;
    HcclExecTimeoutSet execTimeOutSet;

    //  环境变量参数
    s32 linkTimeOut;
    double execTimeOut;
    u32 hcclIfBasePort;
    u32 rdmaTrafficClass;
    u32 rdmaServerLevel;
    u32 rdmaTimeOut;         // RDMA超时时间，配置范围5-24，默认值为20
    u32 rdmaRetryCnt;        // RDMA重传次数，配置范围1-7，默认值为7
    u32 taskExceptionSwitch; // task_exception_handler ctx子任务维度开关，默认0（关闭）
    u32 qpsPerConnection;    // 用于多QP散列场景下，指定两个rank之间的连接个数
    u32 multiQpThreshold;    // 多QP散列下，每个QP分担的数据量阈值，小于这个阈值则不切分多QP
    u32 retryMaxCnt;         // 重执行最大尝试次数，配置范围[0,60000]，默认值为3
    u32 retryIntervalTime;   // 重执行间隔时间，配置范围[0,3600000]，默认值1000
    u32 retryHoldTime;       // 首次重执行等待时间，配置范围[0,3600000]，默认值1000

    // cann版本信息
    std::string cannVersion;

    //  配置文件参数
    u32 intraRoceSwitch; // server内的通信方式 与intraPcieSwitch组合使用，默认为0

    std::string hcclWhiteListFile;
    bool profilingMode;
    std::string profilingOption;
    u64 profConfig; // Msprof profiling配置
    u32 enableWhitelist;
    hccl::HcclIpAddress hcclControlIfIp;
    s32 hcclSocketFamily;
    HcclSocketIfName hcclSocketIfName;
    bool isTcpMode;

    u32 esMaxPsTable; // ES场景获取每条流上的最大表数
    u32 streamNum;
    bool dumpDebug;
    ProtocolType protocolType = ProtocolType::RESERVED;
    MasterInfo masterInfo;
    bool opCounterEnable;
    u64 cclBufferSize;
    bool enableFfts;
    u8 hcclDeterministic;
    bool isDeterministic;
    bool enablePipeline;
    bool enableEntryLog;
    bool interHccsDisable;
    bool aicpuUnfold;
    uint8_t aicpuCacheEnable;
    bool aivMode;
    bool remoteIsHdc = false;
    bool hcclRetryConfig[HCCL_RETRY_ENABLE_LEVEL_NUM];
    std::string logicSuperPodId;
    bool rdmaFastPost;
    std::string multiQpSrcPortConfigPath;
    s32 increSaveExecTimeOut;
    u64 debugConfig;
    ExternalInput() { SetDefaultParams(); }
    void SetDefaultParams()
    {
        initialized = false;
        execTimeOutSet = HcclExecTimeoutSet::HCCL_EXEC_TIMEOUT_NOT_SET;
        linkTimeOut = HCCL_LINK_TIME_OUT_S;     // 120秒超时
        execTimeOut = NOTIFY_DEFAULT_WAIT_TIME; // HCCL 默认的Notify wait超时时间设置
        rdmaTrafficClass = HCCL_RDMA_TC_DEFAULT;
        rdmaServerLevel = HCCL_RDMA_SL_DEFAULT;
        rdmaTimeOut = HCCL_RDMA_TIMEOUT_DEFAULT;
        rdmaRetryCnt = HCCL_RDMA_RETRY_CNT_DEFAULT;
        taskExceptionSwitch = 0;
        qpsPerConnection = HCCL_QPS_PER_CONNECTION_DEFAULT;
        multiQpThreshold = HCCL_MULTI_QP_THRESHOLD_DEFAULT;
        retryMaxCnt = HCCL_RETRY_MAXCNT_DEFAULT;
        retryHoldTime = HCCL_RETRY_HOLD_TIME_DEFAULT;
        retryIntervalTime = HCCL_RETRY_INTERVAL_DEFAULT;
        cannVersion = "";
        hcclIfBasePort = HCCL_INVALID_PORT; // 无效端口
        intraRoceSwitch = 0;                // server内的通信方式 与intraPcieSwitch组合使用，默认为0

        hcclWhiteListFile = "";
        profilingMode = false;
        profilingOption = "";
        profConfig = 0;
        enableWhitelist = HCCL_WHITELIST_OFF;
        hcclControlIfIp.clear();
        hcclSocketFamily = -1;
        hcclSocketIfName.configIfNames = {};
        hcclSocketIfName.searchExact = false; // false: 默认为前缀匹配
        hcclSocketIfName.searchNot = false;   // false: 默认为匹配

        isTcpMode = false;

        esMaxPsTable = 0;
        streamNum = 1;
        dumpDebug = false;
        protocolType = ProtocolType::RESERVED;

        masterInfo.serverIp.clear();
        masterInfo.port = HCCL_INVALID_PORT;
        masterInfo.serverDeviceId = INVALID_VALUE_RANKID;
        masterInfo.rankSize = INVALID_VALUE_RANKSIZE;
        masterInfo.agentIp.clear();

        cclBufferSize = HCCL_CCL_COMM_DEFAULT_BUFFER_SIZE * HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE;
        enableFfts = true;
        hcclDeterministic
            = DETERMINISTIC_DISABLE; // 确定性配置 0：不支持；1：支持确定性不支持规约保序；2：支持确定性&规约保序
        isDeterministic = false; // 兼容性考虑，提供确定性bool
        enablePipeline = false;
        enableEntryLog = false;
        interHccsDisable = false;
        aicpuCacheEnable = 1; // 默认开启aicpu cache (只有当aicpuUnfold为true时才生效)
        opCounterEnable = true;
        logicSuperPodId = "";
        rdmaFastPost = false;
        multiQpSrcPortConfigPath = "";
        increSaveExecTimeOut = NOTIFY_DEFAULT_WAIT_TIME; // HCCL 默认的Notify wait超时时间设置
        debugConfig = 0;
    }
};

HcclResult InitEnvVarParam();

HcclResult ParseLinkConnTimeOut();

HcclResult ParseExecTimeOut();

HcclResult ParseIntraLinkType();

HcclResult GetIntraLinkTypeDigit();

HcclResult ParseHcclWhitelistFilePath();

HcclResult ParseHcclWhitelistSwitch();

HcclResult ParseHcclIfIp();

HcclResult ParseHcclSocketIfName();

HcclResult ParseHcclSocketFamily();

HcclResult SplitHcclSocketIfName(const std::string& socketIfName, std::vector<std::string>& configIfNames);

HcclResult ParseHcclIfBasePort();

bool CheckEnvLen(const char* envStr, u32 envMaxLen);

HcclResult ParseRDMATrafficClass();

HcclResult ParseRDMAServerLevel();

HcclResult ParseRDMATimeOut(std::pair<u32, u32>& rdmaTimeOutRange);

HcclResult ParseRDMARetryCnt();

HcclResult ParseRdmaQpsPerConnection();

HcclResult ParseMultiQpThreshold();

HcclResult ParseCannVersion();

HcclResult ParseCclBufferSize();

HcclResult ParseDeterministic();

HcclResult ParseHcclPipelineModeEnable();

HcclResult ParseTaskExceptionSwitch();

HcclResult ParseEntryLogEnable();

HcclResult ParseOpExpansion();

HcclResult ResetInitState();

HcclResult ParseInterLinkType();

HcclResult ParseRetryEnable();

HcclResult ParseProfilingConfig();

HcclResult ParseRetryParams();

HcclResult ParseLogicSuperPodId();

HcclResult ParseDebugConfig();

HcclResult SplitHcclRetryEnable(const std::string& retryConfig, std::vector<std::string>& retryEnables);

HcclResult CollectRetryEnableFromConfig(const std::vector<std::string>& retryEnables);

u32 GetEsMaxPsTable();

HcclResult SetEsStreamNum(const u32 streamNum);

u32 GetEsStreamNum();

const u32& GetExternalInputRdmaTrafficClass();

const u32& GetExternalInputRdmaServerLevel();

const u32& GetExternalInputRdmaTimeOut();

const u32& GetExternalInputRdmaRetryCnt();

const hccl::HcclIpAddress& GetExternalInputHcclControlIfIp();

const HcclSocketIfName& GetExternalInputHcclSocketIfName();

HcclResult ParseRdmaFastPost();

HcclResult ParseMultiQpSrcPortConfigPath();
HcclResult SetIncreSaveExecTimeOut(const s32 execTimeout);
const s32& GetIncreSaveExecTimeOut();

const u64& GetProfConfig();

const u64& GetExternalInputDebugConfig();
void SetExternalInputProtocolType(ProtocolType prot);
#endif //  EXTERNALINPUT_H
