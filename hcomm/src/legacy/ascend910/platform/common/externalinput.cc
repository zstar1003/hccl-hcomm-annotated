/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <algorithm>
#include <adapter_rts.h>
#include <thread>
#include <mutex>
#include <unistd.h>
#include <regex>
#include <cmath>
#include "sal.h"
#include "adapter_error_manager.h"
#include "externalinput_pub.h"
#include "mmpa_api.h"
#include "config_plf_log.h"
#include "device_capacity.h"
#include "externalinput.h"

using namespace std;
using namespace hccl;

#define GET_ENV(IdName)                                                   \
    ({                                                                    \
        char* mmSysGetEnvValue = nullptr;                                 \
        MM_SYS_GET_ENV(IdName, mmSysGetEnvValue);                         \
        (mmSysGetEnvValue != nullptr) ? mmSysGetEnvValue : "EmptyString"; \
    })

static std::mutex g_externalInputMutex;
static ExternalInput g_externalInput; //  外部输入参数 （环境变量，配置文件）
static thread_local bool g_ifProf = true;
constexpr u32 HCCL_QPS_PER_CONNECTION_MAX = 32; // HCCL 默认的rank 间QP个数（仅单算子下生效）

HcclResult InitExternalInput()
{
    std::lock_guard<std::mutex> lock(g_externalInputMutex);
    if (g_externalInput.initialized) {
        return HCCL_SUCCESS;
    }
    //  初始化环境变量参数
    CHK_RET(InitEnvVarParam());

    g_externalInput.initialized = true;

    return HCCL_SUCCESS;
}

HcclResult InitExternalInputHeterog()
{
    // 解析server间通信协议
    g_externalInput.remoteIsHdc = true;
    g_externalInput.protocolType = ProtocolType::RDMA;
    HCCL_INFO(
        "Protocol[%d], remoteIsHdc[%d]", static_cast<s32>(g_externalInput.protocolType),
        g_externalInput.remoteIsHdc ? 1 : 0);
    return HCCL_SUCCESS;
}

void SetExternalInputProtocolType(ProtocolType prot) { g_externalInput.protocolType = prot; }

HcclResult ResetInitState()
{
    g_externalInput.SetDefaultParams();
    return HCCL_SUCCESS;
}

HcclResult InitEnvVarParam()
{
    // 解析hccl connect timeout 超时时间
    HcclResult ret = ParseLinkConnTimeOut();
    RPT_ENV_ERR(
        ret != HCCL_SUCCESS, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
        std::vector<std::string>(
            {GET_ENV(MM_ENV_HCCL_CONNECT_TIMEOUT), "HCCL_CONNECT_TIMEOUT",
             "a number greater than or equal to 120s and less than or equal to 7200s"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] In init env variable param, parse link "
            "time out failed. errorno[%d]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), HCCL_ERROR_CODE(ret), ret),
        ret);

    // 解析hccl execute timeout 超时时间
    // 环境变量的解析在options解析之后, 如果options中配置了hccl execute timeout, 则不解析环境变量中的hccl execute
    // timeout
    if (g_externalInput.execTimeOutSet != HcclExecTimeoutSet::HCCL_EXEC_TIMEOUT_SET_BY_OPTIONS) {
        ret = ParseExecTimeOut();
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[%s][%s]errNo[0x%016llx] In init env variable param, parse execute time out failed. errorno[%d]",
                LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), HCCL_ERROR_CODE(ret), ret),
            ret);
    }

    // 解析server内通信方式
    ret = ParseIntraLinkType();
    std::string userInput = "PCIE enable: " + std::string(GET_ENV(MM_ENV_HCCL_INTRA_PCIE_ENABLE))
                            + " or ROCE enable:" + std::string(GET_ENV(MM_ENV_HCCL_INTRA_ROCE_ENABLE));
    RPT_ENV_ERR(
        ret != HCCL_SUCCESS, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
        std::vector<std::string>(
            {userInput, "HCCL_INTRA_PCIE_ENABLE or HCCL_INTRA_ROCE_ENABLE", "0 or 1 (but not both 1)"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] In init env variable param, parse intra "
            "comm type failed. errorno[%d]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), HCCL_ERROR_CODE(ret), ret),
        ret);

    // 解析profiling 配置
    ret = ParseProfilingConfig();
    RPT_ENV_ERR(
        ret != HCCL_SUCCESS, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
        std::vector<std::string>({GET_ENV(MM_ENV_PROFILING_MODE), "PROFILING_MODE", "true or false"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] In init env variable param, parse profiling "
            "config failed. errorno[%d]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), HCCL_ERROR_CODE(ret), ret),
        ret);

    // 解析whitelist switch配置
    ret = ParseHcclWhitelistSwitch();
    RPT_ENV_ERR(
        ret != HCCL_SUCCESS, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
        std::vector<std::string>({GET_ENV(MM_ENV_HCCL_WHITELIST_DISABLE), "HCCL_WHITELIST_DISABLE", "0 or 1"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] In init env variable param, parse whitelist switch failed. errorno[%d]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), HCCL_ERROR_CODE(ret), ret),
        ret);

    // 解析whitelist file配置
    ret = ParseHcclWhitelistFilePath();
    RPT_ENV_ERR(
        ret != HCCL_SUCCESS, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
        std::vector<std::string>(
            {GET_ENV(MM_ENV_HCCL_WHITELIST_FILE), "HCCL_WHITELIST_FILE",
             "absolute file path with length less than 4096"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] In init env variable param, "
            "parse whitelist file failed. errorno[%d]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), HCCL_ERROR_CODE(ret), ret),
        ret);

    // 解析rootinfo IF配置
    ret = ParseHcclIfIp();
    RPT_ENV_ERR(
        ret != HCCL_SUCCESS, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
        std::vector<std::string>({GET_ENV(MM_ENV_HCCL_IF_IP), "HCCL_IF_IP", "\"ip[%ifname]\""}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] In init env variable param, parse rootInfo network interface failed. "
            "errorno[%d]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), HCCL_ERROR_CODE(ret), ret),
        ret);

    // 解析Host Socket IfName配置
    ret = ParseHcclSocketIfName();
    RPT_ENV_ERR(
        ret != HCCL_SUCCESS, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
        std::vector<std::string>(
            {GET_ENV(MM_ENV_HCCL_SOCKET_IFNAME), "HCCL_SOCKET_IFNAME", "Format: [=|^=|<interface>[,<interface>...]]"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] In init env variable param, parse host interface name failed. errorno[%d]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), HCCL_ERROR_CODE(ret), ret),
        ret);

    ret = ParseHcclSocketFamily();
    RPT_ENV_ERR(
        ret != HCCL_SUCCESS, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
        std::vector<std::string>({GET_ENV(MM_ENV_HCCL_SOCKET_FAMILY), "HCCL_SOCKET_FAMILY", "AF_INET or AF_INET6"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] In init env variable param, parse hccl socket family config failed. "
            "errorno[%d]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), HCCL_ERROR_CODE(ret), ret),
        ret);

    // 解析BASE端口
    ret = ParseHcclIfBasePort();
    RPT_ENV_ERR(
        ret != HCCL_SUCCESS, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
        std::vector<std::string>({GET_ENV(MM_ENV_HCCL_IF_BASE_PORT), "HCCL_IF_BASE_PORT", "range[1024,65520]"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] In init env variable param, parse IF base port config failed. errorno[%d]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), HCCL_ERROR_CODE(ret), ret),
        ret);

    // 解析Cann版本
    // CANN版本的校验不影响HCCL业务，所以解析过程中不返回错误
    (void)ParseCannVersion();

    // 解析RDMATrafficClass
    ret = ParseRDMATrafficClass();
    RPT_ENV_ERR(
        ret != HCCL_SUCCESS, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
        std::vector<std::string>(
            {GET_ENV(MM_ENV_HCCL_RDMA_TC), "HCCL_RDMA_TC", "range[0, 255], Must be a multiple of 4"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] In init env variable param, parse HCCL_RDMA_TC failed. errorno[%d]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), HCCL_ERROR_CODE(ret), ret),
        ret);

    // 解析RDMAServerLevel
    ret = ParseRDMAServerLevel();
    RPT_ENV_ERR(
        ret != HCCL_SUCCESS, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
        std::vector<std::string>({GET_ENV(MM_ENV_HCCL_RDMA_SL), "HCCL_RDMA_SL", "range[0, 7]"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] In init env variable param, parse HCCL_RDMA_SL failed. errorno[%d]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), HCCL_ERROR_CODE(ret), ret),
        ret);

    // 解析RDMATimeOut
    std::pair<u32, u32> rdmaTimeOutRange = {0, 0};
    ret = ParseRDMATimeOut(rdmaTimeOutRange);
    std::string vaildRange
        = "range[" + std::to_string(rdmaTimeOutRange.first) + " ," + std::to_string(rdmaTimeOutRange.second) + "]";
    RPT_ENV_ERR(
        ret != HCCL_SUCCESS, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
        std::vector<std::string>({GET_ENV(MM_ENV_HCCL_RDMA_TIMEOUT), "HCCL_RDMA_TIMEOUT", vaildRange}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] In init env variable param, parse HCCL_RDMA_TIMEOUT failed. errorno[%d]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), HCCL_ERROR_CODE(ret), ret),
        ret);

    // 解析RDMARetryCnt
    ret = ParseRDMARetryCnt();
    RPT_ENV_ERR(
        ret != HCCL_SUCCESS, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
        std::vector<std::string>({GET_ENV(MM_ENV_HCCL_RDMA_RETRY_CNT), "HCCL_RDMA_RETRY_CNT", "range[1, 7]"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] In init env variable param, parse HCCL_RDMA_RETRY_CNT failed. errorno[%d]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), HCCL_ERROR_CODE(ret), ret),
        ret);

    // 解析cclbufersize
    ret = ParseCclBufferSize();
    RPT_ENV_ERR(
        ret != HCCL_SUCCESS, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
        std::vector<std::string>({GET_ENV(MM_ENV_HCCL_BUFFSIZE), "HCCL_BUFFSIZE", "equal to or greater than 1(MB)"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] In init env variable param, parse HCCL_BUFFSIZE failed. errorno[%d]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), HCCL_ERROR_CODE(ret), ret),
        ret);

    // 解析hcclDeterministic,是否为确定性计算
    ret = ParseDeterministic();
    RPT_ENV_ERR(
        ret != HCCL_SUCCESS, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
        std::vector<std::string>({GET_ENV(MM_ENV_HCCL_DETERMINISTIC), "HCCL_DETERMINISTIC", "true, false or strict"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] In init env variable param, parse HCCL_DETERMINISTIC failed. errorno[%d]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), HCCL_ERROR_CODE(ret), ret),
        ret);

    // 解析ffts+模式（子任务粒度）下task_exception_handler开关
    ret = ParseTaskExceptionSwitch();
    RPT_ENV_ERR(
        ret != HCCL_SUCCESS, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
        std::vector<std::string>({GET_ENV(MM_ENV_HCCL_DIAGNOSE_ENABLE), "HCCL_DIAGNOSE_ENABLE", "0 or 1"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] In init env variable param, parse HCCL_DIAGNOSE_ENABLE failed. "
            "errorno[%d]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), HCCL_ERROR_CODE(ret), ret),
        ret);

    // 解析Entry日志开关
    ret = ParseEntryLogEnable();
    RPT_ENV_ERR(
        ret != HCCL_SUCCESS, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
        std::vector<std::string>({GET_ENV(MM_ENV_HCCL_ENTRY_LOG_ENABLE), "HCCL_ENTRY_LOG_ENABLE", "0 or 1"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] In init env variable param, parse HCCL_ENTRY_LOG_ENABLE failed. "
            "errorno[%d]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), HCCL_ERROR_CODE(ret), ret),
        ret);

    // 解析超节点内节点间链路选择开关
    ret = ParseInterLinkType();
    RPT_ENV_ERR(
        ret != HCCL_SUCCESS, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
        std::vector<std::string>(
            {GET_ENV(MM_ENV_HCCL_INTER_HCCS_DISABLE), "HCCL_INTER_HCCS_DISABLE", "true or false"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] In init env variable param, parse HCCL_INTER_HCCS_DISABLE failed. "
            "errorno[%d]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), HCCL_ERROR_CODE(ret), ret),
        ret);

    ret = ParseOpExpansion();
    RPT_ENV_ERR(
        ret != HCCL_SUCCESS, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
        std::vector<std::string>(
            {GET_ENV(MM_ENV_HCCL_OP_EXPANSION_MODE), "HCCL_OP_EXPANSION_MODE",
             "Atlas A3: AI_CPU | AIV; Atlas A2 AIV | HOST | HOST_TS; Atlas 300I AI_CPU | HOST"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] In init env variable param, parse HCCL_OP_EXPANSION_MODE failed. "
            "errorno[%d]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), HCCL_ERROR_CODE(ret), ret),
        ret);

    // 解析rank 间的QP个数
    ret = ParseRdmaQpsPerConnection();
    RPT_ENV_ERR(
        ret != HCCL_SUCCESS, "", std::vector<std::string>({"value", "env", "expect"}),
        std::vector<std::string>(
            {GET_ENV(MM_ENV_HCCL_RDMA_QPS_PER_CONNECTION), "HCCL_RDMA_QPS_PER_CONNECTION",
             "The allowed value range is [1, 32], "
             "but the recommended value range is [1, 8]."}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] In init env variable param, "
            "parse HCCL_RDMA_QPS_PER_CONNECTION(range[1,32]) failed. errorno[%d]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), HCCL_ERROR_CODE(ret), ret),
        ret);

    // 解析rank 间多QP切分门限
    ret = ParseMultiQpThreshold();
    RPT_ENV_ERR(
        ret != HCCL_SUCCESS, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
        std::vector<std::string>(
            {GET_ENV(MM_ENV_HCCL_MULTI_QP_THRESHOLD), "HCCL_MULTI_QP_THRESHOLD", "larger than 0, less than 8193(KB)"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] In init env variable param, parse "
            "HCCL_MULTI_QP_THRESHOLD(range[1,8192]) failed. errorno[%d]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), HCCL_ERROR_CODE(ret), ret),
        ret);

#ifndef CCL_KERNEL_AICPU
    // 解析重执行设置
    ret = ParseRetryEnable();
    RPT_ENV_ERR(
        ret != HCCL_SUCCESS, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
        std::vector<std::string>({GET_ENV(MM_ENV_HCCL_OP_RETRY_ENABLE), "HCCL_OP_RETRY_ENABLE", "0 or 1"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] In init env variable param, parse HCCL_OP_RETRY_ENABLE failed. "
            "errorno[%d]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), HCCL_ERROR_CODE(ret), ret),
        ret);
#endif

#ifndef CCL_KERNEL_AICPU
    // 解析重执行最大尝试次数 重执行间隔时间 首次重执行等待时间 */
    ret = ParseRetryParams();
    RPT_ENV_ERR(
        ret != HCCL_SUCCESS, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
        std::vector<std::string>(
            {GET_ENV(MM_ENV_HCCL_OP_RETRY_PARAMS), "HCCL_OP_RETRY_PARAMS",
             "format must be: \"MaxCnt:cnt,HoldTime:time,IntervalTime:time\"."
             "cnt range is [1, 10], time range is [0, 60000]"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] In init env variable param, parse HCCL_OP_RETRY_PARAMS failed. "
            "errorno[%d]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), HCCL_ERROR_CODE(ret), ret),
        ret);
#endif

    ret = ParseLogicSuperPodId();
    RPT_ENV_ERR(
        ret != HCCL_SUCCESS, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
        std::vector<std::string>(
            {GET_ENV(MM_ENV_HCCL_LOGIC_SUPERPOD_ID), "HCCL_LOGIC_SUPERPOD_ID", "length must be less than 128"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] In init env variable param, parse HCCL_LOGIC_SUPERPOD_ID failed. "
            "errorno[%d]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), HCCL_ERROR_CODE(ret), ret),
        ret);

    ret = ParseRdmaFastPost();
    RPT_ENV_ERR(
        ret != HCCL_SUCCESS, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
        std::vector<std::string>(
            {GET_ENV(MM_ENV_HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT), "HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT",
             "true or false"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] In init env variable param, parse "
            "HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT failed. errorno[%d]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), HCCL_ERROR_CODE(ret), ret),
        ret);

    // 解析多QP源端口号配置文件路径
    ret = ParseMultiQpSrcPortConfigPath();
    RPT_ENV_ERR(
        ret != HCCL_SUCCESS, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
        std::vector<std::string>(
            {GET_ENV(MM_ENV_HCCL_RDMA_QP_PORT_CONFIG_PATH), "HCCL_RDMA_QP_PORT_CONFIG_PATH",
             "a valid existing file path"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] In init env variable param, "
            "parse MultiQpSrcPortConfigPath failed. errorno[%d]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), HCCL_ERROR_CODE(ret), ret),
        ret);

    // 解析ParseDebugConfig
    ret = ParseDebugConfig();
    char* env = nullptr; // 环境变量值
    MM_SYS_GET_ENV(MM_ENV_HCCL_DEBUG_CONFIG, env);
    std::string envValue = env != nullptr ? std::string(env) : "null";
    RPT_ENV_ERR(
        ret != HCCL_SUCCESS, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
        std::vector<std::string>({envValue, "HCCL_DEBUG_CONFIG", "ALG,TASK,RESOURCE(optionally prefixed with'^')"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] In init env variable param, "
            "parse HCCL_DEBUG_CONFIG failed. errorno[%d]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), HCCL_ERROR_CODE(ret), ret),
        ret);

    return HCCL_SUCCESS;
}

HcclResult ParseRdmaFastPost()
{
    std::string rdmaFastPostEnv = GET_ENV(MM_ENV_HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT);
    if (rdmaFastPostEnv == "EmptyString") {
        HCCL_RUN_INFO(
            "[HCCL_ENV] HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT set by default to [%s], rdmaFastPost is [%d]",
            rdmaFastPostEnv.c_str(), g_externalInput.rdmaFastPost);
        return HCCL_SUCCESS;
    }
    std::transform(rdmaFastPostEnv.begin(), rdmaFastPostEnv.end(), rdmaFastPostEnv.begin(), ::toupper);
    if (rdmaFastPostEnv == "TRUE") {
        g_externalInput.rdmaFastPost = true;
    } else if (rdmaFastPostEnv == "FALSE") {
        g_externalInput.rdmaFastPost = false;
    } else {
        HCCL_ERROR(
            "HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT is set to [%s], which is incorrect. Please check",
            rdmaFastPostEnv.c_str());
        return HCCL_E_PARA;
    }
    HCCL_RUN_INFO(
        "[HCCL_ENV] HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT set by environment to [%s], rdmaFastPost is [%d].",
        rdmaFastPostEnv.c_str(), g_externalInput.rdmaFastPost);
    return HCCL_SUCCESS;
}

bool IsValidExecTimeOutMs(const std::string& execTimeOutStr)
{
    // 校验配置值为数字格式，小数点最多2位
    std::regex validFormat(R"(^\d+(\.\d{1,2})?$)");
    if (!std::regex_match(execTimeOutStr, validFormat)) {
        return false;
    }

    return true;
}

HcclResult ParseExecTimeOut()
{
    std::string execTimeOutEnv = GET_ENV(MM_ENV_HCCL_EXEC_TIMEOUT);
    if (execTimeOutEnv == "EmptyString") {
        HCCL_RUN_INFO("[HCCL_ENV] HCCL_EXEC_TIMEOUT set by default to [%d]s", NOTIFY_DEFAULT_WAIT_TIME);
        return HCCL_SUCCESS;
    }

    // 校验环境变量长度
    bool isEnvLenValid = CheckEnvLen(execTimeOutEnv.c_str(), MAX_LEN_OF_DIGIT_ENV);
    CHK_PRT_RET(
        !isEnvLenValid,
        HCCL_ERROR(
            "[Parse][ExecuteTimeOut]errNo[0x%016llx] Invalid ExecuteTimeOut env len, len is bigger than "
            "[%u]. errorno[%d]",
            HCCL_ERROR_CODE(HCCL_E_PARA), MAX_LEN_OF_DIGIT_ENV, HCCL_E_PARA),
        HCCL_E_PARA);

    CHK_RET(SetHccLExecTimeOut(execTimeOutEnv.c_str(), HcclExecTimeoutSet::HCCL_EXEC_TIMEOUT_SET_BY_ENV));

    return HCCL_SUCCESS;
}

HcclResult ParseLinkConnTimeOut()
{
    std::string timeOutEnv = GET_ENV(MM_ENV_HCCL_CONNECT_TIMEOUT);
    s32 timeOut = HCCL_LINK_TIME_OUT_S;

    if (timeOutEnv != "EmptyString") {
        // 校验环境变量长度
        bool isEnvLenValid = CheckEnvLen(timeOutEnv.c_str(), MAX_LEN_OF_DIGIT_ENV);
        CHK_PRT_RET(
            !isEnvLenValid,
            HCCL_ERROR(
                "[Parse][LinkConnTimeOut]errNo[0x%016llx] Invalid LinkConnTimeOut env len, len is bigger than "
                "[%u]. errorno[%d]",
                HCCL_ERROR_CODE(HCCL_E_PARA), MAX_LEN_OF_DIGIT_ENV, HCCL_E_PARA),
            HCCL_E_PARA);

        CHK_RET(IsAllDigit(timeOutEnv.c_str()));
        HcclResult ret = SalStrToInt(timeOutEnv, HCCL_BASE_DECIMAL, timeOut);
        // 若转换出错或者设置的超时时间小于2分钟或大于2小时，报错并设为默认值
        if (ret != HCCL_SUCCESS || (timeOut < HCCL_MIN_LINK_TIME_OUT_S) || (timeOut > HCCL_MAX_LINK_TIME_OUT_S)) {
            HCCL_ERROR(
                "[Parse][LinkConnTimeOut]environmental variable HCCL_CONNECT_TIMEOUT error, errNo[0x%016llx]"
                "timeOutEnv[%ss] timeRange[%ds,%ds]",
                HCOM_ERROR_CODE(ret), timeOutEnv.c_str(), HCCL_MIN_LINK_TIME_OUT_S, HCCL_MAX_LINK_TIME_OUT_S);
            g_externalInput.linkTimeOut = HCCL_LINK_TIME_OUT_S;
            return HCCL_E_PARA;
        }
        HCCL_RUN_INFO("[HCCL_ENV] HCCL_CONNECT_TIMEOUT set by environment to [%d]s", timeOut);
    } else {
        HCCL_RUN_INFO("[HCCL_ENV] HCCL_CONNECT_TIMEOUT set by default to [%d]s", timeOut);
    }

    g_externalInput.linkTimeOut = timeOut;
    return HCCL_SUCCESS;
}

HcclResult GetIntraLinkTypeDigit(std::string& intraCommStr, u32& intraCommDig)
{
    CHK_RET(IsAllDigit(intraCommStr.c_str()));
    CHK_RET(SalStrToULong(intraCommStr.c_str(), HCCL_BASE_DECIMAL, intraCommDig));

    if ((intraCommDig != 0) && (intraCommDig != 1)) { // 判断转换后的数字是否为0或1
        HCCL_ERROR("[Get][IntraLinkTypeDigit]environmental digit variable error, intraCommDig[%u]", intraCommDig);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult ParseDeterministic()
{
    std::string hcclDeterministicEnv = GET_ENV(MM_ENV_HCCL_DETERMINISTIC);
    if (hcclDeterministicEnv == "EmptyString") {
        HCCL_RUN_INFO("[HCCL_ENV] HCCL_DETERMINISTIC set by default to [false]");
        return HCCL_SUCCESS;
    }

    std::transform(hcclDeterministicEnv.begin(), hcclDeterministicEnv.end(), hcclDeterministicEnv.begin(), ::toupper);
    if (hcclDeterministicEnv != "STRICT" && hcclDeterministicEnv != "TRUE" && hcclDeterministicEnv != "FALSE") {
        HCCL_ERROR("HCCL_DETERMINISTIC is set to [%s], which is incorrect. Please check", hcclDeterministicEnv.c_str());
        return HCCL_E_PARA;
    }
    if (hcclDeterministicEnv == "STRICT") {
        // 规约保序场景（严格的确定性计算，在确定性的基础上强保证规约顺序一致）
        DevType deviceType = DevType::DEV_TYPE_COUNT;
        CHK_RET(hrtGetDeviceType(deviceType));
        if (deviceType != DevType::DEV_TYPE_910B && deviceType != DevType::DEV_TYPE_910_93) {
            // 规约保序仅支持A2 A3场景
            HCCL_ERROR(
                "HCCL_DETERMINISTIC is set to [%s], Reduce order preservation is not supported for "
                "deviceType[%d], please check",
                hcclDeterministicEnv.c_str(), deviceType);
            return HCCL_E_NOT_SUPPORT;
        }
        g_externalInput.hcclDeterministic = DETERMINISTIC_STRICT;
    } else if (hcclDeterministicEnv == "TRUE") {
        // 确定性计算场景（不保证规约保序）
        g_externalInput.hcclDeterministic = DETERMINISTIC_ENABLE;
    } else {
        g_externalInput.hcclDeterministic = DETERMINISTIC_DISABLE;
    }
    HCCL_RUN_INFO(
        "[HCCL_ENV] HCCL_DETERMINISTIC set by environment to [%s], hcclDeterministic[%u]", hcclDeterministicEnv.c_str(),
        g_externalInput.hcclDeterministic);
    return HCCL_SUCCESS;
}

HcclResult ParseIntraLinkType()
{
    std::string intraPcieEnv = GET_ENV(MM_ENV_HCCL_INTRA_PCIE_ENABLE);
    std::string intraRoceEnv = GET_ENV(MM_ENV_HCCL_INTRA_ROCE_ENABLE);

    u32 intraPcie = 1; // 保存pcie环境变量的解析数字
    u32 intraRoce = 0; // 保存roce环境变量的解析数字

    // 两个通信域环境变量均未设置，默认走pcie
    if (intraPcieEnv == "EmptyString" && intraRoceEnv == "EmptyString") {
        HCCL_RUN_INFO(
            "[HCCL_ENV] HCCL_INTRA_PCIE_ENABLE set by default to [%u], HCCL_INTRA_ROCE_ENABLE set by default to [%u]",
            intraPcie, intraRoce);
        return HCCL_SUCCESS;
    }

    if (intraPcieEnv != "EmptyString") { // 解析HCCL_INTRA_PCIE_ENABLE为数字
        // 校验环境变量长度
        bool isEnvLenValid = CheckEnvLen(intraPcieEnv.c_str(), MAX_LEN_OF_DIGIT_ENV);
        CHK_PRT_RET(
            !isEnvLenValid,
            HCCL_ERROR(
                "[Parse][IntraLinkType]errNo[0x%016llx] Invalid INTRA_PCIE_ENABLE env len, len is bigger than "
                "[%u]. errorno[%d]",
                HCCL_ERROR_CODE(HCCL_E_PARA), MAX_LEN_OF_DIGIT_ENV, HCCL_E_PARA),
            HCCL_E_PARA);
        std::string intraPcieStr(intraPcieEnv);
        CHK_RET(GetIntraLinkTypeDigit(intraPcieStr, intraPcie));
    }

    if (intraRoceEnv != "EmptyString") { // 解析HCCL_INTRA_ROCE_ENABLE为数字
        // 校验环境变量长度
        bool isEnvLenValid = CheckEnvLen(intraRoceEnv.c_str(), MAX_LEN_OF_DIGIT_ENV);
        CHK_PRT_RET(
            !isEnvLenValid,
            HCCL_ERROR(
                "[Parse][IntraLinkType]errNo[0x%016llx] Invalid INTRA_ROCE_ENABLE env len, len is bigger than "
                "[%u]. errorno[%d]",
                HCCL_ERROR_CODE(HCCL_E_PARA), MAX_LEN_OF_DIGIT_ENV, HCCL_E_PARA),
            HCCL_E_PARA);
        std::string intraRoceStr(intraRoceEnv);
        CHK_RET(GetIntraLinkTypeDigit(intraRoceStr, intraRoce));
    }

    // 只配置了roce的环境变量
    if (intraPcieEnv == "EmptyString" && intraRoceEnv != "EmptyString") {
        if (intraRoce == 0) { // roce环境变量值为0，报错
            HCCL_ERROR("[Parse][IntraLinkType]only set HCCL_INTRA_ROCE_ENABLE, and the val is zero, pls set "
                       "HCCL_INTRA_PCIE_ENABLE");
            return HCCL_E_PARA;
        } else { // roce环境变量值为1，走roce
            intraPcie = 0;
        }
        HCCL_RUN_INFO(
            "[HCCL_ENV] HCCL_INTRA_PCIE_ENABLE set by environment to [%u], "
            "HCCL_INTRA_ROCE_ENABLE set by environment to [%u]",
            intraPcie, intraRoce);
    }

    // 只配置了pcie的环境变量
    if (intraPcieEnv != "EmptyString" && intraRoceEnv == "EmptyString") {
        if (intraPcie == 0) { // pcie环境变量值为0，报错
            HCCL_ERROR("[Parse][IntraLinkType]only set HCCL_INTRA_PCIE_ENABLE, and the val is zero, pls set "
                       "HCCL_INTRA_ROCE_ENABLE");
            return HCCL_E_PARA;
        }
        HCCL_RUN_INFO(
            "[HCCL_ENV] HCCL_INTRA_PCIE_ENABLE set by environment to [%u], "
            "HCCL_INTRA_ROCE_ENABLE set by default to [%u]",
            intraPcie, intraRoce);
    }

    // pcie和roce环境变量同时配置且不相等
    if (intraPcieEnv != "EmptyString" && intraRoceEnv != "EmptyString") {
        if ((intraPcie == 0 && intraRoce == 1) || (intraPcie == 1 && intraRoce == 0)) {
            HCCL_RUN_INFO(
                "[HCCL_ENV] HCCL_INTRA_PCIE_ENABLE set by environment to [%u], "
                "HCCL_INTRA_ROCE_ENABLE set by environment to [%u]",
                intraPcie, intraRoce);
        }
    }

    // pcie和roce环境变量同时配置且相等
    if (!(intraPcie ^ intraRoce)) {
        if (intraPcie == 1) { // 同时为1，暂不支持，报错
            HCCL_ERROR(
                "[Parse][IntraLinkType] Enabling intra Pcie and intra Roce at the same time is not supported now.");
            return HCCL_E_PARA;
        } else { // 同时为0，走pcie
            HCCL_WARNING("Pcie and Roce Env both set to zero at the same time, intra comm is default Pcie");
            intraPcie = 1;
        }
        HCCL_RUN_INFO(
            "[HCCL_ENV] HCCL_INTRA_PCIE_ENABLE set by environment to [%u], "
            "HCCL_INTRA_ROCE_ENABLE set by environment to [%u]",
            intraPcie, intraRoce);
    }

    g_externalInput.intraRoceSwitch = intraRoce;
    return HCCL_SUCCESS;
}
HcclResult ParseProfilingConfig()
{
    g_externalInput.profilingMode = false;
    g_externalInput.profilingOption = "";
    std::string profilingEnv = GET_ENV(MM_ENV_PROFILING_MODE);
    CHK_PRT_RET(
        profilingEnv == "EmptyString",
        HCCL_RUN_INFO(
            "[HCCL_ENV] environmental variable PROFILING_MODE and GE profiling option is not set, default: false"),
        HCCL_SUCCESS);
    HCCL_DEBUG("PROFILING_MODE[%s] is set", profilingEnv.c_str());
    CHK_PRT_RET(
        profilingEnv.compare("true") != 0, HCCL_INFO("environmental variable PROFILING_MODE = false"), HCCL_SUCCESS);
    g_externalInput.profilingMode = true;
    profilingEnv = GET_ENV(MM_ENV_PROFILING_OPTIONS);
    CHK_PRT_RET(
        profilingEnv == "EmptyString", HCCL_RUN_INFO("[HCCL_ENV] environmental variable PROFILING_OPTIONS is not set."),
        HCCL_SUCCESS);

    g_externalInput.profilingOption = profilingEnv;
    HCCL_RUN_INFO("[HCCL_ENV] Set Env [PROFILING_MODE]: Value[%s]", g_externalInput.profilingOption.c_str());
    return HCCL_SUCCESS;
}

HcclResult ParseHcclWhitelistFilePath()
{
    CHK_PRT_RET((g_externalInput.enableWhitelist != HCCL_WHITELIST_ON), , HCCL_SUCCESS); // 白名单功能无效时无需解析

    std::string filePath = GET_ENV(MM_ENV_HCCL_WHITELIST_FILE);
    if (filePath == "EmptyString") {
        g_externalInput.hcclWhiteListFile.clear();
        HCCL_RUN_INFO("[HCCL_ENV][Parse][HcclWhitelistFilePath]environmental variable HCCL_WHITELIST_DISABLE is [0],"
                      "but HCCL_WHITELIST_FILE is not set");
    } else {
        u32 len = strnlen(filePath.c_str(), PATH_MAX);
        if (len == (PATH_MAX) || len == 0) {
            HCCL_ERROR(
                "[Parse][HcclWhitelistFilePath]errNo[0x%016llx] env[HCCL_WHITELIST_FILE] is invalid, "
                "len is %u",
                HCCL_ERROR_CODE(HCCL_E_PARA), len);
            return HCCL_E_PARA;
        }
        // 校验文件是否存在
        char realFile[PATH_MAX] = {0};
        if (realpath(filePath.c_str(), realFile) == nullptr) {
            HCCL_RUN_WARNING(
                "[HCCL_ENV][Parse][HcclWhitelistFilePath]path %s is not a valid real path", filePath.c_str());
            g_externalInput.hcclWhiteListFile.clear();
        } else {
            g_externalInput.hcclWhiteListFile = realFile;
        }
        HCCL_RUN_INFO(
            "[HCCL_ENV] HCCL_WHITELIST_FILE set by environment to [%s], realpath[%s].", filePath.c_str(),
            g_externalInput.hcclWhiteListFile.c_str());
    }
    return HCCL_SUCCESS;
}

HcclResult ParseMultiQpSrcPortConfigPath()
{
    std::string filePath = GET_ENV(MM_ENV_HCCL_RDMA_QP_PORT_CONFIG_PATH);
    if (filePath == "EmptyString") {
        g_externalInput.multiQpSrcPortConfigPath.clear();
        HCCL_RUN_INFO(
            "[HCCL_ENV][Parse][MultiQpSrcPortConfigPath]environmental variable HCCL_RDMA_QP_PORT_CONFIG_PATH is empty");
    } else {
        u32 len = filePath.size() > PATH_MAX ? PATH_MAX : filePath.size();
        if (len == (PATH_MAX) || len == 0) {
            HCCL_ERROR(
                "[Parse][MultiQpSrcPortConfigPath]errNo[0x%016llx] env[HCCL_RDMA_QP_PORT_CONFIG_PATH] is invalid,"
                "len is %u",
                HCCL_ERROR_CODE(HCCL_E_PARA), len);
            return HCCL_E_PARA;
        }
        // 校验文件是否存在
        char realFile[PATH_MAX] = {0};
        if (realpath(filePath.c_str(), realFile) == nullptr) {
            HCCL_ERROR(
                "[Parse][MultiQpSrcPortConfigPath]errNo[0x%016llx] path %s is not a valid real path",
                HCOM_ERROR_CODE(HCCL_E_PARA), filePath.c_str());
            return HCCL_E_PARA;
        }
        g_externalInput.multiQpSrcPortConfigPath = realFile;
        HCCL_RUN_INFO(
            "[HCCL_ENV] HCCL_RDMA_QP_PORT_CONFIG_PATH set by environment to [%s], realpath[%s]", filePath.c_str(),
            g_externalInput.multiQpSrcPortConfigPath.c_str());
    }
    return HCCL_SUCCESS;
}

HcclResult ParseHcclWhitelistSwitch()
{
    std::string disableWhitelistEnv = GET_ENV(MM_ENV_HCCL_WHITELIST_DISABLE);
    u32 disableWhitelist = 0;

    if (disableWhitelistEnv != "EmptyString") {
        // 校验环境变量长度
        bool isEnvLenValid = CheckEnvLen(disableWhitelistEnv.c_str(), MAX_LEN_OF_DIGIT_ENV);
        CHK_PRT_RET(
            !isEnvLenValid,
            HCCL_ERROR(
                "[Parse][WhitelistSwitch]errNo[0x%016llx] Invalid HCCL_WHITELIST_DISABLE env len, len is "
                "bigger than [%u]. errorno[%d]",
                HCCL_ERROR_CODE(HCCL_E_PARA), MAX_LEN_OF_DIGIT_ENV, HCCL_E_PARA),
            HCCL_E_PARA);
        HcclResult ret = SalStrToULong(disableWhitelistEnv, HCCL_BASE_DECIMAL, disableWhitelist);
        // 若转换出错或使能开关的值不为0和1，报错并设置为默认值
        if (ret != HCCL_SUCCESS || disableWhitelist > 1) {
            HCCL_ERROR(
                "[Parse][WhitelistSwitch]environmental variable HCCL_WHITELIST_DISABLE[%s] is invalid, "
                "expect[%u ~ %u].",
                disableWhitelistEnv.c_str(), 0, 1);
            g_externalInput.enableWhitelist = HCCL_WHITELIST_OFF;
            return HCCL_E_PARA;
        }
        HCCL_RUN_INFO("[HCCL_ENV] HCCL_WHITELIST_DISABLE set by environment to [%u]", disableWhitelist);
    } else {
        disableWhitelist = 1; // 缺省时关闭白名单
        HCCL_RUN_INFO("[HCCL_ENV] HCCL_WHITELIST_DISABLE set by default to [%u]", disableWhitelist);
    }
    g_externalInput.enableWhitelist = !disableWhitelist;
    return HCCL_SUCCESS;
}

HcclResult ParseHcclIfBasePort()
{
    std::string ifBasePort = GET_ENV(MM_ENV_HCCL_IF_BASE_PORT);
    u32 basePort = HCCL_INVALID_PORT;

    if (ifBasePort != "EmptyString") {
        // 校验环境变量长度
        bool isEnvLenValid = CheckEnvLen(ifBasePort.c_str(), MAX_LEN_OF_DIGIT_ENV);
        CHK_PRT_RET(
            !isEnvLenValid,
            HCCL_ERROR(
                "[Parse][HcclIfBasePort]errNo[0x%016llx] Invalid HcclIfBasePort env len, len is bigger than "
                "[%u]. errorno[%d]",
                HCCL_ERROR_CODE(HCCL_E_PARA), MAX_LEN_OF_DIGIT_ENV, HCCL_E_PARA),
            HCCL_E_PARA);

        CHK_RET(IsAllDigit(ifBasePort.c_str()));
        HcclResult ret = SalStrToULong(ifBasePort, HCCL_BASE_DECIMAL, basePort);
        // 若数字小于1024或者数字大于65520，报错并设置为默认值HOST_CONTROL_BASE_PORT
        if (ret != HCCL_SUCCESS || basePort > HOST_PORT_MAX || basePort < HCCL_BASE_PORT_MIN) {
            HCCL_ERROR(
                "[Parse][HcclIfBasePort]environmental variable HCCL_IF_BASE_PORT error, errNo[0x%016llx]"
                "ifBasePort[%s] portRange[%u,%u]",
                HCOM_ERROR_CODE(ret), ifBasePort.c_str(), HCCL_BASE_PORT_MIN, HOST_PORT_MAX);
            return HCCL_E_PARA;
        }
        HCCL_RUN_INFO("[HCCL_ENV] HCCL_IF_BASE_PORT set by environment to [%u]", basePort);
    } else {
        HCCL_RUN_INFO("[HCCL_ENV] HCCL_IF_BASE_PORT set by default to [%u]", HOST_CONTROL_BASE_PORT);
    }
    g_externalInput.hcclIfBasePort = basePort;
    return HCCL_SUCCESS;
}

HcclResult ParseHcclIfIp()
{
    std::string hcclControlIfIp = GET_ENV(MM_ENV_HCCL_IF_IP);
    if (hcclControlIfIp != "EmptyString") {
        CHK_PRT_RET(
            g_externalInput.hcclControlIfIp.SetReadableAddress(hcclControlIfIp),
            HCCL_ERROR("[Parse][HcclIfIp]IP address[%s] is invalid.", hcclControlIfIp.c_str()), HCCL_E_PARA);
        HCCL_RUN_INFO(
            "[HCCL_ENV] HCCL_IF_IP is set to [%s], ip[%s].", hcclControlIfIp.c_str(),
            g_externalInput.hcclControlIfIp.GetReadableAddress());
    } else {
        HCCL_RUN_INFO("[HCCL_ENV] HCCL_IF_IP is not set");
    }
    return HCCL_SUCCESS;
}

HcclResult ParseHcclSocketFamily()
{
    std::string hcclSocketFamily = GET_ENV(MM_ENV_HCCL_SOCKET_FAMILY);
    if (hcclSocketFamily != "EmptyString") {
        HCCL_RUN_INFO("[HCCL_ENV] HCCL_SOCKET_FAMILY set by environment to [%s]", hcclSocketFamily.c_str());
        if (hcclSocketFamily == "AF_INET") {
            g_externalInput.hcclSocketFamily = AF_INET;
        } else if (hcclSocketFamily == "AF_INET6") {
            g_externalInput.hcclSocketFamily = AF_INET6;
        } else {
            g_externalInput.hcclSocketFamily = -1;
            HCCL_ERROR(
                "[Parse][HcclSocketFamily]environmental variable HCCL_SOCKET_FAMILY[%s] is invalid. it should "
                "be \"AF_INET\" or \"AF_INET6\".",
                hcclSocketFamily.c_str());
            return HCCL_E_PARA;
        }
    } else {
        g_externalInput.hcclSocketFamily = -1;
        HCCL_RUN_INFO("[HCCL_ENV] HCCL_SOCKET_FAMILY is not set and is used by default [AF_INET]");
    }
    return HCCL_SUCCESS;
}

HcclResult ParseHcclSocketIfName()
{
    bool searchNot = false;
    bool searchExact = false;
    std::string hcclSocketIfName = GET_ENV(MM_ENV_HCCL_SOCKET_IFNAME);
    std::string remainSocketIfName = hcclSocketIfName;
    g_externalInput.hcclSocketIfName.configIfNames.clear();

    if (hcclSocketIfName != "EmptyString") {
        // 获取HCCL_SOCKET_IFNAME环境变量匹配规则
        if (!hcclSocketIfName.empty() && hcclSocketIfName.at(0) == '^') {
            searchNot = true;
            // 获取从1位置开始剩余部分环境变量内容
            remainSocketIfName = hcclSocketIfName.substr(1);
        }

        if (!remainSocketIfName.empty() && remainSocketIfName.at(0) == '=') {
            searchExact = true;
            remainSocketIfName = remainSocketIfName.substr(1);
        }

        // 获取用户输入的网卡名列表(使用逗号隔开),将网卡名列表存放到全局vector变量中
        HcclResult ret = SplitHcclSocketIfName(remainSocketIfName, g_externalInput.hcclSocketIfName.configIfNames);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[Parse][HcclSocketIfName]hccl IfName config[%s] is invalid.", hcclSocketIfName.c_str()),
            HCCL_E_PARA);
        HCCL_RUN_INFO("[HCCL_ENV] HCCL_SOCKET_IFNAME set by environment to [%s]", hcclSocketIfName.c_str());
    } else {
        HCCL_RUN_INFO("[HCCL_ENV] HCCL_SOCKET_IFNAME set by default to [%s]", hcclSocketIfName.c_str());
    }
    g_externalInput.hcclSocketIfName.searchNot = searchNot;
    g_externalInput.hcclSocketIfName.searchExact = searchExact;
    return HCCL_SUCCESS;
}

HcclResult SplitHcclSocketIfName(const std::string& socketIfName, std::vector<std::string>& configIfNames)
{
    std::string remainSocketIfName;
    std::size_t found = socketIfName.find(",");
    if ((found == 0) || (found == (socketIfName.length() - 1))) {
        HCCL_ERROR("[Split][HcclSocketIfName] configIfNames config is invalid.");
        return HCCL_E_PARA;
    } else if (found != std::string::npos) {
        remainSocketIfName = socketIfName.substr(found + 1);
    } else {
        // 最后一组配置,剩余的字符串为空
    }
    configIfNames.push_back(socketIfName.substr(0, found));

    if (!remainSocketIfName.empty()) {
        CHK_RET(SplitHcclSocketIfName(remainSocketIfName, configIfNames));
    }

    return HCCL_SUCCESS;
}

HcclResult SetHccLExecTimeOut(const char* execTimeOutStr, const HcclExecTimeoutSet execTimeOutSet)
{
    CHK_PTR_NULL(execTimeOutStr);
    if (!IsValidExecTimeOutMs(execTimeOutStr)) {
        HCCL_ERROR("[SetHccLExecTimeOut]Invalid config value, execTimeOutStr[%s]", execTimeOutStr);
        RPT_ENV_ERR(
            true, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
            std::vector<std::string>(
                {std::string(execTimeOutStr), "HCCL_EXEC_TIMEOUT", "a valid number in the specified range"}));
        return HCCL_E_PARA;
    }
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType)); // 910A和910B要分开
    double hcclExecTimeout = (deviceType == DevType::DEV_TYPE_910_93 || deviceType == DevType::DEV_TYPE_910B) ?
                                 HCCL_EXEC_TIME_OUT_S_910_93 :
                                 HCCL_EXEC_TIME_OUT_S;
    double execTimeOut = hcclExecTimeout;
    g_externalInput.execTimeOut = hcclExecTimeout;
    HcclResult ret = SalStrToDouble(execTimeOutStr, execTimeOut);
    bool flag = false;
    std::string inputValue = (execTimeOutStr ? execTimeOutStr : "NULL");
    if (deviceType == DevType::DEV_TYPE_910_93 || deviceType == DevType::DEV_TYPE_910B) {
        // 910B和910_93算子超时时间范围0s-2147483647s,其中0代表永不超时
        flag = (ret != HCCL_SUCCESS || (execTimeOut < 0) || (execTimeOut > HCCL_EXEC_TIME_OUT_S_910_93));
        RPT_ENV_ERR(
            flag, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
            std::vector<std::string>(
                {inputValue, "HCCL_EXEC_TIMEOUT",
                 "a number greater than or equal to 0s and less "
                 "than or equal to 2147483647s"}));
        CHK_PRT_RET(
            flag,
            HCCL_ERROR(
                "[%s][%s]ExecTimeOut[%s]s is invalid. expect: [0, %d]", LOG_KEYWORDS_INIT_GROUP.c_str(),
                LOG_KEYWORDS_ENV_CONFIG.c_str(), execTimeOutStr, HCCL_EXEC_TIME_OUT_S_910_93),
            HCCL_E_PARA);
    } else {
        // 非910B和910_93算子超时时间范围1s-17340s
        flag = (ret != HCCL_SUCCESS || (execTimeOut <= 0) || (execTimeOut > HCCL_EXEC_TIME_OUT_S));
        RPT_ENV_ERR(
            flag, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
            std::vector<std::string>(
                {inputValue, "HCCL_EXEC_TIMEOUT",
                 "a number greater than or equal to 1s and less than or equal to 17340s"}));
        CHK_PRT_RET(
            flag,
            HCCL_ERROR(
                "[%s][%s]ExecTimeOut[%s]s is invalid. expect: [1, %d]", LOG_KEYWORDS_INIT_GROUP.c_str(),
                LOG_KEYWORDS_ENV_CONFIG.c_str(), execTimeOutStr, HCCL_EXEC_TIME_OUT_S),
            HCCL_E_PARA);
        s32 intPart = static_cast<s32>(execTimeOut / HCCL_INTEVAL_EXEC_TIME_OUT_S);
        intPart = (intPart == 0) ? 1 : intPart;
        execTimeOut = intPart * HCCL_INTEVAL_EXEC_TIME_OUT_S;
    }
    g_externalInput.execTimeOut = execTimeOut;
    g_externalInput.execTimeOutSet = execTimeOutSet;
    if (execTimeOutSet == HcclExecTimeoutSet::HCCL_EXEC_TIMEOUT_SET_BY_ENV) {
        HCCL_RUN_INFO("[HCCL_ENV] HCCL_EXEC_TIMEOUT set by environment to [%.2f]s", execTimeOut);
    } else {
        HCCL_RUN_INFO("[HCCL_ENV] HCCL_EXEC_TIMEOUT set by GE option to [%.2f]s", execTimeOut);
    }
    return HCCL_SUCCESS;
}

bool CheckEnvLen(const char* envStr, u32 envMaxLen)
{
    // 校验环境变量长度
    u32 envLen = strnlen(envStr, envMaxLen + 1);
    if (envLen == (envMaxLen + 1)) {
        HCCL_ERROR(
            "[Check][EnvLen]errNo[0x%016llx] env len is invalid, len is %u", HCCL_ERROR_CODE(HCCL_E_PARA), envLen);
        return false;
    }
    return true;
}
HcclResult SetMasterInfo(
    const string& masterIp, const string& masterPort, const string& masterDeviceId, const string& rankSize,
    const string& rankIp)
{
    HcclResult ret = HCCL_SUCCESS;
    ret = g_externalInput.masterInfo.serverIp.SetReadableAddress(masterIp);
    if (ret != HCCL_SUCCESS) {
        RPT_ENV_ERR(
            ret, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
            std::vector<std::string>({masterIp, "CM_CHIEF_IP", "a valid IPv4/IPv6 address string"}));
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] %s errorno[%d]", LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(),
            HCOM_ERROR_CODE(ret), "Invalid masterIp address", ret);
        return HCCL_E_PARA;
    }

    ret = IsAllDigit(masterPort.c_str());
    ret = (ret == HCCL_SUCCESS) ? SalStrToULong(masterPort, HCCL_BASE_DECIMAL, g_externalInput.masterInfo.port) : ret;
    if (ret != HCCL_SUCCESS || g_externalInput.masterInfo.port > HOST_PORT_MAX) {
        RPT_ENV_ERR(
            HCCL_E_PARA, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
            std::vector<std::string>({masterPort, "CM_CHIEF_PORT", "a unsigned number less than the max port num"}));
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] option masterPort[CM_CHIEF_PORT] error, %s errorno[%d]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), HCOM_ERROR_CODE(ret), "out of range",
            ret);
        return HCCL_E_PARA;
    }

    ret = IsAllDigit(masterDeviceId.c_str());
    ret = (ret == HCCL_SUCCESS) ?
              SalStrToULong(masterDeviceId, HCCL_BASE_DECIMAL, g_externalInput.masterInfo.serverDeviceId) :
              ret;
    u32 maxDeviceNum;
    CHK_RET(GetMaxDevNum(maxDeviceNum));
    if (ret != HCCL_SUCCESS || g_externalInput.masterInfo.serverDeviceId >= maxDeviceNum) {
        RPT_ENV_ERR(
            HCCL_E_PARA, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
            std::vector<std::string>(
                {masterDeviceId, "CM_CHIEF_DEVICE", "a unsigned number less than the max device num"}));
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] option master device id[CM_CHIEF_DEVICE] error, masterDeviceId[%s]"
            "errorno[%d]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), HCOM_ERROR_CODE(ret),
            masterDeviceId.c_str(), ret);
        return HCCL_E_PARA;
    }

    if (rankIp.size() == 0) {
        g_externalInput.masterInfo.agentIp.clear();
    } else {
        ret = g_externalInput.masterInfo.agentIp.SetReadableAddress(rankIp);
        if (ret != HCCL_SUCCESS) {
            RPT_ENV_ERR(
                HCCL_E_PARA, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
                std::vector<std::string>({rankIp, "CM_WORKER_IP", "an available ip"}));
            HCCL_ERROR(
                "[%s][%s]errNo[0x%016llx] masterIp agent address[CM_WORKER_IP][%s] is invalid. errorno[%d]",
                LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), HCOM_ERROR_CODE(ret), rankIp.c_str(),
                ret);
            return HCCL_E_PARA;
        }
    }

    ret = IsAllDigit(rankSize.c_str());
    ret = (ret == HCCL_SUCCESS) ? SalStrToULong(rankSize, HCCL_BASE_DECIMAL, g_externalInput.masterInfo.rankSize) : ret;
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[Parse][HcclMasterInfo] option rankSize[CM_WORKER_SIZE] error, errNo[0x%016llx] rankSize[%s]",
            HCOM_ERROR_CODE(ret), rankSize.c_str());
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}
HcclResult SetTcpMode(const bool isTcpMode)
{
    g_externalInput.isTcpMode = isTcpMode;
    return HCCL_SUCCESS;
}

u32 GetEsMaxPsTable() { return g_externalInput.esMaxPsTable; }

HcclResult SetEsStreamNum(const u32 streamNum)
{
    g_externalInput.streamNum = streamNum;
    return HCCL_SUCCESS;
}

u32 GetEsStreamNum() { return g_externalInput.streamNum; }

HcclResult SetDeterministic(u8 deterministic)
{
    g_externalInput.hcclDeterministic = deterministic;
    return HCCL_SUCCESS;
}

void SetDumpDebugMode(const bool dumpDebug)
{
    g_externalInput.dumpDebug = dumpDebug;
    return;
}

HcclResult SetFftsSwitch(const bool switchStatus)
{
    g_externalInput.enableFfts = switchStatus;
    return HCCL_SUCCESS;
}

HcclResult ParseRDMATrafficClass()
{
    std::string trafficClassEnv = GET_ENV(MM_ENV_HCCL_RDMA_TC);
    u32 rdmaTrafficClass = HCCL_RDMA_TC_DEFAULT;

    if (trafficClassEnv == "EmptyString") {
        HCCL_RUN_INFO("[HCCL_ENV] HCCL_RDMA_TC set by default to [%u]", rdmaTrafficClass);
        return HCCL_SUCCESS;
    }

    // 校验环境变量长度
    bool isEnvLenValid = CheckEnvLen(trafficClassEnv.c_str(), MAX_LEN_OF_DIGIT_ENV);

    CHK_PRT_RET(
        !isEnvLenValid,
        HCCL_ERROR(
            "[Parse][TrafficClass]errNo[0x%016llx] Invalid HCCL_RDMA_TC env len, len is bigger than "
            "[%u]. errorno[%d]",
            HCCL_ERROR_CODE(HCCL_E_PARA), MAX_LEN_OF_DIGIT_ENV, HCCL_E_PARA),
        HCCL_E_PARA);

    g_externalInput.rdmaTrafficClass = HCCL_RDMA_TC_DEFAULT;
    CHK_RET(IsAllDigit(trafficClassEnv.c_str()));

    HcclResult ret = SalStrToULong(trafficClassEnv.c_str(), HCCL_BASE_DECIMAL, rdmaTrafficClass);
    // 若转换出错或者设置的RDMATrafficClass不在有效范围内，报错
    CHK_PRT_RET(
        (ret != HCCL_SUCCESS || rdmaTrafficClass > HCCL_RDMA_TC_MAX),
        HCCL_ERROR(
            "[Parse][TrafficClass]HCCL_RDMA_TC[%s] is invalid. expect: [%u, %u]", trafficClassEnv.c_str(),
            HCCL_RDMA_TC_MIN, HCCL_RDMA_TC_MAX),
        HCCL_E_PARA);
    // 设置的RDMATrafficClass需要是4的整数倍, 否则报错
    if (rdmaTrafficClass % HCCL_RDMA_TC_BASE != 0) {
        HCCL_ERROR("rdmaTrafficClass[%u] is not a multiple of [%u]", rdmaTrafficClass, HCCL_RDMA_TC_BASE);
        return HCCL_E_PARA;
    }
    g_externalInput.rdmaTrafficClass = rdmaTrafficClass;
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_RDMA_TC set by environment to [%u]", rdmaTrafficClass);
    return HCCL_SUCCESS;
}

HcclResult ParseRDMAServerLevel()
{
    std::string serverLevelEnv = GET_ENV(MM_ENV_HCCL_RDMA_SL);
    u32 rdmaServerLevel = HCCL_RDMA_SL_DEFAULT;

    if (serverLevelEnv == "EmptyString") {
        HCCL_RUN_INFO("[HCCL_ENV] HCCL_RDMA_SL set by default to [%u]", rdmaServerLevel);
        return HCCL_SUCCESS;
    }

    // 校验环境变量长度
    bool isEnvLenValid = CheckEnvLen(serverLevelEnv.c_str(), MAX_LEN_OF_DIGIT_ENV);

    CHK_PRT_RET(
        !isEnvLenValid,
        HCCL_ERROR(
            "[Parse][rdmaServerLevel]errNo[0x%016llx] Invalid HCCL_RDMA_SL env len, len is bigger than "
            "[%u]. errorno[%d]",
            HCCL_ERROR_CODE(HCCL_E_PARA), MAX_LEN_OF_DIGIT_ENV, HCCL_E_PARA),
        HCCL_E_PARA);

    g_externalInput.rdmaServerLevel = HCCL_RDMA_SL_DEFAULT;
    CHK_RET(IsAllDigit(serverLevelEnv.c_str()));

    HcclResult ret = SalStrToULong(serverLevelEnv.c_str(), HCCL_BASE_DECIMAL, rdmaServerLevel);
    // 若转换出错或者设置的RDMAServerLevel不在有效范围内，报错
    CHK_PRT_RET(
        (ret != HCCL_SUCCESS || rdmaServerLevel > HCCL_RDMA_SL_MAX),
        HCCL_ERROR(
            "[Parse][rdmaServerLevel]HCCL_RDMA_SL[%s] is invalid. expect: [%u, %u]", serverLevelEnv.c_str(),
            HCCL_RDMA_SL_MIN, HCCL_RDMA_SL_MAX),
        HCCL_E_PARA);
    g_externalInput.rdmaServerLevel = rdmaServerLevel;
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_RDMA_SL set by environment to [%u]", rdmaServerLevel);
    return HCCL_SUCCESS;
}

HcclResult ParseRDMATimeOut(std::pair<u32, u32>& rdmaTimeOutRange)
{
    u32 rdmaTimeOutMax;
#ifndef HCCD
    if (!IsGeneralServer()) {
        DevType deviceType = DevType::DEV_TYPE_COUNT;
        CHK_RET(hrtGetDeviceType(deviceType));
        rdmaTimeOutMax = (deviceType == DevType::DEV_TYPE_910_93 || deviceType == DevType::DEV_TYPE_910B) ?
                             HCCL_RDMA_TIMEOUT_MAX_910_93 :
                             HCCL_RDMA_TIMEOUT_MAX;
    } else {
        rdmaTimeOutMax = HCCL_RDMA_TIMEOUT_MAX;
    }
#else
    rdmaTimeOutMax = HCCL_RDMA_TIMEOUT_MAX;
#endif
    rdmaTimeOutRange.first = HCCL_RDMA_TIMEOUT_MIN;
    rdmaTimeOutRange.second = rdmaTimeOutMax;
    std::string timeOutEnv = GET_ENV(MM_ENV_HCCL_RDMA_TIMEOUT);
    u32 rdmaTimeOut = HCCL_RDMA_TIMEOUT_DEFAULT;
    if (timeOutEnv == "EmptyString") {
        HCCL_RUN_INFO("[HCCL_ENV] HCCL_RDMA_TIMEOUT set by default to [%u]", rdmaTimeOut);
        return HCCL_SUCCESS;
    }

    // 校验环境变量长度
    bool isEnvLenValid = CheckEnvLen(timeOutEnv.c_str(), MAX_LEN_OF_DIGIT_ENV);

    CHK_PRT_RET(
        !isEnvLenValid,
        HCCL_ERROR(
            "[Parse][TrafficClass]errNo[0x%016llx] Invalid HCCL_RDMA_TIMEOUT env len, len is bigger than "
            "[%u]. errorno[%d]",
            HCCL_ERROR_CODE(HCCL_E_PARA), MAX_LEN_OF_DIGIT_ENV, HCCL_E_PARA),
        HCCL_E_PARA);

    g_externalInput.rdmaTimeOut = HCCL_RDMA_TIMEOUT_DEFAULT;
    CHK_RET(IsAllDigit(timeOutEnv.c_str()));

    HcclResult ret = SalStrToULong(timeOutEnv.c_str(), HCCL_BASE_DECIMAL, rdmaTimeOut);
    // 若转换出错或者设置的RDMATrafficClass不在有效范围内，报错
    CHK_PRT_RET(
        (ret != HCCL_SUCCESS || rdmaTimeOut < HCCL_RDMA_TIMEOUT_MIN || rdmaTimeOut > rdmaTimeOutMax),
        HCCL_ERROR(
            "[Parse][RdmaTimeout]HCCL_RDMA_TIMEOUT[%s] is invalid. expect: [%u, %u]", timeOutEnv.c_str(),
            HCCL_RDMA_TIMEOUT_MIN, rdmaTimeOutMax),
        HCCL_E_PARA);

    g_externalInput.rdmaTimeOut = rdmaTimeOut;
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_RDMA_TIMEOUT set by environment to [%u]", rdmaTimeOut);
    return HCCL_SUCCESS;
}

HcclResult ParseRDMARetryCnt()
{
    std::string retryCntEnv = GET_ENV(MM_ENV_HCCL_RDMA_RETRY_CNT);
    u32 rdmaRetryCnt = HCCL_RDMA_RETRY_CNT_DEFAULT;
    if (retryCntEnv == "EmptyString") {
        HCCL_RUN_INFO("[HCCL_ENV] HCCL_RDMA_RETRY_CNT set by default to [%u]", rdmaRetryCnt);
        return HCCL_SUCCESS;
    }

    // 校验环境变量长度
    bool isEnvLenValid = CheckEnvLen(retryCntEnv.c_str(), MAX_LEN_OF_DIGIT_ENV);

    CHK_PRT_RET(
        !isEnvLenValid,
        HCCL_ERROR(
            "[Parse][rdmaRetryCnt]errNo[0x%016llx] Invalid HCCL_RDMA_RETRY_CNT env len, len is bigger than "
            "[%u]. errorno[%d]",
            HCCL_ERROR_CODE(HCCL_E_PARA), MAX_LEN_OF_DIGIT_ENV, HCCL_E_PARA),
        HCCL_E_PARA);

    g_externalInput.rdmaRetryCnt = HCCL_RDMA_RETRY_CNT_DEFAULT;
    CHK_RET(IsAllDigit(retryCntEnv.c_str()));

    HcclResult ret = SalStrToULong(retryCntEnv.c_str(), HCCL_BASE_DECIMAL, rdmaRetryCnt);
    // 若转换出错或者设置的RDMAServerLevel不在有效范围内，报错
    CHK_PRT_RET(
        (ret != HCCL_SUCCESS || rdmaRetryCnt < HCCL_RDMA_RETRY_CNT_MIN || rdmaRetryCnt > HCCL_RDMA_RETRY_CNT_MAX),
        HCCL_ERROR(
            "[Parse][rdmaRetryCnt]HCCL_RDMA_RETRY_CNT[%s] is invalid. expect: [%u, %u]", retryCntEnv.c_str(),
            HCCL_RDMA_RETRY_CNT_MIN, HCCL_RDMA_RETRY_CNT_MAX),
        HCCL_E_PARA);
    g_externalInput.rdmaRetryCnt = rdmaRetryCnt;
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_RDMA_RETRY_CNT set by environment to [%u]", rdmaRetryCnt);
    return HCCL_SUCCESS;
}
#if !defined(CCL_KERNEL_AICPU) && !defined(HCCD)
namespace {
#define PKG_VERSION_MAX_SIZE 128
constexpr auto kVersionInfoKey = "Version=";
constexpr auto kRelPathInfo = "/share/info/";
constexpr auto kInfoFileName = "/version.info";
std::string Trim(const std::string& str)
{
    const size_t first = str.find_first_not_of(" \t\r\n");
    if (std::string::npos == first) {
        return "";
    }
    const size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

HcclResult GetVersionStringInternal(
    const std::string& fullPath, const char_t* pkgName, std::string& versionOut, bool isSilent = false)
{
    std::ifstream ifs(fullPath);
    if (!ifs.is_open()) {
        if (isSilent) {
            HCCL_WARNING("Version file not found at [%s] (Silent check, will retry alternative).", fullPath.c_str());
        } else {
            HCCL_WARNING(
                "Version file not found at [%s]. Please check if package name [%s] is correct and package is "
                "installed.",
                fullPath.c_str(), pkgName);
        }
        return HCCL_E_OPEN_FILE_FAILURE;
    }
    std::string line;
    bool found = false;
    const size_t keyLen = std::strlen(kVersionInfoKey);
    while (std::getline(ifs, line)) {
        const size_t pos = line.find(kVersionInfoKey);
        if (pos != std::string::npos) {
            versionOut = Trim(line.substr(pos + keyLen));
            found = true;
            break;
        }
    }
    ifs.close();
    if (!found || versionOut.empty()) {
        HCCL_WARNING("Keyword [%s] not found in file [%s].", kVersionInfoKey, fullPath.c_str());
        return HCCL_E_NOT_FOUND;
    }
    return HCCL_SUCCESS;
}

std::string GetLocalRealPath(const std::string& path)
{
    if (path.empty()) {
        return "";
    }
    char resolvedPath[PATH_MAX] = {0};
    if (realpath(path.c_str(), resolvedPath) == nullptr) {
        HCCL_WARNING("Failed to get real path for [%s], errno:%d", path.c_str(), errno);
        return "";
    }
    return std::string(resolvedPath);
}

HcclResult GetVersionByPkgName(const std::string& targetPkgName, std::string& versionContent, bool isSilent)
{
    const char* pathEnv = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ASCEND_HOME_PATH, pathEnv);
    if (pathEnv == nullptr) {
        HCCL_WARNING("Can not get env [%s]. Please check if ASCEND_HOME_PATH is set.", "ASCEND_HOME_PATH");
        return HCCL_E_OPEN_FILE_FAILURE;
    }
    std::string homePath(pathEnv);
    homePath = GetLocalRealPath(homePath);
    if (homePath.empty()) {
        HCCL_WARNING("ASCEND_HOME_PATH [%s] does not exist.", homePath.c_str());
        return HCCL_E_OPEN_FILE_FAILURE;
    }
    const std::string fullPath = homePath + kRelPathInfo + targetPkgName + kInfoFileName;
    return GetVersionStringInternal(fullPath, targetPkgName.c_str(), versionContent, isSilent);
}
HcclResult GetPkgVersionContent(const char* pkgName, std::string& versionContent)
{
    const std::string originPkgName(pkgName);
    std::string altPkgName = originPkgName;
    bool hasAlternative = false;

    if (originPkgName.find('-') != std::string::npos) {
        std::replace(altPkgName.begin(), altPkgName.end(), '-', '_');
        hasAlternative = true;
    } else if (originPkgName.find('_') != std::string::npos) {
        std::replace(altPkgName.begin(), altPkgName.end(), '_', '-');
        hasAlternative = true;
    }
    const bool isSilent = hasAlternative;
    HcclResult ret = GetVersionByPkgName(originPkgName, versionContent, isSilent);
    if (ret == HCCL_SUCCESS) {
        return ret;
    }
    if (hasAlternative) {
        HCCL_INFO("Pkg [%s] not found, trying alternative name [%s]...", originPkgName.c_str(), altPkgName.c_str());

        ret = GetVersionByPkgName(altPkgName, versionContent, false);
        if (ret == HCCL_SUCCESS) {
            HCCL_INFO("Found version info using alternative name [%s].", altPkgName.c_str());
            return HCCL_SUCCESS;
        }
    }
    return ret;
}

HcclResult GetPkgVersionStr(char* pkgName, char* versionStr)
{
    HCCL_INFO("start to execute GetPkgVersionStr.");
    CHK_PTR_NULL(versionStr);
    CHK_PTR_NULL(pkgName);
    std::string verInfo;
    const HcclResult ret = GetPkgVersionContent(pkgName, verInfo);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    if (strcpy_s(versionStr, PKG_VERSION_MAX_SIZE, verInfo.c_str()) != EOK) {
        HCCL_WARNING(
            "Copy string failed. Dest buffer size is [%d], source len is [%lu].", PKG_VERSION_MAX_SIZE,
            verInfo.length());
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO("GetPkgVersionStr success. Pkg:[%s], Ver:[%s]", pkgName, versionStr);
    return ret;
}

} // namespace
#endif

HcclResult ParseCannVersion()
{
#if !defined(CCL_KERNEL_AICPU) && !defined(HCCD)
    char hcommPkgName[] = "hcomm";
    char hcclPkgName[] = "hccl";
    constexpr u32 HCCL_VERSION_STR_MAX_LEN = 128;
    std::vector<char> hcommVersion(HCCL_VERSION_STR_MAX_LEN, 0);
    std::vector<char> hcclVersion(HCCL_VERSION_STR_MAX_LEN, 0);
    HcclResult versionRet = GetPkgVersionStr(hcommPkgName, hcommVersion.data());
    CHK_PRT_RET(
        versionRet != HCCL_SUCCESS,
        HCCL_WARNING("[Parse][CannVersion]failed to get hcomm version, ret[%d]", versionRet), HCCL_E_NOT_FOUND);
    versionRet = GetPkgVersionStr(hcclPkgName, hcclVersion.data());
    if (versionRet != HCCL_SUCCESS) {
        HCCL_WARNING("[Parse][CannVersion]failed to get hccl version, ret[%d]", versionRet);
        hcclVersion.assign(HCCL_VERSION_STR_MAX_LEN, 0);
    }
    const size_t hlen = strnlen(hcommVersion.data(), HCCL_VERSION_STR_MAX_LEN);
    const size_t clen = strnlen(hcclVersion.data(), HCCL_VERSION_STR_MAX_LEN);
    if (hlen == 0 || hlen == HCCL_VERSION_STR_MAX_LEN) {
        HCCL_WARNING("[Parse][CannVersion]hcomm version is empty", std::string(hcommVersion.data(), hlen).c_str());
        return HCCL_E_NOT_FOUND;
    }
    g_externalInput.cannVersion = std::string(hcommVersion.data());
    if (clen != 0 && clen != HCCL_VERSION_STR_MAX_LEN) {
        g_externalInput.cannVersion += "_" + std::string(hcclVersion.data());
        HCCL_RUN_INFO(
            "[Parse][CannVersion]success, hcomm version is %s, hccl version is %s ",
            std::string(hcommVersion.data()).c_str(), std::string(hcclVersion.data()).c_str());
    } else {
        HCCL_RUN_INFO(
            "[Parse][CannVersion]success, hcomm version is %s, but hccl package not found.",
            std::string(hcommVersion.data()).c_str());
    }
    return HcclResult::HCCL_SUCCESS;
#else
    HCCL_WARNING("[ParseCannVersion]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult ParseCclBufferSize()
{
    std::string hcclBufferSize = GET_ENV(MM_ENV_HCCL_BUFFSIZE);
    u32 cclBufferSize = HCCL_CCL_COMM_DEFAULT_BUFFER_SIZE;
    if (hcclBufferSize == "EmptyString") {
        HCCL_RUN_INFO("[HCCL_ENV] HCCL_BUFFSIZE set by default to [%u]M", cclBufferSize);
        return HCCL_SUCCESS;
    }

    // 校验环境变量长度
    bool isEnvLenValid = CheckEnvLen(hcclBufferSize.c_str(), MAX_LEN_OF_DIGIT_ENV);

    CHK_PRT_RET(
        !isEnvLenValid,
        HCCL_ERROR(
            "[Parse][CclBufferSize]errNo[0x%016llx] Invalid HCCL_BUFFSIZE env len, len is bigger than "
            "[%u]. errorno[%d]",
            HCCL_ERROR_CODE(HCCL_E_PARA), MAX_LEN_OF_DIGIT_ENV, HCCL_E_PARA),
        HCCL_E_PARA);

    u64 cclBufFixedCalcSize = HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE;
    CHK_RET(IsAllDigit(hcclBufferSize.c_str()));

    HcclResult ret = SalStrToULong(hcclBufferSize.c_str(), HCCL_BASE_DECIMAL, cclBufferSize);
    // 若转换出错或者设置的CclBufferSize不在有效范围内，报错
    CHK_PRT_RET(
        (ret != HCCL_SUCCESS || cclBufferSize < HCCL_CCL_COMM_BUFFER_MIN),
        HCCL_ERROR(
            "[Parse][CclBufferSize]external input CclBufferSize[%uM] should be greater than %uM", cclBufferSize,
            HCCL_CCL_COMM_BUFFER_MIN),
        HCCL_E_PARA);
    g_externalInput.cclBufferSize = static_cast<u64>(cclBufferSize * cclBufFixedCalcSize);
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_BUFFSIZE set by environment to [%u]M", cclBufferSize);
    return HCCL_SUCCESS;
}

void SetIfProfile(bool ifProfile) { g_ifProf = ifProfile; }

const bool& GetIfProfile() { return g_ifProf; }

void SetProfConfig(u64 profConfig)
{
    g_externalInput.profConfig = profConfig;
    HCCL_INFO("[%s]Set profConfig[%x]", __func__, profConfig);
}

HcclResult ParseTaskExceptionSwitch()
{
    // task_exception_handler调测开关，默认关闭 (0)
    std::string taskExceptionSwitchEnv = GET_ENV(MM_ENV_HCCL_DIAGNOSE_ENABLE);
    if (taskExceptionSwitchEnv == "EmptyString") {
        HCCL_RUN_INFO("[HCCL_ENV] HCCL_DIAGNOSE_ENABLE set by default to [0]");
        return HCCL_SUCCESS;
    }
    u32 taskExceptionSwitchConfig = 0;
    bool isEnvLenValid = CheckEnvLen(taskExceptionSwitchEnv.c_str(), MAX_LEN_OF_DIGIT_ENV);
    CHK_PRT_RET(
        !isEnvLenValid,
        HCCL_ERROR(
            "[Parse][TaskExceptionSwitch]errNo[0x%016llx] Invalid"
            " HCCL_DIAGNOSE_ENABLE env len, len is bigger than [%u]. errorno[%d]",
            HCCL_ERROR_CODE(HCCL_E_PARA), MAX_LEN_OF_DIGIT_ENV, HCCL_E_PARA),
        HCCL_E_PARA);
    CHK_RET(IsAllDigit(taskExceptionSwitchEnv.c_str()));
    CHK_RET(SalStrToULong(taskExceptionSwitchEnv.c_str(), HCCL_BASE_DECIMAL, taskExceptionSwitchConfig));
    if ((taskExceptionSwitchConfig != 0) && (taskExceptionSwitchConfig != 1)) {
        HCCL_ERROR(
            "[Get][TaskExceptionSwitch]environmental digit variable error, taskExceptionSwitchConfig[%u]",
            taskExceptionSwitchConfig);
        return HCCL_E_PARA;
    }
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_DIAGNOSE_ENABLE set by environment to [%u]", taskExceptionSwitchConfig);
    g_externalInput.taskExceptionSwitch = taskExceptionSwitchConfig;
    return HCCL_SUCCESS;
}

HcclResult ParseRdmaQpsPerConnection()
{
    g_externalInput.qpsPerConnection = HCCL_QPS_PER_CONNECTION_DEFAULT;
    std::string rdmaQpsPerConnectionEnv = GET_ENV(MM_ENV_HCCL_RDMA_QPS_PER_CONNECTION);
    if (rdmaQpsPerConnectionEnv == "EmptyString") {
        HCCL_RUN_INFO("[HCCL_ENV] HCCL_RDMA_QPS_PER_CONNECTION is set to default value [1]");
        return HCCL_SUCCESS;
    }
    // 校验环境变量长度
    bool isEnvLenValid = CheckEnvLen(rdmaQpsPerConnectionEnv.c_str(), MAX_LEN_OF_DIGIT_ENV);
    CHK_PRT_RET(
        !isEnvLenValid,
        HCCL_ERROR(
            "[Parse][RdmaQpsPerConnectionEnv]errNo[0x%016llx] Invalid RdmaQpsPerConnectionEnv env len,"
            "len is bigger than [%u]. errorno[%d]",
            HCCL_ERROR_CODE(HCCL_E_PARA), MAX_LEN_OF_DIGIT_ENV, HCCL_E_PARA),
        HCCL_E_PARA);

    u32 qpsPerConnection = HCCL_QPS_PER_CONNECTION_DEFAULT;
    CHK_RET(IsAllDigit(rdmaQpsPerConnectionEnv.c_str()));
    HcclResult ret = SalStrToULong(rdmaQpsPerConnectionEnv, HCCL_BASE_DECIMAL, qpsPerConnection);
    if (ret != HCCL_SUCCESS || (qpsPerConnection < HCCL_QPS_PER_CONNECTION_DEFAULT)
        || (qpsPerConnection > HCCL_QPS_PER_CONNECTION_MAX)) {
        HCCL_ERROR(
            "[Parse][RdmaQpsPerConnectionEnv]environmental variable HCCL_RDMA_QPS_PER_CONNECTION error,"
            "errNo[0x%016llx] qpsPerConnection[%s] Range[%u, %u] Recommended Range[1, 8]",
            HCOM_ERROR_CODE(ret), rdmaQpsPerConnectionEnv.c_str(), HCCL_QPS_PER_CONNECTION_DEFAULT,
            HCCL_QPS_PER_CONNECTION_MAX);
        g_externalInput.qpsPerConnection = HCCL_QPS_PER_CONNECTION_DEFAULT;
        return HCCL_E_PARA;
    }
    g_externalInput.qpsPerConnection = qpsPerConnection;
    HCCL_RUN_INFO("[HCCL_ENV] environmental variable HCCL_RDMA_QPS_PER_CONNECTION is set to [%d]", qpsPerConnection);
    return HCCL_SUCCESS;
}

HcclResult ParseMultiQpThreshold()
{
    g_externalInput.multiQpThreshold = HCCL_MULTI_QP_THRESHOLD_DEFAULT;
    std::string strMultiQpThresholdEnv = GET_ENV(MM_ENV_HCCL_MULTI_QP_THRESHOLD);
    if (strMultiQpThresholdEnv == "EmptyString") {
        HCCL_RUN_INFO(
            "[HCCL_ENV] HCCL_MULTI_QP_THRESHOLD is set to default value [%u]KB", HCCL_MULTI_QP_THRESHOLD_DEFAULT);
        return HCCL_SUCCESS;
    }
    // 校验环境变量长度
    bool isEnvLenValid = CheckEnvLen(strMultiQpThresholdEnv.c_str(), MAX_LEN_OF_DIGIT_ENV);
    CHK_PRT_RET(
        !isEnvLenValid,
        HCCL_ERROR(
            "[Parse][MultiQpThresholdEnv]errNo[0x%016llx] Invalid MultiQpThresholdEnv env len,"
            "len is bigger than [%u]. errorno[%d]",
            HCCL_ERROR_CODE(HCCL_E_PARA), MAX_LEN_OF_DIGIT_ENV, HCCL_E_PARA),
        HCCL_E_PARA);

    u32 multiQpThreshold = HCCL_MULTI_QP_THRESHOLD_DEFAULT;
    CHK_RET(IsAllDigit(strMultiQpThresholdEnv.c_str()));
    HcclResult ret = SalStrToULong(strMultiQpThresholdEnv, HCCL_BASE_DECIMAL, multiQpThreshold);
    if (ret != HCCL_SUCCESS || multiQpThreshold == 0 || multiQpThreshold > HCCL_MULTI_QP_THRESHOLD_MAX) {
        HCCL_ERROR(
            "[Parse][MultiQpThresholdEnv]environmental variable HCCL_MULTI_QP_THRESHOLD error,"
            "errNo[0x%016llx] multiQpThreshold[%s] Range[1, %u]",
            HCOM_ERROR_CODE(ret), strMultiQpThresholdEnv.c_str(), HCCL_MULTI_QP_THRESHOLD_MAX);
        g_externalInput.multiQpThreshold = HCCL_MULTI_QP_THRESHOLD_DEFAULT;
        return HCCL_E_PARA;
    }
    g_externalInput.multiQpThreshold = multiQpThreshold;
    HCCL_RUN_INFO("[HCCL_ENV] environmental variable HCCL_MULTI_QP_THRESHOLD is set to [%d]KB", multiQpThreshold);
    return HCCL_SUCCESS;
}

HcclResult ParseEntryLogEnable()
{
    std::string enableEntryLogEnv = GET_ENV(MM_ENV_HCCL_ENTRY_LOG_ENABLE);
    if (enableEntryLogEnv == "EmptyString") {
        HCCL_RUN_INFO("[HCCL_ENV] HCCL_ENTRY_LOG_ENABLE set by default to [0]");
        return HCCL_SUCCESS;
    }
    if (enableEntryLogEnv != "0" && enableEntryLogEnv != "1") {
        HCCL_ERROR(
            "[Parser][EntryLogEnable]environmental variable HCCL_ENTRY_LOG_ENABLE [%s] is invalid, set by "
            "default to [0]",
            enableEntryLogEnv.c_str());
        return HCCL_E_PARA;
    }
    g_externalInput.enableEntryLog = false;
    if (enableEntryLogEnv == "1") {
        g_externalInput.enableEntryLog = true;
    }
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_ENTRY_LOG_ENABLE set by environment to [%u]", g_externalInput.enableEntryLog);
    return HCCL_SUCCESS;
}

HcclResult ParseInterLinkType()
{
    std::string interHccsDisableEnv = GET_ENV(MM_ENV_HCCL_INTER_HCCS_DISABLE);
    if (interHccsDisableEnv == "EmptyString") {
        HCCL_RUN_INFO(
            "[HCCL_ENV] HCCL_INTER_HCCS_DISABLE is not set, default value is %s.",
            g_externalInput.interHccsDisable ? "TRUE" : "FALSE");
        return HCCL_SUCCESS;
    }
    std::transform(interHccsDisableEnv.begin(), interHccsDisableEnv.end(), interHccsDisableEnv.begin(), ::toupper);
    if ("TRUE" == interHccsDisableEnv) {
        g_externalInput.interHccsDisable = true;
    } else if ("FALSE" == interHccsDisableEnv) {
        g_externalInput.interHccsDisable = false;
    } else {
        HCCL_ERROR("HCCL_INTER_HCCS_DISABLE %s is invalid, expect true or false.", interHccsDisableEnv.c_str());
        return HCCL_E_PARA;
    }
    HCCL_RUN_INFO(
        "[HCCL_ENV] environmental variable HCCL_INTER_HCCS_DISABLE is set to [%s], interHccsDisable[%d]",
        interHccsDisableEnv.c_str(), g_externalInput.interHccsDisable);
    return HCCL_SUCCESS;
}

HcclResult ParseOpExpansion()
{
    std::string opExpansionModeEnv = GET_ENV(MM_ENV_HCCL_OP_EXPANSION_MODE);
    g_externalInput.aicpuUnfold = false;
    g_externalInput.aivMode = false;
    g_externalInput.aicpuCacheEnable = 0; // aicpu模式不使能时, aicpu cache也不使能 (即使使能也不生效)
    if (IsGeneralServer()) {
        HCCL_RUN_INFO(
            "[HCCL_ENV] HCCL_OP_EXPANSION_MODE is not set, aicpuUnfold is [%u], aivMode is [%u], aicpuCacheEnable is "
            "[%u]",
            g_externalInput.aicpuUnfold, g_externalInput.aivMode, g_externalInput.aicpuCacheEnable);
        return HCCL_SUCCESS;
    }

#ifndef HCCD
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    // 910_93默认打开AICPU展开
    if (deviceType == DevType::DEV_TYPE_910_93) {
        g_externalInput.aicpuUnfold = true;
        g_externalInput.aicpuCacheEnable = 1; // aicpu cache默认使能
    }
    if (opExpansionModeEnv == "EmptyString") {
        HCCL_RUN_INFO(
            "[HCCL_ENV] HCCL_OP_EXPANSION_MODE is not set, aicpuUnfold is [%u], aivMode is [%u], aicpuCacheEnable is "
            "[%u]",
            g_externalInput.aicpuUnfold, g_externalInput.aivMode, g_externalInput.aicpuCacheEnable);
        return HCCL_SUCCESS;
    }
    if (opExpansionModeEnv == "AI_CPU") {
        if (deviceType == DevType::DEV_TYPE_910) {
            HCCL_WARNING("910 do not support AICPU unfold.");
        } else {
            g_externalInput.aicpuUnfold = true;
            g_externalInput.aicpuCacheEnable = 1; // aicpu cache默认使能
        }
    } else if (opExpansionModeEnv == "AICPU_CacheDisable") {
        if (deviceType == DevType::DEV_TYPE_910) {
            HCCL_WARNING("910 do not support AICPU unfold.");
        } else {
            g_externalInput.aicpuUnfold = true;
            g_externalInput.aicpuCacheEnable = 0; // Disable aicpu cache
        }
    } else if (opExpansionModeEnv == "AIV") {
        g_externalInput.aivMode = true;
    } else if (opExpansionModeEnv == "HOST") {
        g_externalInput.aivMode = false;
        g_externalInput.aicpuUnfold = false;
        g_externalInput.aicpuCacheEnable = 0;
    } else if (opExpansionModeEnv == "HOST_TS") {
        if (deviceType == DevType::DEV_TYPE_910B) {
            g_externalInput.enableFfts = false;
        } else {
            HCCL_WARNING("deviceType[%u] do not support HOST_TS", deviceType);
        }
    } else {
        HCCL_ERROR(
            "HCCL_OP_EXPANSION_MODE is set to [%s], which is incorrect. Please check", opExpansionModeEnv.c_str());
        return HCCL_E_PARA;
    }
#endif
    HCCL_RUN_INFO(
        "[HCCL_ENV] environmental variable HCCL_OP_EXPANSION_MODE is [%s], aicpuUnfold[%u], aivMode[%u], "
        "enableFfts[%u], aicpuCacheEnable[%u]",
        opExpansionModeEnv.c_str(), g_externalInput.aicpuUnfold, g_externalInput.aivMode, g_externalInput.enableFfts,
        g_externalInput.aicpuCacheEnable);
    return HCCL_SUCCESS;
}

HcclResult SplitHcclRetryEnable(
    [[maybe_unused]] const std::string& retryConfig, [[maybe_unused]] std::vector<std::string>& retryEnables)
{
#ifndef CCL_KERNEL_AICPU
    std::string remainRetryConfig;
    std::size_t found = retryConfig.find(",");
    if ((found == 0) || (found == (retryConfig.length() - 1))) {
        HCCL_ERROR("[SplitHcclRetryEnable] algo config is invalid.");
        return HCCL_E_PARA;
    } else if (found != std::string::npos) {
        remainRetryConfig = retryConfig.substr(found + 1);
    } else {
        // 最后一组配置,剩余的字符串为空
    }
    retryEnables.push_back(retryConfig.substr(0, found));

    if (retryEnables.size() > HCCL_RETRY_ENABLE_LEVEL_NUM) {
        HCCL_ERROR(
            "[SplitHcclRetryEnable] retryEnable config is invalid. retryEnable level is more than %u.",
            HCCL_RETRY_ENABLE_LEVEL_NUM);
        return HCCL_E_PARA;
    }
    if (!remainRetryConfig.empty()) {
        CHK_RET(SplitHcclRetryEnable(remainRetryConfig, retryEnables));
    }
#endif
    return HCCL_SUCCESS;
}

HcclResult CollectRetryEnableFromConfig([[maybe_unused]] const std::vector<std::string>& retryEnables)
{
#ifndef CCL_KERNEL_AICPU
    const std::map<std::string, u32> hcclRetryLevelMap
        = {{"L0", HCCL_RETRY_ENABLE_LEVEL_0}, {"L1", HCCL_RETRY_ENABLE_LEVEL_1}, {"L2", HCCL_RETRY_ENABLE_LEVEL_2}};

    std::map<std::string, u32> countHcclRetryLevelMap = {{"L0", 0}, {"L1", 0}, {"L2", 0}};

    const std::map<std::string, bool> hcclRetryEnableMap = {{"0", false}, {"1", true}};
    for (auto retryEnableLevel : retryEnables) {
        u32 level = 0;
        bool retryEnable = false;
        std::size_t found = retryEnableLevel.find(":");
        if ((found == 0) || (found == (retryEnableLevel.length() - 1))) {
            HCCL_ERROR("[CollectRetryEnableFromConfig] Hccl retryEnableLevel is invalid.");
            return HCCL_E_PARA;
        }
        std::string orginalLevel = retryEnableLevel.substr(0, found);
        std::string orginalRetryEnable = retryEnableLevel.substr(found + 1);
        if (orginalLevel == "L0") {
            HCCL_RUN_WARNING("[CollectRetryEnableFromConfig] L0 config does not take effect");
        }
        // 检查是否存在重复配置level
        auto iterCountRetryLevel = countHcclRetryLevelMap.find(orginalLevel);
        if (iterCountRetryLevel == countHcclRetryLevelMap.end()) {
            HCCL_ERROR(
                "[CollectRetryEnableFromConfig] Retry config is invalid, level %s is not supported.",
                orginalLevel.c_str());
            return HCCL_E_PARA;
        }
        if (countHcclRetryLevelMap[orginalLevel] == 1) {
            HCCL_ERROR(
                "[CollectRetryEnableFromConfig] Retry config level[%s] is repeated, expect: L1:0, L2:0",
                orginalLevel.c_str());
            return HCCL_E_PARA;
        }
        countHcclRetryLevelMap[orginalLevel] += 1;
        // 获取level和对应的retryEnable，并赋值给g_externalInput.hcclRetryConfig
        auto iterRetryLevel = hcclRetryLevelMap.find(orginalLevel);
        if (iterRetryLevel == hcclRetryLevelMap.end()) {
            HCCL_ERROR(
                "[CollectRetryEnableFromConfig] Retry config is invalid, level %s is not supported.",
                orginalLevel.c_str());
            return HCCL_E_PARA;
        }
        auto iterRetryEnable = hcclRetryEnableMap.find(orginalRetryEnable);
        if (iterRetryEnable == hcclRetryEnableMap.end()) {
            HCCL_ERROR(
                "[CollectRetryEnableFromConfig] Retry config is invalid, retryEnable %s is not supported.",
                orginalRetryEnable.c_str());
            return HCCL_E_PARA;
        }
        level = iterRetryLevel->second;
        retryEnable = iterRetryEnable->second;
        g_externalInput.hcclRetryConfig[level] = retryEnable;
    }
#endif
    return HCCL_SUCCESS;
}

HcclResult ParseRetryEnable()
{
#ifndef CCL_KERNEL_AICPU
    // 默认都设置成false
    for (u32 level = 0; level < HCCL_RETRY_ENABLE_LEVEL_NUM; ++level) {
        g_externalInput.hcclRetryConfig[level] = false;
    }
    std::string hcclRetryEnable = GET_ENV(MM_ENV_HCCL_OP_RETRY_ENABLE);
    if (hcclRetryEnable == "EmptyString") {
        HCCL_RUN_INFO("[HCCL_ENV][ParseRetryEnable] HCCL_OP_RETRY_ENABLE is not set. The retryEnable of all levels is "
                      "set to false.");
        return HCCL_SUCCESS;
    }
    // 去除空格
    std::string retryConfig = hcclRetryEnable;
    retryConfig.erase(std::remove(retryConfig.begin(), retryConfig.end(), ' '), retryConfig.end());

    if (retryConfig.empty()) {
        HCCL_RUN_INFO(
            "[HCCL_ENV][ParseRetryEnable] Hccl retry config is empty. The retryEnable of all levels is set to false.");
        return HCCL_SUCCESS;
    }

    std::vector<std::string> retryEnables;
    HcclResult ret = SplitHcclRetryEnable(retryConfig, retryEnables);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollectRetryEnableFromConfig] Hccl retry config[%s] is invalid. "
            "expect: L1:0, L2:0",
            retryConfig.c_str()),
        ret);

    CHK_RET(CollectRetryEnableFromConfig(retryEnables));
    HCCL_RUN_INFO(
        "[HCCL_ENV][ParseRetryEnable] HCCL_OP_RETRY_ENABLE set by environment variable to [%s].", retryConfig.c_str());
#endif
    return HCCL_SUCCESS;
}

HcclResult ParseRetryParams()
{
#ifndef CCL_KERNEL_AICPU
    std::string retryParams = GET_ENV(MM_ENV_HCCL_OP_RETRY_PARAMS);
    if (retryParams == "EmptyString") {
        HCCL_RUN_INFO(
            "[HCCL_ENV] HCCL_OP_RETRY_PARAMS is not set, default value MaxCnt is [%u], HoldTime is [%u]ms, "
            "IntervalTime is [%u]ms",
            HCCL_RETRY_MAXCNT_DEFAULT, HCCL_RETRY_HOLD_TIME_DEFAULT, HCCL_RETRY_INTERVAL_DEFAULT);
        return HCCL_SUCCESS;
    }
    u32 maxcnt = 0;
    u32 holdtime = 0;
    u32 intervaltime = 0;
    int ret = 0;
    ret = sscanf_s(retryParams.c_str(), "MaxCnt:%u, HoldTime:%u, IntervalTime:%u", &maxcnt, &holdtime, &intervaltime);
    /* 三个参数全部解析成功，返回值为HCCL_RETRY_PARAM_NUM，否则不等于 */
    if ((ret != HCCL_RETRY_PARAM_NUM) || (maxcnt > HCCL_RETRY_MAXCNT_MAX) || (maxcnt < HCCL_RETRY_MAXCNT_MIN)
        || (holdtime > HCCL_RETRY_HLOD_TIME_MAX) || (intervaltime > HCCL_RETRY_INTERVAL_MAX)) {
        HCCL_ERROR(
            "[Parse][RetryParams]fail, HCCL_OP_RETRY_PARAMS: %s is invalid, format must be: "
            "MaxCnt:cnt,HoldTime:time,IntervalTime:time, cnt range is [1, 10], time range is [0, 60000]ms.",
            retryParams.c_str());
        return HCCL_E_PARA;
    }
    g_externalInput.retryMaxCnt = maxcnt;
    g_externalInput.retryHoldTime = holdtime;
    g_externalInput.retryIntervalTime = intervaltime;

    HCCL_RUN_INFO(
        "[HCCL_ENV] HCCL_OP_RETRY_PARAMS is set, MaxCnt is [%u], HoldTime is [%u]ms, IntervalTime is [%u]ms.", maxcnt,
        holdtime, intervaltime);
#endif
    return HCCL_SUCCESS;
}

HcclResult ParseLogicSuperPodId()
{
    std::string logicSuperPodId = GET_ENV(MM_ENV_HCCL_LOGIC_SUPERPOD_ID);
    if (logicSuperPodId == "EmptyString") {
        HCCL_RUN_INFO(
            "[HCCL_ENV] HCCL_LOGIC_SUPERPOD_ID is not set, default value[%s]", g_externalInput.logicSuperPodId.c_str());
        return HCCL_SUCCESS;
    }

    // 校验环境变量长度
    bool isEnvLenValid = CheckEnvLen(logicSuperPodId.c_str(), MAX_LEN_OF_LOGIC_SUPER_ID);
    CHK_PRT_RET(
        !isEnvLenValid,
        HCCL_ERROR(
            "[Parse][LogicSuperPodId]Invalid HCCL_LOGIC_SUPERPOD_ID env len, len is bigger than [%u].",
            MAX_LEN_OF_LOGIC_SUPER_ID),
        HCCL_E_PARA);

    g_externalInput.logicSuperPodId = logicSuperPodId;
    HCCL_RUN_INFO(
        "[HCCL_ENV] HCCL_LOGIC_SUPERPOD_ID set by environment to [%s]", g_externalInput.logicSuperPodId.c_str());
    return HCCL_SUCCESS;
}

HcclResult SetIncreSaveExecTimeOut(const s32 execTimeout)
{
    if (execTimeout == -1) {
        g_externalInput.increSaveExecTimeOut = execTimeout;
    } else {
        // execTimeOut 为 double 类型（环境变量解析结果），此处取整保存到 s32
        g_externalInput.increSaveExecTimeOut = static_cast<s32>(g_externalInput.execTimeOut);
    }

    HCCL_RUN_INFO(
        "setIncreSaveExecTimeOut execTimeout[%d s] increSaveExecTimeOut[%d s]", execTimeout,
        g_externalInput.increSaveExecTimeOut);
    return HCCL_SUCCESS;
}

HcclResult ParseDebugConfig()
{
    char* env = nullptr; // 环境变量值
    MM_SYS_GET_ENV(MM_ENV_HCCL_DEBUG_CONFIG, env);
    if (env == nullptr) {
        HCCL_RUN_INFO(
            "[HCCL_ENV] HCCL_DEBUG_CONFIG is not set, debugConfig set by default to 0x%llx",
            g_externalInput.debugConfig);
        return HCCL_SUCCESS;
    }

    bool invert = (env[0] == '^');
    g_externalInput.debugConfig
        = invert ? ~0ULL : 0ULL; // 第一个字符是'^', 使用取反模式，用户配置的项关闭，未配置的项打开
    char* configValue = (env[0] == '^') ? env + 1 : env; // 去掉'^'符号
    char* configDup = strdup(configValue);               // 需要使用strdup避免修改字符串常量
    CHK_PTR_NULL(configDup);

    char* left = nullptr;
    char* subConfig = strtok_r(configDup, ",", &left); // 按逗号分割
    while (subConfig != nullptr) {
        u64 mask = 0;
        if (strcasecmp(subConfig, "ALG") == 0) {
            mask = PLF_ALG;
        } else if (strcasecmp(subConfig, "TASK") == 0) {
            mask = PLF_TASK;
        } else if (strcasecmp(subConfig, "RESOURCE") == 0) {
            mask = PLF_RES;
        } else if (strcasecmp(subConfig, "AIV_OPS_EXC") == 0) {
            HCCL_WARNING("HCCL_DEBUG_CONFIG:%s subConfig:%s is deprecated and ignored", env, subConfig);
        } else {
            HCCL_ERROR("HCCL_DEBUG_CONFIG:%s is invalid, subConfig:%s is not supported", env, subConfig);
            free(configDup);
            return HCCL_E_PARA;
        }
        g_externalInput.debugConfig
            = invert ? (g_externalInput.debugConfig & (~mask)) : (g_externalInput.debugConfig | mask);
        subConfig = strtok_r(nullptr, ",", &left);
    }
    free(configDup);
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_DEBUG_CONFIG[%s], set debugConfig[0x%llx]", env, g_externalInput.debugConfig);
    return HCCL_SUCCESS;
}

const u32& GetExternalInputHcclIfBasePort() { return g_externalInput.hcclIfBasePort; }

const u32& GetExternalInputRdmaTrafficClass() { return g_externalInput.rdmaTrafficClass; }

const u32& GetExternalInputRdmaServerLevel() { return g_externalInput.rdmaServerLevel; }

const u32& GetExternalInputRdmaTimeOut() { return g_externalInput.rdmaTimeOut; }

const u32& GetExternalInputRdmaRetryCnt() { return g_externalInput.rdmaRetryCnt; }

const u32& GetExternalInputTaskExceptionSwitch() { return g_externalInput.taskExceptionSwitch; }

const u32& GetExternalInputIntraRoceSwitch() { return g_externalInput.intraRoceSwitch; }

const u32& GetExternalInputHcclEnableWhitelist() { return g_externalInput.enableWhitelist; }

const std::string& GetExternalInputHcclWhiteListFile() { return g_externalInput.hcclWhiteListFile; }

const std::string& GetExternalInputProfilingOption() { return g_externalInput.profilingOption; }

const std::string& GetExternalInputCannVersion() { return g_externalInput.cannVersion; }

const double& GetExternalInputHcclExecTimeOut() { return g_externalInput.execTimeOut; }

const s32& GetExternalInputHcclLinkTimeOut() { return g_externalInput.linkTimeOut; }

const s32& GetExternalInputHcclSocketFamily() { return g_externalInput.hcclSocketFamily; }

const bool& GetExternalInputProfilingMode() { return g_externalInput.profilingMode; }

const bool& GetExternalInputHcclIsTcpMode() { return g_externalInput.isTcpMode; }

const bool& GetExternalInputHcclDumpDebug() { return g_externalInput.dumpDebug; }

const bool& GetExternalInputHcclEnableFfts() { return g_externalInput.enableFfts; }

const u8& GetExternalInputHcclDeterministicV2() { return g_externalInput.hcclDeterministic; }

const bool& GetExternalInputHcclDeterministic()
{
    g_externalInput.isDeterministic = g_externalInput.hcclDeterministic == DETERMINISTIC_ENABLE;
    return g_externalInput.isDeterministic;
}

const bool& GetExternalInputHcclEnablePipline() { return g_externalInput.enablePipeline; }

const bool& GetExternalInputHcclEnableEntryLog() { return g_externalInput.enableEntryLog; }

const bool& GetExternalInputInterHccsDisable() { return g_externalInput.interHccsDisable; }

const u64& GetExternalInputCCLBuffSize() { return g_externalInput.cclBufferSize; }

const HcclExecTimeoutSet& GetExternalInputHcclExecTimeoutSet() { return g_externalInput.execTimeOutSet; }

const hccl::HcclIpAddress& GetExternalInputHcclControlIfIp() { return g_externalInput.hcclControlIfIp; }

const HcclSocketIfName& GetExternalInputHcclSocketIfName() { return g_externalInput.hcclSocketIfName; }

u32 GetExternalInputQpsPerConnection() { return g_externalInput.qpsPerConnection; }

u32 GetExternalInputMultiQpThreshold() { return g_externalInput.multiQpThreshold; }

const ProtocolType& GetExternalInputProtocolType() { return g_externalInput.protocolType; }

const MasterInfo& GetExternalInputMasterInfo() { return g_externalInput.masterInfo; }

const bool& GetExternalInputHcclAicpuUnfold() { return g_externalInput.aicpuUnfold; }

const uint8_t& GetExternalInputAicpuCacheEnable() { return g_externalInput.aicpuCacheEnable; }

const bool& GetExternalInputHcclAivMode() { return g_externalInput.aivMode; }

const bool& GetRemoteIsHdc() { return g_externalInput.remoteIsHdc; }

const bool& GetExternalInputIntraServerRetryEnable()
{
    return g_externalInput.hcclRetryConfig[HCCL_RETRY_ENABLE_LEVEL_0];
}

const bool& GetExternalInputInterServerRetryEnable()
{
    return g_externalInput.hcclRetryConfig[HCCL_RETRY_ENABLE_LEVEL_1];
}

const bool& GetExternalInputInterSuperPodRetryEnable()
{
    return g_externalInput.hcclRetryConfig[HCCL_RETRY_ENABLE_LEVEL_2];
}

const bool& GetExternalInputOpCounter() { return g_externalInput.opCounterEnable; }

const u32& GetExternalInputRetryMaxCnt() { return g_externalInput.retryMaxCnt; }

const u32& GetExternalInputRetryHoldTime() { return g_externalInput.retryHoldTime; }

const u32& GetExternalInputRetryIntervalTime() { return g_externalInput.retryIntervalTime; }

const std::string& GetExternalInputLogicSuperPodId() { return g_externalInput.logicSuperPodId; }

const bool& GetExternalInputRdmaFastPost() { return g_externalInput.rdmaFastPost; }

const std::string& GetExternalInputQpSrcPortConfigPath() { return g_externalInput.multiQpSrcPortConfigPath; }

const s32& GetIncreSaveExecTimeOut() { return g_externalInput.increSaveExecTimeOut; }

const u64& GetProfConfig() { return g_externalInput.profConfig; }

const u64& GetExternalInputDebugConfig() { return g_externalInput.debugConfig; }

void SetExternalInputDebugConfig(u64 value) { g_externalInput.debugConfig = value; }
