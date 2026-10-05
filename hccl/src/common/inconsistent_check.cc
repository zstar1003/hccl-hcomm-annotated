/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "inconsistent_check.h"
#include "hcom.h"
#include "alg_env_config.h"
#include "adapter_error_manager_pub.h"
#include "hccl_res_expt_dl.h"

namespace ops_hccl {
// [中文导读] [AllReduce逐行 S18] 声明NeedInconsistentCheck接口：由能力、配置开关、上下文存在性决定是否登记比较算子参数。
bool NeedInconsistentCheck(HcclComm comm, const OpParam& param)
// [中文导读] [AllReduce逐行 S19] 开始NeedInconsistentCheck的函数体。
{
    // [中文导读] [AllReduce逐行 S20] 交换信息登记能力存在且通信域非空时才考虑一致性校验。
    if (HcommIsSupportHcclCommAddExchangeInfo() && (comm != nullptr)) {
        // 以下场景不校验参数一致性，其余场景均校验：
        // inconsistentCheckSwitch为off
        // inconsistentCheckSwitch为first或空，单算子模式下非首次下发且非增量建链模式，当前以context是否存在作为首次下发判断依据
        // [中文导读] [AllReduce逐行 S24] 声明默认/first策略下可以跳过检查的判断结果。
        bool noCheck
            // [中文导读] [AllReduce逐行 S25] 开关为默认0、OPBASE且算法资源上下文已存在时本次允许跳过检查。
            = (GetInconsistentCheckSwitch() == 0) && (param.opMode == OpMode::OPBASE) && CheckCtxStatus(comm, param);
        // [中文导读] [AllReduce逐行 S26] 声明是否是需要增量建链的算子。
        bool increCreateChannelFlag
            // [中文导读] [AllReduce逐行 S27] BatchSendRecv OPBASE属于增量建链，不能按普通已有上下文规则跳过。
            = (param.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) && (param.opMode == OpMode::OPBASE);
        // [中文导读] [AllReduce逐行 S28] 显式off(-1)或普通非首次OPBASE时不做一致性校验。
        if (GetInconsistentCheckSwitch() == -1 || (noCheck && !increCreateChannelFlag)) {
            // [中文导读] [AllReduce逐行 S29] 返回不需要一致性校验。
            return false;
        // [中文导读] [AllReduce逐行 S30] 上述条件不成立时进入替代分支。
        } else {
            // [中文导读] [AllReduce逐行 S31] 其它支持交换信息的场景需要校验。
            return true;
        // [中文导读] [AllReduce逐行 S32] 结束条件} else。
        }
    // [中文导读] [AllReduce逐行 S33] 结束条件if (HcommIsSupportHcclCommAddExchangeInfo() && (comm != nullptr))。
    }
    // [中文导读] [AllReduce逐行 S34] 没有交换接口能力或空通信域时不做检查。
    return false;
// [中文导读] [AllReduce逐行 S35] 结束NeedInconsistentCheck函数体。
}

// [中文导读] [AllReduce逐行 S37] 声明CheckCtxStatus接口：查资源上下文是否已存在，注意CPU/AIV使用不同的存储engine。
bool CheckCtxStatus(HcclComm comm, const OpParam& param)
// [中文导读] [AllReduce逐行 S38] 开始CheckCtxStatus的函数体。
{
    // [中文导读] [AllReduce逐行 S39] 初始化上下文存在性查询地址。
    void* ctx = nullptr;
    // [中文导读] [AllReduce逐行 S40] 初始化上下文字节长度输出。
    uint64_t size = 0;
    // [中文导读] [AllReduce逐行 S41] 按执行engine选择资源上下文实际存储位置。
    switch (param.engine) {
        // [中文导读] [AllReduce逐行 S42] CPU/DPU执行资源存储在AICPU_TS Device上下文。
        case CommEngine::COMM_ENGINE_CPU:
            // [中文导读] [AllReduce逐行 S43] CPU分支把上下文查询成功与否转换为布尔存在性。
            return (
                // [中文导读] [AllReduce逐行 S44] 按AICPU_TS存储engine查询CPU执行资源上下文。
                HcclEngineCtxGet(comm, param.algTag, CommEngine::COMM_ENGINE_AICPU_TS, &ctx, &size) == HCCL_SUCCESS);
        // [中文导读] [AllReduce逐行 S45] AIV执行资源存储在CPU_TS Host上下文。
        case CommEngine::COMM_ENGINE_AIV:
            // [中文导读] [AllReduce逐行 S46] 查询CPU_TS资源上下文，返回是否存在。
            return (HcclEngineCtxGet(comm, param.algTag, CommEngine::COMM_ENGINE_CPU_TS, &ctx, &size) == HCCL_SUCCESS);
        // [中文导读] [AllReduce逐行 S47] 其它引擎按同名存储engine查询。
        default:
            // [中文导读] [AllReduce逐行 S48] 以EngineCtxGet是否成功返回资源上下文存在性。
            return (HcclEngineCtxGet(comm, param.algTag, param.engine, &ctx, &size) == HCCL_SUCCESS);
    // [中文导读] [AllReduce逐行 S49] 结束代码块。
    }
// [中文导读] [AllReduce逐行 S50] 结束CheckCtxStatus函数体。
}

// [中文导读] [AllReduce逐行 S52] 声明CompareOpExchangeInfos接口：按算法层或CCU kernel通道比较远端算子交换信息。
HcclResult CompareOpExchangeInfos(
    // [中文导读] [AllReduce逐行 S53] 函数参数包含通信域句柄、算子参数、线程/通道/通知资源需求，本行延续接口声明。
    HcclComm comm, const OpParam& param, const AlgResourceRequest& resRequest, const OpExchangeInfo& exchangeInfo)
// [中文导读] [AllReduce逐行 S54] 开始CompareOpExchangeInfos的函数体。
{
    // [中文导读] [AllReduce逐行 S55] 检查用于读取交换信息的通信域非空。
    CHK_PTR_NULL(comm);
    // [中文导读] [AllReduce逐行 S56] 运行时有读取交换信息接口时才比较远端元信息。
    if (HcommIsSupportHcclCommGetExchangeInfo()) {
        // [中文导读] [AllReduce逐行 S57] 非CCU按普通算法层通道请求比较，AICPU主例在此分支。
        if (param.engine != COMM_ENGINE_CCU) {
            // [中文导读] [AllReduce逐行 S58] 遍历每个算法层的通道请求。
            for (u32 level = 0; level < resRequest.channels.size(); level++) {
                // [中文导读] [AllReduce逐行 S59] 与该层每个Peer交换的算子元信息比较。
                CHK_RET(InconsistentCheckParams(comm, exchangeInfo, resRequest.channels[level]));
            // [中文导读] [AllReduce逐行 S60] 结束循环for (u32 level = 0; level < resRequest.channels.size(); level++)。
            }
        // [中文导读] [AllReduce逐行 S61] 上述条件不成立时进入替代分支。
        } else {
            // [中文导读] [AllReduce逐行 S62] CCU改为遍历各kernel请求携带的通道描述。
            for (auto& kernelInfo : resRequest.ccuKernelInfos) {
                // [中文导读] [AllReduce逐行 S63] 比较本个CCU kernel的各Peer元信息。
                CHK_RET(InconsistentCheckParams(comm, exchangeInfo, kernelInfo.channels));
            // [中文导读] [AllReduce逐行 S64] 结束循环for (auto& kernelInfo : resRequest.ccuKernelInfos)。
            }
        // [中文导读] [AllReduce逐行 S65] 结束条件} else。
        }
        // [中文导读] [AllReduce逐行 S66] 输出运行日志，记录CompareOpExchangeInfos当前阶段和相关参数。
        HCCL_RUN_INFO("[CompareOpExchangeInfos] all exchangeInfos checked successfully. algTag[%s]", param.algTag);
    // [中文导读] [AllReduce逐行 S67] 结束条件if (HcommIsSupportHcclCommGetExchangeInfo())。
    }
    // [中文导读] [AllReduce逐行 S68] 按算法层或CCU kernel通道比较远端算子交换信息处理完成，返回成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S69] 结束CompareOpExchangeInfos函数体。
}

// [中文导读] [AllReduce逐行 S71] 声明InconsistentCheckParams接口：逐Peer查询建链交换的元信息，核对CCL大小、root、命令、引擎、归约、dtype、count、group及tag。
HcclResult
// [中文导读] [AllReduce逐行 S72] 函数参数包含通信域句柄，本行延续接口声明。
InconsistentCheckParams(HcclComm comm, const OpExchangeInfo& exchangeInfo, const std::vector<HcclChannelDesc>& channels)
// [中文导读] [AllReduce逐行 S73] 开始InconsistentCheckParams的函数体。
{
    // [中文导读] [AllReduce逐行 S74] 检查逐通道一致性比较的通信域非空。
    CHK_PTR_NULL(comm);
    // [中文导读] [AllReduce逐行 S75] 没有通道的这一层不需要比较对端元信息。
    if (channels.empty()) {
        // [中文导读] [AllReduce逐行 S76] 输出运行日志，记录InconsistentCheckParams当前阶段和相关参数。
        HCCL_INFO("[InconsistentCheckParams] channels is empty.");
        // [中文导读] [AllReduce逐行 S77] 空层成功返回。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S78] 结束条件if (channels.empty())。
    }
    // [中文导读] [AllReduce逐行 S79] 逐一读取每个请求通道的远端rank信息。
    for (auto& channel : channels) {
        // [中文导读] [AllReduce逐行 S80] 清零接收对端算子元信息的结构。
        OpExchangeInfo rmtExchangeInfo{};
        // [中文导读] [AllReduce逐行 S81] 初始化接收的交换数据实际长度为零。
        uint32_t rmtDataLen = 0;
        // [中文导读] [AllReduce逐行 S82] 开始查询该远端rank建链时交换的算子元信息。
        CHK_RET(HcclCommGetExchangeInfo(
            // [中文导读] [AllReduce逐行 S83] 指定预期OpExchangeInfo长度、接收地址及实际接收长度输出。
            comm, channel.remoteRank, sizeof(OpExchangeInfo), reinterpret_cast<void*>(&rmtExchangeInfo), &rmtDataLen));
        // [中文导读] [AllReduce逐行 S84] 接收长度0表示对端没有交换元信息，按源码成功跳过该Peer。
        if (rmtDataLen == 0) {
            // [中文导读] [AllReduce逐行 S85] 输出运行日志，记录InconsistentCheckParams当前阶段和相关参数。
            HCCL_INFO("[InconsistentCheckParams] rmtDataLen is 0. Skip. remoteRank[%u]", channel.remoteRank);
            // [中文导读] [AllReduce逐行 S86] 跳过该Peer，继续下一个通道。
            continue;
        // [中文导读] [AllReduce逐行 S87] 非0长度若不同于本版本结构大小，不能按本地布局解析。
        } else if (rmtDataLen != sizeof(OpExchangeInfo)) {
            // [中文导读] [AllReduce逐行 S88] 输出错误日志，记录InconsistentCheckParams当前阶段和相关参数。
            HCCL_ERROR(
                // [中文导读] [AllReduce逐行 S89] 补充日志格式：[InconsistentCheckParams] locDataLen is not equal to rmtDataLen. remoteRank[%u]", channel.remoteRank)。
                "[InconsistentCheckParams] locDataLen is not equal to rmtDataLen. remoteRank[%u]", channel.remoteRank);
            // [中文导读] [AllReduce逐行 S90] 元信息结构长度不匹配返回参数错误。
            return HCCL_E_PARA;
        // [中文导读] [AllReduce逐行 S91] 结束条件} else if (rmtDataLen != sizeof(OpExchangeInfo))。
        }
        // [中文导读] [AllReduce逐行 S92] 检查各rank CCL中转区容量相同。
        if (exchangeInfo.cclBufferSize != rmtExchangeInfo.cclBufferSize) {
            // [中文导读] [AllReduce逐行 S93] CCL容量不一致时调用错误上报辅助函数。
            CHK_RET(ReportOpExchangeInfoCheckFailed(
                // [中文导读] [AllReduce逐行 S94] 传入远端rank、参数名HcclBufferSize和本端CCL容量字符串。
                channel.remoteRank, exchangeInfo, "HcclBufferSize", std::to_string(exchangeInfo.cclBufferSize),
                // [中文导读] [AllReduce逐行 S95] 补充远端CCL容量字符串，上报函数返回错误则宏立即返回。
                std::to_string(rmtExchangeInfo.cclBufferSize)));
        // [中文导读] [AllReduce逐行 S96] 结束条件if (exchangeInfo.cclBufferSize != rmtExchangeInfo.cclBufferSize)。
        }
        // [中文导读] [AllReduce逐行 S97] 检查通用root字段相同。
        if (exchangeInfo.root != rmtExchangeInfo.root) {
            // [中文导读] [AllReduce逐行 S98] root不一致时开始错误上报。
            CHK_RET(ReportOpExchangeInfoCheckFailed(
                // [中文导读] [AllReduce逐行 S99] 传入本端/远端root，错误上报后向上返回错误。
                channel.remoteRank, exchangeInfo, "RootRankId", exchangeInfo.root, rmtExchangeInfo.root));
        // [中文导读] [AllReduce逐行 S100] 结束条件if (exchangeInfo.root != rmtExchangeInfo.root)。
        }
        // [中文导读] [AllReduce逐行 S101] 验证命令类型一致或满足P2P配对规则。
        CHK_RET(InconsistentCheckOpType(channel.remoteRank, exchangeInfo, rmtExchangeInfo.opType));
        // [中文导读] [AllReduce逐行 S102] 检查最终执行配置engine一致。
        if (exchangeInfo.opExecuteConfig != rmtExchangeInfo.opExecuteConfig) {
            // [中文导读] [AllReduce逐行 S103] 开始上报执行配置不一致。
            CHK_RET(ReportOpExchangeInfoCheckFailed(
                // [中文导读] [AllReduce逐行 S104] 指定OpExecuteConfig为不一致参数名字。
                channel.remoteRank, exchangeInfo, "OpExecuteConfig",
                // [中文导读] [AllReduce逐行 S105] 将本端执行配置转成uint32_t供上报。
                static_cast<uint32_t>(exchangeInfo.opExecuteConfig),
                // [中文导读] [AllReduce逐行 S106] 将远端执行配置转成uint32_t并结束上报调用。
                static_cast<uint32_t>(rmtExchangeInfo.opExecuteConfig)));
        // [中文导读] [AllReduce逐行 S107] 结束条件if (exchangeInfo.opExecuteConfig != rmtExchangeInfo.opExecuteConfig)。
        }
        // [中文导读] [AllReduce逐行 S108] 检查SUM等归约操作类型相同。
        if (exchangeInfo.reduceType != rmtExchangeInfo.reduceType) {
            // [中文导读] [AllReduce逐行 S109] 开始上报归约类型不一致。
            CHK_RET(ReportOpExchangeInfoCheckFailed(
                // [中文导读] [AllReduce逐行 S110] 提供HcclReduceOp参数名及本端归约枚举值。
                channel.remoteRank, exchangeInfo, "HcclReduceOp", static_cast<uint32_t>(exchangeInfo.reduceType),
                // [中文导读] [AllReduce逐行 S111] 提供远端归约枚举值并结束上报。
                static_cast<uint32_t>(rmtExchangeInfo.reduceType)));
        // [中文导读] [AllReduce逐行 S112] 结束条件if (exchangeInfo.reduceType != rmtExchangeInfo.reduceType)。
        }
        // [中文导读] [AllReduce逐行 S113] 检查数据类型相同。
        if (exchangeInfo.dataType != rmtExchangeInfo.dataType) {
            // [中文导读] [AllReduce逐行 S114] 开始上报数据类型不一致。
            CHK_RET(ReportOpExchangeInfoCheckFailed(
                // [中文导读] [AllReduce逐行 S115] 提供HcclDataType参数名及本端类型枚举值。
                channel.remoteRank, exchangeInfo, "HcclDataType", static_cast<uint32_t>(exchangeInfo.dataType),
                // [中文导读] [AllReduce逐行 S116] 提供远端类型枚举值并结束上报。
                static_cast<uint32_t>(rmtExchangeInfo.dataType)));
        // [中文导读] [AllReduce逐行 S117] 结束条件if (exchangeInfo.dataType != rmtExchangeInfo.dataType)。
        }
        // [中文导读] [AllReduce逐行 S118] 检查本次参与集合通信的元素count相同。
        if (exchangeInfo.count != rmtExchangeInfo.count) {
            // [中文导读] [AllReduce逐行 S119] 开始上报数据数量不一致。
            CHK_RET(ReportOpExchangeInfoCheckFailed(
                // [中文导读] [AllReduce逐行 S120] 提供DataCount参数名及本端64位count字符串。
                channel.remoteRank, exchangeInfo, "DataCount", std::to_string(exchangeInfo.count),
                // [中文导读] [AllReduce逐行 S121] 提供远端count字符串并结束上报。
                std::to_string(rmtExchangeInfo.count)));
        // [中文导读] [AllReduce逐行 S122] 结束条件if (exchangeInfo.count != rmtExchangeInfo.count)。
        }
        // [中文导读] [AllReduce逐行 S123] AIV核数限制不同时只打印警告，此分支没有返回错误。
        if (exchangeInfo.aivCoreLimit != rmtExchangeInfo.aivCoreLimit) {
            // [中文导读] [AllReduce逐行 S124] 开始记录核数限制不一致警告。
            HCCL_RUN_WARNING(
                // [中文导读] [AllReduce逐行 S125] 警告格式第一段说明AIV核数限制检查不一致。
                "[InconsistentCheckParams]op information aivCoreLimit check fail."
                // [中文导读] [AllReduce逐行 S126] 警告格式第二段展示远端rank、本端期望和对端核数。
                " remoteRank[%u] expectValue[%u] remotePara[%u]",
                // [中文导读] [AllReduce逐行 S127] 警告中展示远端rank及两端核数限制。
                channel.remoteRank, exchangeInfo.aivCoreLimit, rmtExchangeInfo.aivCoreLimit);
        // [中文导读] [AllReduce逐行 S128] 结束条件if (exchangeInfo.aivCoreLimit != rmtExchangeInfo.aivCoreLimit)。
        }
        // [中文导读] [AllReduce逐行 S129] 限定最大字符串长度，比较通信域名字。
        if (strncmp(exchangeInfo.group, rmtExchangeInfo.group, MAX_LENGTH) != 0) {
            // [中文导读] [AllReduce逐行 S130] 开始上报通信域名字不一致。
            CHK_RET(ReportOpExchangeInfoCheckFailed(
                // [中文导读] [AllReduce逐行 S131] 传入两端GroupName字符串并结束上报。
                channel.remoteRank, exchangeInfo, "GroupName", exchangeInfo.group, rmtExchangeInfo.group));
        // [中文导读] [AllReduce逐行 S132] 结束条件if (strncmp(exchangeInfo.group, rmtExchangeInfo.group, MAX_LENGTH) != 0)。
        }
        // [中文导读] [AllReduce逐行 S133] 限定tag最大长度，比较本次算子tag。
        if (strncmp(exchangeInfo.tag, rmtExchangeInfo.tag, TAG_LENGTH) != 0) {
            // [中文导读] [AllReduce逐行 S134] 初始化当前是否启用Group模式标志为false。
            bool isGroupEnabled = false;
            // [中文导读] [AllReduce逐行 S135] 运行时支持Group状态接口时才查询。
            if (HcommIsSupportHcclGroupStatusGet()) {
                // [中文导读] [AllReduce逐行 S136] 取得是否已启用Group模式。
                CHK_RET(HcclGroupStatusGet(&isGroupEnabled));
            // [中文导读] [AllReduce逐行 S137] 结束条件if (HcommIsSupportHcclGroupStatusGet())。
            }
            // [中文导读] [AllReduce逐行 S138] 只有未启用Group时tag不一致才报错。
            if (!isGroupEnabled) {
                // [中文导读] [AllReduce逐行 S139] 开始上报OpTag不一致。
                CHK_RET(ReportOpExchangeInfoCheckFailed(
                    // [中文导读] [AllReduce逐行 S140] 提供本端和远端tag字符串并结束上报。
                    channel.remoteRank, exchangeInfo, "OpTag", exchangeInfo.tag, rmtExchangeInfo.tag));
            // [中文导读] [AllReduce逐行 S141] 结束条件if (!isGroupEnabled)。
            }
        // [中文导读] [AllReduce逐行 S142] 结束条件if (strncmp(exchangeInfo.tag, rmtExchangeInfo.tag, TAG_LENGTH) != 0)。
        }
        // [中文导读] [AllReduce逐行 S143] 输出运行日志，记录InconsistentCheckParams当前阶段和相关参数。
        HCCL_INFO("[InconsistentCheckParams] success. remoteRank[%u]", channel.remoteRank);
    // [中文导读] [AllReduce逐行 S144] 结束循环for (auto& channel : channels)。
    }
    // [中文导读] [AllReduce逐行 S145] 逐Peer查询建链交换的元信息，核对CCL大小、root、命令、引擎、归约、dtype、count、group及tag处理完成，返回成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S146] 结束InconsistentCheckParams函数体。
}

// [中文导读] [AllReduce逐行 S148] 声明InconsistentCheckOpType接口：集合通信命令应一致，P2P需Send/Recv配对或已启用Group。
HcclResult
// [中文导读] [AllReduce逐行 S149] 函数参数包含InconsistentCheckOpType(uint32_t remoteRank, const OpExchangeInfo& exchangeInfo, const HcclCMDType& rmtOpType，本行延续接口声明。
InconsistentCheckOpType(uint32_t remoteRank, const OpExchangeInfo& exchangeInfo, const HcclCMDType& rmtOpType)
// [中文导读] [AllReduce逐行 S150] 开始InconsistentCheckOpType的函数体。
{
    // [中文导读] [AllReduce逐行 S151] 读取本端算子命令类型。
    HcclCMDType locOpType = exchangeInfo.opType;
    // [中文导读] [AllReduce逐行 S152] P2P命令需要对端Send/Recv配对检查。
    if (locOpType == HcclCMDType::HCCL_CMD_SEND || locOpType == HcclCMDType::HCCL_CMD_RECEIVE) {
        // [中文导读] [AllReduce逐行 S153] 初始化Group模式标志为false。
        bool isGroupEnabled = false;
        // [中文导读] [AllReduce逐行 S154] 有Group状态查询接口时读取模式。
        if (HcommIsSupportHcclGroupStatusGet()) {
            // [中文导读] [AllReduce逐行 S155] 查询是否启用Group。
            CHK_RET(HcclGroupStatusGet(&isGroupEnabled));
        // [中文导读] [AllReduce逐行 S156] 结束条件if (HcommIsSupportHcclGroupStatusGet())。
        }
        // [中文导读] [AllReduce逐行 S157] Group已经启用时跳过P2P命令配对校验。
        if (isGroupEnabled) {
            // [中文导读] [AllReduce逐行 S158] Group的P2P场景成功返回。
            return HCCL_SUCCESS;
        // [中文导读] [AllReduce逐行 S159] 结束条件if (isGroupEnabled)。
        }
        // HcclCMDType::HCCL_CMD_SEND和HcclCMDType::HCCL_CMD_RECEIVE的枚举值需确保大于等于0
        // [中文导读] [AllReduce逐行 S161] 用Send枚举值作为计算对端预期命令的第一项。
        uint32_t expectValue = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_SEND)
                               // [中文导读] [AllReduce逐行 S162] 加上Receive枚举值。
                               + static_cast<uint32_t>(HcclCMDType::HCCL_CMD_RECEIVE)
                               // [中文导读] [AllReduce逐行 S163] 减去本端命令，得到与本端相反的预期P2P命令。
                               - static_cast<uint32_t>(locOpType);
        // [中文导读] [AllReduce逐行 S164] 对端必须属于Send/Receive命令之一。
        if ((rmtOpType != HcclCMDType::HCCL_CMD_SEND && rmtOpType != HcclCMDType::HCCL_CMD_RECEIVE)
            // [中文导读] [AllReduce逐行 S165] 对端与本端命令相同也不满足配对。
            || locOpType == rmtOpType) {
            // [中文导读] [AllReduce逐行 S166] 开始上报P2P命令不配对。
            CHK_RET(ReportOpExchangeInfoCheckFailed(
                // [中文导读] [AllReduce逐行 S167] 提供预期命令和实际远端命令枚举值。
                remoteRank, exchangeInfo, "OpType", expectValue, static_cast<uint32_t>(rmtOpType)));
        // [中文导读] [AllReduce逐行 S168] 结束代码块。
        }
    // [中文导读] [AllReduce逐行 S169] 非P2P集合通信命令应相同，AllReduce在该分支核对ALLREDUCE。
    } else if (locOpType != rmtOpType) {
        // [中文导读] [AllReduce逐行 S170] 集合通信命令不一致时开始错误上报。
        CHK_RET(ReportOpExchangeInfoCheckFailed(
            // [中文导读] [AllReduce逐行 S171] 提供本端和远端命令枚举值。
            remoteRank, exchangeInfo, "OpType", static_cast<uint32_t>(locOpType), static_cast<uint32_t>(rmtOpType)));
    // [中文导读] [AllReduce逐行 S172] 结束条件} else if (locOpType != rmtOpType)。
    }
    // [中文导读] [AllReduce逐行 S173] 集合通信命令应一致，P2P需Send/Recv配对或已启用Group处理完成，返回成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S174] 结束InconsistentCheckOpType函数体。
}

HcclResult ReportOpExchangeInfoCheckFailed(
    uint32_t remoteRank, const OpExchangeInfo& exchangeInfo, const std::string& paraName, uint32_t expectVal,
    uint32_t remotePara)
{
    std::string opInfo = "Unknown";
    CHK_RET(GetOpTypeName(exchangeInfo, opInfo));

    RPT_INPUT_ERR(
        true, "EI0005", std::vector<std::string>({"ccl_op", "group", "para_name", "local_para", "remote_para"}),
        std::vector<std::string>(
            {opInfo, exchangeInfo.group, paraName, std::to_string(expectVal), std::to_string(remotePara)}));
    HCCL_ERROR(
        "[ReportOpExchangeInfoCheckFailed]op information %s check fail. remoteRank[%u] expectValue[%u] remotePara[%u]",
        paraName.c_str(), remoteRank, expectVal, remotePara);
    return HCCL_E_PARA;
}

HcclResult ReportOpExchangeInfoCheckFailed(
    uint32_t remoteRank, const OpExchangeInfo& exchangeInfo, const std::string& paraName, const std::string& expectVal,
    const std::string& remotePara)
{
    std::string opInfo = "Unknown";
    CHK_RET(GetOpTypeName(exchangeInfo, opInfo));
    RPT_INPUT_ERR(
        true, "EI0005", std::vector<std::string>({"ccl_op", "group", "para_name", "local_para", "remote_para"}),
        std::vector<std::string>({opInfo, exchangeInfo.group, paraName, expectVal, remotePara}));
    HCCL_ERROR(
        "[ReportOpExchangeInfoCheckFailed]op information %s check fail. remoteRank[%u] expectValue[%s] remotePara[%s]",
        paraName.c_str(), remoteRank, expectVal.c_str(), remotePara.c_str());
    return HCCL_E_PARA;
}

HcclResult GetOpTypeName(const OpExchangeInfo& exchangeInfo, std::string& opInfo)
{
    for (const auto& pair : HCCL_OPTYPE_NAME_MAP) {
        if (pair.second == exchangeInfo.opType) {
            opInfo = std::string(pair.first);
            break;
        }
    }
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
