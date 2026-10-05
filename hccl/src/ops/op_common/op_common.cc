/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <numeric>
#include <future>
#include <map>
#include <set>
#include <string>
#include <memory>
#include <cstdlib> // 包含getenv函数
#include <cstring> // 包含strcmp函数
#include <stdexcept>

#ifdef HCCL_ALGO_PLUGIN_ENABLE
#include "hccl_algo_plugin_mgr.h" // [HCCL-ALGO-Plugin]
#endif
#include <hccl/hccl_types.h>
#include <hccl/hccl_comm.h>
#include "dev_type.h"
#include "sal.h"
#include "error_codes/rt_error_codes.h"
#include "param_check.h"
#include "inconsistent_check.h"
#include "executor_base.h"
#include "coll_alg_v2_exec_registry.h"
#include "alg_attrs.h"
#include "alg_env_config.h"
#include "adapter_acl.h"
#include "topo_host.h"
#include "adapter_error_manager_pub.h"
#include "hccl_inner.h"
#include "hccl.h"
#include "config_log.h"
#include "load_kernel.h"
#include "alg_param.h"
#include "alg_type.h"
#include "op_common.h"
#include "ccu_fallback.h"
#include "aicpu_timeout.h"
#include "exec_timeout_manager.h"
#include "hccl_aiv_utils.h"
#include "dpu/kernel_launch.h"
#include "order_launch.h"
#include "hcomm_host_profiling_dl.h"
#include "hccl_host_comm_dl.h"
#include "hccl_res_dl.h"
#include "hccl_rank_graph_dl.h"
#include "rt_external.h"
#include "dlhcomm_function.h"
#include "hcomm_primitives_dl.h"
#include "hcomm_diag_dl.h"
#include "hcom.h"
#include "hccl_res_expt_dl.h"
#include "ccu_launch_dl.h"
#include "ccu_res_dl.h"
#include "hccl_ccu_res_dl.h"
#include "comm_engine_utils.h"
#include "utils.h"
#include "selector_engine.h"
#include "hcomm_dlsym.h"

namespace ops_hccl {
thread_local bool needInconsistentCheck = false;
// 用于维护增量建链算子的host ctx信息
constexpr u32 HOST_WAIT_AICPU_NOTIFYIDX = 0;   // host主流wait aicpu流的notify idx
constexpr u32 HOST_NOTIFY_TIMEOUT_OFFSET = 27; // host等待Device通知的超时时间偏移量
constexpr u32 KERNEL_TIMEOUT_OFFSET = 25;      // kernel启动超时时间偏移量
constexpr u32 CPU_TS_NOTIFY_NUM = 3;           // CPU TS thread notify数量

// [中文导读] [AllReduce逐行 S79] 声明UpdateAicpuTimeoutCtx接口：把算子执行超时派生为资源上下文中的通知等待及队列资源等待超时。
void UpdateAicpuTimeoutCtx(const OpParam& param, AlgResourceCtxSerializable& resCtx)
// [中文导读] [AllReduce逐行 S80] 开始UpdateAicpuTimeoutCtx的函数体。
{
    // [中文导读] [AllReduce逐行 S81] 从本次execTimeout派生AICPU通知及队列资源等待超时。
    AicpuTimeout timeout = DeriveAicpuTimeout(param.opConfig.execTimeout);
    // [中文导读] [AllReduce逐行 S82] 保存Device展开期间使用的通知等待超时。
    resCtx.waitTimeout = timeout.waitTimeout;
    // [中文导读] [AllReduce逐行 S83] 保存线程资源获取超时，Device展开入口将读取该值。
    resCtx.fullTimeout = timeout.fullTimeout;
    // [中文导读] [AllReduce逐行 S84] 输出运行日志，记录UpdateAicpuTimeoutCtx当前阶段和相关参数。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S85] 补充日志格式：[AicpuTimeout] execTimeout[%u]s, waitTimeout[%u]s, fullTimeout[%u]s。
        "[AicpuTimeout] execTimeout[%u]s, waitTimeout[%u]s, fullTimeout[%u]s, "
        // [中文导读] [AllReduce逐行 S86] 补充日志格式：hostNotifyTimeout[%u]s, kernelLaunchTimeout[%u]s, hcommDefaultTimeoutSupported[%u].。
        "hostNotifyTimeout[%u]s, kernelLaunchTimeout[%u]s, hcommDefaultTimeoutSupported[%u].",
        // [中文导读] [AllReduce逐行 S87] 提供上述日志的实参，涉及算子参数。
        param.opConfig.execTimeout, timeout.waitTimeout, timeout.fullTimeout, timeout.hostNotifyTimeout,
        // [中文导读] [AllReduce逐行 S88] 提供上述日志的实参：timeout.kernelLaunchTimeout, static_cast<u32>(IsHcommDefaultTimeoutSupported(。
        timeout.kernelLaunchTimeout, static_cast<u32>(IsHcommDefaultTimeoutSupported()));
// [中文导读] [AllReduce逐行 S89] 结束UpdateAicpuTimeoutCtx函数体。
}

#ifdef HCCL_ALGO_PLUGIN_ENABLE
namespace {

    HcclResult TryHandlePluginSelector(
        HcclComm comm, OpParam& param, TopoInfoWithNetLayerDetails* topoInfo, std::string& algName, bool& handled)
    {
        handled = false;
        param.pluginSelected = false;

        CHK_RET(HcclAlgoPluginMgr::Instance().Init());
        if (!HcclAlgoPluginMgr::Instance().IsLoaded()) {
            return HCCL_SUCCESS;
        }

        HcclAlgoPlugin_t* plugin = HcclAlgoPluginMgr::Instance().GetPlugin();
        void* pluginCtx = HcclAlgoPluginMgr::Instance().GetContext();

        HcclAlgoPluginParam pluginParam{};
        FillHcclAlgoPluginParam(param, topoInfo, pluginParam);

        char pluginAlgName[HCCL_ALGO_PLUGIN_ALG_NAME_LEN] = {0};
        if (!plugin->SelectAlg(pluginCtx, &pluginParam, pluginAlgName, sizeof(pluginAlgName))) {
            return HCCL_SUCCESS;
        }

        algName = pluginAlgName;
        param.pluginSelected = true;
        HCCL_INFO(
            "[Selector] plugin algorithm selected, algName=[%s], opType=[%d]", algName.c_str(),
            static_cast<int>(param.opType));

        if (algName == "") {
            HCCL_ERROR("[Selector] select algname fail!");
            return HCCL_E_PTR;
        }

        // 保持原来 Plugin 命中后的公共收尾行为
        CHK_RET(SetOpParamAlgTag(param, algName));
        CHK_RET(SetExecTimeout(param));
        CHK_RET(SetMultipleDimensionSplitRatio(comm, param));

        HCCL_INFO("Success to execute Selector.");
        handled = true;
        return HCCL_SUCCESS;
    }

} // namespace

#define HCCL_ALGO_PLUGIN_SELECTOR_FAST_PATH(comm, param, topoInfo, algName)                      \
    do {                                                                                         \
        bool pluginHandled = false;                                                              \
        CHK_RET(TryHandlePluginSelector((comm), (param), (topoInfo), (algName), pluginHandled)); \
        if (pluginHandled) {                                                                     \
            return HCCL_SUCCESS;                                                                 \
        }                                                                                        \
    } while (0)
#endif

static HcclResult checkValidAlgName(const std::string& algName)
{
    if (algName == "") {
        HCCL_ERROR("[Selector] select algname fail!");
        return HCCL_E_PTR;
    }
    return HCCL_SUCCESS;
}

// [中文导读] [AllReduce逐行 S158] 声明Selector接口：选择引擎和算法，恢复拓扑、加载对应kernel并建立算法tag。
HcclResult
// [中文导读] [AllReduce逐行 S159] 声明Selector接口：选择引擎和算法，恢复拓扑、加载对应kernel并建立算法tag。
Selector(HcclComm comm, OpParam& param, std::unique_ptr<TopoInfoWithNetLayerDetails>& topoInfo, std::string& algName)
// [中文导读] [AllReduce逐行 S160] 开始Selector的函数体。
{
    // 判断通信域状态
    // [中文导读] [AllReduce逐行 S162] 把通信域状态初始化为INVALID，供后续状态查询填充。
    HcclCommStatus commStatus = HCCL_COMM_STATUS_INVALID;
    // [中文导读] [AllReduce逐行 S163] 仅在运行时有通信域状态接口时执行状态检查。
    if (HcommIsSupportHcclCommGetStatus()) {
        // [中文导读] [AllReduce逐行 S164] 查询通信域当前状态；查询失败由CHK_RET直接返回。
        CHK_RET(HcclCommGetStatus(param.commName, &commStatus));
        // [中文导读] [AllReduce逐行 S165] 状态不是READY时阻止本次选择和展开。
        if (commStatus != HCCL_COMM_STATUS_READY) {
            // [中文导读] [AllReduce逐行 S166] 输出错误日志，记录Selector当前阶段和相关参数。
            HCCL_ERROR("commStatus is not ready!, commStatus = %d", static_cast<int>(commStatus));
            // [中文导读] [AllReduce逐行 S167] 通信域尚未就绪时返回暂停错误。
            return HCCL_E_SUSPENDING;
        // [中文导读] [AllReduce逐行 S168] 结束条件if (commStatus != HCCL_COMM_STATUS_READY)。
        }
    // [中文导读] [AllReduce逐行 S169] 结束条件if (HcommIsSupportHcclCommGetStatus())。
    }
    // [中文导读] [AllReduce逐行 S170] 输出运行日志，记录Selector当前阶段和相关参数。
    HCCL_INFO("Start to execute Selector.");
    // [中文导读] [AllReduce逐行 S171] 保存通信域句柄，后续selector和资源计算读取它。
    param.hcclComm = comm;
    // 获取基础拓扑
    // [中文导读] [AllReduce逐行 S173] 读取或创建拓扑缓存，填写topoInfo供算法选择。
    CHK_RET(HcclCalcTopoInfo(comm, param, topoInfo));
    // [中文导读] [AllReduce逐行 S174] 顶层使用UBoE时覆盖执行配置为AICPU_TS。
    if (topoInfo->topLevelUboe) {
        // [中文导读] [AllReduce逐行 S175] 将UBoE场景的引擎配置设置为AICPU_TS。
        param.opExecuteConfig = OpExecuteConfig::AICPU_TS;
    // [中文导读] [AllReduce逐行 S176] 结束条件if (topoInfo->topLevelUboe)。
    }

// [中文导读] [AllReduce逐行 S178] 此段仅在HCCL_ALGO_PLUGIN_ENABLE启用的构建中编译。
#ifdef HCCL_ALGO_PLUGIN_ENABLE
    // [中文导读] [AllReduce逐行 S179] 若插件快速选择宏命中自定义算法，则按该宏控制返回；本例使用内置算法。
    HCCL_ALGO_PLUGIN_SELECTOR_FAST_PATH(comm, param, topoInfo.get(), algName);
// [中文导读] [AllReduce逐行 S180] 结束上述编译期条件控制的源码范围。
#endif

    // 算法选择，选择完后顺便param.algTag设置了，资源的保存是以算子+算法为单位
    // [中文导读] [AllReduce逐行 S183] 新selector开关开启且算子在支持列表中时使用新选择引擎。
    if (IsNewSelectorEnabled() && SelectorEngine::IsOpSupported(param.opType)) {
        // [中文导读] [AllReduce逐行 S184] 新选择引擎根据候选元数据/代价等选择算法。
        CHK_RET(SelectorEngine::Global()->Run(comm, param, topoInfo.get(), algName));
    // [中文导读] [AllReduce逐行 S185] 上述条件不成立时进入替代分支。
    } else {
        // [中文导读] [AllReduce逐行 S186] 创建旧规则选择器对象，作为新选择引擎未启用或不支持时的路径。
        std::shared_ptr<ExecuteSelector> collAlgSelector = std::make_shared<ExecuteSelector>(ExecuteSelector());
        // [中文导读] [AllReduce逐行 S187] 调用规则选择器，写回算法名和执行配置。
        CHK_RET(collAlgSelector->Run(param, topoInfo.get(), algName));
    // [中文导读] [AllReduce逐行 S188] 结束条件} else。
    }
    // [中文导读] [AllReduce逐行 S189] 检查选择结果的算法名字是否合法。
    CHK_RET(checkValidAlgName(algName));
    // [中文导读] [AllReduce逐行 S190] 把最终执行配置转换为param.engine。
    CHK_RET(SetCommEngine(param));
    // AIV_ONLY 模式下禁止回退到非 AIV 引擎，未选中 AIV 时直接返回不支持。
    // [中文导读] [AllReduce逐行 S192] AIV_ONLY模式还需确保最终选中AIV执行引擎。
    if (param.commOpExpansionMode == HcclOpExpansionMode::HCCL_OP_EXPANSION_AIV_ONLY
        // [中文导读] [AllReduce逐行 S193] 补充AIV_ONLY约束：若实际engine不是AIV，则拒绝该选择。
        && param.engine != CommEngine::COMM_ENGINE_AIV) {
        // [中文导读] [AllReduce逐行 S194] 输出错误日志，记录Selector当前阶段和相关参数。
        HCCL_ERROR(
            // [中文导读] [AllReduce逐行 S195] 补充日志格式：[HcclExecOp] opType[%d] currently do not select aiv mode, aiv only not support.。
            "[HcclExecOp] opType[%d] currently do not select aiv mode, aiv only not support.",
            // [中文导读] [AllReduce逐行 S196] 提供上述日志的实参，涉及算子参数。
            static_cast<int>(param.opType));
        // [中文导读] [AllReduce逐行 S197] 返回AIV_ONLY约束不满足的错误。
        return HCCL_E_NOT_SUPPORT;
    // [中文导读] [AllReduce逐行 S198] 结束代码块。
    }
    // 如果一开始读取到的Engine不是aicpu，经过算法选择后回退到aicpu，则需要重新LoadAICPUKernel
    // [中文导读] [AllReduce逐行 S200] AICPU_TS或CPU展开路径需要确保AICPU二进制kernel已加载。
    if ((param.engine == CommEngine::COMM_ENGINE_AICPU_TS) || (param.engine == CommEngine::COMM_ENGINE_CPU)) {
        // [中文导读] [AllReduce逐行 S201] 输出调试日志，记录Selector当前阶段和相关参数。
        HCCL_DEBUG("[Selector] is aicpu mode");
        // [中文导读] [AllReduce逐行 S202] 加载AICPU入口二进制，内部防止重复加载。
        CHK_RET(LoadAICPUKernel()); // 该函数内部有防止重复加载的逻辑
        // 在三级组网下走clos链路时，aicpu引擎目前仅支持NHR算法
        // [中文导读] [AllReduce逐行 S204] 输出警告日志，记录Selector当前阶段和相关参数。
        HCCL_WARNING("under 3-level topology with CLOS link, aicpu engine currently only supports NHR algorithm.");
    // [中文导读] [AllReduce逐行 S205] 结束条件if ((param.engine == CommEngine::COMM_ENGINE_AICPU_TS) || (param.engine == CommEngine::COMM_ENGINE_CPU))。
    }
    // 如果一开始读取到的Engine不是aiv，经过算法选择后回退到aiv，则需要重新RegisterKernel
    // [中文导读] [AllReduce逐行 S207] AIV路径需要注册AIV kernel。
    if (param.engine == CommEngine::COMM_ENGINE_AIV) {
        // [中文导读] [AllReduce逐行 S208] 输出调试日志，记录Selector当前阶段和相关参数。
        HCCL_DEBUG("[Selector] is aiv mode");
        // [中文导读] [AllReduce逐行 S209] 注册AIV kernel，内部防止重复注册。
        CHK_RET(RegisterKernel()); // 该函数内部有防止重复加载的逻辑
    // [中文导读] [AllReduce逐行 S210] 结束条件if (param.engine == CommEngine::COMM_ENGINE_AIV)。
    }

    // [中文导读] [AllReduce逐行 S212] 为选中算法生成资源缓存/执行关联用的algTag。
    CHK_RET(SetOpParamAlgTag(param, algName));
    // 设定执行超时时间
    // [中文导读] [AllReduce逐行 S214] 从通信域配置等信息确定本次执行超时。
    CHK_RET(SetExecTimeout(param));
    // 获取多维度切分比例
    // [中文导读] [AllReduce逐行 S216] 读取多维度算法切分比例并保存到param。
    CHK_RET(SetMultipleDimensionSplitRatio(comm, param));
// [中文导读] [AllReduce逐行 S217] 按CANN_VERSION_NUM >= CANN_VERSION(9, 2, 0)这个编译期版本条件决定是否包含以下分支。
#if CANN_VERSION_NUM >= CANN_VERSION(9, 2, 0)
    // CCU模式跨rank协商：各rank交换opExecuteConfig取最低公共值，若被降级则直接走ReSelector回退到该config对应的算法
    // [中文导读] [AllReduce逐行 S219] CANN 9.2及以上CCU配置跨rank协商，必要时重新选择降级引擎算法。
    CHK_RET(CheckCcuParamAndFallback(comm, param, topoInfo, algName));
// [中文导读] [AllReduce逐行 S220] 结束上述编译期条件控制的源码范围。
#endif
    // [中文导读] [AllReduce逐行 S221] 输出运行日志，记录Selector当前阶段和相关参数。
    HCCL_INFO("Success to execute Selector.");
    // [中文导读] [AllReduce逐行 S222] 选择引擎和算法，恢复拓扑、加载对应kernel并建立算法tag处理完成，返回成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S223] 结束Selector函数体。
}

HcclResult GetHcclDfxOpInfoDataCount(const OpParam& param, const u32& rankSize, uint64_t& sendCount)
{
    sendCount = 0;
    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
        CHK_PTR_NULL(param.all2AllVDataDes.sendCounts);
        sendCount
            += *(reinterpret_cast<const uint64_t*>(param.all2AllVDataDes.sendCounts)); // 非v类算子，只上报入参里的count
    } else if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        CHK_PTR_NULL(param.all2AllVDataDes.sendCounts);
        for (u64 i = 0; i < rankSize; i++) { // v类算子上报累加的count
            sendCount += *(reinterpret_cast<const uint64_t*>(param.all2AllVDataDes.sendCounts) + i);
        }
    } else if (param.opType == HcclCMDType::HCCL_CMD_ALLGATHER_V) {
        CHK_PTR_NULL(param.varData);
        for (u64 i = 0; i < rankSize; i++) {
            sendCount += *(reinterpret_cast<const uint64_t*>(param.varData) + i);
        }
    } else if (param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V) {
        CHK_PTR_NULL(param.varData);
        for (u64 i = rankSize; i < 2 * rankSize; i++) {
            sendCount += *(reinterpret_cast<const uint64_t*>(param.varData) + i);
        }
    } else if (param.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
        CHK_PRT_RET(
            param.batchSendRecvDataDes.sendRecvItemsPtr == nullptr,
            HCCL_ERROR(
                "[%s]fail, tag[%s] sendRecvItemsPtr is nullptr, itemNum[%u]", __func__, param.tag,
                param.batchSendRecvDataDes.itemNum),
            HCCL_E_PTR);
        for (u32 idx = 0; idx < param.batchSendRecvDataDes.itemNum; idx++) {
            HcclSendRecvItem* item = param.batchSendRecvDataDes.sendRecvItemsPtr + idx;
            sendCount += item->count;
        }
    } else {
        sendCount = param.DataDes.count;
    }
    HCCL_INFO(
        "[%s]tag[%s], sendCount[%llu], opType[%u], rankSize[%u]", __func__, param.tag, sendCount, param.opType,
        rankSize);
    return HCCL_SUCCESS;
}

HcclResult GetHcclDfxOpInfoDataType(const OpParam& param, uint32_t& dataType)
{
    dataType = 0;
    if (param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V || param.opType == HcclCMDType::HCCL_CMD_ALLGATHER_V) {
        dataType = static_cast<u32>(param.vDataDes.dataType);
    } else if (
        param.opType == HcclCMDType::HCCL_CMD_ALLTOALL || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV
        || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        dataType = static_cast<u32>(param.all2AllVDataDes.sendType);
    } else if (param.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
        CHK_PRT_RET(
            param.batchSendRecvDataDes.itemNum == 0, HCCL_INFO("[%s]tag[%s] itemNum is 0, skip", __func__, param.tag),
            HCCL_SUCCESS);
        CHK_PRT_RET(
            param.batchSendRecvDataDes.sendRecvItemsPtr == nullptr,
            HCCL_ERROR("[%s]fail, tag[%s] sendRecvItemsPtr is nullptr", __func__, param.tag), HCCL_E_PTR);
        dataType
            = static_cast<u32>(param.batchSendRecvDataDes.sendRecvItemsPtr->dataType); // dfx功能只能上报一个数据类型
    } else {
        dataType = static_cast<u32>(param.DataDes.dataType);
    }
    HCCL_INFO("[%s]tag[%s], dataType[%u], opType[%u]", __func__, param.tag, dataType, param.opType);
    return HCCL_SUCCESS;
}

HcclResult AppendFastLaunchTag(
    OpParam& param, const char* dataTypeStr, const char* reduceOpStr, const char* countStr, const char* rootStr)
{
    char* dst = param.fastLaunchTag;
    size_t remain = sizeof(param.fastLaunchTag);

    auto append_str = [&](const char* s) -> bool {
        if (!s)
            return true;
        size_t len = strlen(s);
        if (len >= remain)
            return false;
        if (memcpy_s(dst, remain, s, len) != EOK) {
            HCCL_ERROR("memcpy_s failed in append_str.");
            return false;
        }
        dst += len;
        remain -= len;
        return true;
    };
    if (!append_str(param.tag) || !append_str("_") || !append_str(dataTypeStr)) {
        goto fail;
    }
    if (reduceOpStr && (!append_str("_") || !append_str(reduceOpStr))) {
        goto fail;
    }
    if (countStr && (!append_str("_") || !append_str(countStr))) {
        goto fail;
    }
    if (rootStr && (!append_str("_r") || !append_str(rootStr))) {
        goto fail;
    }
    *dst = '\0';
    HCCL_INFO("[SetOpParamFastLaunchTag] fastLaunchTag: [%s]", param.fastLaunchTag);
    return HcclResult::HCCL_SUCCESS;

fail:
    HCCL_ERROR("failed to fill fastLaunchTag");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult SetOpParamFastLaunchTag(OpParam& param)
{
    // 1. 数据类型
    const char* dataTypeStr = nullptr;
    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV
        || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        dataTypeStr = GetHcclDataTypeStr(param.all2AllVDataDes.sendType);
    } else {
        dataTypeStr = GetHcclDataTypeStr(param.DataDes.dataType);
    }
    CHK_PRT_RET((!dataTypeStr), HCCL_ERROR("unsupported data type"), HcclResult::HCCL_E_INTERNAL);
    // 2. reduce op
    const char* reduceOpStr = nullptr;
    if (param.opType == HcclCMDType::HCCL_CMD_ALLREDUCE || param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER
        || param.opType == HcclCMDType::HCCL_CMD_REDUCE || param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V) {
        reduceOpStr = GetHcclReduceOpStr(param.reduceType);
        CHK_PRT_RET((!reduceOpStr), HCCL_ERROR("unsupported reduce op"), HcclResult::HCCL_E_INTERNAL);
    }
    // 3. count
    char countBuf[32];
    const char* countStr = nullptr;
    if (param.opType != HcclCMDType::HCCL_CMD_ALLTOALLV) {
        u64 count = (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL) ?
                        *reinterpret_cast<u64*>(param.all2AllVDataDes.sendCounts) :
                        param.DataDes.count;
        int countLen = snprintf_s(countBuf, sizeof(countBuf), sizeof(countBuf) - 1, "%llu", count);
        CHK_PRT_RET((countLen <= 0), HCCL_ERROR("failed to format count"), HcclResult::HCCL_E_INTERNAL);
        countStr = countBuf;
    }
    // 4. root
    char rootBuf[10];
    const char* rootStr = nullptr;
    if (param.opType == HcclCMDType::HCCL_CMD_REDUCE || param.opType == HcclCMDType::HCCL_CMD_SCATTER
        || param.opType == HcclCMDType::HCCL_CMD_BROADCAST) {
        int rootLen
            = snprintf_s(rootBuf, sizeof(rootBuf), sizeof(rootBuf) - 1, "%llu", static_cast<uint64_t>(param.root));
        CHK_PRT_RET((rootLen <= 0), HCCL_ERROR("failed to format root"), HcclResult::HCCL_E_INTERNAL);
        rootStr = rootBuf;
    }
    // 5 一次性拼接
    return AppendFastLaunchTag(param, dataTypeStr, reduceOpStr, countStr, rootStr);
}

static constexpr uint32_t opExpansionModeCcuSched = 5;
static constexpr uint32_t opExpansionModeCcuMs = 4;

bool ShouldGoCcuFastLaunch(HcclComm comm, OpParam& param, CcuFastLaunchCtx** ccuFastLaunchCtx)
{
#if CANN_VERSION_NUM >= CANN_VERSION(9, 1, 0)
    param.hcclComm = comm;
    if (param.opMode == OpMode::OFFLOAD) {
        return false;
    }
    // 1. 引擎为ccu模式
    if (param.engine != CommEngine::COMM_ENGINE_CCU) {
        return false;
    }
    if (SetOpParamFastLaunchTag(param) != HCCL_SUCCESS) {
        return false;
    }

    // 2. 查到engineCtx
    uint64_t size = 0;
    void* fastLaunchCtxPtr = nullptr;
    if (HcclEngineCtxGet(comm, param.fastLaunchTag, CommEngine::COMM_ENGINE_CCU, &fastLaunchCtxPtr, &size)
        == HCCL_SUCCESS) {
        HCCL_INFO("[ShouldGoCcuFastLaunch] get fastLaunchCtx success, size is %u", size);
        *ccuFastLaunchCtx = reinterpret_cast<CcuFastLaunchCtx*>(fastLaunchCtxPtr);
        return true;
    }
    return false;
#else
    (void)comm;
    (void)param;
    (void)ccuFastLaunchCtx;
    return false;
#endif
}

HcclResult ConstructHcclDfxOpInfo(
    const OpParam& param, const char* tag, u32 tagSize, HcclDfxOpInfoCompat& hcclDfxOpInfo, ThreadHandle cpuTsThread)
{
    bool isAclGraph = IsStreamInCaptureMode(param.stream);
    hcclDfxOpInfo.opMode = isAclGraph ? static_cast<u32>(ops_hccl::OpMode::ACLGRAPH) : static_cast<u32>(param.opMode);
    hcclDfxOpInfo.opType = static_cast<u32>(param.opType);
    hcclDfxOpInfo.reduceOp = static_cast<u32>(param.reduceType);
    CHK_RET(GetHcclDfxOpInfoDataType(param, hcclDfxOpInfo.dataType));

    // rankSize获取指定算子的dataCount
    u32 userRankSize{0};
    CHK_RET(HcclGetRankSize(param.hcclComm, &userRankSize));
    CHK_RET(GetHcclDfxOpInfoDataCount(param, userRankSize, hcclDfxOpInfo.dataCount));
    hcclDfxOpInfo.root = param.root;
    hcclDfxOpInfo.engine = param.engine;

    hcclDfxOpInfo.inputMemAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    hcclDfxOpInfo.inputMemSize = param.inputSize;
    hcclDfxOpInfo.outputMemAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    hcclDfxOpInfo.outputMemSize = param.outputSize;

    hcclDfxOpInfo.cpuTsThread = cpuTsThread;
    hcclDfxOpInfo.cpuWaitAicpuNotifyIdx = HOST_WAIT_AICPU_NOTIFYIDX;
    s32 sRet = strncpy_s(hcclDfxOpInfo.algTag, ALG_TAG_LENGTH, tag, tagSize);
    CHK_PRT_RET(
        sRet != EOK, HCCL_ERROR("%s call strncpy_s failed, tag:%s, tagSize:%u, sRet:%d.", __func__, tag, tagSize, sRet),
        HCCL_E_MEMORY);
    HCCL_INFO(
        "[%s]HcclDfxOpInfo param: algTag[%s], opMode[%u], opType[%u], reduceOp[%u], dataType[%u], dataCount[%llu], "
        "root[%u], engine[%s], inputMemAddr[0x%llx], inputMemSize[%llu], outputMemAddr[0x%llx], outputMemSize[%llu], "
        "cpuTsThread[0x%llu], cpuWaitAicpuNotifyIdx[%u]",
        __func__, hcclDfxOpInfo.algTag, hcclDfxOpInfo.opMode, hcclDfxOpInfo.opType, hcclDfxOpInfo.reduceOp,
        hcclDfxOpInfo.dataType, hcclDfxOpInfo.dataCount, hcclDfxOpInfo.root,
        GetEnumToString(GetCommEngineStatusStrMap(), hcclDfxOpInfo.engine).c_str(), hcclDfxOpInfo.inputMemAddr,
        hcclDfxOpInfo.inputMemSize, hcclDfxOpInfo.outputMemAddr, hcclDfxOpInfo.outputMemSize, hcclDfxOpInfo.cpuTsThread,
        hcclDfxOpInfo.cpuWaitAicpuNotifyIdx);
    return HCCL_SUCCESS;
}

bool IsStreamInCaptureMode(aclrtStream stream)
{
    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    aclmdlRI rtModel = nullptr;
    aclError aclRet = aclmdlRICaptureGetInfo(stream, &captureStatus, &rtModel);
    if (aclRet == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
        return false;
    }
    if (aclRet != ACL_SUCCESS) {
        HCCL_ERROR("[%s] aclmdlRICaptureGetInfo fail, ret[%d]", __func__, aclRet);
        return false;
    }
    return captureStatus == aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
}

bool IsAivCacheSupported(const OpParam& param)
{
    return (param.opType == HCCL_CMD_ALLGATHER || param.opType == HCCL_CMD_ALLREDUCE
            || param.opType == HCCL_CMD_REDUCE_SCATTER || param.opType == HCCL_CMD_BROADCAST
            || param.opType == HCCL_CMD_REDUCE || param.opType == HCCL_CMD_ALLTOALL
            || param.opType == HCCL_CMD_ALLTOALLV || param.opType == HCCL_CMD_SCATTER)
           && param.opMode == OpMode::OPBASE && !IsStreamInCaptureMode(param.stream);
}

HcclResult HcclAivCacheCheckAndReplay(HcclComm comm, OpParam& param, bool& cacheHit)
{
    cacheHit = false;
    if (!IsAivCacheSupported(param)) {
        return HCCL_SUCCESS;
    }

    // 提前获取 numBlocksLimit，作为 cache key 的一部分
    // 与 HcclAivKernelEntranceLaunch 逻辑保持一致：先查 aivParam，再 fallback 到 runtime
    u32 numBlocksLimit = 0;
    AivParamStorage* aivParam = nullptr;
    HcclResult aivParamRet = GetAivParamStorageByComm(comm, &aivParam, false);
    if (aivParamRet == HCCL_SUCCESS && aivParam != nullptr) {
        numBlocksLimit = aivParam->aivCoreLimit;
    }
    if (numBlocksLimit == 0) {
        ACLCHECK(aclrtGetResInCurrentThread(ACL_RT_DEV_RES_VECTOR_CORE, &numBlocksLimit));
    }
    param.numBlocksLimit = numBlocksLimit;

    AivOpCacheArgs cacheKey = {};
    cacheKey.commName = param.commName;
    cacheKey.opType = param.opType;
    cacheKey.root = param.root;
    cacheKey.reduceOp = param.reduceType;
    cacheKey.numBlocksLimit = numBlocksLimit;
    if (param.opType == HCCL_CMD_ALLTOALL) {
        cacheKey.dataType = param.all2AllVDataDes.sendType;
        cacheKey.count = static_cast<const u64*>(param.all2AllVDataDes.sendCounts)[0];
    } else if (param.opType == HCCL_CMD_ALLTOALLV) {
        cacheKey.dataType = param.all2AllVDataDes.sendType;
        cacheKey.count = 0; // counts 在 replay 时动态刷新,cache key 不区分
    } else {
        cacheKey.count = param.DataDes.count;
        cacheKey.dataType = param.DataDes.dataType;
    }

    u64 keyHash = CalcAivCacheKeyHash(cacheKey);
    std::string ctxTag;
    CHK_RET(BuildAivCacheCtxTag(keyHash, ctxTag));

    std::string cachedAlgName;
    AivInstruction* instructions = nullptr;
    u32 insCount = 0;
    // Keep cached instruction storage alive through replay, including its non-trivial members.
    std::unique_lock<std::mutex> cacheLock(GetAivCacheMutex());
    CHK_RET(LookupAivCacheCtx(comm, ctxTag, keyHash, cacheHit, cachedAlgName, instructions, insCount));
    if (!cacheHit) {
        return HCCL_SUCCESS;
    }

    HCCL_INFO("[HcclAivCacheCheckAndReplay] cache hit, algName[%s], insCount[%u]", cachedAlgName.c_str(), insCount);

    int result = sprintf_s(param.algName, sizeof(param.algName), "%s", cachedAlgName.c_str());
    CHK_PRT_RET(result <= 0, HCCL_ERROR("[%s] failed to fill param.algName", __func__), HCCL_E_INTERNAL);
    CHK_RET(SetOpParamAlgTag(param, cachedAlgName));
    param.hcclComm = comm;

    uint64_t beginTime = HcommGetProfilingSysCycleTime();

    HcclDfxOpInfoCompat hcclDfxOpInfo{};
    CHK_RET(ConstructHcclDfxOpInfo(param, param.algTag, ALG_TAG_LENGTH, hcclDfxOpInfo, 0));
    param.dataCount = hcclDfxOpInfo.dataCount;
    CHK_RET(HcclDfxRegOpInfoByCommId(param.commName, reinterpret_cast<void*>(&hcclDfxOpInfo)));

    if (param.opType == HCCL_CMD_ALLTOALLV) {
        CHK_RET(ReplayAivInstructionsV(instructions, insCount, param));
    } else {
        CHK_RET(ReplayAivInstructions(instructions, insCount, param));
    }
    cacheLock.unlock();

    CHK_RET(HcclReportAivKernel(comm, beginTime));
    CHK_RET(HcclProfilingReportOp(comm, beginTime));
    return HCCL_SUCCESS;
}

HcclResult HcclExecOpCcuFastLaunch(HcclComm comm, OpParam& param, const CcuFastLaunchCtx* ccuFastLaunchCtx)
{
#if CANN_VERSION_NUM >= CANN_VERSION(9, 1, 0)
    HCCL_INFO("[HcclExecOpCcuFastLaunch] HcclExecOpCcuFastLaunch start");
    std::string algName = ccuFastLaunchCtx->algName;
    HCCL_DEBUG("[HcclExecOpCcuFastLaunch] algName: [%s]", algName.c_str());
    std::unique_ptr<InsCollAlgBase> executor = CollAlgExecRegistryV2::Instance().GetAlgExec(param.opType, algName);
    CHK_PRT_RET(
        executor.get() == nullptr, HCCL_ERROR("Fail to find executor for algName[%s]", algName.c_str()), HCCL_E_PARA);

    void* cclBufferAddr;
    uint64_t cclBufferSize;
    // 从通信域获取CCL buffer
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    // CCL IN使用所有的CCL Buffer，这个其实就是scratch buffer
    param.hcclBuff = HcclMem{HCCL_MEM_TYPE_DEVICE, cclBufferAddr, cclBufferSize};
    // 覆盖主流
    ThreadHandle mainThread;
    CHK_RET(HcclThreadAcquireWithStream(
        comm, param.engine, param.stream, ccuFastLaunchCtx->notifyNumOnMainThread, &mainThread));
    ThreadHandle* threads = ccuFastLaunchCtx->GetThreadHandlePtr();
    threads[0] = mainThread;
    std::vector<ThreadHandle> threadTemps;
    threadTemps.assign(threads, threads + ccuFastLaunchCtx->threadNum);
    uint64_t beginTime = HcommGetProfilingSysCycleTime();
    // Op注册
    HcclDfxOpInfoCompat hcclDfxOpInfo{};
    CHK_RET(ConstructHcclDfxOpInfo(param, param.fastLaunchTag, ALG_TAG_LENGTH, hcclDfxOpInfo, 0));
    param.dataCount = hcclDfxOpInfo.dataCount;
    CHK_RET(HcclDfxRegOpInfoByCommId(param.commName, reinterpret_cast<void*>(&hcclDfxOpInfo)));
    if (IsStreamInCaptureMode(param.stream) && threadTemps.size() > 1) {
        HCCL_INFO("HcclExecOpCcuFastLaunch streamnum %d add slavestream", threadTemps.size());
        CHK_RET(CaptureSlaveStreams(comm, param.stream, threadTemps, param.isCapture));
    }

    HCCL_INFO("[HcclExecOpCcuFastLaunch] FastLaunch start");
    CHK_RET(executor->FastLaunch(param, ccuFastLaunchCtx));
    CHK_RET(HcclProfilingReportOp(comm, beginTime));
    HCCL_INFO("[HcclExecOpCcuFastLaunch] HcclExecOpCcuFastLaunch end");
    return HCCL_SUCCESS;
#else
    (void)comm;
    (void)param;
    (void)ccuFastLaunchCtx;
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult ExecuteAivCacheLogic(
    HcclComm comm, OpParam& param, const std::string& algName, std::unique_ptr<InsCollAlgBase>& executor,
    AlgResourceCtxSerializable& resCtxHost)
{
    bool useCache = IsAivCacheSupported(param);

    std::string ctxTag;
    u64 keyHash = 0;
    if (useCache) {
        AivOpCacheArgs cacheKey = {};
        cacheKey.root = param.root;
        cacheKey.commName = param.commName;
        cacheKey.opType = param.opType;
        cacheKey.reduceOp = param.reduceType;
        cacheKey.numBlocksLimit = param.numBlocksLimit;
        if (param.opType == HCCL_CMD_ALLTOALL) {
            cacheKey.dataType = param.all2AllVDataDes.sendType;
            cacheKey.count = static_cast<const u64*>(param.all2AllVDataDes.sendCounts)[0];
        } else if (param.opType == HCCL_CMD_ALLTOALLV) {
            cacheKey.dataType = param.all2AllVDataDes.sendType;
            cacheKey.count = 0; // counts 在 replay 时动态刷新,cache key 不区分
        } else {
            cacheKey.count = param.DataDes.count;
            cacheKey.dataType = param.DataDes.dataType;
        }
        keyHash = CalcAivCacheKeyHash(cacheKey);
        CHK_RET(BuildAivCacheCtxTag(keyHash, ctxTag));
        g_recordingQueue = std::make_shared<InsQueue>();
        g_baseInputAddr = reinterpret_cast<u64>(param.inputPtr);
        g_baseOutputAddr = reinterpret_cast<u64>(param.outputPtr);
    }

    CHK_RET(executor->Orchestrate(param, resCtxHost));

    if (useCache && g_recordingQueue) {
        AivCacheIndexCtx* indexCtx = nullptr;
        CHK_RET(GetOrCreateAivCacheIndexCtx(comm, &indexCtx));
        CHK_RET(StoreAivCacheCtx(comm, ctxTag, keyHash, algName, indexCtx));
        g_recordingQueue = nullptr;
        g_baseInputAddr = 0;
        g_baseOutputAddr = 0;
    }
    return HCCL_SUCCESS;
}

HcclResult FallbackOp(
    HcclComm comm, OpParam& param, std::unique_ptr<TopoInfoWithNetLayerDetails>& topoInfo, std::string& algName,
    const ResPackGraphMode& resPack)
{
    OpExecuteConfig nextConfig;
#if CANN_VERSION_NUM >= CANN_VERSION(9, 2, 0)
    if (param.opExecuteConfig == OpExecuteConfig::CCU_MS) {
        nextConfig = OpExecuteConfig::CCU_SCHED;
    } else if (param.opExecuteConfig == OpExecuteConfig::CCU_SCHED) {
        nextConfig = OpExecuteConfig::AICPU_TS;
    } else {
        HCCL_ERROR(
            "[FallbackOp] already at AICPU_TS or unknown config[%u], cannot fallback further.",
            static_cast<uint32_t>(param.opExecuteConfig));
        return HCCL_E_NOT_SUPPORT;
    }
#else
    nextConfig = OpExecuteConfig::AICPU_TS;
#endif

    void* fallbackCtx = nullptr;
    uint64_t fallbackCtxSize = sizeof(FallbackCtxData);
    CHK_RET(HcclEngineCtxCreate(comm, param.fallbackTag, CommEngine::COMM_ENGINE_CCU, fallbackCtxSize, &fallbackCtx));
    auto* ctxData = static_cast<FallbackCtxData*>(fallbackCtx);
    CHK_RET(ReSelector(comm, param, topoInfo, algName, nextConfig));
    auto copyRet = sprintf_s(ctxData->algName, sizeof(ctxData->algName), "%s", algName.c_str());
    if (copyRet <= 0) {
        HCCL_ERROR("[%s] failed to fill algName", __func__);
        return HCCL_E_INTERNAL;
    }
    ctxData->opExecuteConfig = param.opExecuteConfig;
    CHK_RET(HcclExecOp(comm, param, topoInfo, algName, resPack));
    return HCCL_SUCCESS;
}

HcclResult ReSelector(
    HcclComm comm, OpParam& param, std::unique_ptr<TopoInfoWithNetLayerDetails>& topoInfo, std::string& algName,
    OpExecuteConfig executeConfig)
{
    (void)comm;
    HCCL_INFO("Start to execute ReSelector.");
    // 按传入的executeConfig回退
    param.opExecuteConfig = executeConfig;
    // 拓扑已有，无需再计算

    // 算法选择，选择完后顺便param.algTag设置了，资源的保存是以算子+算法为单位
    if (IsNewSelectorEnabled() && SelectorEngine::IsOpSupported(param.opType)) {
        CHK_RET(SelectorEngine::Global()->Run(comm, param, topoInfo.get(), algName));
    } else {
        std::shared_ptr<ExecuteSelector> collAlgSelector = std::make_shared<ExecuteSelector>(ExecuteSelector());
        CHK_RET(collAlgSelector->Run(param, topoInfo.get(), algName));
    }
    if (algName == "") {
        HCCL_ERROR("[ReSelector] select algname fail!");
        return HCCL_E_PTR;
    }
    CHK_RET(SetCommEngine(param));
    // AIV_ONLY 模式下禁止回退到非 AIV 引擎，未选中 AIV 时直接返回不支持。
    if (param.commOpExpansionMode == HcclOpExpansionMode::HCCL_OP_EXPANSION_AIV_ONLY
        && param.engine != CommEngine::COMM_ENGINE_AIV) {
        HCCL_ERROR(
            "[HcclExecOp] opType[%d] currently do not select aiv mode, aiv only not support.",
            static_cast<int>(param.opType));
        return HCCL_E_NOT_SUPPORT;
    }
    // 如果一开始读取到的Engine不是aicpu，经过算法选择后回退到aicpu，则需要重新LoadAICPUKernel
    if ((param.engine == CommEngine::COMM_ENGINE_AICPU_TS) || (param.engine == CommEngine::COMM_ENGINE_CPU)) {
        HCCL_DEBUG("[ReSelector] is aicpu mode");
        CHK_RET(LoadAICPUKernel()); // 该函数内部有防止重复加载的逻辑
    }
    // 若传入的executeConfig经Select内部回退(如IsRollBackAiv命中)后最终落到AIV引擎，需注册AIV kernel
    if (param.engine == CommEngine::COMM_ENGINE_AIV) {
        HCCL_DEBUG("[ReSelector] is aiv mode");
        CHK_RET(RegisterKernel());
    }
    CHK_RET(SetOpParamAlgTag(param, algName));
    HCCL_INFO("Success to execute ReSelector.");
    return HCCL_SUCCESS;
}

HcclResult SetOpParamFallbackTag(OpParam& param, const std::string& algName)
{
    auto fallbackRet = sprintf_s(param.fallbackTag, sizeof(param.fallbackTag), "%s_%s", algName.c_str(), "fallback");
    if (fallbackRet <= 0) {
        HCCL_ERROR("[%s] failed to fill fallbackTag", __func__);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

#ifdef HCCL_ALGO_PLUGIN_ENABLE
HcclResult
ExecutePluginAlgorithm(HcclComm comm, OpParam& param, TopoInfoWithNetLayerDetails* topoInfo, const std::string& algName)
{
    if (!HcclAlgoPluginMgr::Instance().IsLoaded()) {
        HCCL_ERROR("[HcclExecOp] pluginSelected=true but PluginBroker is not loaded, algName=[%s]", algName.c_str());
        return HCCL_E_INTERNAL;
    }

    HcclAlgoPlugin_t* plugin = HcclAlgoPluginMgr::Instance().GetPlugin();
    void* pluginCtx = HcclAlgoPluginMgr::Instance().GetContext();

    HcclAlgoPluginParam pluginParam{};
    FillHcclAlgoPluginParam(param, topoInfo, pluginParam);

    int execRet
        = plugin->ExecuteAlg(pluginCtx, algName.c_str(), pluginParam.opName, &pluginParam, static_cast<void*>(comm));

    if (execRet != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcclExecOp] plugin ExecuteAlg failed, algName=[%s], opType=[%d], ret=[%d]", algName.c_str(),
            static_cast<int>(param.opType), execRet);
        return HCCL_E_INTERNAL;
    }

    HCCL_INFO("[HcclExecOp] plugin algorithm execute success, algName=[%s]", algName.c_str());
    return HCCL_SUCCESS;
}
#endif

static HcclResult ReportOpProfilingInfo(HcclComm comm, HcclCMDType opType, uint64_t beginTime)
{
    bool isGroupEnabled = false;
    if (HcommIsSupportHcclGroupStatusGet()) {
        CHK_RET(HcclGroupStatusGet(&isGroupEnabled));
    }
    bool isSendOrRecv = (opType == HcclCMDType::HCCL_CMD_SEND || opType == HcclCMDType::HCCL_CMD_RECEIVE);
    // group模式下aicpu的p2p算子reportOp在groupEnd中作为整体算子上报，不在这里处理
    if (!(isGroupEnabled && isSendOrRecv)) {
        CHK_RET(HcclProfilingReportOp(comm, beginTime));
    }

    return HCCL_SUCCESS;
}

// [中文导读] 多算子共用的执行枢纽：取得executor、查询/准备算法资源，然后按Engine走不同执行入口。
// [中文导读] param携带本次调用参数，topoInfo描述拓扑，algName选择具体算法；不是直接通信的原语。
// [中文导读] 插件、缓存回退和资源不足都会改变路径；资源复用成功时不会再次执行完整申请链。
// [中文导读] [AllReduce逐行 S783] 声明HcclExecOp接口：执行分发：复用/创建算法资源，关联Host与Device线程，按引擎发射或Host编排。
HcclResult HcclExecOp(
    // [中文导读] [AllReduce逐行 S784] 函数参数包含通信域句柄、算子参数、物理拓扑对象、算法名字，本行延续接口声明。
    HcclComm comm, OpParam& param, std::unique_ptr<TopoInfoWithNetLayerDetails>& topoInfo, std::string& algName,
    // [中文导读] [AllReduce逐行 S785] 函数参数包含图模式资源包，本行延续接口声明。
    const ResPackGraphMode& resPack)
// [中文导读] [AllReduce逐行 S786] 开始HcclExecOp的函数体。
{
    // [中文导读] [AllReduce逐行 S787] 记录本次Host执行入口的profiling起始时间。
    uint64_t beginTime = HcommGetProfilingSysCycleTime();
    // [中文导读] [AllReduce逐行 S788] 输出运行日志，记录HcclExecOp当前阶段和相关参数。
    HCCL_INFO("[HcclExecOp]Start to execute HcclExecOp. HcommGetProfilingSysCycleTime[%llu us]", beginTime);

    // [HCCL-ALGO-Plugin]
    // 算法执行阶段：
    // 若Selector()阶段已选中自定义算法，则改为调用PluginBroker的ExecuteAlg()完成通信，
    // 不再复用HCCL原有的资源申请/线程管理/executor->Orchestrate()等执行路径。
    // 调用失败或执行出错时直接返回HCCL_E_INTERNAL，不回退至HCCL原有执行逻辑。
    // 未编译HCCL_ALGO_PLUGIN_ENABLE宏时，本段整体不参与编译；
    // 又因Selector()中param.pluginSelected此时恒为false，执行流程与插件引入前完全一致。
// [中文导读] [AllReduce逐行 S797] 此段仅在HCCL_ALGO_PLUGIN_ENABLE启用的构建中编译。
#ifdef HCCL_ALGO_PLUGIN_ENABLE
    // [中文导读] [AllReduce逐行 S798] 已选中插件算法时走插件执行分支。
    if (param.pluginSelected) {
        // [中文导读] [AllReduce逐行 S799] 调用插件编排函数，失败直接返回而不继续内置算法。
        CHK_RET(ExecutePluginAlgorithm(comm, param, topoInfo.get(), algName));
        // [中文导读] [AllReduce逐行 S800] 记录插件算子的profiling统计。
        CHK_RET(ReportOpProfilingInfo(comm, param.opType, beginTime));
        // [中文导读] [AllReduce逐行 S801] 插件执行分支成功后直接返回。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S802] 结束条件if (param.pluginSelected)。
    }
// [中文导读] [AllReduce逐行 S803] 结束上述编译期条件控制的源码范围。
#endif

    // 当前通信域的某个算法回退过，则下次直接回退
    // [中文导读] 先查询该算法的历史回退上下文；命中时恢复已协商的引擎和算法，并重新进入执行枢纽。
    // [中文导读] [AllReduce逐行 S807] 初始化历史回退上下文地址为空。
    void* fallbackCtx = nullptr;
    // [中文导读] [AllReduce逐行 S808] 初始化历史回退上下文长度为零。
    uint64_t fallbackCtxSize = 0;
    // [中文导读] [AllReduce逐行 S809] 根据原算法名构造历史回退缓存tag。
    CHK_RET(SetOpParamFallbackTag(param, algName));
    // [中文导读] [AllReduce逐行 S810] 历史回退上下文命中时直接恢复上次协商后的算法配置。
    if (HcclEngineCtxGet(comm, param.fallbackTag, param.engine, &fallbackCtx, &fallbackCtxSize) == HCCL_SUCCESS) {
        // [中文导读] [AllReduce逐行 S811] 输出运行日志，记录HcclExecOp当前阶段和相关参数。
        HCCL_INFO("[HcclExecOp] Engine ctx exists, try to fallback.");
        // [中文导读] [AllReduce逐行 S812] 把上下文地址解释为FallbackCtxData对象。
        auto* ctxData = static_cast<FallbackCtxData*>(fallbackCtx);
        // [中文导读] [AllReduce逐行 S813] 从缓存取出已经回退到的算法名字。
        std::string newAlgName = ctxData->algName;
        // [中文导读] [AllReduce逐行 S814] 输出运行日志，记录HcclExecOp当前阶段和相关参数。
        HCCL_INFO(
            // [中文导读] [AllReduce逐行 S815] 补充日志格式：[HcclExecOp] Cached algo[%s], config[%u].", newAlgName.c_str()。
            "[HcclExecOp] Cached algo[%s], config[%u].", newAlgName.c_str(),
            // [中文导读] [AllReduce逐行 S816] 提供上述日志的实参：static_cast<uint32_t>(ctxData->opExecuteConfig。
            static_cast<uint32_t>(ctxData->opExecuteConfig));
        // [中文导读] [AllReduce逐行 S817] 恢复历史回退选择的执行配置。
        param.opExecuteConfig = ctxData->opExecuteConfig;
        // [中文导读] [AllReduce逐行 S818] 按恢复的执行配置重新设置引擎。
        CHK_RET(SetCommEngine(param));
        // [中文导读] [AllReduce逐行 S819] 按回退算法名字重新构造算法tag。
        CHK_RET(SetOpParamAlgTag(param, newAlgName));
        // [中文导读] [AllReduce逐行 S820] 递归进入HcclExecOp执行缓存的回退算法。
        CHK_RET(HcclExecOp(comm, param, topoInfo, newAlgName, resPack));
        // [中文导读] [AllReduce逐行 S821] 回退算法执行成功后结束当前调用层。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S822] 结束条件if (HcclEngineCtxGet(comm, param.fallbackTag, param.engine, &fallbackCtx, &fallbackCtxSize) == HCCL_SUCCESS)。
    }
    // 将算法名字放在param参数中
    // [中文导读] 把选中的算法名写入设备参数，并以通信域和执行模式构造资源关联标识。
    // [中文导读] [AllReduce逐行 S825] 把选择器返回的算法名字复制到设备参数固定字符串区。
    int result = sprintf_s(param.algName, sizeof(param.algName), "%s", algName.c_str());
    // [中文导读] [AllReduce逐行 S826] sprintf_s未成功写入名字时终止执行。
    if (result <= 0) {
        // [中文导读] [AllReduce逐行 S827] 输出错误日志，记录HcclExecOp当前阶段和相关参数。
        HCCL_ERROR("failed to fill param.algName");
        // [中文导读] [AllReduce逐行 S828] 算法名字写入失败返回内部错误。
        return HCCL_E_INTERNAL;
    // [中文导读] [AllReduce逐行 S829] 结束条件if (result <= 0)。
    }
    // 在原先的commName中添加执行模式，得到commModeTag
    // [中文导读] [AllReduce逐行 S831] 保存后续资源调用和Device展开使用的通信域句柄。
    param.hcclComm = comm;
    // [中文导读] [AllReduce逐行 S832] 判断当前是否是单算子OPBASE模式。
    bool isOpBase = param.opMode == OpMode::OPBASE;
    // [中文导读] [AllReduce逐行 S833] 根据执行模式选择通信域资源关联后缀。
    const char* opModeStr = isOpBase ? "_opbase" : "_offload";
    // [中文导读] [AllReduce逐行 S834] 用通信域名和模式后缀构造commModeTag。
    auto ret = sprintf_s(param.commModeTag, sizeof(param.commModeTag), "%s_%s", param.commName, opModeStr);
    // [中文导读] [AllReduce逐行 S835] 检查commModeTag字符串格式化是否成功。
    if (ret <= 0) {
        // [中文导读] [AllReduce逐行 S836] 输出错误日志，记录HcclExecOp当前阶段和相关参数。
        HCCL_ERROR("[%s] failed to fill param.commModeTag", __func__);
        // [中文导读] [AllReduce逐行 S837] 关联tag生成失败返回内部错误。
        return HCCL_E_INTERNAL;
    // [中文导读] [AllReduce逐行 S838] 结束条件if (ret <= 0)。
    }

    // [中文导读] 用算子类型和算法名取得 executor；缺少注册项时不能继续计算资源或编排任务。
    // [中文导读] [AllReduce逐行 S841] 按ALLREDUCE和AicpuAllReduceSoleMeshOneShot等名字查工厂，新建具体executor。
    std::unique_ptr<InsCollAlgBase> executor = CollAlgExecRegistryV2::Instance().GetAlgExec(param.opType, algName);
    // [中文导读] [AllReduce逐行 S842] 开始条件检查宏，条件成立时记录错误并返回下方指定错误码。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S843] 找不到工厂创建出的executor时记录错误并返回参数错误。
        executor.get() == nullptr, HCCL_ERROR("Fail to find executor for algName[%s]", algName.c_str()), HCCL_E_PARA);

    // 资源结构体
    // [中文导读] [AllReduce逐行 S846] 创建Host资源描述对象，稍后记录拓扑、CCL、线程及通道。
    std::unique_ptr<AlgResourceCtxSerializable> resCtxHost = std::make_unique<AlgResourceCtxSerializable>();
    // [中文导读] 将 Host 探测到的批传输能力写进资源上下文，设备展开据此决定批传输或逐片原语。
    // [中文导读] [AllReduce逐行 S848] 保存批传输接口是否可用，Device wrapper据此选择批传输或逐片原语。
    resCtxHost->isHcommBatchTransferOnThreadSupported = HcommIsSupportHcommBatchTransferOnThread();
    // 资源序列化结果
    // [中文导读] [AllReduce逐行 S850] 初始化Device资源上下文返回地址为空。
    void* resCtxSequence = nullptr;
    // [中文导读] [AllReduce逐行 S851] 初始化本次资源尚未复用标志为false。
    bool isResourceReused = false;

    // [中文导读] 将用户stream包装成CPU_TS Thread，再导出设备侧可引用的句柄。
    // [中文导读] “导出”建立跨Engine使用关系，不是新建一条相同的用户流，也不是复制张量。
    // [中文导读] [AllReduce逐行 S855] 初始化用户流CPU_TS线程句柄。
    ThreadHandle cpuTsThread{0};
    // [中文导读] [AllReduce逐行 S856] 初始化用户流导出到AICPU_TS的线程句柄。
    ThreadHandle exportedAicpuTsThread{0};
    // [中文导读] [AllReduce逐行 S857] AICPU_TS/CPU路径需要建立用户流与Device主线程的通知关系。
    if ((param.engine == COMM_ENGINE_AICPU_TS) || (param.engine == COMM_ENGINE_CPU)) {
        // [中文导读] [AllReduce逐行 S858] 将用户ACL stream包装为CPU_TS线程，配置Host侧通知容量。
        CHK_RET(HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, param.stream, CPU_TS_NOTIFY_NUM, &cpuTsThread));
        // Export cpuTsThread
        // [中文导读] [AllReduce逐行 S860] 把用户流线程导出为Device可引用的AICPU_TS句柄。
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &cpuTsThread, COMM_ENGINE_AICPU_TS, &exportedAicpuTsThread));
    // [中文导读] [AllReduce逐行 S861] 结束条件if ((param.engine == COMM_ENGINE_AICPU_TS) || (param.engine == COMM_ENGINE_CPU))。
    }

    // [中文导读] 资源层先复用再申请；只有资源不可用这一返回值触发算法回退，其它错误直接传递。
    // [中文导读] [AllReduce逐行 S864] 声明资源准备返回码，下一行调用会填入结果。
    auto resRet
        // [中文导读] [AllReduce逐行 S865] 复用或创建算法资源，返回Device资源序列化地址及复用标志。
        = HcclGetAlgRes(comm, param, executor, topoInfo.get(), resCtxHost, &resCtxSequence, isResourceReused, resPack);
    // [中文导读] [AllReduce逐行 S866] 仅资源不可用返回码触发算法回退。
    if (resRet == HCCL_E_UNAVAIL) {
        // [中文导读] [AllReduce逐行 S867] 输出警告日志，记录HcclExecOp当前阶段和相关参数。
        HCCL_WARNING("[HcclGetAlgRes] resource unavailable, try to fallback.");
        // [中文导读] [AllReduce逐行 S868] 重新选择并执行降级算法；该函数内部负责保存回退关系。
        CHK_RET(FallbackOp(comm, param, topoInfo, algName, resPack));
        // [中文导读] [AllReduce逐行 S869] 回退执行成功后当前分支直接返回。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S870] 上述条件不成立时进入替代分支。
    } else {
        // [中文导读] [AllReduce逐行 S871] 其它资源错误通过CHK_RET直接向上返回。
        CHK_RET(resRet);
    // [中文导读] [AllReduce逐行 S872] 结束条件} else。
    }

    // [中文导读] [AllReduce逐行 S874] 把资源复用结果写入Device参数，作为缓存有效性判断输入。
    param.cacheValid = isResourceReused;

    // [中文导读] [AllReduce逐行 S876] 输出运行日志，记录HcclExecOp当前阶段和相关参数。
    HCCL_CONFIG_INFO(
        // [中文导读] [AllReduce逐行 S877] 提供上述日志的实参，涉及算子参数、算法名字、算法关联tag。
        HCCL_ALG, "resourceReuse[%d] algName[%s] engine[%s] algTag[%s]", static_cast<int>(param.cacheValid),
        // [中文导读] [AllReduce逐行 S878] 提供上述日志的实参，涉及算子参数、算法名字、算法关联tag。
        param.algName, ENGINE_STR_MAP.at(param.opExecuteConfig), param.algTag);

    // Op注册
    // [中文导读] 注册本次算子的维测元信息，并把统计后的数据数量写回参数。
    // [中文导读] [AllReduce逐行 S882] 创建本次算子的维测元信息结构。
    HcclDfxOpInfoCompat hcclDfxOpInfo{};
    // [中文导读] [AllReduce逐行 S883] 把算子参数转换为维测信息，并关联用户流线程。
    CHK_RET(ConstructHcclDfxOpInfo(param, param.algTag, ALG_TAG_LENGTH, hcclDfxOpInfo, cpuTsThread));
    // [中文导读] [AllReduce逐行 S884] 把维测结构统计出的总数据量写回param。
    param.dataCount = hcclDfxOpInfo.dataCount;
    // [中文导读] [AllReduce逐行 S885] 向通信域登记算子维测信息。
    CHK_RET(HcclDfxRegOpInfoByCommId(param.commName, reinterpret_cast<void*>(&hcclDfxOpInfo)));
    // [中文导读] [AllReduce逐行 S886] 声明Device主线程导出到CPU_TS后的句柄。
    ThreadHandle exportedCpuTsThread;
    // [中文导读] [AllReduce逐行 S887] 声明Device算法主线程句柄。
    ThreadHandle mainThread;
    // [中文导读] [AllReduce逐行 S888] 声明该Device主线程的通知槽总容量。
    u32 notifyNumOnMainThread;
    // [中文导读] [AllReduce逐行 S889] AICPU_TS/CPU路径查询并导出算法主线程。
    if ((param.engine == COMM_ENGINE_AICPU_TS) || (param.engine == COMM_ENGINE_CPU)) {
        // 获取主流信息
        // [中文导读] 取得已有设备主 Thread 和通知容量，再导出到 CPU_TS 以建立用户流与设备流依赖。
        // [中文导读] [AllReduce逐行 S892] 从Host上下文读取算法主线程及通知容量。
        CHK_RET(GetMainThreadInfo(comm, param, mainThread, notifyNumOnMainThread));
        // Export mainThread
        // [中文导读] [AllReduce逐行 S894] 把算法Device主线程导出到CPU_TS，供Host用户流发送通知。
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &mainThread, COMM_ENGINE_CPU_TS, &exportedCpuTsThread));
        // cpuTsThread 添加到param里
        // [中文导读] [AllReduce逐行 S896] 把用户流导出后的Device侧句柄写进入口参数。
        param.opThread = exportedAicpuTsThread;
    // [中文导读] [AllReduce逐行 S897] 结束条件if ((param.engine == COMM_ENGINE_AICPU_TS) || (param.engine == COMM_ENGINE_CPU))。
    }

    // [中文导读] AICPU交设备入口编排，AIV准备核数后执行其缓存/发射逻辑，CCU在Host组织Kernel发射。
    // [中文导读] 三类Engine共享部分控制面资源，但不能据此把Thread原语、AIV核内操作与CCU指令混用。
    // 算法执行
    // [中文导读] [AllReduce逐行 S902] AICPU_TS或CPU引擎在Device入口展开算法任务。
    if ((param.engine == COMM_ENGINE_AICPU_TS) || (param.engine == COMM_ENGINE_CPU)) {
        // [中文导读] AICPU/CPU 分支恢复展开 Thread，并把需要的 Thread 纳入当前流捕获关系。
        // [中文导读] [AllReduce逐行 S904] 声明用于提前展开和保序的Host展开线程句柄。
        ThreadHandle unfoldThread;
        // [中文导读] [AllReduce逐行 S905] 读取通信域对应的Host展开线程。
        CHK_RET(GetUnfoldThreadInfo(comm, param, unfoldThread));
        // 根据主流的捕获状态决定展开流的状态
        // [中文导读] [AllReduce逐行 S907] 按用户流捕获状态关联算法主线程和展开线程。
        CHK_RET(CaptureSlaveStreams(comm, param.stream, {mainThread, unfoldThread}, param.isCapture));
        // aicpu task cache使能
        // [中文导读] 将 AICPU 任务缓存配置传入设备入口；设备端还会检查算法和资源是否满足缓存条件。
        // [中文导读] [AllReduce逐行 S910] 读取环境AICPU task cache开关写入入口参数。
        param.aicpuCacheEnable = GetExternalInputHcclAicpuCacheEnable();
        // [中文导读] [AllReduce逐行 S911] 开始调用AICPU入口发射包装器，后两行补充线程/资源参数。
        CHK_RET(HcclAicpuKernelEntranceLaunch(
            // [中文导读] [AllReduce逐行 S912] 传入Host用户线程、Device主线程导出句柄、通知容量和资源地址。
            comm, param, cpuTsThread, exportedCpuTsThread, notifyNumOnMainThread, resCtxSequence, algName,
            // [中文导读] [AllReduce逐行 S913] 补充展开线程句柄并结束发射调用，错误由CHK_RET返回。
            unfoldThread));
    // [中文导读] [AllReduce逐行 S914] AIV执行路径先确定核数，再使用AIV缓存/发射逻辑。
    } else if (param.engine == COMM_ENGINE_AIV) {
        // [中文导读] AIV 分支使用其专属资源上下文，先确定核数限制，再进入缓存或实际发射逻辑。
        // [中文导读] [AllReduce逐行 S916] 记录AIV分支开始时间。
        uint64_t aivBeginTime = HcommGetProfilingSysCycleTime();
        // [中文导读] [AllReduce逐行 S917] AIV资源地址写入参数，供AIV编排读取。
        param.resCtx = resCtxSequence;
        // [中文导读] [AllReduce逐行 S918] 将AIV资源地址解释为Host资源对象引用。
        AlgResourceCtxSerializable& aivResCtxHost = *static_cast<AlgResourceCtxSerializable*>(resCtxSequence);
        // [中文导读] [AllReduce逐行 S919] 确定AIV核数等入口设置。
        CHK_RET(HcclAivKernelEntranceLaunch(comm, param, topoInfo, aivResCtxHost));
        // [中文导读] [AllReduce逐行 S920] 执行AIV缓存逻辑或实际AIV编排发射。
        CHK_RET(ExecuteAivCacheLogic(comm, param, algName, executor, aivResCtxHost));
        // [中文导读] [AllReduce逐行 S921] 上报AIV kernel profiling。
        CHK_RET(HcclReportAivKernel(comm, aivBeginTime));
    // [中文导读] [AllReduce逐行 S922] CCU执行路径在Host恢复资源并直接调用执行器编排。
    } else if (param.engine == COMM_ENGINE_CCU) {
        // [中文导读] [AllReduce逐行 S923] CCU资源复用时需要从序列化上下文恢复Host对象。
        if (isResourceReused) {
            // 复用资源，则需从engineCtx取得res，进行反序列化
            // [中文导读] CCU 复用路径从缓存的序列化描述恢复 Host 资源，再用本次用户流重新包装主 Thread。
            // [中文导读] [AllReduce逐行 S926] 将CCU缓存资源地址解释为字节指针。
            char* ctx = static_cast<char*>(resCtxSequence);
            // [中文导读] [AllReduce逐行 S927] 按param.ctxSize构造序列化字节序列。
            std::vector<char> seq(ctx, ctx + param.ctxSize);
            // [中文导读] [AllReduce逐行 S928] 恢复CCU资源对象，其中含线程及kernel信息。
            resCtxHost->DeSerialize(seq);
            // 覆盖主流
            // [中文导读] [AllReduce逐行 S930] 声明当前用户流所对应的CCU主线程句柄。
            ThreadHandle thread;
            // [中文导读] [AllReduce逐行 S931] 从本次用户流获取CCU线程，下一行补充通知数和输出句柄。
            CHK_RET(HcclThreadAcquireWithStream(
                // [中文导读] [AllReduce逐行 S932] 使用恢复对象的主线程通知数，输出本次流的线程句柄。
                comm, param.engine, param.stream, resCtxHost->notifyNumOnMainThread, &thread));
            // [中文导读] [AllReduce逐行 S933] 防止复用对象没有线程时覆盖不存在的第一个元素。
            if (resCtxHost->threads.empty()) {
                // [中文导读] [AllReduce逐行 S934] 输出错误日志，记录HcclExecOp当前阶段和相关参数。
                HCCL_ERROR("[%s] reused threads is empty after DeSerialize, cannot overwrite main thread.", __func__);
                // [中文导读] [AllReduce逐行 S935] 缺少复用主线程返回资源不可用。
                return HCCL_E_UNAVAIL;
            // [中文导读] [AllReduce逐行 S936] 结束条件if (resCtxHost->threads.empty())。
            }
            // [中文导读] [AllReduce逐行 S937] 用本次用户流线程替换缓存对象原主线程。
            resCtxHost->threads[0] = thread;
            // 图模式要全部覆盖
            // [中文导读] 图模式复用 CCU 资源时，还需按本次外部从流和临时内存重新覆盖图资源。
            // [中文导读] [AllReduce逐行 S940] OFFLOAD图模式需要按本次GE资源包覆盖复用资源。
            if (param.opMode != OpMode::OPBASE) {
                // [中文导读] [AllReduce逐行 S941] 使用本次GE从流/临时内存更新CCU复用对象。
                CHK_RET(GeReuseResource(comm, param, executor, resCtxHost, topoInfo.get(), resPack));
            // [中文导读] [AllReduce逐行 S942] 结束条件if (param.opMode != OpMode::OPBASE)。
            }
        // [中文导读] [AllReduce逐行 S943] 结束条件if (isResourceReused)。
        }
        // [中文导读] 有从 Thread 的 CCU 算法先建立流捕获关系，再在 Host 调用 executor 编排。
        // [中文导读] [AllReduce逐行 S945] 有从线程的CCU算法需登记流捕获关系。
        if (resCtxHost->slaveThreadNum > 0) {
            // [中文导读] [AllReduce逐行 S946] 把所有CCU线程纳入当前用户流的捕获关系。
            CHK_RET(CaptureSlaveStreams(comm, param.stream, resCtxHost->threads, param.isCapture));
        // [中文导读] [AllReduce逐行 S947] 结束条件if (resCtxHost->slaveThreadNum > 0)。
        }
        // [中文导读] [AllReduce逐行 S948] 在Host调用CCU执行器编排。
        CHK_RET(executor->Orchestrate(param, *resCtxHost));
    // [中文导读] [AllReduce逐行 S949] 上述条件不成立时进入替代分支。
    } else {
        // [中文导读] [AllReduce逐行 S950] 其它引擎复用时同样从序列化数据恢复Host资源对象。
        if (isResourceReused) {
            // 复用资源，则需从engineCtx取得res，进行反序列化
            // [中文导读] [AllReduce逐行 S952] 获取缓存上下文的字节起点。
            char* ctx = static_cast<char*>(resCtxSequence);
            // [中文导读] [AllReduce逐行 S953] 按ctxSize构造缓存字节序列。
            std::vector<char> seq(ctx, ctx + param.ctxSize);
            // [中文导读] [AllReduce逐行 S954] 反序列化为Host资源对象。
            resCtxHost->DeSerialize(seq);
        // [中文导读] [AllReduce逐行 S955] 结束条件if (isResourceReused)。
        }
        // [中文导读] [AllReduce逐行 S956] 其它引擎在Host调用执行器编排。
        CHK_RET(executor->Orchestrate(param, *resCtxHost));
    // [中文导读] [AllReduce逐行 S957] 结束条件} else。
    }
    // op上报
    // [中文导读] [AllReduce逐行 S959] 上报整个算子Host执行阶段profiling。
    CHK_RET(ReportOpProfilingInfo(comm, param.opType, beginTime));
    // [中文导读] [AllReduce逐行 S960] 输出运行日志，记录HcclExecOp当前阶段和相关参数。
    HCCL_INFO("Execute HcclExecOp success.");
    // [中文导读] [AllReduce逐行 S961] 执行分发：复用/创建算法资源，关联Host与Device线程，按引擎发射或Host编排处理完成，返回成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S962] 结束HcclExecOp函数体。
}

HcclResult GeReuseResource(
    HcclComm comm, OpParam& param, std::unique_ptr<InsCollAlgBase>& executor,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, TopoInfoWithNetLayerDetails* topoInfo,
    const ResPackGraphMode& resPack)
{
    // 计算AlgHierarchyInfo
    AlgHierarchyInfoForAllLevel algHierarchyInfo; // 分级通信域信息{localRankId, localRankSize}
    AlgAttrs algAttrs = executor->GetAlgoMeta(std::string(param.algName));
    CHK_RET(executor->CalcAlgHierarchyInfoV2(topoInfo, algHierarchyInfo, algAttrs));
    // 资源计算
    AlgResourceRequest resRequest;
    CHK_RET(executor->CalcRes(comm, param, topoInfo, algHierarchyInfo, resRequest));

    u32 maxNotifyNum = 0;
    for (u32 i = 0; i < resRequest.notifyNumPerThread.size(); i++) {
        if (resRequest.notifyNumPerThread[i] > maxNotifyNum) {
            maxNotifyNum = resRequest.notifyNumPerThread[i];
        }
    }

    u32 threadNum = resRequest.slaveThreadNum;
    u32 slaveStreams = resPack.streams.size();
    if (threadNum > slaveStreams) {
        HCCL_ERROR("[%s] threadNum[%u] exceeds slaveStreams[%u].", __func__, threadNum, slaveStreams);
        return HCCL_E_UNAVAIL;
    }
    if (resCtxHost->threads.size() < static_cast<size_t>(threadNum) + 1) {
        HCCL_ERROR(
            "[%s] reused threads size[%zu] is less than threadNum+1[%u].", __func__, resCtxHost->threads.size(),
            threadNum + 1);
        return HCCL_E_UNAVAIL;
    }
    for (u32 i = 0; i < threadNum; i++) {
        ThreadHandle slaveThread;
        CHK_RET(HcclThreadAcquireWithStream(comm, param.engine, resPack.streams[i], maxNotifyNum, &slaveThread));
        resCtxHost->threads[i + 1] = slaveThread;
    }
    return HCCL_SUCCESS;
}

static HcclResult GetUnfoldStream(HcclComm comm, OpParam& param, ThreadHandle unfoldThread, aclrtStream& resolvedStream)
{
    void* unfoldStream = nullptr;
    auto& HcclThreadResGetInfoFunc = ops_hccl::DlHcommFunction::GetInstance();
    HcclResult ret;
    if (!HcclThreadResGetInfoFunc.dlHcclThreadResGetInfo || param.opMode == OpMode::OFFLOAD) { // 不走提前展开
        resolvedStream = param.stream;
    } else {
        ret = HcclThreadResGetInfoFunc.dlHcclThreadResGetInfo(comm, unfoldThread, 0, sizeof(void*), &unfoldStream);
        if (ret == HCCL_E_NOT_SUPPORT) {
            resolvedStream = param.stream;
        } else if (ret != HCCL_SUCCESS) {
            resolvedStream = param.stream;
            return ret;
        } else {
            resolvedStream = unfoldStream;
        }
    }
    return HCCL_SUCCESS;
}

// [中文导读] Host侧建立输入就绪与执行保序依赖，发射AICPU入口，再在用户Thread上排入完成等待。
// [中文导读] Record/Wait是在任务流中表达依赖，不能理解成Host在每个API返回时都已等到设备完成。
// [中文导读] 具备新接口能力的Send/Recv另走前面的P2P分支，AllToAll沿后面的通用下发路径。
// [中文导读] [AllReduce逐行 S1028] 声明HcclAicpuKernelEntranceLaunch接口：排入Host输入通知、执行两阶段保序、发射AICPU入口，最后排入Host等待Device结果。
HcclResult HcclAicpuKernelEntranceLaunch(
    // [中文导读] [AllReduce逐行 S1029] 函数参数包含通信域句柄、算子参数、用户流CPU_TS线程、Device主线程导出到Host的句柄，本行延续接口声明。
    HcclComm comm, OpParam& param, ThreadHandle cpuTsThread, ThreadHandle exportedCpuTsThread,
    // [中文导读] [AllReduce逐行 S1030] 函数参数包含算法名字、Device资源序列化地址输出、Host展开线程句柄、Device主线程通知容量，本行延续接口声明。
    u32 notifyNumOnMainThread, void* resCtxSequence, std::string& algName, ThreadHandle unfoldThread)
// [中文导读] [AllReduce逐行 S1031] 开始HcclAicpuKernelEntranceLaunch的函数体。
{
    // [中文导读] [AllReduce逐行 S1032] 输出调试日志，记录HcclAicpuKernelEntranceLaunch当前阶段和相关参数。
    HCCL_DEBUG("[HcclAicpuKernelEntranceLaunch]start to run aicpu kernel");
    // [中文导读] [AllReduce逐行 S1033] algName参数在本包装器中未直接使用，显式标记避免未使用告警。
    (void)algName;
    // 当前aicpu launch接口只能有一个输入参数，将Context指针放在param参数中
    // [中文导读] 把序列化资源地址合入单一 kernel 参数，并设置用户流等待设备完成的通知槽。
    // [中文导读] [AllReduce逐行 S1036] 将Device可读资源地址写入单一kernel参数。
    param.resCtx = resCtxSequence;
    // [中文导读] [AllReduce逐行 S1037] 指定用户流等待Device完成的通知槽。
    param.aicpuRecordCpuIdx = HOST_WAIT_AICPU_NOTIFYIDX;

    // [中文导读] [AllReduce逐行 S1039] CPU/DPU引擎需先注册Host DPU kernel回调。
    if (param.engine == COMM_ENGINE_CPU) {
        // 注册dpu回调函数
        // [中文导读] [AllReduce逐行 S1041] 将HcclLaunchDPUKernel登记到当前通信域/算法tag。
        CHK_RET(static_cast<HcclResult>(HcclTaskRegister(comm, param.algTag, HcclLaunchDPUKernel)));
    // [中文导读] [AllReduce逐行 S1042] 结束条件if (param.engine == COMM_ENGINE_CPU)。
    }

    // [中文导读] 仅具备专用发射接口的 Send/Recv 使用 P2P 描述分支；普通集合通信沿后面的通用下发路径。
    // [中文导读] [AllReduce逐行 S1045] 专用AICPU发射能力存在时，检查是否是点对点Send/Recv。
    if (HcommIsSupportHcclAicpuKernelLaunch()
        // [中文导读] [AllReduce逐行 S1046] 仅Send或Receive满足此专用分支；AllReduce走后面的通用kernel入口。
        && (param.opType == HcclCMDType::HCCL_CMD_SEND || param.opType == HcclCMDType::HCCL_CMD_RECEIVE)) {
        // [中文导读] [AllReduce逐行 S1047] 输出运行日志，记录HcclAicpuKernelEntranceLaunch当前阶段和相关参数。
        HCCL_INFO(
            // [中文导读] [AllReduce逐行 S1048] 补充日志格式：[HcclAicpuKernelEntranceLaunch] P2P opType[%d], use HcclAicpuKernelLaunch。
            "[HcclAicpuKernelEntranceLaunch] P2P opType[%d], use HcclAicpuKernelLaunch",
            // [中文导读] [AllReduce逐行 S1049] 提供上述日志的实参，涉及算子参数。
            static_cast<int>(param.opType));

        // 构造 HcclOpDesc
        // [中文导读] [AllReduce逐行 S1052] 声明点对点算子描述，普通AllReduce不使用它。
        HcclOpDesc opInfo;

        // [中文导读] [AllReduce逐行 S1054] 清零P2P描述，安全函数失败由宏返回。
        CHK_SAFETY_FUNC_RET(memset_s(&opInfo, sizeof(HcclOpDesc), 0, sizeof(HcclOpDesc)));
        // [中文导读] [AllReduce逐行 S1055] 设置描述类型1代表P2P。
        opInfo.opDescType = 1; // 1: P2P

        // [中文导读] [AllReduce逐行 S1057] 根据发送或接收命令选择P2P算子名字。
        std::string opNameStr = (param.opType == HcclCMDType::HCCL_CMD_SEND) ? "HcclSend" : "HcclRecv";
        // [中文导读] [AllReduce逐行 S1058] 开始安全复制P2P算子名字。
        CHK_SAFETY_FUNC_RET(
            // [中文导读] [AllReduce逐行 S1059] 把P2P名字写进固定长度opName区。
            strncpy_s(opInfo.opName, HCCL_OP_DESC_OP_NAME_MAX_LEN, opNameStr.c_str(), opNameStr.size()));

        // [中文导读] [AllReduce逐行 S1061] Send使用输入地址，Receive使用输出地址作为P2P数据缓冲区。
        opInfo.p2p.buffer = (param.opType == HcclCMDType::HCCL_CMD_SEND) ? param.inputPtr : param.outputPtr;
        // [中文导读] [AllReduce逐行 S1062] 设置P2P命令类型。
        opInfo.p2p.cmdType = param.opType;
        // [中文导读] [AllReduce逐行 S1063] 设置P2P数据类型。
        opInfo.p2p.dataType = param.DataDes.dataType;
        // [中文导读] [AllReduce逐行 S1064] 设置P2P元素数量。
        opInfo.p2p.count = param.DataDes.count;
        // [中文导读] [AllReduce逐行 S1065] 设置P2P对端rank。
        opInfo.p2p.remoteRank = param.sendRecvRemoteRank;
        // [中文导读] [AllReduce逐行 S1066] 声明解析展开线程后得到的ACL流句柄。
        aclrtStream resolvedStream;
        // [中文导读] [AllReduce逐行 S1067] 尝试查展开流，但本行显式忽略返回值。
        (void)GetUnfoldStream(comm, param, unfoldThread, resolvedStream);
        // [中文导读] [AllReduce逐行 S1068] 输出运行日志，记录HcclAicpuKernelEntranceLaunch当前阶段和相关参数。
        HCCL_INFO("unfoldThread[%llu]", unfoldThread);

        // [中文导读] [AllReduce逐行 S1070] 将解析的展开流放进P2P描述。
        opInfo.p2p.unfoldStream = resolvedStream;
        // 构造 HcclKernelFuncInfo
        // [中文导读] [AllReduce逐行 S1072] 声明P2P kernel函数信息描述。
        HcclKernelFuncInfo funcInfo;
        // [中文导读] [AllReduce逐行 S1073] 安全清零P2P kernel信息。
        CHK_SAFETY_FUNC_RET(memset_s(&funcInfo, sizeof(HcclKernelFuncInfo), 0, sizeof(HcclKernelFuncInfo)));

        // [中文导读] [AllReduce逐行 S1075] 指定AICPU kernel所在动态库名字。
        int soRet = sprintf_s(funcInfo.kernelSoName, sizeof(funcInfo.kernelSoName), "libscatter_aicpu_kernel.so");
        // [中文导读] [AllReduce逐行 S1076] 库名写入失败返回内部错误。
        CHK_PRT_RET(soRet <= 0, HCCL_ERROR("[%s] failed to fill kernelSoName", __func__), HCCL_E_INTERNAL);

        // [中文导读] [AllReduce逐行 S1078] 指定P2P专用kernel入口名字。
        int funcRet = sprintf_s(funcInfo.kernelFuncName, sizeof(funcInfo.kernelFuncName), "HcclLaunchP2pAicpuKernel");
        // [中文导读] [AllReduce逐行 S1079] 入口名字写入失败返回内部错误。
        CHK_PRT_RET(funcRet <= 0, HCCL_ERROR("[%s] failed to fill kernelFuncName", __func__), HCCL_E_INTERNAL);

        // 获取 aicpuThreadHandle
        // [中文导读] [AllReduce逐行 S1082] 声明P2P发射需要的Device主Thread句柄。
        ThreadHandle aicpuThreadHandle;
        // [中文导读] [AllReduce逐行 S1083] 声明P2P主线程通知数。
        u32 mainNotifyNum;
        // [中文导读] [AllReduce逐行 S1084] 查询P2P算法主Thread及通知容量。
        CHK_RET(GetMainThreadInfo(comm, param, aicpuThreadHandle, mainNotifyNum));

        // 调用 HcclAicpuKernelLaunch
        // [中文导读] [AllReduce逐行 S1087] 固定入口参数地址指向本次OpParam。
        void* args = &param;
        // [中文导读] [AllReduce逐行 S1088] kernel参数大小包含OpParam及尾部变长描述字节。
        uint32_t argSize = sizeof(OpParam) + param.varMemSize;

        // [中文导读] [AllReduce逐行 S1090] 写入P2P kernel参数地址。
        funcInfo.args = args;
        // [中文导读] [AllReduce逐行 S1091] 写入P2P kernel参数总长度。
        funcInfo.argSize = argSize;

        // [中文导读] [AllReduce逐行 S1093] 声明P2P kernel launch配置。
        HcclKernelLaunchCfg kernelLaunchCfg;
        // [中文导读] [AllReduce逐行 S1094] 从算子执行超时派生AICPU超时。
        AicpuTimeout timeout = DeriveAicpuTimeout(param.opConfig.execTimeout);
        // [中文导读] [AllReduce逐行 S1095] 声明P2P kernel启动超时变量。
        u16 kernelLaunchTimeout
            // [中文导读] [AllReduce逐行 S1096] 若运行时支持默认超时机制，采用统一派生值。
            = IsHcommDefaultTimeoutSupported() ?
                  // [中文导读] [AllReduce逐行 S1097] 默认超时能力成立时取派生kernelLaunchTimeout。
                  timeout.kernelLaunchTimeout :
                  // [中文导读] [AllReduce逐行 S1098] 不支持默认超时时按execTimeout加启动偏移并转换为u16。
                  ToKernelLaunchTimeout(AddAicpuTimeoutOffset(param.opConfig.execTimeout, KERNEL_TIMEOUT_OFFSET));
        // [中文导读] [AllReduce逐行 S1099] 把启动超时写入P2P kernel发射配置。
        kernelLaunchCfg.timeOut = kernelLaunchTimeout;

        // [中文导读] [AllReduce逐行 S1101] 调用HCOMM专用P2P AICPU发射API。
        CHK_RET(HcclAicpuKernelLaunch(comm, &opInfo, &funcInfo, aicpuThreadHandle, param.stream, &kernelLaunchCfg));

        // [中文导读] [AllReduce逐行 S1103] 输出运行日志，记录HcclAicpuKernelEntranceLaunch当前阶段和相关参数。
        HCCL_INFO("[HcclAicpuKernelEntranceLaunch] P2P launch success, algTag[%s]", param.algTag);
        // [中文导读] [AllReduce逐行 S1104] P2P成功后直接结束，不执行后面的集合通信通用路径。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S1105] 结束代码块。
    }

    // [中文导读] CPU_TS队列执行到此通知时，设备主Thread才可跨过输入等待，避免读取尚未准备好的数据。
    // Host stream通知Device主thread，使用主流上idx最大的notify
    // [中文导读] [AllReduce逐行 S1109] 开始记录Host输入就绪通知，下一行补齐目标线程和槽位。
    CHK_RET(static_cast<HcclResult>(
        // [中文导读] [AllReduce逐行 S1110] 在用户流线程记录通知到Device主线程的最后一个通知槽。
        HcommThreadNotifyRecordOnThread(cpuTsThread, exportedCpuTsThread, notifyNumOnMainThread - 1)));

    // OrderLaunch第一阶段
    // 获取执行超时时间
    // [中文导读] [AllReduce逐行 S1114] 读取保序操作所需执行超时。
    u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();

    // [中文导读] 根据 ACL 图捕获、GE 图模式或单算子模式选择保序机制，保证展开与核发射的先后关系。
    // [中文导读] [AllReduce逐行 S1117] 依据用户流是否图捕获选择ACLGRAPH保序。
    OrderLaunchMode launchMode = param.isCapture ?
                                     // [中文导读] [AllReduce逐行 S1118] 图捕获时使用ACLGRAPH事件链。
                                     OrderLaunchMode::ORDER_LAUNCH_ACLGRAPH :
                                     // [中文导读] [AllReduce逐行 S1119] 非捕获时根据OFFLOAD决定GE保序或OPBASE保序。
                                     (param.opMode == OpMode::OFFLOAD ? OrderLaunchMode::ORDER_LAUNCH_GE :
                                                                        // [中文导读] [AllReduce逐行 S1120] 普通单算子例使用ORDER_LAUNCH_OPBASE。
                                                                        OrderLaunchMode::ORDER_LAUNCH_OPBASE);

    // [中文导读] ACL 图保序需要两个临时事件，由 guard 管理生命周期；其它模式使用已有线程通知。
    // [中文导读] [AllReduce逐行 S1123] 声明第一个保序事件的RAII guard。
    HcclRtEventGuard event0Guard;
    // [中文导读] [AllReduce逐行 S1124] 声明第二个保序事件的RAII guard。
    HcclRtEventGuard event1Guard;
    // [中文导读] [AllReduce逐行 S1125] ACLGRAPH模式才创建两阶段保序事件。
    if (launchMode == OrderLaunchMode::ORDER_LAUNCH_ACLGRAPH) {
        // [中文导读] [AllReduce逐行 S1126] 创建第一阶段保序事件。
        CHK_RET(event0Guard.Create());
        // [中文导读] [AllReduce逐行 S1127] 创建第二阶段保序事件。
        CHK_RET(event1Guard.Create());
    // [中文导读] [AllReduce逐行 S1128] 结束条件if (launchMode == OrderLaunchMode::ORDER_LAUNCH_ACLGRAPH)。
    }
    // [中文导读] 第一阶段在发射 kernel 前把本次展开纳入保序链，避免后续算子提前展开。
    // [中文导读] [AllReduce逐行 S1130] 开始第一阶段Host保序调用。
    CHK_RET(HcclOrderLaunchToOrderStream(
        // [中文导读] [AllReduce逐行 S1131] 把本次展开线程纳入OrderStream顺序，传入Notify槽/事件及超时。
        comm, param, unfoldThread, ORDER_UNFOLD_THREAD_NOTIFY_IDX, execTimeout, launchMode, event0Guard.Get()));

    // AicpuKernel report
    // [中文导读] [AllReduce逐行 S1134] 记录AICPU发射开始时间用于profiling。
    uint64_t beginTime = HcommGetProfilingSysCycleTime();
    // [中文导读] 参数和输入依赖准备好后发射 AICPU 入口，再建立保序链的第二阶段。
    // [中文导读] [AllReduce逐行 S1136] 发射已加载二进制中的HcclLaunchAicpuKernel入口。
    CHK_RET(AicpuKernelLaunch(comm, param, unfoldThread));
    // [中文导读] [AllReduce逐行 S1137] 检查通信域句柄非空。
    CHK_PTR_NULL(comm);

    // OrderLaunch第二阶段
    // [中文导读] [AllReduce逐行 S1140] 开始第二阶段kernel保序调用。
    CHK_RET(HcclOrderLaunchToKernelStream(
        // [中文导读] [AllReduce逐行 S1141] 通过Host保序通知/事件建立本次kernel与后续算子的顺序关系。
        comm, unfoldThread, HOST_ORDER_THREAD_NOTIFY_IDX, execTimeout, launchMode, event1Guard.Get()));

    // [中文导读] [AllReduce逐行 S1143] 记录实际发射的kernel入口名字。
    std::string kernelName = "HcclLaunchAicpuKernel";
    // [中文导读] [AllReduce逐行 S1144] 为profiling接口取得可写类型的入口名字指针。
    char* kernelNameCStr = const_cast<char*>(kernelName.c_str());
    // [中文导读] [AllReduce逐行 S1145] 上报本次AICPU kernel发射profiling信息。
    HcclResult ret = HcclReportAicpuKernel(comm, beginTime, kernelNameCStr);
    // [中文导读] [AllReduce逐行 S1146] profiling上报失败时直接返回其错误。
    if (ret != HCCL_SUCCESS) {
        // [中文导读] [AllReduce逐行 S1147] 输出错误日志，记录HcclAicpuKernelEntranceLaunch当前阶段和相关参数。
        HCCL_ERROR(
            // [中文导读] [AllReduce逐行 S1148] 补充日志格式：[HcclAicpuKernelEntranceLaunch] HcclReportAicpuKernel failed, beginTime %lu, kernelNameCStr %s, ret %d。
            "[HcclAicpuKernelEntranceLaunch] HcclReportAicpuKernel failed, beginTime %lu, kernelNameCStr %s, ret %d ",
            // [中文导读] [AllReduce逐行 S1149] 提供上述日志的实参：beginTime, kernelNameCStr, ret。
            beginTime, kernelNameCStr, ret);
        // [中文导读] [AllReduce逐行 S1150] profiling上报失败时向上返回该错误码。
        return ret;
    // [中文导读] [AllReduce逐行 S1151] 结束条件if (ret != HCCL_SUCCESS)。
    }
    // [中文导读] 这里保护用户流中后续任务对通信输出的消费；最终通知由设备编排结束路径排入。
    // Host stream等待Device的通知
    // [中文导读] [AllReduce逐行 S1154] 从本次执行超时派生Host通知等待超时。
    AicpuTimeout timeout = DeriveAicpuTimeout(param.opConfig.execTimeout);
    // [中文导读] 用户流等待设备完成使用独立超时派生值，通知槽与设备末尾 Record 配对。
    // [中文导读] [AllReduce逐行 S1156] 根据运行时默认超时能力选择Host等待时长。
    u32 hostNotifyWaitTime = IsHcommDefaultTimeoutSupported() ?
                                 // [中文导读] [AllReduce逐行 S1157] 默认超时机制使用派生的hostNotifyTimeout。
                                 timeout.hostNotifyTimeout :
                                 // [中文导读] [AllReduce逐行 S1158] 不支持默认超时时使用执行超时加Host通知偏移。
                                 AddAicpuTimeoutOffset(param.opConfig.execTimeout, HOST_NOTIFY_TIMEOUT_OFFSET);
    // [中文导读] [AllReduce逐行 S1159] 有设置通知默认超时能力时才更新该配置。
    if (HcommIsSupportHcommSetNotifyWaitTimeOut()) {
        // [中文导读] [AllReduce逐行 S1160] 设置本次用户流等待Device结果的默认通知超时。
        CHK_RET(HcclSetNotifyWaitTimeOut(hostNotifyWaitTime));
    // [中文导读] [AllReduce逐行 S1161] 结束条件if (HcommIsSupportHcommSetNotifyWaitTimeOut())。
    }
    // [中文导读] [AllReduce逐行 S1162] 在用户CPU_TS流中排入对Device完成通知的等待。
    CHK_RET(HcclThreadNotifyWaitOnThreadDefault(cpuTsThread, param.aicpuRecordCpuIdx, hostNotifyWaitTime));

    // [中文导读] [AllReduce逐行 S1164] 发射包装及流依赖建立成功后返回；返回不意味着硬件已完成通信。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S1165] 结束HcclAicpuKernelEntranceLaunch函数体。
}

// [中文导读] [AllReduce逐行 S1167] 声明AicpuKernelLaunch接口：从已加载二进制取得入口，复制OpParam参数并通过ACL发射到展开流或用户流。
HcclResult AicpuKernelLaunch(HcclComm comm, OpParam& param, ThreadHandle unfoldThread)
// [中文导读] [AllReduce逐行 S1168] 开始AicpuKernelLaunch的函数体。
{
    // [中文导读] [AllReduce逐行 S1169] 指定本次二进制kernel入口为HcclLaunchAicpuKernel。
    std::string kernelName = "HcclLaunchAicpuKernel";
    // [中文导读] [AllReduce逐行 S1170] 声明ACL函数句柄输出变量。
    aclrtFuncHandle funcHandle;
    // [中文导读] [AllReduce逐行 S1171] 声明ACL kernel参数句柄输出变量。
    aclrtArgsHandle argsHandle;
    // 注意，目前开源HCCL加载AICPU kernel使用的是从json文件加载
    // 详见load_kernel.cc中的LoadAICPUKernel函数，且只实现了scatter的，先共用scatter的
    // [中文导读] 从已加载的二进制 kernel 中取得入口句柄，为本次调用创建运行时参数句柄。
    // [中文导读] [AllReduce逐行 S1175] 声明ACL运行时调用返回码。
    aclError ret
        // [中文导读] [AllReduce逐行 S1176] 原子读取已加载的二进制句柄，从中查找AICPU入口函数。
        = aclrtBinaryGetFunction(g_binKernelHandle.load(std::memory_order_acquire), kernelName.c_str(), &funcHandle);
    // [中文导读] [AllReduce逐行 S1177] 开始条件检查宏，条件成立时记录错误并返回下方指定错误码。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S1178] ACL运行时返回码非成功时触发下面的日志和错误返回。
        ret != ACL_SUCCESS,
        // [中文导读] [AllReduce逐行 S1179] 输出错误日志，记录AicpuKernelLaunch当前阶段和相关参数。
        HCCL_ERROR(
            // [中文导读] [AllReduce逐行 S1180] 补充日志格式：[aclrtBinaryGetFunction]errNo[0x%016llx] get func handle failed。
            "[aclrtBinaryGetFunction]errNo[0x%016llx] get func handle failed, "
            // [中文导读] [AllReduce逐行 S1181] 补充日志格式：kernelName:%s。
            "kernelName:%s",
            // [中文导读] [AllReduce逐行 S1182] 提供上述日志的实参：ret, kernelName.c_str()),。
            ret, kernelName.c_str()),
        // [中文导读] [AllReduce逐行 S1183] 提供上述日志的实参：HCCL_E_RUNTIME。
        HCCL_E_RUNTIME);
    // [中文导读] [AllReduce逐行 S1184] 为该入口函数初始化kernel参数句柄。
    ret = aclrtKernelArgsInit(funcHandle, &argsHandle);
    // [中文导读] [AllReduce逐行 S1185] 开始条件检查宏，条件成立时记录错误并返回下方指定错误码。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S1186] ACL运行时返回码非成功时触发下面的日志和错误返回。
        ret != ACL_SUCCESS,
        // [中文导读] [AllReduce逐行 S1187] 输出错误日志，记录AicpuKernelLaunch当前阶段和相关参数。
        HCCL_ERROR(
            // [中文导读] [AllReduce逐行 S1188] 补充日志格式：[aclrtKernelArgsInit]errNo[0x%016llx] args init failed。
            "[aclrtKernelArgsInit]errNo[0x%016llx] args init failed, "
            // [中文导读] [AllReduce逐行 S1189] 补充日志格式：kernelName:%s。
            "kernelName:%s",
            // [中文导读] [AllReduce逐行 S1190] 提供上述日志的实参：ret, kernelName.c_str()),。
            ret, kernelName.c_str()),
        // [中文导读] [AllReduce逐行 S1191] 提供上述日志的实参：HCCL_E_RUNTIME。
        HCCL_E_RUNTIME);
    // [中文导读] [AllReduce逐行 S1192] 声明本次追加的单个参数句柄。
    aclrtParamHandle paraHandle;
    // [中文导读] 一次下发固定 OpParam 和连续尾部变长描述，因此设备能重建 counts 与位移指针。
    // [中文导读] [AllReduce逐行 S1194] 计算固定OpParam加尾部变长区域的总参数大小。
    size_t paramSize = sizeof(OpParam) + param.varMemSize;
    // [中文导读] [AllReduce逐行 S1195] 把本次OpParam及相邻尾部描述追加到ACL参数对象。
    ret = aclrtKernelArgsAppend(argsHandle, &param, paramSize, &paraHandle);
    // [中文导读] [AllReduce逐行 S1196] 开始条件检查宏，条件成立时记录错误并返回下方指定错误码。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S1197] ACL运行时返回码非成功时触发下面的日志和错误返回。
        ret != ACL_SUCCESS,
        // [中文导读] [AllReduce逐行 S1198] 输出错误日志，记录AicpuKernelLaunch当前阶段和相关参数。
        HCCL_ERROR(
            // [中文导读] [AllReduce逐行 S1199] 补充日志格式：[aclrtKernelArgsAppend]errNo[0x%016llx] args append failed, append。
            "[aclrtKernelArgsAppend]errNo[0x%016llx] args append failed, append "
            // [中文导读] [AllReduce逐行 S1200] 补充日志格式：size %u, kernelName:%s。
            "size %u, kernelName:%s",
            // [中文导读] [AllReduce逐行 S1201] 提供上述日志的实参：ret, paramSize, kernelName.c_str()),。
            ret, paramSize, kernelName.c_str()),
        // [中文导读] [AllReduce逐行 S1202] 提供上述日志的实参：HCCL_E_RUNTIME。
        HCCL_E_RUNTIME);
    // [中文导读] [AllReduce逐行 S1203] 完成kernel参数构造，交运行时校验/固化参数。
    ret = aclrtKernelArgsFinalize(argsHandle);
    // [中文导读] [AllReduce逐行 S1204] 开始条件检查宏，条件成立时记录错误并返回下方指定错误码。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S1205] ACL运行时返回码非成功时触发下面的日志和错误返回。
        ret != ACL_SUCCESS,
        // [中文导读] [AllReduce逐行 S1206] 输出错误日志，记录AicpuKernelLaunch当前阶段和相关参数。
        HCCL_ERROR(
            // [中文导读] [AllReduce逐行 S1207] 补充日志格式：[aclrtKernelArgsFinalize]errNo[0x%016llx] args finalize failed。
            "[aclrtKernelArgsFinalize]errNo[0x%016llx] args finalize failed, "
            // [中文导读] [AllReduce逐行 S1208] 补充日志格式：kernelName:%s。
            "kernelName:%s",
            // [中文导读] [AllReduce逐行 S1209] 提供上述日志的实参：ret, kernelName.c_str()),。
            ret, kernelName.c_str()),
        // [中文导读] [AllReduce逐行 S1210] 提供上述日志的实参：HCCL_E_RUNTIME。
        HCCL_E_RUNTIME);

    // [中文导读] 从算子配置派生 kernel 启动超时，并以 runtime 支持的超时属性填充 launch 配置。
    // [中文导读] [AllReduce逐行 S1213] 从执行配置派生kernel启动超时。
    AicpuTimeout timeout = DeriveAicpuTimeout(param.opConfig.execTimeout);
    // [中文导读] [AllReduce逐行 S1214] 声明最终16位kernel启动超时。
    u16 kernelLaunchTimeout
        // [中文导读] [AllReduce逐行 S1215] 根据HCOMM默认超时机制是否支持选择计算路径。
        = IsHcommDefaultTimeoutSupported() ?
              // [中文导读] [AllReduce逐行 S1216] 支持时直接使用统一派生的启动超时。
              timeout.kernelLaunchTimeout :
              // [中文导读] [AllReduce逐行 S1217] 旧机制使用执行超时加启动偏移，再限制为launch超时类型。
              ToKernelLaunchTimeout(AddAicpuTimeoutOffset(param.opConfig.execTimeout, KERNEL_TIMEOUT_OFFSET));
    // [中文导读] [AllReduce逐行 S1218] 声明kernel启动配置结构。
    aclrtLaunchKernelCfg cfg;
    // [中文导读] [AllReduce逐行 S1219] 声明单个kernel启动属性结构。
    aclrtLaunchKernelAttr attr;
    // [中文导读] [AllReduce逐行 S1220] 指定该属性设置kernel启动超时。
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
    // [中文导读] [AllReduce逐行 S1221] 设置启动超时的实际数值。
    attr.value.timeout = kernelLaunchTimeout;
    // [中文导读] [AllReduce逐行 S1222] 本次启动只设置一个属性。
    cfg.numAttrs = 1;
    // [中文导读] [AllReduce逐行 S1223] 将配置属性数组指向上面构造的超时属性。
    cfg.attrs = &attr;
    // [中文导读] [AllReduce逐行 S1224] 本次AICPU入口按一个block发射。
    constexpr u32 numBlocks = 1;
    // [中文导读] [AllReduce逐行 S1225] 输出运行日志，记录AicpuKernelLaunch当前阶段和相关参数。
    HCCL_INFO("[AicpuKernelLaunch] unfoldThread [%lu]", unfoldThread); // 通过Thread获取展开流stream
    // [中文导读] [AllReduce逐行 S1226] 初始化Host展开流返回地址为空。
    void* unfoldStream = nullptr;
    // [中文导读] [AllReduce逐行 S1227] 取得动态HCOMM函数表，用于检查展开线程查询函数。
    auto& HcclThreadResGetInfoFunc = ops_hccl::DlHcommFunction::GetInstance();
    // [中文导读] 缺少展开流查询能力或处于 OFFLOAD 时用用户流发射；支持时优先使用展开 Thread 对应流。
    // [中文导读] [AllReduce逐行 S1229] 无资源查询函数或OFFLOAD模式时直接在用户流发射。
    if (!HcclThreadResGetInfoFunc.dlHcclThreadResGetInfo || param.opMode == OpMode::OFFLOAD) { // 不走提前展开
        // [中文导读] [AllReduce逐行 S1230] 使用param.stream发射单block AICPU入口。
        ret = aclrtLaunchKernelWithConfig(funcHandle, numBlocks, param.stream, &cfg, argsHandle, nullptr);
    // [中文导读] [AllReduce逐行 S1231] 上述条件不成立时进入替代分支。
    } else {
        // [中文导读] [AllReduce逐行 S1232] 声明展开线程查询返回码。
        HcclResult ret1
            // [中文导读] [AllReduce逐行 S1233] 通过可选HCOMM函数查询unfoldThread关联的ACL stream。
            = HcclThreadResGetInfoFunc.dlHcclThreadResGetInfo(comm, unfoldThread, 0, sizeof(void*), &unfoldStream);
        // [中文导读] [AllReduce逐行 S1234] 查询明确返回NOT_SUPPORT时退回用户流。
        if (ret1 == HCCL_E_NOT_SUPPORT) {
            // [中文导读] [AllReduce逐行 S1235] 在用户流上发射AICPU入口作为不支持提前展开的兼容路径。
            ret = aclrtLaunchKernelWithConfig(funcHandle, numBlocks, param.stream, &cfg, argsHandle, nullptr);
        // [中文导读] [AllReduce逐行 S1236] 查询其它错误直接返回。
        } else if (ret1 != HCCL_SUCCESS) {
            // [中文导读] [AllReduce逐行 S1237] 返回展开线程查询错误，未继续发射。
            return ret1;
        // [中文导读] [AllReduce逐行 S1238] 上述条件不成立时进入替代分支。
        } else {
            // [中文导读] [AllReduce逐行 S1239] 查询成功时在unfoldStream提前展开流上发射入口。
            ret = aclrtLaunchKernelWithConfig(funcHandle, numBlocks, unfoldStream, &cfg, argsHandle, nullptr);
        // [中文导读] [AllReduce逐行 S1240] 结束条件} else。
        }
    // [中文导读] [AllReduce逐行 S1241] 结束条件} else。
    }
    // [中文导读] [AllReduce逐行 S1242] 开始条件检查宏，条件成立时记录错误并返回下方指定错误码。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S1243] 检查最终ACL发射结果。
        ret != ACL_SUCCESS,
        // [中文导读] [AllReduce逐行 S1244] 输出错误日志，记录AicpuKernelLaunch当前阶段和相关参数。
        HCCL_ERROR(
            // [中文导读] [AllReduce逐行 S1245] 补充日志格式：[LoadCustomKernel][aclrtLaunchKernelWithConfig]。
            "[LoadCustomKernel][aclrtLaunchKernelWithConfig]"
            // [中文导读] [AllReduce逐行 S1246] 补充日志格式：errNo[0x%016llx] launch kernel failed。
            "errNo[0x%016llx] launch kernel failed",
            // [中文导读] [AllReduce逐行 S1247] 提供上述日志的实参：ret),。
            ret),
        // [中文导读] [AllReduce逐行 S1248] ACL发射失败时返回源码指定的OPEN_FILE_FAILURE错误码。
        HCCL_E_OPEN_FILE_FAILURE);
    // [中文导读] [AllReduce逐行 S1249] 从已加载二进制取得入口，复制OpParam参数并通过ACL发射到展开流或用户流处理完成，返回成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S1250] 结束AicpuKernelLaunch函数体。
}

// [中文导读] 注意函数名：当前实现只确定并校验AIV核数上限，写回param.numBlocksLimit。
// [中文导读] 它本身不调用核发射API，真正执行还要跟踪调用方后续的ExecuteAivCacheLogic。
HcclResult HcclAivKernelEntranceLaunch(
    HcclComm comm, OpParam& param, const std::unique_ptr<TopoInfoWithNetLayerDetails>& topoInfo,
    AlgResourceCtxSerializable& resCtxHost)
{
    (void)topoInfo;
    HCCL_INFO(
        "[%s] algTag[%s] commModeTag[%s] resCtx(Host)[%p] aivCommInfoPtr(Device)[%p]", __func__, param.algTag,
        param.commModeTag, param.resCtx, resCtxHost.aivCommInfoPtr);
    u32 numBlocksLimit = 0;
    AivParamStorage* aivParam = nullptr;
    HcclResult ret = GetAivParamStorageByComm(comm, &aivParam, false);
    if (ret == HCCL_SUCCESS && aivParam != nullptr) {
        numBlocksLimit = aivParam->aivCoreLimit;
    } else if (param.opMode == OpMode::OFFLOAD) {
        HCCL_ERROR("[%s] GetAivParamStorageByComm fail, ret[%d], aivParam[%p]", __func__, ret, aivParam);
        return HCCL_E_INTERNAL;
    }
    if (numBlocksLimit == 0 && param.opMode == OpMode::OPBASE) {
        ACLCHECK(aclrtGetResInCurrentThread(ACL_RT_DEV_RES_VECTOR_CORE, &numBlocksLimit));
    }
    CHK_PRT_RET(
        numBlocksLimit < 1, HCCL_ERROR("[%s] block num less than 1, block num[%d]", __func__, numBlocksLimit),
        HCCL_E_PARA);
    param.numBlocksLimit = numBlocksLimit;
    HCCL_INFO("[%s] Aiv core limit is [%d].", __func__, numBlocksLimit);
    return HCCL_SUCCESS;
}

HcclResult
CaptureSlaveStreams(HcclComm comm, aclrtStream mainStream, const std::vector<ThreadHandle>& threads, bool& isCapture)
{
    isCapture = false;
    aclmdlRI rtModel = nullptr;
    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    aclError ret = aclmdlRICaptureGetInfo(mainStream, &captureStatus, &rtModel);
    if (ret == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
        HCCL_WARNING("[%s]Stream capture not support.", __func__);
        return HCCL_SUCCESS;
    } else {
        CHK_PRT_RET(
            ret != ACL_SUCCESS, HCCL_ERROR("[%s]aclmdlRICaptureGetInfo fail. return[%d].", __func__, ret),
            HCCL_E_RUNTIME);
    }
    if (captureStatus != aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE) {
        HCCL_INFO("[%s]captureStatus is not active, captureStatus[%d]", __func__, captureStatus);
        return HCCL_SUCCESS;
    }
    isCapture = true;
    // thread[0] is main thread
    auto& HcclThreadResGetInfoFunc = ops_hccl::DlHcommFunction::GetInstance();
    for (size_t i = 1; i < threads.size(); ++i) {
        void* stream = nullptr;
        CHK_PRT_RET(
            !HcclThreadResGetInfoFunc.dlHcclThreadResGetInfo, HCCL_ERROR("AclGraph is not support."),
            HCCL_E_NOT_SUPPORT);
        CHK_RET(HcclThreadResGetInfoFunc.dlHcclThreadResGetInfo(comm, threads[i], 0, sizeof(void*), &stream));
        rtError_t addRet = rtStreamAddToModel(stream, rtModel);
        CHK_PRT_RET(
            addRet != RT_ERROR_NONE, HCCL_ERROR("[%s]rtStreamAddToModel fail. return[%d].", __func__, addRet),
            HCCL_E_RUNTIME);
        HCCL_DEBUG(
            "[%s]add slaveStream to model success, idx[%zu], stream[%p], rtModel[%p]", __func__, i, stream, rtModel);
    }
    HCCL_INFO(
        "[%s]success, captured streams to rtmodel:[%p], slaveStreamNum:[%zu]", __func__, rtModel,
        threads.size() > 0 ? threads.size() - 1 : 0);
    return HCCL_SUCCESS;
}

// [中文导读] 在CPU_TS EngineCtx中缓存拓扑序列化结果。指定的未命中返回码才触发InitRankInfo和Create。
// [中文导读] 命中时反序列化已有拓扑，因此重复执行算子不要求每次重新遍历全部RankGraph接口。
// [中文导读] [AllReduce逐行 S1325] 声明HcclCalcTopoInfo接口：读取或新建基于算子tag的Host拓扑上下文。
HcclResult HcclCalcTopoInfo(HcclComm comm, OpParam& param, std::unique_ptr<TopoInfoWithNetLayerDetails>& topoInfo)
// [中文导读] [AllReduce逐行 S1326] 开始HcclCalcTopoInfo的函数体。
{
    // [中文导读] [AllReduce逐行 S1327] 输出运行日志，记录HcclCalcTopoInfo当前阶段和相关参数。
    HCCL_INFO("[%s] HcclCalcTopoInfo start.", __func__);
    // [中文导读] [AllReduce逐行 S1328] 初始化拓扑上下文查询的字节长度输出。
    uint64_t size = 0;
    // [中文导读] [AllReduce逐行 S1329] 初始化拓扑上下文地址输出为空。
    void* ctx = nullptr;
    // 若获取Context失败，表示对应Context尚未缓存
    // [中文导读] 以算子 tag 查询 CPU_TS 上保存的拓扑序列化上下文，减少重复拓扑读取。
    // [中文导读] [AllReduce逐行 S1332] 按算子tag和CPU_TS存储引擎查询已有Host拓扑上下文。
    HcclResult ret = HcclEngineCtxGet(comm, param.tag, CommEngine::COMM_ENGINE_CPU_TS, &ctx, &size);
    // [中文导读] [AllReduce逐行 S1333] NOT_FOUND或PARA按未缓存拓扑处理。
    if (ret == HCCL_E_NOT_FOUND || ret == HCCL_E_PARA) {
        // 初始化topoInfo
        // [中文导读] 首次查询未命中时初始化拓扑，再序列化到新上下文，供同类后续调用复用。
        // [中文导读] [AllReduce逐行 S1336] 从HCOMM rank graph初始化基础拓扑信息。
        CHK_RET(InitRankInfo(comm, topoInfo.get()));
        // 序列化
        // [中文导读] [AllReduce逐行 S1338] 将初始化的拓扑对象序列化为字节数组。
        std::vector<char> seq = topoInfo->Serialize();
        // [中文导读] [AllReduce逐行 S1339] 保存拓扑序列化数据的大小。
        size = seq.size();
        // 创建新的Context保存
        // [中文导读] [AllReduce逐行 S1341] 创建CPU_TS Host上下文存储拓扑缓存。
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, CommEngine::COMM_ENGINE_CPU_TS, size, &ctx));
        // [中文导读] [AllReduce逐行 S1342] 安全复制拓扑序列化字节到上下文。
        CHK_SAFETY_FUNC_RET(memcpy_s(ctx, size, seq.data(), size));
        // [中文导读] [AllReduce逐行 S1343] 新建拓扑缓存成功后直接返回。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S1344] 结束条件if (ret == HCCL_E_NOT_FOUND || ret == HCCL_E_PARA)。
    }
    // [中文导读] 复用路径把缓存字节恢复为新的拓扑对象，后续 selector 使用恢复后的层次信息。
    // [中文导读] [AllReduce逐行 S1346] 将查询返回的地址解释成序列化字节起点；此处未统一检查其它ret值。
    char* ctxTemp = reinterpret_cast<char*>(ctx);
    // [中文导读] [AllReduce逐行 S1347] 按返回长度构造拓扑序列化数据视图副本。
    std::vector<char> seq(ctxTemp, ctxTemp + size);
    // [中文导读] [AllReduce逐行 S1348] 创建临时拓扑对象接收反序列化结果。
    TopoInfoWithNetLayerDetails topoInfoTemp;
    // [中文导读] [AllReduce逐行 S1349] 恢复拓扑对象的字段。
    topoInfoTemp.DeSerialize(seq);
    // [中文导读] [AllReduce逐行 S1350] 移动构造新的topoInfo对象，供选择器继续使用。
    topoInfo = std::make_unique<TopoInfoWithNetLayerDetails>(std::move(topoInfoTemp));
    // [中文导读] [AllReduce逐行 S1351] 输出运行日志，记录HcclCalcTopoInfo当前阶段和相关参数。
    HCCL_INFO("[%s] HcclCalcTopoInfo end.", __func__);
    // [中文导读] [AllReduce逐行 S1352] 读取或新建基于算子tag的Host拓扑上下文处理完成，返回成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S1353] 结束HcclCalcTopoInfo函数体。
}

// [中文导读] [AllReduce逐行 S1355] 声明CompReqChannelWithExistChannel接口：增量请求只保留尚未存在的远端rank，AllReduce普通主例不进入此辅助分支。
void CompReqChannelWithExistChannel(
    // [中文导读] [AllReduce逐行 S1356] 函数参数包含线程/通道/通知资源需求，本行延续接口声明。
    const std::vector<std::vector<ChannelInfo>>& existChannels, AlgResourceRequest& resRequest)
// [中文导读] [AllReduce逐行 S1357] 开始CompReqChannelWithExistChannel的函数体。
{
    // [中文导读] [AllReduce逐行 S1358] 初始化已存在远端rank集合。
    std::set<u32> existRemoteRankSet = {};
    // [中文导读] [AllReduce逐行 S1359] 创建缺失通道请求列表。
    std::vector<HcclChannelDesc> needAllocChannelDesc;
    // 先把所有已存在的channel的remoteRank整理成集合
    // [中文导读] [AllReduce逐行 S1361] 遍历现有第0层通道。
    for (const ChannelInfo& channel : existChannels[0]) {
        // [中文导读] [AllReduce逐行 S1362] 将已有通道对端rank加入集合。
        existRemoteRankSet.insert(channel.remoteRank);
    // [中文导读] [AllReduce逐行 S1363] 结束循环for (const ChannelInfo& channel : existChannels[0])。
    }
    // 在集合中查找有没有request的channel
    // [中文导读] [AllReduce逐行 S1365] 遍历本次第0层新请求。
    for (const HcclChannelDesc& channelDesc : resRequest.channels[0]) {
        // [中文导读] [AllReduce逐行 S1366] 对端rank不在已有集合中才保留此请求。
        if (existRemoteRankSet.find(channelDesc.remoteRank) == existRemoteRankSet.end()) {
            // [中文导读] [AllReduce逐行 S1367] 把缺失通道请求加入待分配列表。
            needAllocChannelDesc.push_back(channelDesc);
        // [中文导读] [AllReduce逐行 S1368] 结束条件if (existRemoteRankSet.find(channelDesc.remoteRank) == existRemoteRankSet.end())。
        }
    // [中文导读] [AllReduce逐行 S1369] 结束循环for (const HcclChannelDesc& channelDesc : resRequest.channels[0])。
    }
    // [中文导读] [AllReduce逐行 S1370] 用缺失请求替换原资源通道请求。
    resRequest.channels = {needAllocChannelDesc};
    // [中文导读] [AllReduce逐行 S1371] 增量请求筛选结束。
    return;
// [中文导读] [AllReduce逐行 S1372] 结束CompReqChannelWithExistChannel函数体。
}

// [中文导读] 以算法tag和上下文Engine查询已准备的资源。返回成功表示可复用，不是本次又完成了一轮建链。
// [中文导读] AIV的算法描述放在CPU_TS上下文，CPU路径的描述放在AICPU_TS；存储位置不等于执行Engine。
// [中文导读] BatchSendRecv增量建链及部分非OPBASE模式明确不走这里的普通复用快路径。
// [中文导读] [AllReduce逐行 S1377] 声明TryReuseResource接口：按算法tag及实际上下文存储engine查询已存在的资源。
static HcclResult TryReuseResource(
    // [中文导读] [AllReduce逐行 S1378] 函数参数包含通信域句柄、算子参数、Device资源序列化地址输出、上下文字节长度、增量建链标志，本行延续接口声明。
    HcclComm comm, OpParam& param, bool& increCreateChannelFlag, void** resCtxSequence, uint64_t& size,
    // [中文导读] [AllReduce逐行 S1379] 函数参数包含资源复用标志，本行延续接口声明。
    bool& isResourceReused)
// [中文导读] [AllReduce逐行 S1380] 开始TryReuseResource的函数体。
{
    // 增量建链模式下不能复用资源
    // [中文导读] BatchSendRecv 单算子需要增量建链，因此先关闭普通整套资源复用快路径。
    // [中文导读] [AllReduce逐行 S1383] BatchSendRecv OPBASE需要增量通道，不走普通整套资源复用。
    if (param.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV && param.opMode == OpMode::OPBASE) {
        // [中文导读] [AllReduce逐行 S1384] 标记本次资源准备为增量建链。
        increCreateChannelFlag = true;
        // [中文导读] [AllReduce逐行 S1385] 用NOT_FOUND指示普通资源复用未命中。
        return HCCL_E_NOT_FOUND;
    // [中文导读] [AllReduce逐行 S1386] 结束条件if (param.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV && param.opMode == OpMode::OPBASE)。
    }
    // 非OPBASE模式且非CCU引擎不能复用资源
    // [中文导读] [AllReduce逐行 S1388] 非OPBASE且非CCU时不允许这条普通资源复用路径。
    if (param.opMode != OpMode::OPBASE && param.engine != CommEngine::COMM_ENGINE_CCU) {
        // [中文导读] [AllReduce逐行 S1389] 返回未找到，交后续重新准备资源。
        return HCCL_E_NOT_FOUND;
    // [中文导读] [AllReduce逐行 S1390] 结束条件if (param.opMode != OpMode::OPBASE && param.engine != CommEngine::COMM_ENGINE_CCU)。
    }
    // [中文导读] [AllReduce逐行 S1391] 初始化资源上下文返回地址。
    void* ctx = nullptr;
    // 这种情况下资源已经有了
    // [中文导读] [AllReduce逐行 S1393] 默认资源存储engine等于执行engine。
    CommEngine ctxEngine = param.engine;
    // [中文导读] [AllReduce逐行 S1394] AIV执行资源对象实际存放在Host CPU_TS上下文。
    if (param.engine == CommEngine::COMM_ENGINE_AIV) {
        // AIV模式固定利用利用algTag申请1块host内存resCtx
        // [中文导读] [AllReduce逐行 S1396] AIV查询转为CPU_TS存储engine。
        ctxEngine = COMM_ENGINE_CPU_TS;
    // [中文导读] [AllReduce逐行 S1397] CPU/DPU展开资源对象实际使用Device AICPU_TS上下文。
    } else if (param.engine == COMM_ENGINE_CPU) {
        // host dpu申请device内存用于存放resctx
        // [中文导读] [AllReduce逐行 S1399] CPU查询转为AICPU_TS存储engine。
        ctxEngine = COMM_ENGINE_AICPU_TS;
    // [中文导读] [AllReduce逐行 S1400] 结束条件} else if (param.engine == COMM_ENGINE_CPU)。
    }
    // [中文导读] 查询成功后返回已有上下文地址与大小，同时标记资源复用；不在这里重新申请通道。
    // [中文导读] [AllReduce逐行 S1402] 按algTag和实际存储engine查询资源上下文。
    if (HcclEngineCtxGet(comm, param.algTag, ctxEngine, &ctx, &size) == HCCL_SUCCESS) {
        // [中文导读] [AllReduce逐行 S1403] 输出调试日志，记录TryReuseResource当前阶段和相关参数。
        HCCL_DEBUG("Already have context, skip create, ctxSize is %llu", size);
        // [中文导读] [AllReduce逐行 S1404] 标记本次算法资源来自复用。
        isResourceReused = true;
        // [中文导读] [AllReduce逐行 S1405] 返回已有序列化资源地址给调用方。
        *resCtxSequence = ctx;
        // [中文导读] [AllReduce逐行 S1406] 保存资源上下文字节长度到Device参数。
        param.ctxSize = size;
        // [中文导读] [AllReduce逐行 S1407] 命中资源上下文后直接成功返回。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S1408] 结束条件if (HcclEngineCtxGet(comm, param.algTag, ctxEngine, &ctx, &size) == HCCL_SUCCESS)。
    }
    // [中文导读] [AllReduce逐行 S1409] 资源未命中返回NOT_FOUND，由HcclGetAlgRes继续资源计算。
    return HCCL_E_NOT_FOUND;
// [中文导读] [AllReduce逐行 S1410] 结束TryReuseResource函数体。
}

// [中文导读] 先尝试复用；未命中才计算算法分层和资源请求，再按Engine分配Thread、Channel及专用资源。
// [中文导读] executor->CalcRes给出“需要什么”，GetAlgResWithEngine负责“怎样获得”，两者不要混为创建API。
// [中文导读] 参数一致性检查按条件启用，其交换对象是算子元信息，不是用户输入输出数据。
// [中文导读] [AllReduce逐行 S1415] 声明HcclGetAlgRes接口：优先复用，未命中时计算拓扑层次和资源请求，分配资源并按条件核对一致性。
HcclResult HcclGetAlgRes(
    // [中文导读] [AllReduce逐行 S1416] 函数参数包含通信域句柄、算子参数、物理拓扑对象、具体算法执行器，本行延续接口声明。
    HcclComm comm, OpParam& param, std::unique_ptr<InsCollAlgBase>& executor, TopoInfoWithNetLayerDetails* topoInfo,
    // [中文导读] [AllReduce逐行 S1417] 函数参数包含Host资源描述对象、Device资源序列化地址输出、资源复用标志，本行延续接口声明。
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, void** resCtxSequence, bool& isResourceReused,
    // [中文导读] [AllReduce逐行 S1418] 函数参数包含图模式资源包，本行延续接口声明。
    const ResPackGraphMode& resPack)
// [中文导读] [AllReduce逐行 S1419] 开始HcclGetAlgRes的函数体。
{
    // [中文导读] [AllReduce逐行 S1420] 输出运行日志，记录HcclGetAlgRes当前阶段和相关参数。
    HCCL_INFO("[HcclGetAlgRes] Start to execute HcclGetAlgRes.");

    // 获取当前算法的完整属性
    // [中文导读] 读取选定算法的引擎与层次属性，作为拓扑分层和资源计算的依据。
    // [中文导读] [AllReduce逐行 S1424] 从具体executor获取选中算法的属性元数据。
    AlgAttrs algoMeta = executor->GetAlgoMeta(std::string(param.algName));
    // [中文导读] [AllReduce逐行 S1425] 输出运行日志，记录HcclGetAlgRes当前阶段和相关参数。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S1426] 补充日志格式：[HcclGetAlgRes] algName=%s, engine=%d, opType=%d, algoTypes.size=%zu.", param.algName。
        "[HcclGetAlgRes] algName=%s, engine=%d, opType=%d, algoTypes.size=%zu.", param.algName,
        // [中文导读] [AllReduce逐行 S1427] 提供上述日志的实参，涉及上下文字节长度。
        static_cast<int>(algoMeta.engine), static_cast<int>(algoMeta.opType), algoMeta.algoTypes.size());

    // [中文导读] 先尝试已有资源上下文；成功则直接返回，避免再次计算和申请同一套资源。
    // [中文导读] [AllReduce逐行 S1430] 默认不进行增量通道创建。
    bool increCreateChannelFlag = false;
    // [中文导读] [AllReduce逐行 S1431] 初始化资源上下文长度变量。
    uint64_t size = 0;
    // [中文导读] [AllReduce逐行 S1432] 尝试按算法tag复用已存在资源。
    if (TryReuseResource(comm, param, increCreateChannelFlag, resCtxSequence, size, isResourceReused) == HCCL_SUCCESS) {
        // [中文导读] [AllReduce逐行 S1433] 命中时跳过拓扑分层和资源申请。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S1434] 结束条件if (TryReuseResource(comm, param, increCreateChannelFlag, resCtxSequence, size, isResourceReused) == HCCL_SUCCESS)。
    }

    // context创建前做是否需要参数一致性的判断，context未创建则判断为首次下发该算子
    // [中文导读] 未命中资源时判断是否需要首次参数一致性检查，后续建链会使用这个标志。
    // [中文导读] [AllReduce逐行 S1438] 未命中资源时决定是否需要跨rank算子参数一致性校验。
    needInconsistentCheck = NeedInconsistentCheck(comm, param);

    // 计算AlgHierarchyInfo
    // [中文导读] 先把物理拓扑转换为算法层次通信域，再由 executor 计算 Thread/Channel 等需求。
    // [中文导读] [AllReduce逐行 S1442] 创建算法层级通信域描述对象。
    AlgHierarchyInfoForAllLevel algHierarchyInfo; // 分级通信域信息{localRankId, localRankSize}
    // [中文导读] [AllReduce逐行 S1443] 调用具体executor把物理拓扑匹配为算法层级通信域。
    CHK_RET(executor->CalcAlgHierarchyInfoV2(topoInfo, algHierarchyInfo, algoMeta));
    // 资源计算
    // [中文导读] [AllReduce逐行 S1445] 输出运行日志，记录HcclGetAlgRes当前阶段和相关参数。
    HCCL_INFO("[HcclGetAlgRes] executor->CalcRes.");
    // [中文导读] [AllReduce逐行 S1446] 创建本算法线程、通道、通知等资源需求对象。
    AlgResourceRequest resRequest;
    // [中文导读] [AllReduce逐行 S1447] 调用具体executor::CalcRes计算资源需求，尚未物理创建通道。
    CHK_RET(executor->CalcRes(comm, param, topoInfo, algHierarchyInfo, resRequest));
    // [中文导读] 按选定引擎获得资源，资源不足保留专门返回码供更上层协商或回退。
    // [中文导读] [AllReduce逐行 S1449] 开始按引擎准备实际资源。
    auto ret = GetAlgResWithEngine(
        // [中文导读] [AllReduce逐行 S1450] 传入需求、Host对象、层次信息和Device资源地址等输出。
        comm, param, resRequest, resCtxHost, topoInfo, algHierarchyInfo, resCtxSequence, size, increCreateChannelFlag,
        // [中文导读] [AllReduce逐行 S1451] 补充GE资源包并结束资源准备调用。
        resPack);
    // [中文导读] [AllReduce逐行 S1452] 资源不可用时保留UNAVAIL供HcclExecOp回退。
    if (ret == HCCL_E_UNAVAIL) {
        // [中文导读] [AllReduce逐行 S1453] 向上返回资源不可用。
        return HCCL_E_UNAVAIL;
    // [中文导读] [AllReduce逐行 S1454] 结束条件if (ret == HCCL_E_UNAVAIL)。
    }
    // [中文导读] [AllReduce逐行 S1455] 其它非成功资源错误直接传递。
    CHK_RET(ret);

    // [中文导读] [AllReduce逐行 S1457] Host资源对象存在时整理资源数量日志。
    if (resCtxHost != nullptr) {
        // 拼接各level的channel数量信息
        // [中文导读] [AllReduce逐行 S1459] 初始化按层通道数量的日志字符串。
        std::string channelNumInfo;
        // [中文导读] [AllReduce逐行 S1460] 遍历Host资源对象的算法通信层。
        for (size_t i = 0; i < resCtxHost->channels.size(); i++) {
            // [中文导读] [AllReduce逐行 S1461] 第二层及以后在数量信息之间添加分隔符。
            if (i > 0)
                // [中文导读] [AllReduce逐行 S1462] 追加通道层统计的逗号分隔符。
                channelNumInfo += ", ";
            // [中文导读] [AllReduce逐行 S1463] 把当前层序号和该层通道数量追加到日志信息。
            channelNumInfo += "level" + std::to_string(i) + "[" + std::to_string(resCtxHost->channels[i].size()) + "]";
        // [中文导读] [AllReduce逐行 S1464] 结束循环for (size_t i = 0; i < resCtxHost->channels.size(); i++)。
        }
        // [中文导读] [AllReduce逐行 S1465] 输出运行日志，记录HcclGetAlgRes当前阶段和相关参数。
        HCCL_RUN_INFO(
            // [中文导读] [AllReduce逐行 S1466] 补充日志格式：[HcclGetAlgRes] engine[%s], algTag[%s], resource allocated: thread num[%u]。
            "[HcclGetAlgRes] engine[%s], algTag[%s], resource allocated: thread num[%u], "
            // [中文导读] [AllReduce逐行 S1467] 补充日志格式：channel num per level[%s], ccu kernel num[%u].。
            "channel num per level[%s], ccu kernel num[%u].",
            // [中文导读] [AllReduce逐行 S1468] 提供上述日志的实参，涉及算子参数、算法关联tag。
            GetEnumToString(GetCommEngineStatusStrMap(), param.engine).c_str(), param.algTag,
            // [中文导读] [AllReduce逐行 S1469] 提供上述日志的实参，涉及Host资源描述对象、上下文字节长度、主从线程列表。
            resCtxHost->threads.size(), channelNumInfo.c_str(), resCtxHost->ccuKernels.size());
    // [中文导读] [AllReduce逐行 S1470] 结束条件if (resCtxHost != nullptr)。
    }

    // 参数一致性校验
    // [中文导读] 资源建立后比较远端交换的算子元信息，确认当前调用参数在通信域中一致。
    // [中文导读] [AllReduce逐行 S1474] 仅启用一致性校验时比较各rank的交换参数。
    if (needInconsistentCheck) {
        // [中文导读] [AllReduce逐行 S1475] 创建本端OpExchangeInfo结构。
        OpExchangeInfo exchangeInfo{};
        // [中文导读] [AllReduce逐行 S1476] 构造本端算子元信息，包括count、类型、归约类型等。
        CHK_RET(FillOpExchangeInfo(comm, param, exchangeInfo));
        // [中文导读] [AllReduce逐行 S1477] 与本次资源请求涉及的对端交换信息比较。
        CHK_RET(CompareOpExchangeInfos(comm, param, resRequest, exchangeInfo));
    // [中文导读] [AllReduce逐行 S1478] 结束条件if (needInconsistentCheck)。
    }

    // [中文导读] [AllReduce逐行 S1480] 优先复用，未命中时计算拓扑层次和资源请求，分配资源并按条件核对一致性处理完成，返回成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S1481] 结束HcclGetAlgRes函数体。
}

// [中文导读] [AllReduce逐行 S1483] 声明FillOpExchangeInfo接口：构造建链时交换的算子元信息。
HcclResult FillOpExchangeInfo(HcclComm comm, const OpParam& param, OpExchangeInfo& exchangeInfo)
// [中文导读] [AllReduce逐行 S1484] 开始FillOpExchangeInfo的函数体。
{
    // [中文导读] [AllReduce逐行 S1485] 检查通信域句柄非空。
    CHK_PTR_NULL(comm);
    // [中文导读] [AllReduce逐行 S1486] 创建不使用的CCL地址变量，查询仅需返回缓冲区大小。
    void* cclBufferAddr = nullptr; // 不使用，仅为调用HcclGetHcclBuffer获取cclBufferSize
    // [中文导读] [AllReduce逐行 S1487] 取得本域CCL容量，写进一致性交换信息。
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &exchangeInfo.cclBufferSize));
    // [中文导读] [AllReduce逐行 S1488] 保存本次root字段，AllReduce虽不依赖root也使用通用结构。
    exchangeInfo.root = param.root;
    // [中文导读] [AllReduce逐行 S1489] 保存本次命令ALLREDUCE。
    exchangeInfo.opType = param.opType;
    // [中文导读] [AllReduce逐行 S1490] 保存最终执行配置，供不同rank比较。
    exchangeInfo.opExecuteConfig = param.opExecuteConfig;
    // [中文导读] [AllReduce逐行 S1491] 保存本次SUM等归约类型。
    exchangeInfo.reduceType = param.reduceType;
    // [中文导读] [AllReduce逐行 S1492] 按算子数据描述布局填入dtype和count。
    CHK_RET(FillOpExchangeInfoWithDataDes(param, exchangeInfo));

    // [中文导读] [AllReduce逐行 S1494] 初始化AIV核数限制查询结果为零。
    u32 numBlocksLimit = 0;
    // [中文导读] [AllReduce逐行 S1495] 初始化通信域AIV参数对象地址。
    AivParamStorage* aivParam = nullptr;
    // [中文导读] [AllReduce逐行 S1496] 尝试查询已有通信域AIV参数，但不创建新对象。
    HcclResult ret = GetAivParamStorageByComm(comm, &aivParam, false);
    // [中文导读] [AllReduce逐行 S1497] 已有AIV参数对象时读取核数限制。
    if (ret == HCCL_SUCCESS && aivParam != nullptr) {
        // [中文导读] [AllReduce逐行 S1498] 从AIV参数对象取核数限制。
        numBlocksLimit = aivParam->aivCoreLimit;
        // [中文导读] [AllReduce逐行 S1499] 将核数限制写到算子交换结构。
        exchangeInfo.aivCoreLimit = numBlocksLimit;
    // [中文导读] [AllReduce逐行 S1500] 结束条件if (ret == HCCL_SUCCESS && aivParam != nullptr)。
    }
    // [中文导读] [AllReduce逐行 S1501] 无既有核数限制且OPBASE时查询当前运行时VectorCore数量。
    if (numBlocksLimit == 0 && param.opMode == OpMode::OPBASE) {
        // [中文导读] [AllReduce逐行 S1502] 通过ACL读取当前线程绑定资源的VectorCore数量。
        ACLCHECK(aclrtGetResInCurrentThread(ACL_RT_DEV_RES_VECTOR_CORE, &numBlocksLimit));
        // [中文导读] [AllReduce逐行 S1503] 保存运行时核数限制到交换结构。
        exchangeInfo.aivCoreLimit = numBlocksLimit;
    // [中文导读] [AllReduce逐行 S1504] 结束条件if (numBlocksLimit == 0 && param.opMode == OpMode::OPBASE)。
    }

    // [中文导读] [AllReduce逐行 S1506] 查询通信域名字写入交换结构。
    CHK_RET(HcclGetCommName(comm, exchangeInfo.group));
    // [中文导读] [AllReduce逐行 S1507] 强制域名字数组末端为字符串终止符。
    exchangeInfo.group[MAX_LENGTH - 1] = '\0';
    // [中文导读] [AllReduce逐行 S1508] 把本次算子tag复制到交换结构。
    s32 sRet = strncpy_s(exchangeInfo.tag, TAG_LENGTH, param.tag, TAG_LENGTH);
    // [中文导读] [AllReduce逐行 S1509] 开始条件检查宏，条件成立时记录错误并返回下方指定错误码。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S1510] 检查tag安全复制是否成功，失败时打印tag和返回码。
        sRet != EOK, HCCL_ERROR("[%s] call strncpy_s failed, param.tag[%s], return[%d].", __func__, param.tag, sRet),
        // [中文导读] [AllReduce逐行 S1511] 字符串复制失败返回内存错误。
        HCCL_E_MEMORY);

    // [中文导读] [AllReduce逐行 S1513] 输出运行日志，记录FillOpExchangeInfo当前阶段和相关参数。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S1514] 补充日志格式：[%s] success. exchangeInfo dump: cclBufferSize[%llu], root[%u], opType[%u], opExecuteConfig[%u]。
        "[%s] success. exchangeInfo dump: cclBufferSize[%llu], root[%u], opType[%u], opExecuteConfig[%u], "
        // [中文导读] [AllReduce逐行 S1515] 补充日志格式：reduceType[%u], dataType[%u], count[%llu], aivCoreLimit[%u], group[%s], tag[%s]。
        "reduceType[%u], dataType[%u], count[%llu], aivCoreLimit[%u], group[%s], tag[%s]",
        // [中文导读] [AllReduce逐行 S1516] 提供上述日志的实参：__func__, exchangeInfo.cclBufferSize, exchangeInfo.root, exchangeInfo.opType, exchangeInfo.opExecuteConfig,。
        __func__, exchangeInfo.cclBufferSize, exchangeInfo.root, exchangeInfo.opType, exchangeInfo.opExecuteConfig,
        // [中文导读] [AllReduce逐行 S1517] 提供上述日志的实参：exchangeInfo.reduceType, exchangeInfo.dataType, exchangeInfo.count, exchangeInfo.aivCoreLimit,。
        exchangeInfo.reduceType, exchangeInfo.dataType, exchangeInfo.count, exchangeInfo.aivCoreLimit,
        // [中文导读] [AllReduce逐行 S1518] 提供上述日志的实参，涉及算法名字键。
        exchangeInfo.group, exchangeInfo.tag);
    // [中文导读] [AllReduce逐行 S1519] 构造建链时交换的算子元信息处理完成，返回成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S1520] 结束FillOpExchangeInfo函数体。
}

// [中文导读] [AllReduce逐行 S1522] 声明FillOpExchangeInfoWithDataDes接口：按算子参数布局填充数据类型和count，AllReduce走default。
HcclResult FillOpExchangeInfoWithDataDes(const OpParam& param, OpExchangeInfo& exchangeInfo)
// [中文导读] [AllReduce逐行 S1523] 开始FillOpExchangeInfoWithDataDes的函数体。
{
    // [中文导读] [AllReduce逐行 S1524] 按算子类型选择数据描述字段布局。
    switch (param.opType) {
        // [中文导读] [AllReduce逐行 S1525] BatchSendRecv布局没有统一count，在该分支不填dtype/count。
        case HcclCMDType::HCCL_CMD_BATCH_SEND_RECV:
            // [中文导读] [AllReduce逐行 S1526] 结束当前算子类型分支，退出switch。
            break;
        // [中文导读] [AllReduce逐行 S1527] 固定AllToAll从all2AllVDataDes布局读取数据类型。
        case HcclCMDType::HCCL_CMD_ALLTOALL:
            // [中文导读] [AllReduce逐行 S1528] 保存AllToAll发送数据类型。
            exchangeInfo.dataType = param.all2AllVDataDes.sendType;
            // [中文导读] [AllReduce逐行 S1529] 检查AllToAll的sendCounts指针非空。
            CHK_PTR_NULL(param.all2AllVDataDes.sendCounts);
            // [中文导读] [AllReduce逐行 S1530] 从AllToAll第一个sendCounts元素取得统一count。
            exchangeInfo.count = static_cast<u64*>(param.all2AllVDataDes.sendCounts)[0];
            // [中文导读] [AllReduce逐行 S1531] 结束当前算子类型分支，退出switch。
            break;
        // [中文导读] [AllReduce逐行 S1532] 变长AllToAllV使用all2AllVDataDes发送数据类型。
        case HcclCMDType::HCCL_CMD_ALLTOALLV:
        // [中文导读] [AllReduce逐行 S1533] AllToAllVC与AllToAllV共享此分支。
        case HcclCMDType::HCCL_CMD_ALLTOALLVC:
            // [中文导读] [AllReduce逐行 S1534] 保存AllToAllV/VC的发送数据类型，不在这里填统一count。
            exchangeInfo.dataType = param.all2AllVDataDes.sendType;
            // [中文导读] [AllReduce逐行 S1535] 结束当前算子类型分支，退出switch。
            break;
        // [中文导读] [AllReduce逐行 S1536] 变长AllGatherV使用vDataDes布局。
        case HcclCMDType::HCCL_CMD_ALLGATHER_V:
        // [中文导读] [AllReduce逐行 S1537] 变长ReduceScatterV与AllGatherV共享数据类型读取逻辑。
        case HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V:
            // [中文导读] [AllReduce逐行 S1538] 保存变长集合通信的vDataDes数据类型。
            exchangeInfo.dataType = param.vDataDes.dataType;
            // [中文导读] [AllReduce逐行 S1539] 结束当前算子类型分支，退出switch。
            break;
        // [中文导读] [AllReduce逐行 S1540] 其它算子含AllReduce进入统一DataDes路径。
        default:
            // [中文导读] [AllReduce逐行 S1541] AllReduce将DataDes.dataType写入交换结构。
            exchangeInfo.dataType = param.DataDes.dataType;
            // [中文导读] [AllReduce逐行 S1542] AllReduce将DataDes.count写入交换结构。
            exchangeInfo.count = param.DataDes.count;
            // [中文导读] [AllReduce逐行 S1543] 结束当前算子类型分支，退出switch。
            break;
    // [中文导读] [AllReduce逐行 S1544] 结束代码块。
    }
    // [中文导读] [AllReduce逐行 S1545] 按算子参数布局填充数据类型和count，AllReduce走default处理完成，返回成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S1546] 结束FillOpExchangeInfoWithDataDes函数体。
}

// [中文导读] 一致性检查启用时，登记供建链流程交换的OpExchangeInfo，包括类型、数量和执行配置等。
// [中文导读] 该信息和内存注册后交换的地址/访问描述不是同一类对象；本函数也没有发起张量搬运。
// [中文导读] [AllReduce逐行 S1550] 声明AddExchangeInfo接口：启用参数一致性检查时登记下一次建链交换的OpExchangeInfo。
HcclResult AddExchangeInfo(HcclComm comm, const OpParam& param)
// [中文导读] [AllReduce逐行 S1551] 开始AddExchangeInfo的函数体。
{
    // [中文导读] [AllReduce逐行 S1552] 检查登记交换信息的通信域非空。
    CHK_PTR_NULL(comm);
    // [中文导读] [AllReduce逐行 S1553] 仅needInconsistentCheck为true时登记元信息。
    if (needInconsistentCheck) {
        // [中文导读] [AllReduce逐行 S1554] 创建本端算子交换结构。
        OpExchangeInfo exchangeInfo{};
        // [中文导读] [AllReduce逐行 S1555] 填充本端count、dtype、reduceType、配置等交换字段。
        CHK_RET(FillOpExchangeInfo(comm, param, exchangeInfo));
        // [中文导读] [AllReduce逐行 S1556] 将结构字节登记到HCOMM通信域，供随后的Acquire建链读取。
        CHK_RET(HcclCommAddExchangeInfo(comm, &exchangeInfo, sizeof(exchangeInfo)));
        // [中文导读] [AllReduce逐行 S1557] 输出运行日志，记录AddExchangeInfo当前阶段和相关参数。
        HCCL_INFO("[%s] success.", __func__);
    // [中文导读] [AllReduce逐行 S1558] 结束条件if (needInconsistentCheck)。
    }
    // [中文导读] [AllReduce逐行 S1559] 启用参数一致性检查时登记下一次建链交换的OpExchangeInfo处理完成，返回成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S1560] 结束AddExchangeInfo函数体。
}

#if CANN_VERSION_NUM >= CANN_VERSION(9, 1, 0)
static HcclResult ReleaseCcuAcquiredChannels(HcclComm comm, AlgResourceRequest& resRequest)
{
    if (!HcommIsSupportHcclChannelDestroy() || resRequest.acquiredChannels.empty()) {
        return HCCL_SUCCESS;
    }
    HcclResult ret = HcclChannelDestroy(comm, resRequest.acquiredChannels.data(), resRequest.acquiredChannels.size());
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING(
            "[ReleaseCcuAcquiredChannels] HcclChannelDestroy failed, ret[%d], channelNum[%zu].", ret,
            resRequest.acquiredChannels.size());
    }
    HCCL_INFO("[ReleaseCcuAcquiredChannels] release [%zu] channels.", resRequest.acquiredChannels.size());
    resRequest.acquiredChannels.clear();
    return HCCL_SUCCESS;
}
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 1, 0)

// [中文导读] 资源准备的Engine分发层。每次仅进入选定分支，不是依次准备AICPU、AIV和CCU全部资源。
// [中文导读] CCU资源不足的返回值可交上层协商或回退，不应把所有非成功结果一律解释成永久通信失败。
// [中文导读] [AllReduce逐行 S1582] 声明GetAlgResWithEngine接口：按CPU/AICPU_TS/AIV/CCU引擎分发资源准备，CCU可跨rank协商回退。
HcclResult GetAlgResWithEngine(
    // [中文导读] [AllReduce逐行 S1583] 函数参数包含通信域句柄、算子参数、线程/通道/通知资源需求，本行延续接口声明。
    HcclComm comm, OpParam& param, AlgResourceRequest& resRequest,
    // [中文导读] [AllReduce逐行 S1584] 函数参数包含物理拓扑对象、Host资源描述对象，本行延续接口声明。
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, TopoInfoWithNetLayerDetails* topoInfo,
    // [中文导读] [AllReduce逐行 S1585] 函数参数包含Device资源序列化地址输出、算法分层通信域、上下文字节长度、增量建链标志，本行延续接口声明。
    AlgHierarchyInfoForAllLevel& algHierarchyInfo, void** resCtxSequence, uint64_t& size, bool increCreateChannelFlag,
    // [中文导读] [AllReduce逐行 S1586] 函数参数包含图模式资源包，本行延续接口声明。
    const ResPackGraphMode& resPack)
// [中文导读] [AllReduce逐行 S1587] 开始GetAlgResWithEngine的函数体。
{
    // host侧资源
    // [中文导读] [AllReduce逐行 S1589] RESERVED引擎分支目前没有资源准备动作。
    if (param.engine == COMM_ENGINE_RESERVED) {
        // COMM_ENGINE_RESERVED
    // [中文导读] [AllReduce逐行 S1591] CPU引擎走Host DPU资源申请。
    } else if (param.engine == COMM_ENGINE_CPU) {
        // [中文导读] [AllReduce逐行 S1592] 开始调用GetAlgResDPU。
        CHK_RET(GetAlgResDPU(
            // [中文导读] [AllReduce逐行 S1593] 传入DPU资源请求、拓扑、资源对象和序列化输出。
            comm, param, resRequest, resCtxHost, topoInfo, algHierarchyInfo, resCtxSequence, size,
            // [中文导读] [AllReduce逐行 S1594] 补充增量标志和GE资源包，结束DPU资源调用。
            increCreateChannelFlag, resPack));
    // [中文导读] [AllReduce逐行 S1595] CPU_TS分支当前没有资源准备实现。
    } else if (param.engine == COMM_ENGINE_CPU_TS) {
        // COMM_ENGINE_CPU_TS
    // [中文导读] [AllReduce逐行 S1597] 裸AICPU分支当前没有资源准备实现。
    } else if (param.engine == COMM_ENGINE_AICPU) {
        // COMM_ENGINE_AICPU
    // [中文导读] [AllReduce逐行 S1599] AICPU_TS调用AICPU线程/通道资源准备，是本例主分支。
    } else if (param.engine == COMM_ENGINE_AICPU_TS) {
        // [中文导读] [AllReduce逐行 S1600] 开始调用GetAlgResAICPU。
        CHK_RET(GetAlgResAICPU(
            // [中文导读] [AllReduce逐行 S1601] 传入本次资源需求、Host对象、算法层次及Device地址输出。
            comm, param, resRequest, resCtxHost, topoInfo, algHierarchyInfo, resCtxSequence, size,
            // [中文导读] [AllReduce逐行 S1602] 补充增量标志和GE资源包，结束AICPU资源准备调用。
            increCreateChannelFlag, resPack));
    // [中文导读] [AllReduce逐行 S1603] AIV引擎申请其专属资源。
    } else if (param.engine == COMM_ENGINE_AIV) {
        // [中文导读] [AllReduce逐行 S1604] 调用AIV资源准备，返回AIV上下文地址。
        CHK_RET(GetAlgResAiv(comm, param, resRequest, topoInfo, algHierarchyInfo, resCtxSequence));
    // [中文导读] [AllReduce逐行 S1605] CCU引擎申请其专属资源。
    } else if (param.engine == COMM_ENGINE_CCU) {
        // [中文导读] CCU 申请结果还需参与多 Rank 资源协商；本端成功不能单独代表全域都能执行。
        // [中文导读] [AllReduce逐行 S1607] 开始获取CCU资源并保留返回码用于跨rank协商。
        auto ret = GetAlgResCcu(
            // [中文导读] [AllReduce逐行 S1608] 传入CCU需求、上下文、拓扑层次和GE资源包。
            comm, param, resRequest, resCtxHost, topoInfo, algHierarchyInfo, resCtxSequence, size, resPack);
        // 多卡CCU资源协商回退
// [中文导读] [AllReduce逐行 S1610] 按CANN_VERSION_NUM >= CANN_VERSION(9, 2, 0)这个编译期版本条件决定是否包含以下分支。
#if CANN_VERSION_NUM >= CANN_VERSION(9, 2, 0)
        // [中文导读] [AllReduce逐行 S1611] CANN 9.2及以上，本端成功或资源不足都需跨rank协商。
        if (ret == HCCL_E_UNAVAIL || ret == HCCL_SUCCESS) {
            // [中文导读] 把本端是否取得资源转为协商输入，任何 Rank 不可用时全域转入回退。
            // [中文导读] [AllReduce逐行 S1613] 以本端CCU资源是否成功作为协商输入。
            bool localResAvailable = (ret == HCCL_SUCCESS);
            // [中文导读] [AllReduce逐行 S1614] 协商各rank是否均有CCU资源可执行。
            auto negRet = CheckCcuResNegotiation(comm, param, localResAvailable);
            // [中文导读] [AllReduce逐行 S1615] 协商结果UNAVAIL表示全域应回退。
            if (negRet == HCCL_E_UNAVAIL) {
                // 多卡协商失败，释放本端已申请的CCU通道资源
                // [中文导读] 协商决定回退时释放本端已经取得的 CCU 通道，再返回资源不可用。
                // [中文导读] [AllReduce逐行 S1618] 请求释放本端已取得的CCU通道。
                ReleaseCcuAcquiredChannels(comm, resRequest);
                // [中文导读] [AllReduce逐行 S1619] 将跨rank资源不可用结果传回HcclExecOp。
                return HCCL_E_UNAVAIL;
            // [中文导读] [AllReduce逐行 S1620] 结束条件if (negRet == HCCL_E_UNAVAIL)。
            }
            // [中文导读] [AllReduce逐行 S1621] 其它协商错误直接向上传递。
            CHK_RET(negRet);
        // [中文导读] [AllReduce逐行 S1622] 上述条件不成立时进入替代分支。
        } else {
            // [中文导读] [AllReduce逐行 S1623] CCU申请的非UNAVAIL错误直接传递。
            CHK_RET(ret);
        // [中文导读] [AllReduce逐行 S1624] 结束条件} else。
        }
// [中文导读] [AllReduce逐行 S1625] 切换到上述编译期条件不成立的兼容实现。
#else
        // [中文导读] [AllReduce逐行 S1626] 较旧CANN分支只有本端CCU资源不可用判断。
        if (ret == HCCL_E_UNAVAIL) {
            // [中文导读] [AllReduce逐行 S1627] 将CCU资源不可用向上传递。
            return HCCL_E_UNAVAIL;
        // [中文导读] [AllReduce逐行 S1628] 结束条件if (ret == HCCL_E_UNAVAIL)。
        }
        // [中文导读] [AllReduce逐行 S1629] 其它CCU申请错误直接传递。
        CHK_RET(ret);
// [中文导读] [AllReduce逐行 S1630] 结束上述编译期条件控制的源码范围。
#endif
    // [中文导读] [AllReduce逐行 S1631] 上述条件不成立时进入替代分支。
    } else {
        // [中文导读] [AllReduce逐行 S1632] 输出错误日志，记录GetAlgResWithEngine当前阶段和相关参数。
        HCCL_ERROR(
            // [中文导读] [AllReduce逐行 S1633] 补充日志格式：fail to get engine, invalid engine type[%s].。
            "fail to get engine, invalid engine type[%s].",
            // [中文导读] [AllReduce逐行 S1634] 提供上述日志的实参，涉及算子参数。
            GetEnumToString(GetCommEngineStatusStrMap(), param.engine).c_str());
        // [中文导读] [AllReduce逐行 S1635] 无法识别的引擎返回参数错误。
        return HCCL_E_PARA;
    // [中文导读] [AllReduce逐行 S1636] 结束条件} else。
    }
    // [中文导读] [AllReduce逐行 S1637] 保存资源序列化长度到Device参数。
    param.ctxSize = size;
    // [中文导读] [AllReduce逐行 S1638] 按CPU/AICPU_TS/AIV/CCU引擎分发资源准备，CCU可跨rank协商回退处理完成，返回成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S1639] 结束GetAlgResWithEngine函数体。
}

// [中文导读] [AllReduce逐行 S1641] 声明CacheHostCtxToEngine接口：增量建链首次保存Host资源副本，失败按源码执行Device/Host上下文回滚。
HcclResult CacheHostCtxToEngine(
    // [中文导读] [AllReduce逐行 S1642] 函数参数包含通信域句柄、算法关联tag，本行延续接口声明。
    HcclComm comm, const char* algTag, const std::string& hostCacheTag, const std::vector<char>& hostCtxSeq)
// [中文导读] [AllReduce逐行 S1643] 开始CacheHostCtxToEngine的函数体。
{
    // [中文导读] [AllReduce逐行 S1644] 初始化增量Host缓存上下文地址为空。
    void* hostCtxPtr = nullptr;
    // [中文导读] [AllReduce逐行 S1645] 开始创建CPU_TS Host缓存上下文。
    HcclResult createRet = HcclEngineCtxCreate(
        // [中文导读] [AllReduce逐行 S1646] 按hostCacheTag为Host序列化副本分配存储，大小为hostCtxSeq长度。
        comm, hostCacheTag.c_str(), CommEngine::COMM_ENGINE_CPU_TS, hostCtxSeq.size(), &hostCtxPtr);
    // [中文导读] [AllReduce逐行 S1647] Host缓存创建失败时回滚此前创建的Device资源上下文。
    if (createRet != HCCL_SUCCESS) {
        // [中文导读] [AllReduce逐行 S1648] 输出错误日志，记录CacheHostCtxToEngine当前阶段和相关参数。
        HCCL_ERROR("failed to create host EngineCtx for caching, ret[%d].", createRet);
        // [中文导读] [AllReduce逐行 S1649] 尝试销毁AICPU_TS算法上下文，记录销毁返回码。
        HcclResult destroyRet = HcclEngineCtxDestroy(comm, algTag, COMM_ENGINE_AICPU_TS);
        // [中文导读] [AllReduce逐行 S1650] Device上下文销毁也失败时只打印错误，仍返回原createRet。
        if (destroyRet != HCCL_SUCCESS) {
            // [中文导读] [AllReduce逐行 S1651] 输出错误日志，记录CacheHostCtxToEngine当前阶段和相关参数。
            HCCL_ERROR("failed to destroy device ctx on host ctx create failure rollback, ret[%d].", destroyRet);
        // [中文导读] [AllReduce逐行 S1652] 结束条件if (destroyRet != HCCL_SUCCESS)。
        }
        // [中文导读] [AllReduce逐行 S1653] 返回原Host缓存创建失败的错误码。
        return createRet;
    // [中文导读] [AllReduce逐行 S1654] 结束条件if (createRet != HCCL_SUCCESS)。
    }
    // [中文导读] [AllReduce逐行 S1655] 安全复制Host资源序列化字节到新缓存地址。
    errno_t memcpyRet = memcpy_s(hostCtxPtr, hostCtxSeq.size(), hostCtxSeq.data(), hostCtxSeq.size());
    // [中文导读] [AllReduce逐行 S1656] Host缓存字节复制失败时销毁两侧缓存上下文。
    if (memcpyRet != EOK) {
        // [中文导读] [AllReduce逐行 S1657] 输出错误日志，记录CacheHostCtxToEngine当前阶段和相关参数。
        HCCL_ERROR("memcpy_s failed writing to host EngineCtx cache, ret=%d.", memcpyRet);
        // [中文导读] [AllReduce逐行 S1658] 请求销毁CPU_TS Host缓存，本行未检查返回码。
        HcclEngineCtxDestroy(comm, hostCacheTag.c_str(), CommEngine::COMM_ENGINE_CPU_TS);
        // [中文导读] [AllReduce逐行 S1659] 请求销毁AICPU_TS Device上下文，本行未检查返回码。
        HcclEngineCtxDestroy(comm, algTag, COMM_ENGINE_AICPU_TS);
        // [中文导读] [AllReduce逐行 S1660] 缓存复制失败返回内部错误。
        return HCCL_E_INTERNAL;
    // [中文导读] [AllReduce逐行 S1661] 结束条件if (memcpyRet != EOK)。
    }
    // [中文导读] [AllReduce逐行 S1662] 增量建链首次保存Host资源副本，失败按源码执行Device/Host上下文回滚处理完成，返回成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S1663] 结束CacheHostCtxToEngine函数体。
}

// [中文导读] [AllReduce逐行 S1665] 声明ReuseCachedDeviceCtx接口：增量请求没有新Peer时直接查已有Device序列化上下文。
HcclResult ReuseCachedDeviceCtx(HcclComm comm, const OpParam& param, void** resCtxSequence, uint64_t& ctxSize)
// [中文导读] [AllReduce逐行 S1666] 开始ReuseCachedDeviceCtx的函数体。
{
    // [中文导读] [AllReduce逐行 S1667] 初始化已有Device上下文查询地址。
    void* ctx = nullptr;
    // [中文导读] [AllReduce逐行 S1668] 初始化已有Device上下文查询长度。
    uint64_t size = 0;
    // [中文导读] [AllReduce逐行 S1669] 声明Device上下文查询返回码。
    HcclResult ret;
    // [中文导读] [AllReduce逐行 S1670] CPU/DPU资源实际存储在AICPU_TS，所以查询需转换engine。
    if (param.engine == COMM_ENGINE_CPU) {
        // [中文导读] [AllReduce逐行 S1671] 按AICPU_TS存储engine查CPU执行资源。
        ret = HcclEngineCtxGet(comm, param.algTag, COMM_ENGINE_AICPU_TS, &ctx, &size);
    // [中文导读] [AllReduce逐行 S1672] 上述条件不成立时进入替代分支。
    } else {
        // [中文导读] [AllReduce逐行 S1673] 其它引擎按param.engine查询资源上下文。
        ret = HcclEngineCtxGet(comm, param.algTag, param.engine, &ctx, &size);
    // [中文导读] [AllReduce逐行 S1674] 结束条件} else。
    }
    // [中文导读] [AllReduce逐行 S1675] 查询成功时把已有上下文地址与大小返回调用方。
    if (ret == HCCL_SUCCESS) {
        // [中文导读] [AllReduce逐行 S1676] 返回已有Device资源序列化地址。
        *resCtxSequence = ctx;
        // [中文导读] [AllReduce逐行 S1677] 返回已有Device资源序列化长度。
        ctxSize = size;
        // [中文导读] [AllReduce逐行 S1678] 增量无新增Peer时成功复用Device上下文。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S1679] 结束条件if (ret == HCCL_SUCCESS)。
    }
    // [中文导读] [AllReduce逐行 S1680] 输出错误日志，记录ReuseCachedDeviceCtx当前阶段和相关参数。
    HCCL_ERROR("failed to get device ctx.");
    // [中文导读] [AllReduce逐行 S1681] 未能查询到已有Device上下文时返回原始错误码。
    return ret;
// [中文导读] [AllReduce逐行 S1682] 结束ReuseCachedDeviceCtx函数体。
}

// [中文导读] [AllReduce逐行 S1684] 声明IncrementalCreateChannel接口：创建缺失通道并更新Device序列化上下文及Host缓存，AllReduce普通主例不进入。
HcclResult IncrementalCreateChannel(
    // [中文导读] [AllReduce逐行 S1685] 函数参数包含通信域句柄、算子参数、线程/通道/通知资源需求，本行延续接口声明。
    HcclComm comm, const OpParam& param, AlgResourceRequest& resRequest, AlgResourceCtxSerializable& hostCtxObj,
    // [中文导读] [AllReduce逐行 S1686] 函数参数包含Device资源序列化地址输出、资源上下文字节长度，本行延续接口声明。
    const std::string& hostCacheTag, void** resCtxSequence, uint64_t& ctxSize)
// [中文导读] [AllReduce逐行 S1687] 开始IncrementalCreateChannel的函数体。
{
    // [中文导读] [AllReduce逐行 S1688] 为筛选后的新增Peer取得通道并追加到Host资源对象。
    HcclResult ret = HcclGetChannel(comm, param, resRequest, &hostCtxObj);
    // [中文导读] [AllReduce逐行 S1689] 新增通道获取失败直接返回该错误。
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("failed to incrementally create channel."), ret);
    // [中文导读] [AllReduce逐行 S1690] CPU/DPU的旧资源上下文存于AICPU_TS。
    if (param.engine == COMM_ENGINE_CPU) {
        // [中文导读] [AllReduce逐行 S1691] 请求销毁旧AICPU_TS序列化资源上下文。
        ret = HcclEngineCtxDestroy(comm, param.algTag, COMM_ENGINE_AICPU_TS);
    // [中文导读] [AllReduce逐行 S1692] 上述条件不成立时进入替代分支。
    } else {
        // [中文导读] [AllReduce逐行 S1693] 其它引擎按param.engine销毁旧Device资源上下文。
        ret = HcclEngineCtxDestroy(comm, param.algTag, param.engine);
    // [中文导读] [AllReduce逐行 S1694] 结束条件} else。
    }
    // [中文导读] [AllReduce逐行 S1695] 旧Device上下文销毁失败时打印错误，但仍继续下面的重建流程。
    if (ret != HCCL_SUCCESS) {
        // [中文导读] [AllReduce逐行 S1696] 输出错误日志，记录IncrementalCreateChannel当前阶段和相关参数。
        HCCL_ERROR("failed to destroy device Ctx, ret[%d].", ret);
    // [中文导读] [AllReduce逐行 S1697] 结束条件if (ret != HCCL_SUCCESS)。
    }
    // [中文导读] [AllReduce逐行 S1698] 将增量更新后的Host资源对象序列化。
    std::vector<char> newSeq = hostCtxObj.Serialize();
    // [中文导读] [AllReduce逐行 S1699] 创建新的Device上下文并复制更新后的资源字节。
    ret = HcclMemcpyCtxHostToDevice(comm, param, newSeq, resCtxSequence, ctxSize);
    // [中文导读] [AllReduce逐行 S1700] 重建Device上下文失败时尝试清理已有Host缓存。
    if (ret != HCCL_SUCCESS) {
        // [中文导读] [AllReduce逐行 S1701] 输出错误日志，记录IncrementalCreateChannel当前阶段和相关参数。
        HCCL_ERROR("failed to memcpy hostCtx to device after incremental channel creation, ret[%d].", ret);
        // [中文导读] [AllReduce逐行 S1702] 请求销毁CPU_TS Host缓存并保存清理返回码。
        HcclResult destroyRet = HcclEngineCtxDestroy(comm, hostCacheTag.c_str(), CommEngine::COMM_ENGINE_CPU_TS);
        // [中文导读] [AllReduce逐行 S1703] Host缓存清理失败时只打印错误，保留原Device重建错误。
        if (destroyRet != HCCL_SUCCESS) {
            // [中文导读] [AllReduce逐行 S1704] 输出错误日志，记录IncrementalCreateChannel当前阶段和相关参数。
            HCCL_ERROR("failed to destroy host ctx on incremental path failure rollback, ret[%d].", destroyRet);
        // [中文导读] [AllReduce逐行 S1705] 结束条件if (destroyRet != HCCL_SUCCESS)。
        }
        // [中文导读] [AllReduce逐行 S1706] 返回原Device资源复制失败错误。
        return ret;
    // [中文导读] [AllReduce逐行 S1707] 结束条件if (ret != HCCL_SUCCESS)。
    }
    // [中文导读] [AllReduce逐行 S1708] 更新Device成功后销毁旧CPU_TS Host缓存。
    HcclResult destroyRet = HcclEngineCtxDestroy(comm, hostCacheTag.c_str(), CommEngine::COMM_ENGINE_CPU_TS);
    // [中文导读] [AllReduce逐行 S1709] 旧Host缓存销毁失败时打印错误，继续创建更新缓存。
    if (destroyRet != HCCL_SUCCESS) {
        // [中文导读] [AllReduce逐行 S1710] 输出错误日志，记录IncrementalCreateChannel当前阶段和相关参数。
        HCCL_ERROR("failed to destroy old host EngineCtx for cache update, ret[%d].", destroyRet);
    // [中文导读] [AllReduce逐行 S1711] 结束条件if (destroyRet != HCCL_SUCCESS)。
    }
    // [中文导读] [AllReduce逐行 S1712] 初始化更新后的Host缓存存储地址。
    void* newHostCtxPtr = nullptr;
    // [中文导读] [AllReduce逐行 S1713] 开始创建更新CPU_TS Host缓存。
    HcclResult cacheRet = HcclEngineCtxCreate(
        // [中文导读] [AllReduce逐行 S1714] 按更新序列化长度创建Host资源缓存上下文。
        comm, hostCacheTag.c_str(), CommEngine::COMM_ENGINE_CPU_TS, newSeq.size(), &newHostCtxPtr);
    // [中文导读] [AllReduce逐行 S1715] 更新Host缓存创建失败时回滚新Device上下文。
    if (cacheRet != HCCL_SUCCESS) {
        // [中文导读] [AllReduce逐行 S1716] 输出错误日志，记录IncrementalCreateChannel当前阶段和相关参数。
        HCCL_ERROR("failed to create host EngineCtx for cache update, ret[%d].", cacheRet);
        // [中文导读] [AllReduce逐行 S1717] 请求销毁param.engine算法Device资源上下文，保存销毁错误。
        HcclResult devDestroyRet = HcclEngineCtxDestroy(comm, param.algTag, param.engine);
        // [中文导读] [AllReduce逐行 S1718] 回滚Device上下文销毁失败时只打印错误。
        if (devDestroyRet != HCCL_SUCCESS) {
            // [中文导读] [AllReduce逐行 S1719] 输出错误日志，记录IncrementalCreateChannel当前阶段和相关参数。
            HCCL_ERROR("failed to destroy device ctx on host cache update failure rollback, ret[%d].", devDestroyRet);
        // [中文导读] [AllReduce逐行 S1720] 结束条件if (devDestroyRet != HCCL_SUCCESS)。
        }
        // [中文导读] [AllReduce逐行 S1721] 返回原Host缓存创建错误。
        return cacheRet;
    // [中文导读] [AllReduce逐行 S1722] 结束条件if (cacheRet != HCCL_SUCCESS)。
    }
    // [中文导读] [AllReduce逐行 S1723] 安全复制更新资源字节到新Host缓存。
    errno_t memcpyRet = memcpy_s(newHostCtxPtr, newSeq.size(), newSeq.data(), newSeq.size());
    // [中文导读] [AllReduce逐行 S1724] 更新Host缓存字节复制失败时请求销毁两侧上下文。
    if (memcpyRet != EOK) {
        // [中文导读] [AllReduce逐行 S1725] 输出错误日志，记录IncrementalCreateChannel当前阶段和相关参数。
        HCCL_ERROR("memcpy_s failed writing to updated host EngineCtx cache, ret=%d.", memcpyRet);
        // [中文导读] [AllReduce逐行 S1726] 请求销毁更新后的Host缓存，本行未检查返回码。
        HcclEngineCtxDestroy(comm, hostCacheTag.c_str(), CommEngine::COMM_ENGINE_CPU_TS);
        // [中文导读] [AllReduce逐行 S1727] 请求销毁新的Device资源上下文，本行未检查返回码。
        HcclEngineCtxDestroy(comm, param.algTag, param.engine);
        // [中文导读] [AllReduce逐行 S1728] 更新Host缓存复制失败返回内部错误。
        return HCCL_E_INTERNAL;
    // [中文导读] [AllReduce逐行 S1729] 结束条件if (memcpyRet != EOK)。
    }
    // [中文导读] [AllReduce逐行 S1730] 输出运行日志，记录IncrementalCreateChannel当前阶段和相关参数。
    HCCL_INFO("Incrementally add channel success");
    // [中文导读] [AllReduce逐行 S1731] 创建缺失通道并更新Device序列化上下文及Host缓存，AllReduce普通主例不进入处理完成，返回成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S1732] 结束IncrementalCreateChannel函数体。
}

// [中文导读] 常规路径准备Host资源描述后序列化并复制到设备，供AICPU入口恢复。
// [中文导读] increCreateChannelFlag描述增量建链模式；该分支可比较已有Peer，只补缺失通道。
// [中文导读] [AllReduce逐行 S1736] 声明GetAlgResAICPU接口：首次请求构造Host资源对象并复制Device，增量请求可筛除已存在通道。
HcclResult GetAlgResAICPU(
    // [中文导读] [AllReduce逐行 S1737] 函数参数包含通信域句柄、算子参数、线程/通道/通知资源需求，本行延续接口声明。
    HcclComm comm, const OpParam& param, AlgResourceRequest& resRequest,
    // [中文导读] [AllReduce逐行 S1738] 函数参数包含物理拓扑对象、Host资源描述对象，本行延续接口声明。
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, TopoInfoWithNetLayerDetails* topoInfo,
    // [中文导读] [AllReduce逐行 S1739] 函数参数包含Device资源序列化地址输出、算法分层通信域、资源上下文字节长度，本行延续接口声明。
    AlgHierarchyInfoForAllLevel& algHierarchyInfo, void** resCtxSequence, uint64_t& ctxSize,
    // [中文导读] [AllReduce逐行 S1740] 函数参数包含图模式资源包、增量建链标志，本行延续接口声明。
    bool increCreateChannelFlag, const ResPackGraphMode& resPack)
// [中文导读] [AllReduce逐行 S1741] 开始GetAlgResAICPU的函数体。
{
    // [中文导读] 增量建链除了设备上下文，还保留 Host 序列化副本用于比较已有 Peer 通道。
    // [中文导读] [AllReduce逐行 S1743] 构造增量通道资源的Host缓存tag。
    std::string hostCacheTag = std::string(param.algTag) + "_hostCache";
    // [中文导读] [AllReduce逐行 S1744] 初始化Host缓存资源地址为空。
    void* hostCtxPtr = nullptr;
    // [中文导读] [AllReduce逐行 S1745] 初始化Host缓存字节长度输出。
    uint64_t hostCtxSize = 0;
    // [中文导读] [AllReduce逐行 S1746] 声明Host缓存查询结果。
    HcclResult hostCtxRet
        // [中文导读] [AllReduce逐行 S1747] 查CPU_TS Host序列化资源缓存；普通AllReduce即便命中也走首次/普通资源准备分支。
        = HcclEngineCtxGet(comm, hostCacheTag.c_str(), CommEngine::COMM_ENGINE_CPU_TS, &hostCtxPtr, &hostCtxSize);
    // [中文导读] 普通申请或首次增量申请需填充拓扑、算法层次及实际资源，随后复制设备上下文。
    // [中文导读] [AllReduce逐行 S1749] 非增量请求或无Host缓存时构造完整资源对象。
    if (!increCreateChannelFlag || hostCtxRet != HCCL_SUCCESS) {
        // [中文导读] [AllReduce逐行 S1750] 保存Host通信域地址，用于Device资源缓存是否陈旧的判断。
        resCtxHost->commInfoPtr = static_cast<void*>(comm);
        // [中文导读] [AllReduce逐行 S1751] 把已选择时得到的拓扑复制到资源对象。
        resCtxHost->topoInfo = *topoInfo;
        // [中文导读] [AllReduce逐行 S1752] 保存算法匹配出的分层通信域信息。
        resCtxHost->algHierarchyInfo = algHierarchyInfo;
        // [中文导读] [AllReduce逐行 S1753] 准备CCL中转缓冲区、执行线程及通信通道。
        HcclResult ret = HcclAllocAlgResourceAICPU(comm, param, resRequest, resCtxHost, resPack);
        // [中文导读] [AllReduce逐行 S1754] 实际资源准备失败直接返回该错误。
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("failed to alloc alg resource."), ret);
        // [中文导读] 序列化的是执行资源描述，设备入口随后反序列化使用其中的 Thread、Channel 和拓扑。
        // [中文导读] [AllReduce逐行 S1756] 将资源对象序列化为字节序列。
        std::vector<char> hostCtxSeq = resCtxHost->Serialize();
        // [中文导读] [AllReduce逐行 S1757] 创建Device上下文并拷贝资源序列化数据。
        ret = HcclMemcpyCtxHostToDevice(comm, param, hostCtxSeq, resCtxSequence, ctxSize);
        // [中文导读] [AllReduce逐行 S1758] Device资源复制失败直接返回该错误。
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("failed to memcpy hostCtx to device."), ret);
        // [中文导读] [AllReduce逐行 S1759] 仅增量建链模式额外保存Host序列化副本。
        if (increCreateChannelFlag) {
            // [中文导读] [AllReduce逐行 S1760] 把Host资源字节存入CPU_TS上下文缓存。
            CHK_RET(CacheHostCtxToEngine(comm, param.algTag, hostCacheTag, hostCtxSeq));
        // [中文导读] [AllReduce逐行 S1761] 结束条件if (increCreateChannelFlag)。
        }
    // [中文导读] [AllReduce逐行 S1762] 上述条件不成立时进入替代分支。
    } else {
        // [中文导读] 已存在增量 Host 缓存时先恢复对象，再筛出尚未拥有通道的 Peer。
        // [中文导读] [AllReduce逐行 S1764] 增量缓存命中时从Host缓存地址构造字节序列。
        std::vector<char> cachedData(static_cast<char*>(hostCtxPtr), static_cast<char*>(hostCtxPtr) + hostCtxSize);
        // [中文导读] [AllReduce逐行 S1765] 创建用于恢复增量资源的临时Host对象。
        AlgResourceCtxSerializable hostCtxObj;
        // [中文导读] [AllReduce逐行 S1766] 反序列化现有Host资源副本。
        hostCtxObj.DeSerialize(cachedData);
        // [中文导读] [AllReduce逐行 S1767] 从请求中剔除已有对端通道。
        CompReqChannelWithExistChannel(hostCtxObj.channels, resRequest);
        // [中文导读] 没有新增 Peer 时直接复用设备上下文；有新增请求才更新通道和两侧缓存。
        // [中文导读] [AllReduce逐行 S1769] 没有新增通道时直接复用Device上下文。
        if (resRequest.channels[0].size() == 0) {
            // [中文导读] [AllReduce逐行 S1770] 读取已有Device资源上下文并返回地址和大小。
            return ReuseCachedDeviceCtx(comm, param, resCtxSequence, ctxSize);
        // [中文导读] [AllReduce逐行 S1771] 结束条件if (resRequest.channels[0].size() == 0)。
        }
        // [中文导读] [AllReduce逐行 S1772] 创建缺失通道后更新Host及Device资源副本。
        CHK_RET(IncrementalCreateChannel(comm, param, resRequest, hostCtxObj, hostCacheTag, resCtxSequence, ctxSize));
    // [中文导读] [AllReduce逐行 S1773] 结束条件} else。
    }

    // [中文导读] [AllReduce逐行 S1775] 输出运行日志，记录GetAlgResAICPU当前阶段和相关参数。
    HCCL_INFO("Execute GetAlgResAICPU success.");
    // [中文导读] [AllReduce逐行 S1776] 首次请求构造Host资源对象并复制Device，增量请求可筛除已存在通道处理完成，返回成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S1777] 结束GetAlgResAICPU函数体。
}

// [中文导读] Create申请设备侧上下文存储，Copy写入序列化资源描述，出参返回设备地址和长度。
// [中文导读] 拷贝的是线程/通道等执行描述，不是本次集合通信的用户张量。
// [中文导读] [AllReduce逐行 S1781] 声明HcclMemcpyCtxHostToDevice接口：创建AICPU资源上下文存储并复制序列化字节。
HcclResult HcclMemcpyCtxHostToDevice(
    // [中文导读] [AllReduce逐行 S1782] 函数参数包含通信域句柄、算子参数、Device资源序列化地址输出、资源上下文字节长度，本行延续接口声明。
    HcclComm comm, const OpParam& param, const std::vector<char>& seq, void** resCtxSequence, uint64_t& ctxSize)
// [中文导读] [AllReduce逐行 S1783] 开始HcclMemcpyCtxHostToDevice的函数体。
{
    // [中文导读] [AllReduce逐行 S1784] 读取待复制序列化字节长度。
    uint64_t size = seq.size();
    // [中文导读] [AllReduce逐行 S1785] 初始化Device上下文地址输出。
    void* ctx = nullptr;
    // 创建Context, aicpu和host dpu申请device内存
    // [中文导读] [AllReduce逐行 S1787] 为algTag创建AICPU_TS Device上下文存储区。
    CHK_RET(HcclEngineCtxCreate(comm, param.algTag, COMM_ENGINE_AICPU_TS, size, &ctx));
    // 从Host内存拷贝到Device Context内存上
    // [中文导读] [AllReduce逐行 S1789] 从Host复制资源描述字节到Device上下文，偏移为0。
    CHK_RET(HcclEngineCtxCopy(comm, COMM_ENGINE_AICPU_TS, param.algTag, seq.data(), size, 0));
    // 将内存强转为AlgResourceCtx结构体
    // [中文导读] [AllReduce逐行 S1791] 返回Device上下文地址供kernel参数引用。
    *resCtxSequence = ctx;
    // [中文导读] [AllReduce逐行 S1792] 返回上下文字节长度供反序列化。
    ctxSize = size;
    // [中文导读] [AllReduce逐行 S1793] 输出运行日志，记录HcclMemcpyCtxHostToDevice当前阶段和相关参数。
    HCCL_INFO("Memcpy hostCtx to device success.");
    // [中文导读] [AllReduce逐行 S1794] 创建AICPU资源上下文存储并复制序列化字节处理完成，返回成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S1795] 结束HcclMemcpyCtxHostToDevice函数体。
}

// [中文导读] AICPU资源准备的读取顺序：取得本地CCL区、申请主从Thread、按Peer申请Channel。
// [中文导读] HcclGetHcclBuffer取得域持有的中转区；不能把Get调用次数当成物理内存分配次数。
// [中文导读] [AllReduce逐行 S1799] 声明HcclAllocAlgResourceAICPU接口：获得CCL缓冲区、线程和通道，写入可序列化资源对象。
HcclResult HcclAllocAlgResourceAICPU(
    // [中文导读] [AllReduce逐行 S1800] 函数参数包含通信域句柄、算子参数、线程/通道/通知资源需求，本行延续接口声明。
    HcclComm comm, const OpParam& param, AlgResourceRequest& resRequest,
    // [中文导读] [AllReduce逐行 S1801] 函数参数包含图模式资源包、Host资源描述对象，本行延续接口声明。
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, const ResPackGraphMode& resPack)
// [中文导读] [AllReduce逐行 S1802] 开始HcclAllocAlgResourceAICPU的函数体。
{
    // [中文导读] [AllReduce逐行 S1803] 输出运行日志，记录HcclAllocAlgResourceAICPU当前阶段和相关参数。
    HCCL_INFO("Start to execute AllocAlgResource.");
    // [中文导读] [AllReduce逐行 S1804] 声明本域CCL缓冲区地址输出变量。
    void* cclBufferAddr;
    // [中文导读] [AllReduce逐行 S1805] 声明本域CCL缓冲区容量输出变量。
    uint64_t cclBufferSize;
    // 从通信域获取CCL buffer
    // [中文导读] [AllReduce逐行 S1807] 查询通信域持有的CCL中转缓冲区。
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    // CCL IN使用所有的CCL Buffer，这个其实就是scratch buffer
    // [中文导读] 把通信域中转缓冲区及算法请求的通知容量记入上下文，供模板分片与同步使用。
    // [中文导读] [AllReduce逐行 S1810] 以Device内存类型保存CCL中转区地址和大小。
    resCtxHost->cclMem = HcclMem{HCCL_MEM_TYPE_DEVICE, cclBufferAddr, cclBufferSize};
    // [中文导读] [AllReduce逐行 S1811] 保存算法主线程内部通知需求数。
    resCtxHost->notifyNumOnMainThread = resRequest.notifyNumOnMainThread;
    // [中文导读] [AllReduce逐行 S1812] 保存算法从线程数量。
    resCtxHost->slaveThreadNum = resRequest.slaveThreadNum;
    // [中文导读] [AllReduce逐行 S1813] 根据本次算子超时更新资源对象的通知/队列等待超时。
    UpdateAicpuTimeoutCtx(param, *resCtxHost);
    // [中文导读] [AllReduce逐行 S1814] 保存各从线程通知需求数。
    resCtxHost->notifyNumPerThread = resRequest.notifyNumPerThread;
    // [中文导读] 先取得执行 Thread，再取得每层通道；这一步准备资源，不进行用户数据传输。
    // [中文导读] [AllReduce逐行 S1816] 申请或复用本次算法使用的执行线程/展开线程。
    CHK_RET(HcclGetThread(comm, param, resRequest, resCtxHost, resPack));
    // [中文导读] [AllReduce逐行 S1817] 按Peer和通信层取得通道及远端内存属性。
    CHK_RET(HcclGetChannel(comm, param, resRequest, resCtxHost.get()));
    // [中文导读] [AllReduce逐行 S1818] 获得CCL缓冲区、线程和通道，写入可序列化资源对象处理完成，返回成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S1819] 结束HcclAllocAlgResourceAICPU函数体。
}

// [中文导读] [AllReduce逐行 S1821] 声明HcclGetThreadWithConfig接口：使用逐Thread配置申请设备主从线程，必要时申请Host展开线程。
static HcclResult HcclGetThreadWithConfig(
    // [中文导读] [AllReduce逐行 S1822] 函数参数包含通信域句柄、算子参数、线程/通道/通知资源需求、主从线程总数，本行延续接口声明。
    HcclComm comm, const OpParam& param, AlgResourceRequest& resRequest, u32 threadNum,
    // [中文导读] [AllReduce逐行 S1823] 函数参数包含Host资源描述对象、主从线程列表、展开线程是否已存在，本行延续接口声明。
    std::vector<ThreadHandle>& threads, std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, bool unfoldReady)
// [中文导读] [AllReduce逐行 S1824] 开始HcclGetThreadWithConfig的函数体。
{
    // [中文导读] [AllReduce逐行 S1825] 创建每个主从线程对应的ThreadConfig对象数组。
    std::vector<ThreadConfig> threadConfigs(threadNum);
    // [中文导读] [AllReduce逐行 S1826] 通过HCOMM接口初始化ThreadConfig ABI字段。
    CHK_RET(static_cast<HcclResult>(ThreadConfigInit(threadConfigs.data(), threadNum)));
    // [中文导读] 主 Thread 在算法需要的通知之外增加一个 Host/Device 依赖槽，各从 Thread 则按各自需求配置。
    // [中文导读] [AllReduce逐行 S1828] 主线程通知需求额外加1，最后一个槽用于Host输入就绪通知。
    threadConfigs[0].notifyNumPerThread = resRequest.notifyNumOnMainThread + 1; // 主流上多一个用于host-device同步
    // [中文导读] [AllReduce逐行 S1829] 输出调试日志，记录HcclGetThreadWithConfig当前阶段和相关参数。
    HCCL_DEBUG("[HcclGetThread] AICPU thread[0] notify num[%u].", threadConfigs[0].notifyNumPerThread);
    // [中文导读] [AllReduce逐行 S1830] 开始条件检查宏，条件成立时记录错误并返回下方指定错误码。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S1831] 检查各从线程的通知需求数组足够长。
        resRequest.notifyNumPerThread.size() < threadNum - 1,
        // [中文导读] [AllReduce逐行 S1832] 输出错误日志，记录HcclGetThreadWithConfig当前阶段和相关参数。
        HCCL_ERROR(
            // [中文导读] [AllReduce逐行 S1833] 补充日志格式：[HcclGetThread] notifyNumPerThread size[%zu] is less than slaveThreadNum[%u].。
            "[HcclGetThread] notifyNumPerThread size[%zu] is less than slaveThreadNum[%u].",
            // [中文导读] [AllReduce逐行 S1834] 错误日志中展示实际需求数组长度和从线程数。
            resRequest.notifyNumPerThread.size(), threadNum - 1),
        // [中文导读] [AllReduce逐行 S1835] 需求数组不足返回内部错误。
        HCCL_E_INTERNAL);
    // [中文导读] [AllReduce逐行 S1836] 从第1个线程开始逐一配置从线程通知容量。
    for (u32 i = 1; i < threadNum; i++) {
        // [中文导读] [AllReduce逐行 S1837] 第i个从线程取需求数组的第i-1个元素。
        threadConfigs[i].notifyNumPerThread = resRequest.notifyNumPerThread[i - 1];
        // [中文导读] [AllReduce逐行 S1838] 输出调试日志，记录HcclGetThreadWithConfig当前阶段和相关参数。
        HCCL_DEBUG("[HcclGetThread] AICPU thread[%u] notify num[%u].", i, threadConfigs[i].notifyNumPerThread);
    // [中文导读] [AllReduce逐行 S1839] 结束循环for (u32 i = 1; i < threadNum; i++)。
    }
    // [中文导读] [AllReduce逐行 S1840] 开始以逐Thread配置接口申请线程。
    CHK_RET(HcclThreadAcquireWithConfig(
        // [中文导读] [AllReduce逐行 S1841] 使用AICPU执行域及TS线程类型申请主从线程。
        comm, COMM_ENGINE_AICPU, threadNum, THREAD_TYPE_TS, threadConfigs.data(), threads.data()));
    // 申请展开流对应的Thread
    // [中文导读] [AllReduce逐行 S1843] 尚无展开线程时申请Host CPU展开线程。
    if (!unfoldReady) {
        // [中文导读] [AllReduce逐行 S1844] 创建Host展开线程配置结构。
        ThreadConfig unfoldThreadConfig;
        // [中文导读] [AllReduce逐行 S1845] 初始化该展开线程ThreadConfig ABI字段。
        CHK_RET(static_cast<HcclResult>(ThreadConfigInit(&unfoldThreadConfig, 1)));
        // 展开流需要一个Notify用于 AICPU按序下发
        // [中文导读] [AllReduce逐行 S1847] 展开线程保序只需要ORDER_UNFOLD_THREAD_NOTIFY_NUM个通知槽。
        unfoldThreadConfig.notifyNumPerThread = ORDER_UNFOLD_THREAD_NOTIFY_NUM;
        // [中文导读] [AllReduce逐行 S1848] 开始申请Host展开线程。
        CHK_RET(HcclThreadAcquireWithConfig(
            // [中文导读] [AllReduce逐行 S1849] 使用COMM_ENGINE_CPU及TS线程类型输出unfoldThread句柄。
            comm, COMM_ENGINE_CPU, 1, THREAD_TYPE_TS, &unfoldThreadConfig, &resCtxHost->unfoldThread));
    // [中文导读] [AllReduce逐行 S1850] 结束条件if (!unfoldReady)。
    }
    // [中文导读] [AllReduce逐行 S1851] 保存Device算法主线程句柄和包含Host同步槽的总通知数。
    CHK_RET(SaveMainThreadInfo(comm, param, threads[0], resRequest.notifyNumOnMainThread + 1));
    // [中文导读] [AllReduce逐行 S1852] 使用逐Thread配置申请设备主从线程，必要时申请Host展开线程处理完成，返回成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S1853] 结束HcclGetThreadWithConfig函数体。
}

// [中文导读] [AllReduce逐行 S1855] 声明GetMaxNotifyNum接口：从主从线程通知需求求最大值，用于旧接口的统一通知配置。
static u32 GetMaxNotifyNum(const std::vector<u32>& notifyNumPerThread, u32 initNotifyNum)
// [中文导读] [AllReduce逐行 S1856] 开始GetMaxNotifyNum的函数体。
{
    // [中文导读] [AllReduce逐行 S1857] 把初始主线程通知需求作为最大值起点。
    u32 maxNotifyNum = initNotifyNum;
    // [中文导读] [AllReduce逐行 S1858] 逐个读取从线程通知需求。
    for (u32 notifyNum : notifyNumPerThread) {
        // [中文导读] [AllReduce逐行 S1859] 从线程需求大于当前最大值时更新。
        if (notifyNum > maxNotifyNum) {
            // [中文导读] [AllReduce逐行 S1860] 保存较大的通知需求。
            maxNotifyNum = notifyNum;
        // [中文导读] [AllReduce逐行 S1861] 结束条件if (notifyNum > maxNotifyNum)。
        }
    // [中文导读] [AllReduce逐行 S1862] 结束循环for (u32 notifyNum : notifyNumPerThread)。
    }
    // [中文导读] [AllReduce逐行 S1863] 返回主从线程通知需求最大值。
    return maxNotifyNum;
// [中文导读] [AllReduce逐行 S1864] 结束GetMaxNotifyNum函数体。
}

// [中文导读] 设备主Thread和从Thread承载通信任务，另一个unfoldThread用于入口展开保序。
// [中文导读] 有WithConfig能力时可逐Thread设置通知数；旧接口分支按最大通知数统一申请，二者互斥。
// [中文导读] 主Thread额外保留Host/Device同步槽位，不能全部分给算法内部主从同步使用。
// [中文导读] [AllReduce逐行 S1869] 声明HcclGetAicpuThread接口：申请算法执行线程并复用/创建Host展开线程。
static HcclResult HcclGetAicpuThread(
    // [中文导读] [AllReduce逐行 S1870] 函数参数包含通信域句柄、算子参数、线程/通道/通知资源需求，本行延续接口声明。
    HcclComm comm, const OpParam& param, AlgResourceRequest& resRequest,
    // [中文导读] [AllReduce逐行 S1871] 函数参数包含Host资源描述对象，本行延续接口声明。
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost)
// [中文导读] [AllReduce逐行 S1872] 开始HcclGetAicpuThread的函数体。
{
    // [中文导读] [AllReduce逐行 S1873] 本算法线程总数等于从线程数量加一个主线程。
    u32 threadNum = resRequest.slaveThreadNum + 1;
    // [中文导读] [AllReduce逐行 S1874] 创建对应数量的线程句柄输出数组。
    std::vector<ThreadHandle> threads(threadNum);
    // [中文导读] [AllReduce逐行 S1875] 默认展开线程尚未准备好。
    bool unfoldReady = false;
    // [中文导读] [AllReduce逐行 S1876] 初始化可复用的展开线程句柄变量。
    ThreadHandle existingUnfoldThread = 0;
    // [中文导读] 展开 Thread 以通信域关联标识复用，与每个算法独立保存的主 Thread 信息不同。
    // [中文导读] [AllReduce逐行 S1878] 成功读到通信域已有展开线程时复用。
    if (GetUnfoldThreadInfo(comm, param, existingUnfoldThread) == HCCL_SUCCESS) {
        // [中文导读] [AllReduce逐行 S1879] 将旧展开线程句柄写入本次资源对象。
        resCtxHost->unfoldThread = existingUnfoldThread;
        // [中文导读] [AllReduce逐行 S1880] 标记无需重新申请展开线程。
        unfoldReady = true;
        // [中文导读] [AllReduce逐行 S1881] 输出运行日志，记录HcclGetAicpuThread当前阶段和相关参数。
        HCCL_INFO("[HcclGetThread] reuse unfoldThread [%lu]", resCtxHost->unfoldThread);
    // [中文导读] [AllReduce逐行 S1882] 结束条件if (GetUnfoldThreadInfo(comm, param, existingUnfoldThread) == HCCL_SUCCESS)。
    }
    // [中文导读] 具备配置接口时按 Thread 定义通知数量，否则统一采用本算法的最大通知数量。
    // [中文导读] [AllReduce逐行 S1884] 具备逐Thread配置接口时调用新接口申请主从线程。
    if (HcommIsSupportHcclThreadAcquireWithConfig()) {
        // [中文导读] [AllReduce逐行 S1885] 以通知需求配置申请AICPU执行线程，必要时创建展开线程。
        CHK_RET(HcclGetThreadWithConfig(comm, param, resRequest, threadNum, threads, resCtxHost, unfoldReady));
    // [中文导读] [AllReduce逐行 S1886] 上述条件不成立时进入替代分支。
    } else {
        // [中文导读] [AllReduce逐行 S1887] 旧接口按所有线程需求最大值统一申请通知容量。
        u32 maxNotifyNum = GetMaxNotifyNum(resRequest.notifyNumPerThread, resRequest.notifyNumOnMainThread);
        // [中文导读] [AllReduce逐行 S1888] 输出调试日志，记录HcclGetAicpuThread当前阶段和相关参数。
        HCCL_DEBUG("[HcclGetThread] require maxNotifyNum[%u] for all AICPU threads.", maxNotifyNum);
        // [中文导读] [AllReduce逐行 S1889] 统一通知容量加Host同步槽，申请AICPU_TS主从线程。
        CHK_RET(HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, threadNum, maxNotifyNum + 1, threads.data()));
        // [中文导读] [AllReduce逐行 S1890] 尚无展开线程时调用旧接口另行申请Host展开线程。
        if (!unfoldReady) {
            // 展开流需要一个Notify用于 AICPU按序下发
            // [中文导读] [AllReduce逐行 S1892] 开始旧接口展开线程申请调用。
            CHK_RET(
                // [中文导读] [AllReduce逐行 S1893] 使用CPU执行域和固定保序通知数申请展开线程。
                HcclThreadAcquire(comm, COMM_ENGINE_CPU, 1, ORDER_UNFOLD_THREAD_NOTIFY_NUM, &resCtxHost->unfoldThread));
        // [中文导读] [AllReduce逐行 S1894] 结束条件if (!unfoldReady)。
        }
        // [中文导读] [AllReduce逐行 S1895] 保存主线程句柄和最大通知容量加Host同步槽。
        CHK_RET(SaveMainThreadInfo(comm, param, threads[0], maxNotifyNum + 1));
    // [中文导读] [AllReduce逐行 S1896] 结束条件} else。
    }
    // [中文导读] [AllReduce逐行 S1897] 本次新建展开线程才需要写入Host缓存。
    if (!unfoldReady) {
        // [中文导读] [AllReduce逐行 S1898] 按通信域名保存新展开线程句柄。
        CHK_RET(SaveUnfoldThreadInfo(comm, param, resCtxHost->unfoldThread));
    // [中文导读] [AllReduce逐行 S1899] 结束条件if (!unfoldReady)。
    }
    // [中文导读] [AllReduce逐行 S1900] 输出运行日志，记录HcclGetAicpuThread当前阶段和相关参数。
    HCCL_INFO("[HcclGetThread] unfoldThread [%lu]", resCtxHost->unfoldThread);
    // [中文导读] [AllReduce逐行 S1901] 输出调试日志，记录HcclGetAicpuThread当前阶段和相关参数。
    HCCL_DEBUG("threads ptr is %p\n", threads.data());
    // [中文导读] [AllReduce逐行 S1902] 遍历本次申请的所有算法线程。
    for (u32 i = 0; i < threadNum; i++) {
        // [中文导读] [AllReduce逐行 S1903] 将每个主从线程句柄写入可序列化资源列表。
        resCtxHost->threads.push_back(threads[i]);
    // [中文导读] [AllReduce逐行 S1904] 结束循环for (u32 i = 0; i < threadNum; i++)。
    }
    // [中文导读] [AllReduce逐行 S1905] 申请算法执行线程并复用/创建Host展开线程处理完成，返回成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S1906] 结束HcclGetAicpuThread函数体。
}

// [中文导读] [AllReduce逐行 S1908] 声明HcclGetThread接口：按engine选择AICPU线程资源或Host流包装/图模式从流。
HcclResult HcclGetThread(
    // [中文导读] [AllReduce逐行 S1909] 函数参数包含通信域句柄、算子参数、线程/通道/通知资源需求，本行延续接口声明。
    HcclComm comm, const OpParam& param, AlgResourceRequest& resRequest,
    // [中文导读] [AllReduce逐行 S1910] 函数参数包含图模式资源包、Host资源描述对象，本行延续接口声明。
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, const ResPackGraphMode& resPack)
// [中文导读] [AllReduce逐行 S1911] 开始HcclGetThread的函数体。
{
    // [中文导读] [AllReduce逐行 S1912] 把ThreadAcquireWithConfig能力标志写到资源对象，Device据此定位Host同步槽。
    resCtxHost->isHcclThreadAcquireWithConfigSupported = HcommIsSupportHcclThreadAcquireWithConfig();
    // [中文导读] [AllReduce逐行 S1913] AICPU_TS或CPU引擎走AICPU线程资源获取。
    if ((param.engine == COMM_ENGINE_AICPU_TS) || (param.engine == COMM_ENGINE_CPU)) {
        // [中文导读] [AllReduce逐行 S1914] 调用AICPU主从线程及Host展开线程准备函数。
        CHK_RET(HcclGetAicpuThread(comm, param, resRequest, resCtxHost));
    // [中文导读] [AllReduce逐行 S1915] 上述条件不成立时进入替代分支。
    } else {
        // host模式下，将主流封装为thread，并创建主流上的notify
        // [中文导读] [AllReduce逐行 S1917] 声明其它引擎主Thread句柄。
        ThreadHandle thread;
        // [中文导读] [AllReduce逐行 S1918] 开始按用户ACL stream包装其它引擎主线程。
        CHK_RET(
            // [中文导读] [AllReduce逐行 S1919] 用算法主线程通知需求配置流包装，并输出Thread句柄。
            HcclThreadAcquireWithStream(comm, param.engine, param.stream, resRequest.notifyNumOnMainThread, &thread));
        // [中文导读] [AllReduce逐行 S1920] 将该主线程加入资源对象。
        resCtxHost->threads.push_back(thread);

        // [中文导读] [AllReduce逐行 S1922] 计算其它引擎从线程的最大通知需求。
        u32 maxNotifyNum = GetMaxNotifyNum(resRequest.notifyNumPerThread, 0);
        // [中文导读] [AllReduce逐行 S1923] 从GE资源包或普通接口准备其它引擎的从线程。
        CHK_RET(GeGetThread(comm, param, resRequest, resCtxHost, resPack, maxNotifyNum));
    // [中文导读] [AllReduce逐行 S1924] 结束条件} else。
    }

    // [中文导读] [AllReduce逐行 S1926] 仅DEBUG日志级别开启时遍历打印线程信息。
    if (UNLIKELY(HcclCheckLogLevel(DLOG_DEBUG))) {
        // [中文导读] [AllReduce逐行 S1927] 输出调试日志，记录HcclGetThread当前阶段和相关参数。
        HCCL_DEBUG("[HcclGetThread] slaveThreadNum[%u]", resRequest.slaveThreadNum);
        // [中文导读] [AllReduce逐行 S1928] 遍历所有主从线程句柄进行调试输出。
        for (u32 i = 0; i < resRequest.slaveThreadNum + 1; i++) {
            // [中文导读] [AllReduce逐行 S1929] 输出调试日志，记录HcclGetThread当前阶段和相关参数。
            HCCL_DEBUG("[HcclGetThread] threads[%u]=[%llu]", i, resCtxHost->threads[i]);
        // [中文导读] [AllReduce逐行 S1930] 结束循环for (u32 i = 0; i < resRequest.slaveThreadNum + 1; i++)。
        }
    // [中文导读] [AllReduce逐行 S1931] 结束条件if (UNLIKELY(HcclCheckLogLevel(DLOG_DEBUG)))。
    }
    // [中文导读] [AllReduce逐行 S1932] 按engine选择AICPU线程资源或Host流包装/图模式从流处理完成，返回成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S1933] 结束HcclGetThread函数体。
}

HcclResult GeGetThread(
    HcclComm comm, const OpParam& param, AlgResourceRequest& resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, const ResPackGraphMode& resPack, u32 maxNotifyNum)
{
    if (param.opMode == OpMode::OPBASE) {
        u32 threadNum = resRequest.slaveThreadNum;
        if (threadNum > 0) {
            std::vector<ThreadHandle> threads(threadNum);
            if (HcommIsSupportHcclThreadAcquireWithConfig()) {
                std::vector<ThreadConfig> threadConfigs(threadNum);
                CHK_RET(static_cast<HcclResult>(ThreadConfigInit(threadConfigs.data(), threadNum)));
                CHK_PRT_RET(
                    resRequest.notifyNumPerThread.size() < threadNum,
                    HCCL_ERROR(
                        "[GeGetThread] notifyNumPerThread size[%zu] is less than slaveThreadNum[%u].",
                        resRequest.notifyNumPerThread.size(), threadNum),
                    HCCL_E_INTERNAL);
                for (u32 i = 0; i < threadNum; i++) {
                    threadConfigs[i].notifyNumPerThread = resRequest.notifyNumPerThread[i];
                }
                CHK_RET(HcclThreadAcquireWithConfig(
                    comm, COMM_ENGINE_CPU, threadNum, THREAD_TYPE_TS, threadConfigs.data(), threads.data()));
            } else {
                CHK_RET(HcclThreadAcquire(comm, param.engine, threadNum, maxNotifyNum, threads.data()));
            }
            for (u32 i = 0; i < threadNum; i++) {
                resCtxHost->threads.push_back(threads[i]);
            }
        }
    } else {
        u32 slaveStreams = resPack.streams.size();
        u32 threadNum = resRequest.slaveThreadNum;
        if (threadNum > slaveStreams) {
            HCCL_ERROR(
                "Thread Num Should less than slave streams. slaveStreams[%llu], threadNums[%llu]", slaveStreams,
                threadNum);
            return HCCL_E_UNAVAIL;
        }

        for (u32 i = 0; i < threadNum; i++) {
            ThreadHandle slaveThread;
            CHK_RET(HcclThreadAcquireWithStream(comm, param.engine, resPack.streams[i], maxNotifyNum, &slaveThread));
            resCtxHost->threads.push_back(slaveThread);
        }
    }

    return HCCL_SUCCESS;
}

// [中文导读] [AllReduce逐行 S1984] 声明SaveMainThreadInfo接口：按算法tag保存主Thread和通知容量。
HcclResult SaveMainThreadInfo(HcclComm comm, const OpParam& param, ThreadHandle thread, u32 notifyNum)
// [中文导读] [AllReduce逐行 S1985] 开始SaveMainThreadInfo的函数体。
{
    // [中文导读] [AllReduce逐行 S1986] 主线程缓存长度为一个ThreadHandle加一个u32通知数。
    uint64_t size = sizeof(ThreadHandle) + sizeof(u32);
    // [中文导读] [AllReduce逐行 S1987] 初始化Host主线程信息上下文地址。
    void* ctx = nullptr;
    // 申请一块host类型内存，保存主流信息
    // [中文导读] [AllReduce逐行 S1989] 按算法tag创建CPU_TS Host存储区。
    CHK_RET(HcclEngineCtxCreate(comm, param.algTag, CommEngine::COMM_ENGINE_CPU_TS, size, &ctx));
    // 填充主流handle信息
    // [中文导读] [AllReduce逐行 S1991] 将Host上下文起始地址解释为ThreadHandle存储位置。
    ThreadHandle* threadPtr = reinterpret_cast<ThreadHandle*>(ctx);
    // [中文导读] [AllReduce逐行 S1992] 把算法Device主线程句柄写入上下文首段。
    *threadPtr = thread;
    // 填充主流notify数量信息
    // [中文导读] [AllReduce逐行 S1994] 将上下文地址转成字节指针便于移动偏移。
    char* curPtr = reinterpret_cast<char*>(ctx);
    // [中文导读] [AllReduce逐行 S1995] 跳过ThreadHandle长度，定位通知数存储段。
    curPtr += sizeof(ThreadHandle);
    // [中文导读] [AllReduce逐行 S1996] 将通知数存储段解释为u32指针。
    u32* notifyNumPtr = reinterpret_cast<u32*>(curPtr);
    // [中文导读] [AllReduce逐行 S1997] 保存主线程总通知槽数。
    *notifyNumPtr = notifyNum;
    // [中文导读] [AllReduce逐行 S1998] 输出运行日志，记录SaveMainThreadInfo当前阶段和相关参数。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S1999] 补充日志格式：[SaveMainThreadInfo]threadPtr[%p], thread[%lu], notifyNumPtr[%p], notifyNum[%lu]", threadPtr, thread。
        "[SaveMainThreadInfo]threadPtr[%p], thread[%lu], notifyNumPtr[%p], notifyNum[%lu]", threadPtr, thread,
        // [中文导读] [AllReduce逐行 S2000] 提供上述日志的实参，涉及主线程通知容量。
        notifyNumPtr, notifyNum);
    // [中文导读] [AllReduce逐行 S2001] 按算法tag保存主Thread和通知容量处理完成，返回成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S2002] 结束SaveMainThreadInfo函数体。
}

// [中文导读] [AllReduce逐行 S2004] 声明SaveUnfoldThreadInfo接口：按通信域名字保存Host展开Thread。
HcclResult SaveUnfoldThreadInfo(HcclComm comm, const OpParam& param, ThreadHandle unfoldThread)
// [中文导读] [AllReduce逐行 S2005] 开始SaveUnfoldThreadInfo的函数体。
{
    // [中文导读] [AllReduce逐行 S2006] 展开线程缓存只存放一个ThreadHandle。
    uint64_t size = sizeof(ThreadHandle);
    // [中文导读] [AllReduce逐行 S2007] 初始化展开线程Host缓存地址。
    void* ctx = nullptr;
    // 申请一块host类型内存，保存展开流信息
    // [中文导读] [AllReduce逐行 S2009] 创建保存通信域展开线程缓存tag的固定长度字符数组。
    char unfoldAlgTag[ALG_TAG_LENGTH] = {0};
    // [中文导读] [AllReduce逐行 S2010] 用通信域名加_unfold构造展开线程缓存tag。
    int ret = snprintf_s(unfoldAlgTag, sizeof(unfoldAlgTag), sizeof(unfoldAlgTag) - 1, "%s_unfold", param.commName);
    // [中文导读] [AllReduce逐行 S2011] 展开tag构造失败返回内部错误。
    CHK_PRT_RET(ret <= 0, HCCL_ERROR("[%s] failed to fill unfoldAlgTag", __func__), HCCL_E_INTERNAL);
    // [中文导读] [AllReduce逐行 S2012] 按通信域展开tag创建CPU_TS Host上下文。
    CHK_RET(HcclEngineCtxCreate(comm, unfoldAlgTag, CommEngine::COMM_ENGINE_CPU_TS, size, &ctx));
    // 填充展开流handle信息
    // [中文导读] [AllReduce逐行 S2014] 将Host上下文首地址解释为ThreadHandle存储位置。
    ThreadHandle* threadPtr = reinterpret_cast<ThreadHandle*>(ctx);
    // [中文导读] [AllReduce逐行 S2015] 保存Host展开线程句柄。
    *threadPtr = unfoldThread;
    // [中文导读] [AllReduce逐行 S2016] 输出运行日志，记录SaveUnfoldThreadInfo当前阶段和相关参数。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S2017] 补充日志格式：[SaveUnfoldThreadInfo]unfoldAlgTag[%s], threadPtr[%p], unfoldThread[%lu]", unfoldAlgTag, threadPtr。
        "[SaveUnfoldThreadInfo]unfoldAlgTag[%s], threadPtr[%p], unfoldThread[%lu]", unfoldAlgTag, threadPtr,
        // [中文导读] [AllReduce逐行 S2018] 提供上述日志的实参，涉及Host展开线程句柄。
        unfoldThread);
    // [中文导读] [AllReduce逐行 S2019] 按通信域名字保存Host展开Thread处理完成，返回成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S2020] 结束SaveUnfoldThreadInfo函数体。
}

// [中文导读] [AllReduce逐行 S2022] 声明GetUnfoldThreadInfo接口：读取通信域的Host展开Thread。
HcclResult GetUnfoldThreadInfo(HcclComm comm, const OpParam& param, ThreadHandle& unfoldThread)
// [中文导读] [AllReduce逐行 S2023] 开始GetUnfoldThreadInfo的函数体。
{
    // [中文导读] [AllReduce逐行 S2024] 设置展开线程缓存查询预期长度为sizeof(ThreadHandle)。
    uint64_t size = sizeof(ThreadHandle);
    // [中文导读] [AllReduce逐行 S2025] 初始化Host上下文地址输出。
    void* ctx = nullptr;
    // [中文导读] [AllReduce逐行 S2026] 创建展开线程缓存tag数组，初始化为空。
    char unfoldAlgTag[ALG_TAG_LENGTH] = {0};
    // [中文导读] [AllReduce逐行 S2027] 按通信域名字构造_unfold tag。
    int ret = snprintf_s(unfoldAlgTag, sizeof(unfoldAlgTag), sizeof(unfoldAlgTag) - 1, "%s_unfold", param.commName);
    // [中文导读] [AllReduce逐行 S2028] tag字符串格式化失败返回内部错误。
    CHK_PRT_RET(ret <= 0, HCCL_ERROR("[%s] failed to fill unfoldAlgTag", __func__), HCCL_E_INTERNAL);
    // [中文导读] [AllReduce逐行 S2029] 查询CPU_TS Host展开线程上下文。
    HcclResult getRet = HcclEngineCtxGet(comm, unfoldAlgTag, CommEngine::COMM_ENGINE_CPU_TS, &ctx, &size);
    // [中文导读] [AllReduce逐行 S2030] 查询失败时直接返回，调用方可决定是否新建展开线程。
    if (getRet != HCCL_SUCCESS) {
        // [中文导读] [AllReduce逐行 S2031] 返回展开线程上下文查询错误。
        return getRet;
    // [中文导读] [AllReduce逐行 S2032] 结束条件if (getRet != HCCL_SUCCESS)。
    }
    // 获取展开流handle信息
    // [中文导读] [AllReduce逐行 S2034] 把Host上下文解释为ThreadHandle存储位置。
    ThreadHandle* threadPtr = reinterpret_cast<ThreadHandle*>(ctx);
    // [中文导读] [AllReduce逐行 S2035] 读取Host展开线程句柄到输出参数。
    unfoldThread = *threadPtr;
    // [中文导读] [AllReduce逐行 S2036] 输出运行日志，记录GetUnfoldThreadInfo当前阶段和相关参数。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S2037] 补充日志格式：[GetUnfoldThreadInfo]unfoldAlgTag[%s], threadPtr[%p], unfoldThread[%lu]", unfoldAlgTag, threadPtr。
        "[GetUnfoldThreadInfo]unfoldAlgTag[%s], threadPtr[%p], unfoldThread[%lu]", unfoldAlgTag, threadPtr,
        // [中文导读] [AllReduce逐行 S2038] 提供上述日志的实参，涉及Host展开线程句柄。
        unfoldThread);
    // [中文导读] [AllReduce逐行 S2039] 读取通信域的Host展开Thread处理完成，返回成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S2040] 结束GetUnfoldThreadInfo函数体。
}

// [中文导读] [AllReduce逐行 S2042] 声明GetMainThreadInfo接口：读取算法主Thread及通知容量。
HcclResult GetMainThreadInfo(HcclComm comm, const OpParam& param, ThreadHandle& thread, u32& notifyNum)
// [中文导读] [AllReduce逐行 S2043] 开始GetMainThreadInfo的函数体。
{
    // [中文导读] [AllReduce逐行 S2044] 主线程信息的预期长度为ThreadHandle加u32通知数。
    uint64_t size = sizeof(ThreadHandle) + sizeof(u32);
    // [中文导读] [AllReduce逐行 S2045] 初始化主线程信息Host上下文查询地址。
    void* ctx = nullptr;
    // [中文导读] [AllReduce逐行 S2046] 按本算法algTag查询CPU_TS Host主线程信息上下文。
    CHK_RET(HcclEngineCtxGet(comm, param.algTag, CommEngine::COMM_ENGINE_CPU_TS, &ctx, &size));

    // 获取主流handle信息
    // [中文导读] [AllReduce逐行 S2049] 将缓存首地址解释为ThreadHandle存储位置。
    ThreadHandle* threadPtr = reinterpret_cast<ThreadHandle*>(ctx);
    // [中文导读] [AllReduce逐行 S2050] 读取算法Device主线程句柄。
    thread = *threadPtr;
    // 获取主流notify数量信息
    // [中文导读] [AllReduce逐行 S2052] 将上下文地址转换为字节指针。
    char* curPtr = reinterpret_cast<char*>(ctx);
    // [中文导读] [AllReduce逐行 S2053] 跳过ThreadHandle定位通知数存储段。
    curPtr += sizeof(ThreadHandle);
    // [中文导读] [AllReduce逐行 S2054] 将通知数所在字节解释为u32指针。
    u32* notifyNumPtr = reinterpret_cast<u32*>(curPtr);
    // [中文导读] [AllReduce逐行 S2055] 读取主线程总通知槽数到输出参数。
    notifyNum = *notifyNumPtr;
    // [中文导读] [AllReduce逐行 S2056] 输出运行日志，记录GetMainThreadInfo当前阶段和相关参数。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S2057] 补充日志格式：[GetMainThreadInfo]threadPtr[%p], thread[%lu], notifyNumPtr[%p], notifyNum[%lu]", threadPtr, thread。
        "[GetMainThreadInfo]threadPtr[%p], thread[%lu], notifyNumPtr[%p], notifyNum[%lu]", threadPtr, thread,
        // [中文导读] [AllReduce逐行 S2058] 提供上述日志的实参，涉及主线程通知容量。
        notifyNumPtr, notifyNum);
    // [中文导读] [AllReduce逐行 S2059] 读取算法主Thread及通知容量处理完成，返回成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S2060] 结束GetMainThreadInfo函数体。
}

// [中文导读] 按算法层级收集请求，再按本地Endpoint位于Device还是Host拆分，分别选建链Engine。
// [中文导读] 图模式先注册输入输出内存，普通单算子模式不因此自动执行同样的注册分支。
// [中文导读] [AllReduce逐行 S2064] 声明HcclGetChannel接口：按算法层和端点所在位置分组申请通道，OFFLOAD额外注册用户区。
HcclResult HcclGetChannel(
    // [中文导读] [AllReduce逐行 S2065] 函数参数包含通信域句柄、算子参数、Host资源描述对象、线程/通道/通知资源需求，本行延续接口声明。
    HcclComm comm, const OpParam& param, AlgResourceRequest& resRequest, AlgResourceCtxSerializable* resCtxHost)
// [中文导读] [AllReduce逐行 S2066] 开始HcclGetChannel的函数体。
{
    // [中文导读] [AllReduce逐行 S2067] 创建图模式用户内存注册信息；OPBASE本例不注册图用户区。
    MemRegInfo memRegInfo;
    // [中文导读] 图模式先注册输入和输出内存，使建链能交换对应用户区的远端访问描述。
    // [中文导读] [AllReduce逐行 S2069] OFFLOAD图模式才注册本次用户输入输出区。
    if (param.opMode == OpMode::OFFLOAD) {
        // [中文导读] [AllReduce逐行 S2070] 输出运行日志，记录HcclGetChannel当前阶段和相关参数。
        HCCL_INFO("[HcclGetChannelImpl] start to RegGraphModeBuffers");
        // [中文导读] [AllReduce逐行 S2071] 开始图模式用户区注册调用。
        CHK_RET(
            // [中文导读] [AllReduce逐行 S2072] 注册用户输入/输出内存并取得tag及内存注册句柄。
            RegGraphModeBuffers(comm, param, memRegInfo.inputBuffTag, memRegInfo.outputBuffTag, memRegInfo.memHandles));
    // [中文导读] [AllReduce逐行 S2073] 结束条件if (param.opMode == OpMode::OFFLOAD)。
    }
    // [中文导读] [AllReduce逐行 S2074] 将资源对象的通道层数量扩展为算法请求的层数。
    resCtxHost->channels.resize(resRequest.channels.size());
    // [中文导读] 逐算法层处理建链请求，每层再按 Endpoint 所在位置拆成 Device 和 Host 两组。
    // [中文导读] [AllReduce逐行 S2076] 逐一处理算法层级的通道请求。
    for (u32 level = 0; level < resRequest.channels.size(); level++) {
        // 获取子通信域的建链请求
        // [中文导读] [AllReduce逐行 S2078] 取当前算法层的建链请求数组引用。
        std::vector<HcclChannelDesc>& levelNChannelRequest = resRequest.channels[level];
        // [中文导读] [AllReduce逐行 S2079] 创建当前层Device端点通道请求组。
        std::vector<HcclChannelDesc> deviceChannelRequest;
        // [中文导读] [AllReduce逐行 S2080] 创建当前层Host端点通道请求组。
        std::vector<HcclChannelDesc> hostChannelRequest;
        // [中文导读] [AllReduce逐行 S2081] 遍历当前层各个通道请求。
        for (auto& channelRequest : levelNChannelRequest) {
            // [中文导读] [AllReduce逐行 S2082] 本地端点位于Device时归入Device建链组。
            if (channelRequest.localEndpoint.loc.locType == ENDPOINT_LOC_TYPE_DEVICE) {
                // [中文导读] [AllReduce逐行 S2083] 复制该通道描述到Device建链请求组。
                deviceChannelRequest.emplace_back(channelRequest);
            // [中文导读] [AllReduce逐行 S2084] 本地端点位于Host时归入Host建链组。
            } else if (channelRequest.localEndpoint.loc.locType == ENDPOINT_LOC_TYPE_HOST) {
                // [中文导读] [AllReduce逐行 S2085] 复制该通道描述到Host建链请求组。
                hostChannelRequest.emplace_back(channelRequest);
            // [中文导读] [AllReduce逐行 S2086] 结束条件} else if (channelRequest.localEndpoint.loc.locType == ENDPOINT_LOC_TYPE_HOST)。
            }
        // [中文导读] [AllReduce逐行 S2087] 结束循环for (auto& channelRequest : levelNChannelRequest)。
        }
        // device建链
        // [中文导读] [AllReduce逐行 S2089] 开始取得当前层Device组通道。
        CHK_RET(
            // [中文导读] [AllReduce逐行 S2090] Device端点使用AICPU_TS执行域申请通道。
            HcclGetChannelImpl(level, comm, param, deviceChannelRequest, COMM_ENGINE_AICPU_TS, resCtxHost, memRegInfo));
        // host建链
        // [中文导读] [AllReduce逐行 S2092] Host端点使用CPU执行域申请通道，空组直接返回成功。
        CHK_RET(HcclGetChannelImpl(level, comm, param, hostChannelRequest, COMM_ENGINE_CPU, resCtxHost, memRegInfo));
    // [中文导读] [AllReduce逐行 S2093] 结束循环for (u32 level = 0; level < resRequest.channels.size(); level++)。
    }
    // [中文导读] [AllReduce逐行 S2094] 按算法层和端点所在位置分组申请通道，OFFLOAD额外注册用户区处理完成，返回成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S2095] 结束HcclGetChannel函数体。
}

// [中文导读] 把已取得的ChannelHandle补成算法所需ChannelInfo，包含Peer、协议、端口属性和远端CCL地址。
// [中文导读] 远端地址是通信描述，不能在Host上当成本机可解引用指针使用。
// [中文导读] [AllReduce逐行 S2099] 声明BuildChannelInfo接口：从句柄构造模板使用的通道描述，查询端口属性与远端CCL地址。
static HcclResult BuildChannelInfo(
    // [中文导读] [AllReduce逐行 S2100] 函数参数包含通信域句柄、算子参数、HCOMM通道句柄、通道请求描述、本端rank，本行延续接口声明。
    HcclComm comm, const OpParam& param, const HcclChannelDesc& channelDesc, ChannelHandle channelHandle, u32 userRank,
    // [中文导读] [AllReduce逐行 S2101] 函数参数包含模板通道信息、图模式内存注册信息，本行延续接口声明。
    MemRegInfo& memRegInfo, ChannelInfo& channel)
// [中文导读] [AllReduce逐行 S2102] 开始BuildChannelInfo的函数体。
{
    // 对于真实建链的链路进行填充
    // [中文导读] 把取得的句柄与请求中的远端 Rank、协议和通知容量组合成模板可使用的通道描述。
    // [中文导读] [AllReduce逐行 S2105] 初始化模板通道有效标志为true。
    channel.isValid = true;
    // [中文导读] [AllReduce逐行 S2106] 保存本通道远端用户rank。
    channel.remoteRank = channelDesc.remoteRank;
    // [中文导读] [AllReduce逐行 S2107] 保存通道采用的传输协议。
    channel.protocol = channelDesc.channelProtocol;
    // [中文导读] [AllReduce逐行 S2108] 保存远端端点位于Host或Device的位置类型。
    channel.locationType = channelDesc.remoteEndpoint.loc.locType;
    // [中文导读] [AllReduce逐行 S2109] 保存本通道通知容量。
    channel.notifyNum = channelDesc.notifyNum;
    // [中文导读] [AllReduce逐行 S2110] 保存HCOMM通道句柄供数据原语使用。
    channel.handle = channelHandle;
// [中文导读] [AllReduce逐行 S2111] 仅未定义AICPU_COMPILE时编译以下Host端查询逻辑。
#ifndef AICPU_COMPILE
    // [中文导读] [AllReduce逐行 S2112] 拷贝本地端点描述用于查询物理端口属性。
    EndpointDesc localEndpoint = channelDesc.localEndpoint;
    // [中文导读] [AllReduce逐行 S2113] 定义端口带宽系数查询结果类型为32位无符号整数。
    using portSizeType = uint32_t;
    // [中文导读] [AllReduce逐行 S2114] 保存该属性查询结果字节长度。
    const uint32_t portSizeTypeSize = sizeof(portSizeType);
    // [中文导读] [AllReduce逐行 S2115] 将端口带宽系数结果初始化为0。
    portSizeType portSize = 0;
    // [中文导读] 查询端口带宽系数，后面的多通道分片会依据这些端口组信息分配数据。
    // [中文导读] [AllReduce逐行 S2117] 开始查询本地端点的带宽系数。
    CHK_RET(HcclRankGraphGetEndpointInfo(
        // [中文导读] [AllReduce逐行 S2118] 按本端rank和Endpoint查询ENDPOINT_ATTR_BW_COEFF属性。
        comm, userRank, &localEndpoint, ENDPOINT_ATTR_BW_COEFF, portSizeTypeSize, static_cast<void*>(&portSize)));
    // [中文导读] [AllReduce逐行 S2119] 保存端口分组带宽系数到ChannelInfo。
    channel.portGroupSize = portSize;
    // [中文导读] [AllReduce逐行 S2120] 开始条件检查宏，条件成立时记录错误并返回下方指定错误码。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S2121] 带宽系数为0时视为内部错误，不能继续建立有效通道描述。
        portSize == 0, HCCL_ERROR("[HcclGetChannelImpl] userRank [%d], portSize [%u] is 0.", userRank, portSize),
        // [中文导读] [AllReduce逐行 S2122] 带宽系数非法时返回内部错误。
        HcclResult::HCCL_E_INTERNAL);
    // [中文导读] [AllReduce逐行 S2123] 初始化端点dieId为无效值。
    EndpointAttrDieId dieId = INVALID_VALUE_RANKID;
    // [中文导读] [AllReduce逐行 S2124] 保存dieId属性查询的字节长度。
    const uint32_t dieIdSize = sizeof(EndpointAttrDieId);
    // [中文导读] [AllReduce逐行 S2125] 声明端点dieId查询结果返回码。
    HcclResult dieIdRet = HcclRankGraphGetEndpointInfo(
        // [中文导读] [AllReduce逐行 S2126] 查询本地端点所属die编号。
        comm, userRank, &localEndpoint, ENDPOINT_ATTR_DIE_ID, dieIdSize, static_cast<void*>(&dieId));
    // [中文导读] [AllReduce逐行 S2127] 查询成功时保存die编号。
    if (dieIdRet == HCCL_SUCCESS) {
        // [中文导读] [AllReduce逐行 S2128] 写入查询所得dieId。
        channel.dieId = dieId;
    // [中文导读] [AllReduce逐行 S2129] 上述条件不成立时进入替代分支。
    } else {
        // [中文导读] [AllReduce逐行 S2130] 输出警告日志，记录BuildChannelInfo当前阶段和相关参数。
        HCCL_WARNING(
            // [中文导读] [AllReduce逐行 S2131] 补充日志格式：[HcclGetChannelImpl] failed to get dieId for userRank[%u], remoteRank[%u]。
            "[HcclGetChannelImpl] failed to get dieId for userRank[%u], remoteRank[%u], "
            // [中文导读] [AllReduce逐行 S2132] 补充日志格式：ret[0x%016llx]. POD convergence adjustment will not be used for this channel.。
            "ret[0x%016llx]. POD convergence adjustment will not be used for this channel.",
            // [中文导读] [AllReduce逐行 S2133] 提供上述日志的实参，涉及模板通道信息、本端rank。
            userRank, channel.remoteRank, HCCL_ERROR_CODE(dieIdRet));
    // [中文导读] [AllReduce逐行 S2134] 结束条件} else。
    }
// [中文导读] [AllReduce逐行 S2135] 非Host编译不查询上述物理端口属性。
#endif
    // [中文导读] 查出此通道对端的 CCL 中转区地址和容量，保存在远端描述中供 Read/Write 构造切片。
    // [中文导读] [AllReduce逐行 S2137] 初始化远端CCL缓冲区地址输出为空。
    void* remoteCclBufferAddr = nullptr;
    // [中文导读] [AllReduce逐行 S2138] 初始化远端CCL容量输出为零。
    uint64_t remoteCclBufferSize = 0;
    // [中文导读] [AllReduce逐行 S2139] 根据通道句柄查询对端CCL中转区地址和大小。
    CHK_RET(HcclChannelGetHcclBuffer(comm, channelHandle, &remoteCclBufferAddr, &remoteCclBufferSize));
    // [中文导读] [AllReduce逐行 S2140] 以Device内存类型保存远端CCL区描述，Host不能直接解引用该远端地址。
    channel.remoteCclMem = HcclMem{HCCL_MEM_TYPE_DEVICE, remoteCclBufferAddr, remoteCclBufferSize};
    // [中文导读] [AllReduce逐行 S2141] 输出运行日志，记录BuildChannelInfo当前阶段和相关参数。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S2142] 补充日志格式：[%s]remoteRank[%u] protocol[%u] portGroupSize[%u] dieId[%u]。
        "[%s]remoteRank[%u] protocol[%u] portGroupSize[%u] dieId[%u] "
        // [中文导读] [AllReduce逐行 S2143] 补充日志格式：remoteCclBufferAddr[0x%llx] remoteCclBufferSize[%u]。
        "remoteCclBufferAddr[0x%llx] remoteCclBufferSize[%u]",
        // [中文导读] [AllReduce逐行 S2144] 提供上述日志的实参，涉及通道请求描述、模板通道信息。
        __func__, channelDesc.remoteRank, channelDesc.channelProtocol, channel.portGroupSize, channel.dieId,
        // [中文导读] [AllReduce逐行 S2145] 提供上述日志的实参：remoteCclBufferAddr, remoteCclBufferSize。
        remoteCclBufferAddr, remoteCclBufferSize);

    // [中文导读] [AllReduce逐行 S2147] OFFLOAD模式还需查询对端已注册的用户输入输出区。
    if (param.opMode == OpMode::OFFLOAD) {
        // [中文导读] [AllReduce逐行 S2148] 通过注册tag取得该通道对端的图模式用户区描述。
        CHK_RET(GetGraphModeBuffers(comm, channelHandle, memRegInfo.inputBuffTag, memRegInfo.outputBuffTag, channel));
    // [中文导读] [AllReduce逐行 S2149] 结束条件if (param.opMode == OpMode::OFFLOAD)。
    }
    // [中文导读] [AllReduce逐行 S2150] 从句柄构造模板使用的通道描述，查询端口属性与远端CCL地址处理完成，返回成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S2151] 结束BuildChannelInfo函数体。
}

// [中文导读] 一组同类通道的申请边界：空请求直接返回；有请求才登记一致性信息并调用HcclChannelAcquire。
// [中文导读] 获取句柄之后再查远端CCL等属性。Acquire内部既可能复用，也可能创建和连接底层资源。
// [中文导读] [AllReduce逐行 S2155] 声明HcclGetChannelImpl接口：登记交换信息并申请通道，逐条构建ChannelInfo存入对应层。
HcclResult HcclGetChannelImpl(
    // [中文导读] [AllReduce逐行 S2156] 函数参数包含通信域句柄、算子参数、建链描述请求数组、算法通信层序号，本行延续接口声明。
    const u32 level, HcclComm comm, const OpParam& param, std::vector<HcclChannelDesc>& channelRequest,
    // [中文导读] [AllReduce逐行 S2157] 函数参数包含Host资源描述对象、图模式内存注册信息、通道执行域，本行延续接口声明。
    const CommEngine commEngine, AlgResourceCtxSerializable* resCtxHost, MemRegInfo& memRegInfo)
// [中文导读] [AllReduce逐行 S2158] 开始HcclGetChannelImpl的函数体。
{
    // 获取子通信域的建链数量
    // [中文导读] 无请求的这一组无需登记交换信息或调用建链接口，直接成功返回。
    // [中文导读] [AllReduce逐行 S2161] 请求组为空时无需登记交换信息或创建通道。
    if (channelRequest.empty()) {
        // [中文导读] [AllReduce逐行 S2162] 输出运行日志，记录HcclGetChannelImpl当前阶段和相关参数。
        HCCL_INFO("[HcclGetChannelImpl] channelRequest is empty");
        // [中文导读] [AllReduce逐行 S2163] 空组直接成功返回。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S2164] 结束条件if (channelRequest.empty())。
    }
    // [中文导读] [AllReduce逐行 S2165] 保存本组申请通道数量。
    u32 channelNum = channelRequest.size();
    // [中文导读] [AllReduce逐行 S2166] 创建通道句柄返回数组。
    std::vector<ChannelHandle> levelNChannels;
    // [中文导读] [AllReduce逐行 S2167] 将句柄数组长度扩展为本组请求数量。
    levelNChannels.resize(channelNum);
    // [中文导读] 图模式把此前注册得到的内存句柄挂入每条请求，以便通道交换远端用户区描述。
    // [中文导读] [AllReduce逐行 S2169] OFFLOAD模式才附加此前注册的用户内存句柄。
    if (param.opMode == OpMode::OFFLOAD) {
        // [中文导读] [AllReduce逐行 S2170] 逐一设置本组图模式通道描述。
        for (auto& channelDesc : channelRequest) {
            // [中文导读] [AllReduce逐行 S2171] 把图模式内存注册句柄数组地址挂到通道描述。
            channelDesc.memHandles = memRegInfo.memHandles.data();
            // [中文导读] [AllReduce逐行 S2172] 设置图模式内存注册句柄数量。
            channelDesc.memHandleNum = memRegInfo.memHandles.size();
        // [中文导读] [AllReduce逐行 S2173] 结束循环for (auto& channelDesc : channelRequest)。
        }
    // [中文导读] [AllReduce逐行 S2174] 结束条件if (param.opMode == OpMode::OFFLOAD)。
    }
    // [中文导读] [AllReduce逐行 S2175] 本组有请求时登记算子交换信息并取得通道。
    if (channelNum > 0) {
        // 参数一致性校验信息注册到通信域，HcclChannelAcquire内部存在读清动作，每次调用前均需注册
        // [中文导读] 每次 Acquire 前重新登记算子元信息，因为下层建链流程会读取并清理这份交换信息。
        // [中文导读] [AllReduce逐行 S2178] 按一致性校验标志登记供本次建链读取的算子元信息。
        CHK_RET(AddExchangeInfo(comm, param));
        // [中文导读] [AllReduce逐行 S2179] 通过HCOMM通信域API取得整组通道；内部可能复用或新建。
        CHK_RET(HcclChannelAcquire(comm, commEngine, channelRequest.data(), channelNum, levelNChannels.data()));
    // [中文导读] [AllReduce逐行 S2180] 结束条件if (channelNum > 0)。
    }

    // [中文导读] 取得所有句柄后逐条补齐远端内存及端口属性，存入对应算法层的通道列表。
    // [中文导读] [AllReduce逐行 S2183] 遍历本组取得的通道句柄。
    for (u32 idx = 0; idx < channelNum; idx++) {
        // [中文导读] [AllReduce逐行 S2184] 创建单个模板通道描述对象。
        ChannelInfo channel;
        // [中文导读] [AllReduce逐行 S2185] 开始填充模板ChannelInfo。
        CHK_RET(BuildChannelInfo(
            // [中文导读] [AllReduce逐行 S2186] 以通道请求和实际句柄查询并填入端口/远端CCL属性。
            comm, param, channelRequest[idx], levelNChannels[idx], resCtxHost->topoInfo.userRank, memRegInfo, channel));
        // [中文导读] [AllReduce逐行 S2187] 将填好的通道描述追加到本算法层的资源通道列表。
        resCtxHost->channels[level].push_back(channel);
    // [中文导读] [AllReduce逐行 S2188] 结束循环for (u32 idx = 0; idx < channelNum; idx++)。
    }
    // [中文导读] [AllReduce逐行 S2189] 登记交换信息并申请通道，逐条构建ChannelInfo存入对应层处理完成，返回成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S2190] 结束HcclGetChannelImpl函数体。
}

HcclResult RegGraphModeBuffers(
    HcclComm comm, const OpParam& param, char* inputBuffTag, char* outputBuffTag,
    std::vector<HcclMemHandle>& memHandles)
{
    HCCL_INFO("[RegGraphModeBuffers] param.tag[%s]", param.tag);
    auto retIn = sprintf_s(inputBuffTag, MAX_MEM_TAG_LENGTH, "%s_%s", param.tag, "InputBuffer");
    auto retOut = sprintf_s(outputBuffTag, MAX_MEM_TAG_LENGTH, "%s_%s", param.tag, "OutputBuffer");
    if (retIn <= 0 || retOut <= 0) {
        HCCL_ERROR("[RegGraphModeBuffers]failed to fill BuffTag");
        return HcclResult::HCCL_E_INTERNAL;
    }

    HCCL_INFO("[RegGraphModeBuffers] graph mode registry remote buffer");
    if (param.inputPtr != nullptr && param.inputSize != 0) {
        HcclMemHandle inputHandle = nullptr;
        CHK_RET(HcclRegstryBuff(comm, inputBuffTag, param.inputPtr, param.inputSize, &inputHandle));
        CHK_PTR_NULL(inputHandle);
        memHandles.emplace_back(inputHandle);
    }
    if (param.outputPtr != nullptr && param.outputSize != 0) {
        HcclMemHandle outputHandle = nullptr;
        CHK_RET(HcclRegstryBuff(comm, outputBuffTag, param.outputPtr, param.outputSize, &outputHandle));
        CHK_PTR_NULL(outputHandle);
        memHandles.emplace_back(outputHandle);
    }
    HCCL_INFO("[RegGraphModeBuffers]memHandles size[%d]", memHandles.size());
    return HCCL_SUCCESS;
}

HcclResult GetGraphModeBuffers(
    HcclComm comm, ChannelHandle channelHandle, const char* inputBuffTag, const char* outputBuffTag,
    ChannelInfo& channel)
{
    void* remoteInputBufferAddr = nullptr;
    uint64_t remoteInputBufferSize = 0;
    CHK_RET(HcclGetRemoteBuff(comm, channelHandle, inputBuffTag, &remoteInputBufferAddr, &remoteInputBufferSize));
    if (remoteInputBufferAddr != nullptr && remoteInputBufferSize > 0) {
        channel.remoteInputGraphMode = HcclMem{HCCL_MEM_TYPE_DEVICE, remoteInputBufferAddr, remoteInputBufferSize};
    }

    void* remoteOutputBufferAddr = nullptr;
    uint64_t remoteOutputBufferSize = 0;
    CHK_RET(HcclGetRemoteBuff(comm, channelHandle, outputBuffTag, &remoteOutputBufferAddr, &remoteOutputBufferSize));
    if (remoteOutputBufferAddr != nullptr && remoteOutputBufferSize > 0) {
        channel.remoteOutputGraphMode = HcclMem{HCCL_MEM_TYPE_DEVICE, remoteOutputBufferAddr, remoteOutputBufferSize};
    }
    return HCCL_SUCCESS;
}

// [中文导读] CCU路径把资源描述序列化缓存到对应EngineCtx，后续复用时在Host恢复并重新绑定用户流。
// [中文导读] 这里缓存描述不等于执行CCU指令，Kernel实际发射发生在算法编排阶段。
HcclResult GetAlgResCcu(
    HcclComm comm, const OpParam& param, AlgResourceRequest& resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, TopoInfoWithNetLayerDetails* topoInfo,
    AlgHierarchyInfoForAllLevel& algHierarchyInfo, void** resCtxSequence, uint64_t& ctxSize,
    const ResPackGraphMode& resPack)
{
    // [中文导读] 把选择阶段的拓扑和算法层次保存进资源对象，供 CCU 编排和后续复用恢复。
    resCtxHost->topoInfo = *topoInfo;
    resCtxHost->algHierarchyInfo = algHierarchyInfo;

    // 创建资源，并填充到Host内存上
    // [中文导读] 先取得实际资源；资源不可用向上传递以触发回退，其它错误按原返回码结束。
    HcclResult ret = HcclAllocAlgResourceCcu(comm, param, resRequest, resCtxHost, resPack);
    if (ret == HCCL_E_UNAVAIL) {
        HCCL_WARNING("[HcclAllocAlgResourceCcu] resource unavailable, try to fallback.");
        return HCCL_E_UNAVAIL;
    } else if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("failed to alloc alg resource.");
        return ret;
    }
    // 序列化
    // [中文导读] 资源准备成功后序列化到 CCU EngineCtx，缓存的是 Host 可恢复的资源描述。
    std::vector<char> seq = resCtxHost->Serialize();
    uint64_t size = seq.size();

    void* ctx = nullptr;
    CHK_RET(HcclEngineCtxCreate(comm, param.algTag, param.engine, size, &ctx));
    CHK_SAFETY_FUNC_RET(memcpy_s(ctx, size, seq.data(), size));
    *resCtxSequence = ctx;
    ctxSize = size;
    HCCL_INFO("Execute GetAlgResCCU success.");
    return HCCL_SUCCESS;
}

// [中文导读] CCU除CCL和Thread外还需要通道与Kernel/指令资源；后两步受编译版本条件控制。
// [中文导读] 通道或CCU资源返回UNAVAIL时向上报告，以便选择回退，不能继续假设Kernel已经注册成功。
HcclResult HcclAllocAlgResourceCcu(
    HcclComm comm, const OpParam& param, AlgResourceRequest& resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, const ResPackGraphMode& resPack)
{
    HCCL_INFO("Start to execute AllocAlgResource.");
    void* cclBufferAddr;
    uint64_t cclBufferSize;
    // 从通信域获取CCL buffer
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    // CCL IN使用所有的CCL Buffer，这个其实就是scratch buffer
    resCtxHost->cclMem = HcclMem{HCCL_MEM_TYPE_DEVICE, cclBufferAddr, cclBufferSize};
    resCtxHost->notifyNumOnMainThread = resRequest.notifyNumOnMainThread;
    resCtxHost->slaveThreadNum = resRequest.slaveThreadNum;
    resCtxHost->notifyNumPerThread = resRequest.notifyNumPerThread;
    resCtxHost->dieSplitRatio = resRequest.dieSplitRatio;
    // CCU模式下不构造ChannelInfo，端口信息由executor在CalcRes阶段采集，此处透传给执行阶段
    // [中文导读] CCU 路径不构造普通 ChannelInfo，直接保留 executor 计算的并行端口信息供执行使用。
    resCtxHost->parallelPortInfo = resRequest.parallelPortInfo;
    CHK_RET(HcclGetThread(comm, param, resRequest, resCtxHost, resPack));
#if CANN_VERSION_NUM >= CANN_VERSION(9, 1, 0)
    // 资源回退
    // [中文导读] 先准备 CCU kernel 所需通道，通道资源不足时不继续注册 kernel。
    auto ret = HcclGetChannelForCcu(comm, param, resRequest);
    if (ret == HCCL_E_UNAVAIL) {
        // 进行资源回退
        HCCL_WARNING("[HcclGetChannelForCcu] channel unavailable, try to fallback.");
        return HCCL_E_UNAVAIL;
    } else {
        CHK_RET(ret);
    }

    // opMode 透传：CCU_MS 与 CCU_SCHED 使用不同的默认资源阈值（MS 模式 LOOP/CCU_BUF 阈值更大）
    // [中文导读] 再按展开模式取得 CCU 指令和 kernel 资源；不同模式会影响动态申请阈值。
    ret = HcclGetCcuKernel(comm, param.commOpExpansionMode, resRequest, resCtxHost);
    if (ret == HCCL_E_UNAVAIL) {
        // 资源不足导致回退：打印算子信息方便定位
        HCCL_RUN_INFO(
            "[HcclGetCcuKernel] ccu resource insufficient, fallback to AICPU. "
            "algTag[%s], algName[%s], opType[%u], kernelNum[%zu], kernel[0] name[%s].",
            param.algTag, param.algName, static_cast<uint32_t>(param.opType), resRequest.ccuKernelInfos.size(),
            resRequest.ccuKernelInfos.empty() ? "N/A" : resRequest.ccuKernelInfos[0].kernelFuncName);
        return HCCL_E_UNAVAIL;
    } else {
        CHK_RET(ret);
    }
#endif
    return HCCL_SUCCESS;
}

#if CANN_VERSION_NUM >= CANN_VERSION(9, 1, 0)
static HcclResult CcuAcquireKernelChannels(
    HcclComm comm, const OpParam& param, u32 userRank, CcuKernelInfo& kernelInfo, AlgResourceRequest& resRequest,
    std::vector<ChannelHandle>& kernelChannels)
{
    std::vector<HcclChannelDesc>& kernelChannelRequest = kernelInfo.channels;
    u32 channelNum = kernelChannelRequest.size();
    // 参数一致性校验信息注册到通信域，HcclChannelAcquire内部存在读清动作，每次调用前均需注册
    CHK_RET(AddExchangeInfo(comm, param));
    // 查询已存在的通道，existingChannels[i]非0表示通道已存在可复用，为0表示需新建
    std::vector<ChannelHandle> existingChannels;
    if (HcommIsSupportHcclChannelQuery()) {
        existingChannels.assign(channelNum, 0);
        auto queryRet
            = HcclChannelQuery(comm, param.engine, kernelChannelRequest.data(), channelNum, existingChannels.data());
        if (queryRet != HCCL_SUCCESS) {
            HCCL_WARNING("[HcclChannelQuery] failed, ret[%d], treat all as new channels.", queryRet);
        }
    }

    auto ret = HcclChannelAcquire(comm, param.engine, kernelChannelRequest.data(), channelNum, kernelChannels.data());
    // 需要资源回退。返回资源不够
    if (ret == HCCL_E_UNAVAIL) {
        HCCL_WARNING("[HcclChannelAcquire] channel unavailable, channel num[%u].", channelNum);
        // 释放当前kernel之前已申请的新增通道
        ReleaseCcuAcquiredChannels(comm, resRequest);
        return HCCL_E_UNAVAIL;
    } else {
        CHK_RET(ret);
    }

    // 记录新增通道（existingChannels为0表示该通道是本次HcclChannelAcquire新建的）
    if (!existingChannels.empty()) {
        for (u32 i = 0; i < channelNum; ++i) {
            if (existingChannels[i] == 0) {
                resRequest.acquiredChannels.push_back(kernelChannels[i]);
            }
        }
    }
    // 从首条channel获取dieId，作为kernel所属dieId保存（同一kernel的所有channel在同一die上）
    EndpointDesc localEndpoint = kernelChannelRequest[0].localEndpoint;
    using DieIdType = uint32_t;
    const uint32_t dieIdTypeSize = sizeof(DieIdType);
    DieIdType dieId = 0;
    CHK_RET(HcclRankGraphGetEndpointInfo(
        comm, userRank, &localEndpoint, ENDPOINT_ATTR_DIE_ID, dieIdTypeSize, static_cast<void*>(&dieId)));
    kernelInfo.dieId = dieId;
    return HCCL_SUCCESS;
}

HcclResult HcclGetChannelForCcu(HcclComm comm, const OpParam& param, AlgResourceRequest& resRequest)
{
    // OpParam.userRank 并非所有算子路径都会赋值（仅 Reduce 赋值），这里直接从 comm 查询本端全局 rank
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET(HcclGetRankId(comm, &userRank));
    // 以kernel为粒度申请channel
    for (CcuKernelInfo& kernelInfo : resRequest.ccuKernelInfos) {
        u32 channelNum = kernelInfo.channels.size();
        std::vector<ChannelHandle> kernelChannels;
        kernelChannels.resize(channelNum);
        if (channelNum > 0) {
            CHK_RET(CcuAcquireKernelChannels(comm, param, userRank, kernelInfo, resRequest, kernelChannels));
        }
        auto* kernelArgBase = static_cast<CcuKernelArgBase*>(kernelInfo.kernelArg);
        if (!kernelArgBase) {
            HCCL_ERROR("[HcclGetChannelForCcu] kernelArg ptr is err.");
            return HCCL_E_INTERNAL;
        }
        for (u32 i = 0; i < channelNum; ++i) {
            kernelArgBase->channels[i] = kernelChannels[i];
        }
        kernelArgBase->channelCount = channelNum;
        HCCL_INFO("[HcclGetChannelForCcu] Get [%lu] channels, dieId[%u]", channelNum, kernelInfo.dieId);
    }
    return HCCL_SUCCESS;
}

static constexpr uint32_t CCU_DEFAULT_RES_FRACTION_ADDRESS = 400;
static constexpr uint32_t CCU_DEFAULT_RES_FRACTION_LOOP = 16;
static constexpr uint32_t CCU_DEFAULT_RES_FRACTION_CCU_BUF = 128;
static constexpr uint32_t CCU_DEFAULT_RES_FRACTION_VARIABLE = 400;
static constexpr uint32_t CCU_DEFAULT_RES_FRACTION_EVENT = 48;
static constexpr uint32_t CCU_DEFAULT_RES_FRACTION_CCU_THREAD = 2;

static constexpr uint32_t CCU_DEFAULT_RES_FRACTION_LOOP_MS = 128;
static constexpr uint32_t CCU_DEFAULT_RES_FRACTION_CCU_BUF_MS = 1024;
static constexpr uint32_t CCU_DEFAULT_RES_FRACTION_EVENT_MS = 160;

// V2(960) 资源规格，与 V1 一致的也单独定义，方便后续独立调整
static constexpr uint32_t CCU_DEFAULT_RES_FRACTION_ADDRESS_V2 = 0;
static constexpr uint32_t CCU_DEFAULT_RES_FRACTION_LOOP_V2 = 16;
static constexpr uint32_t CCU_DEFAULT_RES_FRACTION_CCU_BUF_V2 = 128;
static constexpr uint32_t CCU_DEFAULT_RES_FRACTION_VARIABLE_V2 = 800;
static constexpr uint32_t CCU_DEFAULT_RES_FRACTION_EVENT_V2 = 64;
static constexpr uint32_t CCU_DEFAULT_RES_FRACTION_CCU_THREAD_V2 = 2;

static constexpr uint32_t CCU_DEFAULT_RES_FRACTION_LOOP_MS_V2 = 128;
static constexpr uint32_t CCU_DEFAULT_RES_FRACTION_CCU_BUF_MS_V2 = 1024;
static constexpr uint32_t CCU_DEFAULT_RES_FRACTION_EVENT_MS_V2 = 400;

// 即使本算子未在所有 die 上下 kernel，
// 也需为所有 die 创建 reqDesc，保证后续算子在该 die 上有 kernel 时容量充足。
static constexpr uint32_t CCU_DEFAULT_DIE_NUM = 2;

static const std::vector<HcommCcuResType>& GetCcuResTypes()
{
    static const std::vector<HcommCcuResType> types = {
        HCOMM_CCU_RES_TYPE_LOOP,        HCOMM_CCU_RES_TYPE_CCU_BUF, HCOMM_CCU_RES_TYPE_VARIABLE,
        HCOMM_CCU_RES_TYPE_ADDRESS,     HCOMM_CCU_RES_TYPE_EVENT,   HCOMM_CCU_RES_TYPE_CCU_THREAD,
        HCOMM_CCU_RES_TYPE_INSTRUCTION,
    };
    return types;
}

// 实例创建相关的资源类型列表（不含 INSTRUCTION）。
// INSTRUCTION 仅用于查询
static const std::vector<HcommCcuResType>& GetCcuInsCreateResTypes()
{
    static const std::vector<HcommCcuResType> types = {
        HCOMM_CCU_RES_TYPE_LOOP,    HCOMM_CCU_RES_TYPE_CCU_BUF, HCOMM_CCU_RES_TYPE_VARIABLE,
        HCOMM_CCU_RES_TYPE_ADDRESS, HCOMM_CCU_RES_TYPE_EVENT,   HCOMM_CCU_RES_TYPE_CCU_THREAD,
    };
    return types;
}

// V1(950) 默认资源规格
static uint32_t GetDefaultResFractionV1(HcommCcuResType resType, HcclOpExpansionMode opExpansionMode)
{
    bool isCcuMs = (opExpansionMode == HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_CCU_MS);
    switch (resType) {
        case HCOMM_CCU_RES_TYPE_ADDRESS:
            return CCU_DEFAULT_RES_FRACTION_ADDRESS;
        case HCOMM_CCU_RES_TYPE_LOOP:
            return isCcuMs ? CCU_DEFAULT_RES_FRACTION_LOOP_MS : CCU_DEFAULT_RES_FRACTION_LOOP;
        case HCOMM_CCU_RES_TYPE_CCU_BUF:
            return isCcuMs ? CCU_DEFAULT_RES_FRACTION_CCU_BUF_MS : CCU_DEFAULT_RES_FRACTION_CCU_BUF;
        case HCOMM_CCU_RES_TYPE_VARIABLE:
            return CCU_DEFAULT_RES_FRACTION_VARIABLE;
        case HCOMM_CCU_RES_TYPE_EVENT:
            return isCcuMs ? CCU_DEFAULT_RES_FRACTION_EVENT_MS : CCU_DEFAULT_RES_FRACTION_EVENT;
        case HCOMM_CCU_RES_TYPE_CCU_THREAD:
            return CCU_DEFAULT_RES_FRACTION_CCU_THREAD;
        default:
            return 0;
    }
}

// V2(960) 默认资源规格
static uint32_t GetDefaultResFractionV2(HcommCcuResType resType, HcclOpExpansionMode opExpansionMode)
{
    bool isCcuMs = (opExpansionMode == HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_CCU_MS);
    switch (resType) {
        case HCOMM_CCU_RES_TYPE_ADDRESS:
            return CCU_DEFAULT_RES_FRACTION_ADDRESS_V2;
        case HCOMM_CCU_RES_TYPE_LOOP:
            return isCcuMs ? CCU_DEFAULT_RES_FRACTION_LOOP_MS_V2 : CCU_DEFAULT_RES_FRACTION_LOOP_V2;
        case HCOMM_CCU_RES_TYPE_CCU_BUF:
            return isCcuMs ? CCU_DEFAULT_RES_FRACTION_CCU_BUF_MS_V2 : CCU_DEFAULT_RES_FRACTION_CCU_BUF_V2;
        case HCOMM_CCU_RES_TYPE_VARIABLE:
            return CCU_DEFAULT_RES_FRACTION_VARIABLE_V2;
        case HCOMM_CCU_RES_TYPE_EVENT:
            return isCcuMs ? CCU_DEFAULT_RES_FRACTION_EVENT_MS_V2 : CCU_DEFAULT_RES_FRACTION_EVENT_V2;
        case HCOMM_CCU_RES_TYPE_CCU_THREAD:
            return CCU_DEFAULT_RES_FRACTION_CCU_THREAD_V2;
        default:
            return 0;
    }
}

// 根据设备类型分发：960 走 V2，其余走 V1
// opExpansionMode 为 CCU_MS 时 LOOP/CCU_BUF/EVENT 使用 MS 模式专用阈值，其他资源类型与其他模式保持一致
static uint32_t GetDefaultResFraction(HcommCcuResType resType, HcclOpExpansionMode opExpansionMode)
{
    HcclDevType deviceType;
    bool isV2 = (HcclGetDeviceType(deviceType) == HCCL_SUCCESS) && (deviceType == HcclDevType::DEV_TYPE_960);
    return isV2 ? GetDefaultResFractionV2(resType, opExpansionMode) : GetDefaultResFractionV1(resType, opExpansionMode);
}

// 将 HcommCcuResType 转字符串
static const char* GetCcuResTypeName(HcommCcuResType resType)
{
    switch (resType) {
        case HCOMM_CCU_RES_TYPE_LOOP:
            return "LOOP";
        case HCOMM_CCU_RES_TYPE_CCU_BUF:
            return "CCU_BUF";
        case HCOMM_CCU_RES_TYPE_VARIABLE:
            return "VARIABLE";
        case HCOMM_CCU_RES_TYPE_ADDRESS:
            return "ADDRESS";
        case HCOMM_CCU_RES_TYPE_EVENT:
            return "EVENT";
        case HCOMM_CCU_RES_TYPE_CCU_THREAD:
            return "CCU_THREAD";
        case HCOMM_CCU_RES_TYPE_INSTRUCTION:
            return "INSTRUCTION";
        default:
            return "UNKNOWN";
    }
}

static bool IsCcuDynamicResApiSupported()
{
    return HcommIsSupportHcommCcuInsResDescCreate() && HcommIsSupportHcommCcuInsResDescDestroy()
           && HcommIsSupportHcommCcuInsResDescSetNum() && HcommIsSupportHcommCcuInsResDescQueryNum()
           && HcommIsSupportHcommCcuInsCreate() && HcommIsSupportHcommCcuInsDestroy()
           && HcommIsSupportHcommCcuInsQueryResDesc() && HcommIsSupportHcommCcuQueryRemainResDesc()
           && HcommIsSupportHcommCcuKernelQueryResReq() && HcommIsSupportHcclCommAssignCcuIns()
           && HcommIsSupportHcclCommQueryAssignedCcuIns();
}

// 按 dieId 维护资源描述符集合；HcommCcuInsResDescCreate 接口要求每个 desc 必须绑定一个 dieId，
// 因此同一 die 上多个 kernel 的需求聚合到同一个 desc，不同 die 各自维护独立 desc。
using ResDescByDie = std::map<uint32_t, HcommCcuResDescHandle>;

// 销毁集合中所有 desc 并清空，避免资源泄漏
static void DestroyAllDescs(ResDescByDie& descs)
{
    for (auto& kv : descs) {
        if (kv.second != 0) {
            HcommCcuInsResDescDestroy(kv.second);
            kv.second = 0;
        }
    }
    descs.clear();
}

// 查询单个 kernel 的资源需求，按 (dieId, resGroup, resType) 累加到 groupedResMap。
static HcclResult AccumulateKernelRes(
    const CcuKernelInfo& kernelInfo,
    std::map<uint32_t, std::map<u32, std::map<HcommCcuResType, uint32_t>>>& groupedResMap)
{
    HcommCcuResDescHandle kernelDesc = 0;
    CcuResult createRet = HcommCcuInsResDescCreate(kernelInfo.dieId, &kernelDesc);
    CHK_PRT_RET(
        createRet != CCU_SUCCESS,
        HCCL_ERROR(
            "[AccumulateKernelRes] HcommCcuInsResDescCreate dieId[%u] failed: ccuRet -> %d", kernelInfo.dieId,
            createRet),
        ConvertCcuToHccl(createRet));

    const void* kernelArgs[] = {kernelInfo.kernelArg};
    constexpr uint32_t kernelArgNum = 1;
    CcuResult queryRet = HcommCcuKernelQueryResReq(
        reinterpret_cast<void*>(kernelInfo.kernelFunc), kernelArgs, kernelArgNum, kernelDesc);
    if (queryRet != CCU_SUCCESS) {
        HCCL_ERROR("[AccumulateKernelRes] HcommCcuKernelQueryResReq failed: ccuRet -> %d", queryRet);
        HcommCcuInsResDescDestroy(kernelDesc);
        return ConvertCcuToHccl(queryRet);
    }

    for (HcommCcuResType resType : GetCcuResTypes()) {
        uint32_t resNum = 0;
        CcuResult qRet = HcommCcuInsResDescQueryNum(kernelDesc, resType, &resNum);
        if (qRet != CCU_SUCCESS) {
            HCCL_ERROR("[AccumulateKernelRes] HcommCcuInsResDescQueryNum failed: ccuRet -> %d", qRet);
            HcommCcuInsResDescDestroy(kernelDesc);
            return ConvertCcuToHccl(qRet);
        }
        HCCL_INFO(
            "[AccumulateKernelRes] kernel[%s] dieId[%u] resGroup[%u] resType[%s] resNum[%u].",
            kernelInfo.kernelFuncName, kernelInfo.dieId, kernelInfo.resGroup, GetCcuResTypeName(resType), resNum);
        groupedResMap[kernelInfo.dieId][kernelInfo.resGroup][resType] += resNum;
    }
    HcommCcuInsResDescDestroy(kernelDesc);
    return HCCL_SUCCESS;
}

static HcclResult IsResCapSufficient(
    uint32_t dieId, HcommCcuResDescHandle resCap, HcommCcuResDescHandle resReq, bool& sufficient,
    std::string& insuffSummary)
{
    sufficient = true;
    insuffSummary.clear();
    HCCL_INFO("[IsResCapSufficient] start, dieId[%u].", dieId);
    for (HcommCcuResType resType : GetCcuInsCreateResTypes()) {
        uint32_t capNum = 0;
        uint32_t reqNum = 0;
        CcuResult capRet = HcommCcuInsResDescQueryNum(resCap, resType, &capNum);
        if (capRet != CCU_SUCCESS) {
            HCCL_ERROR(
                "[IsResCapSufficient] dieId[%u] query cap failed, resType[%s]: ccuRet -> %d", dieId,
                GetCcuResTypeName(resType), capRet);
            return ConvertCcuToHccl(capRet);
        }
        CcuResult reqRet = HcommCcuInsResDescQueryNum(resReq, resType, &reqNum);
        if (reqRet != CCU_SUCCESS) {
            HCCL_ERROR(
                "[IsResCapSufficient] dieId[%u] query req failed, resType[%s]: ccuRet -> %d", dieId,
                GetCcuResTypeName(resType), reqRet);
            return ConvertCcuToHccl(reqRet);
        }
        HCCL_INFO(
            "[IsResCapSufficient] dieId[%u] resType[%s] cap[%u] req[%u] %s.", dieId, GetCcuResTypeName(resType), capNum,
            reqNum, capNum >= reqNum ? "sufficient" : "insufficient");
        if (capNum < reqNum) {
            sufficient = false;
            if (!insuffSummary.empty()) {
                insuffSummary += ",";
            }
            insuffSummary += std::string(GetCcuResTypeName(resType)) + "(need=" + std::to_string(reqNum)
                             + ",remain=" + std::to_string(capNum) + ")";
        }
    }
    HCCL_INFO(
        "[IsResCapSufficient] dieId[%u] %s.", dieId,
        sufficient ? "all resTypes sufficient" : "some resTypes insufficient");
    return HCCL_SUCCESS;
}

static HcclResult CalcMaxResReqWithDefault(
    uint32_t dieId, HcclOpExpansionMode opExpansionMode, HcommCcuResDescHandle resReq, HcommCcuResDescHandle outMax)
{
    HCCL_INFO(
        "[CalcMaxResReqWithDefault] start, dieId[%u], opExpansionMode[%u].", dieId,
        static_cast<uint32_t>(opExpansionMode));
    for (HcommCcuResType resType : GetCcuInsCreateResTypes()) {
        uint32_t reqNum = 0;
        CcuResult qRet = HcommCcuInsResDescQueryNum(resReq, resType, &reqNum);
        if (qRet != CCU_SUCCESS) {
            HCCL_ERROR("[CalcMaxResReqWithDefault] dieId[%u] query req failed: ccuRet -> %d", dieId, qRet);
            return ConvertCcuToHccl(qRet);
        }
        uint32_t defaultNum = GetDefaultResFraction(resType, opExpansionMode);
        uint32_t maxNum = std::max(reqNum, defaultNum);
        CcuResult setRet = HcommCcuInsResDescSetNum(outMax, resType, maxNum);
        if (setRet != CCU_SUCCESS) {
            HCCL_ERROR("[CalcMaxResReqWithDefault] dieId[%u] set failed: ccuRet -> %d", dieId, setRet);
            return ConvertCcuToHccl(setRet);
        }
        HCCL_INFO(
            "[CalcMaxResReqWithDefault] dieId[%u] resType[%s] req[%u] default[%u] -> max[%u].", dieId,
            GetCcuResTypeName(resType), reqNum, defaultNum, maxNum);
    }
    HCCL_INFO("[CalcMaxResReqWithDefault] dieId[%u] finish.", dieId);
    return HCCL_SUCCESS;
}

static HcclResult RegisterCcuKernels(
    CcuInsHandle insHandle, AlgResourceRequest& resRequest, std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost)
{
    u32 totalKernelNum = std::accumulate(resRequest.ccuKernelNum.begin(), resRequest.ccuKernelNum.end(), 0u);
    CHK_PRT_RET(
        totalKernelNum != resRequest.ccuKernelInfos.size(), HCCL_ERROR("[RegisterCcuKernels]ccuKernel num not match!"),
        HCCL_E_INTERNAL);
    HCCL_INFO("[RegisterCcuKernels] start, totalKernelNum[%u], insHandle[%p].", totalKernelNum, insHandle);

    // 遍历计算最大resGroup号
    auto maxIt = std::max_element(
        resRequest.ccuKernelInfos.begin(), resRequest.ccuKernelInfos.end(),
        [](const CcuKernelInfo& a, const CcuKernelInfo& b) {
            return a.resGroup < b.resGroup;
        });
    u32 maxResGroup = (maxIt != resRequest.ccuKernelInfos.end()) ? maxIt->resGroup : 0;
    resCtxHost->ccuKernels.resize(totalKernelNum);

    for (u32 currentResGroup = 0; currentResGroup <= maxResGroup; currentResGroup++) {
        HCCL_INFO("[RegisterCcuKernels] register resGroup[%u] start, maxResGroup[%u].", currentResGroup, maxResGroup);
        CcuResult regStartRet = HcommCcuKernelRegisterStart(insHandle);
        CHK_PRT_RET(
            regStartRet != CCU_SUCCESS, HCCL_ERROR("ccu kernel register start failed: ccuRet -> %d", regStartRet),
            ConvertCcuToHccl(regStartRet));
        for (u32 i = 0; i < totalKernelNum; i++) {
            CcuKernelInfo& kernelInfo = resRequest.ccuKernelInfos[i];
            if (kernelInfo.resGroup != currentResGroup)
                continue;
            CcuKernelHandle kernelHandle;
            const void* kernelArgs[] = {kernelInfo.kernelArg};
            CcuResult regRet = HcommCcuKernelRegister(
                insHandle, kernelInfo.dieId, kernelInfo.kernelFuncName, reinterpret_cast<void*>(kernelInfo.kernelFunc),
                kernelArgs, 1, &kernelHandle);
            if (regRet == CCU_E_UNAVAIL) {
                HCCL_WARNING("[RegisterCcuKernels] kernel[%s] unavailable, fallback.", kernelInfo.kernelFuncName);
                return HCCL_E_UNAVAIL;
            }
            CHK_PRT_RET(
                regRet != CCU_SUCCESS,
                HCCL_ERROR("ccu kernel register failed: ccuRet -> %d, kernel[%s]", regRet, kernelInfo.kernelFuncName),
                ConvertCcuToHccl(regRet));
            resCtxHost->ccuKernels[i] = kernelHandle;
        }
        CcuResult regEndRet = HcommCcuKernelRegisterEnd(insHandle);
        CHK_PRT_RET(
            regEndRet != CCU_SUCCESS, HCCL_ERROR("ccu kernel register end failed: ccuRet -> %d", regEndRet),
            ConvertCcuToHccl(regEndRet));
        HCCL_INFO("[RegisterCcuKernels] register resGroup[%u] finish.", currentResGroup);
    }
    resCtxHost->ccuKernelNum = resRequest.ccuKernelNum;
    HCCL_INFO("[RegisterCcuKernels] finish, totalKernelNum[%u] registered.", totalKernelNum);
    return HCCL_SUCCESS;
}

// 探测 die 是否使能：调用 HcommCcuQueryRemainResDesc，返回 CCU_E_UNAVAIL 表示未使能。
// support flag 已在 IsCcuDynamicResApiSupported 中校验，本函数不重复判断。
static HcclResult IsDieEnabledForPadding(uint32_t dieId, bool& enabled)
{
    enabled = false;
    // 创建临时 probe desc 绑定 dieId，查询后立即销毁
    HcommCcuResDescHandle probeDesc = 0;
    CcuResult createRet = HcommCcuInsResDescCreate(dieId, &probeDesc);
    CHK_PRT_RET(
        createRet != CCU_SUCCESS,
        HCCL_ERROR(
            "[IsDieEnabledForPadding] HcommCcuInsResDescCreate dieId[%u] failed: ccuRet -> %d", dieId, createRet),
        ConvertCcuToHccl(createRet));
    CcuResult queryRet = HcommCcuQueryRemainResDesc(probeDesc);
    // 立即销毁 probe desc，避免句柄泄漏
    HcommCcuInsResDescDestroy(probeDesc);
    if (queryRet == CCU_E_UNAVAIL) {
        enabled = false;
        HCCL_INFO("[IsDieEnabledForPadding] dieId[%u] is not enabled (CCU_E_UNAVAIL), skip padding.", dieId);
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(
        queryRet != CCU_SUCCESS,
        HCCL_ERROR(
            "[IsDieEnabledForPadding] HcommCcuQueryRemainResDesc dieId[%u] failed: ccuRet -> %d", dieId, queryRet),
        ConvertCcuToHccl(queryRet));
    enabled = true;
    HCCL_INFO("[IsDieEnabledForPadding] dieId[%u] is enabled, need padding.", dieId);
    return HCCL_SUCCESS;
}

// 聚合所有 kernel 的资源需求到 reqDescs（按 dieId 分组）。
// 聚合规则：同 (dieId, resGroup) 内逐 kernel 相加、同 dieId 不同 resGroup 之间取最大。
// 为硬件上所有使能的 die 创建 reqDesc（未使能 die 跳过），防止后续算子容量不足触发回退。
static HcclResult BuildAggregatedResReq(AlgResourceRequest& resRequest, ResDescByDie& reqDescs)
{
    u32 totalKernelNum = resRequest.ccuKernelInfos.size();
    HCCL_INFO("[BuildAggregatedResReq] start, kernelNum[%u].", totalKernelNum);

    // 按 (dieId, resGroup, resType) 聚合所有 kernel 的资源需求
    std::map<uint32_t, std::map<u32, std::map<HcommCcuResType, uint32_t>>> groupedResMap;
    for (u32 i = 0; i < totalKernelNum; i++) {
        CHK_RET(AccumulateKernelRes(resRequest.ccuKernelInfos[i], groupedResMap));
    }

    // 补齐缺失的 die。区分两类 die：
    // 1. 首算子 kernel 所在的 die：已在 groupedResMap 中，必须申请资源，不判断使能状态。
    // 2. kernel 不所在的 die（groupedResMap 中缺失的 die）：通过 IsDieEnabledForPadding 探测是否使能，
    //    使能才创建空条目，后续 CreateFinalReqDescs 会按默认阈值为其申请资源（防止后续算子回退）；
    //    未使能（单 die 场景）跳过补齐，避免冗余申请。
    for (uint32_t dieId = 0; dieId < CCU_DEFAULT_DIE_NUM; dieId++) {
        if (groupedResMap.find(dieId) != groupedResMap.end())
            continue;
        bool enabled = false;
        CHK_RET(IsDieEnabledForPadding(dieId, enabled));
        if (enabled) {
            groupedResMap[dieId] = {};
            HCCL_INFO("[BuildAggregatedResReq] dieId[%u] has no kernel but enabled, create empty entry.", dieId);
        }
    }

    // 为每个 die 创建独立 reqDesc，写入同 dieId 不同 resGroup 取最大后的资源数
    for (auto& dieEntry : groupedResMap) {
        uint32_t dieId = dieEntry.first;
        auto& resGroupMap = dieEntry.second;
        HcommCcuResDescHandle reqDesc = 0;
        CcuResult createRet = HcommCcuInsResDescCreate(dieId, &reqDesc);
        if (createRet != CCU_SUCCESS) {
            HCCL_ERROR(
                "[BuildAggregatedResReq] HcommCcuInsResDescCreate dieId[%u] failed: ccuRet -> %d", dieId, createRet);
            DestroyAllDescs(reqDescs);
            return ConvertCcuToHccl(createRet);
        }
        for (HcommCcuResType resType : GetCcuInsCreateResTypes()) {
            uint32_t maxNum = 0;
            for (auto& groupEntry : resGroupMap) {
                auto it = groupEntry.second.find(resType);
                if (it != groupEntry.second.end() && it->second > maxNum)
                    maxNum = it->second;
            }
            CcuResult setRet = HcommCcuInsResDescSetNum(reqDesc, resType, maxNum);
            if (setRet != CCU_SUCCESS) {
                HCCL_ERROR("[BuildAggregatedResReq] HcommCcuInsResDescSetNum failed: ccuRet -> %d", setRet);
                HcommCcuInsResDescDestroy(reqDesc);
                DestroyAllDescs(reqDescs);
                return ConvertCcuToHccl(setRet);
            }
            HCCL_INFO(
                "[BuildAggregatedResReq] dieId[%u] resType[%s] aggregated maxNum[%u].", dieId,
                GetCcuResTypeName(resType), maxNum);
        }
        reqDescs[dieId] = reqDesc;
    }
    HCCL_INFO("[BuildAggregatedResReq] finish, dieNum[%zu].", reqDescs.size());
    return HCCL_SUCCESS;
}

// 复用已有 CcuIns：对每个 die 创建 capDesc 查询容量并与 reqDesc 比较，全部充足才注册 kernels；
// 任一 die 不足返回 HCCL_E_UNAVAIL 触发回退。函数内部销毁 reqDescs。
static HcclResult ReuseExistingCcuIns(
    CcuInsHandle insHandle, ResDescByDie& reqDescs, AlgResourceRequest& resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost)
{
    HCCL_INFO("[ReuseExistingCcuIns] reuse existing CcuIns, insHandle[%p], dieNum[%zu].", insHandle, reqDescs.size());
    bool allSufficient = true;
    std::string allDieInsuffSummary; // 收集所有 die 的不足资源摘要，用于回退时记 RUN_INFO
    for (auto& dieEntry : reqDescs) {
        uint32_t dieId = dieEntry.first;
        HcommCcuResDescHandle reqDesc = dieEntry.second;
        HcommCcuResDescHandle capDesc = 0;
        CcuResult capCreateRet = HcommCcuInsResDescCreate(dieId, &capDesc);
        if (capCreateRet != CCU_SUCCESS) {
            HCCL_ERROR(
                "[ReuseExistingCcuIns] HcommCcuInsResDescCreate capDesc dieId[%u] failed: ccuRet -> %d", dieId,
                capCreateRet);
            DestroyAllDescs(reqDescs);
            return ConvertCcuToHccl(capCreateRet);
        }
        CcuResult qRet = HcommCcuInsQueryResDesc(insHandle, capDesc);
        if (qRet != CCU_SUCCESS) {
            HCCL_ERROR("[ReuseExistingCcuIns] HcommCcuInsQueryResDesc dieId[%u] failed: ccuRet -> %d", dieId, qRet);
            HcommCcuInsResDescDestroy(capDesc);
            DestroyAllDescs(reqDescs);
            return ConvertCcuToHccl(qRet);
        }
        bool sufficient = false;
        std::string insuffSummary;
        HcclResult capRet = IsResCapSufficient(dieId, capDesc, reqDesc, sufficient, insuffSummary);
        if (capRet != HCCL_SUCCESS) {
            // 查询接口本身失败，当作错误上报，不继续检查其他 die，不触发回退
            HCCL_ERROR("[ReuseExistingCcuIns] IsResCapSufficient dieId[%u] failed: ret -> %d", dieId, capRet);
            HcommCcuInsResDescDestroy(capDesc);
            DestroyAllDescs(reqDescs);
            return capRet;
        }
        HCCL_INFO("[ReuseExistingCcuIns] dieId[%u] sufficient[%d].", dieId, sufficient);
        if (!sufficient) {
            allSufficient = false;
            if (!allDieInsuffSummary.empty()) {
                allDieInsuffSummary += "; ";
            }
            allDieInsuffSummary += "dieId[" + std::to_string(dieId) + "]: " + insuffSummary;
        }
        HcommCcuInsResDescDestroy(capDesc);
    }
    DestroyAllDescs(reqDescs);

    if (!allSufficient) {
        HCCL_WARNING("[ReuseExistingCcuIns] existing CcuIns resource insufficient, try to fallback.");
        HCCL_RUN_INFO("[ReuseExistingCcuIns] insufficient res detail: %s", allDieInsuffSummary.c_str());
        return HCCL_E_UNAVAIL;
    }
    return RegisterCcuKernels(insHandle, resRequest, resCtxHost);
}

// 为每个 die 创建 finalReqDesc = max(reqDesc, 默认阈值)，避免按实际需求申请造成资源碎片。出参由调用方销毁。
static HcclResult CreateFinalReqDescs(
    ResDescByDie& reqDescs, HcclOpExpansionMode opExpansionMode, std::vector<HcommCcuResDescHandle>& finalReqDescs)
{
    for (auto& dieEntry : reqDescs) {
        uint32_t dieId = dieEntry.first;
        HcommCcuResDescHandle reqDesc = dieEntry.second;
        HcommCcuResDescHandle finalReqDesc = 0;
        CcuResult fCreateRet = HcommCcuInsResDescCreate(dieId, &finalReqDesc);
        if (fCreateRet != CCU_SUCCESS) {
            HCCL_ERROR(
                "[CreateFinalReqDescs] HcommCcuInsResDescCreate dieId[%u] failed: ccuRet -> %d", dieId, fCreateRet);
            for (auto d : finalReqDescs) {
                HcommCcuInsResDescDestroy(d);
            }
            finalReqDescs.clear();
            return ConvertCcuToHccl(fCreateRet);
        }
        HcclResult maxRet = CalcMaxResReqWithDefault(dieId, opExpansionMode, reqDesc, finalReqDesc);
        if (maxRet != HCCL_SUCCESS) {
            HcommCcuInsResDescDestroy(finalReqDesc);
            for (auto d : finalReqDescs) {
                HcommCcuInsResDescDestroy(d);
            }
            finalReqDescs.clear();
            return maxRet;
        }
        finalReqDescs.push_back(finalReqDesc);
    }
    return HCCL_SUCCESS;
}

// 遍历每个 die 查询硬件剩余资源（HcommCcuQueryRemainResDesc），与 finalReqDescs 对比收集不足资源摘要，
// 拼成 "dieId[x]: LOOP(need=128,remain=16); dieId[y]: ..."。任一 die 查询/对比失败仅 WARNING 并 continue。
static void CollectInsufficientResFromRemain(
    const std::vector<HcommCcuResDescHandle>& finalReqDescs, const std::vector<uint32_t>& finalReqDieIds,
    std::string& allDieInsuffSummary)
{
    for (size_t i = 0; i < finalReqDescs.size() && i < finalReqDieIds.size(); i++) {
        uint32_t dieId = finalReqDieIds[i];
        HcommCcuResDescHandle remainDesc = 0;
        CcuResult rCreateRet = HcommCcuInsResDescCreate(dieId, &remainDesc);
        if (rCreateRet != CCU_SUCCESS) {
            HCCL_WARNING(
                "[CollectInsufficientResFromRemain] create remainDesc dieId[%u] failed: ccuRet -> %d, skip.", dieId,
                rCreateRet);
            continue;
        }
        CcuResult rQueryRet = HcommCcuQueryRemainResDesc(remainDesc);
        if (rQueryRet != CCU_SUCCESS) {
            HCCL_WARNING(
                "[CollectInsufficientResFromRemain] query remainDesc dieId[%u] failed: ccuRet -> %d, skip.", dieId,
                rQueryRet);
            HcommCcuInsResDescDestroy(remainDesc);
            continue;
        }
        bool sufficient = false;
        std::string insuffSummary;
        HcclResult cmpRet = IsResCapSufficient(dieId, remainDesc, finalReqDescs[i], sufficient, insuffSummary);
        HcommCcuInsResDescDestroy(remainDesc);
        if (cmpRet != HCCL_SUCCESS) {
            HCCL_WARNING(
                "[CollectInsufficientResFromRemain] compare remainDesc dieId[%u] failed: ret -> %d, skip.", dieId,
                cmpRet);
            continue;
        }
        if (!sufficient) {
            if (!allDieInsuffSummary.empty()) {
                allDieInsuffSummary += "; ";
            }
            allDieInsuffSummary += "dieId[" + std::to_string(dieId) + "]: " + insuffSummary;
        }
    }
}

// 新建 CcuIns：取需求与默认阈值的最大值创建实例并绑定到 comm，使后续算子走复用路径。
// 函数内部销毁 reqDescs。
static HcclResult CreateAndAssignNewCcuIns(
    HcclComm comm, HcclOpExpansionMode opExpansionMode, ResDescByDie& reqDescs, AlgResourceRequest& resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost)
{
    HCCL_INFO(
        "[CreateAndAssignNewCcuIns] no existing CcuIns, create new one, opExpansionMode[%u], dieNum[%zu].",
        static_cast<uint32_t>(opExpansionMode), reqDescs.size());
    // 收集 dieId 顺序（CreateFinalReqDescs 按 reqDescs 即 map 升序遍历，与此处顺序一致），
    // 用于资源不足时查询每个 die 的剩余资源做对比
    std::vector<uint32_t> finalReqDieIds;
    for (const auto& entry : reqDescs) {
        finalReqDieIds.push_back(entry.first);
    }

    std::vector<HcommCcuResDescHandle> finalReqDescs;
    HcclResult createRet = CreateFinalReqDescs(reqDescs, opExpansionMode, finalReqDescs);
    DestroyAllDescs(reqDescs);
    if (createRet != HCCL_SUCCESS) {
        return createRet;
    }

    CcuInsHandle newInsHandle = 0;
    CcuResult createInsRet = HcommCcuInsCreate(finalReqDescs.data(), finalReqDescs.size(), &newInsHandle);
    HCCL_INFO("[CreateAndAssignNewCcuIns] HcommCcuInsCreate ret[%d], newInsHandle[%p].", createInsRet, newInsHandle);
    if (createInsRet == CCU_E_UNAVAIL) {
        HCCL_WARNING("[CreateAndAssignNewCcuIns] HcommCcuInsCreate unavailable, try to fallback.");
        std::string allDieInsuffSummary;
        CollectInsufficientResFromRemain(finalReqDescs, finalReqDieIds, allDieInsuffSummary);
        HCCL_RUN_INFO("[CreateAndAssignNewCcuIns] insufficient res detail: %s", allDieInsuffSummary.c_str());
        for (auto d : finalReqDescs) {
            HcommCcuInsResDescDestroy(d);
        }
        return HCCL_E_UNAVAIL;
    } else if (createInsRet != CCU_SUCCESS) {
        HCCL_ERROR("[CreateAndAssignNewCcuIns] HcommCcuInsCreate failed: ccuRet -> %d", createInsRet);
        for (auto d : finalReqDescs) {
            HcommCcuInsResDescDestroy(d);
        }
        return ConvertCcuToHccl(createInsRet);
    }

    HcclResult assignRet = HcclCommAssignCcuIns(comm, newInsHandle);
    HCCL_INFO("[CreateAndAssignNewCcuIns] HcclCommAssignCcuIns ret[%d].", assignRet);
    if (assignRet != HCCL_SUCCESS) {
        HCCL_ERROR("[CreateAndAssignNewCcuIns] HcclCommAssignCcuIns failed: ret -> %d", assignRet);
        HcommCcuInsDestroy(newInsHandle);
        for (auto d : finalReqDescs) {
            HcommCcuInsResDescDestroy(d);
        }
        return assignRet;
    }
    for (auto d : finalReqDescs) {
        HcommCcuInsResDescDestroy(d);
    }
    return RegisterCcuKernels(newInsHandle, resRequest, resCtxHost);
}

// CCU kernel 动态资源申请主流程：1.聚合资源需求 -> 2.查询可复用 CcuIns
// -> 3a.容量充足则复用并注册 / 3b.新建实例并绑定 comm 后注册。接口返回 CCU_E_UNAVAIL 时触发回退。
static HcclResult HcclGetCcuKernelDynamic(
    HcclComm comm, HcclOpExpansionMode opExpansionMode, AlgResourceRequest& resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost)
{
    HCCL_INFO(
        "[HcclGetCcuKernelDynamic] start, opExpansionMode[%u], kernelNum[%zu].", static_cast<uint32_t>(opExpansionMode),
        resRequest.ccuKernelInfos.size());

    // 步骤1：聚合资源需求，按 dieId 分组
    ResDescByDie reqDescs;
    HcclResult buildRet = BuildAggregatedResReq(resRequest, reqDescs);
    if (buildRet != HCCL_SUCCESS) {
        // 防御性清理：BuildAggregatedResReq 失败时契约上已清理，这里再清理一次防止内部契约被破坏
        DestroyAllDescs(reqDescs);
        return buildRet;
    }

    // 步骤2：查询当前 comm 是否已绑定 CcuIns 实例
    // 接口语义：未绑定 CcuIns 时返回 HCCL_E_UNAVAIL（不是 insNum=0），需走新建路径
    CcuInsHandle insHandle = 0;
    uint32_t insNum = 0;
    bool hasReusableIns = false;
    HcclResult queryRet = HcclCommQueryAssignedCcuIns(comm, &insHandle, &insNum);
    if (queryRet == HCCL_SUCCESS) {
        hasReusableIns = (insNum != 0);
        HCCL_INFO(
            "[HcclGetCcuKernelDynamic] HcclCommQueryAssignedCcuIns success, insHandle[%p] insNum[%u].", insHandle,
            insNum);
    } else if (queryRet == HCCL_E_UNAVAIL) {
        HCCL_INFO(
            "[HcclGetCcuKernelDynamic] HcclCommQueryAssignedCcuIns returns UNAVAIL, no reusable CcuIns, will create "
            "new.");
    } else {
        HCCL_ERROR("[HcclGetCcuKernelDynamic] HcclCommQueryAssignedCcuIns failed: ret -> %d", queryRet);
        DestroyAllDescs(reqDescs);
        return queryRet;
    }

    // 步骤3：有可复用实例走复用路径，否则新建；reqDescs 所有权转移给子函数
    // opExpansionMode 透传下去：新建路径需要根据模式取不同的默认阈值（MS 模式 LOOP/CCU_BUF 阈值更大）
    HcclResult finalRet = hasReusableIns ?
                              ReuseExistingCcuIns(insHandle, reqDescs, resRequest, resCtxHost) :
                              CreateAndAssignNewCcuIns(comm, opExpansionMode, reqDescs, resRequest, resCtxHost);
    // 资源不足导致回退时记一条 run info，便于运维统计动态资源申请的回退频率
    if (finalRet == HCCL_E_UNAVAIL) {
        HCCL_RUN_INFO(
            "[HcclGetCcuKernelDynamic] ccu dynamic resource unavailable, fallback to legacy flow, "
            "hasReusableIns[%d], kernelNum[%zu].",
            hasReusableIns, resRequest.ccuKernelInfos.size());
    }
    HCCL_INFO("[HcclGetCcuKernelDynamic] finish, finalRet[%d].", finalRet);
    return finalRet;
}

HcclResult HcclGetCcuKernel(
    HcclComm comm, HcclOpExpansionMode opExpansionMode, AlgResourceRequest& resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost)
{
    if (IsCcuDynamicResApiSupported()) {
        HCCL_INFO(
            "[HcclGetCcuKernel] use dynamic resource apply flow, opExpansionMode[%u].",
            static_cast<uint32_t>(opExpansionMode));
        return HcclGetCcuKernelDynamic(comm, opExpansionMode, resRequest, resCtxHost);
    }

    // 兼容旧 hcomm 包
    HCCL_INFO("[HcclGetCcuKernel] use legacy pre-allocated resource flow.");
    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(
        insNum != 1, HCCL_ERROR("[HcclGetCcuKernel] HcclCommQueryCcuIns fail! insNum is [%u]", insNum),
        HCCL_E_INTERNAL);
    return RegisterCcuKernels(insHandle, resRequest, resCtxHost);
}
#endif /* CANN_VERSION_NUM >= CANN_VERSION(9, 1, 0) */

// [中文导读] AIV的算法资源描述本体保存在CPU_TS上下文，里面再引用Device上的通信信息区。
// [中文导读] 这是Host/Device两类存储角色，不表示算子改成在CPU_TS上执行向量计算。
HcclResult GetAlgResAiv(
    HcclComm comm, const OpParam& param, AlgResourceRequest& resRequest, TopoInfoWithNetLayerDetails* topoInfo,
    AlgHierarchyInfoForAllLevel& algHierarchyInfo, void** resCtxSequence)
{
    // [中文导读] AIV 的算法资源对象在 CPU_TS 上分配，里面的设备通信信息指针由后面的专用资源准备填充。
    uint64_t size = sizeof(AlgResourceCtxSerializable);
    CHK_RET(HcclEngineCtxCreate(comm, param.algTag, CommEngine::COMM_ENGINE_CPU_TS, size, resCtxSequence));

    AlgResourceCtxSerializable* resCtxHost = static_cast<AlgResourceCtxSerializable*>(*resCtxSequence);
    resCtxHost->topoInfo = *topoInfo;
    resCtxHost->algHierarchyInfo = algHierarchyInfo;

    CHK_RET(HcclAllocAlgResourceAiv(comm, param, resRequest, resCtxHost));
    return HCCL_SUCCESS;
}

// [中文导读] 首次准备AIV标记/通信信息区时创建、清零并注册到域；复用时恢复注册句柄和已有地址表。
// [中文导读] 建链后读取Peer的CCL及标记区地址，按Rank填表，再把两张地址表复制到Device供AIV使用。
// [中文导读] 此阶段完成的是资源与地址准备，AIV核内的数据访问和同步不等同于AICPU Thread原语。
HcclResult HcclAllocAlgResourceAiv(
    HcclComm comm, const OpParam& param, AlgResourceRequest& resRequest, AlgResourceCtxSerializable* resCtxHost)
{
    HCCL_INFO("[%s]Start to execute.", __func__);
    HcclMemHandle memHandle; // 注册到通信域内存的handle，用于建链
    // 获取存放AIV对端信息和标记区的空间
    void* buffersIn[MAX_RANK_SIZE] = {};
    void* buffersOut[MAX_RANK_SIZE] = {};
    uint64_t commInfoSize = 0;
    HcclResult ret
        = HcclEngineCtxGet(comm, param.commModeTag, param.engine, &(resCtxHost->aivCommInfoPtr), &commInfoSize);
    // [中文导读] 按通信域和执行模式查询设备通信信息区，首次未命中才申请并清零标记区。
    if (ret == HCCL_E_NOT_FOUND || ret == HCCL_E_PARA) {
        CHK_RET(HcclEngineCtxCreate(
            comm, param.commModeTag, param.engine, AIV_TAG_BUFF_LEN, &(resCtxHost->aivCommInfoPtr)));
        // 清零
        ACLCHECK(haclrtMemset(resCtxHost->aivCommInfoPtr, AIV_TAG_BUFF_LEN, 0, AIV_TAG_BUFF_LEN));
        if (HcommIsSupportHcclCommRegCommStateCallback()) {
            CHK_RET(HcclCommRegCommStateCallback(param.commModeTag, ClearAivTagCb, resCtxHost->aivCommInfoPtr));
        }
        // 注册到通信域，支持建链时交换
        // [中文导读] 把 AIV 标记及地址信息区注册到通信域，建链时才能向 Peer 交换该区的远端描述。
        CommMem regMem{COMM_MEM_TYPE_DEVICE, resCtxHost->aivCommInfoPtr, AIV_TAG_BUFF_LEN};
        CHK_RET(HcclCommMemReg(comm, param.commModeTag, &regMem, &memHandle));
        void* memHandleCachePtr = nullptr; // 当前AIV存放注册内存的memHandle使用
        CHK_RET(HcclEngineCtxCreate(
            comm, param.commModeTag, CommEngine::COMM_ENGINE_CPU_TS, sizeof(HcclMemHandle), &memHandleCachePtr));
        static_cast<HcclMemHandle*>(memHandleCachePtr)[0] = memHandle;
    } else {
        // [中文导读] 复用设备区时从 CPU_TS 上恢复其注册句柄，并校验缓存容量与句柄类型匹配。
        void* memHandleCachePtr = nullptr;
        uint64_t memHandleCacheSize = 0;
        HcclResult ret = HcclEngineCtxGet(
            comm, param.commModeTag, CommEngine::COMM_ENGINE_CPU_TS, &memHandleCachePtr, &memHandleCacheSize);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS || memHandleCacheSize != sizeof(HcclMemHandle),
            HCCL_ERROR(
                "[%s]commModeTag[%s] aiv memHandle not found in cache, ptr[%p] size[%llu]", __func__, param.commModeTag,
                memHandleCachePtr, memHandleCacheSize),
            HCCL_E_INTERNAL);
        memHandle = static_cast<HcclMemHandle*>(memHandleCachePtr)[0];

        // 有的算子，如send recv，可能多个算子对端不同导致buffer被覆盖，需先获取一遍之前的信息
        // [中文导读] 先读回设备上的已有 Peer 地址表，避免后续只更新本次 Peer 时覆盖其它已保存条目。
        CHK_RET(haclrtMemcpy(
            buffersIn, MAX_RANK_SIZE * sizeof(void*), resCtxHost->aivCommInfoPtr, MAX_RANK_SIZE * sizeof(void*),
            ACL_MEMCPY_DEVICE_TO_HOST));
        CHK_RET(haclrtMemcpy(
            buffersOut, MAX_RANK_SIZE * sizeof(void*),
            static_cast<u8*>(resCtxHost->aivCommInfoPtr) + AIV_TAG_ADDR_OFFSET, MAX_RANK_SIZE * sizeof(void*),
            ACL_MEMCPY_DEVICE_TO_HOST));
    }
    HCCL_INFO(
        "[%s]commModeTag[%s] regMemAddr[%p] memHandle[%p]", __func__, param.commModeTag, resCtxHost->aivCommInfoPtr,
        memHandle);

    void* cclBufferAddr;
    uint64_t cclBufferSize;
    // 从通信域获取CCL buffer
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    HCCL_INFO("[%s]local cclBufferAddr[%p] cclBufferSize[%llu]", __func__, cclBufferAddr, cclBufferSize);
    resCtxHost->cclMem = HcclMem{HCCL_MEM_TYPE_DEVICE, cclBufferAddr, cclBufferSize};

    // [中文导读] 将本 Rank 的 CCL 数据区和 AIV 信息区地址放入本地槽，随后补齐各远端 Rank 的地址。
    buffersIn[resCtxHost->topoInfo.userRank] = cclBufferAddr;
    buffersOut[resCtxHost->topoInfo.userRank] = resCtxHost->aivCommInfoPtr;

    // 迭代每个子通信域的建链请求，创建链路
    // [中文导读] 每层建链请求都携带 AIV 注册句柄，使下层可交换本端标记区的访问描述。
    for (u32 level = 0; level < resRequest.channels.size(); level++) {
        // 获取子通信域的建链请求
        std::vector<HcclChannelDesc>& levelNChannelRequest = resRequest.channels[level];
        for (auto& channelDesc : levelNChannelRequest) {
            channelDesc.memHandles = &memHandle;
            channelDesc.memHandleNum = 1;
        }
        // 获取子通信域的建链数量
        u32 validChannelNum = levelNChannelRequest.size();
        std::vector<ChannelHandle> levelNChannels;
        levelNChannels.resize(validChannelNum);
        HCCL_INFO("[%s]level[%u] validChannelNum[%u]", __func__, level, validChannelNum);

        if (validChannelNum > 0) {
            // 参数一致性校验信息注册到通信域，HcclChannelAcquire内部存在读清动作，每次调用前均需注册
            CHK_RET(AddExchangeInfo(comm, param));
            CHK_RET(HcclChannelAcquire(
                comm, param.engine, levelNChannelRequest.data(), validChannelNum, levelNChannels.data()));
        }

        for (u32 idx = 0; idx < validChannelNum; idx++) {
            HcclChannelDesc& channelDesc = levelNChannelRequest[idx];
            CHK_PRT_RET(
                channelDesc.remoteRank >= MAX_RANK_SIZE,
                HCCL_ERROR(
                    "[%s] remoteRank[%u] exceeds MAX_RANK_SIZE[%u]", __func__, channelDesc.remoteRank, MAX_RANK_SIZE),
                HCCL_E_PARA);
            // [中文导读] 取得每条通道对应的远端 CCL 区，把远端 Rank 的数据区地址填入输入地址表。
            void* remoteBufferAddr;
            uint64_t remoteBufferSize;
            CHK_RET(HcclChannelGetHcclBuffer(comm, levelNChannels[idx], &remoteBufferAddr, &remoteBufferSize));
            HCCL_INFO(
                "[%s]remoteRank[%u] cclBufferAddr[%p] cclBufferSize[%llu]", __func__, channelDesc.remoteRank,
                remoteBufferAddr, remoteBufferSize);
            buffersIn[channelDesc.remoteRank] = remoteBufferAddr;

            // [中文导读] 从通道查询远端注册内存，并按此实现约定取末项地址作为该 Rank 的 AIV 标记区。
            u32 memNum;
            CommMem* remoteMems;
            char** memTags;
            CHK_RET(HcclChannelGetRemoteMems(comm, levelNChannels[idx], &memNum, &remoteMems, &memTags));
            CHK_PRT_RET(memNum == 0, HCCL_ERROR("[%s] HcclChannelGetRemoteMems memNum is 0", __func__), HCCL_E_PARA);
            HCCL_RUN_INFO(
                "[%s]remoteRank[%u] memNum[%u] regMemAddr[%p] regMemSize[%llu] memTag[%s]", __func__,
                channelDesc.remoteRank, memNum, remoteMems[memNum - 1].addr, remoteMems[memNum - 1].size,
                memTags[memNum - 1]);
            buffersOut[channelDesc.remoteRank] = remoteMems[memNum - 1].addr;
        }
    }

    // [中文导读] 建链和地址收集完成后，把 CCL 地址表及标记区地址表写回设备；此处仍只是 AIV 核的执行准备。
    CHK_RET(haclrtMemcpy(
        resCtxHost->aivCommInfoPtr, MAX_RANK_SIZE * sizeof(void*), buffersIn, MAX_RANK_SIZE * sizeof(void*),
        ACL_MEMCPY_HOST_TO_DEVICE));
    CHK_RET(haclrtMemcpy(
        static_cast<u8*>(resCtxHost->aivCommInfoPtr) + AIV_TAG_ADDR_OFFSET, MAX_RANK_SIZE * sizeof(void*), buffersOut,
        MAX_RANK_SIZE * sizeof(void*), ACL_MEMCPY_HOST_TO_DEVICE));

    HCCL_INFO("[%s] Alloc res success.", __func__);
    return HCCL_SUCCESS;
}

HcclResult GetAlgResDPU(
    HcclComm comm, const OpParam& param, AlgResourceRequest& resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, TopoInfoWithNetLayerDetails* topoInfo,
    AlgHierarchyInfoForAllLevel& algHierarchyInfo, void** resCtxSequence, uint64_t& ctxSize,
    bool increCreateChannelFlag, const ResPackGraphMode& resPack)
{
    // 申请共享内存
    uint64_t shmemSize = 100 * 1024 * 1024;
    void* shmemPtr = nullptr;
    bool newCreated;
    CHK_RET(HcclDevMemAcquire(comm, "DPUTAG", &shmemSize, &shmemPtr, &newCreated));
    resCtxHost->npu2DpuShmemPtr = shmemPtr;
    constexpr uint64_t DPU2NPU_SHMEM_RATIO = 2;
    resCtxHost->dpu2NpuShmemPtr = static_cast<void*>(static_cast<uint8_t*>(shmemPtr) + shmemSize / DPU2NPU_SHMEM_RATIO);

    CHK_RET(GetAlgResAICPU(
        comm, param, resRequest, resCtxHost, topoInfo, algHierarchyInfo, resCtxSequence, ctxSize,
        increCreateChannelFlag, resPack));

    HCCL_INFO("Execute GetAlgResAICPU success.");
    return HCCL_SUCCESS;
}

// [中文导读] [AllReduce逐行 S3263] 声明CheckCount接口：限制输入元素count不超过系统支持的SYS_MAX_COUNT。
HcclResult CheckCount(const u64 count)
// [中文导读] [AllReduce逐行 S3264] 开始CheckCount的函数体。
{
    // [中文导读] [AllReduce逐行 S3265] 超过SYS_MAX_COUNT的元素数量返回参数错误。
    if (UNLIKELY(count > SYS_MAX_COUNT)) {
        // [中文导读] [AllReduce逐行 S3266] 输出错误日志，记录CheckCount当前阶段和相关参数。
        HCCL_ERROR(
            // [中文导读] [AllReduce逐行 S3267] 补充日志格式：[Check][Count]errNo[0x%016llx] count[%llu] is invalid(bigger than MAX count[%llu])。
            "[Check][Count]errNo[0x%016llx] count[%llu] is invalid(bigger than MAX count[%llu])",
            // [中文导读] [AllReduce逐行 S3268] 输出错误日志，记录CheckCount当前阶段和相关参数。
            HCCL_ERROR_CODE(HCCL_E_PARA), count, SYS_MAX_COUNT);
        // [中文导读] [AllReduce逐行 S3269] 元素count超出系统范围时返回HCCL_E_PARA。
        return HCCL_E_PARA;
    // [中文导读] [AllReduce逐行 S3270] 结束条件if (UNLIKELY(count > SYS_MAX_COUNT))。
    }
    // [中文导读] [AllReduce逐行 S3271] 限制输入元素count不超过系统支持的SYS_MAX_COUNT处理完成，返回成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S3272] 结束CheckCount函数体。
}

// [中文导读] [AllReduce逐行 S3274] 声明CheckDataType接口：检查枚举值合法性并按归约/非归约场景排除不支持dtype。
HcclResult CheckDataType(const HcclDataType dataType, bool needReduce)
// [中文导读] [AllReduce逐行 S3275] 开始CheckDataType的函数体。
{
    // [中文导读] [AllReduce逐行 S3276] 创建错误上报字段名字列表，用于后面的dtype诊断。
    const std::vector<std::string> infoTitle({"ccl_op", "value", "parameter", "expect"});
    // 查询是否为合法的HcclDataType枚举值
    // [中文导读] [AllReduce逐行 S3278] 通过VALID_HCCL_DATA_TYPES集合判断dataType是否是有效枚举。
    bool notValid = VALID_HCCL_DATA_TYPES.find(dataType) == VALID_HCCL_DATA_TYPES.end();
    // [中文导读] [AllReduce逐行 S3279] needReduce为true走归约dtype限制，AllReduce满足该条件。
    if (needReduce) {
        // reduce场景不支持的数据类型
        // [中文导读] [AllReduce逐行 S3281] 定义归约不支持的数据类型集合。
        static const std::set<HcclDataType> REDUCE_UNSUPPORTED
            // [中文导读] [AllReduce逐行 S3282] 归约排除UINT8/16/32及INT128。
            = {HCCL_DATA_TYPE_UINT8, HCCL_DATA_TYPE_UINT16,  HCCL_DATA_TYPE_UINT32,  HCCL_DATA_TYPE_INT128,
               // [中文导读] [AllReduce逐行 S3283] 归约同时排除HIF8和各FP8格式。
               HCCL_DATA_TYPE_HIF8,  HCCL_DATA_TYPE_FP8E4M3, HCCL_DATA_TYPE_FP8E5M2, HCCL_DATA_TYPE_FP8E8M0};
        // [中文导读] [AllReduce逐行 S3284] 枚举不合法或处于归约不支持列表时诊断并返回错误。
        if (notValid || REDUCE_UNSUPPORTED.find(dataType) != REDUCE_UNSUPPORTED.end()) {
            // [中文导读] [AllReduce逐行 S3285] 开始归约数据类型错误上报。
            RPT_INPUT_ERR(
                // [中文导读] [AllReduce逐行 S3286] 指定EI0003错误码和上报字段列表。
                true, "EI0003", infoTitle,
                // [中文导读] [AllReduce逐行 S3287] 构造归约dtype错误上报的字符串参数数组。
                std::vector<std::string>(
                    // [中文导读] [AllReduce逐行 S3288] 上报当前dtype名字及按needReduce生成的支持类型字符串。
                    {"CheckDataType", GetDataTypeEnumStr(dataType), "dataType", GetSupportDataType(needReduce)}));
            // [中文导读] [AllReduce逐行 S3289] 输出错误日志，记录CheckDataType当前阶段和相关参数。
            HCCL_ERROR(
                // [中文导读] [AllReduce逐行 S3290] 补充日志格式：[Check][DataType]errNo[0x%016llx] data type[%s] not supported, support range=[%s]。
                "[Check][DataType]errNo[0x%016llx] data type[%s] not supported, support range=[%s]",
                // [中文导读] [AllReduce逐行 S3291] 输出错误日志，记录CheckDataType当前阶段和相关参数。
                HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), GetDataTypeEnumStr(dataType).c_str(),
                // [中文导读] [AllReduce逐行 S3292] 提供上述日志的实参：GetSupportDataType(needReduce).c_str(。
                GetSupportDataType(needReduce).c_str());
            // [中文导读] [AllReduce逐行 S3293] 返回归约数据类型不支持错误。
            return HCCL_E_NOT_SUPPORT;
        // [中文导读] [AllReduce逐行 S3294] 结束条件if (notValid || REDUCE_UNSUPPORTED.find(dataType) != REDUCE_UNSUPPORTED.end())。
        }
    // [中文导读] [AllReduce逐行 S3295] 上述条件不成立时进入替代分支。
    } else {
        // 非reduce场景不支持INT128
        // [中文导读] [AllReduce逐行 S3297] 非归约场景排除非法枚举及INT128。
        if (notValid || dataType == HCCL_DATA_TYPE_INT128) {
            // [中文导读] [AllReduce逐行 S3298] 开始非归约数据类型错误上报。
            RPT_INPUT_ERR(
                // [中文导读] [AllReduce逐行 S3299] 指定EI0003错误码和上报字段列表。
                true, "EI0003", infoTitle,
                // [中文导读] [AllReduce逐行 S3300] 创建非归约dtype诊断参数数组。
                std::vector<std::string>(
                    // [中文导读] [AllReduce逐行 S3301] 上报当前数据类型名字及dataType参数名。
                    {"CheckDataType", GetDataTypeEnumStr(dataType), "dataType",
                     // [中文导读] [AllReduce逐行 S3302] 补充非归约支持的数据类型列表字符串并结束上报。
                     GetSupportDataType(needReduce).c_str()}));
            // [中文导读] [AllReduce逐行 S3303] 输出错误日志，记录CheckDataType当前阶段和相关参数。
            HCCL_ERROR(
                // [中文导读] [AllReduce逐行 S3304] 补充日志格式：[Check][DataType]errNo[0x%016llx] data type[%s] not supported, support range=[%s]。
                "[Check][DataType]errNo[0x%016llx] data type[%s] not supported, support range=[%s]",
                // [中文导读] [AllReduce逐行 S3305] 输出错误日志，记录CheckDataType当前阶段和相关参数。
                HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), GetDataTypeEnumStr(dataType).c_str(),
                // [中文导读] [AllReduce逐行 S3306] 提供上述日志的实参：GetSupportDataType(needReduce).c_str(。
                GetSupportDataType(needReduce).c_str());
            // [中文导读] [AllReduce逐行 S3307] 返回非归约数据类型不支持错误。
            return HCCL_E_NOT_SUPPORT;
        // [中文导读] [AllReduce逐行 S3308] 结束条件if (notValid || dataType == HCCL_DATA_TYPE_INT128)。
        }
    // [中文导读] [AllReduce逐行 S3309] 结束条件} else。
    }
    // [中文导读] [AllReduce逐行 S3310] 检查枚举值合法性并按归约/非归约场景排除不支持dtype处理完成，返回成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S3311] 结束CheckDataType函数体。
}

std::string GetSupportDataType(bool needReduce)
{
    std::vector<HcclDataType> supportList = {HCCL_DATA_TYPE_INT8,  HCCL_DATA_TYPE_INT16, HCCL_DATA_TYPE_INT32,
                                             HCCL_DATA_TYPE_INT64, HCCL_DATA_TYPE_FP16,  HCCL_DATA_TYPE_FP32};
    if (needReduce) {
        supportList.insert(supportList.end(), {HCCL_DATA_TYPE_BFP16, HCCL_DATA_TYPE_UINT64, HCCL_DATA_TYPE_FP64});
    } else {
        supportList.insert(
            supportList.end(), {HCCL_DATA_TYPE_UINT8, HCCL_DATA_TYPE_UINT16, HCCL_DATA_TYPE_UINT32,
                                HCCL_DATA_TYPE_UINT64, HCCL_DATA_TYPE_FP64, HCCL_DATA_TYPE_HIF8, HCCL_DATA_TYPE_FP8E4M3,
                                HCCL_DATA_TYPE_FP8E5M2, HCCL_DATA_TYPE_FP8E8M0});
        supportList.push_back(HCCL_DATA_TYPE_BFP16);
    }

    std::string supportInfo = "";
    for (u32 i = 0; i < supportList.size(); i++) {
        if (i != 0) {
            supportInfo += ", ";
        }
        supportInfo += GetDataTypeEnumStr(supportList[i]);
    }

    return supportInfo;
}

// [中文导读] [AllReduce逐行 S3338] 声明CheckReduceOp接口：PROD归约额外检查dtype支持列表，其它reduce类型本函数不做附加检查。
HcclResult CheckReduceOp(const HcclDataType dataType, const HcclReduceOp op)
// [中文导读] [AllReduce逐行 S3339] 开始CheckReduceOp的函数体。
{
    // [中文导读] [AllReduce逐行 S3340] 创建PROD归约专用支持数据类型列表。
    std::vector<HcclDataType> prodSupportList
        // [中文导读] [AllReduce逐行 S3341] PROD支持INT8/INT32/INT64/UINT64等整数类型。
        = {HCCL_DATA_TYPE_INT8, HCCL_DATA_TYPE_INT32, HCCL_DATA_TYPE_INT64, HCCL_DATA_TYPE_UINT64,
           // [中文导读] [AllReduce逐行 S3342] PROD同时支持FP16/FP32/FP64，列表中没有INT16和BFP16。
           HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_FP32,  HCCL_DATA_TYPE_FP64};
    // [中文导读] [AllReduce逐行 S3343] 创建归约数据类型错误上报字段列表。
    const std::vector<std::string> infoTitle({"ccl_op", "value", "parameter", "expect"});
    // [中文导读] [AllReduce逐行 S3344] 仅PROD归约进入本函数附加dtype支持检查；SUM等跳过。
    if (op == HcclReduceOp::HCCL_REDUCE_PROD) {
        // [中文导读] [AllReduce逐行 S3345] 在PROD支持列表中找不到当前dtype则返回不支持。
        if (std::find(prodSupportList.begin(), prodSupportList.end(), dataType) == prodSupportList.end()) {
            // [中文导读] [AllReduce逐行 S3346] 开始PROD数据类型不支持的错误上报。
            RPT_INPUT_ERR(
                // [中文导读] [AllReduce逐行 S3347] 指定EI0003错误码和上报字段名字。
                true, "EI0003", infoTitle,
                // [中文导读] [AllReduce逐行 S3348] 构造PROD dtype诊断参数数组。
                std::vector<std::string>(
                    // [中文导读] [AllReduce逐行 S3349] 上报当前dtype及PROD允许的数据类型字符串。
                    {"CheckReduceDataType", GetDataTypeEnumStr(dataType), "dataType", GetReduceProdSupportDataType()}));
            // [中文导读] [AllReduce逐行 S3350] 输出错误日志，记录CheckReduceOp当前阶段和相关参数。
            HCCL_ERROR(
                // [中文导读] [AllReduce逐行 S3351] 补充日志格式：[Check][ReduceOp][DataType]errNo[0x%016llx] reduceop is [%s] data type[%s] not supported, support。
                "[Check][ReduceOp][DataType]errNo[0x%016llx] reduceop is [%s] data type[%s] not supported, support "
                // [中文导读] [AllReduce逐行 S3352] 补充日志格式：range=[%s]。
                "range=[%s]",
                // [中文导读] [AllReduce逐行 S3353] 输出错误日志，记录CheckReduceOp当前阶段和相关参数。
                HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), GetReduceOpEnumStr(op).c_str(),
                // [中文导读] [AllReduce逐行 S3354] 提供上述日志的实参：GetDataTypeEnumStr(dataType).c_str(), GetReduceProdSupportDataType().c_str(。
                GetDataTypeEnumStr(dataType).c_str(), GetReduceProdSupportDataType().c_str());
            // [中文导读] [AllReduce逐行 S3355] 返回PROD数据类型不支持错误。
            return HCCL_E_NOT_SUPPORT;
        // [中文导读] [AllReduce逐行 S3356] 结束条件if (std::find(prodSupportList.begin(), prodSupportList.end(), dataType) == prodSupportList.end())。
        }
    // [中文导读] [AllReduce逐行 S3357] 结束条件if (op == HcclReduceOp::HCCL_REDUCE_PROD)。
    }
    // [中文导读] [AllReduce逐行 S3358] PROD归约额外检查dtype支持列表，其它reduce类型本函数不做附加检查处理完成，返回成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S3359] 结束CheckReduceOp函数体。
}

std::string GetReduceProdSupportDataType()
{
    std::vector<HcclDataType> supportList
        = {HCCL_DATA_TYPE_INT8, HCCL_DATA_TYPE_INT32, HCCL_DATA_TYPE_INT64, HCCL_DATA_TYPE_UINT64,
           HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_FP32,  HCCL_DATA_TYPE_FP64};
    std::string supportInfo = "";
    for (u32 i = 0; i < supportList.size(); i++) {
        if (i != 0) {
            supportInfo += ", ";
        }
        supportInfo += GetDataTypeEnumStr(supportList[i]);
    }

    return supportInfo;
}

// [中文导读] [AllReduce逐行 S3377] 声明SetCommEngine接口：将最终opExecuteConfig转换为执行engine。
HcclResult SetCommEngine(OpParam& param)
// [中文导读] [AllReduce逐行 S3378] 开始SetCommEngine的函数体。
{
    // 使用一个静态的映射表来关联配置和引擎值
    // [中文导读] [AllReduce逐行 S3380] 建立执行配置到通信引擎的静态映射表。
    static const std::unordered_map<OpExecuteConfig, CommEngine> ConfigToEngineMap = {
        // [中文导读] [AllReduce逐行 S3381] HOSTCPU_TS配置对应CPU_TS线程执行域。
        {OpExecuteConfig::HOSTCPU_TS, COMM_ENGINE_CPU_TS},
        // [中文导读] [AllReduce逐行 S3382] AICPU_TS配置对应AICPU_TS执行域，本例在此映射。
        {OpExecuteConfig::AICPU_TS, COMM_ENGINE_AICPU_TS},
        // [中文导读] [AllReduce逐行 S3383] AIV配置对应AIV执行域。
        {OpExecuteConfig::AIV, COMM_ENGINE_AIV},
        // [中文导读] [AllReduce逐行 S3384] AIV_ONLY配置也映射为AIV执行域，强制模式约束在Selector中另查。
        {OpExecuteConfig::AIV_ONLY, COMM_ENGINE_AIV}, // AIV_ONLY 和 AIV 映射到同一引擎
        // [中文导读] [AllReduce逐行 S3385] CCU_MS配置对应CCU执行域。
        {OpExecuteConfig::CCU_MS, COMM_ENGINE_CCU},
        // [中文导读] [AllReduce逐行 S3386] CCU_SCHED配置也对应CCU执行域。
        {OpExecuteConfig::CCU_SCHED, COMM_ENGINE_CCU},
        // [中文导读] [AllReduce逐行 S3387] 裸AICPU配置对应AICPU执行域。
        {OpExecuteConfig::AICPU, COMM_ENGINE_AICPU},
        // [中文导读] [AllReduce逐行 S3388] HOSTCPU配置对应CPU执行域。
        {OpExecuteConfig::HOSTCPU, COMM_ENGINE_CPU},
    // [中文导读] [AllReduce逐行 S3389] 结束配置到引擎的映射表初始化。
    };

    // [中文导读] [AllReduce逐行 S3391] 按当前param.opExecuteConfig查询映射表。
    auto it = ConfigToEngineMap.find(param.opExecuteConfig);
    // [中文导读] [AllReduce逐行 S3392] 配置存在于映射表时写回引擎。
    if (it != ConfigToEngineMap.end()) {
        // [中文导读] [AllReduce逐行 S3393] 保存该配置对应的通信执行域到param.engine。
        param.engine = it->second;
        // [中文导读] [AllReduce逐行 S3394] 执行配置转换成功返回。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S3395] 结束条件if (it != ConfigToEngineMap.end())。
    }

    // [中文导读] [AllReduce逐行 S3397] 输出错误日志，记录SetCommEngine当前阶段和相关参数。
    HCCL_ERROR(
        // [中文导读] [AllReduce逐行 S3398] 补充日志格式：[op_common][SetCommEngine] Unsupported or unknown opExecuteConfig: {%d}。
        "[op_common][SetCommEngine] Unsupported or unknown opExecuteConfig: {%d}",
        // [中文导读] [AllReduce逐行 S3399] 提供上述日志的实参，涉及算子参数。
        static_cast<int>(param.opExecuteConfig));
    // [中文导读] [AllReduce逐行 S3400] 未知执行配置返回不支持错误。
    return HCCL_E_NOT_SUPPORT;
// [中文导读] [AllReduce逐行 S3401] 结束代码块。
}

// [中文导读] [AllReduce逐行 S3403] 声明SingleRankProc接口：单rank算子旁支只需本地复制输入到输出，不执行多rankAllReduce算法。
HcclResult SingleRankProc(HcclComm comm, OpParam& param)
// [中文导读] [AllReduce逐行 S3404] 开始SingleRankProc的函数体。
{
    // [中文导读] [AllReduce逐行 S3405] 记录单rank处理profiling开始时间。
    uint64_t beginTime = HcommGetProfilingSysCycleTime();
    // [中文导读] [AllReduce逐行 S3406] 输出运行日志，记录SingleRankProc当前阶段和相关参数。
    HCCL_INFO("[SingleRankProc]Start to execute HcclExecOp. HcommGetProfilingSysCycleTime[%llu us]", beginTime);
    // [中文导读] [AllReduce逐行 S3407] AIV_ONLY模式不能使用此单rank处理旁支。
    if (param.commOpExpansionMode == HcclOpExpansionMode::HCCL_OP_EXPANSION_AIV_ONLY) {
        // [中文导读] [AllReduce逐行 S3408] 输出错误日志，记录SingleRankProc当前阶段和相关参数。
        HCCL_ERROR(
            // [中文导读] [AllReduce逐行 S3409] 补充日志格式：[SingleRankProc] opType[%s] currently do not select aiv mode, aiv only not support。
            "[SingleRankProc] opType[%s] currently do not select aiv mode, aiv only not support, "
            // [中文导读] [AllReduce逐行 S3410] 补充日志格式：please ensure rankNum is greater than one。
            "please ensure rankNum is greater than one",
            // [中文导读] [AllReduce逐行 S3411] 提供上述日志的实参，涉及算子参数。
            GetHcclCMDTypeStr(param.opType));
        // [中文导读] [AllReduce逐行 S3412] 单rank但强制AIV_ONLY时返回不支持。
        return HCCL_E_NOT_SUPPORT;
    // [中文导读] [AllReduce逐行 S3413] 结束条件if (param.commOpExpansionMode == HcclOpExpansionMode::HCCL_OP_EXPANSION_AIV_ONLY)。
    }
    // [中文导读] [AllReduce逐行 S3414] 单rank的Send/Receive按源码成功返回，不执行数据复制。
    if (param.opType == HcclCMDType::HCCL_CMD_SEND || param.opType == HcclCMDType::HCCL_CMD_RECEIVE) {
        // [中文导读] [AllReduce逐行 S3415] 输出警告日志，记录SingleRankProc当前阶段和相关参数。
        HCCL_WARNING("[%s] ranksize == 1 is not support BATCHSENDRECV SEND RECV", __func__);
        // [中文导读] [AllReduce逐行 S3416] 返回单rankP2P分支成功状态。
        return HcclResult::HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S3417] 结束条件if (param.opType == HcclCMDType::HCCL_CMD_SEND || param.opType == HcclCMDType::HCCL_CMD_RECEIVE)。
    }
    // [中文导读] [AllReduce逐行 S3418] 输入输出地址相同时无需本地复制。
    if (param.inputPtr == param.outputPtr) {
        // [中文导读] [AllReduce逐行 S3419] 输出警告日志，记录SingleRankProc当前阶段和相关参数。
        HCCL_WARNING("[%s] sendBuf == recvBuf, return success", __func__);
        // [中文导读] [AllReduce逐行 S3420] 原地单rank场景直接成功返回。
        return HcclResult::HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S3421] 结束条件if (param.inputPtr == param.outputPtr)。
    }
    // [中文导读] [AllReduce逐行 S3422] 初始化需要本地复制的字节数量为零。
    u64 len{0};
    // [中文导读] [AllReduce逐行 S3423] AllToAll和AllToAllV使用其发送dtype/count布局。
    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV
        // [中文导读] [AllReduce逐行 S3424] AllToAllVC同样使用all2AllVDataDes布局。
        || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        // [中文导读] [AllReduce逐行 S3425] 读取AllToAll发送数据类型对应的元素字节大小。
        len = DATATYPE_SIZE_TABLE[param.all2AllVDataDes.sendType]
              // [中文导读] [AllReduce逐行 S3426] 单Rank的AllToAll长度续行：取首个sendCounts元素，乘前行的数据类型字节数得到拷贝长度；AllReduce不走此分支。
              * *(static_cast<const u64*>(param.all2AllVDataDes.sendCounts));
    // [中文导读] [AllReduce逐行 S3427] AllGatherV或ReduceScatterV使用vDataDes布局。
    } else if (param.opType == HCCL_CMD_ALLGATHER_V || param.opType == HCCL_CMD_REDUCE_SCATTER_V) {
        // [中文导读] [AllReduce逐行 S3428] vDataDes数据类型大小乘counts第一项得到有效复制长度。
        len = DATATYPE_SIZE_TABLE[param.vDataDes.dataType] * *(static_cast<const u64*>(param.vDataDes.counts));
    // [中文导读] [AllReduce逐行 S3429] 上述条件不成立时进入替代分支。
    } else {
        // [中文导读] [AllReduce逐行 S3430] 普通AllReduce按DataDes.count乘dtype字节数计算复制长度。
        len = DATATYPE_SIZE_TABLE[param.DataDes.dataType] * param.DataDes.count;
    // [中文导读] [AllReduce逐行 S3431] 结束条件} else。
    }
    // [中文导读] [AllReduce逐行 S3432] 输出运行日志，记录SingleRankProc当前阶段和相关参数。
    HCCL_INFO("[%s] sendBuf[%p], recvBuf[%p], len[%llu]", __func__, param.inputPtr, param.outputPtr, len);
    // [中文导读] [AllReduce逐行 S3433] 有非零数据字节时才创建流线程并安排复制。
    if (len > 0) {
        // [中文导读] [AllReduce逐行 S3434] 初始化本次用户流CPU_TS线程句柄。
        ThreadHandle cpuTsThread{0};
        // [中文导读] [AllReduce逐行 S3435] 将本次ACL用户流包装为CPU_TS线程，通知容量为1。
        CHK_RET(HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, param.stream, 1, &cpuTsThread));
        // [中文导读] [AllReduce逐行 S3436] 创建单rank算子维测信息结构。
        HcclDfxOpInfoCompat hcclDfxOpInfo{}; // Op注册
        // [中文导读] [AllReduce逐行 S3437] 保存单算子或图模式到维测信息。
        hcclDfxOpInfo.opMode = static_cast<u32>(param.opMode);
        // [中文导读] [AllReduce逐行 S3438] 保存当前命令类型到维测信息。
        hcclDfxOpInfo.opType = static_cast<u32>(param.opType);
        // [中文导读] [AllReduce逐行 S3439] 保存当前SUM等归约类型到维测信息。
        hcclDfxOpInfo.reduceOp = static_cast<u32>(param.reduceType);
        // [中文导读] [AllReduce逐行 S3440] 按算子描述布局读取数据类型写进维测结构。
        CHK_RET(GetHcclDfxOpInfoDataType(param, hcclDfxOpInfo.dataType));
        // [中文导读] [AllReduce逐行 S3441] 初始化通信域rank数量查询结果。
        u32 userRankSize{0}; // rankSize获取指定算子的dataCount
        // [中文导读] [AllReduce逐行 S3442] 查询本次通信域rank数量。
        CHK_RET(HcclGetRankSize(comm, &userRankSize));
        // [中文导读] [AllReduce逐行 S3443] 按算子类型及rank数计算维测记录的元素总数。
        CHK_RET(GetHcclDfxOpInfoDataCount(param, userRankSize, hcclDfxOpInfo.dataCount));
        // [中文导读] [AllReduce逐行 S3444] 保存通用root字段到维测结构。
        hcclDfxOpInfo.root = param.root;
        // [中文导读] [AllReduce逐行 S3445] 保存本次执行引擎到维测结构。
        hcclDfxOpInfo.engine = param.engine;
        // [中文导读] [AllReduce逐行 S3446] 将本次用户流对应CPU_TS线程放进维测结构。
        hcclDfxOpInfo.cpuTsThread = cpuTsThread;
        // [中文导读] [AllReduce逐行 S3447] 设置维测结构Host结果通知槽字段。
        hcclDfxOpInfo.cpuWaitAicpuNotifyIdx = HOST_WAIT_AICPU_NOTIFYIDX;
        // [中文导读] [AllReduce逐行 S3448] 用SingleRankProc名字构造本地复制旁支的算法tag。
        CHK_RET(SetOpParamAlgTag(param, "SingleRankProc"));
        // [中文导读] [AllReduce逐行 S3449] 将生成的算法tag复制到维测固定字符串区。
        s32 sRet = strncpy_s(hcclDfxOpInfo.algTag, ALG_TAG_LENGTH, param.algTag, ALG_TAG_LENGTH);
        // [中文导读] [AllReduce逐行 S3450] 开始条件检查宏，条件成立时记录错误并返回下方指定错误码。
        CHK_PRT_RET(
            // [中文导读] [AllReduce逐行 S3451] tag安全复制失败时触发日志和错误返回。
            sRet != EOK,
            // [中文导读] [AllReduce逐行 S3452] 输出错误日志，记录SingleRankProc当前阶段和相关参数。
            HCCL_ERROR("%s call strncpy_s failed, param.algTag %s, return %d.", __func__, param.algTag, sRet),
            // [中文导读] [AllReduce逐行 S3453] 算法tag复制失败返回内存错误。
            HCCL_E_MEMORY);
        // [中文导读] [AllReduce逐行 S3454] 向通信域登记单rank算子的维测信息。
        CHK_RET(HcclDfxRegOpInfoByCommId(param.commName, reinterpret_cast<void*>(&hcclDfxOpInfo)));
        // [中文导读] [AllReduce逐行 S3455] 在用户CPU_TS线程安排input到output的len字节本地复制。
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(cpuTsThread, param.outputPtr, param.inputPtr, len)));
    // [中文导读] [AllReduce逐行 S3456] 结束条件if (len > 0)。
    }
    // [中文导读] [AllReduce逐行 S3457] 上报单rank算子profiling时间。
    CHK_RET(HcclProfilingReportOp(comm, beginTime));
    // [中文导读] [AllReduce逐行 S3458] 单rank复制任务安排成功返回，实际完成仍由用户流执行。
    return HcclResult::HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S3459] 结束SingleRankProc函数体。
}

HcclResult HcclCheckTag(const char* tag)
{
    CHK_PTR_NULL(tag);

    u32 tagLen = strnlen(tag, HCCL_TAG_MAX_LEN + 1);
    if (UNLIKELY((tagLen == (HCCL_TAG_MAX_LEN + 1) || tagLen == 0))) {
        HCCL_ERROR("[Check][Tag]errNo[0x%016llx] tag is too long", HCOM_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

static HcclResult BuildCcuExtraTag(const OpParam& param, std::string& ccuExtraTag)
{
    HcclDataType tmpDataType;
    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV
        || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        tmpDataType = param.all2AllVDataDes.sendType;
    } else if (
        param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V || param.opType == HcclCMDType::HCCL_CMD_ALLGATHER_V) {
        tmpDataType = param.vDataDes.dataType;
    } else {
        tmpDataType = param.DataDes.dataType;
    }
    ccuExtraTag = "_" + HCOM_DATA_TYPE_STR_MAP.at(tmpDataType);

    if (param.opType == HcclCMDType::HCCL_CMD_ALLREDUCE || param.opType == HcclCMDType::HCCL_CMD_REDUCE
        || param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER
        || param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V) {
        ccuExtraTag += "_" + HCOM_REDUCE_OP_STR_MAP.at(param.reduceType);
    }

    if (param.opType == HcclCMDType::HCCL_CMD_REDUCE || param.opType == HcclCMDType::HCCL_CMD_SCATTER
        || param.opType == HcclCMDType::HCCL_CMD_BROADCAST) {
        ccuExtraTag += "_r" + std::to_string(param.root);
    }
    return HCCL_SUCCESS;
}

// [中文导读] [AllReduce逐行 S3500] 声明SetOpParamAlgTag接口：构造算法资源关联tag，CCU额外添加dtype/reduce字段。
HcclResult SetOpParamAlgTag(OpParam& param, const std::string& algName)
// [中文导读] [AllReduce逐行 S3501] 开始SetOpParamAlgTag的函数体。
{
    // [中文导读] [AllReduce逐行 S3502] 拷贝算法名字符串用于组装算法tag。
    std::string temp = algName; // 创建algName的副本

    // [中文导读] [AllReduce逐行 S3504] 声明tag的执行位置后缀字符串。
    const char* launchMode
        // [中文导读] [AllReduce逐行 S3505] AICPU或AICPU_TS引擎的tag后缀为device。
        = (((param.engine == CommEngine::COMM_ENGINE_AICPU) || (param.engine == CommEngine::COMM_ENGINE_AICPU_TS)) ?
               // [中文导读] [AllReduce逐行 S3506] Device展开模式使用device字符串后缀。
               "device" :
               // [中文导读] [AllReduce逐行 S3507] 其它引擎使用host字符串后缀。
               "host");
    // [中文导读] [AllReduce逐行 S3508] 声明tag格式化返回的实际字符数。
    int len;
    // 图模式下去掉param.tag前缀，避免tag不同导致algTag不同而无法复用资源
    // [中文导读] [AllReduce逐行 S3510] CCU OFFLOAD图模式不带用户tag前缀，便于复用同算法资源。
    if (param.opMode == OpMode::OFFLOAD && param.engine == CommEngine::COMM_ENGINE_CCU) {
        // [中文导读] [AllReduce逐行 S3511] 开始格式化CCU图模式关联tag。
        len = snprintf_s(
            // [中文导读] [AllReduce逐行 S3512] 构造Graph_算法名_执行位置的固定数组字符串。
            param.algTag, sizeof(param.algTag), sizeof(param.algTag), "Graph_%s_%s", temp.c_str(), launchMode);
    // [中文导读] [AllReduce逐行 S3513] 上述条件不成立时进入替代分支。
    } else {
        // [中文导读] [AllReduce逐行 S3514] 普通场景开始格式化算法关联tag。
        len = snprintf_s(
            // [中文导读] [AllReduce逐行 S3515] 构造算子tag_算法名_执行位置；本例算法名字为AicpuAllReduceSoleMeshOneShot。
            param.algTag, sizeof(param.algTag), sizeof(param.algTag), "%s_%s_%s", param.tag, temp.c_str(), launchMode);
    // [中文导读] [AllReduce逐行 S3516] 结束条件} else。
    }
    // [中文导读] [AllReduce逐行 S3517] 检查snprintf返回负数或超出固定tag容量。
    if (len < 0 || len >= sizeof(param.algTag)) {
        // [中文导读] [AllReduce逐行 S3518] 输出错误日志，记录SetOpParamAlgTag当前阶段和相关参数。
        HCCL_ERROR("failed to fill param.algTag");
        // [中文导读] [AllReduce逐行 S3519] 算法tag构造失败返回内部错误。
        return HcclResult::HCCL_E_INTERNAL;
    // [中文导读] [AllReduce逐行 S3520] 结束条件if (len < 0 || len >= sizeof(param.algTag))。
    }

    // ccu模式，考虑kernel是否能复用，需要添加dataType和reduceType
    // [中文导读] [AllReduce逐行 S3523] CCU算法tag还需增加数据类型及归约类型等专属字段。
    if (param.engine == CommEngine::COMM_ENGINE_CCU) {
        // [中文导读] [AllReduce逐行 S3524] 用try包围CCU枚举字符串查询，捕获越界异常。
        try {
            // [中文导读] [AllReduce逐行 S3525] 创建存放CCU附加字段的临时字符串。
            std::string ccuExtraTag;
            // [中文导读] [AllReduce逐行 S3526] 调用BuildCcuExtraTag生成CCU额外资源关联字段，非本AICPU例。
            CHK_RET(BuildCcuExtraTag(param, ccuExtraTag));
            // [中文导读] [AllReduce逐行 S3527] 计算基础tag之后固定字符数组还剩的字节容量。
            size_t remainBytes = sizeof(param.algTag) - len;

            // [中文导读] [AllReduce逐行 S3529] 从algTag+len处追加CCU附加字符串。
            int len_ccu = snprintf_s(param.algTag + len, remainBytes, remainBytes, "%s", ccuExtraTag.c_str());
            // [中文导读] [AllReduce逐行 S3530] 开始条件检查宏，条件成立时记录错误并返回下方指定错误码。
            CHK_PRT_RET(
                // [中文导读] [AllReduce逐行 S3531] 检查追加字符串失败或超过剩余tag容量。
                (len_ccu < 0 || len_ccu >= sizeof(param.algTag) - len),
                // [中文导读] [AllReduce逐行 S3532] CCU附加tag写入失败记录日志并返回内部错误。
                HCCL_ERROR("failed to fill alg tag with ccu dataType"), HCCL_E_INTERNAL);
        // [中文导读] [AllReduce逐行 S3533] 捕获CCU数据类型/归约枚举映射查询的out_of_range异常。
        } catch (const std::out_of_range& e) {
            // [中文导读] [AllReduce逐行 S3534] 输出错误日志，记录SetOpParamAlgTag当前阶段和相关参数。
            HCCL_ERROR("[SetOpParamAlgTag] dataType or reduceType out of range: %s", e.what());
            // [中文导读] [AllReduce逐行 S3535] 枚举映射越界时返回参数错误。
            return HCCL_E_PARA;
        // [中文导读] [AllReduce逐行 S3536] 结束代码块。
        }
    // [中文导读] [AllReduce逐行 S3537] 结束条件if (param.engine == CommEngine::COMM_ENGINE_CCU)。
    }
    // [中文导读] [AllReduce逐行 S3538] 基础tag及必要的CCU附加字段均构造成功，返回HCCL_SUCCESS。
    return HcclResult::HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S3539] 结束SetOpParamAlgTag函数体。
}

// [中文导读] [AllReduce逐行 S3541] 声明展开模式入口，comm提供域配置，param接收结果。
HcclResult HcclGetOpExpansionMode(HcclComm comm, OpParam& param)
// [中文导读] [AllReduce逐行 S3542] 进入展开模式函数体。
{
    // [中文导读] [AllReduce逐行 S3543] 将候选finalMode置为INVALID，等待配置决策覆盖。
    HcclOpExpansionMode finalMode = HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_INVALID;
    // 第一步：决定使用哪种模式
    // [中文导读] [AllReduce逐行 S3545] 调用DecideHcclOpExpansionMode读取通信域或环境模式。
    HcclResult ret = DecideHcclOpExpansionMode(comm, finalMode);
    // [中文导读] [AllReduce逐行 S3546] 检查模式决策是否失败。
    if (ret != HCCL_SUCCESS) {
        // [中文导读] [AllReduce逐行 S3547] 记录配置决策错误码。
        HCCL_ERROR("DecideHcclOpExpansionMode failed, ret: %d", ret);
        // [中文导读] [AllReduce逐行 S3548] 返回模式决策错误，不执行Engine设置。
        return ret;
    // [中文导读] [AllReduce逐行 S3549] 结束决策失败分支。
    }
    // [中文导读] [AllReduce逐行 S3550] 把有效模式保存到param，供AIV_ONLY等后续条件使用。
    param.commOpExpansionMode = finalMode;

    // 第二步：应用选择的模式到param
    // [中文导读] [AllReduce逐行 S3553] 调用ApplyOpExpansionMode把模式映射为执行配置和Engine。
    ret = ApplyOpExpansionMode(param, finalMode);
    // [中文导读] [AllReduce逐行 S3554] 检查模式映射及必要Kernel加载是否失败。
    if (ret != HCCL_SUCCESS) {
        // [中文导读] [AllReduce逐行 S3555] 记录模式应用错误。
        HCCL_ERROR("ApplyOpExpansionMode failed, ret: %d", ret);
        // [中文导读] [AllReduce逐行 S3556] 向调用者传播模式应用错误。
        return ret;
    // [中文导读] [AllReduce逐行 S3557] 结束模式应用失败分支。
    }
    // [中文导读] [AllReduce逐行 S3558] 决策与应用均成功，返回成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S3559] 结束展开模式入口。
}

HcclResult HcclGetHcclAlgo(HcclComm comm, std::string& hcclAlgo)
{
    if (GetHcommVersion() <= CANN_VERSION(9, 2, 0, 1)) {
        HCCL_INFO("[HcclGetHcclAlgo] HcclConfigGetInfo not supported, skip.");
        return HcclResult::HCCL_SUCCESS;
    }
    auto& hcommFunction = ops_hccl::DlHcommFunction::GetInstance();
    if (!hcommFunction.dlHcclConfigGetInfo) {
        HCCL_INFO("[HcclGetHcclAlgo] HcclConfigGetInfo not supported, skip.");
        return HcclResult::HCCL_SUCCESS;
    }
    std::vector<char> buf(HCCL_COMM_ALGO_MAX_LENGTH, '\0');
    uint32_t infoLen = static_cast<uint32_t>(buf.size());
    HcclResult ret = hcommFunction.dlHcclConfigGetInfo(
        comm, static_cast<HcclConfigType>(HCCL_CONFIG_TYPE_HCCL_ALGO), infoLen, buf.data());
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_WARNING("[HcclGetHcclAlgo] HcclConfigGetInfo failed, ret: %d", ret);
        return HcclResult::HCCL_SUCCESS;
    }
    hcclAlgo.assign(buf.data(), strnlen(buf.data(), buf.size()));
    HCCL_DEBUG("[HcclGetHcclAlgo] hcclAlgo from comm: [%s]", hcclAlgo.c_str());

    return HcclResult::HCCL_SUCCESS;
}

// [中文导读] [AllReduce逐行 S3586] 声明展开模式决策入口，通过引用finalMode返回最终模式。
HcclResult DecideHcclOpExpansionMode(HcclComm comm, HcclOpExpansionMode& finalMode)
// [中文导读] [AllReduce逐行 S3587] 进入模式决策函数体。
{
    // [中文导读] [AllReduce逐行 S3588] 保存域配置模式，初值为INVALID。
    HcclOpExpansionMode configOpExpansionMode = HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_INVALID;
    // [中文导读] [AllReduce逐行 S3589] 先标记尚未从域配置取得模式。
    bool useConfigOpExpansionMode = false;
    // [中文导读] [AllReduce逐行 S3590] 取得HCOMM动态符号表以探测配置查询接口。
    auto& hcommFunction = ops_hccl::DlHcommFunction::GetInstance();
    // [中文导读] [AllReduce逐行 S3591] 接口存在才尝试读取域展开模式。
    if (hcommFunction.dlHcclConfigGetInfo) {
        // [中文导读] [AllReduce逐行 S3592] 指定输出缓冲容量为展开模式枚举大小。
        uint32_t infoLen = sizeof(HcclOpExpansionMode);
        // [中文导读] [AllReduce逐行 S3593] 调用配置查询并在失败时立即返回。
        CHK_RET(hcommFunction.dlHcclConfigGetInfo(
            // [中文导读] [AllReduce逐行 S3594] 从comm查询OP_EXPANSION_MODE，写入configOpExpansionMode。
            comm, HcclConfigType::HCCL_CONFIG_TYPE_OP_EXPANSION_MODE, infoLen, &configOpExpansionMode));
        // [中文导读] [AllReduce逐行 S3595] 将查询结果设为当前最终候选模式。
        finalMode = configOpExpansionMode;
        // [中文导读] [AllReduce逐行 S3596] 记录本次已取得通信域配置。
        useConfigOpExpansionMode = true;
    // [中文导读] [AllReduce逐行 S3597] 域配置接口缺失时进入兼容默认路径。
    } else {
        // [中文导读] [AllReduce逐行 S3598] 记录无法使用域配置，将考虑环境配置。
        HCCL_INFO("[DecideHcclOpExpansionMode] HcclConfigGetInfo is not supported, use environment mode.");
        // [中文导读] [AllReduce逐行 S3599] 暂将候选设CCU_MS，后续环境条件可能覆盖。
        finalMode = static_cast<HcclOpExpansionMode>(opExpansionModeCcuMs);
    // [中文导读] [AllReduce逐行 S3600] 结束域配置可用性分支。
    }

    // A5仅通过HcclConfigGetInfo获取展开模式，其他型号保留环境变量方式
    // [中文导读] [AllReduce逐行 S3603] 创建设备类型输出变量，初值为非法边界值。
    HcclDevType deviceType = HcclDevType::DEV_TYPE_COUNT;
    // [中文导读] [AllReduce逐行 S3604] 查询当前设备类型，失败向上传播。
    CHK_RET(HcclGetDeviceType(deviceType));
    // [中文导读] [AllReduce逐行 S3605] 仅非新流程设备或未取得域配置时允许环境变量路径。
    if (!shouldGoOutPlace(deviceType) || !useConfigOpExpansionMode) {
        // [中文导读] [AllReduce逐行 S3606] 环境优先判断是否配置AICPU展开。
        if (GetExternalInputHcclAicpuUnfold() == true) {
            // [中文导读] [AllReduce逐行 S3607] AICPU配置生效时选择AI_CPU模式。
            finalMode = HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_AI_CPU;
        // [中文导读] [AllReduce逐行 S3608] 否则检查AIV_ONLY环境选项。
        } else if (GetExternalInputHcclAivOnlyMode() == true) {
            // [中文导读] [AllReduce逐行 S3609] AIV_ONLY生效时选择强制AIV模式。
            finalMode = HcclOpExpansionMode::HCCL_OP_EXPANSION_AIV_ONLY;
        // [中文导读] [AllReduce逐行 S3610] 否则检查普通AIV环境选项。
        } else if (GetExternalInputHcclAivMode() == true) {
            // [中文导读] [AllReduce逐行 S3611] 普通AIV生效时选择AIV模式。
            finalMode = HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_AIV;
        // [中文导读] [AllReduce逐行 S3612] 否则检查CCU_MS环境选项。
        } else if (GetExternalInputHcclCcuMSMode()) {
            // [中文导读] [AllReduce逐行 S3613] CCU_MS生效时选择对应枚举值。
            finalMode = static_cast<HcclOpExpansionMode>(opExpansionModeCcuMs);
        // [中文导读] [AllReduce逐行 S3614] 否则检查CCU_SCHED环境选项。
        } else if (GetExternalInputHcclCcuSchedMode()) {
            // [中文导读] [AllReduce逐行 S3615] CCU_SCHED生效时选择对应枚举值。
            finalMode = static_cast<HcclOpExpansionMode>(opExpansionModeCcuSched);
        // [中文导读] [AllReduce逐行 S3616] 结束环境模式优先序；均未设置时保留之前候选。
        }
        // [中文导读] [AllReduce逐行 S3617] 检查环境覆盖结果是否与已有域配置冲突。
        if (useConfigOpExpansionMode && configOpExpansionMode != finalMode) {
            // [中文导读] [AllReduce逐行 S3618] 开始记录模式冲突诊断。
            HCCL_DEBUG(
                // [中文导读] [AllReduce逐行 S3619] 提供域配置与环境模式的日志占位。
                "[DecideHcclOpExpansionMode] configOpExpansionMode: %d, environment mode: %d, conflict, use "
                // [中文导读] [AllReduce逐行 S3620] 补充诊断说明，当前这个路径采用环境模式。
                "environment mode.",
                // [中文导读] [AllReduce逐行 S3621] 把两种模式值写入冲突日志。
                configOpExpansionMode, finalMode);
        // [中文导读] [AllReduce逐行 S3622] 结束配置冲突日志分支。
        }
    // [中文导读] [AllReduce逐行 S3623] 结束允许环境模式覆盖的条件路径。
    }
    // [中文导读] [AllReduce逐行 S3624] 记录最终选择的展开模式。
    HCCL_INFO("[DecideHcclOpExpansionMode] finalMode: %d.", finalMode);

    // [中文导读] [AllReduce逐行 S3626] 将最终模式通过引用带回并返回成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S3627] 结束模式决策函数。
}

HcclResult GetUbMultiChannelNum(HcclComm comm, u32& multiChannelNum)
{
    multiChannelNum = 1;
    auto& hcommFunction = ops_hccl::DlHcommFunction::GetInstance();
    if (!hcommFunction.dlHcclConfigGetInfo) {
        HCCL_INFO("[GetUbMultiChannelNum] HcclConfigGetInfo not supported, use default.");
        return HCCL_SUCCESS;
    }
    u32 cfgNum = 0;
    uint32_t infoLen = sizeof(u32);
    HcclResult ret = hcommFunction.dlHcclConfigGetInfo(
        comm, static_cast<HcclConfigType>(HCCL_CONFIG_TYPE_UB_MULTI_CHANNEL_NUM), infoLen, &cfgNum);
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING("[GetUbMultiChannelNum] HcclConfigGetInfo failed, ret[%d], use default.", ret);
        return HCCL_SUCCESS;
    }
    multiChannelNum = cfgNum;
    return HCCL_SUCCESS;
}

// [中文导读] [AllReduce逐行 S3649] 声明模式应用入口，修改本次OpParam的执行配置与Engine。
HcclResult ApplyOpExpansionMode(OpParam& param, HcclOpExpansionMode finalMode)
// [中文导读] [AllReduce逐行 S3650] 进入模式应用函数体。
{
    // [中文导读] [AllReduce逐行 S3651] 按finalMode分发到相应执行引擎。
    switch (finalMode) {
        // [中文导读] [AllReduce逐行 S3652] 匹配AICPU_TS展开模式。
        case HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_AI_CPU:
            // [中文导读] [AllReduce逐行 S3653] 设置opExecuteConfig=AICPU_TS供selector使用。
            param.opExecuteConfig = OpExecuteConfig::AICPU_TS;
            // [中文导读] [AllReduce逐行 S3654] 设置资源/原语使用的CommEngine=AICPU_TS。
            param.engine = CommEngine::COMM_ENGINE_AICPU_TS;
            // [中文导读] [AllReduce逐行 S3655] 调用LoadAICPUKernel准备入口，失败则立即返回。
            CHK_RET(LoadAICPUKernel());
            // [中文导读] [AllReduce逐行 S3656] 记录选定的AICPU_TS模式。
            HCCL_DEBUG("[ApplyOpExpansionMode] AICPU mode selected.");
            // [中文导读] [AllReduce逐行 S3657] 结束当前模式分支，避免落入下一case。
            break;
        // [中文导读] [AllReduce逐行 S3658] 匹配AIV展开模式。
        case HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_AIV:
            // [中文导读] [AllReduce逐行 S3659] 设置opExecuteConfig=AIV供selector使用。
            param.opExecuteConfig = OpExecuteConfig::AIV;
            // [中文导读] [AllReduce逐行 S3660] 设置资源/原语使用的CommEngine=AIV。
            param.engine = CommEngine::COMM_ENGINE_AIV;
            // [中文导读] [AllReduce逐行 S3661] 调用RegisterKernel准备入口，失败则立即返回。
            CHK_RET(RegisterKernel());
            // [中文导读] [AllReduce逐行 S3662] 记录选定的AIV模式。
            HCCL_DEBUG("[ApplyOpExpansionMode] AIV mode selected.");
            // [中文导读] [AllReduce逐行 S3663] 结束当前模式分支，避免落入下一case。
            break;
        // [中文导读] [AllReduce逐行 S3664] 匹配AIV_ONLY展开模式。
        case HcclOpExpansionMode::HCCL_OP_EXPANSION_AIV_ONLY:
            // [中文导读] [AllReduce逐行 S3665] 设置opExecuteConfig=AIV_ONLY供selector使用。
            param.opExecuteConfig = OpExecuteConfig::AIV_ONLY;
            // [中文导读] [AllReduce逐行 S3666] 设置资源/原语使用的CommEngine=AIV。
            param.engine = CommEngine::COMM_ENGINE_AIV;
            // [中文导读] [AllReduce逐行 S3667] 调用RegisterKernel准备入口，失败则立即返回。
            CHK_RET(RegisterKernel());
            // [中文导读] [AllReduce逐行 S3668] 记录选定的AIV_ONLY模式。
            HCCL_DEBUG("[ApplyOpExpansionMode] AIV_ONLY mode selected.");
            // [中文导读] [AllReduce逐行 S3669] 结束当前模式分支，避免落入下一case。
            break;
        // [中文导读] [AllReduce逐行 S3670] 匹配CCU_MS展开模式枚举。
        case static_cast<HcclOpExpansionMode>(opExpansionModeCcuMs):
            // [中文导读] [AllReduce逐行 S3671] 设置opExecuteConfig=CCU_MS区分CCU选择策略。
            param.opExecuteConfig = OpExecuteConfig::CCU_MS;
            // [中文导读] [AllReduce逐行 S3672] 两种CCU配置均使用COMM_ENGINE_CCU资源引擎。
            param.engine = CommEngine::COMM_ENGINE_CCU;
            // [中文导读] [AllReduce逐行 S3673] 记录选定的CCU_MS模式。
            HCCL_DEBUG("[ApplyOpExpansionMode] CCU_MS mode selected.");
            // [中文导读] [AllReduce逐行 S3674] 结束该CCU模式分支，避免穿透。
            break;
        // [中文导读] [AllReduce逐行 S3675] 匹配CCU_SCHED展开模式枚举。
        case static_cast<HcclOpExpansionMode>(opExpansionModeCcuSched):
            // [中文导读] [AllReduce逐行 S3676] 设置opExecuteConfig=CCU_SCHED区分CCU选择策略。
            param.opExecuteConfig = OpExecuteConfig::CCU_SCHED;
            // [中文导读] [AllReduce逐行 S3677] 两种CCU配置均使用COMM_ENGINE_CCU资源引擎。
            param.engine = CommEngine::COMM_ENGINE_CCU;
            // [中文导读] [AllReduce逐行 S3678] 记录选定的CCU_SCHED模式。
            HCCL_DEBUG("[ApplyOpExpansionMode] CCU_SCHED mode selected.");
            // [中文导读] [AllReduce逐行 S3679] 结束该CCU模式分支，避免穿透。
            break;
        // [中文导读] [AllReduce逐行 S3680] 未识别模式进入兼容AICPU_TS兜底。
        default:
            // 回退到aicpu
            // [中文导读] [AllReduce逐行 S3682] 记录无效模式并说明回退到AICPU_TS。
            HCCL_WARNING("[ApplyOpExpansionMode] Invalid HcclOpExpansionMode: %d, fallback to AICPU_TS.", finalMode);
            // [中文导读] [AllReduce逐行 S3683] 把执行配置设为AICPU_TS。
            param.opExecuteConfig = OpExecuteConfig::AICPU_TS;
            // [中文导读] [AllReduce逐行 S3684] 把资源Engine设为AICPU_TS。
            param.engine = CommEngine::COMM_ENGINE_AICPU_TS;
            // [中文导读] [AllReduce逐行 S3685] 加载AICPU Kernel，加载失败由CHK_RET向上传播。
            CHK_RET(LoadAICPUKernel());
            // [中文导读] [AllReduce逐行 S3686] 结束默认分支，离开switch。
            break;
    // [中文导读] [AllReduce逐行 S3687] 结束模式分发switch。
    }
    // [中文导读] [AllReduce逐行 S3688] 模式应用及必要加载成功。
    return HcclResult::HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S3689] 结束模式应用函数。
}

HcclResult
HcclRegstryBuff(HcclComm comm, const char* memTag, void* bufferPtr, uint64_t bufferSize, HcclMemHandle* memHandle)
{
    CHK_PTR_NULL(memHandle);
    CommMem regMem{COMM_MEM_TYPE_DEVICE, bufferPtr, bufferSize};
    CHK_RET(HcclCommMemReg(comm, memTag, &regMem, memHandle));
    HCCL_INFO("[%s] regMemAddr[%p] regMemSize[%llu]", __func__, regMem.addr, regMem.size);
    CHK_PTR_NULL(*memHandle);
    return HCCL_SUCCESS;
}

HcclResult
HcclGetRemoteBuff(HcclComm comm, ChannelHandle channel, const char* memTag, void** bufferPtr, uint64_t* bufferSize)
{
    CHK_PTR_NULL(bufferPtr);
    CHK_PTR_NULL(bufferSize);

    u32 memNum;
    CommMem* remoteMemList;
    char** memTags;
    CHK_RET(HcclChannelGetRemoteMems(comm, channel, &memNum, &remoteMemList, &memTags));
    HCCL_INFO("[%s] HcclChannelGetRemoteMems memNum[%u]", __func__, memNum);
    for (u32 i = 0; i < memNum; i++) {
        HCCL_INFO("[%s] memNum[%u/%u] memTags[%s]", __func__, i + 1, memNum, memTags[i]);
        if (strcmp(memTags[i], memTag) == 0) {
            *bufferPtr = remoteMemList[i].addr;
            *bufferSize = remoteMemList[i].size;
            HCCL_INFO(
                "[%s] Found memNum[%u/%u]: addr=%p, size=%llu", __func__, i + 1, memNum, remoteMemList[i].addr,
                remoteMemList[i].size);
            break;
        }
    }
    if (*bufferPtr == nullptr) {
        HCCL_WARNING("[%s] Failed to find %s in remote mem list", __func__, memTag);
    }
    return HCCL_SUCCESS;
}

HcclResult LogHcclExit(const std::string& opName, const char* tag, HcclUs startut, bool forceLog)
{
    if (forceLog || GetExternalInputHcclEnableEntryLog()) {
        HcclUs endut = TIME_NOW();
        std::string endInfo = opName + ":success,take time: " + std::to_string(DURATION_US(endut - startut).count())
                              + " us, tag: " + tag;
        HCCL_RUN_INFO("%s", endInfo.c_str());
    }
    return HCCL_SUCCESS;
}

HcclResult GetAivParamStorageByComm(HcclComm comm, AivParamStorage** aivParam, bool ifCreate)
{
    if (comm == nullptr || aivParam == nullptr) {
        HCCL_ERROR("[GetAivParamStorageByComm] Invalid parameters");
        return HCCL_E_PARA;
    }

    void* aivParamCtx = nullptr;
    uint64_t size = sizeof(AivParamStorage);

    const char* aivParamTag = "AivParamStorage";
    if (HcclEngineCtxGet(comm, aivParamTag, CommEngine::COMM_ENGINE_CPU_TS, &aivParamCtx, &size) != HCCL_SUCCESS) {
        if (ifCreate) {
            CHK_RET(HcclEngineCtxCreate(comm, aivParamTag, CommEngine::COMM_ENGINE_CPU_TS, size, &aivParamCtx));
        } else {
            HCCL_WARNING("[GetAivParamStorageByComm] Call HcclEngineCtxGet failed.");
            return HCCL_E_PARA;
        }
    }

    *aivParam = static_cast<AivParamStorage*>(aivParamCtx);

    return HCCL_SUCCESS;
}

HcclResult GetAivParamStorage(const char* group, AivParamStorage** aivParam)
{
    if (group == nullptr || aivParam == nullptr) {
        HCCL_ERROR("[GetAivParamStorage] Invalid parameters");
        return HCCL_E_PARA;
    }

    HcclComm comm = nullptr;
    CHK_RET(HcomGetCommHandleByGroup(group, &comm));

    return GetAivParamStorageByComm(comm, aivParam, true);
}

template <typename...>
using VoidT = void;

template <typename T, typename = void>
struct HasSplitRatioConfigType : std::false_type {};

template <typename T>
struct HasSplitRatioConfigType<T, VoidT<decltype(T::HCCL_CONFIG_TYPE_MULTIPLE_DIMENSION_SPLIT_RATIO)>> :
    std::true_type {};

HcclResult QuerySplitRatioByConfigGetInfo(HcclComm comm, HcclConfigType cfgType, double& ratio, bool& isConfigured)
{
    ratio = 0.0;
    isConfigured = false;
    auto& hcommFunction = ops_hccl::DlHcommFunction::GetInstance();
    if (!hcommFunction.dlHcclConfigGetInfo) {
        HCCL_INFO("[QuerySplitRatioByConfigGetInfo] HcclConfigGetInfo is not supported, skip comm config.");
        return HCCL_SUCCESS;
    }
    double commRatio = 0.0;
    const uint32_t infoLen = sizeof(commRatio);
    HcclResult ret = hcommFunction.dlHcclConfigGetInfo(comm, cfgType, infoLen, &commRatio);
    if (ret == HCCL_E_NOT_SUPPORT) {
        HCCL_INFO("[QuerySplitRatioByConfigGetInfo] comm config not set or not supported, ret[%d].", ret);
        return HCCL_SUCCESS;
    }
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[QuerySplitRatioByConfigGetInfo] HcclConfigGetInfo failed, ret[%d].", ret);
        return ret;
    }
    if (!std::isfinite(commRatio) || commRatio < 0.0 || commRatio > 1.0) {
        HCCL_ERROR("[QuerySplitRatioByConfigGetInfo] comm ratio[%f] is not finite or out of range[0, 1].", commRatio);
        return HCCL_E_PARA;
    }
    if (IsDoubleEqual(commRatio, 0.0)) {
        HCCL_INFO("[QuerySplitRatioByConfigGetInfo] comm split ratio is not configured.");
        return HCCL_SUCCESS;
    }
    ratio = commRatio;
    isConfigured = true;
    HCCL_INFO("[QuerySplitRatioByConfigGetInfo] comm ratio[%f] is configured.", commRatio);
    return HCCL_SUCCESS;
}

template <typename ConfigType>
HcclResult QueryCommSplitRatio(HcclComm comm, double& ratio, bool& isConfigured, std::false_type)
{
    HCCL_INFO("[QueryCommSplitRatio] Current Hcomm headers do not support split ratio config, skip comm config.");
    ratio = 0.0;
    isConfigured = false;
    return HCCL_SUCCESS;
}

template <typename ConfigType>
HcclResult QueryCommSplitRatio(HcclComm comm, double& ratio, bool& isConfigured, std::true_type)
{
    return QuerySplitRatioByConfigGetInfo(
        comm, ConfigType::HCCL_CONFIG_TYPE_MULTIPLE_DIMENSION_SPLIT_RATIO, ratio, isConfigured);
}

HcclResult GetCommMultipleDimensionSplitRatio(HcclComm comm, double& ratio, bool& isConfigured)
{
    return QueryCommSplitRatio<HcclConfigType>(comm, ratio, isConfigured, HasSplitRatioConfigType<HcclConfigType>{});
}

HcclResult SetMultipleDimensionSplitRatio(HcclComm comm, OpParam& param)
{
    constexpr double defaultRatio = 0.5;

    double commRatio = 0.0;
    bool isCommConfigured = false;
    HcclResult ret = GetCommMultipleDimensionSplitRatio(comm, commRatio, isCommConfigured);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    if (isCommConfigured) {
        param.opConfig.multipleDimensionSplitRatio = commRatio;
        param.opConfig.multipleDimensionSplitRatioSource = MultipleDimensionSplitRatioSource::COMM_CONFIG;
        HCCL_INFO("[SetMultipleDimensionSplitRatio] ratioSource[COMM_CONFIG], configuredRatio[%f]", commRatio);
        return HCCL_SUCCESS;
    }

    double envRatio = 0.0;
    if (GetExternalInputMultipleDimensionSplitRatio(envRatio)) {
        if (!std::isfinite(envRatio) || envRatio < 0.0 || envRatio > 1.0) {
            HCCL_WARNING(
                "[SetMultipleDimensionSplitRatio] env ratio[%f] is out of range, use default ratio[%f]", envRatio,
                defaultRatio);
        } else {
            param.opConfig.multipleDimensionSplitRatio = envRatio;
            param.opConfig.multipleDimensionSplitRatioSource = MultipleDimensionSplitRatioSource::ENV_CONFIG;
            HCCL_INFO("[SetMultipleDimensionSplitRatio] ratioSource[ENV_CONFIG], configuredRatio[%f]", envRatio);
            return HCCL_SUCCESS;
        }
    }

    param.opConfig.multipleDimensionSplitRatio = defaultRatio;
    param.opConfig.multipleDimensionSplitRatioSource = MultipleDimensionSplitRatioSource::BUILTIN_FORMULA;
    HCCL_INFO("[SetMultipleDimensionSplitRatio] ratioSource[BUILTIN_FORMULA], configuredRatio[%f]", defaultRatio);
    return HCCL_SUCCESS;
}

// 判断通过最高一个level的网络全部没有device的可达链路，并且有host的可达链路
HcclResult CheckHostDPUOnly(const HcclComm comm, const TopoInfoWithNetLayerDetails* topoInfo, bool& hostDPUOnly)
{
    hostDPUOnly = false;
    HCCL_INFO("Start CheckHostDPUOnly");
    // 只有一个server，不使用DPU
    if (topoInfo->serverNum == 1) {
        HCCL_INFO("Not using hostdpu because serverNum is 1");
        return HCCL_SUCCESS;
    }

    // 只有一层topo，不使用DPU
    if (topoInfo->topoLevelNums == 1) {
        HCCL_INFO("Not using hostdpu because topoLevelNums is 1");
        return HCCL_SUCCESS;
    }

    uint32_t* netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    if ((netLayers == nullptr) || (netLayerNum == 0)) {
        HCCL_WARNING("HcclRankGraphGetLayers fail");
        return HCCL_E_INTERNAL;
    }

    bool hostDPU = false;
    for (uint32_t layerIdx = 0; layerIdx < netLayerNum; layerIdx++) {
        uint32_t netLayer = netLayers[layerIdx];
        // 只校验最后一个level
        if (netLayer < (topoInfo->topoLevelNums - 1)) {
            HCCL_INFO("Skip checking layer[%u], topoLevelNums is [%u]", netLayer, topoInfo->topoLevelNums);
            continue;
        }
        uint32_t* topoInsts = nullptr;
        uint32_t topoInsNum = 0;
        CHK_RET(HcclRankGraphGetTopoInstsByLayer(comm, netLayer, &topoInsts, &topoInsNum));
        if ((topoInsts == nullptr) || (topoInsNum == 0)) {
            HCCL_WARNING("HcclRankGraphGetTopoInstsByLayer fail, netLayer[%u]", netLayer);
            return HCCL_E_INTERNAL;
        }
        for (uint32_t topoInsIdx = 0; topoInsIdx < topoInsNum; topoInsIdx++) {
            uint32_t topoInstId = topoInsts[topoInsIdx];
            HCCL_INFO("Start checking topoInstId[%u]", topoInstId);
            CommTopo topoType;
            CHK_RET(HcclRankGraphGetTopoType(comm, netLayer, topoInstId, &topoType));
            if (topoType != COMM_TOPO_CLOS) {
                HCCL_INFO("Not using hostdpu because topo type is not COMM_TOPO_CLOS");
                continue;
            }
            uint32_t* ranks = nullptr;
            uint32_t rankNum = 0;
            CHK_RET(HcclRankGraphGetRanksByTopoInst(comm, netLayer, topoInstId, &ranks, &rankNum));
            // 校验当前rank与其他所有rank连通
            if (rankNum != topoInfo->userRankSize) {
                HCCL_INFO("Not using hostdpu because current rank is not fully connected to all other ranks");
                continue;
            }
            uint32_t endPointNums = 0;
            CHK_RET(HcclRankGraphGetEndpointNum(comm, netLayer, topoInstId, &endPointNums));
            EndpointDesc endPointDescs[endPointNums];
            CHK_RET(HcclRankGraphGetEndpointDesc(comm, netLayer, topoInstId, &endPointNums, endPointDescs));
            for (uint32_t endPointIdx = 0; endPointIdx < endPointNums; endPointIdx++) {
                EndpointDesc endPointDesc = endPointDescs[endPointIdx];
                if (endPointDesc.loc.locType == ENDPOINT_LOC_TYPE_DEVICE) {
                    HCCL_INFO(
                        "Not using hostdpu because there is links on device in netLayer[%u] in endPointIdx[%u]",
                        netLayer, endPointIdx);
                    return HCCL_SUCCESS;
                } else if (endPointDesc.loc.locType == ENDPOINT_LOC_TYPE_HOST) {
                    HCCL_INFO("Found a host endPoint in netLayer[%u] endPointIdx[%u]", netLayer, endPointIdx);
                    hostDPU = true;
                }
            }
        }
    }
    if (hostDPU) {
        HCCL_INFO("Using host dpu trans.");
        hostDPUOnly = true;
    }
    return HCCL_SUCCESS;
}

// 设置执行超时时间
HcclResult SetExecTimeout(OpParam& param)
{
    double execTimeoutValue = 0;
    if (!GetExternalInputExecTimeout(execTimeoutValue)) {
        param.opConfig.execTimeout = CUSTOM_TIMEOUT;
        HCCL_INFO("[OpCommon] Exec timeout is not set, use default value: %u seconds", CUSTOM_TIMEOUT);
    } else {
        // 验证转换后的值是否合理
        if (execTimeoutValue < 0 || execTimeoutValue > UINT32_MAX) {
            HCCL_WARNING(
                "[OpCommon] Exec timeout value %.2f s out of range, use default: %u seconds", execTimeoutValue,
                CUSTOM_TIMEOUT);
            param.opConfig.execTimeout = CUSTOM_TIMEOUT;
        } else {
            param.opConfig.execTimeout = static_cast<uint32_t>(execTimeoutValue);
            if (param.opConfig.execTimeout == 0) {
                HCCL_INFO("[OpCommon] Exec timeout is disabled (never timeout).");
            } else {
                HCCL_INFO("[OpCommon] Set exec timeout to: %u seconds", param.opConfig.execTimeout);
            }
        }
    }
    return HCCL_SUCCESS;
}

bool IsHostDpu(HcclComm comm)
{
    HcclResult ret;
    bool hostDpuOnly = false;

    HcclDevType deviceType = HcclDevType::DEV_TYPE_COUNT;
    ret = HcclGetDeviceType(deviceType);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[IsHostDpu]HcclGetDeviceType fail, ret:%d", ret);
        return false;
    }
    if (deviceType != HcclDevType::DEV_TYPE_910B) {
        return false;
    }

    uint32_t* level0SizeList = nullptr;
    uint32_t level0RankListNum = 0;
    ret = HcclRankGraphGetInstSizeListByLayer(
        comm, static_cast<uint32_t>(HcclNetLayer::HCCL_NetLayer_L0), &level0SizeList, &level0RankListNum);
    if (ret != HCCL_SUCCESS) {
        return false;
    }

    // 获取 rankSize
    u32 rankSize = 0;
    ret = HcclGetRankSize(comm, &rankSize);
    if (ret != HCCL_SUCCESS) {
        return false;
    }

    // 获取 topoLevelNums
    uint32_t* netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    if (ret != HCCL_SUCCESS) {
        return false;
    }

    TopoInfoWithNetLayerDetails topoInfo;
    topoInfo.serverNum = level0RankListNum;
    topoInfo.topoLevelNums = netLayerNum;
    topoInfo.userRankSize = rankSize;
    ret = CheckHostDPUOnly(comm, &topoInfo, hostDpuOnly);
    if (ret == HCCL_SUCCESS && hostDpuOnly) {
        return true;
    }
    return false;
}

// 判定当前通信域是否为「框间 host-DPU」场景。逻辑与 IsHostDpu 一致，
// 但不限定 910B —— 950 Barrier 新流程（框内 AICPU + 框间 DPU）仅在该场景启用。
bool IsBarrierHostDpu(HcclComm comm)
{
    HcclResult ret;
    bool hostDpuOnly = false;

    uint32_t* level0SizeList = nullptr;
    uint32_t level0RankListNum = 0;
    ret = HcclRankGraphGetInstSizeListByLayer(
        comm, static_cast<uint32_t>(HcclNetLayer::HCCL_NetLayer_L0), &level0SizeList, &level0RankListNum);
    if (ret != HCCL_SUCCESS) {
        return false;
    }

    u32 rankSize = 0;
    ret = HcclGetRankSize(comm, &rankSize);
    if (ret != HCCL_SUCCESS) {
        return false;
    }

    uint32_t* netLayers = nullptr;
    uint32_t netLayerNum = 0;
    ret = HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum);
    if (ret != HCCL_SUCCESS) {
        return false;
    }

    TopoInfoWithNetLayerDetails topoInfo;
    topoInfo.serverNum = level0RankListNum;
    topoInfo.topoLevelNums = netLayerNum;
    topoInfo.userRankSize = rankSize;
    ret = CheckHostDPUOnly(comm, &topoInfo, hostDpuOnly);
    if (ret == HCCL_SUCCESS && hostDpuOnly) {
        return true;
    }
    return false;
}

void CheckAndSetSymmetricMemory(OpParam& param)
{
    size_t inputOffset = 0;
    size_t outputOffset = 0;

    if (param.inputPtr != nullptr && param.inputSize > 0) {
        HcclResult ret
            = HcclCommSymWinGet(param.hcclComm, param.inputPtr, param.inputSize, &param.inputSymWindow, &inputOffset);
        if (ret != HCCL_SUCCESS || param.inputSymWindow == nullptr) {
            HCCL_INFO(
                "[%s] input[%p], size[%llu] is not support symmetric memory, ret[%d]", __func__, param.inputPtr,
                param.inputSize, ret);
            return;
        }
    }

    if (param.outputPtr != nullptr && param.outputSize > 0) {
        HcclResult ret = HcclCommSymWinGet(
            param.hcclComm, param.outputPtr, param.outputSize, &param.outputSymWindow, &outputOffset);
        if (ret != HCCL_SUCCESS || param.outputSymWindow == nullptr) {
            HCCL_INFO(
                "[%s] output[%p], size[%llu] is not support symmetric memory, ret[%d]", __func__, param.outputPtr,
                param.outputSize, ret);
            return;
        }
    }
    param.supportSymmetricMemory = true;
    param.inputOffset = inputOffset;
    param.outputOffset = outputOffset;
    return;
}
} // namespace ops_hccl
