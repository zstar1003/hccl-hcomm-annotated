/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <atomic>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <vector>
#include <string>
#include "hccl/hccl_res.h"
#include "stream_pub.h"
#include "hccl_comm_pub.h"
#include "hccl_independent_common.h"
#include "coll_comm_profiling.h"
#include "comm_engine_utils.h"
#include "coll_comm_mgr.h"
#include "orion_adapter_rts.h"
#include "hccl_common.h"
#include "dfx_dlprof_function.h"
#include "adapter_rts.h"
#include "hcclCommOp.h"
using namespace hccl;
constexpr u32 MAX_EXPORT_THREAD_NUM = 40U;
static const std::unordered_set<HcclDedicatedThreadType> ORDER_LAUNCH_TYPES = {
    HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_OPBASE,
    HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_ACLGRAPH,
    HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_GE,
    HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE,
};

// [中文导读] [AllReduce逐行 S38] HcclGetNotifyNumInThread的接口声明：通信域句柄、当前执行Thread句柄、请求的通信引擎、通知槽数量；这些参数属于本函数调用边界。
HcclResult HcclGetNotifyNumInThread(HcclComm comm, ThreadHandle thread, CommEngine engine, uint32_t* notifyNum)
// [中文导读] [AllReduce逐行 S39] 进入HcclGetNotifyNumInThread函数体：查询指定引擎 Thread 的通知槽容量，新域与兼容域分别进入资源管理器。
{
    // [中文导读] [AllReduce逐行 S40] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
    CHK_PRT_RET(comm == nullptr, HCCL_ERROR("[%s] comm is null", __func__), HCCL_E_PTR);
    // [中文导读] [AllReduce逐行 S41] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S42] 向条件错误检查提供`!IsValidCommEngine(engine),`（请求的通信引擎），用于确定触发条件或形成对应诊断。
        !IsValidCommEngine(engine),
        // [中文导读] [AllReduce逐行 S43] 记录HcclGetNotifyNumInThread的错误诊断；日志本身不执行传输。
        HCCL_ERROR(
            // [中文导读] [AllReduce逐行 S44] 为当前HcclGetNotifyNumInThread诊断/异常表达式提供格式文本，将报告请求的通信引擎；这一物理行没有数据搬运副作用。
            "[%s] commEngine[%s] is invalid", __func__, GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str()),
        // [中文导读] [AllReduce逐行 S45] 记录HcclGetNotifyNumInThread的状态/性能诊断；日志本身不执行传输。
        HCCL_E_PARA);
    // [中文导读] [AllReduce逐行 S46] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
    CHK_PRT_RET(notifyNum == nullptr, HCCL_ERROR("[%s] notifyNum is null", __func__), HCCL_E_PTR);

    // [中文导读] [AllReduce逐行 S48] 设置域的兼容外层对象为/按`static_cast<hccl::hcclComm*>(comm)`（域的兼容外层对象、通信域句柄）。
    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    // [中文导读] [AllReduce逐行 S49] 设置通信域名称为/按`hcclComm->GetIdentifier()`（域的兼容外层对象的GetIdentifier字段）；取得通信域标识字符串。
    std::string commId = hcclComm->GetIdentifier();
    // [中文导读] [AllReduce逐行 S50] 记录HcclGetNotifyNumInThread的状态/性能诊断；日志本身不执行传输。
    HCCL_RUN_INFO(
        // [中文导读] [AllReduce逐行 S51] 为当前HcclGetNotifyNumInThread诊断/异常表达式提供格式文本，将报告通信域名称的c_str字段；这一物理行没有数据搬运副作用。
        "Entry-%s:comm[%s] engine[%s]", __func__, commId.c_str(),
        // [中文导读] [AllReduce逐行 S52] 为把枚举转换成诊断名称补入`GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str())`（请求的通信引擎）；本行是参数/结构化初始化续行。
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
    // [中文导读] [AllReduce逐行 S53] 设置当前调用状态为/按`HCCL_SUCCESS`。
    HcclResult ret = HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S54] 仅当`(hcclComm->IsCommunicatorV2())`（域的兼容外层对象的IsCommunicatorV2字段）成立时进入此分支；判断通信域是否使用V2对象实现。
    if (hcclComm->IsCommunicatorV2()) {
        // [中文导读] [AllReduce逐行 S55] 设置V2通信域对象为/按`hcclComm->GetCollComm()`（域的兼容外层对象的GetCollComm字段）；从域外层对象取得V2通信域。
        hccl::CollComm* collComm = hcclComm->GetCollComm();
        // [中文导读] [AllReduce逐行 S56] 检查`collComm`（V2通信域对象）不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_PTR_NULL(collComm);
        // [中文导读] [AllReduce逐行 S57] 设置CommEngineResMgr* engineResMgr为/按`collComm->GetCommEngineResMgr()`（V2通信域对象的GetCommEngineResMgr字段）；取得域引擎资源管理器。
        CommEngineResMgr* engineResMgr = collComm->GetCommEngineResMgr();
        // [中文导读] [AllReduce逐行 S58] 检查`engineResMgr`不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_PTR_NULL(engineResMgr);
        // [中文导读] [AllReduce逐行 S59] 设置当前调用状态为/按`engineResMgr->HcclGetNotifyNumInThread(thread, engine, notifyNum)`（当前执行Thread句柄、请求的通信引擎、通知槽数量）；查询所选引擎Thread的通知资源数量。
        ret = engineResMgr->HcclGetNotifyNumInThread(thread, engine, notifyNum);
    // [中文导读] [AllReduce逐行 S60] 进入上一个条件未命中的替代路径，按本函数的资源/设备/协议分流继续处理。
    } else {
        // [中文导读] [AllReduce逐行 S61] 设置auto& engineResMgr为/按`hcclComm->GetIndependentOp().GetCommEngineResMgr()`（域的兼容外层对象的GetIndependentOp字段）；取得旧兼容域的独立算子资源入口；取得域引擎资源管理器。
        auto& engineResMgr = hcclComm->GetIndependentOp().GetCommEngineResMgr();
        // [中文导读] [AllReduce逐行 S62] 设置当前调用状态为/按`engineResMgr.HcclGetNotifyNumInThread(thread, engine, notifyNum)`（当前执行Thread句柄、请求的通信引擎、通知槽数量）；查询所选引擎Thread的通知资源数量。
        ret = engineResMgr.HcclGetNotifyNumInThread(thread, engine, notifyNum);
    // [中文导读] [AllReduce逐行 S63] 结束`if (hcclComm->IsCommunicatorV2())`（域的兼容外层对象的IsCommunicatorV2字段）分支/循环；控制流返回外层。
    }

    // [中文导读] [AllReduce逐行 S65] 仅当`(ret != HCCL_SUCCESS)`（当前调用状态）成立时进入此分支。
    if (ret != HCCL_SUCCESS) {
        // [中文导读] [AllReduce逐行 S66] 记录HcclGetNotifyNumInThread的错误诊断；日志本身不执行传输。
        HCCL_ERROR(
            // [中文导读] [AllReduce逐行 S67] 为当前HcclGetNotifyNumInThread诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
            "[HcclGetNotifyNumInThread] Failed to get notifyNum for engine[%s] ret[%d]",
            // [中文导读] [AllReduce逐行 S68] 为把枚举转换成诊断名称补入`GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), ret)`（请求的通信引擎、当前调用状态）；本行是参数/结构化初始化续行。
            GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), ret);
        // [中文导读] [AllReduce逐行 S69] 返回ret，表示保留当前调用的状态（可能成功或失败）；此路径停止本函数的后续处理。
        return ret;
    // [中文导读] [AllReduce逐行 S70] 结束`if (ret != HCCL_SUCCESS)`（当前调用状态）分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S71] 记录HcclGetNotifyNumInThread的状态/性能诊断；日志本身不执行传输。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S72] 为当前HcclGetNotifyNumInThread诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
        "[HcclGetNotifyNumInThread] threads for engine[%s], notifyNum[%u]",
        // [中文导读] [AllReduce逐行 S73] 为把枚举转换成诊断名称补入`GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), *notifyNum)`（请求的通信引擎、通知槽数量）；本行是参数/结构化初始化续行。
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), *notifyNum);
    // [中文导读] [AllReduce逐行 S74] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S75] 结束HcclGetNotifyNumInThread函数体；控制流返回外层。
}

// [中文导读] [AllReduce逐行 S77] HcclThreadAcquireWithConfigDfx的接口声明：按 AICPU/其他引擎上报申请信息或注册 Thread 的 DFX 回调；这些参数属于本函数调用边界。
HcclResult HcclThreadAcquireWithConfigDfx(
    // [中文导读] [AllReduce逐行 S78] HcclThreadAcquireWithConfigDfx的接口声明：V2通信域对象、通信域名称、请求的通信引擎、性能观测起始时间戳、执行Thread条数；这些参数属于本函数调用边界。
    hccl::CollComm* collComm, const std::string& commId, CommEngine engine, u64 beginTime, uint32_t threadNum,
    // [中文导读] [AllReduce逐行 S79] HcclThreadAcquireWithConfigDfx的接口声明：执行Thread句柄数组、实际执行流ID数组；这些参数属于本函数调用边界。
    ThreadHandle* threads, std::vector<uint32_t>& threadId)
// [中文导读] [AllReduce逐行 S80] 进入HcclThreadAcquireWithConfigDfx函数体：按 AICPU/其他引擎上报申请信息或注册 Thread 的 DFX 回调。
{
    // [中文导读] [AllReduce逐行 S81] 检查`threads`（执行Thread句柄数组）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(threads);
    // [中文导读] [AllReduce逐行 S82] 设置HcclCommDfx* hcclCommDfx为/按`collComm->GetHcclCommDfx()`（V2通信域对象的GetHcclCommDfx字段）；调用GetHcclCommDfx，使用V2通信域对象的GetHcclCommDfx字段。
    HcclCommDfx* hcclCommDfx = collComm->GetHcclCommDfx();
    // [中文导读] [AllReduce逐行 S83] 检查`hcclCommDfx`不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(hcclCommDfx);
    // [中文导读] AICPU 记录实际流 ID、域与 Rank 信息并上报初始化 Kernel，其他引擎逐 Thread 绑定 DFX 回调。
    // [中文导读] [AllReduce逐行 S85] 仅当`(engine == CommEngine::COMM_ENGINE_AICPU)`（请求的通信引擎）成立时进入此分支。
    if (engine == CommEngine::COMM_ENGINE_AICPU) {
        // [中文导读] [AllReduce逐行 S86] 准备流/域/Rank观测描述的局部存储/结构描述，初始化方式以本行声明为准。
        Mc2CommInfo mc2CommInfo;
        // [中文导读] [AllReduce逐行 S87] 设置流/域/Rank观测描述的FreeStreamId字段为/按`0`。
        mc2CommInfo.FreeStreamId = 0;
        // [中文导读] [AllReduce逐行 S88] 设置流/域/Rank观测描述的streamsId字段为/按`threadId`（实际执行流ID数组）。
        mc2CommInfo.streamsId = threadId;
        // [中文导读] [AllReduce逐行 S89] 设置流/域/Rank观测描述的groupname字段为/按`commId`（通信域名称）。
        mc2CommInfo.groupname = commId;
        // [中文导读] [AllReduce逐行 S90] 设置流/域/Rank观测描述的myRankId字段为/按`collComm->GetMyRankId()`（V2通信域对象的GetMyRankId字段）；调用GetMyRankId，使用V2通信域对象的GetMyRankId字段。
        mc2CommInfo.myRankId = collComm->GetMyRankId();
        // [中文导读] [AllReduce逐行 S91] 设置流/域/Rank观测描述的rankSize字段为/按`collComm->GetRankSize()`（V2通信域对象的GetRankSize字段）；取得域Rank总数。
        mc2CommInfo.rankSize = collComm->GetRankSize();
        // [中文导读] [AllReduce逐行 S92] 取得父域Rank编号供观测关联；返回非成功时由检查宏立即向上传递。
        CHK_RET(collComm->GetParentRankId(mc2CommInfo.parentRankId));
        // [中文导读] [AllReduce逐行 S93] 上报域/Rank/流关联观测信息；传入/处理流/域/Rank观测描述。
        hcclCommDfx->ReportMc2CommInfo(mc2CommInfo);
        // [中文导读] [AllReduce逐行 S94] 记录HcclThreadAcquireWithConfigDfx的状态/性能诊断；日志本身不执行传输。
        HCCL_INFO("[HcclThreadAcquireWithConfigDfx] ReportThreadAcquireKernel begin");
        // [中文导读] [AllReduce逐行 S95] 设置const std::string KernelName为/按`"RunAicpuIndOpThreadInit"`。
        const std::string KernelName = "RunAicpuIndOpThreadInit";
        // 这个地方获取不到当前是单算子还是图模式，所以全部都不保存
        // [中文导读] [AllReduce逐行 S97] 上报初始化Kernel的观测信息；取得调用线程ID用于观测；返回非成功时由检查宏立即向上传递。
        CHK_RET(hcclCommDfx->ReportKernel(beginTime, commId, KernelName, SalGetTid(), false));
        // [中文导读] [AllReduce逐行 S98] 记录HcclThreadAcquireWithConfigDfx的状态/性能诊断；日志本身不执行传输。
        HCCL_INFO("[HcclThreadAcquireWithConfigDfx] ReportThreadAcquireKernel success");
    // [中文导读] [AllReduce逐行 S99] 进入上一个条件未命中的替代路径，按本函数的资源/设备/协议分流继续处理。
    } else {
        // [中文导读] [AllReduce逐行 S100] 设置auto hcclCommDfxCallBack为/按`collComm->GetDfxCallback()`（V2通信域对象的GetDfxCallback字段）；调用GetDfxCallback，使用V2通信域对象的GetDfxCallback字段。
        auto hcclCommDfxCallBack = collComm->GetDfxCallback();
        // [中文导读] [AllReduce逐行 S101] 按`(u32 num = 0; num < threadNum; ++num)`（执行Thread条数）遍历本批条目/分片；各次处理保持数组对应关系。
        for (u32 num = 0; num < threadNum; ++num) {
            // [中文导读] [AllReduce逐行 S102] 设置当前调用状态为/按`HcommThreadRegisterDfx(threads[num], hcclCommDfxCallBack)`（执行Thread句柄数组）；给非AICPU执行Thread注册观测回调。
            int ret = HcommThreadRegisterDfx(threads[num], hcclCommDfxCallBack);
            // [中文导读] [AllReduce逐行 S103] 仅当`(ret != HCCL_SUCCESS)`（当前调用状态）成立时进入此分支。
            if (ret != HCCL_SUCCESS) {
                // [中文导读] [AllReduce逐行 S104] 记录HcclThreadAcquireWithConfigDfx的错误诊断；日志本身不执行传输。
                HCCL_ERROR(
                    // [中文导读] [AllReduce逐行 S105] 为当前HcclThreadAcquireWithConfigDfx诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
                    "[HcclThreadAcquireWithConfigDfx] ReportThreadAcquireKernel HcommThreadRegisterDfx failed"
                    // [中文导读] [AllReduce逐行 S106] 为当前HcclThreadAcquireWithConfigDfx诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
                    " ret:[%d], num:[%u]",
                    // [中文导读] [AllReduce逐行 S107] 为前述多行表达式补入`ret, num)`（当前调用状态）；本行是参数/结构化初始化续行。
                    ret, num);
                // [中文导读] [AllReduce逐行 S108] 返回HCCL_E_INTERNAL，表示内部处理失败；此路径停止本函数的后续处理。
                return HCCL_E_INTERNAL;
            // [中文导读] [AllReduce逐行 S109] 结束`if (ret != HCCL_SUCCESS)`（当前调用状态）分支/循环；控制流返回外层。
            }
        // [中文导读] [AllReduce逐行 S110] 结束`for (u32 num = 0; num < threadNum; ++num)`（执行Thread条数）分支/循环；控制流返回外层。
        }
    // [中文导读] [AllReduce逐行 S111] 结束`if (engine == CommEngine::COMM_ENGINE_AICPU)`（请求的通信引擎）分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S112] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S113] 结束HcclThreadAcquireWithConfigDfx函数体；控制流返回外层。
}

HcclResult
ValidateThreadAcquireParams(CommEngine engine, ThreadType type, const ThreadConfig* config, uint32_t threadNum)
{
    CHK_PRT_RET(
        type == THREAD_TYPE_INVALID,
        HCCL_ERROR("[%s] thread type[%d] is invalid", __func__, static_cast<int32_t>(type)), HCCL_E_PARA);
    CHK_PRT_RET(
        !IsValidCommEngine(engine),
        HCCL_ERROR("[%s] commEngine[%d] is invalid", __func__, static_cast<int32_t>(engine)), HCCL_E_PARA);
    CHK_PRT_RET(threadNum == 0, HCCL_ERROR("[%s] threadNum[%u] is invalid", __func__, threadNum), HCCL_E_PARA);
    CHK_PRT_RET(config == nullptr, HCCL_ERROR("[%s] config is null", __func__), HCCL_E_PTR);
    // [中文导读] 逐项检查 ThreadConfig 的 ABI 魔数，要求调用方先初始化配置数组，避免把未初始化字段当成配置。
    for (uint32_t i = 0; i < threadNum; ++i) {
        CHK_PRT_RET(
            config[i].header.magicWord != HCOMM_THREAD_CONFIG_MAGIC_WORD,
            HCCL_ERROR(
                "[%s] config[%u] magicWord[0x%x] mismatch, expected[0x%x], call ThreadConfigInit first", __func__, i,
                config[i].header.magicWord, HCOMM_THREAD_CONFIG_MAGIC_WORD),
            HCCL_E_PARA);
    }
    // [中文导读] 配置接口要求用 CPU/AICPU 加 ThreadType 表示执行方式，旧 TS 引擎枚举在这里被明确拒绝。
    CHK_PRT_RET(
        engine == CommEngine::COMM_ENGINE_AICPU_TS || engine == CommEngine::COMM_ENGINE_CPU_TS,
        HCCL_ERROR(
            "[%s] commEngine[%d] CPU_TS/AICPU_TS not supported, use CPU/AICPU engine with THREAD_TYPE_TS instead",
            __func__, static_cast<int32_t>(engine)),
        HCCL_E_PARA);
    CHK_PRT_RET(
        engine == CommEngine::COMM_ENGINE_AIV || engine == CommEngine::COMM_ENGINE_CCU,
        HCCL_ERROR(
            "[%s] commEngine[%d] AIV/CCU not supported, supported engines: CPU/AICPU", __func__,
            static_cast<int32_t>(engine)),
        HCCL_E_PARA);
    return HCCL_SUCCESS;
}

// [中文导读] 控制面按ThreadConfig数组申请执行Thread，每条Thread可有不同通知数；出参threads返回执行句柄。
// [中文导读] 此接口用CPU/AICPU配合THREAD_TYPE_TS表达TS线程，校验不接受旧CPU_TS/AICPU_TS枚举。
// [中文导读] 新通信域经CommEngineResMgr::HcclThreadAcquireV2管理资源，随后注册DFX信息；不是创建OS业务线程。
// [中文导读] [AllReduce逐行 S154] HcclThreadAcquireWithConfig的接口声明：校验显式ThreadConfig后申请执行Thread与通知槽，V2和兼容域分别管理资源；这些参数属于本函数调用边界。
HcclResult HcclThreadAcquireWithConfig(
    // [中文导读] [AllReduce逐行 S155] HcclThreadAcquireWithConfig的接口声明：通信域句柄、请求的通信引擎、执行Thread条数、Thread类型、Thread配置数组；这些参数属于本函数调用边界。
    HcclComm comm, CommEngine engine, uint32_t threadNum, ThreadType type, const ThreadConfig* config,
    // [中文导读] [AllReduce逐行 S156] HcclThreadAcquireWithConfig的接口声明：执行Thread句柄数组；这些参数属于本函数调用边界。
    ThreadHandle* threads)
// [中文导读] [AllReduce逐行 S157] 进入HcclThreadAcquireWithConfig函数体：校验显式ThreadConfig后申请执行Thread与通知槽，V2和兼容域分别管理资源。
{
    // [中文导读] [AllReduce逐行 S158] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
    CHK_PRT_RET(comm == nullptr, HCCL_ERROR("[%s] comm is null", __func__), HCCL_E_PTR);
    // [中文导读] [AllReduce逐行 S159] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
    CHK_PRT_RET(threads == nullptr, HCCL_ERROR("[%s] threads is null", __func__), HCCL_E_PTR);
    // [中文导读] 在调用资源管理器前完成整批请求校验；threadNum 同时决定配置项数和出参句柄数量。
    // [中文导读] [AllReduce逐行 S161] 调用ValidateThreadAcquireParams，使用请求的通信引擎、Thread类型、Thread配置数组、执行Thread条数；返回非成功时由检查宏立即向上传递。
    CHK_RET(ValidateThreadAcquireParams(engine, type, config, threadNum));

    // [中文导读] [AllReduce逐行 S163] 设置性能观测起始时间戳为/按`Hccl::DfxDlProfFunction::GetInstance().dlMsprofSysCycleTime()`；取得该管理器单例。
    u64 beginTime = Hccl::DfxDlProfFunction::GetInstance().dlMsprofSysCycleTime();
    // [中文导读] [AllReduce逐行 S164] 设置域的兼容外层对象为/按`static_cast<hccl::hcclComm*>(comm)`（域的兼容外层对象、通信域句柄）。
    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    // [中文导读] [AllReduce逐行 S165] 设置通信域名称为/按`hcclComm->GetIdentifier()`（域的兼容外层对象的GetIdentifier字段）；取得通信域标识字符串。
    std::string commId = hcclComm->GetIdentifier();
    // [中文导读] [AllReduce逐行 S166] 记录HcclThreadAcquireWithConfig的状态/性能诊断；日志本身不执行传输。
    HCCL_RUN_INFO(
        // [中文导读] [AllReduce逐行 S167] 为当前HcclThreadAcquireWithConfig诊断/异常表达式提供格式文本，将报告通信域名称的c_str字段；这一物理行没有数据搬运副作用。
        "Entry-%s:comm[%s] engine[%s] ThreadNum[%u].", __func__, commId.c_str(),
        // [中文导读] [AllReduce逐行 S168] 为把枚举转换成诊断名称补入`GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), threadNum)`（请求的通信引擎、执行Thread条数）；本行是参数/结构化初始化续行。
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), threadNum);

    // [中文导读] [AllReduce逐行 S170] 设置当前调用状态为/按`HCCL_SUCCESS`。
    HcclResult ret = HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S171] 准备实际执行流ID数组的局部存储/结构描述，初始化方式以本行声明为准。
    std::vector<uint32_t> threadId;
    // [中文导读] [AllReduce逐行 S172] 仅当`(hcclComm->IsCommunicatorV2())`（域的兼容外层对象的IsCommunicatorV2字段）成立时进入此分支；判断通信域是否使用V2对象实现。
    if (hcclComm->IsCommunicatorV2()) {
        // [中文导读] [AllReduce逐行 S173] 设置V2通信域对象为/按`hcclComm->GetCollComm()`（域的兼容外层对象的GetCollComm字段）；从域外层对象取得V2通信域。
        hccl::CollComm* collComm = hcclComm->GetCollComm();
        // [中文导读] [AllReduce逐行 S174] 检查`collComm`（V2通信域对象）不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_PTR_NULL(collComm);
        // [中文导读] [AllReduce逐行 S175] 设置CommEngineResMgr* engineResMgr为/按`collComm->GetCommEngineResMgr()`（V2通信域对象的GetCommEngineResMgr字段）；取得域引擎资源管理器。
        CommEngineResMgr* engineResMgr = collComm->GetCommEngineResMgr();
        // [中文导读] [AllReduce逐行 S176] 检查`engineResMgr`不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_PTR_NULL(engineResMgr);
        // [中文导读] V2 管理器返回执行句柄及用于观测的 threadId；资源申请成功后再注册 DFX。
        // [中文导读] [AllReduce逐行 S178] 设置当前调用状态为/按`engineResMgr->HcclThreadAcquireV2(engine, threadNum, type, config, threads, threadId)`（请求的通信引擎、执行Thread条数、Thread类型、Thread配置数组、执行Thread句柄数组、实际执行流ID数组）；按CPU/AICPU引擎、TS类型及配置申请域内Thread/通知资源。
        ret = engineResMgr->HcclThreadAcquireV2(engine, threadNum, type, config, threads, threadId);
        // [中文导读] [AllReduce逐行 S179] 仅当`(ret != HCCL_SUCCESS)`（当前调用状态）成立时进入此分支。
        if (ret != HCCL_SUCCESS) {
            // [中文导读] [AllReduce逐行 S180] 记录HcclThreadAcquireWithConfig的错误诊断；日志本身不执行传输。
            HCCL_ERROR(
                // [中文导读] [AllReduce逐行 S181] 为当前HcclThreadAcquireWithConfig诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
                "[%s] failed to create threads for engine[%s], threadsNum[%u], ret[%d].", __func__,
                // [中文导读] [AllReduce逐行 S182] 为把枚举转换成诊断名称补入`GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), threadNum, ret)`（请求的通信引擎、执行Thread条数、当前调用状态）；本行是参数/结构化初始化续行。
                GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), threadNum, ret);
            // [中文导读] [AllReduce逐行 S183] 返回ret，表示保留当前调用的状态（可能成功或失败）；此路径停止本函数的后续处理。
            return ret;
        // [中文导读] [AllReduce逐行 S184] 结束`if (ret != HCCL_SUCCESS)`（当前调用状态）分支/循环；控制流返回外层。
        }
        // [中文导读] [AllReduce逐行 S185] 注册新申请Thread的域/流观测信息或回调；返回非成功时由检查宏立即向上传递。
        CHK_RET(HcclThreadAcquireWithConfigDfx(collComm, commId, engine, beginTime, threadNum, threads, threadId));
        // [中文导读] [AllReduce逐行 S186] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S187] 进入上一个条件未命中的替代路径，按本函数的资源/设备/协议分流继续处理。
    } else {
        // [中文导读] [AllReduce逐行 S188] 设置auto& engineResMgr为/按`hcclComm->GetIndependentOp().GetCommEngineResMgr()`（域的兼容外层对象的GetIndependentOp字段）；取得旧兼容域的独立算子资源入口；取得域引擎资源管理器。
        auto& engineResMgr = hcclComm->GetIndependentOp().GetCommEngineResMgr();
        // [中文导读] [AllReduce逐行 S189] 设置当前调用状态为/按`engineResMgr.HcclThreadAcquire(engine, threadNum, type, config, threads, threadId)`（请求的通信引擎、执行Thread条数、Thread类型、Thread配置数组、执行Thread句柄数组、实际执行流ID数组）；进入兼容资源管理器申请Thread配置。
        ret = engineResMgr.HcclThreadAcquire(engine, threadNum, type, config, threads, threadId);
        // [中文导读] [AllReduce逐行 S190] 仅当`(engine == CommEngine::COMM_ENGINE_AICPU)`（请求的通信引擎）成立时进入此分支。
        if (engine == CommEngine::COMM_ENGINE_AICPU) {
            // 上报流
            // [中文导读] [AllReduce逐行 S192] 仅当`(threadNum != threadId.size())`（执行Thread条数、实际执行流ID数组的size字段）成立时进入此分支；读取容器登记项数。
            if (threadNum != threadId.size()) {
                // [中文导读] [AllReduce逐行 S193] 记录HcclThreadAcquireWithConfig的错误诊断，字段包含执行Thread条数、实际执行流ID数组的size字段；日志本身不执行传输。
                HCCL_ERROR("[%s] threadNum [%u] != threadId.size[%zu]", __func__, threadNum, threadId.size());
                // [中文导读] [AllReduce逐行 S194] 返回HCCL_E_PARA，表示参数不满足此分支要求；此路径停止本函数的后续处理。
                return HCCL_E_PARA;
            // [中文导读] [AllReduce逐行 S195] 结束`if (threadNum != threadId.size())`（执行Thread条数、实际执行流ID数组的size字段）分支/循环；控制流返回外层。
            }
            // [中文导读] [AllReduce逐行 S196] 兼容AICPU申请后上报实际流ID；返回非成功时由检查宏立即向上传递。
            CHK_RET(HcclStreamProfilingReport(comm, threadNum, threadId.data()));
        // [中文导读] [AllReduce逐行 S197] 结束`if (engine == CommEngine::COMM_ENGINE_AICPU)`（请求的通信引擎）分支/循环；控制流返回外层。
        }
    // [中文导读] [AllReduce逐行 S198] 结束`if (hcclComm->IsCommunicatorV2())`（域的兼容外层对象的IsCommunicatorV2字段）分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S199] 仅当`(ret != HCCL_SUCCESS)`（当前调用状态）成立时进入此分支。
    if (ret != HCCL_SUCCESS) {
        // [中文导读] [AllReduce逐行 S200] 记录HcclThreadAcquireWithConfig的错误诊断；日志本身不执行传输。
        HCCL_ERROR(
            // [中文导读] [AllReduce逐行 S201] 为当前HcclThreadAcquireWithConfig诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
            "[%s] failed to create threads for engine[%s], threadsNum[%u], ret[%d].", __func__,
            // [中文导读] [AllReduce逐行 S202] 为把枚举转换成诊断名称补入`GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), threadNum, ret)`（请求的通信引擎、执行Thread条数、当前调用状态）；本行是参数/结构化初始化续行。
            GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), threadNum, ret);
        // [中文导读] [AllReduce逐行 S203] 返回ret，表示保留当前调用的状态（可能成功或失败）；此路径停止本函数的后续处理。
        return ret;
    // [中文导读] [AllReduce逐行 S204] 结束`if (ret != HCCL_SUCCESS)`（当前调用状态）分支/循环；控制流返回外层。
    }

    // [中文导读] [AllReduce逐行 S206] 记录HcclThreadAcquireWithConfig的状态/性能诊断，字段包含执行Thread条数、请求的通信引擎；日志本身不执行传输。
    HCCL_INFO("[%s] Allocated %u threads for engine[%d]", __func__, threadNum, engine);
    // [中文导读] [AllReduce逐行 S207] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S208] 结束HcclThreadAcquireWithConfig函数体；控制流返回外层。
}

// [中文导读] [AllReduce逐行 S210] ConvertEngineToTsType的接口声明：请求的通信引擎；这些参数属于本函数调用边界。
static CommEngine ConvertEngineToTsType(CommEngine engine)
// [中文导读] [AllReduce逐行 S211] 进入ConvertEngineToTsType函数体：将 CPU_TS/AICPU_TS 转为配置接口使用的 CPU/AICPU 枚举。
{
    // [中文导读] [AllReduce逐行 S212] 仅当`(engine == COMM_ENGINE_CPU_TS)`（请求的通信引擎）成立时进入此分支。
    if (engine == COMM_ENGINE_CPU_TS) {
        // [中文导读] [AllReduce逐行 S213] 直接返回`COMM_ENGINE_CPU`；将当前查询结果/句柄交给调用者。
        return COMM_ENGINE_CPU;
    // [中文导读] [AllReduce逐行 S214] 结束`if (engine == COMM_ENGINE_CPU_TS)`（请求的通信引擎）分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S215] 仅当`(engine == COMM_ENGINE_AICPU_TS)`（请求的通信引擎）成立时进入此分支。
    if (engine == COMM_ENGINE_AICPU_TS) {
        // [中文导读] [AllReduce逐行 S216] 直接返回`COMM_ENGINE_AICPU`；将当前查询结果/句柄交给调用者。
        return COMM_ENGINE_AICPU;
    // [中文导读] [AllReduce逐行 S217] 结束`if (engine == COMM_ENGINE_AICPU_TS)`（请求的通信引擎）分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S218] 直接返回`engine`（请求的通信引擎）；将当前查询结果/句柄交给调用者。
    return engine;
// [中文导读] [AllReduce逐行 S219] 结束ConvertEngineToTsType函数体；控制流返回外层。
}

// [中文导读] 兼容形态的线程申请：所有Thread采用相同notifyNumPerThread，内部转换为ThreadConfig数组。
// [中文导读] CPU_TS/AICPU_TS分别映射为CPU/AICPU，并设置THREAD_TYPE_TS，再交域内Engine资源管理器。
// [中文导读] [AllReduce逐行 S223] HcclThreadAcquire的接口声明：把兼容 TS 枚举转换为 CPU/AICPU 配合 TS 配置，申请执行 Thread 与通知槽；这些参数属于本函数调用边界。
HcclResult HcclThreadAcquire(
    // [中文导读] [AllReduce逐行 S224] HcclThreadAcquire的接口声明：通信域句柄、请求的通信引擎、执行Thread条数、每个Thread的通知槽数、执行Thread句柄数组；这些参数属于本函数调用边界。
    HcclComm comm, CommEngine engine, uint32_t threadNum, uint32_t notifyNumPerThread, ThreadHandle* threads)
// [中文导读] [AllReduce逐行 S225] 进入HcclThreadAcquire函数体：把兼容 TS 枚举转换为 CPU/AICPU 配合 TS 配置，申请执行 Thread 与通知槽。
{
    // [中文导读] [AllReduce逐行 S226] 设置性能观测起始时间戳为/按`Hccl::DfxDlProfFunction::GetInstance().dlMsprofSysCycleTime()`；取得该管理器单例。
    u64 beginTime = Hccl::DfxDlProfFunction::GetInstance().dlMsprofSysCycleTime();
    // [中文导读] [AllReduce逐行 S227] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
    CHK_PRT_RET(comm == nullptr, HCCL_ERROR("[%s] comm is null", __func__), HCCL_E_PTR);
    // [中文导读] [AllReduce逐行 S228] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
    CHK_PRT_RET(threads == nullptr, HCCL_ERROR("[%s] threads is null", __func__), HCCL_E_PTR);
    // [中文导读] [AllReduce逐行 S229] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S230] 向条件错误检查提供`!IsValidCommEngine(engine),`（请求的通信引擎），用于确定触发条件或形成对应诊断。
        !IsValidCommEngine(engine),
        // [中文导读] [AllReduce逐行 S231] 记录HcclThreadAcquire的错误诊断，字段包含请求的通信引擎；日志本身不执行传输。
        HCCL_ERROR("[%s] commEngine[%d] is invalid", __func__, static_cast<int32_t>(engine)), HCCL_E_PARA);
    // [中文导读] [AllReduce逐行 S232] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
    CHK_PRT_RET(threadNum == 0, HCCL_ERROR("[%s] threadNum[%u] is invalid", __func__, threadNum), HCCL_E_PARA);

    // [中文导读] [AllReduce逐行 S234] 设置域的兼容外层对象为/按`static_cast<hccl::hcclComm*>(comm)`（域的兼容外层对象、通信域句柄）。
    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    // [中文导读] [AllReduce逐行 S235] 设置通信域名称为/按`hcclComm->GetIdentifier()`（域的兼容外层对象的GetIdentifier字段）；取得通信域标识字符串。
    std::string commId = hcclComm->GetIdentifier();
    // [中文导读] [AllReduce逐行 S236] 记录HcclThreadAcquire的状态/性能诊断；日志本身不执行传输。
    HCCL_RUN_INFO(
        // [中文导读] [AllReduce逐行 S237] 为当前HcclThreadAcquire诊断/异常表达式提供格式文本，将报告通信域名称的c_str字段、请求的通信引擎；这一物理行没有数据搬运副作用。
        "Entry-%s:comm[%s] engine[%u] ThreadNum[%u] notifyNumPerThread[%u]", __func__, commId.c_str(), engine,
        // [中文导读] [AllReduce逐行 S238] 为调用c_str，使用通信域名称的c_str字段、请求的通信引擎、执行Thread条数、每个Thread的通知槽数补入`threadNum, notifyNumPerThread)`（执行Thread条数、每个Thread的通知槽数）；本行是参数/结构化初始化续行。
        threadNum, notifyNumPerThread);

    // [中文导读] 把旧 TS 引擎转换为 CPU/AICPU，并为每条执行 Thread 建立 ABI 已初始化的 TS 配置。
    // [中文导读] [AllReduce逐行 S241] 设置转换后的CPU/AICPU引擎为/按`ConvertEngineToTsType(engine)`（请求的通信引擎）；将旧TS引擎枚举转为CPU/AICPU配置引擎。
    CommEngine newEngine = ConvertEngineToTsType(engine);
    // [中文导读] [AllReduce逐行 S242] 设置Thread类型为/按`THREAD_TYPE_TS`。
    ThreadType type = THREAD_TYPE_TS;
    // [中文导读] [AllReduce逐行 S243] 设置Thread配置数组为/按`std::make_unique<ThreadConfig[]>(threadNum)`（执行Thread条数）。
    std::unique_ptr<ThreadConfig[]> config = std::make_unique<ThreadConfig[]>(threadNum);
    // [中文导读] [AllReduce逐行 S244] 检查`config`（Thread配置数组）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(config);
    // [中文导读] [AllReduce逐行 S245] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S246] 向条件错误检查提供`ThreadConfigInit(config.get(), threadNum) != 0, HCCL_ERROR("[%s] ThreadConfigInit failed", __func__),`（Thread配置数组的get字段、执行Thread条数），用于确定触发条件或形成对应诊断。
        ThreadConfigInit(config.get(), threadNum) != 0, HCCL_ERROR("[%s] ThreadConfigInit failed", __func__),
        // [中文导读] [AllReduce逐行 S247] 记录HcclThreadAcquire的状态/性能诊断；日志本身不执行传输。
        HCCL_E_INTERNAL);
    // [中文导读] 在窄化为 uint16_t 前限制通知槽位数量；各配置项随后采用同一 notifyNumPerThread。
    // [中文导读] [AllReduce逐行 S249] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S250] 向条件错误检查提供`notifyNumPerThread >= HCCL_THREAD_NOTIFY_MAX_NUM,`（每个Thread的通知槽数），用于确定触发条件或形成对应诊断。
        notifyNumPerThread >= HCCL_THREAD_NOTIFY_MAX_NUM,
        // [中文导读] [AllReduce逐行 S251] 记录HcclThreadAcquire的错误诊断，字段包含每个Thread的通知槽数；日志本身不执行传输。
        HCCL_ERROR("[%s] notifyNumPerThread[%u] exceeds HCCL_THREAD_NOTIFY_MAX_NUM", __func__, notifyNumPerThread),
        // [中文导读] [AllReduce逐行 S252] 记录HcclThreadAcquire的状态/性能诊断；日志本身不执行传输。
        HCCL_E_PARA);
    // [中文导读] [AllReduce逐行 S253] 按`(u32 i = 0; i < threadNum; i++)`（本批条目下标、执行Thread条数）遍历本批条目/分片；各次处理保持数组对应关系。
    for (u32 i = 0; i < threadNum; i++) {
        // [中文导读] [AllReduce逐行 S254] 设置Thread配置数组、本批条目下标为/按`static_cast<uint16_t>(notifyNumPerThread)`（每个Thread的通知槽数）。
        config[i].notifyNumPerThread = static_cast<uint16_t>(notifyNumPerThread);
    // [中文导读] [AllReduce逐行 S255] 结束`for (u32 i = 0; i < threadNum; i++)`（本批条目下标、执行Thread条数）分支/循环；控制流返回外层。
    }

    // [中文导读] [AllReduce逐行 S257] 设置当前调用状态为/按`HCCL_SUCCESS`。
    HcclResult ret = HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S258] 准备实际执行流ID数组的局部存储/结构描述，初始化方式以本行声明为准。
    std::vector<uint32_t> threadId;
    // [中文导读] [AllReduce逐行 S259] 仅当`(hcclComm->IsCommunicatorV2())`（域的兼容外层对象的IsCommunicatorV2字段）成立时进入此分支；判断通信域是否使用V2对象实现。
    if (hcclComm->IsCommunicatorV2()) {
        // [中文导读] [AllReduce逐行 S260] 设置V2通信域对象为/按`hcclComm->GetCollComm()`（域的兼容外层对象的GetCollComm字段）；从域外层对象取得V2通信域。
        hccl::CollComm* collComm = hcclComm->GetCollComm();
        // [中文导读] [AllReduce逐行 S261] 检查`collComm`（V2通信域对象）不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_PTR_NULL(collComm);
        // [中文导读] [AllReduce逐行 S262] 设置CommEngineResMgr* engineResMgr为/按`collComm->GetCommEngineResMgr()`（V2通信域对象的GetCommEngineResMgr字段）；取得域引擎资源管理器。
        CommEngineResMgr* engineResMgr = collComm->GetCommEngineResMgr();
        // [中文导读] [AllReduce逐行 S263] 检查`engineResMgr`不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_PTR_NULL(engineResMgr);
        // [中文导读] V2 使用转换后的引擎和 TS 类型申请，原始枚举仅用于入口诊断。
        // [中文导读] [AllReduce逐行 S265] 设置当前调用状态为/按`engineResMgr->HcclThreadAcquireV2(newEngine, threadNum, type, config.get(), threads, threadId)`（转换后的CPU/AICPU引擎、执行Thread条数、Thread类型、Thread配置数组的get字段、执行Thread句柄数组、实际执行流ID数组）；按CPU/AICPU引擎、TS类型及配置申请域内Thread/通知资源。
        ret = engineResMgr->HcclThreadAcquireV2(newEngine, threadNum, type, config.get(), threads, threadId);
        // [中文导读] [AllReduce逐行 S266] 仅当`(ret != HCCL_SUCCESS)`（当前调用状态）成立时进入此分支。
        if (ret != HCCL_SUCCESS) {
            // [中文导读] [AllReduce逐行 S267] 记录HcclThreadAcquire的错误诊断；日志本身不执行传输。
            HCCL_ERROR(
                // [中文导读] [AllReduce逐行 S268] 为当前HcclThreadAcquire诊断/异常表达式提供格式文本，将报告转换后的CPU/AICPU引擎、执行Thread条数；这一物理行没有数据搬运副作用。
                "[%s] failed to create threads for engine[%d], threadsNum[%u], ret[%d]", __func__, newEngine, threadNum,
                // [中文导读] [AllReduce逐行 S269] 为前述多行表达式补入`ret)`（当前调用状态）；本行是参数/结构化初始化续行。
                ret);
            // [中文导读] [AllReduce逐行 S270] 返回ret，表示保留当前调用的状态（可能成功或失败）；此路径停止本函数的后续处理。
            return ret;
        // [中文导读] [AllReduce逐行 S271] 结束`if (ret != HCCL_SUCCESS)`（当前调用状态）分支/循环；控制流返回外层。
        }
        // [中文导读] [AllReduce逐行 S272] 注册新申请Thread的域/流观测信息或回调；返回非成功时由检查宏立即向上传递。
        CHK_RET(HcclThreadAcquireWithConfigDfx(collComm, commId, newEngine, beginTime, threadNum, threads, threadId));
    // [中文导读] [AllReduce逐行 S273] 进入上一个条件未命中的替代路径，按本函数的资源/设备/协议分流继续处理。
    } else {
        // [中文导读] [AllReduce逐行 S274] 设置auto& engineResMgr为/按`hcclComm->GetIndependentOp().GetCommEngineResMgr()`（域的兼容外层对象的GetIndependentOp字段）；取得旧兼容域的独立算子资源入口；取得域引擎资源管理器。
        auto& engineResMgr = hcclComm->GetIndependentOp().GetCommEngineResMgr();
        // [中文导读] [AllReduce逐行 S275] 设置当前调用状态为/按`engineResMgr.HcclThreadAcquire(newEngine, threadNum, type, config.get(), threads, threadId)`（转换后的CPU/AICPU引擎、执行Thread条数、Thread类型、Thread配置数组的get字段、执行Thread句柄数组、实际执行流ID数组）；进入兼容资源管理器申请Thread配置。
        ret = engineResMgr.HcclThreadAcquire(newEngine, threadNum, type, config.get(), threads, threadId);
        // [中文导读] [AllReduce逐行 S276] 仅当`(newEngine == CommEngine::COMM_ENGINE_AICPU)`（转换后的CPU/AICPU引擎）成立时进入此分支。
        if (newEngine == CommEngine::COMM_ENGINE_AICPU) {
            // [中文导读] [AllReduce逐行 S277] 仅当`(threadNum != threadId.size())`（执行Thread条数、实际执行流ID数组的size字段）成立时进入此分支；读取容器登记项数。
            if (threadNum != threadId.size()) {
                // [中文导读] [AllReduce逐行 S278] 记录HcclThreadAcquire的错误诊断，字段包含执行Thread条数、实际执行流ID数组的size字段；日志本身不执行传输。
                HCCL_ERROR("[%s] threadNum [%u] != threadId.size[%zu]", __func__, threadNum, threadId.size());
                // [中文导读] [AllReduce逐行 S279] 返回HCCL_E_PARA，表示参数不满足此分支要求；此路径停止本函数的后续处理。
                return HCCL_E_PARA;
            // [中文导读] [AllReduce逐行 S280] 结束`if (threadNum != threadId.size())`（执行Thread条数、实际执行流ID数组的size字段）分支/循环；控制流返回外层。
            }
            // [中文导读] [AllReduce逐行 S281] 兼容AICPU申请后上报实际流ID；返回非成功时由检查宏立即向上传递。
            CHK_RET(HcclStreamProfilingReport(comm, threadNum, threadId.data()));
        // [中文导读] [AllReduce逐行 S282] 结束`if (newEngine == CommEngine::COMM_ENGINE_AICPU)`（转换后的CPU/AICPU引擎）分支/循环；控制流返回外层。
        }
    // [中文导读] [AllReduce逐行 S283] 结束`if (hcclComm->IsCommunicatorV2())`（域的兼容外层对象的IsCommunicatorV2字段）分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S284] 仅当`(ret != HCCL_SUCCESS)`（当前调用状态）成立时进入此分支。
    if (ret != HCCL_SUCCESS) {
        // [中文导读] [AllReduce逐行 S285] 记录HcclThreadAcquire的错误诊断；日志本身不执行传输。
        HCCL_ERROR(
            // [中文导读] [AllReduce逐行 S286] 为当前HcclThreadAcquire诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
            "[%s] Failed to create threads for engine[%s], threadNum[%u], ret[%d]", __func__,
            // [中文导读] [AllReduce逐行 S287] 为把枚举转换成诊断名称补入`GetEnumToString(GetCommEngineStatusStrMap(), newEngine).c_str(), threadNum, ret)`（转换后的CPU/AICPU引擎、执行Thread条数、当前调用状态）；本行是参数/结构化初始化续行。
            GetEnumToString(GetCommEngineStatusStrMap(), newEngine).c_str(), threadNum, ret);
        // [中文导读] [AllReduce逐行 S288] 返回ret，表示保留当前调用的状态（可能成功或失败）；此路径停止本函数的后续处理。
        return ret;
    // [中文导读] [AllReduce逐行 S289] 结束`if (ret != HCCL_SUCCESS)`（当前调用状态）分支/循环；控制流返回外层。
    }

    // [中文导读] [AllReduce逐行 S291] 记录HcclThreadAcquire的状态/性能诊断；日志本身不执行传输。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S292] 为当前HcclThreadAcquire诊断/异常表达式提供格式文本，将报告执行Thread条数；这一物理行没有数据搬运副作用。
        "[%s] Allocated %u threads for engine[%s], notifyPerThread[%u]", __func__, threadNum,
        // [中文导读] [AllReduce逐行 S293] 为把枚举转换成诊断名称补入`GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), notifyNumPerThread)`（请求的通信引擎、每个Thread的通知槽数）；本行是参数/结构化初始化续行。
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), notifyNumPerThread);
    // [中文导读] [AllReduce逐行 S294] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S295] 结束HcclThreadAcquire函数体；控制流返回外层。
}

HcclResult HcclThreadAcquireWithStreamDfx(
    hccl::CollComm* collComm, const std::string& commId, CommEngine engine, ThreadHandle thread)
{
    auto hcclCommDfxCallback = collComm->GetDfxCallback();
    int ret = HcommThreadRegisterDfx(thread, hcclCommDfxCallback);
    if (ret != 0) {
        HCCL_ERROR("[HcclThreadAcquire] HcclThreadAcquire HcommThreadRegisterDfx failed, ret:[%d]", ret);
        return HCCL_E_INTERNAL;
    }
    if (engine == CommEngine::COMM_ENGINE_AICPU) {
        Thread* threadPtr = reinterpret_cast<Thread*>(thread);
        CHK_PTR_NULL(threadPtr);
        // [中文导读] AICPU 路径从包装后的 Thread 读取其 SQ ID，用实际执行流建立域的观测信息。
        Stream* threadStream = threadPtr->GetStream();
        CHK_PTR_NULL(threadStream);
        Mc2CommInfo mc2CommInfo;
        mc2CommInfo.FreeStreamId = 0;
        mc2CommInfo.streamsId.push_back(static_cast<u32>(threadStream->sqId()));
        mc2CommInfo.groupname = commId;
        mc2CommInfo.myRankId = collComm->GetMyRankId();
        mc2CommInfo.rankSize = collComm->GetRankSize();
        CHK_RET(collComm->GetParentRankId(mc2CommInfo.parentRankId));
        HcclCommDfx* hcclCommDfx = collComm->GetHcclCommDfx();
        CHK_PTR_NULL(hcclCommDfx);
        hcclCommDfx->ReportMc2CommInfo(mc2CommInfo);
    }
    return HCCL_SUCCESS;
}

// [中文导读] 把已有用户stream纳入通信Thread抽象，附带通知资源；stream由调用者传入，不是此处新建用户流。
// [中文导读] [AllReduce逐行 S327] HcclThreadAcquireWithStream的接口声明：将用户已有 Stream 包装成通信执行 Thread，而非另建用户流；这些参数属于本函数调用边界。
HcclResult HcclThreadAcquireWithStream(
    // [中文导读] [AllReduce逐行 S328] HcclThreadAcquireWithStream的接口声明：通信域句柄、请求的通信引擎、承载任务的执行流、通知槽数量、当前执行Thread句柄；这些参数属于本函数调用边界。
    HcclComm comm, CommEngine engine, aclrtStream stream, uint32_t notifyNum, ThreadHandle* thread)
// [中文导读] [AllReduce逐行 S329] 进入HcclThreadAcquireWithStream函数体：将用户已有 Stream 包装成通信执行 Thread，而非另建用户流。
{
    // [中文导读] [AllReduce逐行 S330] 检查`comm`（通信域句柄）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(comm);
    // [中文导读] [AllReduce逐行 S331] 检查`stream`（承载任务的执行流）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(stream);
    // [中文导读] [AllReduce逐行 S332] 检查`thread`（当前执行Thread句柄）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(thread);
    // [中文导读] [AllReduce逐行 S333] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S334] 向条件错误检查提供`!IsValidCommEngine(engine),`（请求的通信引擎），用于确定触发条件或形成对应诊断。
        !IsValidCommEngine(engine),
        // [中文导读] [AllReduce逐行 S335] 记录HcclThreadAcquireWithStream的错误诊断，字段包含请求的通信引擎；日志本身不执行传输。
        HCCL_ERROR("[%s] commEngine[%d] is invalid", __func__, static_cast<int32_t>(engine)), HCCL_E_PARA);

    // [中文导读] 先统一兼容枚举，再把用户提供的 stream 交给对应域的资源管理器包装成通信 Thread。
    // [中文导读] [AllReduce逐行 S338] 设置转换后的CPU/AICPU引擎为/按`ConvertEngineToTsType(engine)`（请求的通信引擎）；将旧TS引擎枚举转为CPU/AICPU配置引擎。
    CommEngine newEngine = ConvertEngineToTsType(engine);

    // [中文导读] [AllReduce逐行 S340] 设置域的兼容外层对象为/按`static_cast<hccl::hcclComm*>(comm)`（域的兼容外层对象、通信域句柄）。
    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    // [中文导读] [AllReduce逐行 S341] 设置通信域名称为/按`hcclComm->GetIdentifier()`（域的兼容外层对象的GetIdentifier字段）；取得通信域标识字符串。
    std::string commId = hcclComm->GetIdentifier();
    // [中文导读] [AllReduce逐行 S342] 记录HcclThreadAcquireWithStream的状态/性能诊断；日志本身不执行传输。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S343] 为当前HcclThreadAcquireWithStream诊断/异常表达式提供格式文本，将报告通信域名称的c_str字段；这一物理行没有数据搬运副作用。
        "Entry-%s:comm[%s] engine[%s] notifyNum[%u] stream[%p]", __func__, commId.c_str(),
        // [中文导读] [AllReduce逐行 S344] 为把枚举转换成诊断名称补入`GetEnumToString(GetCommEngineStatusStrMap(), newEngine).c_str(), notifyNum, stream)`（转换后的CPU/AICPU引擎、通知槽数量、承载任务的执行流）；本行是参数/结构化初始化续行。
        GetEnumToString(GetCommEngineStatusStrMap(), newEngine).c_str(), notifyNum, stream);
    // [中文导读] [AllReduce逐行 S345] 设置当前调用状态为/按`HCCL_SUCCESS`。
    HcclResult ret = HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S346] 仅当`(hcclComm->IsCommunicatorV2())`（域的兼容外层对象的IsCommunicatorV2字段）成立时进入此分支；判断通信域是否使用V2对象实现。
    if (hcclComm->IsCommunicatorV2()) {
        // [中文导读] [AllReduce逐行 S347] 设置V2通信域对象为/按`hcclComm->GetCollComm()`（域的兼容外层对象的GetCollComm字段）；从域外层对象取得V2通信域。
        hccl::CollComm* collComm = hcclComm->GetCollComm();
        // [中文导读] [AllReduce逐行 S348] 检查`collComm`（V2通信域对象）不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_PTR_NULL(collComm);
        // [中文导读] [AllReduce逐行 S349] 设置CommEngineResMgr* engineResMgr为/按`collComm->GetCommEngineResMgr()`（V2通信域对象的GetCommEngineResMgr字段）；取得域引擎资源管理器。
        CommEngineResMgr* engineResMgr = collComm->GetCommEngineResMgr();
        // [中文导读] [AllReduce逐行 S350] 检查`engineResMgr`不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_PTR_NULL(engineResMgr);
        // [中文导读] [AllReduce逐行 S351] 设置当前调用状态为/按`engineResMgr->HcclThreadAcquireWithStream(newEngine, stream, notifyNum, thread)`（转换后的CPU/AICPU引擎、承载任务的执行流、通知槽数量、当前执行Thread句柄）；把用户提供Stream包装为通信Thread并准备通知资源。
        ret = engineResMgr->HcclThreadAcquireWithStream(newEngine, stream, notifyNum, thread);
        // [中文导读] [AllReduce逐行 S352] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
        CHK_PRT_RET(
            // [中文导读] [AllReduce逐行 S353] 指定条件命中时要返回的错误状态`ret != HCCL_SUCCESS, HCCL_ERROR("[%s] HcclThreadAcquireWithStream failed, ret[%d]", __func__, ret), ret)`（当前调用状态），未命中则继续原处理路径。
            ret != HCCL_SUCCESS, HCCL_ERROR("[%s] HcclThreadAcquireWithStream failed, ret[%d]", __func__, ret), ret);
        // [中文导读] [AllReduce逐行 S354] 调用HcclThreadAcquireWithStreamDfx，使用V2通信域对象、通信域名称、转换后的CPU/AICPU引擎、当前执行Thread句柄；返回非成功时由检查宏立即向上传递。
        CHK_RET(HcclThreadAcquireWithStreamDfx(collComm, commId, newEngine, *thread));
    // [中文导读] [AllReduce逐行 S355] 进入上一个条件未命中的替代路径，按本函数的资源/设备/协议分流继续处理。
    } else {
        // [中文导读] [AllReduce逐行 S356] 设置auto& engineResMgr为/按`hcclComm->GetIndependentOp().GetCommEngineResMgr()`（域的兼容外层对象的GetIndependentOp字段）；取得旧兼容域的独立算子资源入口；取得域引擎资源管理器。
        auto& engineResMgr = hcclComm->GetIndependentOp().GetCommEngineResMgr();
        // [中文导读] [AllReduce逐行 S357] 设置当前调用状态为/按`engineResMgr.HcclThreadAcquireWithStream(newEngine, stream, notifyNum, thread)`（转换后的CPU/AICPU引擎、承载任务的执行流、通知槽数量、当前执行Thread句柄）；把用户提供Stream包装为通信Thread并准备通知资源。
        ret = engineResMgr.HcclThreadAcquireWithStream(newEngine, stream, notifyNum, thread);
        // [中文导读] [AllReduce逐行 S358] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
        CHK_PRT_RET(
            // [中文导读] [AllReduce逐行 S359] 指定条件命中时要返回的错误状态`ret != HCCL_SUCCESS, HCCL_ERROR("[%s] HcclThreadAcquireWithStream failed, ret[%d]", __func__, ret), ret)`（当前调用状态），未命中则继续原处理路径。
            ret != HCCL_SUCCESS, HCCL_ERROR("[%s] HcclThreadAcquireWithStream failed, ret[%d]", __func__, ret), ret);
    // [中文导读] [AllReduce逐行 S360] 结束`if (hcclComm->IsCommunicatorV2())`（域的兼容外层对象的IsCommunicatorV2字段）分支/循环；控制流返回外层。
    }

    // [中文导读] [AllReduce逐行 S362] 记录HcclThreadAcquireWithStream的状态/性能诊断；日志本身不执行传输。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S363] 为当前HcclThreadAcquireWithStream诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
        "[HcclThreadAcquireWithStream] Allocated thread for engine[%s], stream[%p], notifyNum[%u]",
        // [中文导读] [AllReduce逐行 S364] 为把枚举转换成诊断名称补入`GetEnumToString(GetCommEngineStatusStrMap(), newEngine).c_str(), stream, notifyNum)`（转换后的CPU/AICPU引擎、承载任务的执行流、通知槽数量）；本行是参数/结构化初始化续行。
        GetEnumToString(GetCommEngineStatusStrMap(), newEngine).c_str(), stream, notifyNum);
    // [中文导读] [AllReduce逐行 S365] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S366] 结束HcclThreadAcquireWithStream函数体；控制流返回外层。
}

HcclResult HcclDedicatedThreadAcquire(
    HcclComm comm, HcclDedicatedThreadType useType, uint32_t notifyNumPerThread, ThreadHandle* thread)
{
    EXCEPTION_HANDLE_BEGIN
    CHK_PRT_RET(comm == nullptr, HCCL_ERROR("[%s] comm is null", __func__), HCCL_E_PTR);
    CHK_PRT_RET(thread == nullptr, HCCL_ERROR("[%s] thread is null", __func__), HCCL_E_PTR);
    CHK_PRT_RET(
        useType == HCCL_DED_THREAD_TYPE_INVALID, HCCL_ERROR("[%s] dedThreadType is invalid", __func__), HCCL_E_PARA);

    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    const std::string& commId = hcclComm->GetIdentifier();
    HCCL_INFO(
        "Entry-%s:comm[%s] dedThreadType[%u] notifyNumPerThread[%u]", __func__, commId.c_str(), useType,
        notifyNumPerThread);
    hccl::CollComm* collComm = hcclComm->GetCollComm();
    CHK_PTR_NULL(collComm);
    /* 保序场景：委托给 OrderLaunchThreadMgr（进程粒度） */
    // [中文导读] 保序专用 Thread 由当前设备的进程级 OrderLaunchThreadMgr 分配，其他专用类型继续走域资源管理器。
    if (ORDER_LAUNCH_TYPES.find(useType) != ORDER_LAUNCH_TYPES.end()) {
        s32 deviceLogicId = Hccl::HrtGetDevice();
        auto& resMgr = hccl::CollCommMgr::GetInstance().GetOrderLaunchThreadMgr(deviceLogicId);
        ThreadHandle th = 0;
        HcclResult ret = resMgr.OrderLaunchThreadAcquire(useType, collComm, commId, notifyNumPerThread, th);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[%s] OrderLaunchThreadAcquire fail, ret[%d], useType[%d]", __func__, ret, static_cast<s32>(useType)),
            ret);
        *thread = th;
        return HCCL_SUCCESS;
    }

    CommEngineResMgr* engineResMgr = collComm->GetCommEngineResMgr();
    CHK_PTR_NULL(engineResMgr);
    CHK_RET(engineResMgr->HcclDedicatedThreadAcquire(useType, notifyNumPerThread, thread));
    HCCL_INFO(
        "[%s] success, dedThreadType[%u], thread[0x%llx], notifyNumPerThread[%u]", __func__, useType, *thread,
        notifyNumPerThread);
    EXCEPTION_HANDLE_END

    return HCCL_SUCCESS;
}

HcclResult HcclAllocNotify(
    HcclComm comm, CommEngine commEngine, ::NotifyType notifyType, uint32_t notifyNum, NotifyHandle** notifyHandleList)
{
    CHK_PRT_RET(comm == nullptr, HCCL_ERROR("[%s] comm is null", __func__), HCCL_E_PARA);
    CHK_PRT_RET(
        !IsValidCommEngine(commEngine),
        HCCL_ERROR(
            "[%s] commEngine[%s] is invalid", __func__,
            GetEnumToString(GetCommEngineStatusStrMap(), commEngine).c_str()),
        HCCL_E_PARA);
    CHK_PRT_RET(
        !IsValidNotify(notifyType), HCCL_ERROR("[%s] notifyType[%u] is invalid", __func__, notifyType), HCCL_E_PARA);
    CHK_PRT_RET(
        notifyNum > NOTIFY_MAX_NUM || notifyNum == 0, HCCL_ERROR("[%s] notifyNum[%u] is invalid", __func__, notifyNum),
        HCCL_E_PARA);
    CHK_PRT_RET(notifyHandleList == nullptr, HCCL_ERROR("[%s] notifyHandleList is null", __func__), HCCL_E_PARA);
    CHK_PRT_RET(*notifyHandleList != nullptr, HCCL_ERROR("[%s] notifyHandleList is not null", __func__), HCCL_E_PARA);

    if (commEngine == CommEngine::COMM_ENGINE_CPU || commEngine == CommEngine::COMM_ENGINE_CPU_TS
        || commEngine == CommEngine::COMM_ENGINE_CCU) {
        if (notifyType != ::NOTIFY_TYPE_RTS_NOTIFY) {
            HCCL_ERROR(
                "[%s] commEngine[%s] and notifyType[%u] are mismatch", __func__,
                GetEnumToString(GetCommEngineStatusStrMap(), commEngine).c_str(), notifyType);
            return HCCL_E_PARA;
        }
    } else {
        if (notifyType != ::NOTIFY_TYPE_DEVICE_MEM) {
            HCCL_ERROR(
                "[%s] commEngine[%s] and notifyType[%u] are mismatch", __func__,
                GetEnumToString(GetCommEngineStatusStrMap(), commEngine).c_str(), notifyType);
            return HCCL_E_PARA;
        }
    }

    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    std::string commId = hcclComm->GetIdentifier();
    HCCL_RUN_INFO(
        "Entry-%s:comm[%s] commEngine[%s] notifyType[%u] notifyNum[%u]", __func__, commId.c_str(),
        GetEnumToString(GetCommEngineStatusStrMap(), commEngine).c_str(), notifyType, notifyNum);
    HcclResult ret = HCCL_SUCCESS;
    if (hcclComm->IsCommunicatorV2()) {
        hccl::CollComm* collComm = hcclComm->GetCollComm();
        CHK_PTR_NULL(collComm);
        CommEngineResMgr* engineResMgr = collComm->GetCommEngineResMgr();
        CHK_PTR_NULL(engineResMgr);
        ret = engineResMgr->HcclAllocNotify(commEngine, notifyType, notifyNum, notifyHandleList);
    } else {
        auto& engineResMgr = hcclComm->GetIndependentOp().GetCommEngineResMgr();
        ret = engineResMgr.HcclAllocNotify(commEngine, notifyType, notifyNum, notifyHandleList);
    }

    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[%s] Failed to create notify for commEngine[%s]", __func__,
            GetEnumToString(GetCommEngineStatusStrMap(), commEngine).c_str());
        return ret;
    }

    HCCL_RUN_INFO(
        "[%s] Allocated notify for commEngine[%s], notifyType[%u], notifyNum[%u]", __func__,
        GetEnumToString(GetCommEngineStatusStrMap(), commEngine).c_str(), notifyType, notifyNum);
    return HCCL_SUCCESS;
}

HcclResult HcommFreeNotify(HcclComm comm, uint32_t notifyNum, NotifyHandle* notifyHandleList)
{
    CHK_PRT_RET(comm == nullptr, HCCL_ERROR("[%s] comm is null", __func__), HCCL_E_PARA);
    CHK_PRT_RET(notifyHandleList == nullptr, HCCL_ERROR("[%s] notifyHandleList is null", __func__), HCCL_E_PARA);
    CHK_PRT_RET(
        notifyNum > NOTIFY_MAX_NUM || notifyNum == 0, HCCL_ERROR("[%s] notifyNum[%u] is invalid", __func__, notifyNum),
        HCCL_E_PARA);
    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    std::string commId = hcclComm->GetIdentifier();
    HCCL_RUN_INFO("Entry-%s:comm[%s] notifyNum[%u]", __func__, commId.c_str(), notifyNum);
    HcclResult ret = HCCL_SUCCESS;
    if (hcclComm->IsCommunicatorV2()) {
        hccl::CollComm* collComm = hcclComm->GetCollComm();
        CHK_PTR_NULL(collComm);
        CommEngineResMgr* engineResMgr = collComm->GetCommEngineResMgr();
        CHK_PTR_NULL(engineResMgr);
        ret = engineResMgr->HcommFreeNotify(notifyNum, notifyHandleList);
    } else {
        auto& engineResMgr = hcclComm->GetIndependentOp().GetCommEngineResMgr();
        ret = engineResMgr.HcommFreeNotify(notifyNum, notifyHandleList);
    }
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] Failed to free notify", __func__);
        return ret;
    }

    HCCL_RUN_INFO("[%s] Free notify for notifyNum[%u]", __func__, notifyNum);
    return HCCL_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif
HcclResult HcclThreadExportToCommEngine(
    HcclComm comm, uint32_t threadNum, const ThreadHandle* threads, CommEngine dstCommEngine,
    ThreadHandle* exportedThreads)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(threads);
    CHK_PTR_NULL(exportedThreads);
    CHK_PRT_RET(
        !IsValidCommEngine(dstCommEngine),
        HCCL_ERROR(
            "[%s] commEngine[%s] is invalid", __func__,
            GetEnumToString(GetCommEngineStatusStrMap(), dstCommEngine).c_str()),
        HCCL_E_PARA);
    // [中文导读] 一次导出接受 1 至 40 个 Thread；目标引擎合法后才进入域管理器的转换逻辑。
    if (threadNum == 0 || threadNum > MAX_EXPORT_THREAD_NUM) {
        HCCL_ERROR("[%s] threadNum[%u] is 0 or greater than %u", __func__, threadNum, MAX_EXPORT_THREAD_NUM);
        return HCCL_E_PARA;
    }

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    std::string commId = hcclComm->GetIdentifier();
    HCCL_INFO(
        "Entry-[%s]:comm[%s], threadNum[%u], commEngine[%s], threadsPtr[%p], exportedThreadsPtr[%p]", __func__,
        commId.c_str(), threadNum, GetEnumToString(GetCommEngineStatusStrMap(), dstCommEngine).c_str(), threads,
        exportedThreads);
    HcclResult ret;
    if (hcclComm->IsCommunicatorV2()) {
        hccl::CollComm* collComm = hcclComm->GetCollComm();
        CHK_PTR_NULL(collComm);
        CommEngineResMgr* engineResMgr = collComm->GetCommEngineResMgr();
        CHK_PTR_NULL(engineResMgr);
        // [中文导读] 由资源管理器产生目标引擎句柄；导出表示转换资源访问形态，适配层没有创建 OS 线程。
        ret = engineResMgr->HcclThreadExportToCommEngine(threadNum, threads, dstCommEngine, exportedThreads);
    } else {
        auto& engineResMgr = hcclComm->GetIndependentOp().GetCommEngineResMgr();
        ret = engineResMgr.HcclThreadExportToCommEngine(threadNum, threads, dstCommEngine, exportedThreads);
    }

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] Thread export failed. Export threadNum[%u], commEngine[%s], threadsPtr[%p], exportedThreadsPtr[%p]",
            __func__, threadNum, GetEnumToString(GetCommEngineStatusStrMap(), dstCommEngine).c_str(), threads,
            exportedThreads),
        ret);
    HCCL_INFO("[%s]:comm[%s] export success.", __func__, commId.c_str());
    return HCCL_SUCCESS;
}
#ifdef __cplusplus
}
#endif

HcclResult
HcclThreadResGetInfo(HcclComm comm, ThreadHandle thread, ThreadResType resType, uint32_t infoLen, void** info)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(info);
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    std::string commId = hcclComm->GetIdentifier();
    HCCL_INFO(
        "Entry-[%s]:comm[%s], thread[0x%llx], resType[%d], infoLen[%u], info[%p]", __func__, commId.c_str(),
        static_cast<unsigned long long>(thread), static_cast<int32_t>(resType), infoLen, info);
    HcclResult ret = HCCL_SUCCESS;
    if (hcclComm->IsCommunicatorV2()) {
        hccl::CollComm* collComm = hcclComm->GetCollComm();
        CHK_PTR_NULL(collComm);
        CommEngineResMgr* engineResMgr = collComm->GetCommEngineResMgr();
        CHK_PTR_NULL(engineResMgr);
        ret = engineResMgr->HcclThreadResGetInfo(thread, resType, infoLen, info);
    } else {
        DevType devType;
        CHK_RET(hrtGetDeviceType(devType));
        if (devType != DevType::DEV_TYPE_910B) { // 910B HOST网卡需要走此流程，不打印错误日志
            HCCL_ERROR("[%s] communicatorType is not supported.", __func__);
        }
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] thread resource get info failed. thread[0x%llx], resType[%d], infoLen[%u], info[%p]", __func__,
            static_cast<unsigned long long>(thread), static_cast<int32_t>(resType), infoLen, info),
        ret);
    HCCL_INFO("[%s]:comm[%s] get thread resource success.", __func__, commId.c_str());
    return HCCL_SUCCESS;
}
