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
#include "hccl_mem.h"
#include "stream_pub.h"
#include "hccl_communicator.h"
#include "hccl_comm_pub.h"
#include "param_check_pub.h"
#include "op_base.h"
#include "hccl_res.h"
#include "symmetric_memory/symmetric_memory.h"
#include "hccl_team_c_adpt.h"

using namespace hccl;

HcclResult HcclCommMemReg(HcclComm comm, const char* memTag, const CommMem* mem, HcclMemHandle* memHandle)

{
    CHK_PRT_RET(comm == nullptr, HCCL_ERROR("[HcclCommMemReg]comm is null"), HCCL_E_PTR);
    CHK_PRT_RET(memTag == nullptr, HCCL_ERROR("[HcclCommMemReg]memTag is null"), HCCL_E_PTR);
    CHK_PRT_RET(
        strlen(memTag) == 0 || strlen(memTag) > HCCL_RES_TAG_MAX_LEN,
        HCCL_ERROR("[HcclCommMemReg]memTag length is %zu", strlen(memTag)), HCCL_E_PARA);
    // [中文导读] 保留的对称内存和 Team 同步内存前缀属于内部协议，拒绝用户标签占用这些命名空间。
    std::string memTagStr(memTag);
    CHK_PRT_RET(
        memTagStr.compare(0, strlen(HCCL_SYMMETRIC_MEMORY_TAG_PREFIX), HCCL_SYMMETRIC_MEMORY_TAG_PREFIX) == 0,
        HCCL_ERROR(
            "[HcclCommMemReg]memTag[%s] uses reserved symmetric memory prefix[%s]", memTag,
            HCCL_SYMMETRIC_MEMORY_TAG_PREFIX),
        HCCL_E_PARA);
    CHK_PRT_RET(
        memTagStr.compare(0, strlen(HCCL_TEAM_SYNCMEM_TAG_PREFIX), HCCL_TEAM_SYNCMEM_TAG_PREFIX) == 0,
        HCCL_ERROR(
            "[HcclCommMemReg]memTag[%s] uses reserved team syncmem prefix[%s]", memTag, HCCL_TEAM_SYNCMEM_TAG_PREFIX),
        HCCL_E_PARA);
    // [中文导读] 检查内存描述和句柄出参，再限定内存类型、非空地址与非零字节容量。
    CHK_PRT_RET(mem == nullptr, HCCL_ERROR("[HcclCommMemReg]mem is null"), HCCL_E_PTR);
    CHK_PRT_RET(memHandle == nullptr, HCCL_ERROR("[HcclCommMemReg]memHandle is null"), HCCL_E_PTR);
    CHK_PRT_RET(
        (mem->type != COMM_MEM_TYPE_DEVICE) && (mem->type != COMM_MEM_TYPE_HOST) && (mem->type != COMM_MEM_TYPE_CCU),
        HCCL_ERROR("[HcclCommMemReg]memoryType[%d] must be device or host or ccu", mem->type), HCCL_E_PARA);
    CHK_PRT_RET(mem->addr == nullptr, HCCL_ERROR("[HcclCommMemReg]addr is null"), HCCL_E_PTR);
    CHK_PRT_RET(
        mem->size == 0, HCCL_ERROR("[HcclCommMemReg]size[%llu] invalid", static_cast<unsigned long long>(mem->size)),
        HCCL_E_PARA);

#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
        std::string commId = hcclComm->GetIdentifier();
        HCCL_RUN_INFO("Entry-%s:comm[%s]", __func__, commId.c_str());
        hccl::CollComm* collComm = hcclComm->GetCollComm();
        CHK_PTR_NULL(collComm);
        auto myRank = collComm->GetMyRank();
        CHK_PTR_NULL(myRank);
        // [中文导读] V2 把标签和原内存描述登记到域的 CommMems，返回域级注册句柄供后续 Channel 选择内存。
        CommMems* commMem = myRank->GetCommMems();
        HcclResult ret = HCCL_SUCCESS;
        ret = commMem->CommRegMem(memTagStr, *mem, memHandle);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_ERROR("[HcclCommMemReg]Bind failed. memTag[%s], ret[%d]", memTag, ret), ret);
        HCCL_INFO("[HcclCommMemReg] success: raw handle[%p]", *memHandle);
        return HCCL_SUCCESS;
    }());
#endif
    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    CHK_PTR_NULL(hcclComm);
    // [中文导读] 兼容路径转换描述类型并清零注册属性；HOST 保留，其他允许类型映射到 DEVICE。
    HcclMem hcclMem;
    hcclMem.addr = mem->addr;
    hcclMem.size = mem->size;
    hcclMem.type = (mem->type == COMM_MEM_TYPE_HOST) ? HCCL_MEM_TYPE_HOST : HCCL_MEM_TYPE_DEVICE;
    HcclRegMemAttr attr;
    attr.value = 0;
    HcclResult ret = hcclComm->GetIndependentOp().GetCommMemMgr().CommRegMem(memTagStr, hcclMem, attr, memHandle);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[HcclCommMemReg]legacy Bind failed. memTag[%s], ret[%d]", memTag, ret), ret);
    HCCL_INFO("[HcclCommMemReg]legacy success: raw handle[%p]", *memHandle);

    return HCCL_SUCCESS;
}

HcclResult HcclCommDeregMem(HcclComm comm, const char* memTag, const void* memHandle)
{
    CHK_PRT_RET(comm == nullptr, HCCL_ERROR("[HcclCommDeregMem]comm is null"), HCCL_E_PTR);
    CHK_PRT_RET(memHandle == nullptr, HCCL_ERROR("[HcclCommDeregMem]memHandle is null"), HCCL_E_PTR);
    CHK_PRT_RET(memTag == nullptr, HCCL_ERROR("[HcclCommDeregMem]memTag is null"), HCCL_E_PARA);
    CHK_PRT_RET(strlen(memTag) == 0, HCCL_ERROR("[HcclCommDeregMem]memTag length is 0"), HCCL_E_PARA);

    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    std::string commId = hcclComm->GetIdentifier();
    HCCL_RUN_INFO("Entry-%s: comm[%s], handle[%p]", __func__, commId.c_str(), memHandle);

    // 解绑某算子下的该句柄
    HcclResult ret = HCCL_SUCCESS;
    if (hcclComm->IsCommunicatorV2()) {
        hccl::CollComm* collComm = hcclComm->GetCollComm();
        CHK_PTR_NULL(collComm);
        CommMemMgr* commMemMgr = collComm->GetCommMemMgr();
        CHK_PTR_NULL(commMemMgr);
        ret = commMemMgr->CommUnregMem(std::string(memTag), memHandle);
    } else {
        auto& commMemMgr = hcclComm->GetIndependentOp().GetCommMemMgr();
        ret = commMemMgr.CommUnregMem(std::string(memTag), memHandle);
    }

    // [中文导读] 当前域没有绑定该句柄时按成功处理，重复解绑可结束而不必让调用方恢复不存在的绑定。
    CHK_PRT_RET(
        ret == HCCL_E_NOT_FOUND, HCCL_WARNING("[HcclCommDeregMem]handle not bound in this domain. raw[%p]", memHandle),
        HCCL_SUCCESS);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[HcclCommDeregMem] unBind failed. handle[%p], ret[%d]", memHandle, ret), ret);
    HCCL_INFO("[HcclCommDeregMem]success: raw handle[%p]", memHandle);
    return HCCL_SUCCESS;
}

// [中文导读] [AllReduce逐行 S132] GetHcclBufferWithClearFlag的接口声明：通信域句柄、CCL缓冲区地址出参、字节容量或单片字节数；这些参数属于本函数调用边界。
HcclResult GetHcclBufferWithClearFlag(HcclComm comm, void** buffer, uint64_t* size, bool clearFlag)
// [中文导读] [AllReduce逐行 S133] 进入GetHcclBufferWithClearFlag函数体：V2 本地 CCL 获取帮助函数，单 Rank 返回空区；可选内存清零。
{
    // [中文导读] [AllReduce逐行 S134] 设置域的兼容外层对象为/按`static_cast<hccl::hcclComm*>(comm)`（域的兼容外层对象、通信域句柄）。
    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    // [中文导读] [AllReduce逐行 S135] 设置通信域名称为/按`hcclComm->GetIdentifier()`（域的兼容外层对象的GetIdentifier字段）；取得通信域标识字符串。
    const std::string& commId = hcclComm->GetIdentifier();
    // [中文导读] [AllReduce逐行 S136] 设置V2通信域对象为/按`hcclComm->GetCollComm()`（域的兼容外层对象的GetCollComm字段）；从域外层对象取得V2通信域。
    hccl::CollComm* collComm = hcclComm->GetCollComm();
    // [中文导读] [AllReduce逐行 S137] 检查`collComm`（V2通信域对象）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(collComm);
    // [中文导读] 单 Rank 不需要跨 Rank 中转区，因此明确返回空地址和零容量，并保持成功状态。
    // [中文导读] [AllReduce逐行 S139] 仅当`(collComm->GetRankSize() == 1)`（V2通信域对象的GetRankSize字段）成立时进入此分支；取得域Rank总数。
    if (collComm->GetRankSize() == 1) {
        // [中文导读] [AllReduce逐行 S140] 记录GetHcclBufferWithClearFlag的警告诊断，字段包含通信域名称的c_str字段；日志本身不执行传输。
        HCCL_RUN_WARNING("[%s] comm[%s] is single rank, return nullptr", __func__, commId.c_str());
        // [中文导读] [AllReduce逐行 S141] 设置CCL缓冲区地址出参为/按`nullptr`。
        *buffer = nullptr;
        // [中文导读] [AllReduce逐行 S142] 设置字节容量或单片字节数为/按`0`。
        *size = 0;
        // [中文导读] [AllReduce逐行 S143] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S144] 结束`if (collComm->GetRankSize() == 1)`（V2通信域对象的GetRankSize字段）分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S145] 设置本Rank资源管理对象为/按`collComm->GetMyRank()`（V2通信域对象的GetMyRank字段）；从V2域取得本Rank资源管理器。
    auto myRank = collComm->GetMyRank();
    // [中文导读] [AllReduce逐行 S146] 检查`myRank`（本Rank资源管理对象）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(myRank);
    // [中文导读] [AllReduce逐行 S147] 设置交给底层注册的内存描述为/按`myRank->GetCommMems()`（本Rank资源管理对象的GetCommMems字段）；取得本Rank域内存管理对象。
    CommMems* commMem = myRank->GetCommMems();
    // [中文导读] [AllReduce逐行 S148] 检查`commMem`（交给底层注册的内存描述）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(commMem);
    // [中文导读] 先取得域内缓冲区，再把 clearFlag 交给清零管理逻辑；同一帮助函数承接普通获取和清零获取。
    // [中文导读] [AllReduce逐行 S150] 调用GetHcclBuffer，使用交给底层注册的内存描述的GetHcclBuffer字段、CCL缓冲区地址出参、字节容量或单片字节数；返回非成功时由检查宏立即向上传递。
    CHK_RET(commMem->GetHcclBuffer(*buffer, *size));
    // [中文导读] [AllReduce逐行 S151] clearFlag为真时处理取得的本地CCL区清零；为假时跳过；返回非成功时由检查宏立即向上传递。
    CHK_RET(commMem->HcclBufferMemset(*buffer, *size, clearFlag));

    // [中文导读] [AllReduce逐行 S153] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S154] 结束GetHcclBufferWithClearFlag函数体；控制流返回外层。
}

// [中文导读] 取得域管理的本地CCL中转区地址及字节容量，不是取得远端CCL，也不是交换用户张量。
// [中文导读] 新流程通过CommMems获取；单Rank可成功返回nullptr和0，不能把“成功”一概等同于存在非空中转区。
// [中文导读] 内存生命周期交域/内存管理器，调用者不应把返回指针当成自己本次单独分配的内存去释放。
// [中文导读] [AllReduce逐行 S159] HcclGetHcclBuffer的接口声明：通信域句柄、CCL缓冲区地址出参、字节容量或单片字节数；这些参数属于本函数调用边界。
HcclResult HcclGetHcclBuffer(HcclComm comm, void** buffer, uint64_t* size)
// [中文导读] [AllReduce逐行 S160] 进入HcclGetHcclBuffer函数体：取得域内本地 CCL 中转区地址与字节容量，保留单 Rank 和版本分流。
{
    // [中文导读] [AllReduce逐行 S161] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
    CHK_PRT_RET(buffer == nullptr, HCCL_ERROR("[%s] buffer is null", __func__), HCCL_E_PTR);
    // [中文导读] [AllReduce逐行 S162] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
    CHK_PRT_RET(comm == nullptr, HCCL_ERROR("[%s] comm is null", __func__), HCCL_E_PTR);
    // [中文导读] [AllReduce逐行 S163] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
    CHK_PRT_RET(size == nullptr, HCCL_ERROR("[%s] size is null", __func__), HCCL_E_PTR);

// [中文导读] [AllReduce逐行 S165] 编译条件`if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))`限定后续实现，区分Host/设备或构建能力分支。
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    // [中文导读] [AllReduce逐行 S166] 定义逐片处理回调，捕获当前连接/配置上下文；回调参数描述本地与远端同一分片及其首尾位置。
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        // [中文导读] [AllReduce逐行 S167] 直接返回`GetHcclBufferWithClearFlag(comm, buffer, size, false)`（通信域句柄、CCL缓冲区地址出参、字节容量或单片字节数）；从V2域取得本地CCL区并按clearFlag控制清零。
        return GetHcclBufferWithClearFlag(comm, buffer, size, false);
    // [中文导读] [AllReduce逐行 S168] 关闭并立即调用前述lambda；其新域查询结果交由外层异常/返回宏处理。
    }());
// [中文导读] [AllReduce逐行 S169] 结束前述编译条件控制的实现片段。
#endif

    // [中文导读] [AllReduce逐行 S171] 设置域的兼容外层对象为/按`static_cast<hccl::hcclComm*>(comm)`（域的兼容外层对象、通信域句柄）。
    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    // [中文导读] [AllReduce逐行 S172] 设置通信域名称为/按`hcclComm->GetIdentifier()`（域的兼容外层对象的GetIdentifier字段）；取得通信域标识字符串。
    std::string commId = hcclComm->GetIdentifier();
    // [中文导读] [AllReduce逐行 S173] 设置V2通信域对象为/按`hcclComm->GetCollComm()`（域的兼容外层对象的GetCollComm字段）；从域外层对象取得V2通信域。
    CollComm* collComm = hcclComm->GetCollComm();
    // [中文导读] [AllReduce逐行 S174] 设置本Rank资源管理对象为/按`nullptr`。
    hccl::MyRank* myRank = nullptr;
    // [中文导读] [AllReduce逐行 S175] 仅当`(collComm != nullptr)`（V2通信域对象）成立时进入此分支。
    if (collComm != nullptr) {
        // [中文导读] [AllReduce逐行 S176] 设置本Rank资源管理对象为/按`collComm->GetMyRank()`（V2通信域对象的GetMyRank字段）；从V2域取得本Rank资源管理器。
        myRank = collComm->GetMyRank();
        // [中文导读] [AllReduce逐行 S177] 仅当`(collComm->GetRankSize() == 1)`（V2通信域对象的GetRankSize字段）成立时进入此分支；取得域Rank总数。
        if (collComm->GetRankSize() == 1) {
            // [中文导读] [AllReduce逐行 S178] 记录HcclGetHcclBuffer的警告诊断，字段包含通信域名称的c_str字段；日志本身不执行传输。
            HCCL_RUN_WARNING("[%s] comm[%s] is single rank, return nullptr", __func__, commId.c_str());
            // [中文导读] [AllReduce逐行 S179] 设置CCL缓冲区地址出参为/按`nullptr`。
            *buffer = nullptr;
            // [中文导读] [AllReduce逐行 S180] 设置字节容量或单片字节数为/按`0`。
            *size = 0;
            // [中文导读] [AllReduce逐行 S181] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
            return HCCL_SUCCESS;
        // [中文导读] [AllReduce逐行 S182] 结束`if (collComm->GetRankSize() == 1)`（V2通信域对象的GetRankSize字段）分支/循环；控制流返回外层。
        }
        // [中文导读] 连接模式的兼容域可通过 MyRank 获取 CCL 区；此分支没有调用带 clearFlag 的清零逻辑。
        // [中文导读] [AllReduce逐行 S184] 仅当`(hcclComm->GetConnectMode() != 0 && myRank != nullptr)`（域的兼容外层对象的GetConnectMode字段、本Rank资源管理对象）成立时进入此分支；取得兼容域的特殊连接模式。
        if (hcclComm->GetConnectMode() != 0 && myRank != nullptr) {
            // [中文导读] [AllReduce逐行 S185] 设置交给底层注册的内存描述为/按`myRank->GetCommMems()`（本Rank资源管理对象的GetCommMems字段）；取得本Rank域内存管理对象。
            CommMems* commMem = myRank->GetCommMems();
            // [中文导读] [AllReduce逐行 S186] 检查`commMem`（交给底层注册的内存描述）不是空对象；宏命中失败条件时立即返回对应指针错误。
            CHK_PTR_NULL(commMem);
            // [中文导读] [AllReduce逐行 S187] 调用GetHcclBuffer，使用交给底层注册的内存描述的GetHcclBuffer字段、CCL缓冲区地址出参、字节容量或单片字节数；返回非成功时由检查宏立即向上传递。
            CHK_RET(commMem->GetHcclBuffer(*buffer, *size));
            // [中文导读] [AllReduce逐行 S188] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
            return HCCL_SUCCESS;
        // [中文导读] [AllReduce逐行 S189] 结束`if (hcclComm->GetConnectMode() != 0 && myRank != nullptr)`（域的兼容外层对象的GetConnectMode字段、本Rank资源管理对象）分支/循环；控制流返回外层。
        }
    // [中文导读] [AllReduce逐行 S190] 结束`if (collComm != nullptr)`（V2通信域对象）分支/循环；控制流返回外层。
    }

    // [中文导读] [AllReduce逐行 S192] 记录HcclGetHcclBuffer的状态/性能诊断，字段包含通信域名称的c_str字段；日志本身不执行传输。
    HCCL_RUN_INFO("Entry-%s:comm[%s]", __func__, commId.c_str());
    // [中文导读] [AllReduce逐行 S193] 设置当前调用状态为/按`HCCL_SUCCESS`。
    HcclResult ret = HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S194] 准备`CommBuffer commBuffer`的局部存储/结构描述，初始化方式以本行声明为准。
    CommBuffer commBuffer;

    // [中文导读] [AllReduce逐行 S196] 设置auto& commMemMgr为/按`hcclComm->GetIndependentOp().GetCommMemMgr()`（域的兼容外层对象的GetIndependentOp字段）；取得旧兼容域的独立算子资源入口；取得兼容域内存管理器。
    auto& commMemMgr = hcclComm->GetIndependentOp().GetCommMemMgr();
    // [中文导读] [AllReduce逐行 S197] 设置当前调用状态为/按`commMemMgr.GetHcclBuffer(&commBuffer)`；调用GetHcclBuffer。
    ret = commMemMgr.GetHcclBuffer(&commBuffer);
    // [中文导读] [AllReduce逐行 S198] 仅当`(ret != HCCL_SUCCESS)`（当前调用状态）成立时进入此分支。
    if (ret != HCCL_SUCCESS) {
        // [中文导读] [AllReduce逐行 S199] 记录HcclGetHcclBuffer的错误诊断，字段包含当前调用状态；日志本身不执行传输。
        HCCL_ERROR("[%s] Failed to get local cclBuffer ret[%d]", __func__, ret);
        // [中文导读] [AllReduce逐行 S200] 返回ret，表示保留当前调用的状态（可能成功或失败）；此路径停止本函数的后续处理。
        return ret;
    // [中文导读] [AllReduce逐行 S201] 结束`if (ret != HCCL_SUCCESS)`（当前调用状态）分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S202] 设置CCL缓冲区地址出参为/按`commBuffer.addr`。
    *buffer = commBuffer.addr;
    // [中文导读] [AllReduce逐行 S203] 设置字节容量或单片字节数为/按`commBuffer.size`。
    *size = commBuffer.size;
    // [中文导读] [AllReduce逐行 S204] 记录HcclGetHcclBuffer的状态/性能诊断；日志本身不执行传输。
    HCCL_RUN_INFO(
        // [中文导读] [AllReduce逐行 S205] 为当前HcclGetHcclBuffer诊断/异常表达式提供格式文本，将报告通信域名称的c_str字段、CCL缓冲区地址出参；这一物理行没有数据搬运副作用。
        "Entry-%s: success: comm[%s], buffer[%p] size[%llu]", __func__, commId.c_str(), *buffer,
        // [中文导读] [AllReduce逐行 S206] 为调用c_str，使用通信域名称的c_str字段、CCL缓冲区地址出参、字节容量或单片字节数补入`static_cast<unsigned long long>(*size))`（字节容量或单片字节数）；本行是参数/结构化初始化续行。
        static_cast<unsigned long long>(*size));
    // [中文导读] [AllReduce逐行 S207] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S208] 结束HcclGetHcclBuffer函数体；控制流返回外层。
}

HcclResult HcclGetHcclBufferCleared(HcclComm comm, void** buffer, uint64_t* size)
{
    CHK_PRT_RET(buffer == nullptr, HCCL_ERROR("[%s] buffer is null", __func__), HCCL_E_PTR);
    CHK_PRT_RET(comm == nullptr, HCCL_ERROR("[%s] comm is null", __func__), HCCL_E_PTR);
    CHK_PRT_RET(size == nullptr, HCCL_ERROR("[%s] size is null", __func__), HCCL_E_PTR);

#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    // [中文导读] 清零获取只在此 Host 新流程分支实现；不支持的编译/兼容路径返回 NOT_SUPPORT。
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        return GetHcclBufferWithClearFlag(comm, buffer, size, true);
    }());
#endif

    HCCL_ERROR("HcclGetHcclBufferCleared is not supported");
    return HCCL_E_NOT_SUPPORT;
}
