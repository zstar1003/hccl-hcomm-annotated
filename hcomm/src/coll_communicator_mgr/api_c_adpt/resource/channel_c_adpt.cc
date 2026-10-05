/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl/hccl_res.h"
#include "log.h"
#include "hccl_comm_pub.h"
#include "independent_op.h"
#include "channel_manager.h"
#include "hcomm_c_adpt.h"
#include "param_check_pub.h"
#include "hccl_one_sided_conn.h"
#include <array>
#include <vector>

using namespace hccl;

// [中文导读] 查询通道可用的通知槽位数量，不创建通知、不发送通知；V2和兼容域由不同实现提供结果。
// [中文导读] [AllReduce逐行 S25] HcclChannelGetNotifyNum的接口声明：通信域句柄、传输通道句柄、通知槽数量；这些参数属于本函数调用边界。
HcclResult HcclChannelGetNotifyNum(HcclComm comm, ChannelHandle channel, uint32_t* notifyNum)
// [中文导读] [AllReduce逐行 S26] 进入HcclChannelGetNotifyNum函数体：查询通道本地通知槽容量，适配 V2 与兼容域。
{
    // [中文导读] [AllReduce逐行 S27] 检查`notifyNum`（通知槽数量）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(notifyNum);
    // [中文导读] [AllReduce逐行 S28] 检查`comm`（通信域句柄）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(comm);

    // [中文导读] [AllReduce逐行 S30] 设置域的兼容外层对象为/按`static_cast<hccl::hcclComm*>(comm)`（域的兼容外层对象、通信域句柄）。
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    // [中文导读] [AllReduce逐行 S31] 设置当前调用状态为/按`HCCL_SUCCESS`。
    HcclResult ret = HCCL_SUCCESS;
    // [中文导读] 按通信域实现选择通知容量查询：V2 进入基础通道接口，兼容域进入 IndependentOp 管理器。
    // [中文导读] [AllReduce逐行 S33] 仅当`(hcclComm->IsCommunicatorV2())`（域的兼容外层对象的IsCommunicatorV2字段）成立时进入此分支；判断通信域是否使用V2对象实现。
    if (hcclComm->IsCommunicatorV2()) {
        // [中文导读] [AllReduce逐行 S34] 设置当前调用状态为/按`static_cast<HcclResult>(HcommChannelGetNotifyNum(channel, notifyNum))`（传输通道句柄、通知槽数量）；查询基础层Channel的通知容量。
        ret = static_cast<HcclResult>(HcommChannelGetNotifyNum(channel, notifyNum));
    // [中文导读] [AllReduce逐行 S35] 进入上一个条件未命中的替代路径，按本函数的资源/设备/协议分流继续处理。
    } else {
        // [中文导读] [AllReduce逐行 S36] 设置auto& channelMgr为/按`hcclComm->GetIndependentOp().GetChannelManager()`（域的兼容外层对象的GetIndependentOp字段）；取得旧兼容域的独立算子资源入口；取得兼容通道管理器。
        auto& channelMgr = hcclComm->GetIndependentOp().GetChannelManager();
        // [中文导读] [AllReduce逐行 S37] 设置当前调用状态为/按`channelMgr.ChannelCommGetNotifyNum(channel, notifyNum)`（传输通道句柄、通知槽数量）；查询兼容ChannelManager保存的通知容量。
        ret = channelMgr.ChannelCommGetNotifyNum(channel, notifyNum);
    // [中文导读] [AllReduce逐行 S38] 结束`if (hcclComm->IsCommunicatorV2())`（域的兼容外层对象的IsCommunicatorV2字段）分支/循环；控制流返回外层。
    }

    // [中文导读] [AllReduce逐行 S40] 仅当`(ret != HCCL_SUCCESS)`（当前调用状态）成立时进入此分支。
    if (ret != HCCL_SUCCESS) {
        // [中文导读] [AllReduce逐行 S41] 记录HcclChannelGetNotifyNum的错误诊断；日志本身不执行传输。
        HCCL_ERROR(
            // [中文导读] [AllReduce逐行 S42] 为当前HcclChannelGetNotifyNum诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
            "[%s] Failed to get channel notifyNum, group[%s], channel[%llu], ret[%d]", __func__,
            // [中文导读] [AllReduce逐行 S43] 为取得通信域标识字符串补入`hcclComm->GetIdentifier().c_str(), static_cast<unsigned long long>(channel), ret)`（域的兼容外层对象的GetIdentifier字段、传输通道句柄、当前调用状态）；本行是参数/结构化初始化续行。
            hcclComm->GetIdentifier().c_str(), static_cast<unsigned long long>(channel), ret);
        // [中文导读] [AllReduce逐行 S44] 返回ret，表示保留当前调用的状态（可能成功或失败）；此路径停止本函数的后续处理。
        return ret;
    // [中文导读] [AllReduce逐行 S45] 结束`if (ret != HCCL_SUCCESS)`（当前调用状态）分支/循环；控制流返回外层。
    }

    // [中文导读] [AllReduce逐行 S47] 记录HcclChannelGetNotifyNum的状态/性能诊断；日志本身不执行传输。
    HCCL_RUN_INFO(
        // [中文导读] [AllReduce逐行 S48] 为当前HcclChannelGetNotifyNum诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
        "[%s] get channel notifyNum success, group[%s], channel[%llu], notifyNum[%u], ret[%d]", __func__,
        // [中文导读] [AllReduce逐行 S49] 为取得通信域标识字符串补入`hcclComm->GetIdentifier().c_str(), static_cast<unsigned long long>(channel), *notifyNum, ret)`（域的兼容外层对象的GetIdentifier字段、传输通道句柄、通知槽数量、当前调用状态）；本行是参数/结构化初始化续行。
        hcclComm->GetIdentifier().c_str(), static_cast<unsigned long long>(channel), *notifyNum, ret);
    // [中文导读] [AllReduce逐行 S50] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S51] 结束HcclChannelGetNotifyNum函数体；控制流返回外层。
}

HcclResult CommChannelDestroy(HcclComm comm, ChannelHandle* channelList, uint32_t channelNum)
{
    // [中文导读] 销毁按 channelList 数组逐项处理，channelNum 是通道条数，零条请求在适配层拒绝。
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(channelList);
    CHK_PRT_RET(
        channelNum == 0, HCCL_ERROR("[%s]Invalid channelNum, channelNum[%u]", __func__, channelNum), HCCL_E_PARA);
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    HcclResult ret = HCCL_SUCCESS;
    // [中文导读] 资源仍由创建它的域管理器销毁；V2 和兼容域分别取得各自 ChannelManager。
    if (hcclComm->IsCommunicatorV2()) {
        CollComm* collComm = hcclComm->GetCollComm();
        CHK_PTR_NULL(collComm);
        ChannelManager* channelMgr = collComm->GetChannelManager();
        CHK_PTR_NULL(channelMgr);
        ret = channelMgr->ChannelCommDestroy(channelList, channelNum);
    } else {
        auto& channelMgr = hcclComm->GetIndependentOp().GetChannelManager();
        ret = channelMgr.ChannelCommDestroy(channelList, channelNum);
    }

    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[%s] Failed to destroy channel, group[%s], channelList[%p], channelNum[%u], ret[%d]", __func__,
            hcclComm->GetIdentifier().c_str(), channelList, channelNum, ret);
        return ret;
    }

    HCCL_RUN_INFO(
        "[%s] destroy channel success, group[%s], channelList[%p], channelNum[%u], ret[%d]", __func__,
        hcclComm->GetIdentifier().c_str(), channelList, channelNum, ret);
    return HCCL_SUCCESS;
}

// [中文导读] 取得此Channel对应Peer的CCL区地址和容量，供算法构造远端读写切片。
// [中文导读] 返回的是远端内存描述，不是把对端CCL内容复制回本机，也不是重新分配一块本地缓冲区。
// [中文导读] [AllReduce逐行 S89] HcclChannelGetHcclBuffer的接口声明：通信域句柄、传输通道句柄、CCL缓冲区地址出参、字节容量或单片字节数；这些参数属于本函数调用边界。
HcclResult HcclChannelGetHcclBuffer(HcclComm comm, ChannelHandle channel, void** buffer, uint64_t* size)
// [中文导读] [AllReduce逐行 S90] 进入HcclChannelGetHcclBuffer函数体：返回通道对端 CCL 地址和字节容量，新域/兼容域查询不同。
{
    // [中文导读] [AllReduce逐行 S91] 检查`comm`（通信域句柄）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(comm);
    // [中文导读] [AllReduce逐行 S92] 检查`buffer`（CCL缓冲区地址出参）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(buffer);
    // [中文导读] [AllReduce逐行 S93] 检查`size`（字节容量或单片字节数）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(size);
// [中文导读] [AllReduce逐行 S94] 编译条件`if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))`限定后续实现，区分Host/设备或构建能力分支。
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    // [中文导读] Host 编译的新流程直接从 MyRank 查询远端缓冲区；宏分支成功后按其定义结束当前接口。
    // [中文导读] [AllReduce逐行 S96] 定义逐片处理回调，捕获当前连接/配置上下文；回调参数描述本地与远端同一分片及其首尾位置。
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        // [中文导读] [AllReduce逐行 S97] 为Host编译时检查设备是否支持V2，支持则直接返回新流程表达式的状态补入`hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm)`（域的兼容外层对象、通信域句柄）；本行是参数/结构化初始化续行。
        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
        // [中文导读] [AllReduce逐行 S98] 设置V2通信域对象为/按`hcclComm->GetCollComm()`（域的兼容外层对象的GetCollComm字段）；从域外层对象取得V2通信域。
        CollComm* collComm = hcclComm->GetCollComm();
        // [中文导读] [AllReduce逐行 S99] 检查`collComm`（V2通信域对象）不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_PTR_NULL(collComm);
        // [中文导读] [AllReduce逐行 S100] 设置本Rank资源管理对象为/按`collComm->GetMyRank()`（V2通信域对象的GetMyRank字段）；从V2域取得本Rank资源管理器。
        auto myRank = collComm->GetMyRank();
        // [中文导读] [AllReduce逐行 S101] 检查`myRank`（本Rank资源管理对象）不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_PTR_NULL(myRank);
        // [中文导读] [AllReduce逐行 S102] 按Channel查询对端CCL地址与字节容量；返回非成功时由检查宏立即向上传递。
        CHK_RET(myRank->ChannelGetHcclBuffer(channel, buffer, size));
        // [中文导读] [AllReduce逐行 S103] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S104] 关闭并立即调用前述lambda；其新域查询结果交由外层异常/返回宏处理。
    }());
// [中文导读] [AllReduce逐行 S105] 结束前述编译条件控制的实现片段。
#endif
    // [中文导读] [AllReduce逐行 S106] 设置域的兼容外层对象为/按`static_cast<hccl::hcclComm*>(comm)`（域的兼容外层对象、通信域句柄）。
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    // [中文导读] [AllReduce逐行 S107] 设置V2通信域对象为/按`hcclComm->GetCollComm()`（域的兼容外层对象的GetCollComm字段）；从域外层对象取得V2通信域。
    CollComm* collComm = hcclComm->GetCollComm();
    // [中文导读] [AllReduce逐行 S108] 设置本Rank资源管理对象为/按`nullptr`。
    hccl::MyRank* myRank = nullptr;
    // [中文导读] [AllReduce逐行 S109] 仅当`(collComm != nullptr)`（V2通信域对象）成立时进入此分支。
    if (collComm != nullptr) {
        // [中文导读] [AllReduce逐行 S110] 设置本Rank资源管理对象为/按`collComm->GetMyRank()`（V2通信域对象的GetMyRank字段）；从V2域取得本Rank资源管理器。
        myRank = collComm->GetMyRank();
    // [中文导读] [AllReduce逐行 S111] 结束`if (collComm != nullptr)`（V2通信域对象）分支/循环；控制流返回外层。
    }
    // [中文导读] 兼容域若开启连接模式且存在 MyRank，也沿新资源查询路径返回远端 CCL 描述。
    // [中文导读] [AllReduce逐行 S113] 仅当`(collComm != nullptr && hcclComm->GetConnectMode() != 0 && myRank != nullptr)`（V2通信域对象、域的兼容外层对象的GetConnectMode字段、本Rank资源管理对象）成立时进入此分支；取得兼容域的特殊连接模式。
    if (collComm != nullptr && hcclComm->GetConnectMode() != 0 && myRank != nullptr) {
        // [中文导读] [AllReduce逐行 S114] 按Channel查询对端CCL地址与字节容量；返回非成功时由检查宏立即向上传递。
        CHK_RET(myRank->ChannelGetHcclBuffer(channel, buffer, size));
        // [中文导读] [AllReduce逐行 S115] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S116] 结束`if (collComm != nullptr && hcclComm->GetConnectMode() != 0 && myRank != nullptr)`（V2通信域对象、域的兼容外层对象的GetConnectMode字段、本Rank资源管理对象）分支/循环；控制流返回外层。
    }

    // [中文导读] [AllReduce逐行 S118] 准备`CommBuffer commBuffer`的局部存储/结构描述，初始化方式以本行声明为准。
    CommBuffer commBuffer;
    // [中文导读] [AllReduce逐行 S119] 设置auto& channelMgr为/按`hcclComm->GetIndependentOp().GetChannelManager()`（域的兼容外层对象的GetIndependentOp字段）；取得旧兼容域的独立算子资源入口；取得兼容通道管理器。
    auto& channelMgr = hcclComm->GetIndependentOp().GetChannelManager();
    // [中文导读] [AllReduce逐行 S120] 设置当前调用状态为/按`channelMgr.ChannelCommGetHcclBuffer(channel, &commBuffer)`（传输通道句柄）；进入兼容ChannelManager查询对端CCL描述。
    HcclResult ret = channelMgr.ChannelCommGetHcclBuffer(channel, &commBuffer);
    // [中文导读] [AllReduce逐行 S121] 仅当`(ret != HCCL_SUCCESS)`（当前调用状态）成立时进入此分支。
    if (ret != HCCL_SUCCESS) {
        // [中文导读] [AllReduce逐行 S122] 记录HcclChannelGetHcclBuffer的错误诊断；日志本身不执行传输。
        HCCL_ERROR(
            // [中文导读] [AllReduce逐行 S123] 为当前HcclChannelGetHcclBuffer诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
            "[%s] Failed to get channel hccl buffer, group[%s], channel[%llu], ret[%d]", __func__,
            // [中文导读] [AllReduce逐行 S124] 为取得通信域标识字符串补入`hcclComm->GetIdentifier().c_str(), static_cast<unsigned long long>(channel), ret)`（域的兼容外层对象的GetIdentifier字段、传输通道句柄、当前调用状态）；本行是参数/结构化初始化续行。
            hcclComm->GetIdentifier().c_str(), static_cast<unsigned long long>(channel), ret);
        // [中文导读] [AllReduce逐行 S125] 返回ret，表示保留当前调用的状态（可能成功或失败）；此路径停止本函数的后续处理。
        return ret;
    // [中文导读] [AllReduce逐行 S126] 结束`if (ret != HCCL_SUCCESS)`（当前调用状态）分支/循环；控制流返回外层。
    }
    // [中文导读] 兼容查询成功后才拆出远端地址和字节容量，调用者据此构造访问范围。
    // [中文导读] [AllReduce逐行 S128] 设置CCL缓冲区地址出参为/按`commBuffer.addr`。
    *buffer = commBuffer.addr;
    // [中文导读] [AllReduce逐行 S129] 设置字节容量或单片字节数为/按`commBuffer.size`。
    *size = commBuffer.size;

    // [中文导读] [AllReduce逐行 S131] 记录HcclChannelGetHcclBuffer的状态/性能诊断；日志本身不执行传输。
    HCCL_RUN_INFO(
        // [中文导读] [AllReduce逐行 S132] 为当前HcclChannelGetHcclBuffer诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
        "[%s] get channel hccl buffer success, group[%s], channel[%llu], "
        // [中文导读] [AllReduce逐行 S133] 为当前HcclChannelGetHcclBuffer诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
        "buffer[type:%d, addr:%p, size:%llu], ret[%d]",
        // [中文导读] [AllReduce逐行 S134] 为取得通信域标识字符串补入`__func__, hcclComm->GetIdentifier().c_str(), static_cast<unsigned long long>(channel), commBuffer.type,`（域的兼容外层对象的GetIdentifier字段、传输通道句柄）；本行是参数/结构化初始化续行。
        __func__, hcclComm->GetIdentifier().c_str(), static_cast<unsigned long long>(channel), commBuffer.type,
        // [中文导读] [AllReduce逐行 S135] 为取得通信域标识字符串补入`commBuffer.addr, static_cast<unsigned long long>(commBuffer.size), ret)`（当前调用状态）；本行是参数/结构化初始化续行。
        commBuffer.addr, static_cast<unsigned long long>(commBuffer.size), ret);
    // [中文导读] [AllReduce逐行 S136] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S137] 结束HcclChannelGetHcclBuffer函数体；控制流返回外层。
}

// [中文导读] 查询建链交换得到的远端内存集合及标签；AIV等路径用它定位对端的通信信息/标记区。
// [中文导读] 本函数并不把这些内存交给调用方独立拥有，不应自行free返回的内部列表。
// [中文导读] 旧实现可用固定标签代替真实tag，读取代码时必须区分V2与兼容分支。
HcclResult
HcclChannelGetRemoteMems(HcclComm comm, ChannelHandle channel, uint32_t* memNum, CommMem** remoteMems, char*** memTags)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(remoteMems);
    CHK_PTR_NULL(memTags);
    CHK_PTR_NULL(memNum);

#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
        CollComm* collComm = hcclComm->GetCollComm();
        CHK_PTR_NULL(collComm);
        auto myRank = collComm->GetMyRank();
        CHK_PTR_NULL(myRank);
        CHK_RET(myRank->ChannelGetRemoteMems(channel, memNum, remoteMems, memTags));
        return HCCL_SUCCESS;
    }());
#endif

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    std::string commId = hcclComm->GetIdentifier();
    HCCL_RUN_INFO("legacy Entry-%s:comm[%s]", __func__, commId.c_str());
    HcclMem* remoteMem = nullptr;
    HcclResult ret
        = hcclComm->GetIndependentOp().GetChannelManager().ChannelCommGetRemoteMem(channel, &remoteMem, memNum);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[HcclChannelGetRemoteMems]legacy failed. channel[%llu], ret[%d]", channel, ret), ret);
    // [中文导读] 兼容实现把已有 HcclMem 描述视为 CommMem 列表返回；这里只转换描述表示，没有搬运内存内容。
    *remoteMems = reinterpret_cast<CommMem*>(remoteMem);
    if (*memNum > 0) {
        // A2/A3 tag为非真实tag，无法获取到，统一使用固定字符串。
        static const char* HCCL_BUFFER_TAG = "HcclBuffer";
        // [中文导读] 标签指针数组按调用线程保存；其内容会被同线程后续查询复用，调用者需及时读取。
        static thread_local std::array<char*, MAX_REMOTE_MEM_NUM> tagPtrs;
        CHK_PRT_RET(
            *memNum > MAX_REMOTE_MEM_NUM,
            HCCL_ERROR("[HcclChannelGetRemoteMems] memNum[%u] exceeds max[%u]", *memNum, MAX_REMOTE_MEM_NUM),
            HCCL_E_PARA);
        // [中文导读] 先保证远端内存条数不超过标签数组容量，再为每项填入兼容路径的固定标签。
        for (uint32_t i = 0; i < *memNum; ++i) {
            tagPtrs[i] = const_cast<char*>(HCCL_BUFFER_TAG);
        }
        *memTags = tagPtrs.data();
    } else {
        *memTags = nullptr;
    }
    HCCL_INFO("[HcclChannelGetRemoteMems]legacy success: memNum[%u]", *memNum);
    return HCCL_SUCCESS;
}
