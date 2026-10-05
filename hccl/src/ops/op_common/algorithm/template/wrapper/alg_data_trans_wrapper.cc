/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alg_data_trans_wrapper.h"
#include "exec_timeout_manager.h"
#include "hcomm_primitives_dl.h"
#include <atomic>
#include <limits>
#include <algorithm>
#include <type_traits>
#include <vector>

namespace ops_hccl {

namespace {
    // [中文导读] 这里的同名Wait是HCCL局部包装，实际经Default适配层选择HCOMM能力与超时策略。
    // [中文导读] 查调用链时应继续进入hcomm_primitives_dl.cc，不要把这个局部函数当作底层实现。
    // [中文导读] [AllReduce逐行 S25] 声明返回类型 HcclResult，分别由错误码传播或候选匹配协议解释。
    HcclResult
    // [中文导读] [AllReduce逐行 S26] 定义 HcommChannelNotifyWaitOnThread 入口：局部等待包装转调 HcclChannelNotifyWaitOnThreadDefault，按运行时能力和默认超时选择下层接口。
    HcommChannelNotifyWaitOnThread(ThreadHandle thread, ChannelHandle channel, u32 localNotifyIdx, u32 timeout)
    // [中文导读] [AllReduce逐行 S27] 进入 HcommChannelNotifyWaitOnThread 的实现作用域；局部等待包装转调 HcclChannelNotifyWaitOnThreadDefault，按运行时能力和默认超时选择下层接口。
    {
        // [中文导读] [AllReduce逐行 S28] 直接返回 调用 HcclChannelNotifyWaitOnThreadDefault 完成当前参数所指的子步骤 的结果，调用者取得本分支结果。
        return HcclChannelNotifyWaitOnThreadDefault(thread, channel, localNotifyIdx, timeout);
    // [中文导读] [AllReduce逐行 S29] 结束 HcommChannelNotifyWaitOnThread 实现；其返回状态或已写回字段由调用者接收。
    }

    // [中文导读] [AllReduce逐行 S31] 定义 HcommThreadNotifyWaitOnThread 入口：局部等待包装转调 HcclThreadNotifyWaitOnThreadDefault，不是同名下层原语定义。
    HcclResult HcommThreadNotifyWaitOnThread(ThreadHandle thread, u32 notifyIdx, u32 timeout)
    // [中文导读] [AllReduce逐行 S32] 进入 HcommThreadNotifyWaitOnThread 的实现作用域；局部等待包装转调 HcclThreadNotifyWaitOnThreadDefault，不是同名下层原语定义。
    {
        // [中文导读] [AllReduce逐行 S33] 直接返回 调用 HcclThreadNotifyWaitOnThreadDefault 完成当前参数所指的子步骤 的结果，调用者取得本分支结果。
        return HcclThreadNotifyWaitOnThreadDefault(thread, notifyIdx, timeout);
    // [中文导读] [AllReduce逐行 S34] 结束 HcommThreadNotifyWaitOnThread 实现；其返回状态或已写回字段由调用者接收。
    }

    enum HcommBatchTransferSupportState {
        HCOMM_BATCH_TRANSFER_UNINIT = -1,
        HCOMM_BATCH_TRANSFER_UNSUPPORTED = 0,
        HCOMM_BATCH_TRANSFER_SUPPORTED = 1,
    };

    std::atomic<int> g_hcommBatchTransferSupportState{HCOMM_BATCH_TRANSFER_UNINIT};

    // [中文导读] [AllReduce逐行 S44] 定义 GetSliceAddr 入口：实际访问地址等于 DataSlice 基址加字节偏移。
    void* GetSliceAddr(const DataSlice& slice)
    // [中文导读] [AllReduce逐行 S45] 进入 GetSliceAddr 的实现作用域；实际访问地址等于 DataSlice 基址加字节偏移。
    {
        // [中文导读] [AllReduce逐行 S46] 直接返回 static_cast<void*>(static_cast<s8*>(slice.addr_) + slice.累计字节偏移_)，调用者取得本分支结果。
        return static_cast<void*>(static_cast<s8*>(slice.addr_) + slice.offset_);
    // [中文导读] [AllReduce逐行 S47] 结束 GetSliceAddr 实现；其返回状态或已写回字段由调用者接收。
    }

    void TraceDataSlice(
        const char* funcName, const char* transType, u32 sliceIdx, u32 sliceNum, const DataSlice& srcSlice,
        const DataSlice& dstSlice, const void* src, const void* dst, u64 len, HcclDataType dataType,
        HcclReduceOp reduceOp)
    {
        HCCL_DEBUG(
            "[AlgDataTransWrapper][%s][%s] sliceIdx[%u], sliceNum[%u], srcBase[%p], "
            "srcOffset[%llu], srcAddr[%p], srcSize[%llu], srcCount[%llu], dstBase[%p], "
            "dstOffset[%llu], dstAddr[%p], dstSize[%llu], dstCount[%llu], len[%llu], "
            "dataType[%d], reduceOp[%d].",
            funcName, transType, sliceIdx, sliceNum, srcSlice.addr_, static_cast<unsigned long long>(srcSlice.offset_),
            src, static_cast<unsigned long long>(srcSlice.size_), static_cast<unsigned long long>(srcSlice.count_),
            dstSlice.addr_, static_cast<unsigned long long>(dstSlice.offset_), dst,
            static_cast<unsigned long long>(dstSlice.size_), static_cast<unsigned long long>(dstSlice.count_),
            static_cast<unsigned long long>(len), static_cast<int>(dataType), static_cast<int>(reduceOp));
    }

    void TraceBatchSummary(
        const char* funcName, const char* transType, u32 totalSliceNum, u32 validSliceNum, const ChannelInfo& channel)
    {
        HCCL_DEBUG(
            "[AlgDataTransWrapper][%s][%s] totalSliceNum[%u], validSliceNum[%u], "
            "channelHandle[%llu].",
            funcName, transType, totalSliceNum, validSliceNum, static_cast<unsigned long long>(channel.handle));
    }

    // [中文导读] [AllReduce逐行 S75] 定义 MakeBatchTransDesc 入口：把字节长度及本地/远端源目标地址按 READ 或 WRITE 填入批描述联合体。
    HcclHcommBatchTransferDesc MakeBatchTransDesc(HcclHcommTransferType transType, void* dst, void* src, u64 len)
    // [中文导读] [AllReduce逐行 S76] 进入 MakeBatchTransDesc 的实现作用域；把字节长度及本地/远端源目标地址按 READ 或 WRITE 填入批描述联合体。
    {
        // [中文导读] [AllReduce逐行 S77] 设置 desc 为 {}；该值供下方当前分支使用。
        HcclHcommBatchTransferDesc desc = {};
        // [中文导读] [AllReduce逐行 S78] 设置 批描述传输类型 为 transType；该值供下方当前分支使用。
        desc.transType = transType;
        // [中文导读] [AllReduce逐行 S79] 分支条件为 transType 等于 远端 READ 类型；成立进入本块，未成立继续后续分支。
        if (transType == HCCL_HCOMM_TRANSFER_TYPE_READ) {
            // [中文导读] [AllReduce逐行 S80] 设置 READ 字节长度 为 len；该值供下方当前分支使用。
            desc.transferInfo.read.len = len;
            // [中文导读] [AllReduce逐行 S81] 设置 READ 本地目标地址 为 dst；该值供下方当前分支使用。
            desc.transferInfo.read.dst = dst;
            // [中文导读] [AllReduce逐行 S82] 设置 READ 远端源地址 为 src；该值供下方当前分支使用。
            desc.transferInfo.read.src = src;
        // [中文导读] [AllReduce逐行 S83] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
        } else {
            // [中文导读] [AllReduce逐行 S84] 设置 WRITE 字节长度 为 len；该值供下方当前分支使用。
            desc.transferInfo.write.len = len;
            // [中文导读] [AllReduce逐行 S85] 设置 WRITE 远端目标地址 为 dst；该值供下方当前分支使用。
            desc.transferInfo.write.dst = dst;
            // [中文导读] [AllReduce逐行 S86] 设置 WRITE 本地源地址 为 src；该值供下方当前分支使用。
            desc.transferInfo.write.src = src;
        // [中文导读] [AllReduce逐行 S87] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
        // [中文导读] [AllReduce逐行 S88] 直接返回 desc，调用者取得本分支结果。
        return desc;
    // [中文导读] [AllReduce逐行 S89] 结束 MakeBatchTransDesc 实现；其返回状态或已写回字段由调用者接收。
    }

    // [中文导读] [AllReduce逐行 S91] 定义 MakeBatchReduceDesc 入口：把元素 count、数据类型、归约运算及地址填入批归约描述。
    HcclHcommBatchTransferDesc MakeBatchReduceDesc(
        // [中文导读] [AllReduce逐行 S92] 续接 MakeBatchReduceDesc 的入口参数/基类初始化：HcclHcommTransferType transType, void* dst, void* src, u64 count, HcclDataType dataType, HcclReduceOp reduceOp)；引用参数按声明的 const 限制读写。
        HcclHcommTransferType transType, void* dst, void* src, u64 count, HcclDataType dataType, HcclReduceOp reduceOp)
    // [中文导读] [AllReduce逐行 S93] 进入 MakeBatchReduceDesc 的实现作用域；把元素 count、数据类型、归约运算及地址填入批归约描述。
    {
        // [中文导读] [AllReduce逐行 S94] 设置 desc 为 {}；该值供下方当前分支使用。
        HcclHcommBatchTransferDesc desc = {};
        // [中文导读] [AllReduce逐行 S95] 设置 批描述传输类型 为 transType；该值供下方当前分支使用。
        desc.transType = transType;
        // [中文导读] [AllReduce逐行 S96] 设置 批归约元素数量 为 count；该值供下方当前分支使用。
        desc.transferInfo.reduce.count = count;
        // [中文导读] [AllReduce逐行 S97] 设置 批归约目标地址 为 dst；该值供下方当前分支使用。
        desc.transferInfo.reduce.dst = dst;
        // [中文导读] [AllReduce逐行 S98] 设置 批归约源地址 为 src；该值供下方当前分支使用。
        desc.transferInfo.reduce.src = src;
        // [中文导读] [AllReduce逐行 S99] 设置 HCOMM 归约数据类型 为 static_cast<HcommDataType>(dataType)；该值供下方当前分支使用。
        desc.transferInfo.reduce.dataType = static_cast<HcommDataType>(dataType);
        // [中文导读] [AllReduce逐行 S100] 设置 HCOMM 归约运算 为 static_cast<HcommReduceOp>(reduceOp)；该值供下方当前分支使用。
        desc.transferInfo.reduce.reduceOp = static_cast<HcommReduceOp>(reduceOp);
        // [中文导读] [AllReduce逐行 S101] 直接返回 desc，调用者取得本分支结果。
        return desc;
    // [中文导读] [AllReduce逐行 S102] 结束 MakeBatchReduceDesc 实现；其返回状态或已写回字段由调用者接收。
    }

    // [中文导读] [AllReduce逐行 S104] 定义 FuseNotifyToLastWriteReduceDesc 入口：仅当末条描述是 WRITE_REDUCE 才替换成 WRITE_REDUCE_WITH_NOTIFY；普通 WRITE 不融合。
    bool FuseNotifyToLastWriteReduceDesc(std::vector<HcclHcommBatchTransferDesc>& descs, uint32_t notifyIdx)
    // [中文导读] [AllReduce逐行 S105] 进入 FuseNotifyToLastWriteReduceDesc 的实现作用域；仅当末条描述是 WRITE_REDUCE 才替换成 WRITE_REDUCE_WITH_NOTIFY；普通 WRITE 不融合。
    {
        // [中文导读] [AllReduce逐行 S106] 分支条件为 descs.empty(；成立进入本块，未成立继续后续分支。
        if (descs.empty()) {
            // [中文导读] [AllReduce逐行 S107] 当前能力/拓扑/匹配检查未满足，返回 false 供调用者走替代路径。
            return false;
        // [中文导读] [AllReduce逐行 S108] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
        // [中文导读] [AllReduce逐行 S109] 设置 & last 为 descs.back()；该值供下方当前分支使用。
        const HcclHcommBatchTransferDesc& last = descs.back();
        // [中文导读] [AllReduce逐行 S110] 分支条件为 last.transType 不等于 WRITE_REDUCE 类型；成立进入本块，未成立继续后续分支。
        if (last.transType != HCCL_HCOMM_TRANSFER_TYPE_WRITE_REDUCE) {
            // [中文导读] [AllReduce逐行 S111] 当前能力/拓扑/匹配检查未满足，返回 false 供调用者走替代路径。
            return false;
        // [中文导读] [AllReduce逐行 S112] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
        // [中文导读] [AllReduce逐行 S113] 设置 count 为 last.transferInfo.reduce.count；该值供下方当前分支使用。
        const auto count = last.transferInfo.reduce.count;
        // [中文导读] [AllReduce逐行 S114] 设置 dst 为 last.transferInfo.reduce.dst；该值供下方当前分支使用。
        void* dst = last.transferInfo.reduce.dst;
        // [中文导读] [AllReduce逐行 S115] 设置 src 为 last.transferInfo.reduce.src；该值供下方当前分支使用。
        void* src = last.transferInfo.reduce.src;
        // [中文导读] [AllReduce逐行 S116] 设置 reduceOp 为 last.transferInfo.reduce.reduceOp；该值供下方当前分支使用。
        const auto reduceOp = last.transferInfo.reduce.reduceOp;
        // [中文导读] [AllReduce逐行 S117] 设置 dataType 为 last.transferInfo.reduce.dataType；该值供下方当前分支使用。
        const auto dataType = last.transferInfo.reduce.dataType;

        // [中文导读] [AllReduce逐行 S119] 设置 替换末条的融合通知归约描述 为 {}；该值供下方当前分支使用。
        HcclHcommBatchTransferDesc fusedDesc = {};
        // [中文导读] [AllReduce逐行 S120] 设置 替换末条的融合通知归约描述.transType 为 携带通知的 WRITE_REDUCE 类型；该值供下方当前分支使用。
        fusedDesc.transType = HCCL_HCOMM_TRANSFER_TYPE_WRITE_REDUCE_WITH_NOTIFY;
        // [中文导读] [AllReduce逐行 S121] 设置 替换末条的融合通知归约描述.transferInfo.writeReduceWithNotify.count 为 count；该值供下方当前分支使用。
        fusedDesc.transferInfo.writeReduceWithNotify.count = count;
        // [中文导读] [AllReduce逐行 S122] 设置 替换末条的融合通知归约描述.transferInfo.writeReduceWithNotify.dst 为 dst；该值供下方当前分支使用。
        fusedDesc.transferInfo.writeReduceWithNotify.dst = dst;
        // [中文导读] [AllReduce逐行 S123] 设置 替换末条的融合通知归约描述.transferInfo.writeReduceWithNotify.src 为 src；该值供下方当前分支使用。
        fusedDesc.transferInfo.writeReduceWithNotify.src = src;
        // [中文导读] [AllReduce逐行 S124] 设置 替换末条的融合通知归约描述.transferInfo.writeReduceWithNotify.reduceOp 为 reduceOp；该值供下方当前分支使用。
        fusedDesc.transferInfo.writeReduceWithNotify.reduceOp = reduceOp;
        // [中文导读] [AllReduce逐行 S125] 设置 替换末条的融合通知归约描述.transferInfo.writeReduceWithNotify.dataType 为 dataType；该值供下方当前分支使用。
        fusedDesc.transferInfo.writeReduceWithNotify.dataType = dataType;
        // [中文导读] [AllReduce逐行 S126] 设置 替换末条的融合通知归约描述.transferInfo.writeReduceWithNotify.notifyIdx 为 notifyIdx；该值供下方当前分支使用。
        fusedDesc.transferInfo.writeReduceWithNotify.notifyIdx = notifyIdx;

        // [中文导读] [AllReduce逐行 S128] 设置 descs.back() 为 替换末条的融合通知归约描述；该值供下方当前分支使用。
        descs.back() = fusedDesc;
        // [中文导读] [AllReduce逐行 S129] 开始 HCCL_DEBUG 诊断输出，记录 FuseNotifyToLastWriteReduceDesc 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_DEBUG(
            // [中文导读] [AllReduce逐行 S130] 续接 FuseNotifyToLastWriteReduceDesc 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "[AlgDataTransWrapper] FuseNotifyToLastWriteReduceDesc: fused last descriptor to "
            // [中文导读] [AllReduce逐行 S131] 续接 FuseNotifyToLastWriteReduceDesc 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "WRITE_REDUCE_WITH_NOTIFY, notifyIdx[%u].",
            // [中文导读] [AllReduce逐行 S132] 为 FuseNotifyToLastWriteReduceDesc 的诊断/错误宏提供实参：notifyIdx，与前面的格式占位依次对应。
            notifyIdx);
        // [中文导读] [AllReduce逐行 S133] 当前能力/拓扑/匹配检查满足，返回 true。
        return true;
    // [中文导读] [AllReduce逐行 S134] 结束 FuseNotifyToLastWriteReduceDesc 实现；其返回状态或已写回字段由调用者接收。
    }

    // [中文导读] [AllReduce逐行 S136] 声明模板类型参数，将拓扑匹配器与具体算法模板在编译期绑定。
    template <typename ProcessSliceFunc>
    // [中文导读] [AllReduce逐行 S137] 定义 RunBatchTransfer 入口：过滤零长度片并生成批描述，可融合末条 WriteReduce 的通知；有有效描述才提交批传输。
    HcclResult RunBatchTransfer(
        // [中文导读] [AllReduce逐行 S138] 续接 RunBatchTransfer 的入口参数/基类初始化：const ThreadHandle& thread, const ChannelInfo& channel, const std::vector<DataSlice>& 源数据片列表,；引用参数按声明的 const 限制读写。
        const ThreadHandle& thread, const ChannelInfo& channel, const std::vector<DataSlice>& srcSlices,
        // [中文导读] [AllReduce逐行 S139] 续接 RunBatchTransfer 的入口参数/基类初始化：const std::vector<DataSlice>& 目标数据片列表, const char* funcName, const char* transType,；引用参数按声明的 const 限制读写。
        const std::vector<DataSlice>& dstSlices, const char* funcName, const char* transType,
        // [中文导读] [AllReduce逐行 S140] 续接 RunBatchTransfer 的入口参数/基类初始化：ProcessSliceFunc processSlice, bool 末条归约描述通知融合开关 = false, uint32_t notifyIdx = 0, bool* 已融合完成通知标志 = nullptr)；引用参数按声明的 const 限制读写。
        ProcessSliceFunc processSlice, bool fusePostNotify = false, uint32_t notifyIdx = 0, bool* notifyFused = nullptr)
    // [中文导读] [AllReduce逐行 S141] 进入 RunBatchTransfer 的实现作用域；过滤零长度片并生成批描述，可融合末条 WriteReduce 的通知；有有效描述才提交批传输。
    {
        // [中文导读] [AllReduce逐行 S142] 设置 待处理数据片数 为 源数据片列表.size()；该值供下方当前分支使用。
        u32 repeatNum = srcSlices.size();
        // [中文导读] 先把非零数据片整理为批传输描述，零长度片不加入搬运列表。
        // [中文导读] [AllReduce逐行 S144] 建立本阶段局部对象 std::vector<HcclHcommBatchTransferDesc> 有效批传输描述列表，供 RunBatchTransfer 下方参数组装和子调用使用。
        std::vector<HcclHcommBatchTransferDesc> transferDescs;

        // [中文导读] [AllReduce逐行 S146] 逐片处理对应源/目标数据片，零长度片由块内检查跳过；边界/迭代规则为 (int i = 0; i 小于 待处理数据片数; i++。
        for (int i = 0; i < repeatNum; i++) {
            // [中文导读] [AllReduce逐行 S147] 设置 srcSlice 为 源数据片列表[i]；该值供下方当前分支使用。
            const DataSlice srcSlice = srcSlices[i];
            // [中文导读] [AllReduce逐行 S148] 设置 dstSlice 为 目标数据片列表[i]；该值供下方当前分支使用。
            const DataSlice dstSlice = dstSlices[i];
            // [中文导读] [AllReduce逐行 S149] 分支条件为 源片字节数 等于 0；成立进入本块，未成立继续后续分支。
            if (srcSlice.size_ == 0) {
                // [中文导读] [AllReduce逐行 S150] 开始 HCCL_WARNING 诊断输出，记录 RunBatchTransfer 当前阶段的参数、候选或错误；日志本身不决定返回码。
                HCCL_WARNING("[AlgDataTransWrapper] %s: size is 0.", funcName);
                // [中文导读] [AllReduce逐行 S151] 跳过当前遍历项的剩余步骤，直接处理下一项。
                continue;
            // [中文导读] [AllReduce逐行 S152] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
            }
            // [中文导读] [AllReduce逐行 S153] 调用 processSlice 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。
            CHK_RET(processSlice(i, srcSlice, dstSlice, transferDescs, repeatNum));
        // [中文导读] [AllReduce逐行 S154] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }

        // [中文导读] 允许融合时尝试把完成通知合到最后一条 WriteReduce 描述；失败仍由外层单独记录通知。
        // [中文导读] [AllReduce逐行 S157] 分支条件为 末条归约描述通知融合开关 且 已融合完成通知标志 不等于 nullptr；成立进入本块，未成立继续后续分支。
        if (fusePostNotify && notifyFused != nullptr) {
            // [中文导读] [AllReduce逐行 S158] 设置 *已融合完成通知标志 为 FuseNotifyToLastWriteReduceDesc(有效批传输描述列表, notifyIdx)；该值供下方当前分支使用。
            *notifyFused = FuseNotifyToLastWriteReduceDesc(transferDescs, notifyIdx);
        // [中文导读] [AllReduce逐行 S159] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }

        // [中文导读] 只在存在有效描述时提交批传输，握手通知由调用它的协议层负责。
        // [中文导读] [AllReduce逐行 S162] 分支条件为 有效批传输描述列表.size() 大于 0；成立进入本块，未成立继续后续分支。
        if (transferDescs.size() > 0) {
            // [中文导读] [AllReduce逐行 S163] 记录有效描述数及通道句柄；本行实参为 TraceBatchSummary(funcName, transType, 待处理数据片数, 有效批传输描述列表.size(), channel)。
            TraceBatchSummary(funcName, transType, repeatNum, transferDescs.size(), channel);
            // [中文导读] [AllReduce逐行 S164] 将描述数组提交到 HCOMM 批传输适配层；返回值非成功时立即从当前函数返回该错误。
            CHK_RET(static_cast<HcclResult>(HcclHcommBatchTransferOnThread(
                // [中文导读] [AllReduce逐行 S165] 调用 data 完成当前参数所指的子步骤；本行实参为 thread, 当前通道句柄, 有效批传输描述列表.data(), static_cast<u32>(有效批传输描述列表.size()))))。
                thread, channel.handle, transferDescs.data(), static_cast<u32>(transferDescs.size()))));
        // [中文导读] [AllReduce逐行 S166] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
        // [中文导读] [AllReduce逐行 S167] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S168] 结束 RunBatchTransfer 实现；其返回状态或已写回字段由调用者接收。
    }

    // [中文导读] [AllReduce逐行 S170] 声明模板类型参数，将拓扑匹配器与具体算法模板在编译期绑定。
    template <typename ProcessSliceFunc>
    // [中文导读] [AllReduce逐行 S171] 定义 RunBatchTransferAndNotify 入口：批传输之后保证发送方向发 DATA_SIGNAL；若通知已融合到末条 WriteReduce 则不重复发送。
    HcclResult RunBatchTransferAndNotify(
        // [中文导读] [AllReduce逐行 S172] 续接 RunBatchTransferAndNotify 的入口参数/基类初始化：const ThreadHandle& thread, const ChannelInfo& sendChannel, const std::vector<DataSlice>& 源数据片列表,；引用参数按声明的 const 限制读写。
        const ThreadHandle& thread, const ChannelInfo& sendChannel, const std::vector<DataSlice>& srcSlices,
        // [中文导读] [AllReduce逐行 S173] 续接 RunBatchTransferAndNotify 的入口参数/基类初始化：const std::vector<DataSlice>& 目标数据片列表, const char* funcName, const char* transType,；引用参数按声明的 const 限制读写。
        const std::vector<DataSlice>& dstSlices, const char* funcName, const char* transType,
        // [中文导读] [AllReduce逐行 S174] 续接 RunBatchTransferAndNotify 的入口参数/基类初始化：ProcessSliceFunc processSlice, bool 末条归约描述通知融合开关)；引用参数按声明的 const 限制读写。
        ProcessSliceFunc processSlice, bool fusePostNotify)
    // [中文导读] [AllReduce逐行 S175] 进入 RunBatchTransferAndNotify 的实现作用域；批传输之后保证发送方向发 DATA_SIGNAL；若通知已融合到末条 WriteReduce 则不重复发送。
    {
        // [中文导读] [AllReduce逐行 S176] 设置 已融合完成通知标志 为 false；该值供下方当前分支使用。
        bool notifyFused = false;
        // [中文导读] [AllReduce逐行 S177] 过滤空片、构造描述并提交批接口；返回值非成功时立即从当前函数返回该错误。
        CHK_RET(RunBatchTransfer(
            // [中文导读] [AllReduce逐行 S178] 续接本次错误检查/子调用实参：thread, sendChannel, 源数据片列表, 目标数据片列表, funcName, transType, processSlice, 末条归约描述通知融合开关；返回行为由所在完整宏决定。
            thread, sendChannel, srcSlices, dstSlices, funcName, transType, processSlice, fusePostNotify,
            // [中文导读] [AllReduce逐行 S179] 续接本次错误检查/子调用实参：数据完成通知槽, &已融合完成通知标志；返回行为由所在完整宏决定。
            NOTIFY_IDX_DATA_SIGNAL, &notifyFused));
        // [中文导读] [AllReduce逐行 S180] 分支条件为 !已融合完成通知标志；成立进入本块，未成立继续后续分支。
        if (!notifyFused) {
            // [中文导读] [AllReduce逐行 S181] 执行下方完整子调用；返回值非成功时立即从当前函数返回该错误。
            CHK_RET(static_cast<HcclResult>(
                // [中文导读] [AllReduce逐行 S182] 在 Peer 通道上记录远端可观察的通知；本行实参为 HcommChannelNotifyRecordOnThread(thread, 发送 Peer 的通道句柄, 数据完成通知槽)))。
                HcommChannelNotifyRecordOnThread(thread, sendChannel.handle, NOTIFY_IDX_DATA_SIGNAL)));
        // [中文导读] [AllReduce逐行 S183] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
        // [中文导读] [AllReduce逐行 S184] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S185] 结束 RunBatchTransferAndNotify 实现；其返回状态或已写回字段由调用者接收。
    }

    HcclResult CheckReduceSlicePair(
        const DataSlice& srcSlice, const DataSlice& dstSlice, HcclDataType dataType, const char* funcName)
    {
        CHK_PRT_RET(
            srcSlice.count_ * DATATYPE_SIZE_TABLE[dataType] != srcSlice.size_,
            HCCL_ERROR(
                "[AlgDataTransWrapper] %s: src slice count [%u] is not mate to src slice size [%u], "
                "dataType is [%d].",
                funcName, srcSlice.count_, srcSlice.size_, dataType),
            HcclResult::HCCL_E_INTERNAL);
        CHK_PRT_RET(
            dstSlice.count_ * DATATYPE_SIZE_TABLE[dataType] != dstSlice.size_,
            HCCL_ERROR(
                "[AlgDataTransWrapper] %s: dst slice count [%u] is not mate to dst slice size [%u], "
                "dataType is [%d].",
                funcName, dstSlice.count_, dstSlice.size_, dataType),
            HcclResult::HCCL_E_INTERNAL);
        return HCCL_SUCCESS;
    }

    HcclResult RunWriteReduceAndNotify(
        const ThreadHandle& thread, const ChannelInfo& sendChannel, const std::vector<DataSlice>& srcSlices,
        const std::vector<DataSlice>& dstSlices, HcclDataType dataType, HcclReduceOp reduceOp, const char* funcName)
    {
        const u32 repeatNum = srcSlices.size();
        // [中文导读] 先找最后一个非空归约片，确定哪次 WriteReduce 可以携带完成通知。
        int lastValidIdx = -1;
        for (int i = 0; i < repeatNum; i++) {
            if (srcSlices[i].size_ != 0) {
                lastValidIdx = i;
            }
        }
        for (int i = 0; i < repeatNum; i++) {
            const DataSlice srcSlice = srcSlices[i];
            const DataSlice dstSlice = dstSlices[i];
            if (srcSlice.size_ == 0) {
                HCCL_WARNING("[AlgDataTransWrapper] %s: size is 0.", funcName);
                continue;
            }
            CHK_RET(CheckReduceSlicePair(srcSlice, dstSlice, dataType, funcName));
            void* dst = GetSliceAddr(dstSlice);
            void* src = GetSliceAddr(srcSlice);
            TraceDataSlice(
                funcName, "WRITE_REDUCE", i, repeatNum, srcSlice, dstSlice, src, dst, srcSlice.count_, dataType,
                reduceOp);
            // [中文导读] 最后有效归约片使用带通知原语，其它片只归约，确保通知排在所有有效片之后。
            if (i == lastValidIdx) {
                CHK_RET(static_cast<HcclResult>(HcommWriteReduceWithNotifyOnThread(
                    thread, sendChannel.handle, dst, src, srcSlice.count_, static_cast<HcommDataType>(dataType),
                    static_cast<HcommReduceOp>(reduceOp), NOTIFY_IDX_DATA_SIGNAL)));
            } else {
                CHK_RET(static_cast<HcclResult>(HcommWriteReduceOnThread(
                    thread, sendChannel.handle, dst, src, srcSlice.count_, static_cast<HcommDataType>(dataType),
                    static_cast<HcommReduceOp>(reduceOp))));
            }
        }
        // [中文导读] 没有数据可归约时仍单独发送完成通知，保持与对端等待的协议匹配。
        if (lastValidIdx < 0) {
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(thread, sendChannel.handle, NOTIFY_IDX_DATA_SIGNAL)));
        }
        return HCCL_SUCCESS;
    }

    // Template for SendRecv batch write operations (shared by SendRecvBatchWrite and SendRecvBatchWriteReduce)
    // [中文导读] [AllReduce逐行 S252] 声明模板类型参数，将拓扑匹配器与具体算法模板在编译期绑定。
    template <typename SendRecvInfoType, typename ProcessSliceFunc, typename FallbackFunc>
    // [中文导读] [AllReduce逐行 S253] 定义 DoSendRecvBatchTx 入口：双向 Write 握手：向 recv Peer 发 ACK，等 send Peer 的 ACK，批写并发 DATA_SIGNAL，等 recv Peer 完成。
    HcclResult DoSendRecvBatchTx(
        // [中文导读] [AllReduce逐行 S254] 续接 DoSendRecvBatchTx 的入口参数/基类初始化：const SendRecvInfoType& sendRecvInfo, const ThreadHandle& thread, const char* funcName, const char* transType,；引用参数按声明的 const 限制读写。
        const SendRecvInfoType& sendRecvInfo, const ThreadHandle& thread, const char* funcName, const char* transType,
        // [中文导读] [AllReduce逐行 S255] 续接 DoSendRecvBatchTx 的入口参数/基类初始化：ProcessSliceFunc processSlice, FallbackFunc fallback, bool 末条归约描述通知融合开关 = false)；引用参数按声明的 const 限制读写。
        ProcessSliceFunc processSlice, FallbackFunc fallback, bool fusePostNotify = false)
    // [中文导读] [AllReduce逐行 S256] 进入 DoSendRecvBatchTx 的实现作用域；双向 Write 握手：向 recv Peer 发 ACK，等 send Peer 的 ACK，批写并发 DATA_SIGNAL，等 recv Peer 完成。
    {
        // [中文导读] 运行时不支持批传输时调用逐片实现，握手顺序和双向收发语义保持一致。
        // [中文导读] [AllReduce逐行 S258] 分支条件为 !IsHcommBatchTransferOnThreadSupported(；成立进入本块，未成立继续后续分支。
        if (!IsHcommBatchTransferOnThreadSupported()) {
            // [中文导读] [AllReduce逐行 S259] 直接返回 调用 fallback 完成当前参数所指的子步骤 的结果，调用者取得本分支结果。
            return fallback(sendRecvInfo, thread);
        // [中文导读] [AllReduce逐行 S260] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
        // [中文导读] [AllReduce逐行 S261] 设置 const std::vector<DataSlice> 源数据片列表 为 sendRecvInfo.sendRecvSlices_.txSlicesList_.源数据片列表_；该值供下方当前分支使用。
        const std::vector<DataSlice> srcSlices = sendRecvInfo.sendRecvSlices_.txSlicesList_.srcSlices_;
        // [中文导读] [AllReduce逐行 S262] 设置 const std::vector<DataSlice> 目标数据片列表 为 sendRecvInfo.sendRecvSlices_.txSlicesList_.目标数据片列表_；该值供下方当前分支使用。
        const std::vector<DataSlice> dstSlices = sendRecvInfo.sendRecvSlices_.txSlicesList_.dstSlices_;
        // [中文导读] [AllReduce逐行 S263] 设置 const ChannelInfo& sendChannel 为 sendRecvInfo.sendRecvChannels_.txChannel_；该值供下方当前分支使用。
        const ChannelInfo& sendChannel = sendRecvInfo.sendRecvChannels_.txChannel_;
        // [中文导读] [AllReduce逐行 S264] 设置 const ChannelInfo& recvChannel 为 sendRecvInfo.sendRecvChannels_.rxChannel_；该值供下方当前分支使用。
        const ChannelInfo& recvChannel = sendRecvInfo.sendRecvChannels_.rxChannel_;
        // 向write rank发送tx同步，确保该rank的hcclBuffer可用
        // 这里只是在host上向device下任务，所以实际在host侧不会因为wait而阻塞
        // [中文导读] [AllReduce逐行 S267] 先向接收 Peer 记录 ACK，告诉其本端接收槽可以被写入。
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, recvChannel.handle, NOTIFY_IDX_ACK)));
        // 获取执行超时时间
        // [中文导读] [AllReduce逐行 S269] 设置 当前实际执行配置Timeout 为 ExecTimeoutManager::Instance().GetExecTimeout()；该值供下方当前分支使用。
        u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
        // [中文导读] [AllReduce逐行 S270] 执行下方完整子调用；返回值非成功时立即从当前函数返回该错误。
        CHK_RET(static_cast<HcclResult>(
            // [中文导读] [AllReduce逐行 S271] 等待发送 Peer 的 ACK，确认对端接收槽允许本端开始 Write。
            HcommChannelNotifyWaitOnThread(thread, sendChannel.handle, NOTIFY_IDX_ACK, execTimeout)));

        // 写完之后做后同步告诉对面写完了
        // [中文导读] [AllReduce逐行 S274] 提交本地到发送 Peer 的批传输，再向发送 Peer 记录数据完成通知。
        CHK_RET(RunBatchTransferAndNotify(
            // [中文导读] [AllReduce逐行 S275] 续接本次错误检查/子调用实参：thread, sendChannel, 源数据片列表, 目标数据片列表, funcName, transType, processSlice, 末条归约描述通知融合开关；返回行为由所在完整宏决定。
            thread, sendChannel, srcSlices, dstSlices, funcName, transType, processSlice, fusePostNotify));
        // [中文导读] [AllReduce逐行 S276] 执行下方完整子调用；返回值非成功时立即从当前函数返回该错误。
        CHK_RET(static_cast<HcclResult>(
            // [中文导读] [AllReduce逐行 S277] 等待接收 Peer 写入本端的数据完成，形成 Write 双向闭环。
            HcommChannelNotifyWaitOnThread(thread, recvChannel.handle, NOTIFY_IDX_DATA_SIGNAL, execTimeout)));
        // [中文导读] [AllReduce逐行 S278] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S279] 结束 DoSendRecvBatchTx 实现；其返回状态或已写回字段由调用者接收。
    }

    // Template for Send batch operations (shared by SendBatchWrite and SendBatchWriteReduce)
    template <typename DataInfoType, typename ProcessSliceFunc, typename FallbackFunc>
    HcclResult DoSendBatchTx(
        const DataInfoType& sendInfo, const ThreadHandle& thread, const char* funcName, const char* transType,
        ProcessSliceFunc processSlice, FallbackFunc fallback, bool fusePostNotify = false)
    {
        if (!IsHcommBatchTransferOnThreadSupported()) {
            return fallback(sendInfo, thread);
        }
        const std::vector<DataSlice> srcSlices = sendInfo.slices_.srcSlices_;
        const std::vector<DataSlice> dstSlices = sendInfo.slices_.dstSlices_;
        const ChannelInfo& sendChannel = sendInfo.channel_;
        u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread, sendChannel.handle, NOTIFY_IDX_ACK, execTimeout)));

        return RunBatchTransferAndNotify(
            thread, sendChannel, srcSlices, dstSlices, funcName, transType, processSlice, fusePostNotify);
    }

    // Template for Recv batch read operations (shared by RecvBatchRead and RecvBatchReadReduce)
    template <typename DataInfoType, typename ProcessSliceFunc, typename FallbackFunc>
    HcclResult DoRecvBatchRx(
        const DataInfoType& recvInfo, const ThreadHandle& thread, const char* funcName, const char* transType,
        ProcessSliceFunc processSlice, FallbackFunc fallback)
    {
        if (!IsHcommBatchTransferOnThreadSupported()) {
            return fallback(recvInfo, thread);
        }
        const std::vector<DataSlice> srcSlices = recvInfo.slices_.srcSlices_;
        const std::vector<DataSlice> dstSlices = recvInfo.slices_.dstSlices_;
        const ChannelInfo& recvChannel = recvInfo.channel_;
        u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread, recvChannel.handle, NOTIFY_IDX_ACK, execTimeout)));

        CHK_RET(RunBatchTransfer(thread, recvChannel, srcSlices, dstSlices, funcName, transType, processSlice));

        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, recvChannel.handle, NOTIFY_IDX_DATA_SIGNAL)));
        return HCCL_SUCCESS;
    }

    // Template for SendRecv batch read operations (shared by SendRecvBatchRead and SendRecvBatchReadReduce)
    // [中文导读] [AllReduce逐行 S325] 声明模板类型参数，将拓扑匹配器与具体算法模板在编译期绑定。
    template <typename SendRecvInfoType, typename ProcessSliceFunc, typename FallbackFunc>
    // [中文导读] [AllReduce逐行 S326] 定义 DoSendRecvBatchRx 入口：双向 Read 握手：告诉 send Peer 本端可读，等 recv Peer 可读，批读归约，通知 recv Peer，再等 send Peer 读完。
    HcclResult DoSendRecvBatchRx(
        // [中文导读] [AllReduce逐行 S327] 续接 DoSendRecvBatchRx 的入口参数/基类初始化：const SendRecvInfoType& sendRecvInfo, const ThreadHandle& thread, const char* funcName, const char* transType,；引用参数按声明的 const 限制读写。
        const SendRecvInfoType& sendRecvInfo, const ThreadHandle& thread, const char* funcName, const char* transType,
        // [中文导读] [AllReduce逐行 S328] 续接 DoSendRecvBatchRx 的入口参数/基类初始化：ProcessSliceFunc processSlice, FallbackFunc fallback)；引用参数按声明的 const 限制读写。
        ProcessSliceFunc processSlice, FallbackFunc fallback)
    // [中文导读] [AllReduce逐行 S329] 进入 DoSendRecvBatchRx 的实现作用域；双向 Read 握手：告诉 send Peer 本端可读，等 recv Peer 可读，批读归约，通知 recv Peer，再等 send Peer 读完。
    {
        // [中文导读] [AllReduce逐行 S330] 分支条件为 !IsHcommBatchTransferOnThreadSupported(；成立进入本块，未成立继续后续分支。
        if (!IsHcommBatchTransferOnThreadSupported()) {
            // [中文导读] [AllReduce逐行 S331] 直接返回 调用 fallback 完成当前参数所指的子步骤 的结果，调用者取得本分支结果。
            return fallback(sendRecvInfo, thread);
        // [中文导读] [AllReduce逐行 S332] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
        // [中文导读] [AllReduce逐行 S333] 设置 const std::vector<DataSlice> 源数据片列表 为 sendRecvInfo.sendRecvSlices_.rxSlicesList_.源数据片列表_；该值供下方当前分支使用。
        const std::vector<DataSlice> srcSlices = sendRecvInfo.sendRecvSlices_.rxSlicesList_.srcSlices_;
        // [中文导读] [AllReduce逐行 S334] 设置 const std::vector<DataSlice> 目标数据片列表 为 sendRecvInfo.sendRecvSlices_.rxSlicesList_.目标数据片列表_；该值供下方当前分支使用。
        const std::vector<DataSlice> dstSlices = sendRecvInfo.sendRecvSlices_.rxSlicesList_.dstSlices_;
        // [中文导读] [AllReduce逐行 S335] 设置 const ChannelInfo& sendChannel 为 sendRecvInfo.sendRecvChannels_.txChannel_；该值供下方当前分支使用。
        const ChannelInfo& sendChannel = sendRecvInfo.sendRecvChannels_.txChannel_;
        // [中文导读] [AllReduce逐行 S336] 设置 const ChannelInfo& recvChannel 为 sendRecvInfo.sendRecvChannels_.rxChannel_；该值供下方当前分支使用。
        const ChannelInfo& recvChannel = sendRecvInfo.sendRecvChannels_.rxChannel_;
        // [中文导读] [AllReduce逐行 S337] 向发送 Peer 记录 ACK，表示本端源数据可供对端 Read。
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, sendChannel.handle, NOTIFY_IDX_ACK)));
        // [中文导读] [AllReduce逐行 S338] 设置 当前实际执行配置Timeout 为 ExecTimeoutManager::Instance().GetExecTimeout()；该值供下方当前分支使用。
        u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
        // [中文导读] [AllReduce逐行 S339] 执行下方完整子调用；返回值非成功时立即从当前函数返回该错误。
        CHK_RET(static_cast<HcclResult>(
            // [中文导读] [AllReduce逐行 S340] 等待接收 Peer 的 ACK，确认本端可读取其远端输入。
            HcommChannelNotifyWaitOnThread(thread, recvChannel.handle, NOTIFY_IDX_ACK, execTimeout)));

        // [中文导读] [AllReduce逐行 S342] 过滤空片、构造描述并提交批接口；返回值非成功时立即从当前函数返回该错误。
        CHK_RET(RunBatchTransfer(thread, recvChannel, srcSlices, dstSlices, funcName, transType, processSlice));

        // [中文导读] [AllReduce逐行 S344] 执行下方完整子调用；返回值非成功时立即从当前函数返回该错误。
        CHK_RET(static_cast<HcclResult>(
            // [中文导读] [AllReduce逐行 S345] 读归约完成后通知接收 Peer，使对端知道其源缓冲区可复用。
            HcommChannelNotifyRecordOnThread(thread, recvChannel.handle, NOTIFY_IDX_DATA_SIGNAL)));
        // [中文导读] [AllReduce逐行 S346] 执行下方完整子调用；返回值非成功时立即从当前函数返回该错误。
        CHK_RET(static_cast<HcclResult>(
            // [中文导读] [AllReduce逐行 S347] 等待发送 Peer 读取本端输入完成，保证本端源缓冲区安全复用。
            HcommChannelNotifyWaitOnThread(thread, sendChannel.handle, NOTIFY_IDX_DATA_SIGNAL, execTimeout)));
        // [中文导读] [AllReduce逐行 S348] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S349] 结束 DoSendRecvBatchRx 实现；其返回状态或已写回字段由调用者接收。
    }

} // namespace

// [中文导读] [AllReduce逐行 S353] 定义 InitHcommBatchTransferOnThreadSupported 入口：以原子比较交换初始化本进程批传输支持状态，后续上下文能力不一致时返回内部错误。
HcclResult InitHcommBatchTransferOnThreadSupported(bool isSupported)
// [中文导读] [AllReduce逐行 S354] 进入 InitHcommBatchTransferOnThreadSupported 的实现作用域；以原子比较交换初始化本进程批传输支持状态，后续上下文能力不一致时返回内部错误。
{
    // [中文导读] [AllReduce逐行 S355] 设置 target 为 isSupported ? 批接口明确支持状态 : 批接口不支持状态；该值供下方当前分支使用。
    int target = isSupported ? HCOMM_BATCH_TRANSFER_SUPPORTED : HCOMM_BATCH_TRANSFER_UNSUPPORTED;
    // [中文导读] [AllReduce逐行 S356] 设置 expected 为 批传输支持状态尚未初始化；该值供下方当前分支使用。
    int expected = HCOMM_BATCH_TRANSFER_UNINIT;
    // [中文导读] [AllReduce逐行 S357] 分支条件为 进程级原子批能力状态.compare_exchange_strong(expected, target；成立进入本块，未成立继续后续分支。
    if (g_hcommBatchTransferSupportState.compare_exchange_strong(expected, target)) {
        // [中文导读] [AllReduce逐行 S358] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S359] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // [中文导读] [AllReduce逐行 S361] 分支条件为 expected 不等于 target；成立进入本块，未成立继续后续分支。
    if (expected != target) {
        // [中文导读] [AllReduce逐行 S362] 开始 HCCL_ERROR 诊断输出，记录 InitHcommBatchTransferOnThreadSupported 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_ERROR(
            // [中文导读] [AllReduce逐行 S363] 续接 InitHcommBatchTransferOnThreadSupported 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "[AlgDataTransWrapper] HcommBatchTransferOnThread support mismatch, cached[%d], ctx[%d].", expected,
            // [中文导读] [AllReduce逐行 S364] 为 InitHcommBatchTransferOnThreadSupported 的诊断/错误宏提供实参：target，与前面的格式占位依次对应。
            target);
        // [中文导读] [AllReduce逐行 S365] 终止当前函数并向上返回 内部错误；调用者 CHK_RET 决定是否继续向上传播。
        return HCCL_E_INTERNAL;
    // [中文导读] [AllReduce逐行 S366] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S367] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S368] 结束 InitHcommBatchTransferOnThreadSupported 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S370] 定义 IsHcommBatchTransferOnThreadSupported 入口：读取原子能力状态，只有明确 SUPPORTED 才采用批传输协议。
bool IsHcommBatchTransferOnThreadSupported()
// [中文导读] [AllReduce逐行 S371] 进入 IsHcommBatchTransferOnThreadSupported 的实现作用域；读取原子能力状态，只有明确 SUPPORTED 才采用批传输协议。
{
    // [中文导读] [AllReduce逐行 S372] 直接返回 调用 load 完成当前参数所指的子步骤 的结果，调用者取得本分支结果。
    return g_hcommBatchTransferSupportState.load() == HCOMM_BATCH_TRANSFER_SUPPORTED;
// [中文导读] [AllReduce逐行 S373] 结束 IsHcommBatchTransferOnThreadSupported 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] 发送侧Write协议：等待对端ACK允许写入，逐片从本地src写到远端dst，再通知DATA_SIGNAL。
// [中文导读] size_是字节数；零长片不搬运，但前后握手仍存在。远端dst来自预先建立的Channel/内存描述。
HcclResult SendWrite(const DataInfo& sendInfo, const ThreadHandle& thread)
{
    // [中文导读] 提取一一对应的发送源片、远端目标片及发送通道，下面逐片使用同一通道提交。
    const std::vector<DataSlice> srcSlices = sendInfo.slices_.srcSlices_;
    const std::vector<DataSlice> dstSlices = sendInfo.slices_.dstSlices_;
    const ChannelInfo& sendChannel = sendInfo.channel_;
    u32 sliceNum = srcSlices.size();
    // 获取执行超时时间
    // [中文导读] 先排入对端允许写入的 ACK 等待，目标缓冲区就绪后才执行实际数据写入。
    u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, sendChannel.handle, NOTIFY_IDX_ACK, execTimeout)));
    for (int i = 0; i < sliceNum; i++) {
        const DataSlice srcSlice = srcSlices[i];
        const DataSlice dstSlice = dstSlices[i];
        // [中文导读] 空片跳过数据原语，循环后的 DATA_SIGNAL 仍需发送以结束本次握手。
        if (srcSlice.size_ == 0) {
            HCCL_WARNING("[AlgDataTransWrapper] SendWrite: size is 0.");
            continue;
        }
        // [中文导读] 把片基址和字节偏移组合成实际地址，普通 Write 的长度使用字节数。
        void* dst = GetSliceAddr(dstSlice);
        void* src = GetSliceAddr(srcSlice);
        TraceDataSlice(
            "SendWrite", "WRITE", i, sliceNum, srcSlice, dstSlice, src, dst, srcSlice.size_, sendInfo.dataType_,
            HcclReduceOp::HCCL_REDUCE_RESERVED);
        CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(thread, sendChannel.handle, dst, src, srcSlice.size_)));
    }
    // [中文导读] 所有写任务排入后记录数据完成通知，对端用它约束接收后的复制或消费。
    CHK_RET(
        static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, sendChannel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    return HCCL_SUCCESS;
}

HcclResult SendBatchWrite(const DataInfo& sendInfo, const ThreadHandle& thread)
{
    auto processSlice = [&sendInfo](
                            int i, const DataSlice& srcSlice, const DataSlice& dstSlice,
                            std::vector<HcclHcommBatchTransferDesc>& transferDescs, u32 repeatNum) -> HcclResult {
        void* dst = GetSliceAddr(dstSlice);
        void* src = GetSliceAddr(srcSlice);
        TraceDataSlice(
            "SendBatchWrite", "BATCH_WRITE", i, repeatNum, srcSlice, dstSlice, src, dst, srcSlice.size_,
            sendInfo.dataType_, HcclReduceOp::HCCL_REDUCE_RESERVED);
        transferDescs.push_back(MakeBatchTransDesc(HCCL_HCOMM_TRANSFER_TYPE_WRITE, dst, src, srcSlice.size_));
        return HCCL_SUCCESS;
    };
    return DoSendBatchTx(sendInfo, thread, "SendBatchWrite", "BATCH_WRITE", processSlice, SendWrite);
}

// [中文导读] 被写入的一侧只发送ACK并等待DATA_SIGNAL，数据由对端Write送来；没有再调用一次Read。
HcclResult RecvWrite(const DataInfo& recvInfo, const ThreadHandle& thread)
{
    // [中文导读] 接收端先允许对端写入，再等待数据完成；本函数不重复搬运已经由对端写来的数据。
    const ChannelInfo& recvChannel = recvInfo.channel_;
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, recvChannel.handle, NOTIFY_IDX_ACK)));
    // 获取执行超时时间
    u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, recvChannel.handle, NOTIFY_IDX_DATA_SIGNAL, execTimeout)));
    return HCCL_SUCCESS;
}

/*
 这个SendRecv是以notify的视角去看的，针对一个thread上的notify，即有record操作也有wait操作。
 为什么是SendRecv：因为是一个双向的写，rank 0需要向rank 1写，而rank 1也需要向rank 0写，
 因此对于rank 0来说需要向rank 1 record告诉rank 1自己准备好了可以写了，
 而rank 0也需要wait一下rank 1的record知道rank 1那边也可以写了。
*/
// [中文导读] 同一Thread既收又发：向接收通道对端发ACK→等发送通道ACK→逐片Write→发完成→等接收完成。
// [中文导读] tx/rx可以指向不同Peer。顺序约束通过通知任务表达，不是Host在每一步同步阻塞。
// [中文导读] [AllReduce逐行 S448] 定义 SendRecvWrite 入口：无批能力时逐片 Write 双向协议，接收方向只用通知等待对端写入。
HcclResult SendRecvWrite(const SendRecvInfo& sendRecvInfo, const ThreadHandle& thread)
// [中文导读] [AllReduce逐行 S449] 进入 SendRecvWrite 的实现作用域；无批能力时逐片 Write 双向协议，接收方向只用通知等待对端写入。
{
    // [中文导读] [AllReduce逐行 S450] 设置 const std::vector<DataSlice> 源数据片列表 为 sendRecvInfo.sendRecvSlices_.txSlicesList_.源数据片列表_；该值供下方当前分支使用。
    const std::vector<DataSlice> srcSlices = sendRecvInfo.sendRecvSlices_.txSlicesList_.srcSlices_;
    // [中文导读] [AllReduce逐行 S451] 设置 const std::vector<DataSlice> 目标数据片列表 为 sendRecvInfo.sendRecvSlices_.txSlicesList_.目标数据片列表_；该值供下方当前分支使用。
    const std::vector<DataSlice> dstSlices = sendRecvInfo.sendRecvSlices_.txSlicesList_.dstSlices_;
    // [中文导读] 发送通道承担本端写入，接收通道承担对端写入本端，两条通道可以面向不同 Peer。
    // [中文导读] [AllReduce逐行 S453] 设置 const ChannelInfo& sendChannel 为 sendRecvInfo.sendRecvChannels_.txChannel_；该值供下方当前分支使用。
    const ChannelInfo& sendChannel = sendRecvInfo.sendRecvChannels_.txChannel_;
    // [中文导读] [AllReduce逐行 S454] 设置 const ChannelInfo& recvChannel 为 sendRecvInfo.sendRecvChannels_.rxChannel_；该值供下方当前分支使用。
    const ChannelInfo& recvChannel = sendRecvInfo.sendRecvChannels_.rxChannel_;
    // [中文导读] [AllReduce逐行 S455] 设置 待处理数据片数 为 源数据片列表.size()；该值供下方当前分支使用。
    u32 repeatNum = srcSlices.size();
    // 向write rank发送tx同步，确保该rank的hcclBuffer可用
    // 这里只是在host上向device下任务，所以实际在host侧不会因为wait而阻塞
    // [中文导读] 先告诉接收方向的对端本端可写，再等待发送方向的对端允许本端写入。
    // [中文导读] [AllReduce逐行 S459] 在 Peer 通道上记录远端可观察的通知；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, recvChannel.handle, NOTIFY_IDX_ACK)));
    // 获取执行超时时间
    // [中文导读] [AllReduce逐行 S461] 设置 当前实际执行配置Timeout 为 ExecTimeoutManager::Instance().GetExecTimeout()；该值供下方当前分支使用。
    u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
    // [中文导读] [AllReduce逐行 S462] 执行下方完整子调用；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(static_cast<HcclResult>(
        // [中文导读] [AllReduce逐行 S463] 等待本端通道通知槽满足依赖；本行实参为 HcommChannelNotifyWaitOnThread(thread, 发送 Peer 的通道句柄, ACK 通知槽, 当前实际执行配置Timeout)))。
        HcommChannelNotifyWaitOnThread(thread, sendChannel.handle, NOTIFY_IDX_ACK, execTimeout)));
    // [中文导读] 只对发送方向的源目标片执行 Write，接收方向数据由另一端执行 Write 送来。
    // [中文导读] [AllReduce逐行 S465] 逐片处理对应源/目标数据片，零长度片由块内检查跳过；边界/迭代规则为 (int i = 0; i 小于 待处理数据片数; i++。
    for (int i = 0; i < repeatNum; i++) {
        // tx同步完成后准备将自己的userIn上的数据写到对方的hcclBuffer上
        // [中文导读] [AllReduce逐行 S467] 设置 srcSlice 为 源数据片列表[i]；该值供下方当前分支使用。
        const DataSlice srcSlice = srcSlices[i];
        // [中文导读] [AllReduce逐行 S468] 设置 dstSlice 为 目标数据片列表[i]；该值供下方当前分支使用。
        const DataSlice dstSlice = dstSlices[i];
        // [中文导读] [AllReduce逐行 S469] 分支条件为 源片字节数 等于 0；成立进入本块，未成立继续后续分支。
        if (srcSlice.size_ == 0) {
            // [中文导读] [AllReduce逐行 S470] 开始 HCCL_WARNING 诊断输出，记录 SendRecvWrite 当前阶段的参数、候选或错误；日志本身不决定返回码。
            HCCL_WARNING("[AlgDataTransWrapper] SendRecvWrite: size is 0.");
            // [中文导读] [AllReduce逐行 S471] 跳过当前遍历项的剩余步骤，直接处理下一项。
            continue;
        // [中文导读] [AllReduce逐行 S472] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
        // [中文导读] [AllReduce逐行 S473] 设置 dst 为 GetSliceAddr(dstSlice)；该值供下方当前分支使用。
        void* dst = GetSliceAddr(dstSlice);
        // [中文导读] [AllReduce逐行 S474] 设置 src 为 GetSliceAddr(srcSlice)；该值供下方当前分支使用。
        void* src = GetSliceAddr(srcSlice);
        // [中文导读] [AllReduce逐行 S475] 记录源目标基址、偏移、实际地址和类型信息；本行实参为 TraceDataSlice(。
        TraceDataSlice(
            // [中文导读] [AllReduce逐行 S476] 续接本次调用的字符串常量 "SendRecvWrite", "WRITE", i, repeatNum, srcSlice, dstSlice, src, dst, srcSlice.size_,，由所在注册/日志/条件语句整体使用。
            "SendRecvWrite", "WRITE", i, repeatNum, srcSlice, dstSlice, src, dst, srcSlice.size_,
            // [中文导读] [AllReduce逐行 S477] 续接 SendRecvWrite 当前语句的具体实参/字段：sendRecvInfo.当前元素类型, HcclReduceOp::HCCL_REDUCE_RESERVED)；由其完整表达式完成参数组装、检查或结果写回。
            sendRecvInfo.dataType_, HcclReduceOp::HCCL_REDUCE_RESERVED);
        // [中文导读] [AllReduce逐行 S478] 把本地源片写入远端目标片；返回值非成功时立即从当前函数返回该错误。
        CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(thread, sendChannel.handle, dst, src, srcSlice.size_)));
    // [中文导读] [AllReduce逐行 S479] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // 写完之后做后同步告诉对面写完了
    // [中文导读] [AllReduce逐行 S481] 执行下方完整子调用；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(
        // [中文导读] [AllReduce逐行 S482] 在 Peer 通道上记录远端可观察的通知；本行实参为 static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, 发送 Peer 的通道句柄, 数据完成通知槽)))。
        static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, sendChannel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    // [中文导读] 在发送完成通知之后等待接收方向的 DATA_SIGNAL，使双向交换依赖在本 Thread 上闭合。
    // [中文导读] [AllReduce逐行 S484] 执行下方完整子调用；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(static_cast<HcclResult>(
        // [中文导读] [AllReduce逐行 S485] 等待本端通道通知槽满足依赖；本行实参为 HcommChannelNotifyWaitOnThread(thread, 接收 Peer 的通道句柄, 数据完成通知槽, 当前实际执行配置Timeout)))。
        HcommChannelNotifyWaitOnThread(thread, recvChannel.handle, NOTIFY_IDX_DATA_SIGNAL, execTimeout)));
    // [中文导读] [AllReduce逐行 S486] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S487] 结束 SendRecvWrite 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S489] 定义 SendRecvBatchWrite 入口：逐片构造 WRITE 描述，调用双向批发送包装；不支持批接口则 fallback 到 SendRecvWrite。
HcclResult SendRecvBatchWrite(const SendRecvInfo& sendRecvInfo, const ThreadHandle& thread)
// [中文导读] [AllReduce逐行 S490] 进入 SendRecvBatchWrite 的实现作用域；逐片构造 WRITE 描述，调用双向批发送包装；不支持批接口则 fallback 到 SendRecvWrite。
{
    // [中文导读] [AllReduce逐行 S491] 设置 processSlice 为 [&sendRecvInfo](；该值供下方当前分支使用。
    auto processSlice = [&sendRecvInfo](
                            // [中文导读] [AllReduce逐行 S492] 续接 SendRecvBatchWrite 当前语句的具体实参/字段：int i, const DataSlice& srcSlice, const DataSlice& dstSlice,；由其完整表达式完成参数组装、检查或结果写回。
                            int i, const DataSlice& srcSlice, const DataSlice& dstSlice,
                            // [中文导读] [AllReduce逐行 S493] 建立本阶段局部对象 std::vector<HcclHcommBatchTransferDesc>& 有效批传输描述列表, u32 待处理数据片数) -> HcclResult {，供 SendRecvBatchWrite 下方参数组装和子调用使用。
                            std::vector<HcclHcommBatchTransferDesc>& transferDescs, u32 repeatNum) -> HcclResult {
        // [中文导读] [AllReduce逐行 S494] 设置 dst 为 GetSliceAddr(dstSlice)；该值供下方当前分支使用。
        void* dst = GetSliceAddr(dstSlice);
        // [中文导读] [AllReduce逐行 S495] 设置 src 为 GetSliceAddr(srcSlice)；该值供下方当前分支使用。
        void* src = GetSliceAddr(srcSlice);
        // [中文导读] [AllReduce逐行 S496] 记录源目标基址、偏移、实际地址和类型信息；本行实参为 TraceDataSlice(。
        TraceDataSlice(
            // [中文导读] [AllReduce逐行 S497] 续接本次调用的字符串常量 "SendRecvBatchWrite", "BATCH_WRITE", i, repeatNum, srcSlice, dstSlice, src, dst, srcSlice.size_,，由所在注册/日志/条件语句整体使用。
            "SendRecvBatchWrite", "BATCH_WRITE", i, repeatNum, srcSlice, dstSlice, src, dst, srcSlice.size_,
            // [中文导读] [AllReduce逐行 S498] 续接 SendRecvBatchWrite 当前语句的具体实参/字段：sendRecvInfo.当前元素类型, HcclReduceOp::HCCL_REDUCE_RESERVED)；由其完整表达式完成参数组装、检查或结果写回。
            sendRecvInfo.dataType_, HcclReduceOp::HCCL_REDUCE_RESERVED);
        // [中文导读] [AllReduce逐行 S499] 对 有效批传输描述列表 追加 MakeBatchTransDesc(远端 WRITE 类型, dst, src, 源片字节数)，准备或更新本阶段列表。
        transferDescs.push_back(MakeBatchTransDesc(HCCL_HCOMM_TRANSFER_TYPE_WRITE, dst, src, srcSlice.size_));
        // [中文导读] [AllReduce逐行 S500] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S501] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    };
    // [中文导读] [AllReduce逐行 S502] 批能力就绪则构造并批提交 WRITE；未支持则调用 SendRecvWrite，ACK/DATA_SIGNAL 协议由通用 Tx 包装维护。
    return DoSendRecvBatchTx(sendRecvInfo, thread, "SendRecvBatchWrite", "BATCH_WRITE", processSlice, SendRecvWrite);
// [中文导读] [AllReduce逐行 S503] 结束 SendRecvBatchWrite 实现；其返回状态或已写回字段由调用者接收。
}

HcclResult SendRecvBatchWriteReduce(const SendRecvReduceInfo& sendRecvInfo, const ThreadHandle& thread)
{
    auto processSlice = [&sendRecvInfo](
                            int i, const DataSlice& srcSlice, const DataSlice& dstSlice,
                            std::vector<HcclHcommBatchTransferDesc>& transferDescs, u32 repeatNum) -> HcclResult {
        CHK_PRT_RET(
            srcSlice.count_ * DATATYPE_SIZE_TABLE[sendRecvInfo.dataType_] != srcSlice.size_,
            HCCL_ERROR(
                "[AlgDataTransWrapper] SendRecvBatchWriteReduce: src slice count [%u] is not mate to src slice "
                "size [%u], dataType is [%d].",
                srcSlice.count_, srcSlice.size_, sendRecvInfo.dataType_),
            HcclResult::HCCL_E_INTERNAL);
        CHK_PRT_RET(
            dstSlice.count_ * DATATYPE_SIZE_TABLE[sendRecvInfo.dataType_] != dstSlice.size_,
            HCCL_ERROR(
                "[AlgDataTransWrapper] SendRecvBatchWriteReduce: dst slice count [%u] is not mate to dst slice "
                "size [%u], dataType is [%d].",
                dstSlice.count_, dstSlice.size_, sendRecvInfo.dataType_),
            HcclResult::HCCL_E_INTERNAL);
        void* dst = GetSliceAddr(dstSlice);
        void* src = GetSliceAddr(srcSlice);
        u64 len = srcSlice.count_ * DATATYPE_SIZE_TABLE[sendRecvInfo.dataType_];
        TraceDataSlice(
            "SendRecvBatchWriteReduce", "BATCH_WRITE_REDUCE", i, repeatNum, srcSlice, dstSlice, src, dst, len,
            sendRecvInfo.dataType_, sendRecvInfo.reduceType_);
        transferDescs.push_back(MakeBatchReduceDesc(
            HCCL_HCOMM_TRANSFER_TYPE_WRITE_REDUCE, dst, src, srcSlice.count_, sendRecvInfo.dataType_,
            sendRecvInfo.reduceType_));
        return HCCL_SUCCESS;
    };
    return DoSendRecvBatchTx(
        sendRecvInfo, thread, "SendRecvBatchWriteReduce", "BATCH_WRITE_REDUCE", processSlice, SendRecvWriteReduce,
        true);
}

HcclResult SendWriteReduce(const DataReduceInfo& sendInfo, const ThreadHandle& thread)
{
    const std::vector<DataSlice> srcSlices = sendInfo.slices_.srcSlices_;
    const std::vector<DataSlice> dstSlices = sendInfo.slices_.dstSlices_;
    const ChannelInfo& sendChannel = sendInfo.channel_;
    // 获取执行超时时间
    u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, sendChannel.handle, NOTIFY_IDX_ACK, execTimeout)));
    return RunWriteReduceAndNotify(
        thread, sendChannel, srcSlices, dstSlices, sendInfo.dataType_, sendInfo.reduceType_, "SendWriteReduce");
}

HcclResult SendBatchWriteReduce(const DataReduceInfo& sendInfo, const ThreadHandle& thread)
{
    auto processSlice = [&sendInfo](
                            int i, const DataSlice& srcSlice, const DataSlice& dstSlice,
                            std::vector<HcclHcommBatchTransferDesc>& transferDescs, u32 repeatNum) -> HcclResult {
        CHK_PRT_RET(
            srcSlice.count_ * DATATYPE_SIZE_TABLE[sendInfo.dataType_] != srcSlice.size_,
            HCCL_ERROR(
                "[AlgDataTransWrapper] SendBatchWriteReduce: src slice count [%u] is not mate to src slice "
                "size [%u], dataType is [%d].",
                srcSlice.count_, srcSlice.size_, sendInfo.dataType_),
            HcclResult::HCCL_E_INTERNAL);
        CHK_PRT_RET(
            dstSlice.count_ * DATATYPE_SIZE_TABLE[sendInfo.dataType_] != dstSlice.size_,
            HCCL_ERROR(
                "[AlgDataTransWrapper] SendBatchWriteReduce: dst slice count [%u] is not mate to dst slice "
                "size [%u], dataType is [%d].",
                dstSlice.count_, dstSlice.size_, sendInfo.dataType_),
            HcclResult::HCCL_E_INTERNAL);
        void* dst = GetSliceAddr(dstSlice);
        void* src = GetSliceAddr(srcSlice);
        u64 len = srcSlice.count_ * DATATYPE_SIZE_TABLE[sendInfo.dataType_];
        TraceDataSlice(
            "SendBatchWriteReduce", "BATCH_WRITE_REDUCE", i, repeatNum, srcSlice, dstSlice, src, dst, len,
            sendInfo.dataType_, sendInfo.reduceType_);
        transferDescs.push_back(MakeBatchReduceDesc(
            HCCL_HCOMM_TRANSFER_TYPE_WRITE_REDUCE, dst, src, srcSlice.count_, sendInfo.dataType_,
            sendInfo.reduceType_));
        return HCCL_SUCCESS;
    };
    return DoSendBatchTx(
        sendInfo, thread, "SendBatchWriteReduce", "BATCH_WRITE_REDUCE", processSlice, SendWriteReduce, true);
}

HcclResult RecvWriteReduce(const DataReduceInfo& recvInfo, const ThreadHandle& thread)
{
    const ChannelInfo& recvChannel = recvInfo.channel_;
    // 获取执行超时时间
    u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, recvChannel.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, recvChannel.handle, NOTIFY_IDX_DATA_SIGNAL, execTimeout)));
    return HCCL_SUCCESS;
}

HcclResult SendRecvWriteReduce(const SendRecvReduceInfo& sendRecvInfo, const ThreadHandle& thread)
{
    const std::vector<DataSlice> srcSlices = sendRecvInfo.sendRecvSlices_.txSlicesList_.srcSlices_;
    const std::vector<DataSlice> dstSlices = sendRecvInfo.sendRecvSlices_.txSlicesList_.dstSlices_;
    const ChannelInfo& sendChannel = sendRecvInfo.sendRecvChannels_.txChannel_;
    const ChannelInfo& recvChannel = sendRecvInfo.sendRecvChannels_.rxChannel_;
    // 向write rank发送tx同步，确保该rank的hcclBuffer可用
    // 这里只是在host上向device下任务，所以实际在host侧不会因为wait而阻塞
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, recvChannel.handle, NOTIFY_IDX_ACK)));
    // 获取执行超时时间
    u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, sendChannel.handle, NOTIFY_IDX_ACK, execTimeout)));
    // 写完之后做后同步告诉对面写完了
    CHK_RET(RunWriteReduceAndNotify(
        thread, sendChannel, srcSlices, dstSlices, sendRecvInfo.dataType_, sendRecvInfo.reduceType_,
        "SendRecvWriteReduce"));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, recvChannel.handle, NOTIFY_IDX_DATA_SIGNAL, execTimeout)));
    return HCCL_SUCCESS;
}

// [中文导读] Read协议的数据提供方：通知对端数据可读，再等对端读完。真正的数据Read由接收方发起。
HcclResult SendRead(const DataInfo& sendInfo, const ThreadHandle& thread)
{
    // [中文导读] 数据提供方发送可读 ACK，再等待读取方的完成通知；自身没有 Read 数据原语。
    const ChannelInfo& sendChannel = sendInfo.channel_;
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, sendChannel.handle, NOTIFY_IDX_ACK)));
    // 获取执行超时时间
    u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, sendChannel.handle, NOTIFY_IDX_DATA_SIGNAL, execTimeout)));
    return HCCL_SUCCESS;
}

// [中文导读] Read协议的接收方：等远端ACK后把远端src逐片读入本地dst，最后发DATA_SIGNAL允许对端继续。
HcclResult RecvRead(const DataInfo& recvInfo, const ThreadHandle& thread)
{
    const std::vector<DataSlice> srcSlices = recvInfo.slices_.srcSlices_;
    const std::vector<DataSlice> dstSlices = recvInfo.slices_.dstSlices_;
    const ChannelInfo& recvChannel = recvInfo.channel_;
    u32 repeatNum = srcSlices.size();
    // 获取执行超时时间
    // [中文导读] 接收方先等待远端数据可读，再逐片发起远端读取。
    u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, recvChannel.handle, NOTIFY_IDX_ACK, execTimeout)));
    for (int i = 0; i < repeatNum; i++) {
        const DataSlice srcSlice = srcSlices[i];
        const DataSlice dstSlice = dstSlices[i];
        if (srcSlice.size_ == 0) {
            HCCL_WARNING("[AlgDataTransWrapper] RecvRead: size is 0.");
            continue;
        }
        // [中文导读] Read 的源地址属于对端，目标地址属于本端；长度仍是数据片的字节容量。
        void* dst = GetSliceAddr(dstSlice);
        void* src = GetSliceAddr(srcSlice);
        TraceDataSlice(
            "RecvRead", "READ", i, repeatNum, srcSlice, dstSlice, src, dst, srcSlice.size_, recvInfo.dataType_,
            HcclReduceOp::HCCL_REDUCE_RESERVED);
        CHK_RET(static_cast<HcclResult>(HcommReadOnThread(thread, recvChannel.handle, dst, src, srcSlice.size_)));
    }
    // [中文导读] 所有读任务排入后通知对端读完，使对端可在依赖满足后复用它的数据区。
    CHK_RET(
        static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, recvChannel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    return HCCL_SUCCESS;
}

HcclResult RecvBatchRead(const DataInfo& recvInfo, const ThreadHandle& thread)
{
    auto processSlice = [&recvInfo](
                            int i, const DataSlice& srcSlice, const DataSlice& dstSlice,
                            std::vector<HcclHcommBatchTransferDesc>& transferDescs, u32 repeatNum) -> HcclResult {
        void* dst = GetSliceAddr(dstSlice);
        void* src = GetSliceAddr(srcSlice);
        TraceDataSlice(
            "RecvBatchRead", "BATCH_READ", i, repeatNum, srcSlice, dstSlice, src, dst, srcSlice.size_,
            recvInfo.dataType_, HcclReduceOp::HCCL_REDUCE_RESERVED);
        transferDescs.push_back(MakeBatchTransDesc(HCCL_HCOMM_TRANSFER_TYPE_READ, dst, src, srcSlice.size_));
        return HCCL_SUCCESS;
    };
    return DoRecvBatchRx(recvInfo, thread, "RecvBatchRead", "BATCH_READ", processSlice, RecvRead);
}

HcclResult SendRecvRead(const SendRecvInfo& sendRecvInfo, const ThreadHandle& thread)
{
    // [中文导读] 双向 Read 只遍历接收方向的片，发送方向由对端拉取本端数据。
    const std::vector<DataSlice> srcSlices = sendRecvInfo.sendRecvSlices_.rxSlicesList_.srcSlices_;
    const std::vector<DataSlice> dstSlices = sendRecvInfo.sendRecvSlices_.rxSlicesList_.dstSlices_;
    const ChannelInfo& sendChannel = sendRecvInfo.sendRecvChannels_.txChannel_;
    const ChannelInfo& recvChannel = sendRecvInfo.sendRecvChannels_.rxChannel_;
    u32 repeatNum = srcSlices.size();
    // 向read rank发送rx同步，确保该rank的hcclBuffer可用
    // 这里只是在host上向device下任务，所以实际在host侧不会因为wait而阻塞
    // [中文导读] 先让发送方向的对端读取本端，再等待接收方向对端的数据准备好。
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, sendChannel.handle, NOTIFY_IDX_ACK)));
    // 获取执行超时时间
    u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, recvChannel.handle, NOTIFY_IDX_ACK, execTimeout)));
    for (int i = 0; i < repeatNum; i++) {
        // rx同步完成后准备将数据从对方的hcclBuffer上读到自己的userIn上
        const DataSlice srcSlice = srcSlices[i];
        const DataSlice dstSlice = dstSlices[i];
        if (srcSlice.size_ == 0) {
            HCCL_WARNING("[AlgDataTransWrapper] SendRecvRead: size is 0.");
            continue;
        }
        void* dst = GetSliceAddr(dstSlice);
        void* src = GetSliceAddr(srcSlice);
        TraceDataSlice(
            "SendRecvRead", "READ", i, repeatNum, srcSlice, dstSlice, src, dst, srcSlice.size_, sendRecvInfo.dataType_,
            HcclReduceOp::HCCL_REDUCE_RESERVED);
        CHK_RET(static_cast<HcclResult>(HcommReadOnThread(thread, recvChannel.handle, dst, src, srcSlice.size_)));
    }
    // 写完之后做后同步告诉对面写完了
    CHK_RET(
        static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, recvChannel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    // [中文导读] 本端读完通知接收方通道对端后，还要等待发送方通道对端已读完本端数据。
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, sendChannel.handle, NOTIFY_IDX_DATA_SIGNAL, execTimeout)));
    return HCCL_SUCCESS;
}

HcclResult SendRecvBatchRead(const SendRecvInfo& sendRecvInfo, const ThreadHandle& thread)
{
    auto processSlice = [&sendRecvInfo](
                            int i, const DataSlice& srcSlice, const DataSlice& dstSlice,
                            std::vector<HcclHcommBatchTransferDesc>& transferDescs, u32 repeatNum) -> HcclResult {
        void* dst = GetSliceAddr(dstSlice);
        void* src = GetSliceAddr(srcSlice);
        TraceDataSlice(
            "SendRecvBatchRead", "BATCH_READ", i, repeatNum, srcSlice, dstSlice, src, dst, srcSlice.size_,
            sendRecvInfo.dataType_, HcclReduceOp::HCCL_REDUCE_RESERVED);
        transferDescs.push_back(MakeBatchTransDesc(HCCL_HCOMM_TRANSFER_TYPE_READ, dst, src, srcSlice.size_));
        return HCCL_SUCCESS;
    };
    return DoSendRecvBatchRx(sendRecvInfo, thread, "SendRecvBatchRead", "BATCH_READ", processSlice, SendRecvRead);
}

HcclResult SendReadReduce(const DataReduceInfo& sendInfo, const ThreadHandle& thread)
{
    const ChannelInfo& sendChannel = sendInfo.channel_;
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, sendChannel.handle, NOTIFY_IDX_ACK)));
    // 获取执行超时时间
    u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, sendChannel.handle, NOTIFY_IDX_DATA_SIGNAL, execTimeout)));
    return HCCL_SUCCESS;
}

HcclResult RecvReadReduce(const DataReduceInfo& recvInfo, const ThreadHandle& thread)
{
    const std::vector<DataSlice> srcSlices = recvInfo.slices_.srcSlices_;
    const std::vector<DataSlice> dstSlices = recvInfo.slices_.dstSlices_;
    const ChannelInfo& recvChannel = recvInfo.channel_;
    u32 repeatNum = srcSlices.size();
    // 获取执行超时时间
    u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, recvChannel.handle, NOTIFY_IDX_ACK, execTimeout)));
    for (int i = 0; i < repeatNum; i++) {
        const DataSlice srcSlice = srcSlices[i];
        const DataSlice dstSlice = dstSlices[i];
        if (srcSlice.size_ == 0) {
            HCCL_WARNING("[AlgDataTransWrapper] RecvReadReduce: size is 0.");
            continue;
        }
        CHK_PRT_RET(
            srcSlice.count_ * DATATYPE_SIZE_TABLE[recvInfo.dataType_] != srcSlice.size_,
            HCCL_ERROR(
                "[AlgDataTransWrapper] RecvReadReduce: src slice count [%u] is not mate to src slice size "
                "[%u], dataType is [%d].",
                srcSlice.count_, srcSlice.size_, recvInfo.dataType_),
            HcclResult::HCCL_E_INTERNAL);
        CHK_PRT_RET(
            dstSlice.count_ * DATATYPE_SIZE_TABLE[recvInfo.dataType_] != dstSlice.size_,
            HCCL_ERROR(
                "[AlgDataTransWrapper] RecvReadReduce: dst slice count [%u] is not mate to dst slice size "
                "[%u], dataType is [%d].",
                dstSlice.count_, dstSlice.size_, recvInfo.dataType_),
            HcclResult::HCCL_E_INTERNAL);
        void* dst = GetSliceAddr(dstSlice);
        void* src = GetSliceAddr(srcSlice);
        TraceDataSlice(
            "RecvReadReduce", "READ_REDUCE", i, repeatNum, srcSlice, dstSlice, src, dst, srcSlice.count_,
            recvInfo.dataType_, recvInfo.reduceType_);
        CHK_RET(static_cast<HcclResult>(HcommReadReduceOnThread(
            thread, recvChannel.handle, dst, src, srcSlice.count_, static_cast<HcommDataType>(recvInfo.dataType_),
            static_cast<HcommReduceOp>(recvInfo.reduceType_))));
    }
    CHK_RET(
        static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, recvChannel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    return HCCL_SUCCESS;
}

HcclResult RecvBatchReadReduce(const DataReduceInfo& recvInfo, const ThreadHandle& thread)
{
    auto processSlice = [&recvInfo](
                            int i, const DataSlice& srcSlice, const DataSlice& dstSlice,
                            std::vector<HcclHcommBatchTransferDesc>& transferDescs, u32 repeatNum) -> HcclResult {
        CHK_PRT_RET(
            srcSlice.count_ * DATATYPE_SIZE_TABLE[recvInfo.dataType_] != srcSlice.size_,
            HCCL_ERROR(
                "[AlgDataTransWrapper] RecvBatchReadReduce: src slice count [%u] is not mate to src slice "
                "size [%u], dataType is [%d].",
                srcSlice.count_, srcSlice.size_, recvInfo.dataType_),
            HcclResult::HCCL_E_INTERNAL);
        CHK_PRT_RET(
            dstSlice.count_ * DATATYPE_SIZE_TABLE[recvInfo.dataType_] != dstSlice.size_,
            HCCL_ERROR(
                "[AlgDataTransWrapper] RecvBatchReadReduce: dst slice count [%u] is not mate to dst slice "
                "size [%u], dataType is [%d].",
                dstSlice.count_, dstSlice.size_, recvInfo.dataType_),
            HcclResult::HCCL_E_INTERNAL);
        void* dst = GetSliceAddr(dstSlice);
        void* src = GetSliceAddr(srcSlice);
        u64 len = srcSlice.count_ * DATATYPE_SIZE_TABLE[recvInfo.dataType_];
        TraceDataSlice(
            "RecvBatchReadReduce", "BATCH_READ_REDUCE", i, repeatNum, srcSlice, dstSlice, src, dst, len,
            recvInfo.dataType_, recvInfo.reduceType_);
        transferDescs.push_back(MakeBatchReduceDesc(
            HCCL_HCOMM_TRANSFER_TYPE_READ_REDUCE, dst, src, srcSlice.count_, recvInfo.dataType_, recvInfo.reduceType_));
        return HCCL_SUCCESS;
    };
    return DoRecvBatchRx(recvInfo, thread, "RecvBatchReadReduce", "BATCH_READ_REDUCE", processSlice, RecvReadReduce);
}

// [中文导读] [AllReduce逐行 S826] 定义 SendRecvReadReduce 入口：逐片把远端输入读并归约到本端输出，执行相应双向 ACK/DATA_SIGNAL 握手。
HcclResult SendRecvReadReduce(const SendRecvReduceInfo& sendRecvInfo, const ThreadHandle& thread)
// [中文导读] [AllReduce逐行 S827] 进入 SendRecvReadReduce 的实现作用域；逐片把远端输入读并归约到本端输出，执行相应双向 ACK/DATA_SIGNAL 握手。
{
    // [中文导读] [AllReduce逐行 S828] 设置 const std::vector<DataSlice> 源数据片列表 为 sendRecvInfo.sendRecvSlices_.rxSlicesList_.源数据片列表_；该值供下方当前分支使用。
    const std::vector<DataSlice> srcSlices = sendRecvInfo.sendRecvSlices_.rxSlicesList_.srcSlices_;
    // [中文导读] [AllReduce逐行 S829] 设置 const std::vector<DataSlice> 目标数据片列表 为 sendRecvInfo.sendRecvSlices_.rxSlicesList_.目标数据片列表_；该值供下方当前分支使用。
    const std::vector<DataSlice> dstSlices = sendRecvInfo.sendRecvSlices_.rxSlicesList_.dstSlices_;
    // [中文导读] [AllReduce逐行 S830] 设置 const ChannelInfo& sendChannel 为 sendRecvInfo.sendRecvChannels_.txChannel_；该值供下方当前分支使用。
    const ChannelInfo& sendChannel = sendRecvInfo.sendRecvChannels_.txChannel_;
    // [中文导读] [AllReduce逐行 S831] 设置 const ChannelInfo& recvChannel 为 sendRecvInfo.sendRecvChannels_.rxChannel_；该值供下方当前分支使用。
    const ChannelInfo& recvChannel = sendRecvInfo.sendRecvChannels_.rxChannel_;
    // [中文导读] [AllReduce逐行 S832] 设置 待处理数据片数 为 源数据片列表.size()；该值供下方当前分支使用。
    u32 repeatNum = srcSlices.size();
    // 向write rank发送tx同步，确保该rank的hcclBuffer可用
    // 这里只是在host上向device下任务，所以实际在host侧不会因为wait而阻塞
    // [中文导读] [AllReduce逐行 S835] 在 Peer 通道上记录远端可观察的通知；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, sendChannel.handle, NOTIFY_IDX_ACK)));
    // 获取执行超时时间
    // [中文导读] [AllReduce逐行 S837] 设置 当前实际执行配置Timeout 为 ExecTimeoutManager::Instance().GetExecTimeout()；该值供下方当前分支使用。
    u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
    // [中文导读] [AllReduce逐行 S838] 执行下方完整子调用；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(static_cast<HcclResult>(
        // [中文导读] [AllReduce逐行 S839] 等待本端通道通知槽满足依赖；本行实参为 HcommChannelNotifyWaitOnThread(thread, 接收 Peer 的通道句柄, ACK 通知槽, 当前实际执行配置Timeout)))。
        HcommChannelNotifyWaitOnThread(thread, recvChannel.handle, NOTIFY_IDX_ACK, execTimeout)));
    // [中文导读] [AllReduce逐行 S840] 逐片处理对应源/目标数据片，零长度片由块内检查跳过；边界/迭代规则为 (int i = 0; i 小于 待处理数据片数; i++。
    for (int i = 0; i < repeatNum; i++) {
        // tx同步完成后准备将自己的userIn上的数据写到对方的hcclBuffer上
        // [中文导读] [AllReduce逐行 S842] 设置 srcSlice 为 源数据片列表[i]；该值供下方当前分支使用。
        const DataSlice srcSlice = srcSlices[i];
        // [中文导读] [AllReduce逐行 S843] 设置 dstSlice 为 目标数据片列表[i]；该值供下方当前分支使用。
        const DataSlice dstSlice = dstSlices[i];
        // [中文导读] [AllReduce逐行 S844] 分支条件为 源片字节数 等于 0；成立进入本块，未成立继续后续分支。
        if (srcSlice.size_ == 0) {
            // [中文导读] [AllReduce逐行 S845] 开始 HCCL_WARNING 诊断输出，记录 SendRecvReadReduce 当前阶段的参数、候选或错误；日志本身不决定返回码。
            HCCL_WARNING("[AlgDataTransWrapper] SendRecvReadReduce: size is 0.");
            // [中文导读] [AllReduce逐行 S846] 跳过当前遍历项的剩余步骤，直接处理下一项。
            continue;
        // [中文导读] [AllReduce逐行 S847] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
        // [中文导读] [AllReduce逐行 S848] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
        CHK_PRT_RET(
            // [中文导读] [AllReduce逐行 S849] 续接本次错误检查/子调用实参：源片元素数 * 类型到元素字节数的查找表[sendRecvInfo.当前元素类型] 不等于 源片字节数；返回行为由所在完整宏决定。
            srcSlice.count_ * DATATYPE_SIZE_TABLE[sendRecvInfo.dataType_] != srcSlice.size_,
            // [中文导读] [AllReduce逐行 S850] 开始 HCCL_ERROR 诊断输出，记录 SendRecvReadReduce 当前阶段的参数、候选或错误；日志本身不决定返回码。
            HCCL_ERROR(
                // [中文导读] [AllReduce逐行 S851] 续接 SendRecvReadReduce 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
                "[AlgDataTransWrapper] SendRecvReadReduce: src slice count [%u] is not mate to src slice size "
                // [中文导读] [AllReduce逐行 S852] 续接 SendRecvReadReduce 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
                "[%u], dataType is [%d].",
                // [中文导读] [AllReduce逐行 S853] 为 SendRecvReadReduce 的诊断/错误宏提供实参：源片元素数, 源片字节数, sendRecvInfo.当前元素类型，与前面的格式占位依次对应。
                srcSlice.count_, srcSlice.size_, sendRecvInfo.dataType_),
            // [中文导读] [AllReduce逐行 S854] 向条件返回宏提供 HcclResult::内部错误；上方检查成立才退出当前函数。
            HcclResult::HCCL_E_INTERNAL);
        // [中文导读] [AllReduce逐行 S855] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
        CHK_PRT_RET(
            // [中文导读] [AllReduce逐行 S856] 续接本次错误检查/子调用实参：目标片元素数 * 类型到元素字节数的查找表[sendRecvInfo.当前元素类型] 不等于 目标片字节数；返回行为由所在完整宏决定。
            dstSlice.count_ * DATATYPE_SIZE_TABLE[sendRecvInfo.dataType_] != dstSlice.size_,
            // [中文导读] [AllReduce逐行 S857] 开始 HCCL_ERROR 诊断输出，记录 SendRecvReadReduce 当前阶段的参数、候选或错误；日志本身不决定返回码。
            HCCL_ERROR(
                // [中文导读] [AllReduce逐行 S858] 续接 SendRecvReadReduce 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
                "[AlgDataTransWrapper] SendRecvReadReduce: dst slice count [%u] is not mate to dst slice size "
                // [中文导读] [AllReduce逐行 S859] 续接 SendRecvReadReduce 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
                "[%u], dataType is [%d].",
                // [中文导读] [AllReduce逐行 S860] 为 SendRecvReadReduce 的诊断/错误宏提供实参：目标片元素数, 目标片字节数, sendRecvInfo.当前元素类型，与前面的格式占位依次对应。
                dstSlice.count_, dstSlice.size_, sendRecvInfo.dataType_),
            // [中文导读] [AllReduce逐行 S861] 向条件返回宏提供 HcclResult::内部错误；上方检查成立才退出当前函数。
            HcclResult::HCCL_E_INTERNAL);
        // [中文导读] [AllReduce逐行 S862] 设置 dst 为 GetSliceAddr(dstSlice)；该值供下方当前分支使用。
        void* dst = GetSliceAddr(dstSlice);
        // [中文导读] [AllReduce逐行 S863] 设置 src 为 GetSliceAddr(srcSlice)；该值供下方当前分支使用。
        void* src = GetSliceAddr(srcSlice);
        // [中文导读] [AllReduce逐行 S864] 记录源目标基址、偏移、实际地址和类型信息；本行实参为 TraceDataSlice(。
        TraceDataSlice(
            // [中文导读] [AllReduce逐行 S865] 续接本次调用的字符串常量 "SendRecvReadReduce", "READ_REDUCE", i, repeatNum, srcSlice, dstSlice, src, dst, srcSlice.count_,，由所在注册/日志/条件语句整体使用。
            "SendRecvReadReduce", "READ_REDUCE", i, repeatNum, srcSlice, dstSlice, src, dst, srcSlice.count_,
            // [中文导读] [AllReduce逐行 S866] 续接 SendRecvReadReduce 当前语句的具体实参/字段：sendRecvInfo.当前元素类型, sendRecvInfo.reduceType_)；由其完整表达式完成参数组装、检查或结果写回。
            sendRecvInfo.dataType_, sendRecvInfo.reduceType_);
        // [中文导读] [AllReduce逐行 S867] 把远端源片读入并归约到本地目标片；返回值非成功时立即从当前函数返回该错误。
        CHK_RET(static_cast<HcclResult>(HcommReadReduceOnThread(
            // [中文导读] [AllReduce逐行 S868] 续接本次错误检查/子调用实参：thread, 接收 Peer 的通道句柄, dst, src, 源片元素数, static_cast<HcommDataType>(sendRecvInfo.当前元素类型；返回行为由所在完整宏决定。
            thread, recvChannel.handle, dst, src, srcSlice.count_, static_cast<HcommDataType>(sendRecvInfo.dataType_),
            // [中文导读] [AllReduce逐行 S869] 续接本次错误检查/子调用实参：static_cast<HcommReduceOp>(sendRecvInfo.reduceType_；返回行为由所在完整宏决定。
            static_cast<HcommReduceOp>(sendRecvInfo.reduceType_))));
    // [中文导读] [AllReduce逐行 S870] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // 写完之后做后同步告诉对面写完了
    // [中文导读] [AllReduce逐行 S872] 执行下方完整子调用；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(
        // [中文导读] [AllReduce逐行 S873] 在 Peer 通道上记录远端可观察的通知；本行实参为 static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, 接收 Peer 的通道句柄, 数据完成通知槽)))。
        static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, recvChannel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    // [中文导读] [AllReduce逐行 S874] 执行下方完整子调用；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(static_cast<HcclResult>(
        // [中文导读] [AllReduce逐行 S875] 等待本端通道通知槽满足依赖；本行实参为 HcommChannelNotifyWaitOnThread(thread, 发送 Peer 的通道句柄, 数据完成通知槽, 当前实际执行配置Timeout)))。
        HcommChannelNotifyWaitOnThread(thread, sendChannel.handle, NOTIFY_IDX_DATA_SIGNAL, execTimeout)));
    // [中文导读] [AllReduce逐行 S876] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S877] 结束 SendRecvReadReduce 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S879] 定义 SendRecvBatchReadReduce 入口：校验每片 count×类型大小等于字节容量，构造 READ_REDUCE 批描述；无批能力转逐片 ReadReduce。
HcclResult SendRecvBatchReadReduce(const SendRecvReduceInfo& sendRecvInfo, const ThreadHandle& thread)
// [中文导读] [AllReduce逐行 S880] 进入 SendRecvBatchReadReduce 的实现作用域；校验每片 count×类型大小等于字节容量，构造 READ_REDUCE 批描述；无批能力转逐片 ReadReduce。
{
    // [中文导读] [AllReduce逐行 S881] 设置 processSlice 为 [&sendRecvInfo](；该值供下方当前分支使用。
    auto processSlice = [&sendRecvInfo](
                            // [中文导读] [AllReduce逐行 S882] 续接 SendRecvBatchReadReduce 当前语句的具体实参/字段：int i, const DataSlice& srcSlice, const DataSlice& dstSlice,；由其完整表达式完成参数组装、检查或结果写回。
                            int i, const DataSlice& srcSlice, const DataSlice& dstSlice,
                            // [中文导读] [AllReduce逐行 S883] 建立本阶段局部对象 std::vector<HcclHcommBatchTransferDesc>& 有效批传输描述列表, u32 待处理数据片数) -> HcclResult {，供 SendRecvBatchReadReduce 下方参数组装和子调用使用。
                            std::vector<HcclHcommBatchTransferDesc>& transferDescs, u32 repeatNum) -> HcclResult {
        // [中文导读] [AllReduce逐行 S884] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
        CHK_PRT_RET(
            // [中文导读] [AllReduce逐行 S885] 续接本次错误检查/子调用实参：源片元素数 * 类型到元素字节数的查找表[sendRecvInfo.当前元素类型] 不等于 源片字节数；返回行为由所在完整宏决定。
            srcSlice.count_ * DATATYPE_SIZE_TABLE[sendRecvInfo.dataType_] != srcSlice.size_,
            // [中文导读] [AllReduce逐行 S886] 开始 HCCL_ERROR 诊断输出，记录 SendRecvBatchReadReduce 当前阶段的参数、候选或错误；日志本身不决定返回码。
            HCCL_ERROR(
                // [中文导读] [AllReduce逐行 S887] 续接 SendRecvBatchReadReduce 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
                "[AlgDataTransWrapper] SendRecvBatchReadReduce: src slice count [%u] is not mate to src slice "
                // [中文导读] [AllReduce逐行 S888] 续接 SendRecvBatchReadReduce 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
                "size [%u], dataType is [%d].",
                // [中文导读] [AllReduce逐行 S889] 为 SendRecvBatchReadReduce 的诊断/错误宏提供实参：源片元素数, 源片字节数, sendRecvInfo.当前元素类型，与前面的格式占位依次对应。
                srcSlice.count_, srcSlice.size_, sendRecvInfo.dataType_),
            // [中文导读] [AllReduce逐行 S890] 向条件返回宏提供 HcclResult::内部错误；上方检查成立才退出当前函数。
            HcclResult::HCCL_E_INTERNAL);
        // [中文导读] [AllReduce逐行 S891] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
        CHK_PRT_RET(
            // [中文导读] [AllReduce逐行 S892] 续接本次错误检查/子调用实参：目标片元素数 * 类型到元素字节数的查找表[sendRecvInfo.当前元素类型] 不等于 目标片字节数；返回行为由所在完整宏决定。
            dstSlice.count_ * DATATYPE_SIZE_TABLE[sendRecvInfo.dataType_] != dstSlice.size_,
            // [中文导读] [AllReduce逐行 S893] 开始 HCCL_ERROR 诊断输出，记录 SendRecvBatchReadReduce 当前阶段的参数、候选或错误；日志本身不决定返回码。
            HCCL_ERROR(
                // [中文导读] [AllReduce逐行 S894] 续接 SendRecvBatchReadReduce 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
                "[AlgDataTransWrapper] SendRecvBatchReadReduce: dst slice count [%u] is not mate to dst slice "
                // [中文导读] [AllReduce逐行 S895] 续接 SendRecvBatchReadReduce 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
                "size [%u], dataType is [%d].",
                // [中文导读] [AllReduce逐行 S896] 为 SendRecvBatchReadReduce 的诊断/错误宏提供实参：目标片元素数, 目标片字节数, sendRecvInfo.当前元素类型，与前面的格式占位依次对应。
                dstSlice.count_, dstSlice.size_, sendRecvInfo.dataType_),
            // [中文导读] [AllReduce逐行 S897] 向条件返回宏提供 HcclResult::内部错误；上方检查成立才退出当前函数。
            HcclResult::HCCL_E_INTERNAL);
        // [中文导读] [AllReduce逐行 S898] 设置 dst 为 GetSliceAddr(dstSlice)；该值供下方当前分支使用。
        void* dst = GetSliceAddr(dstSlice);
        // [中文导读] [AllReduce逐行 S899] 设置 src 为 GetSliceAddr(srcSlice)；该值供下方当前分支使用。
        void* src = GetSliceAddr(srcSlice);
        // [中文导读] [AllReduce逐行 S900] 设置 len 为 源片元素数 * 类型到元素字节数的查找表[sendRecvInfo.当前元素类型]；该值供下方当前分支使用。
        u64 len = srcSlice.count_ * DATATYPE_SIZE_TABLE[sendRecvInfo.dataType_];
        // [中文导读] [AllReduce逐行 S901] 记录源目标基址、偏移、实际地址和类型信息；本行实参为 TraceDataSlice(。
        TraceDataSlice(
            // [中文导读] [AllReduce逐行 S902] 续接本次调用的字符串常量 "SendRecvBatchReadReduce", "BATCH_READ_REDUCE", i, repeatNum, srcSlice, dstSlice, src, dst, len,，由所在注册/日志/条件语句整体使用。
            "SendRecvBatchReadReduce", "BATCH_READ_REDUCE", i, repeatNum, srcSlice, dstSlice, src, dst, len,
            // [中文导读] [AllReduce逐行 S903] 续接 SendRecvBatchReadReduce 当前语句的具体实参/字段：sendRecvInfo.当前元素类型, sendRecvInfo.reduceType_)；由其完整表达式完成参数组装、检查或结果写回。
            sendRecvInfo.dataType_, sendRecvInfo.reduceType_);
        // [中文导读] [AllReduce逐行 S904] 对 有效批传输描述列表 追加 MakeBatchReduceDesc，准备或更新本阶段列表。
        transferDescs.push_back(MakeBatchReduceDesc(
            // [中文导读] [AllReduce逐行 S905] 续接 SendRecvBatchReadReduce 当前语句的具体实参/字段：远端 READ_REDUCE 类型, dst, src, 源片元素数, sendRecvInfo.当前元素类型,；由其完整表达式完成参数组装、检查或结果写回。
            HCCL_HCOMM_TRANSFER_TYPE_READ_REDUCE, dst, src, srcSlice.count_, sendRecvInfo.dataType_,
            // [中文导读] [AllReduce逐行 S906] 续接 SendRecvBatchReadReduce 当前语句的具体实参/字段：sendRecvInfo.reduceType_))；由其完整表达式完成参数组装、检查或结果写回。
            sendRecvInfo.reduceType_));
        // [中文导读] [AllReduce逐行 S907] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S908] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    };
    // [中文导读] [AllReduce逐行 S909] 直接返回 调用 DoSendRecvBatchRx 完成当前参数所指的子步骤 的结果，调用者取得本分支结果。
    return DoSendRecvBatchRx(
        // [中文导读] [AllReduce逐行 S910] 续接 SendRecvBatchReadReduce 当前语句的具体实参/字段：sendRecvInfo, thread, "SendRecvBatchReadReduce", "BATCH_READ_REDUCE", processSlice, SendRecvReadReduce)；由其完整表达式完成参数组装、检查或结果写回。
        sendRecvInfo, thread, "SendRecvBatchReadReduce", "BATCH_READ_REDUCE", processSlice, SendRecvReadReduce);
// [中文导读] [AllReduce逐行 S911] 结束 SendRecvBatchReadReduce 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] 本地片间拷贝：检查非零且等长，地址为base+offset，然后排入HcommLocalCopyOnThread。
// [中文导读] 可用于自身Rank数据或CCL中转区与用户区之间的搬运；没有跨Rank的Channel参数。
// [中文导读] [AllReduce逐行 S915] 定义 LocalCopy 入口：零长片直接成功，非零片要求等长，再将本地源片复制到本地目标片。
HcclResult LocalCopy(const ThreadHandle& thread, const DataSlice& srcSlice, const DataSlice& dstSlice)
// [中文导读] [AllReduce逐行 S916] 进入 LocalCopy 的实现作用域；零长片直接成功，非零片要求等长，再将本地源片复制到本地目标片。
{
    // [中文导读] 零字节本地片无需创建复制任务；非零时还必须检查源目标字节长度相等。
    // [中文导读] [AllReduce逐行 S918] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S919] 为 LocalCopy 的诊断/错误宏提供实参：源片字节数 等于 0, HCCL_WARNING("[AlgDataTransWrapper] LocalCopy: src slice size is [%u].", 源片字节数，与前面的格式占位依次对应。
        srcSlice.size_ == 0, HCCL_WARNING("[AlgDataTransWrapper] LocalCopy: src slice size is [%u].", srcSlice.size_),
        // [中文导读] [AllReduce逐行 S920] 为 LocalCopy 的诊断/错误宏提供实参：HcclResult::成功状态，与前面的格式占位依次对应。
        HcclResult::HCCL_SUCCESS);

    // [中文导读] [AllReduce逐行 S922] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S923] 续接本次错误检查/子调用实参：源片字节数 不等于 目标片字节数；返回行为由所在完整宏决定。
        srcSlice.size_ != dstSlice.size_,
        // [中文导读] [AllReduce逐行 S924] 开始 HCCL_ERROR 诊断输出，记录 LocalCopy 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_ERROR(
            // [中文导读] [AllReduce逐行 S925] 续接 LocalCopy 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "[AlgDataTransWrapper] LocalCopy: src slice size [%u] is not equal to dst slice size [%u].", srcSlice.size_,
            // [中文导读] [AllReduce逐行 S926] 为 LocalCopy 的诊断/错误宏提供实参：目标片字节数，与前面的格式占位依次对应。
            dstSlice.size_),
        // [中文导读] [AllReduce逐行 S927] 向条件返回宏提供 HcclResult::内部错误；上方检查成立才退出当前函数。
        HcclResult::HCCL_E_INTERNAL);
    // [中文导读] 计算本地两片的有效地址后提交普通复制，不涉及远端 Channel 或网络握手。
    // [中文导读] [AllReduce逐行 S929] 设置 srcIn 为 GetSliceAddr(srcSlice)；该值供下方当前分支使用。
    void* srcIn = GetSliceAddr(srcSlice);
    // [中文导读] [AllReduce逐行 S930] 设置 dstOut 为 GetSliceAddr(dstSlice)；该值供下方当前分支使用。
    void* dstOut = GetSliceAddr(dstSlice);
    // [中文导读] [AllReduce逐行 S931] 记录源目标基址、偏移、实际地址和类型信息；本行实参为 TraceDataSlice(。
    TraceDataSlice(
        // [中文导读] [AllReduce逐行 S932] 续接本次调用的字符串常量 "LocalCopy", "LOCAL_COPY", 0, 1, srcSlice, dstSlice, srcIn, dstOut, srcSlice.size_, HCCL_DATA_TYPE_RESERVED,，由所在注册/日志/条件语句整体使用。
        "LocalCopy", "LOCAL_COPY", 0, 1, srcSlice, dstSlice, srcIn, dstOut, srcSlice.size_, HCCL_DATA_TYPE_RESERVED,
        // [中文导读] [AllReduce逐行 S933] 续接 LocalCopy 当前语句的具体实参/字段：HcclReduceOp::HCCL_REDUCE_RESERVED)；由其完整表达式完成参数组装、检查或结果写回。
        HcclReduceOp::HCCL_REDUCE_RESERVED);
    // [中文导读] [AllReduce逐行 S934] 在 Thread 上提交按字节长度的本地拷贝；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, dstOut, srcIn, srcSlice.size_)));
    // [中文导读] [AllReduce逐行 S935] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S936] 结束 LocalCopy 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] 本地归约和普通拷贝不同：HcommLocalReduceOnThread接收元素count及类型/归约运算。
// [中文导读] 指定64位类型或PROD在本包装层转入AicpuReduce，不能假定所有类型都落到同一硬件归约原语。
// [中文导读] [AllReduce逐行 S940] 定义 LocalReduce 入口：64 位类型或 PROD 使用 AICPU 软件归约，普通类型按元素 count 提交本地归约原语。
HcclResult LocalReduce(
    // [中文导读] [AllReduce逐行 S941] 续接 LocalReduce 的入口参数/基类初始化：const ThreadHandle& thread, const DataSlice& srcSlice, const DataSlice& dstSlice, const HcclDataType dataType,；引用参数按声明的 const 限制读写。
    const ThreadHandle& thread, const DataSlice& srcSlice, const DataSlice& dstSlice, const HcclDataType dataType,
    // [中文导读] [AllReduce逐行 S942] 续接 LocalReduce 的入口参数/基类初始化：const HcclReduceOp reduceOp)；引用参数按声明的 const 限制读写。
    const HcclReduceOp reduceOp)
// [中文导读] [AllReduce逐行 S943] 进入 LocalReduce 的实现作用域；64 位类型或 PROD 使用 AICPU 软件归约，普通类型按元素 count 提交本地归约原语。
{
    // [中文导读] 指定 64 位类型或乘积运算使用 AICPU 软件归约，避开这里的硬件本地归约原语。
    // [中文导读] [AllReduce逐行 S945] 分支条件为 dataType 等于 有符号 64 位整数 或 dataType 等于 无符号 64 位整数 或 dataType 等于 FP64 类型；成立进入本块，未成立继续后续分支。
    if (dataType == HCCL_DATA_TYPE_INT64 || dataType == HCCL_DATA_TYPE_UINT64 || dataType == HCCL_DATA_TYPE_FP64
        // [中文导读] [AllReduce逐行 S946] 补充同一条件的 或者 子条件：reduceOp 等于 HcclReduceOp::乘积归约。
        || reduceOp == HcclReduceOp::HCCL_REDUCE_PROD) {
        // [中文导读] [AllReduce逐行 S947] 对特殊类型/运算执行 CPU 软件归约；返回值非成功时立即从当前函数返回该错误。
        CHK_RET(AicpuReduce(thread, srcSlice, dstSlice, dataType, reduceOp));
        // [中文导读] [AllReduce逐行 S948] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S949] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S950] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S951] 为 LocalReduce 的诊断/错误宏提供实参：源片字节数 等于 0, HCCL_WARNING("[AlgDataTransWrapper] LocalReduce: src slice size is [%u].", 源片字节数，与前面的格式占位依次对应。
        srcSlice.size_ == 0, HCCL_WARNING("[AlgDataTransWrapper] LocalReduce: src slice size is [%u].", srcSlice.size_),
        // [中文导读] [AllReduce逐行 S952] 为 LocalReduce 的诊断/错误宏提供实参：HcclResult::成功状态，与前面的格式占位依次对应。
        HcclResult::HCCL_SUCCESS);

    // [中文导读] [AllReduce逐行 S954] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S955] 续接本次错误检查/子调用实参：源片字节数 不等于 目标片字节数；返回行为由所在完整宏决定。
        srcSlice.size_ != dstSlice.size_,
        // [中文导读] [AllReduce逐行 S956] 开始 HCCL_ERROR 诊断输出，记录 LocalReduce 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_ERROR(
            // [中文导读] [AllReduce逐行 S957] 续接 LocalReduce 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "[InsCollAlgFactory] LocalReduce: src slice size [%u] is not equal to dst slice size [%u].", srcSlice.size_,
            // [中文导读] [AllReduce逐行 S958] 为 LocalReduce 的诊断/错误宏提供实参：目标片字节数，与前面的格式占位依次对应。
            dstSlice.size_),
        // [中文导读] [AllReduce逐行 S959] 向条件返回宏提供 HcclResult::内部错误；上方检查成立才退出当前函数。
        HcclResult::HCCL_E_INTERNAL);
    // [中文导读] [AllReduce逐行 S960] 设置 src 为 GetSliceAddr(srcSlice)；该值供下方当前分支使用。
    void* src = GetSliceAddr(srcSlice);
    // [中文导读] [AllReduce逐行 S961] 设置 dst 为 GetSliceAddr(dstSlice)；该值供下方当前分支使用。
    void* dst = GetSliceAddr(dstSlice);
    // [中文导读] [AllReduce逐行 S962] 记录源目标基址、偏移、实际地址和类型信息；本行实参为 TraceDataSlice(。
    TraceDataSlice(
        // [中文导读] [AllReduce逐行 S963] 续接本次调用的字符串常量 "LocalReduce", "LOCAL_REDUCE", 0, 1, srcSlice, dstSlice, src, dst, srcSlice.count_, dataType, reduceOp);，由所在注册/日志/条件语句整体使用。
        "LocalReduce", "LOCAL_REDUCE", 0, 1, srcSlice, dstSlice, src, dst, srcSlice.count_, dataType, reduceOp);
    // [中文导读] 本地归约按元素 count 和类型解释数据，将源元素合并到目标元素。
    // [中文导读] [AllReduce逐行 S965] 在 Thread 上提交按元素数量的本地归约；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
        // [中文导读] [AllReduce逐行 S966] 续接本次错误检查/子调用实参：thread, dst, src, 源片元素数, static_cast<HcommDataType>(dataType；返回行为由所在完整宏决定。
        thread, dst, src, srcSlice.count_, static_cast<HcommDataType>(dataType),
        // [中文导读] [AllReduce逐行 S967] 续接本次错误检查/子调用实参：static_cast<HcommReduceOp>(reduceOp；返回行为由所在完整宏决定。
        static_cast<HcommReduceOp>(reduceOp))));
    // [中文导读] [AllReduce逐行 S968] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S969] 结束 LocalReduce 实现；其返回状态或已写回字段由调用者接收。
}

HcclResult LocalCopySlices(
    const ThreadHandle& thread, const std::vector<DataSlice>& srcSlices, const std::vector<DataSlice>& dstSlices)
{
    // [中文导读] 先保证源目标片数量一一对应，再要求至少一对片，避免访问不存在的首片。
    CHK_PRT_RET(
        srcSlices.size() != dstSlices.size(),
        HCCL_ERROR(
            "[InsCollAlgFactory] [AlgDataTrans] LocalCopySlices: num of src slices [%u], is not equal "
            "to num of dst slices [%u].",
            srcSlices.size(), dstSlices.size()),
        HcclResult::HCCL_E_INTERNAL);

    // 两数组长度必然相等，只检查先使用的srcSlices是否空数组。
    CHK_PRT_RET(
        srcSlices.empty(), HCCL_ERROR("[InsCollAlgFactory] [AlgDataTransWrapper] LocalCopySlices: srcSlices is empty."),
        HcclResult::HCCL_E_INTERNAL);

    // tmpSlices: slices to be transfer in this loop
    // [中文导读] 用首片作为待合并的复制区间，后面的连续片可以扩展它以减少原语次数。
    DataSlice tmpSrcSlice = srcSlices[0];
    DataSlice tmpDstSlice = dstSlices[0];

    for (u32 sliceIdx = 0; sliceIdx < srcSlices.size(); sliceIdx++) {
        if (srcSlices[sliceIdx].size_ == 0) {
            HCCL_WARNING("[AlgDataTransWrapper] LocalCopySlices: size is 0.");
            continue;
        }
        TraceDataSlice(
            "LocalCopySlices", "LOCAL_COPY_SLICE", sliceIdx, srcSlices.size(), srcSlices[sliceIdx], dstSlices[sliceIdx],
            GetSliceAddr(srcSlices[sliceIdx]), GetSliceAddr(dstSlices[sliceIdx]), srcSlices[sliceIdx].size_,
            HCCL_DATA_TYPE_RESERVED, HcclReduceOp::HCCL_REDUCE_RESERVED);
        CHK_PRT_RET(
            srcSlices[sliceIdx].size_ != dstSlices[sliceIdx].size_,
            HCCL_ERROR(
                "[InsCollAlgFactory] [AlgDataTransWrapper] LocalCopySlices: [%u]-th slice, src slice size [%u] "
                "is not equal to dst slice size [%u].",
                sliceIdx, srcSlices[sliceIdx].size_, dstSlices[sliceIdx].size_),
            HcclResult::HCCL_E_INTERNAL);

        // [中文导读] 遍历到最后一片时提交累计区间，结束本次片列表的复制安排。
        if (sliceIdx == (srcSlices.size() - 1)) {
            // last slice
            void* src = GetSliceAddr(tmpSrcSlice);
            void* dst = GetSliceAddr(tmpDstSlice);
            TraceDataSlice(
                "LocalCopySlices", "LOCAL_COPY_MERGED", sliceIdx, srcSlices.size(), tmpSrcSlice, tmpDstSlice, src, dst,
                tmpSrcSlice.size_, HCCL_DATA_TYPE_RESERVED, HcclReduceOp::HCCL_REDUCE_RESERVED);
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, dst, src, tmpSrcSlice.size_)));
        } else if (
            // [中文导读] 只有源与目标两侧都同基址且首尾相接时才扩大累计区间，保留片间映射关系。
            IsContinuousSlice(srcSlices[sliceIdx + 1], tmpSrcSlice)
            && IsContinuousSlice(dstSlices[sliceIdx + 1], tmpDstSlice)) {
            // nxtSlice is continuous with tmpSlice, update tmpSlice
            u64 newTmpSize = tmpSrcSlice.size_ + srcSlices[sliceIdx + 1].size_;
            tmpSrcSlice = DataSlice(tmpSrcSlice.addr_, tmpSrcSlice.offset_, newTmpSize);
            tmpDstSlice = DataSlice(tmpDstSlice.addr_, tmpDstSlice.offset_, newTmpSize);
        } else {
            // nxtSlice is not continuous with tmpSlice, copy tmpSlice, update tmpSlice with nxtSlice
            void* src = GetSliceAddr(tmpSrcSlice);
            void* dst = GetSliceAddr(tmpDstSlice);
            TraceDataSlice(
                "LocalCopySlices", "LOCAL_COPY_MERGED", sliceIdx, srcSlices.size(), tmpSrcSlice, tmpDstSlice, src, dst,
                tmpSrcSlice.size_, HCCL_DATA_TYPE_RESERVED, HcclReduceOp::HCCL_REDUCE_RESERVED);
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, dst, src, tmpSrcSlice.size_)));

            // [中文导读] 遇到不连续的下一片时，先提交已有区间，再以下一片重新开始累计。
            tmpSrcSlice = srcSlices[sliceIdx + 1];
            tmpDstSlice = dstSlices[sliceIdx + 1];
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

bool IsContinuousSlice(const DataSlice& nxtSlice, const DataSlice& currSlice)
{
    if (nxtSlice.addr_ != currSlice.addr_) {
        return false;
    }
    if (nxtSlice.offset_ != currSlice.offset_ + currSlice.size_) {
        return false;
    }
    return true;
}

// [中文导读] 主从Thread前同步：主Thread分别Record，每个从Thread在自己的队列Wait，形成并行工作起点。
// [中文导读] [AllReduce逐行 S1057] 定义 PreSyncInterThreads 入口：验证从 Thread 与通知索引列表等长非空，主流 Record、各从流 Wait 建立前同步。
HcclResult PreSyncInterThreads(
    // [中文导读] [AllReduce逐行 S1058] 续接 PreSyncInterThreads 的入口参数/基类初始化：const ThreadHandle& 主 Thread, const std::vector<ThreadHandle>& 从 Thread 列表,；引用参数按声明的 const 限制读写。
    const ThreadHandle& mainThread, const std::vector<ThreadHandle>& subThreads,
    // [中文导读] [AllReduce逐行 S1059] 续接 PreSyncInterThreads 的入口参数/基类初始化：const std::vector<u32>& notifyIdxMainToSub)；引用参数按声明的 const 限制读写。
    const std::vector<u32>& notifyIdxMainToSub)
// [中文导读] [AllReduce逐行 S1060] 进入 PreSyncInterThreads 的实现作用域；验证从 Thread 与通知索引列表等长非空，主流 Record、各从流 Wait 建立前同步。
{
    // [中文导读] [AllReduce逐行 S1061] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
    CHK_PRT_RET(
        // [中文导读] 前同步要求非空且等长的从 Thread 与通知索引列表，避免缺项或错配。
        // [中文导读] [AllReduce逐行 S1063] 调用 size 完成当前参数所指的子步骤；本行实参为 从 Thread 列表.size() 等于 0 或 notifyIdxMainToSub.size() 等于 0,。
        subThreads.size() == 0 || notifyIdxMainToSub.size() == 0,
        // [中文导读] [AllReduce逐行 S1064] 开始 HCCL_ERROR 诊断输出，记录 PreSyncInterThreads 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_ERROR(
            // [中文导读] [AllReduce逐行 S1065] 续接 PreSyncInterThreads 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "[AlgDataTransWrapper] [PreSyncInterThreads] subThreads size: [%u], notifyIdxMainToSub size [%u] "
            // [中文导读] [AllReduce逐行 S1066] 续接 PreSyncInterThreads 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "0 is not correct.",
            // [中文导读] [AllReduce逐行 S1067] 为 PreSyncInterThreads 的诊断/错误宏提供实参：从 Thread 列表.size(), notifyIdxMainToSub.size(，与前面的格式占位依次对应。
            subThreads.size(), notifyIdxMainToSub.size()),
        // [中文导读] [AllReduce逐行 S1068] 向条件返回宏提供 HcclResult::内部错误；上方检查成立才退出当前函数。
        HcclResult::HCCL_E_INTERNAL);
    // [中文导读] [AllReduce逐行 S1069] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S1070] 调用 size 完成当前参数所指的子步骤；本行实参为 从 Thread 列表.size() 不等于 notifyIdxMainToSub.size(),。
        subThreads.size() != notifyIdxMainToSub.size(),
        // [中文导读] [AllReduce逐行 S1071] 开始 HCCL_ERROR 诊断输出，记录 PreSyncInterThreads 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_ERROR(
            // [中文导读] [AllReduce逐行 S1072] 续接 PreSyncInterThreads 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "[AlgDataTransWrapper] [PreSyncInterThreads] subThreads size: [%u], notifyIdxMainToSub size [%u] "
            // [中文导读] [AllReduce逐行 S1073] 续接 PreSyncInterThreads 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "is not equal.",
            // [中文导读] [AllReduce逐行 S1074] 为 PreSyncInterThreads 的诊断/错误宏提供实参：从 Thread 列表.size(), notifyIdxMainToSub.size(，与前面的格式占位依次对应。
            subThreads.size(), notifyIdxMainToSub.size()),
        // [中文导读] [AllReduce逐行 S1075] 向条件返回宏提供 HcclResult::内部错误；上方检查成立才退出当前函数。
        HcclResult::HCCL_E_INTERNAL);
    // 获取执行超时时间
    // [中文导读] [AllReduce逐行 S1077] 设置 当前实际执行配置Timeout 为 ExecTimeoutManager::Instance().GetExecTimeout()；该值供下方当前分支使用。
    u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
    // 主thread向从thread发送record
    // [中文导读] [AllReduce逐行 S1079] 逐一处理从 Thread 与其对应通知槽；边界/迭代规则为 (u32 tidx = 0; tidx 小于 从 Thread 列表.size(); tidx++。
    for (u32 tidx = 0; tidx < subThreads.size(); tidx++) {
        // [中文导读] 主 Thread 分别向每个从 Thread 的对应槽记录放行通知。
        // [中文导读] [AllReduce逐行 S1081] 执行下方完整子调用；返回值非成功时立即从当前函数返回该错误。
        CHK_RET(static_cast<HcclResult>(
            // [中文导读] [AllReduce逐行 S1082] 记录 Thread 间同步通知；本行实参为 HcommThreadNotifyRecordOnThread(主 Thread, 从 Thread 列表[tidx], notifyIdxMainToSub[tidx])))。
            HcommThreadNotifyRecordOnThread(mainThread, subThreads[tidx], notifyIdxMainToSub[tidx])));
    // [中文导读] [AllReduce逐行 S1083] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // 从thread等待主thread的record
    // [中文导读] [AllReduce逐行 S1086] 逐一处理从 Thread 与其对应通知槽；边界/迭代规则为 (u32 tidx = 0; tidx 小于 从 Thread 列表.size(); tidx++。
    for (u32 tidx = 0; tidx < subThreads.size(); tidx++) {
        // [中文导读] 每个从 Thread 在自己的任务队列等待放行，之后才执行各自的并行工作。
        // [中文导读] [AllReduce逐行 S1088] 执行下方完整子调用；返回值非成功时立即从当前函数返回该错误。
        CHK_RET(static_cast<HcclResult>(
            // [中文导读] [AllReduce逐行 S1089] 等待 Thread 间同步通知；本行实参为 HcommThreadNotifyWaitOnThread(从 Thread 列表[tidx], notifyIdxMainToSub[tidx], 当前实际执行配置Timeout)))。
            HcommThreadNotifyWaitOnThread(subThreads[tidx], notifyIdxMainToSub[tidx], execTimeout)));
    // [中文导读] [AllReduce逐行 S1090] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // [中文导读] [AllReduce逐行 S1092] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
    return HcclResult::HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S1093] 结束 PreSyncInterThreads 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] 主从Thread后同步：主Thread排入各从Thread的完成等待，从Thread在各自队列发回Record。
// [中文导读] C++先写主Thread的Wait不构成Host死锁：这些调用组织不同任务队列上的执行依赖。
// [中文导读] [AllReduce逐行 S1097] 定义 PostSyncInterThreads 入口：主流等待各独立完成索引，各从流尾部 Record，建立主流对所有从流的后同步。
HcclResult PostSyncInterThreads(
    // [中文导读] [AllReduce逐行 S1098] 续接 PostSyncInterThreads 的入口参数/基类初始化：const ThreadHandle& 主 Thread, const std::vector<ThreadHandle>& 从 Thread 列表,；引用参数按声明的 const 限制读写。
    const ThreadHandle& mainThread, const std::vector<ThreadHandle>& subThreads,
    // [中文导读] [AllReduce逐行 S1099] 续接 PostSyncInterThreads 的入口参数/基类初始化：const std::vector<u32>& notifyIdxSubToMain)；引用参数按声明的 const 限制读写。
    const std::vector<u32>& notifyIdxSubToMain)
// [中文导读] [AllReduce逐行 S1100] 进入 PostSyncInterThreads 的实现作用域；主流等待各独立完成索引，各从流尾部 Record，建立主流对所有从流的后同步。
{
    // [中文导读] [AllReduce逐行 S1101] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S1102] 调用 size 完成当前参数所指的子步骤；本行实参为 从 Thread 列表.size() 等于 0 或 notifyIdxSubToMain.size() 等于 0,。
        subThreads.size() == 0 || notifyIdxSubToMain.size() == 0,
        // [中文导读] [AllReduce逐行 S1103] 开始 HCCL_ERROR 诊断输出，记录 PostSyncInterThreads 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_ERROR(
            // [中文导读] [AllReduce逐行 S1104] 续接 PostSyncInterThreads 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "[AlgDataTransWrapper] [PreSyncInterThreads] subThreads size: [%u], notifyIdxSubToMain size [%u] "
            // [中文导读] [AllReduce逐行 S1105] 续接 PostSyncInterThreads 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "0 is not correct.",
            // [中文导读] [AllReduce逐行 S1106] 为 PostSyncInterThreads 的诊断/错误宏提供实参：从 Thread 列表.size(), notifyIdxSubToMain.size(，与前面的格式占位依次对应。
            subThreads.size(), notifyIdxSubToMain.size()),
        // [中文导读] [AllReduce逐行 S1107] 向条件返回宏提供 HcclResult::内部错误；上方检查成立才退出当前函数。
        HcclResult::HCCL_E_INTERNAL);
    // [中文导读] [AllReduce逐行 S1108] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S1109] 调用 size 完成当前参数所指的子步骤；本行实参为 从 Thread 列表.size() 不等于 notifyIdxSubToMain.size(),。
        subThreads.size() != notifyIdxSubToMain.size(),
        // [中文导读] [AllReduce逐行 S1110] 开始 HCCL_ERROR 诊断输出，记录 PostSyncInterThreads 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_ERROR(
            // [中文导读] [AllReduce逐行 S1111] 续接 PostSyncInterThreads 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "[AlgDataTransWrapper] [PreSyncInterThreads] subThreads size: [%u], notifyIdxSubToMain size [%u] "
            // [中文导读] [AllReduce逐行 S1112] 续接 PostSyncInterThreads 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "is not equal.",
            // [中文导读] [AllReduce逐行 S1113] 为 PostSyncInterThreads 的诊断/错误宏提供实参：从 Thread 列表.size(), notifyIdxSubToMain.size(，与前面的格式占位依次对应。
            subThreads.size(), notifyIdxSubToMain.size()),
        // [中文导读] [AllReduce逐行 S1114] 向条件返回宏提供 HcclResult::内部错误；上方检查成立才退出当前函数。
        HcclResult::HCCL_E_INTERNAL);
    // 获取执行超时时间
    // [中文导读] [AllReduce逐行 S1116] 设置 当前实际执行配置Timeout 为 ExecTimeoutManager::Instance().GetExecTimeout()；该值供下方当前分支使用。
    u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
    // 主thread等待所有从thread的record
    // [中文导读] [AllReduce逐行 S1118] 逐一处理从 Thread 与其对应通知槽；边界/迭代规则为 (u32 tidx = 0; tidx 小于 从 Thread 列表.size(); tidx++。
    for (u32 tidx = 0; tidx < subThreads.size(); tidx++) {
        // [中文导读] 主 Thread 对每个从 Thread 的完成槽排入等待，把从流尾部任务汇合到主流。
        // [中文导读] [AllReduce逐行 S1120] 执行下方完整子调用；返回值非成功时立即从当前函数返回该错误。
        CHK_RET(
            // [中文导读] [AllReduce逐行 S1121] 等待 Thread 间同步通知；本行实参为 static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(主 Thread, notifyIdxSubToMain[tidx], 当前实际执行配置Timeout)))。
            static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(mainThread, notifyIdxSubToMain[tidx], execTimeout)));
    // [中文导读] [AllReduce逐行 S1122] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // 从thread向主thread发送record
    // [中文导读] [AllReduce逐行 S1125] 逐一处理从 Thread 与其对应通知槽；边界/迭代规则为 (u32 tidx = 0; tidx 小于 从 Thread 列表.size(); tidx++。
    for (u32 tidx = 0; tidx < subThreads.size(); tidx++) {
        // [中文导读] 各从 Thread 在自己的尾部通知主 Thread，和上面的不同主流等待槽逐一配对。
        // [中文导读] [AllReduce逐行 S1127] 执行下方完整子调用；返回值非成功时立即从当前函数返回该错误。
        CHK_RET(static_cast<HcclResult>(
            // [中文导读] [AllReduce逐行 S1128] 记录 Thread 间同步通知；本行实参为 HcommThreadNotifyRecordOnThread(从 Thread 列表[tidx], 主 Thread, notifyIdxSubToMain[tidx])))。
            HcommThreadNotifyRecordOnThread(subThreads[tidx], mainThread, notifyIdxSubToMain[tidx])));
    // [中文导读] [AllReduce逐行 S1129] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // [中文导读] [AllReduce逐行 S1131] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
    return HcclResult::HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S1132] 结束 PostSyncInterThreads 实现；其返回状态或已写回字段由调用者接收。
}

// FP16: 1位符号 + 5位指数(bias=15) + 10位尾数
// FP32: 1位符号 + 8位指数(bias=127) + 23位尾数
float Fp16ToFp32(uint16_t fp16Bits)
{
    uint32_t sign = (fp16Bits >> 15) & 0x1;
    uint32_t exponent = (fp16Bits >> 10) & 0x1F;
    uint32_t mantissa = fp16Bits & 0x3FF;

    if (exponent == 0) {
        // FP16零或非规格化数（exponent=0）
        if (mantissa == 0) {
            // ±0: 符号位保留，指数和尾数全零
            uint32_t result = sign << 31;
            float f;
            if (memcpy_s(&f, sizeof(f), &result, sizeof(f)) != EOK) {
                HCCL_ERROR("[%s] memcpy_s failed.", __func__);
                return 0.0f;
            }
            return f;
        }
        // FP16非规格化数: 无隐含1，实际指数为-14
        // 需要找到mantissa中最高有效1，将其规格化为FP32的隐含1形式
        // while循环左移mantissa直到bit10=1（对应FP16隐含1的位置），shift记录左移次数
        int shift = 0;
        while ((mantissa & 0x400) == 0) {
            mantissa <<= 1;
            shift++;
        }
        mantissa &= 0x3FF; // 去掉隐含1，保留10位尾数
        // FP32指数 = FP16实际指数(-14) + bias(127) - shift = 127 - 15 + 1 - shift
        // shift越大说明原值越小，fp32Exp可能<=0，此时值太小FP32也无法表示，返回±0
        int32_t fp32Exp = 127 - 15 + 1 - shift;
        if (fp32Exp <= 0) {
            uint32_t result = sign << 31;
            float f;
            if (memcpy_s(&f, sizeof(f), &result, sizeof(f)) != EOK) {
                HCCL_ERROR("[%s] memcpy_s failed.", __func__);
                return 0.0f;
            }
            return f;
        }
        // 组装FP32规格化数: 符号 + 偏移后指数 + 尾数左移13位对齐到23位
        uint32_t result = (sign << 31) | (static_cast<uint32_t>(fp32Exp) << 23) | (mantissa << 13);
        float f;
        if (memcpy_s(&f, sizeof(f), &result, sizeof(f)) != EOK) {
            HCCL_ERROR("[%s] memcpy_s failed.", __func__);
            return 0.0f;
        }
        return f;
    }
    if (exponent == 0x1F) {
        // FP16 Inf或NaN（exponent全1）
        // mantissa==0: Inf; mantissa!=0: NaN。统一构造FP32的Inf/NaN:
        // 指数=0xFF，尾数=mantissa<<13（mantissa为0时就是Inf，非0时就是NaN）
        uint32_t result = (sign << 31) | (0xFF << 23) | (mantissa << 13);
        float f;
        if (memcpy_s(&f, sizeof(f), &result, sizeof(f)) != EOK) {
            HCCL_ERROR("[%s] memcpy_s failed.", __func__);
            return 0.0f;
        }
        return f;
    }
    // FP16规格化数（1<=exponent<=30）
    // FP32指数 = exponent - FP16bias(15) + FP32bias(127) = exponent + 112
    // FP32尾数 = FP16尾数左移13位（10位扩展到23位，低位补零）
    uint32_t result = (sign << 31) | ((exponent + 112) << 23) | (mantissa << 13);
    float f;
    if (memcpy_s(&f, sizeof(f), &result, sizeof(f)) != EOK) {
        HCCL_ERROR("[%s] memcpy_s failed.", __func__);
        return 0.0f;
    }
    return f;
}

// FP32: 1位符号 + 8位指数(bias=127) + 23位尾数
// FP16: 1位符号 + 5位指数(bias=15) + 10位尾数
// FP32规格化数下溢到FP16非规格化数或±0
// fp16Exp = exponent - 127 + 15 <= 0
uint16_t Fp32DenormToFp16(uint32_t sign, uint32_t mantissa, int32_t fp16Exp)
{
    if (fp16Exp < -10) {
        // 太小连FP16非规格化数都无法表示（FP16非规格化数最小有效移位为10），返回±0
        return static_cast<uint16_t>(sign << 15);
    }
    // FP16非规格化数: 无隐含1，需要将FP32的隐含1和尾数一起右移
    // totalShift = 将23位尾数移到FP16非规格化10位尾数所需的右移量
    mantissa |= 0x800000; // 将隐含1放入mantissa高位
    int32_t totalShift = 14 - fp16Exp;
    // round-to-nearest-even舍入: 取guard bit和sticky bit
    uint32_t roundBit = (mantissa >> (totalShift - 1)) & 0x1;
    uint32_t truncated = mantissa & ((1U << (totalShift - 1)) - 1);
    uint32_t sticky = (truncated != 0) ? 1 : 0;
    uint16_t fp16Mant = static_cast<uint16_t>(mantissa >> totalShift);
    fp16Mant += roundBit && (sticky || (fp16Mant & 0x1));
    if (fp16Mant & 0x400) {
        // 舍入进位后尾数溢出，升级为FP16最小规格化数（exponent=1, mantissa=0）
        return static_cast<uint16_t>((sign << 15) | (1 << 10));
    }
    return static_cast<uint16_t>((sign << 15) | fp16Mant);
}

uint16_t Fp32ToFp16(float value)
{
    uint32_t fp32Bits;
    if (memcpy_s(&fp32Bits, sizeof(fp32Bits), &value, sizeof(fp32Bits)) != EOK) {
        HCCL_ERROR("[%s] memcpy_s failed.", __func__);
        return 0;
    }
    uint32_t sign = (fp32Bits >> 31) & 0x1;
    uint32_t exponent = (fp32Bits >> 23) & 0xFF;
    uint32_t mantissa = fp32Bits & 0x7FFFFF;

    if (exponent == 0) {
        // FP32零或非规格化数（exponent=0），值太小无法用FP16表示，返回±0
        return static_cast<uint16_t>(sign << 15);
    } else if (exponent == 0xFF) {
        // FP32 Inf或NaN（exponent全1）
        if (mantissa == 0) {
            // Inf: FP16 exponent全1 + 尾数全0
            return static_cast<uint16_t>((sign << 15) | 0x7C00);
        }
        // NaN: FP16 exponent全1 + 尾数非零；若FP32尾数截断后为0则强制置1确保仍是NaN
        uint16_t fp16Mant = static_cast<uint16_t>(mantissa >> 13);
        if (fp16Mant == 0) {
            fp16Mant = 1;
        }
        return static_cast<uint16_t>((sign << 15) | 0x7C00 | fp16Mant);
    }

    // FP32规格化数: 计算FP16目标指数 fp16Exp = exponent - FP32bias(127) + FP16bias(15)
    int32_t fp16Exp = static_cast<int32_t>(exponent) - 127 + 15;

    if (fp16Exp >= 31) {
        // 上溢: 超出FP16规格化数范围（FP16最大指数=30），饱和到±Inf
        return static_cast<uint16_t>((sign << 15) | 0x7C00);
    } else if (fp16Exp <= 0) {
        return Fp32DenormToFp16(sign, mantissa, fp16Exp);
    }

    // 正常规格化数: 截断23位尾数到10位，做round-to-nearest-even舍入
    uint32_t discarded = mantissa & 0x1FFF; // 被丢弃的低13位
    uint16_t fp16Mant = static_cast<uint16_t>(mantissa >> 13);
    uint32_t roundBit = (discarded >> 12) & 0x1;   // guard bit: 被丢弃的最高位
    uint32_t sticky = (discarded & 0xFFF) ? 1 : 0; // sticky bit: 被丢弃的其余位是否有1
    fp16Mant += roundBit && (sticky || (fp16Mant & 0x1));
    if (fp16Mant == 0x400) {
        // 舍入进位导致10位尾数溢出，指数+1，尾数归零
        fp16Mant = 0;
        fp16Exp++;
    }
    if (fp16Exp >= 31) {
        // 进位后指数溢出，饱和到±Inf
        return static_cast<uint16_t>((sign << 15) | 0x7C00);
    }
    return static_cast<uint16_t>((sign << 15) | (fp16Exp << 10) | fp16Mant);
}

// [中文导读] [AllReduce逐行 S1291] 定义 AicpuReduceFp16 入口：把 FP16 源和目标转换为 FP32 完成软件归约，再把目标结果转回 FP16。
HcclResult AicpuReduceFp16(u8* dst, u8* src, u64 size, const HcclReduceOp reduceOp)
// [中文导读] [AllReduce逐行 S1292] 进入 AicpuReduceFp16 的实现作用域；把 FP16 源和目标转换为 FP32 完成软件归约，再把目标结果转回 FP16。
{
    // [中文导读] [AllReduce逐行 S1293] 设置 count 为 size / sizeof(uint16_t)；该值供下方当前分支使用。
    u64 count = size / sizeof(uint16_t);
    // [中文导读] [AllReduce逐行 S1294] 调用 srcFp32 完成当前参数所指的子步骤；本行实参为 std::vector<float> srcFp32(count)。
    std::vector<float> srcFp32(count);
    // [中文导读] [AllReduce逐行 S1295] 调用 dstFp32 完成当前参数所指的子步骤；本行实参为 std::vector<float> dstFp32(count)。
    std::vector<float> dstFp32(count);
    // [中文导读] [AllReduce逐行 S1296] 设置 uint16_t* srcFp16 为 reinterpret_cast<uint16_t*>(src)；该值供下方当前分支使用。
    uint16_t* srcFp16 = reinterpret_cast<uint16_t*>(src);
    // [中文导读] [AllReduce逐行 S1297] 设置 uint16_t* dstFp16 为 reinterpret_cast<uint16_t*>(dst)；该值供下方当前分支使用。
    uint16_t* dstFp16 = reinterpret_cast<uint16_t*>(dst);
    // [中文导读] [AllReduce逐行 S1298] 在 AicpuReduceFp16 中遍历 (u64 i = 0; i 小于 count; ++i 指定的集合或索引区间；边界/迭代规则为 (u64 i = 0; i 小于 count; ++i。
    for (u64 i = 0; i < count; ++i) {
        // [中文导读] [AllReduce逐行 S1299] 设置 srcFp32[i] 为 Fp16ToFp32(srcFp16[i])；该值供下方当前分支使用。
        srcFp32[i] = Fp16ToFp32(srcFp16[i]);
        // [中文导读] [AllReduce逐行 S1300] 设置 dstFp32[i] 为 Fp16ToFp32(dstFp16[i])；该值供下方当前分支使用。
        dstFp32[i] = Fp16ToFp32(dstFp16[i]);
    // [中文导读] [AllReduce逐行 S1301] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S1302] 设置 ret 为 AicpuReduceTemplate<float>(；该值供下方当前分支使用。
    HcclResult ret = AicpuReduceTemplate<float>(
        // [中文导读] [AllReduce逐行 S1303] 调用 data 完成当前参数所指的子步骤；本行实参为 dstFp32.data(), dstFp32.size() * sizeof(float), srcFp32.data(), srcFp32.size() * sizeof(float), reduceOp)。
        dstFp32.data(), dstFp32.size() * sizeof(float), srcFp32.data(), srcFp32.size() * sizeof(float), reduceOp);
    // [中文导读] [AllReduce逐行 S1304] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S1305] 续接本次错误检查/子调用实参：ret 不等于 HcclResult::成功状态；返回行为由所在完整宏决定。
        ret != HcclResult::HCCL_SUCCESS,
        // [中文导读] [AllReduce逐行 S1306] 开始 HCCL_ERROR 诊断输出，记录 AicpuReduceFp16 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_ERROR("[AicpuReduceFp16] AicpuReduceTemplate failed, ret[%d].", static_cast<int>(ret)), ret);
    // [中文导读] [AllReduce逐行 S1307] 在 AicpuReduceFp16 中遍历 (u64 i = 0; i 小于 count; ++i 指定的集合或索引区间；边界/迭代规则为 (u64 i = 0; i 小于 count; ++i。
    for (u64 i = 0; i < count; ++i) {
        // [中文导读] [AllReduce逐行 S1308] 设置 dstFp16[i] 为 Fp32ToFp16(dstFp32[i])；该值供下方当前分支使用。
        dstFp16[i] = Fp32ToFp16(dstFp32[i]);
    // [中文导读] [AllReduce逐行 S1309] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S1310] 直接返回 ret，调用者取得本分支结果。
    return ret;
// [中文导读] [AllReduce逐行 S1311] 结束 AicpuReduceFp16 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S1313] 定义 AicpuReduce 入口：软件归约按类型选择实例；thread 参数当前未用于排队，调用者须预先保证输入已经就绪。
HcclResult AicpuReduce(
    // [中文导读] [AllReduce逐行 S1314] 续接 AicpuReduce 的入口参数/基类初始化：const ThreadHandle& thread, const DataSlice& srcSlice, const DataSlice& dstSlice, const HcclDataType dataType,；引用参数按声明的 const 限制读写。
    const ThreadHandle& thread, const DataSlice& srcSlice, const DataSlice& dstSlice, const HcclDataType dataType,
    // [中文导读] [AllReduce逐行 S1315] 续接 AicpuReduce 的入口参数/基类初始化：const HcclReduceOp reduceOp)；引用参数按声明的 const 限制读写。
    const HcclReduceOp reduceOp)
// [中文导读] [AllReduce逐行 S1316] 进入 AicpuReduce 的实现作用域；软件归约按类型选择实例；thread 参数当前未用于排队，调用者须预先保证输入已经就绪。
{
    // [中文导读] 此软件归约实现直接在 AICPU 上访问数据，参数 thread 在当前函数中没有用于排队原语。
    // [中文导读] [AllReduce逐行 S1318] 显式标记 thread 在此兼容/default 分支未使用，避免编译器未使用参数警告。
    (void)thread;
    // [中文导读] [AllReduce逐行 S1319] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S1320] 续接本次错误检查/子调用实参：源片字节数 不等于 目标片字节数；返回行为由所在完整宏决定。
        srcSlice.size_ != dstSlice.size_,
        // [中文导读] [AllReduce逐行 S1321] 开始 HCCL_ERROR 诊断输出，记录 AicpuReduce 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_ERROR(
            // [中文导读] [AllReduce逐行 S1322] 续接 AicpuReduce 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "[AlgDataTransWrapper] [AicpuReduce] AicpuReduce: src slice size [%u] "
            // [中文导读] [AllReduce逐行 S1323] 续接 AicpuReduce 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "is not equal to dst slice size [%u].",
            // [中文导读] [AllReduce逐行 S1324] 为 AicpuReduce 的诊断/错误宏提供实参：源片字节数, 目标片字节数，与前面的格式占位依次对应。
            srcSlice.size_, dstSlice.size_),
        // [中文导读] [AllReduce逐行 S1325] 向条件返回宏提供 HcclResult::内部错误；上方检查成立才退出当前函数。
        HcclResult::HCCL_E_INTERNAL);

    // [中文导读] [AllReduce逐行 S1327] 设置 ret 为 HcclResult::成功状态；该值供下方当前分支使用。
    auto ret = HcclResult::HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S1328] 设置 u8* src 为 static_cast<u8*>(GetSliceAddr(srcSlice))；该值供下方当前分支使用。
    u8* src = static_cast<u8*>(GetSliceAddr(srcSlice));
    // [中文导读] [AllReduce逐行 S1329] 设置 u8* dst 为 static_cast<u8*>(GetSliceAddr(dstSlice))；该值供下方当前分支使用。
    u8* dst = static_cast<u8*>(GetSliceAddr(dstSlice));
    // [中文导读] [AllReduce逐行 S1330] 记录源目标基址、偏移、实际地址和类型信息；本行实参为 TraceDataSlice(。
    TraceDataSlice(
        // [中文导读] [AllReduce逐行 S1331] 续接本次调用的字符串常量 "AicpuReduce", "AICPU_REDUCE", 0, 1, srcSlice, dstSlice, src, dst, srcSlice.size_, dataType, reduceOp);，由所在注册/日志/条件语句整体使用。
        "AicpuReduce", "AICPU_REDUCE", 0, 1, srcSlice, dstSlice, src, dst, srcSlice.size_, dataType, reduceOp);
    // [中文导读] 依据元素类型选择软件归约实例，FP16 单独转换为 FP32 执行再转回。
    // [中文导读] [AllReduce逐行 S1333] 续接 AicpuReduce 当前语句的具体实参/字段：switch (dataType) {；由其完整表达式完成参数组装、检查或结果写回。
    switch (dataType) {
        // [中文导读] [AllReduce逐行 S1334] 处理 HcclDataType::INT8 类型 的专用实现，不同类型或运算在其它 case 分开处理。
        case HcclDataType::HCCL_DATA_TYPE_INT8:
            // [中文导读] [AllReduce逐行 S1335] 设置 ret 为 AicpuReduceTemplate<int8_t>(；该值供下方当前分支使用。
            ret = AicpuReduceTemplate<int8_t>(
                // [中文导读] [AllReduce逐行 S1336] 调用 reinterpret_cast 完成当前参数所指的子步骤；本行实参为 reinterpret_cast<int8_t*>(dst), 目标片字节数, reinterpret_cast<int8_t*>(src), 源片字节数,。
                reinterpret_cast<int8_t*>(dst), dstSlice.size_, reinterpret_cast<int8_t*>(src), srcSlice.size_,
                // [中文导读] [AllReduce逐行 S1337] 续接 AicpuReduce 当前语句的具体实参/字段：reduceOp)；由其完整表达式完成参数组装、检查或结果写回。
                reduceOp);
            // [中文导读] [AllReduce逐行 S1338] 退出当前 switch 或内层循环，保留此前选中的结果，不继续后续项。
            break;
        // [中文导读] [AllReduce逐行 S1339] 处理 HcclDataType::HCCL_DATA_TYPE_INT32 的专用实现，不同类型或运算在其它 case 分开处理。
        case HcclDataType::HCCL_DATA_TYPE_INT32:
            // [中文导读] [AllReduce逐行 S1340] 设置 ret 为 AicpuReduceTemplate<int32_t>(；该值供下方当前分支使用。
            ret = AicpuReduceTemplate<int32_t>(
                // [中文导读] [AllReduce逐行 S1341] 调用 reinterpret_cast 完成当前参数所指的子步骤；本行实参为 reinterpret_cast<int32_t*>(dst), 目标片字节数, reinterpret_cast<int32_t*>(src), 源片字节数,。
                reinterpret_cast<int32_t*>(dst), dstSlice.size_, reinterpret_cast<int32_t*>(src), srcSlice.size_,
                // [中文导读] [AllReduce逐行 S1342] 续接 AicpuReduce 当前语句的具体实参/字段：reduceOp)；由其完整表达式完成参数组装、检查或结果写回。
                reduceOp);
            // [中文导读] [AllReduce逐行 S1343] 退出当前 switch 或内层循环，保留此前选中的结果，不继续后续项。
            break;
        // [中文导读] [AllReduce逐行 S1344] 处理 HcclDataType::FP16 类型 的专用实现，不同类型或运算在其它 case 分开处理。
        case HcclDataType::HCCL_DATA_TYPE_FP16:
            // [中文导读] [AllReduce逐行 S1345] 设置 ret 为 AicpuReduceFp16(dst, src, 源片字节数, reduceOp)；该值供下方当前分支使用。
            ret = AicpuReduceFp16(dst, src, srcSlice.size_, reduceOp);
            // [中文导读] [AllReduce逐行 S1346] 退出当前 switch 或内层循环，保留此前选中的结果，不继续后续项。
            break;
        // [中文导读] [AllReduce逐行 S1347] 处理 HcclDataType::FP32 类型 的专用实现，不同类型或运算在其它 case 分开处理。
        case HcclDataType::HCCL_DATA_TYPE_FP32:
            // [中文导读] [AllReduce逐行 S1348] 设置 ret 为 AicpuReduceTemplate<float>(；该值供下方当前分支使用。
            ret = AicpuReduceTemplate<float>(
                // [中文导读] [AllReduce逐行 S1349] 调用 reinterpret_cast 完成当前参数所指的子步骤；本行实参为 reinterpret_cast<float*>(dst), 目标片字节数, reinterpret_cast<float*>(src), 源片字节数, reduceOp)。
                reinterpret_cast<float*>(dst), dstSlice.size_, reinterpret_cast<float*>(src), srcSlice.size_, reduceOp);
            // [中文导读] [AllReduce逐行 S1350] 退出当前 switch 或内层循环，保留此前选中的结果，不继续后续项。
            break;
        // [中文导读] [AllReduce逐行 S1351] 处理 HcclDataType::有符号 64 位整数 的专用实现，不同类型或运算在其它 case 分开处理。
        case HcclDataType::HCCL_DATA_TYPE_INT64:
            // [中文导读] [AllReduce逐行 S1352] 设置 ret 为 AicpuReduceTemplate<int64_t>(；该值供下方当前分支使用。
            ret = AicpuReduceTemplate<int64_t>(
                // [中文导读] [AllReduce逐行 S1353] 调用 reinterpret_cast 完成当前参数所指的子步骤；本行实参为 reinterpret_cast<int64_t*>(dst), 目标片字节数, reinterpret_cast<int64_t*>(src), 源片字节数,。
                reinterpret_cast<int64_t*>(dst), dstSlice.size_, reinterpret_cast<int64_t*>(src), srcSlice.size_,
                // [中文导读] [AllReduce逐行 S1354] 续接 AicpuReduce 当前语句的具体实参/字段：reduceOp)；由其完整表达式完成参数组装、检查或结果写回。
                reduceOp);
            // [中文导读] [AllReduce逐行 S1355] 退出当前 switch 或内层循环，保留此前选中的结果，不继续后续项。
            break;
        // [中文导读] [AllReduce逐行 S1356] 处理 HcclDataType::无符号 64 位整数 的专用实现，不同类型或运算在其它 case 分开处理。
        case HcclDataType::HCCL_DATA_TYPE_UINT64:
            // [中文导读] [AllReduce逐行 S1357] 设置 ret 为 AicpuReduceTemplate<uint64_t>(；该值供下方当前分支使用。
            ret = AicpuReduceTemplate<uint64_t>(
                // [中文导读] [AllReduce逐行 S1358] 调用 reinterpret_cast 完成当前参数所指的子步骤；本行实参为 reinterpret_cast<uint64_t*>(dst), 目标片字节数, reinterpret_cast<uint64_t*>(src), 源片字节数,。
                reinterpret_cast<uint64_t*>(dst), dstSlice.size_, reinterpret_cast<uint64_t*>(src), srcSlice.size_,
                // [中文导读] [AllReduce逐行 S1359] 续接 AicpuReduce 当前语句的具体实参/字段：reduceOp)；由其完整表达式完成参数组装、检查或结果写回。
                reduceOp);
            // [中文导读] [AllReduce逐行 S1360] 退出当前 switch 或内层循环，保留此前选中的结果，不继续后续项。
            break;
        // [中文导读] [AllReduce逐行 S1361] 处理 HcclDataType::FP64 类型 的专用实现，不同类型或运算在其它 case 分开处理。
        case HcclDataType::HCCL_DATA_TYPE_FP64:
            // [中文导读] [AllReduce逐行 S1362] 设置 ret 为 AicpuReduceTemplate<double>(；该值供下方当前分支使用。
            ret = AicpuReduceTemplate<double>(
                // [中文导读] [AllReduce逐行 S1363] 调用 reinterpret_cast 完成当前参数所指的子步骤；本行实参为 reinterpret_cast<double*>(dst), 目标片字节数, reinterpret_cast<double*>(src), 源片字节数,。
                reinterpret_cast<double*>(dst), dstSlice.size_, reinterpret_cast<double*>(src), srcSlice.size_,
                // [中文导读] [AllReduce逐行 S1364] 续接 AicpuReduce 当前语句的具体实参/字段：reduceOp)；由其完整表达式完成参数组装、检查或结果写回。
                reduceOp);
            // [中文导读] [AllReduce逐行 S1365] 退出当前 switch 或内层循环，保留此前选中的结果，不继续后续项。
            break;
        // [中文导读] [AllReduce逐行 S1366] 未列出的类型或状态进入兜底分支；按下面返回码判为不支持或错误。
        default:
            // [中文导读] [AllReduce逐行 S1367] 开始 HCCL_ERROR 诊断输出，记录 AicpuReduce 当前阶段的参数、候选或错误；日志本身不决定返回码。
            HCCL_ERROR("DataType[%d] not support", int(dataType));
            // [中文导读] [AllReduce逐行 S1368] 设置 ret 为 内部错误；该值供下方当前分支使用。
            ret = HCCL_E_INTERNAL;
            // [中文导读] [AllReduce逐行 S1369] 退出当前 switch 或内层循环，保留此前选中的结果，不继续后续项。
            break;
    // [中文导读] [AllReduce逐行 S1370] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S1371] 直接返回 ret，调用者取得本分支结果。
    return ret;
// [中文导读] [AllReduce逐行 S1372] 结束 AicpuReduce 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S1374] 声明模板类型参数，将拓扑匹配器与具体算法模板在编译期绑定。
template <typename T>
// [中文导读] [AllReduce逐行 S1375] 定义 AicpuReduceTemplate 入口：逐元素执行 SUM/PROD/MAX/MIN；INT8/INT32 PROD 通过无符号中间值表达乘积，不支持运算返回内部错误。
HcclResult AicpuReduceTemplate(T* dst, u64 dstSize, T* src, u64 srcSize, const HcclReduceOp reduceOp)
// [中文导读] [AllReduce逐行 S1376] 进入 AicpuReduceTemplate 的实现作用域；逐元素执行 SUM/PROD/MAX/MIN；INT8/INT32 PROD 通过无符号中间值表达乘积，不支持运算返回内部错误。
{
    // [中文导读] [AllReduce逐行 S1377] 分支条件为 dstSize 不等于 srcSize；成立进入本块，未成立继续后续分支。
    if (dstSize != srcSize) {
        // [中文导读] [AllReduce逐行 S1378] 开始 HCCL_ERROR 诊断输出，记录 AicpuReduceTemplate 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_ERROR("srcSize[%llu] should be equal to dstSize[%llu]", srcSize, dstSize);
        // [中文导读] [AllReduce逐行 S1379] 终止当前函数并向上返回 HcclResult::内部错误；调用者 CHK_RET 决定是否继续向上传播。
        return HcclResult::HCCL_E_INTERNAL;
    // [中文导读] [AllReduce逐行 S1380] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S1381] 设置 ret 为 HcclResult::成功状态；该值供下方当前分支使用。
    auto ret = HcclResult::HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S1382] 设置 count 为 dstSize / u64(sizeof(T))；该值供下方当前分支使用。
    u64 count = dstSize / u64(sizeof(T));
    // [中文导读] [AllReduce逐行 S1383] 逐元素执行选定的 CPU 软件归约；边界/迭代规则为 (u64 i = 0; i 小于 count; ++i。
    for (u64 i = 0; i < count; ++i) {
        // [中文导读] [AllReduce逐行 S1384] 设置 T dstData 为 *(dst + i)；该值供下方当前分支使用。
        T dstData = *(dst + i);
        // [中文导读] [AllReduce逐行 S1385] 设置 T srcData 为 *(src + i)；该值供下方当前分支使用。
        T srcData = *(src + i);
        // [中文导读] [AllReduce逐行 S1386] 续接 AicpuReduceTemplate 当前语句的具体实参/字段：switch (reduceOp) {；由其完整表达式完成参数组装、检查或结果写回。
        switch (reduceOp) {
            // [中文导读] [AllReduce逐行 S1387] 处理 HcclReduceOp::求和归约 的专用实现，不同类型或运算在其它 case 分开处理。
            case HcclReduceOp::HCCL_REDUCE_SUM:
                // [中文导读] [AllReduce逐行 S1388] 设置 *(dst + i) 为 srcData + dstData；该值供下方当前分支使用。
                *(dst + i) = srcData + dstData;
                // [中文导读] [AllReduce逐行 S1389] 退出当前 switch 或内层循环，保留此前选中的结果，不继续后续项。
                break;
            // [中文导读] [AllReduce逐行 S1390] 处理 HcclReduceOp::乘积归约 的专用实现，不同类型或运算在其它 case 分开处理。
            case HcclReduceOp::HCCL_REDUCE_PROD:
                // [中文导读] [AllReduce逐行 S1391] 分支条件为 std::is_same<T, int8_t>::value；成立进入本块，未成立继续后续分支。
                if (std::is_same<T, int8_t>::value) {
                    // [中文导读] [AllReduce逐行 S1392] 设置 uint8_t prod 为 static_cast<uint8_t>(srcData) * static_cast<uint8_t>(dstData)；该值供下方当前分支使用。
                    uint8_t prod = static_cast<uint8_t>(srcData) * static_cast<uint8_t>(dstData);
                    // [中文导读] [AllReduce逐行 S1393] 设置 *(dst + i) 为 static_cast<T>(prod)；该值供下方当前分支使用。
                    *(dst + i) = static_cast<T>(prod);
                // [中文导读] [AllReduce逐行 S1394] 分支条件为 std::is_same<T, int32_t>::value；成立进入本块，未成立继续后续分支。
                } else if (std::is_same<T, int32_t>::value) {
                    // [中文导读] [AllReduce逐行 S1395] 设置 prod 为 static_cast<uint32_t>(srcData) * static_cast<uint32_t>(dstData)；该值供下方当前分支使用。
                    uint32_t prod = static_cast<uint32_t>(srcData) * static_cast<uint32_t>(dstData);
                    // [中文导读] [AllReduce逐行 S1396] 设置 *(dst + i) 为 static_cast<T>(prod)；该值供下方当前分支使用。
                    *(dst + i) = static_cast<T>(prod);
                // [中文导读] [AllReduce逐行 S1397] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
                } else {
                    // [中文导读] [AllReduce逐行 S1398] 设置 *(dst + i) 为 srcData * dstData；该值供下方当前分支使用。
                    *(dst + i) = srcData * dstData;
                // [中文导读] [AllReduce逐行 S1399] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
                }
                // [中文导读] [AllReduce逐行 S1400] 退出当前 switch 或内层循环，保留此前选中的结果，不继续后续项。
                break;
            // [中文导读] [AllReduce逐行 S1401] 处理 HcclReduceOp::最大值归约 的专用实现，不同类型或运算在其它 case 分开处理。
            case HcclReduceOp::HCCL_REDUCE_MAX:
                // [中文导读] [AllReduce逐行 S1402] 设置 *(dst + i) 为 std::max(srcData, dstData)；该值供下方当前分支使用。
                *(dst + i) = std::max(srcData, dstData);
                // [中文导读] [AllReduce逐行 S1403] 退出当前 switch 或内层循环，保留此前选中的结果，不继续后续项。
                break;
            // [中文导读] [AllReduce逐行 S1404] 处理 HcclReduceOp::最小值归约 的专用实现，不同类型或运算在其它 case 分开处理。
            case HcclReduceOp::HCCL_REDUCE_MIN:
                // [中文导读] [AllReduce逐行 S1405] 设置 *(dst + i) 为 std::min(srcData, dstData)；该值供下方当前分支使用。
                *(dst + i) = std::min(srcData, dstData);
                // [中文导读] [AllReduce逐行 S1406] 退出当前 switch 或内层循环，保留此前选中的结果，不继续后续项。
                break;
            // [中文导读] [AllReduce逐行 S1407] 未列出的类型或状态进入兜底分支；按下面返回码判为不支持或错误。
            default:
                // [中文导读] [AllReduce逐行 S1408] 开始 HCCL_ERROR 诊断输出，记录 AicpuReduceTemplate 当前阶段的参数、候选或错误；日志本身不决定返回码。
                HCCL_ERROR("ReduceOp[%d] not support", int(reduceOp));
                // [中文导读] [AllReduce逐行 S1409] 设置 ret 为 HcclResult::内部错误；该值供下方当前分支使用。
                ret = HcclResult::HCCL_E_INTERNAL;
                // [中文导读] [AllReduce逐行 S1410] 退出当前 switch 或内层循环，保留此前选中的结果，不继续后续项。
                break;
        // [中文导读] [AllReduce逐行 S1411] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
    // [中文导读] [AllReduce逐行 S1412] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S1413] 直接返回 ret，调用者取得本分支结果。
    return ret;
// [中文导读] [AllReduce逐行 S1414] 结束 AicpuReduceTemplate 实现；其返回状态或已写回字段由调用者接收。
}

} // namespace ops_hccl
