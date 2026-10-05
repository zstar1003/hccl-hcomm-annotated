/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_reduce.h"
#include "op_common_ops.h"
#include "topo_host.h"
#include <algorithm>
#include <future>
#include <map>
#include <string>

using namespace std;
using namespace ops_hccl;
extern "C" unsigned int LaunchAicpuKernel(OpParam* param);

// [中文导读] AllReduce按op合并各Rank对应位置的数据，并让各Rank得到归约结果。
// [中文导读] count是本次参与归约的元素数量，dataType和op共同限定合法运算；不是原始字节长度。
// [中文导读] 入口通过版本/设备判定选择新旧流程，再交AllReduceOutPlace；归约任务不在这里展开。
// [中文导读] [AllReduce逐行 S26] 定义 HcclAllReduce 入口：单算子公开入口：版本/设备兼容分流，非零参数校验，入口日志和 OPBASE 公共调度。
HcclResult HcclAllReduce(
    // [中文导读] [AllReduce逐行 S27] 续接 HcclAllReduce 的入口参数/基类初始化：void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm,；引用参数按声明的 const 限制读写。
    void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm,
    // [中文导读] [AllReduce逐行 S28] 续接 HcclAllReduce 的入口参数/基类初始化：aclrtStream stream)；引用参数按声明的 const 限制读写。
    aclrtStream stream)
// [中文导读] [AllReduce逐行 S29] 进入 HcclAllReduce 的实现作用域；单算子公开入口：版本/设备兼容分流，非零参数校验，入口日志和 OPBASE 公共调度。
{
    // [中文导读] [AllReduce逐行 S30] 开始 HCCL_INFO 诊断输出，记录 HcclAllReduce 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_INFO("Start to run execute HcclAllReduce");
    // [中文导读] 先按运行时版本分流，旧 HCOMM 使用兼容归约入口。
    // [中文导读] [AllReduce逐行 S32] 分支条件为 GetHcommVersion() 小于 CANN_VERSION(9, 0, 0；成立进入本块，未成立继续后续分支。
    if (GetHcommVersion() < CANN_VERSION(9, 0, 0)) { // compat handle
        // [中文导读] [AllReduce逐行 S33] 直接返回 调用 HcclAllReduceInner 完成当前参数所指的子步骤 的结果，调用者取得本分支结果。
        return HcclAllReduceInner(sendBuf, recvBuf, count, dataType, op, comm, stream);
    // [中文导读] [AllReduce逐行 S34] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // [中文导读] 检查设备的新流程支持情况；不支持时沿原 AllReduce 实现执行。
    // [中文导读] [AllReduce逐行 S37] 设置 isOutPlace 为 false；该值供下方当前分支使用。
    bool isOutPlace = false;
    // [中文导读] [AllReduce逐行 S38] 调用 IsOutPlaceDevice 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(IsOutPlaceDevice(isOutPlace));
    // [中文导读] [AllReduce逐行 S39] 分支条件为 !isOutPlace；成立进入本块，未成立继续后续分支。
    if (!isOutPlace) {
        // [中文导读] [AllReduce逐行 S40] 直接返回 调用 HcclAllReduceInner 完成当前参数所指的子步骤 的结果，调用者取得本分支结果。
        return HcclAllReduceInner(sendBuf, recvBuf, count, dataType, op, comm, stream);
    // [中文导读] [AllReduce逐行 S41] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] 零元素归约不展开通信；正常非零调用才进入环境和参数检查。
    // [中文导读] [AllReduce逐行 S43] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
    CHK_PRT_RET(count == 0, HCCL_WARNING("input count is 0, return all reduce success"), HCCL_SUCCESS);

    // [中文导读] [AllReduce逐行 S45] 设置 HcclUs startut 为 TIME_NOW()；该值供下方当前分支使用。
    HcclUs startut = TIME_NOW(); // 走老流程的判断时间不统计在内
    // [中文导读] [AllReduce逐行 S46] 声明本阶段局部变量 OpParam param，实际值由后续查询/计算填写。
    OpParam param;
    // [中文导读] 建立算子 tag 并验证数量、数据类型和归约运算组合，防止非法归约进入算法层。
    // [中文导读] [AllReduce逐行 S48] 完成 AllReduce 入口环境与参数验证；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(AllReduceInitAndCheck(comm, sendBuf, recvBuf, count, dataType, op, stream, param));

    /* 接口交互信息日志 */
    // [中文导读] [AllReduce逐行 S51] 记录 AllReduce 本次调用信息；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(AllReduceEntryLog(sendBuf, recvBuf, count, dataType, op, stream, param.tag, "HcclAllReduce"));

    // 执行AllReduce
    // [中文导读] 下传已校验的归约参数，实际集合通信任务由公共调度和选中算法生成。
    // [中文导读] [AllReduce逐行 S55] 调用 CHK_RET_AND_PRINT_IDE 完成当前参数所指的子步骤；本行实参为 CHK_RET_AND_PRINT_IDE(AllReduceOutPlace(sendBuf, recvBuf, count, dataType, op, comm, stream, param), param.tag)。
    CHK_RET_AND_PRINT_IDE(AllReduceOutPlace(sendBuf, recvBuf, count, dataType, op, comm, stream, param), param.tag);

    // [中文导读] [AllReduce逐行 S57] 调用 LogHcclExit 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(LogHcclExit("HcclAllReduce", param.tag, startut));

    // [中文导读] [AllReduce逐行 S59] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S60] 结束 HcclAllReduce 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S62] 定义 HcclAllReduceGraphMode 入口：通过 group 定位通信域并收集图模式外部从流、scratch 与 tag，进入 OFFLOAD 公共调度。
HcclResult HcclAllReduceGraphMode(
    // [中文导读] [AllReduce逐行 S63] 续接 HcclAllReduceGraphMode 的入口参数/基类初始化：void* sendBuf, void* recvBuf, uint64_t sendCount, HcclDataType dataType, HcclReduceOp op, const char* group,；引用参数按声明的 const 限制读写。
    void* sendBuf, void* recvBuf, uint64_t sendCount, HcclDataType dataType, HcclReduceOp op, const char* group,
    // [中文导读] [AllReduce逐行 S64] 续接 HcclAllReduceGraphMode 的入口参数/基类初始化：aclrtStream stream, const char* tag, void** streams, size_t streamCount, void* scratchMemAddr,；引用参数按声明的 const 限制读写。
    aclrtStream stream, const char* tag, void** streams, size_t streamCount, void* scratchMemAddr,
    // [中文导读] [AllReduce逐行 S65] 续接 HcclAllReduceGraphMode 的入口参数/基类初始化：uint64_t scratchMemSize)；引用参数按声明的 const 限制读写。
    uint64_t scratchMemSize)
// [中文导读] [AllReduce逐行 S66] 进入 HcclAllReduceGraphMode 的实现作用域；通过 group 定位通信域并收集图模式外部从流、scratch 与 tag，进入 OFFLOAD 公共调度。
{
    // [中文导读] [AllReduce逐行 S67] 开始 HCCL_INFO 诊断输出，记录 HcclAllReduceGraphMode 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_INFO("Start to run execute HcclAllReduceGraphMode");
    // 根据group获取通信域
    // [中文导读] 根据图模式的 group 定位通信域，沿用同一套归约参数检查。
    // [中文导读] [AllReduce逐行 S70] 调用 CHK_PTR_NULL 完成当前参数所指的子步骤；本行实参为 CHK_PTR_NULL(group)。
    CHK_PTR_NULL(group);
    // [中文导读] [AllReduce逐行 S71] 设置 HcclComm comm 为 nullptr；该值供下方当前分支使用。
    HcclComm comm = nullptr;
    // [中文导读] [AllReduce逐行 S72] 开始 HCCL_INFO 诊断输出，记录 HcclAllReduceGraphMode 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_INFO("[HcclAllReduceGraphMode] get group name: %s", group);
    // [中文导读] [AllReduce逐行 S73] 调用 HcomGetCommHandleByGroup 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(HcomGetCommHandleByGroup(group, &comm));
    // [中文导读] [AllReduce逐行 S74] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
    CHK_PRT_RET(sendCount == 0, HCCL_WARNING("input sendCount is 0, return all reduce success"), HCCL_SUCCESS);

    // [中文导读] [AllReduce逐行 S76] 设置 HcclUs startut 为 TIME_NOW()；该值供下方当前分支使用。
    HcclUs startut = TIME_NOW(); // 走老流程的判断时间不统计在内
    // [中文导读] [AllReduce逐行 S77] 声明本阶段局部变量 OpParam param，实际值由后续查询/计算填写。
    OpParam param;
    // [中文导读] [AllReduce逐行 S78] 完成 AllReduce 入口环境与参数验证；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(AllReduceInitAndCheck(comm, sendBuf, recvBuf, sendCount, dataType, op, stream, param));

    // 检查tag有效性
    // [中文导读] 图执行提供独立 tag，校验后覆盖初始化生成的默认算子标识。
    // [中文导读] [AllReduce逐行 S82] 调用 HcclCheckTag 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(HcclCheckTag(tag));
    // [中文导读] [AllReduce逐行 S83] 设置 ret 为 sprintf_s(param.tag, sizeof(param.tag), "%s", tag)；该值供下方当前分支使用。
    int ret = sprintf_s(param.tag, sizeof(param.tag), "%s", tag);
    // [中文导读] [AllReduce逐行 S84] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
    CHK_PRT_RET((ret <= 0), HCCL_ERROR("failed to fill param.tag"), HCCL_E_INTERNAL);

    // 拼装ResPackGraphMode
    // [中文导读] 把图执行的外部从流和临时内存组织成资源包，随 OFFLOAD 模式传给公共执行层。
    // [中文导读] [AllReduce逐行 S88] 声明本阶段局部变量 ResPackGraphMode resPack，实际值由后续查询/计算填写。
    ResPackGraphMode resPack;
    // 设置tag
    // [中文导读] [AllReduce逐行 S90] 分支条件为 strncpy_s(resPack.tag, sizeof(resPack.tag), tag, sizeof(resPack.tag) - 1) 不等于 0；成立进入本块，未成立继续后续分支。
    if (strncpy_s(resPack.tag, sizeof(resPack.tag), tag, sizeof(resPack.tag) - 1) != 0) {
        // [中文导读] [AllReduce逐行 S91] 开始 HCCL_ERROR 诊断输出，记录 HcclAllReduceGraphMode 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_ERROR("failed to fill resPack.tag");
        // [中文导读] [AllReduce逐行 S92] 终止当前函数并向上返回 内部错误；调用者 CHK_RET 决定是否继续向上传播。
        return HCCL_E_INTERNAL;
    // [中文导读] [AllReduce逐行 S93] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // 设置streams
    // [中文导读] [AllReduce逐行 S95] 分支条件为 streams 不等于 nullptr 且 streamCount 大于 0；成立进入本块，未成立继续后续分支。
    if (streams != nullptr && streamCount > 0) {
        // [中文导读] [AllReduce逐行 S96] 在 HcclAllReduceGraphMode 中遍历 (size_t i = 0; i 小于 streamCount; i++ 指定的集合或索引区间；边界/迭代规则为 (size_t i = 0; i 小于 streamCount; i++。
        for (size_t i = 0; i < streamCount; i++) {
            // [中文导读] [AllReduce逐行 S97] 对 resPack 追加 static_cast<aclrtStream>(streams[i])，准备或更新本阶段列表。
            resPack.streams.push_back(static_cast<aclrtStream>(streams[i]));
        // [中文导读] [AllReduce逐行 S98] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
    // [中文导读] [AllReduce逐行 S99] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // 设置scratchMem
    // [中文导读] [AllReduce逐行 S101] 设置 resPack.scratchMemAddr 为 scratchMemAddr；该值供下方当前分支使用。
    resPack.scratchMemAddr = scratchMemAddr;
    // [中文导读] [AllReduce逐行 S102] 设置 resPack.scratchMemSize 为 scratchMemSize；该值供下方当前分支使用。
    resPack.scratchMemSize = scratchMemSize;
    // [中文导读] [AllReduce逐行 S103] 设置 tagStr 为 tag；该值供下方当前分支使用。
    std::string tagStr = tag;

    /* 接口交互信息日志 */
    // [中文导读] [AllReduce逐行 S106] 记录 AllReduce 本次调用信息；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(AllReduceEntryLog(
        // [中文导读] [AllReduce逐行 S107] 续接本次错误检查/子调用实参：sendBuf, recvBuf, sendCount, dataType, op, stream, param.tag, "HcclAllReduceGraphMode", true；返回行为由所在完整宏决定。
        sendBuf, recvBuf, sendCount, dataType, op, stream, param.tag, "HcclAllReduceGraphMode", true));
    // 执行AllReduce
    // [中文导读] [AllReduce逐行 S109] 调用 CHK_RET_AND_PRINT_IDE 完成当前参数所指的子步骤；本行实参为 CHK_RET_AND_PRINT_IDE(。
    CHK_RET_AND_PRINT_IDE(
        // [中文导读] [AllReduce逐行 S110] 调用 AllReduceOutPlaceGraphMode 完成当前参数所指的子步骤；本行实参为 AllReduceOutPlaceGraphMode(sendBuf, recvBuf, sendCount, dataType, op, comm, stream, resPack, param),。
        AllReduceOutPlaceGraphMode(sendBuf, recvBuf, sendCount, dataType, op, comm, stream, resPack, param),
        // [中文导读] [AllReduce逐行 S111] 调用 c_str 完成当前参数所指的子步骤；本行实参为 tagStr.c_str())。
        tagStr.c_str());
    // [中文导读] [AllReduce逐行 S112] 调用 LogHcclExit 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(LogHcclExit("HcclAllReduceGraphMode", param.tag, startut, true));

    // [中文导读] [AllReduce逐行 S114] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S115] 结束 HcclAllReduceGraphMode 实现；其返回状态或已写回字段由调用者接收。
}

namespace ops_hccl {
// [中文导读] [AllReduce逐行 S118] 定义 AllReduceInitAndCheck 入口：解析环境，检查四个关键句柄，创建通信域关联 tag，验证 Rank、元素数量、可归约类型与运算组合。
HcclResult AllReduceInitAndCheck(
    // [中文导读] [AllReduce逐行 S119] 续接 AllReduceInitAndCheck 的入口参数/基类初始化：HcclComm comm, void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,；引用参数按声明的 const 限制读写。
    HcclComm comm, void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
    // [中文导读] [AllReduce逐行 S120] 续接 AllReduceInitAndCheck 的入口参数/基类初始化：const aclrtStream stream, OpParam& param)；引用参数按声明的 const 限制读写。
    const aclrtStream stream, OpParam& param)
// [中文导读] [AllReduce逐行 S121] 进入 AllReduceInitAndCheck 的实现作用域；解析环境，检查四个关键句柄，创建通信域关联 tag，验证 Rank、元素数量、可归约类型与运算组合。
{
    // 入口的地方先解析环境变量，在初始化环境变量的时候需要设置为AICPU展开
    // [中文导读] 读取环境配置，再验证指针和通信域信息，保证后续调度拥有有效基础参数。
    // [中文导读] [AllReduce逐行 S124] 解析环境配置供模式与算法选择使用；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(InitEnvConfig());

    // 参数校验等工作
    // [中文导读] [AllReduce逐行 S127] 拒绝空输入、输出、通信域或流句柄；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(CheckAllReduceInputPara(comm, sendBuf, recvBuf, stream));
    // [中文导读] [AllReduce逐行 S128] 设置 rankSize 为 INVALID_VALUE_RANKSIZE；该值供下方当前分支使用。
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    // [中文导读] [AllReduce逐行 S129] 调用 HcclGetRankSize 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    // [中文导读] [AllReduce逐行 S130] 设置 userRank 为 INVALID_VALUE_RANKID；该值供下方当前分支使用。
    u32 userRank = INVALID_VALUE_RANKID;
    // [中文导读] [AllReduce逐行 S131] 调用 HcclGetRankId 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(HcclGetRankId(comm, &userRank));
    // [中文导读] [AllReduce逐行 S132] 调用 HcclGetCommName 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(HcclGetCommName(comm, param.commName));

    // topoInfo的tag，所有相同的算子可以共享
    // [中文导读] 以通信域名称生成 AllReduce tag，供拓扑、资源上下文与错误日志关联使用。
    // [中文导读] [AllReduce逐行 S136] 设置 ret 为 sprintf_s(param.tag, sizeof(param.tag), "AllReduce_%s", param.commName)；该值供下方当前分支使用。
    int ret = sprintf_s(param.tag, sizeof(param.tag), "AllReduce_%s", param.commName);
    // [中文导读] [AllReduce逐行 S137] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
    CHK_PRT_RET((ret <= 0), "failed to fill param.tag", HCCL_E_INTERNAL);

    // [中文导读] [AllReduce逐行 S139] 调用 HcclCheckTag 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(HcclCheckTag(param.tag));
    // [中文导读] [AllReduce逐行 S140] 调用 CHK_RET_AND_PRINT_IDE 完成当前参数所指的子步骤；本行实参为 CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, userRank), param.tag)。
    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, userRank), param.tag);
    // [中文导读] 依次检查元素数量、可归约的数据类型以及该类型支持的归约运算。
    // [中文导读] [AllReduce逐行 S142] 调用 CheckCount 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(CheckCount(count));
    // [中文导读] [AllReduce逐行 S143] 调用 CheckDataType 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(CheckDataType(dataType, true));
    // [中文导读] [AllReduce逐行 S144] 调用 CheckReduceOp 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(CheckReduceOp(dataType, op));

    // [中文导读] [AllReduce逐行 S146] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S147] 结束 AllReduceInitAndCheck 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S149] 声明返回类型 HcclResult，分别由错误码传播或候选匹配协议解释。
HcclResult
// [中文导读] [AllReduce逐行 S150] 定义 CheckAllReduceInputPara 入口：分别报告并拒绝空 stream、comm、sendBuf、recvBuf；不在此计算网络资源。
CheckAllReduceInputPara(const HcclComm comm, const void* sendBuf, const void* recvBuf, const aclrtStream stream)
// [中文导读] [AllReduce逐行 S151] 进入 CheckAllReduceInputPara 的实现作用域；分别报告并拒绝空 stream、comm、sendBuf、recvBuf；不在此计算网络资源。
{
    // 入参合法性校验
    // [中文导读] [AllReduce逐行 S153] 调用 RPT_INPUT_ERR 完成当前参数所指的子步骤；本行实参为 RPT_INPUT_ERR(。
    RPT_INPUT_ERR(
        // [中文导读] [AllReduce逐行 S154] 续接 CheckAllReduceInputPara 当前语句的具体实参/字段：stream 等于 nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),；由其完整表达式完成参数组装、检查或结果写回。
        stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        // [中文导读] [AllReduce逐行 S155] 建立本阶段局部对象 std::vector<std::string>({"HcclAllReduce", "nullptr", "stream", "non-null pointer"}))，供 CheckAllReduceInputPara 下方参数组装和子调用使用。
        std::vector<std::string>({"HcclAllReduce", "nullptr", "stream", "non-null pointer"}));
    // [中文导读] [AllReduce逐行 S156] 调用 CHK_PTR_NULL 完成当前参数所指的子步骤；本行实参为 CHK_PTR_NULL(stream)。
    CHK_PTR_NULL(stream);
    // [中文导读] [AllReduce逐行 S157] 调用 RPT_INPUT_ERR 完成当前参数所指的子步骤；本行实参为 RPT_INPUT_ERR(。
    RPT_INPUT_ERR(
        // [中文导读] [AllReduce逐行 S158] 续接 CheckAllReduceInputPara 当前语句的具体实参/字段：comm 等于 nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),；由其完整表达式完成参数组装、检查或结果写回。
        comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        // [中文导读] [AllReduce逐行 S159] 建立本阶段局部对象 std::vector<std::string>({"HcclAllReduce", "nullptr", "comm", "non-null pointer"}))，供 CheckAllReduceInputPara 下方参数组装和子调用使用。
        std::vector<std::string>({"HcclAllReduce", "nullptr", "comm", "non-null pointer"}));
    // [中文导读] [AllReduce逐行 S160] 调用 CHK_PTR_NULL 完成当前参数所指的子步骤；本行实参为 CHK_PTR_NULL(comm)。
    CHK_PTR_NULL(comm);
    // [中文导读] [AllReduce逐行 S161] 调用 RPT_INPUT_ERR 完成当前参数所指的子步骤；本行实参为 RPT_INPUT_ERR(。
    RPT_INPUT_ERR(
        // [中文导读] [AllReduce逐行 S162] 续接 CheckAllReduceInputPara 当前语句的具体实参/字段：sendBuf 等于 nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),；由其完整表达式完成参数组装、检查或结果写回。
        sendBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        // [中文导读] [AllReduce逐行 S163] 建立本阶段局部对象 std::vector<std::string>({"HcclAllReduce", "nullptr", "sendBuf", "non-null pointer"}))，供 CheckAllReduceInputPara 下方参数组装和子调用使用。
        std::vector<std::string>({"HcclAllReduce", "nullptr", "sendBuf", "non-null pointer"}));
    // [中文导读] [AllReduce逐行 S164] 调用 CHK_PTR_NULL 完成当前参数所指的子步骤；本行实参为 CHK_PTR_NULL(sendBuf)。
    CHK_PTR_NULL(sendBuf);
    // [中文导读] [AllReduce逐行 S165] 调用 RPT_INPUT_ERR 完成当前参数所指的子步骤；本行实参为 RPT_INPUT_ERR(。
    RPT_INPUT_ERR(
        // [中文导读] [AllReduce逐行 S166] 续接 CheckAllReduceInputPara 当前语句的具体实参/字段：recvBuf 等于 nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),；由其完整表达式完成参数组装、检查或结果写回。
        recvBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        // [中文导读] [AllReduce逐行 S167] 建立本阶段局部对象 std::vector<std::string>({"HcclAllReduce", "nullptr", "recvBuf", "non-null pointer"}))，供 CheckAllReduceInputPara 下方参数组装和子调用使用。
        std::vector<std::string>({"HcclAllReduce", "nullptr", "recvBuf", "non-null pointer"}));
    // [中文导读] [AllReduce逐行 S168] 调用 CHK_PTR_NULL 完成当前参数所指的子步骤；本行实参为 CHK_PTR_NULL(recvBuf)。
    CHK_PTR_NULL(recvBuf);

    // [中文导读] [AllReduce逐行 S170] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S171] 结束 CheckAllReduceInputPara 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S173] 定义 FillAllReduceOpParam 入口：把用户地址、元素 count、类型、归约 op、模式与设备身份组装成统一 OpParam；输入输出字节容量相等。
HcclResult FillAllReduceOpParam(
    // [中文导读] [AllReduce逐行 S174] 续接 FillAllReduceOpParam 的入口参数/基类初始化：void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, const HcclComm comm,；引用参数按声明的 const 限制读写。
    void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, const HcclComm comm,
    // [中文导读] [AllReduce逐行 S175] 续接 FillAllReduceOpParam 的入口参数/基类初始化：aclrtStream stream, OpMode opMode, OpParam& param)；引用参数按声明的 const 限制读写。
    aclrtStream stream, OpMode opMode, OpParam& param)
// [中文导读] [AllReduce逐行 S176] 进入 FillAllReduceOpParam 的实现作用域；把用户地址、元素 count、类型、归约 op、模式与设备身份组装成统一 OpParam；输入输出字节容量相等。
{
    // [中文导读] 归约前后元素数量相同，因此输入和输出使用相同的字节容量。
    // [中文导读] [AllReduce逐行 S178] 设置 单元素字节数 为 类型到元素字节数的查找表[dataType]；该值供下方当前分支使用。
    u32 perDataSize = DATATYPE_SIZE_TABLE[dataType];
    // [中文导读] [AllReduce逐行 S179] 设置 outputSize 为 count * 单元素字节数；该值供下方当前分支使用。
    u64 outputSize = count * perDataSize;
    // [中文导读] [AllReduce逐行 S180] 设置 inputSize 为 outputSize；该值供下方当前分支使用。
    u64 inputSize = outputSize;

    // [中文导读] 保存流、通信域和归约模式，并先关闭对称内存标志，后面按实际条件重新开启。
    // [中文导读] [AllReduce逐行 S183] 设置 param.hcclComm 为 comm；该值供下方当前分支使用。
    param.hcclComm = comm;
    // [中文导读] [AllReduce逐行 S184] 设置 param.stream 为 stream；该值供下方当前分支使用。
    param.stream = stream;
    // [中文导读] [AllReduce逐行 S185] 设置 归约运算 为 op；该值供下方当前分支使用。
    param.reduceType = op;
    // [中文导读] [AllReduce逐行 S186] 设置 调用模式 为 opMode；该值供下方当前分支使用。
    param.opMode = opMode;
    // [中文导读] [AllReduce逐行 S187] 设置 对称内存标志 为 false；该值供下方当前分支使用。
    param.supportSymmetricMemory = false;

    // [中文导读] [AllReduce逐行 S189] 设置 HcclDevType deviceType 为 HcclDevType::DEV_TYPE_COUNT；该值供下方当前分支使用。
    HcclDevType deviceType = HcclDevType::DEV_TYPE_COUNT;
    // [中文导读] [AllReduce逐行 S190] 调用 HcclGetDeviceType 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(HcclGetDeviceType(deviceType));

    // 参数准备
    // [中文导读] 同时记录字节容量和 count/dataType 元素描述；底层归约原语使用元素数量。
    // [中文导读] [AllReduce逐行 S194] 设置 用户输入基址 为 sendBuf；该值供下方当前分支使用。
    param.inputPtr = sendBuf;
    // [中文导读] [AllReduce逐行 S195] 设置 输入字节容量 为 inputSize；该值供下方当前分支使用。
    param.inputSize = inputSize;
    // [中文导读] [AllReduce逐行 S196] 设置 用户输出基址 为 recvBuf；该值供下方当前分支使用。
    param.outputPtr = recvBuf;
    // [中文导读] [AllReduce逐行 S197] 设置 输出字节容量 为 outputSize；该值供下方当前分支使用。
    param.outputSize = outputSize;
    // [中文导读] [AllReduce逐行 S198] 设置 本 Rank 输入元素数 为 count；该值供下方当前分支使用。
    param.DataDes.count = count;
    // [中文导读] [AllReduce逐行 S199] 设置 输入元素类型 为 dataType；该值供下方当前分支使用。
    param.DataDes.dataType = dataType;
    // [中文导读] [AllReduce逐行 S200] 设置 param.opType 为 HcclCMDType::HCCL_CMD_ALLREDUCE；该值供下方当前分支使用。
    param.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    // [中文导读] [AllReduce逐行 S201] 设置 param.enableDetour 为 false；该值供下方当前分支使用。
    param.enableDetour = false;
    // [中文导读] [AllReduce逐行 S202] 设置 param.deviceType 为 deviceType；该值供下方当前分支使用。
    param.deviceType = deviceType;
    // [中文导读] [AllReduce逐行 S203] 设置 归约运算 为 op；该值供下方当前分支使用。
    param.reduceType = op;
    // [中文导读] [AllReduce逐行 S204] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S205] 结束 FillAllReduceOpParam 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S207] 定义 AllReduceOutPlaceCommon 入口：新流程分发：兼容 CCU、快速发射、AIV 回放、单 Rank、对称内存筛选，最后选算法并进入 HcclExecOp。
HcclResult AllReduceOutPlaceCommon(
    // [中文导读] [AllReduce逐行 S208] 续接 AllReduceOutPlaceCommon 的入口参数/基类初始化：void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm,；引用参数按声明的 const 限制读写。
    void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm,
    // [中文导读] [AllReduce逐行 S209] 续接 AllReduceOutPlaceCommon 的入口参数/基类初始化：aclrtStream stream, OpMode opMode, const ResPackGraphMode& resPack, OpParam& param)；引用参数按声明的 const 限制读写。
    aclrtStream stream, OpMode opMode, const ResPackGraphMode& resPack, OpParam& param)
// [中文导读] [AllReduce逐行 S210] 进入 AllReduceOutPlaceCommon 的实现作用域；新流程分发：兼容 CCU、快速发射、AIV 回放、单 Rank、对称内存筛选，最后选算法并进入 HcclExecOp。
{
    // [中文导读] [AllReduce逐行 S211] 开始 HCCL_INFO 诊断输出，记录 AllReduceOutPlaceCommon 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_INFO("Start to execute AllReduceOutPlace");

    // [中文导读] 为单算子或图模式补齐同一份 OpParam，再从通信域取得展开方式。
    // [中文导读] [AllReduce逐行 S214] 建立统一算子描述；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(FillAllReduceOpParam(sendBuf, recvBuf, count, dataType, op, comm, stream, opMode, param));

    // [中文导读] [AllReduce逐行 S216] 调用 HcclGetOpExpansionMode 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(HcclGetOpExpansionMode(comm, param));

    // 9.0.0 ccu模式走老流程
    // [中文导读] 9.0.0 的单算子 CCU 调度回退到兼容实现，以避开当前新资源路径。
    // [中文导读] [AllReduce逐行 S220] 分支条件为 opMode 等于 单算子 OPBASE 且 GetHcommVersion() 等于 CANN_VERSION(9, 0, 0；成立进入本块，未成立继续后续分支。
    if (opMode == OpMode::OPBASE && GetHcommVersion() == CANN_VERSION(9, 0, 0)
        // [中文导读] [AllReduce逐行 S221] 补充同一条件的 并且 子条件：实际通信引擎 等于 CommEngine::CCU 引擎。
        && param.engine == CommEngine::COMM_ENGINE_CCU) {
        // [中文导读] [AllReduce逐行 S222] 直接返回 调用 HcclAllReduceInner 完成当前参数所指的子步骤 的结果，调用者取得本分支结果。
        return HcclAllReduceInner(sendBuf, recvBuf, count, dataType, op, comm, stream);
    // [中文导读] [AllReduce逐行 S223] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // [中文导读] 单算子先尝试复用 CCU 快速发射信息，命中后跳过常规拓扑和资源准备。
    // [中文导读] [AllReduce逐行 S226] 设置 CcuFastLaunchCtx* ccuFastLaunchCtx 为 nullptr；该值供下方当前分支使用。
    CcuFastLaunchCtx* ccuFastLaunchCtx = nullptr;
    // [中文导读] [AllReduce逐行 S227] 分支条件为 (opMode 等于 单算子 OPBASE) 且 ShouldGoCcuFastLaunch(comm, param, &ccuFastLaunchCtx；成立进入本块，未成立继续后续分支。
    if ((opMode == OpMode::OPBASE) && ShouldGoCcuFastLaunch(comm, param, &ccuFastLaunchCtx)) {
        // [中文导读] [AllReduce逐行 S228] 直接返回 调用 HcclExecOpCcuFastLaunch 完成当前参数所指的子步骤 的结果，调用者取得本分支结果。
        return HcclExecOpCcuFastLaunch(comm, param, ccuFastLaunchCtx);
    // [中文导读] [AllReduce逐行 S229] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // [中文导读] AIV 缓存回放成功时提前返回；未命中才运行下面的单 Rank 或 selector 分支。
    // [中文导读] [AllReduce逐行 S232] 分支条件为 实际通信引擎 等于 CommEngine::AIV 引擎；成立进入本块，未成立继续后续分支。
    if (param.engine == CommEngine::COMM_ENGINE_AIV) {
        // [中文导读] [AllReduce逐行 S233] 设置 aivCacheHit 为 false；该值供下方当前分支使用。
        bool aivCacheHit = false;
        // [中文导读] [AllReduce逐行 S234] 调用 HcclAivCacheCheckAndReplay 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。
        CHK_RET(HcclAivCacheCheckAndReplay(comm, param, aivCacheHit));
        // [中文导读] [AllReduce逐行 S235] 分支条件为 aivCacheHit；成立进入本块，未成立继续后续分支。
        if (aivCacheHit) {
            // [中文导读] [AllReduce逐行 S236] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
            return HCCL_SUCCESS;
        // [中文导读] [AllReduce逐行 S237] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
    // [中文导读] [AllReduce逐行 S238] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // 单卡校验
    // [中文导读] 从通信域取得规模；只有一个 Rank 时由本地处理完成归约语义。
    // [中文导读] [AllReduce逐行 S242] 声明本阶段局部变量 u32 userRankSize，实际值由后续查询/计算填写。
    u32 userRankSize;
    // [中文导读] [AllReduce逐行 S243] 调用 HcclGetRankSize 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(HcclGetRankSize(comm, &userRankSize));
    // [中文导读] [AllReduce逐行 S244] 分支条件为 userRankSize 等于 1；成立进入本块，未成立继续后续分支。
    if (userRankSize == 1) {
        // [中文导读] [AllReduce逐行 S245] 开始 HCCL_WARNING 诊断输出，记录 AllReduceOutPlaceCommon 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_WARNING("[%s] ranksize == 1, enter SingleRankProc", __func__);
        // [中文导读] [AllReduce逐行 S246] 调用 SingleRankProc 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。
        CHK_RET(SingleRankProc(comm, param));
        // [中文导读] [AllReduce逐行 S247] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
        return HcclResult::HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S248] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // [中文导读] 运行时版本和单算子模式允许时，先探测输入输出是否满足对称内存条件。
    // [中文导读] [AllReduce逐行 S251] 分支条件为 GetHcommVersion() 至少 CANN_VERSION(9, 1, 0) 且 调用模式 等于 单算子 OPBASE；成立进入本块，未成立继续后续分支。
    if (GetHcommVersion() >= CANN_VERSION(9, 1, 0) && param.opMode == OpMode::OPBASE) {
        // [中文导读] [AllReduce逐行 S252] 调用 CheckAndSetSymmetricMemory 完成当前参数所指的子步骤；本行实参为 CheckAndSetSymmetricMemory(param)。
        CheckAndSetSymmetricMemory(param);
    // [中文导读] [AllReduce逐行 S253] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // [中文导读] 依据拓扑和归约参数选择算法，再用选择结果收紧对称内存支持范围。
    // [中文导读] [AllReduce逐行 S256] 建立本阶段局部对象 std::string algName，供 AllReduceOutPlaceCommon 下方参数组装和子调用使用。
    std::string algName;
    // [中文导读] [AllReduce逐行 S257] 设置 std::unique_ptr<TopoInfoWithNetLayerDetails> topoInfo 为 std::make_unique<TopoInfoWithNetLayerDetails>()；该值供下方当前分支使用。
    std::unique_ptr<TopoInfoWithNetLayerDetails> topoInfo = std::make_unique<TopoInfoWithNetLayerDetails>();
    // [中文导读] [AllReduce逐行 S258] 根据拓扑和配置选算法；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(Selector(comm, param, topoInfo, algName));

    // [中文导读] Mesh CLOS 只保留特定两层 Omni 算法的对称内存路径，其余情况清除候选标志。
    // [中文导读] [AllReduce逐行 S261] 分支条件为 第零层拓扑形状 等于 Mesh 与 CLOS 混合拓扑 且 !第零层 PCIe 混合标志；成立进入本块，未成立继续后续分支。
    if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix) {
        // [中文导读] [AllReduce逐行 S262] 建立本阶段局部对象 const bool isTwoLevelMeshNhrOmni，供 AllReduceOutPlaceCommon 下方参数组装和子调用使用。
        const bool isTwoLevelMeshNhrOmni
            // [中文导读] [AllReduce逐行 S263] 续接 AllReduceOutPlaceCommon 当前语句的具体实参/字段：= algName 等于 "AicpuAllReducePipeLineMeshNHR" 且 topoInfo->topoLevelNums 等于 TOPO_LEVEL_NUM_1；由其完整表达式完成参数组装、检查或结果写回。
            = algName == "AicpuAllReducePipeLineMeshNHR" && topoInfo->topoLevelNums == TOPO_LEVEL_NUM_1;
        // [中文导读] [AllReduce逐行 S264] 分支条件为 !isTwoLevelMeshNhrOmni；成立进入本块，未成立继续后续分支。
        if (!isTwoLevelMeshNhrOmni) {
            // [中文导读] [AllReduce逐行 S265] 设置 对称内存标志 为 false；该值供下方当前分支使用。
            param.supportSymmetricMemory = false;
        // [中文导读] [AllReduce逐行 S266] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
    // [中文导读] [AllReduce逐行 S267] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
    } else {
        // [中文导读] 普通 Mesh 还要求 AICPU_TS，并排除这里列出的 64 位类型及乘积归约。
        // [中文导读] [AllReduce逐行 S269] 建立本阶段局部对象 const bool supportSymmetricMemory，供 AllReduceOutPlaceCommon 下方参数组装和子调用使用。
        const bool supportSymmetricMemory
            // [中文导读] [AllReduce逐行 S270] 续接 AllReduceOutPlaceCommon 当前语句的具体实参/字段：= (实际通信引擎 等于 CommEngine::AICPU_TS 引擎 且 第零层拓扑形状 等于 单层 Mesh1D；由其完整表达式完成参数组装、检查或结果写回。
            = (param.engine == CommEngine::COMM_ENGINE_AICPU_TS && topoInfo->level0Topo == Level0Shape::MESH_1D
               // [中文导读] [AllReduce逐行 S271] 补充同一条件的 并且 子条件：输入元素类型 不等于 HcclDataType::有符号 64 位整数。
               && param.DataDes.dataType != HcclDataType::HCCL_DATA_TYPE_INT64
               // [中文导读] [AllReduce逐行 S272] 补充同一条件的 并且 子条件：输入元素类型 不等于 HcclDataType::无符号 64 位整数。
               && param.DataDes.dataType != HcclDataType::HCCL_DATA_TYPE_UINT64
               // [中文导读] [AllReduce逐行 S273] 补充同一条件的 并且 子条件：输入元素类型 不等于 HcclDataType::FP64 类型。
               && param.DataDes.dataType != HcclDataType::HCCL_DATA_TYPE_FP64
               // [中文导读] [AllReduce逐行 S274] 补充同一条件的 并且 子条件：归约运算 不等于 HcclReduceOp::乘积归约。
               && param.reduceType != HcclReduceOp::HCCL_REDUCE_PROD);
        // [中文导读] [AllReduce逐行 S275] 分支条件为 !supportSymmetricMemory；成立进入本块，未成立继续后续分支。
        if (!supportSymmetricMemory) {
            // [中文导读] [AllReduce逐行 S276] 设置 对称内存标志 为 false；该值供下方当前分支使用。
            param.supportSymmetricMemory = false;
        // [中文导读] [AllReduce逐行 S277] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
    // [中文导读] [AllReduce逐行 S278] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // [中文导读] 把最终算法与归约配置送入公共执行层；支持标志已经按拓扑和运算条件过滤。
    // [中文导读] [AllReduce逐行 S281] 进入资源与执行器公共调度；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(HcclExecOp(comm, param, topoInfo, algName, resPack));
    // [中文导读] [AllReduce逐行 S282] 开始 HCCL_INFO 诊断输出，记录 AllReduceOutPlaceCommon 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_INFO("Execute AllReduceOutPlace success.");
    // [中文导读] [AllReduce逐行 S283] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S284] 结束 AllReduceOutPlaceCommon 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S286] 定义 AllReduceEntryLog 入口：可配置或强制打印设备、流与输入输出地址等入口信息；日志组装失败仅警告继续。
HcclResult AllReduceEntryLog(
    // [中文导读] [AllReduce逐行 S287] 续接 AllReduceEntryLog 的入口参数/基类初始化：void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, aclrtStream stream,；引用参数按声明的 const 限制读写。
    void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, aclrtStream stream,
    // [中文导读] [AllReduce逐行 S288] 续接 AllReduceEntryLog 的入口参数/基类初始化：const char* tag, const std::string& opName, bool forceLog)；引用参数按声明的 const 限制读写。
    const char* tag, const std::string& opName, bool forceLog)
// [中文导读] [AllReduce逐行 S289] 进入 AllReduceEntryLog 的实现作用域；可配置或强制打印设备、流与输入输出地址等入口信息；日志组装失败仅警告继续。
{
    // [中文导读] [AllReduce逐行 S290] 分支条件为 forceLog 或 GetExternalInputHcclEnableEntryLog(；成立进入本块，未成立继续后续分支。
    if (forceLog || GetExternalInputHcclEnableEntryLog()) {
        // [中文导读] [AllReduce逐行 S291] 设置 s32 deviceId 为 0；该值供下方当前分支使用。
        s32 deviceId = 0;
        // [中文导读] [AllReduce逐行 S292] 调用 ACLCHECK 完成当前参数所指的子步骤；本行实参为 ACLCHECK(aclrtGetDevice(&deviceId))。
        ACLCHECK(aclrtGetDevice(&deviceId));
        // [中文导读] [AllReduce逐行 S293] 设置 s32 streamId 为 0；该值供下方当前分支使用。
        s32 streamId = 0;
        // [中文导读] [AllReduce逐行 S294] 调用 ACLCHECK 完成当前参数所指的子步骤；本行实参为 ACLCHECK(aclrtStreamGetId(stream, &streamId))。
        ACLCHECK(aclrtStreamGetId(stream, &streamId));
        // [中文导读] [AllReduce逐行 S295] 续接 AllReduceEntryLog 当前语句的具体实参/字段：char stackLogBuffer[LOG_TMPBUF_SIZE]；由其完整表达式完成参数组装、检查或结果写回。
        char stackLogBuffer[LOG_TMPBUF_SIZE];
        // [中文导读] [AllReduce逐行 S296] 设置 s32 ret 为 snprintf_s(；该值供下方当前分支使用。
        s32 ret = snprintf_s(
            // [中文导读] [AllReduce逐行 S297] 续接 AllReduceEntryLog 当前语句的具体实参/字段：stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,；由其完整表达式完成参数组装、检查或结果写回。
            stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            // [中文导读] [AllReduce逐行 S298] 续接本次调用的字符串常量 "tag[%s], sendBuf[%p], recvBuf[%p], count[%llu], dataType[%s], reduceOp[%s], streamId[%d], deviceId[%d]",，由所在注册/日志/条件语句整体使用。
            "tag[%s], sendBuf[%p], recvBuf[%p], count[%llu], dataType[%s], reduceOp[%s], streamId[%d], deviceId[%d]",
            // [中文导读] [AllReduce逐行 S299] 调用 GetDataTypeEnumStr 完成当前参数所指的子步骤；本行实参为 tag, sendBuf, recvBuf, count, GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str(),。
            tag, sendBuf, recvBuf, count, GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str(),
            // [中文导读] [AllReduce逐行 S300] 续接 AllReduceEntryLog 当前语句的具体实参/字段：streamId, deviceId)；由其完整表达式完成参数组装、检查或结果写回。
            streamId, deviceId);

        // [中文导读] [AllReduce逐行 S302] 为 AllReduceEntryLog 的诊断/错误宏提供实参：CHK_PRT_CONT(ret 等于 -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag，与前面的格式占位依次对应。
        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag));
        // [中文导读] [AllReduce逐行 S303] 设置 logInfo 为 "Entry-" + opName + ":" + std::string(stackLogBuffer)；该值供下方当前分支使用。
        std::string logInfo = "Entry-" + opName + ":" + std::string(stackLogBuffer);
        // [中文导读] [AllReduce逐行 S304] 调用 HCCL_RUN_INFO 完成当前参数所指的子步骤；本行实参为 HCCL_RUN_INFO("%s", logInfo.c_str())。
        HCCL_RUN_INFO("%s", logInfo.c_str());
    // [中文导读] [AllReduce逐行 S305] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S306] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S307] 结束 AllReduceEntryLog 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S309] 定义 AllReduceOutPlaceGraphMode 入口：用 OFFLOAD 和外部图资源包转调公共 AllReduce 实现。
HcclResult AllReduceOutPlaceGraphMode(
    // [中文导读] [AllReduce逐行 S310] 续接 AllReduceOutPlaceGraphMode 的入口参数/基类初始化：void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm,；引用参数按声明的 const 限制读写。
    void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm,
    // [中文导读] [AllReduce逐行 S311] 续接 AllReduceOutPlaceGraphMode 的入口参数/基类初始化：aclrtStream stream, const ResPackGraphMode& resPack, OpParam& param)；引用参数按声明的 const 限制读写。
    aclrtStream stream, const ResPackGraphMode& resPack, OpParam& param)
// [中文导读] [AllReduce逐行 S312] 进入 AllReduceOutPlaceGraphMode 的实现作用域；用 OFFLOAD 和外部图资源包转调公共 AllReduce 实现。
{
    // [中文导读] [AllReduce逐行 S313] 开始 HCCL_INFO 诊断输出，记录 AllReduceOutPlaceGraphMode 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_INFO("Start to execute AllReduceOutPlaceGraphMode");
    // [中文导读] [AllReduce逐行 S314] 执行下方完整子调用；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(
        // [中文导读] [AllReduce逐行 S315] 进入共享 AllReduce 模式分发；本行实参为 AllReduceOutPlaceCommon(sendBuf, recvBuf, count, dataType, op, comm, stream, 图模式 OFFLOAD, resPack, param))。
        AllReduceOutPlaceCommon(sendBuf, recvBuf, count, dataType, op, comm, stream, OpMode::OFFLOAD, resPack, param));
    // [中文导读] [AllReduce逐行 S316] 开始 HCCL_INFO 诊断输出，记录 AllReduceOutPlaceGraphMode 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_INFO("Execute AllReduceOutPlaceGraphMode success.");
    // [中文导读] [AllReduce逐行 S317] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S318] 结束 AllReduceOutPlaceGraphMode 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S320] 定义 AllReduceOutPlace 入口：用 OPBASE 和默认资源包转调公共 AllReduce 实现。
HcclResult AllReduceOutPlace(
    // [中文导读] [AllReduce逐行 S321] 续接 AllReduceOutPlace 的入口参数/基类初始化：void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm,；引用参数按声明的 const 限制读写。
    void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm,
    // [中文导读] [AllReduce逐行 S322] 续接 AllReduceOutPlace 的入口参数/基类初始化：aclrtStream stream, OpParam& param)；引用参数按声明的 const 限制读写。
    aclrtStream stream, OpParam& param)
// [中文导读] [AllReduce逐行 S323] 进入 AllReduceOutPlace 的实现作用域；用 OPBASE 和默认资源包转调公共 AllReduce 实现。
{
    // [中文导读] [AllReduce逐行 S324] 开始 HCCL_INFO 诊断输出，记录 AllReduceOutPlace 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_INFO("Start to execute AllReduceOutPlace");
    // [中文导读] [AllReduce逐行 S325] 进入共享 AllReduce 模式分发；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(AllReduceOutPlaceCommon(
        // [中文导读] [AllReduce逐行 S326] 调用 ResPackGraphMode 完成当前参数所指的子步骤；本行实参为 sendBuf, recvBuf, count, dataType, op, comm, stream, 单算子 OPBASE, ResPackGraphMode(), param))。
        sendBuf, recvBuf, count, dataType, op, comm, stream, OpMode::OPBASE, ResPackGraphMode(), param));
    // [中文导读] [AllReduce逐行 S327] 开始 HCCL_INFO 诊断输出，记录 AllReduceOutPlace 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_INFO("Execute AllReduceOutPlace success.");
    // [中文导读] [AllReduce逐行 S328] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S329] 结束 AllReduceOutPlace 实现；其返回状态或已写回字段由调用者接收。
}

} // namespace ops_hccl
