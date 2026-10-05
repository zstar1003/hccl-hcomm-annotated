/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_all_reduce_sole_executor.h"
#include "alg_attrs_registry.h"
#include "ins_temp_all_reduce_mesh_1D_one_shot.h"
#include "ins_temp_all_reduce_mesh_1D_two_shot.h"
#include "ins_temp_all_reduce_nhr.h"
#include "ins_temp_all_reduce_mesh_1D_two_shot_mesh_chunk.h"
#include "ins_temp_all_reduce_aicpu_reduce_nhr.h"
#ifndef AICPU_COMPILE
#include "aiv_temp_all_reduce_mesh_1D_oneshot.h"
#include "aiv_temp_all_reduce_mesh_1D_twoshot.h"
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#include "ccu_temp_all_reduce_mesh_1D_one_shot.h"
#include "ccu_temp_all_reduce_mesh_1D_mem2mem.h"
#include "ccu_temp_all_reduce_mesh_1D.h"
#include "ccu_temp_all_reduce_nhr_1D_mem2mem.h"
#include "ccu_temp_all_reduce_mesh_1D_2die_oneshot.h"
#include "ccu_temp_all_reduce_mesh_1D_mem2mem_2die_oneshot.h"
#include "ccu_temp_all_reduce_nhr_mem2mem_1D_multi_jetty.h"
#include "ccu_temp_all_reduce_concurrent_mesh_nhr.h"
#include "topo_match_concurrent.h"
#endif /* CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0) */
#endif

#include "alg_attrs_registry.h"
#include "auto_selector_base.h"
#ifndef AICPU_COMPILE
#include "hccl_aiv_utils.h"
#endif

namespace ops_hccl {
constexpr u32 MAX_RANK_NUM_FOR_CONCURRENT_ALGO = 4; // 与selector保持一致：并发算法的卡数上限

template <typename AlgTopoMatch, typename InsAlgTemplate>
InsV2AllReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InsV2AllReduceSoleExecutor()
{}

// [中文导读] [AllReduce逐行 S47] 声明模板类型参数，将拓扑匹配器与具体算法模板在编译期绑定。
template <typename AlgTopoMatch, typename InsAlgTemplate>
// [中文导读] [AllReduce逐行 S48] 定义 CalcCostCoeff 入口：以算法属性匹配实际物理层并获取端口信息，调用模板计算成本系数；OneShot 模板只为 ≤8 Rank 提供系数。
std::vector<CostModelParam> InsV2AllReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcCostCoeff(
    // [中文导读] [AllReduce逐行 S49] 续接 CalcCostCoeff 的入口参数/基类初始化：HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param)；引用参数按声明的 const 限制读写。
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param)
// [中文导读] [AllReduce逐行 S50] 进入 CalcCostCoeff 的实现作用域；以算法属性匹配实际物理层并获取端口信息，调用模板计算成本系数；OneShot 模板只为 ≤8 Rank 提供系数。
{
    // [中文导读] [AllReduce逐行 S51] 显式标记 comm 在此兼容/default 分支未使用，避免编译器未使用参数警告。
    (void)comm;
    // [中文导读] [AllReduce逐行 S52] 声明本阶段局部变量 AlgHierarchyInfoForAllLevel algHierarchyInfo，实际值由后续查询/计算填写。
    AlgHierarchyInfoForAllLevel algHierarchyInfo;
// [中文导读] [AllReduce逐行 S53] 以下资源/成本逻辑仅在 Host 编译版本执行；AICPU_COMPILE 构建跳过该段。
#ifndef AICPU_COMPILE
    // [中文导读] [AllReduce逐行 S54] 设置 const AlgAttrs* attrs 为 AlgAttrsRegistry::Instance().Get(std::string(algName))；该值供下方当前分支使用。
    const AlgAttrs* attrs = AlgAttrsRegistry::Instance().Get(std::string(algName));
// [中文导读] [AllReduce逐行 S55] 转入该版本或构建条件不成立时的兼容分支。
#else
    // AICPU 独立核库(scatter_aicpu_kernel.so)不链接 host-only 的 AlgAttrsRegistry,
    // device 侧亦无 costmodel 调用链, 置空走 skip 分支
    // [中文导读] [AllReduce逐行 S58] 设置 const AlgAttrs* attrs 为 nullptr；该值供下方当前分支使用。
    const AlgAttrs* attrs = nullptr;
// [中文导读] [AllReduce逐行 S59] 结束该编译期能力/Host 边界，后续公共返回逻辑在相应构建中保留。
#endif
    // 探测路径直接调 MatchTopo（不走 CalcAlgHierarchyInfoV2 的 CHK_RET）：
    // costmodel 迭代时"不匹配"是正常事件，避免执行路径语义的 ERROR 日志刷屏
    // [中文导读] [AllReduce逐行 S62] 声明本阶段局部变量 AlgTopoMatch topoMatch，实际值由后续查询/计算填写。
    AlgTopoMatch topoMatch;
    // [中文导读] [AllReduce逐行 S63] 续接 CalcCostCoeff 当前语句的具体实参/字段：HcclResult matchRet；由其完整表达式完成参数组装、检查或结果写回。
    HcclResult matchRet
        // [中文导读] [AllReduce逐行 S64] 按算法属性筛选并构造算法拓扑；本行实参为 = (attrs 不等于 nullptr) ? topoMatch.MatchTopo(topoInfo, algHierarchyInfo, *attrs) : HcclResult::参数错误。
        = (attrs != nullptr) ? topoMatch.MatchTopo(topoInfo, algHierarchyInfo, *attrs) : HcclResult::HCCL_E_PARA;
    // [中文导读] [AllReduce逐行 S65] 分支条件为 matchRet 不等于 HcclResult::成功状态；成立进入本块，未成立继续后续分支。
    if (matchRet != HcclResult::HCCL_SUCCESS) {
        // [中文导读] [AllReduce逐行 S66] 开始 HCCL_INFO 诊断输出，记录 CalcCostCoeff 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_INFO("[CalcCostCoeff] algName=%s topo match not support, skip.", algName);
        // [中文导读] [AllReduce逐行 S67] 直接返回 {}，调用者取得本分支结果。
        return {};
    // [中文导读] [AllReduce逐行 S68] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S69] 设置 rankSize 为 通信域 Rank 总数；该值供下方当前分支使用。
    u32 rankSize = topoInfo->userRankSize;
    // [中文导读] [AllReduce逐行 S70] 设置 isPod 为 Pod 拓扑标志；该值供下方当前分支使用。
    bool isPod = topoInfo->isPod;
    // [中文导读] [AllReduce逐行 S71] 续接 CalcCostCoeff 当前语句的具体实参/字段：CommTopo netTypeLevel0；由其完整表达式完成参数组装、检查或结果写回。
    CommTopo netTypeLevel0
        // [中文导读] [AllReduce逐行 S72] 设置  为 GetPhysicalLevelTopoType(topoInfo, static_cast<u32>(algHierarchyInfo.physicalIdxForAlgoLevels[0][0]))；该值供下方当前分支使用。
        = GetPhysicalLevelTopoType(topoInfo, static_cast<u32>(algHierarchyInfo.physicalIdxForAlgoLevels[0][0]));
    // [中文导读] [AllReduce逐行 S73] 建立本阶段局部对象 std::vector<u32> portNumLevel0，供 CalcCostCoeff 下方参数组装和子调用使用。
    std::vector<u32> portNumLevel0
        // [中文导读] [AllReduce逐行 S74] 设置  为 GetPhysicalLevelPortNums(topoInfo, static_cast<u32>(algHierarchyInfo.physicalIdxForAlgoLevels[0][0]))；该值供下方当前分支使用。
        = GetPhysicalLevelPortNums(topoInfo, static_cast<u32>(algHierarchyInfo.physicalIdxForAlgoLevels[0][0]));
    // [中文导读] [AllReduce逐行 S75] 分支条件为 portNumLevel0.empty(；成立进入本块，未成立继续后续分支。
    if (portNumLevel0.empty()) {
        // [中文导读] [AllReduce逐行 S76] 开始 HCCL_WARNING 诊断输出，记录 CalcCostCoeff 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_WARNING("[CalcCostCoeff] portNum is empty");
        // [中文导读] [AllReduce逐行 S77] 直接返回 {}，调用者取得本分支结果。
        return {};
    // [中文导读] [AllReduce逐行 S78] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S79] 开始 HCCL_INFO 诊断输出，记录 CalcCostCoeff 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S80] 续接 CalcCostCoeff 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
        "[CalcCostCoeff] rankSize=%d, portNumLevel0=%d, netTypeLevel0=%d", rankSize, portNumLevel0,
        // [中文导读] [AllReduce逐行 S81] 为 CalcCostCoeff 的诊断/错误宏提供实参：static_cast<int>(netTypeLevel0，与前面的格式占位依次对应。
        static_cast<int>(netTypeLevel0));
    // [中文导读] [AllReduce逐行 S82] 直接返回 调用 InsAlgTemplate::CalcCostCoeff 完成当前参数所指的子步骤 的结果，调用者取得本分支结果。
    return InsAlgTemplate::CalcCostCoeff(CalcCostCoeffParam{
        // [中文导读] [AllReduce逐行 S83] 续接 CalcCostCoeff 当前语句的具体实参/字段：rankSize, 1.0f / rankSize, netTypeLevel0, BufferType::INPUT, BufferType::OUTPUT, BufferType::HCCL_BUFFER,；由其完整表达式完成参数组装、检查或结果写回。
        rankSize, 1.0f / rankSize, netTypeLevel0, BufferType::INPUT, BufferType::OUTPUT, BufferType::HCCL_BUFFER,
        // [中文导读] [AllReduce逐行 S84] 续接 CalcCostCoeff 当前语句的具体实参/字段：portNumLevel0, isPod})；由其完整表达式完成参数组装、检查或结果写回。
        portNumLevel0, isPod});
// [中文导读] [AllReduce逐行 S85] 结束 CalcCostCoeff 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S87] 声明模板类型参数，将拓扑匹配器与具体算法模板在编译期绑定。
template <typename AlgTopoMatch, typename InsAlgTemplate>
// [中文导读] [AllReduce逐行 S88] 定义 GetAlgNetMeta 入口：匹配物理层并提供该算法的网络类型、数据比例、Rank 数及 SUM 聚合方式。
AlgNetMeta InsV2AllReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::GetAlgNetMeta(
    // [中文导读] [AllReduce逐行 S89] 续接 GetAlgNetMeta 的入口参数/基类初始化：const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param, const char* algName) const；引用参数按声明的 const 限制读写。
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param, const char* algName) const
// [中文导读] [AllReduce逐行 S90] 进入 GetAlgNetMeta 的实现作用域；匹配物理层并提供该算法的网络类型、数据比例、Rank 数及 SUM 聚合方式。
{
    // [中文导读] [AllReduce逐行 S91] 显式标记 param 在此兼容/default 分支未使用，避免编译器未使用参数警告。
    (void)param;
    // [中文导读] [AllReduce逐行 S92] 声明本阶段局部变量 AlgHierarchyInfoForAllLevel algHierarchyInfo，实际值由后续查询/计算填写。
    AlgHierarchyInfoForAllLevel algHierarchyInfo;
// [中文导读] [AllReduce逐行 S93] 以下资源/成本逻辑仅在 Host 编译版本执行；AICPU_COMPILE 构建跳过该段。
#ifndef AICPU_COMPILE
    // [中文导读] [AllReduce逐行 S94] 设置 const AlgAttrs* attrs 为 AlgAttrsRegistry::Instance().Get(std::string(algName))；该值供下方当前分支使用。
    const AlgAttrs* attrs = AlgAttrsRegistry::Instance().Get(std::string(algName));
// [中文导读] [AllReduce逐行 S95] 转入该版本或构建条件不成立时的兼容分支。
#else
    // AICPU 独立核库(scatter_aicpu_kernel.so)不链接 host-only 的 AlgAttrsRegistry,
    // device 侧亦无 costmodel 调用链, 置空走 skip 分支
    // [中文导读] [AllReduce逐行 S98] 设置 const AlgAttrs* attrs 为 nullptr；该值供下方当前分支使用。
    const AlgAttrs* attrs = nullptr;
// [中文导读] [AllReduce逐行 S99] 结束该编译期能力/Host 边界，后续公共返回逻辑在相应构建中保留。
#endif
    // 探测路径直接调 MatchTopo：无 CHK_RET 的 ERROR，且免去 V2 调用所需的多层 const_cast
    // [中文导读] [AllReduce逐行 S101] 声明本阶段局部变量 AlgTopoMatch topoMatch，实际值由后续查询/计算填写。
    AlgTopoMatch topoMatch;
    // [中文导读] [AllReduce逐行 S102] 续接 GetAlgNetMeta 当前语句的具体实参/字段：HcclResult matchRet；由其完整表达式完成参数组装、检查或结果写回。
    HcclResult matchRet
        // [中文导读] [AllReduce逐行 S103] 续接 GetAlgNetMeta 当前语句的具体实参/字段：= (attrs 不等于 nullptr) ?；由其完整表达式完成参数组装、检查或结果写回。
        = (attrs != nullptr) ?
              // [中文导读] [AllReduce逐行 S104] 按算法属性筛选并构造算法拓扑；本行实参为 topoMatch.MatchTopo(const_cast<TopoInfoWithNetLayerDetails*>(topoInfo), algHierarchyInfo, *attrs) :。
              topoMatch.MatchTopo(const_cast<TopoInfoWithNetLayerDetails*>(topoInfo), algHierarchyInfo, *attrs) :
              // [中文导读] [AllReduce逐行 S105] 续接 GetAlgNetMeta 当前语句的具体实参/字段：HcclResult::参数错误；由其完整表达式完成参数组装、检查或结果写回。
              HcclResult::HCCL_E_PARA;
    // [中文导读] [AllReduce逐行 S106] 分支条件为 matchRet 不等于 HcclResult::成功状态；成立进入本块，未成立继续后续分支。
    if (matchRet != HcclResult::HCCL_SUCCESS) {
        // [中文导读] [AllReduce逐行 S107] 开始 HCCL_INFO 诊断输出，记录 GetAlgNetMeta 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_INFO("[GetAlgNetMeta] algName=%s topo match not support, return empty.", algName);
        // [中文导读] [AllReduce逐行 S108] 直接返回 {}，调用者取得本分支结果。
        return {};
    // [中文导读] [AllReduce逐行 S109] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S110] 设置 rankSize 为 通信域 Rank 总数；该值供下方当前分支使用。
    u32 rankSize = topoInfo->userRankSize;
    // [中文导读] [AllReduce逐行 S111] 续接 GetAlgNetMeta 当前语句的具体实参/字段：CommTopo netTypeLevel0；由其完整表达式完成参数组装、检查或结果写回。
    CommTopo netTypeLevel0
        // [中文导读] [AllReduce逐行 S112] 设置  为 GetPhysicalLevelTopoType(topoInfo, static_cast<u32>(algHierarchyInfo.physicalIdxForAlgoLevels[0][0]))；该值供下方当前分支使用。
        = GetPhysicalLevelTopoType(topoInfo, static_cast<u32>(algHierarchyInfo.physicalIdxForAlgoLevels[0][0]));
    // [中文导读] [AllReduce逐行 S113] 声明本阶段局部变量 AlgNetMeta meta，实际值由后续查询/计算填写。
    AlgNetMeta meta;
    // [中文导读] [AllReduce逐行 S114] 对 meta 追加 netTypeLevel0，准备或更新本阶段列表。
    meta.netTypes.push_back(netTypeLevel0);
    // [中文导读] [AllReduce逐行 S115] 设置 meta.intraGroupMode 为 CostAggMode::SUM；该值供下方当前分支使用。
    meta.intraGroupMode = CostAggMode::SUM;
    // [中文导读] [AllReduce逐行 S116] 设置 meta.groupSizes 为 {1}；该值供下方当前分支使用。
    meta.groupSizes = {1};
    // [中文导读] [AllReduce逐行 S117] 设置 meta.dataRatios 为 {1.0f / rankSize}；该值供下方当前分支使用。
    meta.dataRatios = {1.0f / rankSize};
    // [中文导读] [AllReduce逐行 S118] 设置 meta.rankSizes 为 {rankSize}；该值供下方当前分支使用。
    meta.rankSizes = {rankSize};
    // [中文导读] [AllReduce逐行 S119] 直接返回 meta，调用者取得本分支结果。
    return meta;
// [中文导读] [AllReduce逐行 S120] 结束 GetAlgNetMeta 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S122] 声明模板类型参数，将拓扑匹配器与具体算法模板在编译期绑定。
template <typename AlgTopoMatch, typename InsAlgTemplate>
// [中文导读] [AllReduce逐行 S123] 定义 CalcAlgHierarchyInfo 入口：旧层次接口用默认 AlgAttrs 匹配一层拓扑；当前主链使用带属性的 V2。
HcclResult InsV2AllReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcAlgHierarchyInfo(
    // [中文导读] [AllReduce逐行 S124] 续接 CalcAlgHierarchyInfo 的入口参数/基类初始化：HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)；引用参数按声明的 const 限制读写。
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
// [中文导读] [AllReduce逐行 S125] 进入 CalcAlgHierarchyInfo 的实现作用域；旧层次接口用默认 AlgAttrs 匹配一层拓扑；当前主链使用带属性的 V2。
{
    // [中文导读] [AllReduce逐行 S126] 显式标记 comm 在此兼容/default 分支未使用，避免编译器未使用参数警告。
    (void)comm;
    // [中文导读] [AllReduce逐行 S127] 声明本阶段局部变量 AlgTopoMatch topoMatch，实际值由后续查询/计算填写。
    AlgTopoMatch topoMatch;
    // [中文导读] [AllReduce逐行 S128] 按算法属性筛选并构造算法拓扑；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(topoMatch.MatchTopo(topoInfo, algHierarchyInfo, AlgAttrs{}));
    // [中文导读] [AllReduce逐行 S129] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S130] 结束 CalcAlgHierarchyInfo 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S132] 声明模板类型参数，将拓扑匹配器与具体算法模板在编译期绑定。
template <typename AlgTopoMatch, typename InsAlgTemplate>
// [中文导读] [AllReduce逐行 S133] 定义 CalcAlgHierarchyInfoV2 入口：用注册算法属性匹配物理层，产生算法通信域 Rank 列表及物理层映射。
HcclResult InsV2AllReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcAlgHierarchyInfoV2(
    // [中文导读] [AllReduce逐行 S134] 续接 CalcAlgHierarchyInfoV2 的入口参数/基类初始化：TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& algAttrs)；引用参数按声明的 const 限制读写。
    TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& algAttrs)
// [中文导读] [AllReduce逐行 S135] 进入 CalcAlgHierarchyInfoV2 的实现作用域；用注册算法属性匹配物理层，产生算法通信域 Rank 列表及物理层映射。
{
    // [中文导读] [AllReduce逐行 S136] 声明本阶段局部变量 AlgTopoMatch topoMatch，实际值由后续查询/计算填写。
    AlgTopoMatch topoMatch;
    // [中文导读] [AllReduce逐行 S137] 按算法属性筛选并构造算法拓扑；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(topoMatch.MatchTopo(topoInfo, algHierarchyInfo, algAttrs));
    // [中文导读] [AllReduce逐行 S138] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S139] 结束 CalcAlgHierarchyInfoV2 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S141] 声明模板类型参数，将拓扑匹配器与具体算法模板在编译期绑定。
template <typename AlgTopoMatch, typename InsAlgTemplate>
// [中文导读] [AllReduce逐行 S142] 定义 CalcRes 入口：构造具体 InsAlgTemplate 并计算资源请求，函数本身不创建实际 Thread/Channel。
HcclResult InsV2AllReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRes(
    // [中文导读] [AllReduce逐行 S143] 续接 CalcRes 的入口参数/基类初始化：HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,；引用参数按声明的 const 限制读写。
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    // [中文导读] [AllReduce逐行 S144] 续接 CalcRes 的入口参数/基类初始化：const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)；引用参数按声明的 const 限制读写。
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
// [中文导读] [AllReduce逐行 S145] 进入 CalcRes 的实现作用域；构造具体 InsAlgTemplate 并计算资源请求，函数本身不创建实际 Thread/Channel。
{
    // 构建template
    // [中文导读] [AllReduce逐行 S147] 建立本阶段局部对象 std::shared_ptr<InsAlgTemplate> algTemplate，供 CalcRes 下方参数组装和子调用使用。
    std::shared_ptr<InsAlgTemplate> algTemplate
        // [中文导读] [AllReduce逐行 S148] 设置  为 std::make_shared<InsAlgTemplate>(param, 本地用户 Rank, 匹配出的算法通信域层列表[0])；该值供下方当前分支使用。
        = std::make_shared<InsAlgTemplate>(param, topoInfo->userRank, algHierarchyInfo.infos[0]);
    // 调用计算资源的函数
    // [中文导读] [AllReduce逐行 S150] 调用 CalcRes 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(algTemplate->CalcRes(comm, param, topoInfo, resourceRequest));
    // [中文导读] [AllReduce逐行 S151] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S152] 结束 CalcRes 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S154] 声明模板类型参数，将拓扑匹配器与具体算法模板在编译期绑定。
template <typename AlgTopoMatch, typename InsAlgTemplate>
// [中文导读] [AllReduce逐行 S155] 定义 Orchestrate 入口：恢复算法资源，检查 count×类型字节大小是否溢出，进入按 CCL 容量分块的模板执行。
HcclResult InsV2AllReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    // [中文导读] [AllReduce逐行 S156] 续接 Orchestrate 的入口参数/基类初始化：const OpParam& param, const AlgResourceCtxSerializable& resCtx)；引用参数按声明的 const 限制读写。
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
// [中文导读] [AllReduce逐行 S157] 进入 Orchestrate 的实现作用域；恢复算法资源，检查 count×类型字节大小是否溢出，进入按 CCL 容量分块的模板执行。
{
    // [中文导读] [AllReduce逐行 S158] 开始 HCCL_INFO 诊断输出，记录 Orchestrate 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_INFO("[InsV2AllReduceSoleExecutor][Orchestrate] Orchestrate Start.");
    // maxTmpMemSize_设定为cclIn的大小，op中将申请的HcclBuff全给了cclIn
    // [中文导读] [AllReduce逐行 S160] 取得实际 CCL 内存容量，作为 AICPU_TS 分轮传输容量依据。
    maxTmpMemSize_ = resCtx.cclMem.size;
    // 给channels_和threads_赋值
    // [中文导读] [AllReduce逐行 S162] 保存统一参数中的对称内存标志，供模板分轮及直接读归约判断。
    supportSymmetricMemory_ = param.supportSymmetricMemory;
    // [中文导读] [AllReduce逐行 S163] 保留已申请 Thread 列表，后续模板使用主 Thread 与各 Peer 从 Thread。
    threads_ = resCtx.threads;
    // [中文导读] [AllReduce逐行 S164] 分支条件为 supportSymmetricMemory_；成立进入本块，未成立继续后续分支。
    if (supportSymmetricMemory_) {
        // [中文导读] [AllReduce逐行 S165] 保存对称内存偏移参数供直接访问远端窗口。
        inputOffset_ = param.inputOffset;
        // [中文导读] [AllReduce逐行 S166] 保存输出在对称内存窗口中的偏移。
        outputOffset_ = param.outputOffset;
        // [中文导读] [AllReduce逐行 S167] 保存用户输入的对称内存窗口描述。
        inputSymWindow_ = param.inputSymWindow;
        // [中文导读] [AllReduce逐行 S168] 保存用户输出的对称内存窗口描述。
        outputSymWindow_ = param.outputSymWindow;
    // [中文导读] [AllReduce逐行 S169] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S170] AIV 和 CCU 不需要在这里恢复普通 AICPU 通道映射。
    if (param.engine != CommEngine::COMM_ENGINE_AIV && param.engine != CommEngine::COMM_ENGINE_CCU) {
        // [中文导读] [AllReduce逐行 S171] 恢复按远端 Rank 分组的 ChannelInfo，模板据此定位每个 Peer 的首个通道。
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    // [中文导读] [AllReduce逐行 S172] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S173] 从统一描述读取总输入元素数量。
    dataCount_ = param.DataDes.count;
    // [中文导读] [AllReduce逐行 S174] 读取输入元素类型以查询每元素字节数。
    dataType_ = param.DataDes.dataType;
    // [中文导读] [AllReduce逐行 S175] 从类型表得到元素字节数，供溢出校验与轮次偏移计算。
    dataTypeSize_ = DATATYPE_SIZE_TABLE[param.DataDes.dataType];
    // [中文导读] [AllReduce逐行 S176] 准备校验元素数与元素字节数相乘是否会超过 u64 可表示范围。
    if (dataCount_ > UINT64_MAX / dataTypeSize_) {
        // [中文导读] [AllReduce逐行 S177] 开始 HCCL_ERROR 诊断输出，记录 Orchestrate 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_ERROR(
            // [中文导读] [AllReduce逐行 S178] 续接 Orchestrate 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "[InsV2AllReduceSoleExecutor][Orchestrate] dataCount[%llu] * dataTypeSize_[%llu] is greater than "
            // [中文导读] [AllReduce逐行 S179] 续接 Orchestrate 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "UINT64_MAX",
            // [中文导读] [AllReduce逐行 S180] 为 Orchestrate 的诊断/错误宏提供实参：总元素数, 单元素字节数，与前面的格式占位依次对应。
            dataCount_, dataTypeSize_);
        // [中文导读] [AllReduce逐行 S181] 终止当前函数并向上返回 内部错误；调用者 CHK_RET 决定是否继续向上传播。
        return HCCL_E_INTERNAL;
    // [中文导读] [AllReduce逐行 S182] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S183] 验证后计算全部输入的字节数，后续轮次按元素数量推进。
    dataSize_ = dataCount_ * dataTypeSize_;
    // [中文导读] [AllReduce逐行 S184] 传恢复的资源上下文，按 CCL 容量组织模板执行轮次。
    HcclResult ret = OrchestrateLoop(param, resCtx);
    // [中文导读] [AllReduce逐行 S185] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S186] 续接本次错误检查/子调用实参：ret 不等于 成功状态；返回行为由所在完整宏决定。
        ret != HCCL_SUCCESS,
        // [中文导读] [AllReduce逐行 S187] 开始 HCCL_ERROR 诊断输出，记录 Orchestrate 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_ERROR(
            // [中文导读] [AllReduce逐行 S188] 续接 Orchestrate 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "[InsV2AllReduceSoleExecutor][Orchestrate]errNo[0x%016llx] AllReduce executor kernel run failed",
            // [中文导读] [AllReduce逐行 S189] 为 Orchestrate 的诊断/错误宏提供实参：HCCL_ERROR_CODE(ret，与前面的格式占位依次对应。
            HCCL_ERROR_CODE(ret)),
        // [中文导读] [AllReduce逐行 S190] 为 Orchestrate 的诊断/错误宏提供实参：ret，与前面的格式占位依次对应。
        ret);
    // [中文导读] [AllReduce逐行 S191] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S192] 结束 Orchestrate 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S194] 声明模板类型参数，将拓扑匹配器与具体算法模板在编译期绑定。
template <typename AlgTopoMatch, typename InsAlgTemplate>
// [中文导读] [AllReduce逐行 S195] 定义 OrchestrateLoop 入口：计算对齐后的块容量与轮数，把每轮元素数/字节偏移交给模板；对称内存单轮覆盖全量。
HcclResult InsV2AllReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::OrchestrateLoop(
    // [中文导读] [AllReduce逐行 S196] 续接 OrchestrateLoop 的入口参数/基类初始化：const OpParam& param, const AlgResourceCtxSerializable& resCtx)；引用参数按声明的 const 限制读写。
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
// [中文导读] [AllReduce逐行 S197] 进入 OrchestrateLoop 的实现作用域；计算对齐后的块容量与轮数，把每轮元素数/字节偏移交给模板；对称内存单轮覆盖全量。
{
    // [中文导读] [AllReduce逐行 S198] 开始 HCCL_INFO 诊断输出，记录 OrchestrateLoop 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_INFO("[InsV2AllReduceSoleExecutor][OrchestrateLoop] Start");
    // 准备资源
    // [中文导读] [AllReduce逐行 S200] 建立模板资源包，随后填入已申请的引擎资源、通道和Thread。
    TemplateResource templateAlgRes;
    // [中文导读] [AllReduce逐行 S201] 仅CCU执行引擎需要给模板提供已申请的kernel列表。
    if (param.engine == COMM_ENGINE_CCU) {
        // [中文导读] [AllReduce逐行 S202] 将资源上下文中的CCU kernel列表交给模板。
        templateAlgRes.ccuKernels = resCtx.ccuKernels;
    // [中文导读] [AllReduce逐行 S203] 非 AIV 且恢复的通道映射非空时，提供第零层 Peer 通道。
    }
    // [中文导读] [AllReduce逐行 S204] 非AIV且恢复的通道映射非空时，提供第零层Peer通道。
    if (param.engine != CommEngine::COMM_ENGINE_AIV && remoteRankToChannelInfo_.size() > 0) {
        // [中文导读] [AllReduce逐行 S205] 将第零层远端Rank通道映射交给具体算法模板。
        templateAlgRes.channels = remoteRankToChannelInfo_[0];
    // [中文导读] [AllReduce逐行 S206] 将已申请 Thread 列表交给模板，Thread0为主，其余用于Peer通信。
    }
    // [中文导读] [AllReduce逐行 S207] 将已申请Thread列表交给模板，Thread0为主，其余用于Peer通信。
    templateAlgRes.threads = resCtx.threads;
    // [中文导读] [AllReduce逐行 S208] 保留AIV通信信息指针供AIV模板使用。
    templateAlgRes.aivCommInfoPtr = resCtx.aivCommInfoPtr;
    // [中文导读] [AllReduce逐行 S209] 传递双Die分流比例供支持该功能的模板使用。
    templateAlgRes.dieSplitRatio = resCtx.dieSplitRatio;
    // 准备数据
    // [中文导读] [AllReduce逐行 S211] 准备模板每轮的数据描述对象。
    TemplateDataParams tempAlgParams;
    // [中文导读] [AllReduce逐行 S212] 记录用户输入原始基址，每轮通过字节偏移定位其数据区段。
    tempAlgParams.buffInfo.inputPtr = param.inputPtr;
    // [中文导读] [AllReduce逐行 S213] 记录用户输出原始基址，每轮结果写对应字节区段。
    tempAlgParams.buffInfo.outputPtr = param.outputPtr;
    // [中文导读] [AllReduce逐行 S214] 将已申请的 CCL 内存对象作为模板临时缓冲区。
    tempAlgParams.buffInfo.hcclBuff = resCtx.cclMem;
    // [中文导读] [AllReduce逐行 S215] 标记输入属于用户 INPUT 缓冲区。
    tempAlgParams.buffInfo.inBuffType = BufferType::INPUT;
    // [中文导读] [AllReduce逐行 S216] 标记输出属于用户 OUTPUT 缓冲区。
    tempAlgParams.buffInfo.outBuffType = BufferType::OUTPUT;
    // [中文导读] [AllReduce逐行 S217] 标记 scratch 属于 HCCL_BUFFER 缓冲区。
    tempAlgParams.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    // [中文导读] [AllReduce逐行 S218] 保留用户输入总字节容量。
    tempAlgParams.buffInfo.inputSize = param.inputSize;
    // [中文导读] [AllReduce逐行 S219] 保留用户输出总字节容量。
    tempAlgParams.buffInfo.outputSize = param.outputSize;
    // [中文导读] [AllReduce逐行 S220] 仅 OFFLOAD 打开这一远端内存访问参数，OPBASE为false。
    tempAlgParams.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;
    // 不需要重复；repeat用于处理rank存在多块不连续数据块的情况（all-reduce不涉及）
    // [中文导读] [AllReduce逐行 S222] AllReduce 处理一个连续输入块，不使用离散块重复。
    tempAlgParams.repeatNum = 1;
    // [中文导读] [AllReduce逐行 S223] 单连续输入块不需要重复输入跨度。
    tempAlgParams.inputRepeatStride = 0;
    // [中文导读] [AllReduce逐行 S224] 单连续输出块不需要重复输出跨度。
    tempAlgParams.outputRepeatStride = 0;

    // 构建template
    // [中文导读] [AllReduce逐行 S227] 建立由注册绑定确定的具体算法模板对象。
    std::shared_ptr<InsAlgTemplate> algTemplate
        // [中文导读] [AllReduce逐行 S228] 构造模板时传算子参数、本地用户 Rank 和匹配的第零层通信域。
        = std::make_shared<InsAlgTemplate>(param, resCtx.topoInfo.userRank, resCtx.algHierarchyInfo.infos[0]);
    // [中文导读] [AllReduce逐行 S229] AICPU_TS 的 Pod 或 NHRMultiLink 需要额外设置各 Peer 通道数量。
    if (param.engine == CommEngine::COMM_ENGINE_AICPU_TS
        // [中文导读] [AllReduce逐行 S230] 只有 Pod 或明确的多链路 NHR 注册名进入该通道数量配置。
        && (resCtx.topoInfo.isPod || std::string(param.algName) == "AicpuAllReduceSoleNHRMultiLink")) {
        // [中文导读] [AllReduce逐行 S231] 按模板已有 ChannelInfo 设置每个 Peer 的通道数量，错误立即上送。
        CHK_RET(algTemplate->SetchannelsPerRank(templateAlgRes.channels));
    // [中文导读] [AllReduce逐行 S232] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S233] 向具体模板查询处理完整输入需要的临时内存倍数。
    u32 templateScratchMultiplier
        // [中文导读] [AllReduce逐行 S234] 按输入/输出缓冲区类型查询倍数；OneShot固定返回Rank数。
        = algTemplate->CalcScratchMultiple(tempAlgParams.buffInfo.inBuffType, tempAlgParams.buffInfo.outBuffType);

    // 计算最小传输大小
    // [中文导读] [AllReduce逐行 S237] 先初始化每轮字节容量上限，随后按引擎和 scratch 约束计算。
    u64 maxDataSizePerLoop = 0;
    // [中文导读] [AllReduce逐行 S238] 读取本轮复用的实际 CCL 总字节容量。
    maxTmpMemSize_ = tempAlgParams.buffInfo.hcclBuff.size;
    // [中文导读] [AllReduce逐行 S239] AICPU_TS 每轮传输上限取实际 CCL 容量；其它引擎使用 UB_MAX_DATA_SIZE。
    u64 transportBoundDataSize = (param.engine == CommEngine::COMM_ENGINE_AICPU_TS) ? maxTmpMemSize_ : UB_MAX_DATA_SIZE;
    // [中文导读] [AllReduce逐行 S240] 开始 HCCL_INFO 诊断输出，记录 OrchestrateLoop 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_INFO("[InsV2AllReduceSoleExecutor]maxTmpMemSize_ [%u]", maxTmpMemSize_);
    // [中文导读] [AllReduce逐行 S241] 非零临时内存倍数时，按多个完整输入槽划分可用容量。
    if (templateScratchMultiplier != 0) {
        // [中文导读] [AllReduce逐行 S242] 准备按容量、模板倍数和最小片对齐计算每轮 scratch 字节上限。
        u64 scratchBoundDataSize
            // [中文导读] [AllReduce逐行 S243] 先用 CCL 容量除以模板 scratch 倍数，再向下对齐到 HCCL_MIN_SLICE_ALIGN。
            = maxTmpMemSize_ / templateScratchMultiplier / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN;
        // [中文导读] [AllReduce逐行 S244] 每轮字节上限取传输上限和 scratch 上限的较小值，避免每 Rank 槽越界。
        maxDataSizePerLoop = std::min(transportBoundDataSize, scratchBoundDataSize);
    // [中文导读] [AllReduce逐行 S245] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
    } else {
        // [中文导读] [AllReduce逐行 S246] 无需额外 scratch 的模板只受引擎传输容量上限限制。
        maxDataSizePerLoop = transportBoundDataSize;
    // [中文导读] [AllReduce逐行 S247] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // 单次循环处理的数据量大小
    // [中文导读] [AllReduce逐行 S249] 元素字节数为零时打印错误并准备返回内部错误。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S250] 无效类型大小会终止分轮计算。
        dataTypeSize_ == 0, HCCL_ERROR("[InsV2AllReduceSoleExecutor][OrchestrateOpbase] dataTypeSize_ is 0"),
        // [中文导读] [AllReduce逐行 S251] 向条件返回宏提供 内部错误；上方检查成立才退出当前函数。
        HCCL_E_INTERNAL);
    // [中文导读] [AllReduce逐行 S252] 字节容量除以单元素字节数得到每轮最多能处理的元素数量。
    u64 maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize_;
    // [中文导读] [AllReduce逐行 S253] 开始 HCCL_INFO 诊断输出，记录 OrchestrateLoop 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S254] 续接 OrchestrateLoop 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
        "[InsV2AllReduceSoleExecutor][OrchestrateOpbase] maxDataCountPerLoop[%llu], maxDataSizePerLoop[%llu], "
        // [中文导读] [AllReduce逐行 S255] 续接 OrchestrateLoop 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
        "transportBoundDataSize[%llu], templateScratchMultiplier[%llu]",
        // [中文导读] [AllReduce逐行 S256] 为 OrchestrateLoop 的诊断/错误宏提供实参：每轮最大元素数, 每轮最大字节数, 引擎传输上限, 每轮临时内存倍数，与前面的格式占位依次对应。
        maxDataCountPerLoop, maxDataSizePerLoop, transportBoundDataSize, templateScratchMultiplier);
    // [中文导读] [AllReduce逐行 S257] 每轮必须至少容纳一个完整元素，否则不能推进输入。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S258] CCL或传输容量算出的元素上限为零时返回内部错误。
        maxDataCountPerLoop == 0,
        // [中文导读] [AllReduce逐行 S259] 开始 HCCL_ERROR 诊断输出，记录 OrchestrateLoop 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_ERROR("[InsV2AllReduceSoleExecutor][OrchestrateOpbase] maxDataCountPerLoop is 0"), HCCL_E_INTERNAL);
    // 计算loopTimes
    // [中文导读] [AllReduce逐行 S261] 执行轮数取总元素数除以每轮上限的商，下一行补有余数时的一轮。
    u64 loopTimes = dataCount_ / maxDataCountPerLoop
                    // [中文导读] [AllReduce逐行 S262] 余数非零时加 1，实现向上取整；是轮数表达式续行，不能在中间插破坏表达式的注释。
                    + static_cast<u64>(dataCount_ % maxDataCountPerLoop != 0); // 计算迭代轮次（ceil取整）
    // count已经处理的数据
    // [中文导读] [AllReduce逐行 S264] 尚未处理任何输入元素，首轮输入/输出字节偏移从0开始。
    u64 processedDataCount = 0;

    // [中文导读] [AllReduce逐行 S266] 对称内存模板直接访问完整用户窗口，不按普通CCL槽容量分多轮。
    if (param.supportSymmetricMemory) {
        // [中文导读] [AllReduce逐行 S267] 对称内存路径强制仅一轮，末轮元素数计算覆盖全部输入。
        loopTimes = 1;
        // [中文导读] [AllReduce逐行 S268] 开始 HCCL_INFO 诊断输出，记录 OrchestrateLoop 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_INFO("[InsV2AllReduceSoleExecutor][OrchestrateLoop] %s: symmetric memory enabled", param.algName);
    // [中文导读] [AllReduce逐行 S269] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // [中文导读] [AllReduce逐行 S271] 依次执行所有输入区段，每轮复用同一套资源与CCL。
    for (u64 loop = 0; loop < loopTimes; loop++) {
        // dataCount_实际总数据量 和 maxDataCountPerLoop 一次搬运数据量之间不一定是整除关系，需要对尾块进行处理
        // [中文导读] [AllReduce逐行 S273] 末轮取未处理剩余元素数，其余轮取每轮最大元素数。
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxDataCountPerLoop;
        // [中文导读] [AllReduce逐行 S274] 把本轮元素数写入模板参数，归约原语按该count处理。
        tempAlgParams.count = currDataCount;
        // [中文导读] [AllReduce逐行 S275] 输入字节偏移等于已处理元素数乘单元素字节数。
        tempAlgParams.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
        // [中文导读] [AllReduce逐行 S276] 输出使用相同字节偏移，每轮结果写回原输入区段对应的输出区段。
        tempAlgParams.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
        // [中文导读] [AllReduce逐行 S277] CCL 起始偏移每轮复用为 0；不同 Peer 用模板内 Rank 槽区分。
        tempAlgParams.buffInfo.hcclBuffBaseOff = 0;

        // [中文导读] [AllReduce逐行 S279] 本轮完整数据片字节数为当前元素数乘元素字节数。
        tempAlgParams.sliceSize = currDataCount * dataTypeSize_;
        // [中文导读] [AllReduce逐行 S280] AllReduce 每轮只有一块完整连续数据，因此尾片大小等于本轮片大小。
        tempAlgParams.tailSize = tempAlgParams.sliceSize;
        // [中文导读] [AllReduce逐行 S281] AllReduce当前轮没有多片输入布局，输入片间跨度置0。
        tempAlgParams.inputSliceStride = 0;
        // [中文导读] [AllReduce逐行 S282] AllReduce当前轮没有多片输出布局，输出片间跨度置0。
        tempAlgParams.outputSliceStride = 0;
        // [中文导读] [AllReduce逐行 S283] 开始 HCCL_INFO 诊断输出，记录 OrchestrateLoop 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_INFO(
            // [中文导读] [AllReduce逐行 S284] 续接 OrchestrateLoop 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "[InsV2AllReduceSoleExecutor] loop [%u] tempAlgParams.inputSliceStride [%u],"
            // [中文导读] [AllReduce逐行 S285] 续接 OrchestrateLoop 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "tempAlgParams.outputSliceStride [%u] tempAlgParams.sliceSize [%u]",
            // [中文导读] [AllReduce逐行 S286] 为 OrchestrateLoop 的诊断/错误宏提供实参：loop, 输入分片跨度, 输出分片跨度, 本轮完整数据片字节数，与前面的格式占位依次对应。
            loop, tempAlgParams.inputSliceStride, tempAlgParams.outputSliceStride, tempAlgParams.sliceSize);
        // [中文导读] [AllReduce逐行 S287] 开始 HCCL_INFO 诊断输出，记录 OrchestrateLoop 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_INFO(
            // [中文导读] [AllReduce逐行 S288] 续接 OrchestrateLoop 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "[InsV2AllReduceSoleExecutor] loop [%u] tempAlgParams.buffInfo.inBuffBaseOff [%u],"
            // [中文导读] [AllReduce逐行 S289] 续接 OrchestrateLoop 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "tempAlgParams.buffInfo.outBuffBaseOff [%u]",
            // [中文导读] [AllReduce逐行 S290] 为 OrchestrateLoop 的诊断/错误宏提供实参：loop, 本轮用户输入字节偏移, 本轮用户输出字节偏移，与前面的格式占位依次对应。
            loop, tempAlgParams.buffInfo.inBuffBaseOff, tempAlgParams.buffInfo.outBuffBaseOff);

        // [中文导读] [AllReduce逐行 S292] 把本轮参数与已申请资源交给具体模板；模板错误经 CHK_RET 立即向调用者传播。
        CHK_RET(algTemplate->KernelRun(param, tempAlgParams, templateAlgRes));
        // [中文导读] [AllReduce逐行 S293] 本轮模板提交成功后推进已处理元素数，下一轮输入输出基址偏移随之增加。
        processedDataCount += currDataCount;
    // [中文导读] [AllReduce逐行 S294] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
// [中文导读] [AllReduce逐行 S295] 以下资源/成本逻辑仅在 Host 编译版本执行；AICPU_COMPILE 构建跳过该段。
#ifndef AICPU_COMPILE
    // [中文导读] [AllReduce逐行 S296] 仅单轮、CCU、非OFFLOAD时保存快速发射上下文；主例AICPU_TS不命中。
    if (loopTimes == 1 && param.engine == CommEngine::COMM_ENGINE_CCU && param.opMode != OpMode::OFFLOAD) {
        // [中文导读] [AllReduce逐行 S297] 保存CCU模板资源及主Thread通知数量，供以后快速发射复用。
        CHK_RET(FastLaunchSaveCtx(param, templateAlgRes, resCtx.notifyNumOnMainThread));
    // [中文导读] [AllReduce逐行 S298] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
// [中文导读] [AllReduce逐行 S299] 结束该编译期能力/Host 边界，后续公共返回逻辑在相应构建中保留。
#endif

    // [中文导读] [AllReduce逐行 S301] 开始 HCCL_INFO 诊断输出，记录 OrchestrateLoop 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_INFO("[InsV2AllReduceSoleExecutor][OrchestrateLoop] End.");
    // [中文导读] [AllReduce逐行 S302] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S303] 结束 OrchestrateLoop 实现；其返回状态或已写回字段由调用者接收。
}

#ifndef AICPU_COMPILE
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::FastLaunchSaveCtx(
    const OpParam& param, const TemplateResource& templateAlgRes, u32 notifyNumOnMainThread) const
{
    HCCL_INFO("[InsV2AllReduceSoleExecutor] loopTimes==1, save fast launch ctx.");
    u32 threadNum = templateAlgRes.threads.size();
    u32 ccuKernelNum = templateAlgRes.submitInfos.size();
    if (ccuKernelNum < 1) {
        HCCL_INFO("[InsV2AllReduceSoleExecutor] ccu kernel num is 0, no need to save.");
        return HCCL_SUCCESS;
    }
    HCCL_INFO(
        "[InsV2AllReduceSoleExecutor][HcclEngineCtxCreate] threadNum[%llu], ccuKernelNum[%llu]", threadNum,
        ccuKernelNum);

    u64 size = CcuFastLaunchCtx::GetCtxSize(threadNum, ccuKernelNum);
    // 申请ctx
    void* ctxPtr = nullptr;
    HCCL_INFO("[InsV2AllReduceSoleExecutor][HcclEngineCtxCreate] Tag[%s], size[%llu]", param.fastLaunchTag, size);
    CHK_RET(HcclEngineCtxCreate(param.hcclComm, param.fastLaunchTag, CommEngine::COMM_ENGINE_CCU, size, &ctxPtr));

    CcuFastLaunchCtx* ccuFastLaunchCtx = reinterpret_cast<CcuFastLaunchCtx*>(ctxPtr);
    // 1 算法名
    CHK_SAFETY_FUNC_RET(strcpy_s(ccuFastLaunchCtx->algName, sizeof(ccuFastLaunchCtx->algName), param.algName));
    HCCL_INFO("[InsV2AllReduceSoleExecutor][FastLaunchSaveCtx] algName[%s]", ccuFastLaunchCtx->algName);

    // 2 thread
    ccuFastLaunchCtx->threadNum = threadNum;
    ccuFastLaunchCtx->notifyNumOnMainThread = notifyNumOnMainThread;
    ThreadHandle* threads = ccuFastLaunchCtx->GetThreadHandlePtr();
    for (u32 i = 0; i < threadNum; i++) {
        threads[i] = templateAlgRes.threads[i];
    }

    // 3 ccu kernel handle, taskArg入参
    ccuFastLaunchCtx->ccuKernelNum[0] = ccuKernelNum;
    CcuKernelSubmitInfo* kernelSubmitInfos = ccuFastLaunchCtx->GetCcuKernelSubmitInfoPtr();
    for (u32 i = 0; i < ccuKernelNum; i++) {
        kernelSubmitInfos[i] = templateAlgRes.submitInfos[i];
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::FastLaunch(
    const OpParam& param, const CcuFastLaunchCtx* fastLaunchCtx)
{
    HCCL_INFO("[InsV2AllReduceSoleExecutor][FastLaunch] Start.");
    TemplateFastLaunchCtx tempFastLaunchCtx;
    // 1 取thread
    ThreadHandle* threads = fastLaunchCtx->GetThreadHandlePtr();
    tempFastLaunchCtx.threads.assign(threads, threads + fastLaunchCtx->threadNum);
    HCCL_INFO("[InsV2AllReduceSoleExecutor][FastLaunch] threadNum[%llu]", fastLaunchCtx->threadNum);

    // 2 取arg
    CcuKernelSubmitInfo* ccuKernelSubmitInfos = fastLaunchCtx->GetCcuKernelSubmitInfoPtr();
    tempFastLaunchCtx.ccuKernelSubmitInfos.assign(
        ccuKernelSubmitInfos, ccuKernelSubmitInfos + fastLaunchCtx->ccuKernelNum[0]);
    HCCL_INFO("[InsV2AllReduceSoleExecutor][FastLaunch] ccuKernelNum[%llu]", fastLaunchCtx->ccuKernelNum[0]);
    tempFastLaunchCtx.buffInfo.inputPtr = param.inputPtr;
    tempFastLaunchCtx.buffInfo.outputPtr = param.outputPtr;
    tempFastLaunchCtx.buffInfo.hcclBuff = param.hcclBuff;

    // 3 调template
    std::unique_ptr<InsAlgTemplate> algTemplate = std::make_unique<InsAlgTemplate>();
    CHK_RET(algTemplate->FastLaunch(param, tempFastLaunchCtx));
    HCCL_INFO("[InsV2AllReduceSoleExecutor][FastLaunch] End.");
    return HCCL_SUCCESS;
}
#endif

// [中文导读] [AllReduce逐行 S377] 开始注册 AllReduce 的 OneShot 执行器实现。
REGISTER_EXEC_V2(
    // [中文导读] [AllReduce逐行 S378] 将 AllReduce 算子和 AicpuAllReduceSoleMeshOneShot 名称绑定到 SoleExecutor，并指定 TopoMatchOneLevel。
    HcclCMDType::HCCL_CMD_ALLREDUCE, AicpuAllReduceSoleMeshOneShot, InsV2AllReduceSoleExecutor, TopoMatchOneLevel,
    // [中文导读] [AllReduce逐行 S379] 该执行器的具体通信模板绑定为 InsTempAllReduceMesh1DOneShot；到此结束执行器注册宏。
    InsTempAllReduceMesh1DOneShot);
// [中文导读] [AllReduce逐行 S380] 开始声明 OneShot 算法属性，供候选过滤与拓扑匹配检查。
REGISTER_ALG_ATTRS(
    // [中文导读] [AllReduce逐行 S381] 注册属性设置最多支持一个算法拓扑层。
    AicpuAllReduceSoleMeshOneShot, topo.maxTopoLevelNum = 1;
    // [中文导读] [AllReduce逐行 S382] 仅声明 Mesh1D 或 Mesh1DClos 的第零层形状，并允许第零层 PCIe 混合。
    topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_MESH_1D_CLOS; topo.isSupportLevel0PcieMix = true;
    // [中文导读] [AllReduce逐行 S383] 要求全层 Mesh 覆盖，并建立额外拓扑检查回调。
    topo.requireAllMeshConnected = true; topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        // [中文导读] [AllReduce逐行 S384] 混合 Mesh1DClos 时进入附加检查；纯 Mesh1D 在下方直接通过。
        if (topo->level0Topo == Level0Shape::MESH_1D_CLOS) {
            // [中文导读] [AllReduce逐行 S385] 混合拓扑不含 PCIe 时要求 Mesh/Clos 规模关系满足下一检查。
            if (!topo->level0PcieMix) {
                // [中文导读] [AllReduce逐行 S386] 初始化 Mesh 与 Clos 规模相等的输出标志为 false。
                bool isEqual = false;
                // [中文导读] [AllReduce逐行 S387] 查询 Mesh 与 Clos Rank 数是否相等；此注册回调未检查查询返回码。
                AutoSelectorBase::CheckMeshNumEqualToClosNum(topo, isEqual);
                // [中文导读] [AllReduce逐行 S388] 仅 Mesh 与 Clos Rank 数相等且用户 Rank 数不超过4时允许该非 PCIe 混合拓扑。
                return isEqual && topo->userRankSize <= 4;
            // [中文导读] [AllReduce逐行 S389] 结束非 PCIe 混合拓扑约束分支。
            }
            // [中文导读] [AllReduce逐行 S390] 含 PCIe 的已允许 Mesh1DClos 直接通过附加回调，仍需外层其它属性满足。
            return true;
        // [中文导读] [AllReduce逐行 S391] 结束 Mesh1DClos 的自定义拓扑检查分支。
        }
        // [中文导读] [AllReduce逐行 S392] 其它已被外层形状属性允许的拓扑通过此附加检查。
        return true;
    // [中文导读] [AllReduce逐行 S393] 结束自定义属性回调和算法属性注册宏；不是每次算子的网络执行步骤。
    });
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_ALLREDUCE, AicpuAllReduceSoleMeshTwoShot, InsV2AllReduceSoleExecutor, TopoMatchOneLevel,
    InsTempAllReduceMesh1DTwoShot);
REGISTER_ALG_ATTRS(
    AicpuAllReduceSoleMeshTwoShot, topo.maxTopoLevelNum = 1;
    topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_MESH_1D_CLOS; topo.isSupportLevel0PcieMix = true;
    topo.requireAllMeshConnected = true; topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        if (topo->level0Topo == Level0Shape::MESH_1D_CLOS) {
            if (!topo->level0PcieMix) {
                bool isEqual = false;
                AutoSelectorBase::CheckMeshNumEqualToClosNum(topo, isEqual);
                return isEqual && topo->userRankSize <= 4;
            }
            return true;
        }
        return true;
    });
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_ALLREDUCE, AicpuAllReduceSoleNHR, InsV2AllReduceSoleExecutor, TopoMatchOneLevel,
    InsTempAllReduceNHR);
REGISTER_ALG_ATTRS(AicpuAllReduceSoleNHR,
                   topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_CLOS | LEVEL0_TOPO_MESH_1D_CLOS;
                   topo.isSupportLevel1Nhr = true; op.isSupportProd = false;
                   op.unsupportedDataTypes
                   = {HcclDataType::HCCL_DATA_TYPE_INT64, HcclDataType::HCCL_DATA_TYPE_UINT64,
                      HcclDataType::HCCL_DATA_TYPE_FP64};);
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_ALLREDUCE, AicpuAllReduceSoleNHRMultiLink, InsV2AllReduceSoleExecutor, TopoMatchOneLevel,
    InsTempAllReduceNHR);
REGISTER_ALG_ATTRS(
    AicpuAllReduceSoleNHRMultiLink, topo.maxTopoLevelNum = TOPO_LEVEL_NUM_3; topo.supportLevel0Topos = LEVEL0_TOPO_CLOS;
    topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        return topo->level0Topo == Level0Shape::CLOS;
    };
    op.isSupportProd = false;
    op.unsupportedDataTypes
    = {HcclDataType::HCCL_DATA_TYPE_INT64, HcclDataType::HCCL_DATA_TYPE_UINT64, HcclDataType::HCCL_DATA_TYPE_FP64});
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_ALLREDUCE, AicpuAllReduceSoleMeshChunkTwoShot, InsV2AllReduceSoleExecutor, TopoMatchOneLevel,
    InsTempAllReduceMesh1DTwoShotMeshChunk);
REGISTER_ALG_ATTRS(
    AicpuAllReduceSoleMeshChunkTwoShot, topo.maxTopoLevelNum = 1;
    topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_MESH_1D_CLOS; topo.isSupportLevel0PcieMix = true;
    topo.requireAllMeshConnected = true; op.isSupportProd = false;
    op.unsupportedDataTypes
    = {HcclDataType::HCCL_DATA_TYPE_INT64, HcclDataType::HCCL_DATA_TYPE_UINT64, HcclDataType::HCCL_DATA_TYPE_FP64};
    topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        if (topo->level0Topo == Level0Shape::MESH_1D_CLOS) {
            return topo->level0PcieMix;
        }
        return true;
    });
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_ALLREDUCE, AicpuAllReduceSoleNHRAicpuReduce, InsV2AllReduceSoleExecutor, TopoMatchOneLevel,
    InsTempAllReduceAicpuReduceNHR);
REGISTER_ALG_ATTRS(AicpuAllReduceSoleNHRAicpuReduce,
                   topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_CLOS | LEVEL0_TOPO_MESH_1D_CLOS;
                   topo.isSupportLevel0PcieMix = true; topo.isSupportLevel1Nhr = true);

#ifndef AICPU_COMPILE
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_ALLREDUCE, AivAllReduceSoleMeshOneShot, InsV2AllReduceSoleExecutor, TopoMatchOneLevel,
    AivTempAllReduceMesh1DOneShot);
REGISTER_ALG_ATTRS(
    AivAllReduceSoleMeshOneShot, topo.maxTopoLevelNum = TOPO_LEVEL_NUM_2; topo.isSupportLevel0PcieMix = true;
    topo.maxSupportRankSize = MAX_RANK_SIZE; topo.isSupportLevel1Nhr = true;

    op.isSupportProd = false; op.unsupportedDataTypes = UNSUPPORTED_UINT64_FP64;
    op.opCustomCheck = [](const OpParam& opParam, const TopoInfoWithNetLayerDetails*) -> bool {
        void* bufAddr = nullptr;
        uint64_t bufSize = 0;
        if (HcclGetHcclBuffer(opParam.hcclComm, &bufAddr, &bufSize) != HCCL_SUCCESS) {
            return false;
        }
        u64 dataSize = opParam.DataDes.count * DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
        return dataSize <= bufSize * AIV_MAX_CCL_LOOP_NUM;
    });
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_ALLREDUCE, AivAllReduceSoleMeshTwoShot, InsV2AllReduceSoleExecutor, TopoMatchOneLevel,
    AivTempAllReduceMesh1DTwoShot);
REGISTER_ALG_ATTRS(
    AivAllReduceSoleMeshTwoShot, topo.maxTopoLevelNum = TOPO_LEVEL_NUM_2; topo.isSupportLevel1Nhr = true;
    topo.maxSupportRankSize = MAX_RANK_SIZE;
    topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_CLOS | LEVEL0_TOPO_MESH_1D_CLOS;
    topo.isSupportLevel0PcieMix = true; topo.isSupportLevel1Nhr = true;

    op.isSupportProd = false; op.unsupportedDataTypes = UNSUPPORTED_UINT64_FP64;
    op.opCustomCheck = [](const OpParam& opParam, const TopoInfoWithNetLayerDetails*) -> bool {
        void* bufAddr = nullptr;
        uint64_t bufSize = 0;
        if (HcclGetHcclBuffer(opParam.hcclComm, &bufAddr, &bufSize) != HCCL_SUCCESS) {
            return false;
        }
        u64 dataSize = opParam.DataDes.count * DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
        return dataSize <= bufSize * AIV_MAX_CCL_LOOP_NUM;
    });
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_ALLREDUCE, CcuSchedAllReduceSoleNHR, InsV2AllReduceSoleExecutor, TopoMatchOneLevel,
    CcuTempAllReduceNHRMem2Mem1D);
REGISTER_ALG_ATTRS(CcuSchedAllReduceSoleNHR, topo.maxTopoLevelNum = TOPO_LEVEL_NUM_2;
                   topo.maxSupportRankSize = CCU_SCHED_MAX_RANK_SIZE;
                   topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_CLOS; topo.isSupportLevel1Nhr = true;
                   topo.isSupport2DieFullMesh = true; op.isSupportProd = false;
                   op.unsupportedDataTypes
                   = {HcclDataType::HCCL_DATA_TYPE_INT64, HcclDataType::HCCL_DATA_TYPE_UINT64,
                      HcclDataType::HCCL_DATA_TYPE_FP64};
                   op.isSupportInplace = false);
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)

#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_ALLREDUCE, CcuSchedAllReduceSoleMesh, InsV2AllReduceSoleExecutor, TopoMatchOneLevel,
    CcuTempAllReduceMeshMem2Mem1D);
REGISTER_ALG_ATTRS(
    CcuSchedAllReduceSoleMesh, topo.maxTopoLevelNum = TOPO_LEVEL_NUM_2;
    topo.maxSupportRankSize = CCU_SCHED_MAX_RANK_SIZE;
    topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_MESH_1D_CLOS; topo.isSupportLevel0PcieMix = true;
    topo.requireAllMeshConnected = true; op.isSupportProd = false;
    op.unsupportedDataTypes
    = {HcclDataType::HCCL_DATA_TYPE_INT8, HcclDataType::HCCL_DATA_TYPE_INT64, HcclDataType::HCCL_DATA_TYPE_UINT64,
       HcclDataType::HCCL_DATA_TYPE_FP64};
    op.isSupportInplace = false; topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        if (topo->level0Topo == Level0Shape::MESH_1D_CLOS) {
            if (topo->level0PcieMix) {
                return AutoSelectorBase::IsLayerAllConnetedWithTopo(topo, 0, CommTopo::COMM_TOPO_1DMESH);
            }
            bool isEqual = false;
            AutoSelectorBase::CheckMeshNumEqualToClosNum(topo, isEqual);
            return isEqual && topo->userRankSize <= 4;
        }
        return true;
    });
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_ALLREDUCE, CcuMSAllReduceSoleMesh, InsV2AllReduceSoleExecutor, TopoMatchOneLevel,
    CcuTempAllReduceMesh1D);
REGISTER_ALG_ATTRS(CcuMSAllReduceSoleMesh, topo.maxTopoLevelNum = 1;
                   topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_MESH_1D_CLOS;
                   topo.isSupportLevel0PcieMix = true; op.isSupportProd = false;
                   op.unsupportedDataTypes
                   = {HcclDataType::HCCL_DATA_TYPE_INT8, HcclDataType::HCCL_DATA_TYPE_INT64,
                      HcclDataType::HCCL_DATA_TYPE_UINT64, HcclDataType::HCCL_DATA_TYPE_FP64};
                   op.isSupportInplace = false);
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_ALLREDUCE, CcuMSAllReduceSoleMesh2Die, InsV2AllReduceSoleExecutor, TopoMatchOneLevel,
    CcuTempAllreduceMesh1D2DieOneShot);
REGISTER_ALG_ATTRS(CcuMSAllReduceSoleMesh2Die, topo.maxTopoLevelNum = 1; topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D;
                   topo.supportLevel0MeshTypes = MESH_TYPE_TWO_DIE_REGULAR; topo.isSupport2DieFullMesh = true;
                   op.isSupportProd = false; op.unsupportedDataTypes
                                             = {HcclDataType::HCCL_DATA_TYPE_INT8, HcclDataType::HCCL_DATA_TYPE_INT64,
                                                HcclDataType::HCCL_DATA_TYPE_UINT64, HcclDataType::HCCL_DATA_TYPE_FP64};
                   op.isSupportInplace = false);
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_ALLREDUCE, CcuMSAllReduceSoleMeshOneShot, InsV2AllReduceSoleExecutor, TopoMatchOneLevel,
    CcuTempAllReduceMesh1DOneShot);
REGISTER_ALG_ATTRS(
    CcuMSAllReduceSoleMeshOneShot, topo.maxTopoLevelNum = 1;
    topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_MESH_1D_CLOS; topo.isSupportLevel0PcieMix = true;
    topo.requireAllMeshConnected = true; op.isSupportProd = false;
    op.unsupportedDataTypes
    = {HcclDataType::HCCL_DATA_TYPE_INT8, HcclDataType::HCCL_DATA_TYPE_INT64, HcclDataType::HCCL_DATA_TYPE_UINT64,
       HcclDataType::HCCL_DATA_TYPE_FP64};
    op.isSupportInplace = false; topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        if (topo->level0Topo == Level0Shape::MESH_1D_CLOS) {
            bool isEqual = false;
            AutoSelectorBase::CheckMeshNumEqualToClosNum(topo, isEqual);
            return isEqual && topo->userRankSize <= 4;
        }
        return true;
    });
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_ALLREDUCE, CcuSchedAllReduceSoleMesh2Die, InsV2AllReduceSoleExecutor, TopoMatchOneLevel,
    CcuTempAllReduceMesh1DMem2Mem2DieOneShot);
REGISTER_ALG_ATTRS(CcuSchedAllReduceSoleMesh2Die, topo.maxSupportRankSize = CCU_SCHED_MAX_RANK_SIZE;
                   topo.maxTopoLevelNum = 1; topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_MESH_1D_CLOS;
                   topo.supportLevel0MeshTypes = MESH_TYPE_TWO_DIE_REGULAR; topo.isSupportLevel0PcieMix = true;
                   topo.requireAllMeshConnected = true; topo.isSupport2DieFullMesh = true; op.isSupportProd = false;
                   op.unsupportedDataTypes
                   = {HcclDataType::HCCL_DATA_TYPE_INT8, HcclDataType::HCCL_DATA_TYPE_INT64,
                      HcclDataType::HCCL_DATA_TYPE_UINT64, HcclDataType::HCCL_DATA_TYPE_FP64};
                   op.isSupportInplace = false);
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_ALLREDUCE, CcuSchedAllReduceSoleNHRMultiLink, InsV2AllReduceSoleExecutor, TopoMatchOneLevel,
    CcuTempAllReduceNhrMem2Mem1DMultiJetty);
REGISTER_ALG_ATTRS(
    CcuSchedAllReduceSoleNHRMultiLink, topo.maxSupportRankSize = CCU_SCHED_MAX_RANK_SIZE; topo.maxTopoLevelNum = 1;
    topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D_CLOS; op.isSupportProd = false;
    op.unsupportedDataTypes
    = {HcclDataType::HCCL_DATA_TYPE_INT8, HcclDataType::HCCL_DATA_TYPE_INT64, HcclDataType::HCCL_DATA_TYPE_UINT64,
       HcclDataType::HCCL_DATA_TYPE_FP64};
    op.isSupportInplace = false; topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        bool isEqual = false;
        AutoSelectorBase::CheckMeshNumEqualToClosNum(topo, isEqual);
        return !(isEqual && topo->userRankSize <= MAX_RANK_NUM_FOR_CONCURRENT_ALGO);
    };);

#endif /* CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0) */
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_ALLREDUCE, CcuMSAllReduceSoleMeshConcur, InsV2AllReduceSoleExecutor, TopoMatchConcurrentV2,
    CcuTempAllReduceConcurrentMeshNHR);
REGISTER_ALG_ATTRS(CcuMSAllReduceSoleMeshConcur, topo.maxTopoLevelNum = 1;
                   topo.supportDevTypes = {HcclDevType::DEV_TYPE_960}; op.isSupportProd = false;
                   op.unsupportedDataTypes = UNSUPPORTED_INT8_AND_64BIT; op.isSupportInplace = false);
#endif /* CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0) */
#endif
} // namespace ops_hccl
