/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_reduce_auto_selector.h"
#include "selector_registry.h"
#include "hccl_aiv_utils.h"
#include "ins_v2_all_reduce_order_preserved_executor.h"
#include "order_preserved_common.h"

namespace ops_hccl {
constexpr u64 RS_MAX_DATA_SIZE = 16 * 1024 * 1024;
constexpr u64 AR_ONESHOT_1D_MAX_DATA_SIZE = 16 * 1024;
constexpr u64 AR_M2M_1D_MAX_DATA_SIZE = 8 * 1024 * 1024;
constexpr u64 AR_AICPU_1D_SMALL_DATA_SIZE = 8 * 1024 * 1024;
constexpr u64 AR_AICPU_1D_MAX_DATA_SIZE = 32 * 1024 * 1024;
constexpr u64 AR_AICPU_1D_CROSS_SMALL_DATA_SIZE = 32 * 1024 * 1024;
constexpr u64 AR_AICPU_1D_64DATATYPE_DATA_SIZE = 8 * 1024 * 1024;
constexpr u32 MAX_RANK_NUM_FOR_CONCURRENT_ALGO = 4;
constexpr u32 MAX_RANK_NUM_FOR_REDUCE_MS_ALGO = 8;
constexpr u64 AR_FLATTEN_MAX_DATA_SIZE = 512 * 1024;
constexpr u64 AR_CCU_CLOS_1D_SMALL_DATA_SIZE = 8 * 1024 * 1024;
constexpr u64 AR_AICPU_SEQUENCE_DATA_SIZE = 4ULL * 1024 * 1024 * 1024;
constexpr u64 OMNI_PCIE_AR_DATA_SIZE = 32 * 1024 * 1024;
constexpr u64 OMNI_UBX_AR_SCHED_DATA_SIZE = 64 * 1024 * 1024;
constexpr u64 OMNI_UBX_AR_MS_DATA_SIZE = 32 * 1024 * 1024;
constexpr u64 AR_AIV_SMALL_DATA_SIZE_IN_BOARD = 128 * 1024;
constexpr u64 AR_AIV_BOARD_SIZE = 8;
constexpr u32 DEVICE_NUM_PER_MODULE_8 = 8;
constexpr u64 AR_2P_DETOUR_DATA_SIZE = 8 * 1024 * 1024;
constexpr u32 AR_MORE_64P_SEQ_MAX_DATA_SIZE = 32 * 1024 * 1024;
constexpr u32 AR_CCU_MAX_RANK_SIZE = 64;
constexpr u64 AR_CCU_SEQ_MAX_DATA_SIZE = 64 * 1024 * 1024;
// [中文导读] [AllReduce逐行 S40] 定义 SelectCcuMsAlgo 入口：旧自动选择器的 CCU_MS 能力门槛：拒绝 strict、多拓扑层、INT8、PROD 和 64 位类型，匹配 Mesh 候选。
SelectorStatus AllReduceAutoSelector::SelectCcuMsAlgo(
    // [中文导读] [AllReduce逐行 S41] 续接 SelectCcuMsAlgo 的入口参数/基类初始化：const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,；引用参数按声明的 const 限制读写。
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
    // [中文导读] [AllReduce逐行 S42] 续接 SelectCcuMsAlgo 的入口参数/基类初始化：const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& 算法名输出参数) const；引用参数按声明的 const 限制读写。
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName) const
// [中文导读] [AllReduce逐行 S43] 进入 SelectCcuMsAlgo 的实现作用域；旧自动选择器的 CCU_MS 能力门槛：拒绝 strict、多拓扑层、INT8、PROD 和 64 位类型，匹配 Mesh 候选。
{
    // [中文导读] [AllReduce逐行 S44] 显式标记 configAlgMap 在此兼容/default 分支未使用，避免编译器未使用参数警告。
    (void)configAlgMap;
    // [中文导读] [AllReduce逐行 S45] 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuMsAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_DEBUG("[AllReduceAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);

    // 保序模式不支持CCU_MS，需要回退到AICPU
    // [中文导读] [AllReduce逐行 S48] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S49] 调用 IsNeedStrictModeForOrderPreserved 完成当前参数所指的子步骤；本行实参为 IsNeedStrictModeForOrderPreserved(opParam, 通信域 Rank 总数),。
        IsNeedStrictModeForOrderPreserved(opParam, topoInfo->userRankSize),
        // [中文导读] [AllReduce逐行 S50] 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuMsAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_DEBUG("[AllReduceAutoSelector] DETERMINISTIC_STRICT mode not supported for CCU_MS, fallback to AICPU."),
        // [中文导读] [AllReduce逐行 S51] 为 SelectCcuMsAlgo 的诊断/错误宏提供实参：当前选择器不匹配，与前面的格式占位依次对应。
        SelectorStatus::NOT_MATCH);

    // [中文导读] [AllReduce逐行 S53] 分支条件为 topoInfo->topoLevelNums 大于 1；成立进入本块，未成立继续后续分支。
    if (topoInfo->topoLevelNums > 1) {
        // [中文导读] [AllReduce逐行 S54] 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuMsAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_DEBUG("[AllReduceAutoSelector] levelNum > 1 is not supported yet for ccu_ms mode.");
        // [中文导读] [AllReduce逐行 S55] 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。
        return SelectorStatus::NOT_MATCH;
    // [中文导读] [AllReduce逐行 S56] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // MS 模式不支持 int8
    // [中文导读] [AllReduce逐行 S59] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S60] 续接本次错误检查/子调用实参：输入元素类型 等于 HcclDataType::INT8 类型；返回行为由所在完整宏决定。
        opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT8,
        // [中文导读] [AllReduce逐行 S61] 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuMsAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_DEBUG(
            // [中文导读] [AllReduce逐行 S62] 续接 SelectCcuMsAlgo 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "[AllReduceAutoSelector] dataType[%d] is not supported yet for ccu_ms mode.", opParam.DataDes.dataType),
        // [中文导读] [AllReduce逐行 S63] 为 SelectCcuMsAlgo 的诊断/错误宏提供实参：当前选择器不匹配，与前面的格式占位依次对应。
        SelectorStatus::NOT_MATCH);

    // MS 模式不支持 PROD
    // [中文导读] [AllReduce逐行 S66] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S67] 续接本次错误检查/子调用实参：归约运算 等于 HcclReduceOp::乘积归约；返回行为由所在完整宏决定。
        opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD,
        // [中文导读] [AllReduce逐行 S68] 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuMsAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_DEBUG("[AllReduceAutoSelector] ReduceOp[%d] is not supported yet for ccu_ms mode.", opParam.reduceType),
        // [中文导读] [AllReduce逐行 S69] 为 SelectCcuMsAlgo 的诊断/错误宏提供实参：当前选择器不匹配，与前面的格式占位依次对应。
        SelectorStatus::NOT_MATCH);

    // [中文导读] [AllReduce逐行 S71] 分支条件为 Is64BitDataType(输入元素类型；成立进入本块，未成立继续后续分支。
    if (Is64BitDataType(opParam.DataDes.dataType)) {
        // [中文导读] [AllReduce逐行 S72] 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuMsAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_DEBUG("[AllReduceAutoSelector] ccu_ms mode not support INT64, UINT64, FP64.");
        // [中文导读] [AllReduce逐行 S73] 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。
        return SelectorStatus::NOT_MATCH;
    // [中文导读] [AllReduce逐行 S74] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // [中文导读] [AllReduce逐行 S76] 直接返回 调用 SelectMeshAlgo 完成当前参数所指的子步骤 的结果，调用者取得本分支结果。
    return SelectMeshAlgo(topoInfo, opParam, selectAlgName);
// [中文导读] [AllReduce逐行 S77] 结束 SelectCcuMsAlgo 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S79] 定义 SelectMeshUBXAlgo 入口：CCU_MS 下根据 Mesh/CLOS 实例关系、Rank 数和数据阈值选择 UBX 并发、流水线或单 Mesh。
SelectorStatus AllReduceAutoSelector::SelectMeshUBXAlgo(
    // [中文导读] [AllReduce逐行 S80] 续接 SelectMeshUBXAlgo 的入口参数/基类初始化：const TopoInfoWithNetLayerDetails* topoInfo, std::string& 算法名输出参数, u64 本 Rank 数据字节数) const；引用参数按声明的 const 限制读写。
    const TopoInfoWithNetLayerDetails* topoInfo, std::string& selectAlgName, u64 dataSize) const
// [中文导读] [AllReduce逐行 S81] 进入 SelectMeshUBXAlgo 的实现作用域；CCU_MS 下根据 Mesh/CLOS 实例关系、Rank 数和数据阈值选择 UBX 并发、流水线或单 Mesh。
{
    // UBX机型
    // [中文导读] [AllReduce逐行 S83] 设置 isMeshNumEqualToClosNum 为 false；该值供下方当前分支使用。
    bool isMeshNumEqualToClosNum = false;
    // [中文导读] [AllReduce逐行 S84] 设置 isClosNumMultipleOfMeshNum 为 false；该值供下方当前分支使用。
    bool isClosNumMultipleOfMeshNum = false;
    // [中文导读] [AllReduce逐行 S85] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S86] 调用 CheckMeshNumEqualToClosNum 完成当前参数所指的子步骤；本行实参为 CheckMeshNumEqualToClosNum(topoInfo, isMeshNumEqualToClosNum) 不等于 成功状态,。
        CheckMeshNumEqualToClosNum(topoInfo, isMeshNumEqualToClosNum) != HCCL_SUCCESS,
        // [中文导读] [AllReduce逐行 S87] 开始 HCCL_DEBUG 诊断输出，记录 SelectMeshUBXAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_DEBUG("[AllReduceAutoSelector] CheckMeshNumEqualToClosNum failed."), SelectorStatus::NOT_MATCH);
    // [中文导读] [AllReduce逐行 S88] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S89] 调用 CheckClosNumMultipleOfMeshNum 完成当前参数所指的子步骤；本行实参为 CheckClosNumMultipleOfMeshNum(topoInfo, isClosNumMultipleOfMeshNum) 不等于 成功状态,。
        CheckClosNumMultipleOfMeshNum(topoInfo, isClosNumMultipleOfMeshNum) != HCCL_SUCCESS,
        // [中文导读] [AllReduce逐行 S90] 开始 HCCL_DEBUG 诊断输出，记录 SelectMeshUBXAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_DEBUG("[AllReduceAutoSelector] CheckClosNumMultipleOfMeshNum failed."), SelectorStatus::NOT_MATCH);
    // [中文导读] [AllReduce逐行 S91] 分支条件为 isMeshNumEqualToClosNum 且 通信域 Rank 总数 不超过 MAX_RANK_NUM_FOR_CONCURRENT_ALGO；成立进入本块，未成立继续后续分支。
    if (isMeshNumEqualToClosNum && topoInfo->userRankSize <= MAX_RANK_NUM_FOR_CONCURRENT_ALGO) {
        // 4P mesh
        // [中文导读] [AllReduce逐行 S93] 分支条件为 IsSmallData(本 Rank 数据字节数；成立进入本块，未成立继续后续分支。
        if (IsSmallData(dataSize)) {
            // 小数据量，用1d mesh算法
            // [中文导读] [AllReduce逐行 S95] 写回候选算法名：算法名输出参数 = "CcuMSAllReduceSoleMeshOneShot"；此处只选择注册名，执行资源在后续流程申请。
            selectAlgName = "CcuMSAllReduceSoleMeshOneShot";
        // [中文导读] [AllReduce逐行 S96] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
        } else {
            // 大数据量，用mesh+clos并行算法
            // [中文导读] [AllReduce逐行 S98] 写回候选算法名：算法名输出参数 = "CcuMSAllReduceConcurMeshNHRMultiLink"；此处只选择注册名，执行资源在后续流程申请。
            selectAlgName = "CcuMSAllReduceConcurMeshNHRMultiLink";
        // [中文导读] [AllReduce逐行 S99] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
    // [中文导读] [AllReduce逐行 S100] 分支条件为 isClosNumMultipleOfMeshNum 且 !IsSmallData(本 Rank 数据字节数；成立进入本块，未成立继续后续分支。
    } else if (isClosNumMultipleOfMeshNum && !IsSmallData(dataSize)) {
        // [中文导读] [AllReduce逐行 S101] 分支条件为 本 Rank 数据字节数 小于 OMNI_UBX_AR_MS_DATA_SIZE；成立进入本块，未成立继续后续分支。
        if (dataSize < OMNI_UBX_AR_MS_DATA_SIZE) {
            // [中文导读] [AllReduce逐行 S102] 开始 HCCL_DEBUG 诊断输出，记录 SelectMeshUBXAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。
            HCCL_DEBUG("[AllReduceAutoSelector][%s] MESH_1D_CLOS not match.", __func__);
            // [中文导读] [AllReduce逐行 S103] 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。
            return SelectorStatus::NOT_MATCH;
        // [中文导读] [AllReduce逐行 S104] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
        } else {
            // [中文导读] [AllReduce逐行 S105] 写回候选算法名：算法名输出参数 = "CcuMSAllReducePipeLineMeshNHR"；此处只选择注册名，执行资源在后续流程申请。
            selectAlgName = "CcuMSAllReducePipeLineMeshNHR";
        // [中文导读] [AllReduce逐行 S106] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
    // [中文导读] [AllReduce逐行 S107] 分支条件为 通信域 Rank 总数 不超过 MAX_RANK_NUM_FOR_REDUCE_MS_ALGO；成立进入本块，未成立继续后续分支。
    } else if (topoInfo->userRankSize <= MAX_RANK_NUM_FOR_REDUCE_MS_ALGO) {
        // 跨4p回退
        // [中文导读] [AllReduce逐行 S109] 写回候选算法名：算法名输出参数 = "CcuMSAllReduceSoleMesh"；此处只选择注册名，执行资源在后续流程申请。
        selectAlgName = "CcuMSAllReduceSoleMesh";
    // [中文导读] [AllReduce逐行 S110] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
    } else {
        // [中文导读] [AllReduce逐行 S111] 开始 HCCL_DEBUG 诊断输出，记录 SelectMeshUBXAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_DEBUG("[AllReduceAutoSelector] level0Topo[%u] is not supported mesh yet.", topoInfo->level0Topo);
        // [中文导读] [AllReduce逐行 S112] 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。
        return SelectorStatus::NOT_MATCH;
    // [中文导读] [AllReduce逐行 S113] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // [中文导读] [AllReduce逐行 S115] 开始 HCCL_DEBUG 诊断输出，记录 SelectMeshUBXAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_DEBUG("[AllReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    // [中文导读] [AllReduce逐行 S116] 返回 MATCH，已写入候选算法名，旧选择器结束本次优先级尝试。
    return SelectorStatus::MATCH;
// [中文导读] [AllReduce逐行 S117] 结束 SelectMeshUBXAlgo 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S119] 定义 SelectMeshAlgo 入口：CCU_MS 下处理 Mesh1D/UBX；拒绝原地重叠、非规则双 Die 和需要 AICPU 的大数据 2P 绕路。
SelectorStatus AllReduceAutoSelector::SelectMeshAlgo(
    // [中文导读] [AllReduce逐行 S120] 续接 SelectMeshAlgo 的入口参数/基类初始化：const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam, std::string& 算法名输出参数) const；引用参数按声明的 const 限制读写。
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam, std::string& selectAlgName) const
// [中文导读] [AllReduce逐行 S121] 进入 SelectMeshAlgo 的实现作用域；CCU_MS 下处理 Mesh1D/UBX；拒绝原地重叠、非规则双 Die 和需要 AICPU 的大数据 2P 绕路。
{
    // [中文导读] [AllReduce逐行 S122] 设置 单元素字节数 为 类型到元素字节数的查找表[输入元素类型]；该值供下方当前分支使用。
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    // [中文导读] [AllReduce逐行 S123] 设置 本 Rank 数据字节数 为 本 Rank 输入元素数 * 单元素字节数；该值供下方当前分支使用。
    u64 dataSize = opParam.DataDes.count * perDataSize;
    // 2P场景且数据量大于阈值时回退到AICPU
    // [中文导读] [AllReduce逐行 S125] 分支条件为 IsTwoLevelNetLayer(topoInfo, opParam) 且 通信域 Rank 总数 等于 2 且 本 Rank 数据字节数 至少 8MiB 双 Rank 绕路门槛；成立进入本块，未成立继续后续分支。
    if (IsTwoLevelNetLayer(topoInfo, opParam) && topoInfo->userRankSize == 2 && dataSize >= AR_2P_DETOUR_DATA_SIZE) {
        // [中文导读] [AllReduce逐行 S126] 开始 HCCL_DEBUG 诊断输出，记录 SelectMeshAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_DEBUG(
            // [中文导读] [AllReduce逐行 S127] 续接 SelectMeshAlgo 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "[AllReduceAutoSelector] 2P scenario with data size[%llu], "
            // [中文导读] [AllReduce逐行 S128] 续接 SelectMeshAlgo 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "fallback to AICPU for better performance.",
            // [中文导读] [AllReduce逐行 S129] 为 SelectMeshAlgo 的诊断/错误宏提供实参：本 Rank 数据字节数，与前面的格式占位依次对应。
            dataSize);
        // [中文导读] [AllReduce逐行 S130] 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。
        return SelectorStatus::NOT_MATCH;
    // [中文导读] [AllReduce逐行 S131] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S132] 分支条件为 第零层拓扑形状 等于 单层 Mesh1D；成立进入本块，未成立继续后续分支。
    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        // [中文导读] [AllReduce逐行 S133] 分支条件为 IsInputOutputOverlap(opParam) 等于 true；成立进入本块，未成立继续后续分支。
        if (IsInputOutputOverlap(opParam) == true) { // 不支持 inplace 场景
            // [中文导读] [AllReduce逐行 S134] 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。
            return SelectorStatus::NOT_MATCH;
        // [中文导读] [AllReduce逐行 S135] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
        // [中文导读] [AllReduce逐行 S136] 分支条件为 topoInfo->level0MeshType 等于 Level0MeshType::TWO_DIE_REGULAR；成立进入本块，未成立继续后续分支。
        if (topoInfo->level0MeshType == Level0MeshType::TWO_DIE_REGULAR) {
            // [中文导读] [AllReduce逐行 S137] 分支条件为 IsSmallData(本 Rank 数据字节数；成立进入本块，未成立继续后续分支。
            if (IsSmallData(dataSize)) {
                // [中文导读] [AllReduce逐行 S138] 写回候选算法名：算法名输出参数 = "CcuMSAllReduceSoleMesh2Die"；此处只选择注册名，执行资源在后续流程申请。
                selectAlgName = "CcuMSAllReduceSoleMesh2Die";
            // [中文导读] [AllReduce逐行 S139] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
            } else {
                // [中文导读] [AllReduce逐行 S140] 写回候选算法名：算法名输出参数 = "CcuMSAllReduceSequenceMesh2Die"；此处只选择注册名，执行资源在后续流程申请。
                selectAlgName = "CcuMSAllReduceSequenceMesh2Die";
            // [中文导读] [AllReduce逐行 S141] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
            }
        // [中文导读] [AllReduce逐行 S142] 分支条件为 topoInfo->level0MeshType 等于 Level0MeshType::TWO_DIE_NOT_REGULAR；成立进入本块，未成立继续后续分支。
        } else if (topoInfo->level0MeshType == Level0MeshType::TWO_DIE_NOT_REGULAR) {
            // [中文导读] [AllReduce逐行 S143] 开始 HCCL_DEBUG 诊断输出，记录 SelectMeshAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。
            HCCL_DEBUG("[AllReduceAutoSelector][%s] TWO_DIE_NOT_REGULAR not match", __func__);
            // [中文导读] [AllReduce逐行 S144] 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。
            return SelectorStatus::NOT_MATCH;
        // [中文导读] [AllReduce逐行 S145] 分支条件为 IsSmallData(本 Rank 数据字节数；成立进入本块，未成立继续后续分支。
        } else if (IsSmallData(dataSize)) {
            // [中文导读] [AllReduce逐行 S146] 写回候选算法名：算法名输出参数 = "CcuMSAllReduceSoleMeshOneShot"；此处只选择注册名，执行资源在后续流程申请。
            selectAlgName = "CcuMSAllReduceSoleMeshOneShot";
        // [中文导读] [AllReduce逐行 S147] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
        } else {
            // [中文导读] [AllReduce逐行 S148] 分支条件为 IsDevType960() 且 本 Rank 数据字节数 大于 SMALL_COUNT_16M 且 IsTwoLevelNetLayer(topoInfo, opParam；成立进入本块，未成立继续后续分支。
            if (IsDevType960() && dataSize > SMALL_COUNT_16M && IsTwoLevelNetLayer(topoInfo, opParam)) {
                // [中文导读] [AllReduce逐行 S149] 写回候选算法名：算法名输出参数 = "CcuMSAllReduceSoleMeshConcur"；此处只选择注册名，执行资源在后续流程申请。
                selectAlgName = "CcuMSAllReduceSoleMeshConcur";
            // [中文导读] [AllReduce逐行 S150] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
            } else {
                // [中文导读] [AllReduce逐行 S151] 写回候选算法名：算法名输出参数 = "CcuMSAllReduceSoleMesh"；此处只选择注册名，执行资源在后续流程申请。
                selectAlgName = "CcuMSAllReduceSoleMesh";
            // [中文导读] [AllReduce逐行 S152] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
            }
        // [中文导读] [AllReduce逐行 S153] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
    // [中文导读] [AllReduce逐行 S154] 分支条件为 第零层拓扑形状 等于 Mesh 与 CLOS 混合拓扑；成立进入本块，未成立继续后续分支。
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        // [中文导读] [AllReduce逐行 S155] 分支条件为 IsInputOutputOverlap(opParam) 等于 true；成立进入本块，未成立继续后续分支。
        if (IsInputOutputOverlap(opParam) == true) {
            // 不支持 inplace 场景
            // [中文导读] [AllReduce逐行 S157] 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。
            return SelectorStatus::NOT_MATCH;
        // [中文导读] [AllReduce逐行 S158] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
        // [中文导读] [AllReduce逐行 S159] 直接返回 调用 SelectMeshUBXAlgo 完成当前参数所指的子步骤 的结果，调用者取得本分支结果。
        return SelectMeshUBXAlgo(topoInfo, selectAlgName, dataSize);
    // [中文导读] [AllReduce逐行 S160] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
    } else {
        // [中文导读] [AllReduce逐行 S161] 开始 HCCL_DEBUG 诊断输出，记录 SelectMeshAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_DEBUG("[AllReduceAutoSelector] level0Topo[%u] is not supported yet.", topoInfo->level0Topo);
        // [中文导读] [AllReduce逐行 S162] 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。
        return SelectorStatus::NOT_MATCH;
    // [中文导读] [AllReduce逐行 S163] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S164] 开始 HCCL_DEBUG 诊断输出，记录 SelectMeshAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_DEBUG("[AllReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    // [中文导读] [AllReduce逐行 S165] 返回 MATCH，已写入候选算法名，旧选择器结束本次优先级尝试。
    return SelectorStatus::MATCH;
// [中文导读] [AllReduce逐行 S166] 结束 SelectMeshAlgo 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S168] 定义 SelectCcuScheduleAlgo 入口：旧 CCU 调度算法能力检查：拒绝 UB_RTP、三层、strict、PROD、64 位类型，再处理多层或单层拓扑。
SelectorStatus AllReduceAutoSelector::SelectCcuScheduleAlgo(
    // [中文导读] [AllReduce逐行 S169] 续接 SelectCcuScheduleAlgo 的入口参数/基类初始化：const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,；引用参数按声明的 const 限制读写。
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
    // [中文导读] [AllReduce逐行 S170] 续接 SelectCcuScheduleAlgo 的入口参数/基类初始化：const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& 算法名输出参数) const；引用参数按声明的 const 限制读写。
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName) const
// [中文导读] [AllReduce逐行 S171] 进入 SelectCcuScheduleAlgo 的实现作用域；旧 CCU 调度算法能力检查：拒绝 UB_RTP、三层、strict、PROD、64 位类型，再处理多层或单层拓扑。
{
    // [中文导读] [AllReduce逐行 S172] 显式标记 configAlgMap 在此兼容/default 分支未使用，避免编译器未使用参数警告。
    (void)configAlgMap;
    // [中文导读] [AllReduce逐行 S173] 设置 ccuSize 为 AR_CCU_MAX_RANK_SIZE；该值供下方当前分支使用。
    u32 ccuSize = AR_CCU_MAX_RANK_SIZE;
    // [中文导读] [AllReduce逐行 S174] 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuScheduleAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_DEBUG("[AllReduceAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);

    // [中文导读] [AllReduce逐行 S176] 分支条件为 topoInfo->level2UbRtp；成立进入本块，未成立继续后续分支。
    if (topoInfo->level2UbRtp) {
        // [中文导读] [AllReduce逐行 S177] 开始 HCCL_INFO 诊断输出，记录 SelectCcuScheduleAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_INFO(
            // [中文导读] [AllReduce逐行 S178] 续接 SelectCcuScheduleAlgo 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "[AllReduceAutoSelector][%s] ccu schedule is not supported with level2UbRtp, reset to default.", __func__);
        // [中文导读] [AllReduce逐行 S179] 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。
        return SelectorStatus::NOT_MATCH;
    // [中文导读] [AllReduce逐行 S180] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // [中文导读] [AllReduce逐行 S182] 分支条件为 topoInfo->topoLevelNums 至少 TOPO_LEVEL_NUM_3；成立进入本块，未成立继续后续分支。
    if (topoInfo->topoLevelNums >= TOPO_LEVEL_NUM_3) {
        // [中文导读] [AllReduce逐行 S183] 开始 HCCL_INFO 诊断输出，记录 SelectCcuScheduleAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_INFO(
            // [中文导读] [AllReduce逐行 S184] 续接 SelectCcuScheduleAlgo 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "[AllReduceAutoSelector][%s] ccu schedule is not supported when topoLevelNums >= 3(levelNum[%u]), reset to "
            // [中文导读] [AllReduce逐行 S185] 续接 SelectCcuScheduleAlgo 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "default.",
            // [中文导读] [AllReduce逐行 S186] 为 SelectCcuScheduleAlgo 的诊断/错误宏提供实参：__func__, topoInfo->topoLevelNums，与前面的格式占位依次对应。
            __func__, topoInfo->topoLevelNums);
        // [中文导读] [AllReduce逐行 S187] 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。
        return SelectorStatus::NOT_MATCH;
    // [中文导读] [AllReduce逐行 S188] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // 保序模式不支持CCU_SCHED，需要回退到AICPU
    // [中文导读] [AllReduce逐行 S191] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S192] 调用 IsNeedStrictModeForOrderPreserved 完成当前参数所指的子步骤；本行实参为 IsNeedStrictModeForOrderPreserved(opParam, 通信域 Rank 总数),。
        IsNeedStrictModeForOrderPreserved(opParam, topoInfo->userRankSize),
        // [中文导读] [AllReduce逐行 S193] 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuScheduleAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_DEBUG("[AllReduceAutoSelector] DETERMINISTIC_STRICT mode not supported for CCU_SCHED, fallback to AICPU."),
        // [中文导读] [AllReduce逐行 S194] 为 SelectCcuScheduleAlgo 的诊断/错误宏提供实参：当前选择器不匹配，与前面的格式占位依次对应。
        SelectorStatus::NOT_MATCH);

    // ccu 模式不支持 PROD
    // [中文导读] [AllReduce逐行 S197] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S198] 续接本次错误检查/子调用实参：归约运算 等于 HcclReduceOp::乘积归约；返回行为由所在完整宏决定。
        opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD,
        // [中文导读] [AllReduce逐行 S199] 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuScheduleAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_DEBUG(
            // [中文导读] [AllReduce逐行 S200] 续接 SelectCcuScheduleAlgo 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "[AllReduceAutoSelector] ReduceOp[%d] is not supported yet for ccu schedule mode.", opParam.reduceType),
        // [中文导读] [AllReduce逐行 S201] 为 SelectCcuScheduleAlgo 的诊断/错误宏提供实参：当前选择器不匹配，与前面的格式占位依次对应。
        SelectorStatus::NOT_MATCH);

    // [中文导读] [AllReduce逐行 S203] 分支条件为 Is64BitDataType(输入元素类型；成立进入本块，未成立继续后续分支。
    if (Is64BitDataType(opParam.DataDes.dataType)) {
        // [中文导读] [AllReduce逐行 S204] 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuScheduleAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_DEBUG("[AllReduceAutoSelector] ccu_schedule mode not support INT64, UINT64, FP64.");
        // [中文导读] [AllReduce逐行 S205] 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。
        return SelectorStatus::NOT_MATCH;
    // [中文导读] [AllReduce逐行 S206] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S207] 设置 单元素字节数 为 类型到元素字节数的查找表[输入元素类型]；该值供下方当前分支使用。
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    // [中文导读] [AllReduce逐行 S208] 设置 本 Rank 数据字节数 为 本 Rank 输入元素数 * 单元素字节数；该值供下方当前分支使用。
    u64 dataSize = opParam.DataDes.count * perDataSize;

    // [中文导读] [AllReduce逐行 S210] 分支条件为 topoInfo->topoLevelNums 大于 1；成立进入本块，未成立继续后续分支。
    if (topoInfo->topoLevelNums > 1) {
        // [中文导读] [AllReduce逐行 S211] 分支条件为 第零层拓扑形状 等于 单层 Mesh1D；成立进入本块，未成立继续后续分支。
        if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            // [中文导读] [AllReduce逐行 S212] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
            CHK_PRT_RET(
                // [中文导读] [AllReduce逐行 S213] 调用 IsInputOutputOverlap 完成当前参数所指的子步骤；本行实参为 IsInputOutputOverlap(opParam) 等于 true,。
                IsInputOutputOverlap(opParam) == true,
                // [中文导读] [AllReduce逐行 S214] 开始 HCCL_WARNING 诊断输出，记录 SelectCcuScheduleAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。
                HCCL_WARNING("[Algo][AllReduceAutoSelector] ccu_sched does not support inplace allreduce."),
                // [中文导读] [AllReduce逐行 S215] 为 SelectCcuScheduleAlgo 的诊断/错误宏提供实参：当前选择器不匹配，与前面的格式占位依次对应。
                SelectorStatus::NOT_MATCH);
            // Level1Nhr 已在 CalcTopoShape 中设置（GCD==1 时为 true）
            // [中文导读] [AllReduce逐行 S217] 分支条件为 topoInfo->Level1Nhr；成立进入本块，未成立继续后续分支。
            if (topoInfo->Level1Nhr) {
                // [中文导读] [AllReduce逐行 S218] 写回候选算法名：算法名输出参数 = "CcuSchedAllReduceSoleNHR"；此处只选择注册名，执行资源在后续流程申请。
                selectAlgName = "CcuSchedAllReduceSoleNHR";
                // [中文导读] [AllReduce逐行 S219] 开始 HCCL_INFO 诊断输出，记录 SelectCcuScheduleAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。
                HCCL_INFO("[AllReduceAutoSelector] Level1Nhr=true, select [%s]", selectAlgName.c_str());
                // [中文导读] [AllReduce逐行 S220] 返回 MATCH，已写入候选算法名，旧选择器结束本次优先级尝试。
                return SelectorStatus::MATCH;
            // [中文导读] [AllReduce逐行 S221] 分支条件为 topoInfo->当前网络层编号Details.localNetInsSizeOfLayer[0] 等于 1；成立进入本块，未成立继续后续分支。
            } else if (topoInfo->netLayerDetails.localNetInsSizeOfLayer[0] == 1) {
                // [中文导读] [AllReduce逐行 S222] 写回候选算法名：算法名输出参数 = "CcuSchedAllReduceSoleNHR"；此处只选择注册名，执行资源在后续流程申请。
                selectAlgName = "CcuSchedAllReduceSoleNHR";
            // [中文导读] [AllReduce逐行 S223] 分支条件为 topoInfo->is2DieFullMesh；成立进入本块，未成立继续后续分支。
            } else if (topoInfo->is2DieFullMesh) {
                // [中文导读] [AllReduce逐行 S224] 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuScheduleAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。
                HCCL_DEBUG("[AllReduceAutoSelector] 2DieFullMesh is not supported yet for ccu schedule mode.");
                // [中文导读] [AllReduce逐行 S225] 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。
                return SelectorStatus::NOT_MATCH;
            // [中文导读] [AllReduce逐行 S226] 分支条件为 本 Rank 数据字节数 不超过 AR_MORE_64P_SEQ_MAX_DATA_SIZE 且 通信域 Rank 总数 大于 ccuSize；成立进入本块，未成立继续后续分支。
            } else if (dataSize <= AR_MORE_64P_SEQ_MAX_DATA_SIZE && topoInfo->userRankSize > ccuSize) {
                // [中文导读] [AllReduce逐行 S227] 写回候选算法名：算法名输出参数 = "CcuSchedAllReduceSequenceMeshMesh"；此处只选择注册名，执行资源在后续流程申请。
                selectAlgName = "CcuSchedAllReduceSequenceMeshMesh";
            // [中文导读] [AllReduce逐行 S228] 分支条件为 ；成立进入本块，未成立继续后续分支。
            } else if (
                // [中文导读] [AllReduce逐行 S229] 续接 SelectCcuScheduleAlgo 当前语句的具体实参/字段：本 Rank 数据字节数 不超过 RS_MAX_DATA_SIZE 且 通信域 Rank 总数 至少 ccuSize；由其完整表达式完成参数组装、检查或结果写回。
                dataSize <= RS_MAX_DATA_SIZE && topoInfo->userRankSize >= ccuSize
                // [中文导读] [AllReduce逐行 S230] 调用 Is8BitDataType 完成当前参数所指的子步骤；本行实参为 且 !Is8BitDataType(输入元素类型)) {。
                && !Is8BitDataType(opParam.DataDes.dataType)) {
                // [中文导读] [AllReduce逐行 S231] 写回候选算法名：算法名输出参数 = "CcuSchedAllReduceSequenceMeshMesh"；此处只选择注册名，执行资源在后续流程申请。
                selectAlgName = "CcuSchedAllReduceSequenceMeshMesh";
                // [中文导读] [AllReduce逐行 S232] 返回 MATCH，已写入候选算法名，旧选择器结束本次优先级尝试。
                return SelectorStatus::MATCH;
            // [中文导读] [AllReduce逐行 S233] 分支条件为 ；成立进入本块，未成立继续后续分支。
            } else if (
                // [中文导读] [AllReduce逐行 S234] 续接 SelectCcuScheduleAlgo 当前语句的具体实参/字段：本 Rank 数据字节数 不超过 AR_FLATTEN_MAX_DATA_SIZE 且 通信域 Rank 总数 不超过 ccuSize；由其完整表达式完成参数组装、检查或结果写回。
                dataSize <= AR_FLATTEN_MAX_DATA_SIZE && topoInfo->userRankSize <= ccuSize
                // [中文导读] [AllReduce逐行 S235] 调用 IsInputOutputOverlap 完成当前参数所指的子步骤；本行实参为 且 (!IsInputOutputOverlap(opParam)) 且 !Is8BitDataType(输入元素类型)) {。
                && (!IsInputOutputOverlap(opParam)) && !Is8BitDataType(opParam.DataDes.dataType)) {
                // [中文导读] [AllReduce逐行 S236] 写回候选算法名：算法名输出参数 = "CcuSchedAllReduceSoleMesh"；此处只选择注册名，执行资源在后续流程申请。
                selectAlgName = "CcuSchedAllReduceSoleMesh";
                // [中文导读] [AllReduce逐行 S237] 返回 MATCH，已写入候选算法名，旧选择器结束本次优先级尝试。
                return SelectorStatus::MATCH;
            // [中文导读] [AllReduce逐行 S238] 分支条件为 ；成立进入本块，未成立继续后续分支。
            } else if (
                // [中文导读] [AllReduce逐行 S239] 续接 SelectCcuScheduleAlgo 当前语句的具体实参/字段：本 Rank 数据字节数 不超过 AR_CCU_SEQ_MAX_DATA_SIZE 且 通信域 Rank 总数 小于 ccuSize；由其完整表达式完成参数组装、检查或结果写回。
                dataSize <= AR_CCU_SEQ_MAX_DATA_SIZE && topoInfo->userRankSize < ccuSize
                // [中文导读] [AllReduce逐行 S240] 调用 Is8BitDataType 完成当前参数所指的子步骤；本行实参为 且 !Is8BitDataType(输入元素类型)) {。
                && !Is8BitDataType(opParam.DataDes.dataType)) {
                // [中文导读] [AllReduce逐行 S241] 写回候选算法名：算法名输出参数 = "CcuSchedAllReduceSequenceMeshMesh"；此处只选择注册名，执行资源在后续流程申请。
                selectAlgName = "CcuSchedAllReduceSequenceMeshMesh";
                // [中文导读] [AllReduce逐行 S242] 返回 MATCH，已写入候选算法名，旧选择器结束本次优先级尝试。
                return SelectorStatus::MATCH;
            // [中文导读] [AllReduce逐行 S243] 分支条件为 IsSmallDataCCU(本 Rank 数据字节数, 通信域 Rank 总数；成立进入本块，未成立继续后续分支。
            } else if (IsSmallDataCCU(dataSize, topoInfo->userRankSize)) { // 64M以下跑ccu
                // 性能优化改用MS做reduce后不支持int8
                // [中文导读] [AllReduce逐行 S245] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
                CHK_PRT_RET(
                    // [中文导读] [AllReduce逐行 S246] 续接本次错误检查/子调用实参：输入元素类型 等于 HcclDataType::INT8 类型；返回行为由所在完整宏决定。
                    opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT8,
                    // [中文导读] [AllReduce逐行 S247] 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuScheduleAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。
                    HCCL_DEBUG(
                        // [中文导读] [AllReduce逐行 S248] 续接 SelectCcuScheduleAlgo 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
                        "[AllReduceAutoSelector] dataType[%d] is not supported yet for ccu schedule mode with ms "
                        // [中文导读] [AllReduce逐行 S249] 续接 SelectCcuScheduleAlgo 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
                        "reduce. levelNum[%u]",
                        // [中文导读] [AllReduce逐行 S250] 为 SelectCcuScheduleAlgo 的诊断/错误宏提供实参：输入元素类型, topoInfo->topoLevelNums，与前面的格式占位依次对应。
                        opParam.DataDes.dataType, topoInfo->topoLevelNums),
                    // [中文导读] [AllReduce逐行 S251] 为 SelectCcuScheduleAlgo 的诊断/错误宏提供实参：当前选择器不匹配，与前面的格式占位依次对应。
                    SelectorStatus::NOT_MATCH);
                // [中文导读] [AllReduce逐行 S252] 写回候选算法名：算法名输出参数 = "CcuSchedAllReduceParallelMeshNHR"；此处只选择注册名，执行资源在后续流程申请。
                selectAlgName = "CcuSchedAllReduceParallelMeshNHR";
                // [中文导读] [AllReduce逐行 S253] 返回 MATCH，已写入候选算法名，旧选择器结束本次优先级尝试。
                return SelectorStatus::MATCH;
            // [中文导读] [AllReduce逐行 S254] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
            } else {
                // [中文导读] [AllReduce逐行 S255] 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。
                return SelectorStatus::NOT_MATCH; // 64M以上切为aicpu
            // [中文导读] [AllReduce逐行 S256] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
            }
        // [中文导读] [AllReduce逐行 S257] 分支条件为 第零层拓扑形状 等于 CLOS 拓扑 且 (!IsInputOutputOverlap(opParam；成立进入本块，未成立继续后续分支。
        } else if (topoInfo->level0Topo == Level0Shape::CLOS && (!IsInputOutputOverlap(opParam))) {
            // [中文导读] [AllReduce逐行 S258] 分支条件为 本 Rank 数据字节数 小于 AR_CCU_CLOS_1D_SMALL_DATA_SIZE；成立进入本块，未成立继续后续分支。
            if (dataSize < AR_CCU_CLOS_1D_SMALL_DATA_SIZE) {
                // [中文导读] [AllReduce逐行 S259] 写回候选算法名：算法名输出参数 = "CcuSchedAllReduceSoleNHR"；此处只选择注册名，执行资源在后续流程申请。
                selectAlgName = "CcuSchedAllReduceSoleNHR";
                // [中文导读] [AllReduce逐行 S260] 返回 MATCH，已写入候选算法名，旧选择器结束本次优先级尝试。
                return SelectorStatus::MATCH;
            // [中文导读] [AllReduce逐行 S261] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
            } else {
                // [中文导读] [AllReduce逐行 S262] 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。
                return SelectorStatus::NOT_MATCH;
            // [中文导读] [AllReduce逐行 S263] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
            }
        // [中文导读] [AllReduce逐行 S264] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
        } else {
            // [中文导读] [AllReduce逐行 S265] 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuScheduleAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。
            HCCL_DEBUG(
                // [中文导读] [AllReduce逐行 S266] 续接 SelectCcuScheduleAlgo 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
                "[AllReduceAutoSelector] level0Topo[%d] is not supported yet for ccu schedule mode.",
                // [中文导读] [AllReduce逐行 S267] 为 SelectCcuScheduleAlgo 的诊断/错误宏提供实参：第零层拓扑形状，与前面的格式占位依次对应。
                topoInfo->level0Topo);
            // [中文导读] [AllReduce逐行 S268] 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。
            return SelectorStatus::NOT_MATCH;
        // [中文导读] [AllReduce逐行 S269] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
    // [中文导读] [AllReduce逐行 S270] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
    } else {
        // [中文导读] [AllReduce逐行 S271] 直接返回 调用 SelectCcuScheduleLevel0Algo 完成当前参数所指的子步骤 的结果，调用者取得本分支结果。
        return SelectCcuScheduleLevel0Algo(topoInfo, opParam, selectAlgName, dataSize);
    // [中文导读] [AllReduce逐行 S272] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S273] 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuScheduleAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_DEBUG("[AllReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    // [中文导读] [AllReduce逐行 S274] 返回 MATCH，已写入候选算法名，旧选择器结束本次优先级尝试。
    return SelectorStatus::MATCH;
// [中文导读] [AllReduce逐行 S275] 结束 SelectCcuScheduleAlgo 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S277] 定义 SelectCcuScheduleLevel0UBXAlgo 入口：单层 UBX 中按 Mesh/CLOS 关系与大小选择并发、多 Jetty、流水线或 NHR。
SelectorStatus AllReduceAutoSelector::SelectCcuScheduleLevel0UBXAlgo(
    // [中文导读] [AllReduce逐行 S278] 续接 SelectCcuScheduleLevel0UBXAlgo 的入口参数/基类初始化：const TopoInfoWithNetLayerDetails* topoInfo, std::string& 算法名输出参数, const u64 本 Rank 数据字节数) const；引用参数按声明的 const 限制读写。
    const TopoInfoWithNetLayerDetails* topoInfo, std::string& selectAlgName, const u64 dataSize) const
// [中文导读] [AllReduce逐行 S279] 进入 SelectCcuScheduleLevel0UBXAlgo 的实现作用域；单层 UBX 中按 Mesh/CLOS 关系与大小选择并发、多 Jetty、流水线或 NHR。
{
    // UBX机型
    // [中文导读] [AllReduce逐行 S281] 设置 isMeshNumEqualToClosNum 为 false；该值供下方当前分支使用。
    bool isMeshNumEqualToClosNum = false;
    // [中文导读] [AllReduce逐行 S282] 设置 isClosNumMultipleOfMeshNum 为 false；该值供下方当前分支使用。
    bool isClosNumMultipleOfMeshNum = false;
    // [中文导读] [AllReduce逐行 S283] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S284] 调用 CheckMeshNumEqualToClosNum 完成当前参数所指的子步骤；本行实参为 CheckMeshNumEqualToClosNum(topoInfo, isMeshNumEqualToClosNum) 不等于 成功状态,。
        CheckMeshNumEqualToClosNum(topoInfo, isMeshNumEqualToClosNum) != HCCL_SUCCESS,
        // [中文导读] [AllReduce逐行 S285] 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuScheduleLevel0UBXAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_DEBUG("[AllReduceAutoSelector] CheckMeshNumEqualToClosNum failed."), SelectorStatus::NOT_MATCH);
    // [中文导读] [AllReduce逐行 S286] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S287] 调用 CheckClosNumMultipleOfMeshNum 完成当前参数所指的子步骤；本行实参为 CheckClosNumMultipleOfMeshNum(topoInfo, isClosNumMultipleOfMeshNum) 不等于 成功状态,。
        CheckClosNumMultipleOfMeshNum(topoInfo, isClosNumMultipleOfMeshNum) != HCCL_SUCCESS,
        // [中文导读] [AllReduce逐行 S288] 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuScheduleLevel0UBXAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_DEBUG("[AllReduceAutoSelector] CheckClosNumMultipleOfMeshNum failed."), SelectorStatus::NOT_MATCH);
    // [中文导读] [AllReduce逐行 S289] 分支条件为 isMeshNumEqualToClosNum 且 通信域 Rank 总数 不超过 MAX_RANK_NUM_FOR_CONCURRENT_ALGO；成立进入本块，未成立继续后续分支。
    if (isMeshNumEqualToClosNum && topoInfo->userRankSize <= MAX_RANK_NUM_FOR_CONCURRENT_ALGO) {
        // 4P mesh
        // [中文导读] [AllReduce逐行 S291] 分支条件为 IsSmallData(本 Rank 数据字节数；成立进入本块，未成立继续后续分支。
        if (IsSmallData(dataSize)) {
            // 小数据量，用1d mesh算法
            // [中文导读] [AllReduce逐行 S293] 写回候选算法名：算法名输出参数 = "CcuSchedAllReduceSoleMesh"；此处只选择注册名，执行资源在后续流程申请。
            selectAlgName = "CcuSchedAllReduceSoleMesh";
        // [中文导读] [AllReduce逐行 S294] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
        } else {
            // 大数据量，用mesh+clos并行算法
            // [中文导读] [AllReduce逐行 S296] 写回候选算法名：算法名输出参数 = "CcuSchedAllReduceConcurMeshNHRMultiLink"；此处只选择注册名，执行资源在后续流程申请。
            selectAlgName = "CcuSchedAllReduceConcurMeshNHRMultiLink";
        // [中文导读] [AllReduce逐行 S297] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
    // [中文导读] [AllReduce逐行 S298] 分支条件为 isClosNumMultipleOfMeshNum 且 !IsSmallData(本 Rank 数据字节数；成立进入本块，未成立继续后续分支。
    } else if (isClosNumMultipleOfMeshNum && !IsSmallData(dataSize)) {
        // 矩形场景大数据量，用Parallel并行算法
        // [中文导读] [AllReduce逐行 S300] 分支条件为 本 Rank 数据字节数 小于 OMNI_UBX_AR_SCHED_DATA_SIZE；成立进入本块，未成立继续后续分支。
        if (dataSize < OMNI_UBX_AR_SCHED_DATA_SIZE) {
            // [中文导读] [AllReduce逐行 S301] 写回候选算法名：算法名输出参数 = "CcuSchedAllReduceParallelMeshNHRMultiJetty"；此处只选择注册名，执行资源在后续流程申请。
            selectAlgName = "CcuSchedAllReduceParallelMeshNHRMultiJetty";
        // [中文导读] [AllReduce逐行 S302] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
        } else {
            // [中文导读] [AllReduce逐行 S303] 写回候选算法名：算法名输出参数 = "CcuSchedAllReducePipeLineMeshNHR"；此处只选择注册名，执行资源在后续流程申请。
            selectAlgName = "CcuSchedAllReducePipeLineMeshNHR";
        // [中文导读] [AllReduce逐行 S304] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
    // [中文导读] [AllReduce逐行 S305] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
    } else {
        // 其他场景，用1d NHR算法
        // [中文导读] [AllReduce逐行 S307] 写回候选算法名：算法名输出参数 = "CcuSchedAllReduceSoleNHRMultiLink"；此处只选择注册名，执行资源在后续流程申请。
        selectAlgName = "CcuSchedAllReduceSoleNHRMultiLink";
    // [中文导读] [AllReduce逐行 S308] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // [中文导读] [AllReduce逐行 S310] 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuScheduleLevel0UBXAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_DEBUG("[AllReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    // [中文导读] [AllReduce逐行 S311] 返回 MATCH，已写入候选算法名，旧选择器结束本次优先级尝试。
    return SelectorStatus::MATCH;
// [中文导读] [AllReduce逐行 S312] 结束 SelectCcuScheduleLevel0UBXAlgo 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S314] 定义 SelectCcuScheduleLevel0AlgoMesh1D 入口：单层 CCU Mesh 按 INT8 禁限、Rank 缩放数据阈值及双 Die 规则性选择算法。
SelectorStatus AllReduceAutoSelector::SelectCcuScheduleLevel0AlgoMesh1D(
    // [中文导读] [AllReduce逐行 S315] 续接 SelectCcuScheduleLevel0AlgoMesh1D 的入口参数/基类初始化：const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam, std::string& 算法名输出参数,；引用参数按声明的 const 限制读写。
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam, std::string& selectAlgName,
    // [中文导读] [AllReduce逐行 S316] 续接 SelectCcuScheduleLevel0AlgoMesh1D 的入口参数/基类初始化：const u64 本 Rank 数据字节数) const；引用参数按声明的 const 限制读写。
    const u64 dataSize) const
// [中文导读] [AllReduce逐行 S317] 进入 SelectCcuScheduleLevel0AlgoMesh1D 的实现作用域；单层 CCU Mesh 按 INT8 禁限、Rank 缩放数据阈值及双 Die 规则性选择算法。
{
    // 性能优化改用MS做reduce后不支持int8
    // [中文导读] [AllReduce逐行 S319] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S320] 续接本次错误检查/子调用实参：输入元素类型 等于 HcclDataType::INT8 类型；返回行为由所在完整宏决定。
        opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT8,
        // [中文导读] [AllReduce逐行 S321] 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuScheduleLevel0AlgoMesh1D 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_DEBUG(
            // [中文导读] [AllReduce逐行 S322] 续接 SelectCcuScheduleLevel0AlgoMesh1D 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "[AllReduceAutoSelector] dataType[%d] is not supported yet for ccu schedule mode "
            // [中文导读] [AllReduce逐行 S323] 续接 SelectCcuScheduleLevel0AlgoMesh1D 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "with ms reduce.",
            // [中文导读] [AllReduce逐行 S324] 为 SelectCcuScheduleLevel0AlgoMesh1D 的诊断/错误宏提供实参：输入元素类型，与前面的格式占位依次对应。
            opParam.DataDes.dataType),
        // [中文导读] [AllReduce逐行 S325] 为 SelectCcuScheduleLevel0AlgoMesh1D 的诊断/错误宏提供实参：当前选择器不匹配，与前面的格式占位依次对应。
        SelectorStatus::NOT_MATCH);
    // [中文导读] [AllReduce逐行 S326] 声明本阶段局部变量 double Rank 数平方缩放因子，实际值由后续查询/计算填写。
    double ratio;
    // [中文导读] [AllReduce逐行 S327] 分支条件为 通信域 Rank 总数 等于 0；成立进入本块，未成立继续后续分支。
    if (topoInfo->userRankSize == 0) {
        // [中文导读] [AllReduce逐行 S328] 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuScheduleLevel0AlgoMesh1D 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_DEBUG("[AllReduceAutoSelector] the selector userRankSize not set");
        // [中文导读] [AllReduce逐行 S329] 设置 Rank 数平方缩放因子 为 1；该值供下方当前分支使用。
        ratio = 1;
    // [中文导读] [AllReduce逐行 S330] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
    } else {
        // [中文导读] [AllReduce逐行 S331] 设置 Rank 数平方缩放因子 为 默认参考 Rank 数 / 通信域 Rank 总数 / 通信域 Rank 总数；该值供下方当前分支使用。
        ratio = DEFAULT_RANK_SIZE / topoInfo->userRankSize / topoInfo->userRankSize;
    // [中文导读] [AllReduce逐行 S332] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S333] 分支条件为 本 Rank 数据字节数 * Rank 数平方缩放因子 大于 AR_M2M_1D_MAX_DATA_SIZE；成立进入本块，未成立继续后续分支。
    if (dataSize * ratio > AR_M2M_1D_MAX_DATA_SIZE) {
        // [中文导读] [AllReduce逐行 S334] 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。
        return SelectorStatus::NOT_MATCH;
    // [中文导读] [AllReduce逐行 S335] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S336] 分支条件为 topoInfo->level0MeshType 等于 Level0MeshType::TWO_DIE_REGULAR；成立进入本块，未成立继续后续分支。
    if (topoInfo->level0MeshType == Level0MeshType::TWO_DIE_REGULAR) {
        // [中文导读] [AllReduce逐行 S337] 分支条件为 IsSmallData(本 Rank 数据字节数；成立进入本块，未成立继续后续分支。
        if (IsSmallData(dataSize)) {
            // [中文导读] [AllReduce逐行 S338] 写回候选算法名：算法名输出参数 = "CcuSchedAllReduceSoleMesh2Die"；此处只选择注册名，执行资源在后续流程申请。
            selectAlgName = "CcuSchedAllReduceSoleMesh2Die";
        // [中文导读] [AllReduce逐行 S339] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
        } else {
            // [中文导读] [AllReduce逐行 S340] 写回候选算法名：算法名输出参数 = "CcuSchedAllReduceSequenceMesh2Die"；此处只选择注册名，执行资源在后续流程申请。
            selectAlgName = "CcuSchedAllReduceSequenceMesh2Die";
        // [中文导读] [AllReduce逐行 S341] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
    // [中文导读] [AllReduce逐行 S342] 分支条件为 topoInfo->level0MeshType 等于 Level0MeshType::TWO_DIE_NOT_REGULAR；成立进入本块，未成立继续后续分支。
    } else if (topoInfo->level0MeshType == Level0MeshType::TWO_DIE_NOT_REGULAR) {
        // [中文导读] [AllReduce逐行 S343] 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuScheduleLevel0AlgoMesh1D 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_DEBUG("[AllReduceAutoSelector][%s] TWO_DIE_NOT_REGULAR not match", __func__);
        // [中文导读] [AllReduce逐行 S344] 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。
        return SelectorStatus::NOT_MATCH;
    // [中文导读] [AllReduce逐行 S345] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
    } else {
        // [中文导读] [AllReduce逐行 S346] 写回候选算法名：算法名输出参数 = "CcuSchedAllReduceSoleMesh"；此处只选择注册名，执行资源在后续流程申请。
        selectAlgName = "CcuSchedAllReduceSoleMesh";
    // [中文导读] [AllReduce逐行 S347] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S348] 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuScheduleLevel0AlgoMesh1D 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_DEBUG("[AllReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    // [中文导读] [AllReduce逐行 S349] 返回 MATCH，已写入候选算法名，旧选择器结束本次优先级尝试。
    return SelectorStatus::MATCH;
// [中文导读] [AllReduce逐行 S350] 结束 SelectCcuScheduleLevel0AlgoMesh1D 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S352] 定义 SelectCcuScheduleLevel0Algo 入口：单层 CCU 调度按 MESH_1D、MESH_1D_CLOS、CLOS 分流，并限制输入输出重叠与 PCIe 混合覆盖。
SelectorStatus AllReduceAutoSelector::SelectCcuScheduleLevel0Algo(
    // [中文导读] [AllReduce逐行 S353] 续接 SelectCcuScheduleLevel0Algo 的入口参数/基类初始化：const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam, std::string& 算法名输出参数,；引用参数按声明的 const 限制读写。
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam, std::string& selectAlgName,
    // [中文导读] [AllReduce逐行 S354] 续接 SelectCcuScheduleLevel0Algo 的入口参数/基类初始化：const u64 本 Rank 数据字节数) const；引用参数按声明的 const 限制读写。
    const u64 dataSize) const
// [中文导读] [AllReduce逐行 S355] 进入 SelectCcuScheduleLevel0Algo 的实现作用域；单层 CCU 调度按 MESH_1D、MESH_1D_CLOS、CLOS 分流，并限制输入输出重叠与 PCIe 混合覆盖。
{
    // ccu 模式不支持 inplace 场景
    // [中文导读] [AllReduce逐行 S357] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S358] 调用 IsInputOutputOverlap 完成当前参数所指的子步骤；本行实参为 IsInputOutputOverlap(opParam) 等于 true,。
        IsInputOutputOverlap(opParam) == true,
        // [中文导读] [AllReduce逐行 S359] 开始 HCCL_WARNING 诊断输出，记录 SelectCcuScheduleLevel0Algo 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_WARNING("[Algo][AllReduceAutoSelector] ccu_sched does not support inplace allreduce."),
        // [中文导读] [AllReduce逐行 S360] 为 SelectCcuScheduleLevel0Algo 的诊断/错误宏提供实参：当前选择器不匹配，与前面的格式占位依次对应。
        SelectorStatus::NOT_MATCH);
    // [中文导读] [AllReduce逐行 S361] 分支条件为 第零层拓扑形状 等于 单层 Mesh1D；成立进入本块，未成立继续后续分支。
    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        // [中文导读] [AllReduce逐行 S362] 直接返回 调用 SelectCcuScheduleLevel0AlgoMesh1D 完成当前参数所指的子步骤 的结果，调用者取得本分支结果。
        return SelectCcuScheduleLevel0AlgoMesh1D(topoInfo, opParam, selectAlgName, dataSize);
    // [中文导读] [AllReduce逐行 S363] 分支条件为 第零层拓扑形状 等于 Mesh 与 CLOS 混合拓扑；成立进入本块，未成立继续后续分支。
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        // [中文导读] [AllReduce逐行 S364] 分支条件为 第零层 PCIe 混合标志；成立进入本块，未成立继续后续分支。
        if (topoInfo->level0PcieMix) {
            // [中文导读] [AllReduce逐行 S365] 分支条件为 IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH；成立进入本块，未成立继续后续分支。
            if (IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH)) {
                // [中文导读] [AllReduce逐行 S366] 直接返回 调用 SelectCcuScheduleLevel0AlgoMesh1D 完成当前参数所指的子步骤 的结果，调用者取得本分支结果。
                return SelectCcuScheduleLevel0AlgoMesh1D(topoInfo, opParam, selectAlgName, dataSize);
            // [中文导读] [AllReduce逐行 S367] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
            } else {
                // PCIE-SW定制机型，Mesh无法链接全卡时，需要跨pcie链路，不支持ccu模式
                // [中文导读] [AllReduce逐行 S369] 开始 HCCL_WARNING 诊断输出，记录 SelectCcuScheduleLevel0Algo 当前阶段的参数、候选或错误；日志本身不决定返回码。
                HCCL_WARNING("[AllReduceAutoSelector] pcie mixed topo is not supported yet for ccu schedule mode.");
                // [中文导读] [AllReduce逐行 S370] 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。
                return SelectorStatus::NOT_MATCH;
            // [中文导读] [AllReduce逐行 S371] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
            }
        // [中文导读] [AllReduce逐行 S372] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
        } else {
            // [中文导读] [AllReduce逐行 S373] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
            CHK_PRT_RET(
                // [中文导读] [AllReduce逐行 S374] 续接本次错误检查/子调用实参：输入元素类型 等于 HcclDataType::INT8 类型；返回行为由所在完整宏决定。
                opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT8,
                // [中文导读] [AllReduce逐行 S375] 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuScheduleLevel0Algo 当前阶段的参数、候选或错误；日志本身不决定返回码。
                HCCL_DEBUG(
                    // [中文导读] [AllReduce逐行 S376] 续接 SelectCcuScheduleLevel0Algo 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
                    "[AllReduceAutoSelector] dataType[%d] is not supported yet for ccu schedule mode. "
                    // [中文导读] [AllReduce逐行 S377] 续接 SelectCcuScheduleLevel0Algo 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
                    "with ms reduce.",
                    // [中文导读] [AllReduce逐行 S378] 为 SelectCcuScheduleLevel0Algo 的诊断/错误宏提供实参：输入元素类型，与前面的格式占位依次对应。
                    opParam.DataDes.dataType),
                // [中文导读] [AllReduce逐行 S379] 为 SelectCcuScheduleLevel0Algo 的诊断/错误宏提供实参：当前选择器不匹配，与前面的格式占位依次对应。
                SelectorStatus::NOT_MATCH);
            // [中文导读] [AllReduce逐行 S380] 直接返回 调用 SelectCcuScheduleLevel0UBXAlgo 完成当前参数所指的子步骤 的结果，调用者取得本分支结果。
            return SelectCcuScheduleLevel0UBXAlgo(topoInfo, selectAlgName, dataSize);
        // [中文导读] [AllReduce逐行 S381] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
    // [中文导读] [AllReduce逐行 S382] 分支条件为 第零层拓扑形状 等于 CLOS 拓扑；成立进入本块，未成立继续后续分支。
    } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
        // [中文导读] [AllReduce逐行 S383] 分支条件为 第零层 PCIe 混合标志；成立进入本块，未成立继续后续分支。
        if (topoInfo->level0PcieMix) {
            // PCIE-SW定制机型，Mesh无法链接全卡时，需要跨pcie链路，不支持ccu模式
            // [中文导读] [AllReduce逐行 S385] 开始 HCCL_WARNING 诊断输出，记录 SelectCcuScheduleLevel0Algo 当前阶段的参数、候选或错误；日志本身不决定返回码。
            HCCL_WARNING("[AllReduceAutoSelector] pcie mixed topo is not supported yet for ccu schedule mode.");
            // [中文导读] [AllReduce逐行 S386] 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。
            return SelectorStatus::NOT_MATCH;
        // [中文导读] [AllReduce逐行 S387] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
        // [中文导读] [AllReduce逐行 S388] 分支条件为 本 Rank 数据字节数 小于 AR_CCU_CLOS_1D_SMALL_DATA_SIZE；成立进入本块，未成立继续后续分支。
        if (dataSize < AR_CCU_CLOS_1D_SMALL_DATA_SIZE) {
            // [中文导读] [AllReduce逐行 S389] 写回候选算法名：算法名输出参数 = "CcuSchedAllReduceSoleNHR"；此处只选择注册名，执行资源在后续流程申请。
            selectAlgName = "CcuSchedAllReduceSoleNHR";
            // [中文导读] [AllReduce逐行 S390] 返回 MATCH，已写入候选算法名，旧选择器结束本次优先级尝试。
            return SelectorStatus::MATCH;
        // [中文导读] [AllReduce逐行 S391] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
        } else {
            // [中文导读] [AllReduce逐行 S392] 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。
            return SelectorStatus::NOT_MATCH;
        // [中文导读] [AllReduce逐行 S393] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
    // [中文导读] [AllReduce逐行 S394] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
    } else {
        // [中文导读] [AllReduce逐行 S395] 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuScheduleLevel0Algo 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_DEBUG(
            // [中文导读] [AllReduce逐行 S396] 续接 SelectCcuScheduleLevel0Algo 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "[AllReduceAutoSelector] level0Topo[%d] is not supported yet for ccu schedule mode.", topoInfo->level0Topo);
        // [中文导读] [AllReduce逐行 S397] 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。
        return SelectorStatus::NOT_MATCH;
    // [中文导读] [AllReduce逐行 S398] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S399] 开始 HCCL_DEBUG 诊断输出，记录 SelectCcuScheduleLevel0Algo 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_DEBUG("[AllReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    // [中文导读] [AllReduce逐行 S400] 返回 MATCH，已写入候选算法名，旧选择器结束本次优先级尝试。
    return SelectorStatus::MATCH;
// [中文导读] [AllReduce逐行 S401] 结束 SelectCcuScheduleLevel0Algo 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S403] 定义 SelectAicpuAlgo 入口：旧 AICPU 自动算法选择：strict 优先，特殊归约类型、多层 UB/UBOE、NHR 标志、单层 Mesh 分开处理。
SelectorStatus AllReduceAutoSelector::SelectAicpuAlgo(
    // [中文导读] [AllReduce逐行 S404] 续接 SelectAicpuAlgo 的入口参数/基类初始化：const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,；引用参数按声明的 const 限制读写。
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
    // [中文导读] [AllReduce逐行 S405] 续接 SelectAicpuAlgo 的入口参数/基类初始化：const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& 算法名输出参数) const；引用参数按声明的 const 限制读写。
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName) const
// [中文导读] [AllReduce逐行 S406] 进入 SelectAicpuAlgo 的实现作用域；旧 AICPU 自动算法选择：strict 优先，特殊归约类型、多层 UB/UBOE、NHR 标志、单层 Mesh 分开处理。
{
    // [中文导读] [AllReduce逐行 S407] 显式标记 configAlgMap 在此兼容/default 分支未使用，避免编译器未使用参数警告。
    (void)configAlgMap;
    // [中文导读] [AllReduce逐行 S408] 开始 HCCL_DEBUG 诊断输出，记录 SelectAicpuAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_DEBUG("[AllReduceAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);
    // [中文导读] [AllReduce逐行 S409] 设置 单元素字节数 为 类型到元素字节数的查找表[输入元素类型]；该值供下方当前分支使用。
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    // [中文导读] [AllReduce逐行 S410] 设置 本 Rank 数据字节数 为 本 Rank 输入元素数 * 单元素字节数；该值供下方当前分支使用。
    u64 dataSize = opParam.DataDes.count * perDataSize;

    // [中文导读] [AllReduce逐行 S412] 分支条件为 IsNeedStrictModeForOrderPreserved(opParam, 通信域 Rank 总数；成立进入本块，未成立继续后续分支。
    if (IsNeedStrictModeForOrderPreserved(opParam, topoInfo->userRankSize)) {
        // [中文导读] [AllReduce逐行 S413] 分支条件为 通信域 Rank 总数 大于 MAX_RANK_NUM_FOR_ORDER_PRESERVED；成立进入本块，未成立继续后续分支。
        if (topoInfo->userRankSize > MAX_RANK_NUM_FOR_ORDER_PRESERVED) {
            // 内部reducescatter中采用分组all2all + NHR 算法
            // [中文导读] [AllReduce逐行 S415] 写回候选算法名：算法名输出参数 = "AicpuAllReduceStrictOrderedGroupMesh"；此处只选择注册名，执行资源在后续流程申请。
            selectAlgName = "AicpuAllReduceStrictOrderedGroupMesh";
        // [中文导读] [AllReduce逐行 S416] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
        } else {
            // 内部reducescatter中采用非分组all2all + mesh1D 算法
            // [中文导读] [AllReduce逐行 S418] 写回候选算法名：算法名输出参数 = "AicpuAllReduceStrictOrderedMesh"；此处只选择注册名，执行资源在后续流程申请。
            selectAlgName = "AicpuAllReduceStrictOrderedMesh";
        // [中文导读] [AllReduce逐行 S419] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
        // [中文导读] [AllReduce逐行 S420] 开始 HCCL_INFO 诊断输出，记录 SelectAicpuAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_INFO(
            // [中文导读] [AllReduce逐行 S421] 续接 SelectAicpuAlgo 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "[AllReduceAutoSelector] DETERMINISTIC_STRICT mode, rankSize[%u], threshold[%u], "
            // [中文导读] [AllReduce逐行 S422] 续接 SelectAicpuAlgo 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "select [%s]",
            // [中文导读] [AllReduce逐行 S423] 为 SelectAicpuAlgo 的诊断/错误宏提供实参：通信域 Rank 总数, MAX_RANK_NUM_FOR_ORDER_PRESERVED, 算法名输出参数.c_str(，与前面的格式占位依次对应。
            topoInfo->userRankSize, MAX_RANK_NUM_FOR_ORDER_PRESERVED, selectAlgName.c_str());
        // [中文导读] [AllReduce逐行 S424] 返回 MATCH，已写入候选算法名，旧选择器结束本次优先级尝试。
        return SelectorStatus::MATCH;
    // [中文导读] [AllReduce逐行 S425] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // [中文导读] [AllReduce逐行 S427] 续接 SelectAicpuAlgo 当前语句的具体实参/字段：bool 64 位类型或 PROD 的软件归约标志 = 输入元素类型 等于 HcclDataType::有符号 64 位整数；由其完整表达式完成参数组装、检查或结果写回。
    bool isDataTypeOrReduceTypeSpecial = opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT64
                                         // [中文导读] [AllReduce逐行 S428] 补充同一条件的 或者 子条件：输入元素类型 等于 HcclDataType::无符号 64 位整数。
                                         || opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_UINT64
                                         // [中文导读] [AllReduce逐行 S429] 补充同一条件的 或者 子条件：输入元素类型 等于 HcclDataType::FP64 类型。
                                         || opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_FP64
                                         // [中文导读] [AllReduce逐行 S430] 补充同一条件的 或者 子条件：归约运算 等于 HcclReduceOp::乘积归约。
                                         || opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD;

    // [中文导读] [AllReduce逐行 S432] 分支条件为 topoInfo->topoLevelNums 大于 1；成立进入本块，未成立继续后续分支。
    if (topoInfo->topoLevelNums > 1) {
        // [中文导读] [AllReduce逐行 S433] 设置 level0AndLevel1Symetric 为 topoInfo->level0Symmetric 且 topoInfo->level1Symmetric；该值供下方当前分支使用。
        bool level0AndLevel1Symetric = topoInfo->level0Symmetric && topoInfo->level1Symmetric;
        // [中文导读] [AllReduce逐行 S434] 分支条件为 64 位类型或 PROD 的软件归约标志；成立进入本块，未成立继续后续分支。
        if (isDataTypeOrReduceTypeSpecial) {
            // [中文导读] [AllReduce逐行 S435] 写回候选算法名：算法名输出参数 = "AicpuAllReduceSoleNHRAicpuReduce"；此处只选择注册名，执行资源在后续流程申请。
            selectAlgName = "AicpuAllReduceSoleNHRAicpuReduce";
        // [中文导读] [AllReduce逐行 S436] 分支条件为 ；成立进入本块，未成立继续后续分支。
        } else if (
            // [中文导读] [AllReduce逐行 S437] 续接 SelectAicpuAlgo 当前语句的具体实参/字段：level0AndLevel1Symetric 且 topoInfo->deviceNumPerModule 等于 DEVICE_NUM_PER_MODULE_8；由其完整表达式完成参数组装、检查或结果写回。
            level0AndLevel1Symetric && topoInfo->deviceNumPerModule == DEVICE_NUM_PER_MODULE_8
            // [中文导读] [AllReduce逐行 S438] 补充同一条件的 并且 子条件：topoInfo->topLevelUboe。
            && topoInfo->topLevelUboe) {
            // [中文导读] [AllReduce逐行 S439] 分支条件为 topoInfo->topoLevelNums 等于 TOPO_LEVEL_NUM_2；成立进入本块，未成立继续后续分支。
            if (topoInfo->topoLevelNums == TOPO_LEVEL_NUM_2) {
                // [中文导读] [AllReduce逐行 S440] 写回候选算法名：算法名输出参数 = "AicpuAllReducePipeLineMeshNHR"；此处只选择注册名，执行资源在后续流程申请。
                selectAlgName = "AicpuAllReducePipeLineMeshNHR";
            // [中文导读] [AllReduce逐行 S441] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
            } else {
                // [中文导读] [AllReduce逐行 S442] 写回候选算法名：算法名输出参数 = "AicpuAllReducePipeLineMeshNHRNHR"；此处只选择注册名，执行资源在后续流程申请。
                selectAlgName = "AicpuAllReducePipeLineMeshNHRNHR";
            // [中文导读] [AllReduce逐行 S443] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
            }
        // [中文导读] [AllReduce逐行 S444] 分支条件为 level0AndLevel1Symetric 且 topoInfo->topoLevelNums 等于 TOPO_LEVEL_NUM_3 且 topoInfo->topLevelUboe；成立进入本块，未成立继续后续分支。
        } else if (level0AndLevel1Symetric && topoInfo->topoLevelNums == TOPO_LEVEL_NUM_3 && topoInfo->topLevelUboe) {
            // [中文导读] [AllReduce逐行 S445] 写回候选算法名：算法名输出参数 = "AicpuAllReduceParallelMeshNHR"；此处只选择注册名，执行资源在后续流程申请。
            selectAlgName = "AicpuAllReduceParallelMeshNHR";
        // [中文导读] [AllReduce逐行 S446] 分支条件为 topoInfo->Level1Nhr；成立进入本块，未成立继续后续分支。
        } else if (topoInfo->Level1Nhr) {
            // Level1Nhr 已在 CalcTopoShape 中设置（GCD==1 时为 true）
            // [中文导读] [AllReduce逐行 S448] 写回候选算法名：算法名输出参数 = "AicpuAllReduceSoleNHR"；此处只选择注册名，执行资源在后续流程申请。
            selectAlgName = "AicpuAllReduceSoleNHR";
            // [中文导读] [AllReduce逐行 S449] 开始 HCCL_INFO 诊断输出，记录 SelectAicpuAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。
            HCCL_INFO("[AllReduceAutoSelector] Level1Nhr=true, select [%s]", selectAlgName.c_str());
        // [中文导读] [AllReduce逐行 S450] 分支条件为 topoInfo->当前网络层编号Details.localNetInsSizeOfLayer[0] 等于 1；成立进入本块，未成立继续后续分支。
        } else if (topoInfo->netLayerDetails.localNetInsSizeOfLayer[0] == 1) {
            // [中文导读] [AllReduce逐行 S451] 写回候选算法名：算法名输出参数 = "AicpuAllReduceSoleNHR"；此处只选择注册名，执行资源在后续流程申请。
            selectAlgName = "AicpuAllReduceSoleNHR";
        // [中文导读] [AllReduce逐行 S452] 分支条件为 第零层拓扑形状 等于 单层 Mesh1D；成立进入本块，未成立继续后续分支。
        } else if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            // [中文导读] [AllReduce逐行 S453] 分支条件为 topoInfo->topoLevelNums 等于 TOPO_LEVEL_NUM_3；成立进入本块，未成立继续后续分支。
            if (topoInfo->topoLevelNums == TOPO_LEVEL_NUM_3) {
                // [中文导读] [AllReduce逐行 S454] 分支条件为 level0AndLevel1Symetric；成立进入本块，未成立继续后续分支。
                if (level0AndLevel1Symetric) {
                    // [中文导读] [AllReduce逐行 S455] 写回候选算法名：算法名输出参数 = "AicpuAllReduceSequenceMeshConcurNHRNHR"；此处只选择注册名，执行资源在后续流程申请。
                    selectAlgName = "AicpuAllReduceSequenceMeshConcurNHRNHR";
                // [中文导读] [AllReduce逐行 S456] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
                } else {
                    // [中文导读] [AllReduce逐行 S457] 写回候选算法名：算法名输出参数 = "AicpuAllReduceSoleNHR"；此处只选择注册名，执行资源在后续流程申请。
                    selectAlgName = "AicpuAllReduceSoleNHR";
                // [中文导读] [AllReduce逐行 S458] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
                }
            // [中文导读] [AllReduce逐行 S459] 分支条件为 topoInfo->topoLevelNums 等于 TOPO_LEVEL_NUM_2；成立进入本块，未成立继续后续分支。
            } else if (topoInfo->topoLevelNums == TOPO_LEVEL_NUM_2) {
                // [中文导读] [AllReduce逐行 S460] 分支条件为 本 Rank 数据字节数 大于 AR_AICPU_1D_CROSS_SMALL_DATA_SIZE；成立进入本块，未成立继续后续分支。
                if (dataSize > AR_AICPU_1D_CROSS_SMALL_DATA_SIZE) {
                    // [中文导读] [AllReduce逐行 S461] 写回候选算法名：算法名输出参数 = (本 Rank 数据字节数 大于 4GiB 顺序/并发切换门槛) ? "AicpuAllReduceSequenceMeshConcurNHR" :；此处只选择注册名，执行资源在后续流程申请。
                    selectAlgName = (dataSize > AR_AICPU_SEQUENCE_DATA_SIZE) ? "AicpuAllReduceSequenceMeshConcurNHR" :
                                                                               // [中文导读] [AllReduce逐行 S462] 续接本次调用的字符串常量 "AicpuAllReduceParallelMeshNHR";，由所在注册/日志/条件语句整体使用。
                                                                               "AicpuAllReduceParallelMeshNHR";
                // [中文导读] [AllReduce逐行 S463] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
                } else {
                    // [中文导读] [AllReduce逐行 S464] 写回候选算法名：算法名输出参数 = "AicpuAllReduceSoleNHR"；此处只选择注册名，执行资源在后续流程申请。
                    selectAlgName = "AicpuAllReduceSoleNHR";
                // [中文导读] [AllReduce逐行 S465] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
                }
            // [中文导读] [AllReduce逐行 S466] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
            } else {
                // [中文导读] [AllReduce逐行 S467] 写回候选算法名：算法名输出参数 = "AicpuAllReduceSoleNHR"；此处只选择注册名，执行资源在后续流程申请。
                selectAlgName = "AicpuAllReduceSoleNHR";
            // [中文导读] [AllReduce逐行 S468] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
            }
        // [中文导读] [AllReduce逐行 S469] 分支条件为 第零层拓扑形状 等于 CLOS 拓扑；成立进入本块，未成立继续后续分支。
        } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
            // [中文导读] [AllReduce逐行 S470] 写回候选算法名：算法名输出参数 = "AicpuAllReduceSoleNHRMultiLink"；此处只选择注册名，执行资源在后续流程申请。
            selectAlgName = "AicpuAllReduceSoleNHRMultiLink";
        // [中文导读] [AllReduce逐行 S471] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
        } else {
            // [中文导读] [AllReduce逐行 S472] 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。
            return SelectorStatus::NOT_MATCH;
        // [中文导读] [AllReduce逐行 S473] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
    // [中文导读] [AllReduce逐行 S474] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
    } else {
        // [中文导读] [AllReduce逐行 S475] 直接返回 进入 AICPU 单层 Mesh/Clos 阈值分支 的结果，调用者取得本分支结果。
        return SelectMeshAlgoAicpu(topoInfo, opParam, selectAlgName);
    // [中文导读] [AllReduce逐行 S476] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // [中文导读] [AllReduce逐行 S478] 开始 HCCL_DEBUG 诊断输出，记录 SelectAicpuAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_DEBUG("[AllReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    // [中文导读] [AllReduce逐行 S479] 返回 MATCH，已写入候选算法名，旧选择器结束本次优先级尝试。
    return SelectorStatus::MATCH;
// [中文导读] [AllReduce逐行 S480] 结束 SelectAicpuAlgo 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S482] 定义 SelectMeshAlgoAicpuUBX 入口：AICPU 的 UBX 拓扑选择，区别小规模 Mesh 等于 CLOS、特殊类型、矩形大数据和对称内存候选。
SelectorStatus AllReduceAutoSelector::SelectMeshAlgoAicpuUBX(
    // [中文导读] [AllReduce逐行 S483] 续接 SelectMeshAlgoAicpuUBX 的入口参数/基类初始化：const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam, const u64 本 Rank 数据字节数, std::string& 算法名输出参数,；引用参数按声明的 const 限制读写。
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam, const u64 dataSize, std::string& selectAlgName,
    // [中文导读] [AllReduce逐行 S484] 续接 SelectMeshAlgoAicpuUBX 的入口参数/基类初始化：bool 64 位类型或 PROD 的软件归约标志) const；引用参数按声明的 const 限制读写。
    bool isDataTypeOrReduceTypeSpecial) const
// [中文导读] [AllReduce逐行 S485] 进入 SelectMeshAlgoAicpuUBX 的实现作用域；AICPU 的 UBX 拓扑选择，区别小规模 Mesh 等于 CLOS、特殊类型、矩形大数据和对称内存候选。
{
    // UBX机型
    // [中文导读] [AllReduce逐行 S487] 设置 isMeshNumEqualToClosNum 为 false；该值供下方当前分支使用。
    bool isMeshNumEqualToClosNum = false;
    // [中文导读] [AllReduce逐行 S488] 设置 isClosNumMultipleOfMeshNum 为 false；该值供下方当前分支使用。
    bool isClosNumMultipleOfMeshNum = false;
    // [中文导读] [AllReduce逐行 S489] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S490] 调用 CheckMeshNumEqualToClosNum 完成当前参数所指的子步骤；本行实参为 CheckMeshNumEqualToClosNum(topoInfo, isMeshNumEqualToClosNum) 不等于 成功状态,。
        CheckMeshNumEqualToClosNum(topoInfo, isMeshNumEqualToClosNum) != HCCL_SUCCESS,
        // [中文导读] [AllReduce逐行 S491] 开始 HCCL_ERROR 诊断输出，记录 SelectMeshAlgoAicpuUBX 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_ERROR("[Algo][AllReduceAutoSelector] CheckMeshNumEqualToClosNum failed."), SelectorStatus::NOT_MATCH);
    // [中文导读] [AllReduce逐行 S492] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S493] 调用 CheckClosNumMultipleOfMeshNum 完成当前参数所指的子步骤；本行实参为 CheckClosNumMultipleOfMeshNum(topoInfo, isClosNumMultipleOfMeshNum) 不等于 成功状态,。
        CheckClosNumMultipleOfMeshNum(topoInfo, isClosNumMultipleOfMeshNum) != HCCL_SUCCESS,
        // [中文导读] [AllReduce逐行 S494] 开始 HCCL_ERROR 诊断输出，记录 SelectMeshAlgoAicpuUBX 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_ERROR("[Algo][AllReduceAutoSelector] CheckClosNumMultipleOfMeshNum failed."), SelectorStatus::NOT_MATCH);
    // [中文导读] [AllReduce逐行 S495] 分支条件为 isMeshNumEqualToClosNum 且 通信域 Rank 总数 不超过 MAX_RANK_NUM_FOR_CONCURRENT_ALGO；成立进入本块，未成立继续后续分支。
    if (isMeshNumEqualToClosNum && topoInfo->userRankSize <= MAX_RANK_NUM_FOR_CONCURRENT_ALGO) {
        // [中文导读] [AllReduce逐行 S496] 分支条件为 64 位类型或 PROD 的软件归约标志；成立进入本块，未成立继续后续分支。
        if (isDataTypeOrReduceTypeSpecial) {
            // [中文导读] [AllReduce逐行 S497] 写回候选算法名：算法名输出参数 = 本 Rank 数据字节数 不超过 8MiB 特殊类型门槛 ? "AicpuAllReduceSoleMeshOneShot" :；此处只选择注册名，执行资源在后续流程申请。
            selectAlgName = dataSize <= AR_AICPU_1D_64DATATYPE_DATA_SIZE ? "AicpuAllReduceSoleMeshOneShot" :
                                                                           // [中文导读] [AllReduce逐行 S498] 续接本次调用的字符串常量 "AicpuAllReduceSoleMeshTwoShot";，由所在注册/日志/条件语句整体使用。
                                                                           "AicpuAllReduceSoleMeshTwoShot";
        // [中文导读] [AllReduce逐行 S499] 分支条件为 本 Rank 数据字节数 不超过 8MiB OneShot 门槛；成立进入本块，未成立继续后续分支。
        } else if (dataSize <= AR_AICPU_1D_SMALL_DATA_SIZE) {
            // [中文导读] [AllReduce逐行 S500] 写回候选算法名：算法名输出参数 = "AicpuAllReduceSoleMeshOneShot"；此处只选择注册名，执行资源在后续流程申请。
            selectAlgName = "AicpuAllReduceSoleMeshOneShot";
        // [中文导读] [AllReduce逐行 S501] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
        } else {
            // 大数据量，用mesh+clos并行算法
            // [中文导读] [AllReduce逐行 S503] 写回候选算法名：算法名输出参数 = "AicpuAllReduceConcurMeshTwoShotNHR"；此处只选择注册名，执行资源在后续流程申请。
            selectAlgName = "AicpuAllReduceConcurMeshTwoShotNHR";
        // [中文导读] [AllReduce逐行 S504] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
    // [中文导读] [AllReduce逐行 S505] 分支条件为 64 位类型或 PROD 的软件归约标志；成立进入本块，未成立继续后续分支。
    } else if (isDataTypeOrReduceTypeSpecial) {
        // [中文导读] [AllReduce逐行 S506] 写回候选算法名：算法名输出参数 = "AicpuAllReduceSoleNHRAicpuReduce"；此处只选择注册名，执行资源在后续流程申请。
        selectAlgName = "AicpuAllReduceSoleNHRAicpuReduce";
    // [中文导读] [AllReduce逐行 S507] 分支条件为 isClosNumMultipleOfMeshNum 且 IsLargeData(本 Rank 数据字节数；成立进入本块，未成立继续后续分支。
    } else if (isClosNumMultipleOfMeshNum && IsLargeData(dataSize)) {
        // [中文导读] [AllReduce逐行 S508] 分支条件为 对称内存标志；成立进入本块，未成立继续后续分支。
        if (opParam.supportSymmetricMemory) {
            // [中文导读] [AllReduce逐行 S509] 写回候选算法名：算法名输出参数 = "AicpuAllReducePipeLineMeshNHR"；此处只选择注册名，执行资源在后续流程申请。
            selectAlgName = "AicpuAllReducePipeLineMeshNHR";
        // [中文导读] [AllReduce逐行 S510] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
        } else {
            // 矩形场景大数据量，用Parallel并行算法
            // [中文导读] [AllReduce逐行 S512] 写回候选算法名：算法名输出参数 = "AicpuAllReduceParallelMeshNHRMultiJetty"；此处只选择注册名，执行资源在后续流程申请。
            selectAlgName = "AicpuAllReduceParallelMeshNHRMultiJetty";
        // [中文导读] [AllReduce逐行 S513] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
    // [中文导读] [AllReduce逐行 S514] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
    } else {
        // 其他场景，用1d NHR算法
        // [中文导读] [AllReduce逐行 S516] 写回候选算法名：算法名输出参数 = "AicpuAllReduceSoleNHR"；此处只选择注册名，执行资源在后续流程申请。
        selectAlgName = "AicpuAllReduceSoleNHR";
    // [中文导读] [AllReduce逐行 S517] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // [中文导读] [AllReduce逐行 S519] 开始 HCCL_DEBUG 诊断输出，记录 SelectMeshAlgoAicpuUBX 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_DEBUG("[AllReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    // [中文导读] [AllReduce逐行 S520] 返回 MATCH，已写入候选算法名，旧选择器结束本次优先级尝试。
    return SelectorStatus::MATCH;
// [中文导读] [AllReduce逐行 S521] 结束 SelectMeshAlgoAicpuUBX 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S523] 定义 SelectMeshAlgoAicpu 入口：旧单层 AICPU Mesh 选择：普通 FP32 SUM 的非绕路小数据 ≤8MiB 选 OneShot；其它阈值和拓扑选择 TwoShot/Chunk/NHR。
SelectorStatus AllReduceAutoSelector::SelectMeshAlgoAicpu(
    // [中文导读] [AllReduce逐行 S524] 续接 SelectMeshAlgoAicpu 的入口参数/基类初始化：const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam, std::string& 算法名输出参数) const；引用参数按声明的 const 限制读写。
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam, std::string& selectAlgName) const
// [中文导读] [AllReduce逐行 S525] 进入 SelectMeshAlgoAicpu 的实现作用域；旧单层 AICPU Mesh 选择：普通 FP32 SUM 的非绕路小数据 ≤8MiB 选 OneShot；其它阈值和拓扑选择 TwoShot/Chunk/NHR。
{
    // [中文导读] [AllReduce逐行 S526] 设置 单元素字节数 为 类型到元素字节数的查找表[输入元素类型]；该值供下方当前分支使用。
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    // [中文导读] [AllReduce逐行 S527] 设置 本 Rank 数据字节数 为 本 Rank 输入元素数 * 单元素字节数；该值供下方当前分支使用。
    u64 dataSize = opParam.DataDes.count * perDataSize;

    // [中文导读] [AllReduce逐行 S529] 续接 SelectMeshAlgoAicpu 当前语句的具体实参/字段：bool 64 位类型或 PROD 的软件归约标志 = 输入元素类型 等于 HcclDataType::有符号 64 位整数；由其完整表达式完成参数组装、检查或结果写回。
    bool isDataTypeOrReduceTypeSpecial = opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT64
                                         // [中文导读] [AllReduce逐行 S530] 补充同一条件的 或者 子条件：输入元素类型 等于 HcclDataType::无符号 64 位整数。
                                         || opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_UINT64
                                         // [中文导读] [AllReduce逐行 S531] 补充同一条件的 或者 子条件：输入元素类型 等于 HcclDataType::FP64 类型。
                                         || opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_FP64
                                         // [中文导读] [AllReduce逐行 S532] 补充同一条件的 或者 子条件：归约运算 等于 HcclReduceOp::乘积归约。
                                         || opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD;

    // [中文导读] [AllReduce逐行 S534] 声明本阶段局部变量 double Rank 数平方缩放因子，实际值由后续查询/计算填写。
    double ratio;
    // [中文导读] [AllReduce逐行 S535] 分支条件为 通信域 Rank 总数 等于 0；成立进入本块，未成立继续后续分支。
    if (topoInfo->userRankSize == 0) {
        // [中文导读] [AllReduce逐行 S536] 开始 HCCL_WARNING 诊断输出，记录 SelectMeshAlgoAicpu 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_WARNING("[AllReduceAutoSelector] the selector userRankSize not set");
        // [中文导读] [AllReduce逐行 S537] Rank 总数为零时只警告，将阈值缩放因子设为 1，避免下一分支除以零。
        ratio = 1;
    // [中文导读] [AllReduce逐行 S538] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
    } else {
        // [中文导读] [AllReduce逐行 S539] 当 Rank 总数可用时，以默认参考 Rank 数除以 Rank 数平方缩放大数据阈值；此表达式按变量原始类型计算。
        ratio = DEFAULT_RANK_SIZE / topoInfo->userRankSize / topoInfo->userRankSize;
    // [中文导读] [AllReduce逐行 S540] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S541] 设置 可用二层网络标志 为 IsTwoLevelNetLayer(topoInfo, opParam)；该值供下方当前分支使用。
    bool isTwoLevelFlag = IsTwoLevelNetLayer(topoInfo, opParam);
    // [中文导读] [AllReduce逐行 S542] 设置 超过 4GiB 的标志 为 本 Rank 数据字节数 大于 4GiB 顺序/并发切换门槛；该值供下方当前分支使用。
    bool overSequenceDataThreshold = dataSize > AR_AICPU_SEQUENCE_DATA_SIZE;
    // [中文导读] [AllReduce逐行 S543] 分支条件为 第零层拓扑形状 等于 单层 Mesh1D；成立进入本块，未成立继续后续分支。
    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        // [中文导读] [AllReduce逐行 S544] 分支条件为 64 位类型或 PROD 的软件归约标志；成立进入本块，未成立继续后续分支。
        if (isDataTypeOrReduceTypeSpecial) {
            // [中文导读] [AllReduce逐行 S545] 写回候选算法名：算法名输出参数 = 本 Rank 数据字节数 不超过 8MiB 特殊类型门槛 ? "AicpuAllReduceSoleMeshOneShot" :；此处只选择注册名，执行资源在后续流程申请。
            selectAlgName = dataSize <= AR_AICPU_1D_64DATATYPE_DATA_SIZE ? "AicpuAllReduceSoleMeshOneShot" :
                                                                           // [中文导读] [AllReduce逐行 S546] 续接本次调用的字符串常量 "AicpuAllReduceSoleMeshTwoShot";，由所在注册/日志/条件语句整体使用。
                                                                           "AicpuAllReduceSoleMeshTwoShot";
        // [中文导读] [AllReduce逐行 S547] 分支条件为 可用二层网络标志 且 通信域 Rank 总数 等于 2 且 本 Rank 数据字节数 至少 8MiB 双 Rank 绕路门槛；成立进入本块，未成立继续后续分支。
        } else if (isTwoLevelFlag && topoInfo->userRankSize == 2 && dataSize >= AR_2P_DETOUR_DATA_SIZE) {
            // 两p条件下生效
            // [中文导读] [AllReduce逐行 S549] 写回候选算法名：算法名输出参数 = "AicpuAllReduceSoleMeshConcur"；此处只选择注册名，执行资源在后续流程申请。
            selectAlgName = "AicpuAllReduceSoleMeshConcur";
        // [中文导读] [AllReduce逐行 S550] 普通 Mesh1D 输入字节数不超过 8MiB 时进入 OneShot 分支；必须先排除特殊类型与双 Rank 绕路。
        } else if (dataSize <= AR_AICPU_1D_SMALL_DATA_SIZE) {
            // [中文导读] [AllReduce逐行 S551] 选择 AicpuAllReduceSoleMeshOneShot，后续注册表绑定 SoleExecutor 与 OneShot 模板。
            selectAlgName = "AicpuAllReduceSoleMeshOneShot";
        // [中文导读] [AllReduce逐行 S552] 分支条件为 本 Rank 数据字节数 * Rank 数平方缩放因子 大于 32MiB 分块 TwoShot 门槛；成立进入本块，未成立继续后续分支。
        } else if (dataSize * ratio > AR_AICPU_1D_MAX_DATA_SIZE) {
            // [中文导读] [AllReduce逐行 S553] 写回候选算法名：算法名输出参数 = (可用二层网络标志 且 超过 4GiB 的标志) ? "AicpuAllReduceSoleMeshConcur" :；此处只选择注册名，执行资源在后续流程申请。
            selectAlgName = (isTwoLevelFlag && overSequenceDataThreshold) ? "AicpuAllReduceSoleMeshConcur" :
                                                                            // [中文导读] [AllReduce逐行 S554] 续接本次调用的字符串常量 "AicpuAllReduceSoleMeshChunkTwoShot";，由所在注册/日志/条件语句整体使用。
                                                                            "AicpuAllReduceSoleMeshChunkTwoShot";
        // [中文导读] [AllReduce逐行 S555] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
        } else {
            // [中文导读] [AllReduce逐行 S556] 写回候选算法名：算法名输出参数 = "AicpuAllReduceSoleMeshTwoShot"；此处只选择注册名，执行资源在后续流程申请。
            selectAlgName = "AicpuAllReduceSoleMeshTwoShot";
        // [中文导读] [AllReduce逐行 S557] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
    // [中文导读] [AllReduce逐行 S558] 分支条件为 第零层拓扑形状 等于 CLOS 拓扑；成立进入本块，未成立继续后续分支。
    } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
        // [中文导读] [AllReduce逐行 S559] 分支条件为 64 位类型或 PROD 的软件归约标志；成立进入本块，未成立继续后续分支。
        if (isDataTypeOrReduceTypeSpecial) {
            // [中文导读] [AllReduce逐行 S560] 写回候选算法名：算法名输出参数 = "AicpuAllReduceSoleNHRAicpuReduce"；此处只选择注册名，执行资源在后续流程申请。
            selectAlgName = "AicpuAllReduceSoleNHRAicpuReduce";
        // [中文导读] [AllReduce逐行 S561] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
        } else {
            // [中文导读] [AllReduce逐行 S562] 写回候选算法名：算法名输出参数 = "AicpuAllReduceSoleNHRMultiLink"；此处只选择注册名，执行资源在后续流程申请。
            selectAlgName = "AicpuAllReduceSoleNHRMultiLink";
        // [中文导读] [AllReduce逐行 S563] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
    // [中文导读] [AllReduce逐行 S564] 分支条件为 第零层拓扑形状 等于 Mesh 与 CLOS 混合拓扑；成立进入本块，未成立继续后续分支。
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        // [中文导读] [AllReduce逐行 S565] 分支条件为 第零层 PCIe 混合标志；成立进入本块，未成立继续后续分支。
        if (topoInfo->level0PcieMix) {
            // [中文导读] [AllReduce逐行 S566] 分支条件为 IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH；成立进入本块，未成立继续后续分支。
            if (IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH)) {
                // [中文导读] [AllReduce逐行 S567] 分支条件为 64 位类型或 PROD 的软件归约标志；成立进入本块，未成立继续后续分支。
                if (isDataTypeOrReduceTypeSpecial) {
                    // [中文导读] [AllReduce逐行 S568] 写回候选算法名：算法名输出参数 = 本 Rank 数据字节数 不超过 8MiB 特殊类型门槛 ? "AicpuAllReduceSoleMeshOneShot" :；此处只选择注册名，执行资源在后续流程申请。
                    selectAlgName = dataSize <= AR_AICPU_1D_64DATATYPE_DATA_SIZE ? "AicpuAllReduceSoleMeshOneShot" :
                                                                                   // [中文导读] [AllReduce逐行 S569] 续接本次调用的字符串常量 "AicpuAllReduceSoleMeshTwoShot";，由所在注册/日志/条件语句整体使用。
                                                                                   "AicpuAllReduceSoleMeshTwoShot";
                // [中文导读] [AllReduce逐行 S570] 分支条件为 本 Rank 数据字节数 不超过 8MiB OneShot 门槛；成立进入本块，未成立继续后续分支。
                } else if (dataSize <= AR_AICPU_1D_SMALL_DATA_SIZE) {
                    // [中文导读] [AllReduce逐行 S571] 写回候选算法名：算法名输出参数 = "AicpuAllReduceSoleMeshOneShot"；此处只选择注册名，执行资源在后续流程申请。
                    selectAlgName = "AicpuAllReduceSoleMeshOneShot";
                // [中文导读] [AllReduce逐行 S572] 分支条件为 本 Rank 数据字节数 * Rank 数平方缩放因子 大于 32MiB 分块 TwoShot 门槛；成立进入本块，未成立继续后续分支。
                } else if (dataSize * ratio > AR_AICPU_1D_MAX_DATA_SIZE) {
                    // [中文导读] [AllReduce逐行 S573] 写回候选算法名：算法名输出参数 = "AicpuAllReduceSoleMeshChunkTwoShot"；此处只选择注册名，执行资源在后续流程申请。
                    selectAlgName = "AicpuAllReduceSoleMeshChunkTwoShot";
                // [中文导读] [AllReduce逐行 S574] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
                } else {
                    // [中文导读] [AllReduce逐行 S575] 写回候选算法名：算法名输出参数 = "AicpuAllReduceSoleMeshTwoShot"；此处只选择注册名，执行资源在后续流程申请。
                    selectAlgName = "AicpuAllReduceSoleMeshTwoShot";
                // [中文导读] [AllReduce逐行 S576] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
                }
            // [中文导读] [AllReduce逐行 S577] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
            } else {
                // [中文导读] [AllReduce逐行 S578] 分支条件为 64 位类型或 PROD 的软件归约标志；成立进入本块，未成立继续后续分支。
                if (isDataTypeOrReduceTypeSpecial) {
                    // [中文导读] [AllReduce逐行 S579] 写回候选算法名：算法名输出参数 = "AicpuAllReduceSequenceMeshNHRAicpuReduce"；此处只选择注册名，执行资源在后续流程申请。
                    selectAlgName = "AicpuAllReduceSequenceMeshNHRAicpuReduce";
                // [中文导读] [AllReduce逐行 S580] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
                } else {
                    // [中文导读] [AllReduce逐行 S581] 写回候选算法名：算法名输出参数 = (本 Rank 数据字节数 小于 OMNI_PCIE_AR_DATA_SIZE) ? "AicpuAllReduceParallelMeshNHR" :；此处只选择注册名，执行资源在后续流程申请。
                    selectAlgName = (dataSize < OMNI_PCIE_AR_DATA_SIZE) ? "AicpuAllReduceParallelMeshNHR" :
                                                                          // [中文导读] [AllReduce逐行 S582] 续接本次调用的字符串常量 "AicpuAllReducePipeLineMeshNHR";，由所在注册/日志/条件语句整体使用。
                                                                          "AicpuAllReducePipeLineMeshNHR";
                // [中文导读] [AllReduce逐行 S583] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
                }
            // [中文导读] [AllReduce逐行 S584] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
            }
        // [中文导读] [AllReduce逐行 S585] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
        } else {
            // [中文导读] [AllReduce逐行 S586] 直接返回 选择 AICPU UBX Mesh/Clos 混合算法 的结果，调用者取得本分支结果。
            return SelectMeshAlgoAicpuUBX(topoInfo, opParam, dataSize, selectAlgName, isDataTypeOrReduceTypeSpecial);
        // [中文导读] [AllReduce逐行 S587] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
    // [中文导读] [AllReduce逐行 S588] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
    } else {
        // [中文导读] [AllReduce逐行 S589] 开始 HCCL_ERROR 诊断输出，记录 SelectMeshAlgoAicpu 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_ERROR("[AllReduceAutoSelector] topo not match");
        // [中文导读] [AllReduce逐行 S590] 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。
        return SelectorStatus::NOT_MATCH;
    // [中文导读] [AllReduce逐行 S591] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // [中文导读] [AllReduce逐行 S593] 开始 HCCL_DEBUG 诊断输出，记录 SelectMeshAlgoAicpu 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_DEBUG("[AllReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    // [中文导读] [AllReduce逐行 S594] 返回 MATCH，已写入候选算法名，旧选择器结束本次优先级尝试。
    return SelectorStatus::MATCH;
// [中文导读] [AllReduce逐行 S595] 结束 SelectMeshAlgoAicpu 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S597] 定义 SelectAivAlgo 入口：AIV 候选门槛及 OneShot/TwoShot 选择：按网络、strict、运算、类型、Rank、CCL 容量和数据量过滤。
SelectorStatus AllReduceAutoSelector::SelectAivAlgo(
    // [中文导读] [AllReduce逐行 S598] 续接 SelectAivAlgo 的入口参数/基类初始化：const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,；引用参数按声明的 const 限制读写。
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
    // [中文导读] [AllReduce逐行 S599] 续接 SelectAivAlgo 的入口参数/基类初始化：const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& 算法名输出参数) const；引用参数按声明的 const 限制读写。
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName) const
// [中文导读] [AllReduce逐行 S600] 进入 SelectAivAlgo 的实现作用域；AIV 候选门槛及 OneShot/TwoShot 选择：按网络、strict、运算、类型、Rank、CCL 容量和数据量过滤。
{
    // [中文导读] [AllReduce逐行 S601] 显式标记 configAlgMap 在此兼容/default 分支未使用，避免编译器未使用参数警告。
    (void)configAlgMap;
    // [中文导读] [AllReduce逐行 S602] 开始 HCCL_DEBUG 诊断输出，记录 SelectAivAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_DEBUG("[Algo][AllReduceAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);

    // [中文导读] [AllReduce逐行 S604] 分支条件为 topoInfo->level2UbRtp；成立进入本块，未成立继续后续分支。
    if (topoInfo->level2UbRtp) {
        // [中文导读] [AllReduce逐行 S605] 调用 HCCL_AIV_NOT_MATCH_LOG 完成当前参数所指的子步骤；本行实参为 HCCL_AIV_NOT_MATCH_LOG(。
        HCCL_AIV_NOT_MATCH_LOG(
            // [中文导读] [AllReduce逐行 S606] 续接 SelectAivAlgo 当前语句的具体实参/字段：opParam, HCCL_DEBUG, "[AllReduceAutoSelector][%s] aiv is not supported with level2UbRtp, reset to default.",；由其完整表达式完成参数组装、检查或结果写回。
            opParam, HCCL_DEBUG, "[AllReduceAutoSelector][%s] aiv is not supported with level2UbRtp, reset to default.",
            // [中文导读] [AllReduce逐行 S607] 续接 SelectAivAlgo 当前语句的具体实参/字段：__func__)；由其完整表达式完成参数组装、检查或结果写回。
            __func__);
        // [中文导读] [AllReduce逐行 S608] 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。
        return SelectorStatus::NOT_MATCH;
    // [中文导读] [AllReduce逐行 S609] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // [中文导读] [AllReduce逐行 S611] 分支条件为 topoInfo->topoLevelNums 至少 TOPO_LEVEL_NUM_3；成立进入本块，未成立继续后续分支。
    if (topoInfo->topoLevelNums >= TOPO_LEVEL_NUM_3) {
        // [中文导读] [AllReduce逐行 S612] 调用 HCCL_AIV_NOT_MATCH_LOG 完成当前参数所指的子步骤；本行实参为 HCCL_AIV_NOT_MATCH_LOG(。
        HCCL_AIV_NOT_MATCH_LOG(
            // [中文导读] [AllReduce逐行 S613] 续接 SelectAivAlgo 当前语句的具体实参/字段：opParam, HCCL_DEBUG,；由其完整表达式完成参数组装、检查或结果写回。
            opParam, HCCL_DEBUG,
            // [中文导读] [AllReduce逐行 S614] 续接本次调用的字符串常量 "[AllReduceAutoSelector][%s] aiv is not supported when topoLevelNums >= 3(levelNum[%u]), reset to default.",，由所在注册/日志/条件语句整体使用。
            "[AllReduceAutoSelector][%s] aiv is not supported when topoLevelNums >= 3(levelNum[%u]), reset to default.",
            // [中文导读] [AllReduce逐行 S615] 续接 SelectAivAlgo 当前语句的具体实参/字段：__func__, topoInfo->topoLevelNums)；由其完整表达式完成参数组装、检查或结果写回。
            __func__, topoInfo->topoLevelNums);
        // [中文导读] [AllReduce逐行 S616] 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。
        return SelectorStatus::NOT_MATCH;
    // [中文导读] [AllReduce逐行 S617] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // 保序模式不支持AIV，需要回退到AICPU
    // [中文导读] [AllReduce逐行 S620] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S621] 调用 IsNeedStrictModeForOrderPreserved 完成当前参数所指的子步骤；本行实参为 IsNeedStrictModeForOrderPreserved(opParam, 通信域 Rank 总数),。
        IsNeedStrictModeForOrderPreserved(opParam, topoInfo->userRankSize),
        // [中文导读] [AllReduce逐行 S622] 调用 HCCL_AIV_NOT_MATCH_LOG 完成当前参数所指的子步骤；本行实参为 HCCL_AIV_NOT_MATCH_LOG(。
        HCCL_AIV_NOT_MATCH_LOG(
            // [中文导读] [AllReduce逐行 S623] 续接本次错误检查/子调用实参：opParam, HCCL_DEBUG；返回行为由所在完整宏决定。
            opParam, HCCL_DEBUG,
            // [中文导读] [AllReduce逐行 S624] 续接本次调用的字符串常量 "[Algo][AllReduceAutoSelector] DETERMINISTIC_STRICT mode is not supported yet for AIV mode."),，由所在注册/日志/条件语句整体使用。
            "[Algo][AllReduceAutoSelector] DETERMINISTIC_STRICT mode is not supported yet for AIV mode."),
        // [中文导读] [AllReduce逐行 S625] 续接本次错误检查/子调用实参：当前选择器不匹配；返回行为由所在完整宏决定。
        SelectorStatus::NOT_MATCH);

    // aiv 模式不支持 PROD
    // [中文导读] [AllReduce逐行 S628] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S629] 续接本次错误检查/子调用实参：归约运算 等于 HcclReduceOp::乘积归约；返回行为由所在完整宏决定。
        opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD,
        // [中文导读] [AllReduce逐行 S630] 调用 HCCL_AIV_NOT_MATCH_LOG 完成当前参数所指的子步骤；本行实参为 HCCL_AIV_NOT_MATCH_LOG(。
        HCCL_AIV_NOT_MATCH_LOG(
            // [中文导读] [AllReduce逐行 S631] 续接本次错误检查/子调用实参：opParam, HCCL_DEBUG, "[Algo][AllReduceAutoSelector] ReduceOp[%d] is not supported yet for aiv mode."；返回行为由所在完整宏决定。
            opParam, HCCL_DEBUG, "[Algo][AllReduceAutoSelector] ReduceOp[%d] is not supported yet for aiv mode.",
            // [中文导读] [AllReduce逐行 S632] 续接本次错误检查/子调用实参：归约运算；返回行为由所在完整宏决定。
            opParam.reduceType),
        // [中文导读] [AllReduce逐行 S633] 续接本次错误检查/子调用实参：当前选择器不匹配；返回行为由所在完整宏决定。
        SelectorStatus::NOT_MATCH);

    // [中文导读] [AllReduce逐行 S635] 分支条件为 输入元素类型 等于 HcclDataType::无符号 64 位整数；成立进入本块，未成立继续后续分支。
    if (opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_UINT64
        // [中文导读] [AllReduce逐行 S636] 补充同一条件的 或者 子条件：输入元素类型 等于 HcclDataType::FP64 类型。
        || opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_FP64) {
        // [中文导读] [AllReduce逐行 S637] 调用 HCCL_AIV_NOT_MATCH_LOG 完成当前参数所指的子步骤；本行实参为 HCCL_AIV_NOT_MATCH_LOG(opParam, HCCL_DEBUG, "[Algo][AllReduceAutoSelector] aiv mode not support UINT64, FP64.")。
        HCCL_AIV_NOT_MATCH_LOG(opParam, HCCL_DEBUG, "[Algo][AllReduceAutoSelector] aiv mode not support UINT64, FP64.");
        // [中文导读] [AllReduce逐行 S638] 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。
        return SelectorStatus::NOT_MATCH;
    // [中文导读] [AllReduce逐行 S639] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // [中文导读] [AllReduce逐行 S641] 分支条件为 通信域 Rank 总数 大于 MAX_RANK_SIZE；成立进入本块，未成立继续后续分支。
    if (topoInfo->userRankSize > MAX_RANK_SIZE) {
        // [中文导读] [AllReduce逐行 S642] 调用 HCCL_AIV_NOT_MATCH_LOG 完成当前参数所指的子步骤；本行实参为 HCCL_AIV_NOT_MATCH_LOG(。
        HCCL_AIV_NOT_MATCH_LOG(
            // [中文导读] [AllReduce逐行 S643] 续接 SelectAivAlgo 当前语句的具体实参/字段：opParam, HCCL_DEBUG, "[AllReduceAutoSelector] rankSize[%u] larger than [%u]", 通信域 Rank 总数,；由其完整表达式完成参数组装、检查或结果写回。
            opParam, HCCL_DEBUG, "[AllReduceAutoSelector] rankSize[%u] larger than [%u]", topoInfo->userRankSize,
            // [中文导读] [AllReduce逐行 S644] 续接 SelectAivAlgo 当前语句的具体实参/字段：MAX_RANK_SIZE)；由其完整表达式完成参数组装、检查或结果写回。
            MAX_RANK_SIZE);
        // [中文导读] [AllReduce逐行 S645] 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。
        return SelectorStatus::NOT_MATCH;
    // [中文导读] [AllReduce逐行 S646] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // [中文导读] [AllReduce逐行 S648] 声明本阶段局部变量 void* cclBufferAddr，实际值由后续查询/计算填写。
    void* cclBufferAddr;
    // [中文导读] [AllReduce逐行 S649] 声明本阶段局部变量 uint64_t cclBufferSize，实际值由后续查询/计算填写。
    uint64_t cclBufferSize;
    // [中文导读] [AllReduce逐行 S650] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S651] 调用 HcclGetHcclBuffer 完成当前参数所指的子步骤；本行实参为 HcclGetHcclBuffer(opParam.hcclComm, &cclBufferAddr, &cclBufferSize) 不等于 成功状态,。
        HcclGetHcclBuffer(opParam.hcclComm, &cclBufferAddr, &cclBufferSize) != HCCL_SUCCESS,
        // [中文导读] [AllReduce逐行 S652] 调用 HCCL_AIV_NOT_MATCH_LOG 完成当前参数所指的子步骤；本行实参为 HCCL_AIV_NOT_MATCH_LOG(opParam, HCCL_WARNING, "[AllReduceAutoSelector] HcclGetHcclBuffer failed."),。
        HCCL_AIV_NOT_MATCH_LOG(opParam, HCCL_WARNING, "[AllReduceAutoSelector] HcclGetHcclBuffer failed."),
        // [中文导读] [AllReduce逐行 S653] 续接本次错误检查/子调用实参：当前选择器不匹配；返回行为由所在完整宏决定。
        SelectorStatus::NOT_MATCH);
    // [中文导读] [AllReduce逐行 S654] 设置 单元素字节数 为 类型到元素字节数的查找表[输入元素类型]；该值供下方当前分支使用。
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    // [中文导读] [AllReduce逐行 S655] 设置 本 Rank 数据字节数 为 本 Rank 输入元素数 * 单元素字节数；该值供下方当前分支使用。
    u64 dataSize = opParam.DataDes.count * perDataSize;
    // [中文导读] [AllReduce逐行 S656] 分支条件为 当前执行配置 不等于 OpExecuteConfig::AIV_ONLY；成立进入本块，未成立继续后续分支。
    if (opParam.opExecuteConfig != OpExecuteConfig::AIV_ONLY
        // [中文导读] [AllReduce逐行 S657] 补充同一条件的 并且 子条件：本 Rank 数据字节数 至少 AIV_MAX_PER_RANK_DATA_SIZE * 通信域 Rank 总数。
        && dataSize >= AIV_MAX_PER_RANK_DATA_SIZE * topoInfo->userRankSize) {
        // [中文导读] [AllReduce逐行 S658] 开始 HCCL_DEBUG 诊断输出，记录 SelectAivAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_DEBUG(
            // [中文导读] [AllReduce逐行 S659] 续接 SelectAivAlgo 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "[AllReduceAutoSelector][%s] dataSize[%llu] larger than AIV_MAX_PER_RANK_DATA_SIZE[%llu] * rankSize[%u]",
            // [中文导读] [AllReduce逐行 S660] 为 SelectAivAlgo 的诊断/错误宏提供实参：__func__, 本 Rank 数据字节数, AIV_MAX_PER_RANK_DATA_SIZE, 通信域 Rank 总数，与前面的格式占位依次对应。
            __func__, dataSize, AIV_MAX_PER_RANK_DATA_SIZE, topoInfo->userRankSize);
        // [中文导读] [AllReduce逐行 S661] 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。
        return SelectorStatus::NOT_MATCH;
    // [中文导读] [AllReduce逐行 S662] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S663] 分支条件为 本 Rank 数据字节数 大于 cclBufferSize * AIV_MAX_CCL_LOOP_NUM；成立进入本块，未成立继续后续分支。
    if (dataSize > cclBufferSize * AIV_MAX_CCL_LOOP_NUM) {
        // [中文导读] [AllReduce逐行 S664] 调用 HCCL_AIV_NOT_MATCH_LOG 完成当前参数所指的子步骤；本行实参为 HCCL_AIV_NOT_MATCH_LOG(。
        HCCL_AIV_NOT_MATCH_LOG(
            // [中文导读] [AllReduce逐行 S665] 续接 SelectAivAlgo 当前语句的具体实参/字段：opParam, HCCL_DEBUG,；由其完整表达式完成参数组装、检查或结果写回。
            opParam, HCCL_DEBUG,
            // [中文导读] [AllReduce逐行 S666] 续接本次调用的字符串常量 "[AllReduceAutoSelector][%s] dataSize[%llu] too large for cclBufferSize[%llu], maxSupportSize[%llu]",，由所在注册/日志/条件语句整体使用。
            "[AllReduceAutoSelector][%s] dataSize[%llu] too large for cclBufferSize[%llu], maxSupportSize[%llu]",
            // [中文导读] [AllReduce逐行 S667] 续接 SelectAivAlgo 当前语句的具体实参/字段：__func__, 本 Rank 数据字节数, cclBufferSize, cclBufferSize * AIV_MAX_CCL_LOOP_NUM)；由其完整表达式完成参数组装、检查或结果写回。
            __func__, dataSize, cclBufferSize, cclBufferSize * AIV_MAX_CCL_LOOP_NUM);
        // [中文导读] [AllReduce逐行 S668] 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。
        return SelectorStatus::NOT_MATCH;
    // [中文导读] [AllReduce逐行 S669] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // [中文导读] [AllReduce逐行 S671] 分支条件为 第零层拓扑形状 不等于 单层 Mesh1D；成立进入本块，未成立继续后续分支。
    if (topoInfo->level0Topo != Level0Shape::MESH_1D) {
        // [中文导读] [AllReduce逐行 S672] 写回候选算法名：算法名输出参数 = "AivAllReduceSoleMeshTwoShot"；此处只选择注册名，执行资源在后续流程申请。
        selectAlgName = "AivAllReduceSoleMeshTwoShot";
    // [中文导读] [AllReduce逐行 S673] 分支条件为 通信域 Rank 总数 不超过 AR_AIV_BOARD_SIZE；成立进入本块，未成立继续后续分支。
    } else if (topoInfo->userRankSize <= AR_AIV_BOARD_SIZE) {
        // 板内8p场景，按照时延拐点选择算法
        // [中文导读] [AllReduce逐行 S675] 分支条件为 本 Rank 数据字节数 小于 AR_AIV_SMALL_DATA_SIZE_IN_BOARD；成立进入本块，未成立继续后续分支。
        if (dataSize < AR_AIV_SMALL_DATA_SIZE_IN_BOARD) {
            // [中文导读] [AllReduce逐行 S676] 写回候选算法名：算法名输出参数 = "AivAllReduceSoleMeshOneShot"；此处只选择注册名，执行资源在后续流程申请。
            selectAlgName = "AivAllReduceSoleMeshOneShot";
        // [中文导读] [AllReduce逐行 S677] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
        } else {
            // [中文导读] [AllReduce逐行 S678] 写回候选算法名：算法名输出参数 = "AivAllReduceSoleMeshTwoShot"；此处只选择注册名，执行资源在后续流程申请。
            selectAlgName = "AivAllReduceSoleMeshTwoShot";
        // [中文导读] [AllReduce逐行 S679] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
    // [中文导读] [AllReduce逐行 S680] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
    } else {
        // [中文导读] [AllReduce逐行 S681] 分支条件为 IsSmallData(本 Rank 数据字节数；成立进入本块，未成立继续后续分支。
        if (IsSmallData(dataSize)) {
            // [中文导读] [AllReduce逐行 S682] 写回候选算法名：算法名输出参数 = "AivAllReduceSoleMeshOneShot"；此处只选择注册名，执行资源在后续流程申请。
            selectAlgName = "AivAllReduceSoleMeshOneShot";
        // [中文导读] [AllReduce逐行 S683] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
        } else {
            // [中文导读] [AllReduce逐行 S684] 写回候选算法名：算法名输出参数 = "AivAllReduceSoleMeshTwoShot"；此处只选择注册名，执行资源在后续流程申请。
            selectAlgName = "AivAllReduceSoleMeshTwoShot";
        // [中文导读] [AllReduce逐行 S685] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
    // [中文导读] [AllReduce逐行 S686] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // [中文导读] [AllReduce逐行 S688] 开始 HCCL_DEBUG 诊断输出，记录 SelectAivAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_DEBUG("[AllReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    // [中文导读] [AllReduce逐行 S689] 返回 MATCH，已写入候选算法名，旧选择器结束本次优先级尝试。
    return SelectorStatus::MATCH;
// [中文导读] [AllReduce逐行 S690] 结束 SelectAivAlgo 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S692] 定义 SelectDPUAlgo 入口：HostDPU 模式按显式配置及多层拓扑选择 SequenceMeshNHR 或 PipeLineMeshNHRNHR，单层不匹配。
SelectorStatus AllReduceAutoSelector::SelectDPUAlgo(
    // [中文导读] [AllReduce逐行 S693] 续接 SelectDPUAlgo 的入口参数/基类初始化：const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,；引用参数按声明的 const 限制读写。
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
    // [中文导读] [AllReduce逐行 S694] 续接 SelectDPUAlgo 的入口参数/基类初始化：const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& 算法名输出参数) const；引用参数按声明的 const 限制读写。
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName) const
// [中文导读] [AllReduce逐行 S695] 进入 SelectDPUAlgo 的实现作用域；HostDPU 模式按显式配置及多层拓扑选择 SequenceMeshNHR 或 PipeLineMeshNHRNHR，单层不匹配。
{
    // [中文导读] [AllReduce逐行 S696] 建立本阶段局部对象 std::vector<HcclAlgoType> algos，供 SelectDPUAlgo 下方参数组装和子调用使用。
    std::vector<HcclAlgoType> algos
        // [中文导读] [AllReduce逐行 S697] 设置  为 std::vector<HcclAlgoType>(HCCL_ALGO_LEVEL_NUM, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT)；该值供下方当前分支使用。
        = std::vector<HcclAlgoType>(HCCL_ALGO_LEVEL_NUM, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT);
    // [中文导读] [AllReduce逐行 S698] 设置 it 为 configAlgMap.find(opParam.opType)；该值供下方当前分支使用。
    auto it = configAlgMap.find(opParam.opType);
    // [中文导读] [AllReduce逐行 S699] 分支条件为 (it 不等于 configAlgMap.end()) 且 (it->second.size() 大于 1；成立进入本块，未成立继续后续分支。
    if ((it != configAlgMap.end()) && (it->second.size() > 1)) {
        // [中文导读] [AllReduce逐行 S700] 设置 algos 为 it->second；该值供下方当前分支使用。
        algos = it->second;
    // [中文导读] [AllReduce逐行 S701] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // [中文导读] [AllReduce逐行 S703] 开始 HCCL_INFO 诊断输出，记录 SelectDPUAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S704] 续接 SelectDPUAlgo 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
        "hccl algo op config: config opType:%d, level0:%u, level1:%u, level2:%u, level3:%u", opParam.opType, algos[0],
        // [中文导读] [AllReduce逐行 S705] 为 SelectDPUAlgo 的诊断/错误宏提供实参：algos[1], algos[2], algos[3]，与前面的格式占位依次对应。
        algos[1], algos[2], algos[3]);
    // [中文导读] [AllReduce逐行 S706] 分支条件为 topoInfo->topoLevelNums 大于 1；成立进入本块，未成立继续后续分支。
    if (topoInfo->topoLevelNums > 1) {
        // [中文导读] [AllReduce逐行 S707] 分支条件为 (topoInfo->deviceNumPerModule 等于 1) 或 (第零层拓扑形状 等于 单层 Mesh1D；成立进入本块，未成立继续后续分支。
        if ((topoInfo->deviceNumPerModule == 1) || (topoInfo->level0Topo == Level0Shape::MESH_1D)) {
            // [中文导读] [AllReduce逐行 S708] 写回候选算法名：算法名输出参数 = "DpuAllReduceSequenceMeshNHR"; // 对应当前实际执行配置utor最后register的第二个参数；此处只选择注册名，执行资源在后续流程申请。
            selectAlgName = "DpuAllReduceSequenceMeshNHR"; // 对应executor最后register的第二个参数
            // [中文导读] [AllReduce逐行 S709] 开始 HCCL_INFO 诊断输出，记录 SelectDPUAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。
            HCCL_INFO("Using algo DpuAllReduceSequenceMeshNHR");
            // [中文导读] [AllReduce逐行 S710] 返回 MATCH，已写入候选算法名，旧选择器结束本次优先级尝试。
            return SelectorStatus::MATCH;
        // [中文导读] [AllReduce逐行 S711] 分支条件为 第零层拓扑形状 等于 Mesh 与 CLOS 混合拓扑；成立进入本块，未成立继续后续分支。
        } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
            // [中文导读] [AllReduce逐行 S712] 分支条件为 !第零层 PCIe 混合标志；成立进入本块，未成立继续后续分支。
            if (!topoInfo->level0PcieMix) {
                // [中文导读] [AllReduce逐行 S713] 写回候选算法名：算法名输出参数 = "DpuAllReducePipeLineMeshNHRNHR"；此处只选择注册名，执行资源在后续流程申请。
                selectAlgName = "DpuAllReducePipeLineMeshNHRNHR";
                // [中文导读] [AllReduce逐行 S714] 开始 HCCL_INFO 诊断输出，记录 SelectDPUAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。
                HCCL_INFO("Using algo DpuAllReducePipeLineMeshNHRNHR");
                // [中文导读] [AllReduce逐行 S715] 返回 MATCH，已写入候选算法名，旧选择器结束本次优先级尝试。
                return SelectorStatus::MATCH;
            // [中文导读] [AllReduce逐行 S716] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
            } else {
                // [中文导读] [AllReduce逐行 S717] 写回候选算法名：算法名输出参数 = "DpuAllReduceSequenceMeshNHR"；此处只选择注册名，执行资源在后续流程申请。
                selectAlgName = "DpuAllReduceSequenceMeshNHR";
                // [中文导读] [AllReduce逐行 S718] 开始 HCCL_INFO 诊断输出，记录 SelectDPUAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。
                HCCL_INFO("Using algo DpuAllReduceSequenceMeshNHR");
                // [中文导读] [AllReduce逐行 S719] 返回 MATCH，已写入候选算法名，旧选择器结束本次优先级尝试。
                return SelectorStatus::MATCH;
            // [中文导读] [AllReduce逐行 S720] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
            }
        // [中文导读] [AllReduce逐行 S721] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
        } else {
            // [中文导读] [AllReduce逐行 S722] 写回候选算法名：算法名输出参数 = "DpuAllReduceSequenceMeshNHR"；此处只选择注册名，执行资源在后续流程申请。
            selectAlgName = "DpuAllReduceSequenceMeshNHR";
            // [中文导读] [AllReduce逐行 S723] 开始 HCCL_INFO 诊断输出，记录 SelectDPUAlgo 当前阶段的参数、候选或错误；日志本身不决定返回码。
            HCCL_INFO("Using algo DpuAllReduceSequenceMeshNHR");
            // [中文导读] [AllReduce逐行 S724] 返回 MATCH，已写入候选算法名，旧选择器结束本次优先级尝试。
            return SelectorStatus::MATCH;
        // [中文导读] [AllReduce逐行 S725] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
    // [中文导读] [AllReduce逐行 S726] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // [中文导读] [AllReduce逐行 S728] 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。
    return SelectorStatus::NOT_MATCH;
// [中文导读] [AllReduce逐行 S729] 结束 SelectDPUAlgo 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S731] 以优先级18将 AllReduceAutoSelector 注册到 AllReduce 算子的旧选择器表。
REGISTER_SELECTOR_BY_OPTYPE(HcclCMDType::HCCL_CMD_ALLREDUCE, 18, AllReduceAutoSelector);
} // namespace ops_hccl
