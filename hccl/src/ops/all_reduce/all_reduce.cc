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
HcclResult HcclAllReduce(
    void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm,
    aclrtStream stream)
{
    HCCL_INFO("Start to run execute HcclAllReduce");
    // [中文导读] 先按运行时版本分流，旧 HCOMM 使用兼容归约入口。
    if (GetHcommVersion() < CANN_VERSION(9, 0, 0)) { // compat handle
        return HcclAllReduceInner(sendBuf, recvBuf, count, dataType, op, comm, stream);
    }

    // [中文导读] 检查设备的新流程支持情况；不支持时沿原 AllReduce 实现执行。
    bool isOutPlace = false;
    CHK_RET(IsOutPlaceDevice(isOutPlace));
    if (!isOutPlace) {
        return HcclAllReduceInner(sendBuf, recvBuf, count, dataType, op, comm, stream);
    }
    // [中文导读] 零元素归约不展开通信；正常非零调用才进入环境和参数检查。
    CHK_PRT_RET(count == 0, HCCL_WARNING("input count is 0, return all reduce success"), HCCL_SUCCESS);

    HcclUs startut = TIME_NOW(); // 走老流程的判断时间不统计在内
    OpParam param;
    // [中文导读] 建立算子 tag 并验证数量、数据类型和归约运算组合，防止非法归约进入算法层。
    CHK_RET(AllReduceInitAndCheck(comm, sendBuf, recvBuf, count, dataType, op, stream, param));

    /* 接口交互信息日志 */
    CHK_RET(AllReduceEntryLog(sendBuf, recvBuf, count, dataType, op, stream, param.tag, "HcclAllReduce"));

    // 执行AllReduce
    // [中文导读] 下传已校验的归约参数，实际集合通信任务由公共调度和选中算法生成。
    CHK_RET_AND_PRINT_IDE(AllReduceOutPlace(sendBuf, recvBuf, count, dataType, op, comm, stream, param), param.tag);

    CHK_RET(LogHcclExit("HcclAllReduce", param.tag, startut));

    return HCCL_SUCCESS;
}

HcclResult HcclAllReduceGraphMode(
    void* sendBuf, void* recvBuf, uint64_t sendCount, HcclDataType dataType, HcclReduceOp op, const char* group,
    aclrtStream stream, const char* tag, void** streams, size_t streamCount, void* scratchMemAddr,
    uint64_t scratchMemSize)
{
    HCCL_INFO("Start to run execute HcclAllReduceGraphMode");
    // 根据group获取通信域
    // [中文导读] 根据图模式的 group 定位通信域，沿用同一套归约参数检查。
    CHK_PTR_NULL(group);
    HcclComm comm = nullptr;
    HCCL_INFO("[HcclAllReduceGraphMode] get group name: %s", group);
    CHK_RET(HcomGetCommHandleByGroup(group, &comm));
    CHK_PRT_RET(sendCount == 0, HCCL_WARNING("input sendCount is 0, return all reduce success"), HCCL_SUCCESS);

    HcclUs startut = TIME_NOW(); // 走老流程的判断时间不统计在内
    OpParam param;
    CHK_RET(AllReduceInitAndCheck(comm, sendBuf, recvBuf, sendCount, dataType, op, stream, param));

    // 检查tag有效性
    // [中文导读] 图执行提供独立 tag，校验后覆盖初始化生成的默认算子标识。
    CHK_RET(HcclCheckTag(tag));
    int ret = sprintf_s(param.tag, sizeof(param.tag), "%s", tag);
    CHK_PRT_RET((ret <= 0), HCCL_ERROR("failed to fill param.tag"), HCCL_E_INTERNAL);

    // 拼装ResPackGraphMode
    // [中文导读] 把图执行的外部从流和临时内存组织成资源包，随 OFFLOAD 模式传给公共执行层。
    ResPackGraphMode resPack;
    // 设置tag
    if (strncpy_s(resPack.tag, sizeof(resPack.tag), tag, sizeof(resPack.tag) - 1) != 0) {
        HCCL_ERROR("failed to fill resPack.tag");
        return HCCL_E_INTERNAL;
    }
    // 设置streams
    if (streams != nullptr && streamCount > 0) {
        for (size_t i = 0; i < streamCount; i++) {
            resPack.streams.push_back(static_cast<aclrtStream>(streams[i]));
        }
    }
    // 设置scratchMem
    resPack.scratchMemAddr = scratchMemAddr;
    resPack.scratchMemSize = scratchMemSize;
    std::string tagStr = tag;

    /* 接口交互信息日志 */
    CHK_RET(AllReduceEntryLog(
        sendBuf, recvBuf, sendCount, dataType, op, stream, param.tag, "HcclAllReduceGraphMode", true));
    // 执行AllReduce
    CHK_RET_AND_PRINT_IDE(
        AllReduceOutPlaceGraphMode(sendBuf, recvBuf, sendCount, dataType, op, comm, stream, resPack, param),
        tagStr.c_str());
    CHK_RET(LogHcclExit("HcclAllReduceGraphMode", param.tag, startut, true));

    return HCCL_SUCCESS;
}

namespace ops_hccl {
HcclResult AllReduceInitAndCheck(
    HcclComm comm, void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
    const aclrtStream stream, OpParam& param)
{
    // 入口的地方先解析环境变量，在初始化环境变量的时候需要设置为AICPU展开
    // [中文导读] 读取环境配置，再验证指针和通信域信息，保证后续调度拥有有效基础参数。
    CHK_RET(InitEnvConfig());

    // 参数校验等工作
    CHK_RET(CheckAllReduceInputPara(comm, sendBuf, recvBuf, stream));
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET(HcclGetRankId(comm, &userRank));
    CHK_RET(HcclGetCommName(comm, param.commName));

    // topoInfo的tag，所有相同的算子可以共享
    // [中文导读] 以通信域名称生成 AllReduce tag，供拓扑、资源上下文与错误日志关联使用。
    int ret = sprintf_s(param.tag, sizeof(param.tag), "AllReduce_%s", param.commName);
    CHK_PRT_RET((ret <= 0), "failed to fill param.tag", HCCL_E_INTERNAL);

    CHK_RET(HcclCheckTag(param.tag));
    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, userRank), param.tag);
    // [中文导读] 依次检查元素数量、可归约的数据类型以及该类型支持的归约运算。
    CHK_RET(CheckCount(count));
    CHK_RET(CheckDataType(dataType, true));
    CHK_RET(CheckReduceOp(dataType, op));

    return HCCL_SUCCESS;
}

HcclResult
CheckAllReduceInputPara(const HcclComm comm, const void* sendBuf, const void* recvBuf, const aclrtStream stream)
{
    // 入参合法性校验
    RPT_INPUT_ERR(
        stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclAllReduce", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);
    RPT_INPUT_ERR(
        comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclAllReduce", "nullptr", "comm", "non-null pointer"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(
        sendBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclAllReduce", "nullptr", "sendBuf", "non-null pointer"}));
    CHK_PTR_NULL(sendBuf);
    RPT_INPUT_ERR(
        recvBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclAllReduce", "nullptr", "recvBuf", "non-null pointer"}));
    CHK_PTR_NULL(recvBuf);

    return HCCL_SUCCESS;
}

HcclResult FillAllReduceOpParam(
    void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, const HcclComm comm,
    aclrtStream stream, OpMode opMode, OpParam& param)
{
    // [中文导读] 归约前后元素数量相同，因此输入和输出使用相同的字节容量。
    u32 perDataSize = DATATYPE_SIZE_TABLE[dataType];
    u64 outputSize = count * perDataSize;
    u64 inputSize = outputSize;

    // [中文导读] 保存流、通信域和归约模式，并先关闭对称内存标志，后面按实际条件重新开启。
    param.hcclComm = comm;
    param.stream = stream;
    param.reduceType = op;
    param.opMode = opMode;
    param.supportSymmetricMemory = false;

    HcclDevType deviceType = HcclDevType::DEV_TYPE_COUNT;
    CHK_RET(HcclGetDeviceType(deviceType));

    // 参数准备
    // [中文导读] 同时记录字节容量和 count/dataType 元素描述；底层归约原语使用元素数量。
    param.inputPtr = sendBuf;
    param.inputSize = inputSize;
    param.outputPtr = recvBuf;
    param.outputSize = outputSize;
    param.DataDes.count = count;
    param.DataDes.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    param.enableDetour = false;
    param.deviceType = deviceType;
    param.reduceType = op;
    return HCCL_SUCCESS;
}

HcclResult AllReduceOutPlaceCommon(
    void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm,
    aclrtStream stream, OpMode opMode, const ResPackGraphMode& resPack, OpParam& param)
{
    HCCL_INFO("Start to execute AllReduceOutPlace");

    // [中文导读] 为单算子或图模式补齐同一份 OpParam，再从通信域取得展开方式。
    CHK_RET(FillAllReduceOpParam(sendBuf, recvBuf, count, dataType, op, comm, stream, opMode, param));

    CHK_RET(HcclGetOpExpansionMode(comm, param));

    // 9.0.0 ccu模式走老流程
    // [中文导读] 9.0.0 的单算子 CCU 调度回退到兼容实现，以避开当前新资源路径。
    if (opMode == OpMode::OPBASE && GetHcommVersion() == CANN_VERSION(9, 0, 0)
        && param.engine == CommEngine::COMM_ENGINE_CCU) {
        return HcclAllReduceInner(sendBuf, recvBuf, count, dataType, op, comm, stream);
    }

    // [中文导读] 单算子先尝试复用 CCU 快速发射信息，命中后跳过常规拓扑和资源准备。
    CcuFastLaunchCtx* ccuFastLaunchCtx = nullptr;
    if ((opMode == OpMode::OPBASE) && ShouldGoCcuFastLaunch(comm, param, &ccuFastLaunchCtx)) {
        return HcclExecOpCcuFastLaunch(comm, param, ccuFastLaunchCtx);
    }

    // [中文导读] AIV 缓存回放成功时提前返回；未命中才运行下面的单 Rank 或 selector 分支。
    if (param.engine == CommEngine::COMM_ENGINE_AIV) {
        bool aivCacheHit = false;
        CHK_RET(HcclAivCacheCheckAndReplay(comm, param, aivCacheHit));
        if (aivCacheHit) {
            return HCCL_SUCCESS;
        }
    }

    // 单卡校验
    // [中文导读] 从通信域取得规模；只有一个 Rank 时由本地处理完成归约语义。
    u32 userRankSize;
    CHK_RET(HcclGetRankSize(comm, &userRankSize));
    if (userRankSize == 1) {
        HCCL_WARNING("[%s] ranksize == 1, enter SingleRankProc", __func__);
        CHK_RET(SingleRankProc(comm, param));
        return HcclResult::HCCL_SUCCESS;
    }

    // [中文导读] 运行时版本和单算子模式允许时，先探测输入输出是否满足对称内存条件。
    if (GetHcommVersion() >= CANN_VERSION(9, 1, 0) && param.opMode == OpMode::OPBASE) {
        CheckAndSetSymmetricMemory(param);
    }

    // [中文导读] 依据拓扑和归约参数选择算法，再用选择结果收紧对称内存支持范围。
    std::string algName;
    std::unique_ptr<TopoInfoWithNetLayerDetails> topoInfo = std::make_unique<TopoInfoWithNetLayerDetails>();
    CHK_RET(Selector(comm, param, topoInfo, algName));

    // [中文导读] Mesh CLOS 只保留特定两层 Omni 算法的对称内存路径，其余情况清除候选标志。
    if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix) {
        const bool isTwoLevelMeshNhrOmni
            = algName == "AicpuAllReducePipeLineMeshNHR" && topoInfo->topoLevelNums == TOPO_LEVEL_NUM_1;
        if (!isTwoLevelMeshNhrOmni) {
            param.supportSymmetricMemory = false;
        }
    } else {
        // [中文导读] 普通 Mesh 还要求 AICPU_TS，并排除这里列出的 64 位类型及乘积归约。
        const bool supportSymmetricMemory
            = (param.engine == CommEngine::COMM_ENGINE_AICPU_TS && topoInfo->level0Topo == Level0Shape::MESH_1D
               && param.DataDes.dataType != HcclDataType::HCCL_DATA_TYPE_INT64
               && param.DataDes.dataType != HcclDataType::HCCL_DATA_TYPE_UINT64
               && param.DataDes.dataType != HcclDataType::HCCL_DATA_TYPE_FP64
               && param.reduceType != HcclReduceOp::HCCL_REDUCE_PROD);
        if (!supportSymmetricMemory) {
            param.supportSymmetricMemory = false;
        }
    }

    // [中文导读] 把最终算法与归约配置送入公共执行层；支持标志已经按拓扑和运算条件过滤。
    CHK_RET(HcclExecOp(comm, param, topoInfo, algName, resPack));
    HCCL_INFO("Execute AllReduceOutPlace success.");
    return HCCL_SUCCESS;
}

HcclResult AllReduceEntryLog(
    void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, aclrtStream stream,
    const char* tag, const std::string& opName, bool forceLog)
{
    if (forceLog || GetExternalInputHcclEnableEntryLog()) {
        s32 deviceId = 0;
        ACLCHECK(aclrtGetDevice(&deviceId));
        s32 streamId = 0;
        ACLCHECK(aclrtStreamGetId(stream, &streamId));
        char stackLogBuffer[LOG_TMPBUF_SIZE];
        s32 ret = snprintf_s(
            stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], sendBuf[%p], recvBuf[%p], count[%llu], dataType[%s], reduceOp[%s], streamId[%d], deviceId[%d]",
            tag, sendBuf, recvBuf, count, GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str(),
            streamId, deviceId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag));
        std::string logInfo = "Entry-" + opName + ":" + std::string(stackLogBuffer);
        HCCL_RUN_INFO("%s", logInfo.c_str());
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceOutPlaceGraphMode(
    void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm,
    aclrtStream stream, const ResPackGraphMode& resPack, OpParam& param)
{
    HCCL_INFO("Start to execute AllReduceOutPlaceGraphMode");
    CHK_RET(
        AllReduceOutPlaceCommon(sendBuf, recvBuf, count, dataType, op, comm, stream, OpMode::OFFLOAD, resPack, param));
    HCCL_INFO("Execute AllReduceOutPlaceGraphMode success.");
    return HCCL_SUCCESS;
}

HcclResult AllReduceOutPlace(
    void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm,
    aclrtStream stream, OpParam& param)
{
    HCCL_INFO("Start to execute AllReduceOutPlace");
    CHK_RET(AllReduceOutPlaceCommon(
        sendBuf, recvBuf, count, dataType, op, comm, stream, OpMode::OPBASE, ResPackGraphMode(), param));
    HCCL_INFO("Execute AllReduceOutPlace success.");
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
