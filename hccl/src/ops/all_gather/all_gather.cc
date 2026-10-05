/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather.h"
#include "op_common_ops.h"
#include <algorithm>
#include <future>
#include <map>
#include <string>

using namespace std;
using namespace ops_hccl;
extern "C" unsigned int LaunchAicpuKernel(OpParam* param);

// [中文导读] AllGather把各Rank的等长输入收集到每个Rank的输出，sendCount是本Rank的元素数量。
// [中文导读] 本入口负责版本/设备分流与参数检查，实际算法位于AllGatherOutPlace之后的调度和模板。
// [中文导读] 与AllToAll共用资源框架不表示数据切片和原语相同；本算子没有归约运算参数。
HcclResult HcclAllGather(
    void* sendBuf, void* recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    HCCL_INFO("Start to run execute HcclAllGather");

    // [中文导读] 先以运行时 HCOMM 版本判断新接口是否可用；旧版本直接交兼容实现。
    if (GetHcommVersion() < CANN_VERSION(9, 0, 0)) { // compat handle
        return HcclAllGatherInner(sendBuf, recvBuf, sendCount, dataType, comm, stream);
    }

    // [中文导读] 再检查设备是否支持新流程，只有通过这两项门槛才继续新入口的参数准备。
    bool isOutPlace = false;
    CHK_RET(IsOutPlaceDevice(isOutPlace));
    if (!isOutPlace) {
        return HcclAllGatherInner(sendBuf, recvBuf, sendCount, dataType, comm, stream);
    }
    // [中文导读] 空输入直接返回成功，不再生成通信任务；此判断位于后续指针校验之前。
    CHK_PRT_RET(sendCount == 0, HCCL_WARNING("input sendCount is 0, return all gather success"), HCCL_SUCCESS);

    HcclUs startut = TIME_NOW(); // 走老流程的判断时间不统计在内
    std::string opTag;
    // [中文导读] 建立通信域关联的算子 tag，并检查缓冲区、元素数量、类型和本 Rank 身份。
    CHK_RET(AllGatherInitAndCheck(comm, sendBuf, recvBuf, sendCount, dataType, stream, opTag));

    CHK_RET(AllGatherEntryLog(sendBuf, recvBuf, sendCount, dataType, stream, opTag, "HcclAllGather"));

    // 执行AllGather
    // [中文导读] 把已校验的调用交给单算子调度；失败时用 tag 关联错误信息。
    CHK_RET_AND_PRINT_IDE(AllGatherOutPlace(sendBuf, recvBuf, sendCount, dataType, comm, stream, opTag), opTag.c_str());

    CHK_RET(LogHcclExit("HcclAllGather", opTag.c_str(), startut));

    return HCCL_SUCCESS;
}

HcclResult HcclAllGatherGraphMode(
    void* sendBuf, void* recvBuf, uint64_t sendCount, HcclDataType dataType, const char* group, aclrtStream stream,
    const char* tag, void** streams, size_t streamCount, void* scratchMemAddr, uint64_t scratchMemSize)
{
    HCCL_INFO("Start to run execute HcclAllGatherGraphMode");
    // 根据group获取通信域
    // [中文导读] 图模式用 group 查出已有通信域，后面的公共调度仍使用 HcclComm 句柄。
    CHK_PTR_NULL(group);
    HcclComm comm = nullptr;
    HCCL_INFO("[HcclAllGatherGraphMode] get group name: %s", group);
    CHK_RET(HcomGetCommHandleByGroup(group, &comm));
    CHK_PRT_RET(sendCount == 0, HCCL_WARNING("input sendCount is 0, return all gather success"), HCCL_SUCCESS);

    HcclUs startut = TIME_NOW(); // 走老流程的判断时间不统计在内
    std::string opTag;
    CHK_RET(AllGatherInitAndCheck(comm, sendBuf, recvBuf, sendCount, dataType, stream, opTag));

    // 检查tag有效性
    CHK_RET(HcclCheckTag(tag));

    // 拼装ResPackGraphMode
    // [中文导读] 整理图执行提供的 tag、从流及 scratch 内存，供资源层包装和复用。
    ResPackGraphMode resPack;
    // 设置tag
    if (strncpy_s(resPack.tag, sizeof(resPack.tag), tag, sizeof(resPack.tag) - 1) != 0) {
        HCCL_ERROR("failed to fill resPack.tag");
        return HCCL_E_INTERNAL;
    }
    // 设置streams
    // [中文导读] 仅把有效的外部从流加入资源包；这里不创建这些流。
    if (streams != nullptr && streamCount > 0) {
        for (size_t i = 0; i < streamCount; i++) {
            resPack.streams.push_back(static_cast<aclrtStream>(streams[i]));
        }
    }
    std::string tagStr = tag;
    // 设置scratchMem
    // [中文导读] 保留图执行给出的临时内存地址和容量，稍后随 OFFLOAD 调用下传。
    resPack.scratchMemAddr = scratchMemAddr;
    resPack.scratchMemSize = scratchMemSize;

    CHK_RET(AllGatherEntryLog(sendBuf, recvBuf, sendCount, dataType, stream, opTag, "HcclAllGatherGraphMode", true));

    // 执行AllGather
    CHK_RET_AND_PRINT_IDE(
        AllGatherOutPlaceGraphMode(sendBuf, recvBuf, sendCount, dataType, comm, stream, tagStr, resPack),
        tagStr.c_str());

    CHK_RET(LogHcclExit("HcclAllGatherGraphMode", opTag.c_str(), startut, true));

    return HCCL_SUCCESS;
}
namespace ops_hccl {

HcclResult AllGatherInitAndCheck(
    HcclComm comm, void* sendBuf, void* recvBuf, uint64_t sendCount, HcclDataType dataType, aclrtStream stream,
    std::string& opTag)
{
    // 入口的地方先解析环境变量，在初始化环境变量的时候需要设置为AICPU展开
    // [中文导读] 先解析环境配置，为后续展开模式和算法选择提供本进程配置。
    CHK_RET(InitEnvConfig());
    // 参数校验等工作
    // 检查入参指针有效性
    CHK_RET(CheckAllGatherInputPara(comm, sendBuf, recvBuf, stream));
    // tag有效性,是否过长
    // [中文导读] 以通信域名称构造 AllGather tag，使同类算子的拓扑查询可关联同一标识。
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    opTag = "AllGather_" + string(commName);
    CHK_RET(HcclCheckTag(opTag.c_str()));
    // 检查sendCount是否合法(超出系统上限)
    // [中文导读] 校验的是每个 Rank 的元素数量和数据类型，输出总量在公共参数层计算。
    CHK_RET(CheckCount(sendCount));
    // 检查数据类型是否支持
    CHK_RET(CheckDataType(dataType, false));
    // 检查rank有效性，是否超出rankSize
    // [中文导读] 读取通信域规模和本端 Rank，再确认 Rank 编号落在该通信域范围内。
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET(HcclGetRankId(comm, &userRank));
    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, userRank), opTag.c_str());
    return HCCL_SUCCESS;
}

HcclResult
CheckAllGatherInputPara(const HcclComm comm, const void* sendBuf, const void* recvBuf, const aclrtStream stream)
{
    // 入参合法性校验
    RPT_INPUT_ERR(
        stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclAllGather", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);
    RPT_INPUT_ERR(
        comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclAllGather", "nullptr", "comm", "non-null pointer"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(
        sendBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclAllGather", "nullptr", "sendBuf", "non-null pointer"}));
    CHK_PTR_NULL(sendBuf);
    RPT_INPUT_ERR(
        recvBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclAllGather", "nullptr", "recvBuf", "non-null pointer"}));
    CHK_PTR_NULL(recvBuf);

    return HCCL_SUCCESS;
}

HcclResult AllGatherOutPlaceCommon(
    void* sendBuf, void* recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream,
    const std::string& tag, OpMode opMode, const ResPackGraphMode& resPack)
{
    HCCL_INFO("Start to execute AllGatherOutPlaceCommon");
    u32 userRankSize;
    CHK_RET(HcclGetRankSize(comm, &userRankSize));

    // [中文导读] 把本端元素数换算为输入字节数；输出容量按 Rank 数扩展，以容纳所有 Rank 的输入。
    u32 perDataSize = DATATYPE_SIZE_TABLE[dataType];
    u64 inputSize = sendCount * perDataSize;   // all gather 每个rank上一份数据
    u64 outputSize = inputSize * userRankSize; // 每个卡上结果为rankSize份数据

    OpParam param;
    CHK_RET(HcclGetCommName(comm, param.commName));
    param.stream = stream;
    param.opMode = opMode;

    HcclDevType deviceType = HcclDevType::DEV_TYPE_COUNT;
    CHK_RET(HcclGetDeviceType(deviceType));

    // topoInfo的tag，所有相同的算子可以共享
    int ret = sprintf_s(param.tag, sizeof(param.tag), "%s", tag.c_str());
    if (ret <= 0) {
        HCCL_ERROR("failed to fill param.tag");
        return HCCL_E_INTERNAL;
    }

    // 参数准备
    // [中文导读] 统一记录用户地址、字节容量和元素描述，让 selector 与 executor 使用同一份参数。
    param.inputPtr = sendBuf;
    param.inputSize = inputSize;
    param.outputPtr = recvBuf;
    param.outputSize = outputSize;
    param.DataDes.count = sendCount;
    param.DataDes.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLGATHER;
    param.enableDetour = false;
    param.deviceType = deviceType;

    // [中文导读] 根据通信域与配置确定执行引擎，以下快速路径均以这个结果为前提。
    CHK_RET(HcclGetOpExpansionMode(comm, param));

    // 9.0.0 ccu模式走老流程
    // [中文导读] 9.0.0 的单算子 CCU 分支显式回到兼容实现；版本门槛与设备门槛各自独立。
    if (opMode == OpMode::OPBASE && GetHcommVersion() == CANN_VERSION(9, 0, 0)
        && param.engine == CommEngine::COMM_ENGINE_CCU) {
        return HcclAllGatherInner(sendBuf, recvBuf, sendCount, dataType, comm, stream);
    }

    // [中文导读] 先尝试已有 CCU 快速发射上下文，命中后直接发射，无需本次常规选算法流程。
    CcuFastLaunchCtx* ccuFastLaunchCtx = nullptr;
    if (ShouldGoCcuFastLaunch(comm, param, &ccuFastLaunchCtx)) {
        return HcclExecOpCcuFastLaunch(comm, param, ccuFastLaunchCtx);
    }

    // [中文导读] AIV 任务缓存命中时直接回放本次调用；未命中才继续资源与算法准备。
    if (param.engine == CommEngine::COMM_ENGINE_AIV) {
        bool aivCacheHit = false;
        CHK_RET(HcclAivCacheCheckAndReplay(comm, param, aivCacheHit));
        if (aivCacheHit) {
            HCCL_INFO("Execute AllGatherOutPlace success (AIV cache hit).");
            return HCCL_SUCCESS;
        }
    }

    // [中文导读] 单 Rank 走本地处理，无需建立多 Rank 集合通信算法。
    if (userRankSize == 1) {
        HCCL_WARNING("[%s] rankSize == 1, enter SingleRankProc", __func__);
        CHK_RET(SingleRankProc(comm, param));
        return HcclResult::HCCL_SUCCESS;
    }
    // [中文导读] 为多 Rank 构造拓扑对象，由 Selector 同时给出匹配拓扑与具体算法名。
    std::string algName;
    std::unique_ptr<TopoInfoWithNetLayerDetails> topoInfo = std::make_unique<TopoInfoWithNetLayerDetails>();
    CHK_RET(Selector(comm, param, topoInfo, algName));

    // 单个 MESH_1D_CLOS 网络层在 Omni executor 内展开为 Mesh+NHR 两层；
    // 额外网络层会继续展开出 DPU 第三层，因此不能进入当前对称内存路径。
    // [中文导读] 只把这里列出的 Mesh 或两层 Mesh+NHR 条件纳入对称内存候选，额外拓扑层不适用。
    const bool isTwoLevelMeshNhrOmni = algName == "AicpuAllGatherPipeLineMeshNHR"
                                       && topoInfo->topoLevelNums == TOPO_LEVEL_NUM_1
                                       && topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix;
    if (GetHcommVersion() >= CANN_VERSION(9, 1, 0) && param.opMode == OpMode::OPBASE
        && param.engine == CommEngine::COMM_ENGINE_AICPU_TS
        && (topoInfo->level0Topo == Level0Shape::MESH_1D || isTwoLevelMeshNhrOmni)) {
        CheckAndSetSymmetricMemory(param);
    }
    // [中文导读] 最终把选定算法和图资源包交给公共执行枢纽，后者负责取得资源并按引擎展开。
    CHK_RET(HcclExecOp(comm, param, topoInfo, algName, resPack));
    HCCL_INFO("Execute AllGatherOutPlace success.");
    return HCCL_SUCCESS;
}

HcclResult AllGatherOutPlaceGraphMode(
    void* sendBuf, void* recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream,
    const std::string& tag, const ResPackGraphMode& resPack)
{
    HCCL_INFO("Start to execute AllGatherOutPlaceGraphMode");
    CHK_RET(
        AllGatherOutPlaceCommon(sendBuf, recvBuf, sendCount, dataType, comm, stream, tag, OpMode::OFFLOAD, resPack));
    HCCL_INFO("Execute AllGatherOutPlaceGraphMode success.");
    return HCCL_SUCCESS;
}

HcclResult AllGatherOutPlace(
    void* sendBuf, void* recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream,
    const std::string& tag)
{
    HCCL_INFO("Start to execute AllGatherOutPlace");
    CHK_RET(AllGatherOutPlaceCommon(
        sendBuf, recvBuf, sendCount, dataType, comm, stream, tag, OpMode::OPBASE, ResPackGraphMode()));
    HCCL_INFO("Execute AllGatherOutPlace success.");
    return HCCL_SUCCESS;
}

HcclResult AllGatherEntryLog(
    void* sendBuf, void* recvBuf, uint64_t sendCount, HcclDataType dataType, aclrtStream stream, const std::string& tag,
    const std::string& opName, bool forceLog)
{
    /* 接口交互信息日志 */
    if (forceLog || GetExternalInputHcclEnableEntryLog()) {
        s32 deviceId = 0;
        ACLCHECK(aclrtGetDevice(&deviceId));
        s32 streamId = 0;
        ACLCHECK(aclrtStreamGetId(stream, &streamId));
        char stackLogBuffer[LOG_TMPBUF_SIZE];
        s32 ret = snprintf_s(
            stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], sendBuf[%p], recvBuf[%p], sendCount[%llu], dataType[%s], streamId[%d], deviceId[%d]", tag.c_str(),
            sendBuf, recvBuf, sendCount, GetDataTypeEnumStr(dataType).c_str(), streamId, deviceId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "Entry-" + opName + ":" + std::string(stackLogBuffer);
        HCCL_RUN_INFO("%s", logInfo.c_str());
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
