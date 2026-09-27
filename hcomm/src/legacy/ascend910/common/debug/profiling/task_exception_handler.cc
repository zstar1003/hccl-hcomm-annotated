/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <sstream>
#include <iostream>
#include <cstdint>
#include <iomanip>
#include <array>
#include "adapter_rts_common.h"
#include "externalinput_pub.h"
#include "sal_pub.h"
#include "../../../algorithm/pub_inc/common.h"
#include "acl/error_codes/rt_error_codes.h"
#include "task_exception_handler.h"

using namespace hccl;
using namespace std;
std::atomic<int> TaskExceptionHandler::communicatorCount_{0};
std::atomic<bool> TaskExceptionHandler::errMsgFlag_{false};
GetErrStatusVecCallBack g_GetErrStatusVecCallBack = nullptr;
std::mutex g_communicatorCallbackMapMutex;
array<map<s32, GetAicpuTaskExceptionCallBack>, MAX_MODULE_DEVICE_NUM> g_communicatorCallbackMap;
std::mutex g_commHadCallbackArrayMutex;
array<bool, MAX_MODULE_DEVICE_NUM> g_commHadCallbackArray = {false};
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
void RegisterGetErrStatusVecCallBack(GetErrStatusVecCallBack p1)
{
    g_GetErrStatusVecCallBack = p1;
    return;
}

void RegisterGetAicpuTaskExceptionCallBack(s32 streamId, u32 deviceLogicId, GetAicpuTaskExceptionCallBack p1)
{
    if (deviceLogicId >= MAX_MODULE_DEVICE_NUM) {
        HCCL_ERROR(
            "[RegisterGetAicpuTaskExceptionCallBack] deviceLogicId[%u] out of range, max is %u", deviceLogicId,
            MAX_MODULE_DEVICE_NUM - 1);
        return;
    }
    lock_guard<mutex> lock(g_communicatorCallbackMapMutex);
    g_communicatorCallbackMap[deviceLogicId][streamId] = p1;
    return;
}

void UnregisterGetAicpuTaskExceptionCallBack(s32 streamId, u32 deviceLogicId)
{
    if (deviceLogicId >= MAX_MODULE_DEVICE_NUM) {
        HCCL_ERROR(
            "[UnregisterGetAicpuTaskExceptionCallBack] deviceLogicId[%u] out of range, max is %u", deviceLogicId,
            MAX_MODULE_DEVICE_NUM - 1);
        return;
    }
    lock_guard<mutex> lock(g_communicatorCallbackMapMutex);
    auto& deviceMap = g_communicatorCallbackMap[deviceLogicId];
    auto it = deviceMap.find(streamId);
    if (it != deviceMap.end()) {
        deviceMap.erase(it);
    }
    return;
}
#ifdef __cplusplus
}
#endif // __cplusplus
namespace hccl {
namespace hccl_alg {
    std::vector<std::string> GetErrStatusVec(s32 deviceLogicID, const std::string& group = HCCL_WORLD_GROUP)
    {
        if (g_GetErrStatusVecCallBack != nullptr) {
            return g_GetErrStatusVecCallBack(deviceLogicID, group);
        } else {
            HCCL_RUN_WARNING("[GetErrStatusVec]g_GetErrStatusVecCallBack is nullptr.");
        }
        return std::vector<std::string>();
    }
} // namespace hccl_alg
} // namespace hccl

std::string GetTaskName(TaskType taskType, bool isAlgInfo = false);
std::string GetLinkTypeName(LinkType linkInput);
std::string GetAlgTypeStr(AlgType algType);
std::string GetTaskBriefsName(TaskType taskType);

namespace {
constexpr u32 STREAM_COUNT_UPPER_LIMIT = 2048;               // stream 数量最大值2048，防止内存占用量过大
constexpr u32 TASK_COUNT_UPPER_LIMIT = 2048;                 // task 数量最大值2048，防止内存占用量过大
constexpr u32 TASK_COUNT_UPPER_LIMIT_OP_BASE = 65535;        // 单算子模式task数量最大值
constexpr u32 TASK_CONTEXT_SIZE = 50;                        // task 执行失败时打印前序task的数量
constexpr u32 TASK_CONTEXT_INFO_SIZE = LOG_TMPBUF_SIZE - 50; // task 执行失败时打印前序task信息的长度限制
constexpr u32 PRINT_TASK_AIV_INFO_COUNT = 10;
constexpr u32 AIV_KERNEL_FLAG_SIZE_PER_OP = 6;

constexpr u32 MAX_NUM_BLOCKS = 48;
constexpr u32 MAX_RANK_SIZE_SUPERPOD = 768;
constexpr u32 INTERVAL_1VN = 128;
constexpr u32 INTERVAL_NV1 = 128;
constexpr u32 INTERVAL_1V1 = 8;
constexpr u32 PING_PONG_NUM = 2;
constexpr u32 PRINT_NV1_NUM = 4;
constexpr u32 PRINT_1VN_NUM = 4;
constexpr u32 INTERVAL_COUNT = 8;
constexpr u32 NOTIFY_NUM = 3;
constexpr u32 NUM_BLOCKS_PER_RANK = 4;
constexpr u32 CORE_PER_CARDS = 4;
constexpr u32 NOTIFY_GROUPS_1V1 = 2;

u32 maxStrCount = 0;
u32 maxTaskCount = 0;

std::string GetReduceOpString(HcclReduceOp op)
{
    u32 opVal = static_cast<u32>(op);
    return opVal < ProfilerBase::opString.size() ? std::to_string(ProfilerBase::opString[opVal]) :
                                                   "Unknown(" + std::to_string(op) + ")";
}

std::string GetDataTypeString(HcclDataType dataType)
{
    u32 dtVal = static_cast<u32>(dataType);
    return dtVal < ProfilerBase::dataTypeString.size() ? std::to_string(ProfilerBase::dataTypeString[dtVal]) :
                                                         "Unknown(" + std::to_string(dataType) + ")";
}
} // namespace
array<map<int, shared_ptr<deque<TaskInfo>>>, MAX_MODULE_DEVICE_NUM> TaskExceptionHandler::taskMap;
array<std::mutex, MAX_MODULE_DEVICE_NUM> TaskExceptionHandler::taskMapMutex;
array<map<int, shared_ptr<deque<FFTSOpInfo>>>, MAX_MODULE_DEVICE_NUM> TaskExceptionHandler::opMap;
array<std::mutex, MAX_MODULE_DEVICE_NUM> TaskExceptionHandler::opMapMutex;
array<
    std::map<
        int, shared_ptr<std::deque<std::pair<std::shared_ptr<FFTSOpInfo>, std::shared_ptr<std::vector<CtxInfo>>>>>>,
    MAX_MODULE_DEVICE_NUM>
    TaskExceptionHandler::opCtxInfo;
array<std::mutex, MAX_MODULE_DEVICE_NUM> TaskExceptionHandler::opCtxInfoMutex;
array<std::vector<CtxInfo>, MAX_MODULE_DEVICE_NUM> TaskExceptionHandler::ctxInfoArray;
array<std::mutex, MAX_MODULE_DEVICE_NUM> TaskExceptionHandler::ctxInfoVectorMutex;
array<std::map<const std::string, std::pair<const std::string, std::shared_ptr<GroupRankInfo>>>, MAX_MODULE_DEVICE_NUM>
    TaskExceptionHandler::groupRankMap;
array<std::mutex, MAX_MODULE_DEVICE_NUM> TaskExceptionHandler::groupRankMapMutex;
array<std::map<const std::string, std::shared_ptr<std::queue<OpDataInfo>>>, MAX_MODULE_DEVICE_NUM>
    TaskExceptionHandler::tagOpDataMap;
array<std::mutex, MAX_MODULE_DEVICE_NUM> TaskExceptionHandler::tagOpDataMapMutex;
std::array<std::map<const std::string, std::string>, MAX_MODULE_DEVICE_NUM> TaskExceptionHandler::groupUdiMap;
std::array<std::mutex, MAX_MODULE_DEVICE_NUM> TaskExceptionHandler::groupUdiMapMutex;
TaskInfo::TaskInfo(
    u32& streamID, u32& taskID, string& tag, TaskType& taskType, AlgType& algType, u32& index, const TaskParaDMA& para)
    : streamID(streamID),
      taskID(taskID),
      tag(tag),
      taskType(taskType),
      isAlgInfo(false),
      algType(algType),
      index(index)
{
    taskPara.DMA.src = para.src;
    taskPara.DMA.dst = para.dst;
    taskPara.DMA.size = para.size;
    taskPara.DMA.notifyID = para.notifyID;
    taskPara.DMA.linkType = para.linkType;
    taskPara.DMA.remoteUserRank = para.remoteUserRank;
}
TaskInfo::TaskInfo(
    u32& streamID, u32& taskID, string& tag, TaskType& taskType, AlgType& algType, u32& index,
    const TaskParaReduce& para)
    : streamID(streamID),
      taskID(taskID),
      tag(tag),
      taskType(taskType),
      isAlgInfo(false),
      algType(algType),
      index(index)
{
    taskPara.Reduce.src = para.src;
    taskPara.Reduce.dst = para.dst;
    taskPara.Reduce.size = para.size;
    taskPara.Reduce.op = para.op;
    taskPara.Reduce.dataType = para.dataType;
    taskPara.Reduce.linkType = para.linkType;
    taskPara.Reduce.remoteUserRank = para.remoteUserRank;
}
TaskInfo::TaskInfo(
    u32& streamID, u32& taskID, string& tag, TaskType& taskType, AlgType& algType, u32& index,
    const TaskParaNotify& para)
    : streamID(streamID),
      taskID(taskID),
      tag(tag),
      taskType(taskType),
      isAlgInfo(false),
      algType(algType),
      index(index)
{
    taskPara.Notify.notifyID = para.notifyID;
    taskPara.Notify.stage = para.stage;
    taskPara.Notify.remoteUserRank = para.remoteUserRank;
}
TaskInfo::TaskInfo(u32& streamID, u32& taskID, string& tag, const TaskParaAiv& para)
    : streamID(streamID),
      taskID(taskID),
      tag(tag),
      isAlgInfo(true)
{
    taskPara.Aiv.cmdType = para.cmdType;
    taskPara.Aiv.tag = para.tag;
    taskPara.Aiv.size = para.size;
    taskPara.Aiv.numBlocks = para.numBlocks;
    taskPara.Aiv.rankSize = para.rankSize;
    taskPara.Aiv.flagMem = para.flagMem;
    taskPara.Aiv.aivRdmaStep = para.aivRdmaStep;
    taskPara.Aiv.rank = para.rank;
    taskPara.Aiv.isOpbase = para.isOpbase;
}
CtxInfo::CtxInfo(TaskType& taskType, const TaskParaDMA& para) : taskType(taskType)
{
    ctxPara.DMA.src = para.src;
    ctxPara.DMA.dst = para.dst;
    ctxPara.DMA.size = para.size;
    ctxPara.DMA.notifyID = para.notifyID;
    ctxPara.DMA.linkType = para.linkType;
    ctxPara.DMA.remoteUserRank = para.remoteUserRank;
}
CtxInfo::CtxInfo(TaskType& taskType, const TaskParaReduce& para) : taskType(taskType)
{
    ctxPara.Reduce.src = para.src;
    ctxPara.Reduce.dst = para.dst;
    ctxPara.Reduce.size = para.size;
    ctxPara.Reduce.op = para.op;
    ctxPara.Reduce.dataType = para.dataType;
    ctxPara.Reduce.linkType = para.linkType;
    ctxPara.Reduce.remoteUserRank = para.remoteUserRank;
}
CtxInfo::CtxInfo(TaskType& taskType, const TaskParaNotify& para) : taskType(taskType)
{
    ctxPara.Notify.notifyID = para.notifyID;
    ctxPara.Notify.stage = para.stage;
    ctxPara.Notify.remoteUserRank = para.remoteUserRank;
}

string TaskInfo::GetBaseInfoStr() // 防止tag字符串过长，base信息和para信息分开打印
{
    string taskContent;
    taskContent += "streamID:[";
    taskContent += std::to_string(streamID);
    taskContent += "], taskID[";
    taskContent += std::to_string(taskID);
    taskContent += "], taskType[";
    taskContent += GetTaskName(taskType, isAlgInfo);
    taskContent += "], tag[";
    taskContent += tag;
    taskContent += "], ";
    taskContent += GetAlgTypeStr(algType);
    return taskContent;
}

string TaskInfo::GetRankInfo()
{
    u32 remoteRank = INVALID_VALUE_RANKID;
    switch (taskType) {
        case TaskType::TASK_SDMA:
        case TaskType::TASK_RDMA:
            remoteRank = taskPara.DMA.remoteUserRank;
            break;
        case TaskType::TASK_REDUCE_INLINE:
        case TaskType::TASK_REDUCE_TBE:
            remoteRank = taskPara.Reduce.remoteUserRank;
            break;
        case TaskType::TASK_NOTIFY_RECORD:
        case TaskType::TASK_NOTIFY_WAIT:
            remoteRank = taskPara.Notify.remoteUserRank;
            break;
        default:
            return "/";
    }
    return (remoteRank == INVALID_VALUE_RANKID) ? "/" : to_string(remoteRank);
}

string TaskInfo::GetNotifyInfo()
{
    u64 notifyInfo = INVALID_U64;
    switch (taskType) {
        case TaskType::TASK_RDMA:
            notifyInfo = taskPara.DMA.notifyID;
            break;
        case TaskType::TASK_NOTIFY_RECORD:
        case TaskType::TASK_NOTIFY_WAIT:
            notifyInfo = taskPara.Notify.notifyID;
            break;
        default:
            return "/";
    }
    if (notifyInfo == INVALID_U64) {
        return "/";
    } else {
        stringstream paraStr;
        // NotifyId取后八位16进制数进行打印
        paraStr << std::hex << static_cast<u32>(notifyInfo);
        return paraStr.str();
    }
}

string TaskInfo::GetParaInfoStr()
{
    if (isAlgInfo) {
        return GetParaAiv();
    }
    switch (taskType) {
        case TaskType::TASK_SDMA:
        case TaskType::TASK_RDMA:
            return GetParaDMA();
        case TaskType::TASK_REDUCE_INLINE:
        case TaskType::TASK_REDUCE_TBE:
            return GetParaReduce();
        case TaskType::TASK_NOTIFY_RECORD:
        case TaskType::TASK_NOTIFY_WAIT:
            return GetParaNotify();
        default:
            return "unknown task";
    }
}

string TaskInfo::GetParaDMA()
{
    string retStr;
    stringstream paraStr;
    paraStr << "src:" << "[0x" << std::hex << static_cast<u64>(reinterpret_cast<uintptr_t>(taskPara.DMA.src))
            << "], dst:"
            << "[0x" << std::hex << static_cast<u64>(reinterpret_cast<uintptr_t>(taskPara.DMA.dst)) << "], size:"
            << "[0x" << std::hex << static_cast<u64>(taskPara.DMA.size) << "], notify id:"
            << "[0x" << std::hex << std::setw(16) // 16字符长度对齐
            << std::setfill('0') << taskPara.DMA.notifyID << "], link type:[" << GetLinkTypeName(taskPara.DMA.linkType)
            << "], remote rank:["
            << ((taskPara.DMA.remoteUserRank == INVALID_VALUE_RANKID) ? "local" :
                                                                        to_string(taskPara.DMA.remoteUserRank))
            << "]";
    retStr += paraStr.str();
    return retStr;
}

string TaskInfo::GetParaNotify()
{
    string retStr;
    stringstream paraStr;
    paraStr << "notify id:"
            << "[0x" << std::hex << std::setw(16) // 16字节长度对齐
            << std::setfill('0') << taskPara.Notify.notifyID << "], stage:[" << taskPara.Notify.stage
            << "], remote rank:["
            << ((taskPara.Notify.remoteUserRank == INVALID_VALUE_RANKID) ? "local" :
                                                                           to_string(taskPara.Notify.remoteUserRank))
            << "]";
    retStr += paraStr.str();
    return retStr;
}

string TaskInfo::GetParaReduce()
{
    string retStr;
    stringstream paraStr;
    paraStr << "src:" << "[0x" << std::hex << static_cast<u64>(reinterpret_cast<uintptr_t>(taskPara.Reduce.src))
            << "], dst:"
            << "[0x" << std::hex << static_cast<u64>(reinterpret_cast<uintptr_t>(taskPara.Reduce.dst)) << "], size:"
            << "[0x" << std::hex
            << static_cast<u64>(taskPara.Reduce.size * ProfilerBase::sizeOf[taskPara.Reduce.dataType]) << "], op:["
            << GetReduceOpString(taskPara.Reduce.op) << "], data type:[" << GetDataTypeString(taskPara.Reduce.dataType)
            << "], link type:[" << GetLinkTypeName(taskPara.Reduce.linkType) << "], remote rank:["
            << ((taskPara.Reduce.remoteUserRank == INVALID_VALUE_RANKID) ? "local" :
                                                                           to_string(taskPara.Reduce.remoteUserRank))
            << "]";
    retStr += paraStr.str();
    return retStr;
}

string TaskInfo::GetParaAiv()
{
    string retStr;
    stringstream paraStr;
    paraStr << "cmdType:[" << static_cast<int>(taskPara.Aiv.cmdType) << "], "
            << "tag:[" << taskPara.Aiv.tag << "], "
            << "size:[" << taskPara.Aiv.size << "], "
            << "numBlocks:[" << taskPara.Aiv.numBlocks << "], "
            << "rankSize:[" << taskPara.Aiv.rankSize << "], "
            << "aivRdmaStep:[" << taskPara.Aiv.aivRdmaStep << "], "
            << "flagMem:[0x" << std::hex << static_cast<u64>(reinterpret_cast<uintptr_t>(taskPara.Aiv.flagMem)) << "], "
            << "isOpbase:[" << taskPara.Aiv.isOpbase << "]";

    retStr += paraStr.str();
    return retStr;
}

u32 TaskInfo::GetRemoteUserRank() { return taskPara.Notify.remoteUserRank; }

string CtxInfo::GetCtxBaseInfoStr() // 防止tag字符串过长，base信息和para信息分开打印
{
    string taskContent;
    taskContent += "taskType[";
    taskContent += GetTaskName(taskType);
    taskContent += "].";
    return taskContent;
}

string CtxInfo::GetCtxRankInfo()
{
    u32 remoteRank = INVALID_VALUE_RANKID;
    switch (taskType) {
        case TaskType::TASK_SDMA:
        case TaskType::TASK_RDMA:
            remoteRank = ctxPara.DMA.remoteUserRank;
            break;
        case TaskType::TASK_REDUCE_INLINE:
        case TaskType::TASK_REDUCE_TBE:
            remoteRank = ctxPara.Reduce.remoteUserRank;
            break;
        case TaskType::TASK_NOTIFY_RECORD:
        case TaskType::TASK_NOTIFY_WAIT:
            remoteRank = ctxPara.Notify.remoteUserRank;
            break;
        default:
            return "/";
    }
    return (remoteRank == INVALID_VALUE_RANKID) ? "/" : to_string(remoteRank);
}

string CtxInfo::GetCtxNotifyInfo()
{
    u64 notifyInfo = INVALID_U64;
    switch (taskType) {
        case TaskType::TASK_RDMA:
            notifyInfo = ctxPara.DMA.notifyID;
            break;
        case TaskType::TASK_NOTIFY_RECORD:
        case TaskType::TASK_NOTIFY_WAIT:
            notifyInfo = ctxPara.Notify.notifyID;
            break;
        default:
            return "/";
    }
    if (notifyInfo == INVALID_U64) {
        return "/";
    } else {
        stringstream paraStr;
        // NotifyId取后八位16进制数进行打印
        paraStr << std::hex << static_cast<u32>(notifyInfo);
        return paraStr.str();
    }
}

string CtxInfo::GetCtxParaInfoStr()
{
    switch (taskType) {
        case TaskType::TASK_SDMA:
        case TaskType::TASK_RDMA:
            return GetCtxParaDMA();
        case TaskType::TASK_REDUCE_INLINE:
        case TaskType::TASK_REDUCE_TBE:
            return GetCtxParaReduce();
        case TaskType::TASK_NOTIFY_RECORD:
        case TaskType::TASK_NOTIFY_WAIT:
            return GetCtxParaNotify();
        default:
            return "unknown task";
    }
}

string CtxInfo::GetCtxParaDMA()
{
    string retStr;
    stringstream paraStr;
    paraStr << "src:" << "[0x" << std::hex << static_cast<u64>(reinterpret_cast<uintptr_t>(ctxPara.DMA.src))
            << "], dst:"
            << "[0x" << std::hex << static_cast<u64>(reinterpret_cast<uintptr_t>(ctxPara.DMA.dst)) << "], size:"
            << "[0x" << std::hex << static_cast<u64>(ctxPara.DMA.size) << "], notify id:"
            << "[0x" << std::hex << std::setw(16) // 16字符长度对齐
            << std::setfill('0') << ctxPara.DMA.notifyID << "], link type:[" << GetLinkTypeName(ctxPara.DMA.linkType)
            << "], remote rank:["
            << ((ctxPara.DMA.remoteUserRank == INVALID_VALUE_RANKID) ? "local" : to_string(ctxPara.DMA.remoteUserRank))
            << "]";
    retStr += paraStr.str();
    return retStr;
}

string CtxInfo::GetCtxParaNotify()
{
    string retStr;
    stringstream paraStr;
    paraStr << "notify id:"
            << "[0x" << std::hex << std::setw(16) // 16字节长度对齐
            << std::setfill('0') << ctxPara.Notify.notifyID << "], stage:[" << ctxPara.Notify.stage
            << "], remote rank:["
            << ((ctxPara.Notify.remoteUserRank == INVALID_VALUE_RANKID) ? "local" :
                                                                          to_string(ctxPara.Notify.remoteUserRank))
            << "]";
    retStr += paraStr.str();
    return retStr;
}

string CtxInfo::GetCtxParaReduce()
{
    string retStr;
    stringstream paraStr;
    paraStr << "src:" << "[0x" << std::hex << static_cast<u64>(reinterpret_cast<uintptr_t>(ctxPara.Reduce.src))
            << "], dst:"
            << "[0x" << std::hex << static_cast<u64>(reinterpret_cast<uintptr_t>(ctxPara.Reduce.dst)) << "], size:"
            << "[0x" << std::hex
            << static_cast<u64>(ctxPara.Reduce.size * ProfilerBase::sizeOf[ctxPara.Reduce.dataType]) << "], op:["
            << GetReduceOpString(ctxPara.Reduce.op) << "], data type:[" << GetDataTypeString(ctxPara.Reduce.dataType)
            << "], link type:[" << GetLinkTypeName(ctxPara.Reduce.linkType) << "], remote rank:["
            << ((ctxPara.Reduce.remoteUserRank == INVALID_VALUE_RANKID) ? "local" :
                                                                          to_string(ctxPara.Reduce.remoteUserRank))
            << "]";
    retStr += paraStr.str();
    return retStr;
}

u32 CtxInfo::GetCtxRemoteUserRank() { return ctxPara.Notify.remoteUserRank; }

std::string GetTaskName(TaskType taskType, bool isAlgInfo)
{
    std::string taskName;

    if (isAlgInfo) {
        taskName = "Task AIV";
        return taskName;
    }

    switch (taskType) {
        case TaskType::TASK_SDMA:
            taskName += "Memcpy";
            break;
        case TaskType::TASK_RDMA:
            taskName += "RDMASend";
            break;
        case TaskType::TASK_REDUCE_INLINE:
            taskName += "Reduce Inline";
            break;
        case TaskType::TASK_REDUCE_TBE:
            taskName += "Reduce TBE";
            break;
        case TaskType::TASK_NOTIFY_RECORD:
            taskName += "Notify Record";
            break;
        case TaskType::TASK_NOTIFY_WAIT:
            taskName += "Notify Wait";
            break;
        default:
            return "unknown task";
    }

    return taskName;
}
std::string GetTaskBriefsName(TaskType taskType)
{
    std::string taskName;
    switch (taskType) {
        case TaskType::TASK_SDMA:
            taskName += "M";
            break;
        case TaskType::TASK_RDMA:
            taskName += "RS";
            break;
        case TaskType::TASK_REDUCE_INLINE:
            taskName += "IR";
            break;
        case TaskType::TASK_REDUCE_TBE:
            taskName += "R";
            break;
        case TaskType::TASK_NOTIFY_RECORD:
            taskName += "NR";
            break;
        case TaskType::TASK_NOTIFY_WAIT:
            taskName += "NW";
            break;
        default:
            return "unknown task";
    }

    return taskName;
}
std::string GetLinkTypeName(LinkType linkInput)
{
    switch (linkInput) {
        case LinkType::LINK_ONCHIP:
            return "OnChip";
        case LinkType::LINK_HCCS:
            return "HCCS";
        case LinkType::LINK_PCIE:
            return "PCIe";
        case LinkType::LINK_ROCE:
            return "RoCE";
        case LinkType::LINK_SIO:
            return "SIO";
        case LinkType::LINK_HCCS_SW:
            return "HCCS_SW";
        default:
            return "OnChip";
    }
}

std::string GetAlgTypeStr(AlgType algType)
{
    std::string algTypeStr = "";
    algTypeStr += "AlgType(level 0-1-2):[";
    auto alg0It = HCCL_ALGO_LEVEL0_NAME_MAP.find(algType.algoLevel0);
    if (alg0It != HCCL_ALGO_LEVEL0_NAME_MAP.end()) {
        algTypeStr += alg0It->second;
    } else {
        algTypeStr += "null";
    }

    algTypeStr += "-";
    auto alg1It = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType.algoLevel1);
    if (alg1It != HCCL_ALGO_LEVEL1_NAME_MAP.end()) {
        algTypeStr += alg1It->second;
    } else {
        algTypeStr += "null";
    }

    algTypeStr += "-";
    auto alg2It = HCCL_ALGO_LEVEL2_NAME_MAP.find(algType.algoLevel2);
    if (alg2It != HCCL_ALGO_LEVEL2_NAME_MAP.end()) {
        algTypeStr += alg2It->second;
    } else {
        algTypeStr += "null";
    }
    algTypeStr += "].";
    return algTypeStr;
}

string FFTSOpInfo::GetBaseInfoStr() // 防止tag字符串过长，base信息和para信息分开打印
{
    string taskContent;
    taskContent += "streamID:[";
    taskContent += std::to_string(streamID);
    taskContent += "], taskID[";
    taskContent += std::to_string(taskID);
    taskContent += "], tag[";
    taskContent += std::string(tag.get());
    taskContent += "], ";
    taskContent += GetAlgTypeStr(algType);
    return taskContent;
}
TaskExceptionHandler::TaskExceptionHandler(u32 deviceLogicId) : ProfilerBase(deviceLogicId) {}
TaskExceptionHandler::~TaskExceptionHandler() {}
std::string GetAndPrintHeartbeatErr(rtExceptionInfo* exceptionInfo, const std::string& group = HCCL_WORLD_GROUP)
{
    auto errStatusVec = hccl_alg::GetErrStatusVec(exceptionInfo->deviceid, group);
    std::string errMsg = "";
    int errSize = errStatusVec.size();
    if (errSize > 0) {
        int maxListSize = 3; // 放入errMsg中的异常事件最多只有3个
        if (errSize <= maxListSize) {
            errMsg = "\nthere are(is) " + std::to_string(errSize) + " abnormal device(s):\n";
        } else {
            errMsg = "\nthere are " + std::to_string(errSize) + " abnormal device(s), "
                     + "only the first 3 devices are listed:\n";
        }

        for (int i = 0; i < errSize; i++) {
            HCCL_ERROR("%s", errStatusVec[i].c_str());
            if (i < maxListSize) {
                errMsg += ("\t" + errStatusVec[i] + "\n");
            }
        }
    }
    return errMsg;
}
void TaskExceptionHandler::PrintTaskContextInfo(
    const std::shared_ptr<std::vector<CtxInfo>>& taskList, u32 contextId, std::string& stageErrInfo)
{
    HCCL_ERROR(
        "%sTask run failed, context sequence before error task is "
        "[NotifyRecord:NR(rank,id), NotifyWait:NW(rank,id), Memcpy:M(rank), Reduce: R(rank), "
        "InlineReduce:IR(rank), RDMASend:RS(rank,id)]:",
        stageErrInfo.c_str());
    std::string taskContextInfo = "";
    u32 startIndex = (contextId > TASK_CONTEXT_SIZE) ? (contextId - TASK_CONTEXT_SIZE) : 0;
    for (; startIndex < contextId; startIndex++) {
        auto curCtxInfo = taskList->at(startIndex);

        std::string taskStr = GetTaskBriefsName(curCtxInfo.taskType);
        taskStr += "(";
        taskStr += curCtxInfo.GetCtxRankInfo();
        if (curCtxInfo.taskType == TaskType::TASK_NOTIFY_RECORD || curCtxInfo.taskType == TaskType::TASK_NOTIFY_WAIT
            || curCtxInfo.taskType == TaskType::TASK_RDMA) {
            taskStr += ("," + curCtxInfo.GetCtxNotifyInfo());
        }
        taskStr += "),";
        if (taskContextInfo.size() + taskStr.size() >= TASK_CONTEXT_INFO_SIZE) {
            HCCL_ERROR("%s ...", taskContextInfo.c_str());
            taskContextInfo = "";
        }
        taskContextInfo += taskStr;
    }
    HCCL_ERROR("%s end.", taskContextInfo.c_str());
    return;
}

void TaskExceptionHandler::TimeStruct2Str(struct timeval& tv, std::string& opDataContent)
{
    const u32 length = 128;
    char timeStr[length] = {0};
    std::string timeStamp;
    const time_t sec = tv.tv_sec;
    struct tm nowTime = {};
    const struct tm* tmp = localtime_r(&sec, &nowTime);
    if (tmp == nullptr) {
        return;
    }

    int32_t err = snprintf_s(
        timeStr, length, length - 1, "%04d-%02d-%02d-%02d:%02d:%02d.%03ld.%03ld", (nowTime.tm_year + 1900),
        nowTime.tm_mon + 1, nowTime.tm_mday, nowTime.tm_hour, nowTime.tm_min, nowTime.tm_sec, tv.tv_usec / 1000,
        tv.tv_usec % 1000);
    if (err == -1) {
        timeStamp = "unknown time";
    } else {
        timeStamp = timeStr;
    }

    opDataContent += "timeStamp:[";
    opDataContent += timeStamp;
    opDataContent += "]";

    return;
}
void TaskExceptionHandler::PrintOpDataInfo(
    OpDataInfo& opDataInfo, [[maybe_unused]] bool isFftsPlus, std::string& stageErrInfo)
{
    stringstream opDataStr;
    opDataStr << "src" << "[0x" << std::hex << static_cast<u64>(reinterpret_cast<uintptr_t>(opDataInfo.src))
              << "], dst[0x" << std::hex << static_cast<u64>(reinterpret_cast<uintptr_t>(opDataInfo.dst)) << "], ";

    string opStr;
    if (opDataInfo.reduceType != HcclReduceOp::HCCL_REDUCE_RESERVED) {
        opStr += "reduceType[";
        opStr += GetReduceOpEnumStr(opDataInfo.reduceType);
        opStr += "], ";
    }

    string opDataContent;
    TimeStruct2Str(opDataInfo.tv, opDataContent);
    opDataContent += ", deviceId[";
    opDataContent += std::to_string(opDataInfo.deviceId);
    opDataContent += "], index[";
    opDataContent += std::to_string(opDataInfo.index);
    opDataContent += "], count[";
    opDataContent += std::to_string(opDataInfo.count);
    opDataContent += "], ";
    opDataContent += opStr;
    opDataContent += opDataStr.str();
    opDataContent += "dataType[";
    opDataContent += GetDataTypeEnumStr(opDataInfo.dataType);
    opDataContent += "].";

    PrintOpDataErrorLog(stageErrInfo, opDataContent);
    return;
}

bool TaskExceptionHandler::DealExceptionOpData(
    rtExceptionInfo* exceptionInfo, std::string& tag, bool isFftsPlus, u32 index, std::string& stageErrInfo)
{
    bool opDataFound = false;
    std::unique_lock<std::mutex> lock(tagOpDataMapMutex[exceptionInfo->deviceid]);
    auto opDataIt = tagOpDataMap[exceptionInfo->deviceid].find(tag);
    CHK_PRT_RET(
        opDataIt == tagOpDataMap[exceptionInfo->deviceid].end(),
        HCCL_ERROR("tag not found. the fail tag is not from HCCL. tag[%s]", tag.c_str()), false);
    auto& opDataQueIt = opDataIt->second;
    CHK_PRT_RET(opDataQueIt->size() == 0, HCCL_ERROR("[TaskExceptionHandler][Callback] OpData queue size 0"), false);
    auto opDataInfo = opDataQueIt->front();
    while (opDataQueIt->size() > 0) {
        HCCL_DEBUG(
            "[TaskExceptionHandler][Callback]index %u opData index %u size %u", index, opDataQueIt->front().index,
            opDataQueIt->size());
        if (index == opDataQueIt->front().index) {
            opDataInfo = opDataQueIt->front();
            opDataFound = true; // 需要匹配最后下发的task，不能break
        }
        opDataQueIt->pop();
    }
    if (!opDataFound) {
        return false;
    }

    PrintOpDataInfo(opDataInfo, isFftsPlus, stageErrInfo);
    return true;
}

bool TaskExceptionHandler::DealExceptionGroupRank(
    rtExceptionInfo* exceptionInfo, std::string& tag, [[maybe_unused]] bool isFftsPlus,
    std::string& groupRankContentInfo, std::string& stageErrInfo)
{
    std::unique_lock<std::mutex> lock(groupRankMapMutex[exceptionInfo->deviceid]);
    auto groupRankIt = groupRankMap[exceptionInfo->deviceid].find(tag);
    CHK_PRT_RET(
        groupRankIt == groupRankMap[exceptionInfo->deviceid].end(),
        HCCL_INFO("tag not found. the fail tag is not from HCCL. tag[%s]", tag.c_str()), false);

    auto groupUdiIt = groupUdiMap[exceptionInfo->deviceid].find(groupRankIt->second.first);
    CHK_PRT_RET(
        groupUdiIt == groupUdiMap[exceptionInfo->deviceid].end(),
        HCCL_INFO("group not found. the fail group is not from HCCL. group[%s]", groupRankIt->second.first.c_str()),
        false);

    string peerRankStr;
    if ((groupRankIt->second.second)->remoteRankId != INVALID_VALUE_RANKSIZE) {
        peerRankStr += "], peerRankId[";
        peerRankStr += std::to_string((groupRankIt->second.second)->remoteRankId);
    }

    string groupRankContent;
    groupRankContent += "group:[";
    groupRankContent += groupRankIt->second.first;
    groupRankContent += "], user define information[";
    groupRankContent += groupUdiIt->second;
    groupRankContent += "], rankSize[";
    groupRankContent += std::to_string((groupRankIt->second.second)->rankSize);
    groupRankContent += "], rankId[";
    groupRankContent += std::to_string((groupRankIt->second.second)->rankId);
    groupRankContent += peerRankStr;
    groupRankContent += "]";
    groupRankContentInfo = groupRankContent;

    PrintGroupErrorLog(stageErrInfo, groupRankContent, tag);
    return true;
}

bool TaskExceptionHandler::DealExceptionCtx(rtExceptionInfo* exceptionInfo)
{
    std::unique_lock<std::mutex> lock(opCtxInfoMutex[exceptionInfo->deviceid]);
    if (!FindAndValidateContext(exceptionInfo)) {
        return false;
    }

    FFTSOpInfo fftsOpInfo;
    CtxInfo exceptionCtxInfo;
    std::string stageErrInfo = "";

    if (!ProcessContext(exceptionInfo, stageErrInfo, fftsOpInfo, exceptionCtxInfo)) {
        return false;
    }

    u32 index = fftsOpInfo.index;
    std::string groupRankContentInfo = "";
    std::string tag(fftsOpInfo.tag.get());

    DealExceptionGroupRank(exceptionInfo, tag, true, groupRankContentInfo, stageErrInfo);
    DealExceptionOpData(exceptionInfo, tag, true, index, stageErrInfo);
    std::string errMsg = GetAndPrintHeartbeatErr(exceptionInfo, tag);
    if (!errMsgFlag_.exchange(true)) {
        if (exceptionCtxInfo.taskType == TaskType::TASK_NOTIFY_WAIT) {
            RPT_INPUT_ERR(
                true, "EI0002",
                std::vector<std::string>(
                    {"remote_rankid", "base_information", "task_information", "group_rank_content"}),
                std::vector<std::string>(
                    {std::to_string(exceptionCtxInfo.GetCtxRemoteUserRank()),
                     exceptionCtxInfo.GetCtxBaseInfoStr().c_str(), (exceptionCtxInfo.GetCtxParaInfoStr()).c_str(),
                     groupRankContentInfo.c_str()}));
        } else if (
            exceptionCtxInfo.taskType == TaskType::TASK_SDMA
            || exceptionCtxInfo.taskType == TaskType::TASK_REDUCE_INLINE) {
            RPT_INPUT_ERR(
                true, "EI0012",
                std::vector<std::string>(
                    {"remote_rankid", "base_information", "task_information", "group_rank_content"}),
                std::vector<std::string>(
                    {std::to_string(exceptionCtxInfo.GetCtxRemoteUserRank()),
                     exceptionCtxInfo.GetCtxBaseInfoStr().c_str(), (exceptionCtxInfo.GetCtxParaInfoStr()).c_str(),
                     groupRankContentInfo.c_str()}));
        }
    }
    return true;
}

bool TaskExceptionHandler::FindAndValidateContext(rtExceptionInfo* exceptionInfo)
{
    auto mapIt = opCtxInfo[exceptionInfo->deviceid].find(exceptionInfo->streamid);
    if (mapIt == opCtxInfo[exceptionInfo->deviceid].end()) {
        HCCL_RUN_INFO("stream not found. the fail ctx is not from HCCL. streamid[%u]", exceptionInfo->streamid);
        return false;
    }

    auto& queIt = mapIt->second;
    if (queIt->size() == 0) {
        HCCL_ERROR("[TaskExceptionHandler][Callback] CtxOpInfo queue size 0");
        return false;
    }

    if ((*(queIt->front().second)).size() == 0) {
        HCCL_ERROR("[TaskExceptionHandler][Callback] CtxInfoVector size 0");
        return false;
    }

    return true;
}

void TaskExceptionHandler::PrintFftsCtxInfo(FFTSOpInfo& fftsOpInfo)
{
    // 按照每个task占用128字节打印ffts的子图信息
    if (fftsOpInfo.descBuf != nullptr && fftsOpInfo.descBufLen > 0) {
        HCCL_ERROR(
            "==========FftsPlusTask-begin-context, ctx_addr=%p, descBuflen=%u, ctx_num=%lu==========",
            fftsOpInfo.descBuf.get(), fftsOpInfo.descBufLen, fftsOpInfo.descBufLen / 128UL);
        for (uint32_t i = 0U; i < (fftsOpInfo.descBufLen / 128UL); i++) {
            HCCL_ERROR(
                "stream_id=%u, task_id=%u, FftsPlusTask context_id=%u:", fftsOpInfo.streamID, fftsOpInfo.taskID, i);
            uint32_t* buf = reinterpret_cast<uint32_t*>(fftsOpInfo.descBuf.get()) + (i * 32U);
            for (uint32_t j = 0U; j < 32U; j += 8) {
                HCCL_ERROR(
                    "context_id=%u, buf[%02u-%02u]=%08x %08x %08x %08x %08x %08x %08x %08x.", i, j, (j + 7U), buf[j],
                    buf[j + 1U], buf[j + 2U], buf[j + 3U], buf[j + 4U], buf[j + 5U], buf[j + 6U], buf[j + 7U]);
            }
        }
        HCCL_ERROR("==========FftsPlusTask-end-context==========");
    }
    return;
}

bool TaskExceptionHandler::ProcessContext(
    rtExceptionInfo* exceptionInfo, std::string& stageErrInfo, FFTSOpInfo& fftsOpInfo, CtxInfo& exceptionCtxInfo)
{
    auto mapIt = opCtxInfo[exceptionInfo->deviceid].find(exceptionInfo->streamid);
    auto& queIt = mapIt->second;
    fftsOpInfo = *(queIt->front().first);
    exceptionCtxInfo = (*(queIt->front().second))[0];
    uint16_t invalidCtxid = 65535;
    bool ctxFound = false;

    while (queIt->size() > 0) {
        if (exceptionInfo->taskid == queIt->back().first->taskID) {
            fftsOpInfo = *(queIt->back().first);
            if (exceptionInfo->expandInfo.u.fftsPlusInfo.contextId == invalidCtxid) {
                // 子图任务粒度下，RTS返回的异常task不包含contextId时的处理，约定contextId为65535。只记录算子信息
                HCCL_WARNING(
                    "%sTask run failed, invalid contextId, "
                    "base opInformation is %s",
                    stageErrInfo.c_str(), fftsOpInfo.GetBaseInfoStr().c_str());
            } else if (exceptionInfo->expandInfo.u.fftsPlusInfo.contextId >= queIt->back().second->size()) {
                HCCL_ERROR(
                    "%sTask run failed, contextId[%u] is out of vector "
                    "size[%zu], base opInformation is %s",
                    stageErrInfo.c_str(), exceptionInfo->expandInfo.u.fftsPlusInfo.contextId,
                    queIt->back().second->size(), fftsOpInfo.GetBaseInfoStr().c_str());
            } else {
                exceptionCtxInfo = (*(queIt->back().second))[exceptionInfo->expandInfo.u.fftsPlusInfo.contextId];
                ctxFound = true;
            }
            break;
        } else {
            queIt->pop_back();
        }
    }

    auto logKeywordL2
        = exceptionCtxInfo.taskType == TaskType::TASK_NOTIFY_WAIT ? LOG_KEYWORDS_TIMEOUT : LOG_KEYWORDS_RUN_FAILED;
    stageErrInfo = "[" + LOG_KEYWORDS_TASK_EXEC + "][" + logKeywordL2 + "][" + LOG_KEYWORDS_HOST + "]";

    PrintFftsCtxInfo(fftsOpInfo);

    if (!ctxFound) {
        return false;
    }

    if (exceptionCtxInfo.taskType == TaskType::TASK_NOTIFY_WAIT) { // 只在出错task为NotifyWait时打印前序task序列
        PrintTaskContextInfo(queIt->back().second, exceptionInfo->expandInfo.u.fftsPlusInfo.contextId, stageErrInfo);
    }

    queIt->clear();

    PrintBaseErrorLog(stageErrInfo, fftsOpInfo.GetBaseInfoStr());
    PrintContextErrorLog(stageErrInfo, exceptionCtxInfo.GetCtxBaseInfoStr());
    PrintParaErrorLog(stageErrInfo, exceptionCtxInfo.GetCtxParaInfoStr(), std::string(fftsOpInfo.tag.get()));

    return true;
}

bool TaskExceptionHandler::DealExceptionOp(rtExceptionInfo* exceptionInfo)
{
    std::unique_lock<std::mutex> lock(opMapMutex[exceptionInfo->deviceid]);
    bool taskFound = false;
    auto mapIt = opMap[exceptionInfo->deviceid].find(exceptionInfo->streamid);
    CHK_PRT_RET(
        mapIt == opMap[exceptionInfo->deviceid].end(),
        HCCL_RUN_INFO("stream not found. the fail op is not from HCCL. streamid[%u]", exceptionInfo->streamid), false);
    auto& queIt = mapIt->second;
    CHK_PRT_RET(queIt->size() == 0, HCCL_ERROR("[TaskExceptionHandler][Callback] OpInfo queue size 0"), false);
    auto exceptionOpInfo = queIt->back();
    while (queIt->size() > 0) {
        if (exceptionInfo->taskid == queIt->back().taskID) {
            exceptionOpInfo = queIt->back();
            taskFound = true; // 从后往前匹配最后下发的相同taskId
            break;
        }
        queIt->pop_back();
    }
    if (!taskFound) {
        return false;
    }
    queIt->clear();

    auto logKeywordL2
        = exceptionInfo->retcode == ACL_ERROR_RT_FFTS_PLUS_TIMEOUT ? LOG_KEYWORDS_TIMEOUT : LOG_KEYWORDS_RUN_FAILED;
    auto stageErrInfo = "[" + LOG_KEYWORDS_TASK_EXEC + "][" + logKeywordL2 + "][" + LOG_KEYWORDS_HOST + "]";

    PrintBaseErrorLog(stageErrInfo, exceptionOpInfo.GetBaseInfoStr());
    u32 index = exceptionOpInfo.index;
    std::string groupRankContentInfo = "";
    std::string tag(exceptionOpInfo.tag.get());
    DealExceptionGroupRank(exceptionInfo, tag, true, groupRankContentInfo, stageErrInfo);
    DealExceptionOpData(exceptionInfo, tag, true, index, stageErrInfo);
    std::string errMsg = GetAndPrintHeartbeatErr(exceptionInfo, tag);
    if (!errMsgFlag_.exchange(true)) {
        if (exceptionInfo->retcode == ACL_ERROR_RT_FFTS_PLUS_TIMEOUT) {
            RPT_INPUT_ERR(
                true, "EI0002",
                std::vector<std::string>(
                    {"remote_rankid", "base_information", "task_information", "group_rank_content"}),
                std::vector<std::string>(
                    {"unknown", exceptionOpInfo.GetBaseInfoStr().c_str(), errMsg.c_str(),
                     groupRankContentInfo.c_str()}));
        }
    }
    return true;
}

void TaskExceptionHandler::PrintTaskContextInfo(
    const std::shared_ptr<std::deque<TaskInfo>>& taskQue, std::string& stageErrInfo)
{
    HCCL_ERROR(
        "%sTask run failed, context sequence before error task is "
        "[NotifyRecord:NR(rank,id), NotifyWait:NW(rank,id), Memcpy:M(rank), Reduce: R(rank), "
        "InlineReduce:IR(rank), RDMASend:RS(rank,id)]:",
        stageErrInfo.c_str());
    std::string taskContextInfo = "";
    u32 startIndex = (taskQue->size() > TASK_CONTEXT_SIZE) ? (taskQue->size() - TASK_CONTEXT_SIZE) : 0;
    for (; startIndex < taskQue->size(); startIndex++) {
        auto taskInfo = taskQue->at(startIndex);

        std::string taskStr = GetTaskBriefsName(taskInfo.taskType);
        taskStr += "(";
        taskStr += taskInfo.GetRankInfo();
        if (taskInfo.taskType == TaskType::TASK_NOTIFY_RECORD || taskInfo.taskType == TaskType::TASK_NOTIFY_WAIT
            || taskInfo.taskType == TaskType::TASK_RDMA) {
            taskStr += ("," + taskInfo.GetNotifyInfo());
        }
        taskStr += "),";
        if (taskContextInfo.size() + taskStr.size() >= TASK_CONTEXT_INFO_SIZE) {
            HCCL_ERROR("%s%s ...", stageErrInfo.c_str(), taskContextInfo.c_str());
            taskContextInfo = "";
        }
        taskContextInfo += taskStr;
    }
    HCCL_ERROR("%s%s end.", stageErrInfo.c_str(), taskContextInfo.c_str());
    return;
}

void TaskExceptionHandler::ParseTaskSyncFlag(s32* flagMem, u32 flagMemSize, u32 rankSize, u32 rank, u32 index)
{
    u32 chips1v1 = std::min(rankSize * NUM_BLOCKS_PER_RANK, MAX_RANK_SIZE_SUPERPOD) * NOTIFY_NUM * INTERVAL_1V1;
    u32 cores1v1 = MAX_NUM_BLOCKS * NOTIFY_GROUPS_1V1 * INTERVAL_1V1;
    u32 chips1vN = PRINT_1VN_NUM * INTERVAL_1VN * NOTIFY_GROUPS_1V1;
    u32 cores1vN = PRINT_1VN_NUM * INTERVAL_1VN * NOTIFY_GROUPS_1V1;
    u32 chipsNv1 = PRINT_NV1_NUM * INTERVAL_NV1 * NOTIFY_GROUPS_1V1;
    u32 coresNv1 = PRINT_NV1_NUM * INTERVAL_NV1 * NOTIFY_GROUPS_1V1;
    u32 count = rankSize * CORE_PER_CARDS * INTERVAL_COUNT;
    u32 syncCount = (chips1v1 + cores1v1 + chips1vN + cores1vN + chipsNv1 + coresNv1) * PING_PONG_NUM + count;
    u32 total = syncCount * sizeof(u32);
    if (total > flagMemSize) {
        HCCL_ERROR("rank %u opIndex=%u flag mem size %u is too little total %u.", rank, index, flagMemSize, total);
        return;
    }

    s32* buf = flagMem;
    u32 offset = 0;

    const std::string PREFIX[PING_PONG_NUM] = {"ping", "pong"};
    std::string str;
    for (u32 i = 0; i < PING_PONG_NUM; ++i) {
        // print chips1v1
        str = SerializeSyncFlag(buf + offset, rankSize * NUM_BLOCKS_PER_RANK * NOTIFY_NUM, INTERVAL_1V1);
        offset += chips1v1;
        HCCL_ERROR("rank %u opIndex %u chips 1v1 sync flag [%s] %s", rank, index, PREFIX[i].c_str(), str.c_str());

        str = SerializeSyncFlag(buf + offset, MAX_NUM_BLOCKS * NOTIFY_GROUPS_1V1, INTERVAL_1V1);
        offset += cores1v1;
        HCCL_ERROR("rank %u opIndex %u cores 1v1 sync flag [%s] %s", rank, index, PREFIX[i].c_str(), str.c_str());

        str = SerializeSyncFlag(buf + offset, PRINT_1VN_NUM * NOTIFY_GROUPS_1V1, INTERVAL_1VN);
        offset += chips1vN;
        HCCL_ERROR("rank %u opIndex %u chips 1vn sync flag [%s] %s", rank, index, PREFIX[i].c_str(), str.c_str());

        str = SerializeSyncFlag(buf + offset, PRINT_1VN_NUM * NOTIFY_GROUPS_1V1, INTERVAL_1VN);
        offset += cores1vN;
        HCCL_ERROR("rank %u opIndex %u cores 1vn sync flag [%s] %s", rank, index, PREFIX[i].c_str(), str.c_str());

        str = SerializeSyncFlag(buf + offset, PRINT_NV1_NUM * NOTIFY_GROUPS_1V1, INTERVAL_NV1);
        offset += chipsNv1;
        HCCL_ERROR("rank %u opIndex %u chips nv1 sync flag [%s] %s", rank, index, PREFIX[i].c_str(), str.c_str());

        str = SerializeSyncFlag(buf + offset, PRINT_NV1_NUM * NOTIFY_GROUPS_1V1, INTERVAL_NV1);
        offset += coresNv1;
        HCCL_ERROR("rank %u opIndex %u cores nv1 sync flag [%s] %s", rank, index, PREFIX[i].c_str(), str.c_str());
    }
    str = SerializeSyncFlag(buf + offset, rankSize * CORE_PER_CARDS, INTERVAL_COUNT);
    HCCL_ERROR("rank %u opIndex %u sync count [%s]", rank, index, str.c_str());
}

std::string TaskExceptionHandler::SerializeSyncFlag(s32* buf, u32 num, u32 interval)
{
    std::stringstream ss;
    s32* pos = buf;
    for (u32 i = 0; i < num; i = i + 1) {
        ss << std::dec << " " << *pos;
        pos = pos + interval;
    }
    return ss.str();
}

void TaskExceptionHandler::PrintTaskAivBuffer(const std::shared_ptr<std::deque<TaskInfo>>& taskQue)
{
    if (taskQue->empty()) {
        return;
    }
    // width参考aiv_communication_base.cc的MAX_FLAG_SIZE_PER_KERNEL

    u32 flagMemSize = 1024 * 1024;
    auto& taskInfo = taskQue->back();
    u32 realRankSize = taskInfo.taskPara.Aiv.rankSize;
    void* tmpFlagMem = malloc(flagMemSize);
    if (tmpFlagMem == nullptr) {
        return;
    }
    s32* flagMem = static_cast<s32*>(tmpFlagMem);
    hrtMemSyncCopy(
        flagMem, flagMemSize, reinterpret_cast<u8*>(taskInfo.taskPara.Aiv.flagMem), flagMemSize,
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST);

    ParseTaskSyncFlag(flagMem, flagMemSize, realRankSize, taskInfo.taskPara.Aiv.rank, taskInfo.index);
    free(flagMem);
}

void TaskExceptionHandler::PrintTaskAivInfo(const std::shared_ptr<std::deque<TaskInfo>>& taskQue)
{
    HCCL_ERROR("[PrintTaskAivInfo] print start: ");
    // 从后往前遍历，最多打印PRINT_TASK_AIV_INFO_COUNT个taskAiv
    int cnt = PRINT_TASK_AIV_INFO_COUNT;
    for (auto it = taskQue->end() - 1; it >= taskQue->begin(); --it) {
        if (!it->isAlgInfo) {
            continue;
        }
        if (cnt <= 0) {
            break;
        }
        auto taskInfo = *it;
        HCCL_ERROR("[AIV](%s) ", taskInfo.GetParaAiv().c_str());
        cnt--;
    }
    HCCL_ERROR("[PrintTaskAivInfo] print end.");
    return;
}

void splitAndPrintErrStr(const std::string& s)
{
    std::vector<string> parts;
    std::istringstream iss(s);
    std::string part;

    // 将字符串按照空格分隔
    while (iss >> part) {
        parts.push_back(part);
    }

    // 每10组作为一行打印，暂不做通用化处理
    constexpr u32 plen = 10;
    for (size_t i = 0; i < parts.size(); i += plen) {
        std::string line;
        for (size_t j = i; j < i + plen && j < parts.size(); ++j) {
            if (j != i) {
                line += " ";
            }
            line += parts[j];
        }
        HCCL_ERROR("%s", line.c_str());
    }
}

HcclResult TaskExceptionHandler::PrintCommAivInfo()
{
    /*  本函数的目的：在任务失败后，遍历当前device的所有通信域
        对于通信域内存在AIV算子的情况进行统计和打印
        提示用户如果有多个通信域存在AIV算子可能导致执行卡住
    */
    u32 groupHasAivCount = 0;
    u32 groupNoAivCount = 0;
    s32 deviceLogicId = -1;
    std::stringstream groupHasAivInfo;
    std::stringstream groupNoAivInfo;

    HcclResult ret = hrtGetDevice(&deviceLogicId);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[TaskExceptionHandler][PrintCommAivInfo]hrtGetDevice failed, ret[%d]", ret);
        return HCCL_E_PARA;
    }

    // 轮询aivGroupIndexMap_[deviceLogicId]的group，确认是否此group内有aiv算子
    // 对于存在aiv算子的，记录和打印group信息和aiv信息
    if (aivGroupIndexMap_[deviceLogicId].size() == 0) {
        HCCL_ERROR("[TaskExceptionHandler][PrintCommAivInfo] aiv group not record");
        return HCCL_SUCCESS;
    }

    for (auto it = aivGroupIndexMap_[deviceLogicId].begin(); it != aivGroupIndexMap_[deviceLogicId].end(); it++) {
        if (it->second == 0) {
            groupNoAivInfo << "[" << it->first.c_str() << "] ";
            groupNoAivCount++;
        } else {
            groupHasAivInfo << "[" << it->first.c_str() << "] ";
            groupHasAivCount++;
        }
    }

    // 如果遍历发现，存在通信域内执行过aiv算子，则提示有可能有卡死风险；大于0则提示，因为MC2也有可能有aiv算子。
    if (groupHasAivCount != 0) {
        HCCL_ERROR(
            "[TaskExceptionHandler][PrintCommAivInfo] multi groups include aiv alg, may cause execution stuck."
            " has aiv group count[%u]",
            groupHasAivCount);
        HCCL_ERROR("groups has aiv list[groupName]:");
        splitAndPrintErrStr(groupHasAivInfo.str());
    }

    // 通信域不包含aiv算子的，也一并提示
    if (groupNoAivCount != 0) {
        HCCL_ERROR("[TaskExceptionHandler][PrintCommAivInfo] no aiv alg group count[%u].", groupNoAivCount);
        HCCL_ERROR("groups no aiv list[groupName]: ");
        splitAndPrintErrStr(groupNoAivInfo.str());
    }

    return HCCL_SUCCESS;
}

bool TaskExceptionHandler::DealExceptionTask(rtExceptionInfo* exceptionInfo)
{
    std::unique_lock<std::mutex> lock(taskMapMutex[exceptionInfo->deviceid]);
    bool taskFound = false;
    auto mapIt = taskMap[exceptionInfo->deviceid].find(exceptionInfo->streamid);
    CHK_PRT_RET(
        mapIt == taskMap[exceptionInfo->deviceid].end(),
        HCCL_RUN_INFO("stream not found. the fail task is not from HCCL. streamid[%u]", exceptionInfo->streamid),
        false);
    auto& queIt = mapIt->second;
    CHK_PRT_RET(queIt->size() == 0, HCCL_ERROR("[TaskExceptionHandler][Callback] TaskInfo queue size 0"), false);

    // 从后往前匹配最后下发的相同taskId
    auto exceptionTaskInfo = queIt->back();
    while (queIt->size() > 0) {
        if (exceptionInfo->taskid == queIt->back().taskID) {
            exceptionTaskInfo = queIt->back();
            taskFound = true;
            break;
        }
        queIt->pop_back();
    }
    if (!taskFound) {
        return false;
    }

    // 检测是否存在多通信域有aiv算子情况，提示可能导致执行卡住
    CHK_PRT_RET(PrintCommAivInfo(), HCCL_ERROR("[TaskExceptionHandler] PrintCommAivInfo failed."), false);

    std::string logKeywordL2;
    std::string logKeywordL3;

    if (exceptionTaskInfo.isAlgInfo) {
        // aiv场景若根据retCode是否为ACL_ERROR_RT_VECTOR_CORE_TIMEOUT判断是否为超时报错
        logKeywordL2 = exceptionInfo->retcode == ACL_ERROR_RT_VECTOR_CORE_TIMEOUT ? LOG_KEYWORDS_TIMEOUT :
                                                                                    LOG_KEYWORDS_RUN_FAILED;
        logKeywordL3 = LOG_KEYWORDS_AIV;
    } else {
        // 非aiv场景根据当前报错的taskType是否为TASK_NOTIFY_WAIT判断是否为超时报错
        logKeywordL2
            = exceptionTaskInfo.taskType == TaskType::TASK_NOTIFY_WAIT ? LOG_KEYWORDS_TIMEOUT : LOG_KEYWORDS_RUN_FAILED;
        logKeywordL3 = LOG_KEYWORDS_HOST_TS;
    }

    auto stageErrInfo = "[" + LOG_KEYWORDS_TASK_EXEC + "][" + logKeywordL2 + "][" + logKeywordL3 + "]";

    if (exceptionTaskInfo.isAlgInfo) {
        PrintTaskAivBuffer(queIt);
        PrintTaskAivInfo(queIt);
    } else if (exceptionTaskInfo.taskType == TaskType::TASK_NOTIFY_WAIT) {
        queIt->pop_back();
        // 只在出错task为NotifyWait时打印前序task序列
        PrintTaskContextInfo(queIt, stageErrInfo);
    }

    queIt->clear();
    HCCL_ERROR("%sTask from HCCL run failed.", stageErrInfo.c_str());
    // 防止tag字符串过长， 信息分开打印
    PrintBaseErrorLog(stageErrInfo, exceptionTaskInfo.GetBaseInfoStr());
    PrintParaErrorLog(stageErrInfo, exceptionTaskInfo.GetParaInfoStr(), exceptionTaskInfo.tag);
    u32 index = exceptionTaskInfo.index;
    std::string groupRankContentInfo = "";
    if (!exceptionTaskInfo.isAlgInfo) {
        // AlgInfo时不打印group rank等信息
        DealExceptionGroupRank(exceptionInfo, exceptionTaskInfo.tag, false, groupRankContentInfo, stageErrInfo);
    }
    DealExceptionOpData(exceptionInfo, exceptionTaskInfo.tag, false, index, stageErrInfo);
    std::string errMsg = GetAndPrintHeartbeatErr(exceptionInfo, exceptionTaskInfo.tag);
    if (!errMsgFlag_.exchange(true)) {
        if (logKeywordL2 == LOG_KEYWORDS_TIMEOUT) {
            RPT_INPUT_ERR(
                true, "EI0002",
                std::vector<std::string>(
                    {"remote_rankid", "base_information", "task_information", "group_rank_content"}),
                std::vector<std::string>(
                    {std::to_string(exceptionTaskInfo.GetRemoteUserRank()), exceptionTaskInfo.GetBaseInfoStr().c_str(),
                     (exceptionTaskInfo.GetParaInfoStr()).c_str(), groupRankContentInfo.c_str()}));
        } else {
            RPT_INPUT_ERR(
                true, "EI0012",
                std::vector<std::string>(
                    {"remote_rankid", "base_information", "task_information", "group_rank_content"}),
                std::vector<std::string>(
                    {std::to_string(exceptionTaskInfo.GetRemoteUserRank()), exceptionTaskInfo.GetBaseInfoStr().c_str(),
                     (exceptionTaskInfo.GetParaInfoStr()).c_str(), groupRankContentInfo.c_str()}));
        }
    }
    return true;
}

void TaskExceptionHandler::PrintAicpuErrorMessage(rtExceptionInfo* exceptionInfo, bool& isExistAicpuError)
{
    ErrorMessageReport errorMessage;
    unique_lock<std::mutex> lock(g_commHadCallbackArrayMutex);
    if (g_commHadCallbackArray[exceptionInfo->deviceid]) {
        // 防止同一个device上出现通信主流和kernel流均出现task exception时runtime调用两次callback
        // HDC通道信息不是读清，防止aicpu task exception重复上报
        HCCL_WARNING("aicpu error message has been reported. deviceid[%u]", exceptionInfo->deviceid);
        return;
    }
    lock.unlock();
    if (g_communicatorCallbackMap[exceptionInfo->deviceid].find(exceptionInfo->streamid)
        != g_communicatorCallbackMap[exceptionInfo->deviceid].end()) {
        // 找到对应的通信域，并调用回调函数从HDC通道获取AICPU异常信息
        errorMessage = (g_communicatorCallbackMap[exceptionInfo->deviceid])[exceptionInfo->streamid]();
        if (strlen(errorMessage.tag) > 0) {
            isExistAicpuError = true;
            string groupRankContent;
            u32 streamId = static_cast<u32>(errorMessage.streamId);
            std::string tag = std::string(errorMessage.tag);
            u32 index = 0;
            TaskParaNotify para(
                static_cast<u64>(errorMessage.notifyId), errorMessage.stage, errorMessage.remoteUserRank);
            TaskInfo exceptionTaskInfo(
                streamId, errorMessage.taskId, tag, errorMessage.taskType, errorMessage.algType, index, para);
            const auto& logKeywordL2 = exceptionTaskInfo.taskType == TaskType::TASK_NOTIFY_WAIT ?
                                           LOG_KEYWORDS_TIMEOUT :
                                           LOG_KEYWORDS_RUN_FAILED;
            auto stageErrInfo = "[" + LOG_KEYWORDS_TASK_EXEC + "][" + logKeywordL2 + "][" + LOG_KEYWORDS_AICPU + "]";
            HCCL_ERROR("%sTask from HCCL run failed.", stageErrInfo.c_str());
            // 防止tag字符串过长， 信息分开打印
            PrintBaseErrorLog(stageErrInfo, exceptionTaskInfo.GetBaseInfoStr());
            PrintParaErrorLog(stageErrInfo, exceptionTaskInfo.GetParaInfoStr(), exceptionTaskInfo.tag);
            PrintGroupErrorMessage(errorMessage, exceptionTaskInfo, groupRankContent, stageErrInfo);
            PrintOpDataErrorMessage(exceptionInfo->deviceid, errorMessage, stageErrInfo);
            std::string errMsg = GetAndPrintHeartbeatErr(exceptionInfo, tag);
            if (!errMsgFlag_.exchange(true)) {
                if (exceptionTaskInfo.taskType == TaskType::TASK_NOTIFY_WAIT) {
                    RPT_INPUT_ERR(
                        true, "EI0002",
                        std::vector<std::string>(
                            {"remote_rankid", "base_information", "task_information", "group_rank_content"}),
                        std::vector<std::string>(
                            {std::to_string(exceptionTaskInfo.GetRemoteUserRank()),
                             exceptionTaskInfo.GetBaseInfoStr().c_str(), (exceptionTaskInfo.GetParaInfoStr()).c_str(),
                             "none"}));
                } else if (
                    exceptionTaskInfo.taskType == TaskType::TASK_SDMA
                    || exceptionTaskInfo.taskType == TaskType::TASK_REDUCE_INLINE) {
                    RPT_INPUT_ERR(
                        true, "EI0012",
                        std::vector<std::string>(
                            {"remote_rankid", "base_information", "task_information", "group_rank_content"}),
                        std::vector<std::string>(
                            {std::to_string(exceptionTaskInfo.GetRemoteUserRank()),
                             exceptionTaskInfo.GetBaseInfoStr().c_str(),
                             (exceptionTaskInfo.GetParaInfoStr() + errMsg).c_str(), groupRankContent.c_str()}));
                }
            }
            lock.lock();
            g_commHadCallbackArray[exceptionInfo->deviceid] = true;
        }
    } else {
        HCCL_INFO("PrintAicpuErrorMessage streamId[%d] is not found.", exceptionInfo->streamid);
    }
    return;
}

void TaskExceptionHandler::PrintGroupErrorMessage(
    ErrorMessageReport& errorMessage, TaskInfo& exceptionTaskInfo, string& groupRankContent, string& stageErrInfo)
{
    std::string groupUdi;
    std::string groupName = std::string(errorMessage.group);
    ProfilerBase::GetUdiByGroup(groupName, groupUdi);

    groupRankContent += "group:[";
    groupRankContent += std::string(errorMessage.group);
    groupRankContent += "], user define information[";
    groupRankContent += groupUdi;
    groupRankContent += "], rankSize[";
    groupRankContent += std::to_string(errorMessage.rankSize);
    groupRankContent += "], rankId[";
    groupRankContent += std::to_string(errorMessage.rankId);
    groupRankContent += " ";
    groupRankContent += std::to_string(errorMessage.remoteUserRank);
    groupRankContent += "]";

    PrintGroupErrorLog(stageErrInfo, groupRankContent, exceptionTaskInfo.tag);
    return;
}

void TaskExceptionHandler::PrintOpDataErrorMessage(u32 deviceId, ErrorMessageReport& errorMessage, string& stageErrInfo)
{
    stringstream opDataStr;
    opDataStr << "src" << "[0x" << std::hex << errorMessage.srcAddr << "], dst[0x" << std::hex << errorMessage.dstAddr
              << "], ";

    string opStr;
    if (errorMessage.reduceType != HcclReduceOp::HCCL_REDUCE_RESERVED) {
        opStr += "reduceType[";
        opStr += GetReduceOpEnumStr(static_cast<HcclReduceOp>(errorMessage.reduceType));
        opStr += "], ";
    }

    string opDataContent;
    opDataContent += "deviceId:[";
    opDataContent += std::to_string(deviceId);
    opDataContent += "], index[";
    opDataContent += std::to_string(errorMessage.opIndex);
    opDataContent += "], count[";
    opDataContent += std::to_string(errorMessage.count);
    opDataContent += "], ";
    opDataContent += opStr;
    opDataContent += opDataStr.str();
    opDataContent += "dataType[";
    opDataContent += GetDataTypeEnumStr(errorMessage.dataType);
    opDataContent += "].";

    PrintOpDataErrorLog(stageErrInfo, opDataContent);
    return;
}

void TaskExceptionHandler::Callback(rtExceptionInfo* exceptionInfo)
{
    HCCL_RUN_INFO("[TaskExceptionHandler][%s]begin to execute hccl task exception callback function.", __func__);
    bool isExistAicpuError = false;
    if (exceptionInfo == nullptr) {
        HCCL_ERROR("[TaskExceptionHandler][Callback] exceptionInfo is nullptr.");
        return;
    }

    PrintAicpuErrorMessage(exceptionInfo, isExistAicpuError);
    if (isExistAicpuError) {
        // 如果已经有AICPU上报的task exception, 则host侧无需再次重复上报
        return;
    }
    u32 maxDeviceNum;
    HcclResult ret = GetMaxDevNum(maxDeviceNum);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[GetMaxDevNum] get maxDeviceNum error");
        return;
    }
    CHK_PRT_RET(
        exceptionInfo->deviceid >= maxDeviceNum,
        HCCL_WARNING(
            "deviceID[%u] from exceptionInfo is bigger than maxDeviceNum[%u]", exceptionInfo->deviceid,
            maxDeviceNum), );
    SaluSleep(ONE_MILLISECOND_OF_USLEEP); // sleep 1ms，等待task被存入数据结构
    HCCL_DEBUG(
        "[TaskExceptionHandler][Callback]Task run failed, ffts+ task type:%d, TaskExceptionSwitch:%u",
        exceptionInfo->expandInfo.type, GetExternalInputTaskExceptionSwitch());
    if (exceptionInfo->expandInfo.type == RT_EXCEPTION_FFTS_PLUS) {
        if (GetExternalInputTaskExceptionSwitch() == 1) {
            DealExceptionCtx(exceptionInfo); // 子任务粒度
        } else {
            DealExceptionOp(exceptionInfo); // 算子粒度
        }
    } else {
        DealExceptionTask(exceptionInfo);
    }
    return;
}
HcclResult TaskExceptionHandler::Init()
{
    if (communicatorCount_.fetch_add(1) == 0) {
        HCCL_RUN_INFO("[TaskExceptionHandler][%s] register taskFailCallback", __func__);
        CHK_RET(hrtRegTaskFailCallbackByModule(Callback));
        CHK_RET(hrtGetStreamAvailableNum(maxStrCount));
        maxStrCount = (maxStrCount < STREAM_COUNT_UPPER_LIMIT) ? maxStrCount : STREAM_COUNT_UPPER_LIMIT;
    }
    maxTaskCount = TASK_COUNT_UPPER_LIMIT;
    // 单算子模式task过多的特殊处理
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        maxTaskCount = TASK_COUNT_UPPER_LIMIT_OP_BASE;
    }

    HCCL_INFO("get from RTS the max stream count[%u] the max task count[%u]", maxStrCount, maxTaskCount);

    if (GetExternalInputHcclEnableFfts() && GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE
        && GetExternalInputTaskExceptionSwitch() == 1) {
        for (std::vector<CtxInfo>& ctxInfoVector : ctxInfoArray) {
            ctxInfoVector.reserve(100); // vector预留100个ctxInfo空间
        }
    }

    // 对全局变量g_commHadCallbackArray进行初始化
    for (u32 i = 0; i < MAX_MODULE_DEVICE_NUM; i++) {
        g_commHadCallbackArray[i] = false;
    }
    return HCCL_SUCCESS;
}

HcclResult TaskExceptionHandler::DeInit()
{
    if (communicatorCount_.fetch_sub(1) == 1) {
        CHK_RET(hrtRegTaskFailCallbackByModule(nullptr));
        HCCL_RUN_INFO("deInit taskFailCallback");
    }
    return HCCL_SUCCESS;
}

bool IsOneSideTask(u32 streamId)
{
    std::string tag;
    CHK_PRT(ProfilerBase::GetTagByStream(streamId, tag));
    if (tag.find("BatchPut_") != std::string::npos || tag.find("BatchGet_") != std::string::npos) {
        return true;
    }
    return false;
}

HcclResult TaskExceptionHandler::Save(
    u32 captureStreamID, u32 streamID, u32 taskID, TaskType& taskType, const TaskParaNotify& para)
{
    u32 maxDeviceNum;
    CHK_RET(GetMaxDevNum(maxDeviceNum));
    CHK_PRT_RET(
        deviceLogicId_ >= maxDeviceNum,
        HCCL_ERROR(
            "[TaskExceptionHandler][Save]deviceLogicId_[%u] is bigger than maxDeviceNum[%u]", deviceLogicId_,
            maxDeviceNum),
        HCCL_E_INTERNAL);
    HCCL_INFO(
        "[TaskExceptionHandler][%s]Save task info, streamId[%u], taskId[%u], taskType[%d]", __func__, streamID, taskID,
        taskType);
    if (GetExternalInputHcclEnableFfts() && GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE
        && GetExternalInputTaskExceptionSwitch() == 1 && !IsOneSideTask(captureStreamID)) {
        std::unique_lock<std::mutex> lock(ctxInfoVectorMutex[deviceLogicId_]); // 防止存入和读取冲突
        CtxInfo tmpCtxInfo(taskType, para);
        ctxInfoArray[deviceLogicId_].insert(ctxInfoArray[deviceLogicId_].end(), tmpCtxInfo);
        return HCCL_SUCCESS;
    }

    std::string tag;
    CHK_RET(ProfilerBase::GetTagByStream(captureStreamID, tag));
    AlgType algType = AlgType::Reserved();
    CHK_RET(ProfilerBase::GetAlgTypeByStream(captureStreamID, algType));
    u32 index = 0;
    ProfilerBase::GetSubmittedOpCnt(index);

    TaskInfo tmpTaskInfo(streamID, taskID, tag, taskType, algType, index, para);
    CHK_RET(InsertTaskMap(streamID, tmpTaskInfo));

    CHK_RET(InsertRankInfo(tag));
    CHK_RET(InsertOpData(tag));
    return HCCL_SUCCESS;
}

HcclResult TaskExceptionHandler::Save(u32& streamID, u32& taskID, TaskType& taskType, const TaskParaNotify& para)
{
    return Save(streamID, streamID, taskID, taskType, para);
}

HcclResult
TaskExceptionHandler::Save(u32 captureStreamID, u32 streamID, u32 taskID, TaskType& taskType, const TaskParaDMA& para)
{
    u32 maxDeviceNum;
    CHK_RET(GetMaxDevNum(maxDeviceNum));
    CHK_PRT_RET(
        deviceLogicId_ >= maxDeviceNum,
        HCCL_ERROR(
            "[TaskExceptionHandler][Save]deviceLogicId_[%u] is bigger than maxDeviceNum[%u]", deviceLogicId_,
            maxDeviceNum),
        HCCL_E_INTERNAL);
    HCCL_INFO(
        "[TaskExceptionHandler][%s]Save task info, streamId[%u], taskId[%u], taskType[%d]", __func__, streamID, taskID,
        taskType);
    if (GetExternalInputHcclEnableFfts() && GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE
        && GetExternalInputTaskExceptionSwitch() == 1 && !IsOneSideTask(captureStreamID)) {
        std::unique_lock<std::mutex> lock(ctxInfoVectorMutex[deviceLogicId_]); // 防止存入和读取冲突
        CtxInfo tmpCtxInfo(taskType, para);
        ctxInfoArray[deviceLogicId_].insert(ctxInfoArray[deviceLogicId_].end(), tmpCtxInfo);
        return HCCL_SUCCESS;
    }

    std::string tag;
    CHK_RET(ProfilerBase::GetTagByStream(captureStreamID, tag));
    AlgType algType = AlgType::Reserved();
    CHK_RET(ProfilerBase::GetAlgTypeByStream(captureStreamID, algType));
    u32 index = 0;
    ProfilerBase::GetSubmittedOpCnt(index);

    TaskInfo tmpTaskInfo(streamID, taskID, tag, taskType, algType, index, para);
    CHK_RET(InsertTaskMap(streamID, tmpTaskInfo));
    CHK_RET(InsertRankInfo(tag));
    CHK_RET(InsertOpData(tag));
    return HCCL_SUCCESS;
}

HcclResult TaskExceptionHandler::Save(u32& streamID, u32& taskID, TaskType& taskType, const TaskParaDMA& para)
{
    return Save(streamID, streamID, taskID, taskType, para);
}

HcclResult TaskExceptionHandler::Save(
    u32 captureStreamID, u32 streamID, u32 taskID, TaskType& taskType, const TaskParaReduce& para)
{
    u32 maxDeviceNum;
    CHK_RET(GetMaxDevNum(maxDeviceNum));
    CHK_PRT_RET(
        deviceLogicId_ >= maxDeviceNum,
        HCCL_ERROR(
            "[TaskExceptionHandler][Save]deviceLogicId_[%u] is bigger than maxDeviceNum[%u]", deviceLogicId_,
            maxDeviceNum),
        HCCL_E_INTERNAL);
    HCCL_INFO(
        "[TaskExceptionHandler][%s]Save task info, streamId[%u], taskId[%u], taskType[%d]", __func__, streamID, taskID,
        taskType);
    if (GetExternalInputHcclEnableFfts() && GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE
        && GetExternalInputTaskExceptionSwitch() == 1 && !IsOneSideTask(captureStreamID)) {
        std::unique_lock<std::mutex> lock(ctxInfoVectorMutex[deviceLogicId_]); // 防止存入和读取冲突
        CtxInfo tmpCtxInfo(taskType, para);
        ctxInfoArray[deviceLogicId_].insert(ctxInfoArray[deviceLogicId_].end(), tmpCtxInfo);
        return HCCL_SUCCESS;
    }

    std::string tag;
    CHK_RET(ProfilerBase::GetTagByStream(captureStreamID, tag));
    AlgType algType = AlgType::Reserved();
    CHK_RET(ProfilerBase::GetAlgTypeByStream(captureStreamID, algType));
    u32 index = 0;
    ProfilerBase::GetSubmittedOpCnt(index);

    TaskInfo tmpTaskInfo(streamID, taskID, tag, taskType, algType, index, para);
    CHK_RET(InsertTaskMap(streamID, tmpTaskInfo));
    CHK_RET(InsertRankInfo(tag));
    CHK_RET(InsertOpData(tag));
    return HCCL_SUCCESS;
}

HcclResult TaskExceptionHandler::Save(u32 captureStreamID, u32 streamID, u32 taskID, const TaskParaAiv& para)
{
    u32 maxDeviceNum;
    CHK_RET(GetMaxDevNum(maxDeviceNum));
    CHK_PRT_RET(
        deviceLogicId_ >= maxDeviceNum,
        HCCL_ERROR(
            "[TaskExceptionHandler][Save]deviceLogicId_[%u] is bigger than maxDeviceNum[%u]", deviceLogicId_,
            maxDeviceNum),
        HCCL_E_INTERNAL);

    std::string tag;
    CHK_RET(ProfilerBase::GetTagByStream(captureStreamID, tag));
    u32 index = 0;
    ProfilerBase::GetSubmittedOpCnt(index);
    TaskInfo tmpTaskInfo(streamID, taskID, tag, para);
    tmpTaskInfo.index = index;
    CHK_RET(InsertTaskMap(streamID, tmpTaskInfo));
    CHK_RET(InsertRankInfo(tag));
    CHK_RET(InsertOpData(tag));
    return HCCL_SUCCESS;
}

HcclResult TaskExceptionHandler::Save(u32 streamID, u32 taskID, const TaskParaAiv& para)
{
    return Save(streamID, streamID, taskID, para);
}

HcclResult TaskExceptionHandler::Save(u32& streamID, u32& taskID, TaskType& taskType, const TaskParaReduce& para)
{
    return Save(streamID, streamID, taskID, taskType, para);
}

HcclResult
TaskExceptionHandler::Save(u32 captureStreamID, u32 streamID, u32 taskID, const void* descBuf, size_t descBufLen)
{
    u32 maxDeviceNum;
    CHK_RET(GetMaxDevNum(maxDeviceNum));
    CHK_PRT_RET(
        deviceLogicId_ >= maxDeviceNum,
        HCCL_ERROR(
            "[TaskExceptionHandler][Save]deviceLogicId_[%u] is bigger than maxDeviceNum[%u]", deviceLogicId_,
            maxDeviceNum),
        HCCL_E_INTERNAL);
    HCCL_INFO("[TaskExceptionHandler][%s]Save task info, streamId[%u], taskId[%u]", __func__, streamID, taskID);
    std::string tag;
    CHK_RET(ProfilerBase::GetTagByStream(captureStreamID, tag));
    AlgType algType = AlgType::Reserved();
    CHK_RET(ProfilerBase::GetAlgTypeByStream(captureStreamID, algType));
    u32 index = 0;
    ProfilerBase::GetSubmittedOpCnt(index);

    if (GetExternalInputTaskExceptionSwitch() == 1) {
        CHK_RET(InsertOpCtxInfo(streamID, taskID, tag, algType, index, descBuf, descBufLen));
    } else {
        CHK_RET(InsertOpMap(streamID, taskID, tag, algType, index));
    }
    CHK_RET(InsertRankInfo(tag));
    CHK_RET(InsertOpData(tag));
    return HCCL_SUCCESS;
}

HcclResult TaskExceptionHandler::Save(u32& streamID, u32& taskID, const void* descBuf, size_t descBufLen)
{
    return Save(streamID, streamID, taskID, descBuf, descBufLen);
}

HcclResult TaskExceptionHandler::SaveToLog(const TaskParaHost& paraHost)
{
    (void)paraHost;
    return HCCL_SUCCESS;
}

HcclResult TaskExceptionHandler::InsertTaskMap(u32& streamID, TaskInfo& tmpTaskInfo) const
{
    std::unique_lock<std::mutex> lock(taskMapMutex[deviceLogicId_]);
    auto it = taskMap[deviceLogicId_].find(streamID);
    if (it == taskMap[deviceLogicId_].end()) {
        // streamID 复用且不会超过最大stream数量，因此Map的size超过最大stream数量属于异常场景
        HCCL_INFO(
            "streamID is [%u], deviceLogicId is [%u], taskMap size is [%u]", streamID, deviceLogicId_,
            taskMap[deviceLogicId_].size());
        CHK_PRT_RET(
            taskMap[deviceLogicId_].size() >= maxStrCount,
            HCCL_ERROR(
                "[Insert][TaskMap]taskMap size is "
                "bigger than max stream count[%u]. stream add fail",
                maxStrCount),
            HCCL_E_INTERNAL);
        std::shared_ptr<deque<TaskInfo>> tmpTaskInfoQue = nullptr;
        EXCEPTION_CATCH((tmpTaskInfoQue = make_shared<deque<TaskInfo>>()), return HCCL_E_PTR);
        tmpTaskInfoQue->push_back(tmpTaskInfo);
        taskMap[deviceLogicId_].insert({streamID, tmpTaskInfoQue});
    } else { // 由于不允许多线程对同一stream操作，因此此处不需要保留锁，并且此处访问量最多，性能考虑也最好不要加锁
        lock.unlock();
        it->second->push_back(tmpTaskInfo);
        if (it->second->size() > maxTaskCount) {
            it->second->pop_front();
        }
    }
    return HCCL_SUCCESS;
}
HcclResult
TaskExceptionHandler::InsertOpMap(u32& streamID, u32& taskID, string& tag, AlgType& algType, u32& index) const
{
    FFTSOpInfo tmpOpPara;
    char* tmpAddr = new (std::nothrow) char[tag.size() + 1]();
    CHK_PTR_NULL(tmpAddr);
    tmpOpPara.tag.reset(tmpAddr, default_delete<char[]>());
    CHK_SAFETY_FUNC_RET(memcpy_sp(tmpOpPara.tag.get(), tag.size() + 1, tag.data(), tag.size()));
    tmpOpPara.streamID = streamID;
    tmpOpPara.taskID = taskID;
    tmpOpPara.algType = algType;
    tmpOpPara.index = index;
    std::unique_lock<std::mutex> lock(opMapMutex[deviceLogicId_]); // 防止存入和读取冲突
    auto it = opMap[deviceLogicId_].find(streamID);
    if (it == opMap[deviceLogicId_].end()) {
        CHK_PRT_RET(
            opMap[deviceLogicId_].size() >= maxStrCount,
            HCCL_ERROR(
                "[Insert][OpMap]Map size is "
                "bigger than max stream count[%u]. stream add fail",
                maxStrCount),
            HCCL_E_INTERNAL);
        std::shared_ptr<deque<FFTSOpInfo>> tmpOpInfoQue = nullptr;
        EXCEPTION_CATCH((tmpOpInfoQue = make_shared<deque<FFTSOpInfo>>()), return HCCL_E_PTR);
        tmpOpInfoQue->push_back(tmpOpPara);
        opMap[deviceLogicId_].insert({streamID, tmpOpInfoQue});
    } else {
        it->second->push_back(tmpOpPara);
        if (it->second->size() > maxTaskCount) {
            it->second->pop_front();
        }
    }
    return HCCL_SUCCESS;
}
HcclResult TaskExceptionHandler::InsertOpCtxInfo(
    u32& streamID, u32& taskID, string& tag, AlgType& algType, u32& index, const void* descBuf, size_t descBufLen) const
{
    FFTSOpInfo tmpOpInfo;
    char* tmpAddr = new (std::nothrow) char[tag.size() + 1]();
    CHK_PTR_NULL(tmpAddr);
    tmpOpInfo.tag.reset(tmpAddr, default_delete<char[]>());
    CHK_SAFETY_FUNC_RET(memcpy_sp(tmpOpInfo.tag.get(), tag.size() + 1, tag.data(), tag.size()));
    tmpOpInfo.streamID = streamID;
    tmpOpInfo.taskID = taskID;
    tmpOpInfo.algType = algType;
    tmpOpInfo.index = index;
    if (descBuf != nullptr && descBufLen > 0) {
        char* tmpDescBuf = new (std::nothrow) char[descBufLen + 1]();
        CHK_PTR_NULL(tmpDescBuf);
        tmpOpInfo.descBuf.reset(tmpDescBuf, default_delete<char[]>());
        CHK_SAFETY_FUNC_RET(memcpy_sp(tmpOpInfo.descBuf.get(), descBufLen + 1, descBuf, descBufLen));
        tmpOpInfo.descBufLen = descBufLen;
    }
    std::shared_ptr<FFTSOpInfo> tmpOpInfoPtr = nullptr;
    EXCEPTION_CATCH((tmpOpInfoPtr = std::make_shared<FFTSOpInfo>()), return HCCL_E_PTR);
    *tmpOpInfoPtr = tmpOpInfo;
    std::shared_ptr<vector<CtxInfo>> tempCtxVectorPtr = nullptr;
    EXCEPTION_CATCH((tempCtxVectorPtr = std::make_shared<vector<CtxInfo>>()), return HCCL_E_PTR);
    std::unique_lock<std::mutex> lock(ctxInfoVectorMutex[deviceLogicId_]); // 防止存入和读取冲突
    *tempCtxVectorPtr = ctxInfoArray[deviceLogicId_];
    auto tempPair = std::make_pair(tmpOpInfoPtr, tempCtxVectorPtr);
    std::unique_lock<std::mutex> infoLock(opCtxInfoMutex[deviceLogicId_]); // 防止存入和读取冲突
    auto tempDeque = opCtxInfo[deviceLogicId_].find(streamID);
    if (tempDeque == opCtxInfo[deviceLogicId_].end()) {
        CHK_PRT_RET(
            opCtxInfo[deviceLogicId_].size() >= maxStrCount,
            HCCL_ERROR(
                "[Insert][opCtxInfo]Map size is "
                "bigger than max stream count[%u]. stream add fail",
                maxStrCount),
            HCCL_E_INTERNAL);
        std::shared_ptr<std::deque<std::pair<std::shared_ptr<FFTSOpInfo>, std::shared_ptr<std::vector<CtxInfo>>>>>
            tmpOpInfoQue = nullptr;
        EXCEPTION_CATCH(
            (tmpOpInfoQue = std::make_shared<
                 std::deque<std::pair<std::shared_ptr<FFTSOpInfo>, std::shared_ptr<std::vector<CtxInfo>>>>>()),
            return HCCL_E_PTR);
        tmpOpInfoQue->push_back(tempPair);
        opCtxInfo[deviceLogicId_].insert({streamID, tmpOpInfoQue});
    } else {
        tempDeque->second->push_back(tempPair);
        if (tempDeque->second->size() > maxTaskCount) {
            tempDeque->second->pop_front();
        }
    }
    ctxInfoArray[deviceLogicId_].clear();
    return HCCL_SUCCESS;
}

HcclResult TaskExceptionHandler::InsertRankInfo(std::string& tag) const
{
    std::string groupName;
    CHK_RET(ProfilerBase::GetGroupNameByTag(tag, groupName));
    GroupRankInfo groupRankInfo;
    CHK_RET(ProfilerBase::GetRankInfoByGroup(groupName, groupRankInfo));
    std::string groupUdi;
    CHK_RET(ProfilerBase::GetUdiByGroup(groupName, groupUdi));

    HCCL_DEBUG("[TaskExceptionHandler][Callback]InsertRankInfo tag %s group %s", tag.c_str(), groupName.c_str());
    {
        std::unique_lock<std::mutex> groupRankMapLock(groupRankMapMutex[deviceLogicId_]);
        std::shared_ptr<GroupRankInfo> tmpRankInfo = nullptr;
        EXCEPTION_CATCH((tmpRankInfo = std::make_shared<GroupRankInfo>()), return HCCL_E_PTR);
        *tmpRankInfo = groupRankInfo;
        auto groupRankIt = groupRankMap[deviceLogicId_].find(tag);
        if (groupRankIt == groupRankMap[deviceLogicId_].end()) {
            auto tempPair = std::make_pair(groupName, tmpRankInfo);
            groupRankMap[deviceLogicId_].insert({tag, tempPair});
        } else {
            groupRankIt->second.second = tmpRankInfo;
        }
    }

    {
        std::lock_guard<std::mutex> groupUdiMapLock(groupUdiMapMutex[deviceLogicId_]);
        auto groupUdiIt = groupUdiMap[deviceLogicId_].find(groupName);
        if (groupUdiIt == groupUdiMap[deviceLogicId_].end()) {
            groupUdiMap[deviceLogicId_].insert({groupName, groupUdi});
        } else {
            groupUdiIt->second = groupUdi;
        }
    }

    return HCCL_SUCCESS;
}

HcclResult TaskExceptionHandler::InsertOpData(std::string& tag) const
{
    OpDataInfo opDataInfo;
    CHK_RET(ProfilerBase::GetOpDataInfoByTag(tag, opDataInfo));
    std::unique_lock<std::mutex> lock(tagOpDataMapMutex[deviceLogicId_]);
    auto tempDeque = tagOpDataMap[deviceLogicId_].find(tag);
    if (tempDeque == tagOpDataMap[deviceLogicId_].end()) {
        std::shared_ptr<queue<OpDataInfo>> tmpOpDataInfo = nullptr;
        EXCEPTION_CATCH((tmpOpDataInfo = std::make_shared<queue<OpDataInfo>>()), return HCCL_E_PTR);
        tmpOpDataInfo->push(opDataInfo);
        tagOpDataMap[deviceLogicId_].insert({tag, tmpOpDataInfo});
        HCCL_DEBUG("[TaskExceptionHandler][Callback]InsertOpData index %u tag %s", opDataInfo.index, tag.c_str());
    } else {
        HCCL_DEBUG(
            "[TaskExceptionHandler][Callback]InsertOpData index %u opData index %u size %u tag %s", opDataInfo.index,
            tempDeque->second->back().index, (tempDeque->second)->size(), tag.c_str());
        if (tempDeque->second->back().index != opDataInfo.index) { // 需要去重，taskid不同时可能是同一个
            tempDeque->second->push(opDataInfo);
        }
        if ((tempDeque->second)->size() > 3000) { // 队列深度大于3000则老化
            HCCL_DEBUG("[Insert][opDataMap]Map size is [%u], need to pop head data.", (tempDeque->second)->size());
            tempDeque->second->pop();
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TaskExceptionHandler::Flush() { return HCCL_SUCCESS; }

HcclResult TaskExceptionHandler::TaskExceptionHandler::Run(const StepData& stepData)
{
    (void)stepData;
    return HCCL_SUCCESS;
}
