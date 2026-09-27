/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include "task_exception_handler.h"
#include "log.h"
#include "communicator_impl.h"
#include "coll_service_device_mode.h"
#include "mc2_global_mirror_tasks.h"
#include "ccu_dev_mgr.h"
#include "acl/acl_rt.h"
#include "orion_adapter_hccp.h"
#include <adapter_error_manager_pub.h>
#include "hccl_common_v2.h"
#include "hal.h"
#include "orion_adapter_rts.h"
#include "runtime_api_exception.h"

namespace Hccl {

using namespace std;
using namespace CcuRep;

constexpr uint32_t AIV_FLAG_UB_ALIGN_SIZE = 32; // aiv flag对齐规则
constexpr uint32_t TASK_CONTEXT_SIZE = 50;
constexpr uint32_t TASK_CONTEXT_INFO_SIZE = LOG_TMPBUF_SIZE - 50; // task 执行失败时打印前序task信息的长度限制
constexpr int BYTE = 8;                                           // 一字节的位数
constexpr uint64_t CCU_MSG_256MB_LEN = 256 * 1024 * 1024;         // CCU消息长度不能大于256MB

std::array<TaskExceptionHandler*, MAX_MODULE_DEVICE_NUM> TaskExceptionHandlerManager::handlers_;

std::mutex g_communicatorCallbackMapMutexV2;
array<map<s32, GetAicpuTaskExceptionCallBack>, MAX_MODULE_DEVICE_NUM> g_communicatorCallbackMapV2;
std::mutex g_commHadCallbackArrayMutexV2;
array<bool, MAX_MODULE_DEVICE_NUM> g_commHadCallbackArrayV2 = {false};

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
void RegisterGetAicpuTaskExceptionCallBackV2(s32 streamId, u32 deviceLogicId, Hccl::GetAicpuTaskExceptionCallBack p1)
{
    lock_guard<mutex> lock(Hccl::g_communicatorCallbackMapMutexV2);
    Hccl::g_communicatorCallbackMapV2[deviceLogicId][streamId] = p1;
    return;
}
#ifdef __cplusplus
}
#endif // __cplusplus

TaskExceptionHandler::TaskExceptionHandler(int deviceId) : devId_(deviceId) { Register(); }

TaskExceptionHandler::~TaskExceptionHandler() { UnRegister(); }

void TaskExceptionHandler::Register() const
{
    HrtRegTaskExceptionCallbackByModule(Process);
    HCCL_INFO("[TaskExceptionHandler]exception process func registered.");
}

void TaskExceptionHandler::UnRegister() const { HrtUnregTaskExceptionCallbackByModule(Process); }

TaskExceptionHandler* TaskExceptionHandlerManager::GetHandler(size_t devId)
{
    // 检查 devId 是否越界
    if (devId >= MAX_MODULE_DEVICE_NUM) {
        HCCL_ERROR("[TaskExceptionHandler][GetInstance] deviceLogicID[%lu] is invalid", devId);
        return nullptr;
    }
    // 如果对应位置的实例为空，则创建新实例
    if (handlers_[devId] == nullptr) {
        handlers_[devId] = new (std::nothrow) TaskExceptionHandler(devId);
        if (handlers_[devId] == nullptr) {
            HCCL_ERROR(
                "[TaskExceptionHandler][GetInstance] new TaskExceptionHandler failed due to OOM, devId[%lu]", devId);
            return nullptr;
        }
    }
    return handlers_[devId];
}
TaskExceptionHandlerManager::TaskExceptionHandlerManager() { handlers_.fill(nullptr); }

TaskExceptionHandlerManager::~TaskExceptionHandlerManager()
{
    for (auto& instance : handlers_) {
        if (instance != nullptr) {
            delete instance;
            instance = nullptr;
        }
    }
}

static std::pair<u32, u32> GetOpCounter(const TaskInfo& taskInfo)
{
    std::pair<float, float> floatCounter;
    if (taskInfo.dfxOpInfo_ != nullptr && taskInfo.dfxOpInfo_->headOpCounterAddr_ != 0
        && taskInfo.dfxOpInfo_->tailOpCounterAddr_ != 0) {
        u64 size = 4;
        void* headAddr = reinterpret_cast<void*>(taskInfo.dfxOpInfo_->headOpCounterAddr_);
        void* tailAddr = reinterpret_cast<void*>(taskInfo.dfxOpInfo_->tailOpCounterAddr_);
        HrtMemcpy(&floatCounter.first, size, headAddr, size, RT_MEMCPY_DEVICE_TO_HOST);
        HrtMemcpy(&floatCounter.second, size, tailAddr, size, RT_MEMCPY_DEVICE_TO_HOST);
    }

    std::pair<u32, u32> counter;
    counter.first = static_cast<u32>(floatCounter.first);
    counter.second = static_cast<u32>(floatCounter.second);
    HCCL_INFO("[GetOpCounter] end, head:%u, tail:%u", counter.first, counter.second);
    return counter;
}

static void PrintOpCounterInfo(const TaskInfo& taskInfo)
{
    auto count = GetOpCounter(taskInfo);
    if (taskInfo.dfxOpInfo_ == nullptr) {
        HCCL_ERROR(
            "[TaskExceptionHandler][%s] dfxOpInfo is nullptr, skip opIndex. "
            "headOpCounter[%u] tailOpCounter[%u].",
            __func__, static_cast<u32>(count.first), static_cast<u32>(count.second));
        return;
    }
    HCCL_ERROR(
        "[TaskExceptionHandler][%s]Task run failed, headOpCounter[%u] tailOpCounter[%u] opIndex[%u].", __func__,
        static_cast<u32>(count.first), static_cast<u32>(count.second), taskInfo.dfxOpInfo_->opIndex_);
}

static bool IsMC2Exception(rtExceptionInfo_t* exceptionInfo)
{
    return exceptionInfo != nullptr && exceptionInfo->expandInfo.type == RT_EXCEPTION_FUSION
           && exceptionInfo->expandInfo.u.fusionInfo.type == RT_FUSION_AICORE_CCU;
}

void PrintUbRegisters(s32 devLogicId, const RdmaHandle rdmaHandle)
{
    HCCL_INFO("[PrintUbRegisters] start");
    AuxInfoIn in;
    in.cqe.status = 0xffffffff; // 0xffffffff代表查询所有寄存器
    in.auxInfoInType = AuxInfoInType::AUX_INFO_IN_TYPE_CQE;
    in.cqe.sR = 0;
    AuxInfoOut auxInfo;
    auto ret = RaGetAuxInfo(rdmaHandle, in, auxInfo);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[PrintUbRegister]GetUbRegisterInfo failed.");
    }

    bool isAuxInfoExisted = false;
    for (u32 i = 0; i < auxInfo.auxInfoNum; i++) {
        if (auxInfo.auxInfoValues[i] != 0) { // 非零进行打印
            isAuxInfoExisted = true;
            HCCL_ERROR(
                "devLogicId[%d], cqe_aux_info_type[%u], cqe_aux_info_value[0x%x]", devLogicId, auxInfo.auxInfoTypes[i],
                auxInfo.auxInfoValues[i]);
        } else {
            HCCL_INFO(
                "devLogicId[%d], cqe_aux_info_type[%u], cqe_aux_info_value[0x%x]", devLogicId, auxInfo.auxInfoTypes[i],
                auxInfo.auxInfoValues[i]);
        }
    }
    if (!isAuxInfoExisted) {
        HCCL_ERROR("devLogicId[%d], all aux_info values are zero.", devLogicId);
    }
}

void PrintCcuUbRegisters(s32 devLogicId, const ParaCcu& ccuTaskParam)
{
    std::vector<CcuJetty*> ccuJettys;
    HcclResult ret = GetCcuJettys(devLogicId, ccuTaskParam, ccuJettys);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("PrintCcuUbRegisters failed");
    }
    u32 jettyNum = ccuJettys.size();

    std::vector<JettyHandle> jettyHandles;
    for (auto& ccuJetty : ccuJettys) {
        jettyHandles.push_back(ccuJetty->GetJettyHandle());
    }

    std::vector<JettyStatus> jettyStatusVec;
    RaBatchQueryJettyStatus(jettyHandles, jettyStatusVec, jettyNum);

    for (u32 i = 0; i < jettyNum; ++i) {
        if (jettyStatusVec[i] == JettyStatus::ERROR) {
            auto rdmaHandle = ccuJettys[i]->GetRdmaHandle();
            HCCL_ERROR("PrintCcuUbRegisters jettyId[%u]", ccuJettys[i]->GetJettyId());
            PrintUbRegisters(devLogicId, rdmaHandle);
            break;
        }
    }
}

void TaskExceptionHandler::Process(rtExceptionInfo_t* exceptionInfo)
{
    // Task Exception 入口，使用宏捕获执行间异常
    TRY_CATCH_PRINT_ERROR(
        if (exceptionInfo == nullptr) {
            HCCL_ERROR("Exception process failed, rtExceptionInfo is nullptr.");
            return;
        }

        if (IsMC2Exception(exceptionInfo)) {
            ProcessCcuMC2Exception(exceptionInfo);
            return;
        }

        const auto curTask
        = GlobalMirrorTasks::Instance().GetTaskInfo(
            exceptionInfo->deviceid, exceptionInfo->streamid, exceptionInfo->taskid);
        if (curTask == nullptr) {
            // 未找到异常对应的TaskInfo
            HCCL_ERROR(
                "Exception task not found. deviceId[%u], streamId[%u], taskId[%u].", exceptionInfo->deviceid,
                exceptionInfo->streamid, exceptionInfo->taskid);
            return;
        }

        if (curTask->taskParam_.taskType == TaskParamType::TASK_CCU) {
            ProcessCcuException(exceptionInfo, *curTask);
        } else if (curTask->taskParam_.taskType == TaskParamType::TASK_AIV) {
            ProcessAivException(exceptionInfo, *curTask);
        } else { ProcessException(exceptionInfo, *curTask); });
}

/*
 @Desc: AIV 算子异常DFX
*/
void TaskExceptionHandler::ProcessAivException(rtExceptionInfo_t* exceptionInfo, const TaskInfo& taskInfo)
{
    auto stageErrInfo = "[" + LOG_KEYWORDS_TASK_EXEC + "][" + LOG_KEYWORDS_RUN_FAILED + "][" + LOG_KEYWORDS_AIV + "]";
    HCCL_ERROR("%s Task from HCCL run failed.", stageErrInfo.c_str());

    HCCL_ERROR(
        "%s Task run failed, para information is "
        "deviceId[%u] streamId[%u], TaskId[%u], cmdType[%u], "
        "tag[%u],rank[%u],rankSize[%u], dataCount[%u], numBlocks[%u],"
        "dataType:[%u], beginTime:[%llu], flagMem[%p]",
        stageErrInfo.c_str(), exceptionInfo->deviceid, exceptionInfo->streamid, exceptionInfo->taskid,
        taskInfo.taskParam_.taskPara.Aiv.cmdType, taskInfo.taskParam_.taskPara.Aiv.tag,
        taskInfo.taskParam_.taskPara.Aiv.rank, taskInfo.taskParam_.taskPara.Aiv.rankSize,
        taskInfo.taskParam_.taskPara.Aiv.count, taskInfo.taskParam_.taskPara.Aiv.numBlocks,
        taskInfo.taskParam_.taskPara.Aiv.dataType, taskInfo.taskParam_.beginTime,
        taskInfo.taskParam_.taskPara.Aiv.flagMem);

    // 打印算子flag 区域, flag区域比较大，需要通过LOG_TMPBUF_SIZE控制打印的长度
    void* flag_buff_temp = nullptr;
    try {
        flag_buff_temp = HrtMallocHost(taskInfo.taskParam_.taskPara.Aiv.flagMemSize);
        HrtMemcpy(
            flag_buff_temp, taskInfo.taskParam_.taskPara.Aiv.flagMemSize, taskInfo.taskParam_.taskPara.Aiv.flagMem,
            taskInfo.taskParam_.taskPara.Aiv.flagMemSize, RT_MEMCPY_DEVICE_TO_HOST);
    } catch (const RuntimeApiException& e) {
        HCCL_ERROR("[TaskExceptionHandler] [%s] host memory operation fail: %s", __func__, e.what());
        if (flag_buff_temp != nullptr) {
            HrtFreeHost(flag_buff_temp);
        }
        return;
    }

    std::stringstream flagStr;
    int32_t* flagMemInt32 = static_cast<int32_t*>(flag_buff_temp);
    u64 flagCount = taskInfo.taskParam_.taskPara.Aiv.flagMemSize / sizeof(int32_t);
    // aiv 内部是32 byte对齐,即每32字节首位存放一个4字节的有效flag
    u64 alignstep = AIV_FLAG_UB_ALIGN_SIZE / sizeof(int32_t);
    flagStr << stageErrInfo << " Task run failed, para information is deviceId[" << exceptionInfo->deviceid
            << "], streamId[" << exceptionInfo->streamid << "], TaskId[" << exceptionInfo->taskid << "], flag:";
    for (u64 i = 0; (flag_buff_temp != nullptr) && (i < flagCount) && (flagStr.str().size() <= LOG_TMPBUF_SIZE); i++) {
        if (i % alignstep == 0) {
            flagStr << flagMemInt32[i] << " ";
        }
    }
    HCCL_ERROR(flagStr.str().c_str());

    if (flag_buff_temp != nullptr) {
        HrtFreeHost(flag_buff_temp);
    }
    PrintAivPreviousTaskException(exceptionInfo);
}

void TaskExceptionHandler::PrintAivPreviousTaskException(rtExceptionInfo_t* exceptionInfo)
{
    auto stageErrInfo = "[" + LOG_KEYWORDS_TASK_EXEC + "][" + LOG_KEYWORDS_RUN_FAILED + "][" + LOG_KEYWORDS_AIV + "]";
    // 倒序打印前序AIV task信息,找到当前异常task的前50个task(至多)
    auto queue = GlobalMirrorTasks::Instance().GetQueue(exceptionInfo->deviceid, exceptionInfo->streamid);
    if (queue == nullptr) {
        // 未找到异常对应的TaskQueue
        HCCL_ERROR(
            "Exception task queue not found. deviceId[%u], streamId[%u].", exceptionInfo->deviceid,
            exceptionInfo->streamid);
        return;
    }

    u32 taskId = exceptionInfo->taskid;
    auto func = [taskId](const unique_ptr<TaskInfo>& task) {
        return task->taskId_ == taskId;
    };
    auto taskItorPtr = queue->Find(func);
    if (taskItorPtr == nullptr || *taskItorPtr == *queue->End()) {
        // 在队列中未找到异常对应的TaskInfo
        HCCL_ERROR(
            "Exception task not found. deviceId[%u], streamId[%u], taskId[%u].", exceptionInfo->deviceid,
            exceptionInfo->streamid, exceptionInfo->taskid);
        return;
    }

    HCCL_ERROR(
        "%s Task run failed, para information is "
        "deviceId[%u] streamId[%u], TaskId[%u].",
        stageErrInfo.c_str(), exceptionInfo->deviceid, exceptionInfo->streamid, exceptionInfo->taskid);

    for (uint32_t i = 0; i < TASK_CONTEXT_SIZE && *taskItorPtr != *queue->Begin(); --(*taskItorPtr)) {
        if ((**taskItorPtr)->taskId_ > taskId) {
            break;
        }
        if ((**taskItorPtr)->taskId_ != taskId && (**taskItorPtr)->taskParam_.taskType == TaskParamType::TASK_AIV) {
            HCCL_ERROR(
                "%s "
                "previous TaskId[%u],streamId[%u], cmdType[%u], "
                "tag[%u],rank[%u],rankSize[%u], dataCount[%u], numBlocks[%u],"
                "dataType:[%u], beginTime:[%llu], flagMem[%p]",
                stageErrInfo.c_str(), (**taskItorPtr)->taskId_, (**taskItorPtr)->streamId_,
                (**taskItorPtr)->taskParam_.taskPara.Aiv.cmdType, (**taskItorPtr)->taskParam_.taskPara.Aiv.tag,
                (**taskItorPtr)->taskParam_.taskPara.Aiv.rank, (**taskItorPtr)->taskParam_.taskPara.Aiv.rankSize,
                (**taskItorPtr)->taskParam_.taskPara.Aiv.count, (**taskItorPtr)->taskParam_.taskPara.Aiv.numBlocks,
                (**taskItorPtr)->taskParam_.taskPara.Aiv.dataType, (**taskItorPtr)->taskParam_.beginTime,
                (**taskItorPtr)->taskParam_.taskPara.Aiv.flagMem);
        }
        i++;
    }
}

string TaskExceptionHandler::GetGroupRankInfo(const TaskInfo& taskInfo)
{
    if (taskInfo.dfxOpInfo_ == nullptr || taskInfo.dfxOpInfo_->comm_ == nullptr) {
        HCCL_ERROR("[TaskInfo][%s]TaskInfo communicator is nullptr.", __func__);
        return "";
    }
    const CommunicatorImpl* communicator = static_cast<CommunicatorImpl*>(taskInfo.dfxOpInfo_->comm_);
    return StringFormat(
        "group:[%s], rankSize[%u], rankId[%d]", communicator->GetId().c_str(), communicator->GetRankSize(),
        communicator->GetMyRank());
}

inline void PrintBaseErrorLog(const std::string& stageErrInfo, u32 deviceId, const std::string& baseInfo)
{
    HCCL_ERROR(
        "%s Task run failed, base information is deviceID:[%u], %s", stageErrInfo.c_str(), deviceId, baseInfo.c_str());
}

inline void PrintParaErrorLog(const std::string& stageErrInfo, const std::string& paraInfoStr)
{
    HCCL_ERROR("%s Task run failed, para information is %s.", stageErrInfo.c_str(), paraInfoStr.c_str());
}

inline void PrintOpDataErrorLog(const std::string& stageErrInfo, const std::string& opDataContent)
{
    HCCL_ERROR("%s Task run failed, opData information is %s", stageErrInfo.c_str(), opDataContent.c_str());
}

inline void PrintGroupErrorLog(const std::string& stageErrInfo, const std::string& groupRankContent)
{
    HCCL_ERROR("%s Task run failed, groupRank information is %s.", stageErrInfo.c_str(), groupRankContent.c_str());
}

void TaskExceptionHandler::ProcessException(rtExceptionInfo_t* exceptionInfo, const TaskInfo& taskInfo)
{
    HCCL_RUN_INFO("[TaskExceptionHandler][%s]begin to execute hccl task exception callback function.", __func__);
    bool isExistAicpuError = false;
    if (exceptionInfo == nullptr) {
        HCCL_ERROR("[TaskExceptionHandler][ProcessException] exceptionInfo is nullptr.");
        return;
    }
    PrintAicpuErrorMessage(exceptionInfo, isExistAicpuError);
    if (isExistAicpuError) {
        // 如果已经有AICPU上报的task exception, 则host侧无需再次重复上报
        return;
    }
    const auto& logKeywordL2 = taskInfo.taskParam_.taskType == TaskParamType::TASK_NOTIFY_WAIT ?
                                   LOG_KEYWORDS_TIMEOUT :
                                   LOG_KEYWORDS_RUN_FAILED;
    std::string logKeywordL3 = LOG_KEYWORDS_HOST;
    if (taskInfo.taskParam_.taskType == TaskParamType::TASK_AIV) {
        logKeywordL3 = LOG_KEYWORDS_AIV;
    } else if (taskInfo.taskParam_.taskType == TaskParamType::TASK_CCU) {
        logKeywordL3 = LOG_KEYWORDS_CCU;
    }
    auto stageErrInfo = "[" + LOG_KEYWORDS_TASK_EXEC + "][" + logKeywordL2 + "][" + logKeywordL3 + "]";
    HCCL_ERROR("%s Task from HCCL run failed.", stageErrInfo.c_str());
    if (taskInfo.taskParam_.taskType == TaskParamType::TASK_NOTIFY_WAIT) {
        PrintTaskContextInfo(exceptionInfo->deviceid, exceptionInfo->streamid, exceptionInfo->taskid, stageErrInfo);
        HCCL_ERROR("%s EI0002, remoteRank[%u].", stageErrInfo.c_str(), taskInfo.remoteRank_);
        RPT_INPUT_ERR(
            true, "EI0002",
            std::vector<std::string>({"remote_rankid", "base_information", "task_information", "group_rank_content"}),
            std::vector<std::string>(
                {std::to_string(taskInfo.remoteRank_), taskInfo.GetBaseInfo(), taskInfo.GetParaInfo(),
                 GetGroupRankInfo(taskInfo)}));
    }
    PrintBaseErrorLog(stageErrInfo, exceptionInfo->deviceid, taskInfo.GetBaseInfo());
    PrintParaErrorLog(stageErrInfo, taskInfo.GetParaInfo());
    PrintGroupErrorLog(stageErrInfo, GetGroupRankInfo(taskInfo));
    PrintOpCounterInfo(taskInfo);
    PrintOpDataErrorLog(stageErrInfo, taskInfo.GetOpInfo());
}

void TaskExceptionHandler::PrintTaskContextInfo(
    uint32_t deviceId, uint32_t streamId, uint32_t taskId, const std::string& stageErrInfo)
{
    auto queue = GlobalMirrorTasks::Instance().GetQueue(deviceId, streamId);
    if (queue == nullptr) {
        // 未找到异常对应的TaskQueue
        HCCL_ERROR("Exception task queue not found. deviceId[%u], streamId[%u].", deviceId, streamId);
        return;
    }

    auto func = [taskId](const unique_ptr<TaskInfo>& task) {
        return task->taskId_ == taskId;
    };
    auto taskItorPtr = queue->Find(func);
    if (taskItorPtr == nullptr || *taskItorPtr == *queue->End()) {
        // 在队列中未找到异常对应的TaskInfo
        HCCL_ERROR("Exception task not found. deviceId[%u], streamId[%u], taskId[%u].", deviceId, streamId, taskId);
        return;
    }

    // 找到当前异常task的前50个task(至多)
    vector<TaskInfo*> taskContext{};
    for (uint32_t i = 0; i < TASK_CONTEXT_SIZE && *taskItorPtr != *queue->Begin(); ++i, --(*taskItorPtr)) {
        if ((**taskItorPtr)->taskId_ > taskId) {
            break;
        }
        if ((**taskItorPtr)->taskId_ != taskId) {
            taskContext.emplace_back((**taskItorPtr).get());
        }
    }

    if (taskContext.empty()) {
        return;
    }

    HCCL_ERROR(
        "%s Task run failed, context sequence before error task is "
        "[SDMA:M(rank), RDMA:RS(rank,id), SendPayload:SP(rank), InlineReduce:IR(rank), Reduce:R(rank), "
        "NotifyRecord:NR(rank,id), NotifyWait:NW(rank,id), SendNotify:SN(rank,id), "
        "WriteWithNotify:WN(rank,id), WriteReduceWithNotify:WRN(rank,id)]:",
        stageErrInfo.c_str());

    string taskContextInfo = "";
    for (auto it = taskContext.rbegin(); it != taskContext.rend(); ++it) {
        string conciseInfo = (*it)->GetConciseBaseInfo();
        conciseInfo += ",";

        if (taskContextInfo.size() + conciseInfo.size() >= TASK_CONTEXT_INFO_SIZE) {
            HCCL_ERROR("%s %s", stageErrInfo.c_str(), taskContextInfo.c_str());
            taskContextInfo = "";
        }

        taskContextInfo += conciseInfo;
    }
    HCCL_ERROR("%s %s end.", stageErrInfo.c_str(), taskContextInfo.c_str());
}

struct ccum_dfx_info {
    unsigned int query_result; // 0:success, 1:fail
    unsigned int ccum_sqe_recv_cnt;
    unsigned int ccum_sqe_send_cnt;
    unsigned int ccum_mission_dfx;
    unsigned int ccum_sqe_drop_cnt;
    unsigned int ccum_sqe_addr_len_err_drop_cnt;
    unsigned int lqc_ccu_sec_reg0;
    unsigned int ccum_tif_sqe_cnt;
    unsigned int ccum_tif_cqe_cnt;
    unsigned int ccum_cif_sqe_cnt;
    unsigned int ccum_cif_cqe_cnt;
};

void PrintPanicLogInfo(const uint8_t* panicLog)
{
    struct ccum_dfx_info* info = reinterpret_cast<struct ccum_dfx_info*>(const_cast<uint8_t*>(panicLog));
    const uint16_t ccumIsEnable = info->lqc_ccu_sec_reg0 & 1;
    if (info->query_result != 0) {
        HCCL_ERROR("get ccu dfx info fail, ccu dfx info not all correct");
    }
    HCCL_ERROR(
        "CCU DFX INFO: SQE_RECV_CNT[%u] SQE_SEND_CNT[%u] MISSION_DFX[%u] "
        "TIF_SQE_CNT[%u] TIF_CQE_CNT[%u] CIF_SQE_CNT[%u] CIF_CQE_CNT[%u] "
        "SQE_DROP_CNT[%u] SQE_ADDR_LEN_ERR_DROP_CNT[%u] ccumIsEnable[%u]",
        info->ccum_sqe_recv_cnt, info->ccum_sqe_send_cnt, info->ccum_mission_dfx, info->ccum_tif_sqe_cnt,
        info->ccum_tif_cqe_cnt, info->ccum_cif_sqe_cnt, info->ccum_cif_cqe_cnt, info->ccum_sqe_drop_cnt,
        info->ccum_sqe_addr_len_err_drop_cnt, ccumIsEnable);
}

void TaskExceptionHandler::ProcessCcuMC2Exception(rtExceptionInfo_t* exceptionInfo)
{
    set<uint8_t> exDieIds{};
    auto& ccuExDetailInfo = exceptionInfo->expandInfo.u.fusionInfo.u.aicoreCcuInfo.ccuDetailMsg;
    for (uint32_t i = 0; i < ccuExDetailInfo.ccuMissionNum; ++i) {
        const auto& missionInfo = ccuExDetailInfo.missionInfo[i]; // 异常sqe
        HCCL_INFO(
            "[%s] Exception missionInfo: dieId[%u], missionId[%u], startInstrId[%u], status[0x%x], subStatus[0x%x]",
            __func__, missionInfo.dieId, missionInfo.missionId, missionInfo.instrId, missionInfo.status,
            missionInfo.subStatus);
        exDieIds.insert(missionInfo.dieId);
        uint16_t status = static_cast<uint16_t>(missionInfo.status) << BYTE | missionInfo.subStatus;
        if (status == 0) {
            HCCL_INFO("CCU exception status is 0, no need to process.");
            continue;
        }
        PrintPanicLogInfo(missionInfo.panicLog);

        auto serverTaskInfo = MC2GlobalMirrorTasks::GetInstance().GetTaskInfo(
            exceptionInfo->deviceid, missionInfo.dieId, missionInfo.missionId, missionInfo.instrId);
        if (serverTaskInfo == nullptr) {
            HCCL_ERROR(
                "MC2 TaskInfo not found, deviceId[%u], dieId[%u], missionId[%u], instrId[%u].", exceptionInfo->deviceid,
                missionInfo.dieId, missionInfo.missionId, missionInfo.instrId);
            continue;
        }

        DisplayRPCMsg(*serverTaskInfo);
        ParaCcu serverParam = serverTaskInfo->taskParam_.taskPara.Ccu;
        serverParam.execMissionId = missionInfo.missionId;
        vector<CcuErrorInfo> serverErrorInfos{};
        if (GetCcuErrorMsg(
                exceptionInfo->deviceid, status, serverParam, GetGroupRankInfo(*serverTaskInfo), serverErrorInfos)
            != HcclResult::HCCL_SUCCESS) {
            HCCL_ERROR("Get CCU error info failed.");
            continue;
        }

        if (!serverErrorInfos.empty()) {
            HCCL_INFO("Exception instr is in MC2 Server.");
            PrintCcuErrorLog(serverErrorInfos, *serverTaskInfo);
            continue;
        }

        vector<CcuTaskParam> algoTaskParams = GetMC2AlgTaskParam(*serverTaskInfo);
        for (const auto& algoTaskParam : algoTaskParams) {
            HCCL_INFO(
                "MC2 algo TaskParam: dieId[%u], missionId[%u], instrId[%u]", algoTaskParam.dieId,
                algoTaskParam.missionId, algoTaskParam.instStartId);

            auto algoTaskInfo = MC2GlobalMirrorTasks::GetInstance().GetTaskInfo(
                exceptionInfo->deviceid, algoTaskParam.dieId, algoTaskParam.missionId, algoTaskParam.instStartId);
            if (algoTaskInfo == nullptr) {
                HCCL_ERROR(
                    "MC2 TaskInfo not found, deviceId[%u], dieId[%u], missionId[%u], instrId[%u].",
                    exceptionInfo->deviceid, algoTaskParam.dieId, algoTaskParam.missionId, algoTaskParam.instStartId);
                continue;
            }
            ParaCcu algoParam = algoTaskInfo->taskParam_.taskPara.Ccu;
            algoParam.execMissionId = missionInfo.missionId;
            vector<CcuErrorInfo> algoErrorInfos{};
            if (GetCcuErrorMsg(
                    exceptionInfo->deviceid, status, algoParam, GetGroupRankInfo(*algoTaskInfo), algoErrorInfos)
                != HcclResult::HCCL_SUCCESS) {
                HCCL_ERROR("Get CCU error info failed.");
                continue;
            }
            PrintCcuErrorLog(algoErrorInfos, *algoTaskInfo);
        }
    }

    // 清除TaskKill状态, 清除CKE
    const int32_t devLogicId = static_cast<int32_t>(exceptionInfo->deviceid);
    if (CcuCleanTaskKillState(devLogicId) != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR(
            "[TaskExceptionHandler][%s] failed to clean ccu task kill state, "
            "devLogicId[%d].",
            __func__, devLogicId);
    }

    for (const uint8_t dieId : exDieIds) {
        if (CcuCleanDieCkes(devLogicId, dieId) != HcclResult::HCCL_SUCCESS) {
            HCCL_ERROR(
                "[TaskExceptionHandler][%s] failed to clean ccu die ckes, "
                "dieId[%u], devLogicId[%d].",
                __func__, dieId, devLogicId);
        }
    }
}

vector<CcuTaskParam> TaskExceptionHandler::GetMC2AlgTaskParam(const TaskInfo& taskInfo)
{
    if (taskInfo.taskParam_.taskType != TaskParamType::TASK_CCU) {
        HCCL_ERROR("[TaskInfo][%s]Get MC2 Alg TaskParam failed, task type error.", __func__);
        return {};
    }
    if (taskInfo.dfxOpInfo_ == nullptr || taskInfo.dfxOpInfo_->comm_ == nullptr) {
        HCCL_ERROR("[TaskInfo][%s]Get MC2 Alg TaskParam failed, communicator is nullptr.", __func__);
        return {};
    }
    const CommunicatorImpl* communicator = (CommunicatorImpl*)taskInfo.dfxOpInfo_->comm_;
    auto* collServiceBase = communicator->GetCcuCollService();
    if (collServiceBase == nullptr) {
        HCCL_ERROR("[TaskInfo][%s]Failed to get collService from communicator.", __func__);
        return {};
    }
    auto* collServiceCcu = static_cast<CollServiceDeviceMode*>(collServiceBase);
    return collServiceCcu->GetMc2Compont().GetAlgoCcuTaskInfo(taskInfo.taskParam_.taskPara.Ccu.executeId);
}

void TaskExceptionHandler::ProcessCcuException(const rtExceptionInfo_t* exceptionInfo, const TaskInfo& taskInfo)
{
    auto deviceId = exceptionInfo->deviceid;
    auto stageErrInfo = "[" + LOG_KEYWORDS_TASK_EXEC + "][" + LOG_KEYWORDS_RUN_FAILED + "][" + LOG_KEYWORDS_CCU + "]";
    HCCL_ERROR("%s Task from HCCL run failed.", stageErrInfo.c_str());
    PrintBaseErrorLog(stageErrInfo, deviceId, taskInfo.GetBaseInfo());
    PrintGroupErrorLog(stageErrInfo, GetGroupRankInfo(taskInfo));
    PrintOpCounterInfo(taskInfo);
    PrintOpDataErrorLog(stageErrInfo, taskInfo.GetOpInfo());
    auto& ccuExDetailInfo = exceptionInfo->expandInfo.u.ccuInfo;
    for (uint32_t i = 0; i < ccuExDetailInfo.ccuMissionNum; ++i) { // ccuExDetailInfo.ccuMissionNum为1
        const auto& missionInfo = ccuExDetailInfo.missionInfo[i];  // 异常mission
        uint16_t status = static_cast<uint16_t>(missionInfo.status) << BYTE | missionInfo.subStatus;
        std::tuple<std::string, std::string, std::string, std::string> ipInfo
            = TaskExceptionHandler::GetCcuErrorIpInfo(deviceId, status, taskInfo);
        std::string localServerId = std::get<0>(ipInfo);
        std::string localIp = std::get<1>(ipInfo);
        std::string remoteIp = std::get<2>(ipInfo);
        std::string remoteId = std::get<3>(ipInfo);
        RPT_INPUT_ERR(
            true, "EI0018",
            std::vector<std::string>(
                {"localServerId", "localDeviceId", "localDeviceIp", "remoteServerId", "remoteDeviceId",
                 "remoteDeviceIp"}),
            std::vector<std::string>({localServerId, std::to_string(deviceId), localIp, "", remoteId, remoteIp}));
        HCCL_ERROR(
            "%s UB CQE error, localServerId[%s] deviceId[%u] ip[%s], "
            "remoteRankId[%s] ip[%s]",
            stageErrInfo.c_str(), localServerId.c_str(), deviceId, localIp.c_str(), remoteId.c_str(), remoteIp.c_str());

        PrintCcuErrorInfo(deviceId, status, taskInfo);
        // 打印寄存器信息
        PrintPanicLogInfo(missionInfo.panicLog);
    }

    const int32_t devLogicId = static_cast<int32_t>(deviceId);
    if (CcuCleanTaskKillState(devLogicId) != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR(
            "[TaskExceptionHandler][%s] failed to clean ccu task kill state, "
            "devLogicId[%d].",
            __func__, devLogicId);
    }

    const uint8_t dieId = taskInfo.taskParam_.taskPara.Ccu.dieId;
    if (CcuCleanDieCkes(devLogicId, dieId) != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR(
            "[TaskExceptionHandler][%s] failed to clean ccu die ckes, "
            "dieId[%u], devLogicId[%d].",
            __func__, dieId, devLogicId);
    }
}

void TaskExceptionHandler::PrintGroupErrorMessage(
    ErrorMessageReport& errorMessage, [[maybe_unused]] const TaskInfo& exceptionTaskInfo, string& groupRankContent,
    string& stageErrInfo)
{
    groupRankContent += "group:[";
    groupRankContent += std::string(errorMessage.group);
    groupRankContent += "], rankSize[";
    groupRankContent += std::to_string(errorMessage.rankSize);
    groupRankContent += "], localRank[";
    groupRankContent += std::to_string(errorMessage.rankId);
    groupRankContent += "], remoteRank[";
    groupRankContent += std::to_string(errorMessage.remoteUserRank);
    groupRankContent += "]";

    PrintGroupErrorLog(stageErrInfo, groupRankContent);
    return;
}

const std::map<HcclReduceOp, std::string> HCOM_REDUCE_OP_STR_MAP{
    {HcclReduceOp::HCCL_REDUCE_SUM, "sum"},
    {HcclReduceOp::HCCL_REDUCE_PROD, "prod"},
    {HcclReduceOp::HCCL_REDUCE_MAX, "max"},
    {HcclReduceOp::HCCL_REDUCE_MIN, "min"},
    {HcclReduceOp::HCCL_REDUCE_RESERVED, "invalid"}};

inline std::string GetReduceOpEnumStr(HcclReduceOp reduceOp)
{
    auto iter = HCOM_REDUCE_OP_STR_MAP.find(reduceOp);
    if (iter == HCOM_REDUCE_OP_STR_MAP.end()) {
        return "HcclReduceOp(" + std::to_string(reduceOp) + ")";
    } else {
        return iter->second;
    }
}

const std::map<HcclDataType, std::string> HCOM_DATA_TYPE_STR_MAP{
    {HcclDataType::HCCL_DATA_TYPE_INT8, "int8"},       {HcclDataType::HCCL_DATA_TYPE_INT16, "int16"},
    {HcclDataType::HCCL_DATA_TYPE_INT32, "int32"},     {HcclDataType::HCCL_DATA_TYPE_INT64, "int64"},
    {HcclDataType::HCCL_DATA_TYPE_UINT64, "uint64"},   {HcclDataType::HCCL_DATA_TYPE_FP16, "float16"},
    {HcclDataType::HCCL_DATA_TYPE_FP32, "float32"},    {HcclDataType::HCCL_DATA_TYPE_UINT8, "uint8"},
    {HcclDataType::HCCL_DATA_TYPE_UINT16, "uint16"},   {HcclDataType::HCCL_DATA_TYPE_UINT32, "uint32"},
    {HcclDataType::HCCL_DATA_TYPE_FP64, "float64"},    {HcclDataType::HCCL_DATA_TYPE_BFP16, "bfloat16"},
    {HcclDataType::HCCL_DATA_TYPE_INT128, "int128"},   {HcclDataType::HCCL_DATA_TYPE_FP8E4M3, "fp8e4m3"},
    {HcclDataType::HCCL_DATA_TYPE_FP8E5M2, "fp8e5m2"}, {HcclDataType::HCCL_DATA_TYPE_RESERVED, "reserved"}};

inline std::string GetDataTypeEnumStr(HcclDataType dataType)
{
    auto iter = HCOM_DATA_TYPE_STR_MAP.find(dataType);
    if (iter == HCOM_DATA_TYPE_STR_MAP.end()) {
        return "HcclDataType(" + std::to_string(dataType) + ")";
    } else {
        return iter->second;
    }
}

inline std::string GetDataTypeEnumStr(u32 dataType)
{
    auto hcclDataType = static_cast<HcclDataType>(dataType);
    return GetDataTypeEnumStr(hcclDataType);
}

inline std::string GetOpTypeEnumStr(u32 opType)
{
    OpType hcclOpType = static_cast<OpType::Value>(opType);
    return hcclOpType.Describe();
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
    opDataContent += "], opType[";
    opDataContent += GetOpTypeEnumStr(errorMessage.opType);
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

void ReportErrorMsg(
    const TaskInfo& exceptionTaskInfo, const string& groupRankContent, const ErrorMessageReport& errorMessage,
    const rtExceptionInfo_t* exceptionInfo, const std::string& stageErrInfo)
{
    HCCL_INFO("[ReportErrorMsg] start");
    if (exceptionTaskInfo.taskParam_.taskType == TaskParamType::TASK_NOTIFY_WAIT) {
        RPT_INPUT_ERR(
            true, "EI0002",
            std::vector<std::string>({"remote_rankid", "base_information", "task_information", "group_rank_content"}),
            std::vector<std::string>(
                {std::to_string(exceptionTaskInfo.remoteRank_), exceptionTaskInfo.GetBaseInfo().c_str(),
                 (exceptionTaskInfo.GetParaInfo()).c_str(), groupRankContent}));
    } else if (
        exceptionTaskInfo.taskParam_.taskType == TaskParamType::TASK_WRITE_REDUCE_WITH_NOTIFY
        || exceptionTaskInfo.taskParam_.taskType == TaskParamType::TASK_WRITE_WITH_NOTIFY
        || exceptionTaskInfo.taskParam_.taskType == TaskParamType::TASK_UB_INLINE_WRITE
        || exceptionTaskInfo.taskParam_.taskType == TaskParamType::TASK_UB_REDUCE_INLINE
        || exceptionTaskInfo.taskParam_.taskType == TaskParamType::TASK_UB) {
        RPT_INPUT_ERR(
            true, "EI0018",
            std::vector<std::string>(
                {"localServerId", "localDeviceId", "localDeviceIp", "remoteServerId", "remoteDeviceId",
                 "remoteDeviceIp"}),
            std::vector<std::string>(
                {"", std::to_string(exceptionInfo->deviceid), errorMessage.locEid.Describe().c_str(), "", "",
                 errorMessage.rmtEid.Describe().c_str()}));
        HCCL_ERROR(
            "%s UB CQE error, deviceId[%u], localEid[%s], remoteEid[%s]", stageErrInfo.c_str(), exceptionInfo->deviceid,
            errorMessage.locEid.Describe().c_str(), errorMessage.rmtEid.Describe().c_str());
    }
}

void GetTaskParam(TaskParam& taskParam, const ErrorMessageReport& errorMessage)
{
    if (errorMessage.taskType == TaskParamType::TASK_NOTIFY_WAIT) {
        taskParam.taskPara.Notify.notifyID = errorMessage.notifyId;
        taskParam.taskPara.Notify.value = errorMessage.notifyValue;
    } else if (
        errorMessage.taskType == TaskParamType::TASK_UB_REDUCE_INLINE
        || errorMessage.taskType == TaskParamType::TASK_WRITE_REDUCE_WITH_NOTIFY) {
        taskParam.taskPara.Reduce.notifyID = errorMessage.notifyId;
        taskParam.taskPara.Reduce.notifyValue = errorMessage.notifyValue;
        taskParam.taskPara.Reduce.src = reinterpret_cast<void*>(errorMessage.taskSrcAddr);
        taskParam.taskPara.Reduce.dst = reinterpret_cast<void*>(errorMessage.taskDstAddr);
        taskParam.taskPara.Reduce.linkType = errorMessage.linkType;
        taskParam.taskPara.Reduce.size = errorMessage.size;
    } else if (
        errorMessage.taskType == TaskParamType::TASK_UB_INLINE_WRITE
        || errorMessage.taskType == TaskParamType::TASK_WRITE_WITH_NOTIFY) {
        taskParam.taskPara.DMA.notifyID = errorMessage.notifyId;
        taskParam.taskPara.DMA.notifyValue = errorMessage.notifyValue;
        taskParam.taskPara.DMA.src = reinterpret_cast<void*>(errorMessage.taskSrcAddr);
        taskParam.taskPara.DMA.dst = reinterpret_cast<void*>(errorMessage.taskDstAddr);
        taskParam.taskPara.DMA.linkType = errorMessage.linkType;
        taskParam.taskPara.DMA.size = errorMessage.size;
    }
}

void TaskExceptionHandler::PrintAicpuErrorMessage(rtExceptionInfo_t* exceptionInfo, bool& isExistAicpuError)
{
    ErrorMessageReport errorMessage;
    unique_lock<std::mutex> lock(Hccl::g_commHadCallbackArrayMutexV2);
    if (Hccl::g_commHadCallbackArrayV2[exceptionInfo->deviceid]) {
        // 防止同一个device上出现通信主流和kernel流均出现task exception时runtime调用两次callback
        // HDC通道信息不是读清，防止aicpu task exception重复上报
        HCCL_WARNING("aicpu error message has been reported. deviceid[%u]", exceptionInfo->deviceid);
        return;
    }
    lock.unlock();
    if (Hccl::g_communicatorCallbackMapV2[exceptionInfo->deviceid].find(exceptionInfo->streamid)
        != Hccl::g_communicatorCallbackMapV2[exceptionInfo->deviceid].end()) {
        // 找到对应的通信域，并调用回调函数从HDC通道获取AICPU异常信息
        errorMessage = (Hccl::g_communicatorCallbackMapV2[exceptionInfo->deviceid])[exceptionInfo->streamid]();
        if (strlen(errorMessage.tag) > 0) {
            isExistAicpuError = true;
            std::string groupRankContent;
            u32 streamId = static_cast<u32>(errorMessage.streamId);
            TaskParam taskParam{};
            taskParam.taskType = errorMessage.taskType;

            GetTaskParam(taskParam, errorMessage);

            std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
            dfxOpInfo->tag_ = std::string(errorMessage.tag);
            TaskInfo exceptionTaskInfo(
                streamId, errorMessage.taskId, errorMessage.remoteUserRank, taskParam, dfxOpInfo);
            auto logKeywordL2 = exceptionTaskInfo.taskParam_.taskType == TaskParamType::TASK_NOTIFY_WAIT ?
                                    LOG_KEYWORDS_TIMEOUT :
                                    LOG_KEYWORDS_RUN_FAILED;
            auto stageErrInfo = "[" + LOG_KEYWORDS_TASK_EXEC + "][" + logKeywordL2 + "][" + LOG_KEYWORDS_AICPU + "]";
            HCCL_ERROR("%s Task from HCCL run failed.", stageErrInfo.c_str());
            // 防止tag字符串过长， 信息分开打印
            PrintBaseErrorLog(stageErrInfo, exceptionInfo->deviceid, exceptionTaskInfo.GetBaseInfo());
            PrintParaErrorLog(stageErrInfo, exceptionTaskInfo.GetParaInfo());
            PrintGroupErrorMessage(errorMessage, exceptionTaskInfo, groupRankContent, stageErrInfo);
            PrintOpDataErrorMessage(exceptionInfo->deviceid, errorMessage, stageErrInfo);
            HCCL_ERROR(
                "errorMessage taskType[%s], rtCqErrorType[%u], rtCqErrorCode[%u]. ",
                errorMessage.taskType.Describe().c_str(), static_cast<u32>(errorMessage.rtCqErrorType),
                errorMessage.rtCqErrorCode);

            // 打印UB DFX寄存器信息
            if (errorMessage.taskType == TaskParamType::TASK_WRITE_WITH_NOTIFY
                || errorMessage.taskType == TaskParamType::TASK_WRITE_REDUCE_WITH_NOTIFY
                || errorMessage.taskType == TaskParamType::TASK_UB_INLINE_WRITE
                || errorMessage.taskType == TaskParamType::TASK_UB_REDUCE_INLINE
                || errorMessage.taskType == TaskParamType::TASK_UB) {
                HCCL_ERROR(
                    "errorMessage ubCqeStatus[%u], localEid[%s], remoteEid[%s]. ",
                    static_cast<u32>(errorMessage.ubCqeStatus), errorMessage.locEid.Describe().c_str(),
                    errorMessage.rmtEid.Describe().c_str());
                auto addr = IpAddress(errorMessage.locEid);
                u32 devPhyId = HrtGetDevicePhyIdByUserDevId(exceptionInfo->deviceid);
                auto rdmaHandle = RdmaHandleManager::GetInstance().GetByIp(devPhyId, addr);
                PrintUbRegisters(static_cast<s32>(exceptionInfo->deviceid), rdmaHandle);
            }

            ReportErrorMsg(exceptionTaskInfo, groupRankContent, errorMessage, exceptionInfo, stageErrInfo);

            lock.lock();
            Hccl::g_commHadCallbackArrayV2[exceptionInfo->deviceid] = true;
        } else {
            HCCL_WARNING("PrintAicpuErrorMessage No Valid errorMessage!");
        }
    } else {
        HCCL_INFO("PrintAicpuErrorMessage streamId[%u] is not found.", exceptionInfo->streamid);
    }
}

void TaskExceptionHandler::PrintCcuErrorInfo(uint32_t deviceId, uint16_t status, const TaskInfo& taskInfo)
{
    const ParaCcu& ccuTaskParam = taskInfo.taskParam_.taskPara.Ccu;
    vector<CcuErrorInfo> errorInfos{};
    HcclResult ret = GetCcuErrorMsg(deviceId, status, ccuTaskParam, GetGroupRankInfo(taskInfo), errorInfos);
    const uint8_t missionStatus = (status >> 8) & 0xFF;
    if (ret != HcclResult::HCCL_SUCCESS || errorInfos.empty()) {
        HCCL_ERROR(
            "Get CCU error info failed. deviceId[%u], dieId[%u], missionId[%u], executeId[%llu].", deviceId,
            ccuTaskParam.dieId, ccuTaskParam.missionId, ccuTaskParam.executeId);
        return;
    }
    PrintCcuErrorLog(errorInfos, taskInfo);

    if (missionStatus >= 0x01
        && missionStatus <= 0x05) { // 如果是UB错误(missionStatus为[0x01, 0x05])，打印Ub Dfx寄存器信息
        PrintCcuUbRegisters(static_cast<s32>(deviceId), taskInfo.taskParam_.taskPara.Ccu);
    }
}

void TaskExceptionHandler::PrintCcuErrorLog(const std::vector<CcuErrorInfo>& errorInfos, const TaskInfo& taskInfo)
{
    if (errorInfos.empty()) {
        return;
    }
    auto stageErrInfo = "[" + LOG_KEYWORDS_TASK_EXEC + "][" + LOG_KEYWORDS_RUN_FAILED + "][" + LOG_KEYWORDS_CCU + "]";
    HCCL_ERROR("%s Task run failed, ccu runtime information is:", stageErrInfo.c_str());
    for (const auto& errorInfo : errorInfos) {
        HCCL_ERROR("%s [%s]", stageErrInfo.c_str(), GetCcuErrorMsgByType(errorInfo, taskInfo).c_str());
    }
}

string TaskExceptionHandler::GetCcuLenErrorMsg(const uint64_t len)
{
    if ((0 < len) && (len <= CCU_MSG_256MB_LEN)) {
        return "";
    }
    return StringFormat("ccu transMem Len[%llu]B > 256MB or is zero, not support!", len);
}

string TaskExceptionHandler::GetCcuErrorMsgLoop(const CcuErrorInfo& ccuErrorInfo, const TaskInfo& taskInfo)
{
    (void)taskInfo;
    return StringFormat(
        "InstrId[%u]: Loop startInstrId[%u], endInstrId[%u], executorId[%u], "
        "totalIter[%u], curIter[%u], addressStride[0x%x]",
        ccuErrorInfo.instrId, ccuErrorInfo.msg.loop.startInstrId, ccuErrorInfo.msg.loop.endInstrId,
        ccuErrorInfo.msg.loop.loopEngineId, ccuErrorInfo.msg.loop.loopCnt, ccuErrorInfo.msg.loop.loopCurrentCnt,
        ccuErrorInfo.msg.loop.addrStride);
}

string TaskExceptionHandler::GetCcuErrorMsgLoopGroup(const CcuErrorInfo& ccuErrorInfo, const TaskInfo& taskInfo)
{
    (void)taskInfo;
    return StringFormat(
        "InstrId[%u]: LoopGroup startLoopInsId[%u], loopInsCnt[%u], "
        "expandOffset[%u], expandCnt[%u]",
        ccuErrorInfo.instrId, ccuErrorInfo.msg.loopGroup.startLoopInsId, ccuErrorInfo.msg.loopGroup.loopInsCnt,
        ccuErrorInfo.msg.loopGroup.expandOffset, ccuErrorInfo.msg.loopGroup.expandCnt);
}

string TaskExceptionHandler::GetCcuErrorMsgLocPostSem(const CcuErrorInfo& ccuErrorInfo, const TaskInfo& taskInfo)
{
    (void)taskInfo;
    return StringFormat(
        "InstrId[%u]: Set sem[%u], semValue[0x%04x], mask[0x%04x]", ccuErrorInfo.instrId,
        ccuErrorInfo.msg.waitSignal.signalId, ccuErrorInfo.msg.waitSignal.signalValue,
        ccuErrorInfo.msg.waitSignal.signalMask);
}

string TaskExceptionHandler::GetCcuErrorMsgLocWaitSem(const CcuErrorInfo& ccuErrorInfo, const TaskInfo& taskInfo)
{
    (void)taskInfo;
    return StringFormat(
        "InstrId[%u]: Wait sem[%u], semValue[0x%04x], mask[0x%04x]", ccuErrorInfo.instrId,
        ccuErrorInfo.msg.waitSignal.signalId, ccuErrorInfo.msg.waitSignal.signalValue,
        ccuErrorInfo.msg.waitSignal.signalMask);
}

string TaskExceptionHandler::GetCcuErrorMsgRemPostSem(const CcuErrorInfo& ccuErrorInfo, const TaskInfo& taskInfo)
{
    return StringFormat(
        "InstrId[%u]: Post, Use sem[%u], mask[0x%04x], rankId[%d]", ccuErrorInfo.instrId,
        ccuErrorInfo.msg.waitSignal.signalId, ccuErrorInfo.msg.waitSignal.signalMask,
        GetRankIdByChannelId(ccuErrorInfo.msg.waitSignal.channelId[0], taskInfo));
}

string TaskExceptionHandler::GetCcuErrorMsgRemWaitSem(const CcuErrorInfo& ccuErrorInfo, const TaskInfo& taskInfo)
{
    return StringFormat(
        "InstrId[%u]: Wait, Use sem[%u], semValue[0x%04x], mask[0x%04x], rankId[%d]", ccuErrorInfo.instrId,
        ccuErrorInfo.msg.waitSignal.signalId, ccuErrorInfo.msg.waitSignal.signalValue,
        ccuErrorInfo.msg.waitSignal.signalMask,
        GetRankIdByChannelId(ccuErrorInfo.msg.waitSignal.channelId[0], taskInfo));
}

string TaskExceptionHandler::GetCcuErrorMsgRemPostVar(const CcuErrorInfo& ccuErrorInfo, const TaskInfo& taskInfo)
{
    return StringFormat(
        "InstrId[%u]: Post Variable[0x%016llx] To Param[%u], Use sem[%u], mask[0x%04x], rankId[%d]",
        ccuErrorInfo.instrId, ccuErrorInfo.msg.waitSignal.paramValue, ccuErrorInfo.msg.waitSignal.paramId,
        ccuErrorInfo.msg.waitSignal.signalId, ccuErrorInfo.msg.waitSignal.signalMask,
        GetRankIdByChannelId(ccuErrorInfo.msg.waitSignal.channelId[0], taskInfo));
}

string TaskExceptionHandler::GetCcuErrorMsgRemWaitGroup(const CcuErrorInfo& ccuErrorInfo, const TaskInfo& taskInfo)
{
    stringstream ranks;
    for (uint32_t i = 0; i < WAIT_SIGNAL_CHANNEL_SIZE; ++i) {
        const auto channelId = ccuErrorInfo.msg.waitSignal.channelId[i];
        if (channelId == UINT16_MAX) {
            break;
        }
        const auto rankId = GetRankIdByChannelId(channelId, taskInfo);
        if (i != 0) {
            ranks << ", ";
        }
        ranks << to_string(rankId);
    }
    return StringFormat(
        "InstrId[%u]: Wait Group, Use sem[%u], semValue[0x%04x], mask[0x%04x], rankIds[%s]", ccuErrorInfo.instrId,
        ccuErrorInfo.msg.waitSignal.signalId, ccuErrorInfo.msg.waitSignal.signalValue,
        ccuErrorInfo.msg.waitSignal.signalMask, ranks.str().c_str());
}

string TaskExceptionHandler::GetCcuErrorMsgPostSharedVar(const CcuErrorInfo& ccuErrorInfo, const TaskInfo& taskInfo)
{
    (void)taskInfo;
    return StringFormat(
        "InstrId[%u]: Post Shared Variable[%u] from Variable[0x%016llx], "
        "Use sem[%u], mask[0x%04x]",
        ccuErrorInfo.instrId, ccuErrorInfo.msg.waitSignal.paramId, ccuErrorInfo.msg.waitSignal.paramValue,
        ccuErrorInfo.msg.waitSignal.signalId, ccuErrorInfo.msg.waitSignal.signalMask);
}

string TaskExceptionHandler::GetCcuErrorMsgPostSharedSem(const CcuErrorInfo& ccuErrorInfo, const TaskInfo& taskInfo)
{
    (void)taskInfo;
    return StringFormat(
        "InstrId[%u]: Post, Use sem[%u], mask[0x%04x]", ccuErrorInfo.instrId, ccuErrorInfo.msg.waitSignal.signalId,
        ccuErrorInfo.msg.waitSignal.signalMask);
}

string TaskExceptionHandler::GetCcuErrorMsgRead(const CcuErrorInfo& ccuErrorInfo, const TaskInfo& taskInfo)
{
    auto pair = GetAddrPairByChannelId(ccuErrorInfo.msg.transMem.channelId, taskInfo);
    string printMsg = GetCcuLenErrorMsg(ccuErrorInfo.msg.transMem.len);
    return StringFormat(
        "InstrId[%u]: Read Memory[0x%016llx] To Memory[0x%016llx], Len[%llu], "
        "Set sem[%u] with mask[0x%04x], remoteRankId[%d], srcEID[%s], dstEID[%s] %s",
        ccuErrorInfo.instrId, ccuErrorInfo.msg.transMem.rmtAddr, ccuErrorInfo.msg.transMem.locAddr,
        ccuErrorInfo.msg.transMem.len, ccuErrorInfo.msg.transMem.signalId, ccuErrorInfo.msg.transMem.signalMask,
        GetRankIdByChannelId(ccuErrorInfo.msg.transMem.channelId, taskInfo), pair.first.Describe().c_str(),
        pair.second.Describe().c_str(), printMsg.c_str());
}

string TaskExceptionHandler::GetCcuErrorMsgWrite(const CcuErrorInfo& ccuErrorInfo, const TaskInfo& taskInfo)
{
    auto pair = GetAddrPairByChannelId(ccuErrorInfo.msg.transMem.channelId, taskInfo);
    string printMsg = GetCcuLenErrorMsg(ccuErrorInfo.msg.transMem.len);
    return StringFormat(
        "InstrId[%u]: Write Memory[0x%016llx] to Memory[0x%016llx], Len[%llu], "
        "Set sem[%u] with mask[0x%04x], remoteRankId[%d], srcEID[%s], dstEID[%s] %s",
        ccuErrorInfo.instrId, ccuErrorInfo.msg.transMem.locAddr, ccuErrorInfo.msg.transMem.rmtAddr,
        ccuErrorInfo.msg.transMem.len, ccuErrorInfo.msg.transMem.signalId, ccuErrorInfo.msg.transMem.signalMask,
        GetRankIdByChannelId(ccuErrorInfo.msg.transMem.channelId, taskInfo), pair.first.Describe().c_str(),
        pair.second.Describe().c_str(), printMsg.c_str());
}

string TaskExceptionHandler::GetCcuErrorMsgLocalCpy(const CcuErrorInfo& ccuErrorInfo, const TaskInfo& taskInfo)
{
    (void)taskInfo;
    string printMsg = GetCcuLenErrorMsg(ccuErrorInfo.msg.transMem.len);
    return StringFormat(
        "InstrId[%u]: Read Memory[0x%016llx] to Memory[0x%016llx], Len[%llu], "
        "Set sem[%u] with mask[0x%04x] %s",
        ccuErrorInfo.instrId, ccuErrorInfo.msg.transMem.locAddr, ccuErrorInfo.msg.transMem.rmtAddr,
        ccuErrorInfo.msg.transMem.len, ccuErrorInfo.msg.transMem.signalId, ccuErrorInfo.msg.transMem.signalMask,
        printMsg.c_str());
}

string TaskExceptionHandler::GetCcuErrorMsgLocalReduce(const CcuErrorInfo& ccuErrorInfo, const TaskInfo& taskInfo)
{
    (void)taskInfo;
    string printMsg = GetCcuLenErrorMsg(ccuErrorInfo.msg.transMem.len);
    return StringFormat(
        "InstrId[%u]: Read Memory[0x%016llx] to Memory[0x%016llx], Len[%llu], "
        "Set sem[%u] with mask[0x%04x], dataType[%u], opType[%u] %s",
        ccuErrorInfo.instrId, ccuErrorInfo.msg.transMem.locAddr, ccuErrorInfo.msg.transMem.rmtAddr,
        ccuErrorInfo.msg.transMem.len, ccuErrorInfo.msg.transMem.signalId, ccuErrorInfo.msg.transMem.signalMask,
        ccuErrorInfo.msg.transMem.dataType, ccuErrorInfo.msg.transMem.opType, printMsg.c_str());
}

string TaskExceptionHandler::GetCcuErrorMsgBufRead(const CcuErrorInfo& ccuErrorInfo, const TaskInfo& taskInfo)
{
    auto pair = GetAddrPairByChannelId(ccuErrorInfo.msg.bufTransMem.channelId, taskInfo);
    string printMsg = GetCcuLenErrorMsg(ccuErrorInfo.msg.bufTransMem.len);
    return StringFormat(
        "InstrId[%u]: Read Rmt Mem[0x%016llx] To CcuBuffer[%u], Len[%llu], "
        "sem[%u], mask[0x%04x], remoteRankId[%d], srcEID[%s], dstEID[%s] %s",
        ccuErrorInfo.instrId, ccuErrorInfo.msg.bufTransMem.addr, ccuErrorInfo.msg.bufTransMem.bufId,
        ccuErrorInfo.msg.bufTransMem.len, ccuErrorInfo.msg.bufTransMem.signalId,
        ccuErrorInfo.msg.bufTransMem.signalMask, GetRankIdByChannelId(ccuErrorInfo.msg.bufTransMem.channelId, taskInfo),
        pair.first.Describe().c_str(), pair.second.Describe().c_str(), printMsg.c_str());
}

string TaskExceptionHandler::GetCcuErrorMsgBufWrite(const CcuErrorInfo& ccuErrorInfo, const TaskInfo& taskInfo)
{
    auto pair = GetAddrPairByChannelId(ccuErrorInfo.msg.bufTransMem.channelId, taskInfo);
    string printMsg = GetCcuLenErrorMsg(ccuErrorInfo.msg.bufTransMem.len);
    return StringFormat(
        "InstrId[%u]: Write CcuBuffer[%u] To Rmt Mem[0x%016llx], Len[%llu], "
        "sem[%u], mask[0x%04x], remoteRankId[%d], srcEID[%s], dstEID[%s] %s",
        ccuErrorInfo.instrId, ccuErrorInfo.msg.bufTransMem.bufId, ccuErrorInfo.msg.bufTransMem.addr,
        ccuErrorInfo.msg.bufTransMem.len, ccuErrorInfo.msg.bufTransMem.signalId,
        ccuErrorInfo.msg.bufTransMem.signalMask, GetRankIdByChannelId(ccuErrorInfo.msg.bufTransMem.channelId, taskInfo),
        pair.first.Describe().c_str(), pair.second.Describe().c_str(), printMsg.c_str());
}

string TaskExceptionHandler::GetCcuErrorMsgBufLocRead(const CcuErrorInfo& ccuErrorInfo, const TaskInfo& taskInfo)
{
    (void)taskInfo;
    string printMsg = GetCcuLenErrorMsg(ccuErrorInfo.msg.bufTransMem.len);
    return StringFormat(
        "InstrId[%u]: Read Loc Mem[0x%016llx] To CcuBuffer[%u], Len[%llu], sem[%u], mask[0x%04x] %s",
        ccuErrorInfo.instrId, ccuErrorInfo.msg.bufTransMem.addr, ccuErrorInfo.msg.bufTransMem.bufId,
        ccuErrorInfo.msg.bufTransMem.len, ccuErrorInfo.msg.bufTransMem.signalId,
        ccuErrorInfo.msg.bufTransMem.signalMask, printMsg.c_str());
}

string TaskExceptionHandler::GetCcuErrorMsgBufLocWrite(const CcuErrorInfo& ccuErrorInfo, const TaskInfo& taskInfo)
{
    (void)taskInfo;
    string printMsg = GetCcuLenErrorMsg(ccuErrorInfo.msg.bufTransMem.len);
    return StringFormat(
        "InstrId[%u]: Write CcuBuffer[%u] To Loc Mem[0x%016llx], Len[%llu], sem[%u], mask[0x%04x] %s",
        ccuErrorInfo.instrId, ccuErrorInfo.msg.bufTransMem.bufId, ccuErrorInfo.msg.bufTransMem.addr,
        ccuErrorInfo.msg.bufTransMem.len, ccuErrorInfo.msg.bufTransMem.signalId,
        ccuErrorInfo.msg.bufTransMem.signalMask, printMsg.c_str());
}

string TaskExceptionHandler::GetCcuErrorMsgBufReduce(const CcuErrorInfo& ccuErrorInfo, const TaskInfo& taskInfo)
{
    (void)taskInfo;
    stringstream buffIds;
    for (uint32_t i = 0; i < BUF_REDUCE_ID_SIZE; ++i) {
        const auto buffId = ccuErrorInfo.msg.bufReduce.bufIds[i];
        if (buffId == UINT16_MAX) {
            break;
        }
        if (i != 0) {
            buffIds << ", ";
        }
        buffIds << to_string(buffId);
    }

    return StringFormat(
        "InstrId[%u]: Buffer Reduce count[%u], dataType[%u], outputDataType[%u], opType[%u], "
        "sem[%u], mask[0x%04x], CcuBuffers[%s]",
        ccuErrorInfo.instrId, ccuErrorInfo.msg.bufReduce.count, ccuErrorInfo.msg.bufReduce.dataType,
        ccuErrorInfo.msg.bufReduce.outputDataType, ccuErrorInfo.msg.bufReduce.opType,
        ccuErrorInfo.msg.bufReduce.signalId, ccuErrorInfo.msg.bufReduce.signalMask, buffIds.str().c_str());
}

string TaskExceptionHandler::GetCcuErrorMsgDefault(const CcuErrorInfo& ccuErrorInfo)
{
    return StringFormat("InstrId[%u]: CcuErrorType[%s]", ccuErrorInfo.instrId, ccuErrorInfo.type.Describe().c_str());
}

string TaskExceptionHandler::GetCcuErrorMsgMission(const CcuErrorInfo& ccuErrorInfo)
{
    return StringFormat(
        "InstrId[%u]: dieId[%u], missionId[%u], missionError[%s]", ccuErrorInfo.instrId, ccuErrorInfo.dieId,
        ccuErrorInfo.missionId, ccuErrorInfo.msg.mission.missionError);
}

string TaskExceptionHandler::GetCcuErrorMsgByType(const CcuErrorInfo& ccuErrorInfo, const TaskInfo& taskInfo)
{
    if (ccuErrorInfo.type == CcuErrorType::MISSION) {
        return GetCcuErrorMsgMission(ccuErrorInfo);
    }

    using GetCcuErrorMsgFunc = string (*)(const CcuErrorInfo& ccuErrorInfo, const TaskInfo& taskInfo);
    static const map<CcuRepType, GetCcuErrorMsgFunc> handlerMap{
        {CcuRepType::LOOP, &TaskExceptionHandler::GetCcuErrorMsgLoop},
        {CcuRepType::LOOPGROUP, &TaskExceptionHandler::GetCcuErrorMsgLoopGroup},
        {CcuRepType::LOC_POST_SEM, &TaskExceptionHandler::GetCcuErrorMsgLocPostSem},
        {CcuRepType::LOC_WAIT_SEM, &TaskExceptionHandler::GetCcuErrorMsgLocWaitSem},
        {CcuRepType::REM_POST_SEM, &TaskExceptionHandler::GetCcuErrorMsgRemPostSem},
        {CcuRepType::REM_WAIT_SEM, &TaskExceptionHandler::GetCcuErrorMsgRemWaitSem},
        {CcuRepType::REM_POST_VAR, &TaskExceptionHandler::GetCcuErrorMsgRemPostVar},
        {CcuRepType::REM_WAIT_GROUP, &TaskExceptionHandler::GetCcuErrorMsgRemWaitGroup},
        {CcuRepType::POST_SHARED_VAR, &TaskExceptionHandler::GetCcuErrorMsgPostSharedVar},
        {CcuRepType::POST_SHARED_SEM, &TaskExceptionHandler::GetCcuErrorMsgPostSharedSem},
        {CcuRepType::READ, &TaskExceptionHandler::GetCcuErrorMsgRead},
        {CcuRepType::WRITE, &TaskExceptionHandler::GetCcuErrorMsgWrite},
        {CcuRepType::LOCAL_CPY, &TaskExceptionHandler::GetCcuErrorMsgLocalCpy},
        {CcuRepType::LOCAL_REDUCE, &TaskExceptionHandler::GetCcuErrorMsgLocalReduce},
        {CcuRepType::BUF_READ, &TaskExceptionHandler::GetCcuErrorMsgBufRead},
        {CcuRepType::BUF_WRITE, &TaskExceptionHandler::GetCcuErrorMsgBufWrite},
        {CcuRepType::BUF_LOC_READ, &TaskExceptionHandler::GetCcuErrorMsgBufLocRead},
        {CcuRepType::BUF_LOC_WRITE, &TaskExceptionHandler::GetCcuErrorMsgBufLocWrite},
        {CcuRepType::BUF_REDUCE, &TaskExceptionHandler::GetCcuErrorMsgBufReduce}};

    const auto funcIt = handlerMap.find(ccuErrorInfo.repType);
    if (funcIt == handlerMap.end()) {
        return GetCcuErrorMsgDefault(ccuErrorInfo);
    } else {
        return funcIt->second(ccuErrorInfo, taskInfo);
    }
}

void TaskExceptionHandler::DisplayRPCMsg(const TaskInfo& taskInfo)
{
    if (taskInfo.taskParam_.taskType != TaskParamType::TASK_CCU) {
        HCCL_ERROR("[TaskInfo][%s]Get MC2 Alg TaskParam failed, task type error.", __func__);
        return;
    }
    if (taskInfo.dfxOpInfo_ == nullptr || taskInfo.dfxOpInfo_->comm_ == nullptr) {
        HCCL_ERROR("[TaskInfo][%s]Get MC2 Alg TaskParam failed, communicator is nullptr.", __func__);
        return;
    }
    const CommunicatorImpl* communicator = static_cast<CommunicatorImpl*>(taskInfo.dfxOpInfo_->comm_);
    auto* collServiceBase = communicator->GetCcuCollService();
    if (collServiceBase == nullptr) {
        HCCL_ERROR("[TaskInfo][%s]Failed to get collService from communicator.", __func__);
        return;
    }
    auto* collServiceCcu = static_cast<CollServiceDeviceMode*>(collServiceBase);
    collServiceCcu->GetMc2Compont().DisplayRPCMsg();
}

RankId TaskExceptionHandler::GetRankIdByChannelId(uint16_t channelId, const TaskInfo& taskInfo)
{
    if (taskInfo.taskParam_.taskType != TaskParamType::TASK_CCU) {
        HCCL_ERROR("[TaskException][%s]Get RankId failed, task type error.", __func__);
        return INVALID_RANKID;
    }
    if (taskInfo.dfxOpInfo_ == nullptr || taskInfo.dfxOpInfo_->comm_ == nullptr) {
        HCCL_ERROR("[TaskException][%s]Get RankId failed, communicator is nullptr.", __func__);
        return INVALID_RANKID;
    }
    const CommunicatorImpl* communicator = (CommunicatorImpl*)taskInfo.dfxOpInfo_->comm_;
    auto* collServiceBase = communicator->GetCcuCollService();
    if (collServiceBase == nullptr) {
        HCCL_ERROR("[TaskException][%s]Failed to get collService from communicator.", __func__);
        return INVALID_RANKID;
    }
    auto* collServiceCcu = static_cast<CollServiceDeviceMode*>(collServiceBase);
    const uint8_t dieId = taskInfo.taskParam_.taskPara.Ccu.dieId;
    return collServiceCcu->GetCcuInsPreprocessor()->GetCcuComm()->GetCcuJettyMgr()->GetRemoteRankIdByChannelId(
        dieId, channelId);
}

std::pair<IpAddress, IpAddress>
TaskExceptionHandler::GetAddrPairByChannelId(uint16_t channelId, const TaskInfo& taskInfo)
{
    std::pair<IpAddress, IpAddress> dummy = {IpAddress(), IpAddress()};
    if (taskInfo.taskParam_.taskType != TaskParamType::TASK_CCU) {
        HCCL_ERROR(
            "[TaskException][%s]Get AddrPair failed, task type error[%s]", __func__,
            taskInfo.taskParam_.Describe().c_str());
        return dummy;
    }
    if (taskInfo.dfxOpInfo_ == nullptr || taskInfo.dfxOpInfo_->comm_ == nullptr) {
        HCCL_ERROR("[TaskException][%s]Get AddrPair failed, communicator is nullptr.", __func__);
        return dummy;
    }
    const CommunicatorImpl* communicator = (CommunicatorImpl*)taskInfo.dfxOpInfo_->comm_;
    auto* collServiceBase = communicator->GetCcuCollService();
    if (collServiceBase == nullptr) {
        HCCL_ERROR("[TaskException][%s]Failed to get collService from communicator.", __func__);
        return dummy;
    }
    auto* collServiceCcu = static_cast<CollServiceDeviceMode*>(collServiceBase);
    const uint8_t dieId = taskInfo.taskParam_.taskPara.Ccu.dieId;
    return collServiceCcu->GetCcuInsPreprocessor()->GetCcuComm()->GetCcuJettyMgr()->GetAddrPairByChannelId(
        dieId, channelId);
}

std::tuple<std::string, std::string, std::string, std::string> TaskExceptionHandler::GetCcuErrorIpInfo(
    [[maybe_unused]] uint32_t deviceId, [[maybe_unused]] uint16_t status, const TaskInfo& taskInfo)
{
    std::string localServerId = "";
    std::string localIp = "";
    std::string remoteIp = "";
    std::string remoteId = "";

    char serverIdBuf[64] = {0};
    if (get_server_id(serverIdBuf, sizeof(serverIdBuf)) == 0) {
        localServerId = serverIdBuf;
    }

    auto ccuDetailInfo = taskInfo.taskParam_.ccuDetailInfo;
    if (ccuDetailInfo != nullptr && !ccuDetailInfo->empty()
        && ccuDetailInfo->at(0).channelId[0] != INVALID_VALUE_CHANNELID) {
        uint16_t channelId = ccuDetailInfo->at(0).channelId[0];
        auto addrPair = GetAddrPairByChannelId(channelId, taskInfo);
        localIp = addrPair.first.Describe();
        remoteIp = addrPair.second.Describe();
        remoteId = std::to_string(taskInfo.remoteRank_);
    }
    return std::make_tuple(localServerId, localIp, remoteIp, remoteId);
}

} // namespace Hccl
