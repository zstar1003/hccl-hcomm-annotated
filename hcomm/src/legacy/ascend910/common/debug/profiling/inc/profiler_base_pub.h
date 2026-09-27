/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PROFILER_BASE_PUB_H
#define PROFILER_BASE_PUB_H

#include <memory>
#include <mutex>
#include <string>
#include <map>
#include <thread>

#include <hccl/hccl_types.h>
#include "hccl_common.h"
#include "common.h"
#include "workflow_pub.h"
#include "dispatcher_task_types.h"
#include "alg_profiling.h"
#include "profiler_base_pub_extend.h"

namespace hccl {
enum class StepType { STEP_STAGE = 0, STEP_STEP, STEP_MAX };

enum class OpDict { SUM = 0, PROD, MAX, MIN };

enum class DataType { DINT8 = 0, DINT16, DINT32, DFP16, DFP32, DINT64, DUINT64 };

struct GroupRankInfo {
    u32 rankSize{0};
    u32 rankId{0};
    u32 remoteRankId{INVALID_VALUE_RANKSIZE};
};

struct OpDataInfo {
    u64 count{0};
    const void* src{nullptr};
    const void* dst{nullptr};
    u32 index{0};
    u32 rootId{0};
    u32 deviceId{0};
    HcclDataType dataType{HcclDataType::HCCL_DATA_TYPE_RESERVED};
    HcclReduceOp reduceType{HcclReduceOp::HCCL_REDUCE_RESERVED};
    struct timeval tv {};
};

struct StreamRecordInfo {
    s32 planeId;
    AlgType algType;
    std::string tag;
    StreamRecordInfo() = default;
    StreamRecordInfo(s32 plane, const AlgType& type, const std::string& strTag)
        : planeId(plane),
          algType(type),
          tag(strTag)
    {}
    StreamRecordInfo(const StreamRecordInfo& that) : planeId(that.planeId), algType(that.algType), tag(that.tag) {}
    StreamRecordInfo& operator=(const StreamRecordInfo& that)
    {
        if (&that != this) {
            planeId = that.planeId;
            algType = that.algType;
            tag = that.tag;
        }
        return *this;
    }
};

class ProfilerBase {
public:
    /* * 输出文本时, 获取op, dataType的字符串以及单位数据长度的数组 */
    static const std::array<uint32_t, HCCL_REDUCE_RESERVED> opString;
    static const std::array<uint32_t, HCCL_DATA_TYPE_RESERVED> dataTypeString;
    static const std::array<s32, HCCL_DATA_TYPE_RESERVED> sizeOf;

    explicit ProfilerBase(u32 deviceLogicId);
    virtual ~ProfilerBase();

    virtual HcclResult Run(const StepData& stepData) = 0;
    virtual HcclResult Flush() = 0;
    static HcclResult AddStream(s32 streamID, const std::string& tag, s32 planeID, const AlgType& algType);
    static HcclResult DelStream(s32 streamID);
    static HcclResult AddTag(
        const std::string& tag, const std::string& group, const HcclWorkflowMode& workFlowMode, bool isSendRecv = false,
        bool isAiv = false);
    static HcclResult DelTag(const std::string& tag);
    static HcclResult AddOpData(
        const std::string& tag, u64 count, const void* src, const void* dst, HcclDataType dataType, u32 rootId,
        const std::string& group, HcclReduceOp reduceType = HCCL_REDUCE_RESERVED);
    static HcclResult DelOpData(const std::string& tag);
    static HcclResult AddGroupRankInfo(
        const std::string& group, u32 rankSize, u32 rankId, bool isSendRecv = false,
        u32 remoteRankId = INVALID_VALUE_RANKSIZE);
    static HcclResult DelGroupRankInfo(const std::string& tag);
    static HcclResult GetTagByStream(u32& streamID, std::string& tag);
    static HcclResult GetAlgTypeByStream(u32& streamID, AlgType& algType);
    static HcclResult GetGroupNameByTag(const std::string& tag, std::string& group);
    static HcclResult GetRankInfoByGroup(const std::string& group, GroupRankInfo& groupRankInfo);
    static HcclResult GetOpDataInfoByTag(const std::string& tag, OpDataInfo& opDataInfo);
    static HcclResult AddGroupUdi(const std::string& group, const std::string& udi);
    static HcclResult DelGroupUdi(const std::string& group);
    static HcclResult GetUdiByGroup(const std::string& group, std::string& udi);
    static void GetSubmittedOpCnt(u32& index);
    virtual HcclResult Save(u32& streamID, u32& taskID, TaskType& taskType, const TaskParaDMA& para) = 0;
    virtual HcclResult Save(u32& streamID, u32& taskID, TaskType& taskType, const TaskParaReduce& para) = 0;
    virtual HcclResult Save(u32& streamID, u32& taskID, TaskType& taskType, const TaskParaNotify& para) = 0;
    virtual HcclResult Save(u32 streamID, u32 taskID, const TaskParaAiv& para) = 0;
    virtual HcclResult Save(u32& streamID, u32& taskID, const void* descBuf = nullptr, size_t descBufLen = 0) = 0;
    virtual HcclResult Save(u32 captureStreamID, u32 streamID, u32 taskID, TaskType& taskType, const TaskParaDMA& para)
        = 0;
    virtual HcclResult
    Save(u32 captureStreamID, u32 streamID, u32 taskID, TaskType& taskType, const TaskParaReduce& para)
        = 0;
    virtual HcclResult
    Save(u32 captureStreamID, u32 streamID, u32 taskID, TaskType& taskType, const TaskParaNotify& para)
        = 0;
    virtual HcclResult
    Save(u32 captureStreamID, u32 streamID, u32 taskID, const void* descBuf = nullptr, size_t descBufLen = 0)
        = 0;
    virtual HcclResult Save(u32 captureStreamID, u32 streamID, u32 taskID, const TaskParaAiv& paraAiv) = 0;
    virtual HcclResult SaveToLog(const TaskParaHost& paraHost) = 0;

protected:
    static std::array<std::map<s32, StreamRecordInfo>, MAX_MODULE_DEVICE_NUM> streamRecordInfoMap_;
    static std::array<std::map<const std::string, const std::string>, MAX_MODULE_DEVICE_NUM> tagGroupMap_;
    static std::array<std::map<const std::string, const HcclWorkflowMode>, MAX_MODULE_DEVICE_NUM> tagModeMap_;
    static std::array<std::mutex, MAX_MODULE_DEVICE_NUM> streamMutex_;
    static std::array<std::map<const std::string, GroupRankInfo>, MAX_MODULE_DEVICE_NUM> groupRankMap_;
    static std::array<std::map<const std::string, OpDataInfo>, MAX_MODULE_DEVICE_NUM> tagOpDataMap_;
    static std::array<std::map<const std::string, u32>, MAX_MODULE_DEVICE_NUM> groupIndexMap_;
    static std::array<std::map<const std::string, u32>, MAX_MODULE_DEVICE_NUM> aivGroupIndexMap_;
    static std::array<std::map<const std::string, u32>, MAX_MODULE_DEVICE_NUM> sendRecvGroupIndexMap_;
    static std::array<std::map<const std::string, std::string>, MAX_MODULE_DEVICE_NUM> groupUdiMap_;
    const u32 deviceLogicId_;
    static bool isSendRecv_[MAX_MODULE_DEVICE_NUM];
    static u32 index_[MAX_MODULE_DEVICE_NUM];

private:
};
} // namespace hccl

#endif /* PROFILER_BASE_PUB_H */
