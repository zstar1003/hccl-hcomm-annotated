/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __AICPU_ONE_SIDE_SERVICE_H__
#define __AICPU_ONE_SIDE_SERVICE_H__

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <mutex>
#include <hccl/hccl_types.h>
#include "common/aicpu_hccl_def.h"
#include "stream_pub.h"
#include "transport_mem.h"
#include "dispatcher.h"
#include "aicpu_operator_pub.h"
#include <shared_mutex>

namespace hccl {
class HcclOneSideServiceAicpu {
public:
    HcclOneSideServiceAicpu();
    ~HcclOneSideServiceAicpu();

    static HcclResult Process(const OpTilingData* tilingData);
    static std::shared_ptr<HcclOneSideServiceAicpu> GetService(const std::string& tag, const OpTilingData* tilingData);
    static HcclResult CleanAllStreamFunc();
    static HcclResult DisableAllStreamFunc();
    static HcclResult HandleErrCqe();
    static bool isAllDestroy();

private:
    HcclResult Init(const std::string& tag, const OpTilingData* tilingData);
    HcclResult InitOpNotifyObj();
    HcclResult InitStream(
        Stream& stream, HcclComStreamInfo& comStreamInfo, const HcclStreamParam& streamParam, const std::string& tag);
    HcclResult DoProcess(const std::string& tag, const OpTilingData* tilingData);
    HcclResult
    FillMemDetails(MemDetails& localMems, MemDetails& remoteMems, const HcclOneSideOpDescParam* descPtr, u32 index);
    HcclResult PrepareRdmaLink(u32 remoteRankId, const struct HcclQpInfoV2& qpInfo);
    HcclResult DoRdmaProcess(
        HcclCMDType cmdType, u32 remoteRankId, const OpTilingOneSideCommDataDes* vDataPtr,
        const HcclOneSideOpDescParam* desc, u32 descNum);
    HcclResult DoSdmaProcess(
        HcclCMDType cmdType, u32 remoteRankId, const OpTilingOneSideCommDataDes* vDataPtr,
        const HcclOneSideOpDescParam* desc, u32 descNum);
    // for profiling
    HcclResult InitProfiling();
    HcclResult WorkStart(HcclCMDType cmdType, u32 remoteRankId);
    HcclResult WorkEnd(HcclCMDType cmdType, u32 remoteRankId);
    HcclResult ReportMainStreamTask(u16 type);
    HcclResult UpdateProfReportStartSqeIdx();
    HcclResult CombineReportOpInfo(HcclCMDType cmdType, u8 dataType, u64 count);
    HcclResult ReportHcclTaskInfo();
    HcclResult ClearStreamLocalBuff();
    HcclResult CleanStreamFunc();
    HcclResult DisableStreamFunc();
    HcclResult CleanStream(Stream& stream);
    void ResetStreamCqeExceptionStatus(const Stream& stream);
    HcclResult UpdateSqStatus(Stream& stream);
    void HandleCqeMessage(bool isReadClear);
    void PollCqeException(Stream& stream, bool isReadClear, rtLogicCqReport_t& cqeException, CqeStatus& cqeStatus);

    static std::shared_mutex serviceMapMutex_;
    static std::unordered_map<std::string, std::shared_ptr<HcclOneSideServiceAicpu>> services_;

    bool isInited_{false};
    std::string identifier_;
    const HcclOneSideCommResParam* commResParaPtr_{nullptr};
    HcclDispatcher dispatcher_{nullptr};
    Stream execStream_;
    HcclComStreamInfo execComStreamInfo_;
    u32 rankSize_{0};
    u32 rankId_{0};
    u32 logicDevId_{0};
    u32 devId_{0};
    DevType devType_{DevType::DEV_TYPE_COUNT};
    int64_t chipId_{0};
    u32 linkTimeout_{INVALID_UINT};                                    // 单位us
    std::unordered_map<u32, std::shared_ptr<TransportMem>> rdmaLinks_; // 根据remoteRankId索引transport
    std::vector<std::shared_ptr<LocalNotify>> opNotifies_;             // host与device间同步的notify
    bool execStreamEnable_ = true;

    // for profiling
    u64 groupHashId_{0};
    u64 totalCount_{0};
    HcclDataType dataType_{HcclDataType::HCCL_DATA_TYPE_RESERVED};
};
} // namespace hccl

#endif // __AICPU_ONE_SIDE_SERVICE_H__
