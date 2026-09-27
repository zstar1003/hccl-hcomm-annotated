/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef H_COM_PRIVATE_H
#define H_COM_PRIVATE_H

#include "hcom_pub.h"
#include "hccl_comm_pub.h"

HcclResult GetGroupRankInfo(const char* group, RankInfoType rankType, u32 inPara, u32* outPara);
HcclResult GetRankList(u32 rankNum, const u32* rankIds, HcclGroupParams& params);
HcclResult InitHcomMiscInfo(hccl::HcclCommParams& params, const char* rankTable);

/**********************************************************************
功能描述  : 梯度切分功能实现
输入参数  : char* group
            struct model_feature* feature
输出参数  : u32* segment_num
            u32 segment_index[8]
返 回 值  : HcclResult
**********************************************************************/
HcclResult GetGradientSegment(
    const std::string& group, const struct model_feature* feature, std::vector<u32>& segmentList, bool& configured,
    GradSplitForceMode force = GradSplitForceMode::FORCE_NONE,
    OriginalGraphShapeType shapeType = OriginalGraphShapeType::KNOWN_SHAPE);

struct HcomRequestInfo {
    std::string opTag;
    std::string group;
    HcomOpDesc opDesc;
    HcomRequestInfo(const std::string& opTag, const std::string& group, const HcomOpDesc& opDesc)
        : opTag(opTag),
          group(group),
          opDesc(opDesc)
    {}
};

HcomInfo& HcomGetCtxHomInfoById(u32 idx);
HcclResult HcomGetCtxDeviceLogicId(void);
HcomInfo& HcomGetCtxHomInfo(void);
HcomOpTagInfo& HcomGetCtxOpTagInfo(void);
HcomOpTagInfo& HcomGetCtxOpTagInfo(void);
bool& HcomGetCtxAutoTuneMode(void);
HcclResult HcomFlushBackloggedGroups();
HcclResult QueryDestroyFlag(const char* group);
HcclResult HcomGroupUnref(const char* group);
HcclResult HcomCheckInitClusterInfo(const char* rankTableM, const char* identify);

HcclResult HcomRPCInit(
    const char* rankTableM, const char* identify, HcomInfo& hcomInfo,
    WorkMode commWorkMode = WorkMode::HCCL_MODE_NORMAL);
#endif // H_COM_PRIVATE_H
