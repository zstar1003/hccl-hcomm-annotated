/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GRADIENT_SEGMENT_H
#define GRADIENT_SEGMENT_H

#include <map>
#include <memory>
#include <vector>
#include <mutex>
#include <hccl/hccl_types.h>
#include "hccl/base.h"

namespace hccl {
// ResNet-50按梯度数据量的默认切分比率
const float GRADIENT_TOTAL_SIZE_RATIO = 100;
const float GRADIENT_SEGMENT_SIZE_RATIO = 96.54;
extern std::map<std::string, std::vector<u32>> g_segmentIdxMap;
extern std::map<std::string, std::vector<float>> g_segmentSizeMap;
extern std::mutex g_segmentIdxMapLock;
extern std::mutex g_segmentSizeMapLock;

/*
实现梯度切分功能的类，
目前支持通过用户基于梯度层数或数据量进行配置。
若无配置，则按数据量固定切分。
*/
class GradientSegment {
public:
    explicit GradientSegment();
    virtual ~GradientSegment();

    /* 执行梯度切分功能 */
    HcclResult GetGradientSegmentExecutor(
        const std::string& group, const struct model_feature* feature, std::vector<u32>& segment_index,
        bool& isUseFusionLib, GradSplitForceMode force = GradSplitForceMode::FORCE_NONE,
        OriginalGraphShapeType shapeType = OriginalGraphShapeType::KNOWN_SHAPE);

protected:
private:
    HcclResult GetSegmentByIndex(const std::string& group, u32 featGradNum, std::vector<u32>& segList) const;
    HcclResult GetSegmentBySize(
        const std::string& group, u32 featGradNum, std::vector<u32>& segList, const std::vector<float>& accumGradList);
    HcclResult GetSplitResInEachSegment(
        const std::vector<float>& accumGradList, float gradSize, std::vector<u32>& segList, float& allocGradSize,
        float& preSizeLeft);
    HcclResult
    GetSegmentByDefaultRatio(const std::vector<float>& accumGradList, u32 featGradNum, std::vector<u32>& segList);
    HcclResult CheckAndConfigSegment(
        std::vector<float>& segmentSizeProportion, float totalSize, std::vector<float>& segmentSize) const;
    OriginalGraphShapeType shapeType_;
    HcclResult
    GetIdxByBinarySearch(const std::vector<float>& accumGradList, const float& curSize, u32& segGradIdx) const;
    HcclResult
    GetNearIdxByDataSize(const std::vector<float>& accumGradList, u32& segGradIdx, float gradSize, s32 midIdx) const;
    HcclResult GetFixedSizeSegmentByDefaultRatio(
        const std::vector<float>& accumGradList, u32 featGradNum, std::vector<u32>& segList);
    HcclResult
    GetTwoSegmentByDefaultRatio(const std::vector<float>& accumGradList, u32 featGradNum, std::vector<u32>& segList);
};
} // namespace hccl

#endif /* * GRADIENT_SEGMENT_H */
