/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gradient_segment.h"
#include <math.h>
#include "externalinput_pub.h"

namespace hccl {

GradientSegment::GradientSegment() : shapeType_(OriginalGraphShapeType::KNOWN_SHAPE) {}

GradientSegment::~GradientSegment() {}

HcclResult GradientSegment::GetGradientSegmentExecutor(
    const std::string& group, const struct model_feature* feature, std::vector<u32>& segment_index,
    bool& isUseFusionLib, GradSplitForceMode force, OriginalGraphShapeType shapeType)
{
    CHK_PRT_RET(
        group.empty(), HCCL_ERROR("[GradientSegment][GetGradientSegmentExecutor]params invalid, group is empty"),
        HCCL_E_PARA);
    CHK_PTR_NULL(feature);
    isUseFusionLib = false;
    std::vector<u32> segList;
    std::vector<float> accumGradList;
    u32 featGradNum = feature->gradient_num;
    shapeType_ = shapeType;
    // 根据每层数据量，计算每层的累积数据量list
    accumGradList.push_back(feature->gradient_size[0]);
    for (u32 gradIdx = 1; gradIdx < featGradNum; gradIdx++) {
        float accumGrad = accumGradList.back() + feature->gradient_size[gradIdx];
        accumGradList.push_back(accumGrad);
    }
    // 获取基于梯度数据量的切分策略
    HcclResult ret = GetSegmentBySize(group, featGradNum, segList, accumGradList);
    if (ret != HCCL_SUCCESS) {
        if (ret != HCCL_E_PARA) {
            // 获取基于梯度层数的切分策略
            bool bSplitBySize = true;
            if (force == GradSplitForceMode::FORCE_SIZE) {
                bSplitBySize = true;
                HCCL_INFO("force split gradient segment by built-in size ratio.");
            } else {
                bSplitBySize = (GetSegmentByIndex(group, featGradNum, segList) != HCCL_SUCCESS);
            }
            if (bSplitBySize) { // 基于数据量进行默认切分
                isUseFusionLib = true;
                CHK_RET(GetSegmentByDefaultRatio(accumGradList, featGradNum, segList));
            }
        } else {
            return HCCL_E_PARA;
        }
    }
    if (segList.size() > 0) {
        segment_index = segList;
    } else {
        HCCL_ERROR("[Get][GradientSegmentExecutor]segList is empty.");
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult GradientSegment::GetSegmentByDefaultRatio(
    const std::vector<float>& accumGradList, u32 featGradNum, std::vector<u32>& segList)
{
    if (shapeType_ == OriginalGraphShapeType::UNKNOWN_SHAPE) {
        CHK_RET(GetFixedSizeSegmentByDefaultRatio(accumGradList, featGradNum, segList));
    } else {
        CHK_RET(GetTwoSegmentByDefaultRatio(accumGradList, featGradNum, segList));
    }

    return HCCL_SUCCESS;
}

HcclResult GradientSegment::GetTwoSegmentByDefaultRatio(
    const std::vector<float>& accumGradList, u32 featGradNum, std::vector<u32>& segList)
{
    std::vector<u32> segTempList;
    float gradSize = (GRADIENT_SEGMENT_SIZE_RATIO / GRADIENT_TOTAL_SIZE_RATIO) * accumGradList[featGradNum - 1];
    float allocGradSize = 0;
    float preSizeLeft = 0;
    HcclResult ret = GetSplitResInEachSegment(accumGradList, gradSize, segTempList, allocGradSize, preSizeLeft);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Get][TwoSegmentByDefaultRatio]errNo[0x%016llx] get gradIdx with [%f]%% datasize "
            "fail",
            HCCL_ERROR_CODE(HCCL_E_PARA), GRADIENT_SEGMENT_SIZE_RATIO),
        HCCL_E_PARA);

    segList.push_back(segTempList[0]);
    if (featGradNum >= (segList[0] + 2)) { // 将得到梯度索引值加2与总梯度长比较，判断能否切第二段
        segList.push_back(featGradNum - 1); // 通过减1得到最后一层索引将第二段放入切分策略list
    }
    HCCL_DEBUG("<gradient segment size default result>");
    return HCCL_SUCCESS;
}

HcclResult GradientSegment::GetFixedSizeSegmentByDefaultRatio(
    const std::vector<float>& accumGradList, u32 featGradNum, std::vector<u32>& segList)
{
    std::vector<float> segmentSizeRatio
        = {GRADIENT_SEGMENT_SIZE_RATIO, (GRADIENT_TOTAL_SIZE_RATIO - GRADIENT_SEGMENT_SIZE_RATIO)};
    std::vector<float> segmentSize;
    CHK_PRT_RET(
        accumGradList.empty(), HCCL_ERROR("[Get][FixedSizeSegmentByDefaultRatio]accumGradList empty, fail"),
        HCCL_E_PARA);
    // 按默认切分方式切分两段，计算每段的数据量大小
    CHK_RET(CheckAndConfigSegment(segmentSizeRatio, accumGradList[featGradNum - 1], segmentSize));

    float allocGradSize = 0;
    float preSizeLeft = 0;
    for (u32 inputIdx = 0; inputIdx < segmentSize.size(); inputIdx++) {
        // 根据比例不一定能完整切分，将前一段中剩余未切分的梯度数据量累加到下一段中
        segmentSize[inputIdx] += preSizeLeft;
        HcclResult ret
            = GetSplitResInEachSegment(accumGradList, segmentSize[inputIdx], segList, allocGradSize, preSizeLeft);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Get][FixedSizeSegmentByDefaultRatio]errNo[0x%016llx] get gradIdx with [%u] "
                "segment fail",
                HCCL_ERROR_CODE(HCCL_E_PARA), inputIdx),
            HCCL_E_PARA);
        if (segList.back() == (featGradNum - 1)) {
            break;
        }
    }
    if (segList.back() < (featGradNum - 1)) {
        HCCL_WARNING("the last segment point[%u] is less than feednum[%u]", segList.back(), featGradNum - 1);
        segList.push_back(featGradNum - 1);
    }
    if (segList.empty()) {
        HCCL_ERROR("[Get][FixedSizeSegmentByDefaultRatio]<segList is empty>");
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult
GradientSegment::GetSegmentByIndex(const std::string& group, u32 featGradNum, std::vector<u32>& segList) const
{
    (void)featGradNum;
    std::unique_lock<std::mutex> segmentIdxMapLock(g_segmentIdxMapLock);
    auto gsearch = g_segmentIdxMap.find(group); // 读取用户基于梯度层数的切分方案
    if (gsearch != g_segmentIdxMap.end()) {
        segList.assign(gsearch->second.begin(), gsearch->second.end());
    } else {
        return HCCL_E_NOT_FOUND;
    }
    segmentIdxMapLock.unlock();
    HCCL_DEBUG("<gradient segment user set index result>");
    return HCCL_SUCCESS;
}

HcclResult GradientSegment::GetSegmentBySize(
    const std::string& group, u32 featGradNum, std::vector<u32>& segList, const std::vector<float>& accumGradList)
{
    HcclResult ret;
    CHK_PRT_RET(accumGradList.empty(), HCCL_ERROR("[Get][SegmentBySize]accumGradList empty, fail"), HCCL_E_PARA);
    std::unique_lock<std::mutex> segmentSizeMapLock(g_segmentSizeMapLock);
    float totalSize = accumGradList[featGradNum - 1];
    auto gSizeSearch = g_segmentSizeMap.find(group); // 读取用户基于梯度数据量的切分方案
    if (gSizeSearch != g_segmentSizeMap.end()) {
        if (gSizeSearch->second.size() != 0) {
            std::vector<float> segmentSize;
            CHK_RET(CheckAndConfigSegment(gSizeSearch->second, totalSize, segmentSize));
            float allocGradSize = 0;
            float preSizeLeft = 0;
            for (u32 inputIdx = 0; inputIdx < segmentSize.size(); inputIdx++) {
                // 根据比例不一定能完整切分，将前一段中剩余未切分的梯度数据量累加到下一段中
                segmentSize[inputIdx] += preSizeLeft;
                ret = GetSplitResInEachSegment(
                    accumGradList, segmentSize[inputIdx], segList, allocGradSize, preSizeLeft);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR(
                        "[Get][SegmentBySize]errNo[0x%016llx] get gradIdx with segment [%u] fail",
                        HCCL_ERROR_CODE(HCCL_E_PARA), inputIdx),
                    HCCL_E_PARA);
                if (segList.back() == (featGradNum - 1)) {
                    break;
                }
            }
            if (segList.empty()) {
                HCCL_ERROR("[Get][SegmentBySize]<segList is empty>");
                return HCCL_E_PARA;
            } else {
                HCCL_DEBUG("<gradient segment user set size result>");
                return HCCL_SUCCESS;
            }
        }
    }
    return HCCL_E_NOT_FOUND;
}

HcclResult GradientSegment::CheckAndConfigSegment(
    std::vector<float>& segmentSizeProportion, float totalSize, std::vector<float>& segmentSize) const
{
    float proportion = 0;
    float gradSize = 0;

    for (u32 inputIdx = 0; inputIdx < segmentSizeProportion.size(); inputIdx++) {
        proportion = segmentSizeProportion[inputIdx];
        gradSize = (proportion / GRADIENT_TOTAL_SIZE_RATIO) * totalSize;
        segmentSize.push_back(gradSize);
    }
    return HCCL_SUCCESS;
}

HcclResult GradientSegment::GetSplitResInEachSegment(
    const std::vector<float>& accumGradList, float gradSize, std::vector<u32>& segList, float& allocGradSize,
    float& preSizeLeft)
{
    bool bRet = accumGradList.size() == 0;
    CHK_PRT_RET(
        bRet,
        HCCL_ERROR(
            "[Get][SplitResInEachSegment]errNo[0x%016llx] accumGradList is empty!", HCCL_ERROR_CODE(HCCL_E_PARA)),
        HCCL_E_PARA);
    u64 cclBufferSize = GetExternalInputCCLBuffSize() - CCL_COMM_INBUFFER_UNALIGNED_RESERVE_SIZE;
    float commInputSize = static_cast<float>(cclBufferSize);
    float curSize = 0;
    float sizeLeft = gradSize;
    float allocGradSizeTmp = allocGradSize;
    u32 featGradNum = accumGradList.size();
    u32 segGradIdx = 0;
    while (1) {
        if (shapeType_ == OriginalGraphShapeType::UNKNOWN_SHAPE) {
            curSize = (sizeLeft > commInputSize) ? commInputSize : sizeLeft;
            curSize += allocGradSize;
        } else {
            curSize = sizeLeft + allocGradSize;
        }

        CHK_RET(GetIdxByBinarySearch(accumGradList, curSize, segGradIdx));

        if (segList.size() > 0 && segGradIdx <= segList.back()) {
            segGradIdx = segGradIdx + 1;
        }
        if (segGradIdx == featGradNum) {
            HCCL_ERROR("[Get][SplitResInEachSegment]segGradIdx[%u] already on the last layer", segList.back());
            return HCCL_E_PARA;
        }
        segList.push_back(segGradIdx);
        HCCL_DEBUG("<segment index print:[%u]>", segGradIdx);

        // 更新当前段内已分配的梯度数据量和当前段中剩余未分配完的梯度量
        allocGradSize = accumGradList[segGradIdx];
        sizeLeft = gradSize + allocGradSizeTmp - allocGradSize;
        // 满足两个条件退出循环：1，切分到最后一层  2，剩余的梯度量已无法切分
        if (segGradIdx == (featGradNum - 1)) {
            break;
        } else if (segGradIdx < (featGradNum - 1)) {
            if (((accumGradList[segGradIdx + 1] - accumGradList[segGradIdx]) - sizeLeft) > 1e-6) {
                break;
            }
        }
    }
    // 更新大段已分配的梯度数据量和当前段中剩余未分配完的梯度量
    preSizeLeft = sizeLeft;
    return HCCL_SUCCESS;
}

HcclResult GradientSegment::GetIdxByBinarySearch(
    const std::vector<float>& accumGradList, const float& curSize, u32& segGradIdx) const
{
    s32 lowIdx = 0;
    s32 midIdx = 0;
    s32 highIdx = accumGradList.size() - 1;
    // 二分法找到有序数据量list中第一个大于等于curSize大小的索引
    while (lowIdx <= highIdx) {
        midIdx = (lowIdx + highIdx) / 2;                         // 通过除2得到前后索引的中间值
        if (std::fabs(accumGradList[midIdx] - curSize) < 1e-6) { // 中间值等于curSize则直接返回当前索引
            segGradIdx = static_cast<u32>(midIdx);
            return HCCL_SUCCESS;
        } else if (accumGradList[midIdx] > curSize) {
            highIdx = midIdx - 1;
        } else {
            lowIdx = midIdx + 1;
        }
    }
    /* 动态shape并且二分极限数据量始终小于curSize时，直接返回当前的idx，多1会溢出 */
    if ((shapeType_ == OriginalGraphShapeType::UNKNOWN_SHAPE) && (accumGradList[midIdx] < curSize)) {
        segGradIdx = static_cast<u32>(midIdx);
        return HCCL_SUCCESS;
    }
    /* 没找到对应数据量的index，找接近此数据量的index */
    HcclResult ret = GetNearIdxByDataSize(accumGradList, segGradIdx, curSize, midIdx);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Get][IdxByBinarySearch]errNo[0x%016llx] get near Idx with [%f]%% datasize fail",
            HCCL_ERROR_CODE(HCCL_E_PARA), curSize),
        HCCL_E_PARA);
    return HCCL_SUCCESS;
}

HcclResult GradientSegment::GetNearIdxByDataSize(
    const std::vector<float>& accumGradList, u32& segGradIdx, float gradSize, s32 midIdx) const
{
    if (midIdx == 0) { // 如果当前索引在最开始则直接返回
        segGradIdx = static_cast<u32>(midIdx);
        return HCCL_SUCCESS;
    }

    if (shapeType_ == OriginalGraphShapeType::UNKNOWN_SHAPE) {
        segGradIdx = static_cast<u32>(midIdx) - 1;
    } else {
        // 判断前一个index和当前midIdx哪个位置的数据量与需要百分比数据量更为接近, 返回更接近的那个索引
        CHK_PRT_RET(
            accumGradList.empty(), HCCL_ERROR("[Get][NearIdxByDataSize]accumGradList empty, fail"), HCCL_E_PARA);
        float prevIdxGradDiff = gradSize - accumGradList[midIdx - 1];
        float idxGradDiff = accumGradList[midIdx] - gradSize;
        segGradIdx = (prevIdxGradDiff <= idxGradDiff) ? static_cast<u32>(midIdx) - 1 : static_cast<u32>(midIdx);
        HCCL_DEBUG("datasize: getIndex[%d] prevDiff[%.1f] curDiff[%.1f]", midIdx, prevIdxGradDiff, idxGradDiff);
    }
    return HCCL_SUCCESS;
}
} // namespace hccl
