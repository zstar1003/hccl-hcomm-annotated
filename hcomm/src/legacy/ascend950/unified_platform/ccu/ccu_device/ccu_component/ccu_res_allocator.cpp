/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_res_allocator_legacy.h"

#include <algorithm>
#include <climits>

#include "ccu_res_specs_legacy.h"

namespace Hccl {

HcclResult CcuResIdAllocator::Alloc(
    const uint32_t num, const bool consecutive, std::vector<ResInfo>& allocatedResInfos, const std::string& dfxInfo)
{
    CHK_PRT_RET(
        num == 0, HCCL_ERROR("[CcuResIdAllocator][%s] failed, request num is 0.", __func__), HcclResult::HCCL_E_PARA);

    std::unique_lock<std::mutex> lock(innerMutex);
    // 快速判断是否可以分配
    const uint32_t freeSize = capacity_ - allocatedSize;
    if (num > freeSize) {
        HCCL_WARNING(
            "[CcuResIdAllocator][%s] failed, resType[%s], requested num[%u] exceeds "
            "currently free size[%u].",
            __func__, dfxInfo.c_str(), num, freeSize);
        HCCL_RUN_INFO("Insufficient CCU Resource: %s, requestNum[%u], freeNum[%u].", dfxInfo.c_str(), num, freeSize);
        return HcclResult::HCCL_E_UNAVAIL;
    }

    vector<ResInfo> newResInfos;
    uint32_t leftNum = num;
    uint32_t tryStartId = 0;
    // 对于要求连续的资源需要提供一个足够大小的空闲块
    // 对于非连续需要每次不为 0
    const uint32_t limitSize = consecutive ? num - 1 : 0;
    // 顺序优先分配，遍历已分配的连续块，寻找当前块与下个块之间是否有足够大小空间
    resInfos.emplace_back(capacity_, 0); // 临时添加一个尾资源，简化判断逻辑
    for (size_t i = 0; i < resInfos.size(); i++) {
        const auto& resInfo = resInfos[i];
        uint32_t partNum = std::min(resInfo.startId - tryStartId, leftNum);
        if (partNum > limitSize) {
            newResInfos.emplace_back(tryStartId, partNum); // 该空闲块足够大，分配
            leftNum -= partNum;
            if (leftNum == 0) {
                break;
            }
        }
        tryStartId = resInfo.startId + resInfo.num; // 更新当前块起始位置
    }
    resInfos.pop_back(); // 删除临时添加的尾资源
    // 只有连续要求的资源才可能剩余，此时分配失败，新块为空
    if (leftNum != 0) {
        HCCL_WARNING(
            "[CcuResIdAllocator][%s] failed, no enough consecutive free "
            "resource ids for requested num[%u].",
            __func__, num);
        HCCL_RUN_INFO(
            "Insufficient CCU Resource: consecutived %s, requestNum[%u], freeNum[%u].", dfxInfo.c_str(), num, freeSize);
        return HcclResult::HCCL_E_UNAVAIL;
    }

    allocatedSize += num;
    AllocResInfo(newResInfos); // 将分配的所有资源记录
    allocatedResInfos = newResInfos;
    return HcclResult::HCCL_SUCCESS;
}

void CcuResIdAllocator::AllocResInfo(std::vector<ResInfo> newResInfos)
{
    if (resInfos.empty()) { // 首次分配直接添加块
        resInfos.emplace_back(newResInfos.front());
        return;
    }
    // 内部变量始终维护最简的连续块，对需要合并的块更新
    size_t newIdx = 0;
    size_t idx = 0;
    while (newIdx < newResInfos.size() && idx < resInfos.size()) {
        auto& newResInfo = newResInfos[newIdx];
        auto& resInfo = resInfos[idx];
        // 跳过无关的资源块，使得resInfo是newResInfo的后续块
        if (newResInfo.startId >= resInfo.startId) {
            idx++;
            continue;
        }
        // 检查当前块是否与后续块连续，如果连续则合并
        if (newResInfo.startId == resInfo.startId - newResInfo.num) {
            newResInfo.num += resInfo.num;
            resInfos.erase(resInfos.begin() + idx);
        }
        // 如果当前块是首块则插入首块
        if (idx == 0) {
            resInfos.insert(resInfos.begin(), newResInfo);
            newIdx++;
            continue;
        }
        // 分配保证如果当前块不是首块则一定与前一个块连续，更新前一个块
        resInfos[idx - 1].num += newResInfo.num;
        newIdx++;
    }
    // 如果有剩余块一定与最后一个块连续，更新最后一个块
    if (newIdx < newResInfos.size()) {
        resInfos.back().num += newResInfos.back().num;
    }
}

static HcclResult CheckReleasePara(const uint32_t startId, const uint32_t num, const uint32_t capacity)
{
    CHK_PRT_RET(
        num == 0, HCCL_ERROR("[CcuResIdAllocator][%s] failed, resource num is 0.", __func__), HcclResult::HCCL_E_PARA);

    CHK_PRT_RET(
        num > capacity,
        HCCL_ERROR(
            "[CcuResIdAllocator][%s] failed, resource num[%u] "
            "is greater than capacity[%u]",
            __func__, num, capacity),
        HcclResult::HCCL_E_PARA);

    CHK_PRT_RET(
        startId > capacity - num,
        HCCL_ERROR(
            "[CcuResIdAllocator][%s] failed, resource startId[%u] "
            "num[%u] capacity[%u]",
            __func__, startId, num, capacity),
        HcclResult::HCCL_E_PARA);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResIdAllocator::Release(const uint32_t startId, const uint32_t num)
{
    CHK_RET(CheckReleasePara(startId, num, capacity_));

    std::unique_lock<std::mutex> lock(innerMutex);

    // 找到需要释放的资源块
    const size_t resIndex = FindReleaseResIndex(startId);
    CHK_PRT_RET(
        resIndex >= resInfos.size(),
        HCCL_ERROR(
            "[CcuResIdAllocator][%s] failed, resource startId[%u] num[%u] "
            "has not been allocated yet. ",
            __func__, startId, num),
        HcclResult::HCCL_E_PARA);

    // 判断申请释放的资源是否越界
    const auto& resInfo = resInfos[resIndex];
    uint32_t allocatedNum = resInfo.startId + resInfo.num - startId;
    CHK_PRT_RET(
        num > allocatedNum,
        HCCL_ERROR(
            "[CcuResIdAllocator][%s] failed, resource num[%u] is greater "
            "than the allocated num[%u].",
            __func__, num, allocatedNum),
        HcclResult::HCCL_E_PARA);

    // 将资源块释放并更新
    ReleaseResInfo(resIndex, startId, num);
    return HcclResult::HCCL_SUCCESS;
}

constexpr u32 INVALID_UINT = 0xFFFFFFFF;
size_t CcuResIdAllocator::FindReleaseResIndex(const uint32_t startId) const
{
    size_t resIndex = 0;
    const size_t maxIndex = resInfos.size();
    while (resIndex < maxIndex) {
        const auto& resInfo = resInfos[resIndex];
        // 检查resInfo.startId + resInfo.num是否会u32溢出
        if ((resInfo.startId < INVALID_UINT - resInfo.num) && startId >= resInfo.startId + resInfo.num) {
            resIndex++;
            continue;
        }
        if (startId >= resInfo.startId) {
            break; // 资源id属于该资源块
        }
        return maxIndex; // 无法找到已分配资源块，返回错误索引
    }
    return resIndex;
}

void CcuResIdAllocator::ReleaseResInfo(const size_t resIndex, const uint32_t startId, const uint32_t num)
{
    allocatedSize -= num;

    auto& resInfo = resInfos[resIndex];
    // 释放的资源在资源块起始部分
    if (startId == resInfo.startId) {
        // 恰好是整块资源，则全部释放
        if (num == resInfo.num) {
            resInfos.erase(resInfos.begin() + resIndex);
            return;
        }
        // 非整块资源则更新资源块起始位置和大小
        resInfo.startId += num;
        resInfo.num -= num;
        return;
    }

    uint32_t leftNum = startId - resInfo.startId;
    uint32_t rightNum = resInfo.num - leftNum - num;
    // 释放的资源在资源块末尾，更新资源块大小
    if (rightNum == 0) {
        resInfo.num -= num;
        return;
    }
    // 释放的资源在资源块中间，拆分为两个资源块
    resInfo.num = leftNum; // 左部分块更新数据
    // 右部分块需要新增
    resInfos.emplace(resInfos.begin() + resIndex + 1, startId + num, rightNum);
}

CcuResAllocator::CcuResAllocator(const int32_t devLogicId, const uint8_t dieId) : devLogicId_(devLogicId), dieId_(dieId)
{
    auto& ccuResSpecs = CcuResSpecifications::GetInstance(devLogicId);
    // 获取静态定义的资源规格查询函数列表，遍历构造
    for (const auto& pair : GET_RES_SPEC_FUNC_ARRAY) {
        const ResType resType = pair.first;
        const GetResSpecFunc getFunc = pair.second;
        uint32_t capacity = 0; // 获取失败时容量为 0，后续分配按资源不足处理
        (void)(ccuResSpecs.*getFunc)(dieId, capacity);
        auto allocatorPtr = std::make_unique<CcuResIdAllocator>(capacity);
        idAllocatorMap[static_cast<uint8_t>(resType)] = std::move(allocatorPtr);
    }
}

HcclResult CcuResAllocator::Alloc(
    const ResType resType, const uint32_t num, const bool consecutive, std::vector<ResInfo>& resInfos)
{
    auto resTypeIter = idAllocatorMap.find(static_cast<uint8_t>(resType));
    if (resTypeIter == idAllocatorMap.end()) {
        HCCL_ERROR("[CcuResAllocator][%s] failed, invalid resource type[%s].", __func__, resType.Describe().c_str());
        return HcclResult::HCCL_E_PARA;
    }
    return resTypeIter->second->Alloc(num, consecutive, resInfos, resType.Describe());
}

HcclResult CcuResAllocator::Release(const ResType resType, const uint32_t startId, const uint32_t num)
{
    auto resTypeIter = idAllocatorMap.find(static_cast<uint8_t>(resType));
    if (resTypeIter == idAllocatorMap.end()) {
        HCCL_ERROR("[CcuResAllocator][%s] failed, invalid resource type[%s].", __func__, resType.Describe().c_str());
        return HcclResult::HCCL_E_PARA;
    }
    return resTypeIter->second->Release(startId, num);
}

uint32_t CcuResIdAllocator::GetConsecutiveRemainSize() const
{
    uint32_t maxGap = 0;
    uint32_t cursor = 0;
    // resInfo 保证按startId升序排列，因此无需排序
    for (const auto& r : resInfos) {
        if (r.startId > cursor) {
            maxGap = std::max(maxGap, r.startId - cursor);
        }
        cursor = r.num + r.startId;
    }
    if (capacity_ > cursor) {
        maxGap = std::max(maxGap, capacity_ - cursor);
    }
    return maxGap;
}

std::string CcuResIdAllocator::Describe() const
{
    return StringFormat(
        "CcuResIdAllocator[capacity=%u, allocatedSize=%u, "
        "resInfos_size=%u]",
        capacity_, allocatedSize, resInfos.size());
}

uint32_t CcuResAllocator::GetConsecutiveRemainSize(const ResType resType) const
{
    auto it = idAllocatorMap.find(static_cast<uint8_t>(resType));
    if (it == idAllocatorMap.end())
        return 0;
    return it->second->GetConsecutiveRemainSize();
}

std::string CcuResAllocator::Describe() const
{
    return StringFormat(
        "CcuResAllocator[devLogicId=%u, dieId=%u, "
        "idAllocatorSize=[%u]]",
        devLogicId_, dieId_, idAllocatorMap.size());
}

}; // namespace Hccl
