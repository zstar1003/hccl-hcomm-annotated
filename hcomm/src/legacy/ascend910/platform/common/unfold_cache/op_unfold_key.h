/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __OP_UNFOLD_KEY_H__
#define __OP_UNFOLD_KEY_H__

#include <cstdint>
#include <functional>
#include <string>

#include "hccl_types.h"
#include "workflow_pub.h"

namespace hccl {

// 作为展开算子的标识符
// 注意: 由于给定通信域下, 相同数据量相同算子的算法选择是固定的, 不需要在标识符中维护algType或algName
// 注意: 图模式或单算子模式是全局固定的, 不需要在标识符中维护
struct OpUnfoldKey {
    explicit OpUnfoldKey();
    explicit OpUnfoldKey(const OpUnfoldKey& other); // 拷贝构造函数 (make_pair需要)

    HcclResult Init(
        const HcclCMDType curOpType, const HcclDataType curDataType, const HcclReduceOp curReduceType,
        const bool curIsZeroCopy, const bool curIsSymmetricMemory, const uint64_t curInputSize,
        const bool curIsInplacePreSync, const HcclWorkflowMode curWorkflowMode, const bool curIsCapture,
        const uint32_t curRoot = 0);

    // 用于debug
    std::string GetKeyString() const;

    bool operator==(const OpUnfoldKey& other) const;        // 重载operator==用于std::unordered_map中相等比较
    const OpUnfoldKey& operator=(const OpUnfoldKey& other); // 拷贝赋值操作符

    HcclCMDType opType;
    HcclDataType dataType;
    HcclReduceOp reduceType;
    bool isZeroCopy;

    // 是否开启对称内存优化
    bool isSymmetricMemory;

    // inputSize和outputSize是由totalCount(由rankSize+count决定)+dataType决定的, 给定通信域rankSize是固定的
    // 因为已经维护dataType, 所以inputSize/outputSize/totalCount中只需要维护任意一个即可
    // 注意: 对于alltoallv算子, cache查询不依赖具体数据量, inputSize用来区分isBigCount (0: false; 1: true)
    uint64_t inputSize;

    // ReduceScatter和AllReduce在开启重执行、in-place update、UserInMem > HcclBuffSize的时候，会触发前同步
    // (与正常算子展开逻辑不同)
    bool isInplacePreSync;

    // 是否为图模式 (可能存在同一个通信域下的同一个算子, 既执行图模式又执行单算子模式下的算法)
    HcclWorkflowMode workflowMode; // 0: 图模式; 1: 单算子模式

    // 是否为aclgraph (aclgraph资源在graph销毁时释放)
    bool isCapture;

    // 算子的root参数 (仅对 scatter / broadcast / reduce 三类算子生效)
    // 注意: 这三类算子在不同root下, SQE模板内的远端rank id / link句柄 / notify id不同, 必须把root纳入key
    //      其他算子root字段无意义, 固定为0即可
    // 历史兼容性: 默认值0与"root=0"语义重合, 不影响非root算子的cache命中行为
    uint32_t root;
};

}; // namespace hccl

namespace std {

// 全特化std::hash<OpUnfoldKey>, 用于std::unordered_map中计算哈希值
template <>
struct hash<hccl::OpUnfoldKey> {
    size_t operator()(const hccl::OpUnfoldKey& key) const
    {
        // 使用std::hash计算key中每个字段的哈希值
        std::hash<bool> hashBool;
        std::hash<uint8_t> hashUint8;
        std::hash<uint64_t> hashUint64;

        // 假设opType/dataType/reduceType <= 255
        const size_t opTypeHashValue = hashUint8(static_cast<uint8_t>(key.opType));
        const size_t dataTypeHashValue = hashUint8(static_cast<uint8_t>(key.dataType));
        const size_t reduceTypeHashValue = hashUint8(static_cast<uint8_t>(key.reduceType));

        const size_t isZeroCopyHashValue = hashBool(key.isZeroCopy);
        const size_t isSymmetricMemoryHashValue = hashBool(key.isSymmetricMemory);
        const size_t inputSizeHashValue = hashUint64(key.inputSize);
        const size_t isInplacePreSyncHashValue = hashBool(key.isInplacePreSync);

        // 假设workflowMode <= 255
        const size_t workflowModeHashValue = hashUint8(static_cast<uint8_t>(key.workflowMode));

        const size_t isCaptureHashValue = hashBool(key.isCapture);

        // 加入root字段的哈希: 避免不同root (scatter/broadcast/reduce) 在同一桶内冲突
        const size_t rootHashValue = std::hash<uint32_t>{}(key.root);

        // 简单的哈希混合
        size_t hashValue = opTypeHashValue;
        hashValue ^= dataTypeHashValue;
        hashValue ^= reduceTypeHashValue;
        hashValue ^= isZeroCopyHashValue;
        hashValue ^= isSymmetricMemoryHashValue;
        hashValue ^= inputSizeHashValue;
        hashValue ^= isInplacePreSyncHashValue;
        hashValue ^= workflowModeHashValue;
        hashValue ^= isCaptureHashValue;
        hashValue ^= rootHashValue;

        return hashValue;
    }
};

} // namespace std

#endif // __OP_UNFOLD_KEY_H__
