/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FFTS_COMMON_PUB_H
#define FFTS_COMMON_PUB_H

#include <cstdint>
#include <map>
#include <vector>
#include <memory>

#include "hccl/base.h"
#include "adapter_rts_common.h"
#include "hccl_common.h"
#include "common.h"
#include "device_capacity.h"

namespace hccl {
enum class ReduceType { INLINE_REDUCE = 0, TBE_REDUCE };

enum class CopyPattern { ZCOPY = 0, BCOPY };

using HcclOpMetaInfo = struct HcclOpMetaInfoDef {
    HcclCMDType opType = HcclCMDType::HCCL_CMD_INVALID;
    bool isRootRank = false;
    bool isSmallCount = false;
    uint32_t rootRank = INVALID_UINT;
    uint32_t KEY_SIZE = 100;
    bool isDefaultPath = true; // 是否为默认子图路径
    ReduceType reduceType = ReduceType::INLINE_REDUCE;
    CopyPattern copyPattern = CopyPattern::BCOPY;
    u64 alltoallvSendDataSize = 0;
    u64 alltoallvcSendDataSize = 0;
    u32 pipelineSliceNum = 1;
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    u32 algolevel1Type = 0;
    bool hugeData = false;
    u64 sliceNum = 1;
    bool dataSplit = false;
    bool isAivMode = false;
    bool isEnableCache = true;
    u8 deterministic = 0;
    bool preloadCopyOpt = false;

    static bool CheckEnableCache(const HcclOpMetaInfoDef& opMetaInfo)
    {
        if (opMetaInfo.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV
            || opMetaInfo.opType == HcclCMDType::HCCL_CMD_SEND || opMetaInfo.opType == HcclCMDType::HCCL_CMD_RECEIVE
            || opMetaInfo.hugeData) {
            return false;
        }

        if (opMetaInfo.alltoallvSendDataSize > RDMA_SEND_MAX_SIZE
            || opMetaInfo.alltoallvcSendDataSize > RDMA_SEND_MAX_SIZE) {
            return false;
        }
        return !(
            (opMetaInfo.opType == HcclCMDType::HCCL_CMD_ALLTOALLV || opMetaInfo.opType == HcclCMDType::HCCL_CMD_ALLTOALL
             || opMetaInfo.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC)
            && (opMetaInfo.copyPattern == CopyPattern::BCOPY));
    }

    static HcclOpMetaInfoDef GetOneForAllReduce(
        u32 algolevel1Type = 0, HcclDataType dataType = HCCL_DATA_TYPE_RESERVED,
        ReduceType reduceType = ReduceType::INLINE_REDUCE, bool isSmallCount = false, u32 pipelineSliceNum = 1,
        bool hugeData = false, CopyPattern copyPattern = CopyPattern::BCOPY, u64 sliceNum = 1,
        bool isAivModeConfig = false, bool isDefaultPathConfig = true, bool dataSplitConfig = false,
        u8 deterministicConfig = 0)
    {
        HcclOpMetaInfoDef meta;
        meta.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
        meta.copyPattern = copyPattern;
        meta.reduceType = reduceType;
        meta.dataType = dataType;
        meta.isSmallCount = isSmallCount;
        meta.pipelineSliceNum = pipelineSliceNum;
        meta.algolevel1Type = algolevel1Type;
        meta.hugeData = hugeData;
        meta.sliceNum = sliceNum;
        meta.isAivMode = isAivModeConfig;
        meta.isEnableCache = CheckEnableCache(meta);
        meta.isDefaultPath = isDefaultPathConfig;
        meta.dataSplit = dataSplitConfig;
        meta.deterministic = deterministicConfig;
        return meta;
    }

    static HcclOpMetaInfoDef GetOneForAllGather(
        u32 algolevel1Type = 0, bool hugeData = false, bool smallCount = false,
        CopyPattern copyPattern = CopyPattern::BCOPY, bool dataSplit = false, bool isAivMode = false)
    {
        HcclOpMetaInfoDef meta;
        meta.opType = HcclCMDType::HCCL_CMD_ALLGATHER;
        meta.copyPattern = copyPattern;
        meta.algolevel1Type = algolevel1Type;
        meta.hugeData = hugeData;
        meta.isSmallCount = smallCount;
        meta.isEnableCache = CheckEnableCache(meta);
        meta.dataSplit = dataSplit;
        meta.isAivMode = isAivMode;
        return meta;
    }

    static HcclOpMetaInfoDef GetOneForAllGatherV(
        u32 algolevel1Type = 0, bool hugeData = false, bool smallCount = false,
        CopyPattern copyPattern = CopyPattern::BCOPY, bool dataSplit = false)
    {
        HcclOpMetaInfoDef meta;
        meta.opType = HcclCMDType::HCCL_CMD_ALLGATHER_V;
        meta.copyPattern = copyPattern;
        meta.algolevel1Type = algolevel1Type;
        meta.hugeData = hugeData;
        meta.isSmallCount = smallCount;
        meta.isEnableCache = CheckEnableCache(meta);
        meta.dataSplit = dataSplit;
        return meta;
    }

    static HcclOpMetaInfoDef GetOneForBroadcast(
        bool isRootRank, uint32_t rootRank, bool hugeData = false, bool isSmallCount = false, u64 sliceNum = 1,
        CopyPattern copyPattern = CopyPattern::BCOPY)
    {
        HcclOpMetaInfoDef meta;
        meta.opType = HcclCMDType::HCCL_CMD_BROADCAST;
        meta.isSmallCount = isSmallCount; // 是否小数据
        meta.isRootRank = isRootRank;
        meta.rootRank = rootRank;
        meta.hugeData = hugeData;
        meta.sliceNum = sliceNum;
        meta.copyPattern = copyPattern;
        meta.isEnableCache = CheckEnableCache(meta);
        return meta;
    }

    static HcclOpMetaInfoDef GetOneForScatter(uint32_t rootRank, bool hugeData = false)
    {
        HcclOpMetaInfoDef meta;
        meta.opType = HcclCMDType::HCCL_CMD_SCATTER;
        meta.rootRank = rootRank;
        meta.hugeData = hugeData;
        meta.isEnableCache = CheckEnableCache(meta);
        return meta;
    }

    static HcclOpMetaInfoDef GetOneForReduceScatter(
        u32 algolevel1Type = 0, HcclDataType dataType = HCCL_DATA_TYPE_RESERVED,
        ReduceType reduceType = ReduceType::INLINE_REDUCE, bool hugeData = false, bool isSmallCount = false,
        CopyPattern copyPattern = CopyPattern::BCOPY, bool dataSplit = false, u8 deterministicConfig = 0,
        bool isAivModeConfig = false, bool preloadCopyFlag = false)
    {
        HcclOpMetaInfoDef meta;
        meta.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;
        meta.reduceType = reduceType;
        meta.dataType = dataType;
        meta.algolevel1Type = algolevel1Type;
        meta.hugeData = hugeData;
        meta.isSmallCount = isSmallCount; // 是否小数据
        meta.copyPattern = copyPattern;
        meta.isEnableCache = CheckEnableCache(meta);
        meta.dataSplit = dataSplit;
        meta.deterministic = deterministicConfig;
        meta.isAivMode = isAivModeConfig;
        meta.preloadCopyOpt = preloadCopyFlag;
        return meta;
    }

    static HcclOpMetaInfoDef GetOneForReduceScatterV(
        u32 algolevel1Type = 0, HcclDataType dataType = HCCL_DATA_TYPE_RESERVED,
        ReduceType reduceType = ReduceType::INLINE_REDUCE, bool hugeData = false, bool isSmallCount = false,
        CopyPattern copyPattern = CopyPattern::BCOPY, bool dataSplit = false, u8 deterministicConfig = 0)
    {
        HcclOpMetaInfoDef meta;
        meta.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V;
        meta.reduceType = reduceType;
        meta.dataType = dataType;
        meta.algolevel1Type = algolevel1Type;
        meta.hugeData = hugeData;
        meta.isSmallCount = isSmallCount; // 是否小数据
        meta.copyPattern = copyPattern;
        meta.isEnableCache = CheckEnableCache(meta);
        meta.dataSplit = dataSplit;
        meta.deterministic = deterministicConfig;
        return meta;
    }

    static HcclOpMetaInfoDef
    GetOneForAllToAll(CopyPattern copyPattern, u64 dataSize, bool hugeData = false, bool isAivMode = false)
    {
        HcclOpMetaInfoDef meta;
        meta.opType = HcclCMDType::HCCL_CMD_ALLTOALL;
        meta.copyPattern = copyPattern;
        meta.alltoallvcSendDataSize = dataSize;
        meta.hugeData = hugeData;
        meta.isAivMode = isAivMode;
        meta.isEnableCache = CheckEnableCache(meta);
        return meta;
    }

    static HcclOpMetaInfoDef
    GetOneForAllToAllV(CopyPattern copyPattern, u64 dataSize, bool hugeData = false, bool isSmallCount = false)
    {
        HcclOpMetaInfoDef meta;
        meta.opType = HcclCMDType::HCCL_CMD_ALLTOALLV;
        meta.copyPattern = copyPattern;
        meta.alltoallvSendDataSize = dataSize;
        meta.hugeData = hugeData;
        meta.isSmallCount = isSmallCount; // 是否小数据
        meta.isEnableCache = CheckEnableCache(meta);
        return meta;
    }

    static HcclOpMetaInfoDef GetOneForAllToAllVC(CopyPattern copyPattern, u64 dataSize, bool hugeData = false)
    {
        HcclOpMetaInfoDef meta;
        meta.opType = HcclCMDType::HCCL_CMD_ALLTOALLVC;
        meta.copyPattern = copyPattern;
        meta.alltoallvcSendDataSize = dataSize;
        meta.hugeData = hugeData;
        meta.isEnableCache = CheckEnableCache(meta);
        return meta;
    }

    static HcclOpMetaInfoDef GetOneForSend()
    {
        HcclOpMetaInfoDef meta;
        meta.opType = HcclCMDType::HCCL_CMD_SEND;
        meta.isEnableCache = CheckEnableCache(meta);
        return meta;
    }

    static HcclOpMetaInfoDef GetOneForReceive()
    {
        HcclOpMetaInfoDef meta;
        meta.opType = HcclCMDType::HCCL_CMD_RECEIVE;
        meta.isEnableCache = CheckEnableCache(meta);
        return meta;
    }

    static HcclOpMetaInfoDef GetOneForBatchSendRecv()
    {
        HcclOpMetaInfoDef meta;
        meta.opType = HcclCMDType::HCCL_CMD_BATCH_SEND_RECV;
        meta.isEnableCache = CheckEnableCache(meta);
        return meta;
    }

    static HcclOpMetaInfoDef GetOneForReduce(
        bool isRootRank, uint32_t rootRank, u32 algolevel1Type = 0, HcclDataType dataType = HCCL_DATA_TYPE_RESERVED,
        ReduceType reduceType = ReduceType::INLINE_REDUCE, bool hugeDataConfig = false, u8 deterministicConfig = 0)
    {
        HcclOpMetaInfoDef meta;
        meta.opType = HcclCMDType::HCCL_CMD_REDUCE;
        meta.isRootRank = isRootRank;
        meta.rootRank = rootRank;
        meta.reduceType = reduceType;
        meta.dataType = dataType;
        meta.algolevel1Type = algolevel1Type;
        meta.hugeData = hugeDataConfig;
        meta.isEnableCache = CheckEnableCache(meta);
        meta.deterministic = deterministicConfig;
        return meta;
    }

    std::string GetCacheKey() const
    {
#ifndef CCL_KERNEL_AICPU
        std::string key;
        key.reserve(KEY_SIZE);
        key.append(std::to_string(static_cast<int>(opType)));
        key.append(1, '_');
        key.append(1, isRootRank ? '1' : '0');
        key.append(1, '_');
        key.append(std::to_string(static_cast<int>(reduceType)));
        key.append(1, '_');
        key.append(std::to_string(rootRank));
        key.append(1, '_');
        key.append(std::to_string(sliceNum));
        key.append(1, '_');
        key.append(std::to_string(static_cast<int>(dataType)));
        key.append(1, '_');
        key.append(1, isSmallCount ? '1' : '0');
        key.append(1, '_');
        key.append(1, isDefaultPath ? '1' : '0');
        key.append(1, '_');
        key.append(std::to_string(pipelineSliceNum));
        key.append(1, '_');
        key.append(std::to_string(algolevel1Type));
        key.append(1, '_');
        key.append(std::to_string(static_cast<int>(copyPattern)));
        key.append(1, '_');
        key.append(1, dataSplit ? '1' : '0');
        key.append(1, '_');
        key.append(1, isAivMode ? '1' : '0');
        key.append(1, '_');
        key.append(std::to_string(deterministic));
        key.append(1, '_');
        key.append(std::to_string(preloadCopyOpt));
        return key;
#else
        return "";
#endif
    }
};
} // namespace hccl
#endif // FFTS_COMMON_PUB_H
