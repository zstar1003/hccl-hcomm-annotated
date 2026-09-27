/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_TOPO_MATCH_BASE
#define HCCLV2_TOPO_MATCH_BASE

#include <unordered_set>
#include <algorithm>

#include "virtual_topo.h"
#include "rank_gph.h"
#include "log.h"
#include "dev_type.h"
#include "const_val.h"

namespace Hccl {
constexpr int RANK_SIZE_TWO = 2;
constexpr int RANK_SIZE_THREE = 3;
constexpr int RANK_SIZE_FOUR = 4;
constexpr int RANK_SIZE_EIGHT = 8;

constexpr int COMM_LEVEL_SIZE_0 = 0;
constexpr int COMM_LEVEL_SIZE_1 = 1;
constexpr int COMM_LEVEL_SIZE_2 = 2;
constexpr int COMM_LEVEL_SIZE_3 = 3;

const std::vector<std::vector<u32>> SERVER_910A_4_RING_SEQUENCE
    = {{0, 1, 2, 6, 5, 4, 7, 3}, {0, 3, 7, 4, 5, 6, 2, 1}, {0, 2, 3, 1, 5, 7, 6, 4}, {0, 4, 6, 7, 5, 1, 3, 2}};

struct Hccl910AServerValid4PRanksVectorHashFuc {
    std::size_t operator()(const std::vector<s32> key) const
    {
        size_t ret = 0;
        for (auto it : key) {
            ret ^= static_cast<u32>(it);
        }
        return ret;
    }
};

const std::unordered_set<std::vector<s32>, Hccl910AServerValid4PRanksVectorHashFuc> SERVER_910A_VALID_4P_RANKS
    = {{0, 1, 4, 5}, {0, 2, 4, 6}, {0, 3, 4, 7}, {1, 2, 5, 6}, {1, 3, 5, 7}, {2, 3, 6, 7}, {0, 1, 2, 3}, {4, 5, 6, 7}};

const std::vector<u32> SERVER_910A_4P_SEQUENCE = {0, 1, 3, 2};

class TopoMatchBase {
public:
    explicit TopoMatchBase(const RankId vRank, const u32 rankSize, const RankGraph* rankGraph, const DevType devType);
    virtual ~TopoMatchBase();

    virtual std::string Describe() const = 0;

    virtual HcclResult MatchTopo(
        std::vector<std::vector<RankId>>& vTopo, std::vector<RankId>& virtRanks, std::map<RankId, u32>& virtRankMap);

    virtual HcclResult MatchTopo(
        std::vector<std::vector<std::vector<RankId>>>& vTopo, std::vector<std::vector<RankId>>& virtRanks,
        std::vector<std::map<RankId, u32>>& virtRankMap);

    virtual HcclResult SetTargetRanks(std::set<u32>& targetRanks);

    std::set<u32> batchSendRecvtargetRanks_; // for batchsendrecv create links

protected:
    bool IsAllRanksFullMeshConnected(std::set<RankId> rankSet) const;
    u32 GetPathNum(RankId srcRankId, RankId dstRankId) const;
    HcclResult GenVirtRankMapping(std::vector<RankId>& virtRanks, std::map<RankId, u32>& virtRankMap) const;
    HcclResult GenVirtRankMappingMultiLevel(
        std::vector<std::vector<RankId>>& virtRanks, std::vector<std::map<RankId, u32>>& virtRankMap) const;

    HcclResult CalcRankOnSamePlaneOfR0(
        std::vector<std::vector<RankId>>& rankOnSameBoardVector, std::vector<std::vector<RankId>>& rankOnSameSlotVector,
        std::vector<u32>& numRanksPerBoard) const;

    u32 GcdTwo(u32 a, u32 b) const;
    u32 GcdMultiple(const std::vector<u32>& numbers) const;

    HcclResult GenerateLevel1(
        const std::set<RankId>& rankSetLevel1, u32 gcdInstSize, RankId rankId,
        std::vector<std::vector<std::vector<RankId>>>& vTopo, std::vector<std::vector<RankId>>& virtRanks) const;

    template <typename T>
    using Matrix = std::vector<std::vector<T>>;
    template <typename T>
    using Tensor = std::vector<std::vector<std::vector<T>>>;

    template <typename T>
    std::string PrintSet(const std::set<T>& values) const
    {
        std::ostringstream oss;
        for (const auto& value : values) {
            oss << value << " ";
        }
        return oss.str();
    }

    template <typename T>
    std::string PrintVector(const std::vector<T>& values) const
    {
        std::ostringstream oss;
        for (const auto& value : values) {
            oss << value << " ";
        }
        return oss.str();
    }

    template <typename T>
    std::string PrintMatrix(const Matrix<T>& matrix) const
    {
        std::ostringstream oss;
        for (const auto& row : matrix) {
            oss << "{ ";
            for (const auto& val : row) {
                oss << val << " ";
            }
            oss << "}";
        }
        return oss.str();
    }

    template <typename T>
    std::string PrintTensor(const Tensor<T>& tensor) const
    {
        std::ostringstream oss;
        for (const auto& matrix : tensor) {
            oss << "[ " << PrintMatrix(matrix) << " ]";
        }
        return oss.str();
    }

    RankId myRank_ = INVALID_RANKID;
    u32 rankSize_ = 0;
    const RankGraph* rankGraph_ = nullptr;
    DevType devType_ = DevType::DEV_TYPE_NOSOC;
};

} // namespace Hccl

#endif // !HCCLV2_TOPO_MATCH_BASE
