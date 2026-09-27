/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_COLL_ALG_BASE_SELECTOR
#define HCCLV2_COLL_ALG_BASE_SELECTOR

#include <string>
#include "coll_alg_params.h"
#include "coll_operator.h"
#include "virtual_topo.h"
#include "log.h"
#include "env_config_v2.h"

namespace Hccl {
constexpr u32 CCU_MS_MODE = 2;

constexpr u32 BIG_CLOS_RANGE = 8;

enum class SelectorStatus { MATCH, NOT_MATCH };
enum class Level0Shape {
    MESH_1D = 1,
    MESH_2D = 2,
    CLOS = 3,
    MESH_1D_CLOS = 4,
};

const std::map<OpType, std::string> OP_TYPE_TO_AICPU_SOLE_ALG_MAP = {
    {OpType::ALLGATHER, "InsAllGatherMesh"}, {OpType::REDUCESCATTER, "InsReduceScatterNHR"},
    {OpType::ALLREDUCE, "InsAllReduceNHR"},  {OpType::ALLTOALL, "InsAlltoAllMesh"},
    {OpType::ALLTOALLV, "InsAlltoAllvMesh"}, {OpType::ALLTOALLVC, "InsAlltoAllvcMesh"},
};

class BaseSelector {
public:
    virtual ~BaseSelector() {};
    BaseSelector& SetVirtualTopo(RankGraph* rankGraph);
    BaseSelector& SetDevType(DevType devType);
    BaseSelector& SetMyRank(RankId myRank);
    BaseSelector& SetRankSize(u32 rankSize);
    BaseSelector& SetSeverId(std::string severId);
    BaseSelector& SetDeviceNumPerSever(u32 deviceNumPerSever);
    BaseSelector& SetServerNum(u32 serverNum);
    BaseSelector& SetOpConfig(OpExecuteConfig opConfig);
    BaseSelector& SetIsMc2(bool isMc2);

    RankGraph* GetVirtualTopo();
    DevType GetDevType();
    RankId GetMyRank() const;
    u32 GetRankSize() const;
    std::string GetSeverId();
    u32 GetDeviceNumPerSever() const;
    u32 GetServerNum() const;

    bool IsInputOutputOverlap(const std::shared_ptr<Buffer>& inputMem, const std::shared_ptr<Buffer>& outputMem) const;
    virtual SelectorStatus Select(const CollAlgOperator& op, CollAlgParams& params, std::string& primQueueGenName) = 0;

protected:
    struct NetLayerDetails {
        u32 netLayerNum;
        std::set<u32> netLayers;
        std::vector<u32> netInstNumOfLayer;
        std::vector<std::vector<u32>> instSizeListOfLayer;
        std::vector<u32> localNetInsSizeOfLayer;
    };
    struct TopoInstDetails {
        u32 topoInstNum;
        std::vector<u32> sizeOfTopo;
        std::vector<TopoType> typeOfTopo;
        std::vector<std::vector<u32>> ranksInTopo;
        std::map<TopoType, std::vector<u32>> rankNumForTopoType;
    };
    struct TopoInfo {
        u32 levelNum;
        Level0Shape level0Shape;
        NetLayerDetails netLayerDetails;
        std::vector<TopoInstDetails> topoInstDetailsOfLayer;

        bool Level0Nhr{false};
        bool Level1Nhr{false};

        bool level0PcieMix{false};
        bool level0BigClosRange{false};
    };
    u32 Gcd(u32 a, u32 b) const;                           // 自定义实现的 gcd 函数（兼容旧版本 C++）
    u32 GcdOfArray(const std::vector<u32>& numbers) const; // 计算数组中所有元素的最大公约数
    u32 GetLevel0Gcd() const;
    void CalcTopoShape(TopoInfo& topoInfo) const;
    bool
    IsAsymmetricTopoShapeLevel1Nhr(const std::vector<std::vector<u32>>& localIdPerBoard, u32 gcdRankSizeLevel0) const;
    bool IsTopoShapeLevel0Regular(const std::vector<std::vector<u32>>& localIdPerBoard) const;
    bool IsLayerAllConnetedWithTopo(const TopoInfo& topoInfo, const u32 netLayer, const TopoType topoType) const;
    HcclResult CalcLevel0TopoShape(TopoInfo& topoInfo) const;
    HcclResult ExtractNetLayerDetails(TopoInfo& topoInfo) const;
    HcclResult ExtractTopoDetails(TopoInfo& topoInfo) const;
    bool Is2DieFullMesh() const;
    bool IsLevel0PcieMix() const;
    RankGraph* rankGraph_ = nullptr;
    OpExecuteConfig opConfig_;
    DevType devType_;
    RankId myRank_;
    u32 rankSize_;
    std::string severId_;
    u32 deviceNumPerSever_;
    u32 serverNum_;
    bool isMc2_{false};
};

} // namespace Hccl
#endif
