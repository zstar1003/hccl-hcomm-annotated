/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_COLL_ALG_CCU_SELECTOR
#define HCCLV2_COLL_ALG_CCU_SELECTOR

#include "base_selector.h"
#include "coll_alg_params.h"
#include "coll_operator.h"
#include "virtual_topo.h"

namespace Hccl {

enum AlgorithmType {
    CcuAllGatherMesh1D = 0,
    CcuAllGatherMeshMem2Mem1D,
    CcuAllGatherMesh2D,
    CcuReduceScatterMesh1D,
    CcuReduceScatterMeshMem2Mem1D,
    CcuReduceScatterMesh2D,
    CcuAllReduceMesh1D,
    CcuAllReduceMeshMem2Mem1D,
    CcuAllReduceMesh2DOneShot,
    CcuReduceMesh1D,
    CcuReduceMesh2D,
    CcuAlltoAllMesh1D,
    CcuAlltoAllVMesh1D,
    CcuHalfAll2AllVMesh1D
};

class Mc2Selector : public BaseSelector {
public:
    SelectorStatus
    SelectDefaultCcuMsAlgo(const CollAlgOperator& op, const CollAlgParams& params, std::string& primQueueGenName) const;

    SelectorStatus SelectDefaultCcuSchedAlgo(
        const CollAlgOperator& op, const CollAlgParams& params, std::string& primQueueGenName) const;

    SelectorStatus
    SelectDefaultAicpuAlgo(const CollAlgOperator& op, const CollAlgParams& params, std::string& primQueueGenName) const;

    SelectorStatus
    SelectCcuMsAlgo(const CollAlgOperator& op, CollAlgParams& params, std::string& primQueueGenName) const;

    SelectorStatus
    SelectCcuSchedAlgo(const CollAlgOperator& op, CollAlgParams& params, std::string& primQueueGenName) const;

    SelectorStatus
    SelectAicpuAlgo(const CollAlgOperator& op, CollAlgParams& params, std::string& primQueueGenName) const;

    SelectorStatus Select(const CollAlgOperator& op, CollAlgParams& params, std::string& primQueueGenName) override;

    AlgorithmType GetAlgorithmTypeForMC2CCU(const std::string& name);

private:
    std::map<std::string, AlgorithmType> algorithmMap_
        = {{"CcuAllGatherMesh1D", CcuAllGatherMesh1D},
           {"CcuAllGatherMeshMem2Mem1D", CcuAllGatherMeshMem2Mem1D},
           {"CcuAllGatherMesh2D", CcuAllGatherMesh2D},
           {"CcuReduceScatterMesh1D", CcuReduceScatterMesh1D},
           {"CcuReduceScatterMeshMem2Mem1D", CcuReduceScatterMeshMem2Mem1D},
           {"CcuReduceScatterMesh2D", CcuReduceScatterMesh2D},
           {"CcuAllReduceMesh1D", CcuAllReduceMesh1D},
           {"CcuAllReduceMeshMem2Mem1D", CcuAllReduceMeshMem2Mem1D},
           {"CcuAllReduceMesh2DOneShot", CcuAllReduceMesh2DOneShot},
           {"CcuReduceMesh1D", CcuReduceMesh1D},
           {"CcuReduceMesh2D", CcuReduceMesh2D},
           {"CcuAlltoAllMesh1D", CcuAlltoAllMesh1D},
           {"CcuAlltoAllVMesh1D", CcuAlltoAllVMesh1D},
           {"CcuHalfAll2AllVMesh1D", CcuHalfAll2AllVMesh1D}};
};
} // namespace Hccl
#endif
