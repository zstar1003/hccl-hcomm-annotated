/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALG_CONFIGURATOR_H
#define ALG_CONFIGURATOR_H

#include <vector>
#include <map>

#include <hccl/base.h>
#include <hccl/hccl_types.h>
#include "log.h"
#include "hccl_common.h"
#include "common.h"
#include "hccl_impl_pub.h"

namespace hccl {

class AlgConfigurator {
public:
    AlgConfigurator(HcclAlgoAttr& algoAttr, HcclTopoAttr& topoAttr);
    ~AlgConfigurator();
    HcclResult Init(bool isHeterogComm);

    HcclResult SelectAlgType(u32 moduleNum, const DevType deviceType, std::map<HcclCMDType, AlgType>& algType);
    HcclResult SelectCurrOpAlgType(
        u32 moduleNum, const DevType deviceType, HcclCMDType opType, std::map<HcclCMDType, AlgType>& algType);
    HcclResult SetAlgoLevel0(HcclAlgoType algoConfig, AlgTypeLevel0& algType);
    HcclResult SetAlgoLevel1(HcclAlgoType algoConfig, u32 moduleNum, AlgTypeLevel1& algType, HcclCMDType opType);
    HcclResult GetDefaultAlgoLevel1V1(u32 moduleNum, AlgTypeLevel1& algType) const;
    HcclResult SetAlgoLevel2(HcclAlgoType algoConfig, AlgTypeLevel2& algType);

    HcclResult SetAlgoLevel0StandardCard(HcclAlgoType algoConfig, AlgTypeLevel0& algType);
    HcclResult GetDefaultAlgoLevel0StandardCard(AlgTypeLevel0& algType) const;
    HcclResult SetAlgoLevel0Module(HcclAlgoType algoConfig, AlgTypeLevel0& algType);
    HcclResult GetDefaultAlgoLevel0Module(AlgTypeLevel0& algType);
    bool IsHCCSSWNumEqualToTwiceSIONum();
    HcclResult CheckAlgType(const AlgType algType);
    AlgTypeLevel0 GetLevel0AlgType(const AlgType algType) const;
    HcclResult GetTopoTypeByAlgType(const AlgType& algType, const DevType deviceType, TopoType& topoType);
    HcclResult GetAlgType(AlgType& algType, HcclCMDType opType);
    bool SupportDeterministicOptim() const;
    HcclResult SetAlgType(AlgType algType, HcclCMDType opType);
    void GetTopoType(TopoType& topoType);
    void GetAlgTypeDirect(AlgType& algType, HcclCMDType opType);
    HcclResult GetAlgoLevel1DefaultSwitch(bool& isAlgoLevel1Default, HcclCMDType opType);
    const HcclTopoAttr& GetTopoAttr();
    const HcclAlgoAttr& GetAlgoAttr();

private:
    HcclAlgoAttr& algoAttr_;
    HcclTopoAttr& topoAttr_;
    u8 deterministic_; // 确定性计算配置：0-关闭，1-开启确定性（不开启规约保序），2-开启确定性&规约保序 其他数字暂时保留
    TopoType topoType_ = TopoType::TOPO_TYPE_COMMON;

    std::map<HcclCMDType, AlgType> algType_ = {
        {HcclCMDType::HCCL_CMD_INVALID, AlgType()},   {HcclCMDType::HCCL_CMD_BROADCAST, AlgType()},
        {HcclCMDType::HCCL_CMD_ALLREDUCE, AlgType()}, {HcclCMDType::HCCL_CMD_REDUCE, AlgType()},
        {HcclCMDType::HCCL_CMD_SEND, AlgType()},      {HcclCMDType::HCCL_CMD_RECEIVE, AlgType()},
        {HcclCMDType::HCCL_CMD_ALLGATHER, AlgType()}, {HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgType()},
        {HcclCMDType::HCCL_CMD_ALLTOALLV, AlgType()}, {HcclCMDType::HCCL_CMD_ALLTOALLVC, AlgType()},
        {HcclCMDType::HCCL_CMD_ALLTOALL, AlgType()},  {HcclCMDType::HCCL_CMD_GATHER, AlgType()},
        {HcclCMDType::HCCL_CMD_SCATTER, AlgType()},   {HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, AlgType()},
        {HcclCMDType::HCCL_CMD_MAX, AlgType()},       {HcclCMDType::HCCL_CMD_ALL, AlgType()},
    }; // 算法类型

    std::map<HcclCMDType, bool> isAlgoLevel1Default_ = {
        {HcclCMDType::HCCL_CMD_INVALID, false},   {HcclCMDType::HCCL_CMD_BROADCAST, false},
        {HcclCMDType::HCCL_CMD_ALLREDUCE, false}, {HcclCMDType::HCCL_CMD_REDUCE, false},
        {HcclCMDType::HCCL_CMD_SEND, false},      {HcclCMDType::HCCL_CMD_RECEIVE, false},
        {HcclCMDType::HCCL_CMD_ALLGATHER, false}, {HcclCMDType::HCCL_CMD_REDUCE_SCATTER, false},
        {HcclCMDType::HCCL_CMD_ALLTOALLV, false}, {HcclCMDType::HCCL_CMD_ALLTOALLVC, false},
        {HcclCMDType::HCCL_CMD_ALLTOALL, false},  {HcclCMDType::HCCL_CMD_GATHER, false},
        {HcclCMDType::HCCL_CMD_SCATTER, false},   {HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, false},
        {HcclCMDType::HCCL_CMD_MAX, false},       {HcclCMDType::HCCL_CMD_ALL, false},
    };
};
} // namespace hccl

#endif
