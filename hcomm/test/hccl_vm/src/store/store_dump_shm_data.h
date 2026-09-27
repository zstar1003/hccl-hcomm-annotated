/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DUMP_FLAG_DATA_H
#define DUMP_FLAG_DATA_H

#include <cstdint>
#include <map>
#include <vector>

#include "sim_binary_data_type_pub.h"
#include "sim_common_defs.h"
#include "sim_op_db_types.h"

namespace HcclSim {
std::string GenDataId();
HcclVmResult DumpDataToFile(const std::string& dataId);

HcclVmResult DumpHcclVmFlagData(HcclSim::HcclVmFlagData& flagData);
HcclVmResult GetHcclVmFlagData(HcclSim::HcclVmFlagData& waitFlag);
HcclVmResult DumpHcclVmSynthesisData(const std::string& dataId);
HcclVmResult DumpHcclVmInstrData(const std::string& dataId);
HcclVmResult DumpHcclVmTask(const std::string& dataId);
HcclVmResult CreateChannelInfo(HcclVmSynData& hvmSynData);
HcclVmResult CreateJettyInfo(HcclVmSynData& hvmSynData);
HcclVmResult CreateMemoryInfo(HcclVmSynData& hvmSynData, const sim::OpMemInfoTab& memInfo, uint32_t rankId);
HcclVmResult CreateSimTaskMetaData(
    HcclVmTaskMetaData& hvmTaskMetaData,
    const std::map<uint32_t, std::vector<sim::CompositeOpDetail>>& compositeDataMap);
} // namespace HcclSim
#endif
