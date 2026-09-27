/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _SIM_RUNNER_COMMOM_H_
#define _SIM_RUNNER_COMMOM_H_

#include <thread>

#include "acl/acl_base.h"
#include "sim_models.h"
#include "db_sim_runner_db.h"

namespace sim {
aclError GetDeviceByLogicId(uint32_t deviceId, sim::Device& device);
aclError GetDeviceByRankId(uint32_t rankId, sim::Device& device);
aclError GetDeviceByPhysicalId(uint32_t deviceId, sim::Device& device);
aclError UpdateDeviceLogicId(uint64_t serverKey, uint32_t phyDevId, uint32_t logicDevId, uint32_t userId);
aclError UpdateSuperDeviceId(uint32_t logicDevId, uint32_t superDeviceId);
aclError GetCcuFromDeviceByDieId(uint64_t deviceKey, uint8_t dieId, sim::Ccu& ccu);
aclError GetCcuResourceByCcu(uint64_t ccuKey, sim::CcuResource& ccuRes);
aclError GetContextByDevId(uint32_t deviceId, sim::Context& context);
aclError GetPortByName(uint64_t serverKey, uint32_t phyDevId, const std::string& name, sim::Port& port);
aclError GetEndPointByIpAddr(const std::string& ip, sim::EndPoint& endPoint);
aclError GetPortById(uint64_t portId, sim::Port& port);

uint32_t GetAICpuCount(uint64_t deviceId);
uint32_t GetAICoreCount(uint64_t deviceId);
uint32_t GetVectorCoreCount(uint64_t deviceId);
int GetRankIdByDeviceId(uint32_t deviceId);
bool ResetAllDeviceLogicId();
bool GetRankIdByMPI(uint32_t& rankId, uint64_t& serverId);
uint64_t GetCurServerId();
} // namespace sim
#endif // _SIM_RUNNER_COMMOM_H_
