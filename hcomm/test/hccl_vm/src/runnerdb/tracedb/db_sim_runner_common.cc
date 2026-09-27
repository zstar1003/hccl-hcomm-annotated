/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <set>
#include <iostream>
#include <runnerdb/db_sim_runner_common.h>

#include "sim_log.h"
#include "db_sim_runner_common.h"

uint64_t g_cur_server_key = 0;

namespace sim {
aclError GetDeviceByLogicId(uint32_t deviceId, sim::Device& device)
{
    auto ret = RunnerDB::GetOneByPred<sim::Device>([deviceId](const sim::Device& d) {
        return d.server_id == g_cur_server_key && d.logic_id == deviceId;
    });
    if (!ret.second) {
        HCCL_VM_ERROR("cannot find device by logical id {:d}", deviceId);
        return ACL_ERROR_INVALID_PARAM;
    }

    device = ret.first;
    return ACL_SUCCESS;
}

aclError GetDeviceByRankId(uint32_t rankId, sim::Device& device)
{
    // 目前deviceId = rankId
    // 查看rank表，根据rankId获取device
    auto rankRet = RunnerDB::GetOneByPred<sim::Rank>([rankId](const sim::Rank& d) {
        return d.rank_id == rankId;
    });
    if (!rankRet.second) {
        HCCL_VM_ERROR("cannot find rank by rank id {:d}", rankId);
        return ACL_ERROR_INVALID_PARAM;
    }
    auto deviceKey = rankRet.first.device_id;

    auto deviceRet = RunnerDB::GetById<sim::Device>(deviceKey);
    if (!deviceRet.has_value()) {
        // not find
        HCCL_VM_ERROR("can not find device by rank id {:d}", rankId);
        return ACL_ERROR_INVALID_PARAM;
    }
    device = *deviceRet;

    return ACL_SUCCESS;
}

aclError GetDeviceByPhysicalId(uint32_t deviceId, sim::Device& device)
{
    auto ret = RunnerDB::GetOneByPred<sim::Device>([deviceId](const sim::Device& d) {
        return d.server_id == g_cur_server_key && d.physical_id == deviceId;
    });
    if (!ret.second) {
        HCCL_VM_ERROR(
            "[{}] cannot find device by physical id {:d}, server_key {:d}", __func__, deviceId, g_cur_server_key);
        return ACL_ERROR_INVALID_PARAM;
    }

    device = ret.first;
    return ACL_SUCCESS;
}

aclError GetServerByKey(uint64_t serverKey, sim::Server& server)
{
    auto ret = RunnerDB::GetOneByPred<sim::Server>([serverKey](const sim::Server& d) {
        return d.id == serverKey;
    });
    if (!ret.second) {
        HCCL_VM_ERROR("cannot find server by key {:d}", serverKey);
        return ACL_ERROR_INVALID_PARAM;
    }

    server = ret.first;
    return ACL_SUCCESS;
}

aclError GetDeviceByServerKeyAndPhysicalId(uint64_t serverKey, uint32_t deviceId, sim::Device& device)
{
    auto ret = RunnerDB::GetOneByPred<sim::Device>([serverKey, deviceId](const sim::Device& d) {
        return d.server_id == serverKey && d.physical_id == deviceId;
    });
    if (!ret.second) {
        HCCL_VM_ERROR("cannot find device by server key {:d}, physical id {:d}", serverKey, deviceId);
        return ACL_ERROR_INVALID_PARAM;
    }

    device = ret.first;
    return ACL_SUCCESS;
}

aclError UpdateDeviceLogicId(uint64_t serverKey, uint32_t phyDevId, uint32_t logicDevId, uint32_t userId)
{
    sim::Device device{};
    auto ret = GetDeviceByServerKeyAndPhysicalId(serverKey, phyDevId, device);
    if (ret != ACL_SUCCESS) {
        return ACL_ERROR_INVALID_PARAM;
    }

    auto deviceKey = device.id;
    RunnerDB::Update<sim::Device>(deviceKey, [deviceKey, logicDevId, userId](sim::Device& dev) {
        dev.logic_id = logicDevId;
        dev.user_id = userId;
        dev.status = 1; // 设备状态设置为可用
    });

    return ACL_SUCCESS;
}

bool ResetAllDeviceLogicId()
{
    auto allDevices = RunnerDB::GetByPred<sim::Device>([](const sim::Device& d) {
        return true;
    });
    if (allDevices.empty()) {
        HCCL_VM_ERROR("cannot find any device");
        return false;
    }

    for (auto& device : allDevices) {
        auto deviceKey = device.id;
        RunnerDB::Update<sim::Device>(deviceKey, [deviceKey](sim::Device& dev) {
            dev.logic_id = 0xFFFF;
            dev.status = 0; // 重置设备状态，防止残留 status=1 干扰后续 aclrtGetDeviceCount
            dev.user_id = 0xFFFF;
        });
    }

    return true;
}

aclError UpdateSuperDeviceId(uint32_t logicDevId, uint32_t superDeviceId)
{
    sim::Device device{};
    auto ret = GetDeviceByLogicId(logicDevId, device);
    if (ret != ACL_SUCCESS) {
        return ACL_ERROR_INVALID_PARAM;
    }

    auto deviceKey = device.id;
    RunnerDB::Update<sim::Device>(deviceKey, [deviceKey, superDeviceId](sim::Device& dev) {
        dev.super_device_id = superDeviceId;
    });

    return ACL_SUCCESS;
}

aclError GetCcuFromDeviceByDieId(uint64_t deviceKey, uint8_t dieId, sim::Ccu& ccu)
{
    auto ret = RunnerDB::GetOneByPred<sim::Ccu>([deviceKey, dieId](const sim::Ccu& c) {
        return c.device_id == deviceKey && c.die_id == dieId;
    });
    if (!ret.second) {
        HCCL_VM_ERROR("cannot find ccu from device {:d} by die {:d}", deviceKey, static_cast<uint32_t>(dieId));
        return ACL_ERROR_INVALID_PARAM;
    }

    ccu = ret.first;
    return ACL_SUCCESS;
}

aclError GetCcuResourceByCcu(uint64_t ccuKey, sim::CcuResource& ccuRes)
{
    auto ret = RunnerDB::GetOneByPred<sim::CcuResource>([ccuKey](const sim::CcuResource& cr) {
        return cr.ccu_id == ccuKey;
    });
    if (!ret.second) {
        HCCL_VM_ERROR("cannot find ccu resource from ccu {:d}", ccuKey);
        return ACL_ERROR_INVALID_PARAM;
    }

    ccuRes = ret.first;
    return ACL_SUCCESS;
}

aclError GetContextByDevId(uint32_t deviceId, sim::Context& context)
{
    auto ret = RunnerDB::GetOneByPred<sim::Context>([deviceId](const sim::Context& ctx) {
        return ctx.device_id == deviceId && ctx.is_default == 1;
    });
    if (!ret.second) {
        HCCL_VM_ERROR("cannot find context by logical device id {:d}", deviceId);
        return ACL_ERROR_INVALID_PARAM;
    }

    context = ret.first;
    return ACL_SUCCESS;
}

aclError GetPortByName(uint64_t serverKey, uint32_t phyDevId, const std::string& name, sim::Port& port)
{
    sim::Device device{};
    auto ret1 = GetDeviceByServerKeyAndPhysicalId(serverKey, phyDevId, device);
    if (ret1 != ACL_SUCCESS) {
        HCCL_VM_ERROR("get device by physical device id({:d}) failed", phyDevId);
        return ret1;
    }

    auto deviceKey = device.id;
    auto ret2 = RunnerDB::GetOneByPred<sim::Port>([deviceKey, name](const sim::Port& p) {
        return deviceKey == p.device_id && strcmp(p.name, name.c_str()) == 0;
    });
    if (!ret2.second) {
        HCCL_VM_ERROR("cannot find port by device:{:d} name:{}", deviceKey, name.c_str());
        return ACL_ERROR_INVALID_PARAM;
    }

    port = ret2.first;
    return ACL_SUCCESS;
}

aclError GetEndPointByIpAddr(const std::string& ip, sim::EndPoint& endPoint)
{
    auto ret = RunnerDB::GetOneByPred<sim::EndPoint>([ip](const sim::EndPoint& ep) {
        return strcmp(ep.ip_addr, ip.c_str()) == 0;
    });
    if (!ret.second) {
        HCCL_VM_ERROR("cannot find EndPoint by ip addr:{}", ip.c_str());
        return ACL_ERROR_INVALID_PARAM;
    }
    endPoint = ret.first;
    return ACL_SUCCESS;
}

aclError GetPortById(uint64_t portId, sim::Port& port)
{
    auto ret = RunnerDB::GetOneByPred<sim::Port>([portId](const sim::Port& p) {
        return portId == p.id;
    });
    if (!ret.second) {
        HCCL_VM_ERROR("cannot find port by id: {:d}", portId);
        return ACL_ERROR_INVALID_PARAM;
    }

    port = ret.first;
    return ACL_SUCCESS;
}

int GetRankIdByDeviceId(uint32_t deviceId)
{
    auto rank = RunnerDB::GetOneByPred<sim::Rank>([deviceId](const sim::Rank& r) {
        return r.device_id == deviceId;
    });
    if (!rank.second) {
        HCCL_VM_ERROR("can not find any rank device:{:d}", deviceId);
        return 0;
    }

    return rank.first.rank_id;
}

uint32_t GetAICpuCount(uint64_t deviceId)
{
    // ACL_DEV_ATTR_AICPU_CORE_NUM AI CPU数量。
    auto ret = RunnerDB::GetOneByPred<sim::Device>([deviceId](const sim::Device& d) {
        return d.logic_id == deviceId;
    });
    if (!ret.second) {
        HCCL_VM_ERROR("cannot find device by physical id {:d}", deviceId);
        return 0;
    }

    auto deviceIdx = ret.first.id;

    auto aiCpus = RunnerDB::GetByPred<sim::TaskSchedulerDevice>([deviceIdx](const sim::TaskSchedulerDevice& tsDev) {
        return tsDev.device_id == deviceIdx && tsDev.type == (uint8_t)TS_DEV_TYPE_CPU;
    });

    return aiCpus.size();
}

uint32_t GetAICoreCount(uint64_t deviceId)
{
    // ACL_DEV_ATTR_AICORE_CORE_NUM AI CPU数量。
    auto ret = RunnerDB::GetOneByPred<sim::Device>([deviceId](const sim::Device& d) {
        return d.logic_id == deviceId;
    });
    if (!ret.second) {
        HCCL_VM_ERROR("cannot find device by physical id {:d}", deviceId);
        return 0;
    }

    auto deviceIdx = ret.first.id;

    auto scalars = RunnerDB::GetByPred<sim::TaskSchedulerDevice>([deviceIdx](const sim::TaskSchedulerDevice& tsDev) {
        return tsDev.device_id == deviceIdx && tsDev.type == (uint8_t)TS_DEV_TYPE_SCALAR;
    });

    return scalars.size();
}

uint32_t GetVectorCoreCount(uint64_t deviceId)
{
    // ACL_DEV_ATTR_VECTOR_CORE_NUM AI CPU数量。
    auto ret = RunnerDB::GetOneByPred<sim::Device>([deviceId](const sim::Device& d) {
        return d.logic_id == deviceId;
    });
    if (!ret.second) {
        HCCL_VM_ERROR("cannot find device by physical id {:d}", deviceId);
        return 0;
    }

    auto deviceIdx = ret.first.id;

    auto scalars = RunnerDB::GetByPred<sim::TaskSchedulerDevice>([deviceIdx](const sim::TaskSchedulerDevice& tsDev) {
        return tsDev.device_id == deviceIdx && tsDev.type == (uint8_t)TS_DEV_TYPE_SCALAR;
    });

    uint32_t vectorCoreCount = 0;
    for (auto scalar : scalars) {
        auto tsIdx = scalar.id;
        auto vectorCores = RunnerDB::GetByPred<sim::ComputeDie>([tsIdx](const sim::ComputeDie& die) {
            return die.ts_id == tsIdx && die.type == (uint8_t)COMPUTE_TYPE_VECTOR;
        });

        vectorCoreCount += vectorCores.size();
    }
    return vectorCoreCount;
}

uint32_t GetCubeCoreCount(uint64_t deviceId)
{
    auto ret = RunnerDB::GetOneByPred<sim::Device>([deviceId](const sim::Device& d) {
        return d.logic_id == deviceId;
    });
    if (!ret.second) {
        HCCL_VM_ERROR("cannot find device by physical id {:d}", deviceId);
        return 0;
    }

    auto deviceIdx = ret.first.id;

    auto scalars = RunnerDB::GetByPred<sim::TaskSchedulerDevice>([deviceIdx](const sim::TaskSchedulerDevice& tsDev) {
        return tsDev.device_id == deviceIdx && tsDev.type == (uint8_t)TS_DEV_TYPE_SCALAR;
    });

    uint32_t vectorCoreCount = 0;
    for (auto scalar : scalars) {
        auto tsIdx = scalar.id;
        auto vectorCores = RunnerDB::GetByPred<sim::ComputeDie>([tsIdx](const sim::ComputeDie& die) {
            return die.ts_id == tsIdx && die.type == (uint8_t)COMPUTE_TYPE_CUBE;
        });

        vectorCoreCount += vectorCores.size();
    }
    return vectorCoreCount;
}

std::set<uint64_t> GetUsedServerNum()
{
    try {
        auto allRanks = RunnerDB::GetByPred<sim::Rank>([](const sim::Rank& d) {
            return d.state == 1;
        });
        std::set<uint64_t> usedServerIds;
        for (auto& rank : allRanks) {
            auto deviceKey = rank.device_id;
            auto ret = RunnerDB::GetOneByPred<sim::Device>([deviceKey](const sim::Device& d) {
                return d.id == deviceKey;
            });
            if (!ret.second) {
                continue;
            }
            auto serverId = ret.first.server_id;
            usedServerIds.insert(serverId);
        }
        return usedServerIds;
    } catch (const std::exception& e) {
        HCCL_VM_ERROR("exception: {}", e.what());
        return {};
    }
}

bool GetRankIdByMPI(uint32_t& rankId, uint64_t& serverId)
{
    const char* ompiRankStr = std::getenv("OMPI_COMM_WORLD_RANK");
    const char* mpichRankStr = std::getenv("PMI_RANK");
    if (ompiRankStr == nullptr && mpichRankStr == nullptr) {
        // 单server用例，默认serverId为1
        auto usedServerIds = GetUsedServerNum();
        if (usedServerIds.size() == 1) {
            if (*usedServerIds.begin() != 1) {
                HCCL_VM_ERROR("env OMPI_COMM_WORLD_RANK or PMI_RANK are not found, serverId is not 1.");
                return false;
            }
            serverId = 1;
            return true;
        }
        HCCL_VM_ERROR("env OMPI_COMM_WORLD_RANK or PMI_RANK are not found, usedServerIds size is not 1.");
        return false;
    } else if (ompiRankStr != nullptr) {
        rankId = static_cast<uint32_t>(atoi(ompiRankStr));
    } else {
        rankId = static_cast<uint32_t>(atoi(mpichRankStr));
    }
    return true;
}

uint64_t GetCurServerId()
{
    if (g_cur_server_key != 0) {
        return g_cur_server_key;
    }

    uint32_t rankId;
    uint64_t serverId = 0;
    if (!sim::GetRankIdByMPI(rankId, serverId)) {
        HCCL_VM_ERROR("get rankId by MPI fail serverId:{:d}", serverId);
        return 0;
    }

    if (serverId == 0) {
        sim::Device device{};
        auto devRet = sim::GetDeviceByRankId(rankId, device);
        if (devRet != ACL_SUCCESS) {
            HCCL_VM_ERROR("device not found by rankId:{:d}", rankId);
            return 0;
        }
        serverId = device.server_id;
        HCCL_VM_DEBUG("logicId:{:d} rankId:{:d} key:{:d}", device.logic_id, rankId, device.id);
    }
    g_cur_server_key = serverId;
    return serverId;
}
} // namespace sim
