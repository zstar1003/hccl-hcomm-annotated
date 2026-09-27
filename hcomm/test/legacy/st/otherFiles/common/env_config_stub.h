/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_ENV_CONFIG_STUB_H
#define HCCLV2_ENV_CONFIG_STUB_H

#include "base_config_legacy.h"
#include <map>
#include <string>

namespace Hccl {

const std::map<std::string, std::string> defaultEnvCfgMap
    = {{"HCCL_IF_IP", "10.10.10.1"},
       {"HCCL_IF_BASE_PORT", "50000"},
       {"HCCL_SOCKET_IFNAME", "^=eth0,endvnic"},
       {"HCCL_WHITELIST_DISABLE", "0"},
       {"HCCL_WHITELIST_FILE", "llt_whitelist.txt"},
       {"HCCL_NPU_NET_PROTOCOL", "RDMA"},
       {"HCCL_SOCKET_FAMILY", "AF_INET6"},
       {"HCCL_CONNECT_TIMEOUT", "200"},
       {"HCCL_EXEC_TIMEOUT", "1800"},
       {"HCCL_RDMA_TC", "100"},
       {"HCCL_RDMA_SL", "3"},
       {"HCCL_RDMA_TIMEOUT", "6"},
       {"HCCL_RDMA_RETRY_CNT", "5"},
       {"HCCL_INTRA_PCIE_ENABLE", "1"},
       {"HCCL_INTRA_ROCE_ENABLE", "0"},
       {"HCCL_INTER_HCCS_DISABLE", "FALSE"},
       {"PRIM_QUEUE_GEN_NAME", "AllReduceRing"},
       {"HCCL_ALGO", "allreduce=level0:NA;level1:ring"},
       {"HCCL_BUFFSIZE", "200"},
       {"HCCL_OP_EXPANSION_MODE", "AI_CPU"},
       {"HCCL_DETERMINISTIC", "false"},
       {"HCCL_DIAGNOSE_ENABLE", "1"},
       {"HCCL_ENTRY_LOG_ENABLE", "1"},
       {"PROFILING_MODE", "true"},
       {"PROFILING_OPTIONS", "{\"output\":\"/tmp/"
                             "profiling\",\"training_trace\":\"on\",\"task_trace\":\"on\",\"fp_point\":\"\",\"bp_"
                             "point\":\"\",\"aic_metrics\":\"PipeUtilization\"}"},
       {"LD_LIBRARY_PATH", "/temp:/runtime"},
       {"HCCL_DETOUR", "detour:1"},
       {"CHIP_VERIFY_HCCL_CNT_NOTIFY_ENABLE", "1"},
       {"CHIP_VERIFY_HCCL_TOPO", "HCCL_TOPO_4P4K"},
       {"CHIP_VERIFY_ORCHESTRATE_WAY", "PRIM"}};

class EnvConfigStub {
public:
    EnvConfigStub();

    const EnvHostNicConfig& GetHostNicConfig();

    const EnvSocketConfig& GetSocketConfig();

    const EnvRtsConfig& GetRtsConfig();

    const EnvRdmaConfig& GetRdmaConfig();

    const EnvAlgoConfig& GetAlgoConfig();

    const EnvLogConfig& GetLogConfig();

    const EnvDetourConfig& GetDetourConfig();

private:
    EnvHostNicConfig hostNicCfg;
    EnvSocketConfig socketCfg;
    EnvRtsConfig rtsCfg;
    EnvRdmaConfig rdmaCfg;
    EnvAlgoConfig algoCfg;
    EnvLogConfig logCfg;
    EnvDetourConfig detourCfg;
    void Parse();
};

} // namespace Hccl

#endif // HCCLV2_ENV_CONFIG_STUB_H
