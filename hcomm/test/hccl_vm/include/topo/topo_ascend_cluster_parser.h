/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_VM_TOPO_ASCEND_CLUSTER_PARSER_H
#define HCCL_VM_TOPO_ASCEND_CLUSTER_PARSER_H

#include <cstring>
#include <json.hpp>
#include <set>
#include <string>
#include <vector>

#include "topo_cluster_parser.h"
#include "cmd_cluster_model_utils.h"
#include "sim_common_defs.h"

using namespace HcclSim;
using namespace std;
using json = nlohmann::json;

enum class HvmClusterStatus { COMM_DOMAIN_UNINIT, COMM_DOMAIN_INIT_DONE };

// 单例类，用于解析Ascend集群拓扑, 并初始化动态模型数据
class AscendClusterTopoParser {
public:
    // 单例模式
    static AscendClusterTopoParser& GetInstance()
    {
        static AscendClusterTopoParser instance;
        return instance;
    }

    // 禁用拷贝、移动、赋值运算符
    AscendClusterTopoParser(const AscendClusterTopoParser&) = delete;
    AscendClusterTopoParser& operator=(const AscendClusterTopoParser&) = delete;
    AscendClusterTopoParser(AscendClusterTopoParser&&) = delete;
    AscendClusterTopoParser& operator=(AscendClusterTopoParser&&) = delete;

    // 初始化集群拓扑
    HcclVmResult InitClusterTopo(const std::string& clusterDir);
    // 解析ranktable文件，并初始化通信域表项
    HcclVmResult ParseRanktableAndInitCommDomain(const std::string& ranktablePath);
    // 初始化通信域表项
    HcclVmResult InitCommunicationDomain(const TopoMeta& topoMeta, bool withRanktable = false);
    // 获取集群状态
    HvmClusterStatus GetClusterStatus() const { return status_; }
    // 设置集群状态
    void SetClusterStatus(HvmClusterStatus status) { status_ = status; }

private:
    // 私有构造、析构
    AscendClusterTopoParser() = default;
    ~AscendClusterTopoParser() = default;

private:
    HcclVmResult InitClusterStaticTopoData();
    HcclVmResult InitDynamicModelData(const TopoMeta& topoMeta);
    HcclVmResult BuildLevelList(
        const Server& server, int srcDevPhyId, const std::set<int>& commDomainLocalIds, uint32_t spIdx, uint32_t srvIdx,
        json& levelList);
    json BuildRankEntry(
        const Server& server, int srcDevPhyId, const std::set<int>& commDomainLocalIds, uint32_t spIdx, uint32_t srvIdx,
        uint32_t rankId);
    uint64_t InitServer(uint32_t superPodId, const Server& server);
    uint64_t InitDevice(uint64_t serverKey, const Device& device);
    uint64_t InitPort(uint64_t deviceKey, const Port& port);
    uint64_t InitCcu(uint64_t deviceKey, uint32_t dieId);
    uint64_t InitCcuResource(uint64_t ccuKey);
    HcclVmResult AddLinkInfo(
        const LinkPortRef* srcPort, const LinkPortRef* dstPort, uint32_t netLayer,
        const std::vector<Protocol>& protocols);
    HcclVmResult CreateRankTableFile(const TopoMeta& topoMeta);
    HcclVmResult ParseRanktable(const std::string& ranktablePath, TopoMeta& topoMeta);

private:
    std::string outputFile_{"cluster_topo_data_output.txt"};
    std::string clusterDir_;
    Network network_;
    std::map<uint32_t, std::map<uint32_t, uint64_t>> serverIdx2Key_;
    HvmClusterStatus status_{HvmClusterStatus::COMM_DOMAIN_UNINIT};
};

#endif // HCCL_VM_TOPO_ASCEND_CLUSTER_PARSER_H
