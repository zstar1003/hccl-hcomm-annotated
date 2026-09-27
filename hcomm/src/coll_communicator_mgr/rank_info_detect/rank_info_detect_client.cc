/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <thread>
#include <cstdlib>
#include <fstream>
#include <limits.h>
#include "rank_info_detect_client.h"
#include "bootstrap_ip.h"
#include "root_handle_v2.h"
#include "env_config/env_config_v2.h"
#include "host_buffer.h"
#include "binary_stream.h"
#include "hccp_peer_manager.h"
#include "orion_adapter_hccp.h"
#include "orion_adapter_rts.h"
#include "host_socket_handle_manager.h"
#include "socket_manager.h"
#include "topo_addr_info.h"
#include "adapter_error_manager_pub.h"
#include "hccl_log_keywords.h"
#include "network_api_exception.h"
#include "phy_topo_builder.h"
#include "preempt_port_manager_v2.h"
#include "rank_table_report_macro.h"

namespace Hccl {
namespace {
    constexpr u32 HOST_CONTROL_PORT_COUNT = 15;
    constexpr u32 HOST_BACKUP_ADDR_NET_LAYER = 3;
    constexpr const char* BACKUP_ADDR_FIELD = "backup_addr";
    constexpr const char* ADDR_FIELD = "addr";
    constexpr const char* ADDR_TYPE_FIELD = "addr_type";
    // 无UB场景下level0的兜底netInstId前缀，仅用于日志辨识实例用途；
    // 下游识别兜底层一律依据pcie_fallback标记字段，不依赖此名称。
    // 后缀拼接本机host ip：同一服务器内rank共享同一level0实例（PCIe fullmesh），跨服务器实例隔离，
    // 与原始level0按server划分（server mac）的语义一致，跨机流量走上层网络（RoCE等）
    constexpr const char* PCIE_FALLBACK_NET_INST_ID = "l0_pcie_fallback";

    bool IsLevel0Exist(const nlohmann::json& localDevInfoJson)
    {
        if (!localDevInfoJson.contains("level_list") || !localDevInfoJson.at("level_list").is_array()) {
            return false;
        }
        for (const auto& levelJson : localDevInfoJson.at("level_list")) {
            if (levelJson.value<u32>("net_layer", HOST_BACKUP_ADDR_NET_LAYER) == 0) {
                return true;
            }
        }
        return false;
    }

    // 无UB场景兜底：探测不到UB时rootinfo中rank无level0，插入一个带本卡vnic地址的兜底level0，
    // 保证rankGraph按level0建图不抛异常；链路由RankGraphBuilder按PCIE兜底合成。
    // rank_addr_list需填入本卡vnic IP：SocketManager的端口map按各层rank_addr_list的addr索引，
    // 缺失会导致设备端口查询返回0，server随机监听、client连0端口，建链失败。
    void InsertPcieFallbackLevel0(nlohmann::json& localDevInfoJson, u32 devPhyId)
    {
        if (IsLevel0Exist(localDevInfoJson)) {
            return;
        }
        const std::string fallbackNetInstId
            = StringFormat("%s_%s", PCIE_FALLBACK_NET_INST_ID, GetBootstrapIp(devPhyId).GetIpStr().c_str());
        HCCL_WARNING(
            "[%s] no level0 in rootinfo (UB unavailable), insert pcie fallback level0, netInstId[%s].", __func__,
            fallbackNetInstId.c_str());
        IpAddress vnicIp;
        bool vnicQueryOk = false;
        try {
            HrtRaSocketGetVnicIpInfos(devPhyId, DeviceIdType::DEVICE_ID_TYPE_PHY_ID, devPhyId, vnicIp);
            vnicQueryOk = true;
        } catch (const HcclException& e) {
            HCCL_ERROR(
                "[%s] get local vnic ip failed, pcie fallback level0 has no rank_addr entry, devPhyId[%u], reason[%s].",
                __func__, devPhyId, e.what());
        }
        nlohmann::json fallbackLevel;
        fallbackLevel["net_layer"] = 0;
        fallbackLevel["net_instance_id"] = fallbackNetInstId;
        fallbackLevel["net_type"] = "TOPO_FILE_DESC";
        fallbackLevel["net_attr"] = "";
        // 兜底层显式标记：下游（graph builder/communicator）凭此字段识别兜底层，
        // 避免依赖netInstId前缀匹配而与用户自定义拓扑文件中的实例名误撞
        fallbackLevel["pcie_fallback"] = true;
        if (vnicQueryOk) {
            // ports与RankGraphBuilder::BuildPcieFallbackLinks中ConnInterface的端口名保持一致（d2h）
            nlohmann::json addrEntry;
            addrEntry["addr_type"] = "IPV4";
            addrEntry["addr"] = vnicIp.GetIpStr();
            addrEntry["plane_id"] = "0";
            addrEntry["ports"] = nlohmann::json::array({std::string("d2h")});
            fallbackLevel["rank_addr_list"] = nlohmann::json::array({addrEntry});
        } else {
            fallbackLevel["rank_addr_list"] = nlohmann::json::array();
        }
        if (!localDevInfoJson.contains("level_list")) {
            localDevInfoJson["level_list"] = nlohmann::json::array();
        }
        localDevInfoJson["level_list"].push_back(fallbackLevel);
    }

    void BuildHostAddrCandidates(const nlohmann::json& addrJson, std::vector<IpAddress>& candidates)
    {
        // 候选顺序固定为主地址在前、备地址按配置顺序在后，选择时取首个探测成功的地址。
        std::string addrType;
        std::string primaryAddr;
        const std::string msgAddrType = "get host addr_type failed";
        TRY_CATCH_THROW_REPORT_ROOTINFO(
            InvalidParamsException, msgAddrType, (addrType = GetJsonProperty(addrJson, ADDR_TYPE_FIELD)), addrJson,
            ADDR_TYPE_FIELD, "non-empty string");
        const std::string msgPrimaryAddr = "get primary host addr failed";
        TRY_CATCH_THROW_REPORT_ROOTINFO(
            InvalidParamsException, msgPrimaryAddr, (primaryAddr = GetJsonProperty(addrJson, ADDR_FIELD)), addrJson,
            ADDR_FIELD, "non-empty string");
        IpAddress primaryIpAddress;
        const std::string msgParsePrimaryAddr = "parse primary host addr failed";
        TRY_CATCH_THROW(InvalidParamsException, msgParsePrimaryAddr,
                        AddressInfo::ParseAddrByType(addrType, primaryAddr, primaryIpAddress););
        candidates.clear();
        candidates.emplace_back(primaryIpAddress);

        std::vector<IpAddress> backupAddrs;
        AddressInfo::ParseBackupAddrs(addrJson.at(BACKUP_ADDR_FIELD), addrType, backupAddrs);
        candidates.insert(candidates.end(), backupAddrs.begin(), backupAddrs.end());
    }

    void CollectLayer3AddrJsons(nlohmann::json& localDevInfoJson, std::vector<nlohmann::json*>& addrJsons)
    {
        // 仅返回 netLayer3 及以上且配置了 backup_addr 的可写地址节点，其他地址直接沿用主 addr。
        addrJsons.clear();
        CHK_PRT_THROW(
            !localDevInfoJson.contains("level_list") || !localDevInfoJson.at("level_list").is_array(),
            HCCL_ERROR("[%s] level_list is missing or is not an array.", __func__), InvalidParamsException,
            "level_list is missing or is not an array");
        for (auto& levelJson : localDevInfoJson.at("level_list")) {
            if (!levelJson.contains("rank_addr_list") || !levelJson["rank_addr_list"].is_array()) {
                continue;
            }
            u32 netLayer = 0;
            const std::string msgNetLayer = "get net_layer failed";
            TRY_CATCH_THROW_REPORT_ROOTINFO(
                InvalidParamsException, msgNetLayer, (netLayer = GetJsonPropertyUInt(levelJson, "net_layer")),
                levelJson, "net_layer", "0 ~ UINT32_MAX");
            if (netLayer < HOST_BACKUP_ADDR_NET_LAYER) {
                continue;
            }
            for (auto& addrJson : levelJson["rank_addr_list"]) {
                if (!addrJson.contains(BACKUP_ADDR_FIELD)) {
                    HCCL_WARNING("[%s] backup_addr is not configured, use primary addr without probing.", __func__);
                    continue;
                }
                addrJsons.push_back(&addrJson);
            }
        }
    }

    std::string QueryTopoFilePathByDevice()
    {
        const size_t bufSize = 1024;
        auto userDevId = HrtGetDevice();
        auto devPhyId = HrtGetDevicePhyIdByUserDevId(userDevId);
        std::vector<char> buffer(bufSize, '\0');
        int result = TopoAddrInfoGetTopoFilePath(devPhyId, buffer.data(), buffer.size());
        CHK_PRT_THROW(
            result != 0, HCCL_ERROR("[%s] Get topo file path failed.", __func__), InvalidParamsException,
            "Get topo file path failed.");
        return std::string(buffer.data());
    }

    void CheckTopoFilePath(const std::string& topoFilePath)
    {
        char resolvedPath[PATH_MAX] = {0};
        CHK_PRT_THROW(
            realpath(topoFilePath.c_str(), resolvedPath) == nullptr,
            HCCL_ERROR("[%s] topo_file_path[%s] is not a valid real path", __func__, topoFilePath.c_str()),
            InvalidParamsException, "topo_file_path error");
    }

    std::string GetRootInfoTopoFilePath()
    {
        std::string filePath = "/etc/hccl_rootinfo.json";
        JsonParser jsonParser{};
        nlohmann::json parseJson{};
        std::string topoFilePath{};
        std::ifstream file(filePath);
        if (file.good()) {
            jsonParser.ParseFileToJson(filePath, parseJson);
            std::string msgRankTopoFile = "error occurs when parser object of propName \"topo_file_path\"";
            TRY_CATCH_THROW_REPORT_ROOTINFO(
                InvalidParamsException, msgRankTopoFile, (topoFilePath = GetJsonProperty(parseJson, "topo_file_path")),
                parseJson, "topo_file_path", "non-empty string");
        } else {
            topoFilePath = QueryTopoFilePathByDevice();
        }

        CheckTopoFilePath(topoFilePath);
        return topoFilePath;
    }
} // namespace

void RankInfoDetectClient::Setup(RankTableInfo& rankTable)
{
    // 1. 构造localRankTable
    RankTableInfo localRankTable{};
    ConstructRankTable(localRankTable);

    // 若启用单卡多进程抢占端口则执行
    SocketManager::ServerInitAll(localRankTable.ranks[0]);
    HostListenPortDetect(localRankTable.ranks[0]);

    // 2. 连接root节点
    Connect();

    // 3. 发送本端agentId和rankSize
    SendAgentIdAndRankSize();

    // 4. 发送给root节点
    SendLocalRankTable(localRankTable);

    // 5. 接收完整rankTable
    RecvRankTable();
    rankTable = rankTable_;
}

void RankInfoDetectClient::Connect()
{
    clientSocket_->Connect();
    CheckStatus();
}

void RankInfoDetectClient::CheckStatus()
{
    HCCL_DEBUG("[RankInfoDetectClient::%s] start.", __func__);

    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut());

    while (true) {
        bool isTimeout = ((std::chrono::steady_clock::now() - startTime) >= timeout);
        if (isTimeout) {
            RPT_INPUT_ERR(
                isTimeout, "EI0015", std::vector<std::string>({"error_reason"}),
                std::vector<std::string>({StringFormat(
                    "Receiving message from the root node timed out "
                    "Timeout was set to %lld seconds. Check whether node rankId[%u] reports an error.",
                    static_cast<long long>(timeout.count()), rankId_)}));
            HCCL_ERROR(
                "[%s][%s] errNo[0x%016llx] topo exchange agent get socket timeout, timeout[%lld s], rankId[%u]",
                LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_DETECT.c_str(),
                HCOM_ERROR_CODE(HcclResult::HCCL_E_TIMEOUT), static_cast<long long>(timeout.count()), rankId_);
            // 建链超时后，sleep 20s，避免上层应用提前退出，确保其他正常 client 能够收到 server 发出的临终遗言
            sleep(WAIT_ERROR_BROADCAST_TIME);
            THROW<TimeoutException>("client get connection timeout");
        }

        if (clientSocket_->GetStatus() == SocketStatus::OK) {
            HCCL_DEBUG("[RankInfoDetectClient::%s] client get socket connection success.", __func__);
            break;
        }
    }

    HCCL_INFO("[RankInfoDetectClient::%s] end, connect ok.", __func__);
}

void RankInfoDetectClient::SendAgentIdAndRankSize()
{
    HCCL_DEBUG("[RankInfoDetectClient::%s] start.", __func__);

    // 发送agentId
    std::string rankID = std::to_string(rankId_);
    std::string agentID = std::string(16 - rankID.length(), '0') + rankID;
    socketAgent_.SendMsg(agentID.c_str(), agentID.size());

    // 发送rankSize
    socketAgent_.SendMsg(&rankSize_, sizeof(rankSize_));

    HCCL_INFO(
        "[RankInfoDetectClient::%s] send agentID[%s] and rankSize_[%u] end.", __func__, agentID.c_str(), rankSize_);
}

void RankInfoDetectClient::SendLocalRankTable(const RankTableInfo& localRankTable)
{
    HCCL_DEBUG("[RankInfoDetectClient::%s] start.", __func__);

    // 消息格式: [ranktable数据(n字节)][step(4字节)]
    BinaryStream binaryStream;
    localRankTable.GetBinStream(true, binaryStream);
    binaryStream << currentStep_;

    // 字节流转换为vector<char>格式
    vector<char> sendMsg;
    binaryStream.Dump(sendMsg);

    // 发送
    socketAgent_.SendMsg(sendMsg.data(), sendMsg.size());

    HCCL_INFO("[RankInfoDetectClient::%s] end, currentStep_[%u].", __func__, currentStep_);
    currentStep_++;
}

void RankInfoDetectClient::ConstructSingleRank(RankTableInfo& localRankTable)
{
    localRankTable.version = "2.0";
    localRankTable.rankCount = 1;
    NewRankInfo rankInfo{};
    rankInfo.rankId = rankId_;
    rankInfo.deviceId = devPhyId_;
    rankInfo.rankLevelInfos.emplace_back(RankLevelInfo{});
    CHK_PRT_CONT(GetLocalTlsStatus(rankInfo.tlsStatus), HCCL_WARNING("[GetLocalTlsStatus] Can not get TlsStatus"));
    CHK_PRT_CONT(
        GetLocalHostDpuTlsStatus(rankInfo.hostDpuTlsStatus),
        HCCL_WARNING("[GetLocalHostDpuTlsStatus] Can not get Host DPU TlsStatus"));
    localRankTable.ranks.emplace_back(rankInfo);

    // 打印
    localRankTable.Dump();
    HCCL_INFO(
        "[RankInfoDetectClient::%s] end, single rank, localRankTable[%s].", __func__,
        localRankTable.Describe().c_str());
}

void CheckRootInfoJson(const nlohmann::json& parseJson)
{
    // check version
    std::string version{};
    std::string msgVersion = "error occurs when parser rootinfo object of propName \"version\"";
    TRY_CATCH_THROW_REPORT_ROOTINFO(
        InvalidParamsException, msgVersion, (version = GetJsonProperty(parseJson, "version")), parseJson, "version",
        "non-empty string");
    if (version != "2.0") {
        RPT_INPUT_ERR(
            true, "EI0016", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({version, "version", "2.0"}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] version[%s] is not \"2.0\"", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), version.c_str());
        THROW<InvalidParamsException>("version error");
    }

    // parser topo_file_path
    std::string topoFilePath{};
    std::string msgRankTopoFile = "error occurs when parser object of propName \"topo_file_path\"";
    TRY_CATCH_THROW_REPORT_ROOTINFO(
        InvalidParamsException, msgRankTopoFile, (topoFilePath = GetJsonProperty(parseJson, "topo_file_path")),
        parseJson, "topo_file_path", "non-empty string");

    // check topo_file_path
    char resolvedPath[PATH_MAX] = {0};
    bool isInvalidPath = (realpath(topoFilePath.c_str(), resolvedPath) == nullptr);
    if (isInvalidPath) {
        RPT_INPUT_ERR(
            true, "EI0016", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({topoFilePath, "topo_file_path", "valid path"}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] topo_file_path[%s] is not a valid real path", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), topoFilePath.c_str());
        THROW<InvalidParamsException>("topo_file_path error");
    }

    // parser rank_count
    u32 rankCount{};
    std::string msgRankcount = "error occurs when parser object of propName \"rank_count\"";
    TRY_CATCH_THROW_REPORT_ROOTINFO(
        InvalidParamsException, msgRankcount, (rankCount = GetJsonPropertyUInt(parseJson, "rank_count")), parseJson,
        "rank_count", "0 ~ UINT32_MAX");

    // parser rank_list
    nlohmann::json rankJsons{};
    std::string msgRanklist = "error occurs when parser object of propName \"rank_list\"";
    TRY_CATCH_THROW_REPORT_ROOTINFO(
        InvalidParamsException, msgRanklist, (GetJsonPropertyList(parseJson, "rank_list", rankJsons)), parseJson,
        "rank_list", "array");

    // check rank_count
    bool isRankCountMismatch = (rankCount != rankJsons.size());
    if (isRankCountMismatch) {
        RPT_INPUT_ERR(
            true, "EI0016", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({std::to_string(rankCount), "rankCount", std::to_string(rankJsons.size())}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] rankCount[%u] != rank_list size[%zu]", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), rankCount,
            rankJsons.size());
        THROW<InvalidParamsException>("rankCount error");
    }
}

void RankInfoDetectClient::ConstructRankTable(RankTableInfo& localRankTable)
{
    HCCL_INFO("[RankInfoDetectClient::%s] start.", __func__);

    // 单P场景处理
    CHK_PRT_RET((rankSize_ == 1), ConstructSingleRank(localRankTable), );

    // 1. 解析文件topoInfo.json
    std::string filePath = "/etc/hccl_rootinfo.json";
    JsonParser jsonParser{};
    nlohmann::json parseJson{};
    std::ifstream file(filePath);
    if (file.good()) {
        jsonParser.ParseFileToJson(filePath, parseJson);
    } else {
        size_t bufSize;
        s32 result = TopoAddrInfoGetSize(devPhyId_, &bufSize); // 获取rankInfo大小，用于提前分配内存
        CHK_PRT_THROW(
            result != 0 || bufSize > MAX_BUFFER_LEN,
            HCCL_ERROR("[RankInfoDetectClient::%s] Get rankinfo size failed.", __func__), InvalidParamsException,
            "Get rankinfo size failed.");
        std::vector<char> buffer(bufSize, '\0');
        result = TopoAddrInfoGet(devPhyId_, buffer.data(), &bufSize); // 获取rankInfo 并更新大小
        CHK_PRT_THROW(
            result != 0, HCCL_ERROR("[RankInfoDetectClient::%s] Get rankinfo failed.", __func__),
            InvalidParamsException, "Get rankinfo size failed.");
        std::string jsonString(buffer.data(), bufSize);
        // 将生成的info信息转换成json文件
        parseJson = nlohmann::json::parse(jsonString);
    }
    CheckRootInfoJson(parseJson);

    // 2. 获取当前devPhyId_对应的devInfo
    nlohmann::json localDevInfoJson{};
    GetLocalDevInfoJson(parseJson, localDevInfoJson);
    // 3. 在反序列化和上报本地 RankTable 前改写 addr，确保后续全局 RankTable 和 RankGraph 使用选中地址
    SelectLocalHostBackupAddr(localDevInfoJson);

    // 无UB兜底：探测不到level0时插入PCIe兜底层，仅在level_list存在（RoCE主备探测已通过）后执行
    InsertPcieFallbackLevel0(localDevInfoJson, devPhyId_);

    // 4. 组rankTable的json格式
    nlohmann::json localRankTableJson{};
    GetLocalRankTableJson(parseJson, localRankTableJson);
    localRankTableJson["rank_list"].push_back(localDevInfoJson); // 添加localDevInfoJson

    // 5. 反序列化获得RankTableInfo
    std::string msgDeserialize = "error occurs when localRankTable Deserialize";
    TRY_CATCH_THROW(InvalidParamsException, msgDeserialize,
                    localRankTable.Deserialize(localRankTableJson, false, RankTableSource::ROOTINFO););
    CHK_PRT_THROW(
        localRankTable.ranks.empty(), HCCL_ERROR("[RankInfoDetectClient::%s] local rank table has no rank.", __func__),
        InvalidParamsException, "local rank table has no rank");
    CHK_PRT_CONT(
        GetLocalTlsStatus(localRankTable.ranks[0].tlsStatus),
        HCCL_WARNING("[GetLocalTlsStatus] Can not get TlsStatus"));
    CHK_PRT_CONT(
        GetLocalHostDpuTlsStatus(localRankTable.ranks[0].hostDpuTlsStatus),
        HCCL_WARNING("[GetLocalHostDpuTlsStatus] Can not get Host DPU TlsStatus"));
    HCCL_INFO("[RankInfoDetectClient::%s] end.", __func__);
}

s32 RankInfoDetectClient::ProbeHostRoceAddr(const IpAddress& hostAddr) const
{
    // hostAddr 来自 netLayer3 的 rank_addr_list，是 RoCE 数据通信地址；它不同于 RootInfoDetect
    // 使用的 rootHandle.ip，后者只负责控制面 socket 建链。
    RaInterface raInterface{devPhyId_, hostAddr};
    RdmaHandle rdmaHandle = nullptr;
    const s32 ret = HrtRaRdmaInit(HrtNetworkMode::PEER, raInterface, rdmaHandle);
    if (ret != 0) {
        return ret;
    }
    if (rdmaHandle != nullptr) {
        // RDMA handle 仅用于可用性探测，不参与后续通信，探测成功后立即释放。
        HrtRaRdmaDeInit(rdmaHandle, HrtNetworkMode::PEER);
    }
    HCCL_INFO(
        "[%s] host addr probe success, devPhyId[%u], rankId[%u], hostAddr[%s].", __func__, devPhyId_, rankId_,
        hostAddr.Describe().c_str());
    return 0;
}

void RankInfoDetectClient::SelectLocalHostBackupAddr(nlohmann::json& localDevInfoJson)
{
    const bool isLevelListInvalid = localDevInfoJson.empty() || !localDevInfoJson.contains("level_list")
                                    || !localDevInfoJson["level_list"].is_array();
    CHK_PRT_THROW(
        isLevelListInvalid,
        HCCL_ERROR(
            "[%s] level_list is missing or is not an array, devPhyId[%u], rankId[%u].", __func__, devPhyId_, rankId_),
        InvalidParamsException, "level_list is missing or is not an array");
    std::vector<nlohmann::json*> addrJsons;
    const std::string msgCollectLayer3Addr = "collect netLayer3 addr failed";
    TRY_CATCH_THROW(InvalidParamsException, msgCollectLayer3Addr, CollectLayer3AddrJsons(localDevInfoJson, addrJsons););
    if (addrJsons.empty()) {
        HCCL_DEBUG("[%s] no netLayer3+ addr with backup_addr needs probing.", __func__);
        return;
    }
    // 主备选择只依赖 RootInfo 的 net_layer 字段，不读取或构建 PhyTopo。
    // 对每个 netLayer3+ 地址独立探测主 addr；主地址不可用时，再按配置顺序逐个尝试 backup_addr。
    for (auto* addrJson : addrJsons) {
        SelectAvailableHostAddr(*addrJson);
    }
    HCCL_INFO(
        "[%s] end, devPhyId[%u], rankId[%u], addrConfigNum[%zu].", __func__, devPhyId_, rankId_, addrJsons.size());
}

void RankInfoDetectClient::SelectAvailableHostAddr(nlohmann::json& addrJson)
{
    std::vector<IpAddress> candidates;
    BuildHostAddrCandidates(addrJson, candidates);
    HCCL_INFO(
        "[%s] devPhyId[%u], rankId[%u], primaryAddr[%s], backupAddrSize[%zu], "
        "candidateSize[%zu].",
        __func__, devPhyId_, rankId_, candidates.front().Describe().c_str(), candidates.size() - 1, candidates.size());

    for (std::size_t idx = 0; idx < candidates.size(); ++idx) {
        const s32 ret = ProbeHostRoceAddr(candidates[idx]);
        if (ret == 0) {
            UpdateSelectedHostAddr(addrJson, candidates, idx);
            return;
        }
        if (ret == HCCP_ELINKDOWN) {
            HCCL_RUN_WARNING(
                "[%s] host addr[%s] is linkdown, try next candidate, devPhyId[%u], rankId[%u], candidateIndex[%zu].",
                __func__, candidates[idx].Describe().c_str(), devPhyId_, rankId_, idx);
            continue;
        }
        HCCL_ERROR(
            "[%s] host addr[%s] probe failed with non-linkdown error, ret[%d], devPhyId[%u], rankId[%u], "
            "candidateIndex[%zu].",
            __func__, candidates[idx].Describe().c_str(), ret, devPhyId_, rankId_, idx);
        THROW<NetworkApiException>(StringFormat(
            "[%s] host addr candidate is unavailable due to non-linkdown error, devPhyId[%u], rankId[%u], "
            "hostAddr[%s], candidateIndex[%zu], ret[%d]",
            __func__, devPhyId_, rankId_, candidates[idx].Describe().c_str(), idx, ret));
    }
    HCCL_RUN_WARNING(
        "[%s] all host addr candidates are linkdown, keep primary address and continue, "
        "devPhyId[%u], rankId[%u], primaryAddr[%s], candidateSize[%zu].",
        __func__, devPhyId_, rankId_, candidates.front().Describe().c_str(), candidates.size());
}

void RankInfoDetectClient::UpdateSelectedHostAddr(
    nlohmann::json& addrJson, const std::vector<IpAddress>& candidates, std::size_t selectedIndex) const
{
    const std::string oldAddr = addrJson[ADDR_FIELD].get<std::string>();
    // 只改写当前有效 addr，保留 backup_addr 原始配置，并随本地 RankTable 一并上报。
    addrJson[ADDR_FIELD] = candidates[selectedIndex].GetIpStr();
    HCCL_RUN_INFO(
        "[%s] select host addr success, devPhyId[%u], rankId[%u], "
        "selectedNicRole[%s], oldHostAddr[%s], selectedHostAddr[%s], candidateIndex[%zu], tryCount[%zu].",
        __func__, devPhyId_, rankId_, selectedIndex == 0 ? "primary" : "backup", oldAddr.c_str(),
        candidates[selectedIndex].Describe().c_str(), selectedIndex, selectedIndex + 1);
}

void RankInfoDetectClient::GetLocalDevInfoJson(const nlohmann::json& parseJson, nlohmann::json& localDevInfoJson)
{
    HCCL_INFO("[RankInfoDetectClient::%s] start.", __func__);

    // rankList字段对应json内容
    nlohmann::json rankJsons;
    std::string msgRanklist = "error occurs when parser object of propName \"rank_list\"";
    TRY_CATCH_THROW_REPORT_ROOTINFO(
        InvalidParamsException, msgRanklist, (GetJsonPropertyList(parseJson, "rank_list", rankJsons)), parseJson,
        "rank_list", "array");

    // 获取localrankJsons, 匹配deviceId字段与当前devPhyId_匹配的内容
    for (auto& rankJson : rankJsons) {
        u32 devId = 0;
        std::string msgDeviceId = "error occurs when parser object of propName \"device_id\"";
        TRY_CATCH_THROW_REPORT_ROOTINFO(
            InvalidParamsException, msgDeviceId, (devId = GetJsonPropertyUInt(rankJson, "device_id")), rankJson,
            "device_id", "0 ~ UINT32_MAX");
        if (devId == devPhyId_) {
            HCCL_INFO("[RankInfoDetectClient::%s] find localDevInfoJson.", __func__);
            localDevInfoJson = rankJson;
            break;
        }
    }

    if (localDevInfoJson.empty()) {
        HCCL_ERROR("[%s] failed, no device_id matches devPhyId_[%u] in rank_list.", __func__, devPhyId_);
    }

    // 添加rankId
    localDevInfoJson["rank_id"] = rankId_;

    HCCL_INFO("[RankInfoDetectClient::%s] end.", __func__);
}

void RankInfoDetectClient::GetLocalRankTableJson(const nlohmann::json& parseJson, nlohmann::json& localRankTableJson)
{
    HCCL_INFO("[RankInfoDetectClient::%s] start.", __func__);

    std::string version;
    std::string msgVersion = "error occurs when parser object of propName \"version\"";
    TRY_CATCH_THROW_REPORT_ROOTINFO(
        InvalidParamsException, msgVersion, (version = GetJsonProperty(parseJson, "version")), parseJson, "version",
        "non-empty string");
    localRankTableJson["version"] = version;

    std::string detour;
    std::string msgDetour = "error occurs when parser object of propName \"detour\"";
    TRY_CATCH_THROW_REPORT_ROOTINFO(
        InvalidParamsException, msgDetour, (detour = GetJsonProperty(parseJson, "detour", false)), parseJson, "detour",
        "non-empty string");
    if (detour == "true") {
        localRankTableJson["detour"] = detour;
    }

    localRankTableJson["rank_count"] = rankSize_;
    HCCL_INFO("[RankInfoDetectClient::%s] end.", __func__);
}

void RankInfoDetectClient::RecvRankTableMsg(vector<char>& rankInfoMsg)
{
    HCCL_INFO("[RankInfoDetectClient::%s] start.", __func__);

    // 接收数据
    u64 revMsgLen = 0;
    std::unique_ptr<HostBuffer> msg = std::make_unique<HostBuffer>(MAX_BUFFER_LEN);
    char* msgAddr = reinterpret_cast<char*>(msg->GetAddr());
    CHK_PRT_THROW(
        !socketAgent_.RecvMsg(msgAddr, revMsgLen),
        HCCL_ERROR("RankInfoDetectClient::%s, recv rankTable error.", __func__), SocketException, "client recv fail");

    // 以vector<char>格式保存
    rankInfoMsg.resize(revMsgLen);
    rankInfoMsg.assign(msgAddr, msgAddr + revMsgLen);

    HCCL_INFO("[RankInfoDetectClient::%s] end, revMsgLen[%llu].", __func__, revMsgLen);
}

// 解析接收到的rank table信息
void RankInfoDetectClient::ParseRankTable(vector<char>& rankInfoMsg)
{
    HCCL_INFO("[RankInfoDetectClient::%s] start.", __func__);

    // 消息格式: [ranktable大小(u32, 4字节)][ranktable数据(n字节)][step(4字节)][failedAgentIdList]
    BinaryStream binStream(rankInfoMsg);

    // 解析localRankInfo
    rankTable_ = RankTableInfo(binStream);
    rankTable_.Dump();

    // 解析step
    u32 receivedStep;
    binStream >> receivedStep;

    // 解析failedAgentIdList
    std::string failedAgentIdList;
    binStream >> failedAgentIdList;
    if (failedAgentIdList.size() > 0) {
        // 建链失败时，打印 root 节点发来的临终遗言
        HCCL_ERROR(
            "[%s][%s] Failed to connect agent, failedRankIdList[%s]", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_DETECT.c_str(), failedAgentIdList.c_str());
    }

    HCCL_INFO("[RankInfoDetectClient::%s] end.", __func__);
}

void RankInfoDetectClient::RecvRankTable()
{
    // 获取rankTable
    vector<char> rankInfoMsg{};
    RecvRankTableMsg(rankInfoMsg);

    // 解析rankTable
    ParseRankTable(rankInfoMsg);

    // 校验
    VerifyRankTable();
}

void RankInfoDetectClient::VerifyRankTable()
{
    HCCL_INFO("[RankInfoDetectClient::%s] start.", __func__);

    // 校验rankCount符合预期
    if (rankTable_.rankCount != rankSize_) {
        THROW<InvalidParamsException>(StringFormat(
            "[RankInfoDetectClient::%s] rank_count[%u] does not match"
            " rankSize_[%u].",
            __func__, rankTable_.rankCount, rankSize_));
    }

    // 校验rankTable内容
    rankTable_.Check();
    // TLS开关一致性校验
    HcclResult ret = VerifyTlsConsistency();
    CHK_PRT_THROW(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[RankInfoDetectClient::%s] tls consistency verify failed, ret[%d]", __func__, ret),
        InvalidParamsException, "tls consistency verify failed");

    HcclResult hostDpuRet = VerifyHostDpuTlsConsistency();
    CHK_PRT_THROW(
        hostDpuRet != HCCL_SUCCESS,
        HCCL_ERROR("[RankInfoDetectClient::%s] hostDpuTls consistency verify failed, ret[%d]", __func__, hostDpuRet),
        InvalidParamsException, "hostDpuTls consistency verify failed");

    HCCL_INFO("[RankInfoDetectClient::%s] end.", __func__);
}

HcclResult RankInfoDetectClient::GetLocalTlsStatus(TlsStatus& tlsStatus) const
{
    struct RaInfo raInfo {};
    raInfo.mode = NetworkMode::NETWORK_OFFLINE;
    raInfo.phyId = devPhyId_;
    return HrtRaGetTlsStatus(&raInfo, tlsStatus);
}

HcclResult RankInfoDetectClient::GetLocalHostDpuTlsStatus(TlsStatus& tlsStatus) const
{
    struct RaInfo raInfo {};
    raInfo.mode = NetworkMode::NETWORK_PEER_ONLINE;
    raInfo.phyId = devPhyId_;
    return HrtRaGetTlsStatus(&raInfo, tlsStatus);
}

void RankInfoDetectClient::GenerateTlsStatusStr(std::string& tlsStatusStr, const std::vector<u32>& tlsStatusRanks) const
{
    tlsStatusStr.clear();
    for (const auto& rank : tlsStatusRanks) {
        tlsStatusStr += std::to_string(rank) + ",";
    }
    if (!tlsStatusStr.empty() && tlsStatusStr.back() == ',') {
        tlsStatusStr.pop_back();
    }
}

void RankInfoDetectClient::ReportTlsConfigurationError(
    const std::string& tlsType, const std::string& tlsInconsistentTlsType, const std::string& tlsEnableRankStr,
    const std::string& tlsDisableRankStr, const std::string& tlsUnknownRankStr) const
{
    std::string expectMessage = "\"All ranks are consistent. Current status: rankList for enabled tls: "
                                + tlsEnableRankStr + "; rankList for disabled tls: " + tlsDisableRankStr;
    if (!tlsUnknownRankStr.empty()) {
        expectMessage += "; rankList for query failure tls: " + tlsUnknownRankStr;
    }
    expectMessage += "\"";
    std::string errormessage = "Value \"" + tlsInconsistentTlsType + "\" for config \"" + tlsType
                               + "\" is invalid. Expected: " + expectMessage;

    RPT_INPUT_ERR(
        true, "EI0016", std::vector<std::string>({"value", "variable", "expect"}),
        std::vector<std::string>({tlsInconsistentTlsType, "\"" + tlsType + "\"", expectMessage}));

    HCCL_ERROR(
        "[%s][%s] errNo[0x%016llx] %s", LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CHECK.c_str(),
        HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), errormessage.c_str());
}

HcclResult RankInfoDetectClient::VerifyTlsConsistency() const
{
    auto getTlsStatus = [](const NewRankInfo& rankInfo) {
        return rankInfo.tlsStatus;
    };
    return VerifyTlsConsistencyByTlsType("tls", getTlsStatus);
}

HcclResult RankInfoDetectClient::VerifyHostDpuTlsConsistency() const
{
    auto getTlsStatus = [](const NewRankInfo& rankInfo) {
        return rankInfo.hostDpuTlsStatus;
    };
    return VerifyTlsConsistencyByTlsType("hostDpuTls", getTlsStatus);
}

HcclResult RankInfoDetectClient::VerifyTlsConsistencyByTlsType(
    const std::string& tlsType, const std::function<TlsStatus(const NewRankInfo&)>& getTlsStatus) const
{
    bool isSupportCheckTlsStatus = true; // 用于标识是否存在不支持查询Tls开关状态的情况
    bool isTlsConsistent = true;         // 用于标识TLS开关状态是否一致
    std::vector<u32> tlsEnableRank;
    std::vector<u32> tlsDisableRank;
    std::vector<u32> tlsUnknownRank;

    for (const auto& rankInfo : rankTable_.ranks) {
        const TlsStatus tlsStatus = getTlsStatus(rankInfo);
        if (tlsStatus == TlsStatus::ENABLE) {
            tlsEnableRank.push_back(rankInfo.rankId);
        } else if (tlsStatus == TlsStatus::DISABLE) {
            tlsDisableRank.push_back(rankInfo.rankId);
        } else {
            isSupportCheckTlsStatus = false;
            tlsUnknownRank.push_back(rankInfo.rankId);
        }
    }

    // 将卡的信息汇总成string
    std::string tlsEnableRankStr;
    std::string tlsDisableRankStr;
    std::string tlsUnknownRankStr;
    GenerateTlsStatusStr(tlsEnableRankStr, tlsEnableRank);
    GenerateTlsStatusStr(tlsDisableRankStr, tlsDisableRank);
    if (!isSupportCheckTlsStatus) {
        GenerateTlsStatusStr(tlsUnknownRankStr, tlsUnknownRank);
    }

    std::string tlsInconsistentTlsType;
    if (!tlsEnableRank.empty() && !tlsDisableRank.empty()) {
        isTlsConsistent = false;
        tlsInconsistentTlsType = (tlsDisableRank.size() <= tlsEnableRank.size()) ? "Disable" : "Enable";
    }

    // 四种不同情况
    if (isTlsConsistent && isSupportCheckTlsStatus) {
        // 1.通信域所有卡都支持查询TLS开关状态，并且TLS开关状态都是一致的。
        HCCL_INFO("[Verify][%sConsistency] All ranks %sStatus are consistent", tlsType.c_str(), tlsType.c_str());
    } else if (!isTlsConsistent && isSupportCheckTlsStatus) {
        // 2.通信域所有卡都支持查询TLS开关状态，但是TLS开关状态存在不一致，报错。
        ReportTlsConfigurationError(
            tlsType, tlsInconsistentTlsType, tlsEnableRankStr, tlsDisableRankStr, tlsUnknownRankStr);
        return HCCL_E_PARA;
    } else if (isTlsConsistent && !isSupportCheckTlsStatus) {
        // 3.通信域内的部分卡不支持查询TLS开关状态，目前能查询到的卡的TLS开关状态是一致的，打印warning提醒
        HCCL_WARNING(
            "[Verify][%sConsistency] Some ranks do not support to check %sStatus, "
            "not support rankId: [%s]",
            tlsType.c_str(), tlsType.c_str(), tlsUnknownRankStr.c_str());
    } else {
        // 4.通信域内的部分卡不支持查询TLS开关状态，但是目前能查询到的卡的TLS开关状态已经不一致，报错
        ReportTlsConfigurationError(
            tlsType, tlsInconsistentTlsType, tlsEnableRankStr, tlsDisableRankStr, tlsUnknownRankStr);
        return HCCL_E_PARA;
    }

    return HCCL_SUCCESS;
}

void RankInfoDetectClient::HostListenPortDetect(NewRankInfo& rankInfo)
{
    std::string topoPath = GetRootInfoTopoFilePath();
    PhyTopoBuilder::GetInstance().Build(topoPath);
    auto devLogicId = HrtGetDevice();
    u32 devPhyId = rankInfo.deviceId;
    for (auto& rankLevelInfo : rankInfo.rankLevelInfos) {
        shared_ptr<Graph<PhyTopo::Node, PhyTopo::Link>> graph
            = PhyTopo::GetInstance()->GetTopoGraph(rankLevelInfo.netLayer);
        if (graph == nullptr) {
            HCCL_DEBUG("[RankInfoDetectClient::%s]Can't find the layout %u Graph!", __func__, rankLevelInfo.netLayer);
            continue;
        }
        std::vector<std::shared_ptr<PhyTopo::Link>> links = graph->GetEdges(rankInfo.localId);
        for (auto& link : links) {
            if (link->GetSourceIFace()->GetPos() != AddrPosition::HOST) {
                continue;
            }
            const std::set<LinkProtocol>& protocols = link->GetLinkProtocols();
            for (auto& protocol : protocols) {
                LinkProtoType protoType = LinkProtocol2LinkProtoType(protocol);
                if (protoType != LinkProtoType::RDMA || rankLevelInfo.rankAddrs.empty()) {
                    continue;
                }
                HCCL_DEBUG("[SocketManager::%s] find the host rdma link %s", __func__, link->Describe().c_str());
                const IpAddress& hostIp = rankLevelInfo.rankAddrs[0].addr;
                const bool isHostSocketPortRangeConfigured
                    = !EnvConfig::GetInstance().GetHostNicConfig().GetHostSocketPortRange().empty();
                if (isHostSocketPortRangeConfigured) {
                    // Primary/backup candidate selection is completed by SelectLocalHostBackupAddr.
                    // This check only validates the selected address for host socket listening.
                    // It does not switch candidates: linkdown skips listening; other errors terminate initialization.
                    const s32 probeRet = ProbeHostRoceAddr(hostIp);
                    if (probeRet == HCCP_ELINKDOWN) {
                        HCCL_RUN_WARNING(
                            "[RankInfoDetectClient::%s] host dpu address[%s] is linkdown, skip host socket listen and "
                            "continue communication domain initialization.",
                            __func__, hostIp.Describe().c_str());
                        return;
                    }
                    CHK_PRT_THROW(
                        probeRet != 0,
                        HCCL_ERROR(
                            "[RankInfoDetectClient::%s] probe host dpu address[%s] failed, ret[%d].", __func__,
                            hostIp.Describe().c_str(), probeRet),
                        NetworkApiException, "probe host dpu address failed");
                }
                uint32_t hostPort = 0;
                SetupHostListenPort(devLogicId, devPhyId, hostIp, hostPort);
                rankInfo.hostPort = hostPort;
                return;
            }
        }
    }
}

void RankInfoDetectClient::SetupHostListenPort(
    u32 devLogicId, u32 devPhyId, const IpAddress& hostIp, uint32_t& hostPort)
{
    std::lock_guard<std::mutex> lock(hostSocketLock_);
    u32 listenPort = HCCL_INVALID_PORT;
    auto portRange = EnvConfig::GetInstance().GetHostNicConfig().GetHostSocketPortRange();
    u32 basePort = EnvConfig::GetInstance().GetHostNicConfig().GetIfBasePort();
    if (portRange.empty() && basePort != HCCL_INVALID_PORT) {
        listenPort = basePort + devPhyId;
        HCCL_INFO("[RankInfoDetectClient::%s] BasePort is configured, listenPort[%u].", __func__, listenPort);
        hostPort = listenPort;
        return;
    }

    if (portRange.empty()) {
        constexpr u32 HOST_CONTROL_BASE_PORT = 60000; // 控制面起始port
        HCCL_INFO(
            "[RankInfoDetectClient::%s] No port configuration, using default port range[%u, %u]", __func__,
            HOST_CONTROL_BASE_PORT, HOST_CONTROL_BASE_PORT + HOST_CONTROL_PORT_COUNT);
        SocketPortRange defaultRange = {HOST_CONTROL_BASE_PORT, HOST_CONTROL_BASE_PORT + HOST_CONTROL_PORT_COUNT};
        portRange.push_back(defaultRange);
    }

    SocketHandle hostSocketHandle = HostSocketHandleManager::GetInstance().Create(devPhyId, hostIp);
    hostSocket_ = std::make_shared<Socket>(
        hostSocketHandle, hostIp, HCCL_INVALID_PORT, hostIp, "hostport_preempt", SocketRole::SERVER,
        NicType::HOST_NIC_TYPE);
    PreemptPortManager::GetInstance(devLogicId).ListenPreempt(hostSocket_, portRange, listenPort);
    HCCL_INFO("[RankInfoDetectClient::%s] preempt hostPort[%u] success.", __func__, listenPort);

    // 登记到进程级 map，供算子下发阶段复用，避免跨阶段端口竞争
    DevNetPortType portType(ConnectProtoType::RDMA);
    PortData portData(static_cast<RankId>(devPhyId), portType, 0, hostIp);
    SocketManager socketMgr(0, devPhyId, static_cast<u32>(devLogicId), "hostport_preempt");
    hostSocketRegistered_ = socketMgr.RegisterHostListenSocket(portData, hostSocket_);

    hostPort = listenPort;
}

void RankInfoDetectClient::SocketTearDown(u32 devPhyId)
{
    std::lock_guard<std::mutex> lock(hostSocketLock_);
    if (hostSocket_ == nullptr) {
        return;
    }
    const IpAddress& hostIp = hostSocket_->GetLocalIp();
    auto devLogicId = HrtGetDevice();
    if (hostSocketRegistered_) {
        // 已登记到 SocketManager::GetServerSocketMap()，所有权已转移给 map，
        // 由算子下发阶段 HostSocketStopListen refcount 归 0 时清理，此处跳过 Release/Destroy
        HCCL_INFO(
            "[RankInfoDetectClient::%s] hostSocket already registered to ServerSocketMap, "
            "skip Release/Destroy, only release local ref.",
            __func__);
    } else if (
        EnvConfig::GetInstance().GetHostNicConfig().GetHostSocketPortRange().size() > 0
        || EnvConfig::GetInstance().GetHostNicConfig().GetIfBasePort() == HCCL_INVALID_PORT) {
        // 若开启抢占监听端口
        PreemptPortManager::GetInstance(devLogicId).Release(hostSocket_);
        HostSocketHandleManager::GetInstance().Destroy(devPhyId, hostIp);
    }
    hostSocket_ = nullptr;
}

void RankInfoDetectClient::TearDown()
{
    HCCL_INFO("[RankInfoDetectClient::%s] start.", __func__);
    SocketTearDown(devPhyId_);

    // close socket
    clientSocket_->Close();

    // deinit handle
    HostSocketHandleManager::GetInstance().Destroy(devPhyId_, clientSocket_->GetLocalIp());

    // deinit ra in detach thread to avoid block main thread
    s32 deviceLogicId = HrtGetDevice();
    std::thread{[deviceLogicId]() {
        EXCEPTION_CATCH(
            HccpPeerManager::GetInstance().DeInit(deviceLogicId),
            HCCL_ERROR("[RankInfoDetectClient::TearDown] DeInit exception"));
    }}.detach();

    HCCL_INFO("[RankInfoDetectClient::%s] end.", __func__);
}

RankInfoDetectClient::~RankInfoDetectClient() { DECTOR_TRY_CATCH("RankInfoDetectClient", TearDown()); }

} // namespace Hccl
