/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccn_pingpong.h"

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <array>
#include <atomic>
#include <map>
#include <random>
#include <stdexcept>
#include <thread>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <acl/acl.h>
#include <acl/acl_rt.h>

namespace {
struct LocalPingpongLogConfig {
    bool enabled = false;
    std::string outputDir = "/root/output";
};

std::mutex g_localPingpongLogMutex;
LocalPingpongLogConfig g_localPingpongLogConfig;
std::mutex g_rpingCtxMutex;
std::atomic<u32> g_payloadLen{12};
std::atomic<u32> g_intervalMs{1};
std::atomic<bool> g_running{true};

struct RpingTargetCache {
    std::string srcDevIp;
    std::vector<std::string> targetDevIpList;
    std::vector<int> srcPorts;
    std::vector<std::array<char, 16>> srcIpBuffers;
    std::vector<std::array<char, 16>> dstIpBuffers;
    std::vector<HccnRpingTargetInfo> targetInfo;
};

std::map<u32, RpingTargetCache> g_rpingTargetCache;

static_assert(
    sizeof(HccnRpingTargetInfo::payload) >= HCCN_RPING_PAYLOAD_LEN_MAX,
    "HccnRpingTargetInfo::payload is smaller than HCCN_RPING_PAYLOAD_LEN_MAX");

void cleanup_rping_contexts();

bool mkdir_p(const std::string& path)
{
    if (path.empty()) {
        return false;
    }

    std::string current;
    for (size_t i = 0; i < path.size(); i++) {
        current.push_back(path[i]);
        if ((path[i] != '/' && i + 1 != path.size()) || current == "/") {
            continue;
        }
        if (mkdir(current.c_str(), 0700) != 0) {
            if (errno != EEXIST) {
                return false;
            }
            chmod(current.c_str(), 0700);
        }
    }
    return true;
}

std::string make_local_pingpong_timestamp()
{
    const auto now = std::chrono::system_clock::now();
    const auto now_time = std::chrono::system_clock::to_time_t(now);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;

    std::tm tm_buf{};
    localtime_r(&now_time, &tm_buf);

    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y%m%d_%H%M%S") << "_" << std::setw(3) << std::setfill('0') << ms;
    return oss.str();
}

LocalPingpongLogConfig get_local_pingpong_log_config()
{
    std::lock_guard<std::mutex> lock(g_localPingpongLogMutex);
    return g_localPingpongLogConfig;
}

std::ofstream open_local_pingpong_log(
    const LocalPingpongLogConfig& config, u32 devId, const std::string& srcDevIp,
    const std::vector<std::string>& targetDevIpList, const std::vector<int>& srcPorts, int times)
{
    std::ofstream ofs;
    if (!config.enabled) {
        return ofs;
    }

    const std::string npuDir = config.outputDir + "/npu" + std::to_string(devId);
    if (!mkdir_p(npuDir)) {
        std::cerr << "[hccn_pingpong] cannot create local log dir: " << npuDir << std::endl;
        return ofs;
    }

    const std::string filePath = npuDir + "/" + make_local_pingpong_timestamp() + ".txt";
    ofs.open(filePath);
    if (!ofs.is_open()) {
        std::cerr << "[hccn_pingpong] cannot open local log file: " << filePath << std::endl;
        return ofs;
    }

    ofs << "devId=" << devId << "\n";
    ofs << "srcDevIp=" << srcDevIp << "\n";
    ofs << "times=" << times << "\n";
    ofs << "intervalMs=" << g_intervalMs.load() << "\n";
    ofs << "targetNum=" << targetDevIpList.size() << "\n";
    for (size_t i = 0; i < targetDevIpList.size(); i++) {
        ofs << "target[" << i << "]=" << targetDevIpList[i] << ",srcPort=" << srcPorts[i] << "\n";
    }
    ofs << "---\n";
    return ofs;
}

template <size_t N>
void copy_ip_to_buffer(std::array<char, N>& buffer, const std::string& ip)
{
    buffer.fill('\0');
    const size_t copyLen = std::min(ip.size(), buffer.size() - 1);
    std::copy_n(ip.data(), copyLen, buffer.data());
}

void copy_ip_to_buffer(char* buffer, size_t bufferSize, const std::string& ip)
{
    std::fill(buffer, buffer + bufferSize, '\0');
    const size_t copyLen = std::min(ip.size(), bufferSize - 1);
    std::copy_n(ip.data(), copyLen, buffer);
}

void fill_rping_targets(
    const std::string& srcDevIp, const std::vector<std::string>& targetDevIpList, const std::vector<int>& srcPorts,
    u32 socketPort, std::vector<std::array<char, 16>>& srcIpBuffers, std::vector<std::array<char, 16>>& dstIpBuffers,
    std::vector<HccnRpingTargetInfo>& targetInfo)
{
    srcIpBuffers.resize(targetDevIpList.size());
    dstIpBuffers.resize(targetDevIpList.size());
    targetInfo.resize(targetDevIpList.size());

    const u32 payloadLen = g_payloadLen.load();
    thread_local std::mt19937 randomEngine(std::random_device{}());
    std::uniform_int_distribution<int> randomByte(0, 255);
    for (u32 i = 0; i < static_cast<u32>(targetDevIpList.size()); i++) {
        auto& target = targetInfo[i];
        target = HccnRpingTargetInfo{};
        copy_ip_to_buffer(srcIpBuffers[i], srcDevIp);
        copy_ip_to_buffer(dstIpBuffers[i], targetDevIpList[i]);

        target.srcPort = static_cast<u32>(srcPorts[i]);
        target.addrType = HCCN_RPING_ADDR_TYPE_IP;
        target.sl = _sl;
        target.tc = _tc;
        target.port = socketPort;
        target.payloadLen = payloadLen;
        const u32 payloadCapacity = static_cast<u32>(sizeof(target.payload));
        for (u32 byteIndex = 0; byteIndex < std::min(payloadLen, payloadCapacity); byteIndex++) {
            target.payload[byteIndex] = static_cast<char>(randomByte(randomEngine));
        }
        target.srcIp = srcIpBuffers[i].data();
        target.dstIp = dstIpBuffers[i].data();
    }
}

} // namespace

void set_hccn_pingpong_local_log(bool enabled, const std::string& outputDir)
{
    std::lock_guard<std::mutex> lock(g_localPingpongLogMutex);
    g_localPingpongLogConfig.enabled = enabled;
    g_localPingpongLogConfig.outputDir = outputDir.empty() ? "/root/output" : outputDir;
}

void set_hccn_pingpong_payload_len(u32 payloadLen)
{
    if (payloadLen == 0 || payloadLen > HCCN_RPING_PAYLOAD_LEN_MAX) {
        throw std::invalid_argument(
            "[hccn_pingpong] payload length must be in [1, " + std::to_string(HCCN_RPING_PAYLOAD_LEN_MAX) + "]");
    }
    g_payloadLen.store(payloadLen);
}

u32 get_hccn_pingpong_payload_len() { return g_payloadLen.load(); }

void set_hccn_pingpong_interval_ms(u32 intervalMs)
{
    if (intervalMs == 0) {
        throw std::invalid_argument("[hccn_pingpong] interval_ms must be positive");
    }
    g_intervalMs.store(intervalMs);
}

u32 get_hccn_pingpong_interval_ms() { return g_intervalMs.load(); }

u32 _npuNum = 16;
u32 _socketPort = 13886;
u32 _timeout = 10; // ms
u32 _sl = 4;
u32 _tc = 132;

std::vector<HccnRpingCtx> rpingCtx(_npuNum, nullptr);

namespace {
void cleanup_rping_contexts()
{
    g_running.store(false);
    std::lock_guard<std::mutex> lock(g_rpingCtxMutex);
    for (auto& ctx : rpingCtx) {
        if (ctx != nullptr) {
            HccnRpingDeinit(ctx);
            ctx = nullptr;
        }
    }
    g_rpingTargetCache.clear();
}
} // namespace

void hccn_pingpong_cleanup() { cleanup_rping_contexts(); }

int hccn_pingpong_init(u32 devId, const std::string& devIp, bool keepAlive)
{
    if (devId >= rpingCtx.size()) {
        return -1;
    }

    int init_ret = 0;
    {
        std::lock_guard<std::mutex> lock(g_rpingCtxMutex);
        if (rpingCtx[devId] == nullptr) {
            HccnRpingInitAttr* initAttr = new HccnRpingInitAttr();
            initAttr->mode = HCCN_RPING_MODE_ROCE;
            initAttr->port = _socketPort;
            initAttr->npuNum = _npuNum;
            initAttr->bufferSize = 4096 * 50;
            initAttr->sl = _sl;
            initAttr->tc = _tc;
            initAttr->ipAddr = new char[16];
            copy_ip_to_buffer(initAttr->ipAddr, 16, devIp);

            auto aclRet = aclrtSetDevice(devId);
            if (aclRet != ACL_SUCCESS) {
                delete[] initAttr->ipAddr;
                delete initAttr;
                return -1;
            }

            HccnResult ret = HccnRpingInit(devId, initAttr, &rpingCtx[devId]);
            init_ret = ret == HCCN_SUCCESS ? 0 : -1;

            delete[] initAttr->ipAddr;
            delete initAttr;
        }
    }

    if (init_ret != 0) {
        std::cout << "[hccn_pingpong] init dev=" << devId << ", ip=" << devIp << ", ret=" << init_ret << std::endl;
        return init_ret;
    }

    if (keepAlive) {
        std::cout << "[hccn_pingpong] keepalive start, dev=" << devId << ", ip=" << devIp << std::endl;
        while (g_running.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }

    return 0;
}

bool hccn_pingpong_is_initialized(u32 devId, bool keepAlive)
{
    (void)keepAlive;
    if (devId >= rpingCtx.size()) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_rpingCtxMutex);
    return rpingCtx[devId] != nullptr;
}

int hccn_pingpong_set_targets(
    u32 devId, const std::string& srcDevIp, const std::vector<std::string>& targetDevIpList,
    const std::vector<int>& srcPorts)
{
    std::cout << "[hccn_pingpong] set targets, dev=" << devId << ", srcDevIp=" << srcDevIp
              << ", targetNum=" << targetDevIpList.size() << ", payloadLen=" << get_hccn_pingpong_payload_len()
              << ", intervalMs=" << get_hccn_pingpong_interval_ms() << std::endl;
    if (devId >= rpingCtx.size() || targetDevIpList.empty()) {
        return -1;
    }
    if (targetDevIpList.size() != srcPorts.size()) {
        return -1;
    }

    auto aclRet = aclrtSetDevice(devId);
    if (aclRet != ACL_SUCCESS) {
        std::cout << "[hccn_pingpong] aclrtSetDevice failed, devId=" << devId << ", ret=" << aclRet << std::endl;
        return -1;
    }

    auto localLogConfig = get_local_pingpong_log_config();
    auto localLogFile = open_local_pingpong_log(localLogConfig, devId, srcDevIp, targetDevIpList, srcPorts, 0);

    auto& targetCache = g_rpingTargetCache[devId];
    targetCache.srcDevIp = srcDevIp;
    targetCache.targetDevIpList = targetDevIpList;
    targetCache.srcPorts = srcPorts;
    fill_rping_targets(
        targetCache.srcDevIp, targetCache.targetDevIpList, targetCache.srcPorts, _socketPort, targetCache.srcIpBuffers,
        targetCache.dstIpBuffers, targetCache.targetInfo);

    for (u32 i = 0; i < static_cast<u32>(targetDevIpList.size()); i++) {
        const auto& target = targetCache.targetInfo[i];
        std::cout << "[INFO](hccn_pingpong_set_targets) add targetInfo[" << i << "]: clientDev=" << devId
                  << " srcIp=" << target.srcIp << " dstIp=" << target.dstIp << " srcPort=" << target.srcPort
                  << " targetPort=" << target.port << " sl=" << target.sl << " tc=" << target.tc << std::endl;
        if (localLogFile.is_open()) {
            localLogFile << "add targetInfo[" << i << "]: clientDev=" << devId << ",srcIp=" << target.srcIp
                         << ",dstIp=" << target.dstIp << ",srcPort=" << target.srcPort << ",targetPort=" << target.port
                         << ",sl=" << target.sl << ",tc=" << target.tc << "\n";
        }
    }

    HccnResult ret = HccnRpingAddTarget(
        rpingCtx[devId], static_cast<u32>(targetCache.targetInfo.size()), targetCache.targetInfo.data());
    if (ret != HCCN_SUCCESS) {
        if (localLogFile.is_open()) {
            localLogFile << "HccnRpingAddTarget failed,ret=" << ret << "\n";
        }
        g_rpingTargetCache.erase(devId);
        return -1;
    }
    if (localLogFile.is_open()) {
        localLogFile << "HccnRpingAddTarget success\n";
        localLogFile.flush();
    }
    return 0;
}

std::vector<std::vector<HccnRpingResultInfo>> hccn_pingpong_batch_ping(u32 devId, int times)
{
    std::vector<std::vector<HccnRpingResultInfo>> hccnResults;
    if (devId >= rpingCtx.size() || times <= 0) {
        return hccnResults;
    }
    auto targetCacheIt = g_rpingTargetCache.find(devId);
    if (targetCacheIt == g_rpingTargetCache.end()) {
        return hccnResults;
    }
    auto& targetCache = targetCacheIt->second;
    if (targetCache.targetInfo.empty()) {
        return hccnResults;
    }

    const u32 targetNum = static_cast<u32>(targetCache.targetInfo.size());
    hccnResults.resize(targetNum);

    auto localLogConfig = get_local_pingpong_log_config();
    auto localLogFile = open_local_pingpong_log(
        localLogConfig, devId, targetCache.srcDevIp, targetCache.targetDevIpList, targetCache.srcPorts, times);

    auto aclRet = aclrtSetDevice(devId);
    if (aclRet != ACL_SUCCESS) {
        if (localLogFile.is_open()) {
            localLogFile << "aclrtSetDevice failed,ret=" << aclRet << "\n";
        }
        return hccnResults;
    }

    for (int loop = 0; loop < times; loop++) {
        HccnResult ret = HccnRpingBatchPingStart(rpingCtx[devId], 1, get_hccn_pingpong_interval_ms(), _timeout);
        if (ret != HCCN_SUCCESS) {
            if (localLogFile.is_open()) {
                localLogFile << "loop=" << loop << ",HccnRpingBatchPingStart failed,ret=" << ret << "\n";
            }
            break;
        }

        std::vector<HccnRpingResultInfo> resultInfo(targetNum);
        HccnResult getRet = HCCN_E_AGAIN;
        while (getRet == HCCN_E_AGAIN) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            getRet = HccnRpingGetResult(rpingCtx[devId], targetNum, targetCache.targetInfo.data(), resultInfo.data());
        }

        HccnRpingBatchPingStop(rpingCtx[devId]);

        if (localLogFile.is_open()) {
            localLogFile << "loop=" << loop << ",getRet=" << getRet << "\n";
            for (u32 i = 0; i < targetNum; i++) {
                localLogFile << "target=" << i << ",dstIp=" << targetCache.targetDevIpList[i]
                             << ",srcPort=" << targetCache.targetInfo[i].srcPort << ",txPkt=" << resultInfo[i].txPkt
                             << ",rxPkt=" << resultInfo[i].rxPkt << ",minRTT=" << resultInfo[i].minRTT
                             << ",maxRTT=" << resultInfo[i].maxRTT << ",avgRTT=" << resultInfo[i].avgRTT
                             << ",state=" << resultInfo[i].state << "\n";
            }
            localLogFile.flush();
        }

        if (getRet != HCCN_SUCCESS) {
            continue;
        }
        for (u32 i = 0; i < targetNum; i++) {
            hccnResults[i].push_back(resultInfo[i]);
        }
    }

    return hccnResults;
}

std::vector<std::vector<uint64_t>>
hccn_pingpong_to_rpc_result(const std::vector<std::vector<HccnRpingResultInfo>>& hccnResults)
{
    const int kResultSize = 4;
    const int kP90Lat = 0;
    const int kP99Lat = 1;
    const int kMean = 2;
    const int kPass = 3;
    const uint64_t kInvalidResult = 10000;

    std::vector<std::vector<uint64_t>> results(hccnResults.size(), std::vector<uint64_t>(kResultSize, kInvalidResult));

    for (size_t i = 0; i < hccnResults.size(); i++) {
        std::vector<uint64_t> latencies;

        for (const auto& oneResult : hccnResults[i]) {
            if (oneResult.rxPkt > 0) {
                latencies.emplace_back(oneResult.avgRTT);
            }
        }

        if (latencies.empty()) {
            results[i][kPass] = 0;
            continue;
        }

        std::sort(latencies.begin(), latencies.end());

        uint64_t sum = 0;
        for (auto lat : latencies) {
            sum += lat;
        }

        size_t p90Index = static_cast<size_t>((latencies.size() - 1) * 0.90);
        size_t p99Index = static_cast<size_t>((latencies.size() - 1) * 0.99);

        results[i][kP90Lat] = latencies[p90Index];
        results[i][kP99Lat] = latencies[p99Index];
        results[i][kMean] = sum / latencies.size();
        results[i][kPass] = latencies.size();
    }

    return results;
}
