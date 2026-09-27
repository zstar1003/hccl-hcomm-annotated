/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_ENV_FUNC_H
#define HCCLV2_ENV_FUNC_H

#include <vector>
#include <algorithm>
#include <functional>
#include <sstream>
#include <set>
#include <unordered_map>
#include "exception_util.h"
#include "dma_mode.h"
#include "ip_address.h"
#include "op_type.h"
#include "types.h"

namespace Hccl {

constexpr u32 MAX_LEN_OF_DIGIT_ENV = 10; // 数字环境变量最大长度
constexpr u32 NPU_NET_PROTOCOL_MAX_LEN = 127;

constexpr u32 HCCL_ALGO_LEVEL_0 = 0;   // HCCL 算法层级0
constexpr u32 HCCL_ALGO_LEVEL_1 = 1;   // HCCL 算法层级1
constexpr u32 HCCL_ALGO_LEVEL_2 = 2;   // HCCL 算法层级2
constexpr u32 HCCL_ALGO_LEVEL_3 = 3;   // HCCL 算法层级3
constexpr u32 HCCL_ALGO_LEVEL_NUM = 4; // HCCL 算法层级最多4级

constexpr s32 NOTIFY_MAX_WAIT_TIME = 255 * 68; // 非910A2和910A3场景notify wait最大等待时长，由硬件决定
constexpr s32 NOTIFY_MAX_WAIT_TIME_910A3 = 2147483647; // 910A2和910A3场景notify wait最大等待时长，由软件实现
constexpr s32 HCCL_EXEC_TIME_OUT_S
    = NOTIFY_MAX_WAIT_TIME; // 910A2和910A3场景非HCCL默认的Notify wait超时时间设置为最大超时时间 // 改成与A3相同超时时长
constexpr s32 HCCL_EXEC_TIME_OUT_S_910A3
    = NOTIFY_MAX_WAIT_TIME_910A3; // 910A2和910A3 HCCL默认的Notify wait超时时间设置为最大超时时间
constexpr s32 HCCL_INTEVAL_EXEC_TIME_OUT_S = 68; // notifywait的设置参数必须是68的整数倍

constexpr u32 HCCL_CCU_CONTINUOUS_MS_ID_CONFIG_DIR_NUM = 2; // HCCL CCU MS ID配置层级最多2维

constexpr u32 HCCL_CCU_FLAG_NUM = 2; // HCCL NEW CCU 最大是2

constexpr char HCCL_AUTO_PORT_CONFIG[] = "auto";
constexpr u32 HCCL_SOCKET_PORT_RANGE_AUTO = 0;
constexpr u32 MAX_PORT_NUMBER = 65535;

struct SocketIfName {
    std::vector<std::string> configIfNames{}; // 用户输入的网卡名列表
    bool searchNot{false};                    // 匹配还是不匹配，TRUE：不匹配，FALSE：匹配
    bool searchExact{false}; // 精确匹配或前缀匹配，TRUE：精确匹配，FALSE：前缀匹配
    std::string configIfNameStr{""};
    SocketIfName() = default;
    SocketIfName(const std::vector<std::string>& configIfNames, bool searchNot, bool searchExact)
        : configIfNames(configIfNames),
          searchNot(searchNot),
          searchExact(searchExact) {};
};

struct DfsConfig {
    bool taskExceptionEnable{true};
    bool clusterHeartBeatEnable{true};
    int32_t rankConsistentState{0}; // -1:off 0:first 1:on
    DfsConfig() = default;
    DfsConfig(bool taskException, bool clusterHeartBeatEnable, int32_t consistentState)
        : taskExceptionEnable(taskException),
          clusterHeartBeatEnable(clusterHeartBeatEnable),
          rankConsistentState(consistentState) {};
};

enum class NpuProtoType {
    TCP = 1, // 拉远TCP模式
    RDMA,    // 拉远RDMA模式
    RESERVED // 拉远未进行模式使能
};

using SocketPortRange = struct SocketPortRangeDef {
    u32 min;
    u32 max;
    bool operator==(const SocketPortRangeDef& other) const { return min == other.min && max == other.max; }
};

// HCCL通信算法类型
MAKE_ENUM(
    HcclAlgoType,
    HCCL_ALGO_TYPE_DEFAULT, // 默认算法，配置为此时，使用HCCL内藏算法选择逻辑
    HCCL_ALGO_TYPE_RING, HCCL_ALGO_TYPE_PIPELINE, HCCL_ALGO_TYPE_FULLMESH, HCCL_ALGO_TYPE_HDR, HCCL_ALGO_TYPE_PAIRWISE,
    HCCL_ALGO_TYPE_NHR, HCCL_ALGO_TYPE_NB, HCCL_ALGO_TYPE_NULL, HCCL_ALGO_TYPE_NA, HCCL_ALGO_TYPE_NHR_V1,
    HCCL_ALGO_TYPE_AHC)

const std::set<OpType> OP_TYPE_SET
    = {OpType::ALLREDUCE, OpType::BROADCAST, OpType::ALLGATHER,  OpType::REDUCESCATTER, OpType::SEND,
       OpType::RECV,      OpType::BARRIER,   OpType::ALLTOALL,   OpType::REDUCE,        OpType::GATHER,
       OpType::SCATTER,   OpType::ALLTOALLV, OpType::ALLTOALLVC, OpType::BATCHSENDRECV, OpType::DEBUGCASE};

MAKE_ENUM(OrchestrateWay, PRIM, INS)

// HCCL绕路类型
MAKE_ENUM(
    HcclDetourType,
    HCCL_DETOUR_DISABLE,          // 绕路不使能，默认为此值
    HCCL_DETOUR_ENABLE_2P,        // 2P间绕路
    HCCL_DETOUR_ENABLE_4P,        // 4P间绕路
    HCCL_DETOUR_ENABLE_2P_AND_4P) // 2P和4P间绕路

MAKE_ENUM(
    HcclTopoType, HCCL_TOPO_4P4K, HCCL_TOPO_4P4K_2D, HCCL_TOPO_4P1K, HCCL_TOPO_4P1K_2D, HCCL_TOPO_2P2K, HCCL_TOPO_2P1K,
    HCCL_TOPO_1P1K)

MAKE_ENUM(HcclDebugTestCase, HCCL_INTRA_RANK_CNT_NOTIFY, HCCL_INTRA_RANK_NOTIFY, NONE)

/*------------------- string to type cast functions ---------------------------------------
 *   Several template cast functions are provided.
 *   Register your customized cast functions in the second section.
 *-----------------------------------------------------------------------------------------*/

/*------------- common template cast functions -------------*/
template <class T>
inline T Str2T(const std::string& s)
{
    // 检查数字长度
    if (s.size() > MAX_LEN_OF_DIGIT_ENV) {
        THROW<InvalidParamsException>(
            StringFormat("Invalid env len, len[%zu] should not be bigger than %u.", s.size(), MAX_LEN_OF_DIGIT_ENV));
    }
    // 检查是否为全数字
    if (!std::all_of(s.begin(), s.end(), ::isdigit)) {
        THROW<InvalidParamsException>(
            StringFormat("[Init][EnvVarParam]Invalid env config, [%s] contains non-digit char.", s.c_str()));
    }
    return String2T<T>(s);
}

template <>
inline std::string Str2T<std::string>(const std::string& s)
{
    return s;
}

template <>
inline bool Str2T<bool>(const std::string& s)
{
    bool b = true;
    std::string flag = s;
    std::transform(flag.begin(), flag.end(), flag.begin(), ::toupper);
    if (flag == "FALSE") {
        b = false;
    } else if (flag == "TRUE") {
        b = true;
    } else {
        THROW<InvalidParamsException>(StringFormat("Env config \"%s\" is not valid.", s.c_str()));
    }
    return b;
}

template <>
inline IpAddress Str2T<IpAddress>(const std::string& s)
{
    return IpAddress(s);
}

/*------------------ customized cast functions ------------------*/
extern bool CastBin2Bool(const std::string& s);

extern SocketIfName CastSocketIfName(const std::string& s);

extern std::vector<HcclAlgoType> CastAlgoTypeVec(const std::string& s);

extern std::map<OpType, std::vector<HcclAlgoType>> SetHcclAlgoConfig(const std::string& hcclAlgo);

extern HcclResult
SetSpecificAlgType(std::vector<std::string>& algos, std::map<OpType, std::vector<HcclAlgoType>>& hcclAlgoConfig);

extern HcclResult
SetCommonAlgType(std::vector<std::string>& algos, std::map<OpType, std::vector<HcclAlgoType>>& hcclAlgoConfig);

extern HcclResult SplitHcclAlgoLevel(const std::string& algoConfig, std::vector<std::string>& algos);

extern HcclResult ParserHcclAlgoLevel(const std::string& algoLevel, u32& level, HcclAlgoType& algoType);

extern HcclResult ParseAlgoString(std::string opName, std::string& algoString, std::vector<HcclAlgoType>& algType);

extern HcclResult CheckAlgoConfigValid(std::vector<std::string>& algos, bool& anyCommonConfig, bool& anySpecificConfig);

extern HcclResult SplitHcclOpType(const std::string& algoConfig, std::vector<std::string>& algos);

extern HcclDetourType CastDetourType(const std::string& s);

extern HcclAccelerator CastHcclAccelerator(const std::string& s);

extern s32 CastSocketFamily(const std::string& s);

extern std::string CastCannVersion(const std::string& cannEnv);

extern DfsConfig CastDfsConfig(const std::string& dfsConfigEnv);

extern u32 CastBin2UInt(const std::string& s);

extern void CheckSocketIfName(const SocketIfName& config);

extern std::vector<SocketPortRange> CastSocketPortRange(const std::string& s, const std::string& envName);

/*----------------------- env variable validate functions -----------------------------
 *   Several template cast functions are provided.
 *   Register your customized validate functions in the second section.
 *--------------------------------------------------------------------------------------*/

/*-------------- common template validate functions ------------*/
template <class T>
void CheckRange(const T& value, const T min, const T max, bool closed = true)
{
    if (closed) {
        if (value < min || value > max) {
            THROW<InvalidParamsException>("value[%u] is out of range[%u, %u].", value, min, max);
        }
    }
}
// 为了可读性，用编译期函数封装绑定操作，使用者只需关心 T min max 三个字段
template <class T>
constexpr std::function<void(const T&)> CHK_RANGE_CLOSED(const T min, const T max)
{
    return std::bind(CheckRange<T>, std::placeholders::_1, min, max, true);
}

/*--------------- customized validate functions ---------------*/
extern void CheckExecTimeOut(const u32& timeOut);

extern void CheckFilePath(const string& filePath);

extern void CheckRdmaTimeout(const u32& timeout);

/*----------------------- env variable post process functions --------------------------
 *   Register your customized post process functions here if necessary.
 *--------------------------------------------------------------------------------------*/
extern void SetRealPath(string& filePath);

extern void ProcExecTimeOut(u32& timeOut);

extern void ProcRdmaTimeout(u32& timeout);

extern void CheckRDMATrafficClass(const u32& rdmaTrafficClass);

extern void ConvertUnitQpThreshold(u32& multiQpThreshold);

/*----------------------- multi qp src port config --------------------------*/
struct MultiQpSrcPortConfig {
    std::string configDirPath;
    std::unordered_map<std::string, std::vector<std::uint16_t>> ipPairToPorts;

    bool IsAvailable() const { return !ipPairToPorts.empty(); }

    static constexpr u32 CONFIG_FILE_LINE_MAX = 128 * 1024;
    static constexpr u32 CONFIG_SRC_PORT_NUM_MAX = 32;
    static constexpr u32 CONFIG_SRC_PORT_ID_MAX = 65535;
    static constexpr u32 CONFIG_IP_NUM = 2;
};

struct HostRdmaUdpPortsList {
    std::unordered_map<u32, std::vector<std::uint16_t>> portsByPhyId;

    bool IsAvailable() const { return !portsByPhyId.empty(); }
    size_t GetDeviceConfigCount() const { return portsByPhyId.size(); }

    static constexpr u32 CONFIG_VALUE_LEN_MAX = 32 * 1024;
};

extern bool ParseHostRdmaUdpPorts(const std::string& value, std::vector<std::uint16_t>& ports);

extern HostRdmaUdpPortsList CastHostRdmaUdpPortsList(const std::string& s);

extern std::vector<std::uint16_t> GetHostRdmaUdpPortsByPhyId(const HostRdmaUdpPortsList& list, u32 devicePhyId);

extern u32
GetMultiQpPortsNumByIpPair(const MultiQpSrcPortConfig& config, const IpAddress& srcIp, const IpAddress& dstIp);

extern std::vector<std::uint16_t>
GetMultiQpSrcPortsByIpPair(const MultiQpSrcPortConfig& config, const IpAddress& srcIp, const IpAddress& dstIp);

} // namespace Hccl

#endif // HCCLV2_ENV_FUNC_H
