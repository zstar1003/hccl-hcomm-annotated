/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SNAP_SHOT_PARSER_H
#define SNAP_SHOT_PARSER_H

#include <hccl/hccl_types.h>
#include "hccl_params_pub.h"
#include "topo_info.h"
#include "rank_table_info.h"
#include "ccu_rank_group.h"

namespace Hccl {
constexpr u32 SNAPSHOT_VERSION_SIZE = 128;
constexpr u32 GROUP_NAME_SIZE = 128;

struct SnapShotDynamic {
    vector<std::pair<u32, RankId>> levelRankPairs{};
    vector<std::pair<LinkGroup, u32>> linkGroupPair{};
    u32 submittedOpCnt{0};
    u32 opMode{0};
    OpExecuteConfig opExecuteConfig; // 存储算子粒度和通信域加速模式
    OpExecuteConfig commExecuteConfig;
    bool isLoadOp{false}; // 是否已加载过算子
};

struct SnapShotPub {
    char snapshotVersion[SNAPSHOT_VERSION_SIZE]; // 快照版本 一般不超过 128 Byte
    char cannVersion[SNAPSHOT_VERSION_SIZE];     // cann版本 一般不超过 128 Byte
    char hcclVersion[SNAPSHOT_VERSION_SIZE];     // hccl版本 一般不超过 128 Byte
    uint32_t step;
};

struct SnapShotComm {
    HcclCommConfig config;
    Hccl::CommParams commParams;
    RankTableInfo rankTableInfo;
    TopoInfo topoInfo;
    vector<std::pair<u32, RankId>>
        levelRankPairs; /* 该信息通过算法层提供的接口获取，用于获取建链时使用的linkData数据——GetPaths(u32 level, RankId
                           sRankId, RankId dRankId)-> LinkData(const NetInstance::Path &path) */
    u32 submittedOpCnt;
    u32 opMode{0};
    vector<std::pair<LinkGroup, u32>> linkGroupPair{};
    OpExecuteConfig opExecuteConfig; // 存储算子粒度和通信域加速模式
    OpExecuteConfig commExecuteConfig;
    bool isLoadOp{false}; // 是否已加载过算子
};
struct Snapshot {
    uint32_t size;
    char groupName[GROUP_NAME_SIZE]; // 通信域名称 一般不超过 128 Byte
    SnapShotComm snapShotComm;       // comm save(size, name, tembinary)
};

struct SnapShotSubComm {
    HcclCommConfig config;
    Hccl::CommParams commParams;
    vector<RankId> rankIds; /* 子通信域中rank在全局通信域中的rank id组成的vector */
    vector<std::pair<u32, RankId>> levelRankPairs;
    u32 submittedOpCnt;
    u32 opMode{0};
    vector<std::pair<LinkGroup, u32>> linkGroupPair{};
    OpExecuteConfig opExecuteConfig; // 存储算子粒度和通信域加速模式
    OpExecuteConfig commExecuteConfig;
    bool isLoadOp{false}; // 是否已加载过算子
};
struct SubSnapshot {
    uint32_t size;
    char groupName[GROUP_NAME_SIZE]; // 子通信域名称 一般不超过 128 Byte
    SnapShotSubComm snapShotSubComm; // comm save(size, name, tembinary)
};

struct CcuStatusSnapshot {
    std::vector<std::array<char, GROUP_NAME_SIZE>> useMsCommIds{};
    std::vector<std::array<char, GROUP_NAME_SIZE>> useSchedCommIds{};
};

struct SnapShotBuf {
    SnapShotPub snapShotPub;             /* 公共区域 */
    uint32_t groupNum;                   /* 子通信域数量 */
    Snapshot snapshot;                   /* 全局通信域 */
    vector<SubSnapshot> subSnapshot;     /* 子通信域 */
    CcuStatusSnapshot ccuStatusSnapshot; /* ccu使用情况 */
};

/**
 * @brief 快照数据序列化和反序列化 单例
 * @note
 */
class SnapShotParser {
public:
    SnapShotParser(const SnapShotParser&) = delete;
    SnapShotParser& operator=(const SnapShotParser&) = delete;

    static SnapShotParser& GetInstance();

    // 获取存储的完整流
    BinaryStream& GetSnapShotBuf();

    // 恢复通信域，recover调用，把传入的备份流流反序列化到本地结构
    HcclResult ParseSnapshotToLocalBuff(void* snapshotBuf, uint32_t snapshotBufSize, SnapShotBuf& localBuff);

    // 生成 全局通信域 静态短流
    void SerializeCommonInfo(
        const CommParams& commParams, const HcclCommConfig& config, std::unique_ptr<RankTableInfo> ranktableInfo,
        std::shared_ptr<TopoInfo>& topoInfo, BinaryStream& binStream) const;

    // 生成 子局通信域 静态短流
    void SerializeSubCommInfo(
        const CommParams& commParams, const HcclCommConfig& subConfig, const std::vector<u32>& rankId,
        BinaryStream& binStream) const;
    // 生成 单个通信域 动态短流
    HcclResult SerializeDynamicInfo(
        const std::vector<std::pair<u32, RankId>>& levelRankPairs, u32 submittedOpCnt, BinaryStream& binStream) const;
    // 序列化 公共版本信息
    void SerializeCommVersionInfo(BinaryStream& binStream) const;

    // 计算BinaryStream的CRC值
    HcclResult CalcBufCrc32(BinaryStream& buf, u32& crcValue) const;

    // 快照恢复完成后，需要下发算子才算一个完整的流程
    void SetIsNeedLoadOp(bool status);
    bool GetIsNeedLoadOp() const;

private:
    // 全局通信域反序列化
    HcclResult DeserializeCommInfo(BinaryStream& binaryStream, Snapshot& snapShot);

    // 单个子通信域反序列化
    HcclResult DeserializeSubCommInfo(BinaryStream& stream, SubSnapshot& subSnapShot);
    // 单个通信域动态buf解析
    HcclResult DeSnapShotDynamicBuf(BinaryStream& buf, SnapShotDynamic& dynamicInfo) const;
    // 解析 所有通信域 快照动态buf
    HcclResult DeAllSnapShotDynamicBuf(BinaryStream& buf, SnapShotBuf& localBuff);
    // 解析 所有通信域 快照静态buf
    HcclResult DeAllSnapShotStaticBuf(BinaryStream& buf, SnapShotBuf& localBuff);
    // 解析 ccuStatus
    void DeserializeCcuStatusBuf(BinaryStream& buf, SnapShotBuf& localBuff) const;

    // 全局通信域静态信息的序列化
    void SerializeParamsInfo(const CommParams& commParams, BinaryStream& binStream) const;
    void SerializeCommConfigInfo(const HcclCommConfig& config, BinaryStream& binStream) const;
    void SerializeRankTableInfo(std::unique_ptr<RankTableInfo> ranktableInfo, BinaryStream& binStream) const;
    void SerializeTopoInfo(const std::shared_ptr<TopoInfo>& topoInfo, BinaryStream& binStream) const;

    // 全局通信域静态信息的反序列化
    HcclResult DeserializeParamsInfo(BinaryStream& binaryStream, Hccl::CommParams& commParams) const;
    HcclResult DeserializeCommConfigInfo(BinaryStream& binaryStream, HcclCommConfig& config) const;
    HcclResult DeserializeRankTableInfo(BinaryStream& binaryStream, RankTableInfo& rankTableInfo) const;
    HcclResult DeserializeTopoInfo(BinaryStream& binaryStream, TopoInfo& topoInfo) const;

    // 子通信域静态信息的序列化
    void SerializeSubCommParamsInfo(const CommParams& commParams, BinaryStream& binStream) const;
    void SerializeSubCommConfigInfo(const HcclCommConfig& subConfig, BinaryStream& binStream) const;
    void SerializeRankIds(const std::vector<u32>& rankIds, BinaryStream& binStream) const;

    // 子通信域静态信息的反序列化
    HcclResult DeserializeSubCommParamsInfo(BinaryStream& binaryStream, CommParams& subCommParam) const;
    HcclResult DeserializeSubCommConfigInfo(BinaryStream& binaryStream, HcclCommConfig& subConfig) const;
    HcclResult DeserializeRankIds(BinaryStream& binaryStream, vector<RankId>& rankIds) const;

    //  公共版本信息序列化和反序列化
    HcclResult DeserializeCommVersionInfo(BinaryStream& binaryStream, SnapShotPub& snapshotPub) const;

    SnapShotParser() {};
    ~SnapShotParser() {};

    // 校验BinaryStream的CRC值
    HcclResult CheckBufCrc32(BinaryStream& buf, const u32 otherCrcValue) const;

private:
    BinaryStream snapShotStream_;
    bool isNeedLoadOp{false};
};

} // namespace Hccl

#endif // SNAP_SHOT_PARSER_H
