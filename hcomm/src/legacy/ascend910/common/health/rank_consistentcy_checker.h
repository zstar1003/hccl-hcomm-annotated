/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RANK_CONSISTENTCY_CHECKER_H
#define RANK_CONSISTENTCY_CHECKER_H

#include <map>
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <hccl/hccl_types.h>

#include "hccl_common.h"
#include "common.h"
#include "externalinput_pub.h"

namespace hccl {
constexpr u32 DEFAULT_CRC = 0xFFFFFFFF;  // CRC默认值
constexpr u32 MAX_CANN_VERSION_LEN = 50; // CANN版本校验
constexpr u32 MAX_CRC_LEN = 128;         // 最大CRC个数128（CRC最大直径长度：128*sizeof（u32））

using HcclCMDInfo = struct TagHcclCMDInfo {
    HcclCMDType cmdType{HcclCMDType::HCCL_CMD_INVALID};
    char tag[TAG_MAX_LEN + 1] = {0};
    u64 count{0};
    HcclDataType dataType{HCCL_DATA_TYPE_RESERVED};
    HcclReduceOp op{HCCL_REDUCE_RESERVED};
    char group[GROUP_NAME_MAX_LEN + 1] = {0};
    u32 root{0};
    u32 rank{0};
    u32 srTag{0};
    u32 selfRank{0};
    u64 inCclBufferSize{0};
    u64 outCclBufferSize{0};
    u32 aivCoreLimit{0};
    u8 deterministic{0};
};

using HcclCRCInfo = struct TagHcclCRCInfo {
    u32 configFileExist_ = 0;
    u32 crcNum = 0;
    u32 crcArray[MAX_CRC_LEN] = {0};
};

using HcclCheckInfo = struct TagHcclCheckInfo {
    HcclCRCInfo crcInfoGlobal;
    HcclCRCInfo crcInfoOp;
    HcclCMDInfo cmdInfo;
    ProtocolType protocolType = ProtocolType::RESERVED;
    char version[MAX_CANN_VERSION_LEN + 1] = {0};
};

enum class HcclCrcRecordType {
    HCCL_CRC_RECORD_RANKTABLE = 0,
    HCCL_CRC_RECORD_VARING_COUNTS = 1,
    HCCL_CRC_RECORD_VARING_DISPLACEMENTS = 2,
};

const std::map<HcclCrcRecordType, std::string> HCCL_CRC_RECORD_TYPE_STR_MAP{
    {HcclCrcRecordType::HCCL_CRC_RECORD_RANKTABLE, "ranktable"},
    {HcclCrcRecordType::HCCL_CRC_RECORD_VARING_COUNTS, "varing_counts"},
    {HcclCrcRecordType::HCCL_CRC_RECORD_VARING_DISPLACEMENTS, "varing_displacements"},
};

inline std::string GetCRCTypeEnumStr(HcclCrcRecordType crcType)
{
    const auto iter = HCCL_CRC_RECORD_TYPE_STR_MAP.find(crcType);
    if (iter == HCCL_CRC_RECORD_TYPE_STR_MAP.end()) {
        return "Invalid HcclCrcRecordType";
    } else {
        return iter->second;
    }
}

class RankConsistentcyChecker {
public:
    ~RankConsistentcyChecker();

    static RankConsistentcyChecker& GetInstance(s32 deviceLogicId = 0xFF);

    // gather
    HcclResult RecordOpPara(
        HcclCMDType opCMD, const std::string& tag, u64 count, HcclDataType dataType, u32 root, u64 inCclBufferSize,
        u64 outCclBufferSize, const char* group = nullptr, u32 crc = DEFAULT_CRC, u32 aivCoreLimit = 0);
    // all reduce && all gather && broadcast && scatter && reduce && alltoall alltoallv alltoallvc
    HcclResult RecordOpPara(
        HcclCMDType opCMD, const std::string& tag, u64 count, HcclDataType dataType, HcclReduceOp op, u32 root,
        u64 inCclBufferSize, u64 outCclBufferSize, const char* group = nullptr, u32 crc = DEFAULT_CRC,
        u8 deterministic = 0, u32 aivCoreLimit = 0);
    // send && receive
    HcclResult RecordOpPara(
        HcclCMDType opCMD, const std::string& tag, u64 count, HcclDataType dataType, u32 rank, u32 srTag, u32 selfRank,
        u64 inCclBufferSize, u64 outCclBufferSize, const char* group, u32 crc = DEFAULT_CRC);
    // batchsendrecv
    HcclResult RecordOpPara(
        HcclCMDType opCMD, const std::string& tag, u64 inCclBufferSize, u64 outCclBufferSize,
        const char* group = nullptr, u32 crc = DEFAULT_CRC);
    // reduce scatter v && AllGather v
    HcclResult RecordOpPara(
        HcclCMDType opCMD, const std::string& tag, const void* counts, const void* displs, const u32 rankSize,
        HcclDataType dataType, HcclReduceOp op, u64 inCclBufferSize, u64 outCclBufferSize, const char* group = nullptr,
        u32 crc = DEFAULT_CRC, u8 deterministic = 0, u32 aivCoreLimit = 0);

    HcclResult DelOpPara(const std::string& tag);

    HcclResult RecordVerInfo(const std::string& versionInfo);

    u64 GetRankConsistentDataLength();

    void RecordProtocolType(ProtocolType protocolType);

    HcclResult GetCheckFrame(u8* destBuf, u64 maxDestBuf, const std::string& tag);

    HcclResult CheckFrameRecv(const u8* recvBuf, u32 recvBufLen, const std::string& tag);

    void ClearCheckInfo();

    HcclResult CalcStringCrc(const char* str, u32& crc);

    void SetCheckCannVersionSwitch(const bool cannVerCheckSwitch);

private:
    explicit RankConsistentcyChecker();
    // all of that
    HcclResult RecordOpPara(
        HcclCMDType opCMD, const std::string& tag, u64 count, HcclDataType dataType, HcclReduceOp op, u32 root,
        u32 rank, u32 srTag, u32 selfRank, u64 inCclBufferSize, u64 outCclBufferSize, const char* group, u32 crc,
        u8 deterministic = 0, u32 aivCoreLimit = 0);
    // for reduce_scatter_v and all_gatherv
    HcclResult RecordVaringOpPara(const std::string& tag, const void* counts, const void* displs, const u32 rankSize);
    // get CMDinfo by tag
    HcclResult GetOpParaByTag(const std::string& tag, HcclCMDInfo& CMDInfoOutput);
    HcclResult GetCrcByTag(const std::string& tag, HcclCRCInfo& crcInfo);
    HcclResult GenerateCheckFrame(HcclCheckInfo& checkInfo, const std::string& tag);
    void CompareVersion(const HcclCheckInfo& local, const HcclCheckInfo& remote, bool& isDiff);
    bool CompareFrame(HcclCheckInfo& checkInfo, HcclCheckInfo& checkInfoRecv);
    bool CompareCrcInfo(const HcclCMDInfo& hcclCMDInfo, HcclCRCInfo& crcInfo, HcclCRCInfo& crcInfoRecv);
    void ReportCmdInfoCheckFailed(
        const HcclCMDInfo& hcclCMDInfo, const std::string& paraName, const std::string& localPara,
        const std::string& remotePara) const;
    void ReportCmdInfoCheckFailed(
        const HcclCMDInfo& hcclCMDInfo, const std::string& paraName, uint32_t localPara, uint32_t remotePara);
    void ReportCrcCheckFailed(
        const HcclCMDInfo& hcclCMDInfo, HcclCrcRecordType crcType, const uint32_t localCrc,
        const uint32_t remoteCrc) const; // 打印CRC校验失败信息
    void ReportCommonError(
        const HcclCMDInfo& hcclCMDInfo, const std::string& paraName, const std::string& localParaStr,
        const std::string& remoteParaStr, const std::string& errorMsg) const;
    void CompareCmdInfo(HcclCheckInfo& checkInfo, HcclCheckInfo& checkInfoRecv);
    bool CompareSection(const char* pRawData, const char* recvBuf, u32 len);
    HcclResult AddCrc(const u32 crcValue);
    HcclResult ClearCrcInfo(void);
    HcclResult GetCrc(u32 num, u32* crcAddr);
    HcclResult CalcRawDataCrc(const void* ptr, u64 length, u32& crc);

    // 要校验的内容
    std::unordered_map<std::string, HcclCMDInfo> cmdInfoMap_;
    std::unordered_map<std::string, std::map<HcclCrcRecordType, u32>> crcRecords_; // CRC校验码记录
    // cann 版本号
    char cannVersion_[MAX_CANN_VERSION_LEN + 1];
    // cann 版本校验开关
    bool cannVerCheckSwitch_;
    // 算法标志是否已经更新标志，防止发送空数据
    bool cannVerInfoRecordFlag_;
    // CMD是否已经更新标志，防止发送空数据，要校验内容的信息的更新情况（false：未更新）
    std::unordered_map<std::string, bool> infoFlagCmdMap_;
    // config文件是否存在，1表示存在，0表示不存在
    bool configFileExist_;
    ProtocolType protocolType_ = ProtocolType::RESERVED;
    std::vector<u32> crcTable_;
    std::mutex mutex_;
    bool inconsistentCheckFirstDone_ = false; // first模式下是否已完成首次校验
};
} // namespace hccl
#endif // RANK_CONSISTENTCY_CHECKER_H
