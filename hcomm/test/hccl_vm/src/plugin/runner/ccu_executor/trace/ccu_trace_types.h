/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_SIM_CCU_TRACE_TYPES_H
#define HCCL_SIM_CCU_TRACE_TYPES_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "sim_common_defs.h"
#include "ccu_resource_common.h"
#include "ccu_simulator_base.h"

namespace CcuTrace {

// ===== 指令分类枚举 =====
enum class CcuInstrCategory : uint8_t {
    LOAD = 0,    // 加载/存储/算术
    TRANS = 1,   // 数据搬运
    CONTROL = 2, // 控制流
    REDUCE = 3,  // 归约运算
    UNKNOWN = 0xFF
};

inline const char* CcuInstrCategoryToString(CcuInstrCategory cat)
{
    switch (cat) {
        case CcuInstrCategory::LOAD:
            return "Load";
        case CcuInstrCategory::TRANS:
            return "Trans";
        case CcuInstrCategory::CONTROL:
            return "Control";
        case CcuInstrCategory::REDUCE:
            return "Reduce";
        default:
            return "Unknown";
    }
}

// ===== 资源记录（快照用） =====

struct CcuXnRecord {
    uint16_t id{0};
    uint64_t value{0};
};

struct CcuGsaRecord {
    uint16_t id{0};
    uint64_t value{0};
};

struct CcuCkeRecord {
    uint16_t id{0};
    uint16_t value{0};
};

struct CcuMsRecord {
    uint16_t id{0};
    std::vector<uint8_t> data; // 4KB 数据
};

struct CcuChannelRecord {
    uint16_t channelId{0};
    int32_t remoteRankId{-1};
    uint16_t remoteDieId{0};
};

// ===== 资源变化（Delta 用） =====

struct CcuXnChange {
    uint16_t id{0};
    uint64_t valueBefore{0};
    uint64_t valueAfter{0};
};

struct CcuGsaChange {
    uint16_t id{0};
    uint64_t valueBefore{0};
    uint64_t valueAfter{0};
};

struct CcuCkeChange {
    uint16_t id{0};
    uint16_t valueBefore{0};
    uint16_t valueAfter{0};
};

struct CcuMsChange {
    uint16_t id{0};
    uint64_t offset{0}; // 变化起始偏移
    uint32_t length{0}; // 变化长度
    std::vector<uint8_t> dataBefore;
    std::vector<uint8_t> dataAfter;
};

// ===== CCU 资源完整快照（仅包含动态资源） =====
// 注：Channel 表是静态配置（InitChannelInfo 初始化后不变），
// 已移至 CcuChannelSpace，与 CcuInstrSpace 并列存放

struct CcuResourceSnapshot {
    std::vector<CcuXnRecord> xnRecords;
    std::vector<CcuGsaRecord> gsaRecords;
    std::vector<CcuCkeRecord> ckeRecords;
    std::vector<CcuMsRecord> msRecords; // 可选，体积大
};

// ===== 资源变化增量（仅记录变化的部分） =====
// 注：Channel 表是静态配置，不会变化，因此不包含 channelChanges

struct CcuResourceDelta {
    std::vector<CcuXnChange> xnChanges;
    std::vector<CcuGsaChange> gsaChanges;
    std::vector<CcuCkeChange> ckeChanges;
    std::vector<CcuMsChange> msChanges;

    bool IsEmpty() const { return xnChanges.empty() && gsaChanges.empty() && ckeChanges.empty() && msChanges.empty(); }
};

// ===== 执行上下文（Loop/Jump 相关） =====

struct CcuExecutionContext {
    bool inLoop{false};
    uint16_t loopRound{0};
    uint16_t loopExtendIndex{0};

    // GSA 地址偏移参数（仅在 inLoop=true 时有效）
    uint32_t gsaOffset{0};
    uint64_t iterStepGSA{0};
    uint32_t loopExtendNum{0};
    uint32_t curLoopCnt{0};
    uint64_t gsaAddrOffset{0};
    uint16_t addrExpandCoef{0};

    // 资源 ID 偏移参数
    uint16_t msOffset{0};
    uint16_t ckeOffset{0};
    uint16_t xnIdOffset{0};
};

// ===== CKE Wait 自旋合并信息 =====

struct CcuWaitInfo {
    bool hadWait{false};
    uint32_t waitRetryCount{0};
    uint16_t waitCKEId{0};
    uint16_t waitCKEMask{0};
    uint16_t ckeValueOnFirstCheck{0};
    uint16_t ckeValueOnPass{0};
    uint64_t waitDurationNs{0};
};

// ===== 执行失败信息 =====

struct CcuErrorInfo {
    bool hasError{false};
    std::string errorMessage;
    std::string failPhase; // "Parser" / "Run" / "Process"
};

// ===== 跨 CCU 资源变更 =====

struct CcuRemoteCkeChange {
    int32_t remoteRankId{0};
    uint16_t remoteDieId{0};
    uint16_t ckeId{0};
    uint16_t valueBefore{0};
    uint16_t valueAfter{0};
};

struct CcuRemoteMsChange {
    int32_t remoteRankId{0};
    uint16_t remoteDieId{0};
    uint16_t msId{0};
    uint64_t offset{0};
    uint32_t length{0};
    std::vector<uint8_t> dataAfter;
};

struct CcuRemoteMemChange {
    int32_t remoteRankId{0};
    uint64_t remoteAddr{0};
    uint64_t length{0};
    std::vector<uint8_t> dataAfter;
};

struct CcuCrossCcuChanges {
    bool hasCrossCcuChange{false};
    std::vector<CcuRemoteCkeChange> remoteCkeChanges;
    std::vector<CcuRemoteMsChange> remoteMsChanges;
    std::vector<CcuRemoteMemChange> remoteMemChanges;
};

// ===== 指令专属 trace 细节（扁平 key-value 结构，仅含运行时动态参数） =====

struct CcuInstrTraceDetail {
    std::string typeName;
    std::map<std::string, std::string> args; // 运行时动态参数（Loop offset 后的值、SQE 参数值等）
    // 注意：不包含 describeText（指令描述存储在指令空间中，通过 instrId 索引）
};

// ===== 指令空间（per CCU，纯静态快照） =====
// 指令空间与 trace 动态参数解耦：
//   - 指令空间存储静态信息：指令 ID + 预计算的 Describe() 输出
//   - trace entry 通过 (rankId, dieId, instrId) 三元组索引指令空间
//   - 同一指令在 Loop 中执行多遍时，指令空间只有 1 条记录，但 trace 产生 N 条 entry

struct CcuInstrSpaceEntry {
    uint16_t instrId{0};
    std::string instrDescribe; // 预计算的 Describe() 输出（静态，不含运行时值）
};

struct CcuInstrSpace {
    int32_t rankId{0};
    uint16_t dieId{0};
    uint32_t instrCnt{0};
    std::vector<CcuInstrSpaceEntry> instructions;
};

// ===== Channel 空间（per CCU，纯静态快照） =====
// Channel 映射表与指令空间一样，是 CCU 独有的静态资源：
//   - InitChannelInfo() 初始化后不再变化
//   - 指令执行时只读取（通过 GetRmtCcu()），从不修改
//   - 与 XN/GSA/CKE/MS 等动态资源分离存放

struct CcuChannelSpace {
    int32_t rankId{0};
    uint16_t dieId{0};
    std::vector<CcuChannelRecord> channels; // channelId → (remoteRankId, remoteDieId)
};

// ===== 单条指令的 trace 条目 =====

struct CcuTraceEntry {
    // 全局定位信息
    uint32_t globalSeqId{0};
    uint32_t execRound{0};

    // CCU 归属 + 指令索引（三元组索引指令空间）
    int32_t rankId{0};
    uint16_t dieId{0};
    uint32_t instrId{0}; // 指令 ID，与 (rankId, dieId) 一起索引指令空间

    // SQE 归属（引用 sqeTaskRegistry 中的 sqeTaskId）
    uint32_t sqeTaskId{0};

    // 指令类别（仅用于统计聚合，1 字节，冗余自指令空间的 instrType）
    CcuInstrCategory category{CcuInstrCategory::UNKNOWN};

    // 执行状态
    CcuExecState execState{EXEC_NORMAL_INSTR};

    // 执行上下文（Loop/Jump 偏移信息 — 运行时动态值）
    CcuExecutionContext context;

    // 资源变化
    CcuResourceDelta resourceDelta;
    CcuCrossCcuChanges crossCcuChanges;

    // CKE Wait 自旋信息
    CcuWaitInfo waitInfo;

    // 执行失败信息
    CcuErrorInfo errorInfo;

    // 指令专属动态细节（运行时参数，同一指令不同 Loop 轮次可能不同）
    CcuInstrTraceDetail detail;
};

// ===== 非 CCU 任务记录 =====

struct CcuTraceNonCcuEntry {
    uint32_t globalSeqId{0};
    uint32_t execRound{0};
    int32_t rankId{0};
    HccLTaskMetaType taskType{HccLTaskMetaType::CCU_GRAPH};
    int32_t execResult{0}; // HcclVmResult 值（HCCL_SIM_SUCCESS = 0）
    std::string description;
};

// ===== CCU 身份注册表 =====

struct CcuIdentity {
    int32_t rankId{0};
    uint16_t dieId{0};
    RunnerCcuVersion ccuVersion{RunnerCcuVersion::CCU_V1};
    CcuResourceSnapshot initialSnapshot;
};

// ===== SQE 任务注册表 =====

struct CcuSqeTask {
    uint32_t sqeTaskId{0};
    int32_t rankId{0};
    uint16_t dieId{0};
    uint8_t missionId{0};
    uint16_t instStartId{0};
    uint16_t instCnt{0};
    uint32_t key{0};
    std::vector<uint64_t> args;
    uint64_t simulatorPtr{0};
    uint32_t firstExecRound{0};
};

// ===== 运行级元数据 =====

struct CcuRunMetadata {
    uint32_t traceFormatVersion{1};
    std::string algorithmName;
    std::string operatorName;
    uint32_t rankSize{0};
    uint32_t diePerRank{2};
    std::string ccuVersion;
    uint64_t runTimestampNs{0};
    uint32_t totalSqeTaskCount{0};
    uint32_t totalExecRounds{0};
};

// ===== 运行级摘要 =====

struct CcuRunSummary {
    uint32_t totalInstrExecuted{0};
    uint32_t totalFailedInstr{0};
    uint32_t totalCkeWaitSpins{0};
    uint32_t totalHoldEvents{0};
    std::map<std::string, uint32_t> instrCountByCategory;
    std::map<std::string, uint32_t> instrCountByCcu;
};

// ===== 全局上下文（由 SqeuentialExecutor 层维护） =====

struct CcuGlobalContext {
    uint32_t globalSeqId{0};
    uint32_t execRound{0};
    uint32_t currentSqeTaskId{0};
};

// ===== Trace Run（顶层容器，包含一次完整运行的所有数据） =====

struct CcuTraceRun {
    CcuRunMetadata runMetadata;
    std::vector<CcuIdentity> ccuRegistry;
    std::vector<CcuSqeTask> sqeTaskRegistry;

    // ===== 静态配置层（per CCU，初始化后不变）=====
    std::vector<CcuInstrSpace> instrSpaces;     // 指令空间（per CCU，纯静态）
    std::vector<CcuChannelSpace> channelSpaces; // Channel 映射表（per CCU，纯静态）

    // ===== 动态执行层（运行时变化）=====
    std::vector<CcuTraceEntry> globalEntries;
    std::vector<CcuTraceNonCcuEntry> nonCcuEntries;
    std::map<std::string, CcuResourceSnapshot> ccuFinalSnapshots; // key: "rankId_dieId"
    CcuRunSummary runSummary;
};

} // namespace CcuTrace

#endif // HCCL_SIM_CCU_TRACE_TYPES_H
