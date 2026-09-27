/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_CTX_H
#define HCCL_CCU_CTX_H

#include "ccu_device_manager.h"
#include "ccu_ctx_signature.h"
#include "ccu_ctx_arg.h"
#include "ccu_task_arg.h"
#include "ccu_res_pack_legacy.h"
#include "ccu_task_param.h"

#include "ccu_transport.h"
#include "ccu_transport_group.h"
#include "ccu_rep.h"
#include "ccu_context_resource.h"
#include "ccu_instr_info.h"

#include "ccu_rep_context.h"

namespace Hccl {
constexpr uint32_t LOCAL_COPY_MS_PER_LOOP = 8;
constexpr uint32_t CCU_MS_LOCAL_COPY_LOOP_COUNT = 8;

class CcuContext : public CcuRep::CcuRepContext {
public:
    explicit CcuContext(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& transportGroup);
    CcuContext() = default;
    ~CcuContext() override;
    HcclResult Init();

    CcuResReq GetResourceRequest();
    CcuRepResource& GetResource();
    CcuSharedResource& GetExportRes();
    CcuSharedResource& GetImportRes();

    void SetResPack(CcuResPack& resPack);
    CcuResPack* GetResPack() const;
    void SetInstrId(uint32_t instrId);
    uint32_t GetInstrId() const;
    uint32_t GetInstrCount();
    void SetCcuInstrInfo(const CcuRep::CcuInstrInfo& instrInfo);

    HcclResult GeneTaskParam(const CcuTaskArg& arg, std::vector<CcuTaskParam>& taskParams);
    // ccu profiling相关接口
    HcclResult GetCcuProfilingInfo(const CcuTaskArg& arg, std::vector<CcuProfilingInfo>& allCcuProfilingInfo);

    std::vector<CcuTransport*> GetCcuTransports() const { return transports; }

protected:
    // 编程接口
    struct GroupOpConfig {
        uint32_t msInterleave;
        uint32_t loopCount;
        uint64_t memSlice;
    };

    struct GroupOpSizeResource {
        std::vector<CcuRep::MaskSignal> maskSignal;
        std::vector<CcuRep::CcuBuffer> ccuBuffer;
        std::vector<CcuRep::Executor> executor;
    };

    struct GroupOpSize {
        CcuRep::Variable addrOffset;
        CcuRep::Variable loopParam;
        CcuRep::Variable parallelParam;
        CcuRep::Variable residual;
    };

    // 使用Transport中的Variable
    CcuRep::Variable CreateVariable(const CcuTransport& transport, uint32_t varIndex) const;
    CcuRep::Variable CreateVariable();
    CcuRep::Variable CreateContinuousVariable();
    CcuRep::Address CreateAddress();
    CcuRep::Memory CreateMemory();
    CcuRep::Memory CreateMemory(const CcuRep::Variable& token);
    CcuRep::Memory GetRmtBuffer(const CcuTransport& transport, uint32_t index);
    CcuRep::MaskSignal CreateMaskSignal();
    CcuRep::CcuBuffer CreateCcuBuffer();
    CcuRep::Executor CreateExecutor();
    std::vector<CcuRep::CcuBuffer> CreateBlockCcuBuffer(uint32_t count);
    std::vector<CcuRep::Executor> CreateBlockExecutor(uint32_t count);
    std::vector<CcuRep::MaskSignal> CreateBlockMaskSignal(uint32_t count);
    GroupOpSize CreateGroupOpSize();

    // 不同Device不同Context间同步操作
    void LocalPost(const CcuRep::MaskSignal& sig, uint32_t mask = 1);
    void LocalWait(const CcuRep::MaskSignal& sig, uint32_t mask = 1);
    void RemotePost(const CcuTransport& transport, uint32_t signalIndex, uint32_t mask = 1, bool single = false);
    void WriteVariableWithSignal(
        const CcuTransport& transport, const CcuRep::Variable& var, uint32_t varIndex, uint32_t signalIndex,
        uint32_t mask = 1);
    void RemoteWait(const CcuTransport& transport, uint32_t signalIndex, uint32_t mask = 1);
    void GroupWait(const CcuTransportGroup& transportGroup, uint32_t signalIndex, uint32_t mask = 1);
    // 同一Device不同Context间同步操作
    void ExportVariable(const CcuRep::Variable& var, const std::string& tag);
    CcuRep::Variable ImportVariable(const std::string& tag);
    void ExportMaskSignal(const CcuRep::MaskSignal& sig, const std::string& tag);
    CcuRep::MaskSignal ImportMaskSignal(const std::string& tag);

    void LocalCtxPost(const CcuRep::MaskSignal& sig, uint32_t mask = 1);
    void LocalCtxPostVar(
        const CcuRep::Variable& srcVar, const CcuRep::Variable& dstVar, const CcuRep::MaskSignal& sig,
        uint32_t mask = 1);
    // 数据操作
    void Write(
        const CcuTransport& transport, const CcuRep::Memory& rem, const CcuRep::Memory& loc,
        const CcuRep::Variable& len, const CcuRep::MaskSignal& locSig, uint32_t mask = 1);
    void Write(
        const CcuTransport& transport, const CcuRep::Memory& rem, const CcuRep::CcuBuffer& loc,
        const CcuRep::Variable& len, const CcuRep::MaskSignal& locSig, uint32_t mask = 1);
    void Read(
        const CcuTransport& transport, const CcuRep::Memory& loc, const CcuRep::Memory& rem,
        const CcuRep::Variable& len, const CcuRep::MaskSignal& locSig, uint32_t mask = 1);
    void Read(
        const CcuTransport& transport, const CcuRep::CcuBuffer& loc, const CcuRep::Memory& rem,
        const CcuRep::Variable& len, const CcuRep::MaskSignal& locSig, uint32_t mask = 1);
    void WriteReduce(
        const CcuTransport& transport, const CcuRep::Memory& rem, const CcuRep::Memory& loc,
        const CcuRep::Variable& len, DataType dataType, ReduceOp opType, const CcuRep::MaskSignal& locSig,
        uint32_t mask = 1);
    void ReadReduce(
        const CcuTransport& transport, const CcuRep::Memory& loc, const CcuRep::Memory& rem,
        const CcuRep::Variable& len, DataType dataType, ReduceOp opType, const CcuRep::MaskSignal& locSig,
        uint32_t mask = 1);
    void LocalCopy(
        const CcuRep::Memory& dst, const CcuRep::Memory& src, const CcuRep::Variable& len,
        const CcuRep::MaskSignal& locSig, uint32_t mask = 1);
    void LocalCopy(
        const CcuRep::CcuBuffer& dst, const CcuRep::Memory& src, const CcuRep::Variable& len,
        const CcuRep::MaskSignal& locSig, uint32_t mask = 1);
    void LocalCopy(
        const CcuRep::Memory& dst, const CcuRep::CcuBuffer& src, const CcuRep::Variable& len,
        const CcuRep::MaskSignal& locSig, uint32_t mask = 1);
    void LocalReduce(
        const CcuRep::Memory& dst, const CcuRep::Memory& src, const CcuRep::Variable& len, DataType dataType,
        ReduceOp opType, const CcuRep::MaskSignal& locSig, uint32_t mask = 1);
    void LocalReduce(
        const std::vector<CcuRep::CcuBuffer>& bufs, uint32_t count, DataType dataType, DataType outputDataType,
        ReduceOp opType, const CcuRep::MaskSignal& locSig, const CcuRep::Variable& len, uint32_t mask = 1);
    // 参数操作
    void Load(const CcuRep::Variable& var);

    // Variable src中存放内存地址，从地址中加载数据到Variable var中
    void LoadVariable(const CcuRep::Variable& src, const CcuRep::Variable& var, uint32_t num = 1);
    void LoadVariable(uint64_t addr, const CcuRep::Variable& var);
    void StoreVariable(const CcuRep::Variable& var, uint64_t addr);
    void LoadVariable(uint64_t addr, const CcuRep::Variable& var, uint32_t num);
    void StoreVariable(const CcuRep::Variable& var, const CcuRep::Variable& src);
    // 控制逻辑
    // 宏定义IF、WHILE
    CcuRep::FuncCall Func(const std::string& label);
    CcuRep::FuncCall Func(const CcuRep::Variable& funcAddr);
    CcuRep::LoopCall Loop(const std::string& label);
    void LoopGroup(
        const std::vector<CcuRep::LoopCall>& loops, const std::vector<CcuRep::Variable>& loopCfg,
        const CcuRep::Variable& paraCfg, const CcuRep::Variable& offsetCfg);
    // 高阶操作
    std::vector<uint64_t> CalGoSize(uint64_t size);
    static std::vector<uint64_t> CalGoSizeStatic(uint64_t size, GroupOpConfig& moCfg);

    void AllocGoResource(uint32_t parallelDim = CcuRep::CCU_MS_DEFAULT_LOOP_COUNT, uint32_t msPerLoop = 1);
    void Load(GroupOpSize moSize);
    void GroupBroadcast(
        const std::vector<CcuTransport*>& transports, std::vector<CcuRep::Memory> dst, CcuRep::Memory src,
        GroupOpSize goSize);
    void GroupReduce(
        const std::vector<CcuTransport*>& transports, CcuRep::Memory dst, std::vector<CcuRep::Memory> src,
        GroupOpSize goSize, DataType dataType, DataType outputDataType, ReduceOp opType);
    void GroupCopy(CcuRep::Memory dst, CcuRep::Memory src, GroupOpSize goSize);
    void GroupBroadcastWithoutMyRank(
        const std::vector<CcuTransport*>& ccuTransports, std::vector<CcuRep::Memory> dst, CcuRep::Memory src,
        GroupOpSize goSize);
    void GroupReduceWithoutMyRank(
        const std::vector<CcuTransport*>& ccuTransports, CcuRep::Memory& dst, std::vector<CcuRep::Memory>& src,
        GroupOpSize& goSize, DataType dataType, DataType outputDataType, ReduceOp opType);
    // 子类实现
    virtual void Algorithm() = 0;
    virtual std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) = 0;

private:
    void CreateMultiOpCopy();
    void CreateMultiOpBroadcast(const std::vector<CcuTransport*>& transports);
    void CreateMultiOpReduce(
        const std::vector<CcuTransport*>& transports, DataType dataType, DataType outputDataType, ReduceOp opType);
    void CreateMultiOpBroadcastWithoutMyRank(const std::vector<CcuTransport*>& ccuTransports);

    void CreateMultiOpReduceWithoutMyRank(
        const std::vector<CcuTransport*>& ccuTransports, DataType dataType, DataType outputDataType, ReduceOp opType);
    template <typename T>
    T CreateResAssist(std::array<std::vector<T>, MAX_CCU_IODIE_NUM>& resRecord);
    template <typename T>
    std::vector<T> CreateBlockResAssist(uint32_t count, std::array<std::vector<T>, MAX_CCU_IODIE_NUM>& resRecord);

    // CCU Profiling
    uint64_t GetArgIndex(
        const std::unordered_map<uint16_t, uint16_t>& varId2VarIdMap,
        const std::unordered_map<uint16_t, uint32_t>& varId2ArgIndexMap, const std::vector<uint64_t>& taskArgs,
        uint16_t varId) const;
    void AddCcuProfiling(GroupOpSize goSize, const std::vector<CcuTransport*>& transportsIn);
    void AddCcuProfiling(
        GroupOpSize goSize, const std::vector<CcuTransport*>& transportsIn, DataType dataType, DataType outputDataType,
        ReduceOp opType);
    void DumpCcuProfilingInfo(const std::vector<CcuProfilingInfo>& ccuProfilingInfo) const;
    // 该友元函数用于在context类外创建Variable并被context内的资源管理器管理
    friend HcclResult CcuRep::CreateVariable(CcuRep::CcuRepContext* context, CcuRep::Variable& variable);

protected:
    std::vector<CcuTransport*> transports;
    const CcuTransportGroup* transportGroup{nullptr};

    GroupOpConfig moConfig{0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF};
    GroupOpSizeResource moRes;

private:
    CcuSharedResource exportRes;
    CcuSharedResource importRes;
    CcuRepResource res;
    CcuResPack* resPack_{nullptr}; // 资源生命周期目前由框架层维护，通信域销毁时销毁该资源

    CcuRep::CcuInstrInfo instrInfo;

    uint32_t loadArgIndex{0};
    // ccu profiling相关缓存
    std::vector<GroupOpSize> groupOpSizeInfo;
};

}; // namespace Hccl

#endif // HCCL_CCU_CTX_H
