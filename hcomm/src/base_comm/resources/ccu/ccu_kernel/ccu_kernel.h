/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_KERNEL_H
#define HCOMM_CCU_KERNEL_H

#include <cstdint>
#include <functional>
#include <array>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "ccu_task_param_v1.h"

#include "ccu_kernel_resource.h"
#include "ccu_instr_info_v1.h"
#include "ccu_rep_context_v1.h"

#include "ccu_funccall_v1.h"
#include "ccu_loopcall_v1.h"

#include "hccl_types.h"
#include "hcomm_primitives.h"
#include "ccu_datatype_v1.h"
#include "ccu_interface_assist_v1.h"

#include "ccu_res_repo.h"
#include "ccu_dev_mgr_imp.h" // CcuVersion

// 暂时引用方便算法开发
#include "ccu_repeat_v1.h"
#include "ccu_condition_v1.h"
#include "ccu_rep_funcblock_v1.h"
#include "ccu_loopblock_v1.h"
#include "ccu_loopcall_v1.h"
#include "ccu_assist_pub.h"

#ifndef CCU_PROFILING // 和hccl仓兼容性使用
#define CCU_PROFILING
#endif
#include "ccu_types.h"

namespace hcomm {

struct GroupInfo {
    uint16_t loopParamId;
    uint16_t parallelParamId;
    uint16_t residualId;
};

struct GroupOpConfig {
    uint32_t msInterleave;
    uint32_t loopCount;
    uint64_t memSlice;
};

// Kernel 查询信息结构体，由 CcuKernel::GetCcuKernelInfo 在锁内填充。
// 后续新增查询字段时只需在此结构体中添加，无需新增接口。
struct CcuKernelInfo {
    uint32_t maxTaskArgsNum{0}; // taskArgs数组所需的元素个数，注册期间由LoadArg的argId推算: max(argId)+1，空则为0
};

class CcuKernel : public CcuRep::CcuRepContext {
public:
    CcuKernel() = default;
    ~CcuKernel() override;

    HcclResult SetupProfilingInfo(const char* kernelFuncName);
    HcclResult ApplyDieFromChannels();
    HcclResult ValidateAndApplyDie(uint32_t targetDieId);

    CcuResReq GetResourceRequest();
    CcuResRepository& GetResRepository();
    CcuRepResource& GetResource();
    CcuSharedResource& GetExportedRes();
    CcuSharedResource& GetImportedRes();

    void SetResRepository(const CcuResRepository& resRepo);
    void SetCascCntBlock(const std::unordered_map<HcommCcuCascCntHandle, CntXnBlock>& cascCntBlocks);
    void SetInstrId(uint32_t instrId);
    HcclResult GetCascCntBlock(HcommCcuCascCntHandle cntHandle, CntXnBlock& cascCntBlock);
    uint32_t GetInstrId() const;
    uint32_t GetInstrCount();
    // 统计会翻译出 CKE 写者微码 (profiling -> setcke / 非 profiling -> clearcke) 的三种 wait 类 rep
    // 个数 (含 block 子 rep), 用于按 CCU_CKE_RAW_LATENCY 为每个此类 rep 预留指令空间.
    // 每个此类 rep 只发射 1 条 setcke / clearcke.
    uint32_t GetRepNeedToAddLatency() const;
    // 统计会翻译出 LoadX/StoreX/ClearX (half-rtt 写后读写者) 的 rep 个数
    // = #(LOAD_ADD_IMM + STORE_ADD_IMM + CASC_CNT_CLEAR) (含 block 子 rep), 用于按 CCU_XN_RAW_LATENCY
    // 为每个此类 rep 预留指令空间. 每个此类 rep 恰翻译出 1 条 LoadX/StoreX/ClearX, 后端优化最多为其
    // 后续读者补 (CCU_XN_RAW_LATENCY - 1) 条 NOP. 仅 CCU_V2 生效.
    uint32_t GetLsxRepReserveCount() const;
    void SetCcuInstrInfo(const CcuRep::CcuInstrInfo& instrInfo);

    CcuResult GeneTaskParams(const uint64_t* taskArgs, uint32_t argsNum, std::vector<CcuTaskParam>& taskParams);

    void SetInsGenerater(CcuRep::CcuInsGeneratorBase* insGeneratorBase);
    void SetCcuVersion(CcuVersion version) { ccuVersion_ = version; }
    // 该友元函数用于在context类外创建Variable并被context内的资源管理器管理
    friend CcuRep::Variable CcuRep::CreateVariable(CcuRep::CcuRepContext* context);

    HcclResult AddProfilingInfo(
        const ChannelHandle* channels, uint32_t channelNum, HcclDataType dataType, HcclDataType outputDataType,
        HcclReduceOp opType, const std::string& opName);

    HcclResult AddCcuProfiling(
        GroupInfo groupInfo, const std::vector<ChannelHandle> channelHandle, HcclDataType dataType,
        HcclDataType outputDataType, HcclReduceOp opType, const std::string& opName);
    HcclResult AddCcuProfiling(
        const ChannelHandle* channels, uint32_t channelNum, HcclDataType dataType, HcclDataType outputDataType,
        HcclReduceOp opType, const std::string& opName);
    HcclResult
    GetCcuProfilingInfo(const uint64_t* taskArgs, uint32_t argSize, std::vector<CcuProfilingInfo>& allCcuProfilingInfo);

    const std::vector<CcuProfilingInfo>& GetAllCcuProfilingInfo() { return allCcuProfilingInfos_; };

    // process const values
    std::unordered_map<uint64_t, CcuRep::Variable>& GetConstValue2VarMap() { return constValue2VarMap; }
    HcclResult Add2ConstValue2VarMap(std::vector<uint64_t>& values);

    const std::unordered_set<ChannelHandle>& GetChannels() { return channels_; }

    const std::unordered_set<uint32_t>& GetDeclaredLocXns() const { return declaredLocXns_; }

    struct PinnedGroupEntry {
        CcuRep::Variable baseVar;
        uint16_t count;
    };
    const std::vector<PinnedGroupEntry>& GetPinnedRegGroups() const { return pinnedRegGroups_; }
    void AddPinnedRegGroup(const CcuRep::Variable& baseVar, uint16_t count);

public:
    // Alloc 相关接口
    CcuResult VariableAlloc(CcuVariableHandle* varHandle);
    CcuResult AddressAlloc(CcuAddressHandle* addrHandle);
    CcuResult EventAlloc(CcuEventHandle* eventHandle);
    CcuResult BufferAlloc(CcuBufferHandle* bufHandle);
    CcuResult
    LocalAddrAlloc(CcuLocalAddrHandle* localAddrHandle, CcuAddressHandle* addrHandle, CcuVariableHandle* tokenHandle);
    CcuResult RemoteAddrAlloc(
        CcuRemoteAddrHandle* remoteAddrHandle, CcuAddressHandle* addrHandle, CcuVariableHandle* tokenHandle);
    CcuResult BlockVariableAlloc(CcuVariableHandle* varHandles, uint32_t count);
    CcuResult BlockEventAlloc(CcuEventHandle* eventHandles, uint32_t count);
    CcuResult BlockBufferAlloc(CcuBufferHandle* bufHandles, uint32_t count);
    CcuResult VariableCreateByChannel(ChannelHandle channel, uint32_t varIndex, CcuVariableHandle* varHandle);
    CcuResult VariableCreateByAcquire(CcuVariableHandle acqHandle, uint32_t index, CcuVariableHandle* varHandle);
    CcuResult EventCreateByAcquire(CcuEventHandle acqHandle, uint32_t index, CcuEventHandle* eventHandle);

    // 参数加载类 相关接口
    CcuResult LoadArg(CcuVariableHandle varHandle, uint32_t argId);
    // 在锁内填充 Kernel 查询信息；当前包含 maxTaskArgsNum。
    CcuResult GetCcuKernelInfo(CcuKernelInfo& info) const;
    CcuResult LoadVar(uint64_t addr, CcuVariableHandle varHandle, uint32_t num);
    CcuResult CcuLoadVarFromVarAddr(CcuVariableHandle addrHandle, CcuVariableHandle varHandle, uint32_t num);
    CcuResult LoadAddImm(
        CcuVariableHandle varHandle, uint16_t srcNum, CcuVariableHandle offsetHandle, uint16_t immAddValue,
        CcuVariableHandle dstHandle);
    CcuResult AddImmStore(
        CcuVariableHandle varHandle, uint16_t dstNum, CcuVariableHandle offsetHandle, uint16_t immAddValue,
        CcuVariableHandle srcHandle);
    CcuResult StoreVar(uint64_t addr, CcuVariableHandle varHandle, uint32_t num);
    CcuResult CcuStoreVarToVarAddr(CcuVariableHandle addrHandle, CcuVariableHandle varHandle, uint32_t num);

    // Event信号同步类 相关接口
    //  mask 由调用方独立传入（与 Event 句柄解耦），不再设独立的 SetEventMask 接口。
    CcuResult EventRecord(CcuEventHandle eventHandle, uint32_t mask);
    CcuResult EventWait(CcuEventHandle eventHandle, uint32_t mask);
    CcuResult NotifyRecord(const ChannelHandle channel, uint32_t remoteNotifyIdx, uint32_t mask);
    CcuResult NotifyWait(const ChannelHandle channel, uint32_t localNotifyIdx, uint32_t mask);
    CcuResult WriteVariableWithNotify(
        const ChannelHandle channel, CcuVariableHandle varHandle, uint32_t remoteVarIdx, uint32_t remoteNotifyIdx,
        uint32_t mask);
    CcuResult CascCntWait(HcommCcuCascCntHandle cntHandle, uint64_t tgtValue);
    CcuResult CascCntClear(HcommCcuCascCntHandle cntHandle, CcuEventHandle eventHandle, uint32_t mask);
    // 本地（同 device 内跨 core）通知同步：用 notifyTag 字符串作为对端标识，
    // 由调用方约定生产者/消费者使用相同的 tag 字符串完成配对。
    // 与 NotifyRecord/Wait（用 ChannelHandle 标识跨 rank 通道）的对偶。
    // 必须 public：C API ccu_primitives_impl.cc 直接调用。
    CcuResult LocalNotifyRecord(const char* notifyTag, const uint32_t mask);
    CcuResult LocalNotifyWait(const char* notifyTag, const uint32_t mask);
    // 本地数据拷贝 相关接口
    CcuResult LocalCopyMemToBuffer(
        CcuBufferHandle dstHandle, CcuLocalAddrHandle srcHandle, CcuVariableHandle lenHandle,
        CcuEventHandle eventHandle, uint32_t mask);
    CcuResult LocalCopyBufferToMem(
        CcuLocalAddrHandle dstHandle, CcuBufferHandle srcHandle, CcuVariableHandle lenHandle,
        CcuEventHandle eventHandle, uint32_t mask);
    CcuResult LocalCopyMemToMem(
        CcuLocalAddrHandle dstHandle, CcuLocalAddrHandle srcHandle, CcuVariableHandle lenHandle,
        CcuEventHandle eventHandle, uint32_t mask);

    // 本地reduce 相关接口
    CcuResult LocalMemReduce(
        CcuLocalAddrHandle dstHandle, CcuLocalAddrHandle srcHandle, CcuVariableHandle lenHandle, HcclDataType dataType,
        HcclReduceOp opType, CcuEventHandle eventHandle, uint32_t mask);
    CcuResult LocalBufferReduce(
        CcuBufferHandle* bufHandles, uint32_t count, HcclDataType dataType, HcclDataType outputDataType,
        HcclReduceOp opType, CcuVariableHandle lenHandle, CcuEventHandle eventHandle, uint32_t mask);

    // 运算重载 相关接口
    CcuResult VariableAssignImm(CcuVariableHandle varHandle, uint64_t immediate);
    CcuResult VariableAssignVar(CcuVariableHandle varHandle, CcuVariableHandle varA);
    CcuResult
    VariableAddVarToVar(CcuVariableHandle varHandle, CcuVariableHandle varAHandle, CcuVariableHandle varBHandle);
    CcuResult
    VariableSubVarToVar(CcuVariableHandle varHandle, CcuVariableHandle varAHandle, CcuVariableHandle varBHandle);
    CcuResult
    VariableMulVarToVar(CcuVariableHandle varHandle, CcuVariableHandle varAHandle, CcuVariableHandle varBHandle);
    CcuResult VariableAddImmToVar(CcuVariableHandle varHandle, CcuVariableHandle varAHandle, uint16_t immediate);
    CcuResult VariableSubImmToVar(CcuVariableHandle varHandle, CcuVariableHandle varAHandle, uint16_t immediate);
    CcuResult VariableMulImmToVar(CcuVariableHandle varHandle, CcuVariableHandle varAHandle, uint16_t immediate);
    CcuResult
    VariableAndVarToVar(CcuVariableHandle varHandle, CcuVariableHandle varAHandle, CcuVariableHandle varBHandle);
    CcuResult
    VariableOrVarToVar(CcuVariableHandle varHandle, CcuVariableHandle varAHandle, CcuVariableHandle varBHandle);
    CcuResult
    VariableXorVarToVar(CcuVariableHandle varHandle, CcuVariableHandle varAHandle, CcuVariableHandle varBHandle);
    CcuResult VariableNotVar(CcuVariableHandle varHandle, CcuVariableHandle varAHandle);
    CcuResult
    VariableShlVarToVar(CcuVariableHandle varHandle, CcuVariableHandle varAHandle, CcuVariableHandle varBHandle);
    CcuResult
    VariableShrVarToVar(CcuVariableHandle varHandle, CcuVariableHandle varAHandle, CcuVariableHandle varBHandle);
    CcuResult AddressAssignImm(CcuAddressHandle addrHandle, uint64_t immediate);
    CcuResult AddressAssignVar(CcuAddressHandle addrHandle, CcuVariableHandle varHandle);
    CcuResult AddressAssignAddr(CcuAddressHandle dstAddrHandle, CcuAddressHandle srcAddrHandle);
    CcuResult
    AddressAddVarToAddr(CcuAddressHandle resAddrHandle, CcuAddressHandle lhsAddrHandle, CcuVariableHandle rhsVarHandle);
    CcuResult
    AddressAddAddrToAddr(CcuAddressHandle resAddrHandle, CcuAddressHandle addrAHandle, CcuAddressHandle addrBHandle);
    CcuResult AddressAddAssignVar(CcuAddressHandle addrHandle, CcuVariableHandle varHandle);
    CcuResult AddressAddAssignAddr(CcuAddressHandle addrHandle, CcuAddressHandle otherHandle);
    CcuResult AddressAddImmToAddr(CcuAddressHandle resAddrHandle, CcuAddressHandle addrAHandle, uint16_t imm);

    // 远端数据传输操作

    CcuResult ReadMemToMem(
        ChannelHandle channel, CcuLocalAddrHandle localHandle, CcuRemoteAddrHandle remoteHandle,
        CcuVariableHandle lenHandle, CcuEventHandle eventHandle, uint32_t mask);
    CcuResult ReadMemToBuffer(
        ChannelHandle channel, CcuBufferHandle localHandle, CcuRemoteAddrHandle remoteHandle,
        CcuVariableHandle lenHandle, CcuEventHandle eventHandle, uint32_t mask);
    CcuResult ReadMemToMemReduce(
        ChannelHandle channel, CcuLocalAddrHandle localHandle, CcuRemoteAddrHandle remoteHandle,
        CcuVariableHandle lenHandle, HcclDataType dataType, HcclReduceOp opType, CcuEventHandle eventHandle,
        uint32_t mask);
    CcuResult WriteMemToMem(
        ChannelHandle channel, CcuRemoteAddrHandle remoteHandle, CcuLocalAddrHandle localHandle,
        CcuVariableHandle lenHandle, CcuEventHandle eventHandle, uint32_t mask);
    CcuResult WriteBufferToMem(
        ChannelHandle channel, CcuRemoteAddrHandle remoteHandle, CcuBufferHandle localHandle,
        CcuVariableHandle lenHandle, CcuEventHandle eventHandle, uint32_t mask);
    CcuResult WriteMemToMemReduce(
        ChannelHandle channel, CcuRemoteAddrHandle remoteHandle, CcuLocalAddrHandle localHandle,
        CcuVariableHandle lenHandle, HcclDataType dataType, HcclReduceOp opType, CcuEventHandle eventHandle,
        uint32_t mask);
    CcuResult WriteVarAtomicAdd(
        CcuVariableHandle channelIdHandle, CcuRemoteAddrHandle varAddrHandle, CcuVariableHandle addValueHandle,
        CcuEventHandle eventHandle, uint32_t mask);
    CcuResult WriteWithCascCntInc(
        CcuVariableHandle channelIdHandle, CcuRemoteAddrHandle remoteHandle, CcuLocalAddrHandle localHandle,
        CcuVariableHandle lenHandle, CcuRemoteAddrHandle inCntAddr);

    CcuResult IfBegin(CcuVariableHandle varHandle, uint64_t immediate, CcuConditionType condType, const char* label);
    CcuResult
    IfBeginVar(CcuVariableHandle lhsHandle, CcuVariableHandle rhsHandle, CcuConditionType condType, const char* label);
    CcuResult IfElse(const char* label);
    CcuResult IfEnd(const char* label);

    void IfLabelStackPush(const char* label);
    void IfLabelStackMarkBodyDone();
    const char* IfLabelStackPopForElse();
    bool IfLabelStackTopIsClosable();
    const char* IfLabelStackPop();

    void DoWhileLabelStackPush(const char* label);
    const char* DoWhileLabelStackPopForWhile();

    void FlushClosablePendingIfs();
    void Append(std::shared_ptr<CcuRep::CcuRepBase> rep) override;
    CcuResult WhileBegin(CcuVariableHandle varHandle, uint64_t immediate, CcuConditionType condType, const char* label);
    CcuResult WhileBeginVar(
        CcuVariableHandle lhsHandle, CcuVariableHandle rhsHandle, CcuConditionType condType, const char* label);
    CcuResult WhileEnd(const char* label);
    CcuResult DoWhileBegin(const char* label);
    CcuResult DoWhileEnd(CcuVariableHandle varHandle, uint64_t immediate, CcuConditionType condType, const char* label);
    CcuResult DoWhileEndVar(
        CcuVariableHandle lhsHandle, CcuVariableHandle rhsHandle, CcuConditionType condType, const char* label);

    CcuResult LoopCreate(CcuLoop* loop);
    CcuResult LoopBodyEnter(CcuLoop loop);
    CcuResult LoopBodyExit(CcuLoop loop);
    // 建组/加环统一以版本化 cfg 为准;旧 config 的转换在 C ABI 适配层完成。
    CcuResult LoopGroupCreate(CcuLoopGroup* group, uint32_t maxLoopNum, const CcuLoopGroupCfg* cfg);
    CcuResult LoopGroupCreateFromVar(
        CcuLoopGroup* group, uint32_t maxLoopNum, CcuVariableHandle parallelVarHandle,
        CcuVariableHandle offsetVarHandle);
    CcuResult LoopGroupCreateFromVarV2(
        CcuLoopGroup* group, uint32_t maxLoopNum, CcuVariableHandle parallelVarV2Handle,
        CcuVariableHandle offsetVarV2Handle, CcuVariableHandle varOffsetVarHandle);
    CcuResult LoopGroupAddLoop(CcuLoopGroup group, CcuLoop loop, const CcuLoopCfg* cfg);
    CcuResult LoopGroupAddLoopFromVar(CcuLoopGroup group, CcuLoop loop, CcuVariableHandle loopParamVarHandle);
    CcuResult LoopGroupAddLoopFromVarV2(
        CcuLoopGroup group, CcuLoop loop, CcuVariableHandle iterNumVarHandle, CcuVariableHandle addrOffsetVarHandle,
        CcuVariableHandle ctxIdVarHandle);

    CcuResult FuncBlockLookup(const void* funcPtr, uint64_t* outHandle);
    CcuResult FuncBlockBegin(const void* funcPtr, uint64_t* outHandle);
    CcuResult FuncBlockEnd(uint64_t handle);
    CcuResult FuncDefineInArg(uint64_t handle, CcuVariableHandle formal);
    CcuResult FuncCall(uint64_t handle, const CcuVariableHandle* inArgs, uint32_t numIn);

private:
    CcuResult GetVariableByHandle(CcuVariableHandle varHandle, CcuRep::Variable** variable);
    CcuResult GetEventByHandle(CcuEventHandle eventHandle, CcuRep::CompletedEvent** event);
    CcuResult LatchBodyError(CcuResult err);
    CcuResult EnsureLoopEnginePool(uint32_t maxLoopNum);

    CcuResult ValidateTaskArgs(const uint64_t* taskArgs, uint32_t argsNum) const;
    void FillTaskParam(
        CcuTaskParam& param, uint32_t index, uint32_t seqNum, const uint64_t* taskArgs, uint32_t argsNum) const;

    CcuResult ResolveBufRemoteLenEvent(
        CcuBufferHandle bufHandle, CcuRemoteAddrHandle remoteHandle, CcuVariableHandle lenHandle,
        CcuEventHandle eventHandle, CcuRep::CcuBuf** buf, CcuRep::RemoteAddr** remote, CcuRep::Variable** len,
        CcuRep::CompletedEvent** event);

    CcuResult ResolveLocalRemoteLenEvent(
        CcuLocalAddrHandle localHandle, CcuRemoteAddrHandle remoteHandle, CcuVariableHandle lenHandle,
        CcuEventHandle eventHandle, CcuRep::LocalAddr** local, CcuRep::RemoteAddr** remote, CcuRep::Variable** len,
        CcuRep::CompletedEvent** event);

    CcuResult ResolveRemoteLocalLenEvent(
        CcuRemoteAddrHandle remoteHandle, CcuLocalAddrHandle localHandle, CcuVariableHandle lenHandle,
        CcuEventHandle eventHandle, CcuRep::RemoteAddr** remote, CcuRep::LocalAddr** local, CcuRep::Variable** len,
        CcuRep::CompletedEvent** event);

    // 校验从 varHandle 起的 num 个 Variable 句柄对应的内部变量 Id 连续递增，
    // 用于 LoadVar/StoreVar 等接口对“连续变量块”的前置校验。
    CcuResult CheckContinuousVariables(
        CcuVariableHandle varHandle, uint32_t num, const CcuRep::Variable& baseVar, const char* tag);

    // GetCcuProfilingInfo 的子步骤：处理 sqe & waitcke 类型的 profiling 信息，结果直接 push 到 allCcuProfilingInfos_
    // 中。
    HcclResult CollectSqeAndWaitCkeProfilingInfo();
    // GetCcuProfilingInfo 的子步骤：根据 LoopGroup 的 profiling 缓存构建
    // varId -> argIndex 与 varId -> varId 两个查找表，供 LoopGroup 段查询入参使用。
    HcclResult BuildLoopGroupVarIdMaps(
        std::unordered_map<uint16_t, uint32_t>& varId2ArgIndexMap,
        std::unordered_map<uint16_t, uint16_t>& varId2VarIdMap);
    // GetCcuProfilingInfo 的子步骤：处理 LoopGroup 的 profiling 信息，结果 push 到 allCcuProfilingInfos_ 中。
    HcclResult CollectLoopGroupProfilingInfo(
        const uint64_t* taskArgs, uint32_t argSize, const std::unordered_map<uint16_t, uint32_t>& varId2ArgIndexMap,
        const std::unordered_map<uint16_t, uint16_t>& varId2VarIdMap);
    // CollectLoopGroupProfilingInfo 的单次循环体：处理下标 i 对应的 loopParam 与 parallelParam 分支。
    HcclResult CollectSingleLoopGroupProfiling(
        uint32_t i, size_t repSize, const uint64_t* taskArgs, uint32_t argSize,
        const std::unordered_map<uint16_t, uint32_t>& varId2ArgIndexMap,
        const std::unordered_map<uint16_t, uint16_t>& varId2VarIdMap);

    CcuRep::Variable CreateJumpTargetVar();

    struct IfLabelEntry {
        const char* label{nullptr};
        bool bodyDone{false};
    };
    struct DoWhileLabelEntry {
        const char* label{nullptr};
        std::shared_ptr<CcuRep::CcuRepBlock> snapshotBlock{nullptr};
        size_t snapshotRepCount{0};
    };
    std::vector<IfLabelEntry> iflabelStack_;
    std::vector<DoWhileLabelEntry> doWhileLabelStack_;
    bool isFlushing_ = false;

    struct PendingIfContext {
        std::shared_ptr<CcuRep::CcuRepJumpLabel> elseLabel;
        std::shared_ptr<CcuRep::CcuRepJumpLabel> endLabel;
        bool hasElse{false};
    };

    struct PendingWhileContext {
        std::shared_ptr<CcuRep::CcuRepJumpLabel> beginLabel;
        std::shared_ptr<CcuRep::CcuRepJumpLabel> endLabel;
        CcuVariableHandle varHandle;
        uint64_t immediate;
        CcuConditionType condType;
    };

    struct PendingDoWhileContext {
        std::shared_ptr<CcuRep::CcuRepJumpLabel> beginLabel;
    };

    std::unordered_map<CcuVariableHandle, CcuRep::Variable> ccuVarMap_{};

    std::unordered_map<std::string, PendingIfContext> pendingIfCtx_{};
    std::unordered_map<std::string, PendingWhileContext> pendingWhileCtx_{};
    std::unordered_map<std::string, PendingDoWhileContext> pendingDoWhileCtx_{};

    std::unordered_map<CcuEventHandle, CcuRep::CompletedEvent> ccuEventMap_{};

    CcuResult GetBufferByHandle(CcuBufferHandle bufferHandle, CcuRep::CcuBuf** buffer);
    std::unordered_map<CcuBufferHandle, CcuRep::CcuBuf> ccuBufferMap_{};

    CcuResult GetAddressByHandle(CcuAddressHandle addrHandle, CcuRep::Address** address);
    std::unordered_map<CcuAddressHandle, CcuRep::Address> ccuAddrMap_{};

    CcuResult GetLocalAddrByHandle(CcuLocalAddrHandle handle, CcuRep::LocalAddr** localAddr);
    std::unordered_map<CcuLocalAddrHandle, CcuRep::LocalAddr> ccuLocalAddrMap_{};

    CcuResult GetRemoteAddrByHandle(CcuRemoteAddrHandle handle, CcuRep::RemoteAddr** remoteAddr);
    std::unordered_map<CcuRemoteAddrHandle, CcuRep::RemoteAddr> ccuRemoteAddrMap_{};

    std::unordered_set<uint32_t> loadArgUsedSet_{};

protected:
    // 使用channel中的Variable
    HcclResult CreateVariable(const ChannelHandle channel, uint32_t varIndex, CcuRep::Variable* var);
    CcuRep::Variable CreateVariable();
    CcuRep::Variable CreateExpectVar();
    CcuRep::Variable CreateContinuousVariable();
    CcuRep::LocalAddr CreateLocalAddr();
    CcuRep::RemoteAddr CreateRemoteAddr();
    CcuRep::RemoteAddr GetRemoteAddr(const ChannelHandle channel, const uint32_t index);
    CcuRep::LocalNotify CreateLocalNotify();
    CcuRep::CompletedEvent CreateCompletedEvent();
    CcuRep::CcuBuf CreateCcuBuf();
    CcuRep::Executor CreateExecutor();

    HcclResult CreateBlockCcuBuf(const uint32_t count, CcuRep::CcuBuf* ccuBufs);
    HcclResult CreateBlockExecutor(const uint32_t count, CcuRep::Executor* ccuExes);
    HcclResult CreateBlockCompletedEvent(const uint32_t count, CcuRep::CompletedEvent* ccuEvents);

    // 内部 *Nb / RecordEvent / WaitEvent 系列：mask 由调用方独立传入，
    // 不再从 CompletedEvent 上读取。
    HcclResult RecordEvent(CcuRep::CompletedEvent event, uint32_t mask);
    HcclResult WaitEvent(CcuRep::CompletedEvent event, uint32_t mask);

    // 数据操作
    HcclResult WriteNb(
        const ChannelHandle channel, const CcuRep::RemoteAddr& rem, const CcuRep::LocalAddr& loc,
        const CcuRep::Variable& len, CcuRep::CompletedEvent event, uint32_t mask);
    HcclResult WriteNb(
        const ChannelHandle channel, const CcuRep::RemoteAddr& rem, const CcuRep::CcuBuf& loc,
        const CcuRep::Variable& len, CcuRep::CompletedEvent event, uint32_t mask);

    HcclResult ReadNb(
        const ChannelHandle channel, const CcuRep::LocalAddr& loc, const CcuRep::RemoteAddr& rem,
        const CcuRep::Variable& len, CcuRep::CompletedEvent event, uint32_t mask);
    HcclResult ReadNb(
        const ChannelHandle channel, const CcuRep::CcuBuf& loc, const CcuRep::RemoteAddr& rem,
        const CcuRep::Variable& len, CcuRep::CompletedEvent event, uint32_t mask);

    HcclResult WriteReduceNb(
        const ChannelHandle channel, const CcuRep::RemoteAddr& rem, const CcuRep::LocalAddr& loc,
        const CcuRep::Variable& len, HcclDataType dataType, HcclReduceOp opType, CcuRep::CompletedEvent event,
        uint32_t mask);
    HcclResult ReadReduceNb(
        const ChannelHandle channel, const CcuRep::LocalAddr& loc, const CcuRep::RemoteAddr& rem,
        const CcuRep::Variable& len, HcclDataType dataType, HcclReduceOp opType, CcuRep::CompletedEvent event,
        uint32_t mask);

    HcclResult LocalCopyNb(
        const CcuRep::LocalAddr& dst, const CcuRep::LocalAddr& src, const CcuRep::Variable& len,
        CcuRep::CompletedEvent event, uint32_t mask); // dst和src是否都是local
    HcclResult LocalCopyNb(
        const CcuRep::CcuBuf& dst, const CcuRep::LocalAddr& src, const CcuRep::Variable& len,
        CcuRep::CompletedEvent event, uint32_t mask);
    HcclResult LocalCopyNb(
        const CcuRep::LocalAddr& dst, const CcuRep::CcuBuf& src, const CcuRep::Variable& len,
        CcuRep::CompletedEvent event, uint32_t mask);

    HcclResult LocalReduceNb(
        const CcuRep::LocalAddr& dst, const CcuRep::LocalAddr& src, const CcuRep::Variable& len, HcclDataType dataType,
        HcclReduceOp opType, CcuRep::CompletedEvent event, uint32_t mask);
    HcclResult LocalReduceNb(
        const CcuRep::CcuBuf* bufs, uint32_t count, HcclDataType dataType, HcclDataType outputDataType,
        HcclReduceOp opType, const CcuRep::Variable& len, CcuRep::CompletedEvent event, uint32_t mask);

    // 参数操作
    void Load(const CcuRep::Variable& var);

    // Variable src中存放内存地址，从地址中加载数据到Variable var中
    void LoadVariable(const CcuRep::Variable& src, const CcuRep::Variable& var);

    void StoreVariable(const CcuRep::Variable& var, uint64_t addr);
    // 控制逻辑
    // 宏定义IF、WHILE
    CcuRep::FuncCall Func(const std::string& label);
    CcuRep::FuncCall Func(const CcuRep::Variable& funcAddr);
    CcuRep::LoopCall Loop(const std::string& label);

private:
    CcuRep::Address CreateAddress();
    CcuRep::LocalAddr CreateLocalAddr(const CcuRep::Variable& token);

protected:
    GroupOpConfig moConfig_{0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF};

private:
    template <typename T>
    T CreateResAssist(std::array<std::vector<T>, CCU_MAX_IODIE_NUM>& resRecord);
    template <typename T>
    std::vector<T> CreateBlockResAssist(const uint32_t count, std::array<std::vector<T>, CCU_MAX_IODIE_NUM>& resRecord);

private:
    CcuRepResource res_{};
    CcuResRepository resRepo_{};

    std::mutex cascCntBlockMutex_;
    std::unordered_map<HcommCcuCascCntHandle, CntXnBlock> cascCntBlocks_{};

    std::unordered_set<ChannelHandle> channels_{};

    std::unordered_set<uint32_t> declaredLocXns_{};

    std::vector<PinnedGroupEntry> pinnedRegGroups_{};

    CcuRep::CcuInstrInfo instrInfo_{};

    uint32_t loadArgIndex_{0};

    CcuVersion ccuVersion_{CcuVersion::CCU_INVALID};

    // A5(CCU_V1) 专用：CCU_IF,CCU_ELSE 跳转共用一个 Xn；当前仅CCU_IF和CCU_ELSE复用，While和DoWhile暂未复用
    // A5 上 CCU_IF和ELSE的targetVar 仅用于LoadImdToXn + 紧接 Jump，生命周期不跨越其他指令，可安全复用。
    // A6不复用，保持每次独立分配的语义。
    std::unique_ptr<CcuRep::Variable> sharedJumpTargetVar_;

    CcuSharedResource exportedRes_{};
    CcuSharedResource importedRes_{};
    std::vector<GroupInfo> groupOpSizeInfo_;
    std::vector<CcuProfilingInfo> allCcuProfilingInfos_;

    // 记录每个kernel所需常量，适用于A6场景
    std::unordered_map<uint64_t, CcuRep::Variable> constValue2VarMap;

    struct LoopDescriptor {
        std::string label;
        std::shared_ptr<CcuRep::CcuRepLoopBlock> repLoopBlock;
        std::shared_ptr<CcuRep::CcuRepBlock> prevActiveBlock;
        bool bodyDefined{false};
    };

    struct VersionV2LoopRecord {
        CcuRep::Variable iterNumVar;
        CcuRep::Variable addrOffsetVar;
        CcuRep::Variable ctxIdVar;
    };

    struct LoopGroupDescriptor {
        CcuLoopGroupCfg config;
        uint64_t totalLoopNum{0};
        uint32_t loopCount{0};
        CcuRep::Variable parallelVar;
        CcuRep::Variable offsetVar;
        CcuRep::Variable xnOffsetVar;
        std::shared_ptr<CcuRep::CcuRepBase> bundleRep;
        bool isVarBased{false};
        bool isVersionV2{false};
        std::vector<VersionV2LoopRecord> versionV2Loops;
    };

    struct FuncDescriptor {
        const void* funcPtr{nullptr};
        std::string label;
        std::shared_ptr<CcuRep::CcuRepFuncBlock> repFuncBlock;
        std::shared_ptr<CcuRep::CcuRepBlock> prevActiveBlock;
        bool bodyDefined{false};
    };

    std::unordered_map<CcuLoop, LoopDescriptor> loopMap_;
    std::unordered_map<CcuLoopGroup, LoopGroupDescriptor> loopGroupMap_;
    CcuResult LookupLoopGroupAndLoop(
        CcuLoopGroup group, CcuLoop loop, const char* fnName, const char* createFnName, LoopGroupDescriptor*& grpDesc,
        LoopDescriptor*& loopDesc, uint32_t& loopIdx);
    uint32_t loopHandleCounter_{0};
    uint32_t loopGroupHandleCounter_{0};
    uint32_t loopBodyDepth_{0};

    std::unordered_map<uint64_t, FuncDescriptor> funcMap_;
    std::unordered_map<const void*, uint64_t> funcInstanceMap_;
    uint64_t funcHandleCounter_{0};
    bool inFuncBody_{false};
    // loop/func body 内首个非法错误的粘性闩，退出 body 时上抛（void body 无法直接回传）。
    CcuResult bodyError_{CcuResult::CCU_SUCCESS};

    std::unordered_map<CcuLoopExecutors, std::vector<CcuRep::Executor>> loopEnginePools_;
    uint32_t loopEnginePoolCounter_{0};

    std::string name_{};
};

CcuResult GetChannelIdByHandle(ChannelHandle channel, uint32_t& channelId);

} // namespace hcomm

#endif // HCOMM_CCU_KERNEL_H
