/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_service_base.h"
#include "communicator_impl.h"
#include "env_config_v2.h"
#include "dlprof_function_v2.h"
#include "hccl_log_keywords.h"
namespace Hccl {

constexpr u32 ADDR_SIZE = 2;

void CollServiceBase::RegisterOpBufToBufMgr(CollOperator& op)
{
    CHECK_NULLPTR(comm, "[CollServiceBase::RegisterOpBufToBufMgr] comm is nullptr!");
    DataBufManager& dataBufManager = comm->GetDataBufferManager();
    dataBufManager.Register(op.opTag, BufferType::INPUT, op.inputMem);
    dataBufManager.Register(op.opTag, BufferType::OUTPUT, op.outputMem);
    dataBufManager.Register(op.opTag, BufferType::SCRATCH, op.scratchMem);
}

void CollServiceBase::RegisterCclLocRmaBuffer() const // 注册CCL buffer
{
    if (comm->GetRankSize() == 1) {
        return;
    }
    CHECK_NULLPTR(comm, "[CollServiceBase::RegisterCclLocRmaBuffer] comm is nullptr!");
    CHECK_NULLPTR(comm->GetRankGraph(), "[CollServiceBase::RegisterCclLocRmaBuffer] comm->newVirtualTopo is nullptr!");
    auto myself = comm->GetRankGraph()->GetPeer(comm->GetMyRank());
    if (myself == nullptr) {
        THROW<NullPtrException>(StringFormat("CollServiceAiCpuImpl::Init ptr is null"));
    }
    bool p2pRegistered = false;
    const auto& interfacesMap = myself->GetInterfacesMap();
    for (const auto& pair : interfacesMap) {
        const auto& ifaceVec = pair.second;
        for (const auto& connIface : ifaceVec) {
            std::set<LinkProtocol> protocols = connIface->GetLinkProtocols();
            if (protocols.find(LinkProtocol::HCCS) != protocols.end()
                || protocols.find(LinkProtocol::PCIE) != protocols.end()) {
                if (p2pRegistered) {
                    break;
                }
                p2pRegistered = true;
            }
            auto& rmaBufManager = comm->GetLocalRmaBufManager();
            HCCL_INFO("rmaBufManager reg");
            PortData portData(comm->GetMyRank(), *connIface);
            HCCL_INFO("rmaBufManager reg portData[%s]", portData.Describe().c_str());
            if (portData.GetType() == PortDeploymentType::HOST_NET && portData.GetProto() == LinkProtoType::RDMA) {
                HCCL_WARNING("HostNet RDMA port will not be reged here"); // 1825不支持重复注册MR
                continue;
            }
            rmaBufManager.Reg(comm->GetId(), BufferType::SCRATCH, comm->GetCclBuffer(), portData, *(protocols.begin()));
        }
    }
}

void CollServiceBase::RegisterCclBuffer(const std::vector<LinkData>& links) const
{
    HCCL_INFO("RegisterCclBuffer reg links.size(%u)", links.size());
    CHECK_NULLPTR(comm, "[CollServiceBase::RegisterCclBuffer] comm is nullptr!");
    for (auto& link : links) {
        PortData portData = link.GetLocalPort();
        HCCL_INFO("RegisterCclBuffer reg portData[%s]", portData.Describe().c_str());

        auto& rmaBufManager = comm->GetLocalRmaBufManager();
        HCCL_INFO("RegisterCclBuffer reg");
        if (rmaBufManager.Get(comm->GetId(), portData, BufferType::SCRATCH) != nullptr) {
            HCCL_WARNING(
                "RegisterCclBuffer has reged, optag(%s) portData[%s]", comm->GetId().c_str(),
                portData.Describe().c_str());
            continue;
        }
        rmaBufManager.Reg(comm->GetId(), BufferType::SCRATCH, comm->GetCclBuffer(), portData, link.GetLinkProtocol());
    }
}

void CollServiceBase::RegisterOpbasedStream(unique_ptr<Stream> stream)
{
    CHECK_NULLPTR(comm, "[CollServiceBase::RegisterOpbasedStream] comm is nullptr!");
    StreamManager& sm = comm->GetStreamManager();
    CHECK_NULLPTR(sm.opbase, "[CollServiceBase::RegisterOpbasedStream] sm.opbase is nullptr!");
    sm.opbase->RegisterMaster(std::move(stream));
}

void CollServiceBase::RegisterOpbasedLocalRmaBuf(const std::string& opTag) const
{
    std::vector<BufferType> bufTypes = {BufferType::INPUT, BufferType::OUTPUT, BufferType::SCRATCH};
    std::unordered_map<BufferType, shared_ptr<DevBuffer>, std::EnumClassHash> devBuffers;
    CHECK_NULLPTR(comm, "[CollServiceBase::RegisterOpbasedLocalRmaBuf] comm is nullptr!");
    DataBufManager& dataBufManager = comm->GetDataBufferManager();
    for (auto& bufType : bufTypes) {
        auto dataBuf = dataBufManager.Get(opTag, bufType);
        if (dataBuf != nullptr) {
            devBuffers[bufType] = DevBuffer::Create(dataBuf->GetAddr(), dataBuf->GetSize());
        } else {
            HCCL_WARNING("dataBuf[type=%s] is nullptr", bufType.Describe().c_str());
        }
    }

    CHECK_NULLPTR(
        comm->GetRankGraph(), "[CollServiceBase::RegisterOpbasedLocalRmaBuf] comm->newVirtualTopo is nullptr!");
    auto myself = comm->GetRankGraph()->GetPeer(comm->GetMyRank());
    if (myself == nullptr) {
        THROW<NullPtrException>(StringFormat("CollServiceDefaultImpl::Init ptr is null"));
    }
    auto& localRmaBufManager = comm->GetLocalRmaBufManager();
    const auto& interfacesMap = myself->GetInterfacesMap();
    for (const auto& pair : interfacesMap) {
        const auto& ifaceVec = pair.second;
        for (const auto& connIface : ifaceVec) {
            PortData portData(comm->GetMyRank(), *connIface);
            std::set<LinkProtocol> protocols = connIface->GetLinkProtocols();
            for (auto& devBuf : devBuffers) {
                if (localRmaBufManager.Get(comm->GetId(), portData, devBuf.first) != nullptr) {
                    HCCL_WARNING(
                        "RegisterOpbasedLocalRmaBuf has reged, bufferType[%s], optag[%s] portData[%s]",
                        devBuf.first.Describe().c_str(), comm->GetId().c_str(), portData.Describe().c_str());
                    continue;
                }
                if (devBuf.first != BufferType::SCRATCH && portData.GetType() == PortDeploymentType::P2P) {
                    HCCL_WARNING("Input and Output Mem will not be reged at P2P");
                    continue;
                }
                localRmaBufManager.Reg(opTag, devBuf.first, devBuf.second, portData, *(protocols.begin()));
            }
        }
    }
}

void CollServiceBase::RegisterOffloadLocalRmaBuf(const std::string& opTag) const
{
    std::vector<BufferType> bufTypes = {BufferType::INPUT, BufferType::OUTPUT, BufferType::SCRATCH};
    std::unordered_map<BufferType, shared_ptr<DevBuffer>, std::EnumClassHash> devBuffers;
    CHECK_NULLPTR(comm, "[CollServiceBase::RegisterOffloadLocalRmaBuf] comm is nullptr!");
    DataBufManager& dataBufManager = comm->GetDataBufferManager();
    for (auto& bufType : bufTypes) {
        auto dataBuf = dataBufManager.Get(opTag, bufType);
        if (dataBuf != nullptr) {
            devBuffers[bufType] = DevBuffer::Create(dataBuf->GetAddr(), dataBuf->GetSize());
        } else {
            HCCL_WARNING("dataBuf[type=%s] is nullptr", bufType.Describe().c_str());
        }
    }

    CHECK_NULLPTR(
        comm->GetRankGraph(), "[CollServiceBase::RegisterOffloadLocalRmaBuf] comm->newVirtualTopo is nullptr!");
    auto myself = comm->GetRankGraph()->GetPeer(comm->GetMyRank());
    if (myself == nullptr) {
        THROW<NullPtrException>(StringFormat("CollServiceDefaultImpl::Init ptr is null"));
    }
    auto& localRmaBufManager = comm->GetLocalRmaBufManager();
    const auto& interfacesMap = myself->GetInterfacesMap();
    for (const auto& pair : interfacesMap) {
        const auto& ifaceVec = pair.second;
        for (const auto& connIface : ifaceVec) {
            PortData portData(comm->GetMyRank(), *connIface);
            std::set<LinkProtocol> protocols = connIface->GetLinkProtocols();
            for (auto& devBuf : devBuffers) {
                HCCL_INFO("CollServiceBase::RegisterOffloadLocalRmaBuf, devBuf[%s]", devBuf.second->Describe().c_str());
                localRmaBufManager.Reg(opTag, devBuf.first, devBuf.second, portData, *(protocols.begin()));
            }
        }
    }
}

void CollServiceBase::RegisterOffloadMasterStream(const std::string& opTag, unique_ptr<Stream> stream) const
{
    CHECK_NULLPTR(comm, "[CollServiceBase::RegisterOffloadMasterStream] comm is nullptr!");
    StreamManager& sm = comm->GetStreamManager();
    CHECK_NULLPTR(sm.offload, "[CollServiceBase::RegisterOffloadMasterStream] sm.offload is nullptr!");
    sm.offload->RegisterMaster(opTag, std::move(stream));
}

CollServiceBase::CollServiceBase(CommunicatorImpl* comm) : comm(comm) {}

void CollServiceBase::AllocCommResource(
    [[maybe_unused]] void* mc2Tiling, [[maybe_unused]] void** commContext,
    [[maybe_unused]] const AcceleratorState& tilingAccelerator)
{
    THROW<NotSupportException>("AllocCommResource was not support in this mode.");
}

HcclResult CollServiceBase::AllocCollOpResource(
    [[maybe_unused]] CollOperator& op, [[maybe_unused]] const std::string& opAlgTag, [[maybe_unused]] void** addr)
{
    HCCL_ERROR("[%s] is not supported in this mode.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

void CollServiceBase::GetCcuTaskInfo([[maybe_unused]] void* tilingData, [[maybe_unused]] void* ccuTaskGroup)
{
    THROW<NotSupportException>("GetCcuTaskInfo was not support in this mode.");
}

u32 CollServiceBase::GetCcuMc2ServerNum()
{
    THROW<NotSupportException>("GetCcuMc2ServerNum was not support in this mode.");
    return 0;
}

void CollServiceBase::Resume() { THROW<NotSupportException>("Resume was not support in this mode."); }

void CollServiceBase::WaitOpbasedTransportReady() const
{
    CHECK_NULLPTR(comm, "[CollServiceBase::WaitOpbasedTransportReady] comm is nullptr!");
    CHECK_NULLPTR(
        comm->GetMemTransportManager(),
        "[CollServiceBase::WaitOpbasedTransportReady] comm->GetMemTransportManager is nullptr!");
    auto timeout = std::chrono::seconds(EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut());

    HcclUs startTime = std::chrono::steady_clock::now();
    while (true) {
        if (comm->GetMemTransportManager()->IsAllOpbasedTransportReady()) {
            break;
        }
        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            string timeoutMsg = StringFormat("WaitOpbasedTransportReady timeout, commId[%s].", comm->GetId().c_str());
            RPT_INPUT_ERR(true, "EI0006", std::vector<std::string>({"reason"}), std::vector<std::string>({timeoutMsg}));
            HCCL_ERROR(
                "[%s][%s] wait socket establish timeout, %s", LOG_KEYWORDS_INIT_CHANNEL.c_str(),
                LOG_KEYWORDS_TIMEOUT.c_str(), timeoutMsg.c_str());
            comm->GetMemTransportManager()->DumpNotReadyTransportsOpbased();
            THROW<InternalException>(timeoutMsg);
        }
    }
}

void CollServiceBase::WaitOffloadTransportReady(const std::string& opTag) const
{
    CHECK_NULLPTR(comm, "[CollServiceBase::WaitOffloadTransportReady] comm is nullptr!");
    CHECK_NULLPTR(
        comm->GetMemTransportManager(),
        "[CollServiceBase::WaitOffloadTransportReady] comm->GetMemTransportManager is nullptr!");
    auto timeout = std::chrono::seconds(EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut());

    HcclUs startTime = std::chrono::steady_clock::now();
    while (true) {
        if (comm->GetMemTransportManager()->IsAllOffloadTransportReady(opTag)) {
            break;
        }
        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            string timeoutMsg = StringFormat(
                "WaitOffloadTransportReady timeout, opTag[%s] commId[%s].", opTag.c_str(), comm->GetId().c_str());
            RPT_INPUT_ERR(true, "EI0006", std::vector<std::string>({"reason"}), std::vector<std::string>({timeoutMsg}));
            HCCL_ERROR(
                "[%s][%s] wait socket establish timeout, %s", LOG_KEYWORDS_INIT_CHANNEL.c_str(),
                LOG_KEYWORDS_TIMEOUT.c_str(), timeoutMsg.c_str());
            comm->GetMemTransportManager()->DumpNotReadyTransportsOffload(opTag);
            THROW<InternalException>(timeoutMsg);
        }
    }
}

void CollServiceBase::WaitTransportReady(const std::string& opTag) const
{
    CHECK_NULLPTR(comm, "[CollServiceBase::WaitTransportReady] comm is nullptr!");
    CHECK_NULLPTR(
        comm->GetMemTransportManager(),
        "[CollServiceBase::WaitTransportReady] comm->GetMemTransportManager is nullptr!");
    auto timeout = std::chrono::seconds(EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut());

    HcclUs startTime = std::chrono::steady_clock::now();
    while (true) {
        auto op = comm->GetCurrentCollOperator();
        if (op->opMode == OpMode::OPBASE) {
            if (comm->GetMemTransportManager()->IsAllOpbasedTransportReady()) {
                break;
            }
        } else if (op->opMode == OpMode::OFFLOAD) {
            if (comm->GetMemTransportManager()->IsAllOffloadTransportReady(opTag)) {
                break;
            }
        }
        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            RPT_INPUT_ERR(
                true, "EI0006", std::vector<std::string>({"reason"}),
                std::vector<std::string>({"WaitTransportReady timeout, SOCKET_TIMEOUT."}));
            HCCL_ERROR(
                "[%s][%s] wait socket establish timeout, WaitTransportReady timeout, opTag[%s] commId[%s].",
                LOG_KEYWORDS_INIT_CHANNEL.c_str(), LOG_KEYWORDS_TIMEOUT.c_str(), opTag.c_str(), comm->GetId().c_str());
            THROW<InternalException>(
                "WaitTransportReady timeout, opTag[%s] commId[%s].", opTag.c_str(), comm->GetId().c_str());
        }
    }
}

void CollServiceBase::AddOpCounterMems()
{
    HCCL_INFO("[CollServiceBase::%s] start.", __func__);

    u64 size = 4 * 3; // 第一个四字节用于计数加1, 后面两个四字节分别保存headCounter和tailCounter
    counterBuf = std::make_shared<DevBuffer>(size);

    // 初始化第一个四字节置1, 用于计数加1, reduce task add 1
    u64 srcSize = 4;
    float srcValue = 1;
    void* srcAddr = reinterpret_cast<void*>(counterBuf->GetAddr());
    HrtMemcpy(srcAddr, srcSize, &srcValue, srcSize, RT_MEMCPY_HOST_TO_DEVICE);

    // 初始化后面两个四字节置0
    u64 countMemSize = srcSize;
    float startValue = 0; // value为0表示从0开始计数
    void* headCountAddr = reinterpret_cast<void*>(counterBuf->GetAddr() + srcSize);
    void* tailCountAddr = reinterpret_cast<void*>(counterBuf->GetAddr() + srcSize * 2);
    HrtMemcpy(headCountAddr, countMemSize, &startValue, countMemSize, RT_MEMCPY_HOST_TO_DEVICE);
    HrtMemcpy(tailCountAddr, countMemSize, &startValue, countMemSize, RT_MEMCPY_HOST_TO_DEVICE);
    HCCL_INFO(
        "[CollServiceBase::%s] end, counterBuf[%llu] srcAddr[%p] headCountAddr[%p] tailCountAddr[%p].", __func__,
        counterBuf->GetAddr(), srcAddr, headCountAddr, tailCountAddr);
}

std::pair<u32, u32> CollServiceBase::GetOpCount()
{
    HCCL_INFO("[CollServiceBase::%s] start.", __func__);

    std::pair<float, float> floatCounter;
    u64 size = 4;
    if (counterBuf->GetSize() < size * ADDR_SIZE) {
        THROW<InternalException>("counterBuf size[%zu] is less than %u bytes", counterBuf->GetSize(), size * ADDR_SIZE);
    }
    void* headAddr = reinterpret_cast<void*>(counterBuf->GetAddr() + size);
    void* tailAddr = reinterpret_cast<void*>(counterBuf->GetAddr() + size * 2);
    HrtMemcpy(&floatCounter.first, size, headAddr, size, RT_MEMCPY_DEVICE_TO_HOST);
    HrtMemcpy(&floatCounter.second, size, tailAddr, size, RT_MEMCPY_DEVICE_TO_HOST);

    std::pair<u32, u32> counter;
    counter.first = static_cast<u32>(floatCounter.first);
    counter.second = static_cast<u32>(floatCounter.second);

    HCCL_INFO("[CollServiceBase::%s] end, head:%u, tail:%u", __func__, counter.first, counter.second);
    return counter;
}

DevBuffer* CollServiceBase::GetOpCounterBuf() { return counterBuf.get(); }

CollServiceBase::~CollServiceBase()
{
    if (counterBuf == nullptr) {
        return;
    }
    // 用于图模式算子计数打印，待有心跳检测后适配删除
    DECTOR_TRY_CATCH("CollServiceBase", {
        auto count = GetOpCount();
        HCCL_INFO("[CollServiceBase::~CollServiceBase] head:%u, tail:%u", count.first, count.second);
    });
}

// 功能说明：等待transport建链完成
// 输入说明：string &opTag：通信域ID，唯一标记一个通信域
bool CollServiceBase::IsAllTransportRecoveredReady([[maybe_unused]] const std::string& opTag)
{
    CHECK_NULLPTR(comm, "[CollServiceBase::IsAllTransportRecoveredReady] comm is nullptr!");
    auto op = comm->GetCurrentCollOperator();
    if (op->opMode == OpMode::OPBASE) {
        return comm->GetMemTransportManager()->IsAllOpbasedTransportRecoveredReady();
    } else if (op->opMode == OpMode::OFFLOAD) {
        return comm->GetMemTransportManager()->IsAllOffloadTransportRecoveredReady(op->opTag);
    }
    HCCL_ERROR("[CollServiceBase][IsAllTransportRecoveredReady] opMode[%d] is invalid", op->opMode);
    return false;
}

HcclResult CollServiceBase::GetSnapShotDynamicBuf([[maybe_unused]] CollOperator& op, [[maybe_unused]] BinaryStream& buf)
{
    HCCL_ERROR("[%s] not support.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

constexpr u32 TAILADDR_OFFSET_MULTIPLIER = 2;
void CollServiceBase::SaveMirrorDfxOpInfo()
{
    auto dfxOpInfo = std::make_shared<DfxOpInfo>();
    CHECK_NULLPTR(comm, "[CollServiceBase::SaveMirrorDfxOpInfo] comm is nullptr!");

    dfxOpInfo->op_ = *comm->GetCurrentCollOperator();
    dfxOpInfo->tag_ = dfxOpInfo->op_.opTag;
    dfxOpInfo->algType_ = comm->GetCurAlgName().c_str();
    dfxOpInfo->commIndex_ = comm->GetIdIndex();
    dfxOpInfo->comm_ = comm;
    dfxOpInfo->beginTime_ = DlProfFunction::GetInstance().dlMsprofSysCycleTime();
    dfxOpInfo->commId_ = comm->GetId();
    dfxOpInfo->opIndex_ = comm->GetOpIndex();
    u64 size = 4;
    dfxOpInfo->headOpCounterAddr_ = counterBuf->GetAddr() + size;
    dfxOpInfo->tailOpCounterAddr_ = counterBuf->GetAddr() + size * TAILADDR_OFFSET_MULTIPLIER;

    comm->GetMirrorTaskManager().SetCurrDfxOpInfo(dfxOpInfo);
}

void CollServiceBase::AddCountTask(bool isHead)
{
    if (counterBuf == nullptr) {
        AddOpCounterMems();
    }
    CHECK_NULLPTR(comm, "[CollServiceBase::AddCountTask] comm is nullptr!");

    u64 size = sizeof(float);
    void* dst = isHead == true ? reinterpret_cast<void*>(counterBuf->GetAddr() + size) :
                                 reinterpret_cast<void*>(counterBuf->GetAddr() + size * 2);
    void* src = reinterpret_cast<void*>(counterBuf->GetAddr());

    // 下发reduce task
    aclrtReduceKind rtReduceOp = ACL_RT_MEMCPY_SDMA_AUTOMATIC_SUM;
    aclDataType rtDataType = ACL_FLOAT;
    aclrtStream streamPtr = comm->GetStreamManager().GetMaster()->GetPtr();
    CHECK_NULLPTR(streamPtr, "master stream pointer is nullptr!");
    HrtReduceAsync(dst, size, src, size, rtReduceOp, rtDataType, streamPtr);

    HCCL_INFO("[CollServiceBase::AddCountTask] isHead[%d] dst addr[%p] src addr[%p]", isHead, dst, src);
}

void CollServiceBase::ReLoadWithOpBasedMode([[maybe_unused]] CollOperator& op)
{
    THROW<NotSupportException>("ReLoadWithOpBasedMode was not support in this mode.");
}

void CollServiceBase::ReLoadWithOffloadMode([[maybe_unused]] CollOperator& op)
{
    THROW<NotSupportException>("ReLoadWithOffloadMode was not support in this mode.");
}

void CollServiceBase::AllocQueueNotify(const InsQueue& insQueue)
{
    if (insQueue.SizeOfSlaves() == 0)
        return;
    AllocQNotifyForSingleQ(insQueue);

    for (auto slaveIt = insQueue.IterSlaves(); slaveIt.HasNext(); ++slaveIt) {
        AllocQNotifyForSingleQ(*slaveIt);
    }
}

void CollServiceBase::AllocQNotifyForSingleQ(const InsQueue& insQueue) const
{
    auto& queueNotifyManager = comm->GetCcuQueueNotifyManager();
    for (auto it = insQueue.Iter(); it.HasNext(); ++it) {
        const Instruction& ins = *it;
        auto insType = ins.GetType();
        if (insType == InstructionType::LOCAL_POST_TO) {
            const auto& p = static_cast<const InsLocalPostTo&>(ins);
            queueNotifyManager.ApplyFor(p.GetPostQid(), p.GetWaitQid(), p.GetTopicId());
        } else if (insType == InstructionType::LOCAL_WAIT_FROM) {
            const auto& p = static_cast<const InsLocalWaitFrom&>(ins);
            queueNotifyManager.ApplyFor(p.GetPostQid(), p.GetWaitQid(), p.GetTopicId());
        } else if (insType == InstructionType::LOCAL_WAIT_GROUP) {
            auto& queueWaitGroupCntNotifyManager = comm->GetQueueWaitGroupCntNotifyManager();
            const auto& p = static_cast<const InsLocalWaitGroup&>(ins);
            queueWaitGroupCntNotifyManager.ApplyFor(p.GetWaitQid(), p.GetTopicId());
        } else if (insType == InstructionType::LOCAL_BCAST_POST) {
            auto& queueBcastPostCntNotifyManager = comm->GetBcastPostCntNotifyManager();
            const auto& p = static_cast<const InsLocalBcastPost&>(ins);
            queueBcastPostCntNotifyManager.ApplyFor(p.GetPostQid(), p.GetTopicId());
        }
    }
}

HcclResult CollServiceBase::GetAlgExecParam(
    [[maybe_unused]] bool clearEnable, [[maybe_unused]] u32 numBlocks, [[maybe_unused]] void*& commContext,
    [[maybe_unused]] u64& len)
{
    HCCL_ERROR("GetAlgExecParam is not supported in this mode.");
    return HCCL_E_NOT_SUPPORT;
}

} // namespace Hccl
