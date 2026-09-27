/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>

#ifndef private
#define private public
#define protected public
#endif
#include "launch_aicpu.h"
#include "hccl_communicator.h"
#include "hccl_communicator_attrs.h"
#include "framework/aicpu_hccl_process.h"
#include "framework/aicpu_kfc_process.h"
#include "aicpu_hccl_common.h"
#undef private
#undef protected

#include "transport_pub.h"
#include "common/aicpu_kfc_def.h"

using namespace std;
using namespace hccl;

class Communicator_Device_UT : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "Communicator_Device_UT SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "Communicator_Device_UT TearDown" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        s32 portNum = -1;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
    }
};

TEST_F(Communicator_Device_UT, CommunicatorTest)
{
    RankTable_t rankTable;
    HcclCommParams params;
    std::vector<RankInfo> rankList;
    WorldGroupInfo groupCommonData;
    HcclIpAddress localIp{0};
    OpParam opParam;
    AlgDesc algDesc;
    AlgResourceResponse algResource;
    std::vector<RankInfo_t> rankInfoTList;

    HcclCommunicator hcclCommunicator;
    hcclCommunicator.Init(params, rankTable);
    hcclCommunicator.Init(params, rankList, groupCommonData);

    MOCKER_CPP(&HcclCommunicator::InitOneSidedService).stubs().will(returnValue(HCCL_SUCCESS));

    hcclCommunicator.InitOneSidedServiceNetDevCtx(0);
    hcclCommunicator.DeInitOneSidedServiceNetDevCtx();

    hcclCommunicator.GetOneSidedService(nullptr);
    hcclCommunicator.DeinitOneSidedService();

    hcclCommunicator.IsSupportZeroCopy(opParam);

    hcclCommunicator.PrepareZeroCopy("", algDesc, opParam);
    hcclCommunicator.UpdateZeroCopy(opParam, algResource);
    hcclCommunicator.BuildZeroCopyParam();

    hcclCommunicator.InitCommParams(params);
    hcclCommunicator.Is310PDuoCard();

    hcclCommunicator.CheckSingleServerComm(rankInfoTList);
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    hcclCommunicator.CheckDataType(dataType, true);

    hcclCommunicator.InitZeroCopyMemoryAgent();
    hcclCommunicator.DeinitZeroCopyMemoryAgent(true);

    string tag = "test";
    bool findTag = true;
    hcclCommunicator.ClearResMap(tag, findTag);
    hcclCommunicator.ClearOpResource("");

    u64 count = 1024;
    u32 buffer = 0;
    HcomCollOpInfo opInfo;
    opInfo.count = count;
    opInfo.dataType = dataType;
    opInfo.root = 1;
    opInfo.inputAddr = &buffer;
    opInfo.outputAddr = &buffer;
    opInfo.root = 0;
    hcclCommunicator.CreateOpBasedResources(HcclCMDType::HCCL_CMD_ALL, "", opInfo);
    hcclCommunicator.CreateRemoteOpBasedResources(0, tag);
    hcclCommunicator.DestroyRemoteOpBasedMem(tag);

    hcclCommunicator.IsAtomicInit();
    hcclCommunicator.IsNeedNicInit();

    float bandWidth = 0;
    hcclCommunicator.GetBandWidthPerNPU(0, bandWidth);
    hcclCommunicator.GetDeviceNumPerAggregation(buffer);

    hcclCommunicator.InitHccpChannel();

    HcclReduceOp op = HCCL_REDUCE_SUM;
    hcclCommunicator.CheckReduceDataType(dataType, op);

    AlgType algType;
    hcclCommunicator.GetAlgType(algType, HcclCMDType::HCCL_CMD_ALL);
    hcclCommunicator.GetCommParams(params);
    hcclCommunicator.GetCommRankTable(rankTable);

    hcclCommunicator.InitPara();
    hcclCommunicator.IsStandardCard();
    hcclCommunicator.InitOpRetry();

    ServerInfo_t serverInfoT;
    hcclCommunicator.CompareWithServerId(serverInfoT, serverInfoT);
    NetworkInfo_t netWorkInfoT;
    hcclCommunicator.CompareWithNicName(netWorkInfoT, netWorkInfoT);
    RankInfo rankInfo;
    hcclCommunicator.CompareWithUserRank(rankInfo, rankInfo);

    hcclCommunicator.InitPreResource(rankTable);
    hcclCommunicator.InitTcpMode(rankTable);
    hcclCommunicator.IsEnableBackupLink();
    hcclCommunicator.InitRaResource();
    hcclCommunicator.DisablePreResource();

    u64 streamNum;
    hcclCommunicator.GetWorkspaceSubStreamNum(count, dataType, op, tag, streamNum, 0, false, HcclCMDType::HCCL_CMD_ALL);
    hcclCommunicator.DestroyNetworkResources();
    std::vector<rtStream_t> streams;
    hcclCommunicator.SetWorkspaceResource(tag, nullptr, count, streams);
    hcclCommunicator.DestroyWorkspaceResource(tag);

    hcclCommunicator.AtomicInitSet();
    hcclCommunicator.AtomicInitClear();

    hcclCommunicator.GetUserRank();
    hcclCommunicator.GetGroupRank();
    hcclCommunicator.GetRankSize();
    hcclCommunicator.GetNicInitialized();

    bool ifAiv;
    hcclCommunicator.HcclSelectAlg(HcclCMDType::HCCL_CMD_ALL, 0, nullptr, dataType, op, 0, ifAiv, tag);
    u32 numBlocks;
    hcclCommunicator.HcclCalcNumBlocks(HcclCMDType::HCCL_CMD_ALL, 0, nullptr, dataType, 0, tag, numBlocks);

    void* commContext = nullptr;
    hcclCommunicator.HcclGetAlgExecParam(
        tag, HcclCMDType::HCCL_CMD_ALL, 0, nullptr, nullptr, true, dataType, op, commContext, count, 0);

    DevType devType = DevType::DEV_TYPE_910;
    hcclCommunicator.CheckDeviceType(devType);
    hcclCommunicator.CheckReductionOp(op);
    hcclCommunicator.CheckUserRank(0);
    hcclCommunicator.CheckCount(0);

    std::vector<u32> groupRanks;
    hcclCommunicator.GetGroupRanksInfo(groupRanks, rankList);
    hcclCommunicator.GetGroupCommonData(groupCommonData);
    hcclCommunicator.GetWorkspaceMemSize(tag, 0, dataType, buffer, count, devType);
    hcclCommunicator.GetWorkspaceScracthMem(tag, 0);
    hcclCommunicator.GetWorkspaceSubStreams(tag, 0);

    hcclCommunicator.InitProfiling();
    hcclCommunicator.DeinitProfiling();

    hcclCommunicator.RegistTaskExceptionHandler();
    hcclCommunicator.UnRegistTaskExceptionHandler();

    hcclCommunicator.GetInCCLbuffer(commContext, count);
    hcclCommunicator.GetOutCCLbuffer(commContext, count);
    hcclCommunicator.ReleaseCommCCLbuffer();
    hcclCommunicator.ReleaseCommInfos();

    hcclCommunicator.InitProfiler();
    hcclCommunicator.CreateCommCCLbuffer();

    hcclCommunicator.InitCCLbuffer(0, 0);

    hcclCommunicator.GetLocalNicPort(NicType::VNIC_TYPE);
    hcclCommunicator.InitNic(true);
    hcclCommunicator.DeinitNic();

    hcclCommunicator.RegisterRanksToDca();
    hcclCommunicator.RegisterToHeartBeat();
    hcclCommunicator.RegisterToHeartBeat(0, tag);
    hcclCommunicator.UnRegisterToHeartBeat();

    std::vector<void*> globalWorkSpaceAddr;
    hcclCommunicator.SetGlobalWorkSpace(globalWorkSpaceAddr);

    std::vector<HcclDumpInfo> hcclDumpInfo;
    hcclCommunicator.GetandClearOverFlowTasks(hcclDumpInfo);

    s32 deviceId;
    hcclCommunicator.GetDeviceId(deviceId);

    HcclResult result = HCCL_SUCCESS;
    hcclCommunicator.GetCqeError(result);

    hcclCommunicator.SupportDeterministicOptim(findTag);

    hcclCommunicator.GetHccsLinkNum(buffer);

    HcclRtStream rtStream;
    hcclCommunicator.AllGather(tag, nullptr, nullptr, 0, dataType, rtStream, nullptr);
    hcclCommunicator.AllGatherV(tag, nullptr, 0, nullptr, nullptr, nullptr, dataType, rtStream);
    hcclCommunicator.AicpuUnfold(tag, nullptr, nullptr, 0, dataType, op, rtStream, HcclCMDType::HCCL_CMD_ALL);
    hcclCommunicator.AllGatherOutPlace(tag, nullptr, nullptr, 0, dataType, rtStream);
    hcclCommunicator.AllGatherVOutPlace(tag, nullptr, nullptr, 0, commContext, commContext, dataType, rtStream);

    SyncMode syncModel = SyncMode::DEFAULT_TIMEWAITSYNCMODE;
    hcclCommunicator.GetAndSetSyncMode(syncModel, syncModel);
    hcclCommunicator.RestorePreSyncMode(syncModel, syncModel);

    hcclCommunicator.AllReduce(tag, nullptr, nullptr, 0, dataType, op, rtStream, syncModel);
    hcclCommunicator.AllReduceAicpuUnfold(tag, nullptr, nullptr, 0, dataType, op, rtStream);
    hcclCommunicator.AllReduceOutPlace(tag, nullptr, nullptr, 0, dataType, op, rtStream, syncModel);
    rtStream_t stream_t;
    hcclCommunicator.AlltoAllV(nullptr, nullptr, nullptr, dataType, nullptr, nullptr, nullptr, dataType, stream_t, tag);
    hcclCommunicator.AlltoAllVOutPlace(
        nullptr, nullptr, nullptr, dataType, nullptr, nullptr, nullptr, dataType, stream_t, tag);
    hcclCommunicator.AlltoAllVC(nullptr, nullptr, dataType, nullptr, dataType, stream_t, tag);
    hcclCommunicator.AlltoAllVCOutPlace(nullptr, nullptr, dataType, nullptr, dataType, stream_t, tag);
    hcclCommunicator.AlltoAll(nullptr, 0, dataType, nullptr, 0, dataType, stream_t, tag);

    hcclCommunicator.Broadcast(tag, nullptr, 0, dataType, 0, rtStream);
    hcclCommunicator.BroadcastOutPlace(tag, nullptr, 0, dataType, 0, rtStream);

    hcclCommunicator.Scatter(tag, nullptr, nullptr, 0, dataType, 0, rtStream);
    hcclCommunicator.ScatterOutPlace(tag, nullptr, nullptr, 0, dataType, 0, rtStream);

    hcclCommunicator.Reduce(tag, nullptr, nullptr, 0, dataType, op, 0, rtStream);
    hcclCommunicator.ReduceOutPlace(tag, nullptr, nullptr, 0, dataType, op, 0, rtStream);
    HcomCollOpInfo* opInfoPtr = nullptr;
    hcclCommunicator.ReduceScatter(tag, nullptr, nullptr, 0, dataType, op, rtStream, opInfoPtr);
    hcclCommunicator.ReduceScatterOutPlace(tag, nullptr, nullptr, 0, dataType, op, rtStream);
    hcclCommunicator.ReduceScatterV(tag, nullptr, nullptr, nullptr, nullptr, 0, dataType, op, rtStream, opInfoPtr);
    hcclCommunicator.ReduceScatterVOutPlace(tag, nullptr, nullptr, nullptr, nullptr, 0, dataType, op, rtStream);

    hcclCommunicator.BatchSendRecv(tag, nullptr, 0, stream_t);
    hcclCommunicator.Send(tag, nullptr, 0, dataType, 0, stream_t);
    hcclCommunicator.SendOutPlace(tag, nullptr, 0, dataType, 0, stream_t);

    hcclCommunicator.Receive(tag, nullptr, 0, dataType, 0, stream_t);
    hcclCommunicator.ReceiveOutPlace(tag, nullptr, 0, dataType, 0, stream_t);

    AlltoAllOperator* alltoAllOperator = nullptr;
    std::unique_ptr<PreProcessMetaInfo> preMetaInfo = nullptr;
    hcclCommunicator.RegressCalPreOp(alltoAllOperator, opParam, preMetaInfo);
    Stream stream;
    hcclCommunicator.RegressCalPreOp(alltoAllOperator, opParam, preMetaInfo, stream);
    hcclCommunicator.ExecOp(HcclCMDType::HCCL_CMD_ALL, opParam, true);
    DeviceMem deviceMem;
    hcclCommunicator.FreeScratchMemOnOpBaseMode(deviceMem, opParam, HcclCMDType::HCCL_CMD_ALL);
    hcclCommunicator.ExecOpAlltoAll(HcclCMDType::HCCL_CMD_ALL, opParam, true);
    hcclCommunicator.HandleAclGraphFirstOpAivBuff(stream_t);
    HcclCacheInfo cacheInfo;
    hcclCommunicator.ExecOpCache(HcclCMDType::HCCL_CMD_ALL, opParam, cacheInfo);

    hcclCommunicator.StreamIsCapture(stream_t);
    std::vector<Stream> slaveStreams;
    hcclCommunicator.CaptureSlaveStreams(stream_t, slaveStreams);

    LocalResInfoV2* localResHostPtr = nullptr;
    hcclCommunicator.BuildOpLocalScratchMemResParam(algResource, tag, localResHostPtr);

    hcclCommunicator.CheckSetRetryStateToWaitResume();
    hcclCommunicator.BuildOpLocalResParam(algResource, tag);
    hcclCommunicator.AllocAndGetStreamContextBuff(0, count, count);
    hcclCommunicator.UpdateOpIndex(opParam);
    hcclCommunicator.BuildAicpuCustomParam();

    vector<u32> tlv;
    hcclCommunicator.CopyVectorToDeviceMem(0, deviceMem, tlv);
    std::vector<std::vector<std::vector<u32>>> inputVectorInfo;
    hcclCommunicator.BuildOpTopoResTlvParam(tag, inputVectorInfo, deviceMem, count);
    std::vector<std::vector<std::vector<std::vector<u32>>>> inputVectorInfo2;
    hcclCommunicator.BuildOpTopoResVectorTlvParam(tag, inputVectorInfo2, deviceMem, count);
    hcclCommunicator.BuildPairLinkCounter(tag);
    hcclCommunicator.BuildIsUsedRdmaRank(tag);
    hcclCommunicator.BuildNicList(tag);
    hcclCommunicator.BuildBridgeRank(tag);
    hcclCommunicator.BuildCommPlanRank(tag);
    hcclCommunicator.BuildServerAndsuperPodRank(tag);
    hcclCommunicator.BuildOpRetryParam(algResource, tag);
    hcclCommunicator.BuildCommPlaneSubGroupRank(tag);
    hcclCommunicator.BuildOpTopoResParam(tag, algResource);
    LINK link;
    HccltagRemoteResV3 remoteResV3;
    TransportLinkType linkType = TransportLinkType::RESERVED;
    hcclCommunicator.BuildOpRemoteLinkP2pResParam(link, remoteResV3, linkType);
    hcclCommunicator.BuildOpRemoteLinkRoceResParam(link, remoteResV3, true, true, true);

    HcclRankRelationResV2* rankRelationResHostPtr = nullptr;
    HcclRankRelationResV2* rankRelationResDevicePtr = nullptr;
    hcclCommunicator.BuildRemoteResByTag(tag, buffer, rankRelationResHostPtr, rankRelationResDevicePtr, true, true);
    TransportRequest transportRequest;
    hcclCommunicator.BuildRelationResByRemoteRankId(
        transportRequest, link, rankRelationResHostPtr, rankRelationResDevicePtr);
    OpCommTransport opTransportResponse;
    hcclCommunicator.ParseRemoteDataToMem(opTransportResponse, tag, HcclCMDType::HCCL_CMD_ALL, true, true);
    hcclCommunicator.BuildOpRemoteResParam(algResource, tag, HcclCMDType::HCCL_CMD_ALL, true);

    ListCommon* headHostList = nullptr;
    hcclCommunicator.CopyHostListResToDeviceParam(tag, headHostList, 0);
    hcclCommunicator.CopyHostOpResToDeviceParam(tag);

    hcclCommunicator.BuildOpResParam(tag, algResource, tag, HcclCMDType::HCCL_CMD_ALL, stream_t);
    hcclCommunicator.BuildCustomOpResParam();
    hcclCommunicator.RegisterDfxInfo(opParam, algType, slaveStreams, true);
    hcclCommunicator.GetReportHcclMC2Info(stream, slaveStreams);

    hcclCommunicator.OrchestrateAicpu(HcclCMDType::HCCL_CMD_ALL, tag, opParam, algResource, tag, algType, true);
    hcclCommunicator.CalcTinySendRecvMem(opParam, algResource, deviceMem);

    NotifyLoadType notifyLoadType;
    std::vector<std::shared_ptr<LocalNotify>> notifiesMain;
    hcclCommunicator.AllocAlgNotifys(tag, notifyLoadType, 0, notifiesMain, notifiesMain);
    AlgResourceRequest resRequest;
    hcclCommunicator.AllocAlgResource(tag, HcclCMDType::HCCL_CMD_ALL, opParam, resRequest, algResource);
    hcclCommunicator.IncreAllocLink(tag, opParam, resRequest, algResource);

    hcclCommunicator.SetDevicePid(0);
    hcclCommunicator.ReleaseWorkSpacebuffer();

    std::shared_ptr<DeviceMem> bufferPtr;
    std::shared_ptr<HostMem> hostMemPtr;
    hcclCommunicator.AllocAndClearDeviceMem(0, bufferPtr);
    hcclCommunicator.AllocAndClearHostMem(0, hostMemPtr);

    hcclCommunicator.CreateWorkSpace(0, deviceMem);
    u64* workSpace = nullptr;
    hcclCommunicator.GetWorkSpace(workSpace, workSpace);
    hcclCommunicator.InitWorkSpace();
    hcclCommunicator.FillOpParam(HcclCMDType::HCCL_CMD_ALL, opParam, 0, nullptr, nullptr);

    hcclCommunicator.AllocComResource(tag, tag, HcclCMDType::HCCL_CMD_ALL, opParam, stream_t);
    hcclCommunicator.AllocComResourceByTiling(tag, nullptr);
    void** commContextPtr = nullptr;
    hcclCommunicator.CreateCommResource(tag, stream_t, true, commContextPtr);
    hcclCommunicator.Mc2CreateAndLaunchContext(stream_t, true, commContextPtr, tag);
    std::shared_ptr<LocalNotify> localNotify;
    HcclSignalInfo notifyInfo;
    hcclCommunicator.GetAiCpuNotifyData(localNotify, notifyInfo);
    hcclCommunicator.CreateAndGetAiCpuNotify(localNotify, notifyInfo);

    hcclCommunicator.Mc2AiCpuStreamAllocAndGet(0, stream_t);
    hcclCommunicator.Mc2AiCpuInitStreamAllocAndGet(0, stream_t);
    hcclCommunicator.AiCpuKernelLaunch(stream_t, 0, tag);
    aclrtFuncHandle funcHandle;
    aclrtArgsHandle argsHandle;
    AicpuOpTiling opTilingInfo;
    hcclCommunicator.AicpuKfcTilingDataLaunch(opParam, HcclCMDType::HCCL_CMD_ALL, deviceMem, tag, opTilingInfo);
    hcclCommunicator.AicpuInitOpTilingDataBuf(opParam, HcclCMDType::HCCL_CMD_ALL, tag, opTilingInfo, 0);
    hcclCommunicator.AicpuKfcTilingDataLaunchIn(opParam, deviceMem, tag, opTilingInfo, 0, true);
    hcclCommunicator.AicpuKfcTilingDataLaunchExt(
        opParam, HcclCMDType::HCCL_CMD_ALL, deviceMem, tag, opTilingInfo, true);
    HcclWorkflowMode mode = HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;
    hcclCommunicator.AicpuUnfoldKernelLaunch(nullptr, nullptr, stream_t, 0, nullptr, 0, tag, mode, tag);
    hcclCommunicator.AicpuUnfoldKernelLaunchV2(nullptr, nullptr, stream_t, 0, nullptr, 0, tag, mode, tag, true);
    hcclCommunicator.KernelLaunchChooseAicpuOrCustom(nullptr, nullptr, stream_t, 0, nullptr, 0, tag, mode, tag, true);

    hcclCommunicator.InitCombinOpara();
    hcclCommunicator.GetCommResource(tag, commContextPtr);
    hcclCommunicator.GetCommResource(commContext);
    hcclCommunicator.GetAicpuOpStreamNotify(nullptr, 0, nullptr);
    hcclCommunicator.GetAicpuOpStreamAndNotify(nullptr, 0, nullptr);

    hcclCommunicator.SetAicpuNotifyInvalid();
    std::unique_ptr<CommInfo> commInfo = nullptr;
    ;
    hcclCommunicator.ReplaceCommInfoByTag(tag, commInfo);
    level1StreamInfo_t streamInfo;
    hcclCommunicator.CreateMutiStreamResFor310P(tag, streamInfo);
    hcclCommunicator.CreateCommAndStreamRes(tag, stream);
    hcclCommunicator.GetComm(tag, nullptr);
    hcclCommunicator.SetCommResource(0, nullptr, nullptr, nullptr, nullptr, streamInfo, stream);
    hcclCommunicator.ReleaseCommContextbuffer();

    hcclCommunicator.CreateDeviceCommContext(0, deviceMem);
    hcclCommunicator.Break();
    hcclCommunicator.GetAlltoAllStagedWorkSpaceMemSize(nullptr, nullptr, dataType, nullptr, nullptr, dataType, count);
    std::vector<SendRecvInfo> allMeshAggregationSendRecvInfo;
    hcclCommunicator.GetAlltoAllStagedWorkSpaceMemSize(allMeshAggregationSendRecvInfo, count);
    hcclCommunicator.GetAllReduceScratchSize(0, dataType, count);
    std::unordered_map<std::string, std::map<u32, HcclIpAddress>> phyIdNicInfoMap;
    hcclCommunicator.SetWorldGroupInfo(phyIdNicInfoMap, rankList, groupRanks, groupRanks);
    hcclCommunicator.GetTopoDesc(nullptr, 0);
    hcclCommunicator.SetAivModeConfig(true);
    hcclCommunicator.SetAicpuUnfoldConfig(true);

    bool isChangedLink = true;
    hcclCommunicator.CheckExitWaitResumeState(isChangedLink);
    hcclCommunicator.SetMemoryRange(nullptr, 0, 0, 0);
    hcclCommunicator.UnsetMemoryRange(nullptr);
    hcclCommunicator.ActivateCommMemory(nullptr, 0, 0, nullptr, 0);
    hcclCommunicator.DeactivateCommMemory(nullptr);
    std::unordered_map<u32, bool> switchRanks;
    ChangeLinkInfo changeLinkInfo;
    hcclCommunicator.SetSingleLinkInfo(switchRanks, 0, changeLinkInfo);
    hcclCommunicator.SetRemoteRankLinkInfo(switchRanks, changeLinkInfo);
    std::map<u32, bool> remoteRankPortMap;
    hcclCommunicator.ActiveStoppedLink(remoteRankPortMap, opTransportResponse, true);

    hcclCommunicator.PrepareLinkForSwitchNic(switchRanks, changeLinkInfo);
    hcclCommunicator.ParseSwitchRanks(0, nullptr, nullptr, switchRanks);
    std::shared_ptr<HDCommunicate> controlH2D = nullptr;
    hcclCommunicator.SwitchNic(0, nullptr, nullptr, controlH2D, controlH2D);
    hcclCommunicator.GetSwitchRanks(nullptr, nullptr, buffer, nullptr, buffer, isChangedLink, isChangedLink);
    aclrtBinaryLoadOptionType optionType = ACL_RT_BINARY_LOAD_OPT_LAZY_LOAD;
    aclrtBinHandle binHandle;
    hcclCommunicator.LoadCustomFile(nullptr, optionType, 0, binHandle);
    hcclCommunicator.UnloadBinary(binHandle);
    hcclCommunicator.SetOnlyAivModeConfig(true);
}

TEST_F(Communicator_Device_UT, CommunicatorAttrsTest)
{
    RankTable_t rankTable;
    HcclCommParams params;
    std::vector<RankInfo> rankList;
    WorldGroupInfo groupCommonData;
    HcclCommunicatorAttrs commAttrs;
    std::vector<RankInfo_t> rankInfoTList;
    RankInfo rankInfo;
    commAttrs.Init(params, rankTable);
    commAttrs.Init(params, rankList, groupCommonData);

    commAttrs.IsStandardCard();
    commAttrs.Is310PDuoCard();
    commAttrs.IsCommon310P3DUO(rankInfoTList);

    DevType devType = DevType::DEV_TYPE_910;
    commAttrs.CompareWithUserRank(rankInfo, rankInfo);
    commAttrs.CheckDeviceType(devType);
    NICDeployment nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    commAttrs.GetNicInfo(nicDeploy, 0, rankInfoTList, rankInfo);

    commAttrs.InitCommParams(params);
    commAttrs.SetServerId(rankTable);
    commAttrs.SetServerNum(rankInfoTList);
    commAttrs.SetInnerServerAverageDevice(rankTable);
    commAttrs.SetInnerServerAverageDevice(rankList);

    ServRankInfo servRankInfo;
    commAttrs.TransformRankInfoByServerId(rankInfoTList, servRankInfo);
    RankInfo_t rankInfoT;
    commAttrs.CompareWithDevicePhyId(rankInfoT, rankInfoT);
    commAttrs.SetModuleInfo(rankInfoTList);
    commAttrs.SetSuperPodInfo(rankInfoTList);
    u32 moduleIdx = 0;
    commAttrs.GetModuleIdx(rankInfoT, moduleIdx);

    commAttrs.IsDiffDeviceModule(rankInfoTList);
    commAttrs.InitHccsPortNum();
    commAttrs.SetRankInfoList(rankTable);

    commAttrs.CheckRankTable(rankTable, servRankInfo);
    s32 devicePhyId = 0;
    commAttrs.CheckDevPhyId(devicePhyId);
    commAttrs.SortRankInfoList();
    commAttrs.CheckNicDeploy(nicDeploy, devType);
    commAttrs.CheckDevCount(moduleIdx);
    commAttrs.Check2N(moduleIdx);

    commAttrs.SetLocalRankInfo();
    commAttrs.SetLocalRankInfoSubGroup(rankList);
    commAttrs.CheckLocalRankInfo();
    commAttrs.CalMeshAggRankSize(0);
    commAttrs.SetMeshAggregationRankSize(0);
    commAttrs.CalAndSetMeshAggRankSize();

    std::unordered_map<std::string, std::map<u32, HcclIpAddress>> phyIdNicInfoMap;
    std::vector<u32> nicRanksPort;
    commAttrs.SetWorldGroupInfo(phyIdNicInfoMap, rankList, nicRanksPort, nicRanksPort);
    commAttrs.TransformRankList(rankList, rankInfoTList);
    commAttrs.IsEnableRoce();
    commAttrs.IsUsedRdmaLevel0AndIpInvalid();
    commAttrs.IsSupportEnableRoce();
    HcclTopoAttr topoAttr;
    commAttrs.GetTopoAttr(topoAttr);
    HcclAlgoAttr algoAttr;
    commAttrs.GetAlgoAttr(algoAttr);
    commAttrs.GetLocalNicPort(NicType::VNIC_TYPE);
}

TEST_F(Communicator_Device_UT, Ut_WaitAsyncFlag_When_ParamIsNullptr_Expect_ReturnIsHCCL_E_PTR)
{
    MOCKER(GetCurCpuTimestamp).stubs().will(returnValue(static_cast<u64>(0)));
    HcclResult ret = AicpuHcclProcess::WaitAsyncFlag(nullptr, 1, 0);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(Communicator_Device_UT, Ut_AicpuIndOpNotifyInit_When_ParamIsNullptr_Expect_ReturnIsHCCL_E_PTR)
{
    HcclResult ret = AicpuHcclProcess::AicpuIndOpNotifyInit(nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(Communicator_Device_UT, Ut_WaitAsyncFlag_When_FlagValueLessThanOffset_Expect_ReturnIsHCCL_E_PARA)
{
    HcclResult ret = AicpuHcclProcess::WaitAsyncFlag(nullptr, 0, 0);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(Communicator_Device_UT, Ut_WaitAsyncFlag_When_Timeout_Expect_ReturnIsHCCL_E_INTERNAL)
{
    uint32_t flagMem[POST_SEND_FLAG_COUNT] = {0};
    Transport::Buffer buffers[POST_SEND_FLAG_COUNT];
    for (uint32_t i = 0; i < POST_SEND_FLAG_COUNT; i++) {
        buffers[i].addr = &flagMem[i];
        buffers[i].size = sizeof(uint32_t);
    }
    HcclResult ret = AicpuHcclProcess::WaitAsyncFlag(buffers, FLAG_OFFSET, 0);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(Communicator_Device_UT, Ut_WaitAsyncFlag_When_FlagMatched_Expect_ReturnSuccess)
{
    uint32_t flagMem[POST_SEND_FLAG_COUNT] = {0};
    Transport::Buffer buffers[POST_SEND_FLAG_COUNT];
    for (uint32_t i = 0; i < POST_SEND_FLAG_COUNT; i++) {
        buffers[i].addr = &flagMem[i];
        buffers[i].size = sizeof(uint32_t);
    }
    uint32_t flagValue = FLAG_OFFSET;
    flagMem[flagValue - FLAG_OFFSET] = flagValue;
    HcclResult ret = AicpuHcclProcess::WaitAsyncFlag(buffers, flagValue, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(flagMem[flagValue - FLAG_OFFSET], 0U);
}
