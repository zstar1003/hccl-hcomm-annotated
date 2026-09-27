/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <list>
#include <vector>
#include <string>
#include <securec.h>
#include <hccl/hccl_types.h>
#include "hcom_private.h"
#include "config.h"
#include "externalinput_pub.h"
#include "workflow_pub.h"
#include "gradient_segment.h"
#include "hccl/base.h"
#include "rank_consistentcy_checker.h"
#include "param_check_pub.h"
#include "comm_configer.h"

#include "../common/src/topo/topoinfo_detect.h"
#include "profiling_manager.h"
#include "../op_base/src/op_base.h"
#include "adapter_rts_common.h"
#include "adapter_prof.h"
#include "topoinfo_ranktableParser_pub.h"
#include "hccl_communicator.h"
#include "hccl/hcom.h"
#include "topoinfo_ranktableOffline.h"
#include "mmpa_api.h"
#include "hccl_tbe_task.h"
#include "hcom_private_v2.h"
#include "comm_topo_desc.h"
#include "hcom_common.h"
#include "hcom_pub.h"

using namespace std;
using namespace hccl;

HcclResult CallMsprofReportHostApi(
    hccl::hcclComm* hcclComm, HcclCMDType cmdType, uint64_t beginTime, u64 count, HcclDataType dataType)
{
    auto& profilingManager = hccl::ProfilingManager::Instance();
    AlgType algType;
    if (cmdType == HcclCMDType::HCCL_CMD_RECEIVE || cmdType == HcclCMDType::HCCL_CMD_SEND) {
        algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_RESERVED;
        algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RESERVED;
    } else {
        CHK_RET(hcclComm->GetAlgType(algType, cmdType));
    }
    uint64_t groupName = hrtMsprofGetHashId(hcclComm->GetIdentifier().c_str(), hcclComm->GetIdentifier().length());
    CHK_RET(profilingManager.CallMsprofReportHostApi(cmdType, beginTime, count, dataType, algType, groupName));
    hcclComm->SetAivCoreLimit(0);
    HCCL_DEBUG(
        "CallMsprofReportHostApi success, cmdType[%d], count[%llu], dataType[%d], algType[%d], groupName[%llu]",
        cmdType, count, dataType, algType.algoLevel0, groupName);
    return HCCL_SUCCESS;
}

HcclResult HcomCheckInitClusterInfo(const char* rankTableM, const char* identify);
HcclResult HcomFlushBackloggedGroups();
HcclResult HcomCollRemotePairedParaCheck(const HcomRemoteOperationParams& params);

HcclResult HcomInit(const char* rankTableM, const char* identify, WorkMode commWorkMode)
{
    HcclResult ret = HCCL_SUCCESS;
    bool& isAutoTuneModeOpen = HcomGetCtxAutoTuneMode();
    isAutoTuneModeOpen = false;
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();

    /*--------------入参合法性检测---------------------*/
    CHK_PTR_NULL(rankTableM);
    CHK_PTR_NULL(identify);

    /* 防止重复调用初始化 */
    CHK_PRT_RET(
        (hcomInfo.pComm != nullptr),
        HCCL_ERROR(
            "[Init][Result]errNo[0x%016llx] identify[%s], "
            "multiple initialization is not supported",
            HCOM_ERROR_CODE(HCCL_E_UNAVAIL), identify),
        HCCL_E_UNAVAIL);

    /* --------------初始化------------------------- */
    bool errorFlag = false;
    s32 logicDevId = 0;
    hcomInfo.params.commWorkMode = commWorkMode;
    do {
        ret = InitHcomMiscInfo(hcomInfo.params, rankTableM);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS, HCCL_ERROR("[Init][Result]errNo[0x%016llx] init other Info.", HCOM_ERROR_CODE(ret)),
            errorFlag = true);

        DevType deviceType;
        if (commWorkMode != HCCL_MODE_SCHED_OS) {
            CHK_PRT_BREAK(hrtGetDevice(&logicDevId) != HCCL_SUCCESS, , errorFlag = true);
            CHK_RET(hrtGetDeviceType(deviceType));
            // 为适配12包，做此修改
            (void)HcomCheckrtMemcpyAddrAsync(identify);
        } else {
            deviceType = DevType::DEV_TYPE_NOSOC;
        }
        ret = CfgGetClusterInfo(
            rankTableM, identify, hcomInfo.params, hcomInfo.rankTable, GetExternalInputInterSuperPodRetryEnable(),
            deviceType);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Init][Result]errNo[0x%016llx] cfg get ranktable[%p] info "
                "error: identify[%s]",
                HCOM_ERROR_CODE(ret), rankTableM, identify),
            errorFlag = true);

        // HCCL_MODE_SCHED_OS仅在910A等同构集合通信场景才存在，此处仅希望异构资源池化场景进入
        if (commWorkMode != HCCL_MODE_SCHED_OS) {
            /*
            此case仅希望310 soc形态才能进入(此时有reacource.json来配置通信协议)
            但是310有板卡形态有可能进入(无reacource.json)，因此设定serverNum！=1的条件
            因为310板卡形态跑大模型切分(AllReduce)当前都是单机(serverNum=1)，因此无需通信协议解析
            */
            if (hcomInfo.rankTable.serverNum != SINGLE_SERVER_NUM
                && (deviceType == DevType::DEV_TYPE_310P3 || deviceType == DevType::DEV_TYPE_310P1)) {
                CHK_RET(InitExternalInputHeterog());
            }
        }

        const char* group;

        hcomInfo.pComm.reset(new (std::nothrow) hccl::hcclComm(0, 0, HCCL_WORLD_GROUP));

        CHK_PRT_RET(
            hcomInfo.pComm == nullptr,
            HCCL_ERROR("[Init][Result]hcomInfo.pComm is null, "
                       "create failed"),
            HCCL_E_PTR);
        CommConfig commConfig(HCCL_WORLD_GROUP);
        ret = hcomInfo.pComm->init(hcomInfo.params, commConfig, hcomInfo.rankTable);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS, HCCL_ERROR("[Init][Result]errNo[0x%016llx] hcclComm init error", HCOM_ERROR_CODE(ret)),
            errorFlag = true);

        group = hcomInfo.pComm->GetIdentifier().c_str();

        ret = ShowRanktableConfigInfo(hcomInfo.cloudFlag, hcomInfo.params, hcomInfo.rankTable);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[Init][Result]errNo[0x%016llx] put ranktable info error", HCOM_ERROR_CODE(ret)),
            errorFlag = true);
        if (commWorkMode != HCCL_MODE_SCHED_OS) {
            ret = InitWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);
            CHK_PRT_BREAK(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[Init][Result]errNo[0x%016llx] init work flow mode error", HCCL_ERROR_CODE(ret)),
                errorFlag = true);
        }

        ret = HcomFlushBackloggedGroups();
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[Init][Result]errNo[0x%016llx] create backlogged group failed", HCOM_ERROR_CODE(ret)),
            errorFlag = true);

        ret = HcomSetGroupTopoInfo(hcomInfo.pComm->GetIdentifier().c_str(), hcomInfo.rankTable.rankNum);

        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Init][Result]errNo[0x%016llx] SetGroupTopoInfo error, "
                "group[%s]",
                HCOM_ERROR_CODE(ret), group),
            errorFlag = true);
    } while (0);

    if (errorFlag) {
        HCCL_ERROR(
            "[Init][Result]hcom init failed, rankNum[%u], rank[%u], server[%s], device[%d], return[0x%016llx]",
            hcomInfo.rankTable.rankNum, hcomInfo.params.rank, hcomInfo.params.serverId.c_str(), logicDevId,
            HCOM_ERROR_CODE(ret));
        (void)HcomDestroy();
        return ret;
    }
    return HCCL_SUCCESS;
}

HcclResult
HcomInitByString(const char* rankTableM, const char* identify, WorkMode commWorkMode, HcomInitConfig* initConfig)
{
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    CHK_PTR_NULL(rankTableM);
    CHK_PTR_NULL(identify);

    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        CHK_RET(HcomInitByStringV2(rankTableM, identify));
        s32 myRank = std::atoi(identify);
        Hccl::RankId rank = static_cast<Hccl::RankId>(myRank);
        void* commV2 = nullptr;
        CHK_RET(HcomGetCommV2(&commV2));
        CHK_RET(HcomInitCollComm(rank, &commV2, hcomInfo.pComm));
        u32 rankNum = 0;
        CHK_RET(HcomGetRankSize(HCCL_WORLD_GROUP, &rankNum));
        CHK_RET(HcomSetGroupTopoInfo(HCCL_WORLD_GROUP, rankNum));
        return HCCL_SUCCESS;
    }());

    if (initConfig != nullptr) {
        DevType devType;
        CHK_RET(hrtGetDeviceType(devType));

        CHK_RET(HcomSetAlgorithm(initConfig->algo));
        CHK_RET(HcomSetExecTimeOut(initConfig->execTimeOut));
        if (devType != DevType::DEV_TYPE_910_93) {
            CHK_RET(HcomSetDeterministic(initConfig->deterministic));
        } else {
            HCCL_WARNING("ParserHcclDeterministic: device type is 910_93, use default setting");
        }
    }

    HcclUs startut = TIME_NOW();
    HcclResult ret = HCCL_SUCCESS;
    CHK_RET(InitExternalInput());
    CHK_RET(InitEnvConfig());
    CHK_RET(HcomCheckInitClusterInfo(rankTableM, identify));
    HCCL_RUN_INFO(
        "Entry-HcomInitByString, rankTableM[%s], identify[%s], commWorkMode[%d]", rankTableM, identify, commWorkMode);

    ret = HcomInit(rankTableM, identify, commWorkMode);

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[HcomInitByString]errNo[0x%016llx] rankTable[%p] identify[%s] "
            "hcom init failed.",
            HCCL_ERROR_CODE(ret), rankTableM, identify),
        ret);
    hcomInfo.isHcomInit = true;

    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "[HCCL_TRACE]hcom init by string success,take time [%lld]us, rankTableAddr[%p], rankNum[%u], "
        "rank[%u]",
        DURATION_US(TIME_NOW() - startut), rankTableM, hcomInfo.rankTable.rankNum, hcomInfo.params.rank);
    return HCCL_SUCCESS;
}

HcclResult GenerateRootInfo(HcclRootHandle& rootInfo)
{
    std::string identifier = "hccl_world_group";
    CHK_PRT_RET(
        (identifier.length() >= ROOTINFO_INDENTIFIER_MAX_LENGTH),
        HCCL_ERROR("[Setup][Server]rootinfo identifier len[%zu] is invalid.", identifier.length()), HCCL_E_INTERNAL);
    s32 sret
        = memcpy_s(&rootInfo.identifier[0], sizeof(rootInfo.identifier), identifier.c_str(), (identifier.length() + 1));
    CHK_PRT_RET(
        sret != EOK,
        HCCL_ERROR(
            "[Setup][Server]errNo[0x%016llx] memcpy failed. ret[%d], params:"
            "destMaxSize[%zu],count[%zu]",
            HCOM_ERROR_CODE(HCCL_E_MEMORY), sret, sizeof(rootInfo.identifier), (identifier.length() + 1)),
        HCCL_E_MEMORY);

    s32 sRet = strncpy_s(
        rootInfo.ip, sizeof(rootInfo.ip), GetExternalInputMasterInfo().serverIp.GetReadableIP(),
        strlen(GetExternalInputMasterInfo().serverIp.GetReadableIP()));
    CHK_PRT_RET(sRet != EOK, HCCL_ERROR("[Setup][Server]str copy fail. return[%d]", sRet), HCCL_E_INTERNAL);

    rootInfo.port = GetExternalInputMasterInfo().port;
    rootInfo.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    return HCCL_SUCCESS;
}

HcclResult HcomGenerteRanktable(std::string& rankTableM, std::string& rankId)
{
    s32 logicDevId = 0;
    u32 devPhyId = 0;
    CHK_RET(hrtGetDevice(&logicDevId));
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(logicDevId), devPhyId));

    // true代表感知白名单disable配置
    CHK_RET(HcclNetInit(NICDeployment::NIC_DEPLOYMENT_HOST, devPhyId, logicDevId, true));

    HcclIpAddress localHostIp;
    CHK_RET(GetLocalHostIP(localHostIp, devPhyId));
    HcclRootHandle rootHandle;
    CHK_RET(GenerateRootInfo(rootHandle));

    std::shared_ptr<TopoInfoDetect> topoDetectAgent;
    EXCEPTION_CATCH(topoDetectAgent = std::make_shared<TopoInfoDetect>(), return HCCL_E_PTR);
    std::shared_ptr<TopoInfoDetect> topoDetectServer;
    EXCEPTION_CATCH(topoDetectServer = std::make_shared<TopoInfoDetect>(), return HCCL_E_PTR);

    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    bool retryEnable = devType == DevType::DEV_TYPE_910_93
                       && (GetExternalInputInterServerRetryEnable() || GetExternalInputInterSuperPodRetryEnable());
    HCCL_INFO("[HcomGenerteRanktable] retryEnable is [%d]", retryEnable);
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();

    bool isRoot
        = (localHostIp == GetExternalInputMasterInfo().serverIp
           && logicDevId == static_cast<s32>(GetExternalInputMasterInfo().serverDeviceId));
    if (isRoot) {
        HcclResult ret
            = topoDetectServer->SetupServerByMasterInfo(localHostIp, GetExternalInputMasterInfo().port, rootHandle);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[%s][%s]%s failed, localHostIp[%s] and localhostPort[%u] ret[%u]", LOG_KEYWORDS_INIT_GROUP.c_str(),
                LOG_KEYWORDS_RANKTABLE_DETECT.c_str(), __func__, localHostIp.GetReadableAddress(),
                GetExternalInputMasterInfo().port, ret),
            ret);
    }

    CHK_PRT_RET(
        topoDetectAgent->SetupAgentByMasterInfo(localHostIp, rootHandle) != HCCL_SUCCESS,
        HCCL_ERROR("[Init][CommMasterInfo]setup topo detect error"), HCCL_E_INTERNAL);
    RankTable_t rankTable;
    CHK_PRT_RET(
        topoDetectAgent->GetClusterInfo(rankTable) != HCCL_SUCCESS,
        HCCL_ERROR("[Init][CommMasterInfo]GetClusterInfo error"), HCCL_E_INTERNAL);
    u32 rankIdNum = 0;
    CHK_PRT_RET(
        topoDetectAgent->GetRankId(rankIdNum) != HCCL_SUCCESS,
        HCCL_ERROR("[Init][CommMasterInfo]topoDetectAgent error"), HCCL_E_INTERNAL);

    CHK_RET(topoDetectAgent->TransformRankTableStr(rankTable, rankTableM));
    rankId = to_string(rankIdNum);
    CHK_PRT_RET(
        topoDetectAgent->WaitComplete(rootHandle) != HCCL_SUCCESS,
        HCCL_ERROR("[Init][CommMasterInfo]topoDetectAgent teardown fail"), HCCL_E_INTERNAL);

    CHK_PRT_RET(
        topoDetectAgent->GetAgentListenSocket(hcomInfo.params.commPortConfig) != HCCL_SUCCESS,
        HCCL_ERROR("[Init][CommMasterInfo]HcclGetCommListenSockets failed."), HCCL_E_INTERNAL);

    if (retryEnable) {
        hcomInfo.params.commConnections.isRoot = isRoot;
        if (isRoot) {
            hcomInfo.hcclCommTopoInfoDetectServer.insert({rootHandle.identifier, topoDetectServer});
            topoDetectServer->GetServerConnections(hcomInfo.params.commConnections.serverConnections);
        }
        hcomInfo.hcclCommTopoInfoDetectAgent.insert({rootHandle.identifier, topoDetectAgent});
        topoDetectAgent->GetAgentConnection(hcomInfo.params.commConnections.agentConnection);
    }

    return HCCL_SUCCESS;
}

HcclResult HcomInitByMasterInfo(
    const char* masterIp, const char* masterPort, const char* masterDeviceId, const char* rankSize, const char* rankIp,
    HcomInitConfig* initConfig)
{
    CHK_RET(SetMasterInfo(masterIp, masterPort, masterDeviceId, rankSize, rankIp));
    HcclUs startut = TIME_NOW();
    HcclResult ret = HCCL_SUCCESS;

    if (initConfig != nullptr) {
        DevType devType;
        CHK_RET(hrtGetDeviceType(devType));

        CHK_RET(HcomSetAlgorithm(initConfig->algo));
        CHK_RET(HcomSetExecTimeOut(initConfig->execTimeOut));
        if (devType != DevType::DEV_TYPE_910_93) {
            CHK_RET(HcomSetDeterministic(initConfig->deterministic));
        } else {
            HCCL_WARNING("ParserHcclDeterministic: device type is 910_93, use default setting");
        }
    }

    s32 logicDevId = 0;
    CHK_RET(hrtGetDevice(&logicDevId));
    // 读取rankTable文件到内存
    std::string rankTableM;
    std::string identify;
    HCCL_RUN_INFO(
        "Entry-HcomInitByMasterInfo:masterIp[%s], masterPort[%s], master device id[%s], rankSize[%s], rankIp[%s], "
        "deviceId[%d]",
        masterIp, masterPort, masterDeviceId, rankSize, rankIp, logicDevId);

    CHK_RET(InitExternalInput()); // 生成ranktable前需要提前感知部分配置
    CHK_RET(InitEnvConfig());
    ret = HcomGenerteRanktable(rankTableM, identify);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[HcomInitByMasterInfo]errNo[0x%016llx] masterIp[%s], masterPort[%s], "
            "masterDeviceId[%s] rankSize[%s] deviceId[%d] load rankTable error.",
            HCCL_ERROR_CODE(HCCL_E_INTERNAL), masterIp, masterPort, masterDeviceId, rankSize, logicDevId),
        HCCL_E_INTERNAL);
    CHK_RET(HcomCheckInitClusterInfo(rankTableM.c_str(), identify.c_str()));

    // 调用初始化接口
    ret = HcomInit(rankTableM.c_str(), identify.c_str());
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[HcomInitByMasterInfo]errNo[0x%016llx] identify[%s] "
            "hcom init failed.",
            HCCL_ERROR_CODE(ret), identify.c_str()),
        ret);
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    hcomInfo.isHcomInit = true;
    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "[HCCL_TRACE]hcom init by masterinfo success,take time [%lld]us, rankNum[%u], rank[%u], "
        "server[%s], device[%d]",
        DURATION_US(TIME_NOW() - startut), hcomInfo.rankTable.rankNum, hcomInfo.params.rank,
        hcomInfo.params.serverId.c_str(), hcomInfo.params.logicDevId);
    return HCCL_SUCCESS;
}

HcclResult HcomSetProfilingMode(HcomProfilingMode profilingMode, const char* profilingOption)
{
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    HCCL_INFO("Set profiling option[%s].", profilingOption);
    hcomInfo.params.profilingMode = profilingMode;
    hcomInfo.params.profilingOption = profilingOption;
    return HCCL_SUCCESS;
}

HcclResult HcomDestroyOneDeviceHeterog([[maybe_unused]] HcomInfo& hcomInfo) { return HCCL_SUCCESS; }

HcclResult HcomAllGather(
    const char* tag, void* inputPtr, void* outputPtr, u64 inputCount, HcclDataType dataType, const char* group,
    rtStream_t stream)
{
    HcclResult ret;
    HcclUs startut = TIME_NOW();
    uint64_t beginTime = hrtMsprofSysCycleTime();

    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDevice(&deviceLogicId));

    CHK_PRT_RET(inputCount == 0, HCCL_WARNING("input count is 0, return AllGather success"), HCCL_SUCCESS);
    // 参数合法性校验

    RPT_INPUT_ERR(
        inputPtr == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcomAllGather", "nullptr", "inputPtr", "non-null pointer"}));
    CHK_PTR_NULL(inputPtr);
    RPT_INPUT_ERR(
        outputPtr == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcomAllGather", "nullptr", "outputPtr", "non-null pointer"}));
    CHK_PTR_NULL(outputPtr);
    RPT_INPUT_ERR(
        stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcomAllGather", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);

    std::string strGroup = (group == nullptr) ? HCCL_WORLD_GROUP : group;
    s32 streamId = 0;
    ret = hrtGetStreamId(stream, streamId);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[AllGather][Result]Call hrtGetStreamId error[%d].", ret), HCCL_E_RUNTIME);
    /* 接口交互信息日志 */
    HCCL_USER_CRITICAL_LOG(
        "Entry-HcomAllGather:tag[%s], inputPtr[%p], outputPtr[%p], inputCount[%llu], dataType[%s], "
        "group[%s], streamId[%d], deviceLogicId[%d]",
        tag, inputPtr, outputPtr, inputCount, GetDataTypeEnumStr(dataType).c_str(), strGroup.c_str(), streamId,
        deviceLogicId);
    CHK_RET(PrintMemoryAttr(inputPtr));
    CHK_RET(PrintMemoryAttr(outputPtr));

    // HcomAllGatherV2
    HCCLV2_FUNC_RUN(HcomAllGatherV2(tag, inputPtr, outputPtr, inputCount, dataType, group, stream));

    CHK_RET(HcomCheckOpParam(tag, inputCount, dataType, group, stream));
    std::shared_ptr<hccl::hcclComm> hcclComm;
    CHK_RET(HcomGetCommByGroup(strGroup.c_str(), hcclComm));
    ret = hcclComm->AllGather(tag, inputPtr, outputPtr, inputCount, dataType, stream);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[AllGather][Result]errNo[0x%016llx] hcclComm AllGather error, "
            "tag[%s], input_ptr[%p], output_ptr[%p], count[%llu], data_type[%s]",
            HCOM_ERROR_CODE(ret), tag, inputPtr, outputPtr, inputCount, GetDataTypeEnumStr(dataType).c_str()),
        ret);
    CHK_RET(CallMsprofReportHostApi(hcclComm.get(), HcclCMDType::HCCL_CMD_ALLGATHER, beginTime, inputCount, dataType));
    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "hcom AllGather success,take time [%lld]us, tag[%s], input_ptr[%p], output_ptr[%p], "
        "count[%llu], data_type[%s]",
        DURATION_US(TIME_NOW() - startut), tag, inputPtr, outputPtr, inputCount, GetDataTypeEnumStr(dataType).c_str());

    return HCCL_SUCCESS;
}

HcclResult HcomAllGatherV(
    const char* tag, const void* sendBuf, u64 sendCount, const void* recvBuf, const void* recvCounts,
    const void* rdispls, HcclDataType dataType, const char* group, rtStream_t stream)
{
    HcclResult ret;
    HcclUs startut = TIME_NOW();
    uint64_t beginTime = hrtMsprofSysCycleTime();

    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDevice(&deviceLogicId));

    // 参数合法性校验
    RPT_INPUT_ERR(
        recvCounts == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
        std::vector<std::string>({"HcomAllGatherV", "nullptr", "recvCounts", "non-null pointer"}));
    CHK_PTR_NULL(recvCounts);
    RPT_INPUT_ERR(
        rdispls == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
        std::vector<std::string>({"HcomAllGatherV", "nullptr", "rdispls", "non-null pointer"}));
    CHK_PTR_NULL(rdispls);
    RPT_INPUT_ERR(
        stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
        std::vector<std::string>({"HcomAllGatherV", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);
    if (UNLIKELY(sendCount > 0 && sendBuf == nullptr)) {
        RPT_INPUT_ERR(
            true, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
            std::vector<std::string>({"HcomAllGatherV", "nullptr", "sendBuf", "non-null pointer"}));
        CHK_PTR_NULL(sendBuf);
    }

    std::string strGroup = (group == nullptr) ? HCCL_WORLD_GROUP : group;
    s32 streamId = 0;
    ret = hrtGetStreamId(stream, streamId);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[AllGatherV][Result]Call hrtGetStreamId error[%d].", ret), HCCL_E_RUNTIME);
    /* 接口交互信息日志 */
    HCCL_USER_CRITICAL_LOG(
        "Entry-HcomAllGatherV:tag[%s], inputPtr[%p], outputPtr[%p], sendCount[%llu], dataType[%s], "
        "recvCounts[%p], rdispls[%p], group[%s], streamId[%d], deviceLogicId[%d]",
        tag, sendBuf, recvBuf, sendCount, GetDataTypeEnumStr(dataType).c_str(), recvCounts, rdispls, strGroup.c_str(),
        streamId, deviceLogicId);
    if (sendBuf != nullptr) {
        CHK_RET(PrintMemoryAttr(sendBuf));
    }
    if (recvBuf != nullptr) {
        CHK_RET(PrintMemoryAttr(recvBuf));
    }
    HCCLV2_FUNC_RUN(HcomAllGatherVV2(
        tag, const_cast<void*>(sendBuf), sendCount, const_cast<void*>(recvBuf), const_cast<void*>(recvCounts),
        const_cast<void*>(rdispls), dataType, group, stream));
    CHK_RET(HcomCheckOpParam(tag, 0, dataType, group, stream));
    std::shared_ptr<hccl::hcclComm> hcclComm;
    CHK_RET(HcomGetCommByGroup(strGroup.c_str(), hcclComm));
    ret = hcclComm->AllGatherV(tag, sendBuf, sendCount, recvBuf, recvCounts, rdispls, dataType, stream);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[AllGatherV][Result]errNo[0x%016llx] hcclComm AllGatherV error, "
            "tag[%s], input_ptr[%p], output_ptr[%p], count[%llu], data_type[%s]",
            HCOM_ERROR_CODE(ret), tag, sendBuf, recvBuf, sendCount, GetDataTypeEnumStr(dataType).c_str()),
        ret);
    CHK_RET(CallMsprofReportHostApi(hcclComm.get(), HcclCMDType::HCCL_CMD_ALLGATHER_V, beginTime, sendCount, dataType));
    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "hcom AllGatherv success,take time [%lld]us, tag[%s], input_ptr[%p], output_ptr[%p], "
        "count[%llu], data_type[%s]",
        DURATION_US(TIME_NOW() - startut), tag, sendBuf, recvBuf, sendCount, GetDataTypeEnumStr(dataType).c_str());

    return HCCL_SUCCESS;
}

HcclResult HcomGetInitStatus(bool* initiated)
{
    CHK_PTR_NULL(initiated);
    HCCLV2_FUNC_RUN(HcomGetInitStatusV2(*initiated));
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    *initiated = !(hcomInfo.pComm == nullptr);

    HCCL_INFO("Get Hcom Init Status: [%d]", *initiated);
    return HCCL_SUCCESS;
}

HcclResult HcomAllReduce(
    const char* tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op,
    const char* group, rtStream_t stream)
{
    HcclUs startut = TIME_NOW();
    uint64_t beginTime = hrtMsprofSysCycleTime();

    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDevice(&deviceLogicId));

    CHK_PRT_RET(count == 0, HCCL_WARNING("input count is 0, return AllReduce success"), HCCL_SUCCESS);
    // 入参合法性校验

    RPT_INPUT_ERR(
        inputPtr == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcomAllReduce", "nullptr", "inputPtr", "non-null pointer"}));
    CHK_PTR_NULL(inputPtr);
    RPT_INPUT_ERR(
        outputPtr == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcomAllReduce", "nullptr", "outputPtr", "non-null pointer"}));
    CHK_PTR_NULL(outputPtr);
    RPT_INPUT_ERR(
        stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcomAllReduce", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);

    s32 streamId = 0;
    CHK_RET(HcomCheckReductionOp("HcomAllReduce", op));
    CHK_RET(hrtGetStreamId(stream, streamId));

    std::string strGroup = (group == nullptr) ? HCCL_WORLD_GROUP : group;
    /* 接口交互信息日志 */
    HCCL_USER_CRITICAL_LOG(
        "Entry-HcomAllReduce:tag[%s], inputPtr[%p], outputPtr[%p], count[%llu], dataType[%s], op[%s], "
        "group[%s], streamId[%d], deviceLogicId[%d]",
        tag, inputPtr, outputPtr, count, GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str(),
        strGroup.c_str(), streamId, deviceLogicId);
    CHK_RET(PrintMemoryAttr(inputPtr));
    CHK_RET(PrintMemoryAttr(outputPtr));

    // HcomAllReduceV2
    HCCLV2_FUNC_RUN(HcomAllReduceV2(tag, inputPtr, outputPtr, count, dataType, op, group, stream));
    CHK_RET(HcomCheckOpParam(tag, count, dataType, group, stream));
    std::shared_ptr<hccl::hcclComm> hcclComm;
    CHK_RET(HcomGetCommByGroup(strGroup.c_str(), hcclComm));
    /* 入参的正确性由HCCL确保 */
    HcclResult ret = hcclComm->AllReduce(tag, inputPtr, outputPtr, count, dataType, op, stream);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[AllReduce][Result]errNo[0x%016llx] hcclComm AllReduce error, "
            "tag[%s], input_ptr[%p], output_ptr[%p], count[%llu], data_type[%s], op[%s]",
            HCOM_ERROR_CODE(ret), tag, inputPtr, outputPtr, count, GetDataTypeEnumStr(dataType).c_str(),
            GetReduceOpEnumStr(op).c_str()),
        ret);
    CHK_RET(CallMsprofReportHostApi(hcclComm.get(), HcclCMDType::HCCL_CMD_ALLREDUCE, beginTime, count, dataType));
    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "hcom AllReduce success,take time [%lld]us, tag[%s], input_ptr[%p], output_ptr[%p], "
        "count[%llu], data_type[%s], op[%s]",
        DURATION_US(TIME_NOW() - startut), tag, inputPtr, outputPtr, count, GetDataTypeEnumStr(dataType).c_str(),
        GetReduceOpEnumStr(op).c_str());

    return HCCL_SUCCESS;
}

HcclResult HcomBroadcast(
    const char* tag, void* ptr, u64 count, HcclDataType dataType, u32 root, const char* group, rtStream_t stream)
{
    HcclUs startut = TIME_NOW();
    uint64_t beginTime = hrtMsprofSysCycleTime();

    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDevice(&deviceLogicId));

    CHK_PRT_RET(count == 0, HCCL_WARNING("input count is 0, return BroadCast success"), HCCL_SUCCESS);
    // 入参合法性校验
    RPT_INPUT_ERR(
        ptr == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcomBroadcast", "nullptr", "ptr", "non-null pointer"}));
    CHK_PTR_NULL(ptr);
    RPT_INPUT_ERR(
        stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcomBroadcast", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);

    s32 streamId = 0;
    CHK_RET(hrtGetStreamId(stream, streamId));
    std::string strGroup = (group == nullptr) ? HCCL_WORLD_GROUP : group;
    /* 接口交互信息日志 */
    HCCL_USER_CRITICAL_LOG(
        "Entry-HcomBroadcast:tag[%s], ptr[%p], count[%llu], dataType[%s], root[%u], "
        "group[%s], streamId[%d], deviceLogicId[%d]",
        tag, ptr, count, GetDataTypeEnumStr(dataType).c_str(), root, strGroup.c_str(), streamId, deviceLogicId);
    CHK_RET(PrintMemoryAttr(ptr));

    // HcomBroadcastV2
    HCCLV2_FUNC_RUN(HcomBroadcastV2(tag, ptr, count, dataType, root, group, stream));
    CHK_RET(HcomCheckOpParam(tag, count, dataType, group, stream));
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    CHK_RET(HcomCheckUserRank(hcomInfo.params.totalRanks, root));
    std::shared_ptr<hccl::hcclComm> hcclComm;
    CHK_RET(HcomGetCommByGroup(strGroup.c_str(), hcclComm));
    u32 aivCoreLimit = 0;
    CHK_RET(hcclComm->GetNumBlocks(aivCoreLimit));
    /* 入参的正确性由HCCL确保 */
    HcclResult ret = hcclComm->Broadcast(tag, ptr, count, dataType, root, stream);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Broadcast][Result]errNo[0x%016llx] hcclComm BroadCast error,tag[%s], input_ptr[%p],"
            "count[%llu], data_type[%s], root[%u]",
            HCOM_ERROR_CODE(ret), tag, ptr, count, GetDataTypeEnumStr(dataType).c_str(), root),
        ret);
    CHK_RET(CallMsprofReportHostApi(hcclComm.get(), HcclCMDType::HCCL_CMD_BROADCAST, beginTime, count, dataType));
    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "hcom BroadCast success,take time [%lld]us,tag[%s], input_ptr[%p], count[%llu], data_type[%s], "
        "root[%u]",
        DURATION_US(TIME_NOW() - startut), tag, ptr, count, GetDataTypeEnumStr(dataType).c_str(), root);

    return HCCL_SUCCESS;
}

HcclResult HcomReduce(
    const char* tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op, u32 root,
    const char* group, rtStream_t stream)
{
    HcclUs startut = TIME_NOW();
    uint64_t beginTime = hrtMsprofSysCycleTime();

    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDevice(&deviceLogicId));

    CHK_PRT_RET(count == 0, HCCL_WARNING("input count is 0, return Reduce success"), HCCL_SUCCESS);

    // 入参合法性校验
    RPT_INPUT_ERR(
        tag == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcomReduce", "nullptr", "tag", "non-null pointer"}));
    CHK_PTR_NULL(tag);
    RPT_INPUT_ERR(
        inputPtr == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcomReduce", "nullptr", "inputPtr", "non-null pointer"}));
    CHK_PTR_NULL(inputPtr);
    RPT_INPUT_ERR(
        outputPtr == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcomReduce", "nullptr", "outputPtr", "non-null pointer"}));
    CHK_PTR_NULL(outputPtr);
    RPT_INPUT_ERR(
        stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcomReduce", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);
    CHK_RET(HcomCheckReductionOp("HcomReduce", op));

    s32 streamId = 0;
    CHK_RET(hrtGetStreamId(stream, streamId));
    std::string strGroup = (group == nullptr) ? HCCL_WORLD_GROUP : group;
    /* 接口交互信息日志 */
    HCCL_USER_CRITICAL_LOG(
        "Entry-HcomReduce:tag[%s], inputPtr[%p], outputPtr[%p], count[%llu], dataType[%s], op[%s], "
        "root[%u], group[%s], streamId[%d], deviceLogicId[%d]",
        tag, inputPtr, outputPtr, count, GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str(), root,
        strGroup.c_str(), streamId, deviceLogicId);
    CHK_RET(PrintMemoryAttr(inputPtr));
    CHK_RET(PrintMemoryAttr(outputPtr));

    // HcomReduceV2
    HCCLV2_FUNC_RUN(HcomReduceV2(tag, inputPtr, outputPtr, count, dataType, op, root, group, stream));
    CHK_RET(HcomCheckOpParam(tag, count, dataType, group, stream));
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    CHK_PRT_RET(
        hcomInfo.pComm == nullptr,
        HCCL_ERROR("[Reduce][Result]hcomInfo.pComm is null, "
                   "please check if the initialize process is called."),
        HCCL_E_PTR);
    CHK_RET(HcomCheckUserRank(hcomInfo.params.totalRanks, root));

    std::shared_ptr<hccl::hcclComm> hcclComm;
    CHK_RET(HcomGetCommByGroup(group, hcclComm));
    /* 入参的正确性由HCCL确保 */
    HcclResult ret = hcclComm->Reduce(tag, inputPtr, outputPtr, count, dataType, op, root, stream);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Reduce][Result]errNo[0x%016llx] hcclComm Reduce error, tag[%s], "
            "input_ptr[%p], output_ptr[%p], count[%llu], data_type[%s], op[%s], root[%u]",
            HCOM_ERROR_CODE(ret), tag, inputPtr, outputPtr, count, GetDataTypeEnumStr(dataType).c_str(),
            GetReduceOpEnumStr(op).c_str(), root),
        ret);
    CHK_RET(CallMsprofReportHostApi(hcclComm.get(), HcclCMDType::HCCL_CMD_REDUCE, beginTime, count, dataType));
    HcclUs endut = TIME_NOW();
    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "hcom Reduce success, take time [%lld]us, tag[%s], input_ptr[%p], output_ptr[%p], count[%llu], "
        "data_type[%s], op[%s], root[%u]",
        DURATION_US(endut - startut), tag, inputPtr, outputPtr, count, GetDataTypeEnumStr(dataType).c_str(),
        GetReduceOpEnumStr(op).c_str(), root);

    return HCCL_SUCCESS;
}

HcclResult HcomReduceScatter(
    const char* tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op,
    const char* group, rtStream_t stream)
{
    HcclUs startut = TIME_NOW();
    uint64_t beginTime = hrtMsprofSysCycleTime();

    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDevice(&deviceLogicId));

    CHK_PRT_RET(count == 0, HCCL_WARNING("input count is 0, return ReduceScatter success"), HCCL_SUCCESS);
    // 入参合法性校验

    RPT_INPUT_ERR(
        inputPtr == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcomReduceScatter", "nullptr", "inputPtr", "non-null pointer"}));
    CHK_PTR_NULL(inputPtr);
    RPT_INPUT_ERR(
        outputPtr == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcomReduceScatter", "nullptr", "outputPtr", "non-null pointer"}));
    CHK_PTR_NULL(outputPtr);
    RPT_INPUT_ERR(
        stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcomReduceScatter", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);
    CHK_RET(HcomCheckReductionOp("HcomReduceScatter", op));

    s32 streamId = 0;
    CHK_RET(hrtGetStreamId(stream, streamId));
    std::string strGroup = (group == nullptr) ? HCCL_WORLD_GROUP : group;
    /* 接口交互信息日志 */
    HCCL_USER_CRITICAL_LOG(
        "Entry-HcomReduceScatter:tag[%s], inputPtr[%p], outputPtr[%p], count[%llu], dataType[%s], op[%s], "
        "group[%s], streamId[%d], deviceLogicId[%d]",
        tag, inputPtr, outputPtr, count, GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str(),
        strGroup.c_str(), streamId, deviceLogicId);
    CHK_RET(PrintMemoryAttr(inputPtr));
    CHK_RET(PrintMemoryAttr(outputPtr));

    HCCLV2_FUNC_RUN(HcomReduceScatterV2(tag, inputPtr, outputPtr, count, dataType, op, group, stream));
    CHK_RET(HcomCheckOpParam(tag, count, dataType, group, stream));
    std::shared_ptr<hccl::hcclComm> hcclComm;
    CHK_RET(HcomGetCommByGroup(strGroup.c_str(), hcclComm));
    /* 入参的正确性由HCCL确保 */
    HcclResult ret = hcclComm->ReduceScatter(tag, inputPtr, outputPtr, count, dataType, op, stream);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[ReduceScatter][Result]errNo[0x%016llx] hcclComm ReduceScatter "
                    "error, tag[%s], input_ptr[%p], output_ptr[%p], count[%llu], data_type[%s], op[%s]",
                    HCOM_ERROR_CODE(ret), tag, inputPtr, outputPtr, count, GetDataTypeEnumStr(dataType).c_str(),
                    GetReduceOpEnumStr(op).c_str());
                , ret);
    CHK_RET(CallMsprofReportHostApi(hcclComm.get(), HcclCMDType::HCCL_CMD_REDUCE_SCATTER, beginTime, count, dataType));
    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "hcom reduceScatter success, take time [%lld]us, tag[%s], input_ptr[%p], output_ptr[%p], count[%llu], "
        "data_type[%s], op[%s]",
        DURATION_US(TIME_NOW() - startut), tag, inputPtr, outputPtr, count, GetDataTypeEnumStr(dataType).c_str(),
        GetReduceOpEnumStr(op).c_str());

    return HCCL_SUCCESS;
}

HcclResult HcomReduceScatterV(
    const char* tag, void* sendBuf, const void* sendCounts, const void* sdispls, void* recvBuf, u64 recvCount,
    HcclDataType dataType, HcclReduceOp op, const char* group, rtStream_t stream)
{
    HcclUs startut = TIME_NOW();
    uint64_t beginTime = hrtMsprofSysCycleTime();

    // 入参合法性校验
    RPT_INPUT_ERR(
        stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
        std::vector<std::string>({"HcomReduceScatterV", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);
    RPT_INPUT_ERR(
        sendCounts == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
        std::vector<std::string>({"HcomReduceScatterV", "nullptr", "sendCounts", "non-null pointer"}));
    CHK_PTR_NULL(sendCounts);
    RPT_INPUT_ERR(
        sdispls == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
        std::vector<std::string>({"HcomReduceScatterV", "nullptr", "sdispls", "non-null pointer"}));
    CHK_PTR_NULL(sdispls);
    if (UNLIKELY(recvCount > 0 && recvBuf == nullptr)) {
        RPT_INPUT_ERR(
            true, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
            std::vector<std::string>({"HcomReduceScatterV", "nullptr", "recvBuf", "non-null pointer"}));
        CHK_PTR_NULL(recvBuf);
    }

    s32 streamId = 0;
    CHK_RET(hrtGetStreamId(stream, streamId));
    std::string strGroup = (group == nullptr) ? HCCL_WORLD_GROUP : group;
    /* 接口交互信息日志 */
    HCCL_RUN_INFO(
        "Entry-HcomReduceScatterV:tag[%s], inputPtr[%p], outputPtr[%p], count[%llu], dataType[%s], op[%s], "
        "group[%s], streamId[%d]",
        tag, sendBuf, recvBuf, recvCount, GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str(),
        strGroup.c_str(), streamId);
    if (sendBuf != nullptr) {
        CHK_RET(PrintMemoryAttr(sendBuf));
    }
    if (recvBuf != nullptr) {
        CHK_RET(PrintMemoryAttr(recvBuf));
    }

    HCCLV2_FUNC_RUN(HcomReduceScatterVV2(
        tag, sendBuf, const_cast<void*>(sendCounts), const_cast<void*>(sdispls), recvBuf, recvCount, dataType, op,
        group, stream));
    CHK_RET(HcomCheckOpParam(tag, 0, dataType, group, stream));
    std::shared_ptr<hccl::hcclComm> hcclComm;
    CHK_RET(HcomGetCommByGroup(strGroup.c_str(), hcclComm));
    /* 入参的正确性由HCCL确保 */
    HcclResult ret
        = hcclComm->ReduceScatterV(tag, sendBuf, sendCounts, sdispls, recvBuf, recvCount, dataType, op, stream);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[ReduceScatterV][Result]errNo[0x%016llx] hcclComm ReduceScatter "
                    "error, tag[%s], input_ptr[%p], output_ptr[%p], count[%llu], data_type[%s], op[%s]",
                    HCOM_ERROR_CODE(ret), tag, sendBuf, recvBuf, recvCount, GetDataTypeEnumStr(dataType).c_str(),
                    GetReduceOpEnumStr(op).c_str());
                , ret);
    CHK_RET(CallMsprofReportHostApi(hcclComm.get(), HcclCMDType::HCCL_CMD_REDUCE_SCATTER, beginTime, 0, dataType));
    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "hcom ReduceScatterv success, take time [%lld]us, tag[%s], sendBuf[%p], recvBuf[%p], count[%llu], "
        "data_type[%s], op[%s]",
        DURATION_US(TIME_NOW() - startut), tag, sendBuf, recvBuf, recvCount, GetDataTypeEnumStr(dataType).c_str(),
        GetReduceOpEnumStr(op).c_str());

    return HCCL_SUCCESS;
}

/*
 * 点对点发送接口: 需要对应的hcom_receive执行才会实际发送。先分片，条件满足之后改为不分片
 * 发送端需要接收端准备好才会发送
 */
HcclResult HcomSend(
    const char* tag, void* inputPtr, u64 count, HcclDataType dataType, u32 destRank, u32 srTag, const char* group,
    rtStream_t stream)
{
    HcclUs startut = TIME_NOW();
    uint64_t beginTime = hrtMsprofSysCycleTime();

    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDevice(&deviceLogicId));

    CHK_PRT_RET(count == 0, HCCL_WARNING("input count is 0, return send success"), HCCL_SUCCESS);

    // 入参合法性校验
    RPT_INPUT_ERR(
        inputPtr == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcomSend", "nullptr", "inputPtr", "non-null pointer"}));
    CHK_PTR_NULL(inputPtr);
    RPT_INPUT_ERR(
        stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcomSend", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);

    s32 streamId = 0;
    CHK_RET(hrtGetStreamId(stream, streamId));
    std::string strGroup = (group == nullptr) ? HCCL_WORLD_GROUP : group;
    /* 接口交互信息日志 */
    HCCL_USER_CRITICAL_LOG(
        "Entry-HcomSend:tag[%s], inputPtr[%p], count[%llu], dataType[%s], destRank[%u], srTag[%u], "
        "group[%s], streamId[%d], deviceLogicId[%d]",
        tag, inputPtr, count, GetDataTypeEnumStr(dataType).c_str(), destRank, srTag, strGroup.c_str(), streamId,
        deviceLogicId);
    CHK_RET(PrintMemoryAttr(inputPtr));

    // HcomSendV2
    HCCLV2_FUNC_RUN(HcomSendV2(tag, inputPtr, count, dataType, destRank, srTag, group, stream));
    CHK_RET(HcomCheckOpParam(tag, count, dataType, group, stream));
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    CHK_RET(HcomCheckUserRank(hcomInfo.params.totalRanks, destRank));

    std::shared_ptr<hccl::hcclComm> hcclComm;
    CHK_RET(HcomGetCommByGroup(strGroup.c_str(), hcclComm));

    u32 localGroupRank = INVALID_VALUE_RANKID;
    CHK_RET(hcclComm->GetGroupRank(localGroupRank));
    /* 调用HCCL的send, 入参的正确性由HCCL确保 */
    HcclResult ret = hcclComm->send(tag, inputPtr, count, dataType, destRank, stream, srTag, localGroupRank);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Send][Result]errNo[0x%016llx] hcclComm send error, tag[%s], "
            "inputPtr[%p], count[%llu], dataType[%s], destRank[%u], group[%s]",
            HCOM_ERROR_CODE(ret), tag, inputPtr, count, GetDataTypeEnumStr(dataType).c_str(), destRank,
            strGroup.c_str()),
        ret);
    CHK_RET(CallMsprofReportHostApi(hcclComm.get(), HcclCMDType::HCCL_CMD_SEND, beginTime, count, dataType));
    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "hcom send success,time[%lld]us,tag[%s],inputPtr[%p],count[%llu],dataType[%s],destRank[%u],"
        "srTag[%u],localGroupRank[%u]",
        DURATION_US(TIME_NOW() - startut), tag, inputPtr, count, GetDataTypeEnumStr(dataType).c_str(), destRank, srTag,
        localGroupRank);

    return HCCL_SUCCESS;
}

/*
 * 点对点接收接口: 需要对应的hcom_receive执行才会实际发送。先分片，条件满足之后改为不分片
 * 发送端需要接收端准备好才会发送
 */
HcclResult HcomReceive(
    const char* tag, void* outputPtr, u64 count, HcclDataType dataType, u32 srcRank, u32 srTag, const char* group,
    rtStream_t stream)
{
    HcclUs startut = TIME_NOW();
    uint64_t beginTime = hrtMsprofSysCycleTime();

    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDevice(&deviceLogicId));
    CHK_PRT_RET(count == 0, HCCL_WARNING("input count is 0, return receive success"), HCCL_SUCCESS);

    // 入参合法性校验
    RPT_INPUT_ERR(
        outputPtr == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcomReceive", "nullptr", "outputPtr", "non-null pointer"}));
    CHK_PTR_NULL(outputPtr);
    RPT_INPUT_ERR(
        stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcomReceive", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);

    s32 streamId = 0;
    CHK_RET(hrtGetStreamId(stream, streamId));
    std::string strGroup = (group == nullptr) ? HCCL_WORLD_GROUP : group;
    /* 接口交互信息日志 */
    HCCL_USER_CRITICAL_LOG(
        "Entry-HcomReceive:tag[%s], outputPtr[%p], count[%llu], dataType[%s], srcRank[%u], srTag[%u],"
        "group[%s], streamId[%d], deviceLogicId[%d]",
        tag, outputPtr, count, GetDataTypeEnumStr(dataType).c_str(), srcRank, srTag, strGroup.c_str(), streamId,
        deviceLogicId);
    CHK_RET(PrintMemoryAttr(outputPtr));

    // HcomReceiveV2
    HCCLV2_FUNC_RUN(HcomReceiveV2(tag, outputPtr, count, dataType, srcRank, srTag, group, stream));
    CHK_RET(HcomCheckOpParam(tag, count, dataType, group, stream));
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    CHK_RET(HcomCheckUserRank(hcomInfo.params.totalRanks, srcRank));

    std::shared_ptr<hccl::hcclComm> hcclComm;
    HcclResult ret = HcomGetCommByGroup(strGroup.c_str(), hcclComm);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Receive][Result]errNo[0x%016llx] get comm by group[%s] error", HCOM_ERROR_CODE(ret), strGroup.c_str()),
        ret);

    u32 localGroupRank = INVALID_VALUE_RANKID;
    CHK_RET(hcclComm->GetGroupRank(localGroupRank));
    ret = hcclComm->receive(tag, outputPtr, count, dataType, srcRank, stream, srTag, localGroupRank);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Receive][Result]errNo[0x%016llx] hcclComm receive error,tag[%s], "
            "outputPtr[%p], count[%llu], dataType[%s], srcRank[%u], group[%s]",
            HCOM_ERROR_CODE(ret), tag, outputPtr, count, GetDataTypeEnumStr(dataType).c_str(), srcRank,
            strGroup.c_str()),
        ret);

    CHK_RET(CallMsprofReportHostApi(hcclComm.get(), HcclCMDType::HCCL_CMD_RECEIVE, beginTime, count, dataType));
    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "hcom receive success,time[%lld]us,tag[%s],outputPtr[%p],count[%llu],dataType[%s],srcRank[%u],"
        "srTag[%u], localGroupRank[%u]",
        DURATION_US(TIME_NOW() - startut), tag, outputPtr, count, GetDataTypeEnumStr(dataType).c_str(), srcRank, srTag,
        localGroupRank);

    return HCCL_SUCCESS;
}

HcclResult HcclCommGraphAllGather(
    const char* tag, void* inputPtr, void* outputPtr, u64 inputCount, HcclDataType dataType, s64 opBaseHcom,
    rtStream_t stream)
{
    HcclResult ret;
    HcclUs startut = TIME_NOW();
    uint64_t beginTime = hrtMsprofSysCycleTime();
    CHK_PRT_RET(inputCount == 0, HCCL_WARNING("input count is 0, return AllGather success"), HCCL_SUCCESS);
    // 参数合法性校验
    RPT_INPUT_ERR(
        inputPtr == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclCommGraphAllGather", "nullptr", "inputPtr", "non-null pointer"}));
    CHK_PTR_NULL(inputPtr);
    RPT_INPUT_ERR(
        outputPtr == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclCommGraphAllGather", "nullptr", "outputPtr", "non-null pointer"}));
    CHK_PTR_NULL(outputPtr);
    RPT_INPUT_ERR(
        stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclCommGraphAllGather", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);

    s32 streamId = 0;
    ret = hrtGetStreamId(stream, streamId);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[AllGather][Result]Call hrtGetStreamId error[%d].", ret), HCCL_E_RUNTIME);
    /* 接口交互信息日志 */
    HCCL_RUN_INFO(
        "Entry-HcclCommGraphAllGather:tag[%s], inputPtr[%p], outputPtr[%p], inputCount[%llu], dataType[%s], "
        "opBaseHcom[%lld] streamId[%d]",
        tag, inputPtr, outputPtr, inputCount, GetDataTypeEnumStr(dataType).c_str(), opBaseHcom, streamId);
    CHK_RET(PrintMemoryAttr(inputPtr));
    CHK_RET(PrintMemoryAttr(outputPtr));

    // HcclCommGraphAllGatherV2
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        hccl::hcclComm* hcclComm = reinterpret_cast<hccl::hcclComm*>(opBaseHcom);
        CHK_PTR_NULL(hcclComm);
        HcclComm commV2 = hcclComm->GetCommunicatorV2();
        CHK_PTR_NULL(commV2);
        opBaseHcom = reinterpret_cast<s64>(commV2);
        CHK_RET(HcclCommGraphAllGatherV2(tag, inputPtr, outputPtr, inputCount, dataType, opBaseHcom, stream));
        return HCCL_SUCCESS;
    }());
    CHK_RET(HcomCheckOpParam(tag, inputCount, dataType, stream));
    hccl::hcclComm* hcclComm = reinterpret_cast<hccl::hcclComm*>(opBaseHcom);
    CHK_PTR_NULL(hcclComm);
    CHK_RET(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB));
    ret = hcclComm->AllGather(tag, inputPtr, outputPtr, inputCount, dataType, stream);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[AllGather][Result]errNo[0x%016llx] hcclComm AllGather error, "
            "tag[%s], input_ptr[%p], output_ptr[%p], count[%llu], data_type[%s]",
            HCOM_ERROR_CODE(ret), tag, inputPtr, outputPtr, inputCount, GetDataTypeEnumStr(dataType).c_str()),
        ret);
    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_ALLGATHER, beginTime, inputCount, dataType));
    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "HcclCommGraphAllGather success,take time [%lld]us, tag[%s], input_ptr[%p], output_ptr[%p], "
        "count[%llu], data_type[%s], opBaseHcom[%lld]",
        DURATION_US(TIME_NOW() - startut), tag, inputPtr, outputPtr, inputCount, GetDataTypeEnumStr(dataType).c_str(),
        opBaseHcom);

    return HCCL_SUCCESS;
}

HcclResult HcclCommGraphAllReduce(
    const char* tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op, s64 opBaseHcom,
    rtStream_t stream)
{
    HcclUs startut = TIME_NOW();
    uint64_t beginTime = hrtMsprofSysCycleTime();
    CHK_PRT_RET(count == 0, HCCL_WARNING("input count is 0, return AllReduce success"), HCCL_SUCCESS);
    // 入参合法性校验
    RPT_INPUT_ERR(
        inputPtr == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclCommGraphAllReduce", "nullptr", "inputPtr", "non-null pointer"}));
    CHK_PTR_NULL(inputPtr);
    RPT_INPUT_ERR(
        outputPtr == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclCommGraphAllReduce", "nullptr", "outputPtr", "non-null pointer"}));
    CHK_PTR_NULL(outputPtr);
    RPT_INPUT_ERR(
        stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclCommGraphAllReduce", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);
    CHK_RET(HcomCheckReductionOp("HcclCommGraphAllReduce", op));
    s32 streamId = 0;
    CHK_RET(hrtGetStreamId(stream, streamId));

    /* 接口交互信息日志 */
    HCCL_RUN_INFO(
        "Entry-HcclCommGraphAllReduce:tag[%s], inputPtr[%p], outputPtr[%p], count[%llu], dataType[%s], "
        "op[%s], opBaseHcom[%lld], streamId[%d]",
        tag, inputPtr, outputPtr, count, GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str(),
        opBaseHcom, streamId);
    CHK_RET(PrintMemoryAttr(inputPtr));
    CHK_RET(PrintMemoryAttr(outputPtr));

    // HcomGraphAllReduceV2
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        hccl::hcclComm* hcclComm = reinterpret_cast<hccl::hcclComm*>(opBaseHcom);
        CHK_PTR_NULL(hcclComm);
        HcclComm commV2 = hcclComm->GetCommunicatorV2();
        CHK_PTR_NULL(commV2);
        opBaseHcom = reinterpret_cast<s64>(commV2);
        CHK_RET(HcomGraphAllReduceV2(tag, inputPtr, outputPtr, count, dataType, op, opBaseHcom, stream));
        return HCCL_SUCCESS;
    }());
    CHK_RET(HcomCheckOpParam(tag, count, dataType, stream));
    hccl::hcclComm* hcclComm = reinterpret_cast<hccl::hcclComm*>(opBaseHcom);
    CHK_PTR_NULL(hcclComm);
    CHK_RET(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB));
    /* 入参的正确性由HCCL确保 */
    HcclResult ret = hcclComm->AllReduce(tag, inputPtr, outputPtr, count, dataType, op, stream);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[AllReduce][Result]errNo[0x%016llx] hcclComm AllReduce error, "
            "tag[%s], input_ptr[%p], output_ptr[%p], count[%llu], data_type[%s], op[%s]",
            HCOM_ERROR_CODE(ret), tag, inputPtr, outputPtr, count, GetDataTypeEnumStr(dataType).c_str(),
            GetReduceOpEnumStr(op).c_str()),
        ret);
    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_ALLREDUCE, beginTime, count, dataType));
    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "HcclCommGraphAllReduce success,take time [%lld]us, tag[%s], input_ptr[%p], output_ptr[%p], "
        "count[%llu], data_type[%s], op[%s], opBaseHcom[%lld]",
        DURATION_US(TIME_NOW() - startut), tag, inputPtr, outputPtr, count, GetDataTypeEnumStr(dataType).c_str(),
        GetReduceOpEnumStr(op).c_str(), opBaseHcom);

    return HCCL_SUCCESS;
}

HcclResult HcclCommGraphReduce(
    const char* tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op, u32 root,
    s64 opBaseHcom, rtStream_t stream)
{
    HcclUs startut = TIME_NOW();
    uint64_t beginTime = hrtMsprofSysCycleTime();
    CHK_PRT_RET(count == 0, HCCL_WARNING("input count is 0, return Reduce success"), HCCL_SUCCESS);

    // 入参合法性校验
    RPT_INPUT_ERR(
        tag == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclCommGraphReduce", "nullptr", "tag", "non-null pointer"}));
    CHK_PTR_NULL(tag);
    RPT_INPUT_ERR(
        inputPtr == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclCommGraphReduce", "nullptr", "inputPtr", "non-null pointer"}));
    CHK_PTR_NULL(inputPtr);
    RPT_INPUT_ERR(
        outputPtr == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclCommGraphReduce", "nullptr", "outputPtr", "non-null pointer"}));
    CHK_PTR_NULL(outputPtr);
    RPT_INPUT_ERR(
        stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclCommGraphReduce", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);
    CHK_RET(HcomCheckReductionOp("HcclCommGraphReduce", op));
    u32 totalRanks = 0;
    CHK_RET(HcclCommGraphGetRankSize(opBaseHcom, &totalRanks));
    CHK_RET(HcomCheckUserRank(totalRanks, root));
    s32 streamId = 0;
    CHK_RET(hrtGetStreamId(stream, streamId));
    /* 接口交互信息日志 */
    HCCL_RUN_INFO(
        "Entry-HcclCommGraphReduce:tag[%s], inputPtr[%p], outputPtr[%p], count[%llu], dataType[%s], op[%s], "
        "root[%u], opBaseHcom[%lld], streamId[%d]",
        tag, inputPtr, outputPtr, count, GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str(), root,
        opBaseHcom, streamId);
    CHK_RET(PrintMemoryAttr(inputPtr));
    CHK_RET(PrintMemoryAttr(outputPtr));

    // HcomGraphReduceV2
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        hccl::hcclComm* hcclComm = reinterpret_cast<hccl::hcclComm*>(opBaseHcom);
        CHK_PTR_NULL(hcclComm);
        HcclComm commV2 = hcclComm->GetCommunicatorV2();
        CHK_PTR_NULL(commV2);
        opBaseHcom = reinterpret_cast<s64>(commV2);
        CHK_RET(HcomGraphReduceV2(tag, inputPtr, outputPtr, count, dataType, op, root, opBaseHcom, stream));
        return HCCL_SUCCESS;
    }());
    CHK_RET(HcomCheckOpParam(tag, count, dataType));
    hccl::hcclComm* hcclComm = reinterpret_cast<hccl::hcclComm*>(opBaseHcom);
    CHK_PTR_NULL(hcclComm);
    CHK_RET(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB));
    /* 入参的正确性由HCCL确保 */
    HcclResult ret = hcclComm->Reduce(tag, inputPtr, outputPtr, count, dataType, op, root, stream);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Reduce][Result]errNo[0x%016llx] hcclComm Reduce error, tag[%s], "
            "input_ptr[%p], output_ptr[%p], count[%llu], data_type[%s], op[%s], root[%u]",
            HCOM_ERROR_CODE(ret), tag, inputPtr, outputPtr, count, GetDataTypeEnumStr(dataType).c_str(),
            GetReduceOpEnumStr(op).c_str(), root),
        ret);
    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_REDUCE, beginTime, count, dataType));
    HcclUs endut = TIME_NOW();
    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "HcclCommGraphReduce success, take time [%lld]us, tag[%s], input_ptr[%p], output_ptr[%p], "
        "count[%llu], data_type[%s], op[%s], root[%u], opBaseHcom[%lld]",
        DURATION_US(endut - startut), tag, inputPtr, outputPtr, count, GetDataTypeEnumStr(dataType).c_str(),
        GetReduceOpEnumStr(op).c_str(), root, opBaseHcom);

    return HCCL_SUCCESS;
}

HcclResult HcclCommGraphBroadcast(
    const char* tag, void* ptr, u64 count, HcclDataType dataType, u32 root, s64 opBaseHcom, rtStream_t stream)
{
    HcclUs startut = TIME_NOW();
    uint64_t beginTime = hrtMsprofSysCycleTime();
    CHK_PRT_RET(count == 0, HCCL_WARNING("input count is 0, return BroadCast success"), HCCL_SUCCESS);

    // 入参合法性校验
    RPT_INPUT_ERR(
        ptr == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclCommGraphBroadcast", "nullptr", "ptr", "non-null pointer"}));
    CHK_PTR_NULL(ptr);
    RPT_INPUT_ERR(
        stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclCommGraphBroadcast", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);
    u32 totalRanks = 0;
    CHK_RET(HcclCommGraphGetRankSize(opBaseHcom, &totalRanks));
    CHK_RET(HcomCheckUserRank(totalRanks, root));

    s32 streamId = 0;
    CHK_RET(hrtGetStreamId(stream, streamId));
    /* 接口交互信息日志 */
    HCCL_RUN_INFO(
        "Entry-HcclCommGraphBroadcast:tag[%s], ptr[%p], count[%llu], dataType[%s], root[%u], "
        "opBaseHcom[%lld], streamId[%d]",
        tag, ptr, count, GetDataTypeEnumStr(dataType).c_str(), root, opBaseHcom, streamId);

    CHK_RET(PrintMemoryAttr(ptr));
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        hccl::hcclComm* hcclComm = reinterpret_cast<hccl::hcclComm*>(opBaseHcom);
        CHK_PTR_NULL(hcclComm);
        HcclComm commV2 = hcclComm->GetCommunicatorV2();
        CHK_PTR_NULL(commV2);
        opBaseHcom = reinterpret_cast<s64>(commV2);
        CHK_RET(HcomGraphBroadcastV2(tag, ptr, count, dataType, root, opBaseHcom, stream));
        return HCCL_SUCCESS;
    }());
    CHK_RET(HcomCheckOpParam(tag, count, dataType));
    hccl::hcclComm* hcclComm = reinterpret_cast<hccl::hcclComm*>(opBaseHcom);
    CHK_PTR_NULL(hcclComm);
    CHK_RET(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB));
    u32 aivCoreLimit = 0;
    CHK_RET(hcclComm->GetNumBlocks(aivCoreLimit));
    /* 入参的正确性由HCCL确保 */
    HcclResult ret = hcclComm->Broadcast(tag, ptr, count, dataType, root, stream);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Broadcast][Result]errNo[0x%016llx] hcclComm BroadCast error,tag[%s], input_ptr[%p],"
            "count[%llu], data_type[%s], root[%u]",
            HCOM_ERROR_CODE(ret), tag, ptr, count, GetDataTypeEnumStr(dataType).c_str(), root),
        ret);
    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_BROADCAST, beginTime, count, dataType));
    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "HcclCommGraphBroadcast success,take time [%lld]us,tag[%s], input_ptr[%p], count[%llu], "
        "data_type[%s], root[%u], opBaseHcom[%lld]",
        DURATION_US(TIME_NOW() - startut), tag, ptr, count, GetDataTypeEnumStr(dataType).c_str(), root, opBaseHcom);

    return HCCL_SUCCESS;
}
HcclResult HcclCommGraphReduceScatter(
    const char* tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op, s64 opBaseHcom,
    rtStream_t stream)
{
    HcclUs startut = TIME_NOW();
    uint64_t beginTime = hrtMsprofSysCycleTime();
    CHK_PRT_RET(count == 0, HCCL_WARNING("input count is 0, return ReduceScatter success"), HCCL_SUCCESS);
    // 入参合法性校验
    RPT_INPUT_ERR(
        inputPtr == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclCommGraphReduceScatter", "nullptr", "inputPtr", "non-null pointer"}));
    CHK_PTR_NULL(inputPtr);
    RPT_INPUT_ERR(
        outputPtr == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclCommGraphReduceScatter", "nullptr", "outputPtr", "non-null pointer"}));
    CHK_PTR_NULL(outputPtr);
    RPT_INPUT_ERR(
        stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclCommGraphReduceScatter", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);
    CHK_RET(HcomCheckReductionOp("HcclCommGraphReduceScatter", op));

    s32 streamId = 0;
    CHK_RET(hrtGetStreamId(stream, streamId));
    /* 接口交互信息日志 */
    HCCL_RUN_INFO(
        "Entry-HcclCommGraphReduceScatter:tag[%s], inputPtr[%p], outputPtr[%p], count[%llu], dataType[%s], "
        "op[%s], opBaseHcom[%lld], streamId[%d]",
        tag, inputPtr, outputPtr, count, GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str(),
        opBaseHcom, streamId);
    CHK_RET(PrintMemoryAttr(inputPtr));
    CHK_RET(PrintMemoryAttr(outputPtr));

    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        hccl::hcclComm* hcclComm = reinterpret_cast<hccl::hcclComm*>(opBaseHcom);
        CHK_PTR_NULL(hcclComm);
        HcclComm commV2 = hcclComm->GetCommunicatorV2();
        CHK_PTR_NULL(commV2);
        opBaseHcom = reinterpret_cast<s64>(commV2);
        CHK_RET(HcomGraphReduceScatterV2(tag, inputPtr, outputPtr, count, dataType, op, opBaseHcom, stream));
        return HCCL_SUCCESS;
    }());
    CHK_RET(HcomCheckOpParam(tag, count, dataType));
    hccl::hcclComm* hcclComm = reinterpret_cast<hccl::hcclComm*>(opBaseHcom);
    CHK_PTR_NULL(hcclComm);
    CHK_RET(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB));
    /* 入参的正确性由HCCL确保 */
    HcclResult ret = hcclComm->ReduceScatter(tag, inputPtr, outputPtr, count, dataType, op, stream);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[ReduceScatter][Result]errNo[0x%016llx] hcclComm ReduceScatter error, tag[%s],"
                    "input_ptr[%p], output_ptr[%p], count[%llu], data_type[%s], op[%s]",
                    HCOM_ERROR_CODE(ret), tag, inputPtr, outputPtr, count, GetDataTypeEnumStr(dataType).c_str(),
                    GetReduceOpEnumStr(op).c_str());
                , ret);
    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_REDUCE_SCATTER, beginTime, count, dataType));
    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "HcclCommGraphReduceScatter success, take time [%lld]us, tag[%s], input_ptr[%p], output_ptr[%p], "
        "count[%llu], data_type[%s], op[%s], opBaseHcom[%lld]",
        DURATION_US(TIME_NOW() - startut), tag, inputPtr, outputPtr, count, GetDataTypeEnumStr(dataType).c_str(),
        GetReduceOpEnumStr(op).c_str(), opBaseHcom);

    return HCCL_SUCCESS;
}
HcclResult HcclCommGraphSend(
    const char* tag, void* inputPtr, u64 count, HcclDataType dataType, u32 destRank, u32 srTag, s64 opBaseHcom,
    rtStream_t stream)
{
    HcclUs startut = TIME_NOW();
    uint64_t beginTime = hrtMsprofSysCycleTime();
    CHK_PRT_RET(count == 0, HCCL_WARNING("input count is 0, return send success"), HCCL_SUCCESS);

    // 入参合法性校验
    RPT_INPUT_ERR(
        inputPtr == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclCommGraphSend", "nullptr", "inputPtr", "non-null pointer"}));
    CHK_PTR_NULL(inputPtr);
    RPT_INPUT_ERR(
        stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclCommGraphSend", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);
    u32 totalRanks = 0;
    CHK_RET(HcclCommGraphGetRankSize(opBaseHcom, &totalRanks));
    CHK_RET(HcomCheckUserRank(totalRanks, destRank));

    s32 streamId = 0;
    CHK_RET(hrtGetStreamId(stream, streamId));
    /* 接口交互信息日志 */
    HCCL_RUN_INFO(
        "Entry-HcclCommGraphSend:tag[%s], inputPtr[%p], count[%llu], dataType[%s], destRank[%u], srTag[%u], "
        "opBaseHcom[%lld], streamId[%d]",
        tag, inputPtr, count, GetDataTypeEnumStr(dataType).c_str(), destRank, srTag, opBaseHcom, streamId);

    CHK_RET(PrintMemoryAttr(inputPtr));
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        hccl::hcclComm* hcclComm = reinterpret_cast<hccl::hcclComm*>(opBaseHcom);
        CHK_PTR_NULL(hcclComm);
        HcclComm commV2 = hcclComm->GetCommunicatorV2();
        CHK_PTR_NULL(commV2);
        opBaseHcom = reinterpret_cast<s64>(commV2);
        CHK_RET(HcomGraphSendV2(tag, inputPtr, count, dataType, destRank, srTag, opBaseHcom, stream));
        return HCCL_SUCCESS;
    }());
    CHK_RET(HcomCheckOpParam(tag, count, dataType));
    hccl::hcclComm* hcclComm = reinterpret_cast<hccl::hcclComm*>(opBaseHcom);
    CHK_PTR_NULL(hcclComm);
    CHK_RET(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB));

    u32 rankID = 0;
    CHK_RET(HcclCommGraphGetRankId(opBaseHcom, &rankID));
    /* 调用HCCL的send, 入参的正确性由HCCL确保 */
    HcclResult ret = hcclComm->send(tag, inputPtr, count, dataType, destRank, stream, srTag, rankID);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Send][Result]errNo[0x%016llx] hcclComm send error, tag[%s], "
            "inputPtr[%p], count[%llu], dataType[%s], destRank[%u]",
            HCOM_ERROR_CODE(ret), tag, inputPtr, count, GetDataTypeEnumStr(dataType).c_str(), destRank),
        ret);
    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_SEND, beginTime, count, dataType));
    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "HcclCommGraphSend success,time[%lld]us,tag[%s],inputPtr[%p],count[%llu],dataType[%s],destRank[%u],"
        "srTag[%u], opBaseHcom[%lld]",
        DURATION_US(TIME_NOW() - startut), tag, inputPtr, count, GetDataTypeEnumStr(dataType).c_str(), destRank, srTag,
        opBaseHcom);

    return HCCL_SUCCESS;
}
HcclResult HcclCommGraphReceive(
    const char* tag, void* outputPtr, u64 count, HcclDataType dataType, u32 srcRank, u32 srTag, s64 opBaseHcom,
    rtStream_t stream)
{
    HcclUs startut = TIME_NOW();
    uint64_t beginTime = hrtMsprofSysCycleTime();
    CHK_PRT_RET(count == 0, HCCL_WARNING("input count is 0, return receive success"), HCCL_SUCCESS);
    // 入参合法性校验
    RPT_INPUT_ERR(
        outputPtr == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclCommGraphReceive", "nullptr", "outputPtr", "non-null pointer"}));
    CHK_PTR_NULL(outputPtr);
    RPT_INPUT_ERR(
        stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclCommGraphReceive", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);
    u32 totalRanks = 0;
    CHK_RET(HcclCommGraphGetRankSize(opBaseHcom, &totalRanks));
    CHK_RET(HcomCheckUserRank(totalRanks, srcRank));

    s32 streamId = 0;
    CHK_RET(hrtGetStreamId(stream, streamId));
    /* 接口交互信息日志 */
    HCCL_RUN_INFO(
        "Entry-HcclCommGraphReceive:tag[%s], outputPtr[%p], count[%llu], dataType[%s], srcRank[%u], "
        "srTag[%u], opBaseHcom[%lld], streamId[%d]",
        tag, outputPtr, count, GetDataTypeEnumStr(dataType).c_str(), srcRank, srTag, opBaseHcom, streamId);
    CHK_RET(PrintMemoryAttr(outputPtr));
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        hccl::hcclComm* hcclComm = reinterpret_cast<hccl::hcclComm*>(opBaseHcom);
        CHK_PTR_NULL(hcclComm);
        HcclComm commV2 = hcclComm->GetCommunicatorV2();
        CHK_PTR_NULL(commV2);
        opBaseHcom = reinterpret_cast<s64>(commV2);
        CHK_RET(HcomGraphReceiveV2(tag, outputPtr, count, dataType, srcRank, srTag, opBaseHcom, stream));
        return HCCL_SUCCESS;
    }());
    CHK_RET(HcomCheckOpParam(tag, count, dataType));

    hccl::hcclComm* hcclComm = reinterpret_cast<hccl::hcclComm*>(opBaseHcom);
    CHK_PTR_NULL(hcclComm);
    CHK_RET(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB));

    /* 记录指令信息用于一致性校验 */
    u32 rankID = 0;
    CHK_RET(HcclCommGraphGetRankId(opBaseHcom, &rankID));
    HcclResult ret = hcclComm->receive(tag, outputPtr, count, dataType, srcRank, stream, srTag, rankID);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Receive][Result]errNo[0x%016llx] hcclComm receive error,tag[%s], "
            "outputPtr[%p], count[%llu], dataType[%s], srcRank[%u],",
            HCOM_ERROR_CODE(ret), tag, outputPtr, count, GetDataTypeEnumStr(dataType).c_str(), srcRank),
        ret);
    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_RECEIVE, beginTime, count, dataType));
    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "HcclCommGraphReceive success,time[%lld]us, tag[%s], outputPtr[%p], count[%llu], dataType[%s], "
        "srcRank[%u], srTag[%u], opBaseHcom[%lld],",
        DURATION_US(TIME_NOW() - startut), tag, outputPtr, count, GetDataTypeEnumStr(dataType).c_str(), srcRank, srTag,
        opBaseHcom);

    return HCCL_SUCCESS;
}
HcclResult HcclCommGraphGetAlltoAllStagedWorkSpaceMemSize(
    s64 opBaseHcom, u64* sendCounts, u64* sdispls, HcclDataType sendType, u64* recvCounts, u64* rdispls,
    HcclDataType recvType, u64& memSize)
{
    CHK_PTR_NULL(sendCounts);
    CHK_PTR_NULL(sdispls);
    CHK_PTR_NULL(recvCounts);
    CHK_PTR_NULL(rdispls);
    CHK_RET(HcomCheckDataType(sendType));
    CHK_RET(HcomCheckDataType(recvType));

    hccl::hcclComm* hcclComm = reinterpret_cast<hccl::hcclComm*>(opBaseHcom);
    CHK_PTR_NULL(hcclComm);
    CHK_RET(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB));
    CHK_RET(hcclComm->GetAlltoAllStagedWorkSpaceMemSize(
        sendCounts, sdispls, sendType, recvCounts, rdispls, recvType, memSize));
    return HCCL_SUCCESS;
}
HcclResult HcclCommGraphSetWorkspaceResource(
    const std::string& tag, s64 opBaseHcom, std::vector<rtStream_t> stream, void* memPtr, u64 maxSize)
{
    hccl::hcclComm* hcclComm = reinterpret_cast<hccl::hcclComm*>(opBaseHcom);
    CHK_PTR_NULL(hcclComm);
    CHK_RET(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB));

    CHK_RET(hcclComm->SetWorkspaceResource(tag, memPtr, maxSize, stream));
    return HCCL_SUCCESS;
}

HcclResult HcclCommGraphGetRankSize(s64 opBaseHcom, u32* rankSize)
{
    RPT_INPUT_ERR(
        rankSize == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclCommGraphGetRankSize", "nullptr", "rankSize", "non-null pointer"}));
    CHK_PTR_NULL(rankSize);
    bool& isAutoTuneModeOpen = HcomGetCtxAutoTuneMode();
    if (isAutoTuneModeOpen) {
        *rankSize = 1;
        return HCCL_SUCCESS;
    }
    HCCL_INFO("HcclCommGraphGetRankSize:opBaseHcom[%lld]", opBaseHcom);
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        hccl::hcclComm* hcclComm = reinterpret_cast<hccl::hcclComm*>(opBaseHcom);
        CHK_PTR_NULL(hcclComm);
        HcclComm commV2 = hcclComm->GetCommunicatorV2();
        CHK_PTR_NULL(commV2);
        opBaseHcom = reinterpret_cast<s64>(commV2);
        CHK_RET(HcclCommGraphGetRankSizeV2(opBaseHcom, rankSize));
        return HCCL_SUCCESS;
    }());

    hccl::hcclComm* hcclComm = reinterpret_cast<hccl::hcclComm*>(opBaseHcom);
    CHK_PTR_NULL(hcclComm);
    CHK_RET(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB));
    CHK_RET(hcclComm->GetRankSize(*rankSize));

    return HCCL_SUCCESS;
}

HcclResult HcclCommGraphGetRankId(s64 opBaseHcom, u32* rankId)
{
    RPT_INPUT_ERR(
        rankId == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclCommGraphGetRankId", "nullptr", "rankId", "non-null pointer"}));
    CHK_PTR_NULL(rankId);
    bool& isAutoTuneModeOpen = HcomGetCtxAutoTuneMode();
    if (isAutoTuneModeOpen) {
        *rankId = 0;
        return HCCL_SUCCESS;
    }
    HCCL_INFO("HcclCommGraphGetRankId:opBaseHcom[%lld]", opBaseHcom);
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        hccl::hcclComm* hcclComm = reinterpret_cast<hccl::hcclComm*>(opBaseHcom);
        CHK_PTR_NULL(hcclComm);
        HcclComm commV2 = hcclComm->GetCommunicatorV2();
        CHK_PTR_NULL(commV2);
        opBaseHcom = reinterpret_cast<s64>(commV2);
        CHK_RET(HcclCommGraphGetRankIdV2(opBaseHcom, rankId));
        return HCCL_SUCCESS;
    }());

    hccl::hcclComm* hcclComm = reinterpret_cast<hccl::hcclComm*>(opBaseHcom);
    CHK_PTR_NULL(hcclComm);
    CHK_RET(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB));
    CHK_RET(hcclComm->GetUserRank(*rankId));

    return HCCL_SUCCESS;
}

HcclResult HcclCommGraphGetWorkspaceSubStreamNum(
    u64 count, HcclDataType dataType, HcclReduceOp op, const std::string& algName, s64 opBaseHcom, u64& streamNum,
    u64 dataSize, bool ifAiv, HcclCMDType opType)
{
    hccl::hcclComm* hcclComm = reinterpret_cast<hccl::hcclComm*>(opBaseHcom);
    return hcclComm->GetWorkspaceSubStreamNum(count, dataType, op, algName, streamNum, dataSize, ifAiv, opType);
}

HcclResult
HcclCommGraphGetAllReduceScratchSize(s64 opBaseHcom, const u64 count, const HcclDataType dataType, u64& outScratchSize)
{
    hccl::hcclComm* hcclComm = reinterpret_cast<hccl::hcclComm*>(opBaseHcom);
    CHK_PTR_NULL(hcclComm);
    return hcclComm->GetAllReduceScratchSize(count, dataType, outScratchSize);
}

HcclResult HcclCommGraphGetIdentifier(s64 opBaseHcom, std::string& identifier)
{
    hccl::hcclComm* hcclComm = reinterpret_cast<hccl::hcclComm*>(opBaseHcom);
    CHK_PTR_NULL(hcclComm);
    identifier = hcclComm->GetIdentifier();
    return HCCL_SUCCESS;
}

HcclResult GenerateGroupHash(std::string& group, std::string& groupHash)
{
    std::hash<std::string> hashString;
    size_t nameHash = hashString(group);
    groupHash = std::to_string(nameHash);
    return HCCL_SUCCESS;
}

HcclResult GenerateCclOpTag(const std::string& opType, const int64_t& hcomComm, std::string& group, std::string& sTag)
{
    HcomOpTagInfo& opTagInfo = HcomGetCtxOpTagInfo();

    // middle 在获取到hcomComm时，等于identifier；在获取到group时，等于group
    std::string middle;
    if (hcomComm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
        middle = group;
    } else {
        CHK_RET(HcclCommGraphGetIdentifier(hcomComm, middle));
    }
    std::hash<std::string> hashString;
    size_t nameHash = hashString(middle);
    sTag = opType + "_" + std::to_string(nameHash);
    // 多张图node name重复导致tag相同，因此对tag添加索引 tag = op type + node name + identifier name + index

    auto iter = opTagInfo.opIndex.find(middle);
    if (iter == opTagInfo.opIndex.end()) {
        opTagInfo.opIndex.insert({middle, 0});
        iter = opTagInfo.opIndex.find(middle);
        CHK_PRT_RET(
            (iter == opTagInfo.opIndex.end()),
            HCCL_ERROR("[Generate][OpTag]generate tag fail. get the op index failed. ret[%d]", HCCL_E_INTERNAL),
            HCCL_E_INTERNAL);
    }

    sTag = sTag + "_" + std::to_string(iter->second++);

    HCCL_INFO("generate ccl op tag success, tag[%s]", sTag.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcomGetDevId(const char* group, s32* devId)
{
    HCCLV2_FUNC_RUN(HcomGetDevIdV2(group, devId));
    /* 获取rankId */
    std::shared_ptr<hccl::hcclComm> hcclComm;
    if (group != nullptr && HcclGetCommHandle(group, hcclComm) == HCCL_SUCCESS) {
        CHK_RET(hcclComm->GetDeviceId(*devId));
    } else {
        if (group == nullptr) {
            group = HCCL_WORLD_GROUP;
        }
        u32 rankId = 0;
        CHK_RET(HcomGetRankId(group, &rankId));
        u32 worldRankId = 0;
        CHK_RET(HcomGetWorldRankFromGroupRank(group, rankId, &worldRankId));
        HcomInfo& hcomInfo = HcomGetCtxHomInfo();

        for (auto it : hcomInfo.rankTable.rankList) {
            if (worldRankId == it.rankId) {
                u32 deviceLogicId = 0;
                CHK_RET(hrtGetDeviceIndexByPhyId(static_cast<u32>(it.deviceInfo.devicePhyId), deviceLogicId));
                *devId = static_cast<s32>(deviceLogicId);
                return HCCL_SUCCESS;
            }
        }
        HCCL_WARNING("[Get][DevId]rankList has no item with rankId[%u]", worldRankId);
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommGraphGetDevId(s64 opBaseHcom, s32* devId)
{
    hccl::hcclComm* hcclComm = reinterpret_cast<hccl::hcclComm*>(opBaseHcom);
    CHK_PTR_NULL(hcclComm);
    CHK_PTR_NULL(devId);
    CHK_RET(hcclComm->GetDeviceId(*devId));
    HCCL_INFO("HcclCommGraphGetDevId devID[%d]", *devId);
    return HCCL_SUCCESS;
}

HcclResult HcomGetLocalRankSize(const char* group, u32* localRankSize)
{
    RPT_INPUT_ERR(
        localRankSize == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcomGetLocalRankSize", "nullptr", "localRankSize", "non-null pointer"}));
    CHK_PTR_NULL(localRankSize);

    HCCLV2_FUNC_RUN(HcomGetLocalRankSizeV2(group, localRankSize));
    bool& isAutoTuneModeOpen = HcomGetCtxAutoTuneMode();
    if (isAutoTuneModeOpen) {
        *localRankSize = 1;
        return HCCL_SUCCESS;
    }
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    CHK_PRT_RET(
        hcomInfo.pComm == nullptr,
        HCCL_ERROR("[Get][LocalRankSize]hcomInfo.pComm is null, "
                   "please check if the initialize process is called."),
        HCCL_E_PTR);
    HcclResult ret = HcomCheckGroupName(group);
    RPT_INPUT_ERR(
        ret != HCCL_SUCCESS, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>(
            {"HcomGetLocalRankSize",
             {group, strnlen(group, GROUP_NAME_MAX_LEN + 1)},
             "group",
             "a non-empty string of length 1 to " + std::to_string(GROUP_NAME_MAX_LEN)
                 + ", containing only alphanumeric characters and underscores"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] get local ranksize "
            "group name is invalid",
            LOG_KEYWORDS_TASK_EXEC.c_str(), LOG_KEYWORDS_INVALID_ARGUMENT.c_str(), HCOM_ERROR_CODE(ret)),
        ret);

    std::string strGroup = (group == nullptr) ? HCCL_WORLD_GROUP : group;

    u32 rankSize = 0;
    u32 serverNum = 0;
    CHK_RET(GetGroupRankInfo(strGroup.c_str(), RankInfoType::RANK_SIZE_IN_GROUP, 0, &rankSize));
    CHK_RET(GetGroupRankInfo(strGroup.c_str(), RankInfoType::SERVER_NUM_IN_GROUP, 0, &serverNum));

    CHK_PRT_RET(
        serverNum == 0,
        HCCL_ERROR("[Get][LocalRankSize]errNo[0x%016llx] server num is zero", HCOM_ERROR_CODE(HCCL_E_PARA)),
        HCCL_E_PARA);

    *localRankSize = rankSize / serverNum;
    HCCL_INFO("hcom get local rank size success, group[%s]", strGroup.c_str());

    return HCCL_SUCCESS;
}

HcclResult HcomGetRankId(const char* group, u32* rankId)
{
    RPT_INPUT_ERR(
        rankId == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcomGetRankId", "nullptr", "rankId", "non-null pointer"}));
    CHK_PTR_NULL(rankId);
    bool& isAutoTuneModeOpen = HcomGetCtxAutoTuneMode();
    if (isAutoTuneModeOpen) {
        *rankId = 0;
        return HCCL_SUCCESS;
    }

    HcclResult ret = HcomCheckGroupName(group);
    RPT_INPUT_ERR(
        ret != HCCL_SUCCESS, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>(
            {"HcomGetRankId",
             {group, strnlen(group, GROUP_NAME_MAX_LEN + 1)},
             "group",
             "a non-empty string of length 1 to " + std::to_string(GROUP_NAME_MAX_LEN)
                 + ", containing only alphanumeric characters and underscores"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] get_rank_id group name is invalid", LOG_KEYWORDS_TASK_EXEC.c_str(),
            LOG_KEYWORDS_INVALID_ARGUMENT.c_str(), HCOM_ERROR_CODE(ret)),
        ret);

    // HcomGetRankIdV2
    HCCLV2_FUNC_RUN(HcomGetRankIdV2(group, rankId));
    std::shared_ptr<hccl::hcclComm> hcclComm;
    CHK_RET(HcomGetCommByGroup(group, hcclComm));
    CHK_RET(hcclComm->GetGroupRank(*rankId));
    std::string strGroup = (group == nullptr) ? HCCL_WORLD_GROUP : group;
    HCCL_INFO("hcom get rank id success, group[%s], rankId[%u]", strGroup.c_str(), *rankId);

    return HCCL_SUCCESS;
}

HcclResult HcomGetLocalRankId(const char* group, u32* localRankId)
{
    CHK_PTR_NULL(localRankId);
    bool& isAutoTuneModeOpen = HcomGetCtxAutoTuneMode();
    if (isAutoTuneModeOpen) {
        *localRankId = 0;
        return HCCL_SUCCESS;
    }

    HCCLV2_FUNC_RUN(HcomGetLocalRankIdV2(group, localRankId));
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    CHK_PRT_RET(
        hcomInfo.pComm == nullptr,
        HCCL_ERROR("[Get][LocalRankId]hcomInfo.pComm is null, "
                   "please check if the initialize process is called."),
        HCCL_E_PTR);
    CHK_RET(HcomCheckGroupName(group));

    std::string strGroup = (group == nullptr) ? HCCL_WORLD_GROUP : group;

    u32 rankId = 0;
    u32 localRankSize = 0;
    CHK_RET(GetGroupRankInfo(strGroup.c_str(), RankInfoType::RANK_ID_IN_GROUP, 0, &rankId));
    CHK_RET(HcomGetLocalRankSize(strGroup.c_str(), &localRankSize));

    CHK_PRT_RET(
        localRankSize == 0,
        HCCL_ERROR("[Get][LocalRankId]errNo[0x%016llx] local rank size is zero", HCOM_ERROR_CODE(HCCL_E_PARA)),
        HCCL_E_PARA);

    *localRankId = rankId % localRankSize;
    HCCL_INFO("hcom get local rank id success, group[%s]", strGroup.c_str());

    return HCCL_SUCCESS;
}

HcclResult GetRankListHeterog(u32 rankNum, const u32* rankIds, HcclGroupParams& params);
HcclResult HcomCreateGroupImplHeterog(const std::string& group, const std::vector<u32>& rankIds)
{
    HcclUs startut = TIME_NOW();
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    std::string rankId;
    for (u32 i = 0; i < rankIds.size(); i++) {
        if (i < rankIds.size() - 1) {
            rankId += to_string(rankIds[i]) + ',';
        } else if (i == rankIds.size() - 1) {
            rankId += to_string(rankIds[i]);
        }
    }
    /* 接口交互信息日志 */
    HCCL_RUN_INFO(
        "Entry-HcomCreateGroup:group[%s], rankNum[%zu], rankIds[%s]", group.c_str(), rankIds.size(), rankId.c_str());

    CHK_PRT_RET(
        hcomInfo.pComm == nullptr,
        HCCL_ERROR("[Create][Group]hcomInfo.pComm is null, please check if the initialize process is called."),
        HCCL_E_PTR);

    CHK_PRT_RET(
        hcomInfo.rankTable.rankList.empty(), HCCL_ERROR("[Create][Group]group[%s] rankList is empty", group.c_str()),
        HCCL_E_INTERNAL);

    /* 已经存在的group不允许再次创建 */
    std::unique_lock<std::mutex> groupParaLock(hcomInfo.groupParamsLock);
    if (hcomInfo.hcomGroupMap.find(group) != hcomInfo.hcomGroupMap.end()) {
        HCCL_ERROR(
            "[Create][Group]errNo[0x%016llx] group[%s] already exists", HCOM_ERROR_CODE(HCCL_E_PARA), group.c_str());
        return HCCL_E_PARA;
    }
    groupParaLock.unlock();

    HcclGroupParams groupParamsTem;
    CHK_RET(GetRankListHeterog(rankIds.size(), rankIds.data(), groupParamsTem));

    // 如果是groupRank = INVALID_VALUE_RANKID，即本rank不参与create group
    if (groupParamsTem.groupRank == INVALID_VALUE_RANKID) {
        HCCL_ERROR(
            "[Create][Group]errNo[0x%016llx] confirm groupRank from worldRank[%u] error",
            HCOM_ERROR_CODE(HCCL_E_NOT_FOUND), hcomInfo.params.rank);
        return HCCL_E_NOT_FOUND;
    }

    /* 入参的正确性由HCCL确保 */

    CHK_RET(hcomInfo.pComm->CreateGroup(
        group, groupParamsTem.groupRank, hcomInfo.params.rank, groupParamsTem.groupRanks, groupParamsTem.pSubComm));
    CHK_SMART_PTR_NULL(groupParamsTem.pSubComm);

    groupParaLock.lock();
    hcomInfo.hcomGroupMap.insert(std::make_pair(group, groupParamsTem));
    groupParaLock.unlock();

    HCCL_RUN_INFO(
        "hcom create group[%s] success, take time [%lld]us", group.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult HcomAbortGroup(const char* group)
{
    /* 调优模式直接返回success */
    bool& isAutoTuneModeOpen = HcomGetCtxAutoTuneMode();
    if (isAutoTuneModeOpen) {
        return HCCL_SUCCESS;
    }
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    CHK_PRT_RET(
        hcomInfo.pComm == nullptr,
        HCCL_ERROR("[Destroy][Group]hcomInfo.pComm is null, "
                   "please check if the initialize process is called."),
        HCCL_E_PTR);

    HCCL_RUN_INFO("Entry-HcomAbortGroup : group[%s]", group);
    CHK_RET(DestroyFlag(group, true));
    u32 ref = 0;
    CHK_RET(HcomQueryGroupRef(group, ref));
    while (ref != 0) {
        std::shared_ptr<hccl::hcclComm> hcclComm = nullptr;
        CHK_RET(HcomGetCommByGroup(group, hcclComm));
        SaluSleep(ONE_HUNDRED_MICROSECOND_OF_USLEEP);
        CHK_RET(HcomQueryGroupRef(group, ref));
    }

    HCCL_RUN_INFO("hcom abort group[%s] success.", group);
    return HCCL_SUCCESS;
}

HcclResult HcomDestroyGroupImplHeterog(const std::string& group)
{
    /* 接口交互信息日志 */
    HCCL_RUN_INFO("Entry-HcomDestroyGroup:group[%s]", group.c_str());
    CHK_RET(HcomAbortGroup(group.c_str()));

    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    std::unique_lock<std::mutex> groupParaLock(hcomInfo.groupParamsLock);
    auto iter = hcomInfo.hcomGroupMap.find(group);
    if (iter == hcomInfo.hcomGroupMap.end()) {
        HCCL_ERROR(
            "[Destroy][Group]errNo[0x%016llx] group[%s] does not exist", HCOM_ERROR_CODE(HCCL_E_PARA), group.c_str());
        return HCCL_E_PARA;
    }

    CHK_RET(hcomInfo.pComm->DestroyGroup(group));

    (iter->second).groupRanks.clear(); // 清除该服务器内相关group的对应信息

    hcomInfo.hcomGroupMap.erase(group);
    groupParaLock.unlock();

    HCCL_RUN_INFO("hcom destroy group[%s] success.", group.c_str());
    return HCCL_SUCCESS;
}

bool HcomCallBackGroupIsInitHeterog([[maybe_unused]] HcomInfo& hcomInfo) { return false; }

HcclResult GetRankListHeterog(u32 rankNum, const u32* rankIds, HcclGroupParams& params)
{
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    std::vector<RankInfo_t> rankList;
    params.totalRanks = rankNum;
    params.worldRank = hcomInfo.params.rank;
    params.groupRank = INVALID_VALUE_RANKID;
    for (u32 i = 0; i < rankNum; i++) {
        params.groupRanks.push_back(rankIds[i]);
    }

    std::sort(params.groupRanks.begin(), params.groupRanks.end());
    if (params.groupRanks[rankNum - 1] >= hcomInfo.rankTable.rankNum) {
        HCCL_ERROR(
            "[get][RankList]errNo[0x%016llx] groupRanks[%u]:%u is invalid", HCOM_ERROR_CODE(HCCL_E_PARA), rankNum - 1,
            params.groupRanks[rankNum - 1]);
        return HCCL_E_PARA;
    }
    if (hcomInfo.rankTable.rankList.size() <= params.groupRanks[0]) {
        HCCL_ERROR(
            "[get][RankList]errNo[0x%016llx] groupRanks[0] is invalid:[%u]", HCOM_ERROR_CODE(HCCL_E_PARA),
            params.groupRanks[0]);
        return HCCL_E_PARA;
    }
    // groupRanks 个数已经校验非0
    std::string serverId = hcomInfo.rankTable.rankList[params.groupRanks[0]].serverId;
    u32 serverNum = 1; // serverNum初始值应为1，代表groupId为0的serverId;
    RankInfo_t rankInfo;
    for (u32 i = 0; i < rankNum; i++) {
        rankInfo = hcomInfo.rankTable.rankList[params.groupRanks[i]];
        // 校验worldRankID
        if (rankInfo.rankId != params.groupRanks[i]) {
            HCCL_ERROR(
                "[get][RankList]errNo[0x%016llx] in rankList, worldRanks[%u] is invalid", HCOM_ERROR_CODE(HCCL_E_PARA),
                rankInfo.rankId);
            return HCCL_E_PARA;
        }
        if (params.groupRanks[i] == params.worldRank) {
            params.groupRank = i;
        }
        if (rankInfo.serverId != serverId) {
            serverNum++;
            serverId = rankInfo.serverId;
        }
        rankInfo.rankId = i;          // 放入groupRankid
        rankList.push_back(rankInfo); // ranktable中的ranklist是以rankid的顺序排列的
        rankInfo.serverId = "";       // 释放前先指空字符串
    }
    params.serverNum = serverNum;
    bool isStandardCard = false;
    CHK_RET(hcomInfo.pComm->IsStandardCard(isStandardCard));
    if (!isStandardCard && hcomInfo.params.deviceType != DevType::DEV_TYPE_910B
        && hcomInfo.params.deviceType != DevType::DEV_TYPE_910_93) {
        CHK_RET(CheckRankTableConfigInfo(rankList, rankNum, serverNum));
    }
    return HCCL_SUCCESS;
}

HcclResult HcomGetSplitStrategy(
    const char* group, const struct model_feature* feature, u32** segmentIdxPtr, u32* len, bool* configured,
    GradSplitForceMode force, OriginalGraphShapeType shapeType)
{
    CHK_PTR_NULL(feature);
    CHK_PTR_NULL(feature->model_name);
    CHK_PTR_NULL(feature->gradient_size);
    CHK_PTR_NULL(feature->gradient_time);
    CHK_PTR_NULL(segmentIdxPtr);
    CHK_PTR_NULL(len);
    CHK_PTR_NULL(configured);

    bool bRet = feature->gradient_num == 0;
    CHK_PRT_RET(
        bRet, HCCL_ERROR("[Get][SplitStrategy]errNo[0x%016llx] gradient num is zero", HCOM_ERROR_CODE(HCCL_E_PARA)),
        HCCL_E_PARA);
    CHK_RET(HcomCheckGroupName(group));

    std::string strGroup = (group == nullptr) ? HCCL_WORLD_GROUP : group;
    /* 接口交互信息日志 */
    HCCL_RUN_INFO("Entry-HcomGetSplitStrategy:group[%s], feature[%p]", strGroup.c_str(), feature);

    /* 获取梯度切分策略 */
    std::vector<u32> segmentIdx;
    CHK_RET(GetGradientSegment(strGroup, feature, segmentIdx, *configured, force, shapeType));
    *len = static_cast<u32>(segmentIdx.size());
    *segmentIdxPtr = new u32[*len];
    std::copy(segmentIdx.begin(), segmentIdx.end(), *segmentIdxPtr);

    return HCCL_SUCCESS;
}

HcclResult GetGradientSegment(
    const std::string& group, const struct model_feature* feature, std::vector<u32>& segmentList, bool& configured,
    GradSplitForceMode force, OriginalGraphShapeType shapeType)
{
    HcclResult ret;
    HCCL_INFO(
        "<gradient_segment group %s, model gradient num %u, model name %s>", group.c_str(), feature->gradient_num,
        feature->model_name);
    /* 分段算法实现 */
    std::unique_ptr<hccl::GradientSegment> segmentImpl;
    segmentImpl.reset(new (std::nothrow) GradientSegment());
    CHK_SMART_PTR_NULL(segmentImpl);

    /* 校验基于总层数索引是否正确 */
    std::unique_lock<std::mutex> segmentIdxMapLock(g_segmentIdxMapLock);
    auto gIdxSearch = g_segmentIdxMap.find(group);
    if (gIdxSearch != g_segmentIdxMap.end()) {
        bool bRet = (gIdxSearch->second.size() != 0) && (gIdxSearch->second.back() != (feature->gradient_num - 1));
        CHK_PRT_RET(
            bRet,
            HCCL_ERROR(
                "[Get][GradientSegment]illegal segmentIndex maxVal=%u should be equal %u", gIdxSearch->second.back(),
                feature->gradient_num - 1),
            HCCL_E_PARA);
    }
    segmentIdxMapLock.unlock();
    ret = segmentImpl->GetGradientSegmentExecutor(group, feature, segmentList, configured, force, shapeType);
    if (ret == HCCL_SUCCESS) {
        std::string printStr;
        u32 baseIndex = 0;
        for (u32 i = 0; i < segmentList.size(); i++) {
            printStr.append("[");
            printStr.append(std::to_string(baseIndex));
            printStr.append(",");
            printStr.append(std::to_string(segmentList[i]));
            printStr.append("] ");
            baseIndex = segmentList[i] + 1;
        }
        HCCL_RUN_INFO(
            "gradient segment result: segment num: %zu, segment index list: %s ", segmentList.size(), printStr.c_str());
    }
    return ret;
}

HcclResult HcomExecSelectAlg(
    s64 comm, const char* group, HcclCMDType opType, u64 count, HcclDataType dataType, HcclReduceOp op,
    int32_t aivCoreLimit, bool& ifAiv, char* algName)
{
    std::string tempAlgName;
    if (comm != static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
        hccl::hcclComm* hcclComm = reinterpret_cast<hccl::hcclComm*>(comm);
        HcclComm commV2 = hcclComm->GetCommunicatorV2();
        CHK_PTR_NULL(commV2);
        comm = reinterpret_cast<s64>(commV2);
        CHK_RET(HcomSelectAlgV2(comm, group, opType, count, dataType, op, aivCoreLimit, ifAiv, tempAlgName));
    } else {
        std::string strGroup = (group == nullptr) ? HCCL_WORLD_GROUP : group;
        CHK_RET(HcomGraphSelectAlgV2(comm, group, opType, count, dataType, op, aivCoreLimit, ifAiv, tempAlgName));
    }
    int32_t sret = memcpy_s(algName, ALG_NAME_MAX_LEN, tempAlgName.c_str(), (tempAlgName.length() + 1));
    CHK_PRT_RET(
        sret != EOK,
        HCCL_ERROR(
            "[HcomExecSelectAlg][algName]memcpy failed. ret[%d],"
            "params:destMaxSize[%zu],count[%zu]",
            sret, ALG_NAME_MAX_LEN, (tempAlgName.length() + 1)),
        HCCL_E_PARA);
    return HCCL_SUCCESS;
}

HcclResult HcomSelectAlg(
    s64 comm, const char* group, u64 count, void* counts, HcclDataType dataType, HcclReduceOp op, HcclCMDType opType,
    int32_t aivCoreLimit, bool* ifAiv, char* algName)
{
    CHK_PTR_NULL(ifAiv);
    HCCLV2_FUNC_RUN(HcomExecSelectAlg(comm, group, opType, count, dataType, op, aivCoreLimit, *ifAiv, algName));
    HcclWorkflowMode lastWorkflowMode = GetWorkflowMode();
    SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);
    std::string tempAlgName;
    if (comm != static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
        hccl::hcclComm* hcclHcomComm = reinterpret_cast<hccl::hcclComm*>(comm);
        CHK_RET(hcclHcomComm->HcclSelectAlg(opType, count, counts, dataType, op, aivCoreLimit, *ifAiv, tempAlgName));
    } else {
        std::string strGroup = (group == nullptr) ? HCCL_WORLD_GROUP : group;
        std::shared_ptr<hccl::hcclComm> hcclComm;
        CHK_RET(HcomGetCommByGroup(strGroup.c_str(), hcclComm));
        CHK_RET(hcclComm->HcclSelectAlg(opType, count, counts, dataType, op, aivCoreLimit, *ifAiv, tempAlgName));
    }
    int32_t sret = memcpy_s(algName, ALG_NAME_MAX_LEN, tempAlgName.c_str(), (tempAlgName.length() + 1));
    CHK_PRT_RET(
        sret != EOK,
        HCCL_ERROR(
            "[HcomSelectAlg][algName]memcpy failed. ret[%d],"
            "params:destMaxSize[%zu],count[%zu]",
            sret, ALG_NAME_MAX_LEN, (tempAlgName.length() + 1)),
        HCCL_E_PARA);

    SetWorkflowMode(lastWorkflowMode);
    return HCCL_SUCCESS;
}

HcclResult HcomCalcAivCoreNum(
    const char* group, HcclCMDType opType, u64 count, void* counts, HcclDataType dataType, int32_t aivCoreLimit,
    char* algName, u32* numBlocks)
{
    CHK_PTR_NULL(numBlocks);
    std::string algNamV2(algName);
    HCCLV2_FUNC_RUN(HcomCalcNumBlocksV2(group, opType, count, dataType, aivCoreLimit, algNamV2, *numBlocks));
    std::string strGroup = (group == nullptr) ? HCCL_WORLD_GROUP : group;
    std::shared_ptr<hccl::hcclComm> hcclComm;
    CHK_RET(HcomGetCommByGroup(strGroup.c_str(), hcclComm));
    std::string algNam(algName);
    CHK_RET(hcclComm->HcclCalcNumBlocks(opType, count, counts, dataType, aivCoreLimit, algNam, *numBlocks));

    return HCCL_SUCCESS;
}

HcclResult HcomGetAlgExecParam(
    const char* tag, const char* group, u64 count, void* inputPtr, void* outputPtr, HcclCMDType opType,
    bool clearEnable, HcclDataType dataType, HcclReduceOp op, void** commContext, u64* len, u32 aivCoreLimit)
{
    CHK_PTR_NULL(commContext);
    CHK_PTR_NULL(len);
    HCCLV2_FUNC_RUN(HcclGetAlgExecParamV2(
        tag, group, count, inputPtr, outputPtr, opType, clearEnable, dataType, op, *commContext, *len, aivCoreLimit));

    std::string strGroup = (group == nullptr) ? HCCL_WORLD_GROUP : group;
    std::shared_ptr<hccl::hcclComm> hcclComm;
    CHK_RET(HcomGetCommByGroup(strGroup.c_str(), hcclComm));

    CHK_RET(hcclComm->HcclGetAlgExecParam(
        tag, count, inputPtr, outputPtr, opType, clearEnable, dataType, op, *commContext, *len, aivCoreLimit));
    return HCCL_SUCCESS;
}
// 取得所需的从stream数目
HcclResult HcomGetWorkspaceSubStreamNum(
    const char* group, u64& streamNum, u64 dataSize, HcclDataType dataType, u32 aivCoreLimit, HcclReduceOp reduceOp,
    u64 count, HcclCMDType optype)
{
    HCCLV2_FUNC_RUN(HcomGetWorkspaceSubStreamNumV2(group, streamNum, dataSize, dataType, optype));
    std::shared_ptr<hccl::hcclComm> hcclComm{};
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    hcclComm = hcomInfo.pComm;
    CHK_RET(HcomCheckGroupName(group));
    HcclResult ret = HCCL_SUCCESS;
    std::string strGroup = (group == nullptr) ? HCCL_WORLD_GROUP : group;
    if (strGroup != HCCL_WORLD_GROUP && hcomInfo.pComm != nullptr) {
        std::unique_lock<std::mutex> groupParaLock(hcomInfo.groupParamsLock);
        auto iter = hcomInfo.hcomGroupMap.find(strGroup);
        if (iter != hcomInfo.hcomGroupMap.end()) {
            hcclComm = (iter->second).pSubComm;
        } else {
            HCCL_WARNING("[HcomGetWorkspaceSubStreamNum], please check if the initialize process is called.");
            streamNum = 0;
        }
    } else if (hcomInfo.pComm == nullptr) {
        ret = HcclGetCommHandle(group, hcclComm);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_WARNING("[HcomGetWorkspaceSubStreamNum], please check if the initialize process is called."),
            HCCL_SUCCESS);
    }
    CHK_PRT_RET(hcclComm == nullptr, HCCL_ERROR("[HcomGetWorkspaceSubStreamNum] Get Comm is null"), HCCL_E_PTR);

    string algName;
    bool ifAiv = false;
    void* counts = nullptr;
    ret = hcclComm->HcclSelectAlg(optype, count, counts, dataType, reduceOp, aivCoreLimit, ifAiv, algName);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[HcomGetWorkspaceSubStreamNum] HcclSelectAlg failed, ret[%d], optype[%d], count[%llu],"
            "dataType[%d], reduceOp[%d]",
            ret, optype, count, dataType, reduceOp),
        ret);
    CHK_RET(hcclComm->GetWorkspaceSubStreamNum(count, dataType, reduceOp, algName, streamNum, dataSize, ifAiv, optype));
    HCCL_INFO(
        "[%s] get streamNum[%llu] for optype[%d], algName[%s], count[%llu], dataType[%d], reduceOp[%d], ifAiv[%d]",
        __func__, streamNum, optype, algName.c_str(), count, dataType, reduceOp, ifAiv);
    return HCCL_SUCCESS;
}

HcclResult
HcomGetWorkspaceMemSize(const std::string& opType, u64 count, HcclDataType dataType, const char* group, u64& memSize)
{
    HCCLV2_FUNC_RUN(HcomGetWorkspaceMemSizeV2(opType, count, dataType, group, memSize));
    u32 rankSize = 0;
    std::shared_ptr<hccl::hcclComm> hcclComm{};
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    if (hcomInfo.pComm != nullptr) {
        hcclComm = hcomInfo.pComm;
        if (group == nullptr) {
            group = HCCL_WORLD_GROUP;
        }
        CHK_RET(HcomGetRankSize(group, &rankSize));
        CHK_RET(hcclComm->GetWorkspaceMemSize(opType, count, dataType, rankSize, memSize));
    } else if (group != nullptr && HcclGetCommHandle(group, hcclComm) == HCCL_SUCCESS) {
        CHK_RET(hcclComm->GetRankSize(rankSize));
        CHK_RET(hcclComm->GetWorkspaceMemSize(opType, count, dataType, rankSize, memSize));
    } else {
        HCCL_WARNING("[GetWorkspaceMemSize] please check if the initialize process is called.");
        memSize = 0;
        return HCCL_SUCCESS;
    }
    /* 用户申请内存,获取memSize大小 */

    CHK_PRT_RET(
        memSize > DEVICE_MEMORY_MAX_ALLOC_SIZE,
        HCCL_ERROR("[GetWorkspaceMemSize]workspace memory size exceeds %llu bytes.", DEVICE_MEMORY_MAX_ALLOC_SIZE),
        HCCL_E_PARA);
    return HCCL_SUCCESS;
}

HcclResult HcomGetAlltoAllStagedWorkSpaceMemSize(
    const char* group, u64* sendCounts, u64* sdispls, HcclDataType sendType, u64* recvCounts, u64* rdispls,
    HcclDataType recvType, u64& memSize)
{
    CHK_PTR_NULL(sendCounts);
    CHK_PTR_NULL(sdispls);
    CHK_PTR_NULL(recvCounts);
    CHK_PTR_NULL(rdispls);

    HCCLV2_FUNC_RUN(HcomGetAlltoAllStagedWorkSpaceMemSizeV2(
        group, sendCounts, sdispls, sendType, recvCounts, rdispls, recvType, memSize));
    CHK_RET(HcomCheckDataType(sendType));
    CHK_RET(HcomCheckDataType(recvType));
    std::string strGroup = (group == nullptr) ? HCCL_WORLD_GROUP : group;

    std::shared_ptr<hccl::hcclComm> hcclComm;
    CHK_RET(HcomGetCommByGroup(strGroup.c_str(), hcclComm));
    CHK_RET(hcclComm->GetAlltoAllStagedWorkSpaceMemSize(
        sendCounts, sdispls, sendType, recvCounts, rdispls, recvType, memSize));
    return HCCL_SUCCESS;
}

HcclResult HcomGetAlltoAllvcStagedWorkSpaceMemSize(
    const char* group, std::vector<SendRecvInfo>& allMeshAggregationSendRecvInfo, u64& memSize)
{
    HCCLV2_FUNC_RUN(HcomGetAlltoAllvcStagedWorkSpaceMemSizeV2(group, memSize));
    std::string strGroup = (group == nullptr) ? HCCL_WORLD_GROUP : group;
    std::shared_ptr<hccl::hcclComm> hcclComm;
    CHK_RET(HcomGetCommByGroup(strGroup.c_str(), hcclComm));
    CHK_RET(hcclComm->GetAlltoAllStagedWorkSpaceMemSize(allMeshAggregationSendRecvInfo, memSize));
    return HCCL_SUCCESS;
}

HcclResult
HcomGetAllReduceScratchSize(const char* group, const u64 count, const HcclDataType dataType, u64& outScratchSize)
{
    std::shared_ptr<hccl::hcclComm> hcclComm;
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    if (hcomInfo.pComm != nullptr) {
        hcclComm = hcomInfo.pComm;
        CHK_RET(HcomCheckGroupName(group));
        std::string strGroup = (group == nullptr) ? HCCL_WORLD_GROUP : group;
        if (strGroup == HCCL_WORLD_GROUP) {
            hcclComm = hcomInfo.pComm;
        } else {
            std::unique_lock<std::mutex> groupParaLock(hcomInfo.groupParamsLock);
            auto iter = hcomInfo.hcomGroupMap.find(strGroup);
            if (iter != hcomInfo.hcomGroupMap.end()) {
                hcclComm = (iter->second).pSubComm;
                CHK_PRT_RET(hcclComm == nullptr, HCCL_ERROR("[Get][CommByGroup] Get Comm is null"), HCCL_E_PTR);
            } else {
                u64 memSize = SIZE_TABLE[dataType] * count;
                const u32 DEVICE_EIGHT = 8;
                if (memSize <= HCCL_SMALL_COUNT_GRAPH_64_KB) {
                    // 小数据
                    outScratchSize = memSize * (DEVICE_EIGHT - 1);
                }
                HCCL_DEBUG("[HcomGetAllReduceScratchSize] outScratchSize %llu", outScratchSize);
                groupParaLock.unlock();
                return HCCL_SUCCESS;
            }
            groupParaLock.unlock();
        }
    } else if (group == nullptr || HcclGetCommHandle(group, hcclComm) != HCCL_SUCCESS) {
        HCCL_WARNING("[GetAllReduceScratchSize], please check if the initialize process is called.");
        outScratchSize = 0;
        return HCCL_SUCCESS;
    }
    return hcclComm->GetAllReduceScratchSize(count, dataType, outScratchSize);
}

HcclResult HcomGetCCLBufferAvailableSize(u64& size)
{
    size = GetExternalInputCCLBuffSize() - CCL_COMM_INBUFFER_UNALIGNED_RESERVE_SIZE;
    return HCCL_SUCCESS;
}

HcclResult HcomCheckCommValidity(const char* group)
{
    HCCLV2_FUNC_RUN(HcomCheckCommValidityV2(group));
    std::shared_ptr<hccl::hcclComm> hcclComm;
    if (HcomGetCommByGroup(group, hcclComm) != HCCL_SUCCESS) {
        return HCCL_E_PTR;
    }

    return HCCL_SUCCESS;
}

HcclResult
HcomSetWorkspaceResource(const char* tag, const char* group, rtStream_t* stream, s32 len, void* memPtr, u64 maxSize)
{
    if (len < 0) {
        HCCL_ERROR("[HcomSetWorkspaceResource] len is %d", len);
        return HCCL_E_PARA;
    }
    std::vector<rtStream_t> rtStream(stream, stream + len);

    HCCLV2_FUNC_RUN(HcomSetWorkspaceResourceV2(tag, group, rtStream, memPtr, maxSize));
    if (group == nullptr) {
        group = HCCL_WORLD_GROUP;
    }

    std::shared_ptr<hccl::hcclComm> hcclComm;
    if (HcomGetCommByGroup(group, hcclComm) == HCCL_SUCCESS) {
        /* 设定 workspace 内存资源 */
        CHK_RET(hcclComm->SetWorkspaceResource(tag, memPtr, maxSize, rtStream));
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommSetAttachedStream(s64 opBaseHcom, u32 graphId, const std::vector<rtStream_t>& stream)
{
    hccl::hcclComm* hcclComm = reinterpret_cast<hccl::hcclComm*>(opBaseHcom);
    CHK_PTR_NULL(hcclComm);
    CHK_RET(hcclComm->SetAttachedStream(graphId, stream));

    return HCCL_SUCCESS;
}

void HcomSetAutoTuneMode(bool autoTuneMode)
{
    bool& isAutoTuneModeOpen = HcomGetCtxAutoTuneMode();
    isAutoTuneModeOpen = autoTuneMode;
}

HcclResult HcomSetExecTimeOut(const char* execTimeOut)
{
    HCCL_RUN_INFO("HcomSetExecTimeOut:execTimeOut[%s]s", execTimeOut);
    if (execTimeOut == nullptr) {
        return HCCL_SUCCESS;
    }
    CHK_RET(SetHccLExecTimeOut(execTimeOut, HcclExecTimeoutSet::HCCL_EXEC_TIMEOUT_SET_BY_OPTIONS));
    return HCCL_SUCCESS;
}

HcclResult HcomSetAlgorithm(const char* algo)
{
    HCCL_RUN_INFO("HcomSetAlgorithm:algo[%s]", algo);
    if (algo == nullptr) {
        return HCCL_SUCCESS;
    }
    CHK_RET(SetHcclAlgoConfig(algo));
    return HCCL_SUCCESS;
}

HcclResult HcomSetDeterministic(u8 deterministic)
{
    HCCL_RUN_INFO("HcomSetDeterministic:deterministic[%u]", deterministic);
    CHK_RET(SetDeterministic(deterministic));
    return HCCL_SUCCESS;
}

HcclResult HcomGetAlgorithm(u32 level, char** algo)
{
    CHK_PTR_NULL(algo);
    std::string str = "none";
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    CHK_PRT_RET(
        hcomInfo.pComm == nullptr,
        HCCL_ERROR("[Get][Algorithm]hcomInfo.pComm is null, "
                   "please check if the initialize process is called."),
        HCCL_E_PTR);
    AlgType algType;
    CHK_RET(hcomInfo.pComm->GetAlgType(algType, HcclCMDType::HCCL_CMD_ALL));
    if (level == 0) {
        if (algType.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_WHOLE_RING
            || algType.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_8P_RING
            || algType.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_4P_RING
            || algType.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING
            || algType.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING) {
            str = "ring";
        } else if (
            algType.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_NP_MESH
            || algType.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_4P_MESH
            || algType.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_2P_MESH
            || algType.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_1P_MESH) {
            str = "mesh";
        } else if (
            algType.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_RESERVED
            && algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
            str = "NHR";
        } else if (
            algType.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_RESERVED
            && algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1) {
            str = "NHR_V1";
        } else if (
            algType.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_RESERVED
            && algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC) {
            str = "AHC";
        } else if (
            algType.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_RESERVED
            && algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE) {
            str = "AHC_BROKE";
        } else if (
            algType.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_RESERVED
            && algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
            str = "NB";
        } else {
            HCCL_WARNING(
                "[Get][Algorithm] No valid Level 0 AlgType, which is [%d]", static_cast<s32>(algType.algoLevel0));
            return HCCL_E_NOT_FOUND;
        }
    } else if (level == 1) {
        if (algType.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_WHOLE_RING
            || (algType.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_RESERVED
                && algType.algoLevel1 != AlgTypeLevel1::ALG_LEVEL1_RESERVED)) {
            str = "none";
        } else if (algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_HD) {
            str = "H-D";
        } else if (
            algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE
            || algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
            str = "ring";
        } else if (algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
            str = "NHR";
        } else if (algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1) {
            str = "NHR_V1";
        } else if (algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC) {
            str = "AHC";
        } else if (algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE) {
            str = "AHC_BROKE";
        } else if (algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
            str = "NB";
        } else {
            HCCL_WARNING(
                "[Get][Algorithm] No valid Level 1 AlgType, which is [%d]", static_cast<s32>(algType.algoLevel1));
            return HCCL_E_NOT_FOUND;
        }
    }
    *algo = const_cast<char*>(str.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcomGetBandWidthPerNPU(u32 level, float* bandWidth)
{
    CHK_PTR_NULL(bandWidth);
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    CHK_PRT_RET(
        hcomInfo.pComm == nullptr,
        HCCL_ERROR("[Get][BandWidth]hcomInfo.pComm is null, "
                   "please check if the initialize process is called."),
        HCCL_E_PTR);
    CHK_RET(hcomInfo.pComm->GetBandWidthPerNPU(level, *bandWidth));
    return HCCL_SUCCESS;
}

HcclResult HcomReleaseSubComms()
{
    HCCLV2_FUNC_RUN(HcomReleaseSubCommsV2());
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    if (hcomInfo.pComm) {
        CHK_RET(hcomInfo.pComm->ReleaseSubComms());
    }

    auto iter = hcomInfo.hcomGroupMap.begin();
    while (iter != hcomInfo.hcomGroupMap.end()) {
        if (iter->second.pSubComm) {
            CHK_RET(iter->second.pSubComm->ReleaseSubComms());
        }
        iter++;
    }
    return HCCL_SUCCESS;
}

HcclResult HcomAlltoAllV(
    const void* sendBuf, const void* sendCounts, const void* sdispls, HcclDataType sendType, const void* recvBuf,
    const void* recvCounts, const void* rdispls, HcclDataType recvType, const char* group, rtStream_t stream,
    const char* tag)
{
    HcclUs startut = TIME_NOW();
    uint64_t beginTime = hrtMsprofSysCycleTime();
    // 入参合法性校验
    CHK_PTR_NULL(sendCounts);
    CHK_PTR_NULL(sdispls);
    CHK_PTR_NULL(recvCounts);
    CHK_PTR_NULL(rdispls);
    RPT_INPUT_ERR(
        stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcomAlltoAllV", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);
    s32 streamId = 0;
    CHK_RET(hrtGetStreamId(stream, streamId));
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDevice(&deviceLogicId));

    std::string strGroup = (group == nullptr) ? HCCL_WORLD_GROUP : group;
    if (sendBuf != nullptr) {
        CHK_RET(PrintMemoryAttr(sendBuf));
    }
    if (recvBuf != nullptr) {
        CHK_RET(PrintMemoryAttr(recvBuf));
    }

    /* 接口交互信息日志 */
    HCCL_USER_CRITICAL_LOG(
        "Entry-HcomAlltoAllV:tag[%s], sendBuf[%p], recvBuf[%p], sendCounts[%p], recvCounts[%p],"
        "sendType[%s], recvType[%s], group[%s], streamId[%d], deviceLogicId[%d]",
        tag, sendBuf, recvBuf, sendCounts, recvCounts, GetDataTypeEnumStr(sendType).c_str(),
        GetDataTypeEnumStr(recvType).c_str(), strGroup.c_str(), streamId, deviceLogicId);

    // HcomAlltoAllV2
    HCCLV2_FUNC_RUN(HcomAlltoAllVV2(
        sendBuf, sendCounts, sdispls, sendType, recvBuf, recvCounts, rdispls, recvType, group, stream, tag));
    CHK_RET(HcomCheckOpParam(tag, 0, sendType, group, stream));
    CHK_RET(HcomCheckDataType(recvType));
    std::shared_ptr<hccl::hcclComm> hcclComm;
    CHK_RET(HcomGetCommByGroup(strGroup.c_str(), hcclComm));

    // 根据ranksize校验相关入参
    u32 rankSize = 0;
    CHK_RET(hcclComm->GetRankSize(rankSize));
    CHK_RET(HcomCheckAlltoAllVExternalMem(sendBuf, sendCounts, recvBuf, recvCounts, rankSize));
    u32 rankId = 0;
    CHK_RET(hcclComm->GetUserRank(rankId));
    u32 aivCoreLimit = 0;
    CHK_RET(hcclComm->GetNumBlocks(aivCoreLimit));
    HcclWorkflowMode mode = GetWorkflowMode();
    CHK_PRT_RET(
        mode == HcclWorkflowMode::HCCL_WORKFLOW_MODE_RESERVED, HCCL_ERROR("Invalid Workflow Mode[%d]", mode),
        HCCL_E_INTERNAL);
    CHK_RET(hcclComm->AlltoAllV(
        sendBuf, sendCounts, sdispls, sendType, recvBuf, recvCounts, rdispls, recvType, stream, tag));

    u64 sendCount = 0;
    for (u32 i = 0; i < rankSize; i++) {
        sendCount += *(static_cast<const u64*>(sendCounts) + i);
    }
    CHK_RET(CallMsprofReportHostApi(hcclComm.get(), HcclCMDType::HCCL_CMD_ALLTOALLV, beginTime, sendCount, sendType));
    /* 关键状态记录 */
    HcclUs endut = TIME_NOW();
    HCCL_RUN_INFO(
        "HcomAlltoAllV success,take time [%lld]us, tag[%s], sendBuf[%p], recvBuf[%p], sendCounts[%p], "
        "recvCounts[%p], sendType[%s], recvType[%s], group[%s], streamId[%d]",
        DURATION_US(endut - startut), tag, sendBuf, recvBuf, sendCounts, recvCounts,
        GetDataTypeEnumStr(sendType).c_str(), GetDataTypeEnumStr(recvType).c_str(), strGroup.c_str(), streamId);
    return HCCL_SUCCESS;
}

HcclResult HcomAlltoAllVC(
    const void* sendBuf, const void* sendCountMatrix, HcclDataType sendType, const void* recvBuf, HcclDataType recvType,
    const char* group, rtStream_t stream, const char* tag)
{
    HcclUs startut = TIME_NOW();
    uint64_t beginTime = hrtMsprofSysCycleTime();
    // 入参合法性校验
    CHK_PTR_NULL(sendCountMatrix);
    RPT_INPUT_ERR(
        stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcomAlltoAllVC", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);
    s32 streamId = 0;
    CHK_RET(hrtGetStreamId(stream, streamId));
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDevice(&deviceLogicId));

    std::string strGroup = (group == nullptr) ? HCCL_WORLD_GROUP : group;
    if (sendBuf != nullptr) {
        CHK_RET(PrintMemoryAttr(sendBuf));
    }
    if (recvBuf != nullptr) {
        CHK_RET(PrintMemoryAttr(recvBuf));
    }
    // HcomAlltoAllVCV2
    HCCLV2_FUNC_RUN(HcomAlltoAllVCV2(sendBuf, sendCountMatrix, sendType, recvBuf, recvType, group, stream, tag));
    CHK_RET(HcomCheckOpParam(tag, 0, sendType, group, stream));
    CHK_RET(HcomCheckDataType(recvType));

    // 根据ranksize校验相关入参
    std::shared_ptr<hccl::hcclComm> hcclComm;
    CHK_RET(HcomGetCommByGroup(strGroup.c_str(), hcclComm));
    u32 rankSize = 0;
    CHK_RET(hcclComm->GetRankSize(rankSize));
    u32 userRank = 0;
    hcclComm->GetGroupRank(userRank);
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    u32 rank = hcomInfo.params.userRank;
    CHK_RET(HcomCheckAlltoAllVCExternalMem(sendBuf, sendCountMatrix, recvBuf, rankSize, rank));

    u64 sendCountMatrixHash;
    HcomGetHashFromSendCountMatrix(sendCountMatrixHash, sendCountMatrix, rankSize, tag);

    /* 接口交互信息日志 */
    HCCL_USER_CRITICAL_LOG(
        "Entry-HcomAlltoAllVC:tag[%s], sendBuf[%p], sendCountMatrixHash[%llu], sendType[%s], "
        "recvBuf[%p], recvType[%s], group[%s], streamId[%d], deviceLogicId[%d]",
        tag, sendBuf, sendCountMatrixHash, GetDataTypeEnumStr(sendType).c_str(), recvBuf,
        GetDataTypeEnumStr(recvType).c_str(), strGroup.c_str(), streamId, deviceLogicId);

    u32 aivCoreLimit = 0;
    CHK_RET(hcclComm->GetNumBlocks(aivCoreLimit));
    HcclWorkflowMode mode = GetWorkflowMode();
    CHK_PRT_RET(
        mode == HcclWorkflowMode::HCCL_WORKFLOW_MODE_RESERVED, HCCL_ERROR("Invalid Workflow Mode[%d]", mode),
        HCCL_E_INTERNAL);
    /* 入参的正确性由HCCL确保 */
    CHK_RET(hcclComm->AlltoAllVC(sendBuf, sendCountMatrix, sendType, recvBuf, recvType, stream, tag));

    u64 sendCount = 0;
    for (u32 i = 0; i < rankSize; i++) {
        sendCount += *(static_cast<const u64*>(sendCountMatrix) + userRank * rankSize + i);
    }
    CHK_RET(CallMsprofReportHostApi(hcclComm.get(), HcclCMDType::HCCL_CMD_ALLTOALLVC, beginTime, sendCount, sendType));
    /* 关键状态记录 */
    HcclUs endut = TIME_NOW();
    HCCL_RUN_INFO(
        "HcomAlltoAllVC success,take time [%lld]us, tag[%s], sendBuf[%p], sendCountMatrix[%p], "
        "sendType[%s], recvBuf[%p], recvType[%s], group[%s], streamId[%d]",
        DURATION_US(endut - startut), tag, sendBuf, sendCountMatrix, GetDataTypeEnumStr(sendType).c_str(), recvBuf,
        GetDataTypeEnumStr(recvType).c_str(), strGroup.c_str(), streamId);
    return HCCL_SUCCESS;
}

HcclResult HcclCommGraphAlltoAllV(
    const void* sendBuf, const void* sendCounts, const void* sdispls, HcclDataType sendType, const void* recvBuf,
    const void* recvCounts, const void* rdispls, HcclDataType recvType, s64 opBaseHcom, rtStream_t stream,
    const char* tag)
{
    HcclUs startut = TIME_NOW();
    uint64_t beginTime = hrtMsprofSysCycleTime();
    // 入参合法性校验
    CHK_PTR_NULL(sendCounts);
    CHK_PTR_NULL(sdispls);
    CHK_PTR_NULL(recvCounts);
    CHK_PTR_NULL(rdispls);

    CHK_RET(HcomCheckOpParam(tag, 0, sendType, stream));
    CHK_RET(HcomCheckDataType(recvType));
    s32 streamId = 0;
    CHK_RET(hrtGetStreamId(stream, streamId));
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDevice(&deviceLogicId));

    // 根据ranksize校验相关入参
    hccl::hcclComm* hcclComm = reinterpret_cast<hccl::hcclComm*>(opBaseHcom);
    CHK_PTR_NULL(hcclComm);
    CHK_RET(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB));
    u32 rankSize = 0;
    CHK_RET(hcclComm->GetRankSize(rankSize));
    u32 rankId = 0;
    CHK_RET(hcclComm->GetUserRank(rankId));
    CHK_RET(HcomCheckAlltoAllVExternalMem(sendBuf, sendCounts, recvBuf, recvCounts, rankSize));
    if (sendBuf != nullptr) {
        CHK_RET(PrintMemoryAttr(sendBuf));
    }
    if (recvBuf != nullptr) {
        CHK_RET(PrintMemoryAttr(recvBuf));
    }

    /* 接口交互信息日志 */
    HCCL_USER_CRITICAL_LOG(
        "Entry-HcomAlltoAllV:tag[%s], sendBuf[%p], recvBuf[%p], sendCounts[%p], recvCounts[%p],"
        "sendType[%s], recvType[%s], streamId[%d], deviceLogicId[%d]",
        tag, sendBuf, recvBuf, sendCounts, recvCounts, GetDataTypeEnumStr(sendType).c_str(),
        GetDataTypeEnumStr(recvType).c_str(), streamId, deviceLogicId);

    u32 aivCoreLimit = 0;
    CHK_RET(hcclComm->GetNumBlocks(aivCoreLimit));
    HcclWorkflowMode mode = GetWorkflowMode();
    CHK_PRT_RET(
        mode == HcclWorkflowMode::HCCL_WORKFLOW_MODE_RESERVED, HCCL_ERROR("Invalid Workflow Mode[%d]", mode),
        HCCL_E_INTERNAL);
    /* 入参的正确性由HCCL确保 */
    CHK_RET(hcclComm->AlltoAllV(
        sendBuf, sendCounts, sdispls, sendType, recvBuf, recvCounts, rdispls, recvType, stream, tag));
    u64 sendCount = 0;
    for (u32 i = 0; i < rankSize; i++) {
        sendCount += *(static_cast<const u64*>(sendCounts) + i);
    }
    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_ALLTOALLV, beginTime, sendCount, sendType));
    /* 关键状态记录 */
    HcclUs endut = TIME_NOW();
    HCCL_RUN_INFO(
        "HcomAlltoAllV success,take time [%lld]us, tag[%s], sendBuf[%p], recvBuf[%p], sendCounts[%p], "
        "recvCounts[%p], sendType[%s], recvType[%s], streamId[%d]",
        DURATION_US(endut - startut), tag, sendBuf, recvBuf, sendCounts, recvCounts,
        GetDataTypeEnumStr(sendType).c_str(), GetDataTypeEnumStr(recvType).c_str(), streamId);
    return HCCL_SUCCESS;
}

HcclResult HcclCommGraphAlltoAllVC(
    const void* sendBuf, const void* sendCountMatrix, HcclDataType sendType, const void* recvBuf, HcclDataType recvType,
    s64 opBaseHcom, rtStream_t stream, const char* tag)
{
    HcclUs startut = TIME_NOW();
    uint64_t beginTime = hrtMsprofSysCycleTime();
    // 入参合法性校验
    CHK_PTR_NULL(sendCountMatrix);

    CHK_RET(HcomCheckOpParam(tag, 0, sendType, stream));
    CHK_RET(HcomCheckDataType(recvType));
    s32 streamId = 0;
    CHK_RET(hrtGetStreamId(stream, streamId));
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDevice(&deviceLogicId));

    // 根据ranksize校验相关入参
    hccl::hcclComm* hcclComm = reinterpret_cast<hccl::hcclComm*>(opBaseHcom);
    CHK_PTR_NULL(hcclComm);
    CHK_RET(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB));
    u32 rankSize = 0;
    CHK_RET(hcclComm->GetRankSize(rankSize));
    u32 userRank = 0;
    hcclComm->GetGroupRank(userRank);
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    u32 rank = hcomInfo.params.userRank;
    CHK_RET(HcomCheckAlltoAllVCExternalMem(sendBuf, sendCountMatrix, recvBuf, rankSize, rank));
    if (sendBuf != nullptr) {
        CHK_RET(PrintMemoryAttr(sendBuf));
    }
    if (recvBuf != nullptr) {
        CHK_RET(PrintMemoryAttr(recvBuf));
    }

    u64 sendCountMatrixHash;
    HcomGetHashFromSendCountMatrix(sendCountMatrixHash, sendCountMatrix, rankSize, tag);

    /* 接口交互信息日志 */
    HCCL_USER_CRITICAL_LOG(
        "Entry-HcomAlltoAllVC:tag[%s], sendBuf[%p], sendCountMatrixHash[%llu], sendType[%s], "
        "recvBuf[%p], recvType[%s], streamId[%d], deviceLogicId[%d]",
        tag, sendBuf, sendCountMatrixHash, GetDataTypeEnumStr(sendType).c_str(), recvBuf,
        GetDataTypeEnumStr(recvType).c_str(), streamId, deviceLogicId);

    u32 aivCoreLimit = 0;
    CHK_RET(hcclComm->GetNumBlocks(aivCoreLimit));
    HcclWorkflowMode mode = GetWorkflowMode();
    CHK_PRT_RET(
        mode == HcclWorkflowMode::HCCL_WORKFLOW_MODE_RESERVED, HCCL_ERROR("Invalid Workflow Mode[%d]", mode),
        HCCL_E_INTERNAL);
    /* 入参的正确性由HCCL确保 */
    CHK_RET(hcclComm->AlltoAllVC(sendBuf, sendCountMatrix, sendType, recvBuf, recvType, stream, tag));

    u64 sendCount = 0;
    for (u32 i = 0; i < rankSize; i++) {
        sendCount += *(static_cast<const u64*>(sendCountMatrix) + userRank * rankSize + i);
    }
    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_ALLTOALLVC, beginTime, sendCount, sendType));
    /* 关键状态记录 */
    HcclUs endut = TIME_NOW();
    HCCL_RUN_INFO(
        "HcomAlltoAllVC success, take time [%lld]us, tag[%s], sendBuf[%p], recvBuf[%p], sendCountMatrix[%p], "
        "sendType[%s], recvType[%s], streamId[%d]",
        DURATION_US(endut - startut), tag, sendBuf, recvBuf, sendCountMatrix, GetDataTypeEnumStr(sendType).c_str(),
        GetDataTypeEnumStr(recvType).c_str(), streamId);
    return HCCL_SUCCESS;
}

HcclResult HcomUnloadTask(const char* group, const char* tag)
{
    CHK_PTR_NULL(tag);
    HCCLV2_FUNC_RUN(HcomUnloadTaskV2(group, tag));
    std::shared_ptr<hcclComm> hcclComm;
    if (HcomGetCommByGroup(group, hcclComm) == HCCL_SUCCESS) {
        CHK_PRT_RET(
            hcclComm == nullptr,
            HCCL_WARNING("[UnloadAllTask]hcclComm is null, "
                         "please check if the initialize process is called."),
            HCCL_SUCCESS);
        HCCL_INFO("[UnloadTask]HcomUnloadTask: tag[%s]", tag);
        CHK_RET(hcclComm->ClearOpResource(tag));
    }

    return HCCL_SUCCESS;
}

HcclResult HcomGetServerNumAndDeviceNumPerServer(u32* serverNum, u32* deviceNumPerServer, u32* deviceNumPerAggregation)
{
    CHK_PTR_NULL(serverNum);
    CHK_PTR_NULL(deviceNumPerServer);
    CHK_PTR_NULL(deviceNumPerAggregation);
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    CHK_PRT_RET(
        hcomInfo.pComm == nullptr,
        HCCL_ERROR("[GetServerNumAndDeviceNumPerServer]hcomInfo.pComm is null, "
                   "please check if the initialize process is called."),
        HCCL_E_INTERNAL);
    u32 totalDevNum;
    *serverNum = hcomInfo.rankTable.serverNum;
    totalDevNum = hcomInfo.rankTable.deviceNum;
    if ((totalDevNum % *serverNum) != 0) {
        HCCL_ERROR("devicenum is not Integer.");
    }
    *deviceNumPerServer = totalDevNum / *serverNum;

    CHK_RET(hcomInfo.pComm->GetDeviceNumPerAggregation(*deviceNumPerAggregation));

    return HCCL_SUCCESS;
}

static RankTable_t g_rankTableSetInfo;
HcclResult HcomSetRankTableImpl(const char* rankTableStr)
{
    HCCL_RUN_INFO("Entry-HcomSetRankTable: rankTable \"%s\"", rankTableStr);
    u32 rankTableSize = 0;
    HcclResult ret = HcomCheckRankTable(rankTableStr, rankTableSize);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[Init][Result]errNo[0x%016llx] input rankTable error", HCOM_ERROR_CODE(ret)),
        ret);

    std::string identify = "0";
    HcomInfo hcomInfo;
    ret = CfgGetClusterInfoWithoutDev(rankTableStr, identify, hcomInfo.params, hcomInfo.rankTable);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Init][Result]errNo[0x%016llx] cfg get ranktable[%p] info error: "
            "identify[%s]",
            HCOM_ERROR_CODE(ret), rankTableStr, identify.c_str()),
        ret);

    g_rankTableSetInfo = hcomInfo.rankTable;
    return HCCL_SUCCESS;
}

HcclResult HcomGetActualRankSizeImpl(const char* group, u32* rankSize)
{
    (void)group;
    *rankSize = g_rankTableSetInfo.rankNum;
    return HCCL_SUCCESS;
}

HcclResult HcclCommGraphUnloadTask(s64 opBaseHcom, const char* tag)
{
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    CHK_RET(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB));
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        hccl::hcclComm* hcclComm = reinterpret_cast<hccl::hcclComm*>(opBaseHcom);
        CHK_PTR_NULL(hcclComm);
        HcclComm commV2 = hcclComm->GetCommunicatorV2();
        CHK_PTR_NULL(commV2);
        opBaseHcom = reinterpret_cast<s64>(commV2);
        CHK_RET(HcclCommGraphUnloadTaskV2(opBaseHcom, tag));
        return HCCL_SUCCESS;
    }());
#endif
    hccl::hcclComm* hcclComm = reinterpret_cast<hccl::hcclComm*>(opBaseHcom);
    CHK_PRT_RET(
        hcclComm == nullptr,
        HCCL_WARNING("[HcclCommGraphUnloadTask]hcclComm is null, "
                     "please check if the initialize process is called."),
        HCCL_SUCCESS);
    CHK_RET(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB));
    HCCL_INFO("[UnloadTask]HcclCommGraphUnloadTask: tag[%s]", tag);
    CHK_RET(hcclComm->ClearOpResource(tag));
    return HCCL_SUCCESS;
}

HcclResult HcomSetGlobalWorkSpace(const char* group, void** globalWorkSpaceAddr, u32 len)
{
    std::vector<void*> workspaceAddrVec(globalWorkSpaceAddr, globalWorkSpaceAddr + len);
    HCCLV2_FUNC_RUN(HcomSetGlobalWorkSpaceV2(group, workspaceAddrVec));
    std::shared_ptr<hccl::hcclComm> hcclComm;
    std::vector<void*> globalWorkSpaceAdd(globalWorkSpaceAddr, globalWorkSpaceAddr + len);
    CHK_RET(HcomGetCommByGroup(group, hcclComm));
    CHK_RET(hcclComm->SetGlobalWorkSpace(globalWorkSpaceAdd));
    return HCCL_SUCCESS;
}

HcclResult HcclCommSetGlobalWorkSpace(s64 opBaseHcom, std::vector<void*>& globalWorkSpaceAddr)
{
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    if (devType == DevType::DEV_TYPE_950 || devType == DevType::DEV_TYPE_960) {
        HCCL_WARNING(" A5/A6 does not support this interface");
        return HCCL_SUCCESS;
    }

    hccl::hcclComm* hcclComm = reinterpret_cast<hccl::hcclComm*>(opBaseHcom);
    CHK_PTR_NULL(hcclComm);
    CHK_RET(hcclComm->SetGlobalWorkSpace(globalWorkSpaceAddr));
    return HCCL_SUCCESS;
}

HcclResult HcomGetandClearOverFlowTasks(const char* group, hccl::HcclDumpInfo** hcclDumpInfoPtr, s32* len)
{
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    if (devType == DevType::DEV_TYPE_950 || devType == DevType::DEV_TYPE_960) {
        HCCL_WARNING("A5 does not support get and clear hcom over flow tasks.");
        return HCCL_SUCCESS;
    }

    std::shared_ptr<hccl::hcclComm> hcclComm;
    CHK_RET(HcomGetCommByGroup(group, hcclComm));
    std::vector<hccl::HcclDumpInfo> hcclDumpInfo;
    CHK_RET(hcclComm->GetandClearOverFlowTasks(hcclDumpInfo));
    if (hcclDumpInfo.size() > 0) {
        *hcclDumpInfoPtr = static_cast<hccl::HcclDumpInfo*>(malloc(hcclDumpInfo.size() * sizeof(hccl::HcclDumpInfo)));
        if (*hcclDumpInfoPtr == nullptr) {
            HCCL_ERROR(
                "[HcomGetandClearOverFlowTasks][HcclDumpInfo]mem malloc size[%zu] failed.",
                hcclDumpInfo.size() * sizeof(hccl::HcclDumpInfo));
            return HCCL_E_MEMORY;
        }
        int32_t sret = memcpy_s(
            *hcclDumpInfoPtr, hcclDumpInfo.size() * sizeof(hccl::HcclDumpInfo), hcclDumpInfo.data(),
            hcclDumpInfo.size() * sizeof(hccl::HcclDumpInfo));
        CHK_PRT_RET(
            sret != EOK,
            HCCL_ERROR(
                "[HcomGetandClearOverFlowTasks][HcclDumpInfo]memcpy failed. ret[%d], "
                "hcclDumpInfo:size[%zu]",
                sret, hcclDumpInfo.size()),
            HCCL_E_MEMORY);
    }
    *len = hcclDumpInfo.size();
    return HCCL_SUCCESS;
}

HcclResult HcclCommGetandClearOverFlowTasks(s64 opBaseHcom, std::vector<hccl::HcclDumpInfo>& hcclDumpInfo)
{
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    if (devType == DevType::DEV_TYPE_950 || devType == DevType::DEV_TYPE_960) {
        HCCL_WARNING("A5 does not support get and clear hcclcom over flow tasks.");
        return HCCL_SUCCESS;
    }

    hccl::hcclComm* hcclComm = reinterpret_cast<hccl::hcclComm*>(opBaseHcom);
    CHK_PTR_NULL(hcclComm);
    CHK_RET(hcclComm->GetandClearOverFlowTasks(hcclDumpInfo));
    return HCCL_SUCCESS;
}

HcclResult HcomSupportDeterministicOptim(const char* group, bool* isDeterministicOptim)
{
    CHK_PTR_NULL(isDeterministicOptim);
    HCCLV2_FUNC_RUN(HcomSupportDeterministicOptimV2(group, *isDeterministicOptim));
    std::shared_ptr<hccl::hcclComm> hcclComm;
    CHK_RET(HcomGetCommByGroup(group, hcclComm));
    CHK_RET(hcclComm->SupportDeterministicOptim(*isDeterministicOptim));
    return HCCL_SUCCESS;
}

HcclResult HcomGetHccsLinkNum(const char* group, u32* numHccsLink)
{
    std::shared_ptr<hccl::hcclComm> hcclComm;
    CHK_RET(HcomGetCommByGroup(group, hcclComm));
    CHK_RET(hcclComm->GetHccsLinkNum(*numHccsLink));
    return HCCL_SUCCESS;
}

HcclResult HcclCommSupportDeterministicOptim(s64 opBaseHcom, bool& isDeterministicOptim)
{
    hccl::hcclComm* hcclComm = reinterpret_cast<hccl::hcclComm*>(opBaseHcom);
    CHK_PTR_NULL(hcclComm);
    CHK_RET(hcclComm->SupportDeterministicOptim(isDeterministicOptim));
    return HCCL_SUCCESS;
}

std::vector<u64> GenerateSendCountMatrix(u64 count, u32 rankSize)
{
    std::vector<u64> sendCountMatrix(rankSize * rankSize, count);
    return sendCountMatrix;
}

HcclResult HcomAllToAll(
    const void* sendBuf, u64 sendCount, HcclDataType sendType, const void* recvBuf, u64 recvCount,
    HcclDataType recvType, const char* group, rtStream_t stream, const char* tag)
{
    HcclUs startut = TIME_NOW();
    uint64_t beginTime = hrtMsprofSysCycleTime();
    // 入参合法性校验
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    RPT_INPUT_ERR(
        stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcomAllToAll", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);

    s32 streamId = 0;
    CHK_RET(hrtGetStreamId(stream, streamId));
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDevice(&deviceLogicId));

    if (sendBuf != nullptr) {
        CHK_RET(PrintMemoryAttr(sendBuf));
    }
    if (recvBuf != nullptr) {
        CHK_RET(PrintMemoryAttr(recvBuf));
    }

    CHK_PRT_RET(sendCount == 0, HCCL_WARNING("send count is 0, return AllToAll success"), HCCL_SUCCESS);
    CHK_PRT_RET(recvCount == 0, HCCL_WARNING("recv count is 0, return AllToAll success"), HCCL_SUCCESS);

    std::string strGroup = (group == nullptr) ? HCCL_WORLD_GROUP : group;
    /* 接口交互信息日志 */
    HCCL_USER_CRITICAL_LOG(
        "Entry-HcomAllToAll:tag[%s], sendBuf[%p], recvBuf[%p], sendCount[%llu], recvCount[%llu],"
        "sendType[%s], recvType[%s], group[%s], streamId[%d], deviceLogicId[%d]",
        tag, sendBuf, recvBuf, sendCount, recvCount, GetDataTypeEnumStr(sendType).c_str(),
        GetDataTypeEnumStr(recvType).c_str(), strGroup.c_str(), streamId, deviceLogicId);

    // HcomAlltoAllV2
    HCCLV2_FUNC_RUN(HcomAlltoAllV2(sendBuf, sendCount, sendType, recvBuf, recvCount, recvType, group, stream, tag));
    CHK_RET(HcomCheckOpParam(tag, sendCount, sendType, stream));
    CHK_RET(HcomCheckOpParam(tag, recvCount, recvType, stream));
    CHK_RET(HcomCheckDataType(sendType));
    CHK_RET(HcomCheckDataType(recvType));
    // 根据ranksize校验相关入参
    std::shared_ptr<hccl::hcclComm> hcclComm;
    CHK_RET(HcomGetCommByGroup(strGroup.c_str(), hcclComm));
    u32 rankSize = 0, rankId = 0;
    CHK_RET(hcclComm->GetRankSize(rankSize));
    CHK_RET(hcclComm->GetUserRank(rankId));
    u32 aivCoreLimit = 0;
    CHK_RET(hcclComm->GetNumBlocks(aivCoreLimit));

    HcclWorkflowMode mode = GetWorkflowMode();
    CHK_PRT_RET(
        mode == HcclWorkflowMode::HCCL_WORKFLOW_MODE_RESERVED, HCCL_ERROR("Invalid Workflow Mode[%d]", mode),
        HCCL_E_INTERNAL);
    /* 入参的正确性由HCCL确保 */
    CHK_RET(hcclComm->AlltoAll(sendBuf, sendCount, sendType, recvBuf, recvCount, recvType, stream, tag));
    CHK_RET(CallMsprofReportHostApi(hcclComm.get(), HcclCMDType::HCCL_CMD_ALLTOALLV, beginTime, sendCount, sendType));
    /* 关键状态记录 */
    HcclUs endut = TIME_NOW();
    HCCL_RUN_INFO(
        "HcomAllToAll success,take time [%lld]us, tag[%s], sendBuf[%p], recvBuf[%p], sendCount[%llu], "
        "recvCounts[%llu], sendType[%s], recvType[%s], group[%s], streamId[%d]",
        DURATION_US(endut - startut), tag, sendBuf, recvBuf, sendCount, recvCount, GetDataTypeEnumStr(sendType).c_str(),
        GetDataTypeEnumStr(recvType).c_str(), strGroup.c_str(), streamId);
    return HCCL_SUCCESS;
}

HcclResult HcclIgetLookupRequest(
    [[maybe_unused]] HcclComm comm, [[maybe_unused]] s32* tag, [[maybe_unused]] ServiceHandle* handle,
    [[maybe_unused]] uint64_t* keys, [[maybe_unused]] uint64_t keyMaxNum, [[maybe_unused]] HcclRequest* request)
{
    HCCL_ERROR("[Iget][LookupRequest] does not support the HcclIgetLookupRequest interface");
    return HCCL_E_PARA;
}

HcclResult HcomCollRemotePairedParaCheck(const HcomRemoteOperationParams& params)
{
    CHK_PTR_NULL(params.keyAddr);
    CHK_PTR_NULL(params.value);
    CHK_PTR_NULL(params.tableId);
    CHK_PTR_NULL(params.indices);
    CHK_PTR_NULL(params.numUniqued);
    CHK_PTR_NULL(params.psSeg);
    CHK_PTR_NULL(params.psSegNum);

    return HCCL_SUCCESS;
}

HcclResult HcomInitByRankTable(const char* rankTable, uint32_t rankId)
{
    return HcomInitByString(rankTable, std::to_string(rankId).c_str(), HCCL_MODE_SCHED_OS);
}

inline void GenerateHcomSendRecvOpTag(
    HcomOperationType opType, const char* group, u32 tag, u32 selfRank, u32 peerRank, std::string& opTag)
{
    std::string groupStr = (group == nullptr) ? HCCL_WORLD_GROUP : group;
    if (opType == HCOM_OP_TYPE_SEND) {
        opTag = groupStr + "_" + std::to_string(tag) + "_" + std::to_string(selfRank) + "_" + std::to_string(peerRank);
    } else if (opType == HCOM_OP_TYPE_RECV) {
        opTag = groupStr + "_" + std::to_string(tag) + "_" + std::to_string(peerRank) + "_" + std::to_string(selfRank);
    }
    return;
}

HcclResult HcomGetTopoDesc(const char* group, HcclTopoDescs* topoDescs, uint32_t topoSize)
{
    CHK_PTR_NULL(topoDescs);
    CHK_PTR_NULL(group);
    HCCLV2_FUNC_RUN(HcomGetTopoDescV2(group, topoDescs, topoSize));

    std::shared_ptr<hcclComm> hcclComm;
    s32 deviceLogicId = 0;
    CHK_RET(HcclDeviceRefresh(deviceLogicId));
    if (HcomGetCommByGroup(group, hcclComm) == HCCL_SUCCESS) {
        CHK_RET(hcclComm->GetTopoDesc(topoDescs, topoSize));
    } else {
        return HCCL_E_PTR;
    }

    return HCCL_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
HcclResult HcomGetL0TopoTypeEx(const char* group, CommTopo* topoType, uint32_t flag)
{
    HCCLV2_FUNC_RUN(HcomGetL0TopoTypeExV2(group, topoType, flag));
#define IS_SET_DEVICE_MASK 0xfffffffe
    CHK_PTR_NULL(topoType);
    CHK_PTR_NULL(group);

    bool isSetDevice = static_cast<bool>(flag & (~(0xfffffffe)));
    if (isSetDevice) {
        HCCL_ERROR("current only support no setdevice, flag[%u]", flag);
        return HCCL_E_PARA;
    }

    std::string identifier(group);
    return CommTopoDesc::GetInstance().GetL0TopoType(identifier, topoType);
}

HcclResult HcomGetRankSizeEx(const char* group, uint32_t* rankSize, uint32_t flag)
{
    HCCLV2_FUNC_RUN(HcomGetRankSizeExV2(group, rankSize, flag));
#define IS_SET_DEVICE_MASK 0xfffffffe
    CHK_PTR_NULL(rankSize);
    CHK_PTR_NULL(group);

    bool isSetDevice = static_cast<bool>(flag & (~(0xfffffffe)));
    if (isSetDevice) {
        HCCL_ERROR("current only support no setdevice, flag[%u]", flag);
        return HCCL_E_PARA;
    }

    std::string identifier(group);
    return CommTopoDesc::GetInstance().GetRankSize(identifier, rankSize);
}
#ifdef __cplusplus
}
#endif // __cplusplus

HcclResult HcomGetCommCCLBufferSize(const char* group, uint64_t& size)
{
    HCCLV2_FUNC_RUN(HcomGetCommCCLBufferSizeV2());
    CHK_PTR_NULL(group);
    std::shared_ptr<hcclComm> hcclComm;
    CHK_RET(HcomGetCommByGroup(group, hcclComm));
    HcclResult ret = hcclComm->GetCommUserMemSize(size);
    CHK_PRT_RET(ret == HCCL_SUCCESS, HCCL_INFO("[%s]get comm ccl buffer size from user mem size", __func__), ret);
    if (0 == hcclComm->GetConfigInCCLbufferSize()) {
        size = GetExternalInputCCLBuffSize();
        HCCL_INFO("[%s]get comm ccl buffer size from external input", __func__);
    } else {
        size = hcclComm->GetConfigInCCLbufferSize();
        HCCL_INFO("[%s]get comm ccl buffer size from comm config", __func__);
    }
    return HCCL_SUCCESS;
}

bool HcomIsNormalComm([[maybe_unused]] const char* group)
{
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    return (hcomInfo.pComm != nullptr);
}

HcclResult HcomClearAivSyncBuf(const char* group, bool aivClearEnable)
{
    HCCLV2_FUNC_RUN(HcomSetAivClearEnableV2(group, aivClearEnable));
    CHK_PTR_NULL(group);
    std::shared_ptr<hcclComm> hcclComm;
    if (HcomGetCommByGroup(group, hcclComm) == HCCL_SUCCESS) {
        CHK_RET(hcclComm->SetClearAivSyncBuf(aivClearEnable));
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommGraphClearAivSyncBuf(s64 comm, bool aivClearEnable)
{
    hccl::hcclComm* hcclComm = reinterpret_cast<hccl::hcclComm*>(comm);
    if (hcclComm != nullptr) {
        CHK_RET(hcclComm->SetClearAivSyncBuf(aivClearEnable));
    }
    return HCCL_SUCCESS;
}

HcclResult HcomSetAivCoreLimit(const char* group, u32 aivCoreLimit)
{
    CHK_PRT_RET(
        aivCoreLimit == 0, HCCL_ERROR("[HcomSetAivCoreLimit] aivCoreLimit[%u] invalid", aivCoreLimit), HCCL_E_PARA);
    HCCLV2_FUNC_RUN(HcomSetAivCoreLimitV2(group, aivCoreLimit));

    std::shared_ptr<hccl::hcclComm> hcclComm;
    CHK_RET(HcomGetCommByGroup(group, hcclComm));
    CHK_RET(hcclComm->SetAivCoreLimit(aivCoreLimit));

    HCCL_RUN_INFO("HcomSetAivCoreLimit group[%s] aivCoreLimit[%u]", group ? group : HCCL_WORLD_GROUP, aivCoreLimit);
    return HCCL_SUCCESS;
}

HcclResult HcclCommGraphSetAivCoreLimit(s64 comm, u32 aivCoreLimit)
{
    CHK_PRT_RET(
        (comm == 0 || aivCoreLimit == 0),
        HCCL_ERROR("[HcclCommGraphSetAivCoreLimit] comm[%lld] or aivCoreLimit[%u] invalid", comm, aivCoreLimit),
        HCCL_E_PARA);

    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    if (devType == DevType::DEV_TYPE_950 || devType == DevType::DEV_TYPE_960) {
        HCCL_WARNING("A5 does not support get and clear hcclcom set aiv core limit.");
        return HCCL_SUCCESS;
    }

    hccl::hcclComm* hcclComm = reinterpret_cast<hccl::hcclComm*>(comm);
    CHK_RET(hcclComm->SetAivCoreLimit(aivCoreLimit));

    HCCL_RUN_INFO("HcclCommGraphSetAivCoreLimit hcclComm[%p] aivCoreLimit[%u]", hcclComm, aivCoreLimit);
    return HCCL_SUCCESS;
}

HcclResult HcomCalcTaskNum(HcomOpParam* hcomOpParam, u32& taskNum)
{
    CHK_PTR_NULL(hcomOpParam);
    HCCLV2_FUNC_RUN(HcomCalcTaskNumV2(hcomOpParam, taskNum));
    return HCCL_SUCCESS;
}

__attribute__((constructor)) void CallBackInit()
{
    HcomGroupCallbackFuncInstall(
        HcomCreateGroupImplHeterog, HcomCallBackGroupIsInitHeterog, HcomDestroyGroupImplHeterog,
        HcomDestroyOneDeviceHeterog);
}

HcclResult HcomGetGroupNameByOpBase(s64 opBaseHcom, char** groupname)
{
    hccl::hcclComm* hcclComm = reinterpret_cast<hccl::hcclComm*>(opBaseHcom);
    CHK_PTR_NULL(hcclComm);
    *groupname = const_cast<char*>(hcclComm->GetIdentifier().c_str());
    return HCCL_SUCCESS;
}

HcclResult GetGroupNameByOpBaseHcom(s64 opBaseHcom, char** groupname)
{
    hccl::hcclComm* hcclComm = reinterpret_cast<hccl::hcclComm*>(opBaseHcom);
    CHK_PTR_NULL(hcclComm);
    *groupname = const_cast<char*>(hcclComm->GetIdentifier().c_str());
    return HCCL_SUCCESS;
}

HcclResult HcomCreateComResourceByComm(HcclComm comm, u32 streamMode, bool isOpbaseMode, void** commContext, bool isMC2)
{
    CHK_RET(HcclCreateComResourceByComm(comm, streamMode, isOpbaseMode, commContext, isMC2));
    return HCCL_SUCCESS;
}

HcclWorkflowMode HcomGetWorkflowMode() { return GetWorkflowMode(); }

HcclResult HcomSetWorkflowMode(HcclWorkflowMode mode)
{
    SetWorkflowMode(mode);
    return HCCL_SUCCESS;
}

HcclResult GetModuleInfo(DevType devType, const std::vector<RankInfo_t>& rankList, bool& multiModuleDiffDeviceNumMode)
{
    multiModuleDiffDeviceNumMode = false;

    if (devType != DevType::DEV_TYPE_910B || rankList.size() == 0) {
        return HCCL_SUCCESS;
    }

    std::map<u32, std::vector<RankInfo_t>> moduleMap;
    for (RankInfo_t rankInfo : rankList) {
        if (static_cast<s32>(rankInfo.deviceInfo.devicePhyId) == HOST_DEVICE_ID) {
            continue;
        }
        u32 moduleIdx = rankInfo.serverIdx * FACTOR_NUM_TWO + rankInfo.deviceInfo.devicePhyId / DEVICE_PER_MODULE;
        auto iter = moduleMap.find(moduleIdx);
        if (iter == moduleMap.end()) {
            std::vector<RankInfo_t> rankInfoList;
            rankInfoList.push_back(rankInfo);
            moduleMap.insert(std::make_pair(moduleIdx, rankInfoList));
        } else {
            iter->second.push_back(rankInfo);
        }
    }

    // 无NPU参与通信
    if (moduleMap.size() == 0) {
        return HCCL_SUCCESS;
    }
    u32 preDeviceNum = moduleMap.begin()->second.size();
    u32 curDeviceNum = preDeviceNum;
    for (auto moduleInfo : moduleMap) {
        curDeviceNum = moduleInfo.second.size();
        HCCL_DEBUG("[HcomOpUtils][GetModuleInfo] module[%d] contains [%d]devices", moduleInfo.first, curDeviceNum);
        for (auto rankInfo : moduleInfo.second) {
            HCCL_DEBUG(
                "[HcomOpUtils][GetModuleInfo] moduleIdx[%d] Info: rankId[%d], serverId[%s], serverIdx[%d], "
                "devicePhyId[%d]",
                moduleInfo.first, rankInfo.rankId, rankInfo.serverId.c_str(), rankInfo.serverIdx,
                rankInfo.deviceInfo.devicePhyId);
        }
        if (curDeviceNum != preDeviceNum) {
            multiModuleDiffDeviceNumMode = true;
            HCCL_INFO(
                "[HcomOpUtils][GetModuleInfo] different module contains different numbers of cards:[%d]",
                multiModuleDiffDeviceNumMode);
            return HCCL_SUCCESS;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcomCalcOpOnline(HcomOpParam* hcomOpParam, HcomResResponse* hcomResResponse)
{
    HCCL_INFO("[HcomCalcOpOnline] calc online resource start...");
    u64 streamNum = 0;
    u64 opMemSize = 0;
    u32 taskNum = 0;
    HcclResult ret;
    std::string sCollectiveType(hcomOpParam->opType);

    CHK_PTR_NULL(hcomOpParam->socVersion);
    std::string socVersionStr(hcomOpParam->socVersion);
    DevType devType;
    CHK_RET(GetOffDeviceTypeWithoutDev(socVersionStr, devType));

    auto iter = HCCL_OPTYPE_NAME_MAP.find(hcomOpParam->opType);
    HcclCMDType hcclOpType = (iter != HCCL_OPTYPE_NAME_MAP.end()) ? iter->second : HcclCMDType::HCCL_CMD_INVALID;

    u32 dataTypeSize = 0;
    ret = SalGetDataTypeSize(hcomOpParam->dataType, dataTypeSize);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[GetOp][WorkspaceMemSize]op[%s]: get data size failed. ret[%d]", sCollectiveType.c_str(), ret),
        ret);

    CHK_RET(HcomCheckCount(hcomOpParam->count));

    u64 opDataSize = dataTypeSize * hcomOpParam->count;

    CHK_RET(HcomGetWorkspaceSubStreamNum(
        hcomOpParam->group, streamNum, opDataSize, hcomOpParam->dataType, hcomOpParam->aivCoreLimit,
        hcomOpParam->reduceOp, hcomOpParam->count, hcclOpType));
    if (devType == DevType::DEV_TYPE_950 || devType == DevType::DEV_TYPE_960) {
        CHK_RET(HcomGetWorkspaceMemSize(
            hcomOpParam->opType, hcomOpParam->count, hcomOpParam->dataType, hcomOpParam->group, opMemSize));
    } else {
        CHK_RET(GetOpWorkspaceMemSize(false, hcclOpType, hcomOpParam, 0, opMemSize));
    }

    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    u32 serverNum = hcomInfo.rankTable.serverNum;
    u32 deviceNumPerServer = (serverNum == 0) ? 0 : (hcomInfo.rankTable.deviceNum + serverNum - 1) / serverNum;
    HCCL_INFO("get HcomInfo from Context");

    // 获取multiModuleDiffDeviceNumMode信息
    bool multiModuleDiffDeviceNumMode = false;
    ret = GetModuleInfo(devType, hcomInfo.rankTable.rankList, multiModuleDiffDeviceNumMode);
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING("call GetModuleInfo error, failed to get multiModuleDiffDeviceNumMode.");
    }

    if (devType == DevType::DEV_TYPE_950 || devType == DevType::DEV_TYPE_960) {
        CHK_RET(CalcTaskNumV2(hcomOpParam, taskNum));
    } else {
        CHK_RET(CalcTaskNum(
            hcomOpParam, streamNum, deviceNumPerServer, serverNum, multiModuleDiffDeviceNumMode, taskNum, devType));
    }

    hcomResResponse->streamNum = streamNum;
    hcomResResponse->opMemSize = opMemSize;
    hcomResResponse->taskNum = taskNum;

    return HCCL_SUCCESS;
}

HcclResult HcomCalcOpResOffline(HcomOpParam* hcomOpParam, HcomResResponse* hcomResResponse)
{
    HCCL_INFO("[HcomCalcOpResOffline] calc offline resource start...");
    // 获取需回传的信息
    u64 streamNum = 0;
    u64 opMemSize = 0;
    u32 taskNum = 0;

    CHK_PTR_NULL(hcomOpParam->rankTable);
    std::string rankTableString(hcomOpParam->rankTable);

    auto iter = HCCL_OPTYPE_NAME_MAP.find(hcomOpParam->opType);
    HcclCMDType hcclOpType = (iter != HCCL_OPTYPE_NAME_MAP.end()) ? iter->second : HcclCMDType::HCCL_CMD_INVALID;

    CHK_PTR_NULL(hcomOpParam->socVersion);
    std::string socVersionStr(hcomOpParam->socVersion);
    DevType devType;
    CHK_RET(GetOffDeviceTypeWithoutDev(socVersionStr, devType));

    // 先根据逻辑ranktable获取device数量
    s32 deviceNumPerServer = 0;
    s32 serverNum = 0;
    bool multiModuleDiffDeviceNumMode = false;
    RankTable_t clusterInfo;
    // world group 从逻辑ranktable里获取device数量
    if (hcomOpParam->groupListSize == 0) {
        CHK_RET(GetClusterInfoAndDeviceNum(rankTableString, clusterInfo, deviceNumPerServer));
        serverNum = clusterInfo.serverNum;
    } else {
        CHK_RET(GetServerAndDevNumFromGroupList(
            hcomOpParam->groupList, hcomOpParam->groupListSize, rankTableString, devType, serverNum, deviceNumPerServer,
            multiModuleDiffDeviceNumMode));
    }

    if (hcomOpParam->rankSize == 0) {
        hcomOpParam->rankSize = deviceNumPerServer;
    }

    string algName;
    bool ifAiv = false;
    std::shared_ptr<hccl::hcclComm> hcclComm;
    std::string group = hcomOpParam->group == nullptr ? HCCL_WORLD_GROUP : hcomOpParam->group;
    CHK_RET(HcomGetCommByGroup(group.c_str(), hcclComm));
    void* counts = nullptr;
    HcclResult ret = hcclComm->HcclSelectAlg(
        hcclOpType, hcomOpParam->count, counts, hcomOpParam->dataType, hcomOpParam->reduceOp, hcomOpParam->aivCoreLimit,
        ifAiv, algName);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[HcomGetWorkspaceSubStreamNum] HcclSelectAlg failed, ret[%d], optype[%d], count[%llu],"
            "dataType[%d], reduceOp[%d]",
            ret, hcclOpType, hcomOpParam->count, hcomOpParam->dataType, hcomOpParam->reduceOp),
        ret);
    CHK_RET(GetStreamNumOfflineComp(hcclOpType, serverNum, deviceNumPerServer, ifAiv, devType, streamNum, group));
    CHK_RET(GetOpWorkspaceMemSize(true, hcclOpType, hcomOpParam, serverNum, opMemSize));

    if (devType == DevType::DEV_TYPE_950 || devType == DevType::DEV_TYPE_960) {
        // host展开已日落， Task任务数按照当前需求最多的CCU加速模式[AIV, AICPU使用较少]预估
        taskNum = ESTIMATE_CCU_TASK_PER_STREAM;
    }

    hcomResResponse->streamNum = streamNum;
    hcomResResponse->opMemSize = opMemSize;
    hcomResResponse->taskNum = taskNum;

    return HCCL_SUCCESS;
}

HcclResult GetOffDeviceTypeWithoutDev(std::string socVersionStr, DevType& devType)
{
    // 离线编译第一阶段获取devType从SOC_VERSION里获取
    DevType tempDevType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceTypeBySocVersion(socVersionStr, tempDevType));

    if (tempDevType != DevType::DEV_TYPE_910 && tempDevType != DevType::DEV_TYPE_910B
        && tempDevType != DevType::DEV_TYPE_310P1 && tempDevType != DevType::DEV_TYPE_310P3
        && tempDevType != DevType::DEV_TYPE_910_93 && tempDevType != DevType::DEV_TYPE_950
        && tempDevType != DevType::DEV_TYPE_960) {
        HCCL_ERROR("[offline][compilation] cur dev type[%u] is not supported.", tempDevType);
        return HCCL_E_RUNTIME;
    }
    devType = tempDevType;
    HCCL_DEBUG("[offline] Get devtype[%u]....", devType);
    return HCCL_SUCCESS;
}

HcclResult GetStreamNumOfflineComp(
    HcclCMDType hcclOpType, s32 serverNum, s32 deviceNumPerServer, bool ifAiv, DevType devType, u64& streamNum,
    const std::string& group)
{
    switch (devType) {
        case DevType::DEV_TYPE_310P1:
        case DevType::DEV_TYPE_310P3: {
            streamNum = 0;
            break;
        }

        case DevType::DEV_TYPE_910B:
        case DevType::DEV_TYPE_910:
        case DevType::DEV_TYPE_950:
        case DevType::DEV_TYPE_960:
        case DevType::DEV_TYPE_910_93: {
            CHK_RET(
                GetStremNumOfflineByDev(devType, hcclOpType, serverNum, deviceNumPerServer, ifAiv, streamNum, group));
            break;
        }

        default: {
            HCCL_ERROR(
                "[Get][OfflineCompStreamNum] The current device type does not support offline compilation, "
                "The value of device type is [%u]",
                devType);
            return HCCL_E_NOT_SUPPORT;
        }
    }
    HCCL_INFO("[GetStreamNumOfflineComp]stream num is [%llu]", streamNum);
    return HCCL_SUCCESS;
}

HcclResult GetStremNumOfflineByDev(
    const DevType& devType, HcclCMDType hcclOpType, s32 serverNum, s32 deviceNumPerServer, bool ifAiv, u64& streamNum,
    const std::string& group)
{
    if (ifAiv) {
        streamNum = 0; // 离线编译下，从流数量设置为0
        HCCL_INFO("[GetStremNumOfflineByDev] set AIV stream num is 0 When in Aiv mode");
        return HCCL_SUCCESS;
    }

    if (hcclOpType == HcclCMDType::HCCL_CMD_SEND || hcclOpType == HcclCMDType::HCCL_CMD_RECEIVE) {
        streamNum = 0;
        return HCCL_SUCCESS;
    }

    if (devType == DevType::DEV_TYPE_910 && deviceNumPerServer == HCCL_DEVICE_NUM_EIGHT) {
        CHK_RET(GetSubStreamNum(devType, deviceNumPerServer, streamNum, serverNum, group));
    } else if (devType == DevType::DEV_TYPE_910_93) {
        CHK_RET(GetSubStreamNum(devType, deviceNumPerServer, streamNum, serverNum, group));
    } else {
        streamNum = deviceNumPerServer > HCCL_DEVICE_NUM_ONE ? deviceNumPerServer - MINUS_MESH_STREAM_NUM : 0;
    }
    HCCL_INFO(
        "[GetStremNumOfflineByDev] get device num per server is [%u] streamNum [%u]", deviceNumPerServer, streamNum);
    return HCCL_SUCCESS;
}

HcclResult
GetSubStreamNum(const DevType& devType, s32 deviceNum, u64& streamNum, const s32& serverNum, const std::string& group)
{
    if (devType == DevType::DEV_TYPE_910B) {
        constexpr u64 maxStream = 6;
        streamNum = std::min(maxStream, static_cast<u64>(deviceNum) - MINUS_MESH_STREAM_NUM);
        if (CommConfiger::GetInstance().GetCommConfigAlgoConfig(group)[HCCL_ALGO_LEVEL_1]
            == HcclAlgoType::HCCL_ALGO_TYPE_PIPELINE) {
            streamNum = static_cast<u64>(deviceNum);
        }
    } else if (devType == DevType::DEV_TYPE_910_93) {
        if (serverNum == 1) {
            streamNum = static_cast<u64>(deviceNum) - MINUS_MESH_STREAM_NUM;
        } else {
            constexpr u64 streamNumFor91093 = 3;
            streamNum = streamNumFor91093;
        }
    } else {
        if (deviceNum > HCCL_DEVICE_NUM_EIGHT) {
            streamNum = OFFLINE_BUILD_SUB_STEAM_NUM[HCCL_DEVICE_NUM_EIGHT];
        } else if (OFFLINE_BUILD_SUB_STEAM_NUM.count(deviceNum) != 0) {
            streamNum = OFFLINE_BUILD_SUB_STEAM_NUM[deviceNum];
        } else {
            streamNum = 0;
        }
    }

    if (SatisfyIntraSuperPod(devType, deviceNum, true)) {
        streamNum = std::max(static_cast<u64>(deviceNum - 1u), streamNum);
    } else if (FullmeshPairwiseSatisfyHighPerfAlltoallMeshCondition(
                   devType, deviceNum * serverNum, true,
                   CommConfiger::GetInstance().GetCommConfigAlgoConfig(group, HcclCMDType::HCCL_CMD_ALLTOALL))) {
        streamNum = std::max(static_cast<u64>(deviceNum * serverNum - 1u), streamNum);
    }

    HCCL_DEBUG("[GetSubStreamNum]deviceNum[%d], streamNum[%llu]", deviceNum, streamNum);
    return HCCL_SUCCESS;
}

HcclResult GetClusterInfoAndDeviceNum(const std::string rankTableString, RankTable_t& clusterInfo, s32& deviceNum)
{
    HCCL_DEBUG("[get][offlineStreamNum]rankTableString[%s]", rankTableString.c_str());
    TopoinfoRanktableOffline myTopoRanktable(rankTableString);
    CHK_RET(myTopoRanktable.Init());
    CHK_RET(myTopoRanktable.GetClusterInfo(clusterInfo));
    CHK_RET(myTopoRanktable.GetDeviceNumPerServer(deviceNum));
    CHK_PRT_RET(
        deviceNum == 0,
        HCCL_ERROR("[GetStremNumOfflineByDev]cur device num per server is 0, "
                   "maybe ranktable is incomplete"),
        HCCL_E_PARA);
    return HCCL_SUCCESS;
}

HcclResult GetServerAndDevNumFromGroupList(
    const u32* groupList, u32 groupListSize, const std::string rankTableString, DevType devType, s32& serverNum,
    s32& deviceNumPerServer, bool& multiModuleDiffDeviceNumMode)
{
    deviceNumPerServer = 0;
    serverNum = 0;

    if (groupListSize == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(groupList);

    try {
        // 获取并设定stream 数量
        // 能获取到group list时进入离线编译的流程去获取从流个数
        CHK_RET(GetServerAndDevNumFromLogRanktable(
            rankTableString, groupList, groupListSize, devType, serverNum, deviceNumPerServer,
            multiModuleDiffDeviceNumMode));

        HCCL_INFO("deviceNumPerServer:[%d] serverNum:[%d]", deviceNumPerServer, serverNum);
    } catch (const std::exception& e) {
        HCCL_ERROR("[HcomCalcOpRunningParam] exception caught. err[%s]", e.what());
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult GetServerAndDevNumFromLogRanktable(
    const std::string rankTableString, const u32* groupList, u32 groupListSize, DevType devType, s32& serverNum,
    s32& deviceNum, bool& multiModuleDiffDeviceNumMode)
{
    HCCL_INFO("Entry GetServerAndDevNumFromLogRanktable");
    RankTable_t clusterInfo;
    TopoinfoRanktableOffline myTopoRanktable(rankTableString);
    CHK_RET(myTopoRanktable.Init());
    CHK_RET(myTopoRanktable.GetClusterInfo(clusterInfo));

    CHK_RET(GetServerAndDevNumFromRanklist(
        groupList, groupListSize, clusterInfo.rankList, devType, serverNum, deviceNum, multiModuleDiffDeviceNumMode));
    return HCCL_SUCCESS;
}

HcclResult GetServerAndDevNumFromRanklist(
    const u32* groupList, u32 groupListSize, const std::vector<RankInfo_t>& rankList, DevType devType, s32& serverNum,
    s32& deviceNum, bool& multiModuleDiffDeviceNumMode)
{
    u32 serverId = 0;
    std::map<u32, s32> serverAndDevNum;
    deviceNum = 0;
    for (u32 i = 0; i < groupListSize; i++) {
        u32 rankId = groupList[i];
        CHK_RET(GetServerIdByRankId(rankList, rankId, serverId));
        if (serverAndDevNum.find(serverId) == serverAndDevNum.end()) {
            serverAndDevNum[serverId] = 1;
        } else {
            serverAndDevNum[serverId]++;
        }
        // 可能存在不同server内device数量不一致的情况，因此求最大值
        if (serverAndDevNum[serverId] > deviceNum) {
            deviceNum = serverAndDevNum[serverId];
        }
    }
    serverNum = serverAndDevNum.size();

    // 获取multiModuleDiffDeviceNumMode信息
    HcclResult ret = GetModuleInfo(devType, rankList, multiModuleDiffDeviceNumMode);
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING("call GetModuleInfo error, failed to get multiModuleDiffDeviceNumMode.");
    }
    return HCCL_SUCCESS;
}

HcclResult GetServerIdByRankId(const std::vector<RankInfo_t>& rankList, const u32& rankId, u32& serverId)
{
    for (auto& iter : rankList) {
        if (iter.rankId == rankId) {
            serverId = iter.serverIdx;
            return HCCL_SUCCESS;
        }
    }
    return HCCL_E_NOT_FOUND;
}

HcclResult GetOpWorkspaceMemSize(
    bool isOfflineCompilation, HcclCMDType hcclOpType, HcomOpParam* hcomOpParam, s32 serverNum, u64& opMemSize)
{
    HcclResult ret;
    const u32 alignSize = HCCL_ALIGN_SIZE;
    u32 dataTypeSize = 0;
    std::string sCollectiveType(hcomOpParam->opType);

    ret = SalGetDataTypeSize(hcomOpParam->dataType, dataTypeSize);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[GetOp][WorkspaceMemSize]op[%s]: get data size failed. ret[%d]", sCollectiveType.c_str(), ret),
        ret);

    u64 getMemSize = 0;
    u32 rankSize = hcomOpParam->rankSize;
    CHK_RET(GetOpScratchMemSize(
        isOfflineCompilation, hcclOpType, hcomOpParam, getMemSize, dataTypeSize, rankSize, serverNum));

    // 算子所需的内存大小，加上固定32kb长度，并按对齐后回传
    opMemSize = HCCL_WORKSPACE_MEM_32_KB;
    opMemSize += getMemSize;
    opMemSize = (opMemSize + alignSize - 1) / alignSize * alignSize;

    HCCL_INFO(
        "workspace memory size: op[%s], data type[%s], count[%llu], "
        "group[%s], rank size[%u], size[%llu], mem size[%llu].",
        sCollectiveType.c_str(), GetDataTypeEnumStr(hcomOpParam->dataType).c_str(), hcomOpParam->count,
        hcomOpParam->group, rankSize, getMemSize, opMemSize);

    return HCCL_SUCCESS;
}

HcclResult GetOpScratchMemSize(
    bool isOfflineCompilation, HcclCMDType hcclOpType, HcomOpParam* hcomOpParam, u64& opMemSize, u32 dataTypeSize,
    s32 rankSize, s32 serverNum)
{
    CHK_PTR_NULL(hcomOpParam);
    constexpr u8 devType_950 = 6; // 950枚举值为6，需要统一整改
    constexpr u8 devType_960 = 8; // 960枚举值为8
    u64 count = hcomOpParam->count;
    std::string sCollectiveType(hcomOpParam->opType);

    std::string socVersionStr(hcomOpParam->socVersion);
    DevType devType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceTypeBySocVersion(socVersionStr, devType));

    std::shared_ptr<hccl::hcclComm> hcclComm;
    std::string group = hcomOpParam->group == nullptr ? HCCL_WORLD_GROUP : hcomOpParam->group;
    // 获取通信域句柄，因为91095不需要获取通信域句柄以感知aivonly，暂时规避
    if (static_cast<u8>(devType) != devType_950 && static_cast<u8>(devType) != devType_960) {
        CHK_RET(HcomGetCommByGroup(group.c_str(), hcclComm));
    }

    std::vector<HcclAlgoType> algoTypeArr
        = CommConfiger::GetInstance().GetCommConfigAlgoConfig(group, HcclCMDType::HCCL_CMD_ALLTOALLV);
    bool UseOneLayerAlltoAllv
        = (algoTypeArr[0] == HcclAlgoType::HCCL_ALGO_TYPE_NA
           && algoTypeArr[1] == HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE);

    // 是否需要额外申请scratch mem
    if (hcclOpType == HCCL_CMD_REDUCE_SCATTER_V) {
        CHK_RET(GetReduceScatterVScratchMemSize(hcomOpParam, opMemSize));
    } else if (hcclOpType == HCCL_CMD_REDUCE_SCATTER) {
        // ReduceScatter 所需workspace memory: count * 单个数据的size * rank_size
        opMemSize = count * dataTypeSize * rankSize;

        u8 deterministic;
        std::string socVersionStr(hcomOpParam->socVersion);
        DevType devType = DevType::DEV_TYPE_COUNT;
        CHK_RET(hrtGetDeviceTypeBySocVersion(socVersionStr, devType));
        CHK_RET(GetDeterministic(devType, hcomOpParam->geDeterministic, deterministic));
        if (deterministic == DETERMINISTIC_ENABLE || deterministic == DETERMINISTIC_STRICT) {
            const u32 NUM_SIZE_TWO = 2;
            opMemSize *= NUM_SIZE_TWO;
        }
    } else if (hcclOpType == HCCL_CMD_ALLTOALL) {
        // AlltoAll 所需workspace memory ：input mem size
        opMemSize += (count * dataTypeSize * rankSize);
    } else if (hcclOpType == HCCL_CMD_BROADCAST) {
        if (count * dataTypeSize <= HCCL_MID_COUNT_32_MB) {
            opMemSize += count * dataTypeSize * HCCL_MEMSIZE_HD_FACTOR;
        }
    } else if (
        (hcclOpType == HCCL_CMD_ALLTOALLV || hcclOpType == HCCL_CMD_ALLTOALLVC) && !UseOneLayerAlltoAllv
        && static_cast<u32>(rankSize) > HCCL_ALLTOALLV_P2P_SIZE) {
        // 离线编译场景需要重新计算
        if (isOfflineCompilation) {
            if (hcclOpType == HCCL_CMD_ALLTOALLV) {
                HCCL_ERROR("[GetOpScratchMemSize] offline compilation does not support HcomAllToAllV");
                return HCCL_E_PARA;
            }
        }
        s32 deviceLogicId = 0;
        if (!isOfflineCompilation) {
            // 获取deviceLogicID
            CHK_RET(HcomGetDevId(hcomOpParam->group, &deviceLogicId));
            CHK_RET(hrtSetDevice(deviceLogicId));
        }
        if (hcclOpType == HCCL_CMD_ALLTOALLV) {
            CHK_RET(GetAlltoAllvStagedScratchMemSize(hcomOpParam, rankSize, opMemSize));
        } else {
            CHK_RET(GetAlltoAllvcStagedScratchMemSize(hcomOpParam, rankSize, opMemSize));
        }
        if (!isOfflineCompilation) {
            CHK_RET(hrtResetDevice(deviceLogicId));
        }
    } else if (hcclOpType == HCCL_CMD_ALLREDUCE) {
        // 判断 aiv_only
        bool isAivOnlyMode = false;
        u8 deterministic;

        // 91095环境下，暂时不需要感知是否为aivonly模式
        if (static_cast<u8>(devType) != devType_950 && static_cast<u8>(devType) != devType_960) {
            CHK_RET(hcclComm->GetOnlyAivModeConfig(isAivOnlyMode));
        }
        CHK_RET(GetDeterministic(devType, hcomOpParam->geDeterministic, deterministic));

        if (deterministic != DETERMINISTIC_DISABLE) {
            CHK_RET(GetAllReduceScratchMemSize(isOfflineCompilation, hcomOpParam, serverNum, rankSize, opMemSize));
        } else {
            // 数据量大以及aivOnly的情况下需要申请scratch mem
            if (count * dataTypeSize <= HCCL_MID_COUNT_16_MB || isAivOnlyMode) {
                opMemSize += count * dataTypeSize * HCCL_MEMSIZE_HD_FACTOR;
            }
        }
    }

    HCCL_INFO("workspace memory size: op[%s], scratch mem size[%llu]", sCollectiveType.c_str(), opMemSize);
    return HCCL_SUCCESS;
}

HcclResult GetAlltoAllvStagedScratchMemSize(HcomOpParam* hcomOpParam, u32 rankSize, u64& getMemSize)
{
    if (rankSize > ALLTOALLV_RANK_MAX_NUM) {
        HCCL_ERROR("[GetAlltoAllvStagedScratchMemSize] Invalid rankSize[%u]", rankSize);
        return HCCL_E_PARA;
    }
    u64 memSize = 0;

    std::vector<u64> sendCountsUnsigned(rankSize, 0);
    std::vector<u64> sendDisplsUnsigned(rankSize, 0);
    std::vector<u64> recvCountsUnsigned(rankSize, 0);
    std::vector<u64> recvDisplsUnsigned(rankSize, 0);

    for (u32 i = 0; i < rankSize; i++) {
        sendCountsUnsigned[i] = static_cast<u64*>(hcomOpParam->All2AllDataDes.sendCounts)[i];
        sendDisplsUnsigned[i] = static_cast<u64*>(hcomOpParam->All2AllDataDes.sendDispls)[i];
        recvCountsUnsigned[i] = static_cast<u64*>(hcomOpParam->All2AllDataDes.recvCounts)[i];
        recvDisplsUnsigned[i] = static_cast<u64*>(hcomOpParam->All2AllDataDes.recvDispls)[i];
    }

    CHK_RET(HcomGetAlltoAllStagedWorkSpaceMemSize(
        hcomOpParam->group, sendCountsUnsigned.data(), sendDisplsUnsigned.data(), hcomOpParam->All2AllDataDes.sendType,
        recvCountsUnsigned.data(), recvDisplsUnsigned.data(), hcomOpParam->All2AllDataDes.recvType, memSize));

    getMemSize += memSize;

    return HCCL_SUCCESS;
}

HcclResult GetAlltoAllvcStagedScratchMemSize(HcomOpParam* hcomOpParam, u32 rankSize, u64& getMemSize)
{
    if (rankSize > ALLTOALLVC_RANK_MAX_NUM) {
        HCCL_ERROR("[GetAlltoAllvcStagedScratchMemSize] Invalid rankSize[%u]", rankSize);
        return HCCL_E_PARA;
    }
    u64 memSize = 0;
    HcclDataType sendType = hcomOpParam->All2AllDataDes.sendType;
    HcclDataType recvType = hcomOpParam->All2AllDataDes.recvType;

    int64_t* sendCountMatrix = static_cast<int64_t*>(hcomOpParam->All2AllDataDes.sendCountMatrix);

    u32 sendTypeSize = 0, recvTypeSize = 0;
    CHK_RET(SalGetDataTypeSize(sendType, sendTypeSize));
    CHK_RET(SalGetDataTypeSize(recvType, recvTypeSize));
    std::vector<SendRecvInfo> allMeshAggregationSendRecvInfo;
    for (u32 i = 0; i < rankSize; i++) {
        SendRecvInfo sendRecvInfo;
        u64 curSendDispls = 0;
        u64 curRecvDispls = 0;
        for (u32 j = 0; j < rankSize; j++) {
            u64 curSendCounts = sendCountMatrix[i * rankSize + j];
            sendRecvInfo.sendCounts.push_back(curSendCounts);
            sendRecvInfo.sendDispls.push_back(curSendDispls);
            sendRecvInfo.sendLength.push_back(curSendCounts * sendTypeSize);
            sendRecvInfo.sendOffset.push_back(curSendDispls * sendTypeSize);
            HCCL_DEBUG(
                "GetAlltoAllvcStagedScratchMemSize rankID[%u], curSendCounts[%llu], curSendDispls[%llu]", i,
                curSendCounts, curSendDispls);
            curSendDispls += curSendCounts;

            u64 curRecvCounts = sendCountMatrix[i + rankSize * j];
            sendRecvInfo.recvCounts.push_back(curRecvCounts);
            sendRecvInfo.recvDispls.push_back(curRecvDispls);
            sendRecvInfo.recvLength.push_back(curRecvCounts * recvTypeSize);
            sendRecvInfo.recvOffset.push_back(curRecvDispls * recvTypeSize);
            HCCL_DEBUG(
                "GetAlltoAllvcStagedScratchMemSize rankID[%u], curRecvCounts[%llu], curRecvDispls[%llu]", i,
                curRecvCounts, curRecvDispls);
            curRecvDispls += curRecvCounts;
        }
        allMeshAggregationSendRecvInfo.push_back(std::move(sendRecvInfo));
    }

    CHK_RET(HcomGetAlltoAllvcStagedWorkSpaceMemSize(hcomOpParam->group, allMeshAggregationSendRecvInfo, memSize));
    getMemSize += memSize;

    return HCCL_SUCCESS;
}

HcclResult GetReduceScatterVScratchMemSize(HcomOpParam* hcomOpParam, u64& getMemSize)
{
    DevType devType;
    std::string socVerStr(hcomOpParam->socVersion);
    CHK_RET(GetOffDeviceTypeWithoutDev(socVerStr, devType));
    u8 deterministic;
    CHK_RET(GetDeterministic(devType, hcomOpParam->geDeterministic, deterministic));

    const u32 deviceEight = 8;
    const u32 paddingLen = 1024;
    u64 dataTypeSize = SIZE_TABLE[hcomOpParam->dataType];
    u64 ranksize = hcomOpParam->rankSize;
    // 910B 确定性 || 910B 多module
    if (devType == DevType::DEV_TYPE_910B && (deterministic != DETERMINISTIC_DISABLE || ranksize > deviceEight)) {
        u64 maxCount = 0;
        for (u32 i = 0; i < ranksize; i++) {
            // reducescatterv复用HcomOpParam的All2AllDataDes字段
            maxCount = std::max(maxCount, static_cast<u64*>(hcomOpParam->All2AllDataDes.sendCounts)[i]);
        }
        getMemSize = (maxCount * dataTypeSize + paddingLen) * ranksize;
        HCCL_INFO("[GetReduceScatterVScratchMemSize] maxCount[%llu], getMemSize[%llu]", maxCount, getMemSize);
    } else if (devType == DevType::DEV_TYPE_910B && ranksize <= deviceEight) {
        getMemSize = hcomOpParam->count * dataTypeSize * ranksize;
        HCCL_INFO("[GetReduceScatterVScratchMemSize] getMemSize[%llu]", getMemSize);
    } else {
        getMemSize = hcomOpParam->count * dataTypeSize;
    }
    HCCL_DEBUG("[GetReduceScatterVScratchMemSize] rankSize[%llu] getMemSize[%llu]", ranksize, getMemSize);
    return HCCL_SUCCESS;
}

HcclResult GetAllReduceScratchMemSize(
    bool isOfflineCompilation, HcomOpParam* hcomOpParam, s32 serverNum, s32 rankSize, u64& getMemSize)
{
    u64 scratchSize = 0;

    bool no_impl_compile = isOfflineCompilation || hcomOpParam->groupListSize > 0;
    if (no_impl_compile) {
        HcclResult ret = GetAllReduceScratchSizeWithoutDev(hcomOpParam, serverNum, rankSize, scratchSize);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("no_impl_compile [GetAllReduceScratchMemSize] fail ", HCOM_ERROR_CODE(HCCL_E_PARA)),
            HCCL_E_PARA);
    } else {
        CHK_RET(
            HcomGetAllReduceScratchSize(hcomOpParam->group, hcomOpParam->count, hcomOpParam->dataType, scratchSize));
    }
    u64 memSize = SIZE_TABLE[hcomOpParam->dataType] * hcomOpParam->count;

    HCCL_DEBUG(
        "[GetAllReduceScratchMemSize] serverNum=%u, memSize=%llu, rankSize=%u, scratchSize=%llu", serverNum, memSize,
        rankSize, scratchSize);
    getMemSize += scratchSize;

    return HCCL_SUCCESS;
}

HcclResult GetAllReduceScratchSizeWithoutDev(HcomOpParam* hcomOpParam, s32 serverNum, s32 rankSize, u64& scratchSize)
{
    // 检查是否使能确定性计算

    bool supportInlineReduce
        = (hcomOpParam->reduceOp != HCCL_REDUCE_PROD) && (hcomOpParam->dataType != HcclDataType::HCCL_DATA_TYPE_INT64);

    DevType devType;
    std::string socVerStr(hcomOpParam->socVersion);
    CHK_RET(GetOffDeviceTypeWithoutDev(socVerStr, devType));
    bool isAscend910B = (devType == DevType::DEV_TYPE_910B);
    u64 memSize = SIZE_TABLE[hcomOpParam->dataType] * hcomOpParam->count;
    u8 deterministic;
    CHK_RET(GetDeterministic(devType, hcomOpParam->geDeterministic, deterministic));
    if (deterministic != DETERMINISTIC_DISABLE && serverNum <= 1 && isAscend910B && supportInlineReduce) {
        const u32 deviceEight = 8;
        const s32 deviceTwo = 2;

        scratchSize = 0;
        if (serverNum == 0) {
            // 无效serverNum，按最大需求申请
            if (memSize <= HCCL_SMALL_COUNT_GRAPH_64_KB) {
                // 小数据
                scratchSize = memSize * (deviceEight - 1);
            }
        } else if (serverNum == 1 && rankSize > deviceTwo) {
            // 有效serverNum，按实际需求申请
            if (memSize <= HCCL_SMALL_COUNT_GRAPH_64_KB) {
                // 小数据
                if (rankSize == deviceEight) {
                    scratchSize = 0; // Small Count HD
                } else {
                    scratchSize = memSize * (rankSize - 1); // Small Count Reduce+Bcast
                }
            }
        }
    }
    HCCL_DEBUG(
        "[GetAllReduceScratchMemSizeWithoutDev] serverNum[%d], memSize[%llu], rankSize[%d], scratchSize[%llu]",
        serverNum, memSize, rankSize, scratchSize);
    return HCCL_SUCCESS;
}

bool IsNeedCalTaskNum(HcclCMDType opType)
{
    const std::vector<HcclCMDType> hcomNeedCalTaskNumMap
        = {HCCL_CMD_ALLREDUCE, HCCL_CMD_ALLGATHER, HCCL_CMD_REDUCE_SCATTER,
           HCCL_CMD_ALLTOALL,  HCCL_CMD_ALLTOALLV, HCCL_CMD_ALLTOALLVC};
    auto it = std::find(hcomNeedCalTaskNumMap.begin(), hcomNeedCalTaskNumMap.end(), opType);
    return (it != hcomNeedCalTaskNumMap.end()) ? true : false;
}

HcclResult GetDefaultAlgoLevel1(s32 serverNum, AlgTypeLevel1& algType)
{
    u32 num = serverNum;
    if (num >= HCCL_INTER_SERVER_RING_ALGO_MAX_SUPPORT_SERVER_NUM) {
        // server 数为 8 以上：使用 HD 算法
        algType = AlgTypeLevel1::ALG_LEVEL1_HD;
    } else {
        // server 数为 2 的非整数次幂：使用 RING 算法
        // server 数为 2 的整数次幂：使用 HD 算法
        algType
            = (((num & (num - 1)) != 0) || (num == 1)) ? AlgTypeLevel1::ALG_LEVEL1_RING : AlgTypeLevel1::ALG_LEVEL1_HD;
    }

    return HCCL_SUCCESS;
}

HcclResult GetAlgoLevel1(s32 serverNum, std::string& opType, AlgTypeLevel1& algType)
{
    char* mmSysGetEnvValue = nullptr;
    MM_SYS_GET_ENV(MM_ENV_HCCL_ALGO, mmSysGetEnvValue);
    std::string hcclAlgo = (mmSysGetEnvValue != nullptr) ? mmSysGetEnvValue : "EmptyString";
    if (hcclAlgo != "EmptyString") {
        // 删除空格
        std::string rawAlgoConfig = hcclAlgo;
        rawAlgoConfig.erase(std::remove(rawAlgoConfig.begin(), rawAlgoConfig.end(), ' '), rawAlgoConfig.end());

        std::string algoConfig;
        CHK_RET(SplitHcclOpTypeConfig(rawAlgoConfig, opType, algoConfig));

        // 匹配字段"level1:"
        std::string level1 = "level1:";
        std::size_t found = algoConfig.find(level1);
        if ((found == 0) || (found == (algoConfig.length() - level1.size())) || found == std::string::npos) {
            // HCCL_ALGO中"level1:"配置有问题，走默认获取AlgoLevel1方式
            HCCL_WARNING("Level 1 is not configured.");
            CHK_RET(GetDefaultAlgoLevel1(serverNum, algType));
        } else {
            // 截取HCCL_ALGO中"level1:"之后的字段
            std::string remainAlgoConfig = algoConfig.substr(found + level1.size());
            std::string level1AlgoConfig = remainAlgoConfig.substr(0, remainAlgoConfig.find(";"));

            const std::map<std::string, AlgTypeLevel1> hcclAlgoLevel1Map = {
                {"null", AlgTypeLevel1::ALG_LEVEL1_RESERVED},
                {"ring", AlgTypeLevel1::ALG_LEVEL1_WHOLE_RING},
                {"pipeline", AlgTypeLevel1::ALG_LEVEL1_PIPELINE},
                {"fullmesh", AlgTypeLevel1::ALG_LEVEL1_RESERVED},
                {"H-D_R", AlgTypeLevel1::ALG_LEVEL1_HD},
                {"pairwise", AlgTypeLevel1::ALG_LEVEL1_RESERVED},
                {"NHR", AlgTypeLevel1::ALG_LEVEL1_NHR},
                {"NHR_V1", AlgTypeLevel1::ALG_LEVEL1_NHR_V1},
                {"AHC", AlgTypeLevel1::ALG_LEVEL1_AHC},
                {"AHC_BROKE", AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE},
                {"NB", AlgTypeLevel1::ALG_LEVEL1_NB},
                {"NA", AlgTypeLevel1::ALG_LEVEL1_RESERVED},
            };

            auto iterAlgoLevel1 = hcclAlgoLevel1Map.find(level1AlgoConfig);
            if (iterAlgoLevel1 == hcclAlgoLevel1Map.end()) {
                HCCL_ERROR(
                    "[GetAlgoLevel1] algo config is invalid, level %s is not supported.", level1AlgoConfig.c_str());
                return HCCL_E_PARA;
            }

            algType = iterAlgoLevel1->second;
            if (algType == AlgTypeLevel1::ALG_LEVEL1_RESERVED) {
                CHK_RET(GetDefaultAlgoLevel1(serverNum, algType));
            }
        }
    } else {
        CHK_RET(GetDefaultAlgoLevel1(serverNum, algType));
    }

    HCCL_INFO("[GetAlgoLevel1] level1[%u].", algType);
    auto iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType);
    CHK_PRT_RET(
        iter == HCCL_ALGO_LEVEL1_NAME_MAP.end(), HCCL_ERROR("level1: algType[%u] is invalid.", algType),
        HCCL_E_INTERNAL);
    HCCL_DEBUG(
        "hccl algorithm: there are %d server in level1,"
        " the algorithm for setting environment variables is %s algo.",
        serverNum, iter->second.c_str());
    return HCCL_SUCCESS;
}

HcclResult
SplitHcclOpTypeConfig(const std::string& algoConfig, const std::string& opType, std::string& specificAlgoConfig)
{
    // 对algoConfig进行/切分
    std::size_t foundSeparator = algoConfig.find("/");
    if ((foundSeparator == algoConfig.length() - 1) || (foundSeparator == 0)) {
        HCCL_ERROR("[HcomOpUtils][SplitHcclOpType]algo config is invalid at split sign.");
        return HCCL_E_PARA;
    } else if (foundSeparator == std::string::npos) {
        specificAlgoConfig = algoConfig;
        return HCCL_SUCCESS;
    }

    std::string remainAlgoConfig = algoConfig.substr(foundSeparator + 1);
    std::string currentConfig = algoConfig.substr(0, foundSeparator);
    std::size_t foundEqual = currentConfig.find("=");
    if ((foundEqual == algoConfig.length() - 1) || (foundEqual == 0) || (foundEqual == std::string::npos)) {
        HCCL_ERROR("[HcomOpUtils][SplitHcclOpType]algo config is invalid at equal sign.");
        return HCCL_E_PARA;
    }

    std::string currentOpType = currentConfig.substr(0, foundEqual);
    if (currentOpType == opType) {
        specificAlgoConfig = currentConfig;
        return HCCL_SUCCESS;
    }

    if (!remainAlgoConfig.empty()) {
        CHK_RET(SplitHcclOpTypeConfig(remainAlgoConfig, opType, specificAlgoConfig));
    }
    return HCCL_SUCCESS;
}

HcclResult GetDefaultAlgoLevel0Module(s32 deviceNumPerServer, AlgTypeLevel0& algType, std::string soc_version)
{
    if (soc_version == "Ascend910B") {
        algType = AlgTypeLevel0::ALG_LEVEL0_8P_RING;
    } else if (deviceNumPerServer == TASK_NUM_DEVICE_FOUR) {
        algType = AlgTypeLevel0::ALG_LEVEL0_4P_MESH;
    } else {
        algType = AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING;
    }
    auto iter = HCCL_ALGO_LEVEL0_NAME_MAP.find(algType);
    CHK_PRT_RET(
        iter == HCCL_ALGO_LEVEL0_NAME_MAP.end(), HCCL_ERROR("level0: algType[%u] is invalid.", algType),
        HCCL_E_INTERNAL);
    HCCL_DEBUG(
        "hccl algorithm: [Module(aiserver)] there are %d device in level0, using %s algo.", deviceNumPerServer,
        iter->second.c_str());
    return HCCL_SUCCESS;
}

HcclResult
GetAlgType(s32 deviceNumPerServer, s32 serverNum, std::string opType, std::string socVersionStr, AlgType& algType)
{
    AlgTypeLevel0 algType0;
    AlgTypeLevel1 algType1;

    // 因为非标卡计算出来的task num比标卡场景多，因此task num精确评估暂不区分标卡和非标卡
    CHK_RET(GetDefaultAlgoLevel0Module(deviceNumPerServer, algType0, socVersionStr));
    CHK_RET(GetAlgoLevel1(serverNum, opType, algType1));
    algType.algoLevel0 = algType0;
    algType.algoLevel1 = algType1;
    HCCL_INFO("average device count [%d], algorithm type [%u] is selected.", deviceNumPerServer, algType.algoLevel0);
    return HCCL_SUCCESS;
}

HcclResult GetDfxTaskNum(const std::string& sCollectiveType, u32& taskNum)
{
    if (sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLREDUCE) {
        taskNum += DFX_PADDING_TASK_NUM;
    }
    taskNum += DFX_DEFAULT_TASK_NUM;
    HCCL_DEBUG("[GetDfxTaskNum] cur task num[%u].", taskNum);
    return HCCL_SUCCESS;
}

HcclResult
GetToSlaveStreamTaskNum(const std::string& sCollectiveType, u64 streamNum, u64 pipelineSliceNum, u32& taskNum)
{
    u32 taskNumTmp = 0;
    if (sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLREDUCE) {
        taskNumTmp = streamNum * MASTER_STREAM_EVENT_NUM * COM_STEP_NUM;
    } else {
        taskNumTmp = streamNum * MASTER_STREAM_EVENT_NUM;
    }
    if (pipelineSliceNum >= MIN_PIPELINE_SLICE_NUM) {
        taskNumTmp += pipelineSliceNum * PIPELINE_STREAM_EVENT_NUM * COM_STEP_NUM;
    }
    taskNum += taskNumTmp;
    HCCL_DEBUG("[GetToSlaveStreamTaskNum] cur task num[%u].", taskNum);
    return HCCL_SUCCESS;
}

HcclResult GetToMasterStreamTaskNum(const std::string& sCollectiveType, u32& taskNum)
{
    if (sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLREDUCE) {
        taskNum += (SLAVE_STREAM_EVENT_NUM * COM_STEP_NUM);
    } else {
        taskNum += SLAVE_STREAM_EVENT_NUM;
    }
    HCCL_DEBUG("[GetToMasterStreamTaskNum] cur task num[%u].", taskNum);
    return HCCL_SUCCESS;
}

HcclResult GetCombineComTaskNum(
    const std::string& sCollectiveType, s32 serverNum, s32 deviceNumPerServer, u32& intraTaskNum, u32& interTaskNum)
{
    // 打平拓扑server内通信task数量为0
    intraTaskNum = 0;

    interTaskNum = 0;
    u32 commStep = deviceNumPerServer * serverNum - 1; // 默认根据ring算法评估

    // 计算通信task的数量
    if (sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLREDUCE) {
        interTaskNum = ALLREDUCE_DEFAULT_COM_STEP * commStep;
    } else if (sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLGATHER) {
        interTaskNum = ALLGATHER_DEFAULT_COM_STEP * commStep;
    } else if (sCollectiveType == HCCL_KERNEL_OP_TYPE_REDUCESCATTER) {
        interTaskNum = REDUCESCATTER_DEFAULT_COM_STEP * commStep;
    } else if (
        sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLTOALL || sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLTOALLV
        || sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLTOALLVC) {
        interTaskNum = ALLTOALL_DEFAULT_COM_STEP * commStep;
    } else {
        HCCL_ERROR(
            "[HcomOpUtils][GetCombineComTaskNum]The current operator [%s] do not support tasknum "
            "accurate evaluation.",
            sCollectiveType.c_str());
        return HCCL_E_NOT_SUPPORT;
    }

    HCCL_INFO(
        "[HcomOpUtils][GetCombineComTaskNum]op[%s], cur intraTaskNum is[%u], interTaskNum is[%u], commStep[%u].",
        sCollectiveType.c_str(), intraTaskNum, interTaskNum, commStep);
    return HCCL_SUCCESS;
}

HcclResult GetIntraComTaskNum(
    const std::string& sCollectiveType, s32 deviceNumPerServer, [[maybe_unused]] u64 streamNum, const AlgType& algType,
    u32& taskNum, u64 totalSize)
{
    taskNum = 0;
    u32 commStep = 0;
    u32 commStepDeter = 0;

    // 获取通信步骤
    if (algType.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_8P_RING) {
        commStep += ALG_8P_RING_COMM_STEP;
        commStepDeter = commStep;
    } else if (algType.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING) {
        commStep += (deviceNumPerServer - 1);
        commStepDeter = commStep;
    } else if (algType.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_4P_MESH || deviceNumPerServer > TASK_NUM_DEVICE_ONE) {
        // 涉及确定性计算的通信步骤单独计算
        if (totalSize <= HCCL_SMALL_COUNT_GRAPH_64_KB) {
            commStepDeter += (deviceNumPerServer - 1);
        } else {
            commStepDeter
                += (GetExternalInputHcclDeterministicV2() != DETERMINISTIC_DISABLE ?
                        ((deviceNumPerServer - 1) * (deviceNumPerServer - 1)) :
                        (deviceNumPerServer - 1));
        }
        commStep += (deviceNumPerServer - 1);
    }
    // 计算通信task的数量
    if (sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLREDUCE) {
        taskNum = REDUCESCATTER_DEFAULT_COM_STEP * commStepDeter + ALLGATHER_DEFAULT_COM_STEP * commStep;
    } else if (sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLGATHER) {
        taskNum = ALLGATHER_DEFAULT_COM_STEP * commStep;
    } else if (sCollectiveType == HCCL_KERNEL_OP_TYPE_REDUCESCATTER) {
        taskNum = REDUCESCATTER_DEFAULT_COM_STEP * commStepDeter;
    } else if (
        sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLTOALL || sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLTOALLV
        || sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLTOALLVC) {
        commStep = deviceNumPerServer - 1; // 按照pairwise计算server内通信步数
        taskNum = ALLTOALL_DEFAULT_COM_STEP * commStep;
    } else {
        HCCL_ERROR("The current operator does not support accurate tasknum evaluation.");
        return HCCL_E_NOT_SUPPORT;
    }
    HCCL_INFO(
        "[GetIntraComTaskNum] op[%s], cur tasknum is[%u], commStep[%u], totalSize[%llu]", sCollectiveType.c_str(),
        taskNum, commStep, totalSize);
    return HCCL_SUCCESS;
}

HcclResult GetBetweenServersStep(s32 serverNum, u32& commStep)
{
    if ((static_cast<u32>(serverNum) & (static_cast<u32>(serverNum) - 1U)) == 0U) {
        // 如果serverNum是2的整数次幂，使用HD算法评估CollectiveOp的taskNum
        commStep += SalLog2(serverNum);
    } else if (serverNum < SERVER_NUM_EIGHT) {
        // 如果serverNum是2的非整数次幂并且小于8,使用ring算法评估CollectiveOp的taskNum
        commStep += (serverNum - 1);
    } else {
        // 计算大于serverNum的最大2的整数次幂的值;以N为rankSize, 使用HD算法评估CollectiveOp的taskNum
        s32 bit = 0;
        while (serverNum > 0) {
            serverNum = static_cast<s32>(static_cast<u32>(serverNum) >> 1U);
            bit++;
        }
        commStep += bit;
    }
    HCCL_DEBUG("Get BetweenServers Step [%u]", commStep);
    return HCCL_SUCCESS;
}

HcclResult GetInterComTaskNum(
    const std::string& sCollectiveType, s32 serverNum, s32 deviceNumPerServer, DevType devType, u32& taskNum,
    const std::string& group)
{
    taskNum = 0;
    u32 commStep = 0;

    // 获取server间通信步骤
    if (serverNum > SERVER_NUM_ONE) {
        CHK_RET(GetBetweenServersStep(serverNum, commStep)); // 默认情况下根据serverNum按ring或HD算法评估
        // 计算通信task的数量
        if (sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLREDUCE) {
            taskNum = ALLREDUCE_DEFAULT_COM_STEP * commStep;
        } else if (sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLGATHER) {
            taskNum = ALLGATHER_DEFAULT_COM_STEP * commStep;
        } else if (sCollectiveType == HCCL_KERNEL_OP_TYPE_REDUCESCATTER) {
            taskNum = REDUCESCATTER_DEFAULT_COM_STEP * commStep;
        } else if (
            sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLTOALL || sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLTOALLV
            || sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLTOALLVC) {
            std::vector<HcclAlgoType> algoTypeArr = CommConfiger::GetInstance().GetCommConfigAlgoConfig(group);
            bool useOneLevelAlgorithm
                = (algoTypeArr[0] == HcclAlgoType::HCCL_ALGO_TYPE_NA
                   && algoTypeArr[1] == HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE);
            s32 meshNum = (devType == DevType::DEV_TYPE_910) ? serverNum * 2 : serverNum;
            commStep = useOneLevelAlgorithm ? ((meshNum - 1) * deviceNumPerServer) : (meshNum - 1);
            taskNum = ALLTOALL_DEFAULT_COM_STEP * commStep;
        } else {
            HCCL_ERROR("The current operator does not support accurate tasknum evaluation.");
            return HCCL_E_NOT_SUPPORT;
        }
    }
    HCCL_INFO(
        "[GetInterComTaskNum]op[%s], cur tasknum is[%u], commStep[%u].", sCollectiveType.c_str(), taskNum, commStep);
    return HCCL_SUCCESS;
}

HcclResult CalcTaskNum(
    HcomOpParam* hcomOpParam, const u64& streamNum, const s32& deviceNumPerServer, const s32& serverNum,
    bool multiModuleDiffDeviceNumMode, u32& taskNum, DevType devType)
{
    u32 masterTaskNum = 0;
    u32 slaveTaskNum = 0;
    u32 pipelineTaskNum = 0;

    std::string sCollectiveType(hcomOpParam->opType);

    HcclResult ret;
    HcclUs startut = TIME_NOW();

    auto iter = HCCL_OPTYPE_NAME_MAP.find(hcomOpParam->opType);
    HcclCMDType hcclOpType = (iter != HCCL_OPTYPE_NAME_MAP.end()) ? iter->second : HcclCMDType::HCCL_CMD_INVALID;

    string algName;
    bool ifAiv = false;
    std::shared_ptr<hccl::hcclComm> hcclComm;
    // 获取通信域句柄
    std::string group = hcomOpParam->group == nullptr ? HCCL_WORLD_GROUP : hcomOpParam->group;
    CHK_RET(HcomGetCommByGroup(group.c_str(), hcclComm));
    // 判断是否是AIV场景
    void* counts = nullptr;
    ret = hcclComm->HcclSelectAlg(
        hcclOpType, hcomOpParam->count, counts, hcomOpParam->dataType, hcomOpParam->reduceOp, hcomOpParam->aivCoreLimit,
        ifAiv, algName);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[HcomGetWorkspaceSubStreamNum] HcclSelectAlg failed, ret[%d], optype[%d], count[%llu],"
            "dataType[%d], reduceOp[%d]",
            ret, hcomOpParam->opType, hcomOpParam->count, hcomOpParam->dataType, hcomOpParam->reduceOp),
        ret);
    HCCL_INFO(
        "[%s] HcclSelectAlg success ifAiv[%d] algName[%s] optype[%d] count[%llu] dataType[%d] reduceOp[%d]", __func__,
        ifAiv, algName.c_str(), hcomOpParam->opType, hcomOpParam->count, hcomOpParam->dataType, hcomOpParam->reduceOp);
    // AIV和非rdma场景下，task数量固定
    if (ifAiv && algName.find("Rdma") == std::string::npos) {
        taskNum = AIV_DEFAULT_TASK_NUM;
        HCCL_INFO("[%s] GetAndSetTaskNum success taskNum[%u]", __func__, taskNum);
        return HCCL_SUCCESS;
    }

    if (!IsNeedCalTaskNum(hcclOpType)) {
        if (hcclOpType == HCCL_CMD_SEND || hcclOpType == HCCL_CMD_RECEIVE) {
            taskNum = SEND_RECEIVE_TASK_NUM;
        } else {
            taskNum = OP_DEFAULT_TASK_NUM;
        }
    } else {
        AlgType algType;
        std::string socVersionStr(hcomOpParam->socVersion);

        // 获取通信算法
        CHK_RET(GetAlgType(deviceNumPerServer, serverNum, hcomOpParam->opType, socVersionStr, algType));

        // 如果在线编译没有获取到ranktable file,则返回默认task数量
        if ((deviceNumPerServer == 0) && (serverNum == 0)) {
            taskNum = OP_DEFAULT_TASK_NUM;
        } else {
            // 计算Server间pipeline切分数量
            u32 dataTypeSize;
            u64 totalSize = 0;
            ret = SalGetDataTypeSize(hcomOpParam->dataType, dataTypeSize);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[Get][OpWorkspaceMemSize]op[%s]: get data size failed. ret[%d]", sCollectiveType.c_str(), ret),
                ret);

            totalSize = hcomOpParam->count * dataTypeSize;

            u64 pipelineSliceNum
                = CalculatePiplineSliceNum(hcclOpType, totalSize, algType, devType, deviceNumPerServer, serverNum);

            // 计算DFX校验task数量
            CHK_RET(GetDfxTaskNum(sCollectiveType, masterTaskNum));
            // 计算与从stream同步task数量
            CHK_RET(GetToSlaveStreamTaskNum(sCollectiveType, streamNum, pipelineSliceNum, masterTaskNum));
            // 计算与主stream同步task数量
            CHK_RET(GetToMasterStreamTaskNum(sCollectiveType, slaveTaskNum));
            // 计算Server间Pipeline从stream和主stream同步的task数量
            pipelineTaskNum += (pipelineSliceNum >= MIN_PIPELINE_SLICE_NUM) ?
                                   pipelineSliceNum * PIPELINE_STREAM_EVENT_NUM * COM_STEP_NUM :
                                   0;

            u32 intraTaskNum = 0;
            u32 interTaskNum = 0;
            // 获取Server内通信task数量

            if (multiModuleDiffDeviceNumMode) {
                // 获取打平拓扑通信task数量
                CHK_RET(
                    GetCombineComTaskNum(sCollectiveType, serverNum, deviceNumPerServer, intraTaskNum, interTaskNum));
            } else {
                CHK_RET(GetIntraComTaskNum(
                    sCollectiveType, deviceNumPerServer, streamNum, algType, intraTaskNum, totalSize));
                // 获取Server间通信task数量, 从stream没有server间task
                std::string group = hcomOpParam->group == nullptr ? HCCL_WORLD_GROUP : hcomOpParam->group;
                CHK_RET(
                    GetInterComTaskNum(sCollectiveType, serverNum, deviceNumPerServer, devType, interTaskNum, group));
            }

            // 计算通信task
            if (pipelineSliceNum >= MIN_PIPELINE_SLICE_NUM) {
                masterTaskNum += intraTaskNum * pipelineSliceNum;
                slaveTaskNum += intraTaskNum * pipelineSliceNum;
                pipelineTaskNum += interTaskNum * pipelineSliceNum;
            } else {
                masterTaskNum += intraTaskNum + interTaskNum;
                slaveTaskNum += intraTaskNum;
            }
        }
    }
    if (taskNum == 0) {
        taskNum = std::max(masterTaskNum, std::max(slaveTaskNum, pipelineTaskNum));
    }

    HCCL_INFO("GetAndSetTaskNum success, cost time[%lld]us taskNum[%u]", DURATION_US(TIME_NOW() - startut), taskNum);
    return HCCL_SUCCESS;
}

HcclResult CalcTaskNumV2(HcomOpParam* hcomOpParam, u32& taskNum)
{
    HcclUs startut = TIME_NOW();

    auto iter = HCCL_OPTYPE_NAME_MAP.find(hcomOpParam->opType);
    HcclCMDType hcclOpType = (iter != HCCL_OPTYPE_NAME_MAP.end()) ? iter->second : HcclCMDType::HCCL_CMD_INVALID;

    if (!IsNeedCalTaskNum(hcclOpType)) {
        if (hcclOpType == HCCL_CMD_SEND || hcclOpType == HCCL_CMD_RECEIVE) {
            taskNum = SEND_RECEIVE_TASK_NUM;
        } else {
            taskNum = OP_DEFAULT_TASK_NUM;
        }
    } else {
        CHK_RET(HcomCalcTaskNum(hcomOpParam, taskNum));
    }

    HCCL_INFO("GetAndSetTaskNum success, cost time[%lld]us taskNum[%u]", DURATION_US(TIME_NOW() - startut), taskNum);
    return HCCL_SUCCESS;
}

HcclResult HcomGetMemType(
    const char* group, const char* socVersion, bool isMalloc, u32* memType, bool* isTsMem, bool withoutImplCompile,
    bool level2Address)
{
    DevType devType = DevType::DEV_TYPE_COUNT;
    std::string socVersionStr(socVersion);
    const u32 NUM_SIZE_TWO = 2;

    CHK_RET(hrtGetDeviceTypeBySocVersion(socVersionStr, devType));

    if (isMalloc) {
        if (Is310PDevice()) {
            if (devType == DevType::DEV_TYPE_310P3 || devType == DevType::DEV_TYPE_310P1) {
                if (level2Address) { // 310P二级地址刷新时申请内存类型为：RT_MEMORY_TS
                    *isTsMem = true;
                    *memType = static_cast<int>(ACL_MEM_TYPE_LOW_BAND_WIDTH);
                } else {
                    *memType = static_cast<int>(ACL_MEM_TYPE_LOW_BAND_WIDTH);
                }
            } else {
                *memType = static_cast<int>(ACL_MEM_TYPE_HIGH_BAND_WIDTH);
            }
        } else {
            if (devType == DevType::DEV_TYPE_310P3) {
                if (level2Address) { // 310P二级地址刷新时申请内存类型为：RT_MEMORY_TS
                    *isTsMem = true;
                    *memType = static_cast<int>(ACL_MEM_TYPE_LOW_BAND_WIDTH)
                               | static_cast<int>(ACL_MEM_MALLOC_NORMAL_ONLY_P2P);
                } else {
                    *memType = static_cast<int>(ACL_MEM_TYPE_LOW_BAND_WIDTH)
                               | static_cast<int>(ACL_MEM_MALLOC_NORMAL_ONLY_P2P);
                }
            } else if (devType == DevType::DEV_TYPE_310P1) {
                *memType = static_cast<int>(ACL_MEM_TYPE_LOW_BAND_WIDTH);
            } else {
                *memType
                    = static_cast<int>(ACL_MEM_TYPE_HIGH_BAND_WIDTH) | static_cast<int>(ACL_MEM_MALLOC_NORMAL_ONLY_P2P);
            }
        }
        return HCCL_SUCCESS;
    }

    if (devType == DevType::DEV_TYPE_310P3 || devType == DevType::DEV_TYPE_310P1) {
        u32 numHccsLink = 0;
        u32 rankSize = 0;
        if (!withoutImplCompile) {
            CHK_RET(HcomGetRankSize(group, &rankSize));
            CHK_RET(HcomGetHccsLinkNum(group, &numHccsLink));
        }
        if ((withoutImplCompile || !(rankSize == NUM_SIZE_TWO && numHccsLink == NUM_SIZE_TWO))) {
            // 所有形态切换子包后，改用acl_mem类型
            *memType = RT_MEMORY_P2P_DDR;
        }
    }

    return HCCL_SUCCESS;
}

HcclResult GetDeterministic(DevType devType, u8 geDetOption, u8& deterministic)
{
    deterministic = DETERMINISTIC_DISABLE; // 默认为不支持

    char* mmSysGetEnvValue = nullptr;
    MM_SYS_GET_ENV(MM_ENV_HCCL_DETERMINISTIC, mmSysGetEnvValue);
    std::string hcclDeterministicEnv = (mmSysGetEnvValue != nullptr) ? mmSysGetEnvValue : "EmptyString";
    if (hcclDeterministicEnv != "EmptyString") {
        // 环境变量优先
        std::transform(
            hcclDeterministicEnv.begin(), hcclDeterministicEnv.end(), hcclDeterministicEnv.begin(), ::toupper);
        if (hcclDeterministicEnv == "FALSE") {
            deterministic = DETERMINISTIC_DISABLE;
        } else if (hcclDeterministicEnv == "TRUE") {
            deterministic = DETERMINISTIC_ENABLE;
        } else if (hcclDeterministicEnv == "STRICT") {
            CHK_PRT_RET(
                devType != DevType::DEV_TYPE_910B && devType != DevType::DEV_TYPE_910_93,
                HCCL_ERROR(
                    "ParserHcclDeterministic: reduce order preservation is not supported for devType[%d]", devType),
                HCCL_E_NOT_SUPPORT);
            deterministic = DETERMINISTIC_STRICT;
        } else {
            HCCL_ERROR(
                "[GetDeterministic] HCCL_DETERMINISTIC is set to [%s], which is incorrect, expected values: 0 or 1. "
                "Please check",
                hcclDeterministicEnv.c_str());
            return HCCL_E_PARA;
        }
    } else {
        // 未配环境变量，检查ge option
        if (geDetOption == 1) {
            deterministic = DETERMINISTIC_ENABLE;
        } else if (geDetOption == 2) {
            CHK_PRT_RET(
                devType != DevType::DEV_TYPE_910B && devType != DevType::DEV_TYPE_910_93,
                HCCL_ERROR(
                    "ParserHcclDeterministic: reduce order preservation is not supported for devType[%d]", devType),
                HCCL_E_NOT_SUPPORT);
            deterministic = DETERMINISTIC_STRICT;
        }
    }

    return HCCL_SUCCESS;
}

HcclResult HcomGenerateCclOpTag(const char* opType, s64 hcomComm, const char* group, char* sTag)
{
    CHK_PTR_NULL(group);
    std::string groupName(group);
    std::string tag;
    GenerateCclOpTag(opType, hcomComm, groupName, tag);
    int32_t sret = memcpy_s(sTag, CCL_OP_TAG_MAX_LEN, tag.c_str(), (tag.length() + 1));
    CHK_PRT_RET(
        sret != EOK,
        HCCL_ERROR(
            "[HcomGenerateCclOpTag][Tag]memcpy failed. ret[%d],"
            "params:destMaxSize[%zu],count[%zu]",
            sret, CCL_OP_TAG_MAX_LEN, (tag.length() + 1)),
        HCCL_E_PARA);
    return HCCL_SUCCESS;
}

void HcomSetDumpDebugMode(const bool dumpDebug) { SetDumpDebugMode(dumpDebug); }

void HcomSetLaunchKernelMode(bool state) { SetLaunchKernelMode(state); }

HcclResult
HcomTbeMemClean(int64_t addrList[], int64_t sizeList[], uint32_t count, aclrtStream stream, int32_t deviceLogicId)
{
    CHK_RET(HcclTbeMemClean(addrList, sizeList, count, stream, deviceLogicId));
    return HCCL_SUCCESS;
}
