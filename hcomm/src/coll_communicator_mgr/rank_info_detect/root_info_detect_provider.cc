/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "root_info_detect_bridge.h"

#include <memory>

#include "acl/acl_rt.h"
#include "hccl_common_v2.h"
#include "log.h"
#include "orion_adapter_rts.h"
#include "rank_info_detect.h"
#include "securec.h"

namespace Hccl {
namespace {

    HcclResult GetCurrentDeviceIds(s32& userDevId, s32& devPhyId)
    {
        userDevId = HrtGetDevice();
        aclError aclRet = aclrtGetPhyDevIdByUserDevId(userDevId, &devPhyId);
        CHK_PRT_RET(
            aclRet != ACL_SUCCESS,
            HCCL_ERROR("[%s] aclrtGetPhyDevIdByUserDevId failed, userDevId[%d], ret[%d]", __func__, userDevId, aclRet),
            HCCL_E_INTERNAL);
        return HCCL_SUCCESS;
    }

    // Provider 编译进 hcomm；hccl_v2 中保留的兼容入口只通过 RootInfoDetectBridge 转发到这里。
    HcclResult GetRootInfoImpl(HcclRootInfo* rootInfo)
    {
        HCCL_RUN_INFO("Entry-HcclGetRootInfo V950");
        CHK_PTR_NULL(rootInfo);
        HcclUs startut = TIME_NOW();

        HcclRootHandleV2 rootHandle{};
        std::shared_ptr<RankInfoDetect> rankInfoDetectServer;
        EXCEPTION_CATCH((rankInfoDetectServer = std::make_shared<RankInfoDetect>()), return HCCL_E_MEMORY);
        TRY_CATCH_RETURN(rankInfoDetectServer->SetupServer(rootHandle));

        u32 rootHandleLen = sizeof(HcclRootHandleV2);
        CHK_PRT_RET(
            rootHandleLen > HCCL_ROOT_INFO_BYTES,
            HCCL_ERROR(
                "[%s] hccl root info overflow. max length: %u, actual:%zu, identifier[%s]", __func__,
                HCCL_ROOT_INFO_BYTES, rootHandleLen, rootHandle.identifier),
            HCCL_E_INTERNAL);

        // 对外 HcclRootInfo 仍使用固定大小的不透明缓冲区，避免迁移实现后改变既有接口布局。
        s32 sRet = memcpy_s(rootInfo->internal, HCCL_ROOT_INFO_BYTES, &rootHandle, rootHandleLen);
        CHK_PRT_RET(
            sRet != EOK,
            HCCL_ERROR(
                "[%s] memcpy root info fail. errno[%d] params:destMaxSize[%u], count[%u]", __func__, sRet,
                HCCL_ROOT_INFO_BYTES, rootHandleLen),
            HCCL_E_MEMORY);

        s32 userDevId = 0;
        s32 devPhyId = 0;
        CHK_RET(GetCurrentDeviceIds(userDevId, devPhyId));
        HCCL_RUN_INFO(
            "HcclGetRootInfoV2 success, take time [%lld]us, rootinfo: host ip[%s] port[%u] "
            "netMode[%s] identifier[%s], userDevId[%d], devPhyId[%d]",
            DURATION_US(TIME_NOW() - startut), rootHandle.ip, rootHandle.listenPort,
            rootHandle.netMode.Describe().c_str(), rootHandle.identifier, userDevId, devPhyId);
        return HCCL_SUCCESS;
    }

    HcclResult DetectRankTableImpl(
        u32 nRanks, u32 rank, const HcclRootHandleV2& rootHandle, RankTableInfo& rankTable,
        RootInfoDetectBridge::DetectContext& detectContext)
    {
        s32 userDevId = 0;
        s32 devPhyId = 0;
        CHK_RET(GetCurrentDeviceIds(userDevId, devPhyId));
        HCCL_RUN_INFO(
            "[%s] nRanks[%u], rank[%u] entry flat topo detect, rootinfo: host ip[%s] port[%u] "
            "netMode[%s] identifier[%s], userDevId[%d], devPhyId[%d]",
            __func__, nRanks, rank, rootHandle.ip, rootHandle.listenPort, rootHandle.netMode.Describe().c_str(),
            rootHandle.identifier, userDevId, devPhyId);

        std::shared_ptr<RankInfoDetect> rankInfoDetectAgent;
        EXCEPTION_CATCH((rankInfoDetectAgent = std::make_shared<RankInfoDetect>()), return HCCL_E_MEMORY);

        EXCEPTION_CATCH(rankInfoDetectAgent->SetupAgent(nRanks, rank, rootHandle), return HCCL_E_INTERNAL);
        EXCEPTION_CATCH(
            rankInfoDetectAgent->WaitComplete(rootHandle.listenPort, RANKINFO_DETECT_SERVER_STATUS_IDLE),
            return HCCL_E_INTERNAL);

        rankInfoDetectAgent->GetRankTable(rankTable);
        // bridge 不暴露 RankInfoDetect 类型；调用方持有类型擦除后的 shared_ptr，
        // 使 agent 及其探测资源在后续通信域初始化完成前保持有效。
        detectContext = rankInfoDetectAgent;
        HCCL_RUN_INFO("[%s] end.", __func__);
        return HCCL_SUCCESS;
    }

    const RootInfoDetectBridge ROOT_INFO_DETECT_BRIDGE = {GetRootInfoImpl, DetectRankTableImpl};

    // hcomm 链接 hccl_v2，并在自身装载阶段将 provider 回调注册到 hccl_v2 的槽位。
    // 这样兼容入口保持原符号不变，同时不再直接依赖 RankInfoDetect 实现类。
    struct RootInfoDetectBridgeRegistrar {
        RootInfoDetectBridgeRegistrar()
        {
            // 静态构造函数无法向加载方返回错误；重复注册由 bridge 侧日志记录。
            (void)RegisterRootInfoDetectBridge(ROOT_INFO_DETECT_BRIDGE);
        }
    };

    RootInfoDetectBridgeRegistrar g_rootInfoDetectBridgeRegistrar;

} // namespace
} // namespace Hccl
