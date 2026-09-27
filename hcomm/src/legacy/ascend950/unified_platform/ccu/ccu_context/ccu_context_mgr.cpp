/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_ctx_mgr.h"
#include "ins_exe_que.h"
#include "ccu_context_mgr_imp.h"
#include "hccl_common_v2.h"
#include "exception_util.h"

namespace Hccl {
/**
 * @brief 分配资源
 *
 * @param deviceLogicId device逻辑ID
 * @param ctxGroup CCU上下文组
 * @param resPack 资源包
 *
 * @return 返回HcclResult，表示分配资源是否成功
 * @note 此函数用于分配资源
 */
HcclResult CcuCtxMgr::AllocRes(s32 deviceLogicId, CcuCtxGroup& ctxGroup, CcuResPack& resPack)
{
    TRY_CATCH_RETURN(
        HCCL_INFO("[AllocRes] Input params: deviceLogicId[%d], ctxGroup size[%u]", deviceLogicId, ctxGroup.ctxs.size());
        // 入参校验拦截
        CHK_PRT_RET(
            (deviceLogicId < 0 || static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM),
            HCCL_ERROR(
                "[CcuCtxMgr][AllocRes]deviceLogicId[%d] error, MAX_MODULE_DEVICE_NUM[%u]", deviceLogicId,
                MAX_MODULE_DEVICE_NUM),
            HcclResult::HCCL_E_PARA);

        CHK_PRT_RET(
            ctxGroup.ctxs.size() == 0, HCCL_ERROR("[CcuCtxMgr][AllocRes]ctxs size is zero"), HcclResult::HCCL_E_PARA);

        CHK_RET_UNAVAIL(CtxMgrImp::GetInstance(deviceLogicId).AllocRes(ctxGroup, resPack)););

    return HcclResult::HCCL_SUCCESS;
}

/**
 * @brief 释放资源
 *
 * @param deviceLogicId device逻辑ID
 * @param ctxGroup CCU上下文组
 *
 * @return 返回HcclResult，表示释放资源是否成功
 *
 * @note 此函数用于释放指定device设备ID的CCU上下文组的资源
 */
HcclResult CcuCtxMgr::ReleaseRes(s32 deviceLogicId, CcuCtxGroup& ctxGroup)
{
    TRY_CATCH_RETURN(
        HCCL_RUN_INFO(
            "[CcuCtxMgr]ReleaseRes: deviceLogicId[%d], ctxGroup size[%u]", deviceLogicId, ctxGroup.ctxs.size());

        // 入参校验拦截
        CHK_PRT_RET(
            (deviceLogicId < 0 || static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM),
            HCCL_ERROR("[CcuCtxMgr][ReleaseRes]deviceLogicId[%d] error", deviceLogicId), HcclResult::HCCL_E_PARA);

        CHK_PRT_RET(
            ctxGroup.ctxs.size() == 0, HCCL_ERROR("[CcuCtxMgr][ReleaseRes]ctxs size is zero"), HcclResult::HCCL_E_PARA);

        CHK_RET(CtxMgrImp::GetInstance(deviceLogicId).ReleaseRes(ctxGroup));

        HCCL_RUN_INFO(
            "[CcuCtxMgr]ReleaseRes:success deviceLogicId[%d], ctxGroup size[%u]", deviceLogicId,
            ctxGroup.ctxs.size()););

    return HcclResult::HCCL_SUCCESS;
}

/**
 * @brief 获取任务参数
 *
 * @param deviceLogicId device逻辑ID
 * @param ccuTaskArg CCU 任务参数
 * @param executorId 执行器ID
 * @param taskParam 任务参数
 *
 * @return HcclResult 获取参数结果
 * @note 无
 */
HcclResult CcuCtxMgr::GetTaskParam(
    s32 deviceLogicId, CcuTaskArg& ccuTaskArg, const uint64_t executorId,
    std::vector<std::vector<CcuTaskParam>>& taskParam)
{
    TRY_CATCH_RETURN(
        HCCL_INFO("[GetTaskParam] Input params: deviceLogicId[%d], executorId[%llu]", deviceLogicId, executorId);
        // 入参校验拦截
        CHK_PRT_RET(
            (deviceLogicId < 0 || static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM),
            HCCL_ERROR(
                "[CcuCtxMgr][GetTaskParam]deviceLogicId[%d] error, MAX_MODULE_DEVICE_NUM[%u]", deviceLogicId,
                MAX_MODULE_DEVICE_NUM),
            HcclResult::HCCL_E_PARA);

        taskParam = CtxMgrImp::GetInstance(deviceLogicId).GetTaskParam(ccuTaskArg, executorId);

        // 校验taskParam是否为空
        CHK_PRT_RET(
            (taskParam.size() == 0), HCCL_ERROR("[CcuCtxMgr][GetTaskParam]GetTaskParam fail"),
            HcclResult::HCCL_E_PARA););

    return HcclResult::HCCL_SUCCESS;
}

/**
 * @brief 注册扩展指令
 *
 * @param deviceLogicId device逻辑ID
 * @param entity 扩展指令执行实体
 * @param entityId 扩展指令执行实体ID
 * @return HcclResult 返回HcclResult类型的结果
 * @note 无
 */
HcclResult InsExeQue::RegisterExtendInstruction(s32 deviceLogicId, ExtInsExeEntity& entity, ExtInsExeEntityId& entityId)
{
    TRY_CATCH_RETURN(
        // 入参校验拦截
        CHK_PRT_RET(
            (deviceLogicId < 0 || static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM),
            HCCL_ERROR("[CcuCtxMgr][RegisterExtendInstruction]deviceLogicId[%d] error", deviceLogicId),
            HcclResult::HCCL_E_PARA);

        CHK_PRT_RET(
            entity.ctxGroup.ctxs.size() == 0, HCCL_ERROR("[InsExeQue][RegisterExtendInstruction]ctxs size is zero"),
            HcclResult::HCCL_E_PARA);

        entityId = CtxMgrImp::GetInstance(deviceLogicId).Register(entity.ctxGroup, entity.isFuncBlock););
    return HcclResult::HCCL_SUCCESS;
}

/**
 * @brief 注销扩展指令
 *
 * @param deviceLogicId device逻辑ID
 * @param entityId 扩展指令执行实体ID
 * @return HcclResult 返回HcclResult类型的结果
 * @note 此函数用于注销扩展指令
 */
HcclResult InsExeQue::DeregisterExtendInstruction(s32 deviceLogicId, const ExtInsExeEntityId& entityId)
{
    TRY_CATCH_RETURN(
        HCCL_RUN_INFO(
            "[InsExeQue]DeregisterExtendInstruction: deviceLogicId[%d], executorId[%llu]", deviceLogicId, entityId);
        // 入参校验拦截
        CHK_PRT_RET(
            (deviceLogicId < 0 || static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM),
            HCCL_ERROR("[CcuCtxMgr][DeregisterExtendInstruction]deviceLogicId[%d] error", deviceLogicId),
            HcclResult::HCCL_E_PARA);

        CHK_RET(CtxMgrImp::GetInstance(deviceLogicId).UnRegister(entityId));

        HCCL_RUN_INFO(
            "[InsExeQue]DeregisterExtendInstruction:success deviceLogicId[%d], executorId[%llu]", deviceLogicId,
            entityId););
    return HcclResult::HCCL_SUCCESS;
}

/**
 * @brief 获取任务Profiling信息
 *
 * @param deviceLogicId device逻辑ID
 * @param ccuTaskArg CCU 任务参数
 * @param entityId 执行器ID
 * @param profilingInfo 任务profiling信息
 *
 * @return HcclResult 获取profiling结果
 * @note 无
 */
HcclResult CcuCtxMgr::GetProfilingInfo(
    s32 deviceLogicId, CcuTaskArg& ccuTaskArg, uint64_t entityId,
    std::vector<std::vector<CcuProfilingInfo>>& profilingInfo)
{
    TRY_CATCH_RETURN(
        HCCL_INFO("[GetProfilingInfo] Input params: deviceLogicId[%d], entityId[%llu]", deviceLogicId, entityId);
        // 入参校验拦截
        CHK_PRT_RET(
            (deviceLogicId < 0 || static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM),
            HCCL_ERROR(
                "[CcuCtxMgr][GetProfilingInfo]deviceLogicId[%d] error, MAX_MODULE_DEVICE_NUM[%u]", deviceLogicId,
                MAX_MODULE_DEVICE_NUM),
            HcclResult::HCCL_E_PARA);

        profilingInfo = CtxMgrImp::GetInstance(deviceLogicId).GetProfilingInfo(ccuTaskArg, entityId);

        // 校验profilingInfo是否为空
        CHK_PRT_RET(
            (profilingInfo.size() == 0), HCCL_ERROR("[CcuCtxMgr][GetProfilingInfo]GetProfilingInfo fail"),
            HcclResult::HCCL_E_PARA););

    return HcclResult::HCCL_SUCCESS;
}

}; // namespace Hccl
