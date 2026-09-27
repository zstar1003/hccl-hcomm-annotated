/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "command_handle.h"
#include "profiling_manager.h"
#include "prof_common.h"

namespace hccl {
rtError_t CommandHandle(uint32_t rtType, void* data, uint32_t len)
{
    (void)len;
    if (data == nullptr) {
        HCCL_ERROR("[Profiling][CommandHandle] CommandHandle's data is NULL.");
        return FAILED;
    }
    auto& profilingManager = hccl::ProfilingManager::Instance();
    if (rtType == RT_PROF_CTRL_REPORTER) { // 创建 reporter
        HCCL_INFO("[Profiling][CommandHandle] CommandHandle's rtType is %u.", rtType);
        profilingManager.SetMsprofReporterCallback(reinterpret_cast<MsprofReporterCallback>(data));
    } else if (rtType == RT_PROF_CTRL_SWITCH) {
        rtProfCommandHandle_t* profConfigParam = reinterpret_cast<rtProfCommandHandle_t*>(data);
        auto type = profConfigParam->type;
        auto profconfig = profConfigParam->profSwitch;
        HCCL_RUN_INFO(
            "[Profiling][CommandHandle] CommandHandle's rtType is %u. CommandHandle_switch type[%u], "
            "profconfig[%llu], deviceLogicId[%u]",
            rtType, type, profconfig, profConfigParam->devIdList[0]);
        switch (type) {
            case PROF_COMMANDHANDLE_TYPE_INIT:
                profilingManager.PluginInit();
                break;
            case PROF_COMMANDHANDLE_TYPE_START:
                profilingManager.PluginInit();
                profilingManager.StartSubscribe(profconfig);
                break;
            case PROF_COMMANDHANDLE_TYPE_STOP:
                profilingManager.StopSubscribe(profconfig);
                break;
            case PROF_COMMANDHANDLE_TYPE_FINALIZE:
                profilingManager.PluginUnInit();
                break;
            case PROF_COMMANDHANDLE_TYPE_MODEL_SUBSCRIBE:
                profilingManager.PluginInit();
                profilingManager.StartSubscribe(profconfig);
                break;
            case PROF_COMMANDHANDLE_TYPE_MODEL_UNSUBSCRIBE:
                profilingManager.StopSubscribe(profconfig);
                profilingManager.PluginUnInit();
                break;
            default:
                HCCL_RUN_INFO("[Profiling][CommandHandle] Unexpected behavior.");
        }
    }

    return SUCCESS;
}

rtError_t EsCommandHandle(uint32_t rtType, void* data, uint32_t len)
{
    (void)len;
    if (data == nullptr) {
        HCCL_ERROR("[Profiling][EsCommandHandle] EsCommandHandle's data is NULL.");
        return FAILED;
    }

    HCCL_INFO("[Profiling][EsCommandHandle] EsCommandHandle's rtType is %u.", rtType);
    if (rtType == RT_PROF_CTRL_REPORTER) { // 创建 reporter
        HCCL_INFO("rtType[%u] == RT_PROF_CTRL_REPORTER", rtType);
        return SUCCESS;
    }

    auto& profilingManager = hccl::ProfilingManager::Instance();
    rtProfCommandHandle_t* profConfigParam = static_cast<rtProfCommandHandle_t*>(data);
    auto type = profConfigParam->type;
    auto profconfig = profConfigParam->profSwitch;

    HCCL_INFO("[Profiling][EsCommandHandle] EsCommandHandle_switch type[%u], profconfig[%llu]", type, profconfig);
    switch (type) {
        case PROF_COMMANDHANDLE_TYPE_START:
        case PROF_COMMANDHANDLE_TYPE_MODEL_SUBSCRIBE:
            profilingManager.EsStartSubscribe(profconfig);
            break;
        case PROF_COMMANDHANDLE_TYPE_STOP:
        case PROF_COMMANDHANDLE_TYPE_MODEL_UNSUBSCRIBE:
            profilingManager.EsStopSubscribe(profconfig);
            break;
        case PROF_COMMANDHANDLE_TYPE_INIT:
        case PROF_COMMANDHANDLE_TYPE_FINALIZE:
            break;
        default:
            HCCL_ERROR("[Profiling][EsCommandHandle] Unexpected behavior.");
    }

    return SUCCESS;
}

} // namespace hccl
