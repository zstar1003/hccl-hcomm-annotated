/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "comm_configer.h"
#include "alg_env_config.h"

namespace hccl {
CommConfiger& CommConfiger::GetInstance()
{
    static CommConfiger configer;
    return configer;
}

CommConfiger::CommConfiger() : initialized_(true) {}

HcclResult CommConfiger::SetCommConfig(CommConfig config, const std::string& identifier)
{
    std::lock_guard<std::mutex> lock(lock_);
    if (identifier.empty()) {
        HCCL_WARNING("[CommConfiger][SetCommConfig]: identifier is empty.");
        return HCCL_SUCCESS;
    }
    auto commConfigIter = commConfigMap_.find(identifier);
    if (commConfigIter != commConfigMap_.end()) {
        HCCL_WARNING(
            "[CommConfiger][SetCommConfig]: commConfig of identifier[%s] is already existed.", identifier.c_str());
    }
    commConfigMap_[identifier] = config;
    HCCL_INFO("[CommConfiger][SetCommConfig]: commConfig of identifier[%s]", identifier.c_str());
    return HCCL_SUCCESS;
}

HcclResult CommConfiger::SetCommConfigExecTimeOut(s32 execTimeOut, const std::string& identifier)
{
    std::lock_guard<std::mutex> lock(lock_);
    if (identifier.empty()) {
        HCCL_WARNING("[CommConfiger][SetCommConfigExecTimeOut]: identifier is empty.");
        return HCCL_SUCCESS;
    }
    auto commConfigIter = commConfigMap_.find(identifier);
    if (commConfigIter == commConfigMap_.end()) {
        HCCL_WARNING("[CommConfiger][GetCommConfigExecTimeOut]: CommConfig of [%s] is not found.", identifier.c_str());
        return HCCL_SUCCESS;
    }
    return commConfigMap_[identifier].SetConfigExecTimeOut(execTimeOut);
}

s32 CommConfiger::GetCommConfigExecTimeOut(const std::string& identifier)
{
    std::lock_guard<std::mutex> lock(lock_);
    s32 execTimeOut = GetInternalExecTimeOut();
    if (identifier.empty()) {
        HCCL_WARNING("[CommConfiger][GetCommConfigExecTimeOut]: identifier is empty.");
        return execTimeOut;
    }
    auto commConfigIter = commConfigMap_.find(identifier);
    if (commConfigIter == commConfigMap_.end()) {
        HCCL_WARNING("[CommConfiger][GetCommConfigExecTimeOut]: CommConfig of [%s] is not found.", identifier.c_str());
        return execTimeOut;
    }
    execTimeOut = commConfigMap_[identifier].GetConfigExecTimeOut();
    HCCL_INFO(
        "[CommConfiger][GetCommConfigExecTimeOut]: identifier[%s], execTimeOut[%d]s", identifier.c_str(), execTimeOut);
    return execTimeOut;
}

bool CommConfiger::GetCommConfigExecTimeOutSet(const std::string& identifier)
{
    std::lock_guard<std::mutex> lock(lock_);
    bool execTimeOutSet = false;
    if (identifier.empty()) {
        HCCL_WARNING("[CommConfiger][GetCommConfigExecTimeOutSet]: identifier is empty.");
        return execTimeOutSet;
    }
    auto commConfigIter = commConfigMap_.find(identifier);
    if (commConfigIter == commConfigMap_.end()) {
        HCCL_WARNING("[CommConfiger][GetCommConfigExecTimeOut]: CommConfig of [%s] is not found.", identifier.c_str());
        return execTimeOutSet;
    }
    execTimeOutSet = commConfigMap_[identifier].GetConfigExecTimeOutSet();
    HCCL_INFO(
        "[CommConfiger][GetCommConfigExecTimeOutSet]: identifier[%s], execTimeOutSet[%d]", identifier.c_str(),
        execTimeOutSet);
    return execTimeOutSet;
}

std::vector<HcclAlgoType> CommConfiger::GetCommConfigAlgoConfig(const std::string& identifier, HcclCMDType opType)
{
    std::lock_guard<std::mutex> lock(lock_);
    if (identifier.empty()) {
        HCCL_WARNING("[CommConfiger][GetCommConfigAlgoConfig]: identifier is empty.");
        return GetExternalInputHcclAlgoConfig(opType);
    }
    auto commConfigIter = commConfigMap_.find(identifier);
    if (commConfigIter == commConfigMap_.end()) {
        HCCL_WARNING("[CommConfiger][GetCommConfigAlgoConfig]: CommConfig of [%s] is not found.", identifier.c_str());
        return GetExternalInputHcclAlgoConfig(opType);
    }
    return commConfigMap_[identifier].GetConfigHcclAlgo(opType);
}

bool CommConfiger::GetCommConfigInterServerRetryEnable(const std::string& identifier)
{
    std::lock_guard<std::mutex> lock(lock_);
    bool isRetry = GetExternalInputInterServerRetryEnable();
    if (identifier.empty()) {
        HCCL_WARNING("[CommConfiger][GetCommConfigInterServerRetryEnable]: identifier is empty.");
        return isRetry;
    }
    auto commConfigIter = commConfigMap_.find(identifier);
    if (commConfigIter == commConfigMap_.end()) {
        HCCL_WARNING(
            "[CommConfiger][GetCommConfigInterServerRetryEnable]: CommConfig of [%s] is not found.",
            identifier.c_str());
        return isRetry;
    }
    isRetry = commConfigMap_[identifier].GetConfigInterServerRetryEnable();
    HCCL_INFO(
        "[CommConfiger][GetCommConfigInterServerRetryEnable]: identifier[%s], isRetry[%d].", identifier.c_str(),
        isRetry);
    return isRetry;
}

bool CommConfiger::GetCommConfigInterSuperPodRetryEnable(const std::string& identifier)
{
    std::lock_guard<std::mutex> lock(lock_);
    bool isRetry = GetExternalInputInterSuperPodRetryEnable();
    if (identifier.empty()) {
        HCCL_WARNING("[CommConfiger][GetCommConfigInterSuperPodRetryEnable]: identifier is empty.");
        return isRetry;
    }
    auto commConfigIter = commConfigMap_.find(identifier);
    if (commConfigIter == commConfigMap_.end()) {
        HCCL_WARNING(
            "[CommConfiger][GetCommConfigInterSuperPodRetryEnable]: CommConfig of [%s] is not found.",
            identifier.c_str());
        return isRetry;
    }
    isRetry = commConfigMap_[identifier].GetConfigInterSuperPodRetryEnable();
    HCCL_INFO(
        "[CommConfiger][GetCommConfigInterSuperPodRetryEnable]: identifier[%s], isRetry[%d].", identifier.c_str(),
        isRetry);
    return isRetry;
}

u32 CommConfiger::GetCommConfigRetryMaxCnt(const std::string& identifier)
{
    std::lock_guard<std::mutex> lock(lock_);
    u32 retryMaxCnt = GetExternalInputRetryMaxCnt();
    if (identifier.empty()) {
        HCCL_WARNING("[CommConfiger][GetCommConfigRetryMaxCnt]: identifier is empty.");
        return retryMaxCnt;
    }
    auto commConfigIter = commConfigMap_.find(identifier);
    if (commConfigIter == commConfigMap_.end()) {
        HCCL_WARNING("[CommConfiger][GetCommConfigRetryMaxCnt]: CommConfig of [%s] is not found.", identifier.c_str());
        return retryMaxCnt;
    }
    retryMaxCnt = commConfigMap_[identifier].GetConfigRetryMaxCnt();
    HCCL_INFO(
        "[CommConfiger][GetCommConfigRetryMaxCnt]: identifier[%s], retryMaxCnt[%d].", identifier.c_str(), retryMaxCnt);
    return retryMaxCnt;
}

u32 CommConfiger::GetCommConfigRetryHoldTime(const std::string& identifier)
{
    std::lock_guard<std::mutex> lock(lock_);
    u32 retryHoldTime = GetExternalInputRetryHoldTime();
    if (identifier.empty()) {
        HCCL_WARNING("[CommConfiger][GetCommConfigRetryHoldTime]: identifier is empty.");
        return retryHoldTime;
    }
    auto commConfigIter = commConfigMap_.find(identifier);
    if (commConfigIter == commConfigMap_.end()) {
        HCCL_WARNING(
            "[CommConfiger][GetCommConfigRetryHoldTime]: CommConfig of [%s] is not found.", identifier.c_str());
        return retryHoldTime;
    }
    retryHoldTime = commConfigMap_[identifier].GetConfigRetryHoldTime();
    HCCL_INFO(
        "[CommConfiger][GetCommConfigRetryHoldTime]: identifier[%s], retryHoldTime[%d ms].", identifier.c_str(),
        retryHoldTime);
    return retryHoldTime;
}

u32 CommConfiger::GetCommConfigRetryIntervalTime(const std::string& identifier)
{
    std::lock_guard<std::mutex> lock(lock_);
    u32 retryIntervalTime = GetExternalInputRetryIntervalTime();
    if (identifier.empty()) {
        HCCL_WARNING("[CommConfiger][GetCommConfigRetryIntervalTime]: identifier is empty.");
        return retryIntervalTime;
    }
    auto commConfigIter = commConfigMap_.find(identifier);
    if (commConfigIter == commConfigMap_.end()) {
        HCCL_WARNING(
            "[CommConfiger][GetCommConfigRetryIntervalTime]: CommConfig of [%s] is not found.", identifier.c_str());
        return retryIntervalTime;
    }
    retryIntervalTime = commConfigMap_[identifier].GetConfigRetryIntervalTime();
    HCCL_INFO(
        "[CommConfiger][GetCommConfigRetryIntervalTime]: identifier[%s], retryIntervalTime[%d ms].", identifier.c_str(),
        retryIntervalTime);
    return retryIntervalTime;
}

void CommConfiger::UnRegisterToCommConfiger(const std::string& identifier)
{
    if (initialized_ == false) {
        HCCL_WARNING("CommConfiger has been destroyed");
        return;
    }
    std::lock_guard<std::mutex> lock(lock_);
    if (identifier.empty()) {
        HCCL_WARNING("[CommConfiger][UnRegisterToCommConfiger]: identifier is empty.");
        return;
    }
    auto commConfigIter = commConfigMap_.find(identifier);
    if (commConfigIter == commConfigMap_.end()) {
        HCCL_WARNING("[CommConfiger][UnRegisterToCommConfiger]: CommConfig of [%s] is not found.", identifier.c_str());
        return;
    }
    commConfigMap_.erase(identifier);
}

CommConfiger::~CommConfiger()
{
    std::lock_guard<std::mutex> lock(lock_);
    initialized_ = false;
    commConfigMap_.clear();
}
} // namespace hccl
