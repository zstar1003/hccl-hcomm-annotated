/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_plugin_stub.h"

#include <cstdint>

const int HcclPlugin::MAX_SCAN_DEPTH = 2;
const std::string HcclPlugin::PLUGIN_PATH = "/plugin";
const std::string HcclPlugin::MANIFEST_FILE = "/manifest.json";

const std::string HcclPlugin::Manifest::pluginName = "name";
const std::string HcclPlugin::Manifest::pluginVersion = "version";
const std::string HcclPlugin::Manifest::pluginEntry = "entry";
const std::string HcclPlugin::Manifest::pluginDependency::hostVersion = "min_core_version";

HcclPlugin::HcclPlugin(const std::string& pluginPath) { m_pluginPath = pluginPath; }

HcclPlugin::~HcclPlugin() {}

HcclSim::HcclVmResult HcclPlugin::Start() { return HcclSim::HcclVmResult::HCCL_SIM_SUCCESS; }

HcclSim::HcclVmResult HcclPlugin::Stop() { return HcclSim::HcclVmResult::HCCL_SIM_SUCCESS; }

int32_t HcclPlugin::GetPid() const { return m_pid; }

int32_t HcclPlugin::GetStdinFd() const { return m_stdinFd; }

bool HcclPlugin::IsRunning() const { return m_pid > 0; }

std::string HcclPlugin::GetTag() const { return "stub_plugin"; }
