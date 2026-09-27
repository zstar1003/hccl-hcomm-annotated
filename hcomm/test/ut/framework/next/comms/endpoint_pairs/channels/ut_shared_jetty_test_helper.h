/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UT_SHARED_JETTY_TEST_HELPER_H
#define UT_SHARED_JETTY_TEST_HELPER_H

#include <memory>
#include "dev_ub_connection.h"
#include "env_config_v2.h"

inline std::unique_ptr<Hccl::DevUbConnection>
MakeTestJettyConnection(Hccl::DevUbConnection::JettyMode jettyMode = Hccl::DevUbConnection::JettyMode::SELF_CREATE)
{
    Hccl::IpAddress locIp("1.0.0.1");
    Hccl::IpAddress rmtIp("2.0.0.2");
    return std::make_unique<Hccl::DevUbConnection>(
        nullptr, locIp, rmtIp, Hccl::OpMode::OPBASE, false, Hccl::HrtUbJfcMode::STARS_POLL, Hccl::IpAddress(),
        Hccl::IpAddress(), static_cast<u8>(Hccl::UB_QOS_DEFAULT), Hccl::TpManager::TA_TIMEOUT_NOT_SET,
        COMM_ENGINE_RESERVED, Hccl::UB_SQ_DEPTH_NOT_SET, jettyMode);
}

#endif // UT_SHARED_JETTY_TEST_HELPER_H
