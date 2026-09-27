/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_STUB_RANK_TABLE_H
#define HCCLV2_STUB_RANK_TABLE_H

#include <string>

const std::string RankTable1Ser8Dev = R"(
    {
    "server_count":"1",
    "server_list":
    [
        {
            "device":[
                        { "device_id":"0", "rank_id":"0" },
                        { "device_id":"1", "rank_id":"1" },
                        { "device_id":"2", "rank_id":"2" },
                        { "device_id":"3", "rank_id":"3" },
                        { "device_id":"4", "rank_id":"4" },
                        { "device_id":"5", "rank_id":"5" },
                        { "device_id":"6", "rank_id":"6" },
                        { "device_id":"7", "rank_id":"7" }
                    ],
            "server_id":"1"
        }
    ],
    "status":"completed",
    "version":"1.0"
    }
    )";

const std::string RankTable1Ser2Dev = R"(
    {
    "server_count":"1",
    "server_list":
    [
        {
            "device":[
                        { "device_id":"0", "rank_id":"0" },
                        { "device_id":"1", "rank_id":"1" }
                    ],
            "server_id":"1"
        }
    ],
    "status":"completed",
    "version":"1.0"
    }
    )";

void GenRankTableFile(const std::string& rankTable);

void DelRankTableFile();

#endif
