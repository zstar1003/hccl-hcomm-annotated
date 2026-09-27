/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rank_table.h"
#include <fstream>
#include <iostream>
#include <unistd.h>
#include "nlohmann/json.hpp"

void GenRankTableFile1Ser8Dev()
{
    try {
        nlohmann::json rankTableJson = nlohmann::json::parse(RankTable1Ser8Dev);
        std::ofstream out(ranktablePath, std::ofstream::out);
        out << rankTableJson;
    } catch (...) {
        std::cout << ranktablePath << " generate failed!" << std::endl;
        return;
    }
    std::cout << ranktablePath << " generated." << std::endl;
}

void DelRankTableFile()
{
    int res = unlink(ranktablePath);
    if (res == -1) {
        std::cout << ranktablePath << " delete failed!" << std::endl;
        return;
    }
    std::cout << ranktablePath << " deleted." << std::endl;
}

void GenRankTableFile4p()
{
    try {
        nlohmann::json rankTableJson = nlohmann::json::parse(RankTable4p);
        std::ofstream out(ranktable4pPath, std::ofstream::out);
        out << rankTableJson;
    } catch (...) {
        std::cout << ranktable4pPath << " generate failed!" << std::endl;
        return;
    }
    std::cout << ranktable4pPath << " generated." << std::endl;
}

void DelRankTableFile4p()
{
    int res = unlink(ranktable4pPath);
    if (res == -1) {
        std::cout << ranktable4pPath << " delete failed!" << std::endl;
        return;
    }
    std::cout << ranktable4pPath << " deleted." << std::endl;
}

void GenTopoFile()
{
    try {
        nlohmann::json topoJson = nlohmann::json::parse(Topo1Ser8Dev);
        std::ofstream out(topoPath, std::ofstream::out);
        out << topoJson;
    } catch (...) {
        std::cout << topoPath << " generate failed!" << std::endl;
        return;
    }
    std::cout << topoPath << " generated." << std::endl;
}

void DelTopoFile()
{
    int res = unlink(topoPath);
    if (res == -1) {
        std::cout << topoPath << " delete failed!" << std::endl;
        return;
    }
    std::cout << topoPath << " deleted." << std::endl;
}
