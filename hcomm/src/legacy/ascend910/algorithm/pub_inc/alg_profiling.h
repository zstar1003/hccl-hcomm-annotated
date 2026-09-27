/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include <string>
#include <map>
#include <array>
#include "hccl_types.h"
#include "hccl_common.h"
#include "common.h"

namespace hccl {

using TaskCallBack = void (*)(void* userPtr, void* param, u32 length);

struct TaskParaAiv {
    HcclCMDType cmdType;
    u32 tag;
    u64 size;
    u32 numBlocks;
    u32 rankSize;
    s32 aivRdmaStep;
    void* flagMem;
    u32 rank;
    bool isOpbase;
    TaskParaAiv()
        : cmdType(HcclCMDType::HCCL_CMD_INVALID),
          tag(0),
          size(0),
          numBlocks(0),
          rankSize(0),
          aivRdmaStep(0),
          flagMem(nullptr),
          rank(0),
          isOpbase(false)
    {}
    TaskParaAiv(
        HcclCMDType cmdType, u32 tag, u64 size, u32 numBlocks, u32 rankSize, s32 aivRdmaStep, void* flagMem, u32 rank,
        bool isOpbase = false)
        : cmdType(cmdType),
          tag(tag),
          size(size),
          numBlocks(numBlocks),
          rankSize(rankSize),
          aivRdmaStep(aivRdmaStep),
          flagMem(flagMem),
          rank(rank),
          isOpbase(isOpbase)
    {}
};

struct TaskParaGeneral {
    void* stream{nullptr};
    bool isMainStream{false};
    u64 beginTime{0};
    struct TaskParaAiv aiv;

    TaskParaGeneral() : stream(nullptr), isMainStream(false), beginTime(0) {}

    ~TaskParaGeneral() {}
};

class AlgWrap {
public:
    static AlgWrap& GetInstance();
    HcclResult RegisterAlgCallBack(const std::string& comm, void* userPtr, TaskCallBack callback, s32 deviceLogicID);
    void UnregisterAlgCallBack(const std::string& comm);
    HcclResult TaskAivProfiler(const std::string& comm, struct TaskParaGeneral& taskParaGeneral);

private:
    AlgWrap() { initialized_ = true; };
    ~AlgWrap() { initialized_ = false; };

    // initialized 是否已初始化，避免析构后访问类成员
    bool initialized_ = false;
    std::mutex aivCallBackMutex_;
    std::map<std::string, std::array<TaskCallBack, MAX_MODULE_DEVICE_NUM>> aivCallBackMap_;
    std::map<std::string, std::array<void*, MAX_MODULE_DEVICE_NUM>> aivCallBackUserPtrMap_;
};

} // namespace hccl
