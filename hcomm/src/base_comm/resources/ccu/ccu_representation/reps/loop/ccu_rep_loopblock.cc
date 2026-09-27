/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_rep_v1.h"

#include "string_util.h"
#include "exception_util.h"
#include "ccu_api_exception.h"

namespace hcomm {
namespace CcuRep {

    CcuRepLoopBlock::CcuRepLoopBlock(CcuInsGeneratorBase* insGenPtr, const std::string& label)
        : CcuRepBlock(insGenPtr, label)
    {
        type = CcuRepType::LOOP_BLOCK;
    }

    std::string CcuRepLoopBlock::Describe()
    {
        HCCL_INFO("Begin Describe LoopBlock[%s]", GetLabel().c_str());
        for (const auto& rep : GetReps()) {
            HCCL_INFO(" Rep: %s", rep->Describe().c_str());
        }
        return Hccl::StringFormat("LoopBlock[%s]", GetLabel().c_str());
    }

    void CcuRepLoopBlock::DefineArg(Variable var)
    {
        args.push_back(CcuRepArg(var));
        HCCL_INFO("Define Arg: Index[%u], Type[Variable], Id[%u]", args.size(), var.Id());
    }

    void CcuRepLoopBlock::DefineArg(Memory mem)
    {
        args.push_back(CcuRepArg(mem));
        HCCL_INFO("Define Arg: Index[%u], Type[Memory], Id[%u]", args.size(), mem.addr.Id());
    }

    void CcuRepLoopBlock::DefineArg(LocalAddr addr)
    {
        args.push_back(CcuRepArg(addr));
        HCCL_INFO("Define Arg: Index[%u], Type[LocalAddr], Id[%u]", args.size(), addr.addr.Id());
    }

    void CcuRepLoopBlock::DefineArg(RemoteAddr addr)
    {
        args.push_back(CcuRepArg(addr));
        HCCL_INFO("Define Arg: Index[%u], Type[RemoteAddr], Id[%u]", args.size(), addr.addr.Id());
    }

    void CcuRepLoopBlock::DefineArg(const std::vector<Variable> varList)
    {
        args.push_back(CcuRepArg(varList));
        HCCL_INFO("Define Arg: Index[%u], Type[Variable List]: ", args.size());
        for (uint32_t index = 0; index < varList.size(); index++) {
            HCCL_INFO("    Index[%u].Id[%u]", index, varList[index].Id());
        }
    }

    void CcuRepLoopBlock::DefineArg(const std::vector<LocalAddr> addrList)
    {
        args.push_back(CcuRepArg(addrList));
        HCCL_INFO("Define Arg: Index[%u], Type[LocalAddr List]: ", args.size());
        for (uint32_t index = 0; index < addrList.size(); index++) {
            HCCL_INFO("Index[%u].Id[%u]", index, addrList[index].addr.Id());
        }
    }

    void CcuRepLoopBlock::DefineArg(const std::vector<RemoteAddr> addrList)
    {
        args.push_back(CcuRepArg(addrList));
        HCCL_INFO("Define Arg: Index[%u], Type[RemoteAddr List]: ", args.size());
        for (uint32_t index = 0; index < addrList.size(); index++) {
            HCCL_INFO("Index[%u].Id[%u]", index, addrList[index].addr.Id());
        }
    }

    void CcuRepLoopBlock::DefineArg(const std::vector<Memory> memList)
    {
        args.push_back(CcuRepArg(memList));
        HCCL_INFO("Define Arg: Index[%u], Type[Memory List]: ", args.size());
        for (uint32_t index = 0; index < memList.size(); index++) {
            HCCL_INFO("Index[%u].Id[%u]", index, memList[index].addr.Id());
        }
    }

    CcuRepArg& CcuRepLoopBlock::GetArg(uint16_t index)
    {
        if (index >= args.size()) {
            Hccl::THROW<Hccl::CcuApiException>("CcuLoopBlock Arg Index[%u] Out of Range", index);
        }
        return args[index];
    }

}; // namespace CcuRep
}; // namespace hcomm
