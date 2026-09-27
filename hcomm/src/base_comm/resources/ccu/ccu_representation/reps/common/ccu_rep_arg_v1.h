/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_REPRESENTATION_ARG_H
#define HCOMM_CCU_REPRESENTATION_ARG_H

#include <vector>
#include <memory>

#include "ccu_datatype_v1.h"

namespace hcomm {
namespace CcuRep {

    enum class CcuArgType {
        VARIABLE,
        MEMORY,
        VARIABLE_LIST,
        MEMORY_LIST,
        LOCAL_ADDR,      // 1. 新增枚举值
        LOCAL_ADDR_LIST, // 2. List 类型，以防后面需要 vector<LocalAddr>
        REMOTE_ADDR,
        REMOTE_ADDR_LIST,
    };

    struct CcuRepArg {
        explicit CcuRepArg(const Variable& var) : type(CcuArgType::VARIABLE), var(var) {}
        explicit CcuRepArg(const Memory& mem) : type(CcuArgType::MEMORY), mem(mem) {}
        explicit CcuRepArg(const std::vector<Variable>& varList) : type(CcuArgType::VARIABLE_LIST), varList(varList) {}
        explicit CcuRepArg(const std::vector<Memory>& memList) : type(CcuArgType::MEMORY_LIST), memList(memList) {}
        // 新增：LocalAddr 的构造函数
        explicit CcuRepArg(const LocalAddr& addr) : type(CcuArgType::LOCAL_ADDR), localAddr(addr) {}
        // 新增：LocalAddr 列表的构造函数
        explicit CcuRepArg(const std::vector<LocalAddr>& addrList)
            : type(CcuArgType::LOCAL_ADDR_LIST),
              localAddrList(addrList)
        {}
        explicit CcuRepArg(const RemoteAddr& addr) : type(CcuArgType::REMOTE_ADDR), remoteAddr(addr) {}
        explicit CcuRepArg(const std::vector<RemoteAddr>& addrList)
            : type(CcuArgType::REMOTE_ADDR_LIST),
              remoteAddrList(addrList)
        {}

        CcuArgType type;
        Variable var;
        Memory mem;
        std::vector<Variable> varList;
        std::vector<Memory> memList;
        LocalAddr localAddr;                  // 3. 新增成员变量来存储
        std::vector<LocalAddr> localAddrList; // 新增成员变量
        RemoteAddr remoteAddr;
        std::vector<RemoteAddr> remoteAddrList;
    };

}; // namespace CcuRep
}; // namespace hcomm
#endif // HCOMM_CCU_REPRESENTATION_ARG_H
