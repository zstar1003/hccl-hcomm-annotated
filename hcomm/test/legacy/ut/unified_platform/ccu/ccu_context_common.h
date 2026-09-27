/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_CONTEXT_COMMON_H
#define HCCL_CCU_CONTEXT_COMMON_H

#include "ccu_ctx.h"

using namespace Hccl;
using namespace CcuRep;

class CcuCtxArgTest : public CcuCtxArg {
public:
    explicit CcuCtxArgTest(uint32_t rankId, uint32_t rankSize) : rankId(rankId), rankSize(rankSize) {}
    virtual ~CcuCtxArgTest() = default;
    CcuCtxSignature GetCtxSignature() const override
    {
        CcuCtxSignature signature;
        signature.Append("Test");
        return signature;
    }
    uint32_t rankId;
    uint32_t rankSize;
};

class CcuTaskArgTest : public CcuTaskArg {
public:
    explicit CcuTaskArgTest(uint64_t inputAddr, uint64_t outputAddr, uint64_t size)
        : inputAddr(inputAddr),
          outputAddr(outputAddr),
          size(size)
    {}
    uint64_t inputAddr;
    uint64_t outputAddr;
    uint64_t size;
};

class CcuContextAG : public CcuContext {
public:
    CcuContextAG(CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& transportGroup)
        : CcuContext(arg, transports, transportGroup)
    {
        id = dynamic_cast<const CcuCtxArgTest*>(&arg)->rankId;
        size = dynamic_cast<const CcuCtxArgTest*>(&arg)->rankSize;
    }

protected:
    void Algorithm() override
    {
        std::vector<Variable> input;
        std::vector<Variable> output;
        std::vector<Variable> token;
        for (uint32_t i = 0; i < size; i++) {
            input.emplace_back(CreateVariable());
            output.emplace_back(CreateVariable());
            token.emplace_back(CreateVariable());
        }

        Variable offset = CreateVariable();
        GroupOpSize goSize = CreateGroupOpSize();

        Memory src = CreateMemory();
        std::vector<Memory> dst;
        for (uint32_t i = 0; i < size; i++) {
            dst.emplace_back(CreateMemory());
        }

        uint16_t selfBit = 1 << id;
        uint16_t allBit = ((1 << size) - 1) & (~(1 << id));

        Load(input[id]);
        Load(output[id]);
        Load(offset);
        Load(goSize);
        Load(token[id]);

        for (auto t : transports) {
            WriteVariableWithSignal(*t, input[id], 0, 0, selfBit);  // index = 0，传递input信息
            WriteVariableWithSignal(*t, output[id], 1, 1, selfBit); // index = 1，传递output信息
            WriteVariableWithSignal(*t, token[id], 2, 2, selfBit);  // index = 2，传递token信息
        }
        GroupWait(*transportGroup, 0, allBit); // index = 0，传递input信息
        GroupWait(*transportGroup, 1, allBit); // index = 1，传递output信息
        GroupWait(*transportGroup, 2, allBit); // index = 2，传递token信息

        src.addr = input[id];
        src.token = token[id];
        uint32_t dstId = 0;
        uint32_t curId = 0;
        for (uint32_t r = 0; r < size; r++) {
            if (r != id) {
                curId = dstId;
                dstId++;
            } else {
                curId = size - 1;
            }
            dst[curId].addr = output[r];
            dst[curId].addr += offset;
            dst[curId].token = token[r];
        }

        GroupBroadcast(transports, dst, src, goSize);

        for (auto t : transports) {
            RemotePost(*t, 0, selfBit);
        }
        GroupWait(*transportGroup, 0, allBit);
    }
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg)
    {
        auto taskArg = dynamic_cast<const CcuTaskArgTest*>(&arg);
        auto goSize = CalGoSize(taskArg->size);

        return {taskArg->inputAddr, taskArg->outputAddr, 0, goSize[0], goSize[1], goSize[2], goSize[3], 0};
    }

private:
    uint32_t id;
    uint32_t size;
};

class CcuContextRS : public CcuContext {
public:
    CcuContextRS(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& transportGroup)
        : CcuContext(arg, transports, transportGroup)
    {
        id = dynamic_cast<const CcuCtxArgTest*>(&arg)->rankId;
        size = dynamic_cast<const CcuCtxArgTest*>(&arg)->rankSize;
    }

protected:
    void Algorithm() override
    {
        std::vector<Variable> input;
        std::vector<Variable> output;
        std::vector<Variable> token;
        for (uint32_t i = 0; i < size; i++) {
            input.emplace_back(CreateVariable());
            output.emplace_back(CreateVariable());
            token.emplace_back(CreateVariable());
        }

        Variable offset = CreateVariable();
        GroupOpSize goSize = CreateGroupOpSize();

        std::vector<Memory> src;
        Memory dst = CreateMemory();
        for (uint32_t i = 0; i < size; i++) {
            src.emplace_back(CreateMemory());
        }

        uint16_t selfBit = 1 << id;
        uint16_t allBit = ((1 << size) - 1) & (~(1 << id));

        Load(input[id]);
        Load(output[id]);
        Load(offset);
        Load(goSize);
        Load(token[id]);

        for (auto t : transports) {
            WriteVariableWithSignal(*t, input[id], 0, 0, selfBit);  // index = 0，传递input信息
            WriteVariableWithSignal(*t, output[id], 1, 1, selfBit); // index = 1，传递output信息
            WriteVariableWithSignal(*t, token[id], 2, 2, selfBit);  // index = 2，传递token信息
        }
        GroupWait(*transportGroup, 0, allBit); // index = 0，传递input信息
        GroupWait(*transportGroup, 1, allBit); // index = 1，传递output信息
        GroupWait(*transportGroup, 2, allBit); // index = 2，传递token信息

        uint32_t dstId = 0;
        uint32_t curId = 0;
        for (uint32_t r = 0; r < size; r++) {
            if (r != id) {
                curId = dstId;
                dstId++;
            } else {
                curId = size - 1;
            }
            src[curId].addr = input[r];
            src[curId].addr += offset;
            src[curId].token = token[r];
        }
        dst.addr = output[id];
        dst.token = token[id];

        GroupReduce(transports, dst, src, goSize, DataType::FP32, DataType::FP32, ReduceOp::SUM);

        for (const auto& t : transports) {
            RemotePost(*t, 0, selfBit);
        }
        GroupWait(*transportGroup, 0, allBit);
    }
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg)
    {
        auto taskArg = dynamic_cast<const CcuTaskArgTest*>(&arg);
        auto goSize = CalGoSize(taskArg->size);

        return {taskArg->inputAddr, taskArg->outputAddr, 0, goSize[0], goSize[1], goSize[2], goSize[3], 0};
    }

private:
    uint32_t id;
    uint32_t size;
};

class CcuContextTestMultiArgs : public CcuContext {
public:
    CcuContextTestMultiArgs(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& transportGroup)
        : CcuContext(arg, transports, transportGroup)
    {}

protected:
    void Algorithm() override
    {
        std::vector<Variable> input(14);
        for (uint32_t i = 0; i < 14; i++) {
            input.emplace_back(CreateVariable());
        }
        for (uint32_t i = 0; i < 14; i++) {
            Load(input[i]);
        }
    }
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg)
    {
        std::vector<uint64_t> args(14);
        for (int i = 0; i < 14; i++) {
            args[i] = i;
        }
        return args;
    }
};

class CcuContextTestVariable : public CcuContext {
public:
    CcuContextTestVariable(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& transportGroup)
        : CcuContext(arg, transports, transportGroup)
    {}

protected:
    void Algorithm() override
    {
        Variable a = CreateVariable();
        Variable b = CreateVariable();
        Variable c = CreateVariable();
        a = b + c;

        Address aa = CreateAddress();
        Address ab = CreateAddress();
        Address ac = CreateAddress();
        aa = ab + ac;

        aa = ab;
        aa = 0;
        aa = a + ab;
        aa = ab + a;
    }
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) { return {}; }
};

class CcuContextTestDataTransfer : public CcuContext {
public:
    CcuContextTestDataTransfer(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& transportGroup)
        : CcuContext(arg, transports, transportGroup)
    {}

protected:
    void Algorithm() override
    {
        CcuRep::CcuBuffer buf = CreateCcuBuffer();
        Executor executor = CreateExecutor();
        MaskSignal sig = CreateMaskSignal();

        LocalPost(sig, 1);
        RemoteWait(*transports[0], 0, 1);

        Memory loc = CreateMemory();
        Memory rmt = CreateMemory();
        Variable len = CreateVariable();
        Read(*transports[0], loc, rmt, len, sig, 1);
        ReadReduce(*transports[0], loc, rmt, len, DataType::FP32, ReduceOp::SUM, sig, 1);
        Write(*transports[0], rmt, loc, len, sig, 1);
        WriteReduce(*transports[0], rmt, loc, len, DataType::FP32, ReduceOp::SUM, sig, 1);

        Memory src = CreateMemory();
        Memory dst = CreateMemory();
        LocalCopy(dst, src, len, sig, 1);
        LocalReduce(dst, src, len, DataType::FP32, ReduceOp::SUM, sig, 1);
        GroupOpSize goSize = CreateGroupOpSize();
        Load(goSize);
        GroupCopy(dst, src, goSize);
    }
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg)
    {
        auto taskArg = dynamic_cast<const CcuTaskArgTest*>(&arg);
        auto goSize = CalGoSize(taskArg->size);
        return {goSize[0], goSize[1], goSize[2], goSize[3]};
    }
};

class CcuContextTestCondition : public CcuContext {
public:
    CcuContextTestCondition(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& transportGroup)
        : CcuContext(arg, transports, transportGroup)
    {}

protected:
    void Algorithm() override
    {
        Variable iter = CreateVariable();
        Variable var = CreateVariable();
        iter = 1;
        CCU_IF(iter == 1) { var = 1; }
    }
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) { return {}; }
};

class CcuContextTestRepeat : public CcuContext {
public:
    CcuContextTestRepeat(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& transportGroup)
        : CcuContext(arg, transports, transportGroup)
    {}

protected:
    void Algorithm() override
    {
        Variable iter = CreateVariable();
        iter = 1;
        CCU_WHILE(iter != 10)
        {
            CCU_IF(iter == 5) { CCU_BREAK; }
        }
    }
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) { return {}; }
};

class CcuContextTestFunction : public CcuContext {
public:
    CcuContextTestFunction(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& transportGroup)
        : CcuContext(arg, transports, transportGroup)
    {}

protected:
    void Algorithm() override
    {
        {
            FuncBlock fb(this, "test");
            Variable a = CreateVariable();
            std::vector<Variable> aV = {CreateVariable(), CreateVariable()};
            fb.DefineInArg(a);
            fb.DefineInArg(aV);
            fb.DefineOutArg(a);
            fb.DefineOutArg(aV);
        }
        Variable b = CreateVariable();
        std::vector<Variable> bV = {CreateVariable(), CreateVariable()};
        auto fc = Func("test");
        fc.SetInArg(b);
        fc.SetInArg(bV);
        fc.SetOutArg(b);
        fc.SetOutArg(bV);

        Variable funcAddr = CreateVariable();
        Func(funcAddr);
    }
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) { return {}; }
};

class CcuCtxArgSharedRes : public CcuCtxArg {
public:
    explicit CcuCtxArgSharedRes(uint32_t id) : id(id) {}
    virtual ~CcuCtxArgSharedRes() = default;
    CcuCtxSignature GetCtxSignature() const override
    {
        CcuCtxSignature signature;
        signature.Append("Test");
        return signature;
    }
    uint32_t id{0};
};

class CcuTaskArgSharedRes : public CcuTaskArg {
public:
    explicit CcuTaskArgSharedRes() {}
};

class CcuContextTestSharesRes : public CcuContext {
public:
    CcuContextTestSharesRes(
        CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& transportGroup)
        : CcuContext(arg, transports, transportGroup)
    {
        id = ((CcuCtxArgSharedRes&)(arg)).id;
    }

protected:
    void Algorithm() override
    {
        if (id == 0) {
            Variable var = CreateVariable();
            Load(var);
            for (uint32_t otherId = 1; otherId < 3; otherId++) {
                Variable otherVar = ImportVariable("var" + std::to_string(otherId));
                MaskSignal otherSig = ImportMaskSignal("sig" + std::to_string(otherId));
                LocalCtxPost(otherSig, 1);
                LocalCtxPostVar(var, otherVar, otherSig, 1);
            }
        } else {
            Variable var = CreateVariable();
            ExportVariable(var, "var" + std::to_string(id));
            MaskSignal sig;
            ExportMaskSignal(sig, "sig" + std::to_string(id));
        }
    }
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg)
    {
        if (id == 0) {
            return {1024};
        } else {
            return {};
        }
    }

private:
    uint32_t id;
};

#endif // HCCL_CCU_CONTEXT_COMMON_H
