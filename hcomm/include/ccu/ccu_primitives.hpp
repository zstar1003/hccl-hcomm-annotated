/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_PRIMITIVES_HPP
#define CCU_PRIMITIVES_HPP

#include <vector>

#include "ccu_primitives_impl.h"
#include "ccu_control_flow_macro.h"

#include "ccu_variable.hpp"
#include "ccu_address.hpp"
#include "ccu_event.hpp"
#include "ccu_buffer.hpp"
#include "ccu_local_addr.hpp"
#include "ccu_remote_addr.hpp"
#include "ccu_array.hpp"
#include "ccu_func.hpp"
#include "ccu_loop.hpp"

namespace AscendC {
namespace ccu {

    // ==================== 类型别名 ====================
    using LoopConfig = ::CcuLoopConfig;
    using LoopGroupConfig = ::CcuLoopGroupConfig;
    using LoopCfg = ::CcuLoopCfg;
    using LoopGroupCfg = ::CcuLoopGroupCfg;
    using CascCounterHandle = ::HcommCcuCascCntHandle; // Cascade Counter Handle 数据面别名，与res侧同一句柄实体
    // ==================== 资源创建 ====================

    template <typename T>
    inline T GetResByChannel(ChannelHandle /*channel*/, uint32_t /*index*/)
    {
        static_assert(
            sizeof(T) == 0, "ccu::GetResByChannel<T> is not specialized for this type T; "
                            "currently supported: Variable.");
    }
    template <>
    inline Variable GetResByChannel<Variable>(ChannelHandle channel, uint32_t varIndex)
    {
        Variable v{detail::NoAllocTag{}};
        CCU_THROW_IF_FAILED(
            CcuVariableCreateByChannel(channel, varIndex, &v.handle), "CcuVariableCreateByChannel: failed");
        return v;
    }

    // ==================== 事件 ====================
    inline CcuResult EventRecord(Event e, uint16_t mask = 1) { return CcuEventRecord(e.handle, mask); }
    inline CcuResult EventWait(Event e, uint16_t mask = 1) { return CcuEventWait(e.handle, mask); }
    inline CcuResult EventRecord(const char* notifyTag, uint16_t mask = 1)
    {
        return CcuLocalNotifyRecord(notifyTag, mask);
    }
    inline CcuResult EventWait(const char* notifyTag, uint16_t mask = 1) { return CcuLocalNotifyWait(notifyTag, mask); }
    inline CcuResult NotifyRecord(ChannelHandle channel, uint32_t remoteNotifyIdx, uint16_t mask = 1)
    {
        return CcuNotifyRecord(channel, remoteNotifyIdx, mask);
    }
    inline CcuResult NotifyWait(ChannelHandle channel, uint32_t localNotifyIdx, uint16_t mask = 1)
    {
        return CcuNotifyWait(channel, localNotifyIdx, mask);
    }
    inline CcuResult ChannelIdGet(ChannelHandle channel, uint32_t& channelId)
    {
        return CcuChannelIdGet(channel, &channelId);
    }
    inline CcuResult WriteVariableWithNotify(
        ChannelHandle channel, Variable var, uint32_t remoteVarIdx, uint32_t remoteNotifyIdx, uint16_t mask = 1)
    {
        return CcuWriteVariableWithNotify(channel, var.handle, remoteVarIdx, remoteNotifyIdx, mask);
    }
    // Cascade Counter Wait
    inline CcuResult CascCntWait(CascCounterHandle cntHandle, uint64_t tgtValue)
    {
        return CcuCascCntWait(cntHandle, tgtValue);
    }
    // Cascade Counter Clear
    inline CcuResult CascCntClear(CascCounterHandle cntHandle, Event event, uint16_t mask = 1)
    {
        return CcuCascCntClear(cntHandle, event.handle, mask);
    }

    // ==================== 加载 ====================
    inline CcuResult LoadArg(Variable v, uint32_t argId) { return CcuLoadArg(v.handle, argId); }
    inline CcuResult Load(uint64_t addr, Array<Variable>& vArr, uint32_t num)
    {
        return CcuLoadVar(addr, vArr[0].handle, num);
    }
    inline CcuResult Load(uint64_t addr, Variable v) { return CcuLoadVar(addr, v.handle, 1); }
    inline CcuResult Load(Variable addrVar, Array<Variable>& vArr, uint32_t num)
    {
        return CcuLoadVarFromVarAddr(addrVar.handle, vArr[0].handle, num);
    }
    inline CcuResult Load(Variable addrVar, Variable v) { return CcuLoadVarFromVarAddr(addrVar.handle, v.handle, 1); }
    inline CcuResult LoadAddImm(const Array<Variable>& array, Variable offset, uint16_t immAddValue, Variable dst)
    {
        return CcuLoadAddImm(
            array[0].handle, static_cast<uint16_t>(array.size()), offset.handle, immAddValue, dst.handle);
    }
    inline CcuResult AddImmStore(const Array<Variable>& array, Variable offset, uint16_t immAddValue, Variable src)
    {
        return CcuAddImmStore(
            array[0].handle, static_cast<uint16_t>(array.size()), offset.handle, immAddValue, src.handle);
    }
    inline CcuResult Store(uint64_t addr, Array<Variable>& vArr, uint32_t num)
    {
        return CcuStoreVar(addr, vArr[0].handle, num);
    }
    inline CcuResult Store(uint64_t addr, Variable v) { return CcuStoreVar(addr, v.handle, 1); }
    inline CcuResult Store(Variable addrVar, Array<Variable>& vArr, uint32_t num)
    {
        return CcuStoreVarToVarAddr(addrVar.handle, vArr[0].handle, num);
    }
    inline CcuResult Store(Variable addrVar, Variable v) { return CcuStoreVarToVarAddr(addrVar.handle, v.handle, 1); }

    // ==================== 本地拷贝（3 种重载） ====================
    // LocalAddr → LocalAddr,LocalAddr → CcuBuffer,CcuBuffer → LocalAddr
    inline CcuResult LocalCopy(LocalAddr dst, LocalAddr src, Variable len, Event event, uint16_t mask = 1)
    {
        return CcuLocalCopyMemToMem(dst.handle, src.handle, len.handle, event.handle, mask);
    }
    inline CcuResult LocalCopy(CcuBuffer dst, LocalAddr src, Variable len, Event event, uint16_t mask = 1)
    {
        return CcuLocalCopyMemToBuffer(dst.handle, src.handle, len.handle, event.handle, mask);
    }
    inline CcuResult LocalCopy(LocalAddr dst, CcuBuffer src, Variable len, Event event, uint16_t mask = 1)
    {
        return CcuLocalCopyBufferToMem(dst.handle, src.handle, len.handle, event.handle, mask);
    }

    // ==================== 本地 Reduce ====================
    inline CcuResult LocalReduce(
        LocalAddr dst, LocalAddr src, Variable len, HcclDataType dataType, HcclReduceOp opType, Event event,
        uint16_t mask = 1)
    {
        return CcuLocalMemReduce(dst.handle, src.handle, len.handle, dataType, opType, event.handle, mask);
    }
    inline CcuResult LocalReduce(
        CcuBuffer* buffers, uint32_t count, HcclDataType dataType, HcclDataType outputDataType, HcclReduceOp opType,
        Variable len, Event event, uint16_t mask = 1)
    {
        if (buffers == nullptr || count == 0) {
            return CcuResult::CCU_E_PARA;
        }
        std::vector<CcuBufferHandle> bufHandles(count);
        for (uint32_t i = 0; i < count; i++) {
            bufHandles[i] = buffers[i].handle;
        }
        return CcuLocalBufferReduce(
            bufHandles.data(), count, dataType, outputDataType, opType, len.handle, event.handle, mask);
    }

    // ==================== 远端读====================

    // 远端读 LocalAddr ← RemoteAddr
    inline CcuResult
    Read(ChannelHandle ch, LocalAddr local, RemoteAddr remote, Variable len, Event event, uint16_t mask = 1)
    {
        return CcuReadMemToMem(ch, local.handle, remote.handle, len.handle, event.handle, mask);
    }
    // 远端读 CcuBuffer ← RemoteAddr
    inline CcuResult
    Read(ChannelHandle ch, CcuBuffer local, RemoteAddr remote, Variable len, Event event, uint16_t mask = 1)
    {
        return CcuReadMemToBuffer(ch, local.handle, remote.handle, len.handle, event.handle, mask);
    }
    // 远端读 LocalAddr ← RemoteAddr Reduce (Reduce)
    inline CcuResult ReadReduce(
        ChannelHandle ch, LocalAddr local, RemoteAddr remote, Variable len, HcclDataType dataType, HcclReduceOp opType,
        Event event, uint16_t mask = 1)
    {
        return CcuReadMemToMemReduce(ch, local.handle, remote.handle, len.handle, dataType, opType, event.handle, mask);
    }

    // ==================== 远端写 ====================

    // LocalAddr → RemoteAddr
    inline CcuResult
    Write(ChannelHandle ch, RemoteAddr remote, LocalAddr local, Variable len, Event event, uint16_t mask = 1)
    {
        return CcuWriteMemToMem(ch, remote.handle, local.handle, len.handle, event.handle, mask);
    }
    // CcuBuffer → RemoteAddr
    inline CcuResult
    Write(ChannelHandle ch, RemoteAddr remote, CcuBuffer local, Variable len, Event event, uint16_t mask = 1)
    {
        return CcuWriteBufferToMem(ch, remote.handle, local.handle, len.handle, event.handle, mask);
    }
    // LocalAddr → RemoteAddr Reduce (Reduce)
    inline CcuResult WriteReduce(
        ChannelHandle ch, RemoteAddr remote, LocalAddr local, Variable len, HcclDataType dataType, HcclReduceOp opType,
        Event event, uint16_t mask = 1)
    {
        return CcuWriteMemToMemReduce(
            ch, remote.handle, local.handle, len.handle, dataType, opType, event.handle, mask);
    }
    inline CcuResult
    WriteVarAtomicAdd(Variable channelId, RemoteAddr varAddr, Variable addValue, Event event, uint16_t mask = 1)
    {
        return CcuWriteVarAtomicAdd(channelId.handle, varAddr.handle, addValue.handle, event.handle, mask);
    }
    // Write With Cascade Counter Increase
    inline CcuResult
    WriteWithCascCntInc(Variable channelId, RemoteAddr remote, LocalAddr local, Variable len, RemoteAddr inCntAddr)
    {
        return CcuWriteWithCascCntInc(channelId.handle, remote.handle, local.handle, len.handle, inCntAddr.handle);
    }

} // namespace ccu
} // namespace AscendC

#endif // CCU_PRIMITIVES_HPP
