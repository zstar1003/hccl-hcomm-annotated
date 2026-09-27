/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cnt_notify_res_helper.h"
#include "binary_stream.h"
namespace Hccl {
BaseMemTransport::LocCntNotifyRes
CntNotifyResHelper::GetCntNotifyRes(const unordered_map<u32, vector<LocalCntNotify*>>& topicIdCntNotifyVecMap) const
{
    u32 cntNotifyNum = 0;
    for (auto& it : topicIdCntNotifyVecMap) {
        cntNotifyNum += it.second.size();
    }

    BinaryStream binaryStream;
    BaseMemTransport::LocCntNotifyRes result;
    binaryStream << cntNotifyNum;

    u32 cntNotifyIndex = 0;
    for (auto& it : topicIdCntNotifyVecMap) {
        auto topicId = it.first;
        u32 pos = 0;
        for (auto& notify : it.second) {
            binaryStream << topicId;
            binaryStream << pos;
            binaryStream << cntNotifyIndex;
            result.vec.push_back(notify);
            cntNotifyIndex++;
            pos++;
        }
    }
    binaryStream.Dump(result.desc);
    return result;
}

u32 CntNotifyResHelper::GetIndex(vector<char>& desc, u32 topicId, u32 pos) const
{
    BinaryStream binaryStream(desc);
    u32 cntNotifyNum = 0;
    binaryStream >> cntNotifyNum;
    for (u32 index = 0; index < cntNotifyNum; index++) {
        u32 theTopicId;
        u32 thePos;
        u32 theCntNotifyIndex;
        binaryStream >> theTopicId;
        binaryStream >> thePos;
        binaryStream >> theCntNotifyIndex;
        if (theTopicId == topicId && thePos == pos) {
            return theCntNotifyIndex;
        }
    }
    HCCL_ERROR("cannot find topicId=%u, pos=%u in desc=%s", topicId, pos, Bytes2hex(desc.data(), desc.size()).c_str());
    return 0;
}
} // namespace Hccl
