/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <iostream>
#include "FakeStreamMgr.h"

using namespace std;

bool MemCpy(FakeSqe& sqe)
{
    memcpy(sqe.dst, sqe.src, sqe.count);
    return true;
}

bool MemReduce(FakeSqe& sqe)
{
    if (sqe.reduceOp == rtRecudeKind_t::RT_MEMCPY_SDMA_AUTOMATIC_ADD) {
        if (rtDataType_t::RT_DATA_TYPE_FP32 == sqe.dataType) {
            float* dst = (float*)(sqe.dst);
            float* src = (float*)(sqe.src);
            auto dataCount = sqe.count / sizeof(float);
            for (auto index = 0; index < dataCount; ++index) {
                dst[index] += src[index];
            }
            return true;
        }
    }

    return false;
}

bool ExecuteSqe(FakeSqe& sqe, FakeNotifyMgr* notifyMgr)
{
    if (sqe.type == FakeSqeType::NOTIFY_RECORD) {
        return notifyMgr->Record(sqe.notifyId);
    }

    if (sqe.type == FakeSqeType::NOTIFY_WAIT) {
        return notifyMgr->Wait(sqe.notifyId);
    }

    if (sqe.type == FakeSqeType::MEM_CPY) {
        return MemCpy(sqe);
    }

    if (sqe.type == FakeSqeType::SDMA_REDUCE) {
        return MemReduce(sqe);
    }

    return false;
}

int* FakeStreamMgr::CreateStream(int rank)
{
    lock_guard<mutex> lock(lmutex);
    int currentId = streamIdGen++;
    int* streamId = new int(currentId);

    streamIdRankMap[currentId] = rank;

    streamIds.emplace(streamId);
    return streamId;
}

int FakeStreamMgr::GetRank(int streamId)
{
    auto it = streamIdRankMap.find(streamId);
    if (it == streamIdRankMap.end()) {
        throw std::exception();
    }
    return it->second;
}

bool FakeStreamMgr::HasSqe()
{
    int sum = 0;
    for (auto item : streamIds) {
        sum += stores[*item].size();
    }

    return sum > 0;
}

void FakeStreamMgr::Sync(int streamId)
{
    lock_guard<mutex> lock(lmutex);
    // 虽然每个rank都会sync，但只让第一个rank触发执行就行。 后续rank进来时发现已经没有等待的sqe，直接退出。
    while (HasSqe()) {
        for (auto& stream : stores) {
            auto id = stream.first;
            // 开始按序执行sqe，删掉执行完的sqe，碰到执行不下去的时候退出本次循环
            auto it = stream.second.begin();
            while (it != stream.second.end()) {
                auto ret = ExecuteSqe(*it, fakeNotifyMgr.get());
                if (ret) {
                    it = stream.second.erase(it);
                } else {
                    if (it->type == FakeSqeType::NOTIFY_WAIT) {
                        break;
                    }
                }
            }
        }
    }
}

void FakeStreamMgr::Append(int streamId, FakeSqe sqe)
{
    lock_guard<mutex> lock(lmutex);
    if (stores.find(streamId) == stores.end()) {
        stores.emplace(streamId, vector<FakeSqe>());
    }

    stores[streamId].push_back(sqe);
}

void FakeStreamMgr::DestroyStream(int* streamId)
{
    lock_guard<mutex> lock(lmutex);
    // delete streamId;
    streamIds.erase(streamId);
}

FakeNotifyMgr* FakeStreamMgr::GetFakeNotifyMgr() { return fakeNotifyMgr.get(); }

FakeStreamMgr::~FakeStreamMgr()
{
    for (auto item : streamIds) {
        delete item;
    }
}

int* FakeNotifyMgr::CreateNotify(int rank)
{
    lock_guard<mutex> lock(lmutex);
    int currentNotifyId = notifyIdGen++;
    int* notify = new int(currentNotifyId);
    notifyIds.emplace(notify);
    notifyStatusMap[currentNotifyId] = 0;
    notifyRankMap[currentNotifyId] = rank;
    return notify;
}

bool FakeNotifyMgr::Record(int notifyId)
{
    lock_guard<mutex> lock(lmutex);
    auto it = notifyStatusMap.find(notifyId);
    if (it == notifyStatusMap.end()) {
        throw std::exception();
    }

    it->second = 1;
    return true;
}

bool FakeNotifyMgr::Wait(int notifyId)
{
    lock_guard<mutex> lock(lmutex);
    // wait条件满足，可以继续执行返回true； 需要继续等待返回false；

    auto it = notifyStatusMap.find(notifyId);
    if (it == notifyStatusMap.end()) {
        throw std::exception();
    }

    if (it->second == 1) {
        it->second = 0;
        return true;
    }

    return false;
}

void FakeNotifyMgr::DestroyNotify(int* notifyId)
{
    lock_guard<mutex> lock(lmutex);

    if (notifyIds.find(notifyId) == notifyIds.end()) {
        return;
    }

    notifyIds.erase(notifyIds.find(notifyId));
    delete notifyId;
}

int FakeNotifyMgr::GetRank(int notifyId)
{
    auto it = notifyStatusMap.find(notifyId);
    if (it == notifyStatusMap.end()) {
        throw std::exception();
    }
    return it->second;
}

FakeNotifyMgr::~FakeNotifyMgr()
{
    for (auto item : notifyIds) {
        delete item;
    }
}
