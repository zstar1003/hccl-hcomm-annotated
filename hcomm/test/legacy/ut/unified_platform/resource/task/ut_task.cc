/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>
#include "log.h"
#include "task.h"
#include "null_ptr_exception.h"
#include "internal_exception.h"
#include "ipc_remote_notify.h"
#include "ipc_local_notify.h"

using namespace Hccl;
using namespace std;

class TaskTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TaskTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "TaskTest tests tear down." << std::endl; }

    virtual void SetUp()
    {
        InitLocalNotify();
        ipcRemoteNotify = new IpcRemoteNotify();
        ipcLocalNotify = new IpcLocalNotify();

        taskLocalCopy = new TaskLocalCopy(dstAddr, srcAddr, size, kind);
        taskP2pMemcpy = new TaskP2pMemcpy(dstAddr, srcAddr, size, kind);
        remoteRecord = new TaskRemoteRecord(ipcRemoteNotify);
        taskWait = new TaskWait(ipcLocalNotify);
        localRecord = new TaskLocalRecord(ipcLocalNotify);
        sdmaReduce = new TaskSdmaReduce(dstAddr, srcAddr, size, dataType, reduceOp);
        localReduce = new TaskLocalReduce(srcAddr1, srcAddr2, dstAddr, size, dataType, reduceOp);
        rdmaSendDb = new TaskRdmaSend(dbIndex, dbInfo);
        rdmaSendQpn = new TaskRdmaSend(qpn, wqeIndex);
        localAddrCopy = new TaskLocalAddrCopy(dstAddr, srcAddr, size);
        ubDbSend = new TaskUbDbSend(jettyId, funcId, piVal, dieId);
        ubDirectSend = new TaskUbDirectSend(funcId, dieId, jettyId, dwqeSize, dwqe);
        writeDoorbell = new TaskWriteValue(dbAddr, piVal);
        std::cout << "A Test case in TaskTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        delete taskLocalCopy;
        delete taskP2pMemcpy;
        delete remoteRecord;
        delete taskWait;
        delete localRecord;
        delete sdmaReduce;
        delete localReduce;
        delete rdmaSendDb;
        delete rdmaSendQpn;
        delete localAddrCopy;
        delete ipcRemoteNotify;
        delete ipcLocalNotify;
        delete ubDbSend;
        delete ubDirectSend;
        delete writeDoorbell;
        GlobalMockObject::verify();
        std::cout << "A Test case in TaskTest TearDown" << std::endl;
    }

    void InitLocalNotify()
    {
        MOCKER(HrtGetDevice).stubs().will(returnValue(0));
        MOCKER(HrtNotifyCreate).stubs().will(returnValue((void*)(1)));
        MOCKER(HrtIpcSetNotifyName).stubs();
        MOCKER(HrtGetNotifyID).stubs().will(returnValue(1));
        MOCKER(HrtNotifyGetAddr).stubs().will(returnValue((u64)0));
        MOCKER(HrtNotifyGetOffset).stubs().will(returnValue(1));
        MOCKER(HrtGetSocVer).stubs();
        MOCKER(HrtGetDeviceType).stubs().will(returnValue(DevType(DevType::DEV_TYPE_910A2)));
    }

    TaskLocalCopy* taskLocalCopy;
    TaskP2pMemcpy* taskP2pMemcpy;
    TaskSdmaReduce* sdmaReduce;
    TaskLocalReduce* localReduce;
    TaskRdmaSend* rdmaSendDb;
    TaskRdmaSend* rdmaSendQpn;
    TaskLocalAddrCopy* localAddrCopy;

    TaskLocalRecord* localRecord;
    TaskWait* taskWait;
    TaskRemoteRecord* remoteRecord;
    TaskUbDirectSend* ubDirectSend;
    TaskWriteValue* writeDoorbell;
    TaskUbDbSend* ubDbSend;

    u32 dbIndex = 100;
    u64 dbInfo = 200;
    u32 qpn = 100;
    u32 wqeIndex = 200;

    u64 srcAddr = 0x100;
    u64 dstAddr = 0x200;
    u64 size = 0x300;

    u64 srcAddr1 = 0x500;
    u64 srcAddr2 = 0x300;

    u32 funcId = 0;
    u32 dieId = 0;
    u64 dbAddr = 0;
    u32 piVal = 0;
    u32 jettyId = 18;
    u32 dwqeSize = 128;
    u8 dwqe[128]{0};

    MemcpyKind kind = MemcpyKind::D2D;
    DataType dataType = DataType::INT8;
    ReduceOp reduceOp = ReduceOp::MAX;

    IpcRemoteNotify* ipcRemoteNotify;
    IpcLocalNotify* ipcLocalNotify;
};

TEST_F(TaskTest, test_task_print_all)
{
    cout << taskLocalCopy->Describe() << endl;
    cout << taskP2pMemcpy->Describe() << endl;
    cout << remoteRecord->Describe() << endl;
    cout << taskWait->Describe() << endl;
    cout << localRecord->Describe() << endl;
    cout << sdmaReduce->Describe() << endl;
    cout << localReduce->Describe() << endl;
    cout << rdmaSendDb->Describe() << endl;
    cout << rdmaSendQpn->Describe() << endl;
    cout << localAddrCopy->Describe() << endl;
    cout << ubDbSend->Describe() << endl;
    cout << ubDirectSend->Describe() << endl;
    cout << writeDoorbell->Describe() << endl;
}

TEST_F(TaskTest, test_task_ub_direct_send_exception)
{
    EXPECT_THROW(TaskUbDirectSend(funcId, dieId, jettyId, 0, dwqe), InternalException);
}
