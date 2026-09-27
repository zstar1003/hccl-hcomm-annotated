/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

extern "C" {
#include "ut_dispatch.h"
#include "tc_ut_rs.h"
#include "tc_ut_rs_ping.h"
#include "user_log.h"
#include <sys/epoll.h>

extern void RsEpollEventHandleOne(struct rs_cb* rs_cb, struct epoll_event* events);
extern void RsEpollEventInHandle(struct rs_cb* rs_cb, struct epoll_event* events);
}

#include "gtest/gtest.h"
#include <stdio.h>
#include <mockcpp/mockcpp.hpp>

using namespace std;

#define RS_CHECK_POINTER_NULL_RETURN_VOID(ptr) \
    do {                                       \
        if ((ptr) == NULL) {                   \
            hccp_err("pointer is NULL!");      \
            return;                            \
        }                                      \
    } while (0)

void rs_epoll_event_handle_one_stub(struct rs_cb* rs_cb, struct epoll_event* events)
{
    RS_CHECK_POINTER_NULL_RETURN_VOID(events);
    RS_CHECK_POINTER_NULL_RETURN_VOID(rs_cb);
    if (events->events & EPOLLIN) {
        RsEpollEventInHandle(rs_cb, events);
    } else {
        hccp_warn("unknown event(0x%x) !", events->events);
    }

    return;
}

class RS : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "\033[36m--RoCE RS SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase() { std::cout << "\033[36m--RoCE RS TearDown--\033[0m" << std::endl; }
    virtual void SetUp()
    {
        MOCKER(RsEpollEventHandleOne).expects(atMost(100000)).will(invoke(rs_epoll_event_handle_one_stub));
    }
    virtual void TearDown() { GlobalMockObject::verify(); }
};

TEST_M(RS, TcRsInit2);
TEST_M(RS, TcRsDeinit2);

TEST_M(RS, TcRsSocketInit);
TEST_M(RS, TcRsSocketDeinit);

TEST_M(RS, TcRsRdevInit);
TEST_M(RS, TcRsRdevDeinit);

TEST_M(RS, TcRsGetTsqpDepthAbnormal);
TEST_M(RS, TcRsSetTsqpDepthAbnormal);

TEST_M(RS, TcRsSocketListenStart2);
TEST_M(RS, TcRsQpCreate2);

TEST_M(RS, TcRsAbnormal2);

TEST_M(RS, TcRsMrAbnormal2);
TEST_M(RS, TcRsGetGidIndex2);
TEST_M(RS, TcRsQpConnectAsync2);
TEST_M(RS, TcRsSendWr2);

TEST_M(RS, TcRsSocketNodeid2vnic);
TEST_M(RS, TcRsServerValidAsyncInit);
TEST_M(RS, TcRsConnectHandle);
TEST_M(RS, TcRsGetQpContext);
TEST_M(RS, TcRsSocketGetBindByChip);
TEST_M(RS, TcRsSocketBatchAbort);
TEST_M(RS, TcRsTcpRecvTagInHandle);
TEST_M(RS, TcRsServerValidAsyncAbnormal);
TEST_M(RS, TcRsServerValidAsyncAbnormal01);
TEST_M(RS, TcRsNetApiInitFail);

/* pingMesh ut cases */
TEST_M(RS, TcRsPingHandleInit);
TEST_M(RS, TcRsPingHandleDeinit);
TEST_M(RS, TcRsPingInit);
TEST_M(RS, TcRsPingTargetAdd);
TEST_M(RS, TcRsPingTaskStart);
TEST_M(RS, TcRsPingGetResults);
TEST_M(RS, TcRsPingTaskStop);
TEST_M(RS, TcRsPingTargetDel);
TEST_M(RS, TcRsPingDeinit);
TEST_M(RS, TcRsPingUrmaCheckFd);
TEST_M(RS, TcRsPingCbGetIbCtxAndIndex);
