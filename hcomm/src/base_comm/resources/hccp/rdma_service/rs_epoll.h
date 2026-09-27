/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RS_EPOLL_H
#define RS_EPOLL_H

#define _GNU_SOURCE
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "securec.h"
#include "rs.h"
#include "rs_inner.h"

#define RS_CONN_USLEEP_TIME 200000
#define RS_PROMOTE_CONN_USLEEP_TIME 5000
#define RS_EPOLL_EVENT 64

int RsEpollConnectHandleInit(struct rs_cb *rscb);
int RsEpollCtl(int epollfd, int op, int fd, unsigned int state);
int RsEpollCtlFdHandle(int epollfd, int op, int fd, unsigned int state, void *fdHandle);
void RsDestroyEpoll(struct rs_cb *rsCb);
int RsEpollCreateEpollfd(int *epollfd);
int RsEpollDestroyFd(int *fd);
int RsEpollWaitHandle(int eventHandle, struct epoll_event *events, int timeout, unsigned int maxevents,
    unsigned int *eventsNum);
int RsEpollEventListenInHandle(struct rs_cb *rsCb, int fd);
int RsEpollEventQpMrInHandle(struct rs_cb *rsCb, int fd);
int RsSocketCopyConnInfo(struct RsConnInfo *connTmp, struct RsConnInfo *conn);
int RsWhiteListCheckValid(unsigned int chipId, struct RsConnCb *connCb, struct RsConnInfo *conn);
#endif // RS_EPOLL_H
