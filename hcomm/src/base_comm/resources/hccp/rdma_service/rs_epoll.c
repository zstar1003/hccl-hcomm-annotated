/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define _GNU_SOURCE
#include "config_log.h"
#include <unistd.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/eventfd.h>
#include <sys/epoll.h>
#include <sys/syscall.h>
#include <sys/prctl.h>

#include "rs_tls.h"
#include "ssl_adp.h"
#include "securec.h"
#include "rs.h"
#include "rs_inner.h"
#include "ra_rs_comm.h"
#include "ra_rs_err.h"
#include "rs_drv_rdma.h"
#include "dl_hal_function.h"
#include "rs_drv_socket.h"
#include "rs_socket.h"
#ifdef CONFIG_TLV
#include "rs_adp_nslb.h"
#endif
#include "rs_ub_dfx.h"
#include "rs_epoll.h"

#define BIND_MIN_DCPU_NUM 1
#define COMMAND_LENGTH 100
#define ADD_TID_SHELL_PATH 50
#define TID_LENGTH 20
#define HOST ((uint32_t)1)

struct RsPthreadInfo gEpollThreadInfo = {0}; // lint !e17
struct RsPthreadInfo gConnectThreadInfo = {0};

/*
 * opcode:
 *  ADD: EPOLL_CTL_ADD
 *  MOD: EPOLL_CTL_MOD
 *  DEL: EPOLL_CTL_DEL
 */
int RsEpollCtl(int epollfd, int op, int fd, unsigned int state)
{
    int ret;
    struct epoll_event ev;

    ev.events = state;
    ev.data.fd = fd;
    ret = epoll_ctl(epollfd, op, fd, &ev);
    if (ret) {
        hccp_warn_socket("epoll_ctl for fd %d unsuccessful! ret:%d errno:%d op:%d state:%u", fd, ret, errno, op, state);
    }
    return ret;
}

int RsEpollCtlFdHandle(int epollfd, int op, int fd, unsigned int state, void *fdHandle)
{
    int ret;
    struct epoll_event ev;

    ev.events = state;
    ev.data.ptr = fdHandle;
    ret = epoll_ctl(epollfd, op, fd, &ev);
    if (ret) {
        hccp_warn_socket("epoll_ctl for fd %d unsuccessful! ret:%d errno:%d op:%d state:%u", fd, ret, errno, op, state);
    }
    return ret;
}

int RsWlistCheckConnAdd(struct rs_cb *rsCb, struct RsConnInfo *connTmp)
{
    int ret;
    int retClose;
    struct RsConnInfo *conn = NULL;

    if (rsCb->connCb.wlistEnable == 1) {
        ret = RsWhiteListCheckValid(rsCb->chipId, &rsCb->connCb, connTmp);
        if (ret) {
            hccp_info_socket(
                "invalid client node found: chip_id %u, fd %d accept, server ip:%s, client ip:%s, port:%u, "
                "state:%d, tag:%s",
                rsCb->chipId, connTmp->connfd, connTmp->serverIp.readAddr, connTmp->clientIp.readAddr, connTmp->port,
                connTmp->state, connTmp->tag);
            return ret;
        }
    }

    /* add conn node to server list */
    ret = RsAllocConnNode(&conn, connTmp->port);
    if (ret) {
        hccp_err("server IP:0x%s add conn info to list failed, fd:%d, ret:%d", connTmp->serverIp.readAddr,
            connTmp->connfd, ret);
        goto alloc_err;
    }

    ret = RsSocketCopyConnInfo(connTmp, conn);
    if (ret) {
        hccp_err("rs_socket_copy_conn_info failed, ret[%d]", ret);
        goto out;
    }

    if (rsCb->sslEnable == RS_SSL_ENABLE) {
        ret = RsEpollCtl(rsCb->connCb.epollfd, EPOLL_CTL_DEL, connTmp->connfd, EPOLLIN);
        if (ret) {
            hccp_err("rs epoll ctl failed, ret %d", ret);
            goto out;
        }
    }

    RS_PTHREAD_MUTEX_LOCK(&rsCb->connCb.connMutex);
    RsListAddTail(&conn->list, &rsCb->connCb.serverConnList);
    RS_PTHREAD_MUTEX_ULOCK(&rsCb->connCb.connMutex);

    hccp_info_socket("[Server]chip_id %u, fd %d accept, server ip:%s, client ip:%s, state:%d, tag:%s", rsCb->chipId,
        connTmp->connfd, connTmp->serverIp.readAddr, connTmp->clientIp.readAddr, connTmp->state, connTmp->tag);
    ShowConnNode(&(rsCb->connCb.serverConnList));
    return 0;

out:
    free(conn);
    conn = NULL;
alloc_err:
    RS_CLOSE_RETRY_FOR_EINTR(retClose, connTmp->connfd);
    return ret;
}

STATIC int RsSslRecvTagInHandle(struct RsAcceptInfo *acceptInfo, struct RsConnInfo *connTmp)
{
    int expSize = SOCK_CONN_TAG_SIZE + SOCK_CONN_DEV_ID_SIZE;
    char *recvBuff = connTmp->tag;
    struct timeval startTime, now;
    float timeCost = 0.0;
    int size = 0;
    int ret = 0;

    RsGetCurTime(&startTime);
    while (expSize > 0) {
        connTmp->tagSyncTimes++;
        size = ssl_adp_read(acceptInfo->ssl, recvBuff, expSize);
        if (size > 0) {
            expSize -= size;
            recvBuff += size;
        } else {
            ret = RsSslReadInnerCheck(acceptInfo->ssl, acceptInfo->connFd, size, (uint64_t)expSize);
            if (ret == -EAGAIN) {
                connTmp->tagEintrTimes++;
            } else {
                hccp_run_warn("ssl_adp_read unsuccessful, fd:%d ret:%d tagSyncTime:%u tagEintrTime:%u",
                    acceptInfo->connFd, ret, connTmp->tagSyncTimes, connTmp->tagEintrTimes);
                return -ESOCKCLOSED;
            }
        }

        RsGetCurTime(&now);
        HccpTimeInterval(&now, &startTime, &timeCost);
        if (timeCost >= RS_RECV_MAX_TIME) {
            hccp_run_info("recv tag time out, server:{%s:%u} client:%s tagSyncTime:%u tagEintrTime:%u",
                acceptInfo->serverIpAddr.readAddr, acceptInfo->sockPort, acceptInfo->clientIpAddr.readAddr,
                connTmp->tagSyncTimes, connTmp->tagEintrTimes);
            return -ETIME;
        }

        if (timeCost <= 0) {
            RsGetCurTime(&startTime);
        }
    }

    connTmp->serverIp = acceptInfo->serverIpAddr;
    connTmp->clientIp = acceptInfo->clientIpAddr;
    connTmp->connfd = acceptInfo->connFd;
    connTmp->state = RS_CONN_STATE_TAG_SYNC;
    connTmp->port = acceptInfo->sockPort;
    connTmp->ssl = acceptInfo->ssl;

    hccp_info_socket("recv tag success, server:{%s:%u} client:%s timeCost:%fms tagSyncTime:%u tagEintrTime:%u",
        acceptInfo->serverIpAddr.readAddr, acceptInfo->sockPort, acceptInfo->clientIpAddr.readAddr, timeCost,
        connTmp->tagSyncTimes, connTmp->tagEintrTimes);
    return 0;
}

STATIC void RsEpollEventSslRecvTagInHandle(struct rs_cb *rsCb, struct RsAcceptInfo *acceptInfo)
{
    struct RsConnInfo connTmp = {0};
    int retClose;
    int ret;

    ret = RsSslRecvTagInHandle(acceptInfo, &connTmp);
    if (ret != 0) {
        RS_CLOSE_RETRY_FOR_EINTR(retClose, acceptInfo->connFd);
        goto out;
    }

    ret = RsWlistCheckConnAdd(rsCb, &connTmp);
out:
    if (ret != 0) {
        hccp_warn_socket("recv tag or add conn unsuccessful ret:%d", ret);
        ssl_adp_shutdown(acceptInfo->ssl);
        ssl_adp_free(acceptInfo->ssl);
        acceptInfo->ssl = NULL;
    }

    /* do not shutdown ssl */
    RS_PTHREAD_MUTEX_LOCK(&rsCb->connCb.connMutex);
    RsListDel(&acceptInfo->list);
    free(acceptInfo);
    acceptInfo = NULL;
    RS_PTHREAD_MUTEX_ULOCK(&rsCb->connCb.connMutex);

    return;
}

STATIC void RsDoSslHandshake(struct RsAcceptInfo *acceptInfo, struct rs_cb *rscb)
{
    int ret;
    int err;

    ret = ssl_adp_do_handshake(acceptInfo->ssl);
    if (ret == 1) {
        ret = rs_tls_peer_cert_verify(acceptInfo->ssl, rscb);
        if (ret) {
            hccp_err("tls verify peer cert failed");
            return;
        }
        acceptInfo->state = RS_CONN_STATE_SSL_CONNECTED;
    } else {
        err = ssl_adp_get_error(acceptInfo->ssl, ret);
        if (err == SSL_ERROR_WANT_WRITE) {
            hccp_info_socket("return want write");
            return;
        } else if (err == SSL_ERROR_WANT_READ) {
            hccp_info_socket("return want read");
            return;
        } else {
            rs_ssl_err_string(acceptInfo->connFd, err);
            return;
        }
    }

    return;
}

STATIC int RsEpollEventSslAcceptInHandle(struct rs_cb *rsCb, int fd)
{
    int ret;

    struct RsAcceptInfo *acceptInfo = NULL;
    struct RsAcceptInfo *acceptInfo2 = NULL;

    /* Server event: ssl accept */
    RS_LIST_GET_HEAD_ENTRY(acceptInfo, acceptInfo2, &rsCb->connCb.serverAcceptList, list, struct RsAcceptInfo);
    for (; (&acceptInfo->list) != &rsCb->connCb.serverAcceptList;
         acceptInfo = acceptInfo2, acceptInfo2 = list_entry(acceptInfo2->list.next, struct RsAcceptInfo, list)) {
        /* connection request for Server */
        if (fd == acceptInfo->connFd) {
            if (acceptInfo->ssl == NULL) {
                acceptInfo->ssl = ssl_adp_new(rsCb->serverSslCtx);
                CHK_PRT_RETURN(acceptInfo->ssl == NULL, hccp_err("server ssl ctx alloc failed"), -ENOMEM);

                ret = ssl_adp_set_fd(acceptInfo->ssl, acceptInfo->connFd);
                if (ret != 1) {
                    hccp_err("bind connfd and ssl failed, ret %d", ret);
                    ssl_adp_shutdown(acceptInfo->ssl);
                    ssl_adp_free(acceptInfo->ssl);
                    acceptInfo->ssl = NULL;
                    return -EINVAL;
                }

                ssl_adp_set_mode(acceptInfo->ssl, SSL_MODE_AUTO_RETRY);
                ssl_adp_set_accept_state(acceptInfo->ssl);
            }

            if (acceptInfo->state == RS_CONN_STATE_RESET) {
                RsDoSslHandshake(acceptInfo, rsCb);
                return 0;
            }

            if (acceptInfo->state == RS_CONN_STATE_SSL_CONNECTED) {
                RsEpollEventSslRecvTagInHandle(rsCb, acceptInfo);
                return 0;
            }
        }
    }

    return -ENODEV;
}

STATIC int RsEpollTcpRecv(struct rs_cb *rsCb, int fd)
{
    if (rsCb->tcpRecvCallback != NULL) {
        rsCb->tcpRecvCallback(rsCb->fdMap[fd]);
    } else {
        hccp_err("[rs_epoll_tcp_recv]tcp_recv_callback is null.");
        return -EINVAL;
    }
    return 0;
}

STATIC int RsEpollEventHeterogTcpRecvInHandle(struct rs_cb *rsCb, int fd)
{
    int ret;
    struct RsHeterogTcpFdInfo *fdNode = NULL;
    struct RsHeterogTcpFdInfo *fdNode1 = NULL;

    RS_LIST_GET_HEAD_ENTRY(fdNode, fdNode1, &rsCb->heterogTcpFdList, list, struct RsHeterogTcpFdInfo);
    for (; (&fdNode->list) != &rsCb->heterogTcpFdList;
         fdNode = fdNode1, fdNode1 = list_entry(fdNode1->list.next, struct RsHeterogTcpFdInfo, list)) {
        if (fdNode->fd == fd) {
            // 处理tcp recv
            ret = RsEpollTcpRecv(rsCb, fd);
            return ret;
        }
    }
    return -ENODEV;
}

STATIC void RsEpollEventInHandle(struct rs_cb *rsCb, struct epoll_event *events)
{
    int fd = events->data.fd;
    int ret;

    ret = RsEpollEventPingHandle(rsCb, fd);
    if (ret != -ENODEV) {
        hccp_info_socket("the fd:%d is for ping, no need to go on, ret:%d", fd, ret);
        return;
    }

    ret = RsEpollEventListenInHandle(rsCb, fd);
    if (ret != -ENODEV) {
        hccp_info_socket("the fd:%d is tcp listened, no need to go on, ret:%d", fd, ret);
        return;
    }

    if (rsCb->sslEnable == RS_SSL_ENABLE) {
        ret = RsEpollEventSslAcceptInHandle(rsCb, fd);
        if (ret != -ENODEV) {
            hccp_info_socket("the fd:%d is ssl accept, no need to go on, ret:%d", fd, ret);
            return;
        }
    }

    ret = RsEpollEventQpMrInHandle(rsCb, fd);
    if (ret != -ENODEV) {
        hccp_info_socket("the fd:%d is for qp mr, no need to go on, ret:%d", fd, ret);
        return;
    }

    ret = RsEpollEventHeterogTcpRecvInHandle(rsCb, fd);
    if (ret != -ENODEV) {
        hccp_info_socket("the fd:%d is for tcp recv, no need to go on, ret:%d", fd, ret);
        return;
    }

    if (RsIsUdmaSupported()) {
        ret = RsEpollEventJfcInHandle(rsCb, fd);
        if (ret != -ENODEV) {
            hccp_info_socket("the fd:%d is for poll jfc, no need to go on, ret:%d", fd, ret);
            return;
        }

        ret = RsEpollEventUrmaAsyncEventInHandle(rsCb, fd);
        if (ret != -ENODEV) {
            hccp_info_socket("the fd:%d is for urma async event, no need to go on, ret:%d", fd, ret);
            return;
        }
    }

    return;
}

STATIC void RsEpollEventHandleOne(struct rs_cb *rsCb, struct epoll_event *events)
{
    int ret = 0;

    (void)ret;
    RS_CHECK_POINTER_NULL_RETURN_VOID(events);
    RS_CHECK_POINTER_NULL_RETURN_VOID(rsCb);

#ifdef CONFIG_TLV
    if (RsIsTlvSupported()) {
        ret = RsEpollNslbEventHandle(&rsCb->tlvCb.nslbCb, events->data.fd, events->events);
        if (ret != -ENODEV) {
            hccp_info_socket("the fd:%d is nslb event, no need to go on, ret:%d", events->data.fd, ret);
            return;
        }
    }
#endif

    if (events->events & EPOLLIN) {
        if (events->events & EPOLLRDHUP) {
            hccp_dbg("Peer socket has been closed!");
            return;
        }
        RsEpollEventInHandle(rsCb, events);
    } else {
        hccp_warn_socket("unknown event(0x%x)!", events->events);
    }

    return;
}

STATIC int SetAffinity(unsigned int chipId, unsigned int cpuId)
{
    cpu_set_t mask;
    cpu_set_t get;
    int ret;

    CPU_ZERO(&mask);

    // 设置线程CPU亲和力
    CPU_SET(cpuId, &mask);
    ret = pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
    CHK_PRT_RETURN(ret < 0, hccp_err("could not set CPU affinity"), ret);

    // 获取线程cpu亲和力
    CPU_ZERO(&get);
    ret = pthread_getaffinity_np(pthread_self(), sizeof(get), &get);
    CHK_PRT_RETURN(ret, hccp_err("could not get CPU affinity"), ret);

    if (CPU_ISSET(cpuId, &get)) { // 检查cpuid是否在这个集合中
        hccp_info_others("dev is %d thread %llu is running in processor %d.", chipId, pthread_self(), cpuId);
    } else {
        hccp_warn_others("dev is %d thread %llu is not running in processor %d.", chipId, pthread_self(), cpuId);
    }

    return ret;
}

STATIC int BindDataCpu(unsigned int chipId)
{
    int ret;
    unsigned int cpuId;
    uint32_t info = 0;
    uint32_t devNum = 0;
    int64_t ccpuNum;
    int64_t dcpuNum;
    int64_t acpuNum;
    int64_t cpuNum;

    // 判断当前处于host or device侧，如果在host侧，无需进行绑核操作
    ret = DlDrvGetPlatformInfo(&info);
    CHK_PRT_RETURN(ret, hccp_err("get PlatformInfo failed, ret[%d]", ret), ret);
    if (info == HOST) {
        hccp_info_others("host not need bind cpu, info[%u]", info);
        return 0;
    }

    // 获取当前os的device数量
    ret = DlDrvGetDevNum(&devNum);
    CHK_PRT_RETURN(ret, hccp_err("get device_num failed, ret[%d]", ret), ret);
    if (devNum == 0) {
        hccp_info_others("no device need bind cpu, device num %u", devNum);
        return 0;
    }

    // 获取data cpu数量
    ret = DlHalGetDeviceInfo(chipId, MODULE_TYPE_DCPU, INFO_TYPE_CORE_NUM, &dcpuNum);
    CHK_PRT_RETURN(ret, hccp_err("get dcpu_num failed, ret(%d)", ret), ret);

    // 如果dcpu_num < 1，无需绑核直接返回
    if (dcpuNum < BIND_MIN_DCPU_NUM) {
        hccp_info_others("data cpu num %d, device not need bind cpu", dcpuNum);
        return 0;
    }

    // 获取ctrl cpu数量
    ret = DlHalGetDeviceInfo(chipId, MODULE_TYPE_CCPU, INFO_TYPE_CORE_NUM, &ccpuNum);
    CHK_PRT_RETURN(ret, hccp_err("get ccpu_num failed, ret(%d)", ret), ret);

    // 获取ai cpu数量
    ret = DlHalGetDeviceInfo(chipId, MODULE_TYPE_AICPU, INFO_TYPE_CORE_NUM, &acpuNum);
    CHK_PRT_RETURN(ret, hccp_err("get acpu_num failed, ret(%d)", ret), ret);

    // 计算单个device上的核数
    cpuNum = ccpuNum + dcpuNum + acpuNum;
    hccp_info_others("halGetDeviceInf chip = %u dev_num = %u, ccpu = %lld dcpu = %lld acpu = %lld cpuNum = %lld",
        chipId, devNum, ccpuNum, dcpuNum, acpuNum, cpuNum);

    cpuId = (unsigned int)((int64_t)(chipId % devNum) * cpuNum + ccpuNum);

    // 进行绑核
    ret = DlHalBindCgroup(BIND_DATACPU_CGROUP);
    CHK_PRT_RETURN(ret, hccp_err("bind cgroup failed, ret[%d], strerror[%s]", ret, strerror(errno)), ret);
    hccp_info_others("bind cgroup success!");

    ret = SetAffinity(chipId, cpuId);
    CHK_PRT_RETURN(ret, hccp_err("set affinity with cpu[%u] failed", cpuId), ret);

    hccp_info_others("chip_id[%u] bind data cpu[%u] success", chipId, cpuId);

    return ret;
}

STATIC void RsSocketSaveEpollWaitErrInfo(int eventNum, int errNo, struct SocketErrInfo *epollErrInfo)
{
    if (eventNum > 0) {
        return;
    }

    RsSocketSaveErrInfo(RS_CONN_STATE_RESET, errNo, epollErrInfo);
}

STATIC void *RsEpollHandle(void *arg)
{
    struct RsConnCb *connCb = NULL;
    struct rs_cb *rsCb = NULL;
    eventfd_t val;
    int ret;
    int num;
    int i;

    RS_CHECK_POINTER_NULL_RETURN_NULL(arg);

    hccp_info_others("<EPOLL> thread begin! thread_id:%lu, pid:%d, ppid:%d", pthread_self(), getpid(), getppid());

    struct epoll_event events[RS_EPOLL_EVENT];

    CHK_PRT_RETURN(pthread_detach(pthread_self()),
        hccp_err("pthread_detach failed! thread_id:%lu, errno:%d", pthread_self(), errno), NULL);

    (void)prctl(PR_SET_NAME, (uintptr_t) "hccp_epoll", 0, 0, 0);

    rsCb = (struct rs_cb *)arg;
    gRsCb = rsCb;
    connCb = &rsCb->connCb;

    ret = BindDataCpu(rsCb->chipId);
    if (ret) {
        hccp_warn_others("bind data cpu unsuccessful! thread_id:%lu, errno:%d", pthread_self(), errno);
    }

    rsCb->state &= ~RS_STATE_HALT;
    RsGetCurTime(&gEpollThreadInfo.lastCheckTime);
    ret = strncpy_s((char *)gEpollThreadInfo.pthreadName, sizeof(gEpollThreadInfo.pthreadName), "epoll_pthread",
        strlen("epoll_pthread"));
    CHK_PRT_RETURN(ret, hccp_err("strncpy_s pthread name failed, ret[%d]", ret), NULL);

    hccp_run_info("pthread[%s] is alive!", gEpollThreadInfo.pthreadName);

    ret = RsDrvInitCqeErrInfo();
    CHK_PRT_RETURN(ret, hccp_err("rs_drv_init_cqe_err_info failed, ret[%d]", ret), NULL);
    while (1) {
        do {
            num = epoll_wait(connCb->epollfd, events, RS_EPOLL_EVENT, -1);
        } while ((num < 0) && (errno == EINTR));
        RsSocketSaveEpollWaitErrInfo(num, -errno, &connCb->epollErrInfo);

        /* eventfd is for wake up epoll wait, value is ignored */
        if (events[0].data.fd == connCb->eventfd) {
            hccp_warn_others("<EPOLL> SHUT DOWN event eventfd:%d", connCb->eventfd);
            do {
                num = read(connCb->eventfd, &val, sizeof(eventfd_t));
            } while ((num < 0) && (errno == EINTR));
            RS_PTHREAD_MUTEX_LOCK(&rsCb->mutex);
            rsCb->state |= RS_STATE_HALT;
            RS_PTHREAD_MUTEX_ULOCK(&rsCb->mutex);

            break;
        }

        RsHeartbeatAlivePrint(&gEpollThreadInfo);
        RS_PTHREAD_MUTEX_LOCK(&rsCb->mutex);
        /* events handle one by one */
        for (i = 0; i < num; i++) {
            RsEpollEventHandleOne(rsCb, &events[i]);
        }

        RS_PTHREAD_MUTEX_ULOCK(&rsCb->mutex);
    }
    RsDrvDeinitCqeErrInfo();
    hccp_info_others("<EPOLL> QUIT thread_id:%lu, pid:%d, read num:%d", pthread_self(), getpid(), num);

    return NULL;
}

STATIC int RsCreateEpoll(struct rs_cb *rsCb)
{
    int ret, retFd;
    struct RsConnCb *connCb = NULL;

    connCb = &rsCb->connCb;

    connCb->epollfd = epoll_create(1);
    CHK_PRT_RETURN(connCb->epollfd < 0, hccp_err("epollfd[%d] failed, errno[%d]", connCb->epollfd, errno), -EINVAL);

    connCb->eventfd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    if (connCb->eventfd == RS_FD_INVALID) {
        RS_CLOSE_RETRY_FOR_EINTR(retFd, connCb->epollfd);
        connCb->epollfd = RS_FD_INVALID;
        hccp_err("create eventfd for rs cb failed !");
        return -EINVAL;
    }

    ret = RsEpollCtl(connCb->epollfd, EPOLL_CTL_ADD, connCb->eventfd, EPOLLIN);
    if (ret) {
        hccp_err("re epoll ctl failed, epollfd[%d], eventfd[%d]", connCb->epollfd, connCb->eventfd);
        RS_CLOSE_RETRY_FOR_EINTR(retFd, connCb->eventfd);
        connCb->eventfd = RS_FD_INVALID;
        RS_CLOSE_RETRY_FOR_EINTR(retFd, connCb->epollfd);
        connCb->epollfd = RS_FD_INVALID;
        return ret;
    }

    return 0;
}

void RsDestroyEpoll(struct rs_cb *rsCb)
{
    int ret;
    struct RsConnCb *connCb = NULL;

    connCb = &rsCb->connCb;

    ret = RsEpollCtl(connCb->epollfd, EPOLL_CTL_DEL, connCb->eventfd, EPOLLIN);
    if (ret) {
        hccp_warn_socket("re epoll ctl unsuccessful, epollfd[%d], eventfd[%d]", connCb->epollfd, connCb->eventfd);
    }

    RS_CLOSE_RETRY_FOR_EINTR(ret, connCb->eventfd);
    connCb->eventfd = RS_FD_INVALID;

    RS_CLOSE_RETRY_FOR_EINTR(ret, connCb->epollfd);
    connCb->epollfd = RS_FD_INVALID;

    return;
}

STATIC void RsUsleepWaitConn(sem_t *sem, uint32_t timeoutUs)
{
    uint32_t sleepUs = 1000; // 每 1000us 查看sem是否为零
    uint32_t elapsedUs = 0;
    do {
        if (sem_trywait(sem) == 0) {
            return;
        }
        usleep(sleepUs);
        elapsedUs += sleepUs;
    } while (elapsedUs < timeoutUs);
    return;
}

STATIC void *RsConnectHandle(void *arg)
{
    struct RsConnInfo *connTmp2 = NULL;
    struct RsConnInfo *connTmp = NULL;
    bool promoteConnect = false;
    int ret;

    hccp_info_others("<SOCKET> thread begin! thread_id:%lu, pid:%d, ppid:%d", pthread_self(), getpid(), getppid());
    CHK_PRT_RETURN(pthread_detach(pthread_self()),
        hccp_err("pthread_detach failed! thread_id:%lu, errno:%d", pthread_self(), errno), NULL);

    (void)prctl(PR_SET_NAME, (uintptr_t) "hccp_connect", 0, 0, 0);

    RsGetCurTime(&gConnectThreadInfo.lastCheckTime);
    ret = strncpy_s((char *)gConnectThreadInfo.pthreadName, sizeof(gConnectThreadInfo.pthreadName), "connect_pthread",
        strlen("connect_pthread"));
    CHK_PRT_RETURN(ret, hccp_err("strncpy_s pthread name failed, ret[%d]", ret), NULL);
    struct rs_cb *rsCb = (struct rs_cb *)arg;
    if (rsCb == NULL) {
        return NULL;
    }

    gRsCb = rsCb;
    sem_init(&rsCb->connectTrigSem, 0, 0);

    hccp_run_info("pthread[%s] is alive!", gConnectThreadInfo.pthreadName);
    while (1) {
        if (rsCb == NULL) {
            return NULL;
        }

        if (rsCb->connFlag == 0) {
            break;
        }

        promoteConnect = false;
        RsHeartbeatAlivePrint(&gConnectThreadInfo);
        RS_PTHREAD_MUTEX_LOCK(&rsCb->mutex);
        RS_LIST_GET_HEAD_ENTRY(connTmp, connTmp2, &rsCb->connCb.clientConnList, list, struct RsConnInfo);
        for (; (&connTmp->list) != &rsCb->connCb.clientConnList;
             connTmp = connTmp2, connTmp2 = list_entry(connTmp2->list.next, struct RsConnInfo, list)) {
            ret = RsSocketConnectAsync(connTmp, rsCb);
            if (ret != 0 && connTmp->state == RS_CONN_STATE_RESET) {
                connTmp->state = RS_CONN_STATE_ERR;
                hccp_err("[client]rs_socket_connect_async failed at RS_CONN_STATE_RESET state, ret:%d, clientIp:%s "
                         "serverIp:%s server_port:%u tag:%s",
                    ret, connTmp->clientIp.readAddr, connTmp->serverIp.readAddr, connTmp->port, connTmp->tag);
                continue;
            }
            if ((promoteConnect == false) && (RsGetSocketConnectState(connTmp) == 0)) {
                promoteConnect = true;
            }
        }
        RS_PTHREAD_MUTEX_ULOCK(&rsCb->mutex);

        if (promoteConnect) {
            usleep(RS_PROMOTE_CONN_USLEEP_TIME);
        } else {
            RsUsleepWaitConn(&rsCb->connectTrigSem, RS_CONN_USLEEP_TIME);
        }
    }
    sem_destroy(&rsCb->connectTrigSem);
    rsCb->connFlag = RS_CONN_EXIT_FLAG;
    hccp_info_others("<SOCKET> QUIT thread_id:%lu, pid:%d", pthread_self(), getpid());

    return NULL;
}

int RsEpollConnectHandleInit(struct rs_cb *rscb)
{
    int ret;
    pthread_t ntid;

    ret = RsCreateEpoll(rscb);
    CHK_PRT_RETURN(ret, hccp_err("rs_create_epoll failed ! ret:%d", ret), ret);

    hccp_info_socket("rs_create_epoll ok");
    gRsCb = rscb;

    ret = pthread_create(&ntid, NULL, RsEpollHandle, (void *)rscb);
    if (ret != 0) {
        gRsCb = NULL;
        RsDestroyEpoll(rscb);
        hccp_err("pthread_create failed ! ret:%d, errno:%d", ret, errno);
        return -ESYSFUNC;
    }

    pthread_t tidp;
    rscb->connFlag = 1;
    ret = pthread_create(&tidp, NULL, RsConnectHandle, (void *)rscb);
    if (ret != 0) {
        gRsCb = NULL;
        RsDestroyEpoll(rscb);
        hccp_err("conn pthread_create failed ! ret:%d, errno:%d", ret, errno);
        return -ESYSFUNC;
    }

    hccp_info_init("RS INIT OK!");

    return ret;
}

int RsEpollCreateEpollfd(int *epollfd)
{
    RS_CHECK_POINTER_NULL_WITH_RET(epollfd);

    // 1024 specify the max fd num, this arg will be ignored since Linux 2.6.8
    *epollfd = epoll_create(1024);
    CHK_PRT_RETURN(*epollfd < 0, hccp_err("create epollfd[%d] failed, errno[%d]", *epollfd, errno), -EINVAL);

    return 0;
}

int RsEpollDestroyFd(int *fd)
{
    int ret = 0;

    RS_CHECK_POINTER_NULL_WITH_RET(fd);

    RS_CLOSE_RETRY_FOR_EINTR(ret, *fd);
    *fd = RS_FD_INVALID;

    return ret;
}

int RsEpollWaitHandle(int eventHandle, struct epoll_event *events, int timeout, unsigned int maxevents,
    unsigned int *eventsNum)
{
    int eventCount;

    RS_CHECK_POINTER_NULL_WITH_RET(events);
    RS_CHECK_POINTER_NULL_WITH_RET(eventsNum);

    eventCount = epoll_wait(eventHandle, events, (int)maxevents, timeout);
    if (eventCount < 0) {
        hccp_err("[rs_epoll_wait_handle]epoll_wait failed, strerror[%s]", strerror(errno));
        return -EIO;
    }

    *eventsNum = (unsigned int)eventCount;
    return 0;
}

RS_ATTRI_VISI_DEF int RsCreateEventHandle(int *eventHandle)
{
    int ret;

    ret = RsEpollCreateEpollfd(eventHandle);
    CHK_PRT_RETURN(ret, hccp_err("rs_epoll_create_epollfd failed ret(%d)", ret), ret);
    return 0;
}

RS_ATTRI_VISI_DEF int RsCtlEventHandle(int eventHandle, const void *fdHandle, int opcode, enum RaEpollEvent event)
{
    int fd = RS_FD_INVALID;
    unsigned int tmpEvent;
    int ret;

    if (eventHandle < 0) {
        hccp_err("event_handle[%d] is invalid", eventHandle);
        return -EINVAL;
    }
    if (fdHandle == NULL) {
        hccp_err("fd_handle is NULL");
        return -EINVAL;
    }
    if (opcode != EPOLL_CTL_ADD && opcode != EPOLL_CTL_DEL && opcode != EPOLL_CTL_MOD) {
        hccp_err("opcode[%d] invalid, valid opcode includes {%d, %d, %d}", opcode, EPOLL_CTL_ADD, EPOLL_CTL_DEL,
            EPOLL_CTL_MOD);
        return -EINVAL;
    }

    if (event == RA_EPOLLONESHOT) {
        tmpEvent = EPOLLIN | EPOLLET | EPOLLONESHOT;
    } else if (event == RA_EPOLLIN) {
        tmpEvent = EPOLLIN;
    } else if (event == RA_EPOLLOUT) {
        tmpEvent = EPOLLOUT;
    } else if (event == RA_EPOLLOUT_LET_ONESHOT) {
        tmpEvent = EPOLLOUT | EPOLLET | EPOLLONESHOT;
    } else {
        hccp_err("unknown event[%d]", event);
        return -EINVAL;
    }

    tmpEvent = tmpEvent | EPOLLRDHUP;
    fd = ((const struct SocketPeerInfo *)fdHandle)->fd;

    ret = RsEpollCtlFdHandle(eventHandle, opcode, fd, tmpEvent, (void *)fdHandle);
    CHK_PRT_RETURN(ret, hccp_err("rs_epoll_ctl_fd_handle failed ret(%d), fd:%d", ret, fd), ret);
    return 0;
}

RS_ATTRI_VISI_DEF int RsWaitEventHandle(int eventHandle, struct SocketEventInfoT *eventInfos, int timeout,
    unsigned int maxevents, unsigned int *eventsNum)
{
    int ret;

    if (eventHandle < 0) {
        hccp_err("event_handle[%d] is invalid", eventHandle);
        return -EINVAL;
    }

    if (eventInfos == NULL) {
        hccp_err("event_info is NULL");
        return -EINVAL;
    }

    if (timeout < -1) {
        hccp_err("timeout[%d ms] is invalid", timeout);
        return -EINVAL;
    }

    if (maxevents > MAX_SOCKET_EVENT_NUM) {
        hccp_err("maxevents[%u] exceeds %u", maxevents, MAX_SOCKET_EVENT_NUM);
        return -EINVAL;
    }

    if (eventsNum == NULL) {
        hccp_err("events_num is NULL");
        return -EINVAL;
    }

    ret = RsEpollWaitHandle(eventHandle, (struct epoll_event *)eventInfos, timeout, maxevents, eventsNum);
    CHK_PRT_RETURN(ret, hccp_err("rs_epoll_wait_handle failed ret(%d)", ret), ret);

    return 0;
}

RS_ATTRI_VISI_DEF int RsDestroyEventHandle(int *eventHandle)
{
    int ret;

    ret = RsEpollDestroyFd(eventHandle);
    CHK_PRT_RETURN(ret, hccp_err("rs_epoll_destroy_fd failed ret(%d)", ret), ret);
    return 0;
}
