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
#include <config.h>

#include <endian.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <alloca.h>
#include <errno.h>

#include "ibverbs.h"

static pthread_mutex_t dev_list_lock = PTHREAD_MUTEX_INITIALIZER;
static int initialized;
static struct list_head device_list = LIST_HEAD_INIT(device_list);

void verbs_init_cq(struct ibv_cq* cq, struct ibv_context* context, struct ibv_comp_channel* channel, void* cq_context)
{
    cq->context = context;
    cq->channel = channel;

    if (cq->channel) {
        pthread_mutex_lock(&context->mutex);
        ++cq->channel->refcnt;
        pthread_mutex_unlock(&context->mutex);
    }

    cq->cq_context = cq_context;
    cq->comp_events_completed = 0;
    cq->async_events_completed = 0;
    pthread_mutex_init(&cq->mutex, NULL);
    pthread_cond_init(&cq->cond, NULL);
}

static bool has_ioctl_write(struct ibv_context* ctx) { return true; }

/*
 * Ownership of cmd_fd is transferred into this function, and it will either
 * be released during the matching call to verbs_uninit_contxt or during the
 * failure path of this function.
 */
int verbs_init_context(struct verbs_context* context_ex, struct ibv_device* device, int cmd_fd, uint32_t driver_id)
{
    struct ibv_context* context = &context_ex->context;

    ibverbs_device_hold(device);

    context->device = device;
    context->cmd_fd = cmd_fd;
    context->async_fd = -1;
    pthread_mutex_init(&context->mutex, NULL);

    context_ex->context.abi_compat = __VERBS_ABI_IS_EXTENDED;
    context_ex->sz = sizeof(*context_ex);

    /*
     * In order to maintain backward/forward binary compatibility
     * with apps compiled against libibverbs-1.1.8 that use the
     * flow steering addition, we need to set the two
     * ABI_placeholder entries to match the driver set flow
     * entries.  This is because apps compiled against
     * libibverbs-1.1.8 use an inline ibv_create_flow and
     * ibv_destroy_flow function that looks in the placeholder
     * spots for the proper entry points.  For apps compiled
     * against libibverbs-1.1.9 and later, the inline functions
     * will be looking in the right place.
     */
    context_ex->ABI_placeholder1 = (void (*)(void))context_ex->ibv_create_flow;
    context_ex->ABI_placeholder2 = (void (*)(void))context_ex->ibv_destroy_flow;

    context_ex->priv = calloc(1, sizeof(*context_ex->priv));
    if (!context_ex->priv) {
        errno = ENOMEM;
        close(cmd_fd);
        return -1;
    }

    context_ex->priv->driver_id = driver_id;
    context_ex->priv->use_ioctl_write = has_ioctl_write(context);

    return 0;
}

/*
 * Allocate and initialize a context structure. This is called to create the
 * driver wrapper, and context_offset is the number of bytes into the wrapper
 * structure where the verbs_context starts.
 */
void* _verbs_init_and_alloc_context(
    struct ibv_device* device, int cmd_fd, size_t alloc_size, struct verbs_context* context_offset, uint32_t driver_id)
{
    void* drv_context;
    struct verbs_context* context;

    drv_context = calloc(1, alloc_size);
    if (!drv_context) {
        errno = ENOMEM;
        close(cmd_fd);
        return NULL;
    }

    context = drv_context + (uintptr_t)context_offset;

    if (verbs_init_context(context, device, cmd_fd, driver_id))
        goto err_free;

    return drv_context;

err_free:
    free(drv_context);
    return NULL;
}

void verbs_uninit_context(struct verbs_context* context_ex)
{
    free(context_ex->priv);
    close(context_ex->context.cmd_fd);
    close(context_ex->context.async_fd);
    ibverbs_device_put(context_ex->context.device);
}
