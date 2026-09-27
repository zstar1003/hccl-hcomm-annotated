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

#include <stdlib.h>
#include <string.h>
#include <glob.h>
#include <stdio.h>
#include <dlfcn.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <dirent.h>
#include <errno.h>
#include <assert.h>
#include <fnmatch.h>

#include "ibverbs.h"

static int abi_ver;

struct ibv_driver_name {
    struct list_node entry;
    char* name;
};

struct ibv_driver {
    struct list_node entry;
    const struct verbs_device_ops* ops;
};

static LIST_HEAD(driver_name_list);
static LIST_HEAD(driver_list);

static int find_sysfs_devs(struct list_head* tmp_sysfs_dev_list) { return 0; }

static struct verbs_device_ops* tc_driver_ops;

void verbs_register_driver(const struct verbs_device_ops* ops) { tc_driver_ops = (struct verbs_device_ops*)ops; }

static void load_driver(const char* name) { return; }

static void load_drivers(void) {}

static void read_config_file(const char* path) {}

static void read_config(void) {}

/* Match a single modalias value */
static bool match_modalias(const struct verbs_match_ent* ent, const char* value) {}

/* Search a null terminated table of verbs_match_ent's and return the one
 * that matches the device the verbs sysfs device is bound to or NULL.
 */
static const struct verbs_match_ent*
match_modalias_device(const struct verbs_device_ops* ops, struct verbs_sysfs_dev* sysfs_dev)
{
    return NULL;
}

/* Match the device name itself */
static const struct verbs_match_ent* match_name(const struct verbs_device_ops* ops, struct verbs_sysfs_dev* sysfs_dev)
{
    return NULL;
}

/* True if the provider matches the selected rdma sysfs device */
static bool match_device(const struct verbs_device_ops* ops, struct verbs_sysfs_dev* sysfs_dev) { return true; }

static struct verbs_device* try_driver(const struct verbs_device_ops* ops, struct verbs_sysfs_dev* sysfs_dev)
{
    return NULL;
}

static struct verbs_device* try_drivers(struct verbs_sysfs_dev* sysfs_dev) { return NULL; }

static int check_abi_version(const char* path) { return 0; }

static void check_memlock_limit(void) {}

static int same_sysfs_dev(struct verbs_sysfs_dev* sysfs1, struct verbs_sysfs_dev* sysfs2) { return 0; }

/* Match every ibv_sysfs_dev in the sysfs_list to a driver and add a new entry
 * to device_list. Once matched to a driver the entry in sysfs_list is
 * removed.
 */
static void try_all_drivers(struct list_head* sysfs_list, struct list_head* device_list, unsigned int* num_devices) {}

int ibverbs_get_device_list(struct list_head* device_list) { return 0; }

int ibverbs_init(void) { return 0; }

void ibverbs_device_hold(struct ibv_device* dev) {}

void ibverbs_device_put(struct ibv_device* dev) {}
