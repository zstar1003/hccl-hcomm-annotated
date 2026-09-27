/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONFIG_H_IN
#define CONFIG_H_IN

#define HAVE_STATEMENT_EXPR 1
#define HAVE_BUILTIN_TYPES_COMPATIBLE_P 1
#define HAVE_TYPEOF 1
#define HAVE_ISBLANK 1

#define STREAM_CLOEXEC "e"

#define IBV_CONFIG_DIR \
    "/home/unilsw/share/ccl_it6/out/cloud/host/obj/THIRD_PARTY_LIBS/rmda_core/rdma-core-17.1/build/etc/libibverbs.d"
#define RS_CONF_DIR \
    "/home/unilsw/share/ccl_it6/out/cloud/host/obj/THIRD_PARTY_LIBS/rmda_core/rdma-core-17.1/build/etc/rdma/rsocket"
#define IWPM_CONFIG_FILE \
    "/home/unilsw/share/ccl_it6/out/cloud/host/obj/THIRD_PARTY_LIBS/rmda_core/rdma-core-17.1/build/etc/iwpmd.conf"

#define SRP_DEAMON_CONFIG_FILE                                                                           \
    "/home/unilsw/share/ccl_it6/out/cloud/host/obj/THIRD_PARTY_LIBS/rmda_core/rdma-core-17.1/build/etc/" \
    "srp_daemon.conf"
#define SRP_DEAMON_LOCK_PREFIX "/usr/local/var/run/srp_daemon"

#define ACM_CONF_DIR \
    "/home/unilsw/share/ccl_it6/out/cloud/host/obj/THIRD_PARTY_LIBS/rmda_core/rdma-core-17.1/build/etc/rdma"
#define IBACM_LIB_PATH \
    "/home/unilsw/share/ccl_it6/out/cloud/host/obj/THIRD_PARTY_LIBS/rmda_core/rdma-core-17.1/build/lib/ibacm"
#define IBACM_BIN_PATH \
    "/home/unilsw/share/ccl_it6/out/cloud/host/obj/THIRD_PARTY_LIBS/rmda_core/rdma-core-17.1/build/bin"
#define IBACM_PID_FILE "/usr/local/var/run/ibacm.pid"
#define IBACM_PORT_FILE "/usr/local/var/run/ibacm.port"
#define IBACM_LOG_FILE "/usr/local/var/log/ibacm.log"

#define VERBS_PROVIDER_DIR \
    "/home/unilsw/share/ccl_it6/out/cloud/host/obj/THIRD_PARTY_LIBS/rmda_core/rdma-core-17.1/build/lib"
#define VERBS_PROVIDER_SUFFIX "-rdmav17.so"
#define IBVERBS_PABI_VERSION 17

#define HAVE_FUNC_ATTRIBUTE_ALWAYS_INLINE 1

#define HAVE_FUNC_ATTRIBUTE_IFUNC 1

#define HAVE_WORKING_IF_H 1

#define HAVE_FULL_SYMBOL_VERSIONS 1
/* #undef HAVE_LIMITED_SYMBOL_VERSIONS */

#define SIZEOF_LONG 8

#if 3 == 3
#define HAVE_LIBNL3 1
#elif 3 == 1
#define HAVE_LIBNL1 1
#elif 3 == 0
#define NRESOLVE_NEIGH 1
#endif

#endif
