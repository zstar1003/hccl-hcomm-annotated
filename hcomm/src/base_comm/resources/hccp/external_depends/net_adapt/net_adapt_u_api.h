/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NET_ADP_U_API_H
#define NET_ADP_U_API_H

#define NETADP_ATTRI_VISI_DEF __attribute__((visibility("default")))

NETADP_ATTRI_VISI_DEF int net_adapt_init(void);
NETADP_ATTRI_VISI_DEF void net_adapt_uninit(void);
NETADP_ATTRI_VISI_DEF int net_alloc_jfc_id(const char *udev_name, unsigned int jfc_mode, unsigned int *jfc_id);
NETADP_ATTRI_VISI_DEF int net_free_jfc_id(const char *udev_name, unsigned int jfc_mode, unsigned int jfc_id);
NETADP_ATTRI_VISI_DEF int net_alloc_jetty_id(const char *udev_name, unsigned int jetty_mode, unsigned int *jetty_id);
NETADP_ATTRI_VISI_DEF int net_free_jetty_id(const char *udev_name, unsigned int jetty_mode, unsigned int jetty_id);
NETADP_ATTRI_VISI_DEF unsigned long long net_get_cqe_base_addr(unsigned int die_id);
NETADP_ATTRI_VISI_DEF unsigned int net_get_api_version(void);

#endif // NET_ADP_U_API_H
