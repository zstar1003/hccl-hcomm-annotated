/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "network_comm.h"
#include <errno.h>
#include <sys/types.h>
#include <pwd.h>
#include <unistd.h>
#include "securec.h"
#include "user_log.h"

#define ROOT_PATH "/root"

int NetCommGetSelfHome(char *homePath, unsigned int pathLen)
{
    int ret, retVal;
    struct passwd *pwd = getpwuid(getuid());
    CHK_PRT_RETURN(pwd == NULL, roce_err("pwd is NULL! getpwuid failed, errno:%d", errno), -EINVAL);

    if (pwd->pw_name == NULL) {
        roce_err("pwd->pw_name is NULL, errno:%d", errno);
        ret = -EINVAL;
        goto out_get_self_home;
    }

    // root用户的home路径为/root
    // 其他用户的home路径为/home/${user_name}
    if (strncmp(pwd->pw_name, "root", strlen("root") + 1) == 0) {
        ret = sprintf_s((char *)homePath, pathLen, ROOT_PATH);
    } else {
        ret = sprintf_s((char *)homePath, pathLen, "/home/%s", pwd->pw_name);
    }
    if (ret <= 0) {
        roce_err("sprintf_s for user name:%s failed, ret:%d ", pwd->pw_name, ret);
        ret = -ENOMEM;
        goto out_get_self_home;
    }

    ret = 0;
out_get_self_home:
    retVal = memset_s(pwd, sizeof(struct passwd), 0, sizeof(struct passwd));
    if (retVal) {
        roce_err("memset error, retVal[%d]", retVal);
        ret = retVal;
    }

    return ret;
}

int NetGetGatewayAddress(unsigned int phyId, unsigned int *gtwAddr)
{
    unsigned int gtwDst = NET_INVALID_GW;
    unsigned int ethId = 0;
    char buf[BUF_LEN];
    int ret, retVal;
    char *tmp = NULL;
    FILE *fp = NULL;

    fp = fopen("/proc/net/route", "r");
    if (fp == NULL) {
        roce_err("fopen failed, errno:%d", errno);
        return -EINVAL;
    }

    while (fgets(buf, sizeof(buf), fp) != NULL) {
        tmp = buf;
        while (tmp != NULL && (*tmp == ' ')) {
            ++tmp;
        }

        if (tmp == NULL || (strncmp(tmp, "eth", strlen("eth")) != 0)) {
            continue;
        }

        ret = sscanf_s(tmp, "eth%u%x%x", &ethId, &gtwDst, gtwAddr);
        if (ret != NET_THREE_VALUE) {
            roce_err("sscanf buf(%s) to gtw_addr failed. ret(%d)", buf, ret);
            ret = -ENOMEM;
            goto out;
        }

        if (gtwDst == 0 && ethId == phyId) {
            ret = 0;
            goto out;
        }
    }

    roce_err("cannot find gateway by phy_id: %u", phyId);
    ret = -ENOENT;
out:
    retVal = fclose(fp);
    if (retVal) {
        roce_warn("fclose failed, retVal:%d, errno:%d", retVal, errno);
    }
    fp = NULL;
    return ret;
}
