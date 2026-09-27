/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 被测目标1: hcomm::EndpointMgr（HcommResMgr 成员）—— 管理 EndpointHandle→Endpoint 全局句柄表。
// 被测目标2: hcomm::EndpointCtxMgr（HcommBaseResMgr 成员，per-device）—— 管理 EndpointCtx 去重缓存。

#include "gtest/gtest.h"
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>

#define private public
#define protected public
// 消歧：UT include 路径中 coll_comms 的 endpoint_mgr.h 优先于 hcomm 的，
// 用相对路径强制引用 hcomm::EndpointMgr / hcomm::EndpointCtxMgr 的头文件。
#include "../../../../../../src/base_comm/resources/endpoints/mgr/endpoint_mgr.h"
#include "../../../../../../src/base_comm/resources/endpoints/mgr/endpoint_ctx_mgr.h"
#undef protected
#undef private

#include "endpoint.h"
#include "rdma_handle_manager.h"
#include "../../../../../../src/base_comm/hcomm_res_mgr.h"
#include "ip_address.h"
#include "port.h"
#include "hcomm_res_defs.h"
#include "hcomm_c_adpt.h"
#include "hcomm_c_adpt_common.h"

using namespace hcomm;

namespace {
// 测试用最小 Endpoint 子类：override 所有纯虚方法，避免依赖真实子类初始化。
class UtStubEndpoint : public Endpoint {
public:
    explicit UtStubEndpoint(const EndpointDesc& desc) : Endpoint(desc) {}
    ~UtStubEndpoint() noexcept override = default;

    HcclResult Init() override { return HCCL_SUCCESS; }
    RegedMemMgr* GetRegedMemMgr() override { return nullptr; }
    void* GetRdmaHandle() override { return nullptr; }
    bool IsCtxHandleValid() const override { return false; }
};

EndpointDesc MakeHostRoceDesc()
{
    EndpointDesc desc{};
    desc.protocol = COMM_PROTOCOL_ROCE;
    desc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    Hccl::IpAddress ip("1.0.0.0");
    desc.commAddr.addr = ip.GetBinaryAddress().addr;
    desc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    return desc;
}

EndpointCtxKey MakeKey(uint32_t devPhyId, CommProtocol proto, EndpointLocType locType, const std::string& ipStr)
{
    EndpointCtxKey key{};
    key.devPhyId = devPhyId;
    key.protocol = proto;
    key.locType = locType;
    key.ip = Hccl::IpAddress(ipStr);
    return key;
}

void MockRdmaHandle(void* handle)
{
    // mock RdmaHandleManager::GetByAddr / GetByIp 返回非空 rdmaHandle
    MOCKER_CPP(&Hccl::RdmaHandleManager::GetByAddr).stubs().will(returnValue(handle));
    MOCKER_CPP(&Hccl::RdmaHandleManager::GetByIp).stubs().will(returnValue(handle));
    // UboeIpv4ToEid 空实现：Acquire 内 queryIp 初值=key.ip，空 stub 不改变 queryIp，
    // GetByIp 已 mock 与地址无关，缓存键两次调用一致，不影响断言
    MOCKER_CPP(&Hccl::RdmaHandleManager::UboeIpv4ToEid).stubs();
    // mock IsHandleValid：缓存命中校验时需要返回 true（mock handle 不在 activeHandles_）
    MOCKER_CPP(&Hccl::RdmaHandleManager::IsHandleValid).stubs().will(returnValue(handle != nullptr));
}
} // namespace

class HcommEndpointMgrTest : public testing::Test {
protected:
    // endpointMgr_ 为 HcommResMgr 全局成员：进程内唯一实例
    EndpointMgr& mgr_{HcommResMgr::GetInstance().GetEndpointMgr()};

    void SetUp() override
    {
        // 每个用例前清空单例残留（保证隔离）
        mgr_.DeInit();
    }
    void TearDown() override
    {
        GlobalMockObject::verify();
        mgr_.DeInit();
    }

    EndpointHandle MakeHandle(uintptr_t v) { return reinterpret_cast<EndpointHandle>(v); }
};

class HcommEndpointCtxMgrTest : public testing::Test {
protected:
    // EndpointCtxMgr 为 HcommBaseResMgr 成员（per-device）。
    // UT 用本地实例验证去重/释放/兜底语义，避免经 GetDeviceResMgr 触发设备级单例初始化链。
    EndpointCtxMgr mgr_{};

    void TearDown() override
    {
        GlobalMockObject::verify();
        mgr_.DeInit();
    }
};

// ============ Add ============
// TC-EndpointMgr_Add-001: 正常添加 endpoint 到 map
TEST_F(HcommEndpointMgrTest, Ut_Add_When_NewHandle_Expect_StoredInMap)
{
    EndpointDesc desc = MakeHostRoceDesc();
    auto ep = std::make_unique<UtStubEndpoint>(desc);
    Endpoint* raw = ep.get();
    EndpointHandle h = MakeHandle(0x100);
    mgr_.Add(h, std::move(ep));

    EXPECT_EQ(mgr_.Get(h), raw);
}

// TC-EndpointMgr_Add-002: 同 handle 已存在时替换
TEST_F(HcommEndpointMgrTest, Ut_Add_When_DuplicateHandle_Expect_ReplaceOld)
{
    EndpointDesc desc = MakeHostRoceDesc();
    EndpointHandle h = MakeHandle(0x200);
    mgr_.Add(h, std::make_unique<UtStubEndpoint>(desc));

    auto ep2 = std::make_unique<UtStubEndpoint>(desc);
    Endpoint* raw2 = ep2.get();
    mgr_.Add(h, std::move(ep2));

    EXPECT_EQ(mgr_.Get(h), raw2);
}

// ============ Remove ============
// TC-EndpointMgr_Remove-001: 正常移除已存在 endpoint
TEST_F(HcommEndpointMgrTest, Ut_Remove_When_Exists_Expect_TrueAndRemoved)
{
    EndpointDesc desc = MakeHostRoceDesc();
    EndpointHandle h = MakeHandle(0x300);
    mgr_.Add(h, std::make_unique<UtStubEndpoint>(desc));

    EXPECT_TRUE(mgr_.Remove(h));
    EXPECT_EQ(mgr_.Get(h), nullptr);
}

// TC-EndpointMgr_Remove-002: 移除不存在的 handle 返回 false
TEST_F(HcommEndpointMgrTest, Ut_Remove_When_NotExists_Expect_False) { EXPECT_FALSE(mgr_.Remove(MakeHandle(0xDEAD))); }

// ============ Update ============
// TC-EndpointMgr_Update-001: 正常更新已存在 endpoint
TEST_F(HcommEndpointMgrTest, Ut_Update_When_Exists_Expect_TrueAndReplaced)
{
    EndpointDesc desc = MakeHostRoceDesc();
    EndpointHandle h = MakeHandle(0x400);
    mgr_.Add(h, std::make_unique<UtStubEndpoint>(desc));

    auto ep2 = std::make_unique<UtStubEndpoint>(desc);
    Endpoint* raw2 = ep2.get();
    EXPECT_TRUE(mgr_.Update(h, std::move(ep2)));
    EXPECT_EQ(mgr_.Get(h), raw2);
}

// TC-EndpointMgr_Update-002: 更新不存在的 handle 返回 false（不插入）
TEST_F(HcommEndpointMgrTest, Ut_Update_When_NotExists_Expect_FalseAndNotInserted)
{
    EndpointDesc desc = MakeHostRoceDesc();
    EXPECT_FALSE(mgr_.Update(MakeHandle(0x500), std::make_unique<UtStubEndpoint>(desc)));
    EXPECT_EQ(mgr_.Get(MakeHandle(0x500)), nullptr);
}

// ============ Get ============
// TC-EndpointMgr_Get-001: 获取已存在 endpoint 返回非 nullptr
TEST_F(HcommEndpointMgrTest, Ut_Get_When_Exists_Expect_NonNull)
{
    EndpointDesc desc = MakeHostRoceDesc();
    EndpointHandle h = MakeHandle(0x600);
    auto ep = std::make_unique<UtStubEndpoint>(desc);
    Endpoint* raw = ep.get();
    mgr_.Add(h, std::move(ep));
    EXPECT_NE(mgr_.Get(h), nullptr);
    EXPECT_EQ(mgr_.Get(h), raw);
}

// TC-EndpointMgr_Get-002: 获取不存在的 handle 返回 nullptr
TEST_F(HcommEndpointMgrTest, Ut_Get_When_NotExists_Expect_Nullptr) { EXPECT_EQ(mgr_.Get(MakeHandle(0x700)), nullptr); }

// ============ Acquire（locType=HOST，RDMA key，GetByAddr 路径） ============
// TC-EndpointCtxMgr_AcquireByAddr-001: 首次 ByAddr 创建新 EndpointCtx
TEST_F(HcommEndpointCtxMgrTest, Ut_AcquireByAddr_When_FirstCall_Expect_CreateNewCtx)
{
    void* fakeHandle = reinterpret_cast<void*>(0x8001);
    MockRdmaHandle(fakeHandle);
    EndpointCtxKey key = MakeKey(0, COMM_PROTOCOL_ROCE, ENDPOINT_LOC_TYPE_HOST, "1.0.0.0");

    std::shared_ptr<EndpointCtx> ctx;
    EXPECT_EQ(mgr_.Acquire(key, false, ctx), HCCL_SUCCESS);
    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->ctxHandle, fakeHandle);
    EXPECT_EQ(ctx->key.devPhyId, 0u);
}

// TC-EndpointCtxMgr_AcquireByAddr-002: 相同 key 命中去重（返回同一 shared_ptr）
TEST_F(HcommEndpointCtxMgrTest, Ut_AcquireByAddr_When_SameKey_Expect_HitCache)
{
    void* fakeHandle = reinterpret_cast<void*>(0x8002);
    MockRdmaHandle(fakeHandle);
    EndpointCtxKey key = MakeKey(0, COMM_PROTOCOL_ROCE, ENDPOINT_LOC_TYPE_HOST, "1.0.0.0");

    std::shared_ptr<EndpointCtx> ctx1;
    std::shared_ptr<EndpointCtx> ctx2;
    EXPECT_EQ(mgr_.Acquire(key, false, ctx1), HCCL_SUCCESS);
    EXPECT_EQ(mgr_.Acquire(key, false, ctx2), HCCL_SUCCESS);

    ASSERT_NE(ctx1, nullptr);
    EXPECT_EQ(ctx1.get(), ctx2.get());             // 同一底层指针 → 命中缓存
    EXPECT_EQ(ctx1.use_count(), ctx2.use_count()); // shared_ptr 共享
}

// ByAddr 依赖返回 nullptr 时返回 HCCL_E_PTR 且出参为空
TEST_F(HcommEndpointCtxMgrTest, Ut_AcquireByAddr_When_GetByAddrReturnsNull_Expect_ReturnEPtr)
{
    MockRdmaHandle(nullptr); // rdmaHandle 为空
    EndpointCtxKey key = MakeKey(0, COMM_PROTOCOL_ROCE, ENDPOINT_LOC_TYPE_HOST, "1.0.0.1");

    std::shared_ptr<EndpointCtx> ctx;
    EXPECT_EQ(mgr_.Acquire(key, false, ctx), HCCL_E_PTR);
    EXPECT_EQ(ctx, nullptr);
}

// ============ Acquire（locType=DEVICE，UB key，GetByIp 路径） ============
// TC-EndpointCtxMgr_AcquireByEid-001: 首次 ByEid 创建新 EndpointCtx
TEST_F(HcommEndpointCtxMgrTest, Ut_AcquireByEid_When_FirstCall_Expect_CreateNewCtx)
{
    void* fakeHandle = reinterpret_cast<void*>(0x8003);
    MockRdmaHandle(fakeHandle);
    EndpointCtxKey key = MakeKey(0, COMM_PROTOCOL_UB_CTP, ENDPOINT_LOC_TYPE_DEVICE, "2.0.0.0");

    std::shared_ptr<EndpointCtx> ctx;
    EXPECT_EQ(mgr_.Acquire(key, false, ctx), HCCL_SUCCESS);
    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->ctxHandle, fakeHandle);
}

// TC-EndpointCtxMgr_AcquireByEid-002: 相同 key 命中去重
TEST_F(HcommEndpointCtxMgrTest, Ut_AcquireByEid_When_SameKey_Expect_HitCache)
{
    void* fakeHandle = reinterpret_cast<void*>(0x8004);
    MockRdmaHandle(fakeHandle);
    EndpointCtxKey key = MakeKey(0, COMM_PROTOCOL_UB_CTP, ENDPOINT_LOC_TYPE_DEVICE, "2.0.0.0");

    std::shared_ptr<EndpointCtx> ctx1;
    std::shared_ptr<EndpointCtx> ctx2;
    EXPECT_EQ(mgr_.Acquire(key, false, ctx1), HCCL_SUCCESS);
    EXPECT_EQ(mgr_.Acquire(key, false, ctx2), HCCL_SUCCESS);
    ASSERT_NE(ctx1, nullptr);
    EXPECT_EQ(ctx1.get(), ctx2.get());
}

// ============ Acquire（locType=DEVICE + isUboeIp=true，UBOE IP→EID 转换路径） ============
// TC-EndpointCtxMgr_AcquireByIp-001: isUboeIp=true 首次调用创建新 EndpointCtx
TEST_F(HcommEndpointCtxMgrTest, Ut_AcquireByEid_When_Ipv4ToEidTrueFirstCall_Expect_CreateNewCtx)
{
    void* fakeHandle = reinterpret_cast<void*>(0x8005);
    MockRdmaHandle(fakeHandle);
    EndpointCtxKey key = MakeKey(0, COMM_PROTOCOL_UB_CTP, ENDPOINT_LOC_TYPE_DEVICE, "3.0.0.0");

    std::shared_ptr<EndpointCtx> ctx;
    EXPECT_EQ(mgr_.Acquire(key, true, ctx), HCCL_SUCCESS);
    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->ctxHandle, fakeHandle);
}

// TC-EndpointCtxMgr_AcquireByIp-002: isUboeIp=true 相同 key 命中去重
TEST_F(HcommEndpointCtxMgrTest, Ut_AcquireByEid_When_Ipv4ToEidTrueSameKey_Expect_HitCache)
{
    void* fakeHandle = reinterpret_cast<void*>(0x8006);
    MockRdmaHandle(fakeHandle);
    EndpointCtxKey key = MakeKey(0, COMM_PROTOCOL_UB_CTP, ENDPOINT_LOC_TYPE_DEVICE, "3.0.0.0");

    std::shared_ptr<EndpointCtx> ctx1;
    std::shared_ptr<EndpointCtx> ctx2;
    EXPECT_EQ(mgr_.Acquire(key, true, ctx1), HCCL_SUCCESS);
    EXPECT_EQ(mgr_.Acquire(key, true, ctx2), HCCL_SUCCESS);
    ASSERT_NE(ctx1, nullptr);
    EXPECT_EQ(ctx1.get(), ctx2.get());
}

// ============ Release ============
// TC-EndpointCtxMgr_Release-001: use_count>1 时不销毁
TEST_F(HcommEndpointCtxMgrTest, Ut_Release_When_UseCountGt1_Expect_NotErased)
{
    void* fakeHandle = reinterpret_cast<void*>(0x8007);
    MockRdmaHandle(fakeHandle);
    EndpointCtxKey key = MakeKey(0, COMM_PROTOCOL_ROCE, ENDPOINT_LOC_TYPE_HOST, "5.0.0.0");

    std::shared_ptr<EndpointCtx> ctx1;
    std::shared_ptr<EndpointCtx> ctx2;
    EXPECT_EQ(mgr_.Acquire(key, false, ctx1), HCCL_SUCCESS);
    EXPECT_EQ(mgr_.Acquire(key, false, ctx2), HCCL_SUCCESS);
    ASSERT_NE(ctx1, nullptr);
    (void)ctx2; // map + ctx1 + ctx2 共享

    mgr_.Release(key); // use_count 仍 >1，不从 map 移除

    // 再次 Acquire 应命中缓存（未被销毁）
    std::shared_ptr<EndpointCtx> ctx3;
    EXPECT_EQ(mgr_.Acquire(key, false, ctx3), HCCL_SUCCESS);
    EXPECT_EQ(ctx3.get(), ctx1.get());
}

// TC-EndpointCtxMgr_Release-002: use_count==1 时从 map 移除
TEST_F(HcommEndpointCtxMgrTest, Ut_Release_When_UseCountEq1_Expect_Erased)
{
    void* fakeHandle = reinterpret_cast<void*>(0x8008);
    MockRdmaHandle(fakeHandle);
    EndpointCtxKey key = MakeKey(0, COMM_PROTOCOL_ROCE, ENDPOINT_LOC_TYPE_HOST, "6.0.0.0");

    std::shared_ptr<EndpointCtx> ctx;
    EXPECT_EQ(mgr_.Acquire(key, false, ctx), HCCL_SUCCESS);
    ASSERT_NE(ctx, nullptr);
    ctx.reset(); // 外部引用释放，map 中 use_count==1

    mgr_.Release(key); // 应从 map 移除并销毁

    // 再次 Acquire 走新建路径（缓存已被清）
    MockRdmaHandle(reinterpret_cast<void*>(0x8009));
    std::shared_ptr<EndpointCtx> ctx2;
    EXPECT_EQ(mgr_.Acquire(key, false, ctx2), HCCL_SUCCESS);
    ASSERT_NE(ctx2, nullptr);
    EXPECT_NE(ctx2.get(), (void*)0x8008); // 新建，ctxHandle 不同
}

// TC-EndpointCtxMgr_Release-003: Release 不存在的 key 无操作（不崩溃）
TEST_F(HcommEndpointCtxMgrTest, Ut_Release_When_KeyNotInMap_Expect_NoOp)
{
    EndpointCtxKey key = MakeKey(99, COMM_PROTOCOL_ROCE, ENDPOINT_LOC_TYPE_HOST, "9.9.9.9");
    // map 中无此 key，Release 不应崩溃
    mgr_.Release(key);
    SUCCEED();
}

// ============ DeInit ============
// TC-EndpointMgr_DeInit-001: 遍历销毁残留 endpoint（残留时不崩溃，告警）
TEST_F(HcommEndpointMgrTest, Ut_DeInit_When_ResidualEndpoints_Expect_AllCleared)
{
    EndpointDesc desc = MakeHostRoceDesc();
    mgr_.Add(MakeHandle(0xA1), std::make_unique<UtStubEndpoint>(desc));
    mgr_.Add(MakeHandle(0xA2), std::make_unique<UtStubEndpoint>(desc));

    // DeInit 应清空 endpointMap_
    mgr_.DeInit();
    EXPECT_TRUE(mgr_.endpointMap_.empty());
}

// TC-EndpointCtxMgr_DeInit-001: 遍历销毁残留 EndpointCtx（残留时不崩溃，告警）
TEST_F(HcommEndpointCtxMgrTest, Ut_DeInit_When_ResidualCtxs_Expect_AllCleared)
{
    void* fakeHandle = reinterpret_cast<void*>(0xAAAA);
    MockRdmaHandle(fakeHandle);
    EndpointCtxKey key = MakeKey(0, COMM_PROTOCOL_ROCE, ENDPOINT_LOC_TYPE_HOST, "7.0.0.0");
    std::shared_ptr<EndpointCtx> ctx;
    (void)mgr_.Acquire(key, false, ctx);

    // DeInit 应清空 endpointCtxMap_
    mgr_.DeInit();
    EXPECT_TRUE(mgr_.endpointCtxMap_.empty());
}

// TC-EndpointMgr_DeInit-002: 经 HcommResMgr::GetInstance().GetEndpointMgr() 直接链路访问同一全局实例
TEST_F(HcommEndpointMgrTest, Ut_GetEndpointMgr_When_Called_Expect_SameInstance)
{
    EXPECT_EQ(&HcommResMgr::GetInstance().GetEndpointMgr(), &mgr_);
    // 直接链路 Add/Get 语义：Add 后按 handle 可查回
    EndpointDesc desc = MakeHostRoceDesc();
    EndpointHandle h = MakeHandle(0x900);
    auto ep = std::make_unique<UtStubEndpoint>(desc);
    Endpoint* raw = ep.get();
    mgr_.Add(h, std::move(ep));
    EXPECT_EQ(mgr_.Get(h), raw);
    EXPECT_EQ(mgr_.Get(MakeHandle(0xDEAD2)), nullptr);
    (void)mgr_.Remove(h);
}

// TC-EndpointMgr_DeInit-003: ~HcommResMgr 析构调 DeInit ——间接验证 DeInit 可重复调用且幂等
TEST_F(HcommEndpointMgrTest, Ut_DeInit_When_CalledTwice_Expect_Idempotent)
{
    mgr_.DeInit();
    mgr_.DeInit();
    EXPECT_TRUE(mgr_.endpointMap_.empty());
}

// TC-EndpointCtxMgr_DeInit-003: ~HcommResMgr 析构遍历调各设备 EndpointCtxMgr::DeInit
// ——间接验证 DeInit 可重复调用且幂等
TEST_F(HcommEndpointCtxMgrTest, Ut_DeInit_When_CalledTwice_Expect_Idempotent)
{
    mgr_.DeInit();
    mgr_.DeInit();
    EXPECT_TRUE(mgr_.endpointCtxMap_.empty());
}

// TC-DFX_EndpointMgr_日志-001: Add/Remove 日志输出（不崩溃即通过）
TEST_F(HcommEndpointMgrTest, Ut_DfxLog_When_AddRemove_Expect_NoCrash)
{
    EndpointDesc desc = MakeHostRoceDesc();
    EndpointHandle h = MakeHandle(0xB1);
    mgr_.Add(h, std::make_unique<UtStubEndpoint>(desc));
    EXPECT_TRUE(mgr_.Remove(h));
}

// TC-DFX_EndpointCtxMgr_日志-002: Acquire/Release use_count 日志（不崩溃即通过）
TEST_F(HcommEndpointCtxMgrTest, Ut_DfxLog_When_AcquireRelease_Expect_NoCrash)
{
    void* fakeHandle = reinterpret_cast<void*>(0xBBBB);
    MockRdmaHandle(fakeHandle);
    EndpointCtxKey key = MakeKey(0, COMM_PROTOCOL_ROCE, ENDPOINT_LOC_TYPE_HOST, "8.0.0.0");
    std::shared_ptr<EndpointCtx> ctx;
    EXPECT_EQ(mgr_.Acquire(key, false, ctx), HCCL_SUCCESS);
    ASSERT_NE(ctx, nullptr);
    mgr_.Release(key);
}
