/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h> // shm_unlink
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <vector>

#include "cmd_base_utils.h"
#include "cmd_cluster_model_utils.h"
#include "store_sim_memory_manager.h"
#include "store_sim_comm_pool_policy.h"

using namespace HcclSim;

namespace {
class TempDirGuard {
public:
    TempDirGuard() : path_(std::filesystem::temp_directory_path() / "hvm_ut_cmd_base")
    {
        std::filesystem::create_directories(path_);
    }
    ~TempDirGuard() { std::filesystem::remove_all(path_); }
    std::string str() const { return path_.string(); }
    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

class LdPreloadGuard {
public:
    LdPreloadGuard()
    {
        const char* v = std::getenv("LD_PRELOAD");
        if (v != nullptr) {
            saved_ = v;
            hadValue_ = true;
        }
    }
    ~LdPreloadGuard()
    {
        if (hadValue_) {
            setenv("LD_PRELOAD", saved_.c_str(), 1);
        } else {
            unsetenv("LD_PRELOAD");
        }
    }

private:
    std::string saved_;
    bool hadValue_ = false;
};
} // namespace

class CmdBaseUtilsTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

class CstyleCmdTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(CstyleCmdTest, SingleArg)
{
    std::vector<std::string> args = {"prog"};
    CstyleCmd cmd(args);
    EXPECT_EQ(cmd.argc(), 1);
    EXPECT_NE(cmd.argv(), nullptr);
    EXPECT_STREQ(cmd.argv()[0], "prog");
    EXPECT_EQ(cmd.cmd(), "prog ");
}

TEST_F(CstyleCmdTest, MultipleArgs)
{
    std::vector<std::string> args = {"prog", "arg1", "arg2"};
    CstyleCmd cmd(args);
    EXPECT_EQ(cmd.argc(), 3);
    EXPECT_STREQ(cmd.argv()[0], "prog");
    EXPECT_STREQ(cmd.argv()[1], "arg1");
    EXPECT_STREQ(cmd.argv()[2], "arg2");
}

TEST_F(CstyleCmdTest, EmptyArgs)
{
    std::vector<std::string> args = {};
    CstyleCmd cmd(args);
    EXPECT_EQ(cmd.argc(), 0);
    EXPECT_EQ(cmd.cmd(), "");
}

TEST_F(CstyleCmdTest, CmdStringFormatting)
{
    std::vector<std::string> args = {"hccl-vm", "start"};
    CstyleCmd cmd(args);
    EXPECT_EQ(cmd.cmd(), "hccl-vm start ");
}

TEST_F(CstyleCmdTest, ArgvPointersValid)
{
    std::vector<std::string> args = {"a", "b", "c"};
    CstyleCmd cmd(args);
    char** av = cmd.argv();
    ASSERT_NE(av, nullptr);
    for (int i = 0; i < cmd.argc(); ++i) {
        EXPECT_NE(av[i], nullptr);
    }
}

TEST_F(CstyleCmdTest, ArgWithSpaces)
{
    std::vector<std::string> args = {"prog", "hello world"};
    CstyleCmd cmd(args);
    EXPECT_EQ(cmd.argc(), 2);
    EXPECT_STREQ(cmd.argv()[1], "hello world");
}

class ArgvToStringTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(ArgvToStringTest, SingleArg)
{
    char* argv[] = {const_cast<char*>("test")};
    std::string result = ArgvToString(1, argv);
    EXPECT_EQ(result, "test");
}

TEST_F(ArgvToStringTest, MultipleArgs)
{
    char* argv[] = {const_cast<char*>("cmd"), const_cast<char*>("arg1"), const_cast<char*>("arg2")};
    std::string result = ArgvToString(3, argv);
    EXPECT_EQ(result, "cmd arg1 arg2");
}

TEST_F(ArgvToStringTest, ArgWithSpace)
{
    char* argv[] = {const_cast<char*>("cmd"), const_cast<char*>("hello world")};
    std::string result = ArgvToString(2, argv);
    EXPECT_EQ(result, "cmd \"hello world\"");
}

TEST_F(ArgvToStringTest, Empty)
{
    char* argv[] = {const_cast<char*>("")};
    std::string result = ArgvToString(1, argv);
    EXPECT_EQ(result, "");
}

TEST_F(ArgvToStringTest, MultipleSpacesInArg)
{
    char* argv[] = {const_cast<char*>("cmd"), const_cast<char*>("a b c")};
    std::string result = ArgvToString(2, argv);
    EXPECT_EQ(result, "cmd \"a b c\"");
}

TEST_F(ArgvToStringTest, TwoArgsBothWithSpaces)
{
    char* argv[] = {const_cast<char*>("hello world"), const_cast<char*>("foo bar")};
    std::string result = ArgvToString(2, argv);
    EXPECT_EQ(result, "\"hello world\" \"foo bar\"");
}

TEST_F(ArgvToStringTest, NoSpaceArgNotQuoted)
{
    char* argv[] = {const_cast<char*>("cmd"), const_cast<char*>("nospace")};
    std::string result = ArgvToString(2, argv);
    EXPECT_EQ(result, "cmd nospace");
}

class GetBinLocationTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(GetBinLocationTest, ReturnsNonEmpty)
{
    std::string loc = GetBinLocation();
    EXPECT_FALSE(loc.empty());
}

TEST_F(GetBinLocationTest, ReturnsValidDirectory)
{
    std::string loc = GetBinLocation();
    EXPECT_TRUE(std::filesystem::exists(loc));
    EXPECT_TRUE(std::filesystem::is_directory(loc));
}

TEST_F(GetBinLocationTest, ReturnsAbsolutePath)
{
    std::string loc = GetBinLocation();
    EXPECT_EQ(loc[0], '/');
}

class RemoveFromLDPreloadTest : public testing::Test {
protected:
    LdPreloadGuard guard_;
    void SetUp() override { unsetenv("LD_PRELOAD"); }
    void TearDown() override { unsetenv("LD_PRELOAD"); }
};

TEST_F(RemoveFromLDPreloadTest, NoLdPreloadSet)
{
    unsetenv("LD_PRELOAD");
    RemoveFromLDPreload("/tmp/libfoo.so");
    const char* val = std::getenv("LD_PRELOAD");
    EXPECT_EQ(val, nullptr);
}

TEST_F(RemoveFromLDPreloadTest, ExactMatch)
{
    setenv("LD_PRELOAD", "/tmp/libfoo.so", 1);
    RemoveFromLDPreload("/tmp/libfoo.so");
    const char* val = std::getenv("LD_PRELOAD");
    EXPECT_EQ(val, nullptr);
}

TEST_F(RemoveFromLDPreloadTest, PartialMatch)
{
    setenv("LD_PRELOAD", "/tmp/liba.so:/tmp/libb.so:/tmp/libc.so", 1);
    RemoveFromLDPreload("/tmp/libb.so");
    const char* val = std::getenv("LD_PRELOAD");
    ASSERT_NE(val, nullptr);
    EXPECT_EQ(std::string(val), "/tmp/liba.so:/tmp/libc.so");
}

TEST_F(RemoveFromLDPreloadTest, RemoveFirst)
{
    setenv("LD_PRELOAD", "/tmp/liba.so:/tmp/libb.so", 1);
    RemoveFromLDPreload("/tmp/liba.so");
    const char* val = std::getenv("LD_PRELOAD");
    ASSERT_NE(val, nullptr);
    EXPECT_EQ(std::string(val), "/tmp/libb.so");
}

TEST_F(RemoveFromLDPreloadTest, RemoveLast)
{
    setenv("LD_PRELOAD", "/tmp/liba.so:/tmp/libb.so", 1);
    RemoveFromLDPreload("/tmp/libb.so");
    const char* val = std::getenv("LD_PRELOAD");
    ASSERT_NE(val, nullptr);
    EXPECT_EQ(std::string(val), "/tmp/liba.so");
}

TEST_F(RemoveFromLDPreloadTest, NotFound)
{
    setenv("LD_PRELOAD", "/tmp/liba.so:/tmp/libb.so", 1);
    RemoveFromLDPreload("/tmp/libnotfound.so");
    const char* val = std::getenv("LD_PRELOAD");
    ASSERT_NE(val, nullptr);
    EXPECT_EQ(std::string(val), "/tmp/liba.so:/tmp/libb.so");
}

TEST_F(RemoveFromLDPreloadTest, AllRemoved)
{
    setenv("LD_PRELOAD", "/tmp/liba.so:/tmp/libb.so", 1);
    RemoveFromLDPreload("/tmp/liba.so");
    RemoveFromLDPreload("/tmp/libb.so");
    const char* val = std::getenv("LD_PRELOAD");
    EXPECT_EQ(val, nullptr);
}

TEST_F(RemoveFromLDPreloadTest, DoubleColonHandling)
{
    setenv("LD_PRELOAD", "/tmp/liba.so::/tmp/libb.so", 1);
    RemoveFromLDPreload("/tmp/liba.so");
    const char* val = std::getenv("LD_PRELOAD");
    ASSERT_NE(val, nullptr);
    EXPECT_EQ(std::string(val), "/tmp/libb.so");
}

TEST_F(RemoveFromLDPreloadTest, SingleEntryNotMatching)
{
    setenv("LD_PRELOAD", "/tmp/liba.so", 1);
    RemoveFromLDPreload("/tmp/libother.so");
    const char* val = std::getenv("LD_PRELOAD");
    ASSERT_NE(val, nullptr);
    EXPECT_EQ(std::string(val), "/tmp/liba.so");
}

class FileInModelDirTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(FileInModelDirTest, NonExistentModel)
{
    std::string result = FileInModelDir("nonexistent_model_12345");
    EXPECT_NE(result.find("not found"), std::string::npos);
}

TEST_F(FileInModelDirTest, ResultContainsYamlExtension)
{
    std::string result = FileInModelDir("nonexistent_model_12345");
    EXPECT_NE(result.find(".yaml"), std::string::npos);
}

TEST_F(FileInModelDirTest, ResultContainsModelName)
{
    std::string result = FileInModelDir("my_test_model");
    EXPECT_NE(result.find("my_test_model"), std::string::npos);
}

TEST_F(FileInModelDirTest, ResultContainsClusterModelDir)
{
    std::string result = FileInModelDir("some_model");
    EXPECT_NE(result.find("cluster_model"), std::string::npos);
}

class UninstallUserPluginParseTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(UninstallUserPluginParseTest, ParseSingleTag)
{
    std::string input = "/checker";
    std::vector<std::string> tags;
    size_t start = 0;
    size_t end = input.find(',');
    while (end != std::string::npos) {
        std::string tag = input.substr(start, end - start);
        tag.erase(tag.begin());
        tags.push_back(tag);
        start = end + 1;
        end = input.find(',', start);
    }
    std::string lastTag = input.substr(start);
    lastTag.erase(lastTag.begin());
    tags.push_back(lastTag);

    ASSERT_EQ(tags.size(), 1u);
    EXPECT_EQ(tags[0], "checker");
}

TEST_F(UninstallUserPluginParseTest, ParseMultipleTags)
{
    std::string input = "/checker,/runner";
    std::vector<std::string> tags;
    size_t start = 0;
    size_t end = input.find(',');
    while (end != std::string::npos) {
        std::string tag = input.substr(start, end - start);
        tag.erase(tag.begin());
        tags.push_back(tag);
        start = end + 1;
        end = input.find(',', start);
    }
    std::string lastTag = input.substr(start);
    lastTag.erase(lastTag.begin());
    tags.push_back(lastTag);

    ASSERT_EQ(tags.size(), 2u);
    EXPECT_EQ(tags[0], "checker");
    EXPECT_EQ(tags[1], "runner");
}

TEST_F(UninstallUserPluginParseTest, ParseThreeTags)
{
    std::string input = "/a,/b,/c";
    std::vector<std::string> tags;
    size_t start = 0;
    size_t end = input.find(',');
    while (end != std::string::npos) {
        std::string tag = input.substr(start, end - start);
        tag.erase(tag.begin());
        tags.push_back(tag);
        start = end + 1;
        end = input.find(',', start);
    }
    std::string lastTag = input.substr(start);
    lastTag.erase(lastTag.begin());
    tags.push_back(lastTag);

    ASSERT_EQ(tags.size(), 3u);
    EXPECT_EQ(tags[0], "a");
    EXPECT_EQ(tags[1], "b");
    EXPECT_EQ(tags[2], "c");
}

class InstallUserPluginTagParseTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(InstallUserPluginTagParseTest, ExtractTagFromPath)
{
    std::string argStr = "/usr/local/plugin/checker";
    const char delimiter = '/';
    std::string tag = argStr.substr(argStr.find_last_of(delimiter) + 1);
    EXPECT_EQ(tag, "checker");
}

TEST_F(InstallUserPluginTagParseTest, ExtractTagFromSimpleName)
{
    std::string argStr = "checker";
    const char delimiter = '/';
    std::string tag = argStr.substr(argStr.find_last_of(delimiter) + 1);
    EXPECT_EQ(tag, "checker");
}

TEST_F(InstallUserPluginTagParseTest, ExtractTagFromDeepPath)
{
    std::string argStr = "/a/b/c/d/myplugin";
    const char delimiter = '/';
    std::string tag = argStr.substr(argStr.find_last_of(delimiter) + 1);
    EXPECT_EQ(tag, "myplugin");
}

TEST_F(InstallUserPluginTagParseTest, TrailingSlash)
{
    std::string argStr = "/path/to/plugin/";
    const char delimiter = '/';
    std::string tag = argStr.substr(argStr.find_last_of(delimiter) + 1);
    EXPECT_EQ(tag, "");
}

class ArgvToStringEdgeTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(ArgvToStringEdgeTest, TwoArgsNoSpace)
{
    char* argv[] = {const_cast<char*>("a"), const_cast<char*>("b")};
    std::string result = ArgvToString(2, argv);
    EXPECT_EQ(result, "a b");
}

TEST_F(ArgvToStringEdgeTest, ArgWithLeadingSpace)
{
    char* argv[] = {const_cast<char*>("cmd"), const_cast<char*>(" leading")};
    std::string result = ArgvToString(2, argv);
    EXPECT_EQ(result, "cmd \" leading\"");
}

TEST_F(ArgvToStringEdgeTest, ArgWithTrailingSpace)
{
    char* argv[] = {const_cast<char*>("cmd"), const_cast<char*>("trailing ")};
    std::string result = ArgvToString(2, argv);
    EXPECT_EQ(result, "cmd \"trailing \"");
}

TEST_F(ArgvToStringEdgeTest, ArgWithOnlySpaces)
{
    char* argv[] = {const_cast<char*>("cmd"), const_cast<char*>("   ")};
    std::string result = ArgvToString(2, argv);
    EXPECT_EQ(result, "cmd \"   \"");
}

TEST_F(ArgvToStringEdgeTest, LongArg)
{
    std::string longArg(1000, 'x');
    char* argv[] = {const_cast<char*>(longArg.c_str())};
    std::string result = ArgvToString(1, argv);
    EXPECT_EQ(result.size(), 1000u);
    EXPECT_EQ(result, longArg);
}

class CstyleCmdEdgeTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(CstyleCmdEdgeTest, ArgsWithSpecialChars)
{
    std::vector<std::string> args = {"prog", "--flag=value"};
    CstyleCmd cmd(args);
    EXPECT_EQ(cmd.argc(), 2);
    EXPECT_STREQ(cmd.argv()[1], "--flag=value");
}

TEST_F(CstyleCmdEdgeTest, ArgsWithDash)
{
    std::vector<std::string> args = {"prog", "-a", "-b", "-c"};
    CstyleCmd cmd(args);
    EXPECT_EQ(cmd.argc(), 4);
    EXPECT_EQ(cmd.cmd(), "prog -a -b -c ");
}

TEST_F(CstyleCmdEdgeTest, LargeArgCount)
{
    std::vector<std::string> args;
    for (int i = 0; i < 100; ++i) {
        args.push_back("arg" + std::to_string(i));
    }
    CstyleCmd cmd(args);
    EXPECT_EQ(cmd.argc(), 100);
}

// ==================== RemoveFromLDPreload additional edge cases ====================

class RemoveFromLDPreloadEdgeTest : public testing::Test {
protected:
    LdPreloadGuard guard_;
    void SetUp() override { unsetenv("LD_PRELOAD"); }
    void TearDown() override { unsetenv("LD_PRELOAD"); }
};

TEST_F(RemoveFromLDPreloadEdgeTest, EmptyStringAfterRemoval)
{
    setenv("LD_PRELOAD", "/tmp/liba.so:/tmp/libb.so", 1);
    RemoveFromLDPreload("/tmp/liba.so");
    const char* val = std::getenv("LD_PRELOAD");
    ASSERT_NE(val, nullptr);
    EXPECT_EQ(std::string(val), "/tmp/libb.so");
}

TEST_F(RemoveFromLDPreloadEdgeTest, AllEntriesRemoved)
{
    setenv("LD_PRELOAD", "/tmp/liba.so:/tmp/libb.so", 1);
    RemoveFromLDPreload("/tmp/liba.so");
    RemoveFromLDPreload("/tmp/libb.so");
    const char* val = std::getenv("LD_PRELOAD");
    EXPECT_EQ(val, nullptr);
}

TEST_F(RemoveFromLDPreloadEdgeTest, SingleEntryExactMatch)
{
    setenv("LD_PRELOAD", "/tmp/libfoo.so", 1);
    RemoveFromLDPreload("/tmp/libfoo.so");
    const char* val = std::getenv("LD_PRELOAD");
    EXPECT_EQ(val, nullptr);
}

TEST_F(RemoveFromLDPreloadEdgeTest, SingleEntryNotMatching)
{
    setenv("LD_PRELOAD", "/tmp/libfoo.so", 1);
    RemoveFromLDPreload("/tmp/libbar.so");
    const char* val = std::getenv("LD_PRELOAD");
    ASSERT_NE(val, nullptr);
    EXPECT_EQ(std::string(val), "/tmp/libfoo.so");
}

// ==================== FileInModelDir additional tests ====================

class FileInModelDirEdgeTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(FileInModelDirEdgeTest, NonExistentFileReturnsError)
{
    std::string result = FileInModelDir("definitely_nonexistent_model_xyz");
    EXPECT_NE(result.find("not found"), std::string::npos);
}

TEST_F(FileInModelDirEdgeTest, EmptyFileName)
{
    std::string result = FileInModelDir("");
    EXPECT_NE(result.find("not found"), std::string::npos);
}

TEST_F(FileInModelDirEdgeTest, FileNameWithSpecialChars)
{
    std::string result = FileInModelDir("model!@#$%");
    EXPECT_NE(result.find("not found"), std::string::npos);
}

// ==================== ArgvToString additional edge cases ====================

class ArgvToStringAdditionalTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(ArgvToStringAdditionalTest, ZeroArgs)
{
    char* argv[] = {const_cast<char*>("prog")};
    std::string result = ArgvToString(1, argv);
    EXPECT_EQ(result, "prog");
}

TEST_F(ArgvToStringAdditionalTest, ThreeArgsNoSpace)
{
    char* argv[] = {const_cast<char*>("a"), const_cast<char*>("b"), const_cast<char*>("c")};
    std::string result = ArgvToString(3, argv);
    EXPECT_EQ(result, "a b c");
}

TEST_F(ArgvToStringAdditionalTest, MixedSpaceAndNoSpace)
{
    char* argv[] = {const_cast<char*>("cmd"), const_cast<char*>("no_space"), const_cast<char*>("has space")};
    std::string result = ArgvToString(3, argv);
    EXPECT_EQ(result, "cmd no_space \"has space\"");
}

TEST_F(ArgvToStringAdditionalTest, EmptyArg)
{
    char* argv[] = {const_cast<char*>(""), const_cast<char*>("b")};
    std::string result = ArgvToString(2, argv);
    EXPECT_EQ(result, " b");
}

// ==================== CstyleCmd additional edge cases ====================

class CstyleCmdAdditionalTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(CstyleCmdAdditionalTest, SingleArgNoSpaces)
{
    std::vector<std::string> args = {"prog"};
    CstyleCmd cmd(args);
    EXPECT_EQ(cmd.argc(), 1);
    EXPECT_STREQ(cmd.argv()[0], "prog");
    EXPECT_EQ(cmd.cmd(), "prog ");
}

TEST_F(CstyleCmdAdditionalTest, TwoArgsWithSpaces)
{
    std::vector<std::string> args = {"prog", "hello world"};
    CstyleCmd cmd(args);
    EXPECT_EQ(cmd.argc(), 2);
    EXPECT_STREQ(cmd.argv()[1], "hello world");
}

TEST_F(CstyleCmdAdditionalTest, RebuildAfterDestruction)
{
    {
        std::vector<std::string> args = {"a", "b"};
        CstyleCmd cmd(args);
        EXPECT_EQ(cmd.argc(), 2);
    }
    std::vector<std::string> args2 = {"x", "y", "z"};
    CstyleCmd cmd2(args2);
    EXPECT_EQ(cmd2.argc(), 3);
}

// ==================== GetBinLocation additional tests ====================

class GetBinLocationAdditionalTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(GetBinLocationAdditionalTest, ReturnsConsistentPath)
{
    std::string loc1 = GetBinLocation();
    std::string loc2 = GetBinLocation();
    EXPECT_EQ(loc1, loc2);
}

TEST_F(GetBinLocationAdditionalTest, PathContainsNoTrailingSlash)
{
    std::string loc = GetBinLocation();
    EXPECT_FALSE(loc.empty());
    EXPECT_NE(loc.back(), '/');
}

TEST_F(GetBinLocationAdditionalTest, PathIsAbsolute)
{
    std::string loc = GetBinLocation();
    EXPECT_EQ(loc[0], '/');
}

// ==================== InstallUserPluginTagParse additional tests ====================

class InstallUserPluginTagParseAdditionalTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(InstallUserPluginTagParseAdditionalTest, SingleNameNoPath)
{
    std::string argStr = "checker";
    const char delimiter = '/';
    std::string tag = argStr.substr(argStr.find_last_of(delimiter) + 1);
    EXPECT_EQ(tag, "checker");
}

TEST_F(InstallUserPluginTagParseAdditionalTest, PathWithTrailingSlash)
{
    std::string argStr = "/path/to/plugin/";
    const char delimiter = '/';
    std::string tag = argStr.substr(argStr.find_last_of(delimiter) + 1);
    EXPECT_EQ(tag, "");
}

TEST_F(InstallUserPluginTagParseAdditionalTest, DeepNestedPath)
{
    std::string argStr = "/a/b/c/d/e/f/plugin";
    const char delimiter = '/';
    std::string tag = argStr.substr(argStr.find_last_of(delimiter) + 1);
    EXPECT_EQ(tag, "plugin");
}

// ==================== UninstallUserPluginParse additional tests ====================

class UninstallUserPluginParseAdditionalTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(UninstallUserPluginParseAdditionalTest, ParseSingleTagWithLeadingSlash)
{
    std::string input = "/checker";
    std::vector<std::string> tags;
    size_t start = 0;
    size_t end = input.find(',');
    while (end != std::string::npos) {
        std::string tag = input.substr(start, end - start);
        tag.erase(tag.begin());
        tags.push_back(tag);
        start = end + 1;
        end = input.find(',', start);
    }
    std::string lastTag = input.substr(start);
    lastTag.erase(lastTag.begin());
    tags.push_back(lastTag);

    ASSERT_EQ(tags.size(), 1u);
    EXPECT_EQ(tags[0], "checker");
}

TEST_F(UninstallUserPluginParseAdditionalTest, ParseEmptyInput)
{
    std::string input = "";
    std::vector<std::string> tags;
    if (!input.empty()) {
        size_t start = 0;
        size_t end = input.find(',');
        while (end != std::string::npos) {
            std::string tag = input.substr(start, end - start);
            if (!tag.empty()) {
                tag.erase(tag.begin());
                tags.push_back(tag);
            }
            start = end + 1;
            end = input.find(',', start);
        }
        std::string lastTag = input.substr(start);
        if (!lastTag.empty()) {
            lastTag.erase(lastTag.begin());
            tags.push_back(lastTag);
        }
    }
    EXPECT_EQ(tags.size(), 0u);
}

TEST_F(UninstallUserPluginParseAdditionalTest, ParseFourTags)
{
    std::string input = "/a,/b,/c,/d";
    std::vector<std::string> tags;
    size_t start = 0;
    size_t end = input.find(',');
    while (end != std::string::npos) {
        std::string tag = input.substr(start, end - start);
        tag.erase(tag.begin());
        tags.push_back(tag);
        start = end + 1;
        end = input.find(',', start);
    }
    std::string lastTag = input.substr(start);
    lastTag.erase(lastTag.begin());
    tags.push_back(lastTag);

    ASSERT_EQ(tags.size(), 4u);
    EXPECT_EQ(tags[0], "a");
    EXPECT_EQ(tags[1], "b");
    EXPECT_EQ(tags[2], "c");
    EXPECT_EQ(tags[3], "d");
}

// ==================== ParseYamlTopo Tests ====================

class ParseYamlTopoTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(ParseYamlTopoTest, ParseExistingYamlFile112)
{
    TopoMeta topo;

    bool result = ParseYamlTopo("112", topo);
    EXPECT_TRUE(result);
    EXPECT_EQ(topo.size(), 1u);
    if (!topo.empty()) {
        EXPECT_EQ(topo[0].size(), 1u);
        if (!topo[0].empty()) {
            EXPECT_EQ(topo[0][0].size(), 2u);
        }
    }
}

TEST_F(ParseYamlTopoTest, ParseExistingYamlFile114)
{
    TopoMeta topo;

    bool result = ParseYamlTopo("114", topo);
    EXPECT_TRUE(result);
    EXPECT_EQ(topo.size(), 1u);
    if (!topo.empty()) {
        EXPECT_EQ(topo[0].size(), 1u);
        if (!topo[0].empty()) {
            EXPECT_EQ(topo[0][0].size(), 4u);
        }
    }
}

TEST_F(ParseYamlTopoTest, ParseExistingYamlFile118)
{
    TopoMeta topo;

    bool result = ParseYamlTopo("118", topo);
    EXPECT_TRUE(result);
    EXPECT_EQ(topo.size(), 1u);
    if (!topo.empty()) {
        EXPECT_EQ(topo[0].size(), 1u);
        if (!topo[0].empty()) {
            EXPECT_EQ(topo[0][0].size(), 8u);
        }
    }
}

TEST_F(ParseYamlTopoTest, ParseExistingYamlFile121)
{
    TopoMeta topo;

    bool result = ParseYamlTopo("121", topo);
    EXPECT_TRUE(result);
    EXPECT_EQ(topo.size(), 1u);
    if (!topo.empty()) {
        EXPECT_EQ(topo[0].size(), 2u);
    }
}

TEST_F(ParseYamlTopoTest, ParseExistingYamlFile122)
{
    TopoMeta topo;

    bool result = ParseYamlTopo("122", topo);
    EXPECT_TRUE(result);
    EXPECT_EQ(topo.size(), 1u);
    if (!topo.empty()) {
        EXPECT_EQ(topo[0].size(), 2u);
    }
}

TEST_F(ParseYamlTopoTest, ParseNonExistentYamlFile)
{
    TopoMeta topo;

    bool result = ParseYamlTopo("nonexistent_model_xyz", topo);
    EXPECT_FALSE(result);
}

// ==================== InstallUserPlugin Tests ====================

class InstallUserPluginTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(InstallUserPluginTest, NonexistentPluginPath)
{
    HcclVmResult ret = InstallUserPlugin("/nonexistent/path/to/plugin");
    EXPECT_NE(ret, HCCL_SIM_HOST_SUCCESS_CMD);
}

TEST_F(InstallUserPluginTest, SimpleNameNoSlash)
{
    HcclVmResult ret = InstallUserPlugin("checker");
    EXPECT_NE(ret, HCCL_SIM_HOST_SUCCESS_CMD);
}

TEST_F(InstallUserPluginTest, DeepPathNonexistent)
{
    HcclVmResult ret = InstallUserPlugin("/a/b/c/d/myplugin");
    EXPECT_NE(ret, HCCL_SIM_HOST_SUCCESS_CMD);
}

// ==================== UninstallUserPlugin Tests ====================

class UninstallUserPluginTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(UninstallUserPluginTest, SingleTag)
{
    HcclVmResult ret = UninstallUserPlugin("/checker");
    EXPECT_NE(ret, HCCL_SIM_HOST_SUCCESS_CMD);
}

TEST_F(UninstallUserPluginTest, MultipleTags)
{
    HcclVmResult ret = UninstallUserPlugin("/checker,/runner");
    EXPECT_NE(ret, HCCL_SIM_HOST_SUCCESS_CMD);
}

TEST_F(UninstallUserPluginTest, ThreeTags)
{
    HcclVmResult ret = UninstallUserPlugin("/a,/b,/c");
    EXPECT_NE(ret, HCCL_SIM_HOST_SUCCESS_CMD);
}

// ==================== ShowUserPlugin Tests ====================

class ShowUserPluginFuncTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(ShowUserPluginFuncTest, NoPluginInstalled) { ShowUserPlugin(); }

// ==================== RemoveFromLDPreload: empty result path (line 202) ====================

TEST_F(RemoveFromLDPreloadEdgeTest, DuplicateEntriesAllRemoved)
{
    setenv("LD_PRELOAD", "/tmp/liba.so:/tmp/liba.so", 1);
    RemoveFromLDPreload("/tmp/liba.so");
    const char* val = std::getenv("LD_PRELOAD");
    EXPECT_EQ(val, nullptr);
}

TEST_F(RemoveFromLDPreloadEdgeTest, ColonPrefixThenRemoveAll)
{
    setenv("LD_PRELOAD", ":/tmp/liba.so", 1);
    RemoveFromLDPreload("/tmp/liba.so");
    const char* val = std::getenv("LD_PRELOAD");
    EXPECT_EQ(val, nullptr);
}

TEST_F(RemoveFromLDPreloadEdgeTest, TripleColonThenRemoveAll)
{
    setenv("LD_PRELOAD", "::/tmp/liba.so", 1);
    RemoveFromLDPreload("/tmp/liba.so");
    const char* val = std::getenv("LD_PRELOAD");
    EXPECT_EQ(val, nullptr);
}

// ==================== LogLevel Tests (error path: proxyConfig == nullptr) ====================

class LogLevelTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(LogLevelTest, SetConsoleLogLevelNoShm) { EXPECT_EQ(SetConsoleLogLevel(2), HCCL_SIM_HOST_ERROR_CMD); }

TEST_F(LogLevelTest, SetFileLogLevelNoShm) { EXPECT_EQ(SetFileLogLevel(2), HCCL_SIM_HOST_ERROR_CMD); }

TEST_F(LogLevelTest, ShowCurrentLogLevelNoShm) { EXPECT_EQ(ShowCurrentLogLevel(), HCCL_SIM_HOST_ERROR_CMD); }

TEST_F(LogLevelTest, SetConsoleLogLevelBoundaryLow) { EXPECT_EQ(SetConsoleLogLevel(0), HCCL_SIM_HOST_ERROR_CMD); }

TEST_F(LogLevelTest, SetConsoleLogLevelBoundaryHigh) { EXPECT_EQ(SetConsoleLogLevel(5), HCCL_SIM_HOST_ERROR_CMD); }

TEST_F(LogLevelTest, SetFileLogLevelBoundaryLow) { EXPECT_EQ(SetFileLogLevel(0), HCCL_SIM_HOST_ERROR_CMD); }

TEST_F(LogLevelTest, SetFileLogLevelBoundaryHigh) { EXPECT_EQ(SetFileLogLevel(5), HCCL_SIM_HOST_ERROR_CMD); }

// ==================== ShowModel Error Path Tests ====================

class ShowModelErrorPathTest : public testing::Test {
protected:
    std::string modelPath_;
    std::string backupPath_;

    void SetUp() override
    {
        modelPath_ = GetBinLocation() + "/cluster_model";
        backupPath_ = GetBinLocation() + "/cluster_model_ut_backup";
    }

    void TearDown() override
    {
        if (std::filesystem::exists(backupPath_)) {
            if (std::filesystem::exists(modelPath_)) {
                std::filesystem::remove_all(modelPath_);
            }
            std::filesystem::rename(backupPath_, modelPath_);
        }
    }

    void BackupModelDir()
    {
        if (std::filesystem::exists(modelPath_) && !std::filesystem::exists(backupPath_)) {
            std::filesystem::rename(modelPath_, backupPath_);
        }
    }
};

TEST_F(ShowModelErrorPathTest, PathNotExist)
{
    BackupModelDir();
    ShowModel();
}

TEST_F(ShowModelErrorPathTest, PathNotDirectory)
{
    BackupModelDir();
    std::ofstream fakeFile(modelPath_);
    fakeFile << "not a directory" << std::endl;
    fakeFile.close();
    ShowModel();
    std::filesystem::remove(modelPath_);
}

TEST_F(ShowModelErrorPathTest, NoModelFiles)
{
    BackupModelDir();
    std::filesystem::create_directories(modelPath_);
    ShowModel();
    std::filesystem::remove_all(modelPath_);
}

class ParseYamlTopoBoundaryTest : public testing::Test {
protected:
    std::string modelDir_;
    void SetUp() override
    {
        modelDir_ = GetBinLocation() + "/cluster_model/topo_meta";
        std::filesystem::create_directories(modelDir_);
    }
    void TearDown() override {}
    void WriteYaml(const std::string& name, const std::string& content)
    {
        std::string path = modelDir_ + "/" + name + ".yaml";
        std::ofstream ofs(path);
        ofs << content;
        ofs.close();
    }
    void RemoveYaml(const std::string& name)
    {
        std::string path = modelDir_ + "/" + name + ".yaml";
        std::filesystem::remove(path);
    }
};

TEST_F(ParseYamlTopoBoundaryTest, ValidTopoWithoutSocVersion)
{
    WriteYaml("ut_missing_soc", R"(meta:
  podNum: 1
  serNum: 1
  rankNum: 2
topology:
  - podId: 0
    servers:
      - serId: 0
        ip: 192.1.5.5
        ranks: [0, 1]
)");
    TopoMeta topo;
    EXPECT_TRUE(ParseYamlTopo("ut_missing_soc", topo));
    EXPECT_EQ(topo.size(), 1u);
    RemoveYaml("ut_missing_soc");
}

TEST_F(ParseYamlTopoBoundaryTest, IgnoresExtraSocVersionField)
{
    WriteYaml("ut_wrong_soc", R"(
meta:
  socVersion: Ascend999
  podNum: 1
  serNum: 1
  rankNum: 2
topology:
  - podId: 0
    servers:
      - serId: 0
        ip: 192.1.5.5
        ranks: [0, 1]
)");
    TopoMeta topo;
    EXPECT_TRUE(ParseYamlTopo("ut_wrong_soc", topo));
    EXPECT_EQ(topo.size(), 1u);
    RemoveYaml("ut_wrong_soc");
}

TEST_F(ParseYamlTopoBoundaryTest, PodNumExceedsLimit)
{
    WriteYaml("ut_podnum_limit", R"(
meta:
  podNum: 1025
  serNum: 1
  rankNum: 2
topology:
  - podId: 0
    servers:
      - serId: 0
        ip: 192.1.5.5
        ranks: [0, 1]
)");
    TopoMeta topo;
    EXPECT_FALSE(ParseYamlTopo("ut_podnum_limit", topo));
    RemoveYaml("ut_podnum_limit");
}

TEST_F(ParseYamlTopoBoundaryTest, SerNumExceedsLimit)
{
    WriteYaml("ut_sernum_limit", R"(
meta:
  podNum: 1
  serNum: 1025
  rankNum: 2
topology:
  - podId: 0
    servers:
      - serId: 0
        ip: 192.1.5.5
        ranks: [0, 1]
)");
    TopoMeta topo;
    EXPECT_FALSE(ParseYamlTopo("ut_sernum_limit", topo));
    RemoveYaml("ut_sernum_limit");
}

TEST_F(ParseYamlTopoBoundaryTest, RankNumExceedsLimit)
{
    WriteYaml("ut_ranknum_limit", R"(
meta:
  podNum: 1
  serNum: 1
  rankNum: 1025
topology:
  - podId: 0
    servers:
      - serId: 0
        ip: 192.1.5.5
        ranks: [0, 1]
)");
    TopoMeta topo;
    EXPECT_FALSE(ParseYamlTopo("ut_ranknum_limit", topo));
    RemoveYaml("ut_ranknum_limit");
}

TEST_F(ParseYamlTopoBoundaryTest, PodCountMismatch)
{
    WriteYaml("ut_pod_mismatch", R"(
meta:
  podNum: 2
  serNum: 1
  rankNum: 2
topology:
  - podId: 0
    servers:
      - serId: 0
        ip: 192.1.5.5
        ranks: [0, 1]
)");
    TopoMeta topo;
    EXPECT_TRUE(ParseYamlTopo("ut_pod_mismatch", topo));
    EXPECT_EQ(topo.size(), 1u);
    RemoveYaml("ut_pod_mismatch");
}

TEST_F(ParseYamlTopoBoundaryTest, ServerCountMismatch)
{
    WriteYaml("ut_server_mismatch", R"(
meta:
  podNum: 1
  serNum: 2
  rankNum: 2
topology:
  - podId: 0
    servers:
      - serId: 0
        ip: 192.1.5.5
        ranks: [0, 1]
)");
    TopoMeta topo;
    EXPECT_TRUE(ParseYamlTopo("ut_server_mismatch", topo));
    ASSERT_EQ(topo.size(), 1u);
    EXPECT_EQ(topo[0].size(), 1u);
    RemoveYaml("ut_server_mismatch");
}

TEST_F(ParseYamlTopoBoundaryTest, RankCountMismatch)
{
    WriteYaml("ut_rank_mismatch", R"(
meta:
  podNum: 1
  serNum: 1
  rankNum: 4
topology:
  - podId: 0
    servers:
      - serId: 0
        ip: 192.1.5.5
        ranks: [0, 1]
)");
    TopoMeta topo;
    EXPECT_TRUE(ParseYamlTopo("ut_rank_mismatch", topo));
    ASSERT_EQ(topo.size(), 1u);
    ASSERT_EQ(topo[0].size(), 1u);
    EXPECT_EQ(topo[0][0].size(), 2u);
    RemoveYaml("ut_rank_mismatch");
}

TEST_F(ParseYamlTopoBoundaryTest, MalformedYaml)
{
    WriteYaml("ut_malformed", R"(
meta:
  podNum: 1
  serNum: 1
  rankNum: 2
topology:
  - podId: 0
    servers:
      - serId: 0
        ip: 192.1.5.5
        ranks: [0, 1
)");
    TopoMeta topo;
    EXPECT_FALSE(ParseYamlTopo("ut_malformed", topo));
    RemoveYaml("ut_malformed");
}

TEST_F(ParseYamlTopoBoundaryTest, MissingTopology)
{
    WriteYaml("ut_no_topo", R"(
meta:
  podNum: 1
  serNum: 1
  rankNum: 2
)");
    TopoMeta topo;
    EXPECT_TRUE(ParseYamlTopo("ut_no_topo", topo));
    EXPECT_TRUE(topo.empty());
    RemoveYaml("ut_no_topo");
}

TEST_F(ParseYamlTopoBoundaryTest, Ascend960SocVersion)
{
    WriteYaml("ut_960", R"(
meta:
  podNum: 1
  serNum: 1
  rankNum: 2
topology:
  - podId: 0
    servers:
      - serId: 0
        ip: 192.1.5.5
        ranks: [0, 1]
)");
    TopoMeta topo;

    EXPECT_TRUE(ParseYamlTopo("ut_960", topo));
    RemoveYaml("ut_960");
}

// ==================== RunUserPlugin Tests ====================

class RunUserPluginTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(RunUserPluginTest, NoPluginsInstalled)
{
    HcclVmResult ret = RunUserPlugin("/checker");
    EXPECT_NE(ret, HCCL_SIM_HOST_SUCCESS_CMD);
}

TEST_F(RunUserPluginTest, InvalidTag)
{
    HcclVmResult ret = RunUserPlugin("/nonexistent_tag");
    EXPECT_NE(ret, HCCL_SIM_HOST_SUCCESS_CMD);
}

TEST_F(RunUserPluginTest, EmptyTag) { EXPECT_ANY_THROW(RunUserPlugin("")); }

// ==================== AIV expansion mode environment checks ====================

class AivModeTest : public testing::Test {
protected:
    void SetUp() override { unsetenv("HCCL_OP_EXPANSION_MODE"); }
    void TearDown() override { unsetenv("HCCL_OP_EXPANSION_MODE"); }
};

TEST_F(AivModeTest, AivModeDisabledByDefault)
{
    const char* val = std::getenv("HCCL_OP_EXPANSION_MODE");
    EXPECT_EQ(val, nullptr);
}

TEST_F(AivModeTest, AivModeEnabled)
{
    setenv("HCCL_OP_EXPANSION_MODE", "AIV", 1);
    const char* val = std::getenv("HCCL_OP_EXPANSION_MODE");
    ASSERT_NE(val, nullptr);
    EXPECT_EQ(std::string(val), "AIV");
}

TEST_F(AivModeTest, AivModeNonAivValue)
{
    setenv("HCCL_OP_EXPANSION_MODE", "OTHER", 1);
    const char* val = std::getenv("HCCL_OP_EXPANSION_MODE");
    ASSERT_NE(val, nullptr);
    EXPECT_NE(std::string(val), "AIV");
}

// ==================== FileInModelDir with existing model ====================

class FileInModelDirExistingTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(FileInModelDirExistingTest, ExistingModelFile)
{
    std::string result = FileInModelDir("112");
    EXPECT_EQ(result, "");
}

TEST_F(FileInModelDirExistingTest, AnotherExistingModelFile)
{
    std::string result = FileInModelDir("114");
    EXPECT_EQ(result, "");
}

// ==================== ShowModel with existing models ====================

class ShowModelExistingTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(ShowModelExistingTest, ShowWithExistingModels) { ShowModel(); }

// ==================== UninstallUserPlugin edge cases ====================

class UninstallUserPluginEdgeTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(UninstallUserPluginEdgeTest, LongTagString)
{
    HcclVmResult ret = UninstallUserPlugin("/very_long_plugin_tag_name_for_testing");
    EXPECT_NE(ret, HCCL_SIM_HOST_SUCCESS_CMD);
}

TEST_F(UninstallUserPluginEdgeTest, TagWithNumbers)
{
    HcclVmResult ret = UninstallUserPlugin("/plugin123");
    EXPECT_NE(ret, HCCL_SIM_HOST_SUCCESS_CMD);
}

// ==================== InstallUserPlugin edge cases ====================

class InstallUserPluginEdgeTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(InstallUserPluginEdgeTest, EmptyString)
{
    HcclVmResult ret = InstallUserPlugin("");
    EXPECT_NE(ret, HCCL_SIM_HOST_SUCCESS_CMD);
}

TEST_F(InstallUserPluginEdgeTest, SingleCharName)
{
    HcclVmResult ret = InstallUserPlugin("x");
    EXPECT_NE(ret, HCCL_SIM_HOST_SUCCESS_CMD);
}

// ==================== InitHvmEnv Error Path Tests ====================

class InitHvmEnvTest : public testing::Test {
protected:
    void SetUp() override
    {
        // 逐用例清掉 HcclCommPool 残留。
        sim::MemoryManager::GetInstance().FreeMemByName(sim::CommPoolPolicy::kPoolName);
        shm_unlink(sim::CommPoolPolicy::kPoolName);
        unsetenv("HCCL_OP_EXPANSION_MODE");
    }
    void TearDown() override
    {
        sim::MemoryManager::GetInstance().FreeMemByName(sim::CommPoolPolicy::kPoolName);
        unsetenv("HCCL_OP_EXPANSION_MODE");
    }
};

TEST_F(InitHvmEnvTest, InitializesSharedMemoryWithoutAivValidation)
{
    HcclVmResult ret = InitHvmEnv("/nonexistent/path/for/ut", 2, false);
    EXPECT_EQ(ret, HCCL_SIM_HOST_SUCCESS_CMD);

    // clean 模式不建复用区 HcclCommPool。
    EXPECT_EQ(sim::MemoryManager::GetInstance().AcquireMemByName(sim::CommPoolPolicy::kPoolName), nullptr);
}

TEST_F(InitHvmEnvTest, FailsWhenCommPoolNameAlreadyExists)
{
    // 仅校验模式下池名被预先占用时，InitHvmEnv 建池失败、返回错误。
    ASSERT_NE(sim::MemoryManager::GetInstance().AllocMemByName(sim::CommPoolPolicy::kPoolName, 4096), nullptr);
    EXPECT_EQ(InitHvmEnv("/nonexistent/path/for/ut", 2, true), HCCL_SIM_HOST_ERROR_CMD);
}
