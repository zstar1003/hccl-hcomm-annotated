# pre-commit Tool Guide

## Overview

pre-commit is a Git Hooks framework that automatically runs code checking and formatting tools during `git commit`. This project is configured with the following checks:

| Hook | Function | Description |
| ---------------- | ---------------- | -------------------------------- |
| **clang-format** | C and C++ code formatting | Automatically formats code to maintain consistent style |
| **OAT Check** | Open source compliance check | Detects license headers and prevents binary file submission |

## Requirements

- **Git**: 2.0+
- **Python**: 3.8+
- **clang-format**: v18.1.8 (code formatting tool, must match the `rev` in [.pre-commit-config.yaml](../../../.pre-commit-config.yaml))
- **Java**: 17+ (required for OAT tool, can be installed automatically)
- **Maven**: 3.6+ (required for OAT tool, can be installed automatically)

## Installation Steps

### 1. Install pre-commit

```bash
# Option 1: Using pip
pip install pre-commit

# Option 2: Using system package manager (Ubuntu or Debian)
sudo apt install pre-commit
```

### 2. Install Dependency Tools

```bash
# Ubuntu or Debian
sudo apt install clang-format openjdk-17-jre maven

# macOS
brew install clang-format openjdk@17 maven
```

### 3. Install Git Hooks in the Project Path

```bash
# Navigate to the repository root directory
cd /path/to/hcomm
pre-commit install
```

A successful installation displays:

```bash
pre-commit installed at .git/hooks/pre-commit
```

## Usage

### Automatic Checking (Recommended)

Each time you execute `git commit`, pre-commit automatically runs checks:

```bash
git add .
git commit -m "your commit message"
```

Example output:

```text
clang-format.............................................................Passed
OAT Compliance Check.....................................................Passed
```

### Running Checks Manually

```bash
# Run all checks
pre-commit run

# Run a specific type of check
pre-commit run clang-format
pre-commit run oat-check

# Check all files (not limited to the staging area)
pre-commit run --all-files
```

### Skipping Checks (Emergency)

```bash
git commit --no-verify -m "emergency fix"
```

> **Note**: Use this only in emergencies. During normal development, ensure all checks pass.

## Check Descriptions

### 1. clang-format

Automatically formats C and C++ code according to the [.clang-format](../../../.clang-format) configuration in the project root directory.

### 2. OAT Compliance Check

OAT (Open Source Audit Tool) checks open source compliance:

| Check Item | Description |
| -------------- | ------------------------------ |
| License header check | Ensures source files contain the CANN License header |
| Binary file check | Prevents submission of binary files |
| Archive file check | Prevents submission of archive files such as zip and tar |

The first time the OAT check script runs, it automatically:

1. Detects or installs Java 17
2. Detects or installs Maven
3. Clones and compiles the tools_oat tool (approximately 1 to 2 minutes)

## Frequently Asked Questions

### Q1: The OAT check is slow during the first commit

**Cause**: The first run needs to clone and compile the OAT tool.

**Solution**: This is normal. Subsequent commits use the cached JAR and are much faster.

## Related Documents

- [pre-commit Official Documentation](https://pre-commit.com/)
- [clang-format Configuration](https://clang.llvm.org/docs/ClangFormatStyleOptions.html)
- [OAT Tool](https://gitcode.com/openharmony-sig/tools_oat)
- [pre-commit Integration Guide for Code Repositories (Chinese)](https://gitcode.com/cann/infrastructure/blob/main/docs/SC/pre-commit/pre-commit%E9%85%8D%E7%BD%AE%E6%8C%87%E5%AF%BC%E4%B9%A6.md)
