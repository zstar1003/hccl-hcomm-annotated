# Security Statement

## Recommended User Account

For security reasons, avoid using administrator accounts such as root to execute any commands. Follow the principle of least privilege.

## File Permission Control

- It is recommended to set the umask value to 0027 or higher on the host (including the host machine and containers) to ensure that the default maximum permissions for new folders are 750 and for new files are 640.
- Apply security measures such as permission control to sensitive content, including personal private data, business assets, and source files. For example, control the permissions of the installation directory and input public data files. For recommended permission settings, refer to [A - Recommended Maximum Permissions for Files (Folders) in Various Scenarios](#a---recommended-maximum-permissions-for-files-folders-in-various-scenarios).
- Control permissions during installation and usage. Refer to [A - Recommended Maximum Permissions for Files (Folders) in Various Scenarios](#a---recommended-maximum-permissions-for-files-folders-in-various-scenarios) for file permission references.

## Build Security Statement

- When compiling and installing this project from source, the compilation process generates intermediate files. After compilation, control the permissions of these intermediate files to ensure file security.

## Runtime Security Statement

- When a runtime exception occurs, the process exits and prints error information. Locate the specific error cause based on the error message.

## Public URL Statement

The public URLs included in the code of this project are declared as follows:

| Type | Open Source Code Address | File Name | Public IP Address or Public URL or Domain or Email or Archive Address | Purpose |
| :--: | :----------: | :----- | :-------------------------------------------------------------- | :-------------------------------------------- |
| Dependency | Not Applicable | cmake/third_party/makeself-fetch.cmake | https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz | Download makeself source code from gitcode as a build dependency |
| Dependency | Not Applicable | cmake/third_party/json.cmake | https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip | Download json source code from gitcode as a build dependency |
| Dependency | Not Applicable | cmake/third_party/openssl.cmake | https://gitcode.com/cann-src-third-party/openssl/releases/download/openssl-3.0.9/openssl-openssl-3.0.9.tar.gz | Download openssl source code from gitcode as a build dependency |
| Dependency | Not Applicable | cmake/third_party/gtest.cmake | https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz | Download googletest source code from gitcode as a build dependency |
| Dependency | Not Applicable | cmake/third_party/mockcpp.cmake | https://gitcode.com/cann-src-third-party/mockcpp/releases/download/v2.7-h4/mockcpp-2.7.tar.gz | Download mockcpp source code from gitcode as a build dependency |
| Dependency | Not Applicable | cmake/third_party/protobuf.cmake | https://gitcode.com/cann-src-third-party/protobuf/releases/download/v25.1/protobuf-25.1.tar.gz | Download protobuf source code from gitcode as a build dependency |
| Dependency | Not Applicable | rdma-core | https://gitcode.com/cann-src-third-party/rdma-core/releases/download/v42.7-h1/rdma-core-42.7.tar.gz | Download rdma-core source code from gitcode as a build dependency |
| Dependency | Not Applicable | rdma-core-patch | https://gitcode.com/cann-src-third-party/rdma-core/releases/download/v42.7-h1/rdma-core-42.7.patch | Download rdma-core-patch source code from gitcode as a build dependency |

---

## Port Statement

For information about the ports opened by HCCL, the transport layer protocols used, authentication methods, and purposes, refer to the HCCL tab in the [CANN Communication Matrix](https://hiascend.com/document/redirect/CannCommunityCommMatrix).

## Vulnerability Mechanism

[Vulnerability Management](https://gitcode.com/cann/community/blob/master/security/security.md)

## Appendix

### A - Recommended Maximum Permissions for Files (Folders) in Various Scenarios

| Type | Linux Recommended Maximum Permission |
| ---------------------------------- | -------------------- |
| User home directory | 750 (rwxr-x---) |
| Program files (including scripts, library files, and so on) | 550 (r-xr-x---) |
| Program file directory | 550 (r-xr-x---) |
| Configuration files | 640 (rw-r-----) |
| Configuration file directory | 750 (rwxr-x---) |
| Log files (completed or archived) | 440 (r--r-----) |
| Log files (being recorded) | 640 (rw-r-----) |
| Log file directory | 750 (rwxr-x---) |
| Debug files | 640 (rw-r-----) |
| Debug file directory | 750 (rwxr-x---) |
| Temporary file directory | 750 (rwxr-x---) |
| Maintenance and upgrade file directory | 770 (rwxrwx---) |
| Business data files | 640 (rw-r-----) |
| Business data file directory | 750 (rwxr-x---) |
| Key components, private keys, certificates, encrypted file directory | 700 (rwx------) |
| Key components, private keys, certificates, encrypted ciphertext | 600 (rw-------) |
| Encryption and decryption interfaces, encryption and decryption scripts | 500 (r-x------) |
