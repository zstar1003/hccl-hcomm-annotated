# EI0008 Package_Error_Incorrect_HCCL_Version

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: module name, local version, remote version.

```text
The %s versions are inconsistent. The local %s, while the remote %s.
```

Error example 1:

```text
The ops (cann-hccl) versions are inconsistent. The local version is 9.2.0 (version file: /usr/local/Ascend/cann-9.2.0/share/info/hccl/version.info), while the remote version is 9.1.0.
```

Error example 2:

```text
The Toolkit (cann-hcomm) and ops (cann-hccl) versions are inconsistent. The local Toolkit (cann-hcomm) version is 9.1.0 and ops (cann-hccl) version is 9.1.0 (version file: /usr/local/Ascend/cann-9.2.0/share/info/hccl/version.info), while the remote Toolkit (cann-hcomm) version is 9.2.0 and ops (cann-hccl) version is 9.2.0.
```

## Solution

Install the same version. The local Toolkit (cann-hcomm) and ops (cann-hccl) versions must be consistent with the corresponding remote versions.
