# RFC Document Directory

This directory contains technical design documents (RFC - Request for Comments) for the HCOMM repository. These documents are used to align on solutions and document design decisions before code implementation.

## Directory Structure

- `0000-template.md` - RFC writing template
- `INDEX.md` - RFC number registry, a complete list of all assigned RFC numbers
- `NNNN-xxx-xxx.md` - RFC documents (4-digit number plus a brief description)

## Naming Convention

RFC file naming format: `{4-digit-number}-{brief-description}.md`

For example: `0001-add-new-feature.md`

- Number: 4-digit zero-padded (0001 to 9999)
- Description: English lowercase, hyphen-separated, concise

## Numbering Mechanism (Core)

1. **Number reservation PR** (lightweight): Modify only [INDEX.md](./INDEX.md) to append a reservation row
2. **RFC document PR** (heavyweight): Write the RFC document and submit it for review. The number is already locked through the number reservation PR.

**Number reservation rules**:

- Sequential allocation, prioritize the **smallest unused number**
- Never reused. Merged numbers are not recycled even if the RFC is subsequently replaced
- For details, see [INDEX.md](./INDEX.md)

## RFC Lifecycle

1. **Requirement phase**: Submit the requirement as a Requirement type Issue and wait for the SIG group to accept it
2. **Number reservation PR**: Append a reservation row in [INDEX.md](./INDEX.md) (status `reserved`) and submit a number reservation PR
3. **Number reservation PR merged**: The number is occupied. You can start writing the RFC.
4. **Writing phase**: Write the system solution following the [RFC template](./0000-template.md)
5. **Review phase**: The RFC document PR is reviewed. Modify the solution based on feedback during the process.
6. **Decision phase**:
   - **Merge**: The Maintainer approves the review. Add `/lgtm` and `/approve` to merge. Update the status in INDEX.md to `accepted`.
   - **Close**: The review is not approved. Close the PR. The number remains `reserved` and is not recycled (the author can restart the review process).
7. **Implementation phase**: The merged RFC serves as the implementation contract. Code PRs must follow the RFC solution.

## Replacement Relationship

When the implementation of an RFC is replaced by a subsequent RFC:

- Add the following note at the end of the replaced RFC document: `> Superseded by 00NN`
- Update the status of the corresponding row in [INDEX.md](./INDEX.md) from `accepted` to `superseded`
- Do not modify the original number

## Related Links

- [Contribution Guide](../../../CONTRIBUTING_en.md)
- [RFC Template](./0000-template.md)
- [RFC Number Registry](./INDEX.md)
- [SIG Meeting](https://etherpad-cann.meeting.osinfra.cn/p/sig-hccl)
