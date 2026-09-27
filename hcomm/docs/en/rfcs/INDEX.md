# RFC Number Registry

This file registers all assigned RFC numbers. Before adding a new RFC, claim the **smallest unused number** from this table.

## Number Reservation Process (Independent of RFC Document Creation)

1. Check the Allocated Numbers table to find the smallest unused N (typically the last row number plus 1)
2. Append a row for N in this table (fill in the status as `reserved`; the title and author can be placeholders)
3. Submit a **number reservation PR** (containing only the one-line update to this INDEX.md)
4. After the number reservation PR is merged, number N is occupied (status remains `reserved`), and you can start writing the RFC document
5. The RFC document `NNNN-xxx-xxx.md` is submitted in a subsequent independent PR
6. After the RFC document PR is merged, the status changes to `accepted`, and the RFC officially takes effect

## Status Description

| Status | Meaning |
|------|------|
| `reserved` | Number is reserved (number reservation PR merged), RFC document pending submission or review |
| `accepted` | RFC document PR merged, RFC officially takes effect |
| `superseded` | Replaced by a subsequent RFC. See the `Superseded by` note at the end of the original document. |

## Allocated Numbers

| Number | Title | Author | Status | PR |
|------|------|------|------|-----|
|0001|RFC: Topology-Based Cluster Communication Rapid Sub-Health Monitoring | Wang Yunhao (Xidian University)|`accepted`|#2491|

## Numbering Rules

- **Format**: 4-digit zero-padded (0001 to 9999)
- **Never reused**: Merged numbers are not recycled even if the RFC is subsequently replaced
- **Sequential allocation**: In principle, do not skip numbers. The next number is the largest used number plus 1
- **Replacement relationship**: Add `> Superseded by 00NN` at the end of the original RFC document and update the status column in this table to `superseded`
- **Conflict resolution**: If two people claim the same number simultaneously, the latter must rebase and change to the new smallest number

## Related

- [RFC Template](./0000-template.md)
- [RFC Process Description](./README.md)
