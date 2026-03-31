# Initial Commit (3a10344) — uint32 Truncation in 64-bit Paths

## Summary

At commit `3a10344b2f` (Initial commit), the 64-bit code paths (`--edgeSize 8`, `EdgeType=uint64`) contain
pervasive `uint32` truncation of `uint64` values. The 64-bit kernel functions (e.g. `BFS64_Filter_Kernel`)
were copy-pasted from their 32-bit counterparts without updating internal variable types. The Static and
Demand kernels were correctly written with `uint64` types, but **all Filter / NeighborFilter / StaticFilter
kernels** retained `uint32` locals.

When edge counts or vertex IDs exceed 2^32, the upper bits are silently discarded, causing wrong pointer
offsets, wrong `cudaMemcpy` sizes, incorrect edge indexing, and corrupted algorithm results.

---

## Category 1: `graph.cuh` — File Reading & Device Allocation

### 1a. `sizeof(uint32)` hardcoded in edge reading (line 754)

```cpp
infile.read((char *)h_edges2[GPUAffinityMap[0]], numEdges * sizeof(uint32));
```

At the initial commit the only supported format was the native CSR (`.csr`) format, which stores 32-bit
edges. The read always used `sizeof(uint32)` regardless of `EdgeType`. When `EdgeType=uint64`, only half
the buffer is filled, and the data is misinterpreted as 64-bit values (pairs of adjacent 32-bit values
merged).

**Fixed in:** commit `f747663` (bcsr64/bwcsr64 support). The native format read still uses `sizeof(uint32)`,
which is correct for its 32-bit file structure.

### 1b. `sizeof(*d_filterEdges)` uses pointer size (lines ~452-465)

```cpp
cudaMalloc(&d_filterEdges[i], maxEdgesInPartition * sizeof(*d_filterEdges));
```

`d_filterEdges` is `EdgeType *d_filterEdges[N]` (array of pointers). `sizeof(*d_filterEdges)` evaluates to
`sizeof(EdgeType*)` = 8, not `sizeof(EdgeType)`. For `uint32` this allocates 2x memory; for `uint64` it is
coincidentally correct.

**Fixed in:** current HEAD (`sizeof(**d_filterEdges)`).

---

## Category 2: Host-Side Algorithm Functions — Offset Truncation

In every 64-bit function (BFS64, SSSP64, CC64, PR64, PR64_PUSH), local variables that receive values from
`h_offsets` (which is always `uint64*`) are declared as `uint32`:

| Variable | Usage | Impact |
|----------|-------|--------|
| `uint32 start` | Array index into `h_edges2[...][start]` | Wrong pointer offset → wrong data copied |
| `uint32 partitionSize` | `cudaMemcpyAsync(..., partitionSize * sizeof(...))` | Wrong copy size |
| `uint32 neighborStart` | Array index for neighbor GPU partition | Wrong pointer offset |
| `uint32 neighborPartitionSize` | `cudaMemcpyAsync` size for neighbor GPU | Wrong copy size |
| `uint32 partitionStart` | Start offset for static filter processing | Wrong kernel parameter |
| `uint32 partitionEnd` | End offset for static filter processing | Wrong kernel parameter |
| `uint32 processedPartitionSize` | `partitionEnd - partitionStart` | Wrong value |

Each 64-bit function has ~14 truncation points (2 for filter, 2 for neighbor filter, 12 for static filter
sections). Five functions total → **~70 truncation points**.

### Locations per function at commit 3a10344:

**BFS64** (`bfs.cu`, starts line 532):
- `start`/`partitionSize`: lines 703, 706
- `neighborStart`/`neighborPartitionSize`: lines 738, 741
- `partitionStart`/`End`/`processedPartitionSize`: lines 805-813, 858-865, 925-931, 976-984

**SSSP64** (`sssp.cu`, starts line 559):
- `start`/`partitionSize`: lines 724, 727
- `neighborStart`/`neighborPartitionSize`: lines 764, 767
- `partitionStart`/`End`/`processedPartitionSize`: lines 838-846, 898-906, 968-975, 1025-1033

**CC64** (`cc.cu`, starts line 616):
- `start`/`partitionSize`: lines 815, 818
- `neighborStart`/`neighborPartitionSize`: lines 850, 853
- `partitionStart`/`End`/`processedPartitionSize`: lines 906-914, 966-974, 1036-1044, 1093-1101

**PR64** (`pr.cu`, starts line 521):
- `start`/`partitionSize`: lines 675, 678
- `neighborStart`/`neighborPartitionSize`: lines 711, 714
- `partitionStart`/`End`/`processedPartitionSize`: lines 762-770, 820-828, 890-898, 947-955

**PR64_PUSH** (`pr.cu`, starts line 1551):
- `start`/`partitionSize`: lines 1187, 1190
- `neighborStart`/`neighborPartitionSize`: lines 1222, 1225
- `partitionStart`/`End`/`processedPartitionSize`: lines 1273-1281, 1331-1339, 1401-1409, 1458-1466

**Partial fix applied:** `start` and `partitionSize` were fixed in the previous round. The remaining
`neighborStart`, `neighborPartitionSize`, `partitionStart`, `partitionEnd`, and `processedPartitionSize`
variables are fixed in the current HEAD.

---

## Category 3: Kernel-Side — Variable Type Truncation

### 3a. `uint32 startEdge = d_offsets[...]` (offset truncation)

All 64-bit Filter and NeighborFilter kernels store `uint64` offset values from `d_offsets` into `uint32`
local variables. Used to compute relative edge positions within a partition.

| File | Kernel | Line |
|------|--------|------|
| bfs_kernels.cu | BFS64_Filter_Kernel | 210 |
| bfs_kernels.cu | BFS64_NeighborFilter_Kernel | 392 |
| sssp_kernels.cu | SSSP64_Filter_Kernel | 207 |
| sssp_kernels.cu | SSSP64_NeighborFilter_Kernel | 290 |
| cc_kernels.cu | CC64_Filter_Kernel | 290 |
| cc_kernels.cu | CC64_NeighborFilter_Kernel | 436 |
| pr_kernels.cu | PR64_Filter_Kernel | 176 |
| pr_kernels.cu | PR64_NeighborFilter_Kernel | 250 |
| pr_kernels.cu | PR64_Filter_Kernel_PUSH | 695 |
| pr_kernels.cu | PR64_NeighborFilter_Kernel_PUSH | 826 |

**Fixed in:** current HEAD (previous round of fixes).

### 3b. `uint32 neighborId = d_filterEdges[i]` (vertex ID truncation)

When `EdgeType=uint64`, edge arrays (`d_filterEdges`, `d_staticEdges`, `h_edges`) contain `uint64` vertex
IDs, but the receiving variable is `uint32`. For graphs with more than 2^32 vertices, vertex IDs are
silently truncated, causing accesses to wrong `d_values[]` entries.

| File | Kernel | Line (at 3a10344) |
|------|--------|-------------------|
| bfs_kernels.cu | BFS64_Filter_Kernel | 237 |
| bfs_kernels.cu | BFS64_NeighborFilter_Kernel | 418 |
| sssp_kernels.cu | SSSP64_Filter_Kernel | 221 |
| sssp_kernels.cu | SSSP64_NeighborFilter_Kernel | 316 |
| sssp_kernels.cu | SSSP64_Static_Filter_Kernel | 453 |
| cc_kernels.cu | CC64_Filter_Kernel | 317 |
| cc_kernels.cu | CC64_NeighborFilter_Kernel | 462 |
| cc_kernels.cu | CC64_Static_Filter_Kernel | 572 |
| pr_kernels.cu | PR64_Filter_Kernel | 190 |
| pr_kernels.cu | PR64_NeighborFilter_Kernel | 264 |
| pr_kernels.cu | PR64_Static_NeighborFilter_Kernel | (varies) |
| pr_kernels.cu | PR64_Demand_Kernel_PUSH | 642 |
| pr_kernels.cu | PR64_Filter_Kernel_PUSH | 709 |
| pr_kernels.cu | PR64_Static_Filter_Kernel_PUSH | (varies) |
| pr_kernels.cu | PR64_NeighborFilter_Kernel_PUSH | 840 |
| pr_kernels.cu | PR64_Static_Kernel_PUSH | (varies) |

**Fixed in:** current HEAD.

### 3c. `uint32 sourceValue = d_values[warpIdx]` (algorithm value truncation)

In SSSP and CC, the 64-bit filter kernels read `d_values` (which is `uint64*`) into `uint32` locals. This
truncates distance values (SSSP) or component IDs (CC).

| File | Kernel | Line (at 3a10344) |
|------|--------|-------------------|
| sssp_kernels.cu | SSSP64_Filter_Kernel | 215 |
| sssp_kernels.cu | SSSP64_NeighborFilter_Kernel | 301 |
| sssp_kernels.cu | SSSP64_Static_Filter_Kernel | (varies) |
| cc_kernels.cu | CC64_Filter_Kernel | 300 |
| cc_kernels.cu | CC64_NeighborFilter_Kernel | 437 |
| cc_kernels.cu | CC64_Static_Filter_Kernel | (varies) |

**Fixed in:** current HEAD.

### 3d. `uint32 newValue = d_values[warpIdx] + 1` (BFS level truncation)

In BFS, the 64-bit filter kernels compute the new BFS level from `d_values` (uint64*) but store in `uint32`.
Practically harmless since BFS levels rarely exceed 2^32, but inconsistent.

| File | Kernel | Line (at 3a10344) |
|------|--------|-------------------|
| bfs_kernels.cu | BFS64_Filter_Kernel | 231 |
| bfs_kernels.cu | BFS64_NeighborFilter_Kernel | 412 |

**Fixed in:** current HEAD.

---

## Root Cause

The 64-bit kernel variants (`*64_Filter_Kernel`, `*64_NeighborFilter_Kernel`, `*64_Static_Filter_Kernel`)
were duplicated from the 32-bit versions, and the parameter types in the function signature were updated
to `uint64`, but **internal local variable declarations were left as `uint32`**.

The Static and Demand kernels were written correctly with `uint64` internal types.

---

## Note on 32-bit Kernels

The 32-bit filter kernels have the same `uint32 startEdge = d_offsets[...]` pattern, and since `d_offsets`
is always `uint64*` regardless of `EdgeType`, this is technically a truncation too. However, it only
triggers for graphs with >2^32 total edges, which is uncommon in the 32-bit-edge workflow. This is not
fixed and may be addressed in a follow-up.
