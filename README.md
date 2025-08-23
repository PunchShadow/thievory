# Thievory

Thievory is a GPU-accelerated out-of-memory graph processing framework.

## Table of Contents

## Installation

### Prerequisites

In order to compile the code you must install:

- A C++ compiler
- NVIDIA CUDA toolkit
- NUMA (Non-Uniform Memory Access) library

### Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/qqueke/Thievory.git
   cd Thievory
   mkdir results
   ```

   Directory results must be in place so that results are dumped to results/values.bin

2. Update Makefile

   Replace -arch_sm=80 with your GPU architecture
   Replace $(HOME)/local/include and $(HOME)/local/lib with the appropriate paths to your NUMA installation

3. Compile the code
   ```bash
     make
   ```

## Inputs

### Thievory Formats

### Conversion

Thievory comes with a converter from .txt files to Thievory, EMOGI, Subway, HyTGraph, and Liberator input formats

1. Compile the converter
   ```bash
   cd converter
   g++ converter.cpp -o convert
   ```

2. Find out how many lines to skip

   Some .txt graphs come with lines of text that provide description to the graph input, instead of edges. Figure out how many are there in your .txt
   
3. Convert the graph to the desired format by specifying the input location, number of lines to skip, and the format

   Example:
   ```bash
   ./convert --input path/to/graph.txt --skip 4 --type csr
   ```
   
- '--type': Specifies the type of the conversion. Options are:
   - 'csr': Used by Thievory, Ascetic and Liberator
   - 'csc': Used by Thievory on Pull-based PageRank
   - 'wcsr': Used by Ascetic and Liberator on SSSP
   - 'bel': Used by EMOGI
   - 'el': Used by Subway
   - 'wel': Used by Subway on SSSP
   - 'hwel: Used by HyTGraph


## Usage

To run the framework use:
   
   ```bash
   ./main --input <path_to_csr/csc> [options]
   ```

### Options
1. '--input': Specifies the path to the input

2. '--algo': Specify the algorithm to use:
   - 'bfs': Breadth-First Search
   - 'cc': Connected Components
   - 'pr': PageRank
   - 'sssp': Single Source Shortest Path
   - Default: 'bfs'

3. '--edgeSize': Specify the edge size to use:
   - '4': 4 Bytes edge
   - '8': 8 Bytes edge
   - Default: '4'

3. '--type': Specify the PageRank type to use:
   - 'push': Push-based PageRank
   - 'pull': Pull-based PageRank
   - Default: 'push'

4. '--source': Specify the source vertex for BFS and SSSP
   - Default: 1

5. '--gpus': Specify the number of NEIGHBOR GPUs to use
   - Default: 0

6. '--runs': Specify the number of runs to perform
   - Default: 1
