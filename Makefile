MAKEFLAGS += -j$(nproc)
# Compiler
NVCC = nvcc

# Compiler flags
# -Xcompiler -fopenmp for OpenMP apps
NVCCFLAGS = -O0 -g -arch=sm_80 -lnuma -I$(HOME)/local/include -L$(HOME)/local/lib 

# Directories
SRCDIR = src
OBJDIR = bin

BFS_SRCDIR = $(SRCDIR)/bfs
CC_SRCDIR = $(SRCDIR)/cc
SSSP_SRCDIR = $(SRCDIR)/sssp
PR_SRCDIR = $(SRCDIR)/pr

# Source files
MAIN_SRC = $(SRCDIR)/main.cu
GRAPH_SRC = $(SRCDIR)/graph.cuh
COMMON_SRC = $(SRCDIR)/common.cuh
TIMER_SRC = $(SRCDIR)/timer.hpp

# BFS 
BFS_SRC = $(SRCDIR)/bfs/bfs.cu
BFS_KERNELS_SRC = $(SRCDIR)/bfs/bfs_kernels.cu

# CC 
CC_SRC = $(SRCDIR)/cc/cc.cu
CC_KERNELS_SRC = $(SRCDIR)/cc/cc_kernels.cu

# SSSP 
SSSP_SRC = $(SRCDIR)/sssp/sssp.cu
SSSP_KERNELS_SRC = $(SRCDIR)/sssp/sssp_kernels.cu

# PR 
PR_SRC = $(SRCDIR)/pr/pr.cu
PR_KERNELS_SRC = $(SRCDIR)/pr/pr_kernels.cu

# Object files
MAIN_OBJ = $(OBJDIR)/main.o
# GRAPH_OBJ = $(OBJDIR)/graph.o
# COMMON_OBJ = $(OBJDIR)/common.o
# TIMER_OBJ = $(OBJDIR)/timer.o

# BFS
BFS_OBJ = $(OBJDIR)/bfs.o
BFS_KERNELS_OBJ = $(OBJDIR)/bfs_kernels.o

# CC
CC_OBJ = $(OBJDIR)/cc.o
CC_KERNELS_OBJ = $(OBJDIR)/cc_kernels.o

# SSSP
SSSP_OBJ = $(OBJDIR)/sssp.o
SSSP_KERNELS_OBJ = $(OBJDIR)/sssp_kernels.o

# PR
PR_OBJ = $(OBJDIR)/pr.o
PR_KERNELS_OBJ = $(OBJDIR)/pr_kernels.o

# Targets
all: main

main: $(MAIN_OBJ) $(BFS_OBJ) $(BFS_KERNELS_OBJ) $(CC_OBJ) $(CC_KERNELS_OBJ) $(SSSP_OBJ) $(SSSP_KERNELS_OBJ) $(PR_OBJ) $(PR_KERNELS_OBJ)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(OBJDIR)/main.o: $(MAIN_SRC)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# BFS
$(OBJDIR)/bfs.o: $(BFS_SRC) $(COMMON_SRC) $(GRAPH_SRC) $(TIMER_SRC)
	$(NVCC) $(NVCCFLAGS) -I$(BFS_SRCDIR) -c $< -o $@

$(OBJDIR)/bfs_kernels.o: $(BFS_KERNELS_SRC) $(COMMON_SRC)
	$(NVCC) $(NVCCFLAGS) -I$(BFS_SRCDIR) -c $< -o $@

# CC
$(OBJDIR)/cc.o: $(CC_SRC) $(COMMON_SRC) $(GRAPH_SRC) $(TIMER_SRC)
	$(NVCC) $(NVCCFLAGS) -I$(CC_SRCDIR) -c $< -o $@

$(OBJDIR)/cc_kernels.o: $(CC_KERNELS_SRC) $(COMMON_SRC)
	$(NVCC) $(NVCCFLAGS) -I$(CC_SRCDIR) -c $< -o $@

# SSSP
$(OBJDIR)/sssp.o: $(SSSP_SRC) $(COMMON_SRC) $(GRAPH_SRC) $(TIMER_SRC)
	$(NVCC) $(NVCCFLAGS) -I$(SSSP_SRCDIR) -c $< -o $@

$(OBJDIR)/sssp_kernels.o: $(SSSP_KERNELS_SRC) $(COMMON_SRC)
	$(NVCC) $(NVCCFLAGS) -I$(SSSP_SRCDIR) -c $< -o $@

# PR
$(OBJDIR)/pr.o: $(PR_SRC) $(COMMON_SRC) $(GRAPH_SRC) $(TIMER_SRC)
	$(NVCC) $(NVCCFLAGS) -I$(PR_SRCDIR) -c $< -o $@

$(OBJDIR)/pr_kernels.o: $(PR_KERNELS_SRC) $(COMMON_SRC)
	$(NVCC) $(NVCCFLAGS) -I$(PR_SRCDIR) -c $< -o $@

clean:
	rm -f $(OBJDIR)/*.o main
