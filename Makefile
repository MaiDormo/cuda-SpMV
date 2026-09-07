# Compiler configuration
CC := gcc
NVCC := nvcc

# Build type configuration (default to release)
BUILD_TYPE ?= release

# Local build configuration (Set LOCAL=1 for local build)
LOCAL ?= 0

# Base compiler flags
BASE_OPT := -std=c11 -Wall -Wextra -lm
BASE_NV_OPT := -m64 -Xcompiler -fopenmp

# Debug flags
DEBUG_OPT := $(BASE_OPT) -g -O0 -DDEBUG -fsanitize=address -fsanitize=undefined
DEBUG_NV_OPT := $(BASE_NV_OPT) -g -G -O0 -DDEBUG --device-debug

# Set GPU architecture and LTO flags based on build environment
ifeq ($(LOCAL),1)
    # disable LTO for CUDA compatibility
	RELEASE_OPT := $(BASE_OPT) -O3 -DNDEBUG -march=native -funroll-loops
	PROFILE_OPT := $(BASE_OPT) -O3 -g -DNDEBUG -march=native -funroll-loops
else
	# enabled LTO for CUDA compatibility
	RELEASE_OPT := $(BASE_OPT) -O3 -DNDEBUG -march=native -funroll-loops -flto
	PROFILE_OPT := $(BASE_OPT) -O3 -g -DNDEBUG -march=native -funroll-loops
endif

# Release flags with performance optimizations
RELEASE_NV_OPT := $(BASE_NV_OPT) -O3 --gpu-architecture=sm_80 -DNDEBUG --use_fast_math -Xptxas -O3 \
					--maxrregcount=255 \
					--ptxas-options=--warn-on-spills,--optimize-float-atomics \
					-Xcompiler "-Ofast -march=native -mtune=native -funroll-loops -ffast-math -fopenmp -DNDEBUG" \
					-Xptxas -O3,--def-load-cache=ca,--def-store-cache=wb \
					--relocatable-device-code=false \
					--restrict \
					--fmad=true

# Profile flags for performance analysis
PROFILE_OPT := $(BASE_OPT) -O3 -g -DNDEBUG -march=native -funroll-loops
PROFILE_NV_OPT := $(BASE_NV_OPT) -O3 --gpu-architecture=sm_80 -DNDEBUG -lineinfo -g \
					--maxrregcount=255 \
					--ptxas-options=--warn-on-spills,--optimize-float-atomics \
					-Xcompiler "-Ofast -march=native -mtune=native -funroll-loops -ffast-math -fopenmp" \
					-Xptxas -O3,--def-load-cache=ca,--def-store-cache=wb \
					--relocatable-device-code=false \
					--restrict \
					--fmad=true

# Set flags based on build type
ifeq ($(BUILD_TYPE),debug)
	OPT := $(DEBUG_OPT)
	NV_OPT := $(DEBUG_NV_OPT)
	BUILD_SUFFIX := _debug
else ifeq ($(BUILD_TYPE),profile)
	OPT := $(PROFILE_OPT)
	NV_OPT := $(PROFILE_NV_OPT)
	BUILD_SUFFIX := _profile
else
	OPT := $(RELEASE_OPT)
	NV_OPT := $(RELEASE_NV_OPT)
	BUILD_SUFFIX :=
endif

# Directory configuration with build type suffix
BIN_FOLDER := bin$(BUILD_SUFFIX)
OBJ_FOLDER := obj$(BUILD_SUFFIX)
SRC_FOLDER := src
LIB_FOLDER := lib
INCLUDE_FOLDER := include

# Find all .c files in the lib directory
LIB_SOURCES := $(wildcard $(LIB_FOLDER)/*.c)
# Create corresponding object file paths
LIB_OBJECTS := $(patsubst $(LIB_FOLDER)/%.c, $(OBJ_FOLDER)/%.o, $(LIB_SOURCES))

# Get all of the main c files
MAIN_SOURCES := $(wildcard $(SRC_FOLDER)/*.c)
# Get all of the main cu files
MAIN_CUDA_SOURCES := $(wildcard $(SRC_FOLDER)/*.cu)
# Create corresponding binary file paths
EXECUTABLES := $(patsubst $(SRC_FOLDER)/%.c, $(BIN_FOLDER)/%, $(MAIN_SOURCES))
# Create it also for cuda
CUDA_EXECUTABLES := $(patsubst $(SRC_FOLDER)/%.cu, $(BIN_FOLDER)/%.exec, $(MAIN_CUDA_SOURCES))

# Default target
all: $(EXECUTABLES) $(CUDA_EXECUTABLES)

# Build type targets
debug:
	$(MAKE) BUILD_TYPE=debug all

release:
	$(MAKE) BUILD_TYPE=release all

profile:
	$(MAKE) BUILD_TYPE=profile all

# Local build targets
local:
	$(MAKE) LOCAL=1 all

local-debug:
	$(MAKE) LOCAL=1 BUILD_TYPE=debug all

local-release:
	$(MAKE) LOCAL=1 BUILD_TYPE=release all

local-profile:
	$(MAKE) LOCAL=1 BUILD_TYPE=profile all

# Performance optimization target with additional flags
perf: 
	$(MAKE) BUILD_TYPE=release RELEASE_OPT="$(RELEASE_OPT) -fprofile-use=profile.profdata" all

# Rule for creating executables
$(BIN_FOLDER)/%: $(SRC_FOLDER)/%.c $(LIB_OBJECTS)
	@mkdir -p $(BIN_FOLDER)
	@echo "Building $@ with $(BUILD_TYPE) configuration..."
	$(CC) $< $(LIB_OBJECTS) -o $@ $(OPT) -fopenmp

# Rule for creating object files
$(OBJ_FOLDER)/%.o: $(LIB_FOLDER)/%.c
	@mkdir -p $(OBJ_FOLDER)
	@echo "Compiling $< with $(BUILD_TYPE) configuration..."
	$(CC) -c $< -o $@ $(OPT) -fopenmp

# Rule for compiling cuda
# Rule for compiling cuda with environment-specific commands
$(BIN_FOLDER)/%.exec: $(SRC_FOLDER)/%.cu $(LIB_OBJECTS)
	@mkdir -p $(BIN_FOLDER)
	@echo "Building CUDA $@ with $(BUILD_TYPE) configuration (GPU arch: sm80)..."
ifeq ($(LOCAL),1)
	$(NVCC) $< $(LIB_FOLDER)/spmv_kernels.cu $(LIB_OBJECTS) -o $@ $(NV_OPT)
else
	@bash -c "source /etc/profile.d/modules.sh && module load CUDA/12.5.0 && $(NVCC) $< $(LIB_FOLDER)/spmv_kernels.cu $(LIB_OBJECTS) -o $@ $(NV_OPT)"
endif

# Profile-guided optimization (PGO) support
pgo-generate:
	$(MAKE) BUILD_TYPE=release RELEASE_OPT="$(RELEASE_OPT) -fprofile-generate" all

pgo-use:
	$(MAKE) BUILD_TYPE=release RELEASE_OPT="$(RELEASE_OPT) -fprofile-use" all

# Benchmark target for performance testing
benchmark: release
	@echo "Running performance benchmarks..."
	./run_all_benchmarks.sh

# Host-only unit test for the hybrid v2 row plan (no GPU needed).
# Extra .mtx files can be passed with: make test-plan MTX="a.mtx b.mtx"
test-plan:
	@mkdir -p $(BIN_FOLDER)
	$(CC) test/test_hybrid_v2_plan.c $(LIB_FOLDER)/hybrid_v2_plan.c \
		$(LIB_FOLDER)/read_file_lib.c $(LIB_FOLDER)/coo_to_csr.c \
		-o $(BIN_FOLDER)/test_hybrid_v2_plan $(OPT) -fopenmp
	./$(BIN_FOLDER)/test_hybrid_v2_plan $(MTX)

# Create necessary directories
directories:
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER)

# Clean targets
clean:
	rm -rf bin bin_debug bin_profile
	rm -rf obj obj_debug obj_profile

clean-debug:
	rm -rf bin_debug obj_debug

clean-profile:
	rm -rf bin_profile obj_profile

clean-release:
	rm -rf bin obj

# Help target
help:
	@echo "Available targets:"
	@echo "  all (default) - Build with current BUILD_TYPE (default: release)"
	@echo "  debug         - Build with debug flags (-g -O0)"
	@echo "  release       - Build with optimization flags (-O3 -march=native)"
	@echo "  profile       - Build with profiling support"
	@echo "  local         - Build locally on Arch Linux (no module loading)"
	@echo "  local-debug   - Build locally with debug flags"
	@echo "  local-release - Build locally with release flags"
	@echo "  local-profile - Build locally with profile flags"
	@echo "  perf          - Build with maximum performance optimizations"
	@echo "  pgo-generate  - Build with profile generation for PGO"
	@echo "  pgo-use       - Build using generated profiles for PGO"
	@echo "  benchmark     - Build and run performance benchmarks"
	@echo "  clean         - Clean all build artifacts"
	@echo "  clean-debug   - Clean debug build artifacts"
	@echo "  clean-profile - Clean profile build artifacts"
	@echo "  clean-release - Clean release build artifacts"
	@echo "  help          - Show this help message"
	@echo ""
	@echo "Build types can be set with BUILD_TYPE variable:"
	@echo "  make BUILD_TYPE=debug"
	@echo "  make BUILD_TYPE=release"
	@echo "  make BUILD_TYPE=profile"

# Declare phony targets
.PHONY: all debug release profile local local-debug local-release local-profile perf pgo-generate pgo-use benchmark test-plan directories clean clean-debug clean-profile clean-release install help