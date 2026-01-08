CXX := nvcc
CPPFLAGS := 
CXXFLAGS := -O3
LDFLAGS := 
LDLIBS :=
GCOV_CMD := gcov

EXE_DIR := bin/
EXECUTABLE := $(EXE_DIR)cudart
SRC_FILES := $(wildcard src/render.cu)
OBJ_DIR := obj/
OBJ_FILES := $(addprefix $(OBJ_DIR),$(notdir $(SRC_FILES:.cu=.o)))
GCOV_FILES := $(notdir $(addsuffix .gcov,$(SRC_FILES)))
GCDA_FILES := $(wildcard $(OBJ_DIR)/*.gcda)
SRC_PREFIX := src/
SRC_DIRS := $(dir $(SRC_FILES))
VPATH := $(SRC_DIRS)

.PHONY : all dirs clean

all : dirs $(EXECUTABLE)

objs : dirs $(OBJ_FILES)

dirs : $(EXE_DIR) $(OBJ_DIR)

# Placing gcov target in the Makefile in order to easily collect all SRC_FILES w/ correct paths

gcov : dirs $(GCOV_FILES)

# For debugging variables in Makefile, e.g. by "make print-GCOV_FILES"

print-%  : ; @echo $* = $($*)

$(EXE_DIR):
	mkdir -p $(EXE_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Link objects into executable

$(EXECUTABLE) : $(OBJ_FILES)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $(OBJ_FILES) $(LDFLAGS) $(LDLIBS)

# Create objects from source files

$(OBJ_DIR)%.o : %.cu
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

# Process .gcno and .gcda files from obj/ into .cpp.gcov files (and .hpp.gcov, .h.gcov) in root directory
# Rerun Gcov on all files if a single .gcda changes. Other options to consider: --preserve-paths -abcu
./%.cpp.gcov : %.cu $(OBJ_DIR)/%.gcno $(GCDA_FILES)
	$(GCOV_CMD)  --relative-only --source-prefix=$(SRC_PREFIX) --object-directory=$(OBJ_DIR) $<

# Cleanup

clean :
	rm -rf $(OBJ_DIR)*
	rm -rf $(EXECUTABLE)
	rm -rf *.gcov