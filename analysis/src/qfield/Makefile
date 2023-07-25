
#CC := gcc
CC := $(CC)
CFLAGS := -O3 -Wall -Wshadow -std=c11 -fopenmp

#CXX := g++
CXX := $(CXX)
CXXFLAGS := -O3 -Wall -Wshadow -std=c++11 -fopenmp

PRG_SUFFIX_FLAG := 0

LDIR := ../../lib
LODIR = ../../bin/qfield/lib
ODIR := ../../bin/qfield/obj
EDIR := ../../bin/qfield/exe

#CONDA_DIR := /home/s1309877/.local/bin/miniconda3/envs/myUtil/
CONDA_DIR := /home/cchiang2/programs/miniconda3/envs/myUtil/
#CONDA_DIR := /storage/cmstore02/users/cchiang2/conda/miniconda3/envs/pfield/
INC = -I$(CONDA_DIR)/include/ -I$(LDIR)
#INC = -I$(LDIR)
LDFLAGS := -L$(CONDA_DIR)/lib/ -lm -lgomp -larmadillo -lgsl
#LDFLAGS := -lm -lgomp -larmadillo -lgsl

LSRCC := $(wildcard $(LDIR)/*.c)
LOBJC := $(patsubst %.c, $(LODIR)/%.o, $(notdir $(LSRCC)))
LSRCCXX := $(wildcard $(LDIR)/*.cpp)
LOBJCXX := $(patsubst %.cpp, $(LODIR)/%.o, $(notdir $(LSRCCXX)))
SRC := $(wildcard qfield*.cpp)
OBJ := $(patsubst %.cpp, $(ODIR)/%.o, $(SRC))
PRG_SRC := run_qfield.cpp
PRG_OBJ := $(patsubst %.cpp, $(ODIR)/%.o, $(PRG_SRC))
PRG := $(patsubst %.cpp, $(EDIR)/%, $(PRG_SRC))
#PRG_SUFFIX =
#BIN := $(patsubst %, %$(PRG_SUFFIX), $(PRG))

all : make_bin $(LOBJC) $(LOBJCXX) $(OBJ) $(PRG_OBJ) $(PRG)

$(ODIR)/%.o : %.cpp
	$(CXX) $(CXXFLAGS) $(INC) -o $@ -c $<

$(ODIR)/%.o : %.cpp %.hpp
	$(CXX) $(CXXFLAGS) $(INC) -o $@ -c $<

#$(EDIR)/% : $(ODIR)/%.o $(LOBJC) $(LOBJCXX)
#	$(CXX) $(LDFLAGS) $^ -o $@

$(LODIR)/%.o : $(LDIR)/%.c
	$(CC) $(CFLAGS) -o $@ -c $<

$(LODIR)/%.o : $(LDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

$(PRG) : $(LOBJC) $(LOBJCXX) $(OBJ) $(PRG_OBJ)
	$(CXX) $^ -o $@ $(LDFLAGS)

.PHONY : clean make_bin

clean : 
	rm -rf $(EDIR) $(ODIR) $(LODIR)

make_bin : $(LODIR) $(ODIR) $(EDIR)

$(LODIR) : 
	mkdir -p $(LODIR)

$(ODIR) :
	mkdir -p $(ODIR)

$(EDIR) :
	mkdir -p $(EDIR)
