h_files := $(wildcard mat*.h)
exes := $(patsubst %.h,%,$(h_files))
asms := $(addsuffix .s,$(exes))

opts := 
opts += -march=native 
opts += -O3
opts += -Wall
#opts += -fopt-info-vec-optimized 
opts += -std=c++0x

#opts += -fopt-info-vec-missed
#opts += -fopenmp-simd
#opts += -funroll-loops 
# opts += -std=c++0x
CXX := g++
CXXFLAGS := $(opts)

all : $(exes) 
# $(asms)

$(exes) : % : %.h 
	$(CXX) $(CXXFLAGS) -DINTEGRAL_H=\"$*.h\" mat.cc -o $@ -ldr -lmyth

$(asms) : %.s : %.h
	$(CXX) $(CXXFLAGS) -DINTEGRAL_H=\"$*.h\" mat.cc -o $@ -S

clean :
	rm -f $(exes) $(asms)

