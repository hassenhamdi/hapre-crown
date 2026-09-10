# CROWN6 C++ port — build (NEW file; existing sources read-only).
# C++17, -O2. crown6 semantics need -ffp-contract=off (frozen tanh LUT
# bit-exactness: no FMA contraction across the mul/add pairs, matching
# numpy's separate elementwise ufuncs). Applied to ALL C++ TUs for safety.
CXX = g++
CC = gcc
STD = -std=c++17
OPT = -O2
FP = -ffp-contract=off
WARN = -Wall -Wextra
SRC = ..
INC = -I.

all: crown_enc crown_dec crown_train

hapre_c.o: $(SRC)/src/hapre.c
	$(CC) -O2 -c -o $@ $<

crown6_c.o: $(SRC)/src/crown6.c $(SRC)/src/crown6_tanhlut.h
	$(CC) -O2 $(FP) -c -o $@ $<

codec.o: codec.cpp codec.h $(SRC)/src/crown6_tanhlut.h stb_image.h
	$(CXX) $(STD) $(OPT) $(FP) $(WARN) $(INC) -c -o $@ $<

crown_enc.o: crown_enc.cpp codec.h
	$(CXX) $(STD) $(OPT) $(FP) $(WARN) $(INC) -c -o $@ $< -pthread

crown_dec.o: crown_dec.cpp codec.h
	$(CXX) $(STD) $(OPT) $(FP) $(WARN) $(INC) -c -o $@ $< -pthread

crown_enc: crown_enc.o codec.o hapre_c.o crown6_c.o
	$(CXX) $(STD) $(OPT) $(FP) -o $@ $^ -pthread

crown_dec: crown_dec.o codec.o hapre_c.o crown6_c.o
	$(CXX) $(STD) $(OPT) $(FP) -o $@ $^ -pthread

crown_train.o: crown_train.cpp codec.h
	$(CXX) $(STD) $(OPT) $(FP) $(WARN) $(INC) -c -o $@ $< -pthread

crown_train: crown_train.o codec.o hapre_c.o crown6_c.o
	$(CXX) $(STD) $(OPT) $(FP) -o $@ $^ -pthread

# fast: same sources, -O3 -march=native (AVX2+FMA on this host) into SEPARATE objects/binaries.
# Integer costing is bit-exact under vectorization (exact integer arithmetic;
# -ffp-contract=off retained for the float64 tanh-LUT path). The default
# portable binaries above are never clobbered; *_fast must reproduce
# their bytes sha-identically (see cpp/README.md). x86-64 + AVX2 host only.
OPT_FAST = -O3 -march=native
hapre_fast.o: $(SRC)/src/hapre.c
	$(CC) $(OPT_FAST) -c -o $@ $<

crown6_fast.o: $(SRC)/src/crown6.c $(SRC)/src/crown6_tanhlut.h
	$(CC) $(OPT_FAST) $(FP) -c -o $@ $<

codec_fast.o: codec.cpp codec.h $(SRC)/src/crown6_tanhlut.h stb_image.h
	$(CXX) $(STD) $(OPT_FAST) $(FP) $(WARN) $(INC) -c -o $@ $<

crown_enc_fast.o: crown_enc.cpp codec.h
	$(CXX) $(STD) $(OPT_FAST) $(FP) $(WARN) $(INC) -c -o $@ $< -pthread

crown_dec_fast.o: crown_dec.cpp codec.h
	$(CXX) $(STD) $(OPT_FAST) $(FP) $(WARN) $(INC) -c -o $@ $< -pthread

crown_train_fast.o: crown_train.cpp codec.h
	$(CXX) $(STD) $(OPT_FAST) $(FP) $(WARN) $(INC) -c -o $@ $< -pthread

crown_enc_fast: crown_enc_fast.o codec_fast.o hapre_fast.o crown6_fast.o
	$(CXX) $(STD) $(OPT_FAST) $(FP) -o $@ $^ -pthread

crown_dec_fast: crown_dec_fast.o codec_fast.o hapre_fast.o crown6_fast.o
	$(CXX) $(STD) $(OPT_FAST) $(FP) -o $@ $^ -pthread

crown_train_fast: crown_train_fast.o codec_fast.o hapre_fast.o crown6_fast.o
	$(CXX) $(STD) $(OPT_FAST) $(FP) -o $@ $^ -pthread

fast: crown_enc_fast crown_dec_fast crown_train_fast

simd-info:
	@echo "SIMD: $(OPT_FAST) + $(FP) (FMA contraction OFF for tanh-LUT exactness)"
	@$(CXX) $(STD) $(OPT_FAST) -dM -E - < /dev/null | grep -E "__AVX2__|__AVX__|__FMA__|__SSE4" || true

clean:
	rm -f *.o crown_enc crown_dec crown_train crown_enc_fast crown_dec_fast crown_train_fast

.PHONY: all clean fast simd-info