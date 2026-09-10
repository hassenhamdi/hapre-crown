# HAPRE-CROWN codec family — build all native extensions.
CC = gcc
CFLAGS_HAPRE = -O3 -shared -fPIC
CFLAGS_CROWN = -O2 -shared -fPIC
# crown6 needs strict FP contraction for the frozen tanh LUT bit-exactness proof
CFLAGS_CROWN6 = -O2 -shared -fPIC -ffp-contract=off
SRC = src

all: $(SRC)/libhapre.so $(SRC)/libcrown2.so $(SRC)/libcrown3.so $(SRC)/libcrown4.so $(SRC)/libcrown5.so $(SRC)/libcrown6.so

$(SRC)/libhapre.so: $(SRC)/hapre.c
	$(CC) $(CFLAGS_HAPRE) -o $@ $<

$(SRC)/libcrown2.so: $(SRC)/crown2.c
	$(CC) $(CFLAGS_CROWN) -o $@ $<

$(SRC)/libcrown3.so: $(SRC)/crown3.c
	$(CC) $(CFLAGS_CROWN) -o $@ $<

$(SRC)/libcrown4.so: $(SRC)/crown4.c
	$(CC) $(CFLAGS_CROWN) -o $@ $<

$(SRC)/libcrown5.so: $(SRC)/crown5.c
	$(CC) $(CFLAGS_CROWN) -o $@ $<

$(SRC)/libcrown6.so: $(SRC)/crown6.c $(SRC)/crown6_tanhlut.h
	$(CC) $(CFLAGS_CROWN6) -o $@ $<

clean:
	rm -f $(SRC)/*.so
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null; true

check:
	python3 -c "import sys; sys.path.insert(0,'src'); import driver, driver_run; print('imports ok')"

help:
	@echo "targets: all clean check help; see README.md and reproduce.sh"