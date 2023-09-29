CC=gcc
CFLAGS=-Iinclude/ -Wall -std=gnu99 -g
OMPFLAGS=$(CFLAGS) -fopenmp
LIBS=src/data_handling.c
TUNE=-O2

.PHONY: all
all: serial omp

.PHONY: serial
serial:
	$(CC) $(TUNE) $(CFLAGS) -o bin/serial_matmul $(LIBS) src/serial_matmul.c

omp: omp_matmul

omp_matmul:
	$(CC) $(TUNE) $(OMPFLAGS) -o bin/omp_matmul $(LIBS) src/omp_matmul.c
			
.PHONY: clean
clean: clean_serial clean_openmp

.PHONY: clean_serial
clean_serial:
	@rm -rf bin/serial_matmul

.PHONY: clean_openmp
clean_openmp:
	@rm -rf bin/omp_matmul

MAT_A = mat_a.txt
MAT_B = mat_b.txt
MAT_C0 = mat_c0.txt
MAT_C = mat_c.txt

PROBLEM = matmul

SERIAL_BIN = serial_$(PROBLEM)

run_serial: serial
	@echo "*** Running $(SERIAL_BIN) and storing result in $(MAT_C0) ***"
	bin/$(SERIAL_BIN) $(MAT_A) $(MAT_B) > $(MAT_C0)

NUM_THREADS = 16
OMP_BIN = omp_$(PROBLEM)

run_omp:
	@echo "*** Running $(OMP_BIN) and storing result in $(MAT_C) ***"
	bin/$(OMP_BIN) $(NUM_THREADS) $(MAT_A) $(MAT_B) > $(MAT_C)

SEED_A = 406
SEED_B = 506
X_A = 3
Y_A = 4
X_B = 4
Y_B = 3
OUT_A = mat_a.txt
OUT_B = mat_b.txt

gen_matrix:
	python3 gen_matrix.py $(SEED_A) $(X_A) $(Y_A) > $(OUT_A)
	python3 gen_matrix.py $(SEED_B) $(X_B) $(Y_B) > $(OUT_B)

# In self-grader we don't run serial each time, correct output is generated only once
# check:
check: run_serial run_omp
	@echo "*** Running diff ***"
# not present on ARC, but you can install it locally with apt install numdiff or try to use diff
# 	numdiff $(MAT_C0) $(MAT_C)
	diff -iw $(MAT_C0) $(MAT_C)

pack:
	zip -r project1.zip src/ include/

JOB = job.omp

sbatch:
	sbatch $(JOB)
