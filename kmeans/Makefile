CC = gcc
VISFLAGS = -I/usr/include/cairo -lgsl -lgslcblas -lcairo 
CFLAGS_ALL = -Wall -Wextra -Wfloat-equal -Wundef -Wpointer-arith -Wcast-align -Wstrict-prototypes -Wstrict-overflow=5 -Wwrite-strings -Waggregate-return -Wcast-qual -Wswitch-default -Wswitch-enum -Wconversion -Wunreachable-code -Wformat=2 -lm
CFLAGS_FAST = $(CFLAGS_ALL) -O3
CFLAGS_DEBUG = $(CFLAGS_ALL) -g -O0
UFLAGS = -Wshadow

NAME = kmeans
BUILD_DIR = bin
LIB_DIR = lib
ARR_DEBUG = $(BUILD_DIR)/arraylist_debug.o
ARR_FAST = $(BUILD_DIR)/arraylist_fast.o
ARR_SRC = $(LIB_DIR)/arraylist.c
BIN_DEBUG = $(BUILD_DIR)/$(NAME)_debug.o
BIN_FAST = $(BUILD_DIR)/$(NAME)_fast.o
SRC = $(NAME).c
ASM = $(NAME).asm
OUT = out_std.txt
OUT_COMPLETE = out_complete.txt
OUT_SLIDING = out_sliding.txt
OUT_KMEANS = out_kmeans.css
CMD_ARGS = $(OUT_COMPLETE) $(OUT_SLIDING) $(OUT_KMEANS)

.PHONY: all rerun rebuild build clean run debug valgrind asm

all: build

rerun: clean run

rebuild: clean build

build: $(BUILD_DIR) $(BIN_FAST)

clean:
	rm -rf $(BUILD_DIR)
	rm -f $(BIN_FAST) $(BIN_DEBUG) $(ASM) $(OUT) $(OUT_COMPLETE) $(OUT_SLIDING) $(OUT_KMEANS) .gdb_history

run: $(BUILD_DIR) $(BIN_FAST)
	./$(BIN_FAST) $(CMD_ARGS) > $(OUT)

debug: $(BUILD_DIR) $(BIN_DEBUG)
	gdb --args ./$(BIN_DEBUG) $(CMD_ARGS)

valgrind: $(BUILD_DIR) $(BIN_DEBUG)
	valgrind ./$(BIN_DEBUG) $(CMD_ARGS)

asm: $(ASM)

$(ASM): $(ARR_DEBUG) $(SRC)
	$(CC) $(ARR_DEBUG) -S -o $(ASM) $(SRC) $(CFLAGS_FAST)

$(BIN_DEBUG): $(ARR_DEBUG) $(SRC)
	$(CC) $(ARR_DEBUG) -o $(BIN_DEBUG) $(SRC) $(CFLAGS_DEBUG)

$(BIN_FAST): $(ARR_FAST) $(SRC)
	$(CC) $(ARR_FAST) -o $(BIN_FAST) $(SRC) $(CFLAGS_FAST)

$(ARR_DEBUG): $(ARR_SRC)
	$(CC) -o $(ARR_DEBUG) -c $(ARR_SRC) $(CFLAGS_FAST)

$(ARR_FAST): $(ARR_SRC)
	$(CC) -o $(ARR_FAST) -c $(ARR_SRC) $(CFLAGS_FAST)

$(BUILD_DIR):
	mkdir $(BUILD_DIR)
