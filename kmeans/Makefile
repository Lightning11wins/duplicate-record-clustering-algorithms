CC = gcc
VISFLAGS = -I/usr/include/cairo -lgsl -lgslcblas -lcairo 
CFLAGS_ALL = -Wall -Wextra -Wfloat-equal -Wundef -Wpointer-arith -Wcast-align -Wstrict-prototypes -Wstrict-overflow=5 -Wwrite-strings -Waggregate-return -Wcast-qual -Wswitch-default -Wswitch-enum -Wconversion -Wunreachable-code -Wformat=2 -lm
CFLAGS_FAST = $(CFLAGS_ALL) -O3
CFLAGS_DEBUG = $(CFLAGS_ALL) -g -rdynamic -O0
UFLAGS = -Wshadow

NAME = kmeans
BUILD_DIR = bin
LIB_DIR = lib
OUT_DIR = out
DATA_DIR = datasets
LIBS = arraylist timer
LIBOS = $(addprefix $(BUILD_DIR)/, $(addsuffix .o, $(LIBS)))
BIN_DEBUG = $(BUILD_DIR)/$(NAME)_debug.o
BIN_FAST = $(BUILD_DIR)/$(NAME)_fast.o
SRC = $(NAME).c
ASM = $(NAME).asm
PRE = $(NAME)_pre.c
OUT = $(OUT_DIR)/std.txt
OUT_COMPLETE = $(OUT_DIR)/complete.txt
OUT_SLIDING = $(OUT_DIR)/sliding.txt
OUT_KMEANS = $(OUT_DIR)/kmeans.css
CMD_ARGS = $(OUT_COMPLETE) $(OUT_SLIDING) $(OUT_KMEANS)

.PHONY: all rerun rebuild build clean run debug gdb valgrind pre preprocess asm

all: build

rerun: clean run

rebuild: clean build

build: $(BUILD_DIR) $(BIN_FAST)

clean:
	rm -rf $(BUILD_DIR) $(OUT_DIR)
	rm -f $(BIN_FAST) $(BIN_DEBUG) $(PRE) $(ASM) $(OUT) $(OUT_COMPLETE) $(OUT_SLIDING) $(OUT_KMEANS) .gdb_history a.out
	git gc

run: $(BUILD_DIR) $(OUT_DIR) $(BIN_FAST)
	./$(BIN_FAST) $(CMD_ARGS) > $(OUT)

run_debug: $(BUILD_DIR) $(OUT_DIR) $(BIN_DEBUG)
	./$(BIN_DEBUG) $(CMD_ARGS) > $(OUT)

debug: gdb

gdb: $(BUILD_DIR) $(OUT_DIR) $(BIN_DEBUG)
	gdb --args ./$(BIN_DEBUG) $(CMD_ARGS)

valgrind: $(BUILD_DIR) $(OUT_DIR) $(BIN_DEBUG)
	valgrind ./$(BIN_DEBUG) $(CMD_ARGS)

pre: preprocess

preprocess: $(PRE)
	code $^

$(PRE): $(SRC)
	$(CC) -E -o $@ $^ $(CFLAGS_DEBUG)

asm: $(ASM)
	code $^

$(ASM): $(SRC)
	$(CC) -S -o $@ $^ $(CFLAGS_FAST) -g

$(BIN_DEBUG): $(LIBOS) $(SRC)
	$(CC) -o $@ $^ $(CFLAGS_DEBUG)

$(BIN_FAST): $(LIBOS) $(SRC)
	$(CC) -o $@ $^ $(CFLAGS_FAST)

$(LIBOS): $(BUILD_DIR)/%.o: $(LIB_DIR)/%.c
	$(CC) -o $@ -c $^ $(CFLAGS_FAST)

$(BUILD_DIR):
	mkdir $(BUILD_DIR)

$(OUT_DIR):
	mkdir $(OUT_DIR)

$(DATA_DIR):
	mkdir $(DATA_DIR)
