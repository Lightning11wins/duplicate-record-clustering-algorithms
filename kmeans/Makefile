CC = gcc
VISFLAGS = -I/usr/include/cairo -lgsl -lgslcblas -lcairo 
CFLAGS_ALL = -Wall -Wextra -Wfloat-equal -Wundef -Wpointer-arith -Wcast-align -Wstrict-prototypes -Wstrict-overflow=5 -Wwrite-strings -Waggregate-return -Wcast-qual -Wswitch-default -Wswitch-enum -Wconversion -Wunreachable-code -Wformat=2 -lm
CFLAGS_FAST = $(CFLAGS_ALL) -O3
CFLAGS_DEBUG = $(CFLAGS_ALL) -g -O0
UFLAGS = -Wshadow

NAME = kmeans
DIR = bin
BIN_FAST = $(DIR)/$(NAME)_fast.o
BIN_DEBUG = $(DIR)/$(NAME)_debug.o
SRC = $(NAME).c
ASM = $(NAME).asm
OUT = out_std.txt
OUT_COMPLETE = out_complete.css
OUT_CURRENT = out_current.css
OUT_KMEANS = out_kmeans.css
CMD_ARGS = $(OUT_COMPLETE) $(OUT_CURRENT) $(OUT_KMEANS)

.PHONY: all rerun rebuild build clean run debug valgrind asm

all: build

rerun: clean run

rebuild: clean build

build: $(DIR) $(BIN_FAST)

clean:
	rm -rf $(DIR)
	rm -f $(BIN_FAST) $(BIN_DEBUG) $(ASM) $(OUT) $(OUT_COMPLETE) $(OUT_CURRENT) $(OUT_KMEANS) .gdb_history

run: $(DIR) $(BIN_FAST)
	./$(BIN_FAST) $(CMD_ARGS) > $(OUT)

debug: $(DIR) $(BIN_DEBUG)
	gdb --args ./$(BIN_DEBUG) $(CMD_ARGS)

valgrind: $(DIR) $(BIN_DEBUG)
	valgrind ./$(BIN_DEBUG) $(CMD_ARGS)

asm: $(ASM)

$(ASM): $(SRC)
	$(CC) -S -o $(ASM) $(SRC) $(CFLAGS_FAST)

$(BIN_DEBUG): $(SRC)
	$(CC) -o $(BIN_DEBUG) $(SRC) $(CFLAGS_DEBUG)

$(BIN_FAST): $(SRC)
	$(CC) -o $(BIN_FAST) $(SRC) $(CFLAGS_FAST)

$(DIR):
	mkdir $(DIR)
