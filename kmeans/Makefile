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
OUT = output.txt

.PHONY: all rerun rebuild build clean run debug valgrind asm

all: build

rerun: clean run

rebuild: clean build

build: $(DIR) $(BIN_FAST)

clean:
	rm -rf $(DIR)
	rm -f $(BIN_FAST) $(BIN_DEBUG) $(OUT) $(ASM) .gdb_history

run: $(DIR) $(BIN_FAST)
	./$(BIN_FAST) > $(OUT)

debug: $(DIR) $(BIN_DEBUG)
	gdb ./$(BIN_DEBUG)

valgrind: $(DIR) $(BIN_DEBUG)
	valgrind ./$(BIN_DEBUG)

asm: $(ASM)

$(ASM): $(SRC)
	$(CC) -s -o $(ASM) $(SRC) $(CFLAGS_DEBUG)

$(BIN_DEBUG): $(SRC)
	$(CC) -o $(BIN_DEBUG) $(SRC) $(CFLAGS_DEBUG)

$(BIN_FAST): $(SRC)
	$(CC) -o $(BIN_FAST) $(SRC) $(CFLAGS_FAST)

$(DIR):
	mkdir $(DIR)
