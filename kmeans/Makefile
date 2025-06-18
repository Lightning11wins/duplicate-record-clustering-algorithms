CC = gcc
VISFLAGS = -I/usr/include/cairo -lgsl -lgslcblas -lcairo 
CFLAGS = -Wall -Wextra -Wfloat-equal -Wundef -Wpointer-arith -Wcast-align -Wstrict-prototypes -Wstrict-overflow=5 -Wwrite-strings -Waggregate-return -Wcast-qual -Wswitch-default -Wswitch-enum -Wconversion -Wunreachable-code -Wformat=2 -lm -O3
UFLAGS = -Wshadow

NAME = kmeans
EXE = $(NAME).o
SRC = $(NAME).c
OUT = output.txt

.PHONY: all rerun rebuild build clean run debug valgrind

all: build

rerun: clean run

rebuild: clean build

build: $(EXE)

clean:
	rm -f $(EXE) $(OUT)

run: $(EXE)
	./$(EXE) > $(OUT)

debug: $(EXE)
	gdb ./$(EXE)

valgrind: $(EXE)
	valgrind ./$(EXE)

$(EXE): $(SRC)
	$(CC) -o $(EXE) $(SRC) $(CFLAGS)
