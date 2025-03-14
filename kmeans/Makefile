CC = gcc
VISFLAGS = -I/usr/include/cairo -lgsl -lgslcblas -lcairo 
CFLAGS = -lm -Wall -Wextra -O3

NAME = kmeans
EXE = $(NAME).exe
SRC = $(NAME).c
OUT = output.txt

.PHONY: all run debug valgrind clean

all: $(EXE)

run: $(EXE)
	./$(EXE) > $(OUT)

debug: $(EXE)
	gdb ./$(EXE)
		
valgrind: $(EXE)
	valgrind ./$(EXE)

$(EXE): $(SRC)
	$(CC) $(CFLAGS) -o $(EXE) $(SRC)

clean:
	rm -f $(EXE) $(OUT)
