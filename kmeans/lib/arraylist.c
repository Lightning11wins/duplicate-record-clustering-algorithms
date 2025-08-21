#include <stdio.h>
#include <stdlib.h>
#include "arraylist.h"

// Function to initialize the dynamic array
ArrayList* al_init(void) {
	return al_initc(10);
}

ArrayList* al_initc(size_t initialCapacity) {
	ArrayList* list = (ArrayList*)malloc(sizeof(ArrayList));
	if (list == NULL) return NULL;

	list->data = (int*)malloc(initialCapacity * sizeof(int));
	if (list->data == NULL) {
		free(list);
		return NULL;
	}
	list->size = 0;
	list->capacity = initialCapacity;
	list->is_locked = 0;
	return list;
}

static void al_ensure_capacity(ArrayList* list, size_t min_capacity) {
	while (list->capacity < min_capacity) {
		size_t new_size = (list->capacity *= 2) * sizeof(int);
		list->data = (int*)realloc(list->data, new_size);
		if (list->data == NULL) fprintf(stderr, "[ArrayList] realloc(%ld) failed!\n", new_size);
	}
}

void al_add(ArrayList* list, int element) {
	if (list->is_locked) {
		fprintf(stderr, "Attempted to add %d to locked list of size %ld.\n", element, list->size);
		return;
	}
	al_ensure_capacity(list, list->size + 1);
	list->data[list->size++] = element;
}

int al_get(ArrayList* list, size_t index) {
	if (index >= list->size) {
		fprintf(stderr, "[ArrayList] Index %ld out of bounds for length %ld!\n", index, list->size);
		return -1; // Fail
	}
	return list->data[index];
}

void al_lock(ArrayList* list) {
	list->is_locked = 1;
}

void al_unlock(ArrayList* list) {
	list->is_locked = 0;
}

void al_trim_to_size(ArrayList* list) {
	if (list->size < list->capacity) {
		size_t new_size = list->size * sizeof(int);
		list->data = (int*)realloc(list->data, new_size);
		if (list->data == NULL) fprintf(stderr, "[ArrayList] realloc(%ld) failed!\n", new_size);
		list->capacity = list->size;
	}
}

void al_clear(ArrayList* list) {
	if (list->is_locked) {
		fprintf(stderr, "Attempted to clear locked list of size %ld.\n", list->size);
		return;
	}
	list->size = 0;
}

void al_free(ArrayList* list) {
	if (list == NULL) return;
	free(list->data);
	free(list);
}