#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "arraylist.h"
#include "utils.h"

ArrayList* al_init(ArrayList* list) {
	return al_initc(list, AL_DEFAULT_CAPACITY);
}

ArrayList* al_initc(ArrayList* list, size_t initial_capacity) {
	if (!list) return list;
	
	if (initial_capacity < AL_MIN_CAPACITY) {
		fprintf(stderr, "[ArrayList] al_initc() - Increasing initial capacity from requested (%ld) to minimum (%ld).\n", initial_capacity, AL_MIN_CAPACITY);
	}

	list->data = malloc(initial_capacity * sizeof(unsigned int));
	if (list->data == NULL) {
		free(list);
		return NULL;
	}
	list->size = 0;
	list->capacity = initial_capacity;
	list->is_locked = 0;
	return list;
}

ArrayList* al_new(void) {
	return al_newc(AL_DEFAULT_CAPACITY);
}

ArrayList* al_newc(size_t initial_capacity) {
	ArrayList* list = malloc(sizeof(ArrayList));
	al_initc(list, initial_capacity);
	return list;
}

static void al_ensure_capacity(ArrayList* list, size_t min_capacity) {
	while (list->capacity < min_capacity) {
		size_t new_size = (list->capacity *= 2) * sizeof(unsigned int);
		list->data = (unsigned int*)realloc(list->data, new_size);
		if (list->data == NULL) fprintf(stderr, "[ArrayList] al_ensure_capacity() - realloc(%ld) failed!\n", new_size);
	}
}

void al_add(ArrayList* list, unsigned int element) {
	if (list->is_locked) {
		fprintf(stderr, "[ArrayList] al_add() - Attempted to add %d to locked list of size %ld.\n", element, list->size);
		return;
	}
	al_ensure_capacity(list, list->size + 1);
	list->data[list->size++] = element;
}

unsigned int al_get(ArrayList* list, size_t index) {
	if (index >= list->size) {
		fprintf(stderr, "[ArrayList] al_get() - Index %ld out of bounds for length %ld!\n", index, list->size);
		return 0; // Fail
	}
	return list->data[index];
}

void al_lock(ArrayList* list) {
	list->is_locked = true;
}

void al_unlock(ArrayList* list) {
	list->is_locked = false;
}

void al_trim_to_size(ArrayList* list) {
	if (list->size < list->capacity) {
		size_t new_size_bytes = (list->capacity = max(AL_MIN_CAPACITY, list->size)) * sizeof(unsigned int);
		list->data = (unsigned int*)realloc(list->data, new_size_bytes);
		if (list->data == NULL) fprintf(stderr, "[ArrayList] al_trim_to_size() - realloc(%ld) failed!\n", new_size_bytes);
		list->capacity = list->size;
	}
}

void al_clear(ArrayList* list) {
	if (list->is_locked) {
		fprintf(stderr, "[ArrayList] al_clear() - Attempted to clear locked list of size %ld.\n", list->size);
		return;
	}
	list->size = 0;
}

void al_destruct(ArrayList* list) {
	free(list->data);
}

void al_free(ArrayList* list) {
	if (list == NULL) return;
	free(list->data);
	free(list);
}