#ifndef __H_ARRAYLIST
#define __H_ARRAYLIST
#include <stdbool.h>

// Constants
#define AL_DEFAULT_CAPACITY (size_t)16
#define AL_MIN_CAPACITY (size_t)4

// Struct
typedef struct {
	size_t size;
	size_t capacity;
	bool is_locked;
	int* data;
} ArrayList;

// Constructors
ArrayList* al_init(ArrayList* list);
ArrayList* al_initc(ArrayList* list, size_t initial_capacity);
ArrayList* al_new(void);
ArrayList* al_newc(size_t initial_capacity);

// Member methods
void al_add(ArrayList* list, int element);
int  al_get(ArrayList* list, size_t index);
void al_lock(ArrayList* list);
void al_unlock(ArrayList* list);
void al_trim_to_size(ArrayList* list);
void al_clear(ArrayList* list);
void al_destruct(ArrayList* list);
void al_free(ArrayList* list);

#endif