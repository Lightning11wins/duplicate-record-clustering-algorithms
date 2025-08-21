#ifndef __H_ARRAYLIST
#define __H_ARRAYLIST

// Struct
typedef struct {
	size_t size;
	size_t capacity;
	int is_locked;
	int* data;
} ArrayList;

// Constructors
ArrayList* al_init(void);
ArrayList* al_initc(size_t initialCapacity);

// Member methods
void al_add(ArrayList* list, int element);
int  al_get(ArrayList* list, size_t index);
void al_lock(ArrayList* list);
void al_unlock(ArrayList* list);
void al_trim_to_size(ArrayList* list);
void al_clear(ArrayList* list);
void al_free(ArrayList* list);

#endif