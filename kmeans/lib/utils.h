#ifndef UTLIS_H
#define UTLIS_H

#include <string.h>
#include <stdlib.h>

#define INDENT "\t> "

// Readable string equality.
#define streql(str1, str2) (strcmp((str1), (str2)) == 0)
#define strneql(str1, str2, num) (strncmp((str1), (str2), (num)) == 0)

// Min and max.
#define min(a, b) ({ \
	__typeof__ (a) _a = (a); \
	__typeof__ (b) _b = (b); \
	(_a < _b) ? _a : _b; \
})

#define max(a, b) ({ \
	__typeof__ (a) _a = (a); \
	__typeof__ (b) _b = (b); \
	(_a > _b) ? _a : _b; \
})

// Random.
#define rand_dbl() ((double)rand() / (double)RAND_MAX)

#define random(min, max) ({ \
    __typeof__ (min) _min = (min); \
    __typeof__ (max) _max = (max); \
    _min + (rand() % (_max - _min + 1)); \
})

// Control flow.
#define repeat(times, inc) for (__typeof__ (times) _times = (times), inc = 0; inc < _times; inc++)

// Comments.
#define comment(code)
#define super_comment(text) __asm__ volatile("# " text)

// Function signatures.
void fprint_mem(FILE* out);
void fprint_stacktrace(void);
void fail(const char* function_name); 
int check(int result, const char* function_name);
void* check_ptr(void* result, const char* function_name);
void* mallocs(const size_t size);

#endif // UTLIS_H