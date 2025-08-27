#ifndef UTLIS_H
#define UTLIS_H

#define streql(str1, str2) (strcmp((str1), (str2)) == 0)

#define max(a, b) ({ \
	__typeof__ (a) _a = (a); \
	__typeof__ (b) _b = (b); \
	(_a > _b) ? _a : _b; \
})

#define min(a, b) ({ \
	__typeof__ (a) _a = (a); \
	__typeof__ (b) _b = (b); \
	(_a < _b) ? _a : _b; \
})

#define random(min, max) ({ \
    __typeof__ (min) _min = (min); \
    __typeof__ (max) _max = (max); \
    _min + (rand() % (_max - _min + 1)); \
})

#define repeat(times, inc) for (__typeof__ (times) _times = (times), inc = 0; inc < _times; inc++)

#define comment(code)
#define super_comment(text) __asm__ volatile("# " text)

#endif // UTLIS_H