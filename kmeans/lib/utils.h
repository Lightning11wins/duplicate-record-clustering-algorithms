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

#endif // UTLIS_H