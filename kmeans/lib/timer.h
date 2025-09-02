#ifndef TIMER_H
#define TIMER_H

typedef struct {
    double start;
    double end;
    double stored_duration; // Store a duration for future comparison.
} Timer;

double monotonic_seconds(void);

Timer* timer_init(Timer* timer);
Timer* timer_new(void);
void timer_start(Timer* timer);
void timer_stop(Timer* timer);
double timer_get(Timer* timer);
double timer_store(Timer* timer);
void timer_print(const Timer* timer, const char* name);
void timer_print_s(const Timer* timer, const char* name);
void timer_print_cmp(const Timer* timer, const char* name, double cmp);
void timer_free(Timer* timer);

#define timer_benchmark(timer, code) timer_start(timer); code timer_stop(timer)

#endif // TIMER_H