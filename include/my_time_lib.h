#ifndef MY_TIME_LIB_H
#define MY_TIME_LIB_H

#include <sys/time.h>

#define TIMER_DEF(n)     struct timeval temp_1_##n = {0, 0}, temp_2_##n = {0, 0}
#define TIMER_START(n)   gettimeofday(&temp_1_##n, (struct timezone *)0)
#define TIMER_STOP(n)    gettimeofday(&temp_2_##n, (struct timezone *)0)
#define TIMER_ELAPSED(n)                                                       \
  ((temp_2_##n.tv_sec - temp_1_##n.tv_sec) * 1.e6 +                            \
   (temp_2_##n.tv_usec - temp_1_##n.tv_usec))

#endif
