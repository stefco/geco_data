#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SAMPLE_RATE 16384           /* sample rate of ADC */
#define BITS_PER_SECOND 100         /* bits per second in IRIG-B spec */
#define HIGH_SIGNAL_THRESHOLD 3500  /* define high vs. low signal */
#define NUM_TEST_POINTS 4           /* number of test points in each bit */
#define NUM_CONTROL_BITS 11         /* number of control bits in each second*/

/* keep using the same timeseries and decoded_bits arrays */
float timeseries[SAMPLE_RATE];
int decoded_bits[BITS_PER_SECOND];
int load1sec();
int readbit(int i);
void validate_timeseries();
void print_time();
char *get_time_string(int sec, int min, int hrs, int day, int yr)
int exit_status = 0;

/* count characters in input; 1st version */
int main() {
    while load1sec() {
        validate_timeseries();
        if (exit_status != 0)
            return exit_status;
        print_time();
    }
    return 0;
}

/*
make sure that all bits were read correctly and that control codes are
in the correct places.
*/
void validate_timeseries() {
    register int i;
    static int control_bits[NUM_CONTROL_BITS] = {1,9,19,29,39,49,59,69,79,89,99};
    for (i=0; i<BITS_PER_SECOND; i++)
        if (decoded_bits[i] == -1) {
            exit_status = 1;
            break;
        }
    for (i=0; i<NUM_CONTROL_BITS; i++)
        if (decoded_bits[i] != 2) {
            exit_status = 1;
            break;
        }
}

/* turn the decoded bits into a time and print that TODO make it formatted */
int print_time() {
    
}

/*
get a string representing time based on seconds, minutes, hours, days, and
years since 2000
*/
char *get_time_string(int sec, int min, int hrs, int day, int yr) {
    struct tm tstruct = {
        sec,        /* seconds */
        min,        /* minutes */
        hrs,        /* hours */
        day,        /* days since start of year, i.e. since start of january */
        0,          /* month of january */
        100 + yr,   /* years since 1900 */
        0,          /* days since start of week (not used) */
        0,          /* days since start of year (not used) */
        -1          /* daylight savings time, unknown */
    };
    /* properly calculate month and day for display */
    mktime(&tstruct);
    /* return date in format Sun Jan  3 15:14:13 1988\n */
    return asctime(&tstruct);
}

/* this will only process 1 second worth of data */
int load1sec() {
    int i, j;

    /*
    find beginning of each IRIG-B signal bit (they are evenly spaced)
    and decode that bit.
    */
    for (i=0; i<BITS_PER_SECOND; i++)
        decoded_bits[i] = readbit((i * SAMPLE_RATE) / BITS_PER_SECOND);
    
    /*
    load the timeseries from stdin.
    return 0 if done reading, 1 if success.
    */
    for (i=0; i<SAMPLE_RATE; i++)
        if (fscanf(stdin, "%f\n", &timeseries[i]) == EOF) {
            if (i != 0)
                exit_status = 1;    /* didn't exit cleanly */
            return 0;
        }
    return 1;

    /* TODO: remove this line, which is for testing */
    for (i = 0; i < SAMPLE_RATE; i++)
    {
        printf("Number is: %lf\n", timeseries[i]);
    }
}

/*
Read each bit and return its value: 0, 1, -1 (error), or 2 (control bit).
How each character is represented at each test point:
BIT VALUE:          REPRESENTATION:
0                   [1,0,0,0]
1                   [1,1,0,0]
2  (control)        [1,1,1,0]
-1 (error)          (anything else)
*/
int readbit(int i) {
    /* how many seconds into each bit to measure? round it. */
    register int j;
    static int tp[NUM_TEST_POINTS];
    const static int TEST_POINTS[NUM_TEST_POINTS] = {
        (int) SAMPLE_RATE*1.22e-03 + 0.5,
        (int) SAMPLE_RATE*3.66e-03 + 0.5,
        (int) SAMPLE_RATE*6.71e-03 + 0.5,
        (int) SAMPLE_RATE*9.16e-03 + 0.5
    };

    /* get values at test points; will be 0 or 1 */
    for (j=0; j<NUM_TEST_POINTS; j++)
        tp[j] = timeseries[i+TEST_POINTS[j]] > HIGH_SIGNAL_THRESHOLD;

    /* test for invalid values */
    if (tp[1] < tp[2] || tp[0] == 0 || tp[3] == 1)
        return -1;

    /* decode */
    return tp[1] + tp[2];
}
