#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SAMPLE_RATE 16384               // sample rate of ADC
#define BITS_PER_SECOND 100             // bits per second in IRIG-B spec
#define HIGH_SIGNAL_THRESHOLD 3500      // define high vs. low signal

// TODO take timeseries out of global scope, make more functional maybe

// keep using the same timeseries and decoded_bits arrays
float timeseries[SAMPLE_RATE];
int decoded_bits[BITS_PER_SECOND];

// helper functions
void load1sec();
int readbit(int i);

/* count characters in input; 1st version */
int main() {
    load1sec();
}

// this will only process 1 second worth of data
void load1sec() {
    int i, j;

    /*
    find beginning of each IRIG-B signal bit (they are evenly spaced)
    and decode that bit.
    */
    for (i=0; i<BITS_PER_SECOND; i++)
        decoded_bits[i] = readbit((i * SAMPLE_RATE) / BITS_PER_SECOND);
    
    // load the timeseries from stdin
    for (i=0; i<SAMPLE_RATE; i++)
        fscanf(stdin, "%f\n", &timeseries[i]);

    // TODO: remove this line, which is for testing
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
    // how many samples into each bit to measure? TODO make indep. of SAMPLE_RATE
    const int TEST_POINTS[4] = {20, 60, 110, 150};
    // get values at test points; will be 0 or 1
    const int tp[4] = {
        timeseries[i+TEST_POINTS[0]] > HIGH_SIGNAL_THRESHOLD,
        timeseries[i+TEST_POINTS[1]] > HIGH_SIGNAL_THRESHOLD,
        timeseries[i+TEST_POINTS[2]] > HIGH_SIGNAL_THRESHOLD,
        timeseries[i+TEST_POINTS[3]] > HIGH_SIGNAL_THRESHOLD
    };
    // test for invalid values
    if (tp[1] < tp[2] || tp[0] == 0 || tp[3] == 1) return -1;
    return tp[1] + tp[2];   // decode
}
