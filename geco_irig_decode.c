#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SAMPLE_RATE 16384           /* sample rate of ADC */
#define BITS_PER_SECOND 100         /* bits per second in IRIG-B spec */
#define HIGH_SIGNAL_THRESHOLD 3500  /* define high vs. low signal */
#define NUM_TEST_POINTS 4           /* number of test points in each bit */
#define NUM_CONTROL_BITS 11         /* number of control bits in each second*/
#define EXIT_BAD_BIT 64             /* exit code for a bad bit */
#define EXIT_MISPLACED_CTRL_BIT 65  /* exit code for misplaced control bit */
#define EXIT_UNEXPECTED_EOF 66      /* EOF in the middle of second of data */

/* keep using the same timeseries and decoded_bits arrays */
float timeseries[SAMPLE_RATE];
int decoded_bits[BITS_PER_SECOND];
int load1sec();
int readbit(int i);
void validate_timeseries();
void print_time();
char *get_time_string(int sec, int min, int hrs, int day, int yr);
int exit_status = 0;

/* count characters in input; 1st version */
int main() {
    while (load1sec()) {
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
    const static int control_bit_addresses[NUM_CONTROL_BITS] = 
        {0,9,19,29,39,49,59,69,79,89,99};
    /* a value of 1 indicates a control bit */
    static int control_bits[BITS_PER_SECOND]; /* static initializes to zero */
    for (i=0; i<NUM_CONTROL_BITS; i++)
        control_bits[control_bit_addresses[i]] = 1;

    for (i=0; i<BITS_PER_SECOND; i++)
        if (decoded_bits[i] == -1) {
            puts("ERROR: Bad Bit! Dumping decoded bit stream:\n");
            for (i=0; i<BITS_PER_SECOND; i++)
                printf("%d\t%d\n", i, decoded_bits[i]);
            exit_status = EXIT_BAD_BIT;
            break;
        } else if (control_bits[i] && decoded_bits[i] != 2) {
            exit_status = EXIT_MISPLACED_CTRL_BIT;
            puts("ERROR: Misplaced Control Bit at position ");
            printf("%d", i);
            puts("! Dumping decoded bit stream:\n");
            for (i=0; i<BITS_PER_SECOND; i++)
                printf("%d\t%d\t%d\n", i, control_bits[i], decoded_bits[i]);
            break;
        }
}

/* turn the decoded bits into a time and print that TODO make it formatted */
void print_time() {
    int sec, min, hrs, day, yr;
    sec = min = hrs = day = yr = 0;

    /* how many seconds does each bit represent? */
    sec += 1 * decoded_bits[1];
    sec += 2 * decoded_bits[2];
    sec += 4 * decoded_bits[3];
    sec += 8 * decoded_bits[4];
    sec += 10 * decoded_bits[6];
    sec += 20 * decoded_bits[7];
    sec += 40 * decoded_bits[8];

    /* how many minutes does each bit represent? */
    min += 1 * decoded_bits[10];
    min += 2 * decoded_bits[11];
    min += 4 * decoded_bits[12];
    min += 8 * decoded_bits[13];
    min += 10 * decoded_bits[15];
    min += 20 * decoded_bits[16];
    min += 40 * decoded_bits[17];

    /* how many hours does each bit represent? */
    hrs += 1 * decoded_bits[20];
    hrs += 2 * decoded_bits[21];
    hrs += 4 * decoded_bits[22];
    hrs += 8 * decoded_bits[23];
    hrs += 10 * decoded_bits[25];
    hrs += 20 * decoded_bits[26];

    /* how many days does each bit represent? */
    day += 1 * decoded_bits[30];
    day += 2 * decoded_bits[31];
    day += 4 * decoded_bits[32];
    day += 8 * decoded_bits[33];
    day += 10 * decoded_bits[35];
    day += 20 * decoded_bits[36];
    day += 40 * decoded_bits[37];
    day += 80 * decoded_bits[38];
    day += 100 * decoded_bits[40];
    day += 200 * decoded_bits[41];

    /* how many years does each bit represent? */
    yr += 1 * decoded_bits[50];
    yr += 2 * decoded_bits[51];
    yr += 4 * decoded_bits[52];
    yr += 8 * decoded_bits[53];
    yr += 10 * decoded_bits[55];
    yr += 20 * decoded_bits[56];
    yr += 40 * decoded_bits[57];
    yr += 80 * decoded_bits[58];

    /* print the current time */
    printf("%s", get_time_string(sec, min, hrs, day, yr));
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
    load the timeseries from stdin.
    return 0 if done reading.
    */
    for (i=0; i<SAMPLE_RATE; i++)
        if (fscanf(stdin, "%f\n", &timeseries[i]) == EOF) {
            if (i != 0)
                exit_status = EXIT_UNEXPECTED_EOF;
            return 0;
        }

    /*
    find beginning of each IRIG-B signal bit (they are evenly spaced)
    and decode that bit.
    */
    for (i=0; i<BITS_PER_SECOND; i++)
        decoded_bits[i] = readbit((i * SAMPLE_RATE) / BITS_PER_SECOND);

    return 1;
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
    /* TODO remove debug */
    /* printf("[%d %d %d %d]\n", tp[0], tp[1], tp[2], tp[3]); */

    /* test for invalid values */
    if (tp[1] < tp[2] || tp[0] == 0 || tp[3] == 1)
        return -1;

    /* decode */
    return tp[1] + tp[2];
}
