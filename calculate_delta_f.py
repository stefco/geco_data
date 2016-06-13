#!/usr/bin/env python

import os
import subprocess
import math
import sys
import argparse

args = sys.argv
if len(args) == 1 or args[1] == '-h':
    print('This program takes two arguments: the first is the current correction to the frequency as read from the Monitor2 software on the IBM Thinkpad in the mainstorage room, and the second is the drift coefficient.')
    exit()
elif len(args) != 3:
    print('This program takes exactly two arguments')

def calculate_new_delta_f(current_delta_f, drift_coefficient):
    current_delta_f = current_delta_f * 10**-15
    tmp = -1*drift_coefficient/(1+drift_coefficient)
    tmp += current_delta_f
    tmp = tmp * 10**15
    new_delta_f = tmp
    print('The new value to enter into the Monitor2 software is '+ str(new_delta_f))

calculate_new_delta_f(float(args[1]), float(args[2]))
    
