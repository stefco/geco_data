# Plotting Improvements/Document To Do

- [ ] maybe don't connect the dots with lines while the detector is off
- [ ] check and make sure 7ns is our time quantization
  on the y-label); will avoid compression and will show how good we are for the
  full run.
- [ ] put start/end dates, segment name, and channel name in the captions
- [ ] there is a missing second in L1 O1; explain it in the Overleaf caption
- [ ] explain in the L1 captions that L1 has more jitter, piecewise linear
- [ ] explain that L1 is not properly calibrated and remark on how visible the
  difference is, but without making anyone feel kicked in the balls
- [ ] Check JRPC for exact dates
- [ ] make O2 current up to present day
- [ ] recommendation in the instruction: it is very important to calibrate the
  cesium clock at the start of the run.
- [ ] investigate the two zeroes in the L1 O2 plot and put those in the caption
- [ ] send overleaf document later today
- [ ] put the timing installation omnigraffle maps into the overleaf document
- [ ] check the jump time in o1 at H1 in the Y-end CNS-II clocks
- [ ] check loss of signal in H1 O2 CNS-II x-end and y-end (happened twice,
  likely due to non-position-locked loss of satellite signal as already logged)
- [ ] use a fourth type of color (orange circle with x in it?) to mark when
  there is no data available (indicated by zero values), or just don't mark
  them.  But, in either case, find the specific seconds/minutes where we lose
  signal, print them/save them somewhere, and just exclude those from the plot
  (print them to a metadata rider file associated with the plot). Calculate the
  missing segments using similar code to that used for finding missing DATA
  segments when fetching data.
- [ ] look up any other zeros.
- [ ] the CNS II at L1 during O1; why is there this weird slow trend and then
  suddenly everything looks correct? look in alog.
- [ ] check EY CNS II at livingston; what is going on? why all the spurious
  points with delays of multiple milliseconds? are we actually measuring
  anything during the good portions?
- [ ] set the plot scales manually
- [ ] make it possible to specify the x-axis and x-limits as well
- [ ] for LLO EY CNS II, just say that data before it was plugged in is not recorded and exclude it from the x-axis. but make sure to check on that data first.
- [ ] put in giant vertical errorbar to show where missing times are in busy plots
- [ ] exclude missing times from legend when there are none, i.e. don’t do the missing times plot unless there are missing times

# Plots to Make
- [ ] mean-removed plots (otherwise the same)
