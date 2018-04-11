starttime="Feb 25 2018 23:59:37 GMT"
for det in H1 L1; do
    for arm in X Y; do
        channel="${det}:CAL-PCAL${arm}_IRIGB_OUT_DQ"
        echo "Decoded IRIG-B times from ${channel} starting at ${starttime}:"
        nds_query \
                -n nds.ligo.caltech.edu \
                -s $(lalapps_tconvert ${starttime}) \
                -d 30 \
                -v ${channel} \
            | get_vals \
            | geco_irig_decode.py 
    done
done
