for i in {4..36}; do
    thres=`bc <<< "scale=1; $i/1"`
    python measures_v4.py --threshold $thres
    python roi_metric_v2.py --method pf_40x20 --data_type interactive --threshold $thres
    # thres=`bc <<< "scale=2; $i/100*2"`
    # python roi_metric_trend.py --method pf_40x20 --data_type interactive --threshold $thres
    # echo $thres
done
