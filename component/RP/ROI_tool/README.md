
1. 把 inference 完的結果 (raw scroe) 按照格式放到 `"./model/{MODEL}/{DATA_TYPE}.json"`
   
2. 執行 

		python roi_metric.py --method {MODEL} --transpose --threshold {THRESHOLD} --data_type {DATA_TYPE} --save_result

	或

		python roi_metric.py --method {MODEL} --data_type {DATA_TYPE} --save_result 

3. `--transpose` 會將 raw score > `THRESHOLD` 的 actor 視為 risky
    
4. 結果會存到 `"result/{MODEL}/{DATA_TYPE}_result.json"`

5. Example: 
       
   		python roi_metric_v2.py --method RRL --transpose --threshold 0.0 --data_type interactive --save_result



