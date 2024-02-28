import numpy as np
import json
import os
from collections import OrderedDict


data_root = "../../pf_trajectory_40x20_actor-target=60_v4/interactive"
train_town = ["1_", "2_", "3_", "5_", "6_", "7_", "A1"][:] # 1350, (45, 30)
test_town = ["10", "A6", "B3"]   # 515, (47, 11)
town =  train_town+test_town

def main():

	cnt = 0
	basic_cnt = 0
	
	final_reachable_points = OrderedDict()

	for basic in sorted(os.listdir(data_root)):
		if not basic[:2] in town:
			continue

		basic_cnt += 1
		basic_path = os.path.join(data_root, basic)
		final_reachable_points[basic] = OrderedDict()

		for variant in sorted(os.listdir(basic_path)):
			variant_path = os.path.join(basic_path, variant, 'actor_trajectory_json')
			final_reachable_points[basic][variant] = OrderedDict()

			for frame in sorted(os.listdir(variant_path)):
				json_path = os.path.join(variant_path, frame)
				json_data = json.load(open(json_path))

				frame_id = str(int(frame.split('.')[0]))
				final_reachable_points[basic][variant][frame_id] = OrderedDict()
				
				for actor_id in json_data:
					final_reachable_points[basic][variant][frame_id][actor_id] = json_data[actor_id][-1]

			cnt += 1
			print(cnt, basic, variant, 'Done!!!')


	with open('./final_reachable_points.json', 'w') as f:
		json.dump(final_reachable_points, f, indent=4)


if __name__ == '__main__':
	main()


