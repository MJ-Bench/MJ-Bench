First run `generate_json.py` file, input your models, checkpoints, and your expected directory (it will be created under the root directory/your_directory_name): 
e.g. 
```
python generate_json.py --models pick_v1_finetune clip_v2_finetune --checkpoints 198 198 --des_img_path 123testimg
```
Then run `group_evaluation.py` file, input your expected directory path (should be the same as the previous one):
e.g.
```
python group_evaluation.py --des_img_path 123testimg
```
