import os
import Augmentor

root_dir = '../../data/images/train/'

total_dirs = 0
valid_dirs = 0

invalid_dirs = 17
invalid_dict = {}

label_dict = {}

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

with open('../../data/train.txt') as f:
    for line in f:
    	line = line.split()
        file_name = line[0]
        file_label = line[1]

        label_name = file_name.split('/')[2]

        label_dict[label_name] = file_label

k = 0

output_dirs = []

for subdir, dirs, files in os.walk(root_dir):
    for dir in dirs:

    	current_dir = os.path.join(subdir, dir)

    	if total_dirs < 17 or get_immediate_subdirectories(current_dir):
    		invalid_dict[current_dir] = True

    	if not invalid_dict.get(current_dir, False):
    		
    		current_dir_list = current_dir.split('/')
    		current_dir_label_name = current_dir_list[6]
    		current_dir_label_num = label_dict[current_dir_label_name]

    		output_dir = "../" + current_dir_label_name + "_augmented"

    		# if k < 1:

    		p = Augmentor.Pipeline(current_dir, output_dir)
    		p.skew(probability=0.4)
    		p.random_distortion(probability=0.4, grid_width=4, grid_height=4, magnitude=6)
    		p.rotate(probability=0.4, max_left_rotation=15, max_right_rotation=15)

    		p.sample(4000)

    		output_dirs.append((current_dir + '/' + output_dir, current_dir_label_name, current_dir_label_num))

	    		# k += 1

        total_dirs += 1

with open("../../data/train.txt", "a") as f:
	for output_dir in output_dirs:
		for subdir, dirs, files in os.walk(output_dir[0]):
			for file in files:
				subdir_list = subdir.split('/')
				augmented_image_path = [subdir_list[4], subdir_list[5], subdir_list[-1], file]
				training_txt_update = '/'.join(augmented_image_path) + ' ' + output_dir[2] + '\n'

				f.write(training_txt_update)
