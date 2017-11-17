new_val_text = open('../../data/val.txt', 'w')

with open('../../data/old_val.txt', 'r') as f:
	for line in f:
		line = line.split()
		path = line[0]
		label = line[1]

		path_list = path.split('/')
		# print path_list
		path_list[1] = 'val_' + path_list[1]

		new_path = '/'.join(path_list)
		new_line = new_path + ' ' + label + '\n'

		# print new_line

		new_val_text.write(new_line)

new_val_text.close()

