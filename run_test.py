import os
BATCH_SIZES = [4]
EPOCHS = [175]

for i in range(len(EPOCHS)):
	print("{{'batch_size':{0}, 'epochs':{1}}}".format(BATCH_SIZES[i], EPOCHS[i]))
	with open('canshu.txt','w+') as f:
		f.write("{{'batch_size':{0}, 'epochs':{1}}}".format(BATCH_SIZES[i], EPOCHS[i]))
	cmd_str = "--dataset CUB --model VggNet --method baseline --train_aug --save_iter 60 --train_n_way 5 --test_n_way 5 --n_shot 1"
	res = os.popen("python test.py " + cmd_str)
	with open('1Ad2results.txt','a+') as f:
		f.write("\n\n\n\n'batch_size':{0}, 'epochs':{1}\n".format(BATCH_SIZES[i], EPOCHS[i]))
		f.write(res.read())

