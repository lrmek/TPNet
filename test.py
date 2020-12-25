import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.optim
import json
import torch.utils.data.sampler
import os
import glob
import random
import time

import configs
import models
import data.feature_loader as feat_loader
from data.datamgr import SetDataManager
from metatrain import BackboneTrain
from metatest import MetaTest
from io_utils import model_dict, parse_args, get_resume_file, get_best_file, get_assigned_file

import os
import matplotlib.pyplot as plt


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def feature_evaluation(index, cl_data_file, model, n_way=5, n_support=5, n_query=15, adaptation=False):
	class_list = cl_data_file.keys()

	select_class = random.sample(class_list, n_way)
	z_all = []
	z_paths = []
	for cl in select_class:
		img_feat = cl_data_file[cl]
		perm_ids = np.random.permutation(len(img_feat)).tolist()
		z_all.append([np.squeeze(img_feat[perm_ids[i]][0]) for i in range(n_support + n_query)])
		z_paths.append([np.squeeze(img_feat[perm_ids[i]][1]) for i in range(n_support + n_query)])

	z_all = torch.from_numpy(np.array(z_all))
	z_paths = np.array(z_paths)
	model.n_query = n_query
	if adaptation:
		scores = model.set_forward_adaptation(z_all, is_feature=True)                              ###################################
	else:
		scores = model.set_forward(index, z_all, z_paths, is_feature=True)
	pred = scores.data.cpu().numpy().argmax(axis=1)
	y = np.repeat(range(n_way), n_query)
	acc = np.mean(pred == y) * 100
	return acc


if __name__ == '__main__':
	params = parse_args('test')

	acc_all = []

	iter_num = 500

	few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)



	if params.method == 'baseline':
		model = MetaTest(model_dict[params.model], **few_shot_params)

	else:
		raise ValueError('Unknown method')

	model = model.cuda()

	checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (configs.save_dir, params.dataset, params.model, params.method)
	if params.train_aug:
		checkpoint_dir += '_aug'
	if not params.method in ['baseline', 'baseline++']:
		checkpoint_dir += '_%dway_%dshot' % (params.train_n_way, params.n_shot)


	if not params.method in ['baseline', 'baseline++']:
		if params.save_iter != -1:
			modelfile = get_assigned_file(checkpoint_dir, params.save_iter)
		else:
			modelfile = get_best_file(checkpoint_dir)
		if modelfile is not None:
			tmp = torch.load(modelfile)
			model.load_state_dict(tmp['state'])

	split = params.split
	if params.save_iter != -1:
		split_str = split + "_" + str(params.save_iter)
	else:
		split_str = split
	if params.method in ['maml', 'maml_approx']:
		if 'Conv' in params.model:
			if params.dataset in ['omniglot', 'cross_char']:
				image_size = 28
			else:
				image_size = 84
		else:
			image_size = 224

		datamgr = SetDataManager(image_size, n_eposide=iter_num, n_query=15, **few_shot_params)

		if params.dataset == 'cross':
			if split == 'base':
				loadfile = configs.data_dir['miniImagenet'] + 'all.json'
			else:
				loadfile = configs.data_dir['CUB'] + split + '.json'
		elif params.dataset == 'cross_char':
			if split == 'base':
				loadfile = configs.data_dir['omniglot'] + 'noLatin.json'
			else:
				loadfile = configs.data_dir['emnist'] + split + '.json'
		else:
			loadfile = configs.data_dir[params.dataset] + split + '.json'

		novel_loader = datamgr.get_data_loader(loadfile, aug=False)
		if params.adaptation:
			model.task_update_num = 100
		model.eval()
		acc_mean, acc_std = model.test_loop(novel_loader, return_std=True)

	else:
		novel_file = os.path.join(checkpoint_dir.replace("checkpoints", "features"), split_str + ".hdf5")
		print("novel file: ",novel_file)
		cl_data_file = feat_loader.init_loader(novel_file)

		for i in range(iter_num):
			acc = feature_evaluation(i, cl_data_file, model, n_query=15, adaptation=params.adaptation, **few_shot_params)
			acc_all.append(acc)
			print("item: %s, acc: %.4f, avg_acc: %.4f, min: %.4f, max: %.4f" % (i, acc, np.mean(acc_all),np.min(acc_all),np.max(acc_all)))

		acc_all = np.asarray(acc_all)
		acc_mean = np.mean(acc_all)
		acc_std = np.std(acc_all)
		print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
	with open('./record/results.txt', 'a') as f:
		timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
		aug_str = '-aug' if params.train_aug else ''
		aug_str += '-adapted' if params.adaptation else ''
		if params.method in ['baseline', 'baseline++']:
			exp_setting = '%s-%s-%s-%s%s %sshot %sway_test' % (
			params.dataset, split_str, params.model, params.method, aug_str, params.n_shot, params.test_n_way)
		else:
			exp_setting = '%s-%s-%s-%s%s %sshot %sway_train %sway_test' % (
			params.dataset, split_str, params.model, params.method, aug_str, params.n_shot, params.train_n_way,
			params.test_n_way)
		acc_str = '%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num))
		f.write('Time: %s, Setting: %s, Acc: %s \n' % (timestamp, exp_setting, acc_str))
