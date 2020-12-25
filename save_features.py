

import numpy as np
import torch
from torch.autograd import Variable
import os
import glob
import h5py

import configs
import models
from data.datamgr import SimpleDataManager
from metatrain import BackboneTrain
from metatest import MetaTest
from io_utils import model_dict, parse_args, get_resume_file, get_best_file, get_assigned_file

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '9'


def save_features(model, data_loader, outfile):
	f = h5py.File(outfile, 'w')
	dt = h5py.special_dtype(vlen=str)
	max_count = len(data_loader) * data_loader.batch_size
	all_labels = f.create_dataset('all_labels', (max_count,), dtype='i')
	all_paths = f.create_dataset('all_paths', (max_count,), dtype=dt)
	all_feats = None
	count = 0
	for i, (x_path, x, y) in enumerate(data_loader):
		if i % 10 == 0:
			print('{:d}/{:d}'.format(i, len(data_loader)))
		x = x.cuda()
		x_var = Variable(x)
		feats = model(x_var)
		if all_feats is None:
			all_feats = f.create_dataset('all_feats', [max_count] + list(feats.size()[1:]), dtype='f')
		all_feats[count:count + feats.size(0)] = feats.data.cpu().numpy()
		all_labels[count:count + feats.size(0)] = y.cpu().numpy()
		all_paths[count:count + feats.size(0)] = x_path
		count = count + feats.size(0)

	count_var = f.create_dataset('count', (1,), dtype='i')
	count_var[0] = count

	f.close()

if __name__ == '__main__':
	params = parse_args('save_features')
	assert params.method != 'maml' and params.method != 'maml_approx', 'maml do not support save_feature and run'


	image_size = 224

	split = params.split
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

	checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (configs.save_dir, params.dataset, params.model, params.method)
	if params.train_aug:
		checkpoint_dir += '_aug'
	if not params.method in ['baseline', 'baseline++']:
		checkpoint_dir += '_%dway_%dshot' % (params.train_n_way, params.n_shot)

	if params.save_iter != -1:
		modelfile = get_assigned_file(checkpoint_dir, params.save_iter)
	#    elif params.method in ['baseline', 'baseline++'] :
	#        modelfile   = get_resume_file(checkpoint_dir) #comment in 2019/08/03 updates as the validation of baseline/baseline++ is added
	else:
		modelfile = get_best_file(checkpoint_dir)
	print("modelfile: ", modelfile)
	if params.save_iter != -1:
		outfile = os.path.join(checkpoint_dir.replace("checkpoints", "features"),
		                       split + "_" + str(params.save_iter) + ".hdf5")
	else:
		outfile = os.path.join(checkpoint_dir.replace("checkpoints", "features"), split + ".hdf5")

	datamgr = SimpleDataManager(image_size, batch_size=10)
	data_loader = datamgr.get_data_loader(loadfile, aug=False)


	model = model_dict[params.model]()

	model = model.cuda()
	tmp = torch.load(modelfile)
	state = tmp['state']
	state_keys = list(state.keys())
	for i, key in enumerate(state_keys):
		if "feature." in key:
			newkey = key.replace("feature.", "")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
			state[newkey] = state.pop(key)
		else:
			state.pop(key)

	model.load_state_dict(state)
	model.eval()

	dirname = os.path.dirname(outfile)
	if not os.path.isdir(dirname):
		os.makedirs(dirname)
	save_features(model, data_loader, outfile)

