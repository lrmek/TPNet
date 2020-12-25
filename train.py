import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob

import configs
import models
from data.datamgr import SimpleDataManager, SetDataManager
from metatrain import BackboneTrain
from metatest import MetaTest

from io_utils import model_dict, parse_args, get_resume_file

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'



def get_finetune_optimizer(model):
    lr = 0.0001
    weight_list = []
    bias_list = []
    last_weight_list = []
    last_bias_list =[]
    for name,value in model.named_parameters():
        if 'cls' in name:
            print(name)
            if 'weight' in name:
                last_weight_list.append(value)
            elif 'bias' in name:
                last_bias_list.append(value)
        else:
            if 'weight' in name:
                weight_list.append(value)
            elif 'bias' in name:
                bias_list.append(value)

    weight_decay = 0.0001

    opt = torch.optim.SGD([{'params': weight_list, 'lr':lr},
                     {'params':bias_list, 'lr':lr*2},
                     {'params':last_weight_list, 'lr':lr*10},
                     {'params': last_bias_list, 'lr':lr*20}], momentum=0.9, weight_decay=0.0001, nesterov=True)
    # opt = optim.SGD([{'params': weight_list, 'lr':lr},
    #                  {'params':bias_list, 'lr':lr*2},
    #                  {'params':last_weight_list, 'lr':lr*10},
    #                  {'params': last_bias_list, 'lr':lr*20}], momentum=0.9, nesterov=True)

    return opt

def train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params):
	if optimization == 'Adam':
		optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
	elif optimization == "SGD":
		optimizer = get_finetune_optimizer(model)
	else:
		raise ValueError('Unknown optimization, please define by yourself')

	max_acc = 0

	for epoch in range(start_epoch, stop_epoch):
		model.train()
		model.train_loop(epoch, base_loader, optimizer)  # model are called by reference, no need to return
		model.eval()

		if not os.path.isdir(params.checkpoint_dir):
			os.makedirs(params.checkpoint_dir)

		acc = model.test_loop(val_loader)
		if acc > max_acc:  # for baseline and baseline++, we don't use validation in default and we let acc = -1, but we allow options to validate with DB index
			print("best model! save...")
			max_acc = acc
			outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
			torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

		if (epoch % params.save_freq == 0) or (epoch == stop_epoch - 1):
			outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
			torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

	return model



def _model_load(model, pretrained_dict):
	model_dict = model.state_dict()

	# model_dict_keys = [v.replace('module.', '') for v in model_dict.keys() if v.startswith('module.')]
	if list(model_dict.keys())[0].startswith('module.'):
		pretrained_dict = {'module.' + k: v for k, v in pretrained_dict.items()}
	if list(model_dict.keys())[0].startswith('feature.'):
		pretrained_dict = {k.replace("features.","feature."): v for k, v in pretrained_dict.items()}
	pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
	print("Weights cannot be loaded:")
	print([k for k in model_dict.keys() if k not in pretrained_dict.keys()])

	model_dict.update(pretrained_dict)
	model.load_state_dict(model_dict)

if __name__ == '__main__':
	np.random.seed(10)
	params = parse_args('train')

	base_file = configs.data_dir[params.dataset] + 'base.json'
	val_file = configs.data_dir[params.dataset] + 'val.json'



	image_size = 224


	optimization = 'SGD'

	if params.stop_epoch == -1:
		if params.method in ['baseline', 'baseline++']:

			if params.dataset in ['CUB']:
				params.stop_epoch = 500
			else:
				params.stop_epoch = 400
		else:
			if params.n_shot == 1:
				params.stop_epoch = 600
			elif params.n_shot == 5:
				params.stop_epoch = 400
			else:
				params.stop_epoch = 600

	if params.method in ['baseline', 'baseline++']:
		base_datamgr = SimpleDataManager(image_size, batch_size=16)
		base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)
		val_datamgr = SimpleDataManager(image_size, batch_size=64)
		val_loader = val_datamgr.get_data_loader(val_file, aug=False)

		if params.method == 'baseline':
			model = BackboneTrain(model_dict[params.model], params.num_classes)
		elif params.method == 'baseline++':
			model = BackboneTrain(model_dict[params.model], params.num_classes, loss_type='dist')


	else:
		raise ValueError('Unknown method')

	model = model.cuda()

	params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (configs.save_dir, params.dataset, params.model, params.method)
	if params.train_aug:
		params.checkpoint_dir += '_aug'
	if not params.method in ['baseline', 'baseline++']:
		params.checkpoint_dir += '_%dway_%dshot' % (params.train_n_way, params.n_shot)

	if not os.path.isdir(params.checkpoint_dir):
		os.makedirs(params.checkpoint_dir)

	start_epoch = params.start_epoch
	stop_epoch = params.stop_epoch
	if params.method == 'maml' or params.method == 'maml_approx':
		stop_epoch = params.stop_epoch * model.n_task

	if params.resume:

		resume_file = get_resume_file(params.checkpoint_dir)
		resume_file = "/home/liruimin/.cache/torch/checkpoints/vgg16-397923af.pth"
		#resume_file = "/home/liruimin/Resnet-bockbonevgg/checkpoints/omniglot/VggNet_baseline_aug/15.tar"
		if resume_file is not None:
			tmp = torch.load(resume_file)

			try:
				model.load_state_dict(tmp['state'])
				start_epoch = tmp['epoch'] + 1
				print("load success %s"%resume_file)
			except KeyError:
				_model_load(model, tmp)

	elif params.warmup:
		baseline_checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (
		configs.save_dir, params.dataset, params.model, 'baseline')
		if params.train_aug:
			baseline_checkpoint_dir += '_aug'
		warmup_resume_file = get_resume_file(baseline_checkpoint_dir)
		tmp = torch.load(warmup_resume_file)
		if tmp is not None:
			state = tmp['state']
			state_keys = list(state.keys())
			for i, key in enumerate(state_keys):
				if "feature." in key:
					newkey = key.replace("feature.",
					                     "")
					state[newkey] = state.pop(key)
				else:
					state.pop(key)
			model.feature.load_state_dict(state)
		else:
			raise ValueError('No warm_up file')


	model = train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params)
