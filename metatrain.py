import models
import utils

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2

import models
import utils

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'





class BackboneTrain(nn.Module):
	def __init__(self, model_func, num_class, loss_type='softmax',indim = 512):
		super(BackboneTrain, self).__init__()
		self.feature = model_func()


		self.cls = self.classifier(512, num_class)
		self.cls_erase = self.classifier(512, num_class)
		self._initialize_weights()
		self.loss_type = loss_type
		self.num_class = num_class
		self.loss_fn = nn.CrossEntropyLoss()
		self.DBval = False

	def classifier(self, in_planes, out_planes):
		return nn.Sequential(

			nn.Conv2d(in_planes, 512, kernel_size=3, padding=1, dilation=1),
			nn.ReLU(True),

			nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1),
			nn.ReLU(True),
			nn.Conv2d(512, out_planes, kernel_size=1, padding=0)
		)


	def forward(self, x):
		x = Variable(x.cuda())
		x = self.feature(x)

		feat = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)


		out = self.cls(feat)

		logits_1 = torch.mean(torch.mean(out, dim=2), dim=2)
		return logits_1

	def forward_loss(self, x, y):

		out_logits = self.forward(x)
		y = Variable(y.cuda())
		loss = self.loss_fn(out_logits, y)
		return loss




	def _initialize_weights(self):
		for m in self.modules():
		# for m in list(self.children())[1]:
			if isinstance(m, nn.Conv2d):
				nn.init.xavier_uniform(m.weight.data)
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()

	def train_loop(self, epoch, train_loader, optimizer):
		print_freq = 10
		avg_loss = 0


		for i, (x_path,x, y) in enumerate(train_loader):
			optimizer.zero_grad()
			loss = self.forward_loss(x, y)
			loss.backward()
			optimizer.step()

			avg_loss = avg_loss + loss.item()

			if i % print_freq == 0:
				# print(optimizer.state_dict()['param_groups'][0]['lr'])
				print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss / float(i + 1)))

	def test_loop(self, val_loader):
		if self.DBval:
			return self.analysis_loop(val_loader)
		else:
			return -1

	def analysis_loop(self, val_loader, record=None):
		class_file = {}
		for i, (x, y) in enumerate(val_loader):
			x = x.cuda()
			x_var = Variable(x)
			feats = self.feature.forward(x_var).data.cpu().numpy()
			labels = y.cpu().numpy()
			for f, l in zip(feats, labels):
				if l not in class_file.keys():
					class_file[l] = []
				class_file[l].append(f)

		for cl in class_file:
			class_file[cl] = np.array(class_file[cl])

		DB = DBindex(class_file)
		print('DB index = %4.2f' % (DB))
		return 1 / DB  # DB index: the lower the better


def DBindex(cl_data_file):

	class_list = cl_data_file.keys()
	cl_num = len(class_list)
	cl_means = []
	stds = []
	DBs = []
	for cl in class_list:
		cl_means.append(np.mean(cl_data_file[cl], axis=0))
		stds.append(np.sqrt(np.mean(np.sum(np.square(cl_data_file[cl] - cl_means[-1]), axis=1))))

	mu_i = np.tile(np.expand_dims(np.array(cl_means), axis=0), (len(class_list), 1, 1))
	mu_j = np.transpose(mu_i, (1, 0, 2))
	mdists = np.sqrt(np.sum(np.square(mu_i - mu_j), axis=2))

	for i in range(cl_num):
		DBs.append(np.max([(stds[i] + stds[j]) / mdists[i, j] for j in range(cl_num) if j != i]))
	return np.mean(DBs)


