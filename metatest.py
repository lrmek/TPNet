import models
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from meta_template import MetaTemplate
import os
import cv2
import itertools
import matplotlib.pyplot as plt
from PIL import Image


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '8'


class SELayer(nn.Module):
	def __init__(self, channel, reduction=32):
		super(SELayer, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Sequential(
			nn.Linear(channel, channel // reduction, bias=False),
			nn.ReLU(inplace=True),
			nn.Linear(channel // reduction, 5, bias=False),
			nn.Sigmoid()
		)

	def forward(self, x):
		b, c, _, _ = x.size()
		y = self.avg_pool(x).view(b, c)
		y = self.fc(y).view(b, 5, 1, 1)
		# return x * y.expand_as(x)
		return y



class MetaTest(MetaTemplate):
	def __init__(self, model_func, n_way, n_support, alpha=1.0, loss_type="softmax"):
		super(MetaTest, self).__init__(model_func, n_way, n_support)
		self.loss_type = loss_type
		self.classifier = self.classifier(512, n_way)
		self.alpha = alpha
		self._initialize_weights()
		self.SElayer = SELayer(512, 16).cuda()

	def set_forward(self, index, x, x_paths, is_feature=True):
		return self.set_forward_adaptation(index, x, x_paths, is_feature)

	def classifier(self, in_planes, out_planes):
		return nn.Sequential(


			nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1),
			nn.ReLU(True),

			nn.Conv2d(512, out_planes, kernel_size=1, padding=0)
		)

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.xavier_uniform(m.weight.data)
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.01)
				#m.bias.data.zero_()
	def parse_feature(self, x, is_feature):
		x = Variable(x.cuda())
		if is_feature:

			z_all = x

		else:
			x = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
			z_all = self.feature.forward(x)
			z_all = z_all.view(self.n_way, self.n_support + self.n_query, -1)
		z_support = z_all[:, :self.n_support]
		z_query = z_all[:, self.n_support:]

		return z_support, z_query

	def set_forward_adaptation(self, index, x, x_paths, is_feature=True):
		assert is_feature == True, 'Baseline only support testing with feature'
		self._initialize_weights()
		z_support, z_query = self.parse_feature(x, is_feature)
		z_support_path = x_paths[:,:self.n_support]
		z_query_path = x_paths[:,self.n_support:]
		z_support = z_support.contiguous().view(self.n_way * self.n_support, 512, 28, 28)
		z_query = z_query.contiguous().view(self.n_way * self.n_query, 512, 28, 28)

		y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support))
		y_support = Variable(y_support.cuda())

		cls = self.classifier(self.feat_dim, self.n_way)
		cls = cls.cuda()

		cls_erase = self.classifier(self.feat_dim, self.n_way)
		cls_erase = cls_erase.cuda()

		cls_three = self.classifier(self.feat_dim, self.n_way)
		cls_three = cls_three.cuda()



		set_optimizer = torch.optim.SGD(itertools.chain(cls.parameters(), cls_erase.parameters(),self.SElayer.parameters()), lr=0.0005, momentum=0.9, dampening=0.9,
		                                weight_decay=0.001)

		loss_function = nn.CrossEntropyLoss()
		loss_function = loss_function.cuda()
		params = {}
		with open('canshu.txt') as f:
			params = eval(f.read())

		batch_size = params['batch_size']
		support_size = self.n_way * self.n_support
		for epoch in range(params['epochs']):
			rand_id = np.random.permutation(support_size)
			for i in range(0, support_size, batch_size):
				set_optimizer.zero_grad()
				selected_id = torch.from_numpy(rand_id[i: min(i + batch_size, support_size)]).cuda()
				z_batch = z_support[selected_id]
				y_batch = y_support[selected_id]
				s_attention = self.SElayer(z_batch)
				out_scores = cls(z_batch)
				out_scores = out_scores * s_attention.expand_as(out_scores)

				scores_one = torch.mean(torch.mean(out_scores, dim=2), dim=2)
				localization_map_normed = self.get_atten_map(out_scores, y_batch, True)
				self.attention = localization_map_normed
				feat_erase = self.erase_feature_maps(localization_map_normed, z_batch, 0.6)

				s2_attention = self.SElayer(feat_erase)
				out_erase = cls_erase(feat_erase)
				out_erase = out_erase * s2_attention.expand_as(out_erase)

				scores_erase = torch.mean(torch.mean(out_erase, dim=2), dim=2)
				#----three branch-------------------------------------------------------------------------------
				localization_map_normed = self.get_atten_map(out_erase, y_batch, True)
				three_erase = self.erase_feature_maps(localization_map_normed, feat_erase, 0.7)
				out_three = cls_three(three_erase)
				out_three = out_three * s_attention.expand_as(out_three)
				scores_three = torch.mean(torch.mean(out_three, dim=2), dim=2)
				# -------------------------------------------------------------------------------------------------------------

				scores = torch.mean(torch.mean(out_scores, dim=2), dim=2)
				loss_support = loss_function(scores_one, y_batch) + loss_function(scores_erase, y_batch) #  loss_function(scores_three, y_batch)

				loss = loss_support

				loss.backward(retain_graph=True)
				set_optimizer.step()
		q_attention = self.SElayer(z_query)
		query_scores_one = cls(z_query)

		query_scores = query_scores_one * q_attention.expand_as(query_scores_one)
		query_one = torch.mean(torch.mean(query_scores, dim=2), dim=2)  #
		query_pre_y = torch.max(query_one, 1)[1]
		localization_map_normed = self.get_atten_map(query_scores, query_pre_y , True)  #
		self.attention = localization_map_normed

		feat_erase = self.erase_feature_maps(localization_map_normed, z_query, 0.6)  #
		out_erase_two = cls_erase(feat_erase)

		q2_attention = self.SElayer(feat_erase)
		out_erase = out_erase_two * q2_attention.expand_as(out_erase_two)
		query_erase = torch.mean(torch.mean(out_erase, dim=2), dim=2)  # 16*20

		#-----three branch-----------------------------------------------------------------------
		query_pre_three = torch.max(query_erase, 1)[1]
		localization_map_normed = self.get_atten_map(out_erase, query_pre_three, True)
		three_erase = self.erase_feature_maps(localization_map_normed, feat_erase, 0.7)
		out_three = cls_three(three_erase)
		out_three = out_three * q_attention.expand_as(out_three)
		querybrab = torch.add(query_scores, out_erase)  # + 0.5*query_erase_gap
		queryclsi = torch.max(querybrab, out_three)
		scores = torch.mean(torch.mean(queryclsi, dim=2), dim=2)
		#---------------------------------------------------------


		# no attention(branch2 + fuse)
		query_no_one = torch.mean(torch.mean(query_scores_one, dim=2), dim=2)
		query_no_pre_y = torch.max(query_no_one, 1)[1]
		localization_no_map_normed = self.get_atten_map(query_scores_one, query_no_pre_y, True)  #
		feat_no_erase = self.erase_feature_maps(localization_no_map_normed, z_query, 0.6)  #
		out_no_erase_two = cls_erase(feat_no_erase)    # branch2 with no attention
		querybrab_no_fuse = torch.max(query_scores_one, out_no_erase_two)
##---------------------------------------------------------------------------------------------------------

		querybrab =  torch.max(query_scores, out_erase)
		scores = torch.mean(torch.mean(querybrab, dim=2), dim=2)
		#scores = query_one
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		##visuallization
		flag = True
		if(flag):
			flag = False
			pwd = os.getcwd()
			visual_path = os.path.join(pwd, "visualization/CUB/"+str(index)+"/")
			for i in range(z_support.shape[0]):
				os.makedirs(visual_path+x_paths[i][0].split('/')[-2])

		for i in range(z_support.shape[0]):
			for s_path in z_support_path[i]:   #save n support origion picuture
				support_path = os.path.join(os.getcwd(), "visualization/CUB/"+str(index)+"/"+s_path.split('/')[-2]+"/")
				Image.fromarray(plt.imread(s_path)).save(support_path + "support_"+s_path.split('/')[-1])

		if index == 0:
			labels = scores.cpu().detach().numpy().argmax(axis=1)
			for i in range(z_support.shape[0]):
				for j, q_path in enumerate(z_query_path[i]):
					query_path = os.path.join(os.getcwd(), "visualization/CUB/"+str(index)+"/"+q_path.split('/')[-2]+"/")
					img = plt.imread(q_path)
					img_w, img_h = img.shape[:2]
					Image.fromarray(img).save(query_path + "query_" + q_path.split('/')[-1])
					idx = i*z_query_path.shape[1]+j

					query_scores_img = (self.normalize_map(query_scores[idx][labels[idx]].data.cpu().numpy()) * 255).astype(np.uint8)
					query_scores_img = cv2.resize(query_scores_img, dsize=(img_h, img_w))
					query_scores_img = cv2.applyColorMap(query_scores_img, cv2.COLORMAP_JET)
					w_img = cv2.addWeighted(img.astype(np.uint8), 0.5, query_scores_img.astype(np.uint8), 0.5, 0)
					cv2.imwrite(query_path + "branch1_" + q_path.split('/')[-1], w_img)

					out_erase_img = (self.normalize_map(out_erase[idx][labels[idx]].data.cpu().numpy()) * 255).astype(np.uint8)
					out_erase_img = cv2.resize(out_erase_img, dsize=(224, 224))
					out_erase_img = cv2.applyColorMap(out_erase_img, cv2.COLORMAP_JET)
					cv2.imwrite(query_path + "branch2_" + q_path.split('/')[-1], out_erase_img)
					out_erase_img = (self.normalize_map(out_erase[idx][labels[idx]].data.cpu().numpy()) * 255).astype(np.uint8)
					out_erase_img = cv2.resize(out_erase_img, dsize=(img_h, img_w))
					out_erase_img = cv2.applyColorMap(out_erase_img, cv2.COLORMAP_JET)
					w_img = cv2.addWeighted(img.astype(np.uint8), 0.5, out_erase_img.astype(np.uint8), 0.5, 0)
					cv2.imwrite(query_path + "branch2_" + q_path.split('/')[-1], w_img)



					max_img = (self.normalize_map(querybrab[idx][labels[idx]].data.cpu().numpy())*255).astype(np.uint8)
					max_img = cv2.resize(max_img, dsize=(224,224))
					max_img = cv2.applyColorMap(max_img, cv2.COLORMAP_JET)
					cv2.imwrite(query_path + "max_" + q_path.split('/')[-1], max_img)

					max_img = (self.normalize_map(querybrab[idx][labels[idx]].data.cpu().numpy()) * 255).astype(np.uint8)
					max_img = cv2.resize(max_img, dsize=(img_h, img_w))
					max_img = cv2.applyColorMap(max_img, cv2.COLORMAP_JET)
					w_img = cv2.addWeighted(img.astype(np.uint8), 0.5, max_img.astype(np.uint8), 0.5, 0)
					cv2.imwrite(query_path + "max_" + q_path.split('/')[-1], w_img)
#######-------------------------------------no attention--------------------------
					max_noatten_img = (self.normalize_map(querybrab_no_fuse[idx][labels[idx]].data.cpu().numpy()) * 255).astype(np.uint8)
					max_noatten_img = cv2.resize(max_noatten_img, dsize=(224, 224))
					max_noatten_img = cv2.applyColorMap(max_noatten_img, cv2.COLORMAP_JET)
					cv2.imwrite(query_path + "nofuse" + q_path.split('/')[-1], max_noatten_img)
					max_noatten_img = (self.normalize_map(querybrab_no_fuse[idx][labels[idx]].data.cpu().numpy()) * 255).astype(np.uint8)
					max_noatten_img = cv2.resize(max_noatten_img, dsize=(img_h, img_w))
					max_noatten_img = cv2.applyColorMap(max_noatten_img, cv2.COLORMAP_JET)
					w_img = cv2.addWeighted(img.astype(np.uint8), 0.5, max_noatten_img.astype(np.uint8), 0.5, 0)
					cv2.imwrite(query_path + "nofuse" + q_path.split('/')[-1], w_img)
			print("visualization done!")

##over   ----------------------------------------------
		return scores

	def normalize_map(self, atten_map):
		min_val = np.min(atten_map)
		max_val = np.max(atten_map)
		return (atten_map - min_val) / (max_val - min_val)

	def set_forward_loss(self, x):
		raise ValueError('Baseline predict on pretrained feature and do not support finetune models')




	def mixup_data(self, x, y, alpha=1.0):

		'''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
		if alpha > 0.:
			lam = np.random.beta(alpha, alpha)
		else:
			lam = 1.
		#batch_size = x.size()[0]
		batch_size = x.size()[0]

		index = torch.randperm(batch_size)

		mixed_x = lam * x + (1 - lam) * x[index, :]
		y_a, y_b = y, y[index]
		return mixed_x, y_a, y_b, lam

	def mixup_criterion(self, y_a, y_b, lam):
		return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

	def add_heatmap2img(self, img, heatmap):

		heatmap = heatmap * 255
		color_map = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
		img_res = cv2.addWeighted(img.astype(np.uint8), 0.5, color_map.astype(np.uint8), 0.5, 0)

		return img_res

	def get_loss(self, logits, gt_labels):
		if self.onehot == 'True':
			gt = gt_labels.float()
		else:
			gt = gt_labels.long()
		loss_cls = self.loss_cross_entropy(logits[0], gt)
		loss_cls_ers = self.loss_cross_entropy(logits[1], gt)

		loss_val = loss_cls + loss_cls_ers

		return [loss_val, ]

	def get_localization_maps(self):
		map1 = self.normalize_atten_maps(self.map1)
		map_erase = self.normalize_atten_maps(self.map_erase)
		return torch.max(map1, map_erase)

	# return map_erase

	def get_heatmaps(self, gt_label):
		map1 = self.get_atten_map(self.map1, gt_label)
		return [map1, ]

	def get_fused_heatmap(self, gt_label):
		maps = self.get_heatmaps(gt_label=gt_label)
		fuse_atten = maps[0]
		return fuse_atten

	def get_maps(self, gt_label):
		map1 = self.get_atten_map(self.map1, gt_label)
		return [map1, ]

	def erase_feature_maps(self, atten_map_normed, feature_maps, threshold):

		if len(atten_map_normed.size()) > 3:
			atten_map_normed = torch.squeeze(atten_map_normed)
		atten_shape = atten_map_normed.size()

		pos = torch.ge(atten_map_normed, threshold)
		mask = torch.ones(atten_shape).cuda()
		mask[pos.data] = 0.0
		mask = torch.unsqueeze(mask, dim=1)
		# erase
		erased_feature_maps = feature_maps * Variable(mask)

		return erased_feature_maps

	def normalize_atten_maps(self, atten_maps):
		atten_shape = atten_maps.size()

		# --------------------------
		batch_mins, _ = torch.min(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
		batch_maxs, _ = torch.max(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
		atten_normed = torch.div(atten_maps.view(atten_shape[0:-2] + (-1,)) - batch_mins,
		                         batch_maxs - batch_mins)
		atten_normed = atten_normed.view(atten_shape)

		return atten_normed

	def save_erased_img(self, img_path, img_batch=None):
		mean_vals = [0.485, 0.456, 0.406]
		std_vals = [0.229, 0.224, 0.225]
		if img_batch is None:
			img_batch = self.img_erased
		if len(img_batch.size()) == 4:
			batch_size = img_batch.size()[0]
			for batch_idx in range(batch_size):
				imgname = img_path[batch_idx]
				nameid = imgname.strip().split('/')[-1].strip().split('.')[0]


				atten_map = F.upsample(self.attention.unsqueeze(dim=1), (224, 224), mode='bilinear')

				mask = atten_map
				mask = mask.squeeze().cpu().data.numpy()

				img_dat = img_batch[batch_idx]
				img_dat = img_dat.cpu().data.numpy().transpose((1, 2, 0))
				img_dat = (img_dat * std_vals + mean_vals) * 255

				mask = cv2.resize(mask, (321, 321))
				img_dat = self.add_heatmap2img(img_dat, mask)
				save_path = os.path.join('../save_bins/', nameid + '.png')
				cv2.imwrite(save_path, img_dat)

	def get_atten_map(self, feature_maps, gt_labels, normalize=True):
		label = gt_labels.long()

		feature_map_size = feature_maps.size()
		batch_size = feature_map_size[0]

		atten_map = torch.zeros([feature_map_size[0], feature_map_size[2], feature_map_size[3]])
		atten_map = Variable(atten_map.cuda())
		for batch_idx in range(batch_size):
			atten_map[batch_idx, :, :] = torch.squeeze(feature_maps[batch_idx, label.data[batch_idx], :, :])

		if normalize:
			atten_map = self.normalize_atten_maps(atten_map)

		return atten_map


def imshow(img,text="",should_save=False):
    npimg = img.cpu().detach().numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    if npimg.ndim == 3:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    else:
        plt.imshow(npimg)
    plt.show()