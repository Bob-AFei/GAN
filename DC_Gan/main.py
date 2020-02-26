import torch
from ReadData import read_data, show_image

PATH = '../data/data.csv'
data = read_data(PATH, batch_size=16, number = 3, show = True)

from Discriminator import Discriminator
from Generator import Generator

dis = Discriminator()
gen = Generator()

# dis = torch.load('dis.plk')
# gen = torch.load('gen.plk')

import torch.nn as nn
import torch.optim as opt


criteria = nn.BCELoss()
dis_opt = opt.Adam(dis.parameters())
gen_opt = opt.Adam(gen.parameters())
EPCHO = 50


def train ():

	for epcho in range(EPCHO):
		for real_im,y in data:

			# train real data
			dis_opt.zero_grad()
			is_true = dis(real_im.float())
			true_loss = criteria(is_true, y)
			true_loss.backward()

			# train fake data
			fake_im = gen(5)
			is_fake = dis(fake_im)
			fake_loss = criteria(is_fake, torch.zeros(1,5))
			fake_loss.backward()
			dis_opt.step()

			# train generator
			gen_opt.zero_grad()
			fake_im = gen(5)
			is_fake = dis(fake_im)
			gen_loss = criteria(is_fake, torch.ones(1,5))
			gen_loss.backward()
			gen_opt.step()

		
		print('epcho: {}, true_loss: {:.5}, fake_loss: {:.5}, gen_loss: {:.5}'.format(epcho,true_loss.data,fake_loss.data,gen_loss.data))
		torch.save(dis,'dis.plk')
		torch.save(gen,'gen.plk')

def test ():
	imgs = gen(64)
	show_image(imgs.data)

train()
test()




