import torch
import torch.nn as nn

class Generator(nn.Module):
	def __init__ (self):
		super().__init__()
		self.fc = nn.Linear(100,900)
		self.function = nn.Tanh()
		self.main = nn.Sequential(

				# (input-1)*stride + kernal
				
				nn.Conv2d(1,4,3,1),
				nn.BatchNorm2d(4),
				nn.LeakyReLU(),	

				nn.Conv2d(4,8,5,2),
				nn.BatchNorm2d(8),
				nn.LeakyReLU(),

				nn.ConvTranspose2d(8,6,2,1),
				nn.BatchNorm2d(6),
				nn.LeakyReLU(),

				nn.ConvTranspose2d(6,1,4,1),
				nn.BatchNorm2d(1),
			)

	def forward(self,batch_size=10):
		X = self.normal_noise(batch_size)
		X = self.fc(X).view(batch_size,1,30,30)
		X = self.main(X)
		X = self.function(X)
		return X

	def normal_noise (self,batch_size):
		return torch.randn(batch_size,1,100)

gen = Generator()
gen()