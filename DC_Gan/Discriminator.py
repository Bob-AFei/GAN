import torch
import torch.nn as nn

class Discriminator (nn.Module):
	def __init__ (self):
		super().__init__()
		self.main = nn.Sequential(
				nn.Conv2d(1,2,1,1),
				nn.BatchNorm2d(2),
				nn.LeakyReLU(),

				nn.Conv2d(2,3,3,1),
				nn.BatchNorm2d(3),
				nn.LeakyReLU(),

				nn.Conv2d(3,3,3,2),
				nn.BatchNorm2d(3),
				nn.LeakyReLU(),

				nn.Conv2d(3,1,3,1),
				nn.BatchNorm2d(1),
				nn.LeakyReLU()
			)
		# to return one value which indicates the reality
		self.fc = nn.Linear(16,1)
		self.function = nn.Sigmoid()

	def forward(self, X):
		X = self.main(X)
		X = X.view(-1,16)
		X = self.fc(X)
		X = self.function(X)
		return X