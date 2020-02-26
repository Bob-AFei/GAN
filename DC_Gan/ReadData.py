import matplotlib.pyplot as plt
import torchvision,torch,numpy,pandas
from torch.utils.data import Dataset, DataLoader

# this is for the shrink the size of images
# N_SIZE = 2

class RawData(Dataset):
	def __init__ (self, PATH, number):
		data = pandas.read_csv(PATH).values
		if number == -1: self.X = torch.tensor(data[:,1:])
		else: self.X = torch.tensor(data[data[:,0] == number+1,1:])
		self.length = len(self.X)
		# self.y = torch.tensor(data[:,0]-1)

	def __len__ (self):
		return self.length

	def __getitem__ (self, index):
		return self.X[index].view(1,16,16), torch.ones(1)

def show_image (data):
	images = torchvision.utils.make_grid(data/2+0.5)
	np_img = numpy.transpose(images.numpy(),(1,2,0))
	plt.imshow(np_img)
	plt.axis('off')
	plt.show()

def read_data (PATH, batch_size=10, show=True, number = -1):
	dataset = RawData(PATH,number)
	data = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True, num_workers = 4)
	if show: show_image(iter(data).next()[0])
	return data
