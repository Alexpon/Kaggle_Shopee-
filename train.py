import pickle
import torch

from dataset import myDataset
from torch.utils.data import DataLoader

from ipdb import set_trace


BATCH = 4

def main():
    r_file = open('train_data.pkl', 'rb')
    train_data = pickle.load(r_file)
    dataset = myDataset(train_data, image_dir='./train_images')
    dataloader = DataLoader(dataset, shuffle=True, batch_size=BATCH)

    for pos_img, neg_img in dataloader:
        set_trace()
        pass

if __name__ == '__main__':
    main()
