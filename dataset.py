import random
from torch.utils.data import Dataset
from ipdb import set_trace


class myDataset(Dataset):
    def __init__(self, group_list, image_dir):
        self.group_list = group_list
    
    def __getitem__(self, index):
        
        num_pos = len(self.group_list[index]['img'])
        select_id = random.sample(range(num_pos), k=2)
        pos_imgs = [self.group_list[index]['img'][id] for id in select_id]

        neg_imgs = []
        groups_for_neg = random.sample(self.group_list, k=2)
        num_neg1 = len(groups_for_neg[0]['img'])
        select_id = random.choice(range(num_neg1))
        neg_imgs.append(groups_for_neg[0]['img'][select_id])
        num_neg2 = len(groups_for_neg[1]['img'])
        select_id = random.choice(range(num_neg2))
        neg_imgs.append(groups_for_neg[1]['img'][select_id])

        return pos_imgs, neg_imgs

    def __len__(self):
        return len(self.group_list)