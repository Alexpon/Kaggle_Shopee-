import random
import pickle
import pandas as pd

from ipdb import set_trace


df_data = pd.read_csv('train.csv', header=0)
label_group = df_data.groupby('label_group')
print ('# Group =', len(label_group))
print ('Group size', label_group.size())

group_keys = label_group.size().keys()
group_list = []
for key in group_keys:
    group_dir  = {}
    group_items = label_group.get_group(key)
    group_dir['img'] = group_items['image'].to_list()
    group_dir['title'] = group_items['title'].to_list()
    group_list.append(group_dir)

group_size = len(group_list)
random.shuffle(group_list)
train_data = group_list[:-2000]
val_data = group_list[-2000:]

set_trace()
tr_file = open('train_data.pkl', 'wb')
pickle.dump(train_data, tr_file)
tr_file.close()

val_file = open('val_data.pkl', 'wb')
pickle.dump(val_data, val_file)
val_file.close()
