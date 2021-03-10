import time
import utils
import datetime
import h5py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import ujson as json

class MySet(Dataset):
    def __init__(self, input_file):
#         self.content = open('./data/' + input_file, 'r').readlines()
#         self.content = map(lambda x: json.loads(x), self.content)
#         self.lengths = map(lambda x: len(x['lngs']), self.content)
        self.lon_min = 104.04214
        self.lat_min = 30.65294
        self.lon_max = 104.12958
        self.lat_max = 30.72775
#         self.lon_min = 126.506130
#         self.lat_min = 45.657920
#         self.lon_max = 126.771862
#         self.lat_max = 45.830905
        self.num_slot = 71
        self.content = []
        dist_all = []
        time_all = []
        dist_gap_all = []
        time_gap_original_all = []
        lngs_all = []
        lats_all = []
        timeID_all = []
#         year = int(input_file[-11:-7])
        year = 2000 + int(input_file[-9:-7])
        month = int(input_file[-7:-5])
        date = int(input_file[-5:-3])
        with h5py.File("/Project0551/jingyi/deepgtt/data/chengdu/" + input_file) as f:
            if len(list(f.keys())) != self.num_slot:
                self.num_slot = len(list(f.keys()))
            for slot in range(1, self.num_slot+1):
                n = f["/{}/ntrips".format(slot)][...]
                if n == 0: 
                    continue
                trips = [f["/{}/trip/{}".format(slot, i)][...] for i in range(1, n+1)]
                time = [f["/{}/time/{}".format(slot, i)][...].item() for i in range(1, n+1)]
                lon = [np.array(f["/{}/lon/{}".format(slot, i)][...]) + self.lon_min for i in range(1, n+1)]
                lat = [np.array(f["/{}/lat/{}".format(slot, i)][...]) + self.lat_min for i in range(1, n+1)]        
                distance = [f["/{}/distance/{}".format(slot, i)][...][0] for i in range(1, n+1)]
                
                time_gap = [np.array(f["/{}/times/{}".format(slot, i)][...]) for i in range(1, n+1)]

                dist_gap = [np.array(f["/{}/distances/{}".format(slot, i)][...])/1000.0 for i in range(1, n+1)]
                
                # generate cumulative sum array from dist gap array
                dist_gap_sum = [np.concatenate((np.array([0]), np.cumsum(dist_gap[i]))) for i in range(n)]
                
                start_time = datetime.datetime(year, month, date, 0, 0).timestamp()/60
                
                for i in range(n):
                    if int(time_gap[i][0] - start_time) < 0 or int(time_gap[i][0] - start_time - 480) >= 1440:
                        print("error in timeID")
                        print(time_gap[i][0])
                        print(start_time)
                        print(int(time_gap[i][0] - start_time))
                
                # remember to minus time by 480 to calculate timeID for harbin dataset
                if input_file[:5] == "train":
                    content = [{"time": time[i], "dist": distance[i], "dist_gap": dist_gap_sum[i], "time_gap": time_gap[i] - time_gap[i][0], "lngs": lon[i].tolist(), "lats": lat[i].tolist(), "driverID": 0, "dateID": 0, "weekID": 0, "timeID": int(time_gap[i][0] - start_time), "avail": [0 for j in range(len(lon[i]))]} for i in range(n)]
                else:
                    content = [{"time": time[i], "dist": distance[i], "dist_gap": dist_gap_sum[i], "time_gap": np.random.uniform(0, 1, len(lon[i])), "lngs": lon[i].tolist(), "lats": lat[i].tolist(), "driverID": 0, "dateID": 0, "weekID": 0, "timeID": int(time_gap[i][0] - start_time), "avail": [0 for j in range(len(lon[i]))]} for i in range(n)]
                
                # filter out trips containing less than 5 gps points
                content = filter(lambda x: len(x['lngs']) >= 5 and len(x['lngs']) <= 200, content)
                
                self.content.extend(content)
        
        self.lengths = map(lambda x: len(x['lngs']), self.content)
        
#         print(type(self.lengths))
#         print(self.content[0]["time"])
#         print(type(self.content[0]["time"]))
        self.lengths = list(self.lengths)
        
        

    def __getitem__(self, idx):
        return self.content[idx]

    def __len__(self):
        return len(self.content)

def collate_fn(data):
    stat_attrs = ['dist', 'time']
    info_attrs = ['driverID', 'dateID', 'weekID', 'timeID']
    traj_attrs = ['lngs', 'lats', 'avail', 'time_gap', 'dist_gap']
#     traj_attrs = ['lngs', 'lats', 'time_gap', 'dist_gap']
    attr, traj = {}, {}
    
    lens = np.asarray([len(item['lngs']) for item in data])

#     print(data)
    for key in stat_attrs:
        x = torch.FloatTensor([item[key] for item in data])
        attr[key] = utils.normalize(x, key)

    for key in info_attrs:
        attr[key] = torch.LongTensor([item[key] for item in data])

    for key in traj_attrs:
#         print(key)
        # pad to the max length
        seqs = np.asarray([item[key] for item in data])
        mask = np.arange(lens.max()) < lens[:, None]
        if key in ['lngs', 'lats', 'time_gap', 'dist_gap']:
            padded = np.zeros(mask.shape, dtype = np.float32)
        else:
            padded = np.zeros(mask.shape, dtype = np.long)
        padded[mask] = np.concatenate(seqs)

        if key in ['lngs', 'lats', 'time_gap', 'dist_gap']:
            padded = utils.normalize(padded, key)

#         padded = torch.from_numpy(padded).float()
        padded = torch.from_numpy(padded)
        traj[key] = padded
        
    if traj['avail'].min() != 0 or traj['avail'].max() != 0:
        print("error!!!!!")
        print(traj['avail'])

    if attr['timeID'].min() < 0 or attr['timeID'].max() >= 1440:
        print("error!!!!!")
        print(attr['timeID'])
                
                
#         print(key)
#         print(traj[key].shape)

    lens = lens.tolist()
    traj['lens'] = lens

    return attr, traj

class BatchSampler:
    def __init__(self, dataset, batch_size):
        self.count = len(dataset)
        self.batch_size = batch_size
        self.lengths = dataset.lengths
        self.indices = list(range(self.count))

    def __iter__(self):
        '''
        Divide the data into chunks with size = batch_size * 100
        sort by the length in one chunk
        '''
        np.random.shuffle(self.indices)

        chunk_size = self.batch_size * 100

        chunks = (self.count + chunk_size - 1) // chunk_size

        # re-arrange indices to minimize the padding
        for i in range(chunks):
            partial_indices = self.indices[i * chunk_size: (i + 1) * chunk_size]
            partial_indices.sort(key = lambda x: self.lengths[x], reverse = True)
            self.indices[i * chunk_size: (i + 1) * chunk_size] = partial_indices

        # yield batch
        batches = (self.count - 1 + self.batch_size) // self.batch_size

        for i in range(batches):
            yield self.indices[i * self.batch_size: (i + 1) * self.batch_size]

    def __len__(self):
        return (self.count + self.batch_size - 1) // self.batch_size

def get_loader(input_file, batch_size):
    dataset = MySet(input_file = input_file)

    batch_sampler = BatchSampler(dataset, batch_size)

    data_loader = DataLoader(dataset = dataset, \
                             batch_size = 1, \
                             collate_fn = lambda x: collate_fn(x), \
                             num_workers = 4,
                             batch_sampler = batch_sampler,
                             pin_memory = True
    )

    return data_loader
