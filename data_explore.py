import numpy as np
import torch, h5py, os
from collections import namedtuple
import json
import math
import datetime

trainfiles = list(filter(lambda x:x.endswith(".h5"),
                         sorted(os.listdir("/Project0551/jingyi/deepgtt/data/chengdu/"))))


# lon_min = 104.04214
# lat_min = 30.65294
# lon_max = 104.12958
# lat_max = 30.72775
lon_min = 126.506130
lat_min = 45.657920
lon_max = 126.771862
lat_max = 45.830905
dist_all = []
time_all = []
dist_gap_all = []
time_gap_all = []
lngs_all = []
lats_all = []
for fname in trainfiles:
    print("reading file: " + fname)
    with h5py.File("/Project0551/jingyi/deepgtt/data/chengdu/" + fname) as f:
        num_slot = 71
        year = int(fname[:4])
        month = int(fname[4:6])
        date = int(fname[6:8])
#         year = 2000 + int(fname[:2])
#         month = int(fname[2:4])
#         date = int(fname[4:6])
        start_time = datetime.datetime(year, month, date, 0, 0).timestamp()/60
        if len(list(f.keys())) != num_slot:
            num_slot = len(list(f.keys()))
        for slot in range(1, num_slot+1):
            n = f["/{}/ntrips".format(slot)][...]
            if n == 0: 
                continue
            trips = [f["/{}/trip/{}".format(slot, i)][...] for i in range(1, n+1)]
            time = [f["/{}/time/{}".format(slot, i)][...] for i in range(1, n+1)]
            lon = [np.array(f["/{}/lon/{}".format(slot, i)][...]) for i in range(1, n+1)]
            lat = [np.array(f["/{}/lat/{}".format(slot, i)][...]) for i in range(1, n+1)]    
            distance = [f["/{}/distance/{}".format(slot, i)][...][0] for i in range(1, n+1)]

            time_gap = [(np.array(f["/{}/times/{}".format(slot, i)][...])) for i in range(1, n+1)]
            dist_gap = [(np.array(f["/{}/distances/{}".format(slot, i)][...]) / 1000) for i in range(1, n+1)]
            
            dist_gap_sum = [np.concatenate((np.array([0]), np.cumsum(dist_gap[i]))) for i in range(n)]
            
            dist_all.extend(distance)
            time_all.extend(time)
            lngs_all.extend(lon)
            lats_all.extend(lat)
            dist_gap_all.extend(dist_gap_sum)
            time_gap_all.extend(time_gap)

n_trips = len(time_all)
time_all = np.array(time_all)
dist_all = np.array(dist_all)
time_gap_all_adjusted = [(seq - seq[0]) for seq in time_gap_all]
time_gap_all_adjusted = np.concatenate(time_gap_all_adjusted, axis=None)
dist_gap_all = np.concatenate(dist_gap_all, axis=None)
lngs_all = np.concatenate(lngs_all, axis=None)
lats_all = np.concatenate(lats_all, axis=None)

print("total number of trips: " + str(n_trips))
print(time_all[0])
print(time_gap_all_adjusted[0])
print(dist_all[0])
print(dist_gap_all[0])

print("time mean and std:")
print(np.mean(time_all))
print(np.std(time_all))
print("dist mean and std:")
print(np.mean(dist_all))
print(np.std(dist_all))
print("time_gap mean and std:")
print(np.mean(time_gap_all_adjusted))
print(np.std(time_gap_all_adjusted))
print("dist_gap mean and std:")
print(np.mean(dist_gap_all))
print(np.std(dist_gap_all))
print("lngs mean and std:")
print(np.mean(lngs_all) + lon_min)
print(np.std(lngs_all))
print("lats mean and std:")
print(np.mean(lats_all) + lat_min)
print(np.std(lats_all))