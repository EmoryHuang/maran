from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import utils
from torch import LongTensor as LT
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import index_to_mask


class PoiDataset(Dataset):

    def __init__(self, config, checkins, mapping):
        super(PoiDataset, self).__init__()
        self.config = config
        self.checkins = checkins
        self.mapping = mapping
        self.knn = False

        self.checkins_split()

    def checkins_split(self):
        '''split the checkins (pd.DataFrame) and collect.
        '''

        def collect_fn(df):
            # Groupby user_id and collect trajs, times etc.
            max_sequence_length = self.config.max_sequence_length

            # Create batch ptr
            cnt, remain = divmod(len(df) - 1, max_sequence_length)
            user_start_ptr.extend([len(user)] * cnt)
            batch_length.extend([max_sequence_length] * cnt)
            batch_ptr.extend([len(user) + i * max_sequence_length for i in range(cnt)])
            # if remain:
            #     user_start_ptr.extend([len(user)])
            #     batch_length.extend([remain])
            #     batch_ptr.extend([len(user) + cnt * max_sequence_length])

            user.extend(df['user_id'].values)
            traj.extend(df['location_id'].values)
            time.extend(df['time_unix'].values)
            geo.extend(df['geohash_id'].values)

            cur_user = df.iloc[0]['user_id']
            # Create a user-poi graph and record the interval of the last visit to a location
            traj_graph[cur_user] = self.create_freq_graph(df['location_id'].values, 'loc',
                                                          self.config.alpha)
            geo_graph[cur_user] = self.create_freq_graph(df['geohash_id'].values, 'geo',
                                                         self.config.beta)

            # Create local center sequence
            center_traj.extend(
                self.local_center_seq(traj_graph[cur_user], df['location_id'].values))

        # Groupby user_id and collect trajs, times, categories etc.
        user, traj, time, geo = [], [], [], []
        user_start_ptr, batch_ptr, batch_length = [], [], []
        center_traj = []
        traj_graph, geo_graph = {}, {}
        _ = self.checkins.groupby('user_id').apply(collect_fn)

        self.user, self.traj, self.time, self.geo = LT(user), LT(traj), LT(time), LT(geo)
        self.user_start_ptr, self.batch_ptr, self.batch_length = user_start_ptr, batch_ptr, batch_length
        self.center_traj = LT(center_traj)
        self.traj_graph, self.geo_graph = traj_graph, geo_graph

        if self.knn:
            loc2lat, loc2lon, _ = self.mapping
            cand = []
            for i in range(self.config.max_loc_num):
                cur_lat, cur_lon = loc2lat[i], loc2lon[i]
                dis = utils.cal_distance(cur_lat, cur_lon, loc2lat, loc2lon)
                cand.append(dis.topk(500, largest=False)[1])
            cand = torch.stack(cand)

        self.negihbors_ptr = []
        # Get the POI in each user's neighborhood area
        for idx in range(len(batch_ptr)):
            start, end = batch_ptr[idx], batch_ptr[idx] + batch_length[idx]
            long_start = max(user_start_ptr[idx], end - self.config.long_sequence_length)
            if self.knn:
                long_traj = self.traj[long_start:end]
                negihbors = self.knn_sample(long_traj, cand)
                self.negihbors_ptr.append(negihbors)
                continue

            long_geo = self.geo[long_start:end]
            negihbors = self.get_neighborhood_loc(long_geo,
                                                  self.config.min_neighborhood_num)
            self.negihbors_ptr.append(negihbors)

    def knn_sample(self, loc_seq, cand):
        return torch.concat([cand[loc] for loc in loc_seq.numpy()]).unique()

    def get_neighborhood_loc(self, geo_seq, num=1000):
        '''Get the POI in each user's neighborhood area'''
        _, _, geo2neighbor_loc = self.mapping
        ne_loc = torch.concat([geo2neighbor_loc[geo] for geo in geo_seq.numpy()]).unique()
        # random sample
        if len(ne_loc) < num:
            k = num - len(ne_loc)
            ne_random = np.random.choice(range(1, self.config.max_loc_num), k, False)
            ne_loc = torch.concat([ne_loc, torch.from_numpy(ne_random)]).unique()
        return ne_loc

    def create_freq_graph(self, seq, g_type, rate=0.1):
        seq = LT(seq)
        graph = Data(x=seq.unique())

        # # Constructing visit frequency sequences
        node2idx = {node: i for i, node in enumerate(graph.x.numpy())}
        graph.freq = torch.zeros_like(graph.x)
        for node, cnt in Counter(seq.numpy()).items():
            graph.freq[node2idx[node]] = cnt

        # # Determining frequency thresholds
        while (graph.freq >= len(seq) * rate).sum() == 0:
            rate /= 2
        graph.thr = torch.zeros_like(graph.x) + len(seq) * rate

        if g_type == 'loc':
            loc2lat, loc2lon, _ = self.mapping
            # # Determine the local center of each POI based on distance
            graph.center = torch.zeros_like(graph.x)
            # local center idxs
            c_idxs = (graph.freq >= graph.thr).nonzero().view(-1)
            # latitude and longitude of local center
            c_lons, c_lats = loc2lon[graph.x[c_idxs]], loc2lat[graph.x[c_idxs]]
            for i, node in enumerate(graph.x):
                lon, lat = loc2lon[node], loc2lat[node]
                dis = utils.cal_distance(lat, lon, c_lats, c_lons)
                graph.center[i] = c_idxs[dis.argmin()]
            # edge_index
            c_edge = torch.stack([torch.arange(len(graph.x)), graph.center])
            i_seq = torch.from_numpy(np.vectorize(node2idx.get)(seq))
            s_edge = torch.stack([i_seq[:-1], i_seq[1:]])
            graph.edge_index = torch.concat([c_edge, s_edge], dim=1)
        elif g_type == 'geo':
            i_seq = torch.from_numpy(np.vectorize(node2idx.get)(seq))
            graph.edge_index = torch.stack([i_seq[:-1], i_seq[1:]])
        return graph

    def local_center_seq(self, graph, seq):
        center_mapping = dict(zip(graph.x.numpy(), graph.center.numpy()))
        center_seq = np.vectorize(center_mapping.get)(seq)
        return center_seq

    def __len__(self):
        return len(self.batch_ptr)

    def __getitem__(self, idx):
        start, end = self.batch_ptr[idx], self.batch_ptr[idx] + self.batch_length[idx]
        long_start = max(self.user_start_ptr[idx], end - self.config.long_sequence_length)

        user = self.user[start:end]
        traj = self.traj[start:end]
        time = self.time[start:end]
        geo = self.geo[start:end]
        label_traj = self.traj[start + 1:end + 1]
        label_geo = self.geo[start + 1:end + 1]

        if len(user) != self.config.max_sequence_length:
            padding_length = self.config.max_sequence_length - len(user)
            user = F.pad(user, [0, padding_length])
            traj = F.pad(traj, [0, padding_length])
            time = F.pad(time, [0, padding_length])
            geo = F.pad(geo, [0, padding_length])
            center_traj = F.pad(center_traj, [0, padding_length])
            label_traj = F.pad(label_traj, [0, padding_length])
            label_geo = F.pad(label_geo, [0, padding_length])

        long_traj = self.traj[long_start:start]
        long_time = self.time[long_start:start]
        center_traj = self.center_traj[long_start:start]
        if len(long_traj) != self.config.long_sequence_length:
            padding_length = self.config.long_sequence_length - len(long_traj)
            long_traj = F.pad(long_traj, [0, padding_length])
            long_time = F.pad(long_time, [0, padding_length])
            center_traj = F.pad(center_traj, [0, padding_length])

        dt = torch.stack([abs(long_time - time[i]) for i in range(len(time))])

        # POI index to mask, fills elements of predict with value where mask is True.
        negihbors_mask = ~index_to_mask(self.negihbors_ptr[idx],
                                        size=self.config.max_loc_num)
        negihbors_mask[0] = True

        traj_graph = self.traj_graph[int(user[0])]
        geo_graph = self.geo_graph[int(user[0])]

        return user, traj, geo, center_traj, long_traj, dt, label_traj, label_geo, negihbors_mask, traj_graph, geo_graph
