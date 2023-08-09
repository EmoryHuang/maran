from pathlib import Path

import pandas as pd
import torch
import utils
from dataset import PoiDataset
from torch_geometric.loader import DataLoader


class Poidataloader():
    '''load poi data from Gowalla/Foursquare data file
    '''

    def __init__(self, config):
        self.config = config
        self.database = Path(config.database)
        if not self.database.exists():
            self.database.mkdir()

    def load(self, dataset, file):
        self.checkins = self.load_checkins(dataset, file)

        self.dataset = dataset
        setattr(self.config, 'max_user_num', self.user_count() + 1)
        setattr(self.config, 'max_loc_num', self.location_count() + 1)
        setattr(self.config, 'max_geo_num', self.geohash_count() + 1)
        return self.checkins

    def create_POI_mapping(self):
        # Create a global mapping of geohash and category,
        # then create a global graph based on location, geohash, category
        # (Before dividing the dataset and sequence length limits)
        def split_loc(df):
            loc2lat.append(df['latitude'].values[0])
            loc2lon.append(df['longitude'].values[0])

        loc2lat, loc2lon = [0], [0]
        _ = self.checkins.groupby('location_id').apply(split_loc)
        loc2lat, loc2lon = torch.Tensor(loc2lat), torch.Tensor(loc2lon)

        # Create a mapping relationship between geohash and geohash_id
        geo2id = self.checkins.set_index('geohash')['geohash_id'].to_dict()

        # Get the POIs contained in each region
        def geo2loc_collect(df):
            geohash, geohash_id = df.iloc[0][['geohash', 'geohash_id']]
            geo2loc[geohash_id] = torch.from_numpy(df['location_id'].unique())

            geo2neighbor[geohash_id] = [
                geo2id[ne] for ne in utils.geohash_neighbors(geohash)
                if ne in geo2id.keys()
            ]

        geo2loc, geo2neighbor = {}, {}
        _ = self.checkins.groupby('geohash').apply(geo2loc_collect)
        geo2neighbor_loc = {}
        for key, val in geo2neighbor.items():
            geo2neighbor_loc[key] = torch.concat([geo2loc[geo] for geo in val]).unique()
        return loc2lat, loc2lon, geo2neighbor_loc

    def create_dataset(self, mode, dataset):
        dataset_train_path = self.database / f'dataset_{dataset}_train.pkl'
        dataset_test_path = self.database / f'dataset_{dataset}_test.pkl'
        dataset_static_path = self.database / f'dataset_{dataset}_static.pkl'
        if dataset_static_path.exists():
            if mode == 'train':
                self.dataset_train = torch.load(dataset_train_path)
            self.dataset_test = torch.load(dataset_test_path)
            self.dataset_static = torch.load(dataset_static_path)
            self.static_dataloader()
            return

        mapping = self.create_POI_mapping()

        # Dividing the data set into training and test sets
        checkins_train, checkins_test = utils.split_user_train_test(self.checkins, 0.8)
        # checkins_val, checkins_test = utils.split_user_train_test(checkins_test, 0.5)

        # Create dataset
        if mode == 'train':
            self.dataset_train = PoiDataset(self.config, checkins_train.copy(), mapping)
            torch.save(self.dataset_train, dataset_train_path)
        self.dataset_test = PoiDataset(self.config, checkins_test.copy(), mapping)
        torch.save(self.dataset_test, dataset_test_path)
        # self.dataset_val = PoiDataset(self.config, checkins_val.copy())
        # torch.save(self.dataset_val, dataset_val_path)

        self.dataset_static = torch.LongTensor(
            [self.config.max_user_num, self.config.max_loc_num, self.config.max_geo_num])
        torch.save(self.dataset_static, dataset_static_path)
        self.static_dataloader()

    def train_dataloader(self):
        return DataLoader(dataset=self.dataset_train,
                          batch_size=self.config.batch_size,
                          pin_memory=True,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.dataset_test,
                          batch_size=self.config.batch_size,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(dataset=self.dataset_test,
                          batch_size=self.config.batch_size,
                          shuffle=False)

    def static_dataloader(self):
        setattr(self.config, 'max_user_num', int(self.dataset_static[0]))
        setattr(self.config, 'max_loc_num', int(self.dataset_static[1]))
        setattr(self.config, 'max_geo_num', int(self.dataset_static[2]))

    def user_count(self):
        return len(self.checkins['user_id'].unique())

    def location_count(self):
        return len(self.checkins['location_id'].unique())

    def geohash_count(self):
        return len(self.checkins['geohash_id'].unique())

    def checkins_count(self):
        return len(self.checkins)

    def load_checkins(self, dataset, file):
        if (self.database / f'checkins_{dataset}.pkl').exists():
            checkins = pd.read_pickle(self.database / f'checkins_{dataset}.pkl')
            return checkins

        checkins = pd.read_csv(
            file, sep='\t', names=['user', 'time', 'latitude', 'longitude', 'location'])
        checkins['time'] = pd.to_datetime(checkins['time'], errors='coerce')
        checkins = checkins.dropna().drop_duplicates()
        checkins.reset_index(drop=True, inplace=True)
        checkins = utils.geohash_encode(checkins)
        checkins = self.__convert(checkins)
        checkins.to_pickle(self.database / f'checkins_{dataset}.pkl')
        return checkins

    def __convert(self, checkins):

        def item2id(checkins, column):
            # id start from 1
            item = checkins[column].unique()
            item2id = dict(zip(item, range(1, item.size + 1)))
            checkins.insert(checkins.shape[1], f'{column}_id',
                            checkins[column].map(item2id))
            return checkins, item2id

        # user -> user_id, location -> location_id
        checkins, user2id = item2id(checkins, 'user')
        checkins, location2id = item2id(checkins, 'location')
        checkins, geohash2id = item2id(checkins, 'geohash')
        # time -> time_unix
        checkins.insert(checkins.shape[1], 'time_unix',
                        checkins['time'].astype('int') // 10**9)

        checkins = checkins.sort_values(['user_id', 'time_unix'])
        checkins.reset_index(drop=True, inplace=True)
        return checkins