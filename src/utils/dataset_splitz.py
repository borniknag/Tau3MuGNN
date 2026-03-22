# -*- coding: utf-8 -*-

"""
Created on 2021/4/14
@author: Siqi Miao
"""

import os.path as osp
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import permutations, product
try:
    from .root2df import Root2Df
except:
    from utils.root2df import Root2Df

import torch
import torch_geometric
from torch_geometric.nn import radius_graph, knn_graph
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader, DataListLoader
import pickle
import multiprocessing

class Tau3MuDataset(InMemoryDataset):
    def __init__(self, setting, data_config, endcap, debug=False): # Instantiate relevant variables
        self.setting = setting
        self.data_dir = Path(data_config['data_dir'])
        self.conditions = data_config.get('conditions', False)
        self.add_self_loops = data_config.get('add_self_loops', None)
        self.node_feature_names = data_config['node_feature_names']
        self.edge_feature_names = data_config.get('edge_feature_names', [])
        self.only_one_tau = data_config['only_one_tau']
        self.splits = data_config['splits']
        self.pos_neg_ratio = data_config['pos_neg_ratio']
        self.radius = data_config.get('radius', False)
        self.knn = data_config.get('knn', False)
        self.knn_inter = data_config.get('knn_inter', False)
        self.n_hits_min = data_config.get('n_hits_min', 1)
        self.n_hits_max = data_config.get('n_hits_max', np.inf)
        
        global per_station_virtual_node
        per_station_virtual_node = data_config.get('per_station_virtual_node', False)
        
        global signal_dataset ## TODO: Figure out a way to not require these being global variables
        global bkg_dataset
        global far_station
        
        signal_dataset = data_config['signal_dataset']
        bkg_dataset = data_config['bkg_dataset']
        far_station = data_config.get('far_station', False)
        
        if type(self.radius) != list and self.radius != False: # If radius is a list, it will apply a different dR cutoff at each station
            self.radius = [self.radius for i in range(4)]
        
        self.eta_thresh = data_config.get('eta_thresh', False) 
        
        if type(self.eta_thresh) != list:
            self.eta_thresh = [self.eta_thresh for i in range(3)] # If eta_thresh is a list, it will apply a different threshold between 1-2, 2-3, and 3-4
            
        self.virtual_node = data_config.get('virtual_node', False)
        self.cut = data_config.get('cut', False)

        self.debug = debug

        print(f'[INFO] Debug mode: {self.debug}')
        
        print(self.setting)
        print(self.raw_file_names)
        super(Tau3MuDataset, self).__init__(root=self.data_dir)
        self.data, self.slices, self.idx_split = torch.load(self.processed_paths[endcap] , weights_only=False)
        self.x_dim = self.data.x.shape[-1]
        self.edge_attr_dim = self.data.edge_attr.shape[-1] if self.edge_feature_names else 0
        print_splits(self)
    
    @property
    def raw_file_names(self):
        return ['DsTau3muPU0_MTD.pkl', 'DSTau3Mu_pCut1GeV_DF.pkl', 'MinBiasPU200_MTD.pkl'] if 'mix' in self.setting else [signal_dataset, bkg_dataset]

    @property
    def processed_dir(self) -> str:
        cut_id = '-' + self.cut if self.cut else ''
        dataset_name = self.raw_file_names[0].replace('.pkl', '') + '_' + self.raw_file_names[1].replace('.pkl', '')
        return osp.join(self.root, f'processed-{self.setting}{cut_id}_{dataset_name}')

    @property
    def processed_file_names(self):
        if 'half' in self.setting:
            return ['data_pos.pt', 'data_neg.pt']
        else:
            return 'data.pt'

    def download(self):
        print('Please put .pkl or .csv files ino $PROJECT_DIR/data/raw!')
        raise KeyboardInterrupt

    def process(self):
        df = self.get_df()
        
        df = df.sample(frac=1, random_state=42).reset_index(drop=False) # Shuffle the dataset. Set seed to make results reproducible
        df = df.rename(columns={'index': 'og_index'}) # Store the indices of the original dataset for prediction analysis
        
        if self.debug:
            df = df.iloc[:100]
        
        global eta
        global phi
        global r
        global z
        global theta
        
        if 'mu_hit_global_eta' in df.keys():
            eta = 'mu_hit_global_eta'
            phi = 'mu_hit_global_phi'
            r = 'mu_hit_global_r'
            z = 'mu_hit_global_z'
            theta = 'mu_hit_global_theta'
        else:
            eta = 'mu_hit_sim_eta'
            phi = 'mu_hit_sim_phi'
            r = 'mu_hit_sim_r'
            z = 'mu_hit_sim_z'
            theta = 'mu_hit_sim_theta'

        if any('EMTF' in item for item in self.node_feature_names):
            assert 'full' in self.setting, 'half detector setting is currently not supported'
            pt = 'EMTF_mu_pt'
            eta = 'EMTF_mu_eta'
            phi = 'EMTF_mu_phi'
        
        self.feature_names = list(np.unique(self.node_feature_names+self.edge_feature_names))
        
        if 'mu_hit_sim_cosphi' in self.feature_names or 'mu_hit_sim_sinphi' in self.feature_names:
            # Add phi transformations
            r_copy = df[r].to_numpy()
            phi_copy = df[phi].to_numpy()
            cos = []
            sin = []
            x = []
            y = []
            
            for i in range(len(r_copy)):
                
                hit_r = r_copy[i]
                hit_phi = phi_copy[i]
                
                cos.append(np.cos(hit_phi))
                sin.append(np.sin(hit_phi))
                
                x.append(hit_r*np.cos(hit_phi))
                y.append(hit_r*np.sin(hit_phi))
                
            df['mu_hit_sim_cosphi'] = cos
            df['mu_hit_sim_sinphi'] = sin
            df['mu_hit_sim_x'] = x
            df['mu_hit_sim_y'] = y
        
        # TODO: Figure out how to not require these to be global.
        
        if 'half' in self.setting:
            global pos_maxs
            global pos_mins
            global neg_maxs
            global neg_mins
            
            for name in ['mu_hit_nlog_eta', 'mu_hit_nlog_phi', 'mu_hit_dist', 'mu_hit_dot']:
                if name in self.feature_names:
                    self.feature_names.remove(name)
            
            # Need separate mins/maxs for each endcap
            
            pos_maxs = [[] for feature in self.feature_names]
            pos_mins = [[] for feature in self.feature_names]
            
            neg_maxs = [[] for feature in self.feature_names]
            neg_mins = [[] for feature in self.feature_names]
            
            
            #self.feature_names = np.array(feature_names)
    
            
            for i in range(len(df)):
                event = df.iloc[i]
                for j,feature in enumerate(self.feature_names):
                    var = event[feature]

                    endcaps = np.sign(event[z])

                    pos_col = var[endcaps==1]
                    neg_col = var[endcaps==-1]
                    
                    if np.sum(endcaps==1) > 0:
                        pos_maxs[j].append(np.max(pos_col))
                        pos_mins[j].append(np.min(pos_col))
                    
                    if np.sum(endcaps==-1) > 0:
                        neg_maxs[j].append(np.max(neg_col))
                        neg_mins[j].append(np.min(neg_col))
                        
                        
            pos_maxs = np.array(pos_maxs)
            pos_maxs = np.max(pos_maxs, axis=1)
            neg_maxs = np.array(neg_maxs)
            neg_maxs = np.max(neg_maxs, axis=1)
            
            pos_mins = np.array(pos_mins)
            pos_mins = np.min(pos_mins, axis=1)
            neg_mins = np.array(neg_mins)
            neg_mins = np.min(neg_mins, axis=1)
            
            pos_data_list = []
            neg_data_list = []
            data_list = [pos_data_list,neg_data_list]
            
        elif 'full' in self.setting:
            
            global full_maxs
            global full_mins
            
            full_maxs = []
            full_mins = []
            
            for j,feature in enumerate(self.feature_names):
                
                feat = df[feature].to_numpy()
            
                full_maxs.append(np.max(np.concatenate(feat)))
                full_mins.append(np.min(np.concatenate(feat)))
            
            data_list = []
            
        print('[INFO] Processing entries...')
        args_agg = []
    
        for entry in tqdm(df.itertuples(), total=len(df)):
            masked_entry = self.mask_hits(entry, self.conditions, n_hits_min=self.n_hits_min)
            if masked_entry == None: # Don't use events that don't have any hits left after the mask is applied
                continue
            

            if 'GNN_full' in self.setting:
                #args_agg.append(masked_entry)
                
                data_list.append(self._process_one_entry(masked_entry))
            else:
                for i, endcap in enumerate([1,-1]):
                    if 'half' in self.setting:
                        entry = Tau3MuDataset.split_endcap(masked_entry, endcap)
                        if entry == None: # Don't use an endcap that is empty
                            continue
                        else:
                            # half-detector, tau and non-tau endcap
                            data_list[i].append(self._process_one_entry(entry, endcap=endcap))
                    elif 'DT' in self.setting:
                        entry = Tau3MuDataset.split_endcap(masked_entry, endcap)
                        if entry == None:
                            continue
                        else:
                            data = self._process_one_entry(masked_entry, endcap)
                            data_list[i].append(data)
                    
        
        
        if 'half' in self.setting:
            for i in range(2):
                idx_split = Tau3MuDataset.get_idx_split(data_list[i], self.splits, self.pos_neg_ratio)
                data, slices = self.collate(data_list[i])
        
                print('[INFO] Saving data.pt...')
                torch.save((data, slices, idx_split), self.processed_paths[i])
        
        elif 'full' in self.setting:
            #p = multiprocessing.Pool(64)
            #p.starmap_async(self._process_one_entry, args_agg, callback=(lambda x: data_list.append(x)), error_callback=lambda x: print('Failed'))
            #p.close()
            #p.join()  
            
            idx_split = Tau3MuDataset.get_idx_split(data_list, self.splits, self.pos_neg_ratio)
            data, slices = self.collate(data_list)
            print(self.processed_paths)
            print('[INFO] Saving data.pt...')
            torch.save((data, slices, idx_split), self.processed_paths[0])

    def _process_one_entry(self, entry, endcap=0, only_eval=False):
        
        if endcap == 0:
            maxs = full_maxs
            mins = full_mins
        elif endcap == 1:
            maxs = pos_maxs
            mins = pos_mins
        elif endcap == -1:
            maxs = neg_maxs
            mins = neg_mins
        
        if 'GNN' in self.setting:
            edge_index = self.build_graph(entry, self.add_self_loops, self.radius, self.virtual_node, self.eta_thresh, self.knn, self.knn_inter) # Construct graph before min-max norm

            if 'half' in self.setting:
                if entry['n_gen_tau']==1: # If signal event, only return hits on tau endcap
                    if ((entry['gen_tau_eta'] * entry[eta]) > 0).sum() == entry['n_mu_hit']: 
                        only_eval=False
                    else:
                        only_eval=True
            else:
                only_eval = False
            
            for i, feature in enumerate(self.feature_names): # Min-max norm
                entry[feature] = (entry[feature] - mins[i]) / (maxs[i] - mins[i])
            
            edge_attr = Tau3MuDataset.get_edge_features(entry, edge_index, self.edge_feature_names, self.virtual_node)
            x = Tau3MuDataset.get_node_features(entry, self.node_feature_names, self.virtual_node)
            
            
            y = torch.tensor(entry['y']).float().view(-1, 1)
            
            if 'mu_hit_label' in entry:
                if y.item() == 1:
                    if self.virtual_node:
                        node_label = torch.tensor(np.concatenate( (entry['mu_hit_label'],np.array([0])) )).float().view(-1, 1) 
                    else:
                        node_label = torch.tensor(entry['mu_hit_label']).float().view(-1, 1) 
                else:
                    node_label = torch.zeros((x.shape[0], 1)).float()
            else:
                node_label = torch.zeros((x.shape[0], 1)).float()
                
            return Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, node_label=node_label, sample_idx=entry['og_index'], endcap=endcap, only_eval=only_eval)
        else:
            assert 'DT' in self.setting
            
            for i, feature in enumerate(np.unique(self.node_feature_names+self.edge_feature_names)):
                entry[feature] = (entry[feature] - mins[i]) / (maxs[i] - mins[i])
                
            x = Tau3MuDataset.get_node_features(entry, self.node_feature_names, self.virtual_node)
            y = torch.tensor(entry['y']).float().view(-1, 1)
            return Data(x=x, y=y, sample_idx=entry['og_index'], endcap=endcap, only_eval=only_eval)

    # @staticmethod
    # def process_one_entry(entry, setting, add_self_loops, radius, virtual_node, node_feature_names, edge_feature_names):
    #     if 'GNN' in setting:
    #         edge_index = Tau3MuDataset.build_graph(entry, add_self_loops, radius, virtual_node)
    #         edge_attr = Tau3MuDataset.get_edge_features(entry, edge_index, edge_feature_names, virtual_node)
    #         x = Tau3MuDataset.get_node_features(entry, node_feature_names, virtual_node)
    #         y = torch.tensor(entry['y']).float().view(-1, 1)

    #         node_label = None
    #         if 'node_label' in entry:
    #             if y.item() == 1:
    #                 node_label = torch.tensor(entry['node_label']).float().view(-1, 1)
    #             else:
    #                 node_label = torch.zeros((x.shape[0], 1)).float() if not virtual_node else torch.zeros((x.shape[0] - 1, 1)).float()
    #         return Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, node_label=node_label)
    #     else:
    #         assert 'DT' in setting
    #         x = Tau3MuDataset.get_node_features(entry, node_feature_names, virtual_node)
    #         y = torch.tensor(entry['y']).float().view(-1, 1)
    #         return Data(x=x, y=y)

    def get_df_save_path(self):
        save_name = ''
        save_name += 'mix_' if 'mix' in self.setting else 'raw_'
        
        save_name += self.raw_file_names[0].replace('.pkl', '') + '_' + self.raw_file_names[1].replace('.pkl', '') + '_'
        save_name += self.cut if self.cut else 'nocut'
        df_dir = self.data_dir / 'scores' / f'{save_name}'
        df_dir.mkdir(parents=True, exist_ok=True)
        return df_dir / f'{save_name}.pkl'

    def get_df(self):
        df_save_path = self.get_df_save_path()
        if df_save_path.exists():
            print(f'[INFO] Loading {df_save_path}...')
            with open(df_save_path, 'rb') as handle: 
                return pickle.load(handle)
    
        
        dfs = Root2Df(self.data_dir / 'raw').read_df(self.setting)
        neg200 = dfs[self.raw_file_names[1].replace('.pkl', '')]
        pos200 = dfs[self.raw_file_names[0].replace('.pkl', '')]
        pos0 = dfs.get('DsTau3muPU0_MTD', None)
        print(f"\n=== RAW DATA INFO ===")
        print(f"Initial signal events (pos200): {len(pos200)}")
        print(f"Initial background events (neg200): {len(neg200)}")
        print(f"Initial signal ratio: {len(pos200)/len(neg200):.4f}")
        assert self.only_one_tau # Only one-tau is supported
        if self.only_one_tau:
            pos200 = pos200[pos200.n_gen_tau == 1].reset_index(drop=True)
            if pos0 is not None:
                pos0 = pos0[pos0.n_gen_tau == 1].reset_index(drop=True)
            
            print(f"\nAfter only_one_tau filter:")
            print(f"  Signal events: {len(pos200)}")
            print(f"  Background events: {len(neg200)}")
            print(f"  Signal ratio: {len(pos200)/len(neg200):.4f}")
            
            #try:
            #    neg200 = neg200[neg200.n_gen_tau == 1].reset_index(drop=True)
            #except:
            #    pass
            
        if self.cut:
            pos200 = pos200[pos200.apply(lambda x: self.filter_samples(x), axis=1)].reset_index(drop=True)
            if pos0 is not None:
                pos0 = pos0[pos0.apply(lambda x: self.filter_samples(x), axis=1)].reset_index(drop=True)
            print(f"\nAfter cut filter:")
            print(f"  Signal events: {len(pos200)}")
            print(f"  Background events: {len(neg200)}")
            print(f"  Signal ratio: {len(pos200)/len(neg200):.4f}")


        if pos0 is not None and len(pos0) > 100000:
            print('[INFO] Sampling from pos0 to fasten processing & training...')
            pos0 = pos0.sample(100000).reset_index(drop=True)

        if 'mix' in self.setting:
            pos, neg = self.mix(pos0, neg200, pos200, self.setting)
        else:
            pos, neg = pos200, neg200
        
        if 'half' in self.setting:
            min_pos_neg_ratio = len(pos) / (len(neg) * 2)
        else:
            min_pos_neg_ratio = len(pos) / len(neg)
        print(f'[INFO] min_pos_neg_ratio: {min_pos_neg_ratio}')

        pos['y'], neg['y'] = 1, 0
        #assert self.pos_neg_ratio >= min_pos_neg_ratio, f'min_pos_neg_ratio = {min_pos_neg_ratio}! Now pos_neg_ratio = {self.pos_neg_ratio}!'
        print(f"\nFinal (before concat):")
        print(f"  Signal events (pos): {len(pos)}")
        print(f"  Background events (neg): {len(neg)}")
        print(f"  Final signal ratio: {len(pos)/len(neg):.4f}")
        print(f'[INFO] Concatenating pos & neg, saving to {df_save_path}...')
        df = pd.concat((pos, neg), join='outer', ignore_index=True)
        df.to_pickle(df_save_path)
        return df

    def filter_samples(self, x):
        p = np.sqrt(x['gen_mu_e']**2 - 0.1057**2 + 1e-5)
        pt = x['gen_mu_pt']
        abs_eta = np.abs(x['gen_mu_eta'])

        cut_1 = ((p > 2.5).sum() == 3) and ((pt > 0.5).sum() == 3) and ((abs_eta < 2.8).sum() == 3)
        cut_2 = ((pt > 2.0).sum() >= 1) and ((abs_eta < 2.4).sum() >= 1)

        filter_res = True
        if self.cut == 'cut1':
            filter_res *= cut_1
        elif self.cut == 'cut1+2':
            filter_res *= cut_1 * cut_2
        else:
            raise ValueError(f'Unknown filter cut: {self.cut}')

        # filter_res = True
        # if 'cut' in self.filter:
        #     if self.filter['cut'] == 'cut1':
        #         filter_res *= cut_1
        #     elif self.filter['cut'] == 'cut1+2':
        #         filter_res *= cut_1 * cut_2
        #     else:
        #         raise ValueError(f'Unknown filter cut: {self.filter["cut"]}')

        # if 'num_hits' in self.filter:
        #     mask = np.ones(x['n_mu_hit'g], dtype=bool)
        #     for k, v in self.conditions.items():
        #         k = k.split('-')[1]
        #         mask *= eval(f'x["{k}"] {v}')

        #     if isinstance(self.filter['num_hits'], list):
        #         for each_hit_filter in self.filter['num_hits']:
        #             filter_res *= eval('mask.sum()' + each_hit_filter)
        #     else:
        #         filter_res *= eval('mask.sum()' + self.filter['num_hits'])

        return filter_res

    @staticmethod
    def get_intra_station_edges(entry, hit_id, radius=False, knn=False):
        if radius:
            coors = Tau3MuDataset.get_coors_for_hits(entry, hit_id)
            if coors.shape[0] == 0:
                return torch.tensor([]).reshape(2, -1)
            row, col = radius_graph(coors, r=radius, loop=False)  # node id starts from 0
            hit_id = torch.tensor(hit_id)
            row = hit_id[row]  # relabel row
            col = hit_id[col]  # relabel col
            return torch.stack([row, col], dim=0)
        
        elif knn:
            coors = Tau3MuDataset.get_coors_for_hits(entry, hit_id)
            if coors.shape[0] == 0:
                return torch.tensor([]).reshape(2, -1)
            row, col = knn_graph(coors, k=knn, loop=False)  # node id starts from 0
            hit_id = torch.tensor(hit_id)
            row = hit_id[row]  # relabel row
            col = hit_id[col]  # relabel col
            return torch.stack([row, col], dim=0)
            
        else:
            return torch.tensor(list(permutations(hit_id, 2))).T

    @staticmethod
    def get_inter_station_edges(entry, hit_ids, eta_thresh, knn_inter=False):
        # hit_ids: ([idxs of hits in station 1, idxs of hits in station 2, ...])
        # edge_index = list(product(hit_ids[0], hit_ids[1])) + list(product(hit_ids[1], hit_ids[0]))
        
        if eta_thresh:
            station0_coors = Tau3MuDataset.get_coors_for_hits(entry, hit_ids[0])
            station1_coors = Tau3MuDataset.get_coors_for_hits(entry, hit_ids[1])

            edge_index = []
            for i in range(len(station0_coors)):
                for j in range(len(station1_coors)):

                    eta0 = station0_coors[i][0]
                    eta1 = station1_coors[j][0]

                    if abs(eta0-eta1) < eta_thresh:
                        edge_index += [(hit_ids[0][i], hit_ids[1][j]), (hit_ids[1][j], hit_ids[0][i])]
       
        elif knn_inter: # TODO: Make this more efficient. Adds a lot of time to processing
            edge_index = []
            dRs = []
            idxs = []
            station0_coors = Tau3MuDataset.get_coors_for_hits(entry, hit_ids[0])
            station1_coors = Tau3MuDataset.get_coors_for_hits(entry, hit_ids[1])
                
            for i in range(len(station0_coors)):
                for j in range(len(station1_coors)):

                    eta0 = station0_coors[i][0]
                    eta1 = station1_coors[j][0]
                    
                    phi0 = station0_coors[i][1]
                    phi1 = station1_coors[j][1]
                    
                    dRs.append(np.sqrt( (eta0-eta1)**2 + (phi0-phi1)**2 ))
                    idxs.append([i,j])
            
            ranking = np.array(np.argsort(dRs))
            idxs = np.array(idxs)
            knn_idxs = idxs[np.array(ranking[-knn_inter:])]
            
            for idx_pair in knn_idxs:
                i, j = idx_pair
                edge_index += [(hit_ids[0][i], hit_ids[1][j]), (hit_ids[1][j], hit_ids[0][i])]
                        
            
        else:
            edge_index = list(product(hit_ids[0], hit_ids[1])) + list(product(hit_ids[1], hit_ids[0]))
            
        return torch.tensor(edge_index).T

    @staticmethod
    def get_coors_for_hits(entry, hit_id, virtual_node=False):
            
        hit_eta, hit_phi = entry[eta][hit_id], np.deg2rad(entry[phi][hit_id])%(2*np.pi)
        coors = torch.tensor(np.stack((hit_eta, hit_phi)).T)
        return coors

    def build_graph(self, entry, add_self_loops, radius, virtual_node, eta_thresh, knn, knn_inter):
        
        #### IF FULLY CONNECTED, RETURN PERMUTATION GRAPH

        if radius == False:
            node_idxs = [i for i in range(len(entry[self.node_feature_names[0]]))]
            
            if len(node_idxs) == 1:
                return torch.tensor([[0],[0]])
                
            if virtual_node: node_idxs += [len(entry[self.node_feature_names[0]])]
            
            edge_index = torch.tensor(list(permutations(node_idxs, 2))).T
            
            if add_self_loops and edge_index.shape != (0,):
                edge_index, _ = torch_geometric.utils.add_self_loops(edge_index)

            return edge_index
        
        ## IF RADIUS GRAPH
        station2hitids = self.groupby_station(entry['mu_hit_station'])
        
        intra_station_edges = []
        for i, hit_id in enumerate(station2hitids.values()):
            real_edges = self.get_intra_station_edges(entry, hit_id, radius=radius[i], knn=knn)
            
            if per_station_virtual_node:
                virtual_node_id = entry['n_mu_hit'] + (i)
                
                virtual_edges = self.get_virtual_edges(virtual_node_id, hit_id)
                virtual_edges = torch.tensor(virtual_edges).T if len(virtual_edges) != 0 else torch.tensor([]).reshape(2, -1)
                
                intra_station_edges.append(torch.cat((real_edges, virtual_edges), dim=1))
            else:
                intra_station_edges.append(real_edges)
                
        
        intra_station_edges = torch.cat(intra_station_edges, dim=1) if len(intra_station_edges) != 0 else torch.tensor([]).reshape(2, -1)

        # assert torch_geometric.utils.coalesce(intra_station_edges).shape == intra_station_edges.shape

        # We cannot simply iterate four stations since many samples do not hit all the four stations.
        # Some samples may hit station [1, 2, 3], some may hit [1], and some may have hit [1, 2, 4].
        inter_station_edges = []
        ordered_station_ids = sorted(station2hitids.keys())
        for i in range(len(ordered_station_ids) - 1):
            station_0, station_1 = ordered_station_ids[i], ordered_station_ids[i + 1]
            inter_station_edges.append(Tau3MuDataset.get_inter_station_edges(entry, (station2hitids[station_0], station2hitids[station_1]), eta_thresh[i], knn_inter=knn_inter))
        
        if far_station: # Edges for 1->3 and 2->4
            if len(ordered_station_ids) > 2:
            
                for i in range(len(ordered_station_ids) - 2):
                    station_0, station_1 = ordered_station_ids[i], ordered_station_ids[i + 2]
                    inter_station_edges.append(Tau3MuDataset.get_inter_station_edges(entry, (station2hitids[station_0], station2hitids[station_1]), eta_thresh[i+1]))
            
        inter_station_edges = torch.cat(inter_station_edges, dim=1) if len(inter_station_edges) != 0 else torch.tensor([]).reshape(2, -1)
        # assert torch_geometric.utils.coalesce(inter_station_edges).shape == inter_station_edges.shape

        virtual_node_id = len(entry[self.node_feature_names[0]]) + max(entry['mu_hit_station']) if per_station_virtual_node else len(entry[self.node_feature_names[0]])
        
        real_node_ids = [i for i in range(virtual_node_id)]
        virtual_edges = Tau3MuDataset.get_virtual_edges(virtual_node_id, real_node_ids)
        virtual_edges = torch.tensor(virtual_edges).T if len(virtual_edges) != 0 else torch.tensor([]).reshape(2, -1)
        # assert torch_geometric.utils.coalesce(virtual_edges).shape == virtual_edges.shape

        if virtual_node:
            edge_index = torch.cat((intra_station_edges, inter_station_edges, virtual_edges), dim=1).long()
        else:
            edge_index = torch.cat((intra_station_edges, inter_station_edges), dim=1).long()

        if add_self_loops and edge_index.shape != (0,):
            edge_index, _ = torch_geometric.utils.add_self_loops(edge_index)
        # assert torch_geometric.utils.coalesce(edge_index).shape == edge_index.shape
        return edge_index

    @staticmethod
    def get_virtual_edges(virtual_node_id, real_node_ids):
        return list(product([virtual_node_id], real_node_ids)) + list(product(real_node_ids, [virtual_node_id]))

    @staticmethod
    def groupby_station(stations: np.ndarray) -> dict:
        station2hitids = {}
        for hit_id, station_id in enumerate(stations):
            if station2hitids.get(station_id) is None:
                station2hitids[station_id] = []
            station2hitids[station_id].append(hit_id)
        return station2hitids

    @staticmethod
    def get_node_features(entry, feature_names, virtual_node):
        if entry['n_mu_hit'] == 0:
            virtual_node = True
        
        # Directly index the entry using features = entry[feature_names] is extremely slow!
        features = np.stack([entry[feature] for feature in feature_names], axis=1)
        
        if per_station_virtual_node:
            for i in range(max(entry['mu_hit_station'])):
                features = np.concatenate((features, np.zeros((1, features.shape[1]))), axis=0)
                
        if virtual_node:
            features = np.concatenate((features, np.zeros((1, features.shape[1]))))
        return torch.tensor(features, dtype=torch.float)

    @staticmethod
    def get_edge_features(entry, edge_index, feature_names, virtual_node):
        if edge_index.shape == (2, 0):
            return torch.tensor([])
        
        feature_names_copy = feature_names.copy()

        for name in feature_names:
            if name in ['mu_hit_dR', 'mu_hit_dist', 'mu_hit_dot', 'mu_hit_nlog_phi', 'mu_hit_nlog_eta']:
                feature_names_copy.remove(name)
        
        #print(entry.keys())
        # Directly index the entry using features = entry[feature_names] is extremely slow!
        features = np.stack([entry[feature] for feature in feature_names_copy], axis=1)
        
        
        if per_station_virtual_node:
            for i in range(max(entry['mu_hit_station'])):
                features = np.concatenate((features, np.zeros((1, features.shape[1]))), axis=0)
            
        if virtual_node:
            # Initialize the feature of the virtual node with all zeros.
            features = np.concatenate((features, np.zeros((1, features.shape[1]))), axis=0)
        
        #print(features)
        #print(np.shape(features))
        #print(edge_index)
        edge_features = features[edge_index[0]] - features[edge_index[1]]
        
        if 'mu_hit_dR' in feature_names: # TODO: Make this more efficicient
            
             hit_eta, hit_phi = entry[eta], np.deg2rad(entry[phi])%(2*np.pi)
             if virtual_node:
                 hit_eta, hit_phi = np.append(hit_eta, 0), np.append(hit_phi, 0)
             dR = (hit_eta[edge_index[0]] - hit_eta[edge_index[1]])**2 + (hit_phi[edge_index[0]] - hit_phi[edge_index[1]])**2
             dR = dR**0.5
             edge_features = np.concatenate((edge_features, dR.reshape(-1, 1)), axis=1)
        
        if 'mu_hit_dist' in feature_names or 'mu_hit_dot' in feature_names:
            
            x = entry['mu_hit_sim_x']
            y = entry['mu_hit_sim_y']
            z = entry['mu_hit_sim_z']
            
            if virtual_node:
                 x, y, z = np.append(x, 0), np.append(y, 0), np.append(z, 0)
            
            if 'mu_hit_dist' in  feature_names:
                dist = 0
                for q in [x,y,z]:
                    dist += (q[edge_index[0]] - q[edge_index[1]])**2
                
                dist = np.sqrt(dist)
                edge_features = np.concatenate((edge_features, dist.reshape(-1, 1)), axis=1)
            
            if 'mu_hit_dot' in feature_names:
                dot = 0
                for q in [x,y,z]:
                    dot += q[edge_index[0]]*q[edge_index[1]]
                
                edge_features = np.concatenate((edge_features, dot.reshape(-1, 1)), axis=1)
        
        if 'mu_hit_nlog_phi' in feature_names:
            
            hit_phi = entry[phi] 
            if per_station_virtual_node:
                phi = np.concatenate((hit_phi, [0 for i in range(max(entry['mu_hit_station']))]))
                
            if virtual_node:
                hit_phi = np.append(hit_phi,0)
            
                
            dphi = np.abs( hit_phi[edge_index[0]] - hit_phi[edge_index[1]] ) + 1e-7
            nlog = -np.log(dphi)
            
            edge_features = np.concatenate((edge_features, nlog.reshape(-1,1)), axis=1)
        
        if 'mu_hit_nlog_eta' in feature_names:
            
            hit_eta = entry[eta] 
            
            if per_station_virtual_node:
                hit_eta = np.concatenate((hit_eta, [0 for i in range(max(entry['mu_hit_station']))]))
                
            if virtual_node:
                hit_eta = np.append(hit_eta,0)
            
            deta = np.abs( hit_eta[edge_index[0]] - hit_eta[edge_index[1]] ) + 1e-7
            nlog = -np.log(deta)
            
            edge_features = np.concatenate((edge_features, nlog.reshape(-1,1)), axis=1)
            
        return torch.tensor(edge_features, dtype=torch.float)

    @staticmethod
    def get_idx_split(data_list, splits, pos2neg):
        np.random.seed(42)
        assert sum(splits.values()) == 1.0
        y_dist = np.array([data.y.item() for data in data_list])
        only_eval = np.array([data.only_eval for data in data_list])

        pos_idx = np.argwhere((y_dist == 1) & (~only_eval)).reshape(-1)  # if is half-detector model, do not use non-tau endcap for training
        neg_idx = np.argwhere((y_dist == 0) & (~only_eval)).reshape(-1)
        only_eval_idx = np.argwhere(only_eval).reshape(-1)
        
        print('Minimum pos2neg:', len(pos_idx)/len(neg_idx))
        
        try:
            assert len(pos_idx) <= len(neg_idx) * pos2neg, 'The number of negative samples is not enough given the pos_neg_ratio!'
        except:
            print('Number of negative samples not enough. Using minimum possible')
            pos2neg = len(pos_idx)/len(neg_idx)
        
        n_train_pos, n_valid_pos = int(splits['train'] * len(pos_idx)), int(splits['valid'] * len(pos_idx))
        n_train_neg, n_valid_neg = int(n_train_pos / pos2neg), int(n_valid_pos / pos2neg)

        pos_train_idx = pos_idx[:n_train_pos]
        pos_valid_idx = pos_idx[n_train_pos:n_train_pos + n_valid_pos]
        pos_test_idx = pos_idx[n_train_pos + n_valid_pos:]

        neg_train_idx = neg_idx[:n_train_neg]
        neg_valid_idx = neg_idx[n_train_neg:n_train_neg + n_valid_neg]
        neg_test_idx = neg_idx[n_train_neg + n_valid_neg:]

        return {'train': np.concatenate((pos_train_idx, neg_train_idx)).tolist(),
                'valid': np.concatenate((pos_valid_idx, neg_valid_idx)).tolist(),
                'test': np.concatenate((pos_test_idx, neg_test_idx, only_eval_idx)).tolist()}

    @staticmethod
    def mix(pos0, neg200, pos200, setting):
        neg_idx = np.arange(len(neg200))
        np.random.shuffle(neg_idx)

        # first len(pos0) neg data will be used as noise in pos0
        noise_in_pos0 = neg200.loc[neg_idx[:len(pos0)]].reset_index(drop=True)
        # remaining neg data will remain negative
        neg200 = neg200.loc[neg_idx[len(pos0):]].reset_index(drop=True)

        print('[INFO] Mixing data...')
        # mixed_pos = noise_in_pos0
        mixed_pos = []
        for idx, entry in tqdm(pos0.iterrows(), total=len(pos0)):
            for k, v in entry.items():
                if 'gen' in k:  # directly keep gen variables
                    continue
                elif isinstance(v, int):  # accumulate n_mu_hit
                    assert k == 'n_mu_hit'
                    entry['node_label'] = np.concatenate((np.zeros(noise_in_pos0.iloc[idx][k]), np.ones(v)))
                    entry[k] += noise_in_pos0.iloc[idx][k]
                else:  # concat hit features
                    assert isinstance(v, np.ndarray)
                    mixed_hits = np.concatenate((noise_in_pos0.iloc[idx][k], v))
                    entry[k] = mixed_hits
            mixed_pos.append(entry.values)
        mixed_pos = pd.DataFrame(data=mixed_pos, columns=entry.index)

        if 'check' in setting:
            return mixed_pos, pos200
        elif 'sanity' in setting:
            return noise_in_pos0, neg200
        else:
            return mixed_pos, neg200

    @staticmethod
    def split_endcap(masked_entry, endcap):
        

        entry = {}
        endcap_idx = np.sign(masked_entry[z]) == endcap

        for k, v in masked_entry.items():
            if isinstance(v, np.ndarray) and 'gen' not in k and k != 'y' and 'L1' not in k and 'EMTF' not in k:
                assert v.shape[0] == masked_entry['n_mu_hit']
                entry[k] = v[endcap_idx]
            else:
                entry[k] = v
        
        entry['n_mu_hit'] = endcap_idx.sum().item()
        
        if entry['n_mu_hit'] < 1: # Only return an entry if it has hits left
            return None
        
        if masked_entry['y'] == 1: # If signal event, only return hits on tau endcap
            if ((masked_entry['gen_tau_eta'] * entry[eta]) > 0).sum() == entry['n_mu_hit']: 
                entry['y'] = 1
            else:
                entry['y'] = 0
        
        return entry

    def mask_hits(self, entry, conditions, n_hits_min=1):
        n_mu_hit = eval(f'len(entry.{self.node_feature_names[0]})')
        
        if n_mu_hit == 0: return None
        
        if conditions == False:
            masked_entry = {'n_mu_hit': n_mu_hit}
            for k in entry._fields:
                value = getattr(entry, k)
                if isinstance(value, np.ndarray):
                    masked_entry[k] = value.reshape(-1)
                else:
                    if k != 'n_mu_hit':
                        masked_entry[k] = value
            
            return masked_entry
        
        mask = np.ones(n_mu_hit, dtype=bool)
        for k, v in conditions.items():
            k = k.split('-')[1]
            assert isinstance(getattr(entry, k), np.ndarray)
            mask *= eval('entry.' + k + v)
        
        
        n_mu_hit = mask.sum()
        masked_entry = {'n_mu_hit': n_mu_hit}
        
        if not(mask.sum() >= n_hits_min): # Only return an entry if it has hits left
            return None
        
        for k in entry._fields:
            value = getattr(entry, k)
            if isinstance(value, np.ndarray) and 'gen' not in k and k != 'y' and 'L1' not in k and k != 'n_' not in k:
                try: masked_entry[k] = value[mask].reshape(-1)
                except:
                    continue
            else:
                if k != 'n_mu_hit':
                    masked_entry[k] = value
        return masked_entry



def get_data_loaders(setting, data_config, batch_size, endcap=1):
    
    if endcap == 1 or endcap==0:
        idx = 0
    elif endcap == -1:
        idx = 1
    
    dataset = Tau3MuDataset(setting, data_config, idx)
    print('Retrieving Data Loaders from:'+dataset.processed_paths[idx])
    train_loader = DataListLoader(dataset[dataset.idx_split['train']], batch_size=batch_size, shuffle=True)
    valid_loader = DataListLoader(dataset[dataset.idx_split['valid']], batch_size=batch_size, shuffle=False)
    test_loader = DataListLoader(dataset[dataset.idx_split['test']], batch_size=batch_size, shuffle=False)
    return {'train': train_loader, 'valid': valid_loader, 'test': test_loader}, dataset.x_dim, dataset.edge_attr_dim, dataset


def print_splits(dataset):
    def get_pos_neg_count(y):
        pos_y = (y == 1).sum()
        neg_y = (y == 0).sum()
        pos2neg_ratio = pos_y / neg_y
        return pos_y, neg_y, pos2neg_ratio

    print('[Splits]')
    for k, v in dataset.idx_split.items():
        y = dataset.data.y[v]
        pos_y, neg_y, pos2neg_ratio = get_pos_neg_count(y)
        print(f'    {k}: {len(v)}. # pos: {pos_y}, # neg: {neg_y}. Pos:Neg: {pos2neg_ratio:.3f}')
