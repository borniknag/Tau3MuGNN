import shutil
import torch
import yaml
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

from models import Model
from utils import Criterion, Writer, log_epoch, load_checkpoint, save_checkpoint, set_seed, get_data_loaders, add_cuts_to_config
from model_efficiency import main as model_eff
from model_efficiency import eval_one_batch, run_one_epoch, generate_roc

from torch_geometric.nn import DataParallel

def bin_acc(preds, labels, threshold=.5):
    preds = torch.where(preds<threshold, 0, 1)
    
    acc = ((preds==labels).sum())/len(preds)
    
    return acc
        
class Tau3MuGNNs:

    def __init__(self, config, device, log_path, setting):
        self.config = config
        self.device = device
        self.log_path = log_path
        self.writer = Writer(log_path)
        self.log_dir=self.writer.log_dir
        
        endcap = config['model']['endcap']
        
        self.data_loaders, x_dim, edge_attr_dim, _ = get_data_loaders(setting, config['data'], config['optimizer']['batch_size'], endcap=endcap)
        
        self.model = Model(x_dim, edge_attr_dim, config['data']['virtual_node'], config['model'])
        self.model.to(self.device)
        num_gpus = torch.cuda.device_count()
        print(f'Training on {num_gpus} GPUs')
        self.model = DataParallel(self.model, device_ids=[i for i in range(num_gpus)])
            
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config['optimizer']['lr'])
        self.criterion = Criterion(config['optimizer'])
        self.node_clf = config['data'].get('node_clf', False)
        
        #print(self.node_clf)
        print(f'[INFO] Number of trainable parameters: {sum(p.numel() for p in self.model.parameters())}')

    @torch.no_grad()
    def eval_one_batch(self, data):
        self.model.eval()
        event_clf_logits = self.model(data)
        y = torch.cat([event.y for event in data]).to(self.device)
        
        event_loss, loss_dict = self.criterion(event_clf_logits.sigmoid(), y)
        
        if self.node_clf:
            node_clf_logits = self.model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch, ptr=data.ptr, node_clf=True)
            node_loss, _ = self.criterion(node_clf_logits.sigmoid(), data.node_label)
            
            loss = event_loss+ self.node_clf*node_loss
            loss_dict['node_focal'] = node_loss
            node_acc = bin_acc(node_clf_logits.sigmoid(), data.node_label)
            loss_dict['node_acc'] = node_acc
        
        return loss_dict, event_clf_logits.data.cpu()
        
    def train_one_batch(self, data):
        self.model.train()
        
        #print('Shape of x: ', data.x.size())
        #print('Shape of edge_index: ', data.edge_index.size())
        #print('Shape of edge_attr: ', data.edge_attr.size())
        #print('Shape of batch: ', data.batch.size())
        #print('Shape of ptr: ', data.ptr.size())
        #print('ptr: ', data.ptr)
        #print('batch: ', data.batch)


        event_clf_logits = self.model(data)
        
        y = torch.cat([event.y for event in data]).to(self.device)
        
        event_loss, loss_dict = self.criterion(event_clf_logits.sigmoid(), y)

        if not hasattr(self, '_diagnostic_printed'):
            print(f"\n=== DIAGNOSTIC INFO ===")
            print(f"Sample predictions (raw logits): {event_clf_logits[:5].detach().cpu()}")
            print(f"Sample predictions (sigmoid): {event_clf_logits.sigmoid()[:5].detach().cpu()}")
            print(f"Sample labels: {y[:5].detach().cpu()}")
            print(f"Are labels 0/1? Unique: {torch.unique(y)}")
            print(f"Prediction mean: {event_clf_logits.sigmoid().mean():.3f}")
            print(f"Label mean: {y.mean():.3f}")
            print(f"Pos samples: {(y==1).sum().item()}, Neg samples: {(y==0).sum().item()}")
            print(f"======================\n")
            self._diagnostic_printed = True 
        
        if self.node_clf:
            node_clf_logits = self.model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch, ptr=data.ptr, node_clf=True)
            node_loss, _ = self.criterion(node_clf_logits.sigmoid(), data.node_label)
            
            loss = event_loss+ self.node_clf*node_loss
            loss_dict['node_focal'] = node_loss

            node_acc = bin_acc(node_clf_logits.sigmoid(), data.node_label)
            loss_dict['node_acc'] = node_acc
        else:
            loss = event_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss_dict, event_clf_logits.data.cpu()

    def run_one_epoch(self, data_loader, epoch, phase):
        loader_len = len(data_loader)
        run_one_batch = self.train_one_batch if phase == 'train' else self.eval_one_batch
        phase = 'test ' if phase == 'test' else phase  # align tqdm desc bar

        all_loss_dict, all_clf_logits, all_clf_labels, all_sample_idxs = {}, [], [], []
        pbar = tqdm(data_loader, total=loader_len)
        for idx, data_list in enumerate(pbar):
            loss_dict, clf_logits = run_one_batch(data_list)
            y = torch.cat([event.y for event in data_list]).cpu()
            sample_idxs = torch.cat([event.sample_idx for event in data_list]).cpu()
            
            desc = log_epoch(epoch, phase, loss_dict, clf_logits, y, True, sample_idxs)
            for k, v in loss_dict.items():
                all_loss_dict[k] = all_loss_dict.get(k, 0) + v
                
            all_clf_logits.extend(list(clf_logits)), all_clf_labels.extend(list(y)), all_sample_idxs.extend(list(sample_idxs))

            if idx == loader_len - 1:
                all_clf_logits, all_clf_labels = torch.cat(all_clf_logits), torch.cat(all_clf_labels)
                for k, v in all_loss_dict.items():
                    all_loss_dict[k] = v / loader_len
                desc, auroc, recall, avg_loss = log_epoch(epoch, phase, all_loss_dict, all_clf_logits, all_clf_labels, False, all_sample_idxs, writer=self.writer)
                
                signal_mask = (all_clf_labels==1)
                bkg_mask = (all_clf_labels==0)
                
                self.writer.add_histogram(f'{phase}/Signal_Predictions', all_clf_logits[signal_mask].sigmoid(),epoch)
                self.writer.add_histogram(f'{phase}/Bkg_Predictions', all_clf_logits[bkg_mask].sigmoid(),epoch)
                
            pbar.set_description(desc)

        return avg_loss, auroc, recall

    def train(self):
        print(self.log_dir)
        start_epoch = 0
        if self.config['optimizer']['resume']:
            start_epoch = load_checkpoint(self.model, self.optimizer, self.log_path, self.device)

        best_val_auc = 0
        best_test_auc = 0
        best_test_auc = 0
        
        best_epoch = 0
        for epoch in range(start_epoch, self.config['optimizer']['epochs'] + 1):
            self.run_one_epoch(self.data_loaders['train'], epoch, 'train')
            valid_auc = self.run_one_epoch(self.data_loaders['valid'], epoch, 'valid')[1]
            
            if epoch % self.config['eval']['test_interval'] == 0:
                test_auc = self.run_one_epoch(self.data_loaders['test'], epoch, 'test')[1]
                if valid_auc > best_val_auc:
                    save_checkpoint(self.model, self.optimizer, self.log_path, epoch)
                    best_val_auc, best_test_auc, best_epoch = valid_auc, test_auc, epoch

            self.writer.add_scalar('best/best_epoch', best_epoch, epoch)
            self.writer.add_scalar('best/best_val_auc', best_val_auc, epoch)
            self.writer.add_scalar('best/best_test_auc', best_test_auc, epoch)
        
            print('-' * 50)
        
        print('Evaluating Performance')

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train Tau3MuGNNs')
    parser.add_argument('--setting', type=str, help='experiment settings', default='GNN_half_dR_1')
    parser.add_argument('--cut', type=str, help='cut id', default=None)
    parser.add_argument('--cuda', type=int, help='cuda device id, -1 for cpu', default=3)
    parser.add_argument('--comment', type=str, help='comment for log')
    parser.add_argument('--search', help='Use directory SearchConfigs instead of configs')
    
    args = parser.parse_args()
    setting = args.setting
    cuda_id = args.cuda
    cut_id = args.cut
    comment = args.comment
    search = args.search
    print(f'[INFO] Running {setting} on cuda {cuda_id} with cut {cut_id}')

    torch.set_num_threads(5)
    set_seed(42)
    time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    
    if not(search):
        config = yaml.safe_load(Path(f'./configs/{setting}.yml').open('r'))
        config = add_cuts_to_config(config, cut_id)
        path_to_config = Path(f'./configs/{setting}.yml')
    else:
        config = yaml.safe_load(Path(f'{setting}').open('r'))
        path_to_config = setting
        config = add_cuts_to_config(config, cut_id)
        comment = setting.replace('/depot/cms/users/simon73/Tau3MuGNNs_newdata/src/SearchConfigs/GNN_half_dR_1_3station_morevars_', '')
        comment = comment.replace('.yml', '')
        setting = 'GNN_half_dR_1_3station_morevars'
    
    print(comment)
    device = torch.device(f'cuda:{cuda_id}' if cuda_id >= 0 else 'cpu')

    log_cut_name = '' if cut_id is None else f'-{cut_id}'
    
    if comment:
        log_name = f'{time}-{setting}{log_cut_name}_{comment}' if not config['optimizer']['resume'] else config['optimizer']['resume']
    else:
        log_name = f'{time}-{setting}{log_cut_name}' if not config['optimizer']['resume'] else config['optimizer']['resume']

    log_path = Path(config['data']['log_dir']) / log_name
    log_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(f'{path_to_config}', log_path / 'config.yml')

    Tau3MuGNNs(config, device, log_path, setting).train()
    
if __name__ == '__main__':
    import os
    os.chdir('./src')
    main()
