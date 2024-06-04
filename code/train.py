import os
import nni
import csv
import time
import json
import torch
import random
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from model import DDG_RDE_Network, Codebook
from trainer import CrossValidation, recursive_to
from dataset import SkempiDatasetManager, get_SARS_data
from utils import set_seed, check_dir, eval_skempi_three_modes, save_code, load_config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pretrain(epoch, model, optimizer):

    model.train()

    batch = recursive_to(next(dataloader.pretrain_loader), device)
    _, e_q_loss, s_recon_loss, h_recon_loss, x_recon_loss = model(batch)
    loss_vae = e_q_loss * args.loss_weight + s_recon_loss + h_recon_loss + x_recon_loss

    optimizer.zero_grad()
    loss_vae.backward()
    optimizer.step()

    if epoch % 100 == 1:
        print("\033[0;30;43m{} | [pretrain] Epoch {} | Train Loss: {:.5f} | {:.5f} {:.5f} {:.5f} {:.8f}\033[0m".format(time.strftime("%Y-%m-%d %H-%M-%S"), epoch, loss_vae.item(), e_q_loss.item(), s_recon_loss.item(), h_recon_loss.item(), x_recon_loss.item()))
        log_file.write("{} | [pretrain] Epoch {} | Train Loss: {:.5f} | {:.5f} {:.5f} {:.5f} {:.8f}\n".format(time.strftime("%Y-%m-%d %H-%M-%S"), epoch, loss_vae.item(), e_q_loss.item(), s_recon_loss.item(), h_recon_loss.item(), x_recon_loss.item()))
        log_file.flush()


def train(epoch):
    for fold in range(args.num_cvfolds):
        model, optimizer, _ = cv_mgr.get(fold)

        model.train()

        batch = recursive_to(next(dataloader.get_train_loader(fold)), device)
        if args.pre_epoch != 0:
            loss, _ = model(batch, vae_model)
        else:
            loss, _ = model(batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        optimizer.step()
        optimizer.zero_grad()

        if epoch % 100 == 1:
            print("\033[0;30;46m{} | [train] Epoch {} Fold {} | Loss {:.8f}\033[0m".format(time.strftime("%Y-%m-%d %H-%M-%S"), epoch, fold, loss.item()))
            log_file.write("{} | [train] Epoch {} Fold {} | Loss {:.8f}\n".format(time.strftime("%Y-%m-%d %H-%M-%S"), epoch, fold, loss.item()))
            log_file.flush()


def validate(save=False):
    results = []
    val_loss_list = []

    with torch.no_grad():
        for fold in range(args.num_cvfolds):
            model, _, _ = cv_mgr.get(fold)

            model.eval()

            for i, batch in enumerate(tqdm(dataloader.get_val_loader(fold), desc='validate', dynamic_ncols=True)):
                batch = recursive_to(batch, device)
                if args.pre_epoch != 0:
                    loss, output_dict = model(batch, vae_model)
                else:
                    loss, output_dict = model(batch)
                val_loss_list.append(loss.item())

                for complex, num_muts, ddg_true, ddg_pred in zip(batch['complex'], batch['num_muts'], output_dict['ddG_true'], output_dict['ddG_pred']):
                    results.append({
                        'complex': complex,
                        'num_muts': num_muts,
                        'ddG': ddg_true.item(),
                        'ddG_pred': ddg_pred.item()
                    })
    
    results = pd.DataFrame(results)
    results['method'] = 'RDE'
    if save is True:
        results.to_csv(os.path.join(save_dir, 'results.csv'), index=False)
    df_metrics = eval_skempi_three_modes(results).to_numpy()

    print("\033[0;30;43m {} | [val] A-Pea {:.6f} A-Spe {:.6f} | RMSE {:.6f} MAE {:.6f} AUROC {:.6f} | P-Pea {:.6f} P-Spe {:.6f}\033[0m".format(time.strftime("%Y-%m-%d %H-%M-%S"), 
                                df_metrics[0][1], df_metrics[0][2], df_metrics[0][3], df_metrics[0][4], df_metrics[0][5], df_metrics[0][6], df_metrics[0][7]))
    log_file.write("{} | [val] A-Pea {:.6f} A-Spe {:.6f} | RMSE {:.6f} MAE {:.6f} AUROC {:.6f} | P-Pea {:.6f} P-Spe {:.6f}\n".format(time.strftime("%Y-%m-%d %H-%M-%S"), 
                                df_metrics[0][1], df_metrics[0][2], df_metrics[0][3], df_metrics[0][4], df_metrics[0][5], df_metrics[0][6], df_metrics[0][7]))
    log_file.flush()

    for fold in range(args.num_cvfolds):
        _, _, scheduler = cv_mgr.get(fold)
        scheduler.step(np.mean(val_loss_list))
    
    return df_metrics


def pointmut_analysis():
    results = []

    with torch.no_grad():
        for batch in tqdm(mutloader):
            batch = recursive_to(batch, device)

            for fold in range(args.num_cvfolds):
                model, _, _ = cv_mgr.get(fold)
                model.eval()

                if args.pre_epoch != 0:
                    _, output_dict = model(batch, vae_model)
                else:
                    _, output_dict = model(batch)

                for mutstr, ddG_pred in zip(batch['mutstr'], output_dict['ddG_pred'].cpu().tolist()):
                    results.append({
                        'mutstr': mutstr,
                        'ddG_pred': ddG_pred,
                    })
    
    results = pd.DataFrame(results)
    results = results.groupby('mutstr').mean().reset_index()
    results['rank'] = results['ddG_pred'].rank() / len(results)
    if 'interest' in config and config.interest:
        print(results[results['mutstr'].isin(config.interest)])

    print("\033[0;30;43m {} | TH31W {:.5f} AH53F {:.5f} NH57L {:.5f} RH103M {:.5f} LH104F {:.5f}\033[0m".format(time.strftime("%Y-%m-%d %H-%M-%S"), 
                                results['rank'][416], results['rank'][3], results['rank'][294], results['rank'][352], results['rank'][232]))
    log_file.write("{} | TH31W {:.5f} AH53F {:.5f} NH57L {:.5f} RH103M {:.5f} LH104F {:.5f}\n".format(time.strftime("%Y-%m-%d %H-%M-%S"), 
                                results['rank'][416], results['rank'][3], results['rank'][294], results['rank'][352], results['rank'][232]))
    log_file.flush()

    return results



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_cvfolds', type=int, default=3) 
    parser.add_argument('--patch_size', type=int, default=128)

    parser.add_argument('--node_feat_dim', type=int, default=128)
    parser.add_argument('--pair_feat_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=3)

    parser.add_argument('--knn', type=int, default=8)
    parser.add_argument('--cutoff', type=float, default=12.0)
    parser.add_argument('--num_embeddings', type=int, default=512)
    parser.add_argument('--loss_weight', type=float, default=0.01)
    parser.add_argument('--commitment_cost', type=float, default=0.25)
    parser.add_argument('--mask_ratio', type=float, default=0.20)

    parser.add_argument('--pre_epoch', type=int, default=2000)
    parser.add_argument('--max_epoch', type=int, default=10000)
    parser.add_argument('--val_freq', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--ckpt_path', type=str, default=None)
    args = parser.parse_args()
    param_s = args.__dict__
    param = json.loads(open("../configs/param_configs.json", 'r').read())
    param.update(nni.get_next_parameter())
    args = argparse.Namespace(**param)
    config, _ = load_config('../data/7FAE_RBD_Fv_mutation.yml')

    set_seed(args.seed)

    timestamp = time.strftime("%Y-%m-%d %H-%M-%S") + f"-%3d" % ((time.time() - int(time.time())) * 1000)
    save_dir = os.path.join('../results/', timestamp)
    check_dir(os.path.join(save_dir, 'checkpoint'))
    log_file = open(os.path.join(save_dir, "train_log.txt"), 'a+')
    save_code(save_dir)
    with open(os.path.join(save_dir, 'train_config.json'), 'w') as fout:
        json.dump(args.__dict__, fout, indent=2)

    print('Loading datasets...')
    dataloader = SkempiDatasetManager(config=args, num_cvfolds=args.num_cvfolds, num_workers=4)
    mutloader = get_SARS_data(args.batch_size, config)

    print('Building model...')
    cv_mgr = CrossValidation(config=args, num_cvfolds=args.num_cvfolds, model_factory=DDG_RDE_Network).to(device)

    if param_s['ckpt_path'] is not None:
        args.pre_epoch = args.max_epoch = 1

    if args.pre_epoch != 0:

        vae_model = Codebook(args).to(device)
        vae_optimizer = torch.optim.Adam(vae_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        for epoch in range(1, args.pre_epoch+1):
            pretrain(epoch, vae_model, vae_optimizer)

        for p in vae_model.parameters():
            p.requires_grad_(False)

        torch.save(vae_model.state_dict(), os.path.join(save_dir, 'checkpoint', f'vae_model.ckpt'))


    for epoch in range(1, args.max_epoch+1):
        train(epoch)
        if epoch % args.val_freg == 0:
            metrics = validate()
    torch.save(cv_mgr.state_dict(), os.path.join(save_dir, 'checkpoint', f'ddg_model.ckpt'))


    if param_s['ckpt_path'] is None:
        cv_mgr.load_state_dict(torch.load(os.path.join(save_dir, 'checkpoint', f'ddg_model.ckpt')))
    else:
        cv_mgr.load_state_dict(torch.load(param_s['ckpt_path']))
    # res_ranks = pointmut_analysis()
    metrics = validate(save=True)
    best_valid_metric = metrics[0][1] + metrics[0][2] - metrics[0][3] - metrics[0][4] + metrics[0][5] + metrics[0][6] + metrics[0][7]


    nni.report_final_result(best_valid_metric)
    outFile = open('../PerformMetrics.csv','a+', newline='')
    writer = csv.writer(outFile, dialect='excel')

    results = [timestamp]
    for v, k in param.items():
        results.append(k)

    for i in range(3):
        if i == 0:
            results.append(str(metrics[0][6] + metrics[0][7]))
            results.append(str(best_valid_metric))
        results.append(metrics[i][-1])
        for j in range(1, 8):
            results.append(str(metrics[i][j]))

    writer.writerow(results)