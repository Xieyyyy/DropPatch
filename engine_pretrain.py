# import fontTools.t1Lib
import torch
import os
import time
import numpy as np
import pickle
from utils.showcase import show_pos_rec, PatchTST_showcase, dm_showcase
from torch.optim import lr_scheduler
from tools import adjust_lr


def train_pt(model, optimizer, device, train_loader, vali_loader, path, args):
    print("start training...")
    best_score = np.inf

    scheduler = lr_scheduler.OneCycleLR(optimizer=optimizer,
                                        steps_per_epoch=len(train_loader),
                                        pct_start=0.3,
                                        epochs=args.n_epochs_pretrain,
                                        max_lr=args.lr)

    train_loss_epoch_lst = []
    val_loss_epoch_lst = []

    for epoch in range(args.n_epochs_pretrain):
        model.train()

        train_loss = []

        start_time = time.time()

        for i, (batch_x, _) in enumerate(train_loader):
            batch_x = batch_x.float().to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)

            loss = outputs[0]

            loss.backward()
            train_loss.append(loss.item())
            optimizer.step()
            scheduler.step()

        train_loss_epoch = np.average(train_loss)
        vali_loss_epoch = vali(model, vali_loader, device)

        print('Epoch {} \n| train loss: {:.4f} | valid loss: {:.4f}'
              .format(epoch + 1, train_loss_epoch, vali_loss_epoch))

        print('| cost time: {:.2f}'.format(time.time() - start_time))

        train_loss_epoch_lst.append(train_loss_epoch)
        val_loss_epoch_lst.append(vali_loss_epoch)

        save_dict = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model": args.model
        }

        if vali_loss_epoch < best_score:
            print(f'Validation loss decreased ({best_score:.6f} --> {vali_loss_epoch:.6f}).')
            best_score = vali_loss_epoch
            print('saving best model...')
            torch.save(save_dict, path + '/' + 'checkpoint.pth')

        adjust_lr(optimizer, scheduler, epoch + 1, args)

    best_model_path = path + '/checkpoint.pth'
    print(f'best pretrain model path: {best_model_path}')
    model.load_state_dict(torch.load(best_model_path)['state_dict'])

    train_loss_epoch = np.array(train_loss_epoch_lst)
    val_loss_epoch = np.array(val_loss_epoch_lst)

    np.save('outputs/pretrain_loss/' + args.setting + '_train.npy', train_loss_epoch)
    np.save('outputs/pretrain_loss/' + args.setting + '_val.npy', val_loss_epoch)

    return model


def vali(model, vali_loader, device):
    model.eval()
    vali_loss = []
    with torch.no_grad():
        for i, (batch_x, _) in enumerate(vali_loader):
            batch_x = batch_x.float().to(device)

            outputs = model(batch_x)
            loss = outputs[0]

            vali_loss.append(loss.item())

        vali_loss_epoch = np.average(vali_loss)

    model.train()

    return vali_loss_epoch


def test_pt(test_loader, device, model, path, setting_pt, patch_len, showcase, stride=None):
    best_model_path = path + '/checkpoint.pth'
    print(f'best pretrain model path: {best_model_path}')
    model.load_state_dict(torch.load(best_model_path)['state_dict'])
    print("start testing...")
    losses = []
    visual_outs = []
    attns = []

    model.eval()
    with torch.no_grad():
        for i, (batch_x, _) in enumerate(test_loader):
            batch_x = batch_x.float().to(device)
            loss, _, _, model_visual_outs = model(batch_x, test=True)

            if showcase == 1:
                if (i+1) % 1000 == 0:
                    visual_out = dm_showcase(model_visual_outs, batch_x, patch_len)
                    visual_outs.append(visual_out)

                    if 'attn' in model_visual_outs.keys():
                        attns.append(model_visual_outs['attn'])

            losses.append(loss.detach().cpu().numpy())

    loss_mse = np.array(losses)  # num_batch * bs * l * c

    loss_mse_avg = np.average(loss_mse)

    print('loss_mse:{}'.format(loss_mse_avg))
    f = open('./outputs/result_pretrain_test/pretrain_test.txt', 'a')
    f.write(setting_pt + ' \n')
    f.write('test loss: {:.2f}\n'.format(loss_mse_avg))
    f.close()

    if showcase == 1:
        print('saving visualization outputs...')

        with open('./outputs/visualization_results/' + setting_pt + '.pkl', 'wb') as f:
            pickle.dump(visual_outs, f)

    if 'attn' in model_visual_outs.keys():
        print('saving attn outputs...')
        torch.save(attns, './outputs/visualization_results/attn_' + setting_pt + '.pth')

    return
