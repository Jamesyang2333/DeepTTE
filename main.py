import os
import json
import time
import utils
import models
import logger
import inspect
import datetime
import argparse
import data_loader

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable

import numpy as np

parser = argparse.ArgumentParser()
# basic args
parser.add_argument('--task', type = str)
parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--epochs', type = int, default = 10)

# evaluation args
parser.add_argument('--weight_file', type = str)
parser.add_argument('--result_file', type = str)

# cnn args
parser.add_argument('--kernel_size', type = int)

# rnn args
parser.add_argument('--pooling_method', type = str)

# multi-task args
parser.add_argument('--alpha', type = float)

# log file name
parser.add_argument('--log_file', type = str)

args = parser.parse_args()

config = json.load(open('./config-chengdu.json', 'r'))

def train(model, elogger, train_set, eval_set):
    # record the experiment setting
    elogger.log(str(model))
    elogger.log(str(args._get_kwargs()))

    model.train()

    if torch.cuda.is_available():
        print("USING GPU")
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    
    data_iter_set_train = {}
    data_iter_set_eval = {}
    
    for input_file in train_set:
        print('reading file {}'.format(input_file))
        data_iter = data_loader.get_loader(input_file, args.batch_size)
        data_iter_set_train[input_file] = data_iter
        
    for input_file in eval_set:
        print('reading file {}'.format(input_file))
        data_iter = data_loader.get_loader(input_file, args.batch_size)
        data_iter_set_eval[input_file] = data_iter

    best_mse = 1e9
    for epoch in range(args.epochs):
        model.train()
        print ('Training on epoch {}'.format(epoch))
        for input_file in train_set:
            print ('Train on file {}'.format(input_file))

            # data loader, return two dictionaries, attr and traj
#             data_iter = data_loader.get_loader(input_file, args.batch_size)
            data_iter = data_iter_set_train[input_file]

            running_loss = 0.0
            total_mse = 0.0
            total_l1 = 0.0

            for idx, (attr, traj) in enumerate(data_iter):
                # transform the input to pytorch variable
                attr, traj = utils.to_var(attr), utils.to_var(traj)

                pred_dict, loss = model.eval_on_batch(attr, traj, config)
                
                total_mse += F.mse_loss(pred_dict["label"], pred_dict["pred"]).item()
                total_l1 += F.l1_loss(pred_dict["label"], pred_dict["pred"]).item()

                # update the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

#                 running_loss += loss.data[0]
                running_loss += loss.data.item()
#                 print ('\r Progress {:.2f}%, average loss {}'.format((idx + 1) * 100.0 / len(data_iter), running_loss / (idx + 1.0)))
#             print
            elogger.log('Training Epoch {}, File {}, Loss {}, mse {}, l1 {}'.format(epoch, input_file, running_loss / (idx + 1.0), total_mse / (idx + 1.0), total_l1 / (idx + 1.0)))
            print('Training Epoch {}, File {}, Loss {}, mse {}, l1 {}'.format(epoch, input_file, running_loss / (idx + 1.0), total_mse / (idx + 1.0), total_l1 / (idx + 1.0)))

        # evaluate the model after each epoch
        epoch_mse = evaluate(model, elogger, eval_set, data_iter_set_eval, save_result = False)
        
        if epoch_mse < best_mse:
            best_mse = epoch_mse

            # save the weight file after each epoch
            weight_name = '{}_{}'.format(args.log_file, str(datetime.datetime.now()))
            elogger.log('Save weight file {}'.format(weight_name))
    #         torch.save(model.state_dict(), './saved_weights/' + weight_name)
            torch.save(model.state_dict(), '/Project0551/jingyi/deeptte/saved_weights/deeptte-chengdu-1.pt')

def write_result(fs, pred_dict, attr):
    pred = pred_dict['pred'].data.cpu().numpy()
    label = pred_dict['label'].data.cpu().numpy()

    for i in range(pred_dict['pred'].size()[0]):
        fs.write('%.6f %.6f\n' % (label[i][0], pred[i][0]))

        dateID = attr['dateID'].data[i]
        timeID = attr['timeID'].data[i]
        driverID = attr['driverID'].data[i]


def evaluate(model, elogger, files, data_iter_set_eval, save_result = False):
    model.eval()
    if save_result:
        fs = open('%s' % args.result_file, 'w')

    result_mse = 0
    for input_file in files:
        running_loss = 0.0
#         data_iter = data_loader.get_loader(input_file, args.batch_size)
        data_iter = data_iter_set_eval[input_file]
        total_mse = 0.0
        total_l1 = 0.0

        for idx, (attr, traj) in enumerate(data_iter):
            attr, traj = utils.to_var(attr), utils.to_var(traj)
            
            if attr['timeID'].size(0) != attr['dist'].size(0) or attr['dateID'].size(0) != attr['dist'].size(0) or attr['driverID'].size(0) != attr['dist'].size(0):
                print(idx)
                print(attr['dist'].shape)
                print(attr['dateID'].shape)
                print(attr['timeID'].shape)
                print(attr['driverID'].shape)

            pred_dict, loss = model.eval_on_batch(attr, traj, config)

            total_mse += F.mse_loss(pred_dict["label"], pred_dict["pred"]).item()
            total_l1 += F.l1_loss(pred_dict["label"], pred_dict["pred"]).item()
            
#             if idx % 10000 == 9999:
#                 print(idx)
#                 print(total_mse)
#                 print(total_l1)
            

            if save_result: write_result(fs, pred_dict, attr)

            running_loss += loss.data.item()
        
        print("total_mse: " + str(total_mse))
        print("total_l1: " + str(total_l1))
        print("idx: " + str(idx))
        print ('Evaluate on file {}, loss {}, mse {}, l1 {}'.format(input_file, running_loss / (idx + 1.0), total_mse / (idx + 1.0), total_l1 / (idx + 1.0)))
        elogger.log('Evaluate File {}, Loss {}, mse {}, l1 {}'.format(input_file, running_loss / (idx + 1.0), total_mse / (idx + 1.0), total_l1 / (idx + 1.0)))
        result_mse = total_mse / (idx + 1.0)

    if save_result: fs.close()
    return result_mse

def get_kwargs(model_class):
    model_args = inspect.getargspec(model_class.__init__).args
    shell_args = args._get_kwargs()

    kwargs = dict(shell_args)

    for arg, val in shell_args:
        if not arg in model_args:
            kwargs.pop(arg)

    return kwargs

def run():
    # get the model arguments
    kwargs = get_kwargs(models.DeepTTE.Net)

    # model instance
    model = models.DeepTTE.Net(**kwargs)

    # experiment logger
    elogger = logger.Logger(args.log_file)

    if args.task == 'train':
        train(model, elogger, train_set = config['train_set'], eval_set = config['eval_set'])

    elif args.task == 'test':
        # load the saved weight file
        model.load_state_dict(torch.load(args.weight_file))
        if torch.cuda.is_available():
            model.cuda()
            
        data_iter_set_eval = {}
        for input_file in config['test_set']:
            
            print('reading file {}'.format(input_file))
            data_iter = data_loader.get_loader(input_file, args.batch_size)
            data_iter_set_eval[input_file] = data_iter
            
        evaluate(model, elogger, config['test_set'], data_iter_set_eval, save_result = True)

if __name__ == '__main__':
    run()
