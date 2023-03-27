import torch
import torch.nn.functional as F
import numpy as np
import os
import time
from random import randint

from src.utils import python_utils as putils
from src.utils import eval_utils as eutils
from src.worker import plot_utils as plutils
from src.worker import worker_utils as wutils

class Segmenter:

    def __init__(self, expConfig, trainDataLoader, valDataLoader, infDataLoader):
        self.expConfig = expConfig
        self.trainDataLoader = trainDataLoader
        self.valDataLoader = valDataLoader
        self.infDataLoader = infDataLoader
        self.logfile = self.expConfig.OUT_DIR + '/log.txt'

        self.cuda = False
        if next(expConfig.model.parameters()).is_cuda:
            self.cuda = True

        self.accumulated_losses = np.zeros((expConfig.EPOCHS, 2))
        self.accumulated_losses_val = np.zeros((expConfig.EPOCHS, 2))

        # load model
        expConfig.START_EPOCH = 0
        pretrain = False
        if 'infer' in expConfig.MODE or expConfig.RESUME:
            if os.path.isfile(expConfig.RESUME):
                print("=> loading checkpoint '{}'".format(expConfig.RESUME))
                checkpoint = torch.load(expConfig.RESUME)
                print('loading checkpoint as dictionary')
                expConfig.model.load_state_dict(checkpoint['model_state_dict'])
                expConfig.optimizer.load_state_dict(checkpoint['optim_state_dict'])
                expConfig.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                expConfig.START_EPOCH = checkpoint['epoch']
                self.accumulated_losses[:expConfig.START_EPOCH,:] = checkpoint['acc_loss'][:expConfig.START_EPOCH,:]
                self.accumulated_losses_val[:expConfig.START_EPOCH,:] = checkpoint['acc_loss_val'][:expConfig.START_EPOCH,:]
                
                print("=> loaded checkpoint '{}' (epoch {})".format(expConfig.RESUME, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(expConfig.RESUME))
                if 'infer' in expConfig.MODE:
                    exit(1)
                else:
                    pretrain = True
                    print("{}Start from scratch or pretrained model...{}".format(wutils.bcolors.WARNING, wutils.bcolors.ENDC))

        else:
            pretrain = True

        if pretrain:
            if expConfig.PRETRAINED is not None:
                if os.path.isfile(expConfig.PRETRAINED):
                    print("{}Start from pretraining...{}".format(wutils.bcolors.WARNING, wutils.bcolors.ENDC))
                    print("=> loading checkpoint '{}'".format(expConfig.PRETRAINED))
                    checkpoint = torch.load(expConfig.PRETRAINED)
                    print('loading checkpoint as dictionary')

                    print('Overwrite encoder weights')
                    pretrained_dict = checkpoint['model_state_dict']
                    model_dict = expConfig.model.state_dict()
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    model_dict.update(pretrained_dict) 
                    expConfig.model.load_state_dict(model_dict)

                    wutils.save_checkpoint(
                        {'epoch': 0, 'batch': 0, 'model_state_dict': expConfig.model.state_dict(),
                        'best_perf': None,
                        'optim_state_dict': expConfig.optimizer.state_dict(),
                        'scheduler_state_dict': expConfig.lr_scheduler.state_dict(),
                        'acc_loss': self.accumulated_losses, 'acc_loss_val': self.accumulated_losses_val, }, 0,
                        os.path.join(expConfig.OUT_DIR, expConfig.EXPERIMENT_PREFIX), model_name='pretrained')


                else:
                    print("=> no checkpoint found at '{}'".format(expConfig.PRETRAINED))
                    print("{}Start from scratch...{}".format(wutils.bcolors.WARNING, wutils.bcolors.ENDC))

            else:
                print("{}Start from scratch...{}".format(wutils.bcolors.WARNING, wutils.bcolors.ENDC))

        if expConfig.FREEZE_ENCODER:
            self.freeze_weights_encoder()

        print()
        nr_params = wutils.count_parameters(expConfig.model)
        print("{}{} parameters to train{}".format(wutils.bcolors.WARNING, nr_params, wutils.bcolors.ENDC))
        print()


    def freeze_weights_encoder(self):

        print("{}freeze weights encoder{}".format(wutils.bcolors.OKGREEN, wutils.bcolors.ENDC))

        # freeze weights of encoder
        for name, child in self.expConfig.model.named_children():
            # FREEZE
            if name == 'encoders':
                print("{}Encoder is frozen{}".format(wutils.bcolors.OKGREEN, wutils.bcolors.ENDC))
                for gchild in child:
                    for param in gchild.parameters():
                        param.requires_grad = False

    # ===================================================================================================

    def train(self):
        print("Training...")

        exp_config = self.expConfig

        best_perf = 0
        is_best = 0
        plotter = None
        if exp_config.PLOT_PROGRESS:
            plotter = plutils.PlotSegmentationProgress(exp_config.PROGRESS_FILE)

        perf_loss = 0
        for epoch in range(exp_config.START_EPOCH, exp_config.EPOCHS):
            print('EPOCH = {}'.format(epoch))

            for param_group in exp_config.optimizer.param_groups:
                print("Optimizer's learning rate: {}".format(param_group['lr']))

            if epoch > 0:
                # self.expConfig.lr_scheduler.step()
                if self.expConfig.SCHEDULER:
                    self.expConfig.lr_scheduler.step()

            # train for one epoch
            loss, dice, image, label, estim, train_losses = self.train_one_epoch(epoch)
            self.accumulated_losses[epoch, 0] = loss
            self.accumulated_losses[epoch, 1:] = dice

            # evaluate on validation set
            perf_loss, perf_dice, val_image, val_label, val_estim, val_losses = self.validate(epoch)
            self.accumulated_losses_val[epoch, 0] = perf_loss
            self.accumulated_losses_val[epoch, 1:] = perf_dice

            # remember best performance and save checkpoint
            is_best = perf_dice > best_perf
            best_perf = max(perf_dice, best_perf)
            wutils.save_checkpoint({'epoch': epoch + 1, 'model_state_dict': exp_config.model.state_dict(), 'best_perf': best_perf,
                             'optim_state_dict': exp_config.optimizer.state_dict(),
                             'scheduler_state_dict': exp_config.lr_scheduler.state_dict(),
                             'acc_loss': self.accumulated_losses, 'acc_loss_val': self.accumulated_losses_val, }, is_best,
                            os.path.join(exp_config.OUT_DIR, exp_config.EXPERIMENT_PREFIX))

            label_ind = 0
            if exp_config.PLOT_PROGRESS:
                plotter(image[0, :, :].transpose(0, 1),
                        label[label_ind, :, :].transpose(0, 1),
                        estim[0, :, :].transpose(0, 1), train_losses,
                        val_image[0, :, :].transpose(0, 1),
                        val_label[label_ind, :, :].transpose(0, 1),
                        val_estim[0, :, :].transpose(0, 1), val_losses,
                        self.accumulated_losses[0:epoch + 1, :],
                        self.accumulated_losses_val[0:epoch + 1, :], epoch,
                        isatend=(epoch == exp_config.EPOCHS - 1))

        print("training finished!")
        f = open(self.logfile, 'a')
        f.write("training finished!\n")
        f.write("\n")
        f.close()

    # ===================================================================================================

    def train_one_epoch(self, epoch):
        f = open(self.logfile, 'a' if os.path.exists(self.logfile) else 'w')
        f.write('EPOCH = {}\n'.format(epoch))
        for param_group in self.expConfig.optimizer.param_groups:
            f.write("Optimizer's learning rate: {}\n".format(param_group['lr']))

        batch_time = putils.AverageMeter()
        data_time = putils.AverageMeter()
        losses = putils.AverageMeter()
        dice_coeff = putils.AverageMeter()

        # switch to train mode
        self.expConfig.model.train()

        plot_id1 = randint(0, len(self.trainDataLoader) - 1)

        end = time.time()
        for i, (input_, target, _) in enumerate(self.trainDataLoader):

            data_time.update(time.time() - end)

            # Convert single channel target to one-hot encoded num_classes -channel target
            targetHot = putils.OneHot(2)(target.long())

            if self.cuda:
                input_ = input_.type(torch.FloatTensor).cuda() #(async=True)
                target = target.type(torch.FloatTensor).cuda() #(async=True)
                targetHot = targetHot.type(torch.FloatTensor).cuda()  # (async=True)


            target_var = target
            targetHot_var = targetHot.long()

            # compute output
            output = self.expConfig.model(input_)

            loss = self.expConfig.criterion(output, target_var)

            # record loss and measure dice coeff
            losses.update(loss.item(), input_.size(0))

            output = output.data
            output_seg = (output > 0.2).float()
            d_coeff = eutils.dice(output_seg, target.long(), (1,))

            dice_coeff.update(d_coeff[0], input_.size(0))

            # compute gradient and do SGD step
            loss.backward()
            self.expConfig.optimizer.step()
            self.expConfig.optimizer.zero_grad()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i == plot_id1:
                plot_id2 = randint(0, input_.size()[0] - 1)

                if self.expConfig.PLOT_PROGRESS:
                    image = wutils.get_image_to_show(input_, plot_id2).cpu().detach()
                    estim = wutils.get_image_to_show(output_seg, plot_id2).cpu().detach()
                    label = wutils.get_image_to_show(target, plot_id2).cpu().detach()
                    train_losses = np.zeros(2)
                    train_losses[0] = losses.val
                    train_losses[1] = dice_coeff.val

            # output
            if i % self.expConfig.PRINT_FREQ == 0:

                print_str = (f'Epoch: [{epoch}][{i}/{len(self.trainDataLoader)}]\t Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t Data {data_time.val:.3f} ({data_time.avg:.3f})'
                             f'\t Loss {losses.val:.4f} ({losses.avg:.4f})\t Dice {dice_coeff.val:.3f} ({dice_coeff.avg:.3f})')
                print(print_str)
                f.write(print_str)

        print_str = f'TrainAvg: Loss {losses.avg:.4f}\t Dice-score {dice_coeff.avg:.3f}'
        print(print_str)
        f.write(print_str)
        f.close()

        if self.expConfig.PLOT_PROGRESS:
            return losses.avg, dice_coeff.avg, image, label, estim, train_losses
        else:
            return losses.avg, dice_coeff.avg, [], [], [], []
        # return losses.avg, dice_coeff.avg

    # ===================================================================================================

    def validate(self, epoch):
        f = open(self.logfile, 'a')

        print("Validation...")
        f.write("Validation...\n")

        batch_time = putils.AverageMeter()
        losses = putils.AverageMeter()
        dice_coeff = putils.AverageMeter()

        self.expConfig.model.eval()

        plot_id = randint(0, len(self.valDataLoader) - 1)

        end = time.time()
        for i, (input_, target, _) in enumerate(self.valDataLoader):

            targetHot = putils.OneHot(2)(target.long())

            if self.cuda:
                target = target.type(torch.FloatTensor).cuda() #(async=True)
                input_ = input_.type(torch.FloatTensor).cuda() #(async=True)
                targetHot = targetHot.type(torch.FloatTensor).cuda()  # (async=True)

            target_var = target
            targetHot_var = targetHot.long()

            # compute output
            output = self.expConfig.model(input_)

            # print(torch.max(output))
            loss = self.expConfig.criterion(output, target_var)

            # measure dice coefficient and record loss
            losses.update(loss.item(), input_.size(0))
            output = output.data
            output_seg = (output > 0.2).float() #F.softmax(output.data, dim=1)

            d_coeff = eutils.dice(output_seg, target, (1,))

            dice_coeff.update(d_coeff[0], input_.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i == plot_id:
                plot_id2 = randint(0, input_.size()[0] - 1)

                if self.expConfig.PLOT_PROGRESS:
                    image = wutils.get_image_to_show(input_, plot_id2).cpu().detach()
                    estim = wutils.get_image_to_show(output_seg, plot_id2).cpu().detach()
                    label = wutils.get_image_to_show(target, plot_id2).cpu().detach()
                    val_losses = np.zeros(2)
                    val_losses[0] = losses.val
                    val_losses[1] = dice_coeff.val

            if i % self.expConfig.PRINT_FREQ == 0:

                print_str = (f'Val: [{i}/{len(self.valDataLoader)}]\t Time {batch_time.val:.3f} ({batch_time.avg:.3f})'
                             f'\t Loss {losses.val:.4f} ({losses.avg:.4f})\t Dice {dice_coeff.val:.3f} ({dice_coeff.avg:.3f})')
                print(print_str)
                f.write(print_str)

        print_str = f'ValAvg: Loss {losses.avg:.4f}\t Dice-score {dice_coeff.avg:.3f}'
        print(print_str)
        f.write(print_str)
        f.close()

        if self.expConfig.PLOT_PROGRESS:
            return losses.avg, dice_coeff.avg, image, label, estim, val_losses
        else:
            return losses.avg, dice_coeff.avg, [], [], [], []

    # ===================================================================================================

    def infer(self):
        print("Inference...")

        exp_config = self.expConfig
        exp_config.model.eval()

        measures = ['jaccard', 'dice', 'robust (95%) Hausdorff distance', 'surface distance mean']
        performance = np.empty((len(self.infDataLoader)*exp_config.BATCH_SIZE_INF,len(measures)))
        classes = []
        file_names = []

        count = 0
        if len(self.infDataLoader) > 0:

            # for each batch
            for i, (input_, target, info) in enumerate(self.infDataLoader):
                name = info[0]
                clabel = info[1]

                if self.cuda:
                    target = target.type(torch.FloatTensor).cuda() #(async=True)
                    input_ = input_.type(torch.FloatTensor).cuda() #(async=True)

                # compute output
                output = self.expConfig.model(input_)

                output = output.data
                output_seg = (output > 0.2).float()

                # for each image in batch
                for j in range(0, input_.size()[0]):
                    
                    res = eutils.evaluate_segmentation(np.squeeze(target[j,:].cpu().detach().numpy()).astype(int), 
                                                 np.squeeze(output_seg[j,:].cpu().detach().numpy()).astype(int))

                    file_names.append(name[j])
                    classes.append(clabel[j])
                    performance[count, :] = res
                    count += 1

                    print(f"{name[j]} \t {clabel[j]} \t Dice {res[1]:.3f}\t IoU {res[0]:.3f}\t ASD {res[2]:.3f}\t HD95 {res[3]:.3f}")

            eutils.save_to_excel(file_names, performance, measures, 'test', self.expConfig.RESULTS_FILE, classes, task='segment')

            p_mean = performance.mean(axis=0)
            p_std = performance.std(axis=0)

            print('')
            print(f'{wutils.bcolors.OKGREEN}TEST SET{wutils.bcolors.ENDC}')
            print(f"Dice\t {p_mean[1]:.3f} +/- {p_std[1]:.3f}\t IoU\t {p_mean[0]:.3f} +/- {p_std[0]:.3f}\t ASD\t {p_mean[2]:.3f} +/- {p_std[2]:.3f}\t HD95\t {p_mean[3]:.3f} +/- {p_std[3]:.3f}")

            for s in set(classes):
                print(f'{wutils.bcolors.OKGREEN}TEST SET {s}{wutils.bcolors.ENDC}')
                ind = [i for i,c in enumerate(classes) if c==s]

                pc_mean = performance[ind,:].mean(axis=0)
                pc_std = performance[ind,:].std(axis=0)
                print(f"Dice\t {pc_mean[1]:.3f} +/- {pc_std[1]:.3f}\t IoU\t {pc_mean[0]:.3f} +/- {pc_std[0]:.3f}\t ASD\t {pc_mean[2]:.3f} +/- {pc_std[2]:.3f}\t HD95\t {pc_mean[3]:.3f} +/- {pc_std[3]:.3f}")


    # ===================================================================================================

    def enable_dropout(self):
        """ Function to enable the dropout layers during test-time """
        for m in self.expConfig.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                # print(m.__class__)
                m.train()

    
    def infer_mc_dropout(self):
        print("Inference with monte carlo predictions...")

        exp_config = self.expConfig
        exp_config.model.eval()
        self.enable_dropout()

        measures = ['jaccard', 'dice', 'robust (95%) Hausdorff distance', 'surface distance mean']
        performance = np.empty((exp_config.DROPOUT_MC,len(self.infDataLoader)*exp_config.BATCH_SIZE_INF,len(measures)))
        classes = []
        file_names = [] 

       
        if len(self.infDataLoader) > 0:
            for drop in range(exp_config.DROPOUT_MC):

                count = 0
                print("dropout run {}".format(drop))

                classes.append([])
                file_names.append([])
                # for each batch
                for i, (input_, target, info) in enumerate(self.infDataLoader):
                    name = info[0]
                    clabel = info[1]

                    if self.cuda:
                        target = target.type(torch.FloatTensor).cuda() #(async=True)
                        input_ = input_.type(torch.FloatTensor).cuda() #(async=True)

                    # compute output
                    output = self.expConfig.model(input_)

                    output = output.data
                    output_seg = (output > 0.2).float()

                    # for each image in batch
                    for j in range(0, input_.size()[0]):
                        
                        res = eutils.evaluate_segmentation(np.squeeze(target[j,:].cpu().detach().numpy()).astype(int), 
                                                    np.squeeze(output_seg[j,:].cpu().detach().numpy()).astype(int))

                        file_names[drop].append(name[j])
                        classes[drop].append(clabel[j])
                        performance[drop, count, :] = res
                        count += 1

                        print(f"{name[j]} \t {clabel[j]} \t Dice {res[1]:.3f}\t IoU {res[0]:.3f}\t ASD {res[2]:.3f}\t HD95 {res[3]:.3f}")

            for drop in range(exp_config.DROPOUT_MC):
                p_mean = performance[drop,:].mean(axis=0)
                p_std = performance[drop,:].std(axis=0)

                print('')
                print(f'{wutils.bcolors.OKGREEN}Run {drop}{wutils.bcolors.ENDC}')
                print(f'\t{wutils.bcolors.OKGREEN}TEST SET{wutils.bcolors.ENDC}')
                print(f"\tDice\t {p_mean[1]:.3f} +/- {p_std[1]:.3f}\t IoU\t {p_mean[0]:.3f} +/- {p_std[0]:.3f}\t ASD\t {p_mean[2]:.3f} +/- {p_std[2]:.3f}\t HD95\t {p_mean[3]:.3f} +/- {p_std[3]:.3f}")

                for s in set(classes[drop]):
                    print(f'{wutils.bcolors.OKGREEN}\tTEST SET {s}{wutils.bcolors.ENDC}')
                    ind = [i for i,c in enumerate(classes[drop]) if c==s]

                    pc_mean = performance[drop,ind,:].mean(axis=0)
                    pc_std = performance[drop,ind,:].std(axis=0)
                    print(f"\tDice\t {pc_mean[1]:.3f} +/- {pc_std[1]:.3f}\t IoU\t {pc_mean[0]:.3f} +/- {pc_std[0]:.3f}\t ASD\t {pc_mean[2]:.3f} +/- {pc_std[2]:.3f}\t HD95\t {pc_mean[3]:.3f} +/- {pc_std[3]:.3f}")

            performance = performance.reshape(exp_config.DROPOUT_MC*len(self.infDataLoader)*exp_config.BATCH_SIZE,-1)
            file_names = [item for sublist in file_names for item in sublist]
            classes = [item for sublist in classes for item in sublist]
            dropout = [[drop] * len(self.infDataLoader)*exp_config.BATCH_SIZE for drop in range(exp_config.DROPOUT_MC)]
            dropout = [item for sublist in dropout for item in sublist]

            cols = {'dropout run': dropout}
            eutils.save_to_excel(file_names, performance, measures, 'test-mc', self.expConfig.RESULTS_FILE, classes, task='segment', **cols)
