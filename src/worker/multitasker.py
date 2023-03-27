import torch
import torch.nn.functional as F
import numpy as np
import os
import time
from random import randint
from glob import glob as glob

from src.utils import python_utils as putils
from src.utils import eval_utils as eutils
from src.worker import plot_utils as plutils
from src.worker import worker_utils as wutils

class Multitasker:

    def __init__(self, expConfig, trainDataLoader1, trainDataLoader2, valDataLoader, infDataLoader):
        self.expConfig = expConfig
        self.trainDataLoader1 = trainDataLoader1
        self.trainDataLoader2 = trainDataLoader2
        self.valDataLoader = valDataLoader
        self.infDataLoader = infDataLoader
        self.logfile = self.expConfig.OUT_DIR + '/log.txt'

        self.best_perf = 0

        self.cuda = False
        if next(expConfig.model.parameters()).is_cuda:
            self.cuda = True

        self.accumulated_losses = np.zeros((expConfig.EPOCHS, 4 + self.expConfig.N_CLASSES))
        self.accumulated_losses_val = np.zeros((expConfig.EPOCHS, 4 + self.expConfig.N_CLASSES))

        # load model
        expConfig.START_EPOCH = 0
        expConfig.START_BATCH = 0
        pretrain = False
        if expConfig.MODE == 'infer' or expConfig.RESUME:
            if os.path.isfile(expConfig.RESUME):
                print("=> loading checkpoint '{}'".format(expConfig.RESUME))
                checkpoint = torch.load(expConfig.RESUME)
                print('loading checkpoint as dictionary')
                expConfig.model.load_state_dict(checkpoint['model_state_dict'])
                expConfig.optimizer.load_state_dict(checkpoint['optim_state_dict'])
                expConfig.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                expConfig.START_EPOCH = checkpoint['epoch']  # - 1
                self.accumulated_losses[:expConfig.START_EPOCH,:] = checkpoint['acc_loss'][:expConfig.START_EPOCH,:]
                self.accumulated_losses_val[:expConfig.START_EPOCH,:] = checkpoint['acc_loss_val'][:expConfig.START_EPOCH,:]
                
                print("=> loaded checkpoint '{}' (epoch {})".format(expConfig.RESUME, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(expConfig.RESUME))
                if expConfig.MODE == 'infer':
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

                    print('Overwrite encoder and classification weights')
                    pretrained_dict = checkpoint['model_state_dict']
                    model_dict = expConfig.model.state_dict()
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    model_dict.update(pretrained_dict) 
                    expConfig.model.load_state_dict(model_dict)

                    wutils.save_checkpoint(
                        {'epoch': 0, 'batch': 0, 'model_state_dict': expConfig.model.state_dict(),
                        'best_perf': self.best_perf,
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
            # FREEZE encoder
            if name == 'encoders':
                print("{}Encoder is frozen{}".format(wutils.bcolors.OKGREEN, wutils.bcolors.ENDC))
                for gchild in child:
                    for param in gchild.parameters():
                        param.requires_grad = False

            # FREEZE classification head
            elif self.expConfig.FREEZE_CLASS_HEAD:
                if name == 'attention_layers' or name == 'projectors_layers' or name == 'dense':
                    print(f"{wutils.bcolors.OKGREEN}{name} is frozen{wutils.bcolors.ENDC}")
                    for gchild in child:
                        for param in gchild.parameters():
                            param.requires_grad = False
                elif name == 'classify' or name == 'upsampler':
                    print(f"{wutils.bcolors.OKGREEN}{name} is frozen{wutils.bcolors.ENDC}")


    # ===================================================================================================

    def train(self):
        print("Training...")

        exp_config = self.expConfig
        n_attentions = len(self.expConfig.AM_POS)

        self.best_perf = 0
        best_loss = 10
        is_best = 0
        plotter = None
        if exp_config.PLOT_PROGRESS:
            plotter = plutils.PlotMultitaskProgress(n_attentions, exp_config.PROGRESS_FILE)

        perf_loss = 0
        for epoch in range(exp_config.START_EPOCH, exp_config.EPOCHS):
            print('EPOCH = {}'.format(epoch))

            for param_group in exp_config.optimizer.param_groups:
                print("Optimizer's learning rate: {}".format(param_group['lr']))

            # self.adjust_learning_rate(exp_config.optimizer, epoch)
            if epoch > 0:
                # self.expConfig.lr_scheduler.step()
                if self.expConfig.SCHEDULER:
                    self.expConfig.lr_scheduler.step()

            # train for one epoch
            loss, loss_1, loss_2, dice, acc, image1, image2, label, label_class, estim, estim_class, attention_maps, \
            train_losses = self.train_one_epoch(epoch)
            # loss, dice = self.train_one_epoch(epoch)
            self.accumulated_losses[epoch, 0] = loss
            self.accumulated_losses[epoch, 1] = loss_1
            self.accumulated_losses[epoch, 2] = loss_2
            self.accumulated_losses[epoch, 3] = dice
            self.accumulated_losses[epoch, 4:] = acc

            # # evaluate on validation set
            perf_loss, perf_loss_1, perf_loss_2, perf_dice, perf_acc, val_image, val_label, val_label_class, \
            val_estim, val_estim_class, val_attention_maps, val_losses = self.validate(epoch)
            # perf_loss, perf_dice = self.validate(epoch)
            self.accumulated_losses_val[epoch, 0] = perf_loss
            self.accumulated_losses_val[epoch, 1] = perf_loss_1
            self.accumulated_losses_val[epoch, 2] = perf_loss_2
            self.accumulated_losses_val[epoch, 3] = perf_dice
            self.accumulated_losses_val[epoch, 4:] = perf_acc

            # remember best performance and save checkpoint
            is_best = perf_dice > self.best_perf
            self.best_perf = max(perf_dice, self.best_perf)
            wutils.save_checkpoint(
                {'epoch': epoch + 1, 'batch': 0, 'model_state_dict': exp_config.model.state_dict(),
                 'best_perf': self.best_perf,
                 'optim_state_dict': exp_config.optimizer.state_dict(),
                 'scheduler_state_dict': exp_config.lr_scheduler.state_dict(),
                 'acc_loss': self.accumulated_losses, 'acc_loss_val': self.accumulated_losses_val, }, is_best,
                os.path.join(exp_config.OUT_DIR, exp_config.EXPERIMENT_PREFIX))

            label_ind = 0
            if exp_config.PLOT_PROGRESS:  ## or epoch == exp_config.EPOCHS-1:
                plotter(exp_config.CLASSES, image1[0, :, :].transpose(0, 1),
                        image2[0, :, :].transpose(0, 1),
                        label[label_ind, :, :].transpose(0, 1), label_class,
                        estim[0, :, :].transpose(0, 1), estim_class, train_losses,
                        val_image[0, :, :].transpose(0, 1),
                        val_label[label_ind, :, :].transpose(0, 1), val_label_class,
                        val_estim[0, :, :].transpose(0, 1), val_estim_class, val_losses,
                        [att[0, :, :].transpose(0, 1) for att in attention_maps],
                        [vatt[0, :, :].transpose(0, 1) for vatt in val_attention_maps],
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
        losses_1 = putils.AverageMeter()
        losses_2 = putils.AverageMeter()
        dice_coeff = putils.AverageMeter()
        acc_coeff = putils.AccuracyMeter(self.expConfig.N_CLASSES)

        # switch to train mode
        self.expConfig.model.train()

        plot_id1 = randint(0, len(self.trainDataLoader2) - 1)

        end = time.time()

        print("Size self.trainDataLoader1: {}".format(len(self.trainDataLoader1)))
        print("Size self.trainDataLoader2: {}".format(len(self.trainDataLoader2)))

        task1_iterator = iter(self.trainDataLoader1)
        # task2_iterator = iter(self.trainDataLoader2)
        count_task2 = 0
        for i in range(len(self.trainDataLoader1)):

            if i < self.expConfig.START_BATCH:
                continue
            if i == self.expConfig.START_BATCH:
                self.expConfig.START_BATCH = 0

            data_time.update(time.time() - end)

            #
            #   TASK1: CLASSIFICATION
            #

            input_, target, _ = next(task1_iterator)

            if self.cuda:
                input_ = input_.type(torch.FloatTensor).cuda() #(async=True)
                target = target.type(torch.FloatTensor).cuda() #(async=True)

            # compute output for task 1
            _, output_class, attention = self.expConfig.model(input_)

            # classification loss
            loss1 = self.expConfig.criterionT1(output_class, torch.max(target, 1)[1])

            # optimize the tasks separatley
            loss1.backward()
            self.expConfig.optimizer.step()
            self.expConfig.optimizer.zero_grad()

            # record loss
            losses_1.update(loss1.item(), input_.size(0))

            # compute balanced accuracy, sensitivity and specificity
            with torch.no_grad():
                _, predicted = torch.max(output_class, 1)
                _, gt = torch.max(target, 1)

                acc_coeff.update(predicted.cpu().numpy(), gt.cpu().numpy())

            if i == plot_id1:
                plot_id2_t1 = randint(0, input_.size()[0] - 1)

                if self.expConfig.PLOT_PROGRESS:
                    image_t1 = wutils.get_image_to_show(input_, plot_id2_t1).cpu().detach()
                    if attention:
                        attention_maps = [wutils.get_image_to_show(attention[x], plot_id2_t1).cpu().detach() for x in
                                          range(0, len(attention))]
                    else:
                        attention_maps = None
                    estim_class = [output_class[plot_id2_t1].detach(), predicted[plot_id2_t1].item()]
                    label_class = gt[plot_id2_t1].item()
                    train_losses = np.zeros(4)
                    train_losses[1] = losses_1.val

            # ======================================================================================================
            #
            #   TASK2: SEGMENTATION
            #
            loss2 = 0
            if epoch >= self.expConfig.START_SEGM:
                # print("segmentation")

                for b in range(self.expConfig.BETA):
                    # print("Beta: {}".format(b))
                    if count_task2 % len(self.trainDataLoader2) == 0:
                        # print("(re)start dataloader task2 (i={})".format(i))
                        task2_iterator = iter(self.trainDataLoader2)
                        count_task2 = 0
                    input_, target, _ = next(task2_iterator)
                    count_task2 += 1

                    # Convert single channel target to one-hot encoded num_classes -channel target
                    targetHot = putils.OneHot(2)(target.long())

                    if self.cuda:
                        input_ = input_.type(torch.FloatTensor).cuda() #(async=True)
                        target = target.type(torch.FloatTensor).cuda() #(async=True)
                        targetHot = targetHot.type(torch.FloatTensor).cuda()  # (async=True)

                    target_var = target
                    targetHot_var = targetHot.long()

                    # compute output
                    output, _, _ = self.expConfig.model(input_)

                    loss2 = self.expConfig.criterionT2(output, target_var)
                    
                    # optimize the tasks separatley
                    loss2.backward()
                    self.expConfig.optimizer.step()
                    self.expConfig.optimizer.zero_grad()

                # record loss and measure dice coeff
                losses_2.update(loss2.item(), input_.size(0))

                output = output.data
                output_seg = (output > 0.2).float()
                d_coeff = eutils.dice(output_seg, target.long(), (1,))

                dice_coeff.update(d_coeff[0], input_.size(0))
            else:
                loss2 = 0.0
                losses_2.update(0.0, 1)
                dice_coeff.update(0.0, 1)

            # =======================================================================================

            # combined loss
            loss = loss1 + loss2
            losses.update(loss.item(), input_.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i == plot_id1:
                plot_id2_t2 = randint(0, input_.size()[0] - 1)

                if self.expConfig.PLOT_PROGRESS:
                    image_t2 = wutils.get_image_to_show(input_, plot_id2_t2).cpu().detach()
                    estim = wutils.get_image_to_show(output_seg, plot_id2_t2).cpu().detach()
                    label = wutils.get_image_to_show(target, plot_id2_t2).cpu().detach()
                    train_losses[0] = losses.val
                    train_losses[2] = losses_2.val
                    train_losses[3] = dice_coeff.val

            # output
            if i % self.expConfig.PRINT_FREQ == 0:

                acc_str = ' '.join(map(str, ['%.2f%%' % elem for elem in acc_coeff.acc]))
                print_str = (f'Epoch: [{epoch}][{i}/{len(self.trainDataLoader1)}] '
                             f'\tTime {batch_time.val:.3f} ({batch_time.avg:.3f}) \t Data {data_time.val:.3f} ({data_time.avg:.3f})'
                             f'\tLoss {losses.val:.4f} ({losses.avg:.4f})\t LossClass {losses_1.val:.4f} ({losses_1.avg:.4f}) '
                             f'\tLossSeg {losses_2.val:.4f} ({losses_2.avg:.4f}) \t Dice-score {dice_coeff.val:.3f} ({dice_coeff.avg:.3f}) '
                             f'\tAcc {acc_str:s}')
                print(print_str)
                f.write(print_str)

        acc_str = ' '.join(map(str, ['%.2f%%' % elem for elem in acc_coeff.acc]))
        print_str = (f'TrainAvg: Loss {losses.avg:.4f}\t LossClass {losses_1.avg:.4f} \t LossSeg {losses_2.avg:.4f}'
                     f'\t Dice-score {dice_coeff.avg:.3f} \t Acc-score {acc_str:s}')
        print(print_str)
        f.write(print_str)
        f.close()

        if self.expConfig.PLOT_PROGRESS:
            return losses.avg, losses_1.avg, losses_1.avg, dice_coeff.avg, acc_coeff.acc, \
                   image_t1, image_t2, label, label_class, estim, estim_class, attention_maps, train_losses
        else:
            return losses.avg, losses_1.avg, losses_1.avg, dice_coeff.avg, acc_coeff.acc, [], [], [], [], [], [], [], []


    # ===================================================================================================

    def validate(self, epoch, batch=None):
        f = open(self.logfile, 'a')

        print("Validation...")
        f.write("Validation...\n")

        batch_time = putils.AverageMeter()
        losses = putils.AverageMeter()
        losses_1 = putils.AverageMeter()
        losses_2 = putils.AverageMeter()
        dice_coeff = putils.AverageMeter()
        acc_coeff = putils.AccuracyMeter(self.expConfig.N_CLASSES)

        self.expConfig.model.eval()

        plot_id = randint(0, len(self.valDataLoader) - 1)

        end = time.time()
        for i, (input_, target, class_, _) in enumerate(self.valDataLoader):

            targetHot = putils.OneHot(2)(target.long())

            if self.cuda:
                target = target.type(torch.FloatTensor).cuda() #(async=True)
                input_ = input_.type(torch.FloatTensor).cuda() #(async=True)
                class_ = class_.type(torch.FloatTensor).cuda()
                targetHot = targetHot.type(torch.FloatTensor).cuda()  # (async=True)

            target_var = target
            targetHot_var = targetHot.long()

            # compute output
            output, output_class, attention = self.expConfig.model(input_)

            # classification loss
            loss1 = self.expConfig.criterionT1(output_class, torch.max(class_, 1)[1])

            # segmentation loss
            loss2 = self.expConfig.criterionT2(output, target_var)

            # combined loss
            loss = loss1 + loss2

            # measure dice coefficient and record loss
            # record loss
            losses_1.update(loss1.item(), input_.size(0))
            losses_2.update(loss2.item(), input_.size(0))
            losses.update(loss.item(), input_.size(0))
            output = output.data
            output_seg = (output > 0.2).float() #F.softmax(output.data, dim=1)
            d_coeff = eutils.dice(output_seg, target, (1,))

            dice_coeff.update(d_coeff[0], input_.size(0))

            # compute balanced accuracy, sensitivity and specificity
            with torch.no_grad():
                _, predicted = torch.max(output_class, 1)
                _, gt = torch.max(class_, 1)

                acc_coeff.update(predicted.cpu().numpy(), gt.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i == plot_id:
                plot_id2 = randint(0, input_.size()[0] - 1)

                if self.expConfig.PLOT_PROGRESS:
                    image = wutils.get_image_to_show(input_, plot_id2).cpu().detach()
                    estim = wutils.get_image_to_show(output_seg, plot_id2).cpu().detach()
                    label = wutils.get_image_to_show(target, plot_id2).cpu().detach()
                    if attention:
                        attention_maps = [wutils.get_image_to_show(attention[x], plot_id2).cpu().detach() for x in range(0, len(attention))]
                    else:
                        attention_maps = None
                    estim_class = [output_class[plot_id2].detach(), predicted[plot_id2].item()]
                    label_class = gt[plot_id2].item()
                    val_losses = np.zeros(4)
                    val_losses[0] = losses.val
                    val_losses[1] = losses_1.val
                    val_losses[2] = losses_2.val
                    val_losses[3] = dice_coeff.val

            if i % self.expConfig.PRINT_FREQ == 0:
                acc_str = ' '.join(map(str, ['%.2f%%' % elem for elem in acc_coeff.acc]))
                print_str = (f'Val: [{i}/{len(self.valDataLoader)}]\t Time {batch_time.val:.3f} ({batch_time.avg:.3f})' 
                            f'\t Loss {losses.val:.4f} ({losses.avg:.4f})\t LossClass {losses_1.val:.4f} ({losses_1.avg:.4f}) '
                            f'\t LossSeg {losses_2.val:.4f} ({losses_2.avg:.4f}) '
                            f'\t Dice-score {dice_coeff.val:.3f} ({dice_coeff.avg:.3f}) \t Acc {acc_str:s}')
                print(print_str)
                f.write(print_str)

        acc_str = ' '.join(map(str, ['%.2f%%' % elem for elem in acc_coeff.acc]))
        print_str = (f'ValAvg: Loss {losses.avg:.4f} \t LossClass {losses_1.avg:.4f} '
                     f'\t LossSeg {losses_2.avg:.4f} \t Dice-score {dice_coeff.avg:.3f} \t Acc-score {acc_str:s}')
        print(print_str)
        f.write(print_str)
        f.close()

        if self.expConfig.PLOT_PROGRESS:
            return losses.avg, losses_1.avg, losses_1.avg, dice_coeff.avg, acc_coeff.acc, \
                   image, label, label_class, estim, estim_class, attention_maps, val_losses
        else:
            return losses.avg, losses_1.avg, losses_1.avg, dice_coeff.avg, acc_coeff.acc, [], [], [], [], [], [], []

    # ===================================================================================================

    def infer(self):
        print("Inference...")

        exp_config = self.expConfig
        exp_config.model.eval()

        acc_coeff = putils.AccuracyMeter(self.expConfig.N_CLASSES)

        measures = ['jaccard', 'dice', 'robust (95%) Hausdorff distance', 'surface distance mean']
        performance = np.empty((len(self.infDataLoader)*exp_config.BATCH_SIZE_INF,len(measures)))
        classes_str = []
        file_names = []

        classes = [-1] * len(self.infDataLoader)
        class_predictions = [-1] * len(self.infDataLoader)
        class_confidences = [-1] * len(self.infDataLoader)
        # file_names = [-1] * len(self.infDataLoader)

        count = 0
        if len(self.infDataLoader) > 0:

            # for each batch
            for i, (input_, target, class_, info) in enumerate(self.infDataLoader):
                name = info[0]
                clabel = info[1]

                if self.cuda:
                    target = target.type(torch.FloatTensor).cuda() #(async=True)
                    input_ = input_.type(torch.FloatTensor).cuda() #(async=True)
                    class_ = class_.type(torch.FloatTensor).cuda()

                # compute output
                output, output_class, _ = self.expConfig.model(input_)

                # evaluate classification
                # compute balanced accuracy, sensitivity and specificity
                with torch.no_grad():
                    _, predicted = torch.max(output_class, 1)
                    _, gt = torch.max(class_, 1)

                    acc_coeff.update(predicted.cpu().numpy(), gt.cpu().numpy())

                    classes[i] = gt.cpu().numpy()
                    class_predictions[i] = predicted.cpu().numpy()
                    class_confidences[i] = output_class.cpu().numpy()

                # evaluate segmentation
                output = output.data
                output_seg = (output > 0.2).float()

                # for each image in batch
                for j in range(0, input_.size()[0]):
                    
                    res = eutils.evaluate_segmentation(np.squeeze(target[j,:].cpu().detach().numpy()).astype(int), 
                                                 np.squeeze(output_seg[j,:].cpu().detach().numpy()).astype(int))

                    file_names.append(name[j])
                    classes_str.append(clabel[j])
                    performance[count, :] = res
                    count += 1

                    print(f"{name[j]} \t {clabel[j]}")
                    print(f"Dice {res[1]:.3f}\t IoU {res[0]:.3f}\t ASD {res[2]:.3f}\t HD95 {res[3]:.3f}")
                    print(f'\t Pred: {self.expConfig.CLASSES[predicted[j]]} \t True: {self.expConfig.CLASSES[gt[j]]}')
                    print()

            classes = [item for sublist in classes for item in sublist]
            class_predictions = [item for sublist in class_predictions for item in sublist]
            class_confidences = [item for sublist in class_confidences for item in sublist]

            p_mean = performance.mean(axis=0)
            p_std = performance.std(axis=0)

            print('')
            print(f'{wutils.bcolors.OKGREEN}TEST SET{wutils.bcolors.ENDC}')
            print(f"Dice\t {p_mean[1]:.3f} +/- {p_std[1]:.3f}\t IoU\t {p_mean[0]:.3f} +/- {p_std[0]:.3f}\t ASD\t {p_mean[2]:.3f} +/- {p_std[2]:.3f}\t HD95\t {p_mean[3]:.3f} +/- {p_std[3]:.3f}")

            for s in set(classes_str):
                print(f'{wutils.bcolors.OKGREEN}TEST SET {s}{wutils.bcolors.ENDC}')
                ind = [i for i,c in enumerate(classes_str) if c==s]

                pc_mean = performance[ind,:].mean(axis=0)
                pc_std = performance[ind,:].std(axis=0)
                print(f"Dice\t {pc_mean[1]:.3f} +/- {pc_std[1]:.3f}\t IoU\t {pc_mean[0]:.3f} +/- {pc_std[0]:.3f}\t ASD\t {pc_mean[2]:.3f} +/- {pc_std[2]:.3f}\t HD95\t {pc_mean[3]:.3f} +/- {pc_std[3]:.3f}")

            print('')
            acc_str = ' '.join(map(str, ['%.2f%%' % elem for elem in acc_coeff.acc]))
            print(f'Acc mean: {acc_str:s}')

            # ----------------------------------------------------------------------------------

            eutils.save_to_excel(file_names, performance, measures, 'test-segm', self.expConfig.RESULTS_FILE, classes_str, task='segment')
            eutils.evaluation(self.expConfig.CLASSES, classes, class_predictions, class_confidences,
                              'test-class', self.expConfig.RESULTS_FILE, file_names)


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

        acc_coeff = putils.AccuracyMeter(self.expConfig.N_CLASSES)

        measures = ['jaccard', 'dice', 'robust (95%) Hausdorff distance', 'surface distance mean']
        performance = np.empty((exp_config.DROPOUT_MC,len(self.infDataLoader)*exp_config.BATCH_SIZE_INF,len(measures)))
        classes_str = []
        file_names = [] 

        classes = [-1] * len(self.infDataLoader)
        class_predictions = [-1] * len(self.infDataLoader)
        class_confidences = [-1] * len(self.infDataLoader)
       
        if len(self.infDataLoader) > 0:
            for drop in range(exp_config.DROPOUT_MC):

                count = 0
                print("dropout run {}".format(drop))

                classes_str.append([])
                file_names.append([])

                # for each batch
                for i, (input_, target, class_, info) in enumerate(self.infDataLoader):
                    name = info[0]
                    clabel = info[1]

                    if self.cuda:
                        target = target.type(torch.FloatTensor).cuda() #(async=True)
                        input_ = input_.type(torch.FloatTensor).cuda() #(async=True)
                        class_ = class_.type(torch.FloatTensor).cuda()

                    # compute output
                    output, output_class, _ = self.expConfig.model(input_)

                    # evaluate classification (no dropout)
                    if drop==0:
                        with torch.no_grad():
                            _, predicted = torch.max(output_class, 1)
                            _, gt = torch.max(class_, 1)

                            acc_coeff.update(predicted.cpu().numpy(), gt.cpu().numpy())

                            classes[i] = gt.cpu().numpy()
                            class_predictions[i] = predicted.cpu().numpy()
                            class_confidences[i] = output_class.cpu().numpy()

                    # evaluate segmentation
                    output = output.data
                    output_seg = (output > 0.2).float()

                    # for each image in batch
                    for j in range(0, input_.size()[0]):
                        
                        res = eutils.evaluate_segmentation(np.squeeze(target[j,:].cpu().detach().numpy()).astype(int), 
                                                    np.squeeze(output_seg[j,:].cpu().detach().numpy()).astype(int))

                        file_names[drop].append(name[j])
                        classes_str[drop].append(clabel[j])
                        performance[drop, count, :] = res
                        count += 1

                        print(f"{name[j]} \t {clabel[j]}")
                        print(f"Dice {res[1]:.3f}\t IoU {res[0]:.3f}\t ASD {res[2]:.3f}\t HD95 {res[3]:.3f}")
                        if drop==0:
                            print(f'\t Pred: {self.expConfig.CLASSES[predicted[j]]} \t True: {self.expConfig.CLASSES[gt[j]]}')
                            print()


            for drop in range(exp_config.DROPOUT_MC):
                p_mean = performance[drop,:].mean(axis=0)
                p_std = performance[drop,:].std(axis=0)

                print('')
                print(f'{wutils.bcolors.OKGREEN}Run {drop}{wutils.bcolors.ENDC}')
                print(f'\t{wutils.bcolors.OKGREEN}TEST SET{wutils.bcolors.ENDC}')
                print(f"\tDice\t {p_mean[1]:.3f} +/- {p_std[1]:.3f}\t IoU\t {p_mean[0]:.3f} +/- {p_std[0]:.3f}\t ASD\t {p_mean[2]:.3f} +/- {p_std[2]:.3f}\t HD95\t {p_mean[3]:.3f} +/- {p_std[3]:.3f}")

                for s in set(classes_str[drop]):
                    print(f'{wutils.bcolors.OKGREEN}\tTEST SET {s}{wutils.bcolors.ENDC}')
                    ind = [i for i,c in enumerate(classes_str[drop]) if c==s]

                    pc_mean = performance[drop,ind,:].mean(axis=0)
                    pc_std = performance[drop,ind,:].std(axis=0)
                    print(f"\tDice\t {pc_mean[1]:.3f} +/- {pc_std[1]:.3f}\t IoU\t {pc_mean[0]:.3f} +/- {pc_std[0]:.3f}\t ASD\t {pc_mean[2]:.3f} +/- {pc_std[2]:.3f}\t HD95\t {pc_mean[3]:.3f} +/- {pc_std[3]:.3f}")

            print('')
            acc_str = ' '.join(map(str, ['%.2f%%' % elem for elem in acc_coeff.acc]))
            print(f'Acc mean: {acc_str:s}')

            performance = performance.reshape(exp_config.DROPOUT_MC*len(self.infDataLoader)*exp_config.BATCH_SIZE,-1)
            file_names = [item for sublist in file_names for item in sublist]
            classes_str = [item for sublist in classes_str for item in sublist]
            dropout = [[drop] * len(self.infDataLoader)*exp_config.BATCH_SIZE for drop in range(exp_config.DROPOUT_MC)]
            dropout = [item for sublist in dropout for item in sublist]

            classes = [item for sublist in classes for item in sublist]
            class_predictions = [item for sublist in class_predictions for item in sublist]
            class_confidences = [item for sublist in class_confidences for item in sublist]

            cols = {'dropout run': dropout}
            eutils.save_to_excel(file_names, performance, measures, 'test-segm-mc', self.expConfig.RESULTS_FILE, classes_str, task='segment', **cols)

            eutils.evaluation(self.expConfig.CLASSES, classes, class_predictions, class_confidences,
                              'test-class', self.expConfig.RESULTS_FILE, file_names[:len(self.infDataLoader)*exp_config.BATCH_SIZE_INF])
