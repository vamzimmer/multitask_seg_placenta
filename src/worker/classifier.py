import torch
import numpy as np
import os
import time
from random import randint

from src.utils import python_utils as putils
from src.utils import eval_utils as cutils
from src.worker import plot_utils as plutils
from src.worker import worker_utils as wutils


class Classifier:

    def __init__(self, expConfig, trainDataLoader, valDataLoader, infDataLoader):
        self.expConfig = expConfig
        self.trainDataLoader = trainDataLoader
        self.valDataLoader = valDataLoader
        self.infDataLoader = infDataLoader
        self.logfile = self.expConfig.OUT_DIR + '/log.txt'

        self.cuda = False
        if next(expConfig.model.parameters()).is_cuda:
            self.cuda = True

        self.accumulated_losses = np.zeros((expConfig.EPOCHS, 1+self.expConfig.N_CLASSES))
        self.accumulated_losses_val = np.zeros((expConfig.EPOCHS, 1+self.expConfig.N_CLASSES))

        # load model
        expConfig.START_EPOCH = 0
        if expConfig.MODE == 'infer' or expConfig.RESUME:
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
                if expConfig.MODE == 'infer':
                    exit(1)


        print()
        nr_params = wutils.count_parameters(expConfig.model)
        print("{}{} parameters to train{}".format(wutils.bcolors.WARNING, nr_params, wutils.bcolors.ENDC))
        print()

    # ===================================================================================================

    def train(self):
        print("Training...")

        exp_config = self.expConfig
        n_attentions = len(self.expConfig.AM_POS)

        best_perf = 0
        best_loss = 10
        plotter = None
        if exp_config.PLOT_PROGRESS:
            plotter = plutils.PlotClassificationAMapsProgress(n_attentions, exp_config.PROGRESS_FILE)

        # accumulated_losses = np.zeros((exp_config.EPOCHS, 1+self.expConfig.N_CLASSES))
        # accumulated_losses_val = np.zeros((exp_config.EPOCHS, 1+self.expConfig.N_CLASSES))

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
            loss, acc, image, label, estim, attention, train_loss = self.train_one_epoch(epoch)
            self.accumulated_losses[epoch, 0] = loss
            self.accumulated_losses[epoch, 1:] = acc

            # evaluate on validation set
            perf_loss, perf_acc, val_image, val_label, val_estim, val_attention, val_loss = self.validate(epoch)
            self.accumulated_losses_val[epoch, 0] = perf_loss
            self.accumulated_losses_val[epoch, 1:] = perf_acc

            # remember best performance and save checkpoint
            is_best_acc = np.mean(perf_acc) > best_perf
            is_best_loss = np.mean(perf_loss) < best_loss
            best_perf = max(np.mean(perf_acc), best_perf)
            best_loss = min(np.mean(perf_loss), best_loss)
            wutils.save_checkpoint({'epoch': epoch + 1, 'model_state_dict': exp_config.model.state_dict(), 'best_perf': best_perf,
                             'optim_state_dict': exp_config.optimizer.state_dict(),
                             'scheduler_state_dict': exp_config.lr_scheduler.state_dict(),
                             'acc_loss': self.accumulated_losses, 'acc_loss_val': self.accumulated_losses_val, }, 1 if is_best_acc else 0,
                            os.path.join(exp_config.OUT_DIR, exp_config.EXPERIMENT_PREFIX))
            wutils.save_checkpoint(
                {'epoch': epoch + 1, 'model_state_dict': exp_config.model.state_dict(), 'best_perf': best_perf,
                 'optim_state_dict': exp_config.optimizer.state_dict(),
                 'scheduler_state_dict': exp_config.lr_scheduler.state_dict(), 
                 'acc_loss': self.accumulated_losses, 'acc_loss_val': self.accumulated_losses_val, }, 2 if is_best_loss else 0,
                os.path.join(exp_config.OUT_DIR, exp_config.EXPERIMENT_PREFIX))

            if exp_config.PLOT_PROGRESS:
                plotter(exp_config.CLASSES, image[0, :, :].transpose(0, 1),
                        label, estim, train_loss,
                        val_image[0, :, :].transpose(0, 1), val_label, val_estim, val_loss,
                        [att[0, :, :].transpose(0, 1) for att in attention] if n_attentions else None,
                        [vatt[0, :, :].transpose(0, 1) for vatt in val_attention] if n_attentions else None,
                        self.accumulated_losses[0:epoch + 1, :], self.accumulated_losses_val[0:epoch + 1, :], epoch,
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
        acc_coeff = putils.AccuracyMeter(self.expConfig.N_CLASSES)

        # switch to train mode
        self.expConfig.model.train()

        plot_id1 = randint(0, len(self.trainDataLoader) - 1)

        end = time.time()
        for i, (input_, target, _) in enumerate(self.trainDataLoader):

            # measure data loading time
            data_time.update(time.time() - end)

            if self.cuda:
                input_ = input_.type(torch.FloatTensor).cuda() #(async=True)
                target = target.type(torch.FloatTensor).cuda() #(async=True)

            # compute output
            _, output, attention = self.expConfig.model(input_)

            print(output.size())
            print(torch.max(target, 1)[1].size())

            loss = self.expConfig.criterion(output, torch.max(target, 1)[1])

            # record loss and measure dice coeff
            losses.update(loss.item(), input_.size(0))

            # compute gradient and do SGD step
            loss.backward()
            self.expConfig.optimizer.step()
            self.expConfig.optimizer.zero_grad()

            # compute balanced accuracy, sensitivity and specificity
            with torch.no_grad():
                _, predicted = torch.max(output, 1)
                _, gt = torch.max(target, 1)

                acc_coeff.update(predicted.cpu().numpy(), gt.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i == plot_id1:
                plot_id2 = randint(0, input_.size()[0] - 1)

                if self.expConfig.PLOT_PROGRESS:
                    image = wutils.get_image_to_show(input_, plot_id2).cpu().detach()
                    if attention:
                        attention_maps = [wutils.get_image_to_show(attention[x], plot_id2).cpu().detach() for x in range(0, len(attention))]
                    else:
                        attention_maps = None
                    estim = [output[plot_id2].detach(), predicted[plot_id2].item()]
                    label = gt[plot_id2].item()
                    train_loss = losses.val

            # output
            if i % self.expConfig.PRINT_FREQ == 0:
                acc_str = ' '.join(map(str, ['%.2f%%' % elem for elem in acc_coeff.acc]))
                print_str = (f'Epoch: [{epoch}][{i}/{len(self.trainDataLoader)}]\t Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t Data {data_time.val:.3f} ({data_time.avg:.3f})'
                             f'\t Loss {losses.val:.4f} ({losses.avg:.4f})\t Acc {acc_str:s}')
                print(print_str)
                f.write(print_str)

        acc_str = ' '.join(map(str, ['%.2f%%' % elem for elem in acc_coeff.acc]))
        print_str = (f'TrainAvg: Loss {losses.avg:.4f}\t Acc-score {acc_str:s}')
        print(print_str)
        f.write(print_str)
        f.close()

        if self.expConfig.PLOT_PROGRESS:
            return losses.avg, acc_coeff.acc, image, label, estim, attention_maps, train_loss
        else:
            return losses.avg, acc_coeff.acc, [], [], [], [], []

    # ===================================================================================================

    def validate(self, epoch):
        f = open(self.logfile, 'a')

        print("Validation...")
        f.write("Validation...\n")

        batch_time = putils.AverageMeter()
        losses = putils.AverageMeter()
        acc_coeff = putils.AccuracyMeter(self.expConfig.N_CLASSES)

        self.expConfig.model.eval()

        plot_id = randint(0, len(self.valDataLoader) - 1)

        end = time.time()
        for i, (input_, target, _) in enumerate(self.valDataLoader):

            if self.cuda:
                target = target.type(torch.FloatTensor).cuda() #(async=True)
                input_ = input_.type(torch.FloatTensor).cuda() #(async=True)

            # compute output
            _, output, attention = self.expConfig.model(input_)

            # print(torch.max(output))
            loss = self.expConfig.criterion(output, torch.max(target, 1)[1])

            # measure dice coefficient and record loss
            losses.update(loss.item(), input_.size(0))

            # compute balanced accuracy, sensitivity and specificity
            with torch.no_grad():
                _, predicted = torch.max(output, 1)
                _, gt = torch.max(target, 1)

                acc_coeff.update(predicted.cpu().numpy(), gt.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i == plot_id:
                plot_id2 = randint(0, input_.size()[0] - 1)

                if self.expConfig.PLOT_PROGRESS:
                    image = wutils.get_image_to_show(input_, plot_id2).cpu().detach()
                    if attention:
                        attention_maps = [wutils.get_image_to_show(attention[x], plot_id2).cpu().detach() for x in range(0, len(attention))]
                    else:
                        attention_maps = None
                    estim = [output[plot_id2].detach(), predicted[plot_id2].item()]
                    label = gt[plot_id2].item()
                    val_loss = losses.val

            if i % self.expConfig.PRINT_FREQ == 0:

                acc_str = ' '.join(map(str, ['%.2f%%' % elem for elem in acc_coeff.acc]))
                print_str = (f'Val: [{i}/{len(self.valDataLoader)}]\t Time {batch_time.val:.3f} ({batch_time.avg:.3f})'
                             f'\t Loss {losses.val:.4f} ({losses.avg:.4f})\t Acc {acc_str:s}')
                print(print_str)
                f.write(print_str)

        acc_str = ' '.join(map(str, ['%.2f%%' % elem for elem in acc_coeff.acc]))
        print_str = (f'ValAvg: Loss {losses.avg:.4f}\t Acc-score {acc_str:s}')
        print(print_str)
        f.write(print_str)

        if self.expConfig.PLOT_PROGRESS:
            return losses.avg, acc_coeff.acc, image, label, estim, attention_maps, val_loss
        else:
            return losses.avg, acc_coeff.acc, [], [], [], [], []

    # ===================================================================================================

    def infer(self):
        print("Inference...")

        acc_coeff = putils.AccuracyMeter(self.expConfig.N_CLASSES)

        exp_config = self.expConfig
        exp_config.model.eval()

        classes = [-1] * len(self.infDataLoader)
        class_predictions = [-1] * len(self.infDataLoader)
        class_confidences = [-1] * len(self.infDataLoader)
        file_names = [-1] * len(self.infDataLoader)

        if len(self.infDataLoader) > 0:
            for i, (input_, target, info) in enumerate(self.infDataLoader):
                name = info
                # print(name)

                if self.cuda:
                    target = target.type(torch.FloatTensor).cuda() #(async=True)
                    input_ = input_.type(torch.FloatTensor).cuda() #(async=True)

                # compute output
                _, output, _ = self.expConfig.model(input_)

                # compute balanced accuracy, sensitivity and specificity
                with torch.no_grad():
                    _, predicted = torch.max(output, 1)
                    _, gt = torch.max(target, 1)

                    acc_coeff.update(predicted.cpu().numpy(), gt.cpu().numpy())

                    classes[i] = gt.cpu().numpy()
                    class_predictions[i] = predicted.cpu().numpy()
                    class_confidences[i] = output.cpu().numpy()
                    file_names[i] = name

                for j in range(0, input_.size()[0]):
                    print(f'Predict: [{j}/{i}/{len(self.infDataLoader)}]\t Pred: {self.expConfig.CLASSES[predicted[j]]} \t True: {self.expConfig.CLASSES[gt[j]]}')
                    
            file_names = [item for sublist in file_names for item in sublist]
            classes = [item for sublist in classes for item in sublist]
            class_predictions = [item for sublist in class_predictions for item in sublist]
            class_confidences = [item for sublist in class_confidences for item in sublist]

            print('')
            print('TEST SET')
            acc_str = ' '.join(map(str, ['%.2f%%' % elem for elem in acc_coeff.acc]))
            print(f'Acc mean: {acc_str:s}')

            # ----------------------------------------------------------------------------------

            print('')
            # evaluation
            filename = self.expConfig.OUT_DIR + '/results.xlsx'
            cutils.evaluation(self.expConfig.CLASSES, classes, class_predictions, class_confidences,
                              'test', self.expConfig.RESULTS_FILE, file_names)
