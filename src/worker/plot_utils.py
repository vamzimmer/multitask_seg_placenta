import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# colormap for visualizations
mycmap = plt.cm.jet
mycmap._init()
mycmap._lut[:, -1] = np.linspace(0, 0.8, 255 + 4)


class PlotSegmentationProgress(object):

    def __init__(self, out_file):

        self.out_file = out_file
        self.fig = None
        self.ax = []

        self._initialize_plot()

    """
        This function has to be overwritten by classes inheriting from ImageFusion!
    """
    def _initialize_plot(self):
        plt.ion()
        # f, a = plt.subplots(1, 3, figsize=(15, 6))
        self.fig = plt.figure(constrained_layout=True, figsize=(15, 6))
        gs = GridSpec(2, 3, figure=self.fig)
        self.ax.append(self.fig.add_subplot(gs[0, 0]))  # ax00 train ground truth
        self.ax.append(self.fig.add_subplot(gs[1, 0]))  # ax10 validate ground truth
        self.ax.append(self.fig.add_subplot(gs[0, 1]))  # ax01 train prediction
        self.ax.append(self.fig.add_subplot(gs[1, 1]))  # ax11 validate prediction
        self.ax.append(self.fig.add_subplot(gs[:, 2]))  # ax2

    """
       Call
    """
    def __call__(self, image, label, prediction, train_losses,
                 val_image, val_label, val_prediction, val_losses,
                 acc_losses, acc_losses_val, epoch,
                 isatend=False):
        self.ax[0].clear()
        self.ax[0].imshow(image, cmap='gray')
        self.ax[0].imshow(label, cmap=mycmap, alpha=0.7)
        self.ax[0].axis('off')
        self.ax[0].set_title('(Train) Ground truth')
        plt.draw()

        self.ax[1].clear()
        self.ax[1].imshow(val_image, cmap='gray')
        self.ax[1].imshow(val_label, cmap=mycmap, alpha=0.7)
        self.ax[1].axis('off')
        self.ax[1].set_title('(Val) Ground truth')
        plt.draw()

        self.ax[2].clear()
        self.ax[2].imshow(image, cmap='gray')
        self.ax[2].imshow(prediction, cmap=mycmap, alpha=0.7)
        self.ax[2].axis('off')
        self.ax[2].set_title('(Train) Estimated: loss={:.3f}; dice={:.3f}'.format(train_losses[0], train_losses[1]))
        plt.draw()

        self.ax[3].clear()
        self.ax[3].imshow(val_image, cmap='gray')
        self.ax[3].imshow(val_prediction, cmap=mycmap, alpha=0.7)
        self.ax[3].axis('off')
        self.ax[3].set_title('(Val) Estimated: loss={:.3f}; dice={:.3f}'.format(val_losses[0], val_losses[1]))
        plt.draw()

        self.ax[4].clear()
        self.ax[4].plot(acc_losses[:, 0], label='training loss')
        self.ax[4].plot(acc_losses[:, 1], label='training dice')
        self.ax[4].plot(acc_losses_val[:, 0], label='validation loss')
        self.ax[4].plot(acc_losses_val[:, 1], label='validation dice')
        self.ax[4].set_title("Epoch: {} | total loss: {:.4f}".format(epoch, acc_losses[-1, 0]))
        self.ax[4].set_ylabel("loss")
        self.ax[4].set_xlabel("epoch")
        self.ax[4].legend()
        plt.draw()

        plt.pause(0.001)
        plt.show()

        self.fig.savefig(self.out_file)

        if isatend:
            plt.ioff()
            self.fig.savefig(self.out_file)
            plt.close('all')


class PlotClassificationAMapsProgress(object):

    def __init__(self, n_attentions, out_file):

        self.out_file = out_file
        self.n_attentions = n_attentions
        self.fig = None
        self.ax = []

        self._initialize_plot()

    """
        This function has to be overwritten by classes inheriting from ImageFusion!
    """
    def _initialize_plot(self):
        plt.ion()
        # f, a = plt.subplots(1, 3, figsize=(15, 6))
        self.fig = plt.figure(constrained_layout=True, figsize=(15, 6))
        gs = GridSpec(2, 2+self.n_attentions, figure=self.fig)
        self.ax.append(self.fig.add_subplot(gs[0, 0]))  # ax00 train ground truth
        self.ax.append(self.fig.add_subplot(gs[1, 0]))  # ax10 validate ground truth
        self.ax.append([])
        self.ax.append([])
        for na in range(self.n_attentions):
            self.ax[2].append(self.fig.add_subplot(gs[0, 1 + na]))
            self.ax[3].append(self.fig.add_subplot(gs[1, 1 + na]))
        self.ax.append(self.fig.add_subplot(gs[0, self.n_attentions + 1]))
        self.ax.append(self.fig.add_subplot(gs[1, self.n_attentions + 1]))

    """
       Call
    """
    def __call__(self, classes, image, label, prediction, train_loss,
                 val_image, val_label, val_prediction, val_loss,
                 attention, val_attention,
                 acc_losses, val_acc_losses, epoch,
                 isatend=False):
        self.ax[0].clear()
        self.ax[0].imshow(image, cmap='gray')
        self.ax[0].axis('off')
        self.ax[0].set_title('(Train) True label: {:s} ({:s})\nPredicted: {:s} ({:s})\n loss={:.3f}'
                             .format(classes[label], ', '.join(classes), classes[prediction[1]],
                                     ', '.join(map(str, ['%.2f' % elem for elem in prediction[0]])), train_loss))
        plt.draw()

        self.ax[1].clear()
        self.ax[1].imshow(val_image, cmap='gray')
        self.ax[1].axis('off')
        self.ax[1].set_title('(Val) True label: {:s} ({:s})\nPredicted: {:s} ({:s})\n loss={:.3f}'
                       .format(classes[val_label], ', '.join(classes),
                               classes[val_prediction[1]],
                               ', '.join(map(str, ['%.2f' % elem for elem in val_prediction[0]])),
                               val_loss))
        plt.draw()

        for a in range(0, self.n_attentions):
            self.ax[2][a].clear()
            self.ax[2][a].imshow(image, cmap='gray')
            self.ax[2][a].imshow(attention[a], cmap=mycmap, alpha=0.7)
            self.ax[2][a].axis('off')
            self.ax[2][a].set_title('(Train) Attention layer {}'.format(a))

            plt.draw()

            self.ax[3][a].clear()
            self.ax[3][a].imshow(val_image, cmap='gray')
            self.ax[3][a].imshow(val_attention[a], cmap=mycmap, alpha=0.7)
            self.ax[3][a].axis('off')
            self.ax[3][a].set_title('(Val) Attention layer {}'.format(a))

            plt.draw()

        self.ax[-2].clear()
        self.ax[-2].plot(acc_losses[:, 0], label='training loss')
        self.ax[-2].plot(val_acc_losses[:, 0], label='validation loss')
        self.ax[-2].set_title("Epoch: {} \n total loss: {:.4f}".format(epoch, acc_losses[-1, 0]))
        self.ax[-2].set_ylabel("loss")
        self.ax[-2].set_xlabel("epoch")
        self.ax[-2].legend()
        plt.draw()

        self.ax[-1].clear()
        for j in range(len(classes)):
            self.ax[-1].plot(acc_losses[:, j+1], label='train class {:d}'.format(j))
            self.ax[-1].plot(val_acc_losses[:, j+1], label='val class {:d}'.format(j))
        self.ax[-1].set_title("Epoch: {} \n mean acc: {:.4f}".format(epoch, np.mean(acc_losses[-1, :])))
        self.ax[-1].set_ylabel("acc")
        self.ax[-1].set_xlabel("epoch")
        self.ax[-1].legend()
        plt.draw()

        plt.pause(0.001)
        plt.show()
        self.fig.savefig(self.out_file)

        if isatend:
            plt.ioff()
            self.fig.savefig(self.out_file)
            plt.close('all')


class PlotMultitaskProgress(object):

    def __init__(self, n_attentions, out_file):

        self.out_file = out_file
        self.n_attentions = n_attentions
        self.fig = None
        self.ax = []

        self._initialize_plot()

    """
        This function has to be overwritten by classes inheriting from ImageFusion!
    """
    def _initialize_plot(self):
        plt.ion()
        # f, a = plt.subplots(1, 3, figsize=(15, 6))
        self.fig = plt.figure(constrained_layout=True, figsize=(15, 6))
        gs = GridSpec(2, 4+self.n_attentions, figure=self.fig)
        self.ax.append(self.fig.add_subplot(gs[0, 0]))  # ax00 train ground truth
        self.ax.append(self.fig.add_subplot(gs[1, 0]))  # ax10 validate ground truth
        self.ax.append(self.fig.add_subplot(gs[0, 1]))  # ax01 train prediction
        self.ax.append(self.fig.add_subplot(gs[1, 1]))  # ax11 validate prediction
        self.ax.append([])
        self.ax.append([])
        for na in range(self.n_attentions):
            self.ax[4].append(self.fig.add_subplot(gs[0, 2 + na]))
            self.ax[5].append(self.fig.add_subplot(gs[1, 2 + na]))
        self.ax.append(self.fig.add_subplot(gs[:, self.n_attentions + 2]))
        self.ax.append(self.fig.add_subplot(gs[0, self.n_attentions + 3]))
        self.ax.append(self.fig.add_subplot(gs[1, self.n_attentions + 3]))

    """
       Call
    """
    def __call__(self, classes, image1, image2, label, class_label, prediction, class_prediction, train_loss,
                 val_image, val_label, val_class_label, val_prediction, val_class_prediction, val_loss,
                 attention, val_attention,
                 acc_losses, val_acc_losses, epoch,
                 isatend=False):
        self.ax[0].clear()
        self.ax[0].imshow(image2, cmap='gray')
        self.ax[0].imshow(label, cmap=mycmap, alpha=0.7)
        self.ax[0].axis('off')
        self.ax[0].set_title('(Train) Reference')
        plt.draw()

        self.ax[1].clear()
        self.ax[1].imshow(val_image, cmap='gray')
        self.ax[1].imshow(val_label, cmap=mycmap, alpha=0.7)
        self.ax[1].axis('off')
        self.ax[1].set_title('(Val) Reference \n True class: {:s}'
                             .format(classes[val_class_label]))
        plt.draw()

        self.ax[2].clear()
        self.ax[2].imshow(image2, cmap='gray')
        self.ax[2].imshow(prediction, cmap=mycmap, alpha=0.7)
        self.ax[2].axis('off')
        self.ax[2].set_title('(Train) Estimated \nloss={:.3f}; dice={:.3f}'.format(train_loss[0], train_loss[3]))
        plt.draw()

        self.ax[3].clear()
        self.ax[3].imshow(val_image, cmap='gray')
        self.ax[3].imshow(val_prediction, cmap=mycmap, alpha=0.7)
        self.ax[3].axis('off')
        self.ax[3].set_title('(Val) Estimated \n loss={:.3f}; dice={:.3f}'.format(val_loss[0], val_loss[3]))
        plt.draw()

        for a in range(0, self.n_attentions):
            self.ax[4][a].clear()
            self.ax[4][a].imshow(image1, cmap='gray')
            self.ax[4][a].imshow(attention[a], cmap=mycmap, alpha=0.7)
            self.ax[4][a].axis('off')
            self.ax[4][a].set_title('(Train) Attention layer {} \nTrue class: {:s}\n Predicted class: {:s}'
                                    .format(a, classes[class_label], classes[class_prediction[1]]))

            plt.draw()

            self.ax[5][a].clear()
            self.ax[5][a].imshow(val_image, cmap='gray')
            self.ax[5][a].imshow(val_attention[a], cmap=mycmap, alpha=0.7)
            self.ax[5][a].axis('off')
            self.ax[5][a].set_title('(Val) Attention layer {}\n Predicted class: {:s}'
                                    .format(a, classes[val_class_prediction[1]]))

        self.ax[6].clear()
        self.ax[6].plot(acc_losses[:, 0], label='training loss')
        self.ax[6].plot(acc_losses[:, 3], label='training dice')
        self.ax[6].plot(val_acc_losses[:, 0], label='validation loss')
        self.ax[6].plot(val_acc_losses[:, 3], label='validation dice')
        self.ax[6].set_title("Epoch: {} | total loss: {:.4f}".format(epoch, acc_losses[-1, 0]))
        self.ax[6].set_ylabel("loss")
        self.ax[6].set_xlabel("epoch")
        self.ax[6].legend()
        plt.draw()

        self.ax[7].clear()
        self.ax[7].plot(acc_losses[:, 1], label='training loss segmentation')
        self.ax[7].plot(acc_losses[:, 2], label='training loss classification')
        self.ax[7].plot(val_acc_losses[:, 1], label='validation loss segmentation')
        self.ax[7].plot(val_acc_losses[:, 2], label='validation dice classification')
        self.ax[7].set_title("Epoch: {} | total loss segm: {:.4f} \n total loss class: {:.4f}"
                             .format(epoch, acc_losses[-1, 1], acc_losses[-1, 2]))
        self.ax[7].set_ylabel("loss")
        self.ax[7].set_xlabel("epoch")
        self.ax[7].legend()
        plt.draw()

        self.ax[8].clear()
        for j in range(len(classes)):
            self.ax[8].plot(acc_losses[:, j + 4], label='train class {:d}'.format(j))
            self.ax[8].plot(val_acc_losses[:, j + 4], label='val class {:d}'.format(j))
        self.ax[8].set_title("Epoch: {} \n mean acc: {:.4f}".format(epoch, np.mean(acc_losses[-1, :])))
        self.ax[8].set_ylabel("acc")
        self.ax[8].set_xlabel("epoch")
        self.ax[8].legend()
        plt.draw()

        plt.pause(0.001)
        plt.show()

        self.fig.savefig(self.out_file)

        if isatend:
            plt.ioff()
            self.fig.savefig(self.out_file)
            plt.close('all')