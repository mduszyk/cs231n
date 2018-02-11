import matplotlib.pyplot as plt


def plot_learning_curves(loss, train_acc, val_acc, show=True, save_path=None):
    plt.subplot(1, 2, 1)
    plt.title('Training loss')
    plt.xlabel('Iteration')
    plt.plot(loss, 'o', label="loss")
    plt.legend(loc='upper center', ncol=4)

    plt.subplot(1, 2, 2)
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.plot(train_acc, '-o', label="train")
    plt.plot(val_acc, '-o', label="val")
    plt.legend(loc='upper center', ncol=4)

    plt.gcf().set_size_inches(15, 5)

    if show:
        plt.show()

    if save_path is not None:
        plt.savefig(save_path)
