import matplotlib.pyplot as plt


def plot_learning_curves(**kwargs):
    loss = kwargs["loss"]
    train_acc = kwargs["train_acc"]
    val_acc = kwargs["val_acc"]
    show = kwargs.get("show", True)
    save_path = kwargs.get("save_path")

    plt.subplot(1, 2, 1)
    plt.title('Training loss')
    plt.xlabel('Iteration')
    plt.plot(loss, label="loss")
    plt.legend(loc='upper center', ncol=4)

    plt.subplot(1, 2, 2)
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.plot(train_acc, '-o', label="train")
    plt.plot(val_acc, '-o', label="val")
    plt.legend(loc='upper left', ncol=4)

    fig = plt.gcf()
    fig.set_size_inches(15, 5)

    if save_path is not None:
        fig.savefig(save_path)

    if show:
        plt.show()

