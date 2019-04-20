import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


FIGNAME="le_plot"


def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
    __plot_compare(train_losses, valid_losses, "Loss", filename=FIGNAME+"_loss")
    __plot_compare(train_accuracies, valid_accuracies, "Accuracy", filename=FIGNAME+"_acc")
    pass


def plot_confusion_matrix(results, class_names):
    y_true, y_pred = list(zip(*results))
    cm = confusion_matrix(y_true, y_pred)

    normalize = True
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=class_names, yticklabels=class_names,
           title="Normalized Confusion Matrix",
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    # fig.savefig("figures/"+FIGNAME+"_confusion.pdf")


def __plot_compare(y_train, y_valid, title, filename="figure"):
    fig = plt.figure()
    ax = plt.axes()
    l_train = ax.plot(y_train, "r")
    l_valid = ax.plot(y_valid, "b")
    plt.plot(y_train, "r", y_valid, "b")
    plt.title(title)

    plt.legend(labels=("Training", "Valid"),
               loc="upper right")

    # ax.axes.get_xaxis().set_visible(False)

    plt.show()
    # fig.savefig("figures/"+filename+".pdf")
    pass