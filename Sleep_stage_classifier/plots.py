import matplotlib.pyplot as plt
import numpy as np
# TODO: You can use other packages if you want, e.g., Numpy, Scikit-learn, etc.


def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
	# TODO: Make plots for loss curves and accuracy curves.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
	plt.figure()
	plt.plot(np.arange(len(train_losses)), train_losses, label='Training loss')
	plt.plot(np.arange(len(valid_losses)), valid_losses, label='Validation loss')
	plt.ylabel('Loss')
	plt.xlabel('epoch')
	plt.legend(loc="best")
	plt.title('Loss curve')
	plt.savefig('loss curve.png')
    
	plt.figure()
	plt.plot(np.arange(len(train_accuracies)), train_accuracies, label='Train Accuracy')
	plt.plot(np.arange(len(valid_accuracies)), valid_accuracies, label='Validation Accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('epoch')
	plt.legend(loc="best")
	plt.title('Accuracy curve')
	plt.savefig('Accuracy curve.png')


def plot_confusion_matrix(results, class_names):
	# TODO: Make a confusion matrix plot.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
    from sklearn.metrics import confusion_matrix
    from sklearn.utils.multiclass import unique_labels
    from sklearn.metrics import confusion_matrix
    results_array=np.asarray(results)
    cm = confusion_matrix(results_array[:,0], results_array[:,1])
    '''normalize'''
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure()
    title='Normalized Confusion Matrix'
    cmap=plt.cm.Blues
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=class_names, yticklabels=class_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig('Confusion Matrix.png')
    pass
