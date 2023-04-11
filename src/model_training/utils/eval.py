import matplotlib.pyplot as plt
import numpy as np
import itertools


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix without normalization')

    print(cm)

    tresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i, cm[i, j],
            horizontalalignment='center',
            color='white' if cm[i, j] > tresh else 'black'
        )

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    # plt.show()


def keras_model_evaluation(model, x_test, y_test, batch_size):
    # evaluate model on test set
    evaluation = model.evaluate(x_test, y_test, batch_size)
    print('test loss, test acc: ', evaluation)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    print('Generate predictions for test set')
    probas = model.predict(x_test)
    print('predictions on test set: {}'.format(probas))

    # map probabilities to 0 and 1
    y_hat = probas
    for prediction in range(len(y_hat)):
        if y_hat[prediction] >= 0.5:
            y_hat[prediction] = 1
        else:
            y_hat[prediction] = 0

    return probas, y_hat
