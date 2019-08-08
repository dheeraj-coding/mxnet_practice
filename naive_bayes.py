from mxnet import nd, gluon
import matplotlib.pyplot as plt
import random

def show_images(imgs, rows, cols):
    _, axes = plt.subplots(rows, cols)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    return axes

def transform(data, label):
    return (data/128).astype('float32').squeeze(axis=-1), label

def train(train, n_classes):
    X, Y = train[:]
    n_y = nd.zeros(n_classes)
    for y in range(n_classes):
        n_y[y] = (Y==y).sum()
    P_y = n_y / n_y.sum()

    n_x = nd.zeros((n_classes, 28, 28))
    for y in range(n_classes):
        n_x[y] = nd.array(X.asnumpy()[Y==y].sum(axis=0))
    P_xy = (n_x+1) / (n_y+1).reshape((10, 1, 1))
    show_images(P_xy.asnumpy(), 2, 5)

    return P_xy, P_y

def predict(img, P_xy, P_y):
    img = img.expand_dims(axis=0)
    log_P_xy = nd.log(P_xy)
    neg_log_P_xy = nd.log(1-P_xy)
    pxy = log_P_xy * img + neg_log_P_xy * (1-img)
    pxy = pxy.reshape((10, -1)).sum(axis=1)

    probs = pxy+nd.log(P_y)

    return probs.argmax(axis=0).asscalar()

def test(test, n_classes, P_xy, P_y):
    X, Y = test[:]
    correct = 0
    for i, img in enumerate(X):
        result = predict(img, P_xy, P_y)
        if result == Y[i]:
            correct += 1
    acc = (correct/X.shape[0]) * 100
    print("Accuracy {}%".format(acc))
    return acc

    
def main():
    n_classes = 10

    mnist_train = gluon.data.vision.datasets.MNIST(train=True, transform=transform)
    mnist_test = gluon.data.vision.datasets.MNIST(train=False, transform=transform)

    P_xy, P_y = train(mnist_train, n_classes)
    
    acc = test(mnist_test, n_classes, P_xy, P_y)

    test_X, test_Y = mnist_test[:]
    index = random.randint(0, test_X.shape[0])
    result = predict(test_X[index], P_xy, P_y)
    
    print("Predicted value {}".format(result))
    plt.imshow(test_X[index].asnumpy())
    plt.show()
   
if __name__ == "__main__":
    main()
