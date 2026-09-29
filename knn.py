

import random
import sys
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

from data_utils import load_CIFAR10

def _vis_samples():
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    samples_per_class = 10
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y_train == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()


class KNN(object):
    __slots__ = ('X_train', 'y_train', 'num_classes')

    def __init__(self, num_classes:int):
        self.num_classes: int = num_classes
        pass

    def train(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X # (N, D)
        self.y_train = y # (N, )

    def _asset_input_shape(self, input_x: np.ndarray):
        if input_x.shape[-self.X_train.ndim+1:] != self.X_train.shape[1:]:
            raise ValueError(
                f"input_x trailing shape {input_x.shape[-self.X_train.ndim + 1:]} "
                f"does not match X_train sample shape {self.X_train.shape[1:]}"
            )

    def compute_distances_naive(self, input_x: np.ndarray) -> np.ndarray:
        """
            return the (B, N) L2 distance matrix between input_x (B, D) and X_train (N, D).
            int (B, D) -> out (B, Xtrain_B)
            for each input (B) has num_trained amount of l2 distances to each train example
        """
        self._asset_input_shape(input_x)

        num_inputs = input_x.shape[0]
        num_trained = self.X_train.shape[0]
        # each input (B) has num_trained amount of l2 distances to each train example
        out = np.zeros((num_inputs, num_trained))

        for i in range(num_inputs):
            for j in range(num_trained):
                inp_v = input_x[i]
                train_v = self.X_train[j]
                l2 = np.sqrt(np.sum((inp_v - train_v)**2))
                out[i][j] = l2

        return out

    def compute_distances_a_bit_faster(self, input_x: np.ndarray) -> np.ndarray:
        """
            return the (B, N) L2 distance matrix between input_x (B, D) and X_train (N, D).
            int (B, D) -> out (B, Xtrain_B)
            for each input (B) has num_trained amount of l2 distances to each train example
        """
        self._asset_input_shape(input_x)

        num_inputs = input_x.shape[0]
        num_trained = self.X_train.shape[0]
        # each input (B) has num_trained amount of l2 distances to each train example
        out = np.zeros((num_inputs, num_trained))

        for i in range(num_inputs):
            # broadcast one (D, ) at i over the entire train (N, D) to get (N, D)
            # and then sum over the D dimension (axis=1) to get
            # one L2 distance per N (training sample) (N, )
            l2 = np.sqrt(
                np.sum((input_x[i,:] - self.X_train) ** 2, axis=1)
            )
            out[i] = l2

        return out

    def compute_distances_fast(self, input_x: np.ndarray) -> np.ndarray:
        """
            return the (B, N) L2 distance matrix between input_x (B, D) and X_train (N, D).
            int (B, D) -> out (B, Xtrain_B)
            for each input (B) has num_trained amount of l2 distances to each train example
        """

        '''
            l2 distance is a square of a vector: 
            ||a - b||^2 = ||v||^2
            
            meaning it's a vector that dotproducts with itself:
            
            ||v||^2 = sum_i (v_i^2)
            
            definition:
            ||a - b||^2 = sum_i  ( (a_i - b_i)^2 )
            
            expand the square:
            (a_i - b_i)^2 = a_i^2 - 2 * a_i * b_i + b_i^2
            
            so:
            ||a - b||^2 = sum_i ( a_i^2 - 2 * a_i * b_i + b_i^2 )
            
            we are summing sums, we can just rearrage the order how we sum:
            
            sum_i ( a_i^2 - 2 * a_i * b_i + b_i^2 ) = sum_i (a_i^2) - 2 * sum_i (a_i * b_i) + sum_i (b_i^2)
            
            so:
            ||a - b||^2 = sum_i (a_i^2) - 2 * sum_i (a_i * b_i) + sum_i (b_i^2)
            
            and recognize the definitions:
            
            ||a - b||^2 = ||a||^2 - 2 * dotproduct(a, b) + ||b||^2
        '''

        self._asset_input_shape(input_x)

        # this is just doing ||a - b||^2 right away
        # with full broadcast to (B, N, D) this is (B * N * D) float32 allocated -> oom
        #shape_test_1 = input_x[:,None,:] - self.X_train
        #l2 = np.sqrt(np.sum((shape_test_1 **2), axis=-1))

        # galaxy brain math version based on the algebraic identity
        # (B, 1) aka the ||a||^2
        inp_dot = (input_x ** 2).sum(axis=1, keepdims=True)
        # (N, 1) aka  the ||b||^2
        train_dot = (self.X_train ** 2).sum(axis=1, keepdims=True)
        # (B, D) @ (N, D).T -> (B, N) aka dotproduct(a, b) achieved with matmul
        inp_train_dot = input_x @ self.X_train.T

        # ||a||^2 - 2 * dotproduct(a, b) + ||b||^2)
        #  transpose train_dot (N, 1) -> (1, N) so each train norm broadcasts over the (B, N) result
        distances_squared = inp_dot - (2 * inp_train_dot) + train_dot.T
        l2 = np.sqrt(distances_squared)

        # (B, N)
        # each input (B) has num_trained amount of l2 distances to each train example
        return l2

    def predict_labels(self, dist: np.ndarray, k: int = 1) -> np.ndarray:
        '''
        dist is a distance matrix (num_inputs, num_train) with distances for each train point to the input point
        so dist[i, j] is the distance between x_input[i] and X_train[j]
        return np array y of shape (num_inputs, )
        where y [i] is the predicted label for the test point intput_x[i]
        '''

        assert k >= 1, f"k must me greate or equal to 1, was {k}"

        sorted_distances = np.argsort(dist, axis=1)  # lo to hi

        shortest_k_distances = sorted_distances[:, :k]

        k_closest_ys = self.y_train[shortest_k_distances]
        bincounts = np.array([
            np.bincount(row, minlength=self.num_classes)
            for row in k_closest_ys
        ])  # (num_inputs, num_classes) count for each class at the class index

        top_voted = np.argmax(bincounts, axis=1)
        return top_voted


def _show_sample_plot(x_train: np.ndarray, index: int = 0):
    img = x_train[index].reshape(32, 32, 3).astype('uint8')
    plt.imshow(img)
    plt.show()

def _show_single_chanel_plot(x_train: np.ndarray, index: int = 0):
    img = x_train[index].reshape(32, 32, 3)
    plt.subplot(1, 3, 1)
    plt.imshow(img[:, :, 0], cmap='Reds')
    plt.title('R')
    plt.subplot(1, 3, 2)
    plt.imshow(img[:, :, 1], cmap='Greens')
    plt.title('G')
    plt.subplot(1, 3, 3)
    plt.imshow(img[:, :, 2], cmap='Blues')
    plt.title('B')
    plt.show()


import time
def bench(fn, X, label):
    t0 = time.perf_counter()
    out = fn(X)
    dt = time.perf_counter() - t0
    print(f"{label:12s} {dt:.3f}s")
    return out

if __name__ == "__main__":
    print(f"python: {sys.version}")
    print(f"numpy:  {np.__version__}")

    cifar10_dir = r"datasets/cifar-10-batches-py"
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    print('Training data shape: ', X_train.shape)
    print('Training labels shape: ', y_train.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)

    # Subsample the data for more efficient code execution in this exercise
    num_training = 5000
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]

    num_test = 500
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # For a NumPy array or PyTorch tensor of shape (32, 32, 16), the last dimension is the fastest-varying (most contiguous) in memory.
    # This is row-major (C-order) layout, the default for both libraries.
    # Reshape the image data into rows (B, W, H, C) -> (B, W*H*C)
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    print(X_train.shape, X_test.shape)

    cifar_classes = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    knn: KNN = KNN(len(cifar_classes))
    knn.train(X_train, y_train)

    test_slice_count: int = 500
    k: int = 5
    distances = bench(knn.compute_distances_naive, X_test[:test_slice_count],"naive")
    predicted_labels = knn.predict_labels(distances, k=k)

    print([cifar_classes[i] for i in predicted_labels])
    print([cifar_classes[i] for i in y_test[:test_slice_count]])

    accuracy = (predicted_labels == y_test[:test_slice_count]).mean()
    print(f"naive params: {k=}, {test_slice_count=}, {accuracy=:.3f}")


    print("a bit faster...")
    distances2 = bench(knn.compute_distances_a_bit_faster, X_test[:test_slice_count], "a-bit-faster")
    predicted_labels = knn.predict_labels(distances2, k=k)
    print("a bit faster all close: ", np.allclose(distances, distances2))  # True
    accuracy = (predicted_labels == y_test[:test_slice_count]).mean()
    print(f"a bit faster params: {k=}, {test_slice_count=}, {accuracy=:.3f}")

    print("fast...")
    distances3 = bench(knn.compute_distances_fast, X_test[:test_slice_count], "full-vectorized")
    predicted_labels = knn.predict_labels(distances3, k=k)
    print("fast all close: ", np.allclose(distances, distances3))
    accuracy = (predicted_labels == y_test[:test_slice_count]).mean()
    print(f"fast params: {k=}, {test_slice_count=}, {accuracy=:.3f}")



    # CROSS VALIDATE
    num_folds = 5
    k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

    X_train_folds = np.array_split(X_train, num_folds, axis=0)
    y_train_folds = np.array_split(y_train, num_folds, axis=0)
    num_classes = len(cifar_classes)
    k_to_accuracies = {k: [] for k in k_choices}

    for f in range(num_folds):
        X_val = X_train_folds[f]
        y_val = y_train_folds[f]

        # all the other folds become the train set
        X_tr = np.concatenate([X_train_folds[i] for i in range(num_folds) if i != f], axis=0)
        y_tr = np.concatenate([y_train_folds[i] for i in range(num_folds) if i != f], axis=0)

        knn = KNN(num_classes)
        knn.train(X_tr, y_tr)

        dists = knn.compute_distances_fast(X_val)

        for k in k_choices:
            preds = knn.predict_labels(dists, k=k)
            acc = (preds == y_val).mean()
            k_to_accuracies[k].append(acc)

    for k in sorted(k_to_accuracies):
        for accuracy in k_to_accuracies[k]:
            print('k = %d, accuracy = %f' % (k, accuracy))

    # plot
    for k in k_choices:
        accs = k_to_accuracies[k]
        plt.scatter([k] * len(accs), accs, color='blue')
        plt.plot(k, np.mean(accs), marker='o', color='red')

    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.show()

    pass