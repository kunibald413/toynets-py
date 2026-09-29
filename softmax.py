import numpy as np
from data_utils import load_CIFAR10
import sys

import matplotlib.pyplot as plt


def _show_mean_comparison(some_img, mean_img):
    centered_img = some_img - mean_img

    centered_vis = centered_img - centered_img.min()
    centered_vis = (centered_vis / centered_vis.max() * 255).astype('uint8')

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(mean_img.reshape((32, 32, 3)).astype('uint8'))
    axes[0].set_title("Mean image")
    axes[0].axis('off')

    axes[1].imshow(some_img.reshape((32, 32, 3)).astype('uint8'))
    axes[1].set_title("First training image")
    axes[1].axis('off')

    axes[2].imshow(centered_vis.reshape((32, 32, 3)))
    axes[2].set_title("First image - mean")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

def _show_img(img):
    plt.figure(figsize=(4, 4))
    plt.imshow(img.reshape((32, 32, 3)).astype('uint8'))  # visualize the mean image
    plt.show()

if __name__ == "__main__":
    print(f"python: {sys.version}")
    print(f"numpy:  {np.__version__}")

    np.random.seed(1234)

    # load cifar
    cifar10_dir = r"datasets/cifar-10-batches-py"
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # TODO create splits

    # reshape each sample into flat vector 3072 (32*32*3)
    # and normalize (color has max value of 255 uint8/byte)
    X_train = X_train.reshape((X_train.shape[0], -1)) / 255.0
    X_test = X_test.reshape((X_test.shape[0], -1)) / 255.0
    print(f"xtrain shaped: {X_train.shape}")
    print(f"X_test shaped: {X_test.shape}")

    # calculate mean across train samples aka: compute the average image
    # by collapsing the image dimension and elaving one mean value for every pixel position across the entire set)
    # aka collapse rows
    mean_img = X_train.mean(axis=0)
    print(f"mean_img: {mean_img.shape}")

    #_show_mean_comparison(X_train[0], mean_img)

    # substraact mean from all train samples
    X_train = X_train - mean_img
    X_test = X_test - mean_img  # same mean that was created on test


    # create our W (num_classes, D)
    W = np.random.randn(10, mean_img.shape[0]) * 0.001
    print(f"W: {W.shape}")
    print(f"W: {W[:2]}")

    # TODO
    # we want a bias, and instead of create a separate store for the bias
    # we append a column to our W and make sure it results in just adding the bias term
    # so: ... + w_i * x_i
    # we start the weights for this appended column with 0 as we would an external bias
    # and always append a elment of value 1.0 to our input vectors so that they become D + 1
    # so that we get this  ... + w_i * 1 during the matmul

    # so W will be (num_classes, D+1)
    # and our xs will be (B, D+1)

    # implement softmax naive
    # implement softmax vectorized
    # implement loss calculation and gradients
    # add regularization to the loss

    batch_size: int = 50

    for i in range(100):
        x_batch = X_train[:batch_size]
        y_batch = y_train[:batch_size]
        #print(f"x_batch: {x_batch.shape}")
        logits = x_batch @ W.T
        #print(f"logits: {logits.shape}")

        logits_max = logits.max(axis=1, keepdims=True)
        #print(f"logits max axis 1: {logits_max.shape}")

        logits = logits - logits_max
        logits = np.exp(logits)
        #print(f"logits exp: {logits.shape}")

        exp_sum = logits.sum(axis=1, keepdims=True)
        #print(f"exp_sum: {exp_sum.shape}")
        probs = logits / exp_sum  # (B, C)
        #print(f"probs: {probs.shape}")

        #print(f"probs: {probs[0]}")

        #print(probs.sum(axis=1))
        y_probs = probs[np.arange(batch_size), y_batch]
        #print(f"y_probs: {y_probs.shape}")
        #print(f"y_probs: {y_probs}")


        if (i <= 10 or i % 10 == 0):
            loss = -np.log(y_probs).mean() # average loss agross the batch
            print(f"step: %d loss: %.4f" % (i, loss))



        #print(X_train[0][:5])

        dLdlogits = probs.copy()
        dLdlogits[np.arange(batch_size), y_batch] -= 1

        # print(f"y_probs: {dLdlogits.shape}") # B, C

        # we use average for loss, so need to average the grad here as well
        dLdlogits /= batch_size

        # transpose dlogits because matmul is row by vector
        # but the correct grads are currently in the rows of the other matrix,
        # need to transpose so they become vectors and then the dotproduct does what we want
        # (B, C).T -> (C, B) @ (B, D) -> (C, D)
        # logits are a sum of products
        # derivative of a sum is sum of derivatives
        # which is implied in the dotproduct
        # done here in one expression with matmul
        dLdW = dLdlogits.T @ x_batch


        # update weights with sgd
        lr: float = 0.01
        W += lr * -dLdW  # step into negative direction of the gradient

    # cross-validate

    expected_init_loss = -np.log(1.0 / 10)
    print(f"expected first init loss {expected_init_loss}")

    pass
