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


def softmax(logits: np.ndarray) -> np.ndarray:
    # logits (B, C)
    logits_max = logits.max(axis=1, keepdims=True)

    logits = logits - logits_max
    logits = np.exp(logits)

    exp_sum = logits.sum(axis=1, keepdims=True)
    probs = logits / exp_sum

    return probs  # (B, C)


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

    NUM_CLASSES: int = 10
    # create our W (num_classes, D)
    W = np.random.randn(NUM_CLASSES, mean_img.shape[0]) * 0.001
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


    batch_size: int = 64
    train_size: int = X_train.shape[0]
    max_index: int = train_size - batch_size
    train_steps: int = train_size * 2
    base_lr: float = 0.01

    reg: float = 0.002
    for step in range(train_steps):
        start = np.random.randint(0, max_index)
        x_batch = X_train[start:start + batch_size]
        y_batch = y_train[start:start + batch_size]

        logits = x_batch @ W.T  # (B, D) @ (D, C) -> (B, C)
        probs = softmax(logits)  # (B, C)

        y_probs = probs[np.arange(batch_size), y_batch]

        lr: float = base_lr * (1 - (step / train_steps) + 0.00001)
        if (step <= 10 or step % 1000 == 0):
            loss = -np.log(y_probs).mean()  # average loss across the batch
            loss += reg * np.sum(W*W)  # L2 regularization
            print(f"step: %d loss: %.4f lr: %.4f" % (step, loss, lr))


        dLdlogits = probs.copy()
        dLdlogits[np.arange(batch_size), y_batch] -= 1

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

        # regularization gradient
        # loss is: reg * (w[r_1, c_1] * w[r_1, c_1] + w[r_2, c_2] * w[r_2, c_2] + ... etc)
        # we differentiate wrt to w[r,c]
        # through sum rule other terms become just + 0
        # so we just took at w[r,c]^2 and apply product rule (2 * w)
        dLdW += reg * 2 * W

        # update weights with sgd
        W += lr * -dLdW  # step into negative direction of the gradient

    logits = X_test @ W.T  # (B, D) @ (D, C) -> (B, C)
    probs = softmax(logits)  # (B, C)

    # passing logits works too because we interpret
    # increasing logit (or score) as "count" and increased probablity
    labels = np.argmax(probs, axis=1)
    print("predicted: ", labels[:10])
    print("true     : ", y_test[:10])

    accuracy = np.mean(labels == y_test)
    print("accuracy: %.2f perc" % (accuracy * 100))

    # cross-validate

    expected_init_loss = -np.log(1.0 / NUM_CLASSES)
    print(f"expected first init loss {expected_init_loss}")

    pass
