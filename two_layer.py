
import sys
import numpy as np
from data_utils import load_CIFAR10

def affine_forward(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:

    """
    :param x: input data of shape (B, d_1, d_2, d_3 ... d_k)
    :param W: weights of shape (D_OUT, D)
    :param b: biases of shape (D_OUT, )
    :return: out, linear transformation x @ W.T + b of shape (B, D_OUT)
    """
    x = x.reshape(x.shape[0], -1) # (B, D)
    assert x.shape[-1] == W.shape[-1], f"shape mismatch x {x.shape} W {W.shape}"
    # (B, D) @ (D, D_OUT) -> (B, D_OUT)
    out = x @ W.T + b
    return out

def affine_backward(dout: np.ndarray, x: np.ndarray, W: np.ndarray):
    """
    :param dout: upstream derivative of shape (B, D_OUT)
    :param x: input data of shape (B, d_1, d_2, ... d_k)
    :param W: weights of shape (D_OUT, D)
    :param b: bias of shope (D_OUT, )
    :return: dx, dW, db, derivatives with respect to x, W, and b

            dx of shape (B, d_1, d_2, ... d_k)
            dW of shape (D_OUT, D)
            db ias of shape (D_OUT, )
    """

    B = x.shape[0]
    # w * upstream grad and derivative of sum
    # (B, D_OUT) @ (D_OUT, D) -> (B, D)
    dx = (dout @ W).reshape(B, *x.shape[1:])

    # x * upstream grad and derivative of sum
    # (D_OUT, B) @ (B, D) -> (D_OUT, D)
    dW = dout.T @ x.reshape(B, -1)

    # 1 * upstream grad, b is added to every sample
    # and so also here derivative of sum is sum of derivatives
    # if we had B = 1, just that vector (dout[0]) would be the derivative wrt b
    # but we have a batch (B > 1) and have to sum the vectors (along row axis)
    db = dout.sum(axis=0)

    #print(f"dx: {dx.shape}")
    #print(f"dW: {dW.shape}")

    return dx, dW, db


def relu_forward(x: np.ndarray):
    # x if x > 0 else 0
    return np.maximum(0, x)

def relu_backward(dout: np.ndarray, x: np.ndarray):
    assert x.shape == dout.shape, f"shape mismatch dout: {dout.shape} x: {x.shape}"
    # x > 0 creates a matrix with True for elem positions where the x elem is > 0
    # and otherwise False, so a mask kind of
    # then we multiply this mask matrix with our upstream gradient (True/False get treated as 1s/0s)
    # to pass grads through where the input was > 0 and kill them otherwise
    return dout * (x > 0)


def softmax(x: np.ndarray) -> np.ndarray:
    max_x = x.max(axis=1, keepdims=True)

    x = x - max_x
    x_exp = np.exp(x)
    x_exp_sum = x_exp.sum(axis=1, keepdims=True)
    probs = x_exp / x_exp_sum
    return probs

def softmax_loss(x: np.ndarray, y: np.ndarray):
    """
    :param x: logits of shape (B, C)
    :param y: correct labels of shape (B, )
    :return: loss and gradient wrt x
    """
    probs = softmax(x)

    B = x.shape[0]
    dx = probs.copy()
    dx[np.arange(B), y] -= 1.0
    dx /= B

    loss = -np.log(probs[np.arange(B), y]).mean()

    return loss, dx


def _test_fwd_backward():
    xinp = np.random.rand(12, 32, 32, 3)
    y = np.array([2] * 12)
    W = np.random.rand(10, 32*32*3) * 0.0001
    b = np.random.rand(10) * 0.0
    out = affine_forward(xinp, W, b)

    print(f"xinp {xinp.shape}")
    print(f"out {out.shape}")

    upstream_dummy_w = np.random.rand(12, 10)
    dx, dW, db = affine_backward(upstream_dummy_w, xinp, W)

    upstream_dummy_a = np.random.rand(12, 10)
    activation = relu_forward(out)
    relu_backward(upstream_dummy_a, out)

    loss, dlogits = softmax_loss(activation, y)

    print(f"{loss=}")


class TwoLayerNet(object):
    def __init__(
        self,
        input_dim: int = 3*32*32,
        hidden_dim: int = 64,
        num_classes: int = 10,
    ):
        self.reg: float = 0.002
        self.input_dim: int = input_dim
        self.num_classes: int = num_classes
        self.hidden_dim: int = hidden_dim

        weight_scale: float = 0.0001
        self.W1: np.ndarray = np.random.randn(self.hidden_dim, self.input_dim) * weight_scale
        self.b1: np.ndarray = np.array([0.0] * self.hidden_dim)

        self.W2: np.ndarray = np.random.randn(self.num_classes, self.hidden_dim) * weight_scale
        self.b2: np.ndarray = np.array([0.0] * self.num_classes)


    def forward(self, x: np.ndarray, y: np.ndarray = None):
        # x (B, d_1, d_2 ... d_k)
        B = x.shape[0]

        x_hidden = affine_forward(x, self.W1, self.b1)  # (B, H)

        a = relu_forward(x_hidden)  # (B, H)

        logits = affine_forward(a, self.W2, self.b2)  # (B, C)

        if y is None:
            return logits, None

        loss, dlogits = softmax_loss(logits, y)

        da, dW2, db2 = affine_backward(dlogits, a, self.W2)

        dhidden = relu_backward(da, x_hidden)

        dx1, dW1, db1 = affine_backward(dhidden, x, self.W1)

        ## add regularization
        loss += self.reg * np.sum(self.W1**2) + self.reg * np.sum(self.W2**2)

        dW1 += self.reg * 2 * self.W1
        dW2 += self.reg * 2 * self.W2

        backward_result = {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
            "loss": loss
        }

        return logits, backward_result




if __name__ == "__main__":
    print(f"python: {sys.version}")
    print(f"numpy:  {np.__version__}")

    np.random.seed(999)

    num_training: int = 49000
    num_validation: int = 1000
    num_test: int = 1000

    cifar10_dir = r"datasets/cifar-10-batches-py"
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # normalize
    X_train /= 255.0
    X_val /= 255.0
    X_test /= 255.0

    # substract mean
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image


    #  model
    classifier: TwoLayerNet = TwoLayerNet()

    batch_size: int = 64
    train_size: int = X_train.shape[0]
    max_index: int = train_size - batch_size
    train_steps: int = train_size
    base_lr: float = 0.1

    reg: float = 0.002
    for step in range(train_steps):
        start = np.random.randint(0, max_index)
        x_batch = X_train[start:start + batch_size + 1]
        y_batch = y_train[start:start + batch_size + 1]

        logits, bw_res = classifier.forward(x_batch, y_batch)
        # print(f"logits: {logits.shape}")

        loss = bw_res.get("loss")
        if (step <= 10 or step % 100 == 0):
            print(f"step:{step} {loss=}")

        for key in ["W1", "b1", "W2", "b2"]:
            param = getattr(classifier, key)
            grad = bw_res[key]
            param -= base_lr * grad


    expected_init_loss = -np.log(1/classifier.num_classes)
    print(f"{expected_init_loss=}")

    test_logits, _ = classifier.forward(X_test)

    test_probs = softmax(test_logits)

    predicted_labels = np.argmax(test_probs, axis=1)
    test_accuracy = np.mean((predicted_labels == y_test))
    print("test accuracy: %.2f perc" % (test_accuracy * 100))


    pass