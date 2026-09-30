
import numpy as np

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

    print(f"dx: {dx.shape}")
    print(f"dW: {dW.shape}")

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

if __name__ == "__main__":
    np.random.seed(999)

    xinp = np.random.rand(12, 32, 32, 3)
    W = np.random.rand(10, 32*32*3)
    b = np.random.rand(10)
    out = affine_forward(xinp, W, b)

    print(f"xinp {xinp.shape}")
    print(f"out {out.shape}")

    upstream_dummy_w = np.random.rand(12, 10)
    dx, dW, db = affine_backward(upstream_dummy_w, xinp, W)

    upstream_dummy_a = np.random.rand(12, 10)
    activation = relu_forward(out)
    relu_backward(upstream_dummy_a, out)

    pass