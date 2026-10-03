
import sys
import numpy as np

from data_utils import load_CIFAR10
from two_layer import (
    affine_forward,
    affine_backward,
    relu_forward,
    relu_backward,
    softmax,
    softmax_loss,
)


class FC_Net(object):
    def __init__(
        self,
        rng: np. random._generator.Generator,
        hidden_dims: list[int],
        input_dim: int = 32*32*3,
        output_dim: int = 10,
    ):
        assert hidden_dims is not None and len(hidden_dims) >= 1, "hidden dims must at least have one integer"
        self.num_hidden_layers: int = len(hidden_dims)
        self.num_layers: int = self.num_hidden_layers + 1  # add out layers
        self.params: dict[str, np.ndarray] = {}

        layer_id: int = 0
        for layer_idx in range(self.num_hidden_layers):
            inp_dim: int = input_dim if layer_idx == 0 else hidden_dims[layer_idx - 1]
            out_dim: int = hidden_dims[layer_idx]
            W_h: np.ndarray = rng.standard_normal((out_dim, inp_dim)) * np.sqrt(2.0 / inp_dim)
            b_h: np.ndarray = np.zeros(out_dim)
            self.params[f"W{layer_id}"] = W_h
            self.params[f"b{layer_id}"] = b_h
            layer_id += 1

        W_out: np.ndarray = rng.standard_normal((output_dim, hidden_dims[-1])) * np.sqrt(2.0 / hidden_dims[-1])
        b_out: np.ndarray = np.zeros(output_dim)
        self.params[f"W{layer_id}"] = W_out
        self.params[f"b{layer_id}"] = b_out
        layer_id += 1

    def loss(self, X: np.ndarray, y: np.ndarray = None) -> tuple[np.ndarray, float, dict[str, np.ndarray]]:
        """
        :param X: arr of input data of shape (B, d_1, d_2, ... d_k)
        :param y: arr of labels (B, ). y[i] gives the correct label for X[i]
        :return: tuple:
            logits (B, C),
            loss: float
            grads: dictionary with gradients of the loss wrt to the param
        """

        if y is not None:
            assert y.shape[0] == X.shape[0], f"y and X shape mismatch y: {y.shape} X: {X.shape}"

        x = X

        logits: np.ndarray = None
        hidden_outputs: dict[int, np.ndarray] = {}
        activations: dict[int, np.ndarray] = {}


        final_layer_id: int = self.num_layers - 1
        for layer_id in range(self.num_layers):
            W: np.ndarray = self.params.get(f"W{layer_id}")
            b: np.ndarray = self.params.get(f"b{layer_id}")

            #print(f"START LAYER {layer_id} FORWARD W: {W.shape} b {b.shape} x {x.shape}")
            x_h = affine_forward(x, W, b)
            hidden_outputs[layer_id] = x_h.copy()

            #s: np.ndarray = x_h.std(axis=0)  # std per feature across batch
            #print(f"layer: {layer_id} x {x.shape} -> x_h {x_h.shape} \n   std s.mean {s.mean():.4f} s.max: {s.max():.4f} s.min {s.min():.4f}")

            if layer_id < final_layer_id:
                a = relu_forward(x_h)
                activations[layer_id] = a.copy()
                #print(f"    hidden x_h {x_h.shape} -> a {a.shape}")
                x = a
            else:
                logits = x_h
                #print(f"    softmax x_h {x_h.shape} -> a {logits.shape}")

        assert logits is not None, "logits None?"

        if y is None:
            return logits, 0, {}

        grads: dict[str, np.ndarray] = {}  # store grads for the params we want to update

        # x -> x_h = W(x) -> a = act(x_h) -> logits = Wout(a) -> probs = softmax(logits)
        loss, dlogits = softmax_loss(logits, y)

        upstream_grad: np.ndarray = dlogits
        for layer_id in range(final_layer_id, -1, -1):
            #print(f"backward layer: {layer_id}")
            if layer_id < final_layer_id:
                # how does activation (a) change wrt to input (x_h) + chain rule (upstream_grad)
                dhidden = relu_backward(upstream_grad, hidden_outputs[layer_id])
                upstream_grad = dhidden

            # how does the linear output (x_h, logits) change wrt to input (x, a) + chain rule (upstream_grad)
            input_this_layer: np.ndarray = X if layer_id == 0 else activations[layer_id - 1]
            W: np.ndarray = self.params.get(f"W{layer_id}")
            dinp, dW, db = affine_backward(upstream_grad, input_this_layer, W)
            grads[f"W{layer_id}"] = dW
            grads[f"b{layer_id}"] = db
            upstream_grad = dinp

        # print(f"mean loss: {loss}")

        return logits, loss, grads



# fully connected net with arbitrary number of layers

class Data():
    num_training: int = 49000
    num_validation: int = 1000
    num_test: int = 1000

    mean_image: np.ndarray
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

def load_data(extract_features: bool = False) -> Data:
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

    if extract_features:
        from features import get_img_features
        X_train = get_img_features(X_train)
        X_val = get_img_features(X_val)
        X_test = get_img_features(X_test)
    else:
        # normalize
        X_train /= 255.0
        X_val /= 255.0
        X_test /= 255.0

    # substract mean
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    out = Data()
    out.X_train = X_train
    out.y_train = y_train
    out.X_val = X_val
    out.y_val = y_val
    out.X_test = X_test
    out.y_test = y_test
    out.mean_image = mean_image
    return out

def assert_optim_config(param: np.ndarray, grad: np.ndarray, config: dict):
    assert param.shape == grad.shape, f"shape mismatch param vs grad {param.shape} != {grad.shape}"
    assert "lr" in config and isinstance(config["lr"], float), f"missing scalar learning rate 'lr' {config}"

def optim_sgd_vanilla(param: np.ndarray, grad: np.ndarray, config: dict):
    assert_optim_config(param, grad, config)
    param -= config["lr"] * grad

def optim_sgd_momentum(param_name: str, param: np.ndarray, grad: np.ndarray, config: dict):
    assert_optim_config(param, grad, config)
    v_key = f"velocity_{param_name}"
    assert v_key in config, f"missing velocity matrix for {param_name} ({v_key})"
    assert "friction" in config and isinstance(config["friction"], float), f"missing scalar 'friction' value"
    # v = m * v - lr * grad
    new_velocity = config["friction"] * config[v_key] - config["lr"] * grad
    config[v_key] = new_velocity
    param += new_velocity  # velocity is already expressed into negative direction of grad

def optim_rmsprop(param_name: str, param: np.ndarray, grad: np.ndarray, config: dict):
    assert_optim_config(param, grad, config)
    # cache = decay_rate * cache + (1 - decay_rate) * grad ** 2
    # param += -1 * lr * grad / (sqrt(cache) + eps)
    ada_cache_key: str = f"ada_cache_{param_name}"
    assert ada_cache_key in config, f"missing adaptive cache matrix for {param_name} ({ada_cache_key})"
    decay_key: str = "ada_decay_rate"
    assert decay_key in config and isinstance(config[decay_key], float), f"missing scalar '{decay_key}' value"
    eps: float = 1e-8
    dr: float = config[decay_key]
    config[ada_cache_key] = dr * config[ada_cache_key] + ((1 - dr) * grad ** 2)
    param -= config["lr"] * grad / np.sqrt(config[ada_cache_key] + eps)

def optim_adam():
    pass

OPTIM_TYPE_SGD_VANILLA: str = "sgd_vanilla"
OPTIM_TYPE_SGD_MOMENTUM: str = "sgd_momentum"
OPTIM_TYPE_RMSPROP: str = "rmsprop"
def optim_init_config(
        params: dict[str, np.ndarray],
        base_lr: float = 5e-3,
        optim_type: str = OPTIM_TYPE_SGD_VANILLA) -> dict:
    optim_config = {
        "lr": base_lr,
        "base_lr": base_lr,
    }

    if optim_type == OPTIM_TYPE_SGD_MOMENTUM:
        optim_config["friction"] = 0.9
        for k, v in params.items():
            optim_config[f"velocity_{k}"] = np.zeros_like(v)

    elif optim_type == OPTIM_TYPE_RMSPROP:
        optim_config["ada_decay_rate"] = 0.9
        for k, v in params.items():
            optim_config[f"ada_cache_{k}"] = np.zeros_like(v)
        pass
    elif optim_type == OPTIM_TYPE_SGD_VANILLA:
        pass
    else:
        raise ValueError(f"unknown optimizer type: {optim_type}")

    optim_config['optim_type'] = optim_type

    return optim_config

def optim_update(params: dict[str, np.ndarray], grad: dict[str, np.ndarray], optim_config: dict):
    optim_type: str = optim_config.get("optim_type")
    for k, v in params.items():
        if optim_type == OPTIM_TYPE_RMSPROP:
            optim_rmsprop(k, v, grad[k], optim_config)
        elif optim_type == OPTIM_TYPE_SGD_MOMENTUM:
            optim_sgd_momentum(k, v, grad[k], optim_config)
        elif optim_type == OPTIM_TYPE_SGD_VANILLA:
            optim_sgd_vanilla(v, grad[k], optim_config)
        else:
            raise ValueError(f"Can't do optimzer update. Unknown optimizer: {optim_type}")

if __name__ == "__main__":
    print(f"python: {sys.version}")
    print(f"numpy:  {np.__version__}")

    seed: int = 1337
    rng = np.random.default_rng(seed=seed)

    print(rng.standard_normal((2, 3)))

    num_classes: int = 10
    input_dim: int = 32 * 32 * 3  # actually data dependant
    net: FC_Net = FC_Net(rng, [100, 100, 100, 100, 100], input_dim, output_dim=num_classes)

    for k, v in net.params.items():
        print(f"{k=} {v.shape}")

    B: int = 10
    dummy_x = rng.random((B, 32, 32, 3))
    dummy_y = np.array([4] * B)
    logits, loss, grads = net.loss(dummy_x, dummy_y)
    for k, v in net.params.items():
        assert k in grads, f"param {k} has no grads?"

    probs = softmax(logits)

    print(logits.shape)
    print(f"probs: {probs[:2]}")
    print(probs.sum(axis=1))
    print(f"expected init loss: ", -np.log(1/num_classes))
    print(f"loss: {loss}")

    for k, v in grads.items():
        examlpe_grads = v[:5] if len(v.shape) == 1 else v[0][:5]
        print(f"grad for {k} {v.shape} example: {examlpe_grads}...")

    print("loading data...")
    data: Data = load_data()

    # test accuracy: 51.50 perc
    overfit_one_batch: bool = False
    batch_size: int = 100
    train_size: int = data.X_train.shape[0]
    max_index: int = train_size - batch_size
    train_steps: int = 1500 if overfit_one_batch else train_size
    base_lr: float = 5e-3

    optim_type: str = OPTIM_TYPE_RMSPROP
    decay_lr: bool = False

    reg: float = 0.002

    def run_test():
        test_logits, loss, _ = net.loss(data.X_test, data.y_test)
        test_probs = softmax(test_logits)
        predicted_labels = np.argmax(test_probs, axis=1)
        test_accuracy = np.mean((predicted_labels == data.y_test))
        print("test accuracy: %.2f perc" % (test_accuracy * 100))


    optim_config = optim_init_config(net.params, base_lr, optim_type)
    print(f"using optimizer: {optim_type}")

    for step in range(train_steps):
        start = 0 if overfit_one_batch else rng.integers(0, max_index, endpoint=True)  # chance to miss good part of dataset
        x_batch = data.X_train[start:start + batch_size]
        y_batch = data.y_train[start:start + batch_size]

        logits, loss, grad = net.loss(x_batch, y_batch)

        lr = optim_config.get("lr")
        if (step <= 10 or step % 1000 == 0):
            print(f"loss: {loss:.4f} step: {step} lr: {lr:.4f}")

        optim_update(net.params, grad, optim_config)

        if decay_lr: optim_config["lr"] = optim_config["base_lr"] * (1 - step / train_steps)


    run_test()
