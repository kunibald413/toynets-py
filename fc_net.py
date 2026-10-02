
import numpy as np

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
        weight_scale: float = 0.1
    ):
        assert hidden_dims is not None and len(hidden_dims) >= 1, "hidden dims must at least have one integer"
        self.num_hidden_layers: int = len(hidden_dims)
        self.num_layers: int = self.num_hidden_layers + 2  # in and out layers
        self.params: dict[str, np.ndarray] = {}


        layer_id: int = 0
        embed_dim: int = hidden_dims[0]
        W_in: np.ndarray = rng.standard_normal((embed_dim, input_dim)) * weight_scale
        b_in: np.ndarray = np.zeros(embed_dim)
        self.params[f"W{layer_id}"] = W_in
        self.params[f"b{layer_id}"] = b_in
        layer_id += 1

        for layer_idx in range(self.num_hidden_layers):
            inp_dim: int = embed_dim if layer_idx == 0 else hidden_dims[layer_idx - 1]
            out_dim: int = hidden_dims[layer_idx]
            W_h: np.ndarray = rng.standard_normal((out_dim, inp_dim)) * weight_scale
            b_h: np.ndarray = np.zeros(out_dim)
            self.params[f"W{layer_id}"] = W_h
            self.params[f"b{layer_id}"] = b_h
            layer_id += 1

        W_out: np.ndarray = rng.standard_normal((output_dim, hidden_dims[-1])) * weight_scale
        b_out: np.ndarray = np.zeros(output_dim)
        self.params[f"W{layer_id}"] = W_out
        self.params[f"b{layer_id}"] = b_out
        layer_id += 1

    def loss(self, X: np.ndarray, y: np.ndarray = None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        :param X: arr of input data of shape (B, d_1, d_2, ... d_k)
        :param y: arr of labels (B, ). y[i] gives the correct label for X[i]
        :return: tuple:
            logits (B, C),
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

            s: np.ndarray = x_h.std(axis=0)  # std per feature across batch
            print(f"layer: {layer_id} x {x.shape} -> x_h {x_h.shape} \n   std s.mean {s.mean():.4f} s.max: {s.max():.4f} s.min {s.min():.4f}")

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
            return logits, None

        grads: dict[str, np.ndarray] = {}  # store grads for the params we want to update

        # x -> x_h = W(x) -> a = act(x_h) -> logits = Wout(a) -> probs = softmax(logits)
        loss, dlogits = softmax_loss(logits, y)

        upstream_grad: np.ndarray = dlogits
        for layer_id in range(final_layer_id, -1, -1):
            print(f"backward layer: {layer_id}")
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

        print(f"mean loss: {loss}")

        return logits, grads



# fully connected net with arbitrary number of layers


if __name__ == "__main__":
    seed: int = 1337
    rng = np.random.default_rng(seed=seed)

    print(rng.standard_normal((2, 3)))

    net: FC_Net = FC_Net(rng, [32, 16, 8])

    for k, v in net.params.items():
        print(f"{k=} {v.shape}")

    dummy_x = rng.random((2, 32, 32, 3))
    dummy_y = np.array([4] * 2)
    logits, grads = net.loss(dummy_x, dummy_y)

    probs = softmax(logits)

    print(logits.shape)
    print(f"probs: {probs[:2]}")
    print(probs.sum(axis=1))
