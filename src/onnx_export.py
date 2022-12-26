import torch.onnx
from networks import ActorNetwork, CriticNetwork


def convert_onnx(model, name, input_size, n_actions=None):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.eval()

    if n_actions is not None:
        state = torch.randn(1, input_size, requires_grad=True).to(device)
        action = torch.randn(1, n_actions, requires_grad=True).to(device)
        dummy_input = (state, action)
    else:
        dummy_input = torch.randn(1, input_size, requires_grad=True).to(device)

    # Export the model
    torch.onnx.export(
        model,
        dummy_input,
        f"data/{name}.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print("Model exported to ONNX format")


if __name__ == '__main__':
    input_size = 33
    n_actions = 4

    alpha = 0.0001
    beta = 0.0001
    fc1_dims=128 
    fc2_dims=128
    input_dims = input_size

    actor_model = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                    n_actions=n_actions, name="actor_model",
                                    checkpoint_dir="")

    critic_model = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                    n_actions=n_actions, name="critic_model",
                                    checkpoint_dir="")

    convert_onnx(actor_model, "ActorNetwork", input_size)
    convert_onnx(critic_model, "CriticNetwork", input_size, n_actions)