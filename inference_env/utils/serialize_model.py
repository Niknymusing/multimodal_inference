import torch

def load_model(model_class, checkpoint_path):
    """
    Load a PyTorch model from a checkpoint.

    Args:
        model_class: The class of the model (not an instance).
        checkpoint_path: Path to the model's state dict saved with torch.save().

    Returns:
        A PyTorch model instance loaded with trained weights.
    """
    model = model_class()  # Instantiate the model
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))  # Load the state dict
    return model

def save_torchscript_model(model, example_input, save_path, mode='script'):
    """
    Convert a PyTorch model to TorchScript and save it, using tracing or scripting.

    Args:
        model: The PyTorch model instance to serialize.
        example_input: A tensor corresponding to an example input to the model (needed for tracing).
        save_path: Path where the TorchScript model will be saved.
        mode: 'trace' for tracing (default) or 'script' for scripting.

    Returns:
        None
    """
    model.eval()  # Set the model to evaluation mode
    if mode == 'trace':
        # Use tracing for static computation graphs
        with torch.no_grad():  # Disable gradient calculation
            traced_script_module = torch.jit.trace(model, example_input)
    elif mode == 'script':
        # Use scripting for dynamic computation graphs
        traced_script_module = torch.jit.script(model)
    else:
        raise ValueError("Unsupported mode. Use 'trace' or 'script'.")

    traced_script_module.save(save_path)  # Save the TorchScript model
    print(f'Model has been saved as TorchScript at: {save_path}')

# This allows the module to be importable without running any code
if __name__ == "__main__":
    pass
