import importlib


def find_model_using_name(model_name):
    """Import the module "data/[dataset_name]_dataset.py".
    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)

    model = None
    target_model_name = model_name.replace("_", "") + "model"
    from models.base_model import BaseModel

    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        raise NotImplementedError(
            "In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase."
            % (model_filename, target_model_name)
        )

    return model


def get_option_setter(model_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options
