"""
Helpful utils for handling pre-trained models
"""
from pathlib import Path

from jsonargparse import ArgumentParser

from torchdrum.data import OnsetFeatureDataModule
from torchdrum.tasks import TorchDrumParameterMapping


def load_model(config: str, ckpt: str, root: str = None, include_data: bool = False):
    """
    Load model from checkpoint
    """
    # Load the config file and instantiate the model
    config_parser = ArgumentParser()
    config_parser.add_subclass_arguments(
        TorchDrumParameterMapping, "model", fail_untyped=False
    )
    config_parser.add_argument("--trainer", type=dict, default={})
    config_parser.add_argument("--seed_everything", type=int)
    config_parser.add_argument("--ckpt_path", type=str)
    config_parser.add_argument("--optimizer", type=dict)
    config_parser.add_argument("--lr_scheduler", type=dict)

    if include_data:
        config_parser.add_subclass_arguments(OnsetFeatureDataModule, "data")
    else:
        config_parser.add_argument("--data", type=dict, default={})

    config = config_parser.parse_path(config)
    if root is not None:
        if hasattr(config.model.init_args, "preset"):
            config.model.init_args.preset = str(
                Path(root) / config.model.init_args.preset
            )
        if hasattr(config.data.init_args, "audio_path"):
            config.data.init_args.audio_path = str(
                Path(root) / config.data.init_args.audio_path
            )

    init = config_parser.instantiate_classes(config)

    # Load the checkpoint
    print(f"Loading checkpoint from {ckpt}...")
    model = TorchDrumParameterMapping.load_from_checkpoint(
        ckpt,
        model=init.model.model,
        loss_fn=init.model.loss_fn,
        feature=init.model.feature,
        synth=init.model.synth,
        preset=config.model.init_args.preset,
    )

    # Instantiate the datamodule if required
    if include_data:
        datamodule = init.data
        return model, datamodule

    return model, None


# def load_datamodule(config: str):
#     """
#     Load a datamodule from a config file
#     """
#     datamodule_parser = ArgumentParser()
#     datamodule_parser.add_subclass_arguments(KickDataModule, "datamodule")
#     if config is not None:
#         with open(config, "r") as f:
#             config = yaml.safe_load(f)
#             config = {"datamodule": config}
#             datamodule_args = datamodule_parser.parse_object(config)
#             datamodule = datamodule_parser.instantiate_classes(
#                 datamodule_args
#             ).datamodule

#     return datamodule
