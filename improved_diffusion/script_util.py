import argparse
import inspect

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
# from .unet import SuperResModel, UNetModel

# from .unet_sige_cmunext import SparseUNetModel  unet_sige_cmunext_patch32_overlap
# from .unet_sige_cmunext_pad0 import SparseUNetModel
# from .unet_sige_cmunext_patch32_overlap import SparseUNetModel
# from .unet_sige_cmunext import SparseUNetModel  # 连通

# from .unet_sige_cmunext_patch32_overlap_largesmall import SparseUNetModel 
# from .unet_cmunext_dconv_patch32_overlap20 import SparseUNetModel  # dconv

NUM_CLASSES = 1000


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    return dict(
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions="16,8",
        dropout=0.0,
        rrdb_blocks=8,
        deeper_net=False,
        learn_sigma=False,
        sigma_small=False,
        class_cond=False,
        class_name="train",
        expansion=False,
        diffusion_steps=100,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        use_checkpoint=False,
        use_scale_shift_norm=False,
        # seed=None,
        number_of_annotators=None,
        condition_input_channel=1,
        soft_label_training=False,
        consensus_training=False,
        no_annotator_training=False,
        annotators_training=True,
        use_sparse=False,
        unetmode="v0",
        hnpairs="2,2",
        use_bg=True,
        cal_bgnum=True,
        overlap_w=0.1,
        cut_padding=1
    )


def create_model_and_diffusion(
    image_size,
    class_cond,
    learn_sigma,
    sigma_small,
    num_channels,
    num_res_blocks,
    num_heads,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    rrdb_blocks,
    deeper_net,
    class_name,
    expansion,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    # seed,
    number_of_annotators,
    condition_input_channel,
    consensus_training,
    soft_label_training,
    annotators_training,
    no_annotator_training,
    use_sparse,
    unetmode,
    hnpairs,
    use_bg,
    cal_bgnum,
    overlap_w,
    cut_padding,
):
    print("here=======",use_sparse, unetmode)
    # _ = seed  # hack to prevent unused variable
    _ = expansion
    _ = class_name
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        rrdb_blocks=rrdb_blocks,
        deeper_net=deeper_net,
        number_of_annotators=number_of_annotators,
        condition_input_channel=condition_input_channel,
        consensus_training=consensus_training,
        soft_label_training=soft_label_training,
        annotators_training=annotators_training,
        no_annotator_training=no_annotator_training,
        use_sparse=use_sparse,
        unetmode=unetmode,
        hnpairs=hnpairs,
        use_bg=use_bg,
        cal_bgnum=cal_bgnum,
        overlap_w=overlap_w,
        cut_padding=cut_padding
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
    rrdb_blocks,
    deeper_net,
    number_of_annotators,
    condition_input_channel,
    consensus_training,
    soft_label_training,
    annotators_training,
    no_annotator_training,
    use_sparse,
    unetmode,
    hnpairs,
    use_bg,
    cal_bgnum,
    overlap_w,
    cut_padding
):
    if image_size == 480:
        channel_mult = (1, 1, 2, 2, 3, 3)
    elif image_size == 320:
        channel_mult = (1, 1, 2, 2, 3, 3)
    elif image_size == 256:
        if deeper_net:
            channel_mult = (1, 1, 1, 2, 2, 4, 4)
        else:
            channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 224:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 128:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))
    
    hnpair = [int(num) for num in hnpairs.split(",")]
    if unetmode=="acp":
        from .unet import UNetModel
    elif unetmode=="v3padcutabsmau":
        from .unet_v3_padcut_small_autoblock_au import UNetModel, SparseUNetModel
    
    print("------use model UNetModel---------:", unetmode, ", with sparse:", use_sparse)
    if use_sparse:
        return SparseUNetModel(
            hnpairs=hnpair,
            use_bg=use_bg,
            cal_bgnum=cal_bgnum,
            overlap_w=overlap_w,
            cut_padding=cut_padding,
            in_channels=1,
            model_channels=num_channels,
            out_channels=(1 if not learn_sigma else 2),
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            total_num_of_annotators=number_of_annotators,
            use_checkpoint=use_checkpoint,
            num_heads=num_heads,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            rrdb_blocks=rrdb_blocks,
            condition_input_channel=condition_input_channel,
            consensus_training=consensus_training,
            soft_label_training=soft_label_training,
            annotators_training=annotators_training,
            no_annotator_training=no_annotator_training,
        )
    else:
        return UNetModel(
            in_channels=1,
            model_channels=num_channels,
            out_channels=(1 if not learn_sigma else 2),
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            total_num_of_annotators=number_of_annotators,
            use_checkpoint=use_checkpoint,
            num_heads=num_heads,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            rrdb_blocks=rrdb_blocks,
            condition_input_channel=condition_input_channel,
            consensus_training=consensus_training,
            soft_label_training=soft_label_training,
            annotators_training=annotators_training,
            no_annotator_training=no_annotator_training,
        )



def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    
    if not timestep_respacing:
        timestep_respacing = [steps]
    
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
