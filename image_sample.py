"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import datetime
import json
from pathlib import Path

import torch.distributed as dist
from mpi4py import MPI

from qdatasets.kidney import KidneyDataset
from qdatasets.brain_growth import BrainGrowthDataset
from qdatasets.brain_tumor import BrainTumor1Dataset, BrainTumor2Dataset, BrainTumor3Dataset
from qdatasets.prostate import Prostate1Dataset, Prostate2Dataset

from improved_diffusion import dist_util, logger
from improved_diffusion.sampling_util import sampling_major_vote_func 
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from improved_diffusion.utils import set_random_seed
import warnings
warnings.filterwarnings('ignore')
import torch
torch.cuda.empty_cache()

def main():
    args = create_argparser().parse_args()
    # import pdb
    # pdb.set_trace()
    if args.dataname == "kidney":
        dataset = KidneyDataset
    elif args.dataname == "brain_growth":
        dataset = BrainGrowthDataset
    elif args.dataname == "brain_tumor":
        dataset = BrainTumor1Dataset
        args.condition_input_channel = 4
    elif args.dataname == "prostate_1":
        dataset = Prostate1Dataset
    elif args.dataname == "prostate_2":
        dataset = Prostate2Dataset
    
    unetmode = args.unetmode
    use_sparse = args.use_sparse
    data_dir = args.data_dir
    original_logs_path = Path(args.model_path).parent
    timeresp = args.timestep_respacing
    hnpairs = args.hnpairs
    use_bg = args.use_bg
    cal_bgnum = args.cal_bgnum

    args.__dict__.update(json.loads((original_logs_path / 'args.json').read_text()))
    logger.info(args.__dict__)
    dist_util.setup_dist()
    args.n_gen = 25
    args.data_dir = data_dir
    args.use_sparse = use_sparse
    print('+++++',args.use_sparse)
    args.unetmode = unetmode
    args.hnpairs = hnpairs
    args.use_bg = use_bg
    args.cal_bgnum = cal_bgnum
    number_of_generated_instances = args.n_gen
    print("args.generate samples:", args.n_gen)
    logs_path = original_logs_path / f"{Path(args.model_path).stem}_major_vote_{args.n_gen}_{args.fixstr}"
    logger.configure(dir=str(logs_path), log_suffix=f"val_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}")

    test_dataset = dataset(
        data_dir=args.data_dir,
        mode='val',
        image_size=args.image_size,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        soft_label_gt=True
    )
    

    logger.log("creating model and diffusion...")
    
    # acs = args_to_dict(args, model_and_diffusion_defaults().keys())
    args.timestep_respacing = timeresp 
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu"), strict=True
    )
    model.to(dist_util.dev())
    model.eval()
    # print(model)
    with open('model_structure.txt', 'w') as f:
        f.write(str(model))

    if args.__dict__.get("seed") is None:
        seed = 1234
    else:
        seed = int(args.__dict__.get("seed"))
    set_random_seed(seed, deterministic=True)
    logger.log("sampling major vote val")
    (logs_path / "major_vote").mkdir(exist_ok=True)
    step = int(Path(args.model_path).stem.split("_")[-1])
    sampling_major_vote_func(diffusion, model, str(logs_path / "major_vote"), test_dataset, logger, args.clip_denoised,
                                step=step, number_of_generated_instances=args.n_gen,diff_step=args.diffusion_steps)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        n_gen=25,
        data_dir="",
        fixstr="",
        mode="val",
        dataname="kidney",
        use_sparse=False,
        hnpairs="2,2",
        cal_bgnum=True,
        overlap_w=0.1,
        cut_padding=1,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
