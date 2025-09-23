"""
Train a diffusion model on images.
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import blobfile as bf
import git
from mpi4py import MPI

from qdatasets.kidney import KidneyDataset
from qdatasets.brain_growth import BrainGrowthDataset
from qdatasets.brain_tumor import BrainTumor1Dataset, BrainTumor2Dataset, BrainTumor3Dataset
from qdatasets.prostate import Prostate1Dataset, Prostate2Dataset

from improved_diffusion import dist_util, logger
from qdatasets.multi_annotators_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop
from improved_diffusion.utils import set_random_seed, set_random_seed_for_iterations
import warnings
warnings.filterwarnings('ignore')
from tensorboardX import SummaryWriter

def main():
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    args = create_argparser().parse_args()
    args.use_fp16 = False
    args.clip_denoised = True
    args.num_channels = 128
    args.image_size = 256
    args.num_res_blocks = 3
    args.learn_sigma = False
    args.deeper_net = True
    args.condition_input_channel = 1

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


    exp_name = f"{args.dataname}_{args.res_block_type}_{args.lr}_{args.batch_size}_{args.diffusion_steps}_{args.weight_decay}_{args.n_gen}_rank{MPI.COMM_WORLD.Get_rank()}"
    # logs_root = Path(__file__).absolute().parent.parent / "logs"  # PosixPath('/home/jiyang/logs')
    log_path = f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}_{exp_name}"  # logs_root / PosixPath('/home/jiyang/logs/2024-04-19-15-24-00-078839_kidney_learn_sigma_8_2e-05_4_100_0.0_9_0')
    log_path = os.path.join(args.out_dir, log_path)
    os.environ["OPENAI_LOGDIR"] = str(log_path)
    set_random_seed(MPI.COMM_WORLD.Get_rank(), deterministic=True)
    set_random_seed_for_iterations(MPI.COMM_WORLD.Get_rank())
    dist_util.setup_dist()
    logger.configure(dir=str(log_path))

    writer = SummaryWriter(log_dir=os.path.join(log_path,"tensorboard"))
    resume_checkpoint_path = args.resume_checkpoint
    if args.resume_checkpoint:
        resumed_checkpoint_arg = bf.dirname(args.resume_checkpoint)
        args.__dict__.update(json.loads((Path(resumed_checkpoint_arg) / 'args.json').read_text()))
        args.resume_checkpoint = resume_checkpoint_path

    # args.condition_input_channel = 1
    if args.soft_label_training: # F
        args.number_of_annotators = None
    else:
        args.number_of_annotators = dataset.get_number_of_annotators()

    args.annotators_training = not args.consensus_training and not args.soft_label_training and not args.no_annotator_training

    logger.info(args.__dict__)

    (Path(log_path) / 'args.json').write_text(json.dumps(args.__dict__, indent=4))
    logger.info(f"log folder path: {Path(log_path).resolve()}")

    # repo = git.Repo(search_parent_directories=True)
    # sha = repo.head.object.hexsha

    # logger.log(f"git commit hash {sha}")

    logger.log("creating data loader...")
    # import pdb
    # pdb.set_trace()
    data = load_data(
        data_dir=args.data_dir,
        dataset_class=dataset,
        batch_size=args.batch_size,
        image_size=args.image_size,
        erosion=args.erosion,
        soft_label_training=args.soft_label_training,
        consensus_training=args.consensus_training,
        no_annotator_training=args.no_annotator_training
    )
    test_dataset = dataset(
        data_dir=args.data_dir,
        mode='val',
        image_size=args.image_size,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        soft_label_gt=True
    )

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log(f"gpu {MPI.COMM_WORLD.Get_rank()} / {MPI.COMM_WORLD.Get_size()} validation images {len(test_dataset)}")

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        clip_denoised=args.clip_denoised,
        logger=logger,
        image_size=args.image_size,
        val_dataset=test_dataset,
        run_without_test=args.run_without_test,
        args=args,
        writer=writer
        # dist_util=dist_util,
    ).run_loop(max_iter=100001, start_print_iter=args.start_print_iter, number_of_generated_instances=args.n_gen)


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=0.00002,
        weight_decay=0.0,
        lr_anneal_steps=0,
        clip_denoised=False,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        save_interval=200,
        start_print_iter=600,
        log_interval=200,
        run_without_test=False,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        n_gen=9,
        erosion=False,
        soft_label_training=False,
        consensus_training=False,
        no_annotator_training=False,
        annotators_training=False,
        out_dir="",
        res_block_type="Res",  # DResv1
        dataname="kidney",
        unetmode="v3padcutabsmau",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
