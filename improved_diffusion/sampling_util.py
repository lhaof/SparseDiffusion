import math
import os
import pdb
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.utils as tvu
from PIL import Image
from kornia.enhance import denormalize
from matplotlib import pyplot as plt
from mpi4py import MPI
from sklearn.metrics import f1_score, jaccard_score
from torch.utils.data import DataLoader

from . import dist_util
from .metrics import FBound_metric, WCov_metric
from .qubiq_metrics import qubiq_metric
from .utils import set_random_seed_for_iterations
import torch_pruning as tp
# from thop import profile
import csv
import matplotlib.patches as patches
cityspallete = [
    0, 0, 0,
    128, 64, 128,
    244, 35, 232,
    70, 70, 70,
    102, 102, 156,
    190, 153, 153,
    153, 153, 153,
    250, 170, 30,
    220, 220, 0,
    107, 142, 35,
    152, 251, 152,
    0, 130, 180,
    220, 20, 60,
    255, 0, 0,
    0, 0, 142,
    0, 0, 70,
    0, 60, 100,
    0, 80, 100,
    0, 0, 230,
    119, 11, 32,
]


def calculate_metrics(x, gt):
    predict = x.detach().cpu().numpy().astype('uint8')
    target = gt.detach().cpu().numpy().astype('uint8')
    return f1_score(target.flatten(), predict.flatten()), jaccard_score(target.flatten(), predict.flatten()), \
           WCov_metric(predict, target), FBound_metric(predict, target)
def calculate_gather_metrics(num_active_blocks, block_size_h, block_size_w, channels):
    """Calculate GFLOPs and MACs for gather operations"""
    flops_per_element = 2  # Multiplication and addition
    macs_per_element = 3   # Memory access operations
    
    total_elements = num_active_blocks * block_size_h * block_size_w * channels
    total_flops = total_elements * flops_per_element
    total_macs = total_elements * macs_per_element
    
    return total_flops, total_macs

def calculate_scatter_metrics(num_active_blocks, block_size_h, block_size_w, channels):
    """Calculate GFLOPs and MACs for scatter operations"""
    flops_per_element = 2  # Average operation and accumulation
    macs_per_element = 2   # Read and write operations
    
    total_elements = num_active_blocks * block_size_h * block_size_w * channels
    total_flops = total_elements * flops_per_element
    total_macs = total_elements * macs_per_element
    
    return total_flops, total_macs




def sampling_major_vote_func(diffusion_model, ddp_model, output_folder, dataset, logger, clip_denoised, step, number_of_generated_instances=9, diff_step=100):
    ddp_model.eval()
    batch_size = 1
    logger.info(f"number of images to visualize {len(dataset)}")
    loader = DataLoader(dataset, batch_size=batch_size)
    name_unique_prefix = MPI.COMM_WORLD.Get_rank()
    f1_score_list = []
    soft_dice_score_list_9 = []
    soft_dice_score_list_5 = []
    soft_dice_score_list_5_msk = []
    miou_list = []
    fbound_list = []
    wcov_list = []
    time_list = []
    time_dict = {}

    logger.info("Starting warmup phase...")
    with torch.no_grad():
        # 获取第一个样本用于warmup
        warmup_data = next(iter(loader))
        gt_mask_warmup, condition_on_warmup, _ = warmup_data
        condition_on_image_warmup = condition_on_warmup["conditioned_image"]
        former_frame_for_feature_extraction_warmup = condition_on_image_warmup.to(dist_util.dev())
        
        # 设置warmup的model_kwargs
        model_kwargs_warmup = {
            "inference": True,
            "first_time_step": diff_step
        }
        model_kwargs_warmup["number_of_annotators"] = torch.range(1, dataset.number_of_annotators, dtype=torch.int).to(dist_util.dev())
        model_kwargs_warmup["conditioned_image"] = torch.cat([former_frame_for_feature_extraction_warmup] * dataset.number_of_annotators)
        first_dim_warmup = dataset.number_of_annotators
        
        # 执行10次warmup采样
        logger.info("Performing 10 warmup sampling iterations...")
        for i in range(2):
            _ = diffusion_model.p_sample_loop(
                ddp_model,
                (first_dim_warmup, gt_mask_warmup.shape[1], former_frame_for_feature_extraction_warmup.shape[2],
                 former_frame_for_feature_extraction_warmup.shape[3]),
                progress=False,  # 不显示进度以减少日志输出
                clip_denoised=clip_denoised,
                model_kwargs=model_kwargs_warmup
            )
        logger.info("Warmup phase completed.")
     
    with torch.no_grad():
        for index, (gt_mask, condition_on, name) in enumerate(loader):
            print("sampling:",name[0])
            set_random_seed_for_iterations(step + int(name[0].split("_")[0][-2:]))

            gt_mask = (gt_mask + 1.0) / 2.0
            condition_on_image = condition_on["conditioned_image"]
            former_frame_for_feature_extraction = condition_on_image.to(dist_util.dev())

            for i in range(gt_mask.shape[0]):
                cm = plt.get_cmap('bwr')
                gt_img = Image.fromarray((cm((gt_mask[i][0].detach().cpu().numpy()))[:, :, :3] * 255).astype(np.uint8))
                gt_img.save(
                    os.path.join(output_folder, f"{index}_{name_unique_prefix}_{name[0]}_gt_colormap.png"))

            annotator_gt = gt_mask * dataset.number_of_annotators
            for i in range(dataset.number_of_annotators):
                gt_img = Image.fromarray((annotator_gt >= (i + 1))[0][0].detach().cpu().numpy().astype('uint8'))
                gt_img.putpalette(cityspallete)
                gt_img.save(
                    os.path.join(output_folder, f"{index}_{name_unique_prefix}_{name[0]}_gt_annotator_{i + 1}_palette.png"))

            for i in range(condition_on_image.shape[0]):
                denorm_condition_on = denormalize(condition_on_image.clone(), mean=dataset.mean, std=dataset.std)

                tvu.save_image(
                    denorm_condition_on[i,] / denorm_condition_on.max(),
                    os.path.join(output_folder, f"{index}_{name_unique_prefix}_{name[0]}_condition_on_image.png")
                )
            
            model_kwargs = {
                "inference": True,
                "first_time_step":diff_step#diffusion_model.num_timesteps  #len(diffusion_model.betas)
            }
            model_kwargs["number_of_annotators"] = torch.range(1, dataset.number_of_annotators, dtype=torch.int).to(dist_util.dev())
            model_kwargs["conditioned_image"] = torch.cat([former_frame_for_feature_extraction] * dataset.number_of_annotators)
            first_dim = dataset.number_of_annotators

            list_of_x = []
            list_of_seg = []
            for i in range(number_of_generated_instances):#
                
                start_time1 = time.time()
                
                x = diffusion_model.p_sample_loop(
                    ddp_model,
                    (first_dim, gt_mask.shape[1], former_frame_for_feature_extraction.shape[2],
                     former_frame_for_feature_extraction.shape[3]),
                    progress=True,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs
                )
                end_time1 = time.time()
                if index not in time_dict:
                    time_dict[index] = []
                time_dict[index].append(end_time1 - start_time1)
                time_list.append(end_time1-start_time1)
                
                # has_seg_res = hasattr(ddp_model, 'seg_res')
                # if has_seg_res
                segmsk = ddp_model.seg_res
                list_of_seg.append(segmsk)
                # pdb.set_trace()
                list_of_x.append(x)
            # pdb.set_trace()
            x = torch.stack(list_of_x) # torch.Size([N, 3, 1, 256, 256])
            segmsk = torch.stack(list_of_seg) # torch.Size([N, 256, 256])
            # pdb.set_trace()
            segmsk = segmsk.mean(dim=0)

            x = (x + 1.0) / 2.0

            # np.save(os.path.join(output_folder, f"{index}_{name_unique_prefix}_{name[0]}_activeid"), np.array(list_of_active_id))
            # np.save(os.path.join(output_folder, f"{index}_{name_unique_prefix}_{name[0]}_model_result_array"), x.detach().cpu().numpy())
            # np.save(os.path.join(output_folder, f"{index}_{name_unique_prefix}_{name[0]}_gt_mask_array"), gt_mask.detach().cpu().numpy())
            # np.save(os.path.join(output_folder, f"{index}_{name_unique_prefix}_{name[0]}_segmsk_result_array"), segmsk.detach().cpu().numpy())

            if x.shape[-1] != gt_mask.shape[-1] or x.shape[-2] != gt_mask.shape[-2]:
                x = F.interpolate(x, gt_mask.shape[2:], mode='bilinear')

            x = torch.clamp(x, 0.0, 1.0)
            cm = plt.get_cmap('bwr')
            for gen_im in range(x.shape[0]):
                for annotator in range(x.shape[1]):
                    out_img = Image.fromarray(x[gen_im][annotator][0].round().detach().cpu().numpy().astype('uint8'))
                    out_img.putpalette(cityspallete)
                    out_img.save(
                        os.path.join(output_folder, f"{index}_{name_unique_prefix}_{name[0]}_model_output_annotator_{annotator+1}_vote_{gen_im}_binary.png"))

                    out_img = Image.fromarray((cm((x[gen_im][annotator][0].detach().cpu().numpy()))[:, :, :3] * 255).astype(np.uint8))
                    out_img.save(
                        os.path.join(output_folder, f"{index}_{name_unique_prefix}_{name[0]}_model_output_annotator_{annotator+1}_vote_{gen_im}_colormap.png"))

            out_img = Image.fromarray((cm((segmsk.detach().cpu().numpy()))[:, :, :3] * 255).astype(np.uint8))
            out_img.save(
                os.path.join(output_folder, f"{index}_{name_unique_prefix}_{name[0]}_segmsk_colormap.png"))
            
            # pdb.set_trace()
            if segmsk.dim()==2:
                segmsk = segmsk.unsqueeze(0)

            # major vote result
            x = x.mean(dim=0)

            for i in range(x.shape[0]):
                out_img = Image.fromarray(x[i][0].round().detach().cpu().numpy().astype('uint8'))
                out_img.putpalette(cityspallete)
                out_img.save(
                    os.path.join(output_folder, f"{index}_{name_unique_prefix}_{name[0]}_model_output_major_vote_annotator_{i+1}.png"))

                cm = plt.get_cmap('bwr')
                out_img = Image.fromarray((cm((x[i][0].detach().cpu().numpy()))[:, :, :3] * 255).astype(np.uint8))
                out_img.save(
                    os.path.join(output_folder, f"{index}_{name_unique_prefix}_{name[0]}_model_output_major_vote_annotator_{i+1}_colormap.png"))

            if not ddp_model.soft_label_training and not ddp_model.no_annotator_training:
                x = x.round()
                if ddp_model.consensus_training:
                    annotator_gt = gt_mask * dataset.number_of_annotators
                    for i in range(dataset.number_of_annotators):
                        out_im = x[i].int()
                        single_annotator = (annotator_gt >= i+1).int()

                        f1, miou, wcov, fbound = calculate_metrics(out_im[0], single_annotator[0][0])

                        logger.info(
                            f"{name_unique_prefix}_{index}_{name[0]} annotator {i+1} iou {miou}, f1_Score {f1}, WCov {wcov}, boundF {fbound}")

                x = torch.sum(x, dim=0) / dataset.number_of_annotators

                out_img = Image.fromarray((cm((x[0].detach().cpu().numpy()))[:, :, :3] * 255).astype(np.uint8))
                out_img.save(
                    os.path.join(output_folder, f"{index}_{name_unique_prefix}_{name[0]}_model_output_final_colormap.png"))

            # if ddp_model.soft_label_training:
            #     x = x[0]

            # if ddp_model.no_annotator_training:
            #     x = x.round()
            #     x = x.mean(dim=0)
            #     out_img = Image.fromarray((cm((x[0].detach().cpu().numpy()))[:, :, :3] * 255).astype(np.uint8))
            #     out_img.save(
            #         os.path.join(output_folder, f"{index}_{name_unique_prefix}_{name[0]}_model_output_final_colormap.png"))
            # pdb.set_trace()
            for i, (gt_eval, out_im, out_msk) in enumerate(zip(gt_mask, x, segmsk)):# torch.Size([1, 1, 256, 256])/torch.Size([1, 256, 256])

                b = out_im.unsqueeze(0).detach().cpu() # * denorm_condition_mask torch.Size([1, 256, 256])
                # b[b<0.5] = 0 #need to make this as hyper parameter
                avg_dice, dice_score_list = qubiq_metric(b.unsqueeze(0), gt_eval.unsqueeze(0).detach().cpu(), num_of_thresholds=9)
                soft_dice_score_list_9.append(avg_dice)

                logger.info(
                    f"{name_unique_prefix}_{index}_{name[0]} soft dice 9 {soft_dice_score_list_9[-1]}")

                logger.info(
                    f"{name_unique_prefix}_{index}_{name[0]} soft dice list {dice_score_list}")

                avg_dice, dice_score_list = qubiq_metric(b.unsqueeze(0), gt_eval.unsqueeze(0).detach().cpu(), num_of_thresholds=5)
                soft_dice_score_list_5.append(avg_dice)

                logger.info(
                    f"{name_unique_prefix}_{index}_{name[0]} soft dice 5 {soft_dice_score_list_5[-1]}")

                logger.info(
                    f"{name_unique_prefix}_{index}_{name[0]} soft dice list {dice_score_list}")

                # msk
                out_msk = out_msk.unsqueeze(0).detach().cpu()
                avg_dice, dice_score_list = qubiq_metric(out_msk.unsqueeze(0), gt_eval.unsqueeze(0).detach().cpu(), num_of_thresholds=5)
                soft_dice_score_list_5_msk.append(avg_dice)

                logger.info(
                    f"{name_unique_prefix}_{index}_{name[0]} mask soft dice 5 {soft_dice_score_list_5_msk[-1]}")

                logger.info(
                    f"{name_unique_prefix}_{index}_{name[0]} mask soft dice list {dice_score_list}")


                out_im = out_im.round().int()  # torch.Size([256, 256])
                gt_eval = gt_eval.round().int()  # torch.Size([1, 256, 256])

                f1, miou, wcov, fbound = calculate_metrics(out_im, gt_eval[0])
                f1_score_list.append(f1)
                miou_list.append(miou)
                wcov_list.append(wcov)
                fbound_list.append(fbound)

                logger.info(
                    f"{name_unique_prefix}_{index}_{name[0]} iou {miou_list[-1]}, f1_Score {f1_score_list[-1]}, WCov {wcov_list[-1]}, boundF {fbound_list[-1]}")
            # break
    logger.info(f"Time,esb 1 sample: {np.mean(time_list)}")
    logger.info(f"Time,esb 1 case: {np.sum(time_list)/dataset.max_len}")
    logger.info(f"waiting for rest of the processes for barrier 1")

    dist.barrier()
    all_time_dicts = [None] * dist.get_world_size()
    dist.all_gather_object(all_time_dicts, time_dict)
    if dist.get_rank() == 0:  # Only let the main process write the file
    # Combine all dictionaries
        combined_time_dict = {}
        for d in all_time_dicts:
            for case_idx, times in d.items():
                if case_idx not in combined_time_dict:
                    combined_time_dict[case_idx] = times
                else:
                    # In case multiple processes handled the same case
                    combined_time_dict[case_idx].extend(times)
        
        # Determine the maximum number of samples for any case
        max_samples = max(len(times) for times in combined_time_dict.values())
        # Create CSV file
        csv_path = os.path.join(output_folder, "sampling_times.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow([f"Sample_{i+1}" for i in range(max_samples)])
            # Write data for each case
            for case_idx in sorted(combined_time_dict.keys()):
                times = combined_time_dict[case_idx]
                # Pad with empty strings if this case has fewer samples than max_samples
                row = times + [''] * (max_samples - len(times))
                writer.writerow(row)
        logger.info(f"Sampling times saved to {csv_path}")

    logger.info(f"passing barrier 1")
    my_length = len(dataset)
    max_single_len = int(np.ceil(dataset.max_len / dist.get_world_size()))
    logger.info(f"{my_length} {max_single_len}")
    iou_tensor = torch.tensor(miou_list + [torch.tensor(-1)] * (max_single_len - my_length), device=dist_util.dev(), dtype=torch.float64)
    f1_tensor = torch.tensor(f1_score_list + [torch.tensor(-1)] * (max_single_len - my_length), device=dist_util.dev(), dtype=torch.float64)
    wcov_tensor = torch.tensor(wcov_list + [torch.tensor(-1)] * (max_single_len - my_length), device=dist_util.dev(), dtype=torch.float64)
    boundf_tensor = torch.tensor(fbound_list + [torch.tensor(-1)] * (max_single_len - my_length), device=dist_util.dev(), dtype=torch.float64)
    soft_dice_tensor_5 = torch.tensor(soft_dice_score_list_5 + [torch.tensor(-1)] * (max_single_len - my_length), device=dist_util.dev(), dtype=torch.float64)
    soft_dice_tensor_9 = torch.tensor(soft_dice_score_list_9 + [torch.tensor(-1)] * (max_single_len - my_length), device=dist_util.dev(), dtype=torch.float64)
    soft_dice_tensor_5_msk = torch.tensor(soft_dice_score_list_5_msk + [torch.tensor(-1)] * (max_single_len - my_length), device=dist_util.dev(), dtype=torch.float64)
    gathered_miou = [torch.ones_like(iou_tensor) * -1 for _ in range(dist.get_world_size())]
    gathered_f1 = [torch.ones_like(f1_tensor) * -1 for _ in range(dist.get_world_size())]
    gathered_wcov = [torch.ones_like(wcov_tensor) * -1 for _ in range(dist.get_world_size())]
    gathered_boundf = [torch.ones_like(boundf_tensor) * -1 for _ in range(dist.get_world_size())]
    gathered_soft_dice_5 = [torch.ones_like(soft_dice_tensor_5) * -1 for _ in range(dist.get_world_size())]
    gathered_soft_dice_9 = [torch.ones_like(soft_dice_tensor_9) * -1 for _ in range(dist.get_world_size())]
    gathered_soft_dice_5_msk = [torch.ones_like(soft_dice_tensor_5_msk) * -1 for _ in range(dist.get_world_size())]

    logger.info(f"Iou tensor{gathered_miou}")
    logger.info(f"Iou tensor{gathered_f1}")
    logger.info(f"Iou tensor{gathered_wcov}")
    logger.info(f"Iou tensor{gathered_boundf}")
    logger.info(f"Iou tensor{gathered_soft_dice_5}")
    logger.info(f"Iou tensor{gathered_soft_dice_9}")

    logger.info(f"Iou tensor{iou_tensor}")
    logger.info(f"Iou tensor{f1_tensor}")
    logger.info(f"Iou tensor{wcov_tensor}")
    logger.info(f"Iou tensor{boundf_tensor}")
    logger.info(f"Iou tensor{soft_dice_tensor_5}")
    logger.info(f"Iou tensor{soft_dice_tensor_9}")

    dist.all_gather(gathered_miou, iou_tensor)
    dist.all_gather(gathered_f1, f1_tensor)
    dist.all_gather(gathered_wcov, wcov_tensor)
    dist.all_gather(gathered_boundf, boundf_tensor)
    dist.all_gather(gathered_soft_dice_5, soft_dice_tensor_5)
    dist.all_gather(gathered_soft_dice_9, soft_dice_tensor_9)
    dist.all_gather(gathered_soft_dice_5_msk, soft_dice_tensor_5_msk)

    # if dist.get_rank() == 0:
    logger.info(f"dice5 tensor{gathered_soft_dice_5}")

    logger.info("measure total avg")
    logger.info(f"Iou tensor{iou_tensor}")
    logger.info(f"{gathered_miou}")

    gathered_miou = torch.cat(gathered_miou)
    logger.info(f"1")
    gathered_miou = gathered_miou[gathered_miou != -1]
    logger.info(f"mean iou {gathered_miou.mean()}")

    logger.info(f"2")
    gathered_f1 = torch.cat(gathered_f1)
    gathered_f1 = gathered_f1[gathered_f1 != -1]
    logger.info(f"mean f1 {gathered_f1.mean()}")
    gathered_wcov = torch.cat(gathered_wcov)
    gathered_wcov = gathered_wcov[gathered_wcov != -1]
    logger.info(f"mean WCov {gathered_wcov.mean()}")
    gathered_boundf = torch.cat(gathered_boundf)
    gathered_boundf = gathered_boundf[gathered_boundf != -1]
    logger.info(f"mean boundF {gathered_boundf.mean()}")

    gathered_soft_dice_9 = torch.cat(gathered_soft_dice_9)
    gathered_soft_dice_9 = gathered_soft_dice_9[gathered_soft_dice_9 != -1]
    logger.info(f"soft dice 9:")
    logger.info(f"{gathered_soft_dice_9}")
    logger.info(f"mean soft dice 9 {gathered_soft_dice_9.mean()}")

    gathered_soft_dice_5 = torch.cat(gathered_soft_dice_5)
    gathered_soft_dice_5 = gathered_soft_dice_5[gathered_soft_dice_5 != -1]
    logger.info(f"soft dice 5:")
    logger.info(f"{gathered_soft_dice_5}")
    logger.info(f"mean soft dice 5 {gathered_soft_dice_5.mean()}")

    gathered_soft_dice_5_msk = torch.cat(gathered_soft_dice_5_msk)
    gathered_soft_dice_5_msk = gathered_soft_dice_5_msk[gathered_soft_dice_5_msk != -1]
    logger.info(f"_msk soft dice 5:")
    logger.info(f"{gathered_soft_dice_5_msk}")
    logger.info(f"_msk mean soft dice 5 {gathered_soft_dice_5_msk.mean()}")

    logger.info(f"waiting for rest of the processes for barrier 2")
    dist.barrier()
    logger.info(f"passing barrier 2")
    return gathered_soft_dice_5.mean().item()
