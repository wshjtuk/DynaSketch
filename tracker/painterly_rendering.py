import cv2
from IPython.display import display, SVG
from models.painter_params import Painter, PainterOptimizer
from models.loss import Loss
import sketch_utils as utils
import config
from tqdm.auto import tqdm, trange
from torchvision import models, transforms
from PIL import Image
import wandb
import torch.nn.functional as F
import torch.nn as nn
import torch
import PIL
import numpy as np
import traceback
import time
import sys
import os
from PIL import Image
import math
import argparse
import warnings
import time
import random

current_directory = os.path.dirname(os.path.abspath(__file__ ))
cotracker_path = os.path.join(current_directory, 'tracker')
sys.path.append(cotracker_path)
from tracker.demo import track



warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


def load_renderer(args, target_im=None, mask=None):
    renderer = Painter(num_strokes=args.num_paths, args=args,
                       num_segments=args.num_segments,
                       imsize=args.image_scale,
                       device=args.device,
                       target_im=target_im,
                       mask=mask)
    renderer = renderer.to(args.device)
    return renderer


#####################
# get control point #
#####################

def get_point(args):
    start_point = args.origin


def get_frame(video_path, frame_num):
    video = cv2.VideoCapture(video_path)
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = video.read()
    if ret:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    else:
        img = None
    video.release()
    return img


def get_target(args, fps):
    target = get_frame(args.target, fps)
    if target.mode == "RGBA":
        # Create a white rgba background
        new_image = Image.new("RGBA", target.size, "WHITE")
        # Paste the image on the background.
        new_image.paste(target, (0, 0), target)
        target = new_image
    target = target.convert("RGB")
    # masked_im, mask, mask_real = utils.get_mask_u2net(args, target)
    masked_im, mask, mask_real = utils.get_mask_inspyrenet(args, target, fps)
    if args.mask_object:
        target = masked_im
    if args.fix_scale:
        target = utils.fix_image_scale(target)

    transforms_ = []
    if target.size[0] != target.size[1]:
        transforms_.append(transforms.Resize(
            (args.image_scale, args.image_scale), interpolation=PIL.Image.BICUBIC))
    else:
        transforms_.append(transforms.Resize(
            args.image_scale, interpolation=PIL.Image.BICUBIC))
        transforms_.append(transforms.CenterCrop(args.image_scale))
    transforms_.append(transforms.ToTensor())
    data_transforms = transforms.Compose(transforms_)
    target_ = data_transforms(target).unsqueeze(0).to(args.device)
    return target_, mask, mask_real


def main(args, fps, origin_renderer, first_renderer, dict_pred, modify=0):
    loss_func = Loss(args)
    inputs, mask, _ = get_target(args, fps)
    utils.log_input(args.use_wandb, 0, inputs, args.output_dir)
    renderer = load_renderer(args, inputs, mask)
    if modify:
        renderer = origin_renderer
    optimizer = PainterOptimizer(args, renderer)
    counter = 0
    configs_to_save = {"loss_eval": []}
    best_loss, best_fc_loss = 100, 100
    best_iter, best_iter_fc = 0, 0
    min_delta = 1e-5
    terminate = False

    renderer.set_random_noise(0)
    if not modify:
        img = renderer.init_image(stage=0)
    optimizer.init_optimizers()
    # not using tdqm for jupyter demo
    if args.display:
        epoch_range = range(args.num_iter)
    else:
        epoch_range = tqdm(range(args.num_iter))
    if modify:
        epoch_range = tqdm(range(args.num_iter, 2*args.num_iter))
        counter = args.num_iter
    clip_loss = []
    for epoch in epoch_range:
        torch.cuda.empty_cache()
        if not args.display:
            epoch_range.refresh()
        renderer.set_random_noise(epoch)
        if args.lr_scheduler:
            optimizer.update_lr(counter)

        start = time.time()
        optimizer.zero_grad_()
        sketches = renderer.get_image().to(args.device)
        losses_dict = loss_func(sketches, inputs.detach(
        ), renderer.get_color_parameters(), renderer, origin_renderer, first_renderer, dict_pred, counter, optimizer)
        loss = sum(list(losses_dict.values()))
        loss.backward()
        optimizer.step_()
        shapes = renderer.shapes
        if epoch % args.save_interval == 0:
            utils.plot_batch(inputs, sketches, f"{args.output_dir}/jpg_logs", counter,
                             use_wandb=args.use_wandb, title=f"iter{epoch}.jpg")
            renderer.save_svg(
                f"{args.output_dir}/svg_logs", f"svg_iter{epoch}")

        if epoch % args.eval_interval == 0:
            with torch.no_grad():
                # losses_dict_eval = loss_func(sketches, inputs, renderer.get_color_parameters(
                # ), renderer.get_points_parans(), counter, optimizer, mode="eval")
                losses_dict_eval = loss_func(sketches, inputs, renderer.get_color_parameters(
                ), renderer, origin_renderer, first_renderer, dict_pred, counter, optimizer, mode="eval")
                loss_eval = sum(list(losses_dict_eval.values()))
                configs_to_save["loss_eval"].append(loss_eval.item())
                for k in losses_dict_eval.keys():
                    if k not in configs_to_save.keys():
                        configs_to_save[k] = []
                    configs_to_save[k].append(losses_dict_eval[k].item())
                if args.clip_fc_loss_weight:
                    if losses_dict_eval["fc"].item() < best_fc_loss:
                        best_fc_loss = losses_dict_eval["fc"].item(
                        ) / args.clip_fc_loss_weight
                        best_iter_fc = epoch
                # print(
                #     f"eval iter[{epoch}/{args.num_iter}] loss[{loss.item()}] time[{time.time() - start}]")
                clip_loss.append(sum([losses_dict_eval['clip_conv_loss'], losses_dict_eval['clip_conv_loss_layer2'], losses_dict_eval['clip_conv_loss_layer3'], losses_dict_eval['fc']]).item())
                cur_delta = loss_eval.item() - best_loss
                if abs(cur_delta) > min_delta:
                    if cur_delta < 0:
                        best_loss = loss_eval.item()
                        best_iter = epoch
                        terminate = False
                        utils.plot_batch(
                            inputs, sketches, args.output_dir, counter, use_wandb=args.use_wandb, title="best_iter.jpg")
                        renderer.save_svg(args.output_dir, "best_iter")
                        best_renderer = renderer

                if args.use_wandb:
                    wandb.run.summary["best_loss"] = best_loss
                    wandb.run.summary["best_loss_fc"] = best_fc_loss
                    wandb_dict = {"delta": cur_delta,
                                  "loss_eval": loss_eval.item()}
                    for k in losses_dict_eval.keys():
                        wandb_dict[k + "_eval"] = losses_dict_eval[k].item()
                    wandb.log(wandb_dict, step=counter)

                if abs(cur_delta) <= min_delta:
                    if terminate:
                        pass
                    terminate = True

        if counter == 0 and args.attention_init:
            utils.plot_atten(renderer.get_attn(), renderer.get_thresh(), inputs, renderer.get_inds(),
                             args.use_wandb, "{}/{}.jpg".format(
                                 args.output_dir, "attention_map"),
                             args.saliency_model, args.display_logs)

        if args.use_wandb:
            wandb_dict = {"loss": loss.item(), "lr": optimizer.get_lr()}
            for k in losses_dict.keys():
                wandb_dict[k] = losses_dict[k].item()
            wandb.log(wandb_dict, step=counter)
        counter += 1
    # for i in range(len(best_renderer.shapes)):
    #     best_renderer.shape_groups[i].stroke_color = torch.tensor([255, 255, 255, 1], dtype=torch.int32)
    #     img = best_renderer.get_image().to(args.device)
    #     utils.plot_batch(inputs, img, args.output_dir, counter, use_wandb=args.use_wandb, title=f"best_iter_{i}.jpg")
    #     args.track = 0
    #     losses_dict = loss_func(img, inputs.detach(
    #     ), best_renderer.get_color_parameters(), best_renderer, origin_renderer, counter, optimizer)
    #     loss_eval = sum(list(losses_dict_eval.values()))
    #     losses.append(loss_eval)
    #     best_renderer.shape_groups[i].stroke_color = torch.tensor([0., 0., 0., 1.])
    # print(losses)
    renderer.save_svg(args.output_dir, "final_svg")
    path_svg = os.path.join(args.output_dir, "best_iter.svg")
    utils.log_sketch_summary_final(
        path_svg, args.use_wandb, args.device, best_iter, best_loss, "best total")
    clip_loss_min = min(clip_loss)
    return configs_to_save, best_renderer, clip_loss_min


if __name__ == "__main__":
    origin_renderer = None
    args = config.parse_arguments()
    final_config = vars(args)
    video = cv2.VideoCapture(args.target)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    outputDir = args.output_dir
    num_iter = args.num_iter
    if args.use_wandb:
        time_str = time.strftime('%Y%m%d_%H%M', time.localtime(time.time()))
        proj_name = outputDir.split("/")[-3]
        args.wandb_project_name = f"{time_str}_{proj_name}_strokeNum={args.num_paths}"
        print(args.wandb_project_name)
    frame_cut = args.frame_cut
    frame_add = 2
    print(f"frame_count: {frame_count}")
    # if frame_count<=30:
    #     frame_cut = 8
    # elif frame_count<=50:
    #     frame_cut = 10
    # else:
    #     frame_cut = 15
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    # origin_fps = None
    # dict_pred = {}
    # for fps in range(0, frame_count, frame_cut):
    #     _, _, mask = get_target(args, fps)
    #     new_fps = fps
    #     if origin_fps != None:
    #         pred_tracks, pred_visibility = track(args.target, mask, 40, origin_fps, new_fps, args.use_gpu)
    #         dict_pred[fps] = (pred_tracks, pred_visibility)
    #     origin_fps = new_fps
    _, _, mask = get_target(args, 0)
    height, width, _ = mask.shape
    last_not_modified = [1 for i in range(args.num_paths)]
    dict_pred = {}
    mask_fps = {}
    best_iter_img = []
    first_renderer = None
    for fps in range(0, frame_count, frame_cut):
        inputs, mask_, mask = get_target(args, fps)
        mask_fps[fps] = mask
        new_renderer = load_renderer(args, inputs, mask_)
        print(f"current_frame: {fps}")
        ####### modify #######
        eps = 0.3
        args.modified = False
        dict_pred = {}
        dict_pred['0_0'] = None
        origin_fps = 0
        new_fps = fps
        if not origin_renderer == None:
            total_count = 20
            curves = first_renderer.get_curve(total_count)
            curve_num = 0
            origin_renderer.optimize_flag = [
                False for i in range(len(origin_renderer.shapes))]
            origin_fps = 0
            new_fps = fps
            for curve in curves:
                # origin_fps = last_not_modified[curve_num]
                if not f'{origin_fps}_{new_fps}' in dict_pred:
                    pred_tracks, pred_visibility, origin_tracks = track(args.target, mask_fps[origin_fps], 40, origin_fps, new_fps, args.use_gpu)
                    pred_tracks[:,0] = pred_tracks[:,0] / width * origin_renderer.canvas_width
                    pred_tracks[:,1] = pred_tracks[:,1] / height * origin_renderer.canvas_height
                    origin_tracks[:,0] = origin_tracks[:,0] / width * origin_renderer.canvas_width
                    origin_tracks[:,1] = origin_tracks[:,1] / height * origin_renderer.canvas_height
                    dict_pred[f'{origin_fps}_{new_fps}'] = (pred_tracks, pred_visibility, origin_tracks)
                true_count = 0
                next_point = []
                for point in curve:
                    distances = torch.norm(dict_pred[f'{origin_fps}_{new_fps}'][2] - point, dim=1)
                    closest_index = torch.argmin(distances).item()
                    # if dict_pred[f'{origin_fps}_{new_fps}'][1][closest_index] == True:
                    #     true_count += 1
                    if dict_pred[f'{origin_fps}_{new_fps}'][0][closest_index][0] <= width and dict_pred[f'{origin_fps}_{new_fps}'][0][closest_index][0] >=0 \
                        and dict_pred[f'{origin_fps}_{new_fps}'][0][closest_index][1] <= height and dict_pred[f'{origin_fps}_{new_fps}'][0][closest_index][1] >=0:
                        true_count += 1
                    next_point.append(dict_pred[f'{origin_fps}_{new_fps}'][0][closest_index])
                rate = true_count/total_count
                last_not_modified[curve_num] = last_not_modified[curve_num] * rate
                # if last_not_modified[curve_num] <= eps:
                if rate <= eps and args.modify:
                    last_not_modified[curve_num] = 1
                    args.modified = True
                    last_not_modified[curve_num] = new_fps
                    attention_max = 0
                    for point in next_point:
                        try:
                            attention = new_renderer.attention_map[int(point[0]), int(point[1])]
                        except:
                            attention = 0
                        if attention >= attention_max:
                            max_point = point
                    points_list = []
                    for i in range(4):
                        points_list.append((max_point[0] + 0.05 * origin_renderer.canvas_width * (random.random() - 0.5),
                                                                       max_point[1] + 0.05 * origin_renderer.canvas_height * (random.random() - 0.5)))
                    origin_renderer.shapes[curve_num].points = torch.tensor(points_list).to(origin_renderer.device)
                    # origin_renderer.shape_groups[curve_num].stroke_color = torch.tensor(
                    #     [255, 255, 255, 1], dtype=torch.int32)

                    origin_renderer.optimize_flag[curve_num] = True
                    origin_renderer.save_svg(args.output_dir, "best_iter")
                curve_num += 1
            # total_count = 20
            # point_list = dict_pred[fps][0]
            # curves = origin_renderer.get_curve(total_count)
            # curve_num = 0
            # origin_renderer.optimize_flag = [
            #     False for i in range(len(origin_renderer.shapes))]
            # for curve in curves:
            #     false_count = 0
            #     for point in curve:
            #         point[0] = point[0]/origin_renderer.canvas_width * width
            #         point[1] = point[1]/origin_renderer.canvas_height * height
            #         distances = torch.norm(point_list - point, dim=1)
            #         closest_index = torch.argmin(distances).item()
            #         if dict_pred[fps][1][closest_index] == False:
            #             false_count += 1
            #     if false_count/total_count >= eps:
            #         args.modified = True
            #         print(origin_renderer.attn_map)
            #         # origin_renderer.shape_groups[curve_num].stroke_color = torch.tensor(
            #         #     [255, 255, 255, 1], dtype=torch.int32)
            #         origin_renderer.optimize_flag[curve_num] = True
            #         origin_renderer.save_svg(args.output_dir, "best_iter")
            #     curve_num += 1
            if args.modified:
                print("starting modify...")
                args.num_iter = num_iter
                args.path_svg = f"{args.output_dir}/best_iter.svg"
                if args.use_wandb:
                    wandb.init(project=args.wandb_project_name, entity=args.wandb_user,
                            config=args, name=args.wandb_name.replace("frame", f"frame_{fps-frame_cut}") + "_modify", id=wandb.util.generate_id())
                args.track = 1
                configs_to_save, origin_renderer, _ = main(
                    args, fps, origin_renderer, first_renderer, dict_pred[f'{origin_fps}_{new_fps}'], modify=1)
                if args.use_wandb:
                    wandb.finish()
                args.modified = False
        ####### modify #######
        
        if origin_renderer == None:
            args.track = 0
            args.path_svg = "none"
            args.output_dir = args.output_dir.replace("frame", f"frame_{fps}")
            # for i in range(frame_add):
            #     frame = fps + (i + 1) * frame_cut
            #     inputs, mask = get_target(args, frame)
            #     renderer = load_renderer(args, inputs, mask, attention_map_list)
            #     attention_map = renderer.get_attn()
            #     attention_map_list.append(attention_map)
        else:
            args.track = 1
            args.path_svg = f"{args.output_dir}/best_iter.svg"
            args.output_dir = args.output_dir.replace(f"frame_{fps-frame_cut}", f"frame_{fps}")
            # if fps == frame_cut:
            #     args.path_svg = f"/data/wujk2022/2023/CLIPasso/output_sketches/frame_{fps-frame_cut}/frame_{fps-frame_cut}_{args.num_paths}strokes_seed0/best_iter.svg"
            args.num_iter = num_iter//4
            origin_renderer.grouping(10)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        jpg_logs_dir = f"{args.output_dir}/jpg_logs"
        svg_logs_dir = f"{args.output_dir}/svg_logs"
        if not os.path.exists(jpg_logs_dir):
            os.mkdir(jpg_logs_dir)
        if not os.path.exists(svg_logs_dir):
            os.mkdir(svg_logs_dir)
        if args.use_wandb:
            wandb.init(project=args.wandb_project_name, entity=args.wandb_user,
                config=args, name=args.wandb_name.replace("frame", f"frame_{fps}"), id=wandb.util.generate_id())
        if origin_renderer == None:
            min_loss = 1000
            first_output_dir = args.output_dir
            for i in range(3):
                seed = 1000*i
                random.seed(seed)
                np.random.seed(seed)
                os.environ['PYTHONHASHSEED'] = str(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                if i == 0:
                    last_seed = "seed0"
                else:
                    last_seed = f"seed{(i-1)*1000}"
                args.output_dir = args.output_dir.replace(last_seed, f"seed{i*1000}")
                if not os.path.exists(args.output_dir):
                    os.makedirs(args.output_dir)
                jpg_logs_dir = f"{args.output_dir}/jpg_logs"
                svg_logs_dir = f"{args.output_dir}/svg_logs"
                if not os.path.exists(jpg_logs_dir):
                    os.mkdir(jpg_logs_dir)
                if not os.path.exists(svg_logs_dir):
                    os.mkdir(svg_logs_dir)
                if args.use_wandb:
                    wandb.init(project=args.wandb_project_name, entity=args.wandb_user,
                        config=args, name=args.wandb_name.replace("frame", f"frame_{fps}").replace("seed0", f"seed{i*1000}"), id=wandb.util.generate_id())
                configs_to_save, origin_renderer_, min_clip_loss= main(args, fps, origin_renderer, first_renderer, dict_pred[f'{origin_fps}_{new_fps}'])
                if args.use_wandb:
                    wandb.finish()
                if min_clip_loss < min_loss:
                    min_loss = min_clip_loss
                    origin_renderer = origin_renderer_
                    print(f"seed{i}000 is the best...")
                    if not i==0:
                        os.rename(first_output_dir, args.output_dir+"temp")
                        os.rename(args.output_dir, first_output_dir)
                        os.rename(args.output_dir+"temp", args.output_dir)
            args.output_dir = args.output_dir.replace(f"seed{i*1000}", "seed0")
            best_img = f"{args.output_dir}/best_iter.jpg"
            best_iter_img.append(best_img)
            first_renderer = origin_renderer
        else:
            configs_to_save, origin_renderer, _= main(args, fps, origin_renderer, first_renderer, dict_pred[f'{origin_fps}_{new_fps}'])
            best_img = f"{args.output_dir}/best_iter.jpg"
            best_iter_img.append(best_img)
            output_video = f"{args.output_dir}/output_video.mp4"
            frame_rate = 2
            first_img_path = best_iter_img[0]
            first_img = cv2.imread(first_img_path)
            frame_size = (first_img.shape[1], first_img.shape[0])
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(output_video, fourcc, frame_rate, frame_size)
            for img_file in best_iter_img:
                img_path = os.path.join(img_file)
                img = cv2.imread(img_path)
                out.write(img)
            out.release()
            if args.use_wandb:
                # image_array = []
                # for img_file in best_iter_img:
                #     img_path = img_file
                #     img = cv2.imread(img_path)
                #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                #     image_array.append(img)
                # image_array = np.array(image_array)
                # image_array = image_array.transpose(0, 3, 1, 2)
                # wandb.log({"video": wandb.Video(image_array, fps=2)})
                wandb.finish()
        
    
            
            
    for k in configs_to_save.keys():
        final_config[k] = configs_to_save[k]
    np.save(f"{outputDir}/config.npy", final_config)
    
