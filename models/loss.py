import collections
import CLIP_.clip as clip
import torch
import torch.nn as nn
from torchvision import models, transforms

from shapesimilarity import shape_similarity
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import directed_hausdorff

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# svg
import xml.etree.ElementTree as ET

class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.args = args
        self.percep_loss = args.percep_loss

        self.train_with_clip = args.train_with_clip
        self.clip_weight = args.clip_weight
        self.start_clip = args.start_clip

        self.clip_conv_loss = args.clip_conv_loss
        self.clip_fc_loss_weight = args.clip_fc_loss_weight
        self.clip_text_guide = args.clip_text_guide
        self.control_point_loss = args.control_point_loss
        self.control_point_weight = args.control_point_weight
        self.bezier_loss = args.bezier_loss
        self.bezier_weight = args.bezier_weight
        self.track_loss = args.track_loss
        self.track_weight = args.track_weight
        self.losses_to_apply = self.get_losses_to_apply()

        self.loss_mapper = \
            {
                "clip": CLIPLoss(args),
                "clip_conv_loss": CLIPConvLoss(args),
                "control_point": ControlPointLoss(args),
                "bezier_loss": BezierLoss(args),
                "track_loss": TrackLoss(args)
            }

    def get_losses_to_apply(self):
        losses_to_apply = []
        if self.percep_loss != "none":
            losses_to_apply.append(self.percep_loss)
        if self.train_with_clip and self.start_clip == 0: # no
            losses_to_apply.append("clip")
        if self.clip_conv_loss: # yes
            losses_to_apply.append("clip_conv_loss")
        if self.clip_text_guide: # no
            losses_to_apply.append("clip_text")
        if self.control_point_loss and self.args.track and not self.args.modified:
            losses_to_apply.append("control_point")
        if self.bezier_loss and self.args.track and not self.args.modified:
            losses_to_apply.append("bezier_loss")
        if self.track_loss and self.args.track and not self.args.modified:
            losses_to_apply.append("track_loss")
        return losses_to_apply
    

    def update_losses_to_apply(self, epoch):
        if "clip" not in self.losses_to_apply:
            if self.train_with_clip:
                if epoch > self.start_clip:
                    self.losses_to_apply.append("clip")

    def forward(self, sketches, targets, color_parameters, renderer, origin_renderer, first_renderer, dict_pred, epoch, points_optim=None, mode="train"):
        loss = 0
        self.update_losses_to_apply(epoch)

        losses_dict = dict.fromkeys(
            self.losses_to_apply, torch.tensor([0.0]).to(self.args.device))
        loss_coeffs = dict.fromkeys(self.losses_to_apply, 1.0)
        loss_coeffs["clip"] = self.clip_weight
        loss_coeffs["clip_text"] = self.clip_text_guide
        if self.args.target:
            loss_coeffs["control_point"] = self.control_point_weight
            loss_coeffs["bezier_loss"] = self.bezier_weight
            loss_coeffs["track_loss"] = self.track_weight
        else:
            loss_coeffs["control_point"] = 0
            loss_coeffs["bezier_loss"] = 0
            loss_coeffs["track_loss"] = 0
        
        

        for loss_name in self.losses_to_apply:
            if loss_name in ["clip_conv_loss"]:
                conv_loss = self.loss_mapper[loss_name](
                    sketches, targets, mode)
                for layer in conv_loss.keys():
                    losses_dict[layer] = conv_loss[layer]
            elif loss_name == "l2":
                losses_dict[loss_name] = self.loss_mapper[loss_name](
                    sketches, targets).mean()
            elif loss_name == "control_point":
                losses_dict[loss_name] = self.loss_mapper[loss_name](
                    sketches, targets, epoch, renderer, origin_renderer, mode)
            elif loss_name == "bezier_loss":
                losses_dict[loss_name] = self.loss_mapper[loss_name](
                    sketches, targets, epoch, renderer, origin_renderer, mode)
            elif loss_name == "track_loss":
                losses_dict[loss_name] = self.loss_mapper[loss_name](
                    sketches, targets, epoch, renderer, origin_renderer, first_renderer, dict_pred, mode)
            else:
                losses_dict[loss_name] = self.loss_mapper[loss_name](
                    sketches, targets, mode).mean()
            # loss = loss + self.loss_mapper[loss_name](sketches, targets).mean() * loss_coeffs[loss_name]
        
        
        for key in self.losses_to_apply:
            # loss = loss + losses_dict[key] * loss_coeffs[key]
            losses_dict[key] = losses_dict[key] * loss_coeffs[key]
        # print(losses_dict)
        return losses_dict


class CLIPLoss(torch.nn.Module):
    def __init__(self, args):
        super(CLIPLoss, self).__init__()

        self.args = args
        self.model, clip_preprocess = clip.load(
            'ViT-B/32', args.device, jit=False)
        self.model.eval()
        self.preprocess = transforms.Compose(
            [clip_preprocess.transforms[-1]])  # clip normalisation
        self.device = args.device
        self.NUM_AUGS = args.num_aug_clip
        augemntations = []
        if "affine" in args.augemntations:
            augemntations.append(transforms.RandomPerspective(
                fill=0, p=1.0, distortion_scale=0.5))
            augemntations.append(transforms.RandomResizedCrop(
                224, scale=(0.8, 0.8), ratio=(1.0, 1.0)))
        augemntations.append(
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)))
        self.augment_trans = transforms.Compose(augemntations)

        self.calc_target = True
        self.include_target_in_aug = args.include_target_in_aug
        self.counter = 0
        self.augment_both = args.augment_both

    def forward(self, sketches, targets, mode="train"):
        if self.calc_target:
            targets_ = self.preprocess(targets).to(self.device)
            self.targets_features = self.model.encode_image(targets_).detach()
            self.calc_target = False

        if mode == "eval":
            # for regular clip distance, no augmentations
            with torch.no_grad():
                sketches = self.preprocess(sketches).to(self.device)
                sketches_features = self.model.encode_image(sketches)
                return 1. - torch.cosine_similarity(sketches_features, self.targets_features)

        loss_clip = 0
        sketch_augs = []
        img_augs = []
        for n in range(self.NUM_AUGS):
            augmented_pair = self.augment_trans(torch.cat([sketches, targets]))
            sketch_augs.append(augmented_pair[0].unsqueeze(0))

        sketch_batch = torch.cat(sketch_augs)
        # sketch_utils.plot_batch(img_batch, sketch_batch, self.args, self.counter, use_wandb=False, title="fc_aug{}_iter{}_{}.jpg".format(1, self.counter, mode))
        # if self.counter % 100 == 0:
        # sketch_utils.plot_batch(img_batch, sketch_batch, self.args, self.counter, use_wandb=False, title="aug{}_iter{}_{}.jpg".format(1, self.counter, mode))

        sketch_features = self.model.encode_image(sketch_batch)

        for n in range(self.NUM_AUGS):
            loss_clip += (1. - torch.cosine_similarity(
                sketch_features[n:n+1], self.targets_features, dim=1))
        self.counter += 1
        return loss_clip
        # return 1. - torch.cosine_similarity(sketches_features, self.targets_features)


class LPIPS(torch.nn.Module):
    def __init__(self, pretrained=True, normalize=True, pre_relu=True, device=None):
        """
        Args:
            pre_relu(bool): if True, selects features **before** reLU activations
        """
        super(LPIPS, self).__init__()
        # VGG using perceptually-learned weights (LPIPS metric)
        self.normalize = normalize
        self.pretrained = pretrained
        augemntations = []
        augemntations.append(transforms.RandomPerspective(
            fill=0, p=1.0, distortion_scale=0.5))
        augemntations.append(transforms.RandomResizedCrop(
            224, scale=(0.8, 0.8), ratio=(1.0, 1.0)))
        self.augment_trans = transforms.Compose(augemntations)
        self.feature_extractor = LPIPS._FeatureExtractor(
            pretrained, pre_relu).to(device)

    def _l2_normalize_features(self, x, eps=1e-10):
        nrm = torch.sqrt(torch.sum(x * x, dim=1, keepdim=True))
        return x / (nrm + eps)

    def forward(self, pred, target, mode="train"):
        """Compare VGG features of two inputs."""

        # Get VGG features

        sketch_augs, img_augs = [pred], [target]
        if mode == "train":
            for n in range(4):
                augmented_pair = self.augment_trans(torch.cat([pred, target]))
                sketch_augs.append(augmented_pair[0].unsqueeze(0))
                img_augs.append(augmented_pair[1].unsqueeze(0))

        xs = torch.cat(sketch_augs, dim=0)
        ys = torch.cat(img_augs, dim=0)

        pred = self.feature_extractor(xs)
        target = self.feature_extractor(ys)

        # L2 normalize features
        if self.normalize:
            pred = [self._l2_normalize_features(f) for f in pred]
            target = [self._l2_normalize_features(f) for f in target]

        # TODO(mgharbi) Apply Richard's linear weights?

        if self.normalize:
            diffs = [torch.sum((p - t) ** 2, 1)
                     for (p, t) in zip(pred, target)]
        else:
            # mean instead of sum to avoid super high range
            diffs = [torch.mean((p - t) ** 2, 1)
                     for (p, t) in zip(pred, target)]

        # Spatial average
        diffs = [diff.mean([1, 2]) for diff in diffs]

        return sum(diffs)

    class _FeatureExtractor(torch.nn.Module):
        def __init__(self, pretrained, pre_relu):
            super(LPIPS._FeatureExtractor, self).__init__()
            vgg_pretrained = models.vgg16(pretrained=pretrained).features

            self.breakpoints = [0, 4, 9, 16, 23, 30]
            if pre_relu:
                for i, _ in enumerate(self.breakpoints[1:]):
                    self.breakpoints[i + 1] -= 1

            # Split at the maxpools
            for i, b in enumerate(self.breakpoints[:-1]):
                ops = torch.nn.Sequential()
                for idx in range(b, self.breakpoints[i + 1]):
                    op = vgg_pretrained[idx]
                    ops.add_module(str(idx), op)
                # print(ops)
                self.add_module("group{}".format(i), ops)

            # No gradients
            for p in self.parameters():
                p.requires_grad = False

            # Torchvision's normalization: <https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101>
            self.register_buffer("shift", torch.Tensor(
                [0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("scale", torch.Tensor(
                [0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def forward(self, x):
            feats = []
            x = (x - self.shift) / self.scale
            for idx in range(len(self.breakpoints) - 1):
                m = getattr(self, "group{}".format(idx))
                x = m(x)
                feats.append(x)
            return feats


class L2_(torch.nn.Module):
    def __init__(self):
        """
        Args:
            pre_relu(bool): if True, selects features **before** reLU activations
        """
        super(L2_, self).__init__()
        # VGG using perceptually-learned weights (LPIPS metric)
        augemntations = []
        augemntations.append(transforms.RandomPerspective(
            fill=0, p=1.0, distortion_scale=0.5))
        augemntations.append(transforms.RandomResizedCrop(
            224, scale=(0.8, 0.8), ratio=(1.0, 1.0)))
        augemntations.append(
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)))
        self.augment_trans = transforms.Compose(augemntations)
        # LOG.warning("LPIPS is untested")

    def forward(self, pred, target, mode="train"):
        """Compare VGG features of two inputs."""

        # Get VGG features

        sketch_augs, img_augs = [pred], [target]
        if mode == "train":
            for n in range(4):
                augmented_pair = self.augment_trans(torch.cat([pred, target]))
                sketch_augs.append(augmented_pair[0].unsqueeze(0))
                img_augs.append(augmented_pair[1].unsqueeze(0))

        pred = torch.cat(sketch_augs, dim=0)
        target = torch.cat(img_augs, dim=0)
        diffs = [torch.square(p - t).mean() for (p, t) in zip(pred, target)]
        return sum(diffs)


class CLIPVisualEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.featuremaps = None

        for i in range(12):  # 12 resblocks in VIT visual transformer
            self.clip_model.visual.transformer.resblocks[i].register_forward_hook(
                self.make_hook(i))

    def make_hook(self, name):
        def hook(module, input, output):
            if len(output.shape) == 3:
                self.featuremaps[name] = output.permute(
                    1, 0, 2)  # LND -> NLD bs, smth, 768
            else:
                self.featuremaps[name] = output

        return hook

    def forward(self, x):
        self.featuremaps = collections.OrderedDict()
        fc_features = self.clip_model.encode_image(x).float()
        featuremaps = [self.featuremaps[k] for k in range(12)]

        return fc_features, featuremaps


def l2_layers(xs_conv_features, ys_conv_features, clip_model_name):
    return [torch.square(x_conv - y_conv).mean() for x_conv, y_conv in
            zip(xs_conv_features, ys_conv_features)]


def l1_layers(xs_conv_features, ys_conv_features, clip_model_name):
    return [torch.abs(x_conv - y_conv).mean() for x_conv, y_conv in
            zip(xs_conv_features, ys_conv_features)]


def cos_layers(xs_conv_features, ys_conv_features, clip_model_name):
    if "RN" in clip_model_name:
        return [torch.square(x_conv, y_conv, dim=1).mean() for x_conv, y_conv in
                zip(xs_conv_features, ys_conv_features)]
    return [(1 - torch.cosine_similarity(x_conv, y_conv, dim=1)).mean() for x_conv, y_conv in
            zip(xs_conv_features, ys_conv_features)]


class CLIPConvLoss(torch.nn.Module):
    def __init__(self, args):
        super(CLIPConvLoss, self).__init__()
        self.clip_model_name = args.clip_model_name
        assert self.clip_model_name in [
            "RN50",
            "RN101",
            "RN50x4",
            "RN50x16",
            "ViT-B/32",
            "ViT-B/16",
        ]

        self.clip_conv_loss_type = args.clip_conv_loss_type
        self.clip_fc_loss_type = "Cos"  # args.clip_fc_loss_type
        assert self.clip_conv_loss_type in [
            "L2", "Cos", "L1",
        ]
        assert self.clip_fc_loss_type in [
            "L2", "Cos", "L1",
        ]

        self.distance_metrics = \
            {
                "L2": l2_layers,
                "L1": l1_layers,
                "Cos": cos_layers
            }

        self.model, clip_preprocess = clip.load(
            self.clip_model_name, args.device, jit=False)

        if self.clip_model_name.startswith("ViT"):
            self.visual_encoder = CLIPVisualEncoder(self.model)

        else:
            self.visual_model = self.model.visual
            layers = list(self.model.visual.children())
            init_layers = torch.nn.Sequential(*layers)[:8]
            self.layer1 = layers[8]
            self.layer2 = layers[9]
            self.layer3 = layers[10]
            self.layer4 = layers[11]
            self.att_pool2d = layers[12]

        self.args = args

        self.img_size = clip_preprocess.transforms[1].size
        self.model.eval()
        self.target_transform = transforms.Compose([
            transforms.ToTensor(),
        ])  # clip normalisation
        self.normalize_transform = transforms.Compose([
            clip_preprocess.transforms[0],  # Resize
            clip_preprocess.transforms[1],  # CenterCrop
            clip_preprocess.transforms[-1],  # Normalize
        ])

        self.model.eval()
        self.device = args.device
        self.num_augs = self.args.num_aug_clip

        augemntations = []
        if "affine" in args.augemntations:
            augemntations.append(transforms.RandomPerspective(
                fill=0, p=1.0, distortion_scale=0.5))
            augemntations.append(transforms.RandomResizedCrop(
                224, scale=(0.8, 0.8), ratio=(1.0, 1.0)))
        augemntations.append(
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)))
        self.augment_trans = transforms.Compose(augemntations)

        self.clip_fc_layer_dims = None  # self.args.clip_fc_layer_dims
        self.clip_conv_layer_dims = None  # self.args.clip_conv_layer_dims
        self.clip_fc_loss_weight = args.clip_fc_loss_weight
        self.counter = 0

    def forward(self, sketch, target, mode="train"):
        """
        Parameters
        ----------
        sketch: Torch Tensor [1, C, H, W]
        target: Torch Tensor [1, C, H, W]
        """
        #         y = self.target_transform(target).to(self.args.device)
        conv_loss_dict = {}
        x = sketch.to(self.device)
        y = target.to(self.device)
        sketch_augs, img_augs = [self.normalize_transform(x)], [
            self.normalize_transform(y)]
        if mode == "train":
            for n in range(self.num_augs):
                augmented_pair = self.augment_trans(torch.cat([x, y]))
                sketch_augs.append(augmented_pair[0].unsqueeze(0))
                img_augs.append(augmented_pair[1].unsqueeze(0))

        xs = torch.cat(sketch_augs, dim=0).to(self.device)
        ys = torch.cat(img_augs, dim=0).to(self.device)

        if self.clip_model_name.startswith("RN"):
            xs_fc_features, xs_conv_features = self.forward_inspection_clip_resnet(
                xs.contiguous())
            ys_fc_features, ys_conv_features = self.forward_inspection_clip_resnet(
                ys.detach())

        else:
            xs_fc_features, xs_conv_features = self.visual_encoder(xs)
            ys_fc_features, ys_conv_features = self.visual_encoder(ys)

        conv_loss = self.distance_metrics[self.clip_conv_loss_type](
            xs_conv_features, ys_conv_features, self.clip_model_name)

        for layer, w in enumerate(self.args.clip_conv_layer_weights):
            if w:
                conv_loss_dict[f"clip_conv_loss_layer{layer}"] = conv_loss[layer] * w

        if self.clip_fc_loss_weight:
            # fc distance is always cos
            fc_loss = (1 - torch.cosine_similarity(xs_fc_features,
                       ys_fc_features, dim=1)).mean()
            conv_loss_dict["fc"] = fc_loss * self.clip_fc_loss_weight

        self.counter += 1
        return conv_loss_dict

    def forward_inspection_clip_resnet(self, x):
        def stem(m, x):
            for conv, bn in [(m.conv1, m.bn1), (m.conv2, m.bn2), (m.conv3, m.bn3)]:
                x = m.relu(bn(conv(x)))
            x = m.avgpool(x)
            return x
        x = x.type(self.visual_model.conv1.weight.dtype)
        x = stem(self.visual_model, x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        y = self.att_pool2d(x4)
        return y, [x, x1, x2, x3, x4]
        
class ControlPointLoss(torch.nn.Module):
    def __init__(self, args):
        super(ControlPointLoss, self).__init__()
        self.args = args
    def forward(self, sketches, targets, epoch, renderer, origin_renderer, mode="train"):
        # loss = []
        # point_num = 8
        # shape1 = renderer.get_curve(point_num=point_num)
        # shape2 = origin_renderer.get_curve(point_num=point_num)
        # criterion = torch.nn.MSELoss(reduction='mean')
        # for i in range(len(shape1)):
        #     for j in range(point_num - 2):
        #         loss.append(torch.abs(criterion(shape1[i][j+1],shape1[i][j])-criterion(shape2[i][j+1], shape2[i][j])))
        #         loss.append(torch.abs(criterion(shape1[i][j+2],shape1[i][j])-criterion(shape2[i][j+2], shape2[i][j])))
        #         loss.append(torch.abs(criterion(shape1[i][j+2],shape1[i][j+1])-criterion(shape2[i][j+2], shape2[i][j+1])))
        # control_point_loss = sum(loss)
        # return control_point_loss
        point_cut = 8
        origin_curves = origin_renderer.get_curve(point_cut)
        new_curves = renderer.get_curve(point_cut)
        shape2 = torch.stack([torch.stack([point for point in curve]) for curve in origin_curves])
        shape1 = torch.stack([torch.stack([point for point in curve]) for curve in new_curves])
        diff11 = torch.norm(shape1[:, 1:-1] - shape1[:, :-2], dim=2, p=2)
        diff12 = torch.norm(shape1[:, 2:] - shape1[:, :-2], dim=2, p=2)
        diff13 = torch.norm(shape1[:, 1:-1] - shape1[:, 2:], dim=2, p=2)
        diff21 = torch.norm(shape2[:, 1:-1] - shape2[:, :-2], dim=2, p=2)
        diff22 = torch.norm(shape2[:, 2:] - shape2[:, :-2], dim=2, p=2)
        diff23 = torch.norm(shape2[:, 1:-1] - shape2[:, 2:], dim=2, p=2)
        control_point_loss = torch.abs(diff11 - diff21) + torch.abs(diff12 - diff22) + torch.abs(diff13 - diff23)
        control_point_loss = control_point_loss.sum()
        return control_point_loss

class BezierLoss(torch.nn.Module):
    def __init__(self, args):
        super(BezierLoss, self).__init__()
        self.args = args

    def forward(self, sketches, targets, epoch, renderer, origin_renderer, mode="train"):
        new_curve = renderer.get_curve(100)
        origin_curve = origin_renderer.get_curve(100)
        num_curve = len(new_curve)
        matrix = origin_renderer.matrix
        loss = []
        for i in range(num_curve):
            mean1, _, _, angle1 = self.get_ellipse(torch.stack(new_curve[i]))
            mean2, _, _, angle2 = self.get_ellipse(torch.stack(origin_curve[i]))
            theta = angle1 - angle2
            rotation_matrix = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                                [torch.sin(theta), torch.cos(theta)]]).to(self.args.device)
            transform = mean2 - mean1
            for j in range(num_curve):
                if matrix[i, j]:
                    transformed_curve = torch.matmul(torch.stack(new_curve[j]) + transform, torch.inverse(rotation_matrix))
                    center1 = torch.mean(torch.stack(origin_curve[i]), dim=0)
                    center2 = torch.mean(transformed_curve, dim=0)
                    center3 = torch.mean(torch.stack(origin_curve[j]), dim=0)
                    center_diff1 = center2 - center1
                    center_diff2 = center3 - center1
                    center_diff = center_diff2 - center_diff1
                    loss.append(torch.norm(center_diff, p=2))
        bezier_loss = sum(loss)
        return bezier_loss/1000
        # point_cut = 100
        # new_curves = renderer.get_curve(point_cut)
        # origin_curves = origin_renderer.get_curve(point_cut)
        # matrix = origin_renderer.matrix
        # new_curves = [torch.stack(curve) for curve in new_curves]
        # origin_curves = [torch.stack(curve) for curve in origin_curves]
        # origin_curves = torch.stack(origin_curves)
        # new_curves = torch.stack(new_curves)
        # means1, angles1 = self.get_ellipse(new_curves)
        # means2, angles2 = self.get_ellipse(origin_curves)
        # thetas = angles1 - angles2
        # rotation_matrices = torch.stack([torch.stack([torch.cos(theta), -torch.sin(theta), torch.sin(theta), torch.cos(theta)]) for theta in thetas])
        # rotation_matrices = rotation_matrices.view(-1, 2, 2)
        # transforms = means2 - means1
        # transformed_curves = new_curves + transforms[:, None, :]
        # transformed_curves = transformed_curves.permute(0, 2, 1)
        # transformed_curves = torch.matmul(rotation_matrices, transformed_curves)
        # transformed_curves = transformed_curves.permute(0, 2, 1)
        # transformed_curves = transformed_curves.squeeze()
        # origin_curve_means = torch.stack([torch.mean(curve, dim=0) for curve in origin_curves])
        # transformed_curves = torch.mean(transformed_curves, dim=1)
        # center_diff1 = transformed_curves - origin_curve_means
        # center_diff2 = origin_curve_means.unsqueeze(1) - origin_curve_means.unsqueeze(0)
        # M, _, _ = center_diff2.size()
        # center_diff1 = center_diff1.unsqueeze(0)
        # center_diff1 = center_diff1.expand(M, -1, -1)
        # center_diff = center_diff2 - center_diff1
        # center_diff_norms = torch.norm(center_diff, dim=2, p=2)
        # center_diff_norms = center_diff_norms * matrix
        # bezier_loss = torch.sum(center_diff_norms)
        # return bezier_loss / 1000
    
    # def get_ellipse(self, data_points):
    #     batch_size, num_points, _ = data_points.size()
    #     means = torch.mean(data_points, dim=1)
    #     centered_data = data_points - means.view(batch_size, 1, 2)
    #     cov_matrices = torch.bmm(centered_data.transpose(1, 2), centered_data) / (num_points - 1)
    #     eigenvalues, eigenvectors = torch.symeig(cov_matrices, eigenvectors=True)
    #     major_axis_indices = torch.argmax(eigenvalues, dim=1)
    #     major_axes = torch.gather(eigenvectors, 2, major_axis_indices.view(batch_size, 1, 1).expand(-1, -1, 2))
    #     angles = torch.atan2(major_axes[:, :, 1], major_axes[:, :, 0])
    #     return means, angles
    
    def get_ellipse(self, data_points):
        mean = torch.mean(data_points, dim=0)
        centered_data = data_points - mean
        cov_matrix = torch.mm(centered_data.t(), centered_data) / (data_points.size(0) - 1)
        confidence_level = torch.tensor(0.95, requires_grad=True)
        eigenvalues, eigenvectors = torch.eig(cov_matrix, eigenvectors=True)
        major_axis_index = torch.argmax(eigenvalues[:, 0])
        major_axis = eigenvectors[:, major_axis_index]
        angle = torch.atan2(major_axis[1], major_axis[0])
        width = 2 * torch.sqrt(eigenvalues[major_axis_index] * -2 * torch.log(1 - confidence_level))
        height = 2 * torch.sqrt(eigenvalues[1 - major_axis_index] * -2 * torch.log(1 - confidence_level))
        return mean, width, height, angle

    # def grouping(self, start_points, end_points):
    #     combined_points = torch.cat((torch.stack(start_points), torch.stack(end_points)), dim=1)
    #     scaler = StandardScaler()
    #     combined_points_cpu = combined_points.cpu().detach().numpy()
    #     combined_points_scaled = scaler.fit_transform(combined_points_cpu)
    #     combined_points_scaled = torch.from_numpy(combined_points_scaled).to(self.args.device)
    #     eps = 0.2
    #     min_samples = 2
    #     dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    #     labels = dbscan.fit_predict(combined_points_scaled.cpu().numpy())
    #     matrix = torch.zeros((len(start_points), len(start_points)), requires_grad=True)

    #     for i in range(len(start_points)):
    #         for j in range(len(start_points)):
    #             if labels[i] == labels[j]:
    #                 matrix[i, j] = 1
    #     return matrix
    




#         ##############################
#         ##### loss control point #####
#         ##############################
#         if self.args.track:
#             # new path get
#             shapes = renderer.shapes
#             new_start_point = []
#             new_end_point = []
#             origin_start_point = []
#             origin_end_point = []
#             for path in shapes:
#                 new_start_point.append(path.points[0])
#                 new_end_point.append(path.points[3])
#             # origin path get
#             file_path = f"/data/wujk2022/2023/CLIPasso/output_sketches/test1/test1_16strokes_seed0/svg_logs/svg_iter2000.svg"
#             svg_file = open(file_path, 'r')
#             svg_content = svg_file.read()
#             svg_root = ET.fromstring(svg_content)
#             for element in svg_root.iter():
#                 if 'path' in (element.tag):
#                     coordinate = element.attrib['d'].split(' ')
#                     x0 = float(coordinate[1]) # start point
#                     y0 = float(coordinate[2])
#                     origin_start_point.append(torch.tensor([x0, y0]))
#                     x = float(coordinate[8]) # end point
#                     y = float(coordinate[9])
#                     origin_end_point.append(torch.tensor([x, y]))
#                     # x1 = coordinate[4] # start point -> control point
#                     # y1 = coordinate[5]
#                     # x2 = coordinate[6] # end point -> control point
#                     # y2 = coordinate[7]
#             # print("new_start_point: {}, new_end_point: {}".format(new_start_point, new_end_point))
#             # print("origin_start_point: {}, origin_end_point: {}".format(origin_start_point, origin_end_point))
#             criterion_sum = torch.nn.L1Loss(reduction='sum')
#             loss_control_point_start = criterion_sum(new_start_point[0], origin_start_point[0])
#             loss_control_point_end = criterion_sum(new_end_point[0], origin_end_point[0])
#             for i in range(1, 16):
#                 loss_control_point_start = loss_control_point_start + criterion_sum(new_start_point[i], origin_start_point[i])
#                 loss_control_point_end = loss_control_point_end + criterion_sum(new_end_point[i], origin_end_point[i])
#             loss_control_point = (loss_control_point_start + loss_control_point_end)/(224)
#             # loss_control_point.requires_grad_(True)
#             return loss_control_point
#         else:
#             return torch.tensor(0., requires_grad=True)

class TrackLoss(torch.nn.Module):
    def __init__(self, args):
        super(TrackLoss, self).__init__()
        self.args = args

    def forward(self, sketches, targets, epoch, renderer, origin_renderer, first_renderer, dict_pred, mode="train"):
        # point_cut = 5
        # origin_curves = origin_renderer.get_curve(point_cut)
        # new_curves = renderer.get_curve(point_cut)
        # loss = []
        # for i, curve in enumerate(origin_curves):
        #     for j, point in enumerate(curve):
        #         distances = torch.norm(dict_pred[2] - point, dim=1)
        #         closest_index = torch.argmin(distances).item()
        #         if dict_pred[1][closest_index]:
        #             loss.append(torch.norm(new_curves[i][j] - dict_pred[0][closest_index], p=2))
        # track_loss = sum(loss)/224/224
        # return track_loss
        point_cut = 5
        origin_curves = first_renderer.get_curve(point_cut)
        new_curves = renderer.get_curve(point_cut)
        origin_curves = [[point for point in curve] for curve in origin_curves]
        new_curves = [[point for point in curve] for curve in new_curves]
        flat_origin_curves = [point for curve in origin_curves for point in curve]
        flat_new_curves = [point for curve in new_curves for point in curve]
        flat_origin_curves = torch.stack(flat_origin_curves)
        flat_new_curves = torch.stack(flat_new_curves)
        expanded_dict_pred = dict_pred[2].unsqueeze(0).expand(flat_origin_curves.shape[0], -1, -1)
        expanded_flat_origin_curves = flat_origin_curves.unsqueeze(1).expand(-1, dict_pred[2].shape[0], -1)
        distances = torch.norm(expanded_dict_pred - expanded_flat_origin_curves, dim=2)
        closest_indices = torch.argmin(distances, dim=1)
        closest_bool_values = dict_pred[1][closest_indices]
        filtered_new_curves = flat_new_curves[closest_bool_values]
        closest_points = dict_pred[0][closest_indices[closest_bool_values]]
        loss = torch.norm(filtered_new_curves - closest_points, dim=1, p=2)
        track_loss = loss.mean()
        return track_loss