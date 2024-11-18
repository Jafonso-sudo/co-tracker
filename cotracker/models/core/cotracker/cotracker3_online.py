# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from cotracker.models.core.model_utils import sample_features5d, bilinear_sampler
from cotracker.models.core.embeddings import get_1d_sincos_pos_embed_from_grid

from cotracker.models.core.cotracker.blocks import Mlp, BasicEncoder
from cotracker.models.core.cotracker.cotracker import EfficientUpdateFormer

torch.manual_seed(0)


def posenc(x, min_deg, max_deg):
    """Cat x with a positional encoding of x with scales 2^[min_deg, max_deg-1].
    Instead of computing [sin(x), cos(x)], we use the trig identity
    cos(x) = sin(x + pi/2) and do one vectorized call to sin([x, x+pi/2]).
    Args:
      x: torch.Tensor, variables to be encoded. Note that x should be in [-pi, pi].
      min_deg: int, the minimum (inclusive) degree of the encoding.
      max_deg: int, the maximum (exclusive) degree of the encoding.
      legacy_posenc_order: bool, keep the same ordering as the original tf code.
    Returns:
      encoded: torch.Tensor, encoded variables.
    """
    if min_deg == max_deg:
        return x
    scales = torch.tensor(
        [2**i for i in range(min_deg, max_deg)], dtype=x.dtype, device=x.device
    )

    xb = (x[..., None, :] * scales[:, None]).reshape(list(x.shape[:-1]) + [-1])
    four_feat = torch.sin(torch.cat([xb, xb + 0.5 * torch.pi], dim=-1))
    return torch.cat([x] + [four_feat], dim=-1)


class CoTrackerThreeBase(nn.Module):
    def __init__(
        self,
        window_len=8,
        stride=4,
        corr_radius=3,
        corr_levels=4,
        num_virtual_tracks=64,
        model_resolution=(384, 512),
        add_space_attn=True,
        linear_layer_for_vis_conf=True,
    ):
        super(CoTrackerThreeBase, self).__init__()
        self.window_len = window_len
        self.stride = stride
        self.corr_radius = corr_radius
        self.corr_levels = corr_levels
        self.hidden_dim = 256
        self.latent_dim = 128

        self.linear_layer_for_vis_conf = linear_layer_for_vis_conf
        self.fnet = BasicEncoder(input_dim=3, output_dim=self.latent_dim, stride=stride)

        highres_dim = 128
        lowres_dim = 256

        self.num_virtual_tracks = num_virtual_tracks
        self.model_resolution = model_resolution

        self.input_dim = 1110

        self.updateformer = EfficientUpdateFormer(
            space_depth=3,
            time_depth=3,
            input_dim=self.input_dim,
            hidden_size=384,
            output_dim=4,
            mlp_ratio=4.0,
            num_virtual_tracks=num_virtual_tracks,
            add_space_attn=add_space_attn,
            linear_layer_for_vis_conf=linear_layer_for_vis_conf,
        )
        self.corr_mlp = Mlp(in_features=49 * 49, hidden_features=384, out_features=256)

        time_grid = torch.linspace(0, window_len - 1, window_len).reshape(
            1, window_len, 1
        )

        self.register_buffer(
            "time_emb", get_1d_sincos_pos_embed_from_grid(self.input_dim, time_grid[0])
        )

    def get_support_points(self, coords, r, reshape_back=True):
        B, _, N, _ = coords.shape
        device = coords.device
        centroid_lvl = coords.reshape(B, N, 1, 1, 3)

        dx = torch.linspace(-r, r, 2 * r + 1, device=device)
        dy = torch.linspace(-r, r, 2 * r + 1, device=device)

        xgrid, ygrid = torch.meshgrid(dy, dx, indexing="ij")
        zgrid = torch.zeros_like(xgrid, device=device)
        delta = torch.stack([zgrid, xgrid, ygrid], axis=-1)
        delta_lvl = delta.view(1, 1, 2 * r + 1, 2 * r + 1, 3)
        coords_lvl = centroid_lvl + delta_lvl

        if reshape_back:
            return coords_lvl.reshape(B, N, (2 * r + 1) ** 2, 3).permute(0, 2, 1, 3)
        else:
            return coords_lvl

    #                   # B T C H/4 W/4 | B x N (frame) | B x N x 2 (x, y)
    def get_track_feat(self, fmaps, queried_frames, queried_coords, support_radius=0):
        # B 1 N 1 (frame)
        sample_frames = queried_frames[:, None, :, None]
        # B 1 N 3 (frame, x, y)
        sample_coords = torch.cat(
            [
                sample_frames,
                queried_coords[:, None],  # B 1 N 2
            ],
            dim=-1,
        )
        # For each point, get square of size 2r+1 around it B (2r + 1)^2 N 3, where (2r + 1)^2 is 49
        # TODO: Maybe this is where we would introduce the multi-frame for same point
        # right now we are already doing it somehow for (2r + 1)^2 points in the same frame, but we should do
        # it for (2r + 1)^2 x # of different frames where the point is queried
        # B (2r + 1)^2 N 3
        support_points = self.get_support_points(sample_coords, support_radius)
        # B (2r + 1)^2 N C
        support_track_feats = sample_features5d(fmaps, support_points)
        return (
            support_track_feats[:, None, support_track_feats.shape[1] // 2],
            support_track_feats,
        )

    def get_correlation_feat(self, fmaps, queried_coords):
        B, T, D, H_, W_ = fmaps.shape
        N = queried_coords.shape[1]
        r = self.corr_radius
        sample_coords = torch.cat(
            [torch.zeros_like(queried_coords[..., :1]), queried_coords], dim=-1
        )[:, None]
        support_points = self.get_support_points(sample_coords, r, reshape_back=False)
        correlation_feat = bilinear_sampler(
            fmaps.reshape(B * T, D, 1, H_, W_), support_points
        )
        return correlation_feat.view(B, T, D, N, (2 * r + 1), (2 * r + 1)).permute(
            0, 1, 3, 4, 5, 2
        )

    def interpolate_time_embed(self, x, t):
        previous_dtype = x.dtype
        T = self.time_emb.shape[1]

        if t == T:
            return self.time_emb

        time_emb = self.time_emb.float()
        time_emb = F.interpolate(
            time_emb.permute(0, 2, 1), size=t, mode="linear"
        ).permute(0, 2, 1)
        return time_emb.to(previous_dtype)


class CoTrackerThreeOnline(CoTrackerThreeBase):
    def __init__(self, **args):
        super(CoTrackerThreeOnline, self).__init__(**args)

    def init_video_online_processing(self):
        self.online_ind = 0
        self.online_track_feat = [None] * self.corr_levels
        self.online_track_support = [None] * self.corr_levels
        self.online_coords_predicted = None
        self.online_vis_predicted = None
        self.online_conf_predicted = None

    def forward_window(
        self,
        fmaps_pyramid,  # 4 scales x B T C H/4 W/4
        coords,  # B T N 2
        track_feat_support_pyramid,  # 4 scales x B 1 (2r + 1)^2 N C
        vis=None,  # B T N 1
        conf=None,  # B T N 1
        attention_mask=None,  # B T N (Has query for point i appeared in fram j?)
        iters=4,
        add_space_attn=False,
    ):
        B, S, *_ = fmaps_pyramid[0].shape
        N = coords.shape[2]
        r = 2 * self.corr_radius + 1

        coord_preds, vis_preds, conf_preds = [], [], []
        for it in range(iters):
            coords = coords.detach()  # B T N 2
            coords_init = coords.view(B * S, N, 2)
            corr_embs = []
            corr_feats = []
            for i in range(self.corr_levels):
                corr_feat = self.get_correlation_feat(  # B T N (2r + 1) (2r + 1) C
                    fmaps_pyramid[i], coords_init / 2**i
                )
                track_feat_support = (  # B N (2r + 1) (2r + 1) C
                    track_feat_support_pyramid[i]
                    .view(B, 1, r, r, N, self.latent_dim)
                    .squeeze(1)
                    .permute(0, 3, 1, 2, 4)
                )
                corr_volume = torch.einsum(  # B T N (2r + 1) (2r + 1) (2r + 1) (2r + 1)
                    "btnhwc,bnijc->btnhwij", corr_feat, track_feat_support
                )
                corr_emb = self.corr_mlp(
                    corr_volume.reshape(B * S * N, r * r * r * r)
                )  # Num Batches: B * T * N, Input Size: 49 * 49, Output Size: 256
                # B * T * N, 256
                corr_embs.append(corr_emb)

            corr_embs = torch.cat(corr_embs, dim=-1)
            corr_embs = corr_embs.view(
                B, S, N, corr_embs.shape[-1]
            )  # B T N 256 * 4 TODO: We would treat alternates the exact same so far until this point

            transformer_input = [vis, conf, corr_embs]

            rel_coords_forward = coords[:, :-1] - coords[:, 1:]
            rel_coords_backward = coords[:, 1:] - coords[:, :-1]

            rel_coords_forward = torch.nn.functional.pad(
                rel_coords_forward, (0, 0, 0, 0, 0, 1)
            )
            rel_coords_backward = torch.nn.functional.pad(
                rel_coords_backward, (0, 0, 0, 0, 1, 0)
            )

            scale = (
                torch.tensor(
                    [self.model_resolution[1], self.model_resolution[0]],
                    device=coords.device,
                )
                / self.stride
            )
            rel_coords_forward = rel_coords_forward / scale
            rel_coords_backward = rel_coords_backward / scale

            rel_pos_emb_input = posenc(
                torch.cat([rel_coords_forward, rel_coords_backward], dim=-1),
                min_deg=0,
                max_deg=10,
            )  # B T N 84
            transformer_input.append(rel_pos_emb_input)

            x = (
                torch.cat(transformer_input, dim=-1)
                .permute(0, 2, 1, 3)
                .reshape(B * N, S, -1)
            )

            x = x + self.interpolate_time_embed(x, S)
            x = x.view(B, N, S, -1)  # (B N) T D -> B N T D

            delta = self.updateformer(x, add_space_attn=add_space_attn)

            delta_coords = delta[..., :2].permute(0, 2, 1, 3)
            delta_vis = delta[..., 2:3].permute(0, 2, 1, 3)
            delta_conf = delta[..., 3:].permute(0, 2, 1, 3)

            vis = vis + delta_vis
            conf = conf + delta_conf

            coords = coords + delta_coords
            # TODO: Here is where we would apply our rigidity prior
            
            
            coord_preds.append(coords[..., :2] * float(self.stride))

            vis_preds.append(vis[..., 0])
            conf_preds.append(conf[..., 0])
        return coord_preds, vis_preds, conf_preds

    def forward(
        self,
        video,
        queries,
        iters=4,
        is_train=False,
        add_space_attn=True,
        fmaps_chunk_size=200,
        is_online=False,
        init_coords=None,
        init_vis=None,
        init_confidence=None,
        init_length=None,
    ):
        """Predict tracks

        Args:
            video (FloatTensor[B, T, 3]): input videos.
            queries (FloatTensor[B, N, 3]): point queries.
            iters (int, optional): number of updates. Defaults to 4.
            is_train (bool, optional): enables training mode. Defaults to False.
        Returns:
            - coords_predicted (FloatTensor[B, T, N, 2]):
            - vis_predicted (FloatTensor[B, T, N]):
            - train_data: `None` if `is_train` is false, otherwise:
                - all_vis_predictions (List[FloatTensor[B, S, N, 1]]):
                - all_coords_predictions (List[FloatTensor[B, S, N, 2]]):
                - mask (BoolTensor[B, T, N]):
        """

        B, T, C, H, W = video.shape
        device = queries.device
        assert H % self.stride == 0 and W % self.stride == 0

        B, N, __ = queries.shape
        # B = batch size
        # S_trimmed = actual number of frames in the window
        # N = number of tracks
        # C = color channels (3 for RGB)
        # E = positional embedding size
        # LRR = local receptive field radius
        # D = dimension of the transformer input tokens

        # video = B T C H W
        # queries = B N 3
        # coords_init = B T N 2
        # vis_init = B T N 1
        S = self.window_len
        assert S >= 2  # A tracker needs at least two frames to track something
        if is_online:
            assert T <= S, "Online mode: video chunk must be <= window size."
            assert (
                self.online_ind is not None
            ), "Call model.init_video_online_processing() first."
            assert not is_train, "Training not supported in online mode."

        step = S // 2  # How much the sliding window moves at every step

        video = 2 * (video / 255.0) - 1.0
        pad = (
            S - T if is_online else (S - T % S) % S
        )  # We don't want to pad if T % S == 0
        video = video.reshape(B, 1, T, C * H * W)
        if pad > 0:
            padding_tensor = video[:, :, -1:, :].expand(B, 1, pad, C * H * W)
            video = torch.cat([video, padding_tensor], dim=2)
        video = video.reshape(B, -1, C, H, W)
        T_pad = video.shape[1]
        # The first channel is the frame number
        # The rest are the coordinates of points we want to track
        dtype = video.dtype
        # For each query point, which frame it is queried in
        queried_frames = queries[:, :, 0].long()
        # For each query point, where it is queried from (location)
        queried_coords = queries[..., 1:3]
        queried_coords = queried_coords / self.stride
        # Apply stride to ground truths as well
        if init_coords is not None:
            init_coords_strided = init_coords / self.stride

        # We store our predictions here (In online T = S = window size)
        coords_predicted = torch.zeros((B, T, N, 2), device=device)
        vis_predicted = torch.zeros((B, T, N), device=device)
        conf_predicted = torch.zeros((B, T, N), device=device)

        if is_online:
            if self.online_coords_predicted is None:
                # Init online predictions with zeros
                self.online_coords_predicted = coords_predicted
                self.online_vis_predicted = vis_predicted
                self.online_conf_predicted = conf_predicted
            else:
                # Pad online predictions with zeros for the current window
                pad = min(step, T - step)
                coords_predicted = F.pad(
                    self.online_coords_predicted, (0, 0, 0, 0, 0, pad), "constant"
                )
                vis_predicted = F.pad(
                    self.online_vis_predicted, (0, 0, 0, pad), "constant"
                )
                conf_predicted = F.pad(
                    self.online_conf_predicted, (0, 0, 0, pad), "constant"
                )

        # We store our predictions here (For training)
        all_coords_predictions, all_vis_predictions, all_confidence_predictions = (
            [],
            [],
            [],
        )

        C_ = C
        H4, W4 = H // self.stride, W // self.stride

        # Compute convolutional features for the video or for the current chunk in case of online mode
        if (not is_train) and (T > fmaps_chunk_size):
            fmaps = []
            for t in range(0, T, fmaps_chunk_size):
                video_chunk = video[:, t : t + fmaps_chunk_size]
                fmaps_chunk = self.fnet(video_chunk.reshape(-1, C_, H, W))
                T_chunk = video_chunk.shape[1]
                C_chunk, H_chunk, W_chunk = fmaps_chunk.shape[1:]
                fmaps.append(fmaps_chunk.reshape(B, T_chunk, C_chunk, H_chunk, W_chunk))
            fmaps = torch.cat(fmaps, dim=1).reshape(-1, C_chunk, H_chunk, W_chunk)
        else:  # Compute all features at once (Online model or small video)
            fmaps = self.fnet(video.reshape(-1, C_, H, W))
        fmaps = fmaps.permute(0, 2, 3, 1)
        fmaps = fmaps / torch.sqrt(
            torch.maximum(
                torch.sum(torch.square(fmaps), axis=-1, keepdims=True),
                torch.tensor(1e-12, device=fmaps.device),
            )
        )
        fmaps = fmaps.permute(0, 3, 1, 2).reshape(
            B, -1, self.latent_dim, H // self.stride, W // self.stride
        )
        fmaps = fmaps.to(dtype)

        # We compute track features
        fmaps_pyramid = []
        track_feat_pyramid = []
        track_feat_support_pyramid = []
        fmaps_pyramid.append(fmaps)
        for i in range(self.corr_levels - 1):
            fmaps_ = fmaps.reshape(
                B * T_pad, self.latent_dim, fmaps.shape[-2], fmaps.shape[-1]
            )
            fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
            fmaps = fmaps_.reshape(
                B, T_pad, self.latent_dim, fmaps_.shape[-2], fmaps_.shape[-1]
            )
            fmaps_pyramid.append(fmaps)
        # Check which query frames are in the current window (left, left + S) i.e. (left, left + 16)
        if is_online:
            sample_frames = queried_frames[:, None, :, None]  # B 1 N 1
            left = 0 if self.online_ind == 0 else self.online_ind + step
            right = self.online_ind + S
            # B 1 N 1 (True/False) - True if the frame is in the current window
            sample_mask = (sample_frames >= left) & (sample_frames < right)

        for i in range(self.corr_levels):
            # TODO: This could be where the queried frames correlation is computed, could be easy to add intermediates?
            # B 1 N C, B (2r + 1)^2 N C - features from point, features from support neighbour points
            track_feat, track_feat_support = self.get_track_feat(
                fmaps_pyramid[i],
                queried_frames - self.online_ind if is_online else queried_frames,
                queried_coords / 2**i,
                support_radius=self.corr_radius,
            )

            if is_online:
                if self.online_track_feat[i] is None:
                    self.online_track_feat[i] = torch.zeros_like(
                        track_feat, device=device
                    )
                    self.online_track_support[i] = torch.zeros_like(
                        track_feat_support, device=device
                    )

                self.online_track_feat[i] += track_feat * sample_mask
                self.online_track_support[i] += track_feat_support * sample_mask
                track_feat_pyramid.append(
                    self.online_track_feat[i].repeat(1, T_pad, 1, 1)
                )
                track_feat_support_pyramid.append(
                    self.online_track_support[i].unsqueeze(1)
                )
            else:
                track_feat_pyramid.append(track_feat.repeat(1, T_pad, 1, 1))
                track_feat_support_pyramid.append(track_feat_support.unsqueeze(1))

        D_coords = 2
        coord_preds, vis_preds, confidence_preds = [], [], []

        vis_init = torch.zeros((B, S, N, 1), device=device).float()
        conf_init = torch.zeros((B, S, N, 1), device=device).float()
        coords_init = queried_coords.reshape(B, 1, N, 2).expand(B, S, N, 2).float()
        # TODO: THIS IS WHERE WE COULD ADD THE INITIALIZATION HELP
        # - Currently they initialize the coords_init with the queried coords and propagate to entire video
        # - We can do one better by initializing them in the prepend video, and potentially giving a better initialization later on

        num_windows = (T - S + step - 1) // step + 1
        # We process only the current video chunk in the online mode
        indices = [self.online_ind] if is_online else range(0, step * num_windows, step)

        for ind in indices:
            if ind > 0:
                overlap = S - step
                # NOTE: Here they create a mask for queried frames that are in the overlap region
                copy_over = (queried_frames < ind + overlap)[
                    :, None, :, None
                ]  # B 1 N 1
                coords_prev = coords_predicted[:, ind : ind + overlap] / self.stride
                padding_tensor = coords_prev[:, -1:, :, :].expand(-1, step, -1, -1)
                coords_prev = torch.cat([coords_prev, padding_tensor], dim=1)

                vis_prev = vis_predicted[:, ind : ind + overlap, :, None].clone()
                padding_tensor = vis_prev[:, -1:, :, :].expand(-1, step, -1, -1)
                vis_prev = torch.cat([vis_prev, padding_tensor], dim=1)

                conf_prev = conf_predicted[:, ind : ind + overlap, :, None].clone()
                padding_tensor = conf_prev[:, -1:, :, :].expand(-1, step, -1, -1)
                conf_prev = torch.cat([conf_prev, padding_tensor], dim=1)
                # TODO: Here is the final step where they copy over the previous predictions
                coords_init = torch.where(
                    copy_over.expand_as(coords_init), coords_prev, coords_init
                )
                vis_init = torch.where(
                    copy_over.expand_as(vis_init), vis_prev, vis_init
                )
                conf_init = torch.where(
                    copy_over.expand_as(conf_init), conf_prev, conf_init
                )

            attention_mask = (queried_frames < ind + S).reshape(
                B, 1, N
            )  # B 1 N (Which points are queried in the current window or before)
            # import ipdb; ipdb.set_trace()
            coords, viss, confs = self.forward_window(
                fmaps_pyramid=(
                    fmaps_pyramid  # 4 scales x B T C H/4 W/4
                    if is_online
                    else [fmap[:, ind : ind + S] for fmap in fmaps_pyramid]
                ),
                coords=coords_init,
                track_feat_support_pyramid=[
                    attention_mask[:, None, :, :, None]
                    * tfeat  # B 1 (2r + 1)^2 N C (Only the points from queries that have already appeared)
                    for tfeat in track_feat_support_pyramid
                ],
                vis=vis_init,
                conf=conf_init,
                attention_mask=attention_mask.repeat(1, S, 1),  # B S N
                iters=iters,
                add_space_attn=add_space_attn,
            )
            S_trimmed = (
                T if is_online else min(T - ind, S)
            )  # accounts for last window duration
            coords_predicted[:, ind : ind + S] = coords[-1][:, :S_trimmed]
            vis_predicted[:, ind : ind + S] = viss[-1][:, :S_trimmed]
            conf_predicted[:, ind : ind + S] = confs[-1][:, :S_trimmed]
            if (
                init_confidence is not None
                and init_vis is not None
                and init_coords is not None
                and init_length is not None
                and ind < init_length
            ):
                left, right = ind, min(ind + S, init_length)
                conf_predicted[:, left:right] = init_confidence[:, left:right]
                vis_predicted[:, left:right] = init_vis[:, left:right]
                coords_predicted[:, left:right] = init_coords[:, left:right]
            if is_train:
                all_coords_predictions.append(
                    [coord[:, :S_trimmed] for coord in coords]
                )
                all_vis_predictions.append(
                    [torch.sigmoid(vis[:, :S_trimmed]) for vis in viss]
                )
                all_confidence_predictions.append(
                    [torch.sigmoid(conf[:, :S_trimmed]) for conf in confs]
                )
        if is_online:
            self.online_ind += step
            self.online_coords_predicted = coords_predicted
            self.online_vis_predicted = vis_predicted
            self.online_conf_predicted = conf_predicted
        vis_predicted = torch.sigmoid(vis_predicted)
        conf_predicted = torch.sigmoid(conf_predicted)

        if is_train:
            valid_mask = (
                queried_frames[:, None]
                <= torch.arange(0, T, device=device)[None, :, None]
            )
            train_data = (
                all_coords_predictions,
                all_vis_predictions,
                all_confidence_predictions,
                valid_mask,
            )
        else:
            train_data = None

        return coords_predicted, vis_predicted, conf_predicted, train_data
