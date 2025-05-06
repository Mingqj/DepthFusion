import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, kaiming_init, build_conv_layer
from mmcv.runner import force_fp32
from torch import nn
from .transfusion_head import TransFusionHead
from mmdet3d.models.builder import HEADS
from .. import builder
from torch.nn import Softmax
import numpy as np
from ...ops.cross_attention_2d import DeformableAttention2D
from ...ops.cross_attention_1d import DeformableAttention1D

__all__ = ["DepthFusionLargeHead"]

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class FeedForward2D(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FeedForward2D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channel, out_channel, kernel_size=3, padding=2, dilation=2
            ),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

def clip_sigmoid(x, eps=1e-4):
    y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y

@HEADS.register_module()
class DepthFusionLargeHead(TransFusionHead):
    def __init__(self,
                 img_feat_dim=128,
                 feat_bev_img_dim=32,
                 sparse_fuse_layers=2,
                 dense_fuse_layers=2,
                 **kwargs):
        super(DepthFusionLargeHead, self).__init__(**kwargs)

        # global
        self.patch_size = 5
        self.hiddle_channel = 128 * self.patch_size
        self.dim_channel = 256
        self.up_c = nn.Sequential(
            nn.Conv2d(128, self.hiddle_channel, kernel_size=self.patch_size, stride=self.patch_size),
            nn.BatchNorm2d(self.hiddle_channel),
            nn.ReLU(True),
            nn.Conv2d(self.hiddle_channel, self.dim_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.dim_channel, eps=1e-5, momentum=0.1),
            nn.ReLU(True),
        )

        self.up_l = nn.Sequential(
            nn.Conv2d(128, self.hiddle_channel, kernel_size=self.patch_size, stride=self.patch_size),
            nn.BatchNorm2d(self.hiddle_channel),
            nn.ReLU(True),
            nn.Conv2d(self.hiddle_channel, self.dim_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.dim_channel, eps=1e-5, momentum=0.1),
            nn.ReLU(True),
        )

        self.down_l = nn.Sequential(
            nn.Conv2d(self.dim_channel, self.dim_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.dim_channel, eps=1e-5, momentum=0.1),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_channel, 128, kernel_size=self.patch_size, stride=self.patch_size),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )

        self.c = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-5, momentum=0.1),
            nn.ReLU(True),
        )

        self.heatmap_sparse_head = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-5, momentum=0.1),
            nn.ReLU(True),
            nn.Conv2d(128, 10, kernel_size=3, stride=1, padding=1, bias=False),
        )


        self.defor_cross_attention = DeformableAttention2D(
                                        dim = self.dim_channel,
                                        dim_head = 64,
                                        heads = 8,
                                        dropout = 0.1,
                                        downsample_factor = 4,
                                        offset_scale = 4,
                                        offset_groups = None,
                                        offset_kernel_size = 6
                                    )
        self.FFN = FeedForward2D(128, 128)
        self.norm_a1 = nn.BatchNorm2d(128)
        self.norm_a2 = nn.BatchNorm2d(128)

        self.voxel_encoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-5, momentum=0.1),
            nn.ReLU(True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-5, momentum=0.1),
            nn.ReLU(True),
        )

        self.defor_cross_attention_sparse_IF = DeformableAttention1D(
                                        dim = 128,
                                        dim_head = 64,
                                        heads = 8,
                                        dropout = 0.1,
                                        downsample_factor = 4,
                                        offset_scale = 4,
                                        offset_groups = None,
                                        offset_kernel_size = 6
                                    )
        self.defor_cross_attention_sparse_LF = DeformableAttention1D(
                                        dim = 128,
                                        dim_head = 64,
                                        heads = 8,
                                        dropout = 0.1,
                                        downsample_factor = 4,
                                        offset_scale = 4,
                                        offset_groups = None,
                                        offset_kernel_size = 6
                                    )
        self.defor_cross_attention_sparse_FF = DeformableAttention1D(
                                        dim = 128,
                                        dim_head = 64,
                                        heads = 8,
                                        dropout = 0.1,
                                        downsample_factor = 4,
                                        offset_scale = 4,
                                        offset_groups = None,
                                        offset_kernel_size = 6
                                    )
        self.sparse2sparse = nn.Sequential(
            nn.Conv1d(256, 128, 1, bias = False),
            nn.GELU(),
            nn.Conv1d(128, 128, 1, bias = False),
        )
        self.FFN_sparse = FeedForward2D(128, 128)
        self.norm_a3 = nn.BatchNorm2d(128)
        self.norm_a4 = nn.BatchNorm2d(128)

        # De_dense
        dis_width, dis_length = 180 // self.patch_size, 180 // self.patch_size
        self.dist_matrix = torch.ones((dis_width, dis_width))
        position_threshold = int(dis_width * 0.2)
        distance_threshold = np.sqrt((position_threshold - (dis_width + 1) / 2) ** 2 + (position_threshold - (dis_length + 1) / 2) ** 2)
        for i in range(1, dis_width + 1):
            for j in range(1, dis_width + 1):
                if np.sqrt((i - (dis_width + 1) / 2) ** 2 + (j - (dis_width + 1) / 2) ** 2) < distance_threshold:
                    self.dist_matrix[i - 1, j - 1] = distance_threshold
                else:
                    self.dist_matrix[i - 1, j - 1] = np.sqrt((i - (dis_width + 1) / 2) ** 2 + (j - (dis_length + 1) / 2) ** 2)

        min_m = torch.min(self.dist_matrix)
        max_m = torch.max(self.dist_matrix)
        self.dist_matrix = (self.dist_matrix - min_m) / (max_m - min_m) + 1

        # De_sparse
        dis_width, dis_length = 180, 180
        self.dist_matrix_sparse = torch.ones((dis_width, dis_width))
        position_threshold = int(dis_width * 0.2)
        distance_threshold = np.sqrt((position_threshold - (dis_width + 1) / 2) ** 2 + (position_threshold - (dis_length + 1) / 2) ** 2)
        for i in range(1, dis_width + 1):
            for j in range(1, dis_width + 1):
                if np.sqrt((i - (dis_width + 1) / 2) ** 2 + (j - (dis_width + 1) / 2) ** 2) < distance_threshold:
                    self.dist_matrix_sparse[i - 1, j - 1] = distance_threshold
                else:
                    self.dist_matrix_sparse[i - 1, j - 1] = np.sqrt((i - (dis_width + 1) / 2) ** 2 + (j - (dis_length + 1) / 2) ** 2)

        min_m = torch.min(self.dist_matrix_sparse)
        max_m = torch.max(self.dist_matrix_sparse)
        self.dist_matrix_sparse = (self.dist_matrix_sparse - min_m) / (max_m - min_m) + 1
        
        # INVERSE De_sparse
        self.idist_matrix_sparse = (1 - (self.dist_matrix_sparse - 1 )) + 1 

        ########################################################################################
        # fuse net for first stage dense prediction
        cfg = dict(
            type='CustomResNet',
            numC_input=kwargs['hidden_channel'],
            num_layer=[dense_fuse_layers+1, ],
            num_channels=[kwargs['hidden_channel'], ],
            stride=[1, ],
            backbone_output_ids=[0, ])
        self.dense_heatmap_fuse_convs = builder.build_backbone(cfg)

        # fuse net for second stage sparse prediction
        fuse_convs = []
        c_in = img_feat_dim + kwargs['hidden_channel'] + feat_bev_img_dim
        for i in range(sparse_fuse_layers - 1):
            fuse_convs.append(
                ConvModule(
                    c_in,
                    c_in,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias='auto',
                    conv_cfg=dict(type='Conv1d'),
                    norm_cfg=dict(type="BN1d")))
        fuse_convs.append(
            ConvModule(
                c_in,
                kwargs['hidden_channel'],
                kernel_size=1,
                stride=1,
                padding=0,
                bias='auto',
                conv_cfg=dict(type='Conv1d'),
                norm_cfg=dict(type="BN1d")))
        self.fuse_convs = nn.Sequential(*fuse_convs)

        fuse_convs0 = []
        c_in = img_feat_dim + kwargs['hidden_channel'] + feat_bev_img_dim
        for i in range(sparse_fuse_layers - 1):
            fuse_convs0.append(
                ConvModule(
                    c_in,
                    c_in,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias='auto',
                    conv_cfg=dict(type='Conv1d'),
                    norm_cfg=dict(type="BN1d")))
        fuse_convs0.append(
            ConvModule(
                c_in,
                kwargs['hidden_channel'],
                kernel_size=1,
                stride=1,
                padding=0,
                bias='auto',
                conv_cfg=dict(type='Conv1d'),
                norm_cfg=dict(type="BN1d")))
        self.fuse_convs0 = nn.Sequential(*fuse_convs0)

        self._init_weights()

    def _init_weights(self):
        for m in self.dense_heatmap_fuse_convs.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

    @force_fp32()
    def extract_img_feat_from_3dpoints(self, points, img_inputs_list, fuse=True):
        if not isinstance(img_inputs_list[0], list):
            img_inputs_list = [img_inputs_list]
        global2keyego = torch.inverse(img_inputs_list[0][2][:,0,:,:].unsqueeze(1).to(torch.float64))
        point_img_feat_list = []

        b, p, _ = points.shape
        points = points.view(b, 1, -1, 3, 1)
        for img_inputs in img_inputs_list:
            img_feats = img_inputs[0].permute(0, 2, 1, 3, 4).contiguous()
            _, c, n, h, w = img_feats.shape
            with torch.no_grad():

                sensor2ego, ego2global, cam2imgs, post_rots, post_trans, bda = \
                    img_inputs[1:]
                currego2global = ego2global[:,0,:,:].unsqueeze(1).to(torch.float64)
                currego2keyego = global2keyego.matmul(currego2global).to(torch.float32)

                # aug ego to cam
                augego2cam = torch.inverse(bda.view(b, 1, 4, 4).matmul(currego2keyego).matmul(sensor2ego))
                augego2cam = augego2cam.view(b, -1, 1, 4, 4)
                points_cam = augego2cam[..., :3, :3].matmul(points)
                points_cam += augego2cam[:, :, :, :3, 3:4]

                valid = points_cam[..., 2, 0] > 0.5
                points_img = points_cam/points_cam[..., 2:3, :]
                points_img = cam2imgs.view(b, -1, 1, 3, 3).matmul(points_img)

                points_img_x = points_img[..., 0, 0]
                points_img_x = points_img_x * valid
                select_cam_ids = \
                    torch.argmin(torch.abs(points_img_x -
                                           cam2imgs[:, :, 0, 2:3]), dim=1)

                points_img = post_rots.view(b, -1, 1, 3, 3).matmul(points_img) + \
                             post_trans.view(b, -1, 1, 3, 1)

                points_img[..., 2, 0] = points_cam[..., 2, 0]

                points_img = points_img[..., :2, 0]
                index = select_cam_ids[:, None, :, None].expand(-1, -1, -1, 2)
                points_img_selected = \
                    points_img.gather(index=index, dim=1).squeeze(1)

                # img space to feature space
                points_img_selected /= self.test_cfg['img_feat_downsample']

                grid = torch.cat([points_img_selected,
                                  select_cam_ids.unsqueeze(-1)], dim=2)

                normalize_factor = torch.tensor([w - 1.0, h - 1.0, n - 1.0]).to(grid)
                grid = grid / normalize_factor.view(1, 1, 3) * 2.0 - 1.0
                grid = grid.view(b, p, 1, 1, 3)
            point_img_feat = \
                F.grid_sample(img_feats, grid,
                              mode='bilinear',
                              align_corners=True).view(b,c,p)
            point_img_feat_list.append(point_img_feat)
        if not fuse:
            point_img_feat = point_img_feat_list[0]
        else:
            point_img_feat = point_img_feat_list
        return point_img_feat

    def extract_instance_img_feat(self, res_layer, img_inputs, fuse=False):
        center = res_layer["center"]
        height = res_layer["height"]
        center_x = center[:, 0:1, :] * self.bbox_coder.out_size_factor * \
                   self.bbox_coder.voxel_size[0] + self.bbox_coder.pc_range[0]
        center_y = center[:, 1:2, :] * self.bbox_coder.out_size_factor * \
                   self.bbox_coder.voxel_size[1] + self.bbox_coder.pc_range[1]

        ref_points = torch.cat([center_x, center_y, height], dim=1).permute(0, 2, 1)

        img_feat = self.extract_img_feat_from_3dpoints(ref_points, img_inputs, fuse=fuse)
        return img_feat

    def extract_instance_img_feat0(self, res_layer, img_inputs, fuse=False):
        center = res_layer["center0"]
        height = res_layer["height0"]
        center_x = center[:, 0:1, :] * self.bbox_coder.out_size_factor * \
                   self.bbox_coder.voxel_size[0] + self.bbox_coder.pc_range[0]
        center_y = center[:, 1:2, :] * self.bbox_coder.out_size_factor * \
                   self.bbox_coder.voxel_size[1] + self.bbox_coder.pc_range[1]

        ref_points = torch.cat([center_x, center_y, height], dim=1).permute(0, 2, 1)

        img_feat = self.extract_img_feat_from_3dpoints(ref_points, img_inputs, fuse=fuse)
        return img_feat

    def extract_proposal(self, heatmap):
        batch_size = heatmap.shape[0]
        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        # equals to nms radius = voxel_size * out_size_factor * kenel_size
        local_max_inner = F.max_pool2d(heatmap, stride=1, padding=0,
                                       kernel_size=self.nms_kernel_size)
        local_max[:, :, padding:(-padding), padding:(-padding)] = \
            local_max_inner
        ## for Pedestrian & Traffic_cone in nuScenes
        if self.test_cfg["dataset"] == "nuScenes":
            local_max[:, 8,] = F.max_pool2d(heatmap[:, 8], kernel_size=1,
                                            stride=1, padding=0)
            local_max[:, 9,] = F.max_pool2d(heatmap[:, 9], kernel_size=1,
                                            stride=1, padding=0)
        elif self.test_cfg["dataset"] == "Waymo":
            # for Pedestrian & Cyclist in Waymo
            local_max[:, 1,] = F.max_pool2d(heatmap[:, 1], kernel_size=1,
                                            stride=1, padding=0)
            local_max[:, 2,] = F.max_pool2d(heatmap[:, 2], kernel_size=1,
                                            stride=1, padding=0)
        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)

        # top #num_proposals among all classes
        top_proposals = heatmap.view(batch_size, -1)
        top_proposals = top_proposals.argsort(dim=-1, descending=True)
        top_proposals = top_proposals[..., :self.num_proposals]
        top_proposals_class = top_proposals // heatmap.shape[-1]
        top_proposals_index = top_proposals % heatmap.shape[-1]
        top_proposals_index = top_proposals_index.unsqueeze(1)
        return top_proposals_class, top_proposals_index

    def forward_single(self, inputs, img_inputs, bev_feat_img, img_metas):
        """Forward function for CenterPoint.
        Args:
            inputs (torch.Tensor): Input feature map with the shape of
                [B, 512, 128(H), 128(W)]. (consistent with L748)
        Returns:
            list[dict]: Output results for tasks.
        """
        batch_size = inputs.shape[0]

        bev_feat_lidar = self.shared_conv(inputs)
        bev_feat_lidar_flatten = bev_feat_lidar.view(batch_size, bev_feat_lidar.shape[1], -1)  # [BS, C, H*W]

        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(bev_feat_lidar.device)
        
        ##########################################################
        # ori to patch
        up_l = self.up_l(bev_feat_lidar)
        c_expand_c = self.c(bev_feat_img)
        up_c = self.up_c(c_expand_c)

        # De
        De = self.dist_matrix.unsqueeze(0).repeat(up_l.shape[1], 1, 1).unsqueeze(0).repeat(up_l.shape[0], 1, 1, 1).cuda()
        De_sparse = self.dist_matrix_sparse.unsqueeze(0).repeat(up_l.shape[1], 1, 1).unsqueeze(0).repeat(up_l.shape[0], 1, 1, 1).cuda()
        IDe_sparse = self.idist_matrix_sparse.unsqueeze(0).repeat(up_l.shape[1], 1, 1).unsqueeze(0).repeat(up_l.shape[0], 1, 1, 1).cuda()

        # cross-attention
        clfusion_l = self.defor_cross_attention(up_l, up_c, De) 
        dense_fuse_feat_attention = self.down_l(clfusion_l)
    
        # Add & Norm
        dense_fuse_feat_attention = self.norm_a1(dense_fuse_feat_attention + bev_feat_lidar + c_expand_c)
        
        # FFN
        dense_fuse_feat_attention_ffn = self.FFN(dense_fuse_feat_attention)

        # Add & Norm
        dense_fuse_feat_attention = self.norm_a2(dense_fuse_feat_attention_ffn + dense_fuse_feat_attention)

        dense_heatmap = self.heatmap_head(dense_fuse_feat_attention)
        heatmap = dense_heatmap.detach().sigmoid()

        # generate proposal
        top_proposals_class, top_proposals_index = self.extract_proposal(heatmap)
        self.query_labels = top_proposals_class

        # prepare sparse lidar feat of proposal
        index = top_proposals_index.expand(-1, bev_feat_lidar_flatten.shape[1],
                                           -1)
        query_feat_lidar = bev_feat_lidar_flatten.gather(index=index, dim=-1)

        # add category embedding
        one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(0, 2, 1)
        query_cat_encoding = self.class_encoding(one_hot.float())
        query_feat_lidar += query_cat_encoding

        query_pos_index = top_proposals_index.permute(0, 2, 1)
        query_pos_index = query_pos_index.expand(-1, -1, bev_pos.shape[-1])
        query_pos = bev_pos.gather(index=query_pos_index, dim=1)

        # Prediction
        res = dict()
        for task in ['height0', 'center0', 'dim0', 'rot0', 'vel0']:
            res[task] = \
                self.prediction_heads0[0].__getattr__(task)(query_feat_lidar)
        res['center0'] += query_pos.permute(0, 2, 1)

        # generate sparse fuse feat
        query_feat_img = self.extract_instance_img_feat0(res, img_inputs)

        bev_feat_img = bev_feat_img.view(batch_size, bev_feat_img.shape[1], -1)
        index = top_proposals_index.expand(-1, bev_feat_img.shape[1], -1)
        query_feat_img_bev = bev_feat_img.gather(index=index, dim=-1)

        query_feat_fuse = torch.cat([query_feat_lidar, query_feat_img, query_feat_img_bev], dim=1) 
        
        query_feat_fuse = self.fuse_convs0(query_feat_fuse)
        res['heatmap0'] = \
            self.prediction_heads0[0].__getattr__('heatmap0')(query_feat_fuse)

        ###################################################################
        dense_fuse_feat_attention_flatten = dense_fuse_feat_attention.view(batch_size, dense_fuse_feat_attention.shape[1], -1) # [bs, 128, 180 * 180]
        index = top_proposals_index.expand(-1, dense_fuse_feat_attention_flatten.shape[1], -1)
        query_dense_fuse_feat_attention = dense_fuse_feat_attention_flatten.gather(index=index, dim=-1)

        De_sparse_flatten = De_sparse.view(batch_size, De_sparse.shape[-1], -1)
        query_De_sparse = De_sparse_flatten.gather(index=index, dim=-1)

        IDe_sparse_flatten = IDe_sparse.view(batch_size, IDe_sparse.shape[-1], -1)
        query_IDe_sparse = IDe_sparse_flatten.gather(index=index, dim=-1)

        voxel_feat = self.voxel_encoder(inputs)
        voxel_feat_flatten = voxel_feat.view(batch_size, voxel_feat.shape[1], -1)
        index = top_proposals_index.expand(-1, voxel_feat_flatten.shape[1], -1)
        query_feat_voxel = voxel_feat_flatten.gather(index=index, dim=-1)
        
        # sparse feature fusion
        sparse_fuse_feat_IF = self.defor_cross_attention_sparse_IF(query_dense_fuse_feat_attention, query_feat_img, query_De_sparse)
        sparse_fuse_feat_LF = self.defor_cross_attention_sparse_LF(query_dense_fuse_feat_attention, query_feat_voxel, query_IDe_sparse)
        sparse_fuse_feat_flatten = self.sparse2sparse(torch.cat((sparse_fuse_feat_LF, sparse_fuse_feat_IF), 1))
        sparse_fuse_feat_flatten = self.defor_cross_attention_sparse_FF(query_dense_fuse_feat_attention, sparse_fuse_feat_flatten, None)
        
        # merge
        top_proposals_index1 = top_proposals_index.expand(-1, sparse_fuse_feat_flatten.shape[1], -1)
        dense_fuse_feat_attention_sparse = dense_fuse_feat_attention_flatten.scatter_add(dim=2, index=top_proposals_index1, src=sparse_fuse_feat_flatten).view(batch_size, dense_fuse_feat_attention_flatten.shape[1], 180, 180)

        # Norm
        dense_fuse_feat_attention_sparse = self.norm_a3(dense_fuse_feat_attention_sparse)

        # FFN
        dense_fuse_feat_attention_sparse_ffn = self.FFN_sparse(dense_fuse_feat_attention_sparse)

        # Add & Norm
        dense_fuse_feat_attention_sparse = self.norm_a4(dense_fuse_feat_attention_sparse_ffn + dense_fuse_feat_attention_sparse)

        dense_sparse_heatmap = self.heatmap_sparse_head(dense_fuse_feat_attention_sparse)

        heatmap = dense_sparse_heatmap.detach().sigmoid()

        # generate proposal
        top_proposals_class, top_proposals_index = self.extract_proposal(heatmap)
        self.query_labels = top_proposals_class

        # prepare sparse lidar feat of proposal
        index = top_proposals_index.expand(-1, bev_feat_lidar_flatten.shape[1],
                                           -1)
        query_feat_lidar = bev_feat_lidar_flatten.gather(index=index, dim=-1)

        # add category embedding
        one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(0, 2, 1)
        query_cat_encoding = self.class_encoding(one_hot.float())
        query_feat_lidar += query_cat_encoding

        query_pos_index = top_proposals_index.permute(0, 2, 1)
        query_pos_index = query_pos_index.expand(-1, -1, bev_pos.shape[-1])
        query_pos = bev_pos.gather(index=query_pos_index, dim=1)

        # Prediction
        for task in ['height', 'center', 'dim', 'rot', 'vel']:
            res[task] = \
                self.prediction_heads[0].__getattr__(task)(query_feat_lidar)
        res['center'] += query_pos.permute(0, 2, 1)

        # generate sparse fuse feat
        query_feat_img = self.extract_instance_img_feat(res, img_inputs) # [bs, 128, 128]

        bev_feat_img = bev_feat_img.view(batch_size, bev_feat_img.shape[1], -1) # ori: [bs, 128, 180, 180] --> now: [bs, 128, 180 * 180]
        index = top_proposals_index.expand(-1, bev_feat_img.shape[1], -1)
        query_feat_img_bev = bev_feat_img.gather(index=index, dim=-1)

        # ###################################################################

        query_feat_fuse = torch.cat([query_feat_lidar, query_feat_img, query_feat_img_bev], dim=1) # [bs, 128, 200] 200 -> proposals
        
        query_feat_fuse = self.fuse_convs(query_feat_fuse)
        res['heatmap'] = \
            self.prediction_heads[0].__getattr__('heatmap')(query_feat_fuse)

        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)
        res["query_heatmap_score"] = heatmap.gather(
            index=top_proposals_index.expand(-1,  self.num_classes, -1),
            dim=-1)  
        res["dense_heatmap"] = dense_heatmap

        res["dense_heatmap_sparse"] = dense_sparse_heatmap


        return [res]

    def forward(self, feats, img_metas):
        """Forward pass.
        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.
        Returns:
            tuple(list[dict]): Output results. first index by level, second index by layer
        """
        return [self.forward_single(feats[1][0], feats[0], feats[2][0], img_metas)]