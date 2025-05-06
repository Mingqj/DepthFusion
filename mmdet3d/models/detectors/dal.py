import torch
from .bevdet import BEVDet
from mmdet.models import DETECTORS
from mmdet3d.models.utils import FFN
from mmdet3d.models.utils.spconv_voxelize import SPConvVoxelization
from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d


@DETECTORS.register_module()
class DAL(BEVDet):
    def __init__(self, **kwargs):
        super(DAL, self).__init__(**kwargs)

        # image view auxiliary task heads
        self.num_cls = self.pts_bbox_head.num_classes
        heads = dict(heatmap=(self.num_cls, 2))
        input_feat_dim = kwargs['pts_bbox_head']['hidden_channel']
        self.auxiliary_heads = FFN(
                input_feat_dim,
                heads,
                conv_cfg=dict(type="Conv1d"),
                norm_cfg=dict(type="BN1d"),
                bias=True)
        self.auxiliary_heads.init_weights()

        pts_voxel_cfg = kwargs.get('pts_voxel_layer', None)
        if pts_voxel_cfg:
            pts_voxel_cfg['num_point_features'] = 5
            self.pts_voxel_layer = SPConvVoxelization(**pts_voxel_cfg)

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        img = self.prepare_inputs(img)
        x, _  = self.image_encoder(img[0])
        return [x] + img[1:]

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        return (img_feats, pts_feats)
    
    def extract_feat1(self, points, img, img_metas):
        """Extract features from images and points."""
        results = []
        for i in range(len(points)):
            img_feats = self.extract_img_feat(img[i], img_metas[i])
            pts_feats = self.extract_pts_feat(points[i], img_feats, img_metas[i])
            results.append((img_feats, pts_feats))
        return results


    def forward_img_auxiliary_train(self,
                          x,
                          img_metas,
                          gt_bboxes,
                          gt_labels,
                          gt_bboxes_ignore=None,
                          proposals=None,
                          **kwargs):
        max_instance = 150
        num_pos = 0
        centers_augego = x[0].new_zeros((len(gt_bboxes), max_instance, 3))
        box_targets_all = x[0].new_zeros((len(gt_bboxes), max_instance, 10))
        valid_mask = x[0].new_zeros((len(gt_bboxes), max_instance, 1))
        label = x[0].new_zeros((len(gt_bboxes), max_instance, 1)).to(torch.long)
        for sid in range(len(gt_bboxes)):
            centers_augego_tmp = gt_bboxes[sid].gravity_center.to(x[0])
            box_targets_tmp = self.pts_bbox_head.bbox_coder.encode(gt_bboxes[sid].tensor)
            if gt_bboxes_ignore is not None:
                centers_augego_tmp = centers_augego_tmp[gt_bboxes_ignore[sid], :]
                box_targets_tmp = box_targets_tmp[gt_bboxes_ignore[sid], :]
            num_valid_samples = centers_augego_tmp.shape[0]
            num_pos += num_valid_samples
            valid_mask[sid, :num_valid_samples, :] = 1.0
            centers_augego[sid,:num_valid_samples,:] = centers_augego_tmp
            box_targets_all[sid,:num_valid_samples,:] = box_targets_tmp
            label_tmp = gt_labels[sid].unsqueeze(-1)
            if gt_bboxes_ignore is not None:
                label_tmp = label_tmp[gt_bboxes_ignore[sid], :]
            label[sid,:num_valid_samples,:] = label_tmp
        img_feats = self.pts_bbox_head.extract_img_feat_from_3dpoints(
            centers_augego, x, fuse=False)
        heatmap = self.auxiliary_heads.heatmap(img_feats)
        loss_cls_img = self.pts_bbox_head.loss_cls(
            heatmap.permute(0, 2, 1).reshape(-1, self.num_cls),
            label.flatten(),
            valid_mask.flatten(),
            avg_factor=max(num_pos, 1))
        return dict(loss_cls_img=loss_cls_img)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas)
        img_feats_bev = \
            self.img_view_transformer(img_feats + img_inputs[1:7],
                                      depth_from_lidar=kwargs['gt_depth'])

        losses = dict()
        losses_pts = \
            self.forward_pts_train([img_feats, pts_feats, img_feats_bev],
                                   gt_bboxes_3d, gt_labels_3d, img_metas,
                                   gt_bboxes_ignore)
        losses.update(losses_pts)
        losses_img_auxiliary = \
            self.forward_img_auxiliary_train(img_feats,img_metas,
                                             gt_bboxes_3d, gt_labels_3d,
                                             gt_bboxes_ignore,
                                             **kwargs)
        losses.update(losses_img_auxiliary)
        return losses

    def simple_test(self,
                    points,
                    img_metas,
                    img_inputs=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, pts_feats = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas)
        img_feats_bev = \
            self.img_view_transformer(img_feats + img_inputs[1:7],
                                      depth_from_lidar=kwargs['gt_depth'][0])

        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts([img_feats, pts_feats, img_feats_bev],
                                        img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox

        return bbox_list

        # bbox_list = dict()
        # if pts_feats and self.with_pts_bbox:
        #     pts_bbox = self.aug_test_pts2(img_feats, pts_feats, img_feats_bev, img_metas, rescale)
        #     bbox_list.update(pts_bbox=pts_bbox)
        # return [bbox_list]
    
    def aug_test(self,
                    points,
                    img_metas,
                    img_inputs=None,
                    rescale=False,
                    **kwargs):
        results = self.extract_feat1(
            points, img=img_inputs, img_metas=img_metas)
        
        img_feats_bev_list = []
        for i in range(len(img_inputs)):
            img_feats, pts_feats = results[i]
            img_feats_bev = \
                self.img_view_transformer(img_feats + img_inputs[i][1:7],
                                        depth_from_lidar=kwargs['gt_depth'][0])
            img_feats_bev_list.append(img_feats_bev)

        bbox_list = dict()
        pts_bbox = self.aug_test_pts2(results, img_feats_bev_list, img_metas, rescale)
        bbox_list.update(pts_bbox=pts_bbox)
        return [bbox_list]
    
    def aug_test_pts1(self, feats, img_metas, rescale=False):
        """Test function of point cloud branch with augmentaiton.

        The function implementation process is as follows:

            - step 1: map features back for double-flip augmentation.
            - step 2: merge all features and generate boxes.
            - step 3: map boxes back for scale augmentation.
            - step 4: merge results.

        Args:
            feats (list[torch.Tensor]): Feature of point cloud.
            img_metas (list[dict]): Meta information of samples.
            rescale (bool, optional): Whether to rescale bboxes.
                Default: False.

        Returns:
            dict: Returned bboxes consists of the following keys:

                - boxes_3d (:obj:`LiDARInstance3DBoxes`): Predicted bboxes.
                - scores_3d (torch.Tensor): Scores of predicted boxes.
                - labels_3d (torch.Tensor): Labels of predicted boxes.
        """
        # only support aug_test for one sample
        outs_list = []
        for img_meta in zip(img_metas):
            outs = self.pts_bbox_head(feats)
            # merge augmented outputs before decoding bboxes
            for task_id, out in enumerate(outs):
                for key in out[0].keys():
                    if img_meta[0]['pcd_horizontal_flip']:
                        outs[task_id][0][key] = torch.flip(
                            outs[task_id][0][key], dims=[2])
                        if key == 'reg':
                            outs[task_id][0][key][:, 1, ...] = 1 - outs[
                                task_id][0][key][:, 1, ...]
                        elif key == 'rot':
                            outs[task_id][0][
                                key][:, 0,
                                     ...] = -outs[task_id][0][key][:, 0, ...]
                        elif key == 'vel':
                            outs[task_id][0][
                                key][:, 1,
                                     ...] = -outs[task_id][0][key][:, 1, ...]
                    if img_meta[0]['pcd_vertical_flip']:
                        outs[task_id][0][key] = torch.flip(
                            outs[task_id][0][key], dims=[3])
                        if key == 'reg':
                            outs[task_id][0][key][:, 0, ...] = 1 - outs[
                                task_id][0][key][:, 0, ...]
                        elif key == 'rot':
                            outs[task_id][0][
                                key][:, 1,
                                     ...] = -outs[task_id][0][key][:, 1, ...]
                        elif key == 'vel':
                            outs[task_id][0][
                                key][:, 0,
                                     ...] = -outs[task_id][0][key][:, 0, ...]

            outs_list.append(outs)

        preds_dicts = dict()
        scale_img_metas = []

        # concat outputs sharing the same pcd_scale_factor
        for i, (img_meta, outs) in enumerate(zip(img_metas, outs_list)):
            pcd_scale_factor = img_meta['pcd_scale_factor']
            if pcd_scale_factor not in preds_dicts.keys():
                preds_dicts[pcd_scale_factor] = outs
                scale_img_metas.append(img_meta)
            else:
                for task_id, out in enumerate(outs):
                    for key in out[0].keys():
                        preds_dicts[pcd_scale_factor][task_id][0][key] += out[
                            0][key]

        aug_bboxes = []

        for pcd_scale_factor, preds_dict in preds_dicts.items():
            for task_id, pred_dict in enumerate(preds_dict):
                # merge outputs with different flips before decoding bboxes
                for key in pred_dict[0].keys():
                    preds_dict[task_id][0][key] /= len(outs_list) / len(
                        preds_dicts.keys())
            bbox_list = self.pts_bbox_head.get_bboxes(
                preds_dict, img_metas, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])
        
        # print(preds_dicts.keys())
        if len(preds_dicts.keys()) > 1:
            # merge outputs with different scales after decoding bboxes
            merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, scale_img_metas,
                                                self.pts_bbox_head.test_cfg)
            return merged_bboxes
        else:
            for key in bbox_list[0].keys():
                bbox_list[0][key] = bbox_list[0][key].to('cpu')
            return bbox_list[0]

    def aug_test_pts2(self, results, img_feats_bev_list, img_metas, rescale=False):
        """Test function of point cloud branch with augmentaiton."""
        # only support aug_test for one sample
        aug_bboxes = []
        for _idx, img_meta in enumerate(img_metas):
            img_feats, pts_feats = results[_idx]
            img_feats_bev = img_feats_bev_list[_idx]
            outs = self.pts_bbox_head(
                [img_feats, pts_feats, img_feats_bev], img_metas
            )

            bbox_list = self.pts_bbox_head.get_bboxes(outs, img_meta, rescale=rescale)

            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(
            aug_bboxes, img_metas, self.pts_bbox_head.test_cfg
        )
        return merged_bboxes