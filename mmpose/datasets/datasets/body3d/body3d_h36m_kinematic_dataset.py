from typing import OrderedDict
from .body3d_h36m_dataset import Body3DH36MDataset
from ...builder import DATASETS
from ....models.losses.regression_loss import L1Loss
from collections import OrderedDict
import torch as T
@DATASETS.register_module()
class Body3DH36MKinematicDataset(Body3DH36MDataset):
    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 test_mode=False):
        super().__init__(
            ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info,
                 test_mode
        )
    
    KINEMATIC_METRIC = ['angle_l1']
    def evaluate(self, results, res_folder=None, metric='mpjpe',  **kwargs):
        metrics = metric if isinstance(metric, list) else [metric]
        h36_metrics = []
        kinematic_metrics = []
        for _metric in metrics:
            if _metric in super().ALLOWED_METRICS:
                h36_metrics.append(_metric)
            elif _metric in self.KINEMATIC_METRIC:
                kinematic_metrics.append(_metric)
            else:
                raise ValueError(
                    f'Unsupported metric "{_metric}" for human3.6 dataset (kinematic mode).')
        # Do evaluation
        eval_res = OrderedDict()
        # Compute eval results at H36M Dataset task
        if len(h36_metrics) > 0:
            eval_res =  super().evaluate(results, res_folder, h36_metrics)
            criteria = L1Loss()
        
        if len(kinematic_metrics) > 0:
            l1_loss  = 0.0
            for i, result in enumerate(results):
                preds = result['preds_dirc_vector']
                gts   = result['gt_dirc_vector']
                l1_loss += criteria(preds, gts).item()
            l1_loss = l1_loss/len(results)
            eval_res["angle_l1"] = l1_loss
        return eval_res