import csv
import logging
import os
from typing import Any, Dict, List

import monai.metrics as metrics
import numpy as np
import torch
from monai.handlers import from_engine

import wandb

log = logging.getLogger(__name__)


class Segmentation:
    def monai_metrics_after_post_transform(self, y_pred, y, reduction="none"):

        red = "none"

        if reduction == "mean":
            red = "mean_batch"
        elif reduction == "std":
            red = "none"

        dice = metrics.DiceMetric(include_background=False, reduction=red)
        iou = metrics.MeanIoU(include_background=False, reduction=red)
        hausdorff = metrics.HausdorffDistanceMetric(
            include_background=False, reduction=red
        )
        avg_surface_distance = metrics.SurfaceDistanceMetric(
            include_background=False, reduction=red
        )
        confusion = metrics.ConfusionMatrixMetric(
            include_background=False,
            metric_name=(
                "precision",
                "recall",
                "miss rate",
                "fall out",
                "sensitivity",
                "specificity",
            ),
            reduction=red,
            compute_sample=True,
        )
        auc = metrics.ROCAUCMetric()

        batch = type(y_pred) is list
        y_pred = y_pred if type(y_pred) is list else [y_pred]
        y = y if type(y) is list else [y]

        dice = dice(y_pred, y)
        print("dice", dice)
        iou = iou(y_pred, y)
        print("iou", iou)
        # auc = auc(y_pred, y)
        # print("auc", auc)

        hausdorff = hausdorff(y_pred, y)
        print("hausdorff", hausdorff)
        avg_surface_distance = avg_surface_distance(y_pred, y)
        print("avg_surface_distance", avg_surface_distance)
        _conf_matrix = confusion(y_pred, y)

        conf_metrixs = confusion.aggregate()
        precision = conf_metrixs[0]
        recall = conf_metrixs[1]
        miss_rate = conf_metrixs[2]
        fall_out = conf_metrixs[3]
        sensitivity = conf_metrixs[4]
        specificity = conf_metrixs[5]

        print("precision", precision)
        print("recall", recall)
        print("miss_rate", miss_rate)
        print("fall_out", fall_out)
        print("sensitivity", sensitivity)
        print("specificity", specificity)

        ret = []
        for i in range(y_pred[0].shape[0] - 1 if y_pred[0].shape[0] > 1 else 1):

            if batch:
                fun = lambda x: x
                if reduction == "mean":
                    fun = torch.mean
                elif reduction == "std":
                    fun = torch.std
                if len(precision.shape) == 1:
                    ret.append(
                        {
                            "dice": fun(dice[:, i]).item(),
                            "iou": fun(iou[:, i]).item(),
                            "hausdorff": fun(hausdorff[:, i]).item(),
                            "avg_surface_distance": fun(
                                avg_surface_distance[:, i]
                            ).item(),
                        }
                    )
                    if reduction == "mean":
                        ret[i].update(
                            {
                                "precision": fun(precision[i]).item(),
                                "recall": fun(recall[i]).item(),
                                "miss_rate": fun(miss_rate[i]).item(),
                                "fall_out": fun(fall_out[i]).item(),
                                "sensitivity": fun(sensitivity[i]).item(),
                                "specificity": fun(specificity[i]).item(),
                            }
                        )
                    elif reduction == "std":
                        ret[i].update(
                            {
                                "precision": fun(precision[:, i]).item(),
                                "recall": fun(recall[:, i]).item(),
                                "miss_rate": fun(miss_rate[:, i]).item(),
                                "fall_out": fun(fall_out[:, i]).item(),
                                "sensitivity": fun(sensitivity[:, i]).item(),
                                "specificity": fun(specificity[:, i]).item(),
                            }
                        )
                else:
                    ret.append(
                        {
                            "dice": fun(dice[:, i]).item(),
                            "iou": fun(iou[:, i]).item(),
                            "hausdorff": fun(hausdorff[:, i]).item(),
                            "avg_surface_distance": fun(
                                avg_surface_distance[:, i]
                            ).item(),
                            "precision": fun(precision[:, i]).item(),
                            "recall": fun(recall[:, i]).item(),
                            "miss_rate": fun(miss_rate[:, i]).item(),
                            "fall_out": fun(fall_out[:, i]).item(),
                            "sensitivity": fun(sensitivity[:, i]).item(),
                            "specificity": fun(specificity[:, i]).item(),
                        }
                    )
            else:
                ret.append(
                    {
                        "dice": dice[0][i].item(),
                        "iou": iou[0][i].item(),
                        "hausdorff": hausdorff[0][i].item(),
                        "avg_surface_distance": avg_surface_distance[0][i].item(),
                        "precision": precision[0][i].item(),
                        "recall": recall[0][i].item(),
                        "miss_rate": miss_rate[0][i].item(),
                        "fall_out": fall_out[0][i].item(),
                        "sensitivity": sensitivity[0][i].item(),
                        "specificity": specificity[0][i].item(),
                    }
                )

        return ret

    def log_metrics(
        self,
        computed_metrics: List[Dict[str, Any]],
        writer,
        experiment,
        epoch=None,
        name="Metrics",
    ):
        """Log metrics."""
        log.info("%s: \n %s", name, computed_metrics)
        to_log = {}
        for label, metric in enumerate(computed_metrics):
            for key, value in metric.items():
                if epoch is not None:
                    writer.add_scalar(f"{name}/{key}_{label}", value, global_step=epoch)
                else:
                    writer.add_scalar(f"{name}/{key}_{label}", value)
                    to_log[f"{name}/{key}_{label}"] = value
        if epoch is not None:
            wandb.log(to_log, step=epoch)  # type: ignore
            experiment.log_metrics(**to_log, step=epoch)
        else:
            wandb.log(to_log)  # type: ignore
            experiment.log_metrics(**to_log)

    def calc_metrics(
        self,
        data,
        writer,
        experiment=None,
        epoch=None,
        name="Metrics",
        _save_ram=False,
        csv_dir="./dataLoe/output/",
        reduction=None,
    ):
        """Calculate metrics."""
        computed_metrics = self.monai_metrics_after_post_transform(
            *from_engine(["pred", "seg"])(data), reduction=reduction
        )
        computed_metrics = np.asarray(computed_metrics)
        self.log_metrics(computed_metrics, writer, experiment, epoch, name)
        if csv_dir:
            self.write_to_csv(computed_metrics, csv_dir)

    def write_to_csv(self, computed_metrics, csv_dir):
        """Write metrics to csv."""
        import csv
        import os

        header = computed_metrics[0].keys()
        data = [m.values() for m in computed_metrics]

        csv_path = os.path.join(csv_dir, "metrics.csv")
        with open(csv_path, "w", encoding="UTF8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data)
