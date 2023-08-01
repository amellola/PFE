"""Module to handle command line arguments and run the program."""
import json
import logging
import os
import tempfile
from typing import Optional

import click
import SimpleITK as sitk
from monai.config import print_config
from polyaxon_client import settings
from polyaxon_client.tracking import Experiment, get_data_paths, get_outputs_path
from torch.utils.tensorboard import SummaryWriter

import wandb
from deepatlas.lib.configs import DeepatlasConfig, SegmentationConfig
from deepatlas.lib.dataloaders import SegmentationDataLoader
from deepatlas.lib.metrics import SegmentationMetrics
from deepatlas.lib.trainers import DeepatlasContext
from deepatlas.lib.transforms import default_transform_single_dataset_3d_us
from deepatlas.utils.logger import setup_logger

log = logging.getLogger(__name__)
@click.group(chain=True)
@click.option("--polyaxon", is_flag=True, help="enable polyaxon functions")
@click.option("-d", "--debug", is_flag=True, help="enable debug mode")
@click.option(
    "--wandb/--no-wandb",
    "wandb_enable",
    is_flag=True,
    default=False,
    help="enable wandb functions",
)
#@click.option("--wandb-key", type=str, default="628cafbacc3fd4644555da8b21a1aed6326b2465", help="wandb key")
@click.option("--wandb-key", type=str, default=None, help="wandb key")
#@click.option("--wandb-project", type=str, default="FULGUR", help="wandb project")
@click.option("--wandb-project", type=str, default=None, help="wandb project")
@click.pass_context


def main(ctx, polyaxon, debug, wandb_enable, wandb_key, wandb_project):  # noqa: D401
    """Implementation of the DeepAtlas approach to segmentation and registration of medical images."""
    sitk.ProcessObject_SetGlobalWarningDisplay(False)

    ctx.obj["POLYAXON"] = polyaxon
    ctx.obj["DEBUG"] = debug
    setup_logger(debug=debug)
    print_config()
    if not wandb_enable:
        wandb.init(mode="disabled")
    else:
        wandb.login(key=wandb_key)
        wandb.init(project=wandb_project, entity="louise-piecuch")
    if not polyaxon:
        os.environ["POLYAXON_NO_OP"] = "true"
        settings.NO_OP = True
        
@main.command()


@click.option(
    "--batch-size",
    type=int,
    default=8,
    metavar="N",
    help="input batch size for training (default: 8)",
)
@click.option(
    "--test-batch-size",
    type=int,
    default=16,
    metavar="N",
    help="input batch size for testing (default: 16)",
)
@click.option(
    "--cache-num",
    type=int,
    default=1,
    metavar="N",
    help="number of images to cache (default: 1)",
)
@click.option(
    "--resize",
    type=int,
    metavar="N",
    help="input resize parameter to save memory",
)
@click.option(
    "--epochs",
    type=int,
    default=60,
    metavar="N",
    help="number of epochs to train (default: 60)",
)
@click.option(
    "--lr",
    type=float,
    default=0.001,
    metavar="LR",
    help="learning rate (default: 0.001)",
)
@click.option(
    "--pretrain-seg/--no-pretrain-seg",
    default=False,
    help="enable pretraining of segmentation",
)
@click.option("--data-dir", type=str, help="data directory")
@click.option("--device", type=str, default="cuda:0", help="device to use")
@click.option("--limit", type=int, help="limit the number of labels to use")
@click.option("--limit-images", type=int, help="limit the number of images to use")
@click.option(
    "--train-context", type=str, help="train context, for format see docs/code"
)
@click.option("--oasis", is_flag=True, help="download and use oasis dataset")
@click.option(
    "--network",
    type=str,
    default="unet",
    help="network to use for segmentation (unet, unetr)",
)
@click.option("--solo-seg", is_flag=True, help="train without deepatlas")
@click.option(
    "--add-cm-loss/--no-add-cm-loss",
    is_flag=True,
    default=False,
    help="use confidence maps for loss",
)
@click.option("--save-interval", type=int, default=5, help="save interval")
@click.option("--aux-model-dir", type=str, help="auxiliary model directory")
@click.option(
    "--size", type=int, default=512, help="size each image will be cropped to"
)
@click.option(
    "--add-cm-ch/--no-add-cm-ch",
    is_flag=True,
    default=False,
    help="add confidence map as second channel",
)
@click.option("--loss", type=str, default="dice", help="loss function to use")
@click.option("--transformer", type=str, default="default", help="transformer to use")
@click.option(
    "--num-seg-classes", type=int, default=7, help="number of segmentation classes"
)
@click.pass_context
def train(
    ctx,
    batch_size,
    test_batch_size,
    cache_num,
    resize,
    epochs,
    lr,
    pretrain_seg,
    data_dir,
    device,
    limit,
    limit_images,
    train_context,
    oasis,
    network,
    solo_seg,
    add_cm_loss,
    save_interval,
    aux_model_dir,
    size,
    add_cm_ch,
    loss,
    transformer,
    num_seg_classes,
):
    """Train the model."""
    on_polyaxon = ctx.obj["POLYAXON"]
    output_dir = "./data/output"
    checkpoint_dir = "./data/checkpoints/"
    num_segmentation_classes = num_seg_classes

    experiment = Experiment()

    if on_polyaxon:
        output_dir = get_outputs_path()
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        input_dir = get_data_paths()["data1"] + "/" + (data_dir or "")  # "/vanessagd/"
        if aux_model_dir:
            aux_model_dir = get_data_paths()["data1"] + "/" + aux_model_dir

        if oasis:
            data_dir = tempfile.mkdtemp()
            input_dir = data_dir
    else:
        input_dir = data_dir
    if not data_dir:
        oasis = True
        data_dir = tempfile.mkdtemp()
    writer = SummaryWriter(log_dir=output_dir)

    ctx.obj["DATA_DIR"] = data_dir if not oasis else data_dir + "/imagesTs/"
    ctx.obj["RESIZE"] = resize
    ctx.obj["MODEL_DIR"] = checkpoint_dir
    ctx.obj["OUTPUT_DIR"] = output_dir
    ctx.obj["DEVICE"] = device
    ctx.obj["WRITER"] = writer
    ctx.obj["EXPERIMENT"] = experiment
    ctx.obj["SOLO_SEG"] = solo_seg
    ctx.obj["NETWORK"] = network
    # ctx.obj["CONF_MAPS"] = conf_maps
    ctx.obj["ADD_CM_CH"] = add_cm_ch
    ctx.obj["ADD_CM_LOSS"] = add_cm_loss
    ctx.obj["SIZE"] = size
    ctx.obj["TRANSFORMER"] = transformer

    log.debug(train_context)

    wandb.config.update(
        {
            "learning_rate": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "test_batch_size": test_batch_size,
            "cache_num": cache_num,
            "resize": resize,
            "pretrain_seg": pretrain_seg,
            "data_dir": data_dir,
            "device": device,
            "limit": limit,
            "limit_images": limit_images,
            "train_context": train_context,
            "oasis": oasis,
            "network": network,
            "solo_seg": solo_seg,
        }
    )

    def _train_seg():
        log.debug("Training segmentation only")
        seg = SegmentationConfig()
        seg.init(
            name="seg",
            device=device,
            num_segmentation_classes=num_segmentation_classes,
            batch_size=batch_size,
            test_batch_size=test_batch_size,
            resize=resize,
            model_dir=checkpoint_dir,
            data_dir=input_dir,
            lr=lr,
            network=network,
            cm_channel=add_cm_ch,
            cm_loss=add_cm_loss,
            size=size,
            transformer=transformer,
        )

        transformer_seg = seg.transformer()
        dataloader_seg = seg.dataloader(limit_imgs=limit_images, limit_label=limit)
        trainer_seg = seg.trainer()

        if aux_model_dir:
            log.info("Loading checkpoint from %s", aux_model_dir)
            trainer_seg.load_checkpoint(path=aux_model_dir)

        dataloader_seg.load(
            cache_num=cache_num,
            batch_size=batch_size,
            test_batch_size=test_batch_size,
            seg_transforms=transformer_seg.transforms(),
        )
        trainer_seg.train(
            epochs=epochs,
            writer=writer,
            experiment=experiment,
            dataloader=dataloader_seg,
            save_interval=save_interval,
            loss=loss,
        )

    def _train_joint():
        joint = DeepatlasConfig()
        joint.init(
            device=device,
            num_segmentation_classes=num_segmentation_classes,
            batch_size=batch_size,
            test_batch_size=test_batch_size,
            resize=resize,
            model_dir=checkpoint_dir,
            additional_model_dir=aux_model_dir,
            output_dir=output_dir,
            data_dir=input_dir,
            lr_seg=lr,
            oasis=oasis,
            cm_channel=add_cm_ch,
            cm_loss=add_cm_loss,
            size=size,
            seg_transformer=transformer,
        )

        dataloader_seg, dataloader_joint = joint.dataloader(limit_imgs=limit_images, limit_label=limit)
        transformer_seg, transformer_joint = joint.transformer()
        trainer_joint = joint.trainer()

        if pretrain_seg:
            seg = SegmentationConfig()
            seg.init(
                device=device,
                num_segmentation_classes=num_segmentation_classes,
                batch_size=batch_size,
                test_batch_size=test_batch_size,
                resize=resize,
                model_dir=checkpoint_dir,
                name="seg_net",
                lr=lr,
                cm_channel=add_cm_ch,
                size=size,
            )
            seg.network = joint.seg_network
            seg_trainer = seg.trainer()
            dataloader_seg.load(
                cache_num=cache_num,
                batch_size=batch_size,
                test_batch_size=test_batch_size,
                seg_transforms=transformer_seg.transforms(),
            )
            seg_trainer.train(
                epochs=epochs,
                writer=writer,
                experiment=experiment,
                dataloader=dataloader_seg,
                pretrain=True,
                save_interval=save_interval,
            )

        dataloader_joint.load(
            cache_num=cache_num + 1
            if cache_num % 2 == 1
            else cache_num,  # cache_num must be even
            transforms_deepatlas=transformer_joint.transforms(),
            load_seg=not pretrain_seg,
            transforms_seg=transformer_seg.transforms() if not pretrain_seg else None,
        )

        args = {
            "max_epochs": epochs if not pretrain_seg else 2 * epochs,
            "writer": writer,
            "experiment": experiment,
            "dataloader": dataloader_joint,
            "pretrained_epochs": epochs if pretrain_seg else 0,
        }

        if train_context:
            context_args = json.loads(train_context)
            context = DeepatlasContext(**context_args)
            args.update({"context": context})

        trainer_joint.train(**args)

    if solo_seg:
        _train_seg()
    else:
        _train_joint()


@main.command()
@click.option("--device", type=str, default="cuda:0", help="device to use")
@click.option("--data-dir", type=str, help="data directory")
@click.option("--output-dir", type=str, help="output directory")
@click.option(
    "--model-dir", type=str, help="If set loads seg_net.pth and reg_net.pth from folder"
)
@click.option("--seg-model", type=str, help="To specify a specific seg-model path.")
@click.option("--reg-model", type=str, help="To specify a specific reg-model path.")
@click.option(
    "--resize",
    type=int,
    metavar="N",
    help="input resize parameter to save memory",
)
@click.option("--metrics", is_flag=True, help="calculate metrics")
@click.option("--save-ram", is_flag=True, help="save ram by not calculating hausdorff")
@click.option(
    "--solo-seg", is_flag=True, default=False, help="only train segmentation network"
)
@click.option("--network", type=str, default="unet", help="network to use")
@click.option(
    "--add-cm-ch/--no-add-cm-ch",
    is_flag=True,
    default=False,
    help="add confidence map as second channel",
)
@click.option(
    "--size", type=int, default=512, help="size each image will be cropped to"
)
@click.option("--transformer", type=str, default="default", help="transformer to use")
@click.option(
    "--num-seg-classes", type=int, default=7, help="number of segmentation classes"
)
@click.option("--batch-size", type=int, default=1, help="batch size")
@click.pass_context
def infer(
    ctx,
    device,
    data_dir,
    output_dir,
    model_dir,
    seg_model,
    reg_model,
    resize,
    metrics,  # pylint: disable=redefined-outer-name
    save_ram,
    solo_seg,
    network,
    add_cm_ch,
    size,
    transformer,
    num_seg_classes,
    batch_size,
):
    """Only run inference on images with pre-trained models.

    Prepare folder imagesTest with labels/original as ground truth.
    """
    resize = resize or ctx.obj.get("RESIZE")
    device = device or ctx.obj.get("DEVICE")
    solo_seg = solo_seg or ctx.obj.get("SOLO_SEG")
    network = ctx.obj.get("NETWORK") or network
    on_polyaxon = ctx.obj.get("POLYAXON")
    add_cm_ch = ctx.obj.get("ADD_CM_CH") or add_cm_ch
    size = ctx.obj.get("SIZE") or size
    experiment = ctx.obj.get("EXPERIMENT") or Experiment()
    num_segmentation_classes = ctx.obj.get("NUM_SEG_CLASSES") or num_seg_classes
    transformer = ctx.obj.get("TRANSFORMER") or transformer
    summary_output_dir = ctx.obj.get("OUTPUT_DIR") or "data/output"
    
    if on_polyaxon:
        output_dir = get_outputs_path()
        summary_output_dir = get_outputs_path()
        data_dir = (
            get_data_paths()["data1"] + "/" + data_dir
            if data_dir
            else ctx.obj.get("DATA_DIR")
        )
        if model_dir:
            model_dir = (
                get_data_paths()["data1"] + "/" + model_dir
                if model_dir
                else ctx.obj.get("MODEL_DIR")
            )
    else:
        data_dir = data_dir or ctx.obj.get("DATA_DIR")

    if not data_dir:
        log.info("No data directory specified.")
        exit(1)

    output_dir = output_dir or ctx.obj.get("OUTPUT_DIR") or f"{data_dir}/labels/pred"
    writer = ctx.obj.get("WRITER") or SummaryWriter(log_dir=summary_output_dir)

    seg_path: Optional[str] = None
    _reg_path: Optional[str] = None
    if model_dir:
        seg_path = f"{model_dir}/final_seg_net.pth"
        _reg_path = f"{model_dir}/final_reg_net.pth"
    if seg_model:
        seg_path = seg_model
    if reg_model:
        _reg_path = reg_model

    if solo_seg:
        seg = SegmentationConfig()
        seg.init(
            device=device,
            num_segmentation_classes=num_segmentation_classes,
            resize=resize,
            network=network,
            data_dir=data_dir,
            cm_channel=add_cm_ch,
            size=size,
            transformer=transformer,
        )
        seg_inferer = seg.infer()
        transformer_seg = seg.transformer()
        print("HHHHHHHHHIIIIIIIIIII", transformer_seg)
        #print('transformer_seg',transformer_seg)
        dataloader_seg = seg.dataloader()
        if seg_path:
            ## load checkpoints
            if on_polyaxon:
                seg_inferer.load_checkpoint(
                    path=get_data_paths()["data1"] + "/" + seg_path
                )
            else:
                seg_inferer.load_checkpoint(path=seg_path)
    else:
        joint = DeepatlasConfig()
        joint.init(
            device=device,
            num_segmentation_classes=num_segmentation_classes,
            data_dir=data_dir,
            cm_channel=add_cm_ch,
            resize=resize,
            size=size,
        )

        inferer = joint.infer()

        dataloader_seg, _ = joint.dataloader()
        transformer_seg, _ = joint.transformer()
        seg_inferer, _ = inferer.inferer()

        if seg_path:
            ## load checkpoints
            seg_inferer.load_checkpoint(path=seg_path)

    seg_metrics = SegmentationMetrics() if metrics else None

    seg_inferer.infer(
        infer=seg_inferer.inferer(),
        dataloader=dataloader_seg,
        transformer=transformer_seg,
        output_dir=output_dir,
        metrics=seg_metrics,
        writer=writer,
        experiment=experiment,
        save_ram=save_ram,
        batch_size=batch_size,
    )


@main.command()
@click.option("--gt-dir", type=str, help="data directory")
@click.option("--pred-dir", type=str, help="data directory")
@click.option("--output-dir", type=str, help="output directory")
@click.option("--save-ram", is_flag=True, help="save ram by not calculating hausdorff")
@click.pass_context
def metrics(
    ctx,
    gt_dir,
    pred_dir,
    output_dir,
    save_ram,
):
    """Calc metrics from ground truth and predictions."""
    on_polyaxon = ctx.obj["POLYAXON"]
    if on_polyaxon:
        output_dir = get_outputs_path()
        data_dir = get_data_paths()["data1"]
        gt_dir = f"{data_dir}/{gt_dir}" if gt_dir else f"{data_dir}/labels/final"
        log.info("getting ground truth from: %s", gt_dir)
        log.info("getting predictions from: %s", pred_dir)
        log.info("writing metrics to: %s", output_dir)
        experiment = ctx.obj.get("EXPERIMENT") or Experiment()
    else:
        experiment = None

    output_dir = output_dir or ctx.obj.get("OUTPUT_DIR")
    writer = ctx.obj.get("WRITER") or SummaryWriter(log_dir=output_dir)

    joint = DeepatlasConfig()
    joint.init(num_segmentation_classes=7)

    dataloader_seg, _ = joint.dataloader()

    seg_metrics = SegmentationMetrics()

    for data in dataloader_seg.load_gt_and_pred(gt_dir, pred_dir):
        name = data["pred"].meta["filename_or_obj"][0].split("/")[-1]
        seg_metrics.calc_metrics(
            data,
            writer,
            experiment,
            name=name,
            _save_ram=save_ram,
            csv_dir=output_dir,
        )


@main.command()
@click.option("--data-dir", type=str, help="data directory")
@click.option("--output-dir", type=str, help="output directory")
@click.option(
    "--resize", type=int, metavar="N", help="input resize parameter to save memory"
)
@click.pass_context
def transform(
    _ctx,
    data_dir,
    output_dir,
    resize,
):
    """Only transform images and labels and save."""
    dataloader = SegmentationDataLoader(data_dir=data_dir)
    dataloader.load_transform_save(
        data=data_dir,
        transforms=default_transform_single_dataset_3d_us,
        resize=resize,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main(obj={})  # pylint: disable=no-value-for-parameter
