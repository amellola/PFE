"""Results of the Thyroid experiment."""
import csv
import json
import os
import re
import sys
from collections import OrderedDict
from typing import List, cast

import matplotlib.pyplot as plt
import numpy as np

data = {}  # type: ignore
pattern_names = re.compile(r"INFO segmentation ((\w+).(nii|mha))")
pattern_metrics = re.compile(
    r"(\[(\{('[a-zA-Z0-9_]+': (nan|inf|[0-9]+.[0-9e\-\+]+),? ?)+\}\n? ?)+\])",
    re.MULTILINE,
)
pattern_line = re.compile(r"(\{('(\w+)': (\d+.[\de\+\-]+),? ?)+\})")
pattern_metric_names = re.compile(r"'([a-zA-Z0-9_]+)': (nan|inf|[0-9]+.[0-9e\-\+]+)")

DIRECTORY = sys.argv[1]


def catch_inf_and_nan(arg):
    """Catch inf and nan."""
    print("got: ", arg)
    c = {"inf": -float("inf"), "nan": float("nan")}
    return c[arg]


def get_metrics_from_log(filename):
    """Get metrics from log file."""
    with open(DIRECTORY + "/" + filename, mode="rt", encoding="utf-8") as log_file:
        text = log_file.read()
        regex = re.findall(pattern_metrics, text)
        _img_names = re.findall(pattern_names, text)
        line = re.search(pattern_line, text)

        _metric_names = []
        if line:
            _metric_names = [
                match[0] for match in re.findall(pattern_metric_names, line.group(0))
            ]

        _img_names = [name[1] for name in _img_names]

        compiled_metrics = [
            json.loads(
                match[0]
                .replace("'", '"')
                .replace("\n ", ",")
                .replace("nan", "NaN")
                .replace("inf", "Infinity"),
            )
            for match in regex
        ]
        compiled_metrics = {
            name: metric for name, metric in zip(_img_names, compiled_metrics)
        }
        data[filename.replace(".log", "")] = compiled_metrics
        return _img_names, _metric_names


imgs = None
metrics = None

print(DIRECTORY)
files = [f for f in os.listdir(DIRECTORY) if os.path.isfile(DIRECTORY + "/" + f)]
print(files)
for f in files:
    if f.endswith(".log"):
        img_names, metric_names = get_metrics_from_log(f)
        if imgs is None:
            imgs = img_names
        else:
            if set(imgs) != set(img_names):
                raise ValueError(f"Image names in {f} do not match those in {files[0]}")
        if metrics is None:
            metrics = metric_names
        else:
            if set(metrics) != set(metric_names):
                raise ValueError(
                    f"Metric names in {f} do not match those in {files[0]}"
                )


def sort_by_name(elem):
    """Sort by name."""
    name = elem[0]
    channel: int = 0
    loss: int = 0
    conf: int = 0

    if "1ch" in name:
        channel = 1
    elif "2ch" in name:
        channel = 2

    if "diceCE" in name:
        loss = 3
    elif "dice" in name:
        loss = 1
    elif "CE" in name:
        loss = 2

    if "conf" in name:
        conf = 1

    return conf, channel, loss, name


data = OrderedDict(sorted(data.items(), key=sort_by_name))

imgs = cast(List[str], imgs)
metrics = cast(List[str], metrics)

labels = list(range(len(data[list(data.keys())[0]][imgs[0]])))

print(imgs)
print(labels)
print(metrics)

for img in imgs:
    with open(f"{DIRECTORY}/{img}.csv", "w", encoding="utf8") as csv_file:
        writer = csv.writer(
            csv_file,
            delimiter=";",
        )
        writer.writerow([" "])
        writer.writerow([f"{img}"])
        for label in labels:
            writer.writerow([f"label {label}"])
            writer.writerow(["\\"] + list(data.keys()))
            for metric in metrics:
                writer.writerow(
                    [metric]
                    + [
                        str(data[network][img][label][metric]).replace(".", ",")
                        for network in data.keys()
                    ]
                )
            writer.writerow([" "])

# Use matplotlib to create boxplots, one graph per metric with one box per experiment
for metric in metrics:
    print(f"Creating boxplot for metric {metric}")

    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.3)

    metric_data = [
        [cast(float, d[img][l][metric]) for img in imgs for l in labels]
        for d in data.values()
    ]

    bp = ax.boxplot(metric_data, notch=False, sym="+", vert=True, whis=1.5)
    plt.setp(bp["boxes"], color="black")
    plt.setp(bp["whiskers"], color="black")
    plt.setp(bp["fliers"], color="red", marker="+")

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)

    ax.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title=metric,
        xlabel="Experiment",
        ylabel=metric,
    )

    num_boxes = len(data)

    ax.set_xticklabels(data.keys(), rotation=45, fontsize=8)
    ax.set_xticks(range(1, len(data.keys()) + 1))

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    pos = np.arange(num_boxes) + 1

    medians = np.empty(num_boxes)

    for i in range(num_boxes):
        med = bp["medians"][i]
        for j in range(2):
            medians[i] = med.get_ydata()[j]

    upper_labels = [str(round(s, 2)) for s in medians]
    weights = ["bold", "semibold"]
    for tick, label in zip(range(num_boxes), ax.get_xticklabels()):
        k = tick % 2
        ax.text(
            pos[tick],
            0.97,
            upper_labels[tick],
            transform=ax.get_xaxis_transform(),
            horizontalalignment="center",
            size="x-small",
            weight=weights[k],
            color="royalblue",
        )

    plt.savefig(f"{DIRECTORY}/{metric}.png")
    plt.close()
