"""
Main file for training Yolo model on Pascal VOC and COCO dataset
"""

import detect_config
import torch
import torch.optim as optim

from detect_model import YOLOv3
from tqdm import tqdm
from detect_utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples
)
from detect_loss import YoloLoss
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True

#################################Detection Part Start##########################################
def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    loop = tqdm(train_loader, leave=True)
    losses = []
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(detect_config.DEVICE)
        y0, y1, y2 = (
            y[0].to(detect_config.DEVICE),
            y[1].to(detect_config.DEVICE),
            y[2].to(detect_config.DEVICE),
        )

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)

    return mean_loss


model = YOLOv3(num_classes=detect_config.NUM_CLASSES).to(detect_config.DEVICE)
optimizer = optim.Adam(
    model.parameters(), lr=detect_config.LEARNING_RATE, weight_decay=detect_config.WEIGHT_DECAY
)
loss_fn = YoloLoss()
train_loader, test_loader, train_eval_loader = get_loaders(
    train_csv_path=detect_config.DATASET + "/train.csv", test_csv_path=detect_config.DATASET + "/test.csv"
)

if detect_config.LOAD_MODEL:
    load_checkpoint(
        detect_config.CHECKPOINT_FILE, model, optimizer, detect_config.LEARNING_RATE
    )


scaled_anchors = (
    torch.tensor(detect_config.ANCHORS)
    * torch.tensor(detect_config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
).to(detect_config.DEVICE)

scaler = torch.cuda.amp.GradScaler()
detect_loss=0
##########################################Detection Part End#########################################


for epoch in range(detect_config.NUM_EPOCHS):
   
    #plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)
    detect_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)
    
    
    if detect_config.SAVE_MODEL:
        save_checkpoint(model, optimizer, filename=f"checkpoint.pth.tar")
    #print(f"Currently epoch {epoch}")
    #print("On Train Eval loader:")
    #print("On Train loader:")
    #check_class_accuracy(model, train_loader, threshold=detect_config.CONF_THRESHOLD)
    if epoch > 0 and epoch % 6 == 0:
        check_class_accuracy(model, test_loader, threshold=detect_config.CONF_THRESHOLD)
        pred_boxes, true_boxes = get_evaluation_bboxes(
            test_loader,
            model,
            iou_threshold=detect_config.NMS_IOU_THRESH,
            anchors=detect_config.ANCHORS,
            threshold=detect_config.CONF_THRESHOLD,
        )
        mapval = mean_average_precision(
            pred_boxes,
            true_boxes,
            iou_threshold=detect_config.MAP_IOU_THRESH,
            box_format="midpoint",
            num_classes=detect_config.NUM_CLASSES,
        )
        print(f"MAP: {mapval.item()}")
        model.train()

  
