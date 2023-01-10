
"""Model evaluation and performance metric calculation functions."""

import numpy as np
from keras import backend as K


def pos_IoU(y_true, y_pred):
    """"Computes positive intersection over union metric.
        
    Args:
        y_true: groundtruth labels
        y_pred: predictions
    """

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    true_positives = K.sum(y_true_f * y_pred_f)
    ground_truth_positives = K.sum(y_true_f) 
    predicted_positives = K.sum(y_pred_f)

    recall = true_positives / (ground_truth_positives + K.epsilon())
    precision = true_positives / (predicted_positives + K.epsilon())

    positive_IoU = 1 / (1/precision + 1/recall - 1)

    return positive_IoU


def get_IoU(pd, gt):
    """"Computes overall intersection over union metric.
        
    Args:
        pd: predicted labels
        gt: groundtruth labels
    """

    assert(pd.ndim==3)
    i = (pd==1) & (gt==1)
    u = (pd==1) | (gt==1)
    iou = np.sum(i) / np.sum(u)
    return iou

def evaluate_model_iou(imgs, lbls, indexes, model, depth, thresholds=np.linspace(0, 1, 11)):
    """"Computes overall IoU metric results and mean IoUs at threshold points.
        
    Args:
        imgs: CT scans
        lbls: corresponding groundtruth labels
        indexes: indexes of CT scans that the trained model evaluation
        model: the trained U-Net model
        depth: the depth defined for CT scan slices
        thresholds: the threshold value in which model is evaluated
    """
    
    delta = (depth-1)//2
    ious = {th: [] for th in thresholds}
    val_size = len(indexes)
    for step, idx in enumerate(indexes):
        print(f"Step {step+1}/{val_size}")
        
        img = imgs[idx]
        gt = lbls[idx]
        pd = np.empty(gt.shape)
        pd_channels = pd.shape[-1]
        
        print("_" * pd_channels) # show the length of the progress bar
        for channel in range(pd_channels):
            print("#", end="")
            channels = np.arange(channel-delta, channel+delta+1).clip(0, pd_channels-1)
            pd[:,:,channel] = model.predict(img[None,:,:,channels])[0,:,:,0]
        print()
        
        for th in thresholds:
            iou = get_IoU(pd >= th, gt)
            ious[th].append(iou)
            print(f"IoU at threshold {th} = {iou}")
        print()
        
    mean_iou = {}
    for th in thresholds:
        th_mean_iou = np.mean(ious[th])
        mean_iou[th] = th_mean_iou
        print(f"Mean IoU at threshold {th} = {th_mean_iou}")
    mean_iou = {th: np.mean(th_ious) for th, th_ious in ious.items()}
    
    return ious, mean_iou

