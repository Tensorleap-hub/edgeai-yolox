from typing import List

import torch
import torch.nn as nn
from yolox.utils import PostprocessExport

class OnnxDetWithPredictions(nn.Module):
    """Wraps model + postprocess to also emit pre-NMS predictions."""

    def __init__(self, model, post_process):
        super().__init__()
        self.model = model
        self.post_process = post_process

    def forward(self, x):
        preds = self.model(x)
        dets = self.post_process(preds)
        return dets, preds

class OnnxHeadRawOutputs(nn.Module):
    """
    Wraps the model to return raw head outputs (per scale) before decode/NMS.
    Outputs per-level concatenated head tensors: reg|obj|cls along channel dim.
    Returns head_out0, head_out1, head_out2, strides.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Forward through backbone/neck
        fpn_outs = self.model.backbone(x)
        head = self.model.head
        head_outs = []
        for k, (cls_conv, reg_conv, stride_this_level, x_lvl) in enumerate(
            zip(head.cls_convs, head.reg_convs, head.strides, fpn_outs)
        ):
            x_feat = head.stems[k](x_lvl)
            cls_feat = cls_conv(x_feat)
            reg_feat = reg_conv(x_feat)
            cls_output = head.cls_preds[k](cls_feat)
            reg_output = head.reg_preds[k](reg_feat)
            obj_output = head.obj_preds[k](reg_feat)
            output = torch.cat([reg_output, obj_output, cls_output], 1)
            head_outs.append(output)
        return head_outs[0], head_outs[1], head_outs[2]


class OnnxHeadRawWithDet(nn.Module):
    """
    Return raw head per-level outputs plus post-NMS detections.
    Outputs: head_out0, head_out1, head_out2, strides, detections
    """

    def __init__(self, model, post_process=None):
        super().__init__()
        self.model = model
        self.post_process = post_process

    def forward(self, x):

        # Produce per-level raw head outputs
        fpn_outs = self.model.backbone(x)
        head = self.model.head
        head_outs = []
        for k, (cls_conv, reg_conv, stride_this_level, x_lvl) in enumerate(
            zip(head.cls_convs, head.reg_convs, head.strides, fpn_outs)
        ):
            x_ = head.stems[k](x_lvl)
            cls_feat = cls_conv(x_)
            reg_feat = reg_conv(x_)
            cls_output = head.cls_preds[k](cls_feat)
            reg_output = head.reg_preds[k](reg_feat)
            obj_output = head.obj_preds[k](reg_feat)
            output = torch.cat([reg_output, obj_output, cls_output], 1)
            head_outs.append(output)
        strides = torch.tensor(head.strides, dtype=torch.float32)

        # Get detections using the full model forward (flattened raw preds)
        self.model.head.decode_in_inference = True
        preds = self.model(x)

        if self.post_process is not None:
            preds = self.post_process(preds)
            preds = preds[0].unsqueeze(0)

        return preds, head_outs[0], head_outs[1], head_outs[2]


class OnnxHeadRawWithSnippetDet(nn.Module):
    """
    Return raw head per-level outputs plus post-NMS detections.
    Outputs: head_out0, head_out1, head_out2, strides, detections
    """

    def __init__(self, model, post_process=None, snippet:List[int] = None):
        super().__init__()
        self.model = model
        self.post_process = post_process
        if snippet is not None:
            snippet = [int(5+snip) for snip in snippet]
        self.snippet = snippet
        print(snippet)

    def forward(self, x):

        # Produce per-level raw head outputs
        fpn_outs = self.model.backbone(x)
        head = self.model.head
        head_outs = []
        for k, (cls_conv, reg_conv, stride_this_level, x_lvl) in enumerate(
            zip(head.cls_convs, head.reg_convs, head.strides, fpn_outs)
        ):
            x_ = head.stems[k](x_lvl)
            cls_feat = cls_conv(x_)
            reg_feat = reg_conv(x_)
            cls_output = head.cls_preds[k](cls_feat)
            reg_output = head.reg_preds[k](reg_feat)
            obj_output = head.obj_preds[k](reg_feat)
            output = torch.cat([reg_output, obj_output, cls_output], 1)
            head_outs.append(output)
        strides = torch.tensor(head.strides, dtype=torch.float32)

        # Get detections using the full model forward (flattened raw preds)
        self.model.head.decode_in_inference = True
        preds = self.model(x)

        if self.snippet is not None:
            channels_to_keep = [0,1,2,3,4, *self.snippet]
            preds = preds[:,:, channels_to_keep]
            head_outs[0] = head_outs[0][:, channels_to_keep, :, :]
            head_outs[1] = head_outs[1][:, channels_to_keep, :, :]
            head_outs[2] = head_outs[2][:, channels_to_keep, :, :]
        if self.post_process is not None:
            preds = self.post_process(preds)
            preds = preds[0].unsqueeze(0)



        return preds, head_outs[0], head_outs[1], head_outs[2]

