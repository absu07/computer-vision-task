import onnxruntime as ort
import numpy as np
from .functions import distance2bbox

class OnnxRunner:

    def __init__(self, onnx_path:str,
                       feat_stride_fpn:list=[8,16,32],
                       fmc:int=3,
                       num_anchors:int=2,
                       nms_thresh:float=0.4,
                       conf_thresh:float=0.5):
        """
        This class is an OnnxRuntime wrapper.
        The purpose is to seamlessly use the face detection onnx model and 
        mostly, to process the raw output of this model.

        args:
            onnx_path (str): path to the onnx model to use
            feat_stride_fpn (list): defines the stride of Feature Pyramid Network
            fmc (int): defines the shift between network outputs index
            num_anchors (int): number of anchors per grid cell
            nms_thresh (float): IOU threshold above which, nms will discard overlapping BBoxes
            conf_thresh (float): Confidence threshold below which, we do not consider a prediction
        """

        self._onnx_path = onnx_path
        self._runtime_session = ort.InferenceSession(self._onnx_path)
        self._feat_stride_fpn = feat_stride_fpn
        self._fmc = fmc
        self._num_anchors = num_anchors
        self._nms_thresh = nms_thresh
        self._conf_thresh = conf_thresh
        
        self._center_cache = {}
        self._height = 640
        self._width = 640
        

    def run(self, frame:np.ndarray)->list:
        """
        Main function to call to infer the onnx model.
        2 Steps : Inference, model output processing

        args:
            frame (np.ndarray): processed float32 frame RGB, BCHW

        returns:
            (list) : normalized bounding boxes [(x1,y1,x2,y2,score),...] coordinates
                     are scaled between 0 and 1. 
        """
        
        raw_output = self._runtime_session.run(None, {"input.1":frame})
        filtered_output = self.postProcess(raw_output) 
        return filtered_output

    def postProcess(self, raw_output:list)->list:
        """
        The model provided in this assignement has only 6 outputs and it is 
        a grid detection model. This function in few steps : 
            1. For each level of the FPN identified through the stride size in self._feat_stride_fpn,
                we get the bboxes predictions and the scores.
            2. We create the grid related to the stride size and thus we define ours anchor
                points.
            3. We filter out the scores with a too low confidence
            4. We transform the distance vectors, previously called 'bbox' into an actual bounding box
                vector.
            5. We concatenate the bboxes which have a good enough confidence score into a general list
            6. We apply NMS algorithm
            7. We rescale the bboxes to values between 0 and 1 in order
                to eventually rescale them to the initial image size scale.
        
        args:
            raw_output (list): raw list of model output
        
        returns:
            (list) : filtered list of bounding boxes scaled between 0 and 1.
        """
    
        scores_list = []
        bboxes_list = []
        res = []

        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = raw_output[idx][0]
            bbox_preds = raw_output[idx + self._fmc][0]
            bbox_preds = bbox_preds * stride

            height = self._height // stride
            width = self._width // stride

            key = (height, width, stride)

            if(key in self._center_cache):
                anchor_centers = self._center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)

                anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
                if self._num_anchors>1:
                    anchor_centers = np.stack([anchor_centers]*self._num_anchors, axis=1).reshape( (-1,2) )
                if len(self._center_cache)<100:
                    self._center_cache[key] = anchor_centers

            pos_inds = np.where(scores>=self._conf_thresh)[0]

            bboxes = distance2bbox(anchor_centers, bbox_preds)

            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]

            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list)
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)

        det = pre_det[keep, :].tolist()
        
        for bboxes in det:
            x1,y1,x2,y2,score = bboxes
            res.append([x1/self._width,y1/self._height,x2/self._width,y2/self._height,score])

        return res

    def nms(self, dets:list)->list:
        """
        Non Max Suppression Implementation :
        https://www.youtube.com/watch?v=VAo84c1hQX8&ab_channel=DeepLearningAI

        Args:
            dets (list): unfiltered list of bounding boxes having a 'good enough' confidence score

        Returns:
            keep (list): filtered list of bounding boxes
        """

        thresh = self._nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

