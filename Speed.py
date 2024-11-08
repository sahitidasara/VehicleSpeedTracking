from collections import defaultdict
from time import time
import cv2
import numpy as np
from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors


class SpeedEstimator:
    def __init__(self, names, reg_pts=None, view_img=False, line_thickness=2, spdl_dist_thresh=10):
        self.reg_pts = reg_pts if reg_pts is not None else [(640, 400), (640, 400)]
        self.names = names
        self.trk_history = defaultdict(list)
        self.view_img = view_img
        self.tf = line_thickness
        self.spd = {}
        self.trkd_ids = []
        self.spdl = spdl_dist_thresh
        self.trk_pt = {}
        self.trk_pp = {}
        self.env_check = check_imshow(warn=True)

    def estimate_speed(self, im0, tracks):
        if tracks[0].boxes.id is None:
            return im0

        boxes = tracks[0].boxes.xyxy.cpu()
        clss = tracks[0].boxes.cls.cpu().tolist()
        t_ids = tracks[0].boxes.id.int().cpu().tolist()
        annotator = Annotator(im0, line_width=self.tf)
        annotator.draw_region(reg_pts=self.reg_pts, color=(255, 0, 255), thickness=self.tf * 2)

        for box, t_id, cls in zip(boxes, t_ids, clss):
            track = self.trk_history[t_id]
            bbox_center = (float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2))
            track.append(bbox_center)

            if len(track) > 30:
                track.pop(0)

            trk_pts = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            if t_id not in self.trk_pt:
                self.trk_pt[t_id] = 0

            speed_label = f"{int(self.spd[t_id][0] * 0.621371)} m/h" if t_id in self.spd else self.names[int(cls)]
            bbox_color = colors(int(t_id), True)

            annotator.box_label(box, speed_label, bbox_color)
            cv2.polylines(im0, [trk_pts], isClosed=False, color=bbox_color, thickness=self.tf)
            cv2.circle(im0, (int(track[-1][0]), int(track[-1][1])), self.tf * 2, bbox_color, -1)

            # Calculation of object speed
            if not 0 < track[-1][0] < im0.shape[1]:
                return
            if not self.reg_pts[0][0] < track[-1][0] < self.reg_pts[1][0]:
                return
            if self.reg_pts[1][1] - self.spdl < track[-1][1] < self.reg_pts[1][1] + self.spdl:
                direction = "known"
            elif self.reg_pts[0][1] - self.spdl < track[-1][1] < self.reg_pts[0][1] + self.spdl:
                direction = "known"
            else:
                direction = "unknown"

            if self.trk_pt.get(t_id) != 0 and direction != "unknown" and t_id not in self.trkd_ids:
                self.trkd_ids.append(t_id)

                time_difference = time() - self.trk_pt[t_id]
                if time_difference > 0:
                    # Store speed and class ID for each object
                    self.spd[t_id] = (np.abs(track[-1][1] - self.trk_pp[t_id][1]) / time_difference, cls)

            self.trk_pt[t_id] = time()
            self.trk_pp[t_id] = track[-1]

        if self.view_img and self.env_check:
            cv2.imshow("Ultralytics Speed Estimation", im0)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return

        return im0

    def get_speed_data(self):
        """Returns speed data including object ID, speed, class ID, and class name."""
        return [
            {
                "id": obj_id,
                "speed": speed,
                "class_id": cls_id,
                "class_name": self.names.get(cls_id, "Unknown")
            }
            for obj_id, (speed, cls_id) in self.spd.items()
        ]


if __name__ == "__main__":
    names = {0: "person", 1: "car", 2: "truck"}
    speed_estimator = SpeedEstimator(names)
