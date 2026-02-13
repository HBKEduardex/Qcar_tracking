#!/usr/bin/env python3
"""
Yellow Line Position Node — Dual Mode (Point / Polynomial)

Modo RECTA: centroide simple (yellow + edge) → error lateral directo.
Modo CURVA: fit polinomial + offset Ackermann + lookahead.

Publica:
  - /lane/yellow/error   (Float32)
  - /lane/yellow/visible (Bool)
  - /lane/center/error   (Float32)
  - /lane/center/visible (Bool)
  - /lane/curvature      (Float32)  ← NUEVO

Debug: 2 ventanas separadas:
  - lane_detection:  detección básica (centroides, lane center, estado)
  - lane_prediction: curvas fiteadas, Ackermann, lookahead

Author: eduardex
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Bool
from cv_bridge import CvBridge
import numpy as np
import cv2


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class YellowLinePositionNode(Node):

    # ============ Colores debug (BGR) ============
    # Ventana 1 (detection)
    CLR_IMG_CENTER  = (180, 180, 180)   # blanco tenue
    CLR_YELLOW_DOT  = (0, 220, 220)     # amarillo
    CLR_EDGE_DOT    = (0, 180, 0)       # verde
    CLR_LANE_CENTER = (255, 0, 200)     # magenta
    CLR_TEXT        = (255, 255, 255)    # blanco

    # Ventana 2 (prediction)
    CLR_YELLOW_FIT  = (255, 255, 0)     # cyan
    CLR_EDGE_FIT    = (0, 140, 255)     # naranja
    CLR_CENTER_RAW  = (200, 100, 255)   # rosa
    CLR_CENTER_ACK  = (50, 255, 50)     # verde limón
    CLR_LOOKAHEAD   = (0, 0, 255)       # rojo brillante

    def __init__(self):
        super().__init__('yellow_line_position_node')

        # ==================== Topics ====================
        self.declare_parameter('mask_topic', '/segmentation/color_mask')
        self.declare_parameter('yellow_error_topic', '/lane/yellow/error')
        self.declare_parameter('yellow_visible_topic', '/lane/yellow/visible')
        self.declare_parameter('center_error_topic', '/lane/center/error')
        self.declare_parameter('center_visible_topic', '/lane/center/visible')
        self.declare_parameter('curvature_topic', '/lane/curvature')

        # ==================== ROI ====================
        self.declare_parameter('use_bottom_ratio', 0.45)

        # ==================== Thresholds ====================
        self.declare_parameter('min_yellow_pixels', 120)
        self.declare_parameter('min_edge_pixels', 60)
        self.declare_parameter('min_edge_x_ratio', 0.35)

        # ==================== Debug ====================
        self.declare_parameter('show_detection_window', True)
        self.declare_parameter('show_prediction_window', True)

        # ==================== Lane width learning ====================
        self.declare_parameter('default_lane_width_ratio', 0.40)
        self.declare_parameter('lane_width_ema_alpha', 0.20)
        self.declare_parameter('min_lane_width_ratio', 0.25)
        self.declare_parameter('max_lane_width_ratio', 0.95)

        # ==================== Polynomial / Ackermann ====================
        self.declare_parameter('num_sample_rows', 12)
        self.declare_parameter('poly_degree', 2)
        self.declare_parameter('lookahead_ratio', 0.4)
        self.declare_parameter('ackermann_gain', 0.24)
        self.declare_parameter('min_fit_points', 5)
        self.declare_parameter('curvature_threshold', 0.009)
        self.declare_parameter('edge_hold_frames', 15)
        self.declare_parameter('edge_ema_alpha', 0.3)
        self.declare_parameter('curvature_ema_alpha', 0.15)
        self.declare_parameter('curve_exit_frames', 5)
        self.declare_parameter('straight_offset_px', 22)  # positive = shift right (towards edge)
        self.declare_parameter('center_ema_alpha', 0.4)  # smoothing for lane center transitions

        # ==================== Sidewalk validation ====================
        self.declare_parameter('sidewalk_validate', True)        # CALIBRATE: enable/disable sidewalk check
        self.declare_parameter('min_road_ratio', 0.6)            # CALIBRATE: min fraction of sample points on road
        self.declare_parameter('validate_num_points', 8)         # CALIBRATE: how many vertical points to sample

        # ==================== Load params ====================
        self.mask_topic = self.get_parameter('mask_topic').value
        self.yellow_error_topic = self.get_parameter('yellow_error_topic').value
        self.yellow_visible_topic = self.get_parameter('yellow_visible_topic').value
        self.center_error_topic = self.get_parameter('center_error_topic').value
        self.center_visible_topic = self.get_parameter('center_visible_topic').value
        self.curvature_topic = self.get_parameter('curvature_topic').value

        self.use_bottom_ratio = float(self.get_parameter('use_bottom_ratio').value)
        self.min_yellow_pixels = int(self.get_parameter('min_yellow_pixels').value)
        self.min_edge_pixels = int(self.get_parameter('min_edge_pixels').value)
        self.min_edge_x_ratio = float(self.get_parameter('min_edge_x_ratio').value)

        self.show_detection_window = bool(self.get_parameter('show_detection_window').value)
        self.show_prediction_window = bool(self.get_parameter('show_prediction_window').value)

        self.default_lane_width_ratio = float(self.get_parameter('default_lane_width_ratio').value)
        self.lane_width_ema_alpha = float(self.get_parameter('lane_width_ema_alpha').value)
        self.min_lane_width_ratio = float(self.get_parameter('min_lane_width_ratio').value)
        self.max_lane_width_ratio = float(self.get_parameter('max_lane_width_ratio').value)

        self.num_sample_rows = int(self.get_parameter('num_sample_rows').value)
        self.poly_degree = int(self.get_parameter('poly_degree').value)
        self.lookahead_ratio = float(self.get_parameter('lookahead_ratio').value)
        self.ackermann_gain = float(self.get_parameter('ackermann_gain').value)
        self.min_fit_points = int(self.get_parameter('min_fit_points').value)
        self.curvature_threshold = float(self.get_parameter('curvature_threshold').value)
        self.edge_hold_frames = int(self.get_parameter('edge_hold_frames').value)
        self.edge_ema_alpha = float(self.get_parameter('edge_ema_alpha').value)
        self.curvature_ema_alpha = float(self.get_parameter('curvature_ema_alpha').value)
        self.curve_exit_frames = int(self.get_parameter('curve_exit_frames').value)
        self.straight_offset_px = int(self.get_parameter('straight_offset_px').value)
        self.center_ema_alpha = float(self.get_parameter('center_ema_alpha').value)
        self.sidewalk_validate = bool(self.get_parameter('sidewalk_validate').value)
        self.min_road_ratio = float(self.get_parameter('min_road_ratio').value)
        self.validate_num_points = int(self.get_parameter('validate_num_points').value)

        # ==================== State ====================
        self.bridge = CvBridge()
        self.last_lane_center_x = None
        self.center_has_value = False
        self.lane_width_px = None
        self.last_edge_cx = None           # edge memory
        self.edge_lost_count = 0           # frames since edge was last seen
        self.last_edge_poly = None         # polynomial memory
        self.smoothed_curvature = 0.0      # EMA-smoothed curvature
        self.straight_count = 0            # consecutive frames below threshold
        self.smoothed_center_x = None      # EMA-smoothed lane center
        self.last_validation_status = ''    # '', 'FLIP', or 'FALLBACK'

        # ==================== ROS I/O ====================
        self.sub = self.create_subscription(Image, self.mask_topic, self.cb_mask, 10)
        self.pub_y_err = self.create_publisher(Float32, self.yellow_error_topic, 10)
        self.pub_y_vis = self.create_publisher(Bool, self.yellow_visible_topic, 10)
        self.pub_c_err = self.create_publisher(Float32, self.center_error_topic, 10)
        self.pub_c_vis = self.create_publisher(Bool, self.center_visible_topic, 10)
        self.pub_curv = self.create_publisher(Float32, self.curvature_topic, 10)

        if self.show_detection_window:
            cv2.namedWindow('lane_detection', cv2.WINDOW_NORMAL)
        if self.show_prediction_window:
            cv2.namedWindow('lane_prediction', cv2.WINDOW_NORMAL)

        self.get_logger().info("YellowLinePositionNode started (dual mode: point / poly+Ackermann)")
        self.get_logger().info(f"  mask_topic: {self.mask_topic}")
        self.get_logger().info(f"  curvature_topic: {self.curvature_topic}")
        self.get_logger().info(f"  curvature_threshold: {self.curvature_threshold}")
        self.get_logger().info(f"  ackermann_gain: {self.ackermann_gain}")

    # ================================================================
    #  Row-wise helpers (for polynomial mode)
    # ================================================================
    def _sample_yellow_per_row(self, yellow_bin, sample_ys):
        ys_out, xs_out = [], []
        for y in sample_ys:
            row = yellow_bin[y, :]
            nz = np.nonzero(row)[0]
            if nz.size >= 3:
                ys_out.append(float(y))
                xs_out.append(float(np.mean(nz)))
        return np.array(ys_out), np.array(xs_out)

    def _sample_edge_per_row(self, edge_bin, sample_ys, w):
        min_x = int(self.min_edge_x_ratio * w)
        ys_out, xs_out = [], []
        for y in sample_ys:
            row = edge_bin[y, :]
            nz = np.nonzero(row)[0]
            if nz.size >= 1:  # even 1 pixel is useful
                rx = int(nz[-1])
                if rx >= min_x:
                    ys_out.append(float(y))
                    xs_out.append(float(rx))
        return np.array(ys_out), np.array(xs_out)

    def _safe_polyfit(self, ys, xs, deg):
        if len(ys) < max(deg + 1, self.min_fit_points):
            return None
        try:
            return np.polyfit(ys, xs, deg)
        except (np.linalg.LinAlgError, ValueError):
            return None

    def _edge_cx_rightmost_median(self, edge_bin, w):
        """Robust edge: rightmost pixel per row, then median."""
        ys, xs = np.where(edge_bin > 0)
        if xs.size == 0:
            return None
        rightmost = {}
        for y, x in zip(ys, xs):
            prev = rightmost.get(int(y), None)
            if prev is None or x > prev:
                rightmost[int(y)] = int(x)
        rightmost_xs = np.array(list(rightmost.values()), dtype=np.float32)
        if rightmost_xs.size < 4:  # relaxed from 10
            return None
        cx = int(np.median(rightmost_xs))
        if cx < int(self.min_edge_x_ratio * w):
            return None
        return cx

    # ================================================================
    #  Sidewalk validation helpers
    # ================================================================
    def _is_sidewalk(self, roi_bgr, x, y):
        """Check if pixel (x,y) in the BGR mask is sidewalk (black)."""
        h, w = roi_bgr.shape[:2]
        if x < 0 or x >= w or y < 0 or y >= h:
            return True  # out-of-bounds treated as invalid
        b, g, r = int(roi_bgr[y, x, 0]), int(roi_bgr[y, x, 1]), int(roi_bgr[y, x, 2])
        return b < 30 and g < 30 and r < 30  # CALIBRATE: sidewalk color threshold

    def _is_road(self, roi_bgr, x, y):
        """Check if pixel (x,y) in the BGR mask is road (blue)."""
        h, w = roi_bgr.shape[:2]
        if x < 0 or x >= w or y < 0 or y >= h:
            return False
        b, g, r = int(roi_bgr[y, x, 0]), int(roi_bgr[y, x, 1]), int(roi_bgr[y, x, 2])
        return b > 200 and g < 80 and r < 80  # CALIBRATE: road color threshold

    def _road_ratio_at_x(self, roi_bgr, x, h):
        """Sample vertical column at x, return fraction of points on road."""
        n = self.validate_num_points
        if n <= 0:
            return 1.0
        ys = np.linspace(int(h * 0.3), h - 2, n, dtype=int)  # CALIBRATE: sample from 30% to bottom
        road_count = sum(1 for y in ys if self._is_road(roi_bgr, x, int(y)))
        return road_count / n

    def _project_to_road(self, roi_bgr, h, w, prefer_x=None):
        """Fallback: find road centroid in bottom third of ROI.
           Returns the x coordinate closest to prefer_x within road."""
        # Sample the bottom third for road pixels
        y_start = int(h * 0.65)  # CALIBRATE: bottom 35% of ROI
        road_slice = roi_bgr[y_start:, :]
        # Road is blue: B>200, G<80, R<80
        road_mask = (road_slice[:, :, 0] > 200) & (road_slice[:, :, 1] < 80) & (road_slice[:, :, 2] < 80)
        road_xs = np.where(road_mask.any(axis=0))[0]
        if road_xs.size == 0:
            return None
        if prefer_x is not None:
            # Return the road x closest to the preferred position
            dists = np.abs(road_xs.astype(float) - float(prefer_x))
            return int(road_xs[np.argmin(dists)])
        # Otherwise return centroid of right half of road (drive on right)
        mid = w // 2
        right_road = road_xs[road_xs >= mid]
        if right_road.size > 0:
            return int(np.mean(right_road))
        return int(np.mean(road_xs))

    def _validate_and_fix_center(self, roi_bgr, lane_center_x, h, w,
                                  yellow_cx, edge_cx, lane_width_px, offset_px):
        """Validate that lane_center_x is on road, not sidewalk.
           Returns (validated_x, status_string).
           status_string: '' = ok, 'FLIP' = offset flipped, 'FALLBACK' = road projection."""
        if lane_center_x is None or not self.sidewalk_validate:
            return lane_center_x, ''

        # --- Step 1: Check if current center is on road ---
        ratio = self._road_ratio_at_x(roi_bgr, lane_center_x, h)
        if ratio >= self.min_road_ratio:  # CALIBRATE: min_road_ratio
            return lane_center_x, ''  # valid

        # --- Step 2: Flip offset direction ---
        # Determine base point and reverse the offset
        flipped_x = None
        if yellow_cx is not None and edge_cx is not None:
            # Both visible: center was average, flip by shifting opposite to offset
            flipped_x = int(clamp(
                (yellow_cx + edge_cx) / 2.0 - offset_px, 0, w - 1))
        elif edge_cx is not None:
            # Edge only: was shifting left, now shift right
            flipped_x = int(clamp(
                edge_cx + 0.5 * lane_width_px + offset_px, 0, w - 1))
        elif yellow_cx is not None:
            # Yellow only: was shifting right, now shift left
            flipped_x = int(clamp(
                yellow_cx - 0.5 * lane_width_px - offset_px, 0, w - 1))

        if flipped_x is not None:
            flip_ratio = self._road_ratio_at_x(roi_bgr, flipped_x, h)
            if flip_ratio >= self.min_road_ratio:
                return flipped_x, 'FLIP'

        # --- Step 3: Fallback to road projection ---
        fallback_x = self._project_to_road(roi_bgr, h, w,
                                            prefer_x=self.last_lane_center_x)
        if fallback_x is not None:
            return fallback_x, 'FALLBACK'

        # Nothing worked, return original (better than nothing)
        return lane_center_x, ''

    # ================================================================
    #  Main callback
    # ================================================================
    def cb_mask(self, msg: Image):
        mask_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if mask_bgr is None or mask_bgr.size == 0:
            return

        H, W = mask_bgr.shape[:2]
        y0 = int(H * (1.0 - self.use_bottom_ratio))
        roi = mask_bgr[y0:, :]
        h, w = roi.shape[:2]
        center_img_x = w // 2

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        sample_ys = np.linspace(h - 5, 5, self.num_sample_rows, dtype=int)

        # ================== 1) Yellow detection ==================
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_y = np.array([20, 120, 120], dtype=np.uint8)
        upper_y = np.array([40, 255, 255], dtype=np.uint8)
        yellow_bin = cv2.inRange(roi_hsv, lower_y, upper_y)
        yellow_bin = cv2.morphologyEx(yellow_bin, cv2.MORPH_OPEN, kernel, iterations=1)
        yellow_bin = cv2.morphologyEx(yellow_bin, cv2.MORPH_CLOSE, kernel, iterations=1)

        yellow_pixels = int(cv2.countNonZero(yellow_bin))
        yellow_visible = yellow_pixels >= self.min_yellow_pixels

        # Centroid (for point mode)
        yellow_cx = None
        if yellow_visible:
            M = cv2.moments(yellow_bin)
            if M["m00"] > 0:
                yellow_cx = int(M["m10"] / M["m00"])
            else:
                yellow_visible = False

        # Yellow error (always from centroid, backward compat)
        if yellow_visible and yellow_cx is not None:
            yellow_error = (yellow_cx - (w / 2.0)) / (w / 2.0)
            yellow_error = float(clamp(yellow_error, -1.0, 1.0))
        else:
            yellow_error = 0.0

        self.pub_y_vis.publish(Bool(data=yellow_visible))
        self.pub_y_err.publish(Float32(data=yellow_error))

        # ================== 2) Edge detection ==================
        b, g, r = cv2.split(roi)
        edge_bin = ((r > 150) & (g < 100) & (b < 100)).astype(np.uint8) * 255
        edge_bin = cv2.morphologyEx(edge_bin, cv2.MORPH_OPEN, kernel, iterations=1)
        edge_bin = cv2.morphologyEx(edge_bin, cv2.MORPH_CLOSE, kernel, iterations=2)

        edge_pixels = int(cv2.countNonZero(edge_bin))
        edge_visible = edge_pixels >= self.min_edge_pixels

        # Point mode edge (rightmost median)
        edge_cx = None
        if edge_visible:
            edge_cx = self._edge_cx_rightmost_median(edge_bin, w)
            if edge_cx is None:
                edge_visible = False

        # ---- Edge memory (EMA + hold) ----
        if edge_visible and edge_cx is not None:
            # Update EMA
            if self.last_edge_cx is None:
                self.last_edge_cx = float(edge_cx)
            else:
                a = self.edge_ema_alpha
                self.last_edge_cx = (1.0 - a) * self.last_edge_cx + a * float(edge_cx)
            self.edge_lost_count = 0
        else:
            self.edge_lost_count += 1
            # Use memory if within hold window
            if self.last_edge_cx is not None and self.edge_lost_count <= self.edge_hold_frames:
                edge_cx = int(self.last_edge_cx)
                edge_visible = True  # treat as visible (held)

        # ================== 3) Polynomial fitting (always, for curvature) ==================
        y_ys, y_xs = np.array([]), np.array([])
        e_ys, e_xs = np.array([]), np.array([])
        yellow_poly = None
        edge_poly = None
        center_poly = None
        center_poly_ack = None
        curvature = 0.0
        poly_valid = False

        if yellow_visible:
            y_ys, y_xs = self._sample_yellow_per_row(yellow_bin, sample_ys)
            if len(y_ys) >= self.min_fit_points:
                yellow_poly = self._safe_polyfit(y_ys, y_xs, self.poly_degree)

        if edge_visible:
            e_ys, e_xs = self._sample_edge_per_row(edge_bin, sample_ys, w)
            if len(e_ys) >= self.min_fit_points:
                edge_poly = self._safe_polyfit(e_ys, e_xs, self.poly_degree)
                if edge_poly is not None:
                    self.last_edge_poly = edge_poly.copy()  # save for fallback

        # Fallback: use last known edge poly if current detection failed
        if edge_poly is None and self.last_edge_poly is not None and self.edge_lost_count <= self.edge_hold_frames:
            edge_poly = self.last_edge_poly.copy()

        # Center poly
        if edge_poly is not None:
            if yellow_poly is not None:
                center_poly = (np.array(yellow_poly) + np.array(edge_poly)) / 2.0
            else:
                # Edge only → shift left by half lane width
                offset_poly = np.array(edge_poly).copy()
                if self.lane_width_px is not None:
                    offset_poly[-1] -= 0.5 * float(self.lane_width_px)
                else:
                    offset_poly[-1] -= 0.5 * self.default_lane_width_ratio * w
                center_poly = offset_poly
            poly_valid = True
        elif yellow_poly is not None:
            # Yellow only → shift right by half lane width (inverse)
            offset_poly = np.array(yellow_poly).copy()
            if self.lane_width_px is not None:
                offset_poly[-1] += 0.5 * float(self.lane_width_px)
            else:
                offset_poly[-1] += 0.5 * self.default_lane_width_ratio * w
            center_poly = offset_poly
            poly_valid = True

        # Extract curvature (raw)
        raw_curvature = 0.0
        if center_poly is not None and len(center_poly) >= 3:
            raw_curvature = float(center_poly[0])

        # Smooth curvature with EMA
        ca = self.curvature_ema_alpha
        self.smoothed_curvature = (1.0 - ca) * self.smoothed_curvature + ca * raw_curvature
        curvature = self.smoothed_curvature

        # Ackermann offset (for curve mode)
        if center_poly is not None:
            center_poly_ack = np.array(center_poly).copy()
            if len(center_poly_ack) >= 3 and abs(curvature) >= self.curvature_threshold:
                ack_offset = -self.ackermann_gain * curvature * (h ** 2)
                center_poly_ack[-1] += ack_offset

        # Publish curvature
        self.pub_curv.publish(Float32(data=curvature))

        # ================== 4) Learn lane width ==================
        if yellow_visible and edge_visible and yellow_cx is not None and edge_cx is not None:
            width_px_meas = float(edge_cx - yellow_cx)
            min_w = self.min_lane_width_ratio * w
            max_w = self.max_lane_width_ratio * w
            if min_w <= width_px_meas <= max_w:
                if self.lane_width_px is None:
                    self.lane_width_px = width_px_meas
                else:
                    a = float(clamp(self.lane_width_ema_alpha, 0.0, 1.0))
                    self.lane_width_px = (1.0 - a) * self.lane_width_px + a * width_px_meas

        if self.lane_width_px is None:
            self.lane_width_px = float(clamp(self.default_lane_width_ratio, 0.05, 0.99)) * w

        # ================== 5) Decide mode & compute error ==================
        # Hysteresis: enter curve immediately, but require N straight frames to exit
        is_above_thresh = abs(curvature) >= self.curvature_threshold and poly_valid
        if is_above_thresh:
            self.straight_count = 0
            in_curve = True
        else:
            self.straight_count += 1
            in_curve = self.straight_count < self.curve_exit_frames  # stay in curve until confirmed straight

        lane_center_x = None
        lookahead_y = int(h * (1.0 - self.lookahead_ratio))
        lookahead_y = int(clamp(lookahead_y, 2, h - 2))

        if in_curve and center_poly_ack is not None:
            # ---- CURVE MODE: polynomial + Ackermann + lookahead ----
            lane_center_x = int(clamp(np.polyval(center_poly_ack, lookahead_y), 0, w - 1))
            self.last_lane_center_x = lane_center_x
            self.center_has_value = True
        else:
            # ---- STRAIGHT MODE: simple point centroid (original) ----
            if edge_visible and edge_cx is not None:
                if yellow_visible and yellow_cx is not None:
                    lane_center_x = int((yellow_cx + edge_cx) / 2.0)
                else:
                    # Edge only → shift left
                    lane_center_x = int(edge_cx - 0.5 * float(self.lane_width_px))
                lane_center_x = int(clamp(lane_center_x + self.straight_offset_px, 0, w - 1))
                self.last_lane_center_x = lane_center_x
                self.center_has_value = True
            elif yellow_visible and yellow_cx is not None:
                # Yellow only → shift right (inverse)
                lane_center_x = int(yellow_cx + 0.5 * float(self.lane_width_px))
                lane_center_x = int(clamp(lane_center_x + self.straight_offset_px, 0, w - 1))
                self.last_lane_center_x = lane_center_x
                self.center_has_value = True
            else:
                # HOLD
                if self.last_lane_center_x is not None:
                    lane_center_x = self.last_lane_center_x

        # ================== Sidewalk validation ==================
        self.last_validation_status = ''
        if lane_center_x is not None:
            lane_center_x, self.last_validation_status = self._validate_and_fix_center(
                roi, lane_center_x, h, w,
                yellow_cx if yellow_visible else None,
                edge_cx if edge_visible else None,
                float(self.lane_width_px),
                self.straight_offset_px if not in_curve else 0,
            )

        # Smooth lane_center_x with EMA to prevent jumps
        if lane_center_x is not None:
            if self.smoothed_center_x is None:
                self.smoothed_center_x = float(lane_center_x)
            else:
                ca = self.center_ema_alpha
                self.smoothed_center_x = (1.0 - ca) * self.smoothed_center_x + ca * float(lane_center_x)
            lane_center_x = int(clamp(self.smoothed_center_x, 0, w - 1))

        # Error
        if lane_center_x is not None:
            center_error = (lane_center_x - (w / 2.0)) / (w / 2.0)
            center_error = float(clamp(center_error, -1.0, 1.0))
        else:
            center_error = 0.0

        self.pub_c_vis.publish(Bool(data=self.center_has_value))
        self.pub_c_err.publish(Float32(data=center_error))

        # ================== 6) Debug windows ==================
        if self.show_detection_window:
            self._draw_detection(
                roi, h, w, center_img_x,
                yellow_visible, yellow_cx, edge_visible, edge_cx,
                lane_center_x, in_curve,
                yellow_pixels, edge_pixels, yellow_error, center_error, curvature,
            )

        if self.show_prediction_window:
            self._draw_prediction(
                roi, h, w,
                yellow_poly, y_ys, y_xs,
                edge_poly, e_ys, e_xs,
                center_poly, center_poly_ack,
                lane_center_x, lookahead_y,
                curvature, in_curve,
            )

    # ================================================================
    #  Window 1: Detection (clean, point-based)
    # ================================================================
    def _draw_detection(
        self, roi, h, w, center_img_x,
        yellow_visible, yellow_cx, edge_visible, edge_cx,
        lane_center_x, in_curve,
        yellow_pixels, edge_pixels, yellow_error, center_error, curvature,
    ):
        dbg = roi.copy()

        # Image center
        cv2.line(dbg, (center_img_x, 0), (center_img_x, h), self.CLR_IMG_CENTER, 1)

        # Yellow centroid
        if yellow_visible and yellow_cx is not None:
            cv2.circle(dbg, (yellow_cx, int(h * 0.60)), 8, self.CLR_YELLOW_DOT, -1)
            cv2.putText(dbg, "YELLOW", (yellow_cx + 12, int(h * 0.60)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.CLR_YELLOW_DOT, 2)

        # Edge centroid
        if edge_visible and edge_cx is not None:
            cv2.circle(dbg, (edge_cx, int(h * 0.72)), 10, self.CLR_EDGE_DOT, -1)
            cv2.circle(dbg, (edge_cx, int(h * 0.72)), 14, (0, 0, 0), 2)
            cv2.putText(dbg, "EDGE", (edge_cx + 12, int(h * 0.72)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.CLR_EDGE_DOT, 2)

        # Lane center
        if lane_center_x is not None:
            cv2.line(dbg, (lane_center_x, 0), (lane_center_x, h), self.CLR_LANE_CENTER, 2)
            cv2.circle(dbg, (lane_center_x, int(h * 0.85)), 8, self.CLR_LANE_CENTER, -1)

        # Text
        mode_str = "CURVA" if in_curve else "RECTA"
        mode_clr = self.CLR_CENTER_ACK if in_curve else self.CLR_TEXT
        cv2.putText(dbg,
                    f"Yvis={yellow_visible} pix={yellow_pixels} | Evis={edge_visible} pix={edge_pixels}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.CLR_TEXT, 2)
        cv2.putText(dbg,
                    f"Yerr={yellow_error:+.2f} | Cerr={center_error:+.2f} | lw={float(self.lane_width_px):.0f}px",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.CLR_TEXT, 2)
        cv2.putText(dbg,
                    f"MODO: {mode_str} | curv={curvature:.5f}",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_clr, 2)
        if self.last_validation_status:
            val_clr = (0, 165, 255) if self.last_validation_status == 'FLIP' else (0, 0, 255)
            cv2.putText(dbg,
                        f"!! {self.last_validation_status} !!",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, val_clr, 2)

        cv2.imshow('lane_detection', dbg)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            self.show_detection_window = False
            cv2.destroyWindow('lane_detection')

    # ================================================================
    #  Window 2: Prediction (polynomial curves)
    # ================================================================
    def _draw_prediction(
        self, roi, h, w,
        yellow_poly, y_ys, y_xs,
        edge_poly, e_ys, e_xs,
        center_poly, center_poly_ack,
        lane_center_x, lookahead_y,
        curvature, in_curve,
    ):
        dbg = roi.copy()
        draw_ys = np.linspace(5, h - 5, 80, dtype=int)

        def _draw_poly(poly, color, thickness):
            pts = []
            for y in draw_ys:
                x = int(clamp(np.polyval(poly, y), 0, w - 1))
                pts.append((x, int(y)))
            if len(pts) > 1:
                cv2.polylines(dbg, [np.array(pts)], False, color, thickness, cv2.LINE_AA)

        # Yellow fit + sample points
        if yellow_poly is not None:
            _draw_poly(yellow_poly, self.CLR_YELLOW_FIT, 2)
            for yy, xx in zip(y_ys, y_xs):
                cv2.circle(dbg, (int(xx), int(yy)), 3, self.CLR_YELLOW_FIT, -1)

        # Edge fit + sample points
        if edge_poly is not None:
            _draw_poly(edge_poly, self.CLR_EDGE_FIT, 2)
            for yy, xx in zip(e_ys, e_xs):
                cv2.circle(dbg, (int(xx), int(yy)), 3, self.CLR_EDGE_FIT, -1)

        # Center raw (thin)
        if center_poly is not None:
            _draw_poly(center_poly, self.CLR_CENTER_RAW, 1)

        # Center Ackermann (thick, only if in curve)
        if center_poly_ack is not None and in_curve:
            _draw_poly(center_poly_ack, self.CLR_CENTER_ACK, 3)

        # Lookahead
        if lane_center_x is not None:
            cv2.circle(dbg, (lane_center_x, lookahead_y), 8, self.CLR_LOOKAHEAD, -1)
            cv2.circle(dbg, (lane_center_x, lookahead_y), 12, self.CLR_LOOKAHEAD, 2)
            cv2.line(dbg, (0, lookahead_y), (w, lookahead_y), self.CLR_LOOKAHEAD, 1, cv2.LINE_AA)

        # Info text
        mode_str = "CURVA (poly+Ack)" if in_curve else "RECTA (point)"
        mode_clr = self.CLR_CENTER_ACK if in_curve else self.CLR_TEXT
        cv2.putText(dbg, f"MODO: {mode_str}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_clr, 2)
        cv2.putText(dbg, f"curv={curvature:.5f} | thresh={self.curvature_threshold} | ack={self.ackermann_gain}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.CLR_TEXT, 1)

        # Legend
        ly = h - 65
        cv2.putText(dbg, "--- YELLOW fit", (10, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.CLR_YELLOW_FIT, 1)
        cv2.putText(dbg, "--- EDGE fit", (10, ly + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.CLR_EDGE_FIT, 1)
        cv2.putText(dbg, "--- CENTER raw", (10, ly + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.CLR_CENTER_RAW, 1)
        cv2.putText(dbg, "=== CENTER Ackermann", (10, ly + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.CLR_CENTER_ACK, 1)
        cv2.putText(dbg, " o  LOOKAHEAD", (10, ly + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.CLR_LOOKAHEAD, 1)

        cv2.imshow('lane_prediction', dbg)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            self.show_prediction_window = False
            cv2.destroyWindow('lane_prediction')


def main(args=None):
    rclpy.init(args=args)
    node = YellowLinePositionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
