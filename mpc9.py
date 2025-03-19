from time import sleep
import threading
import cv2
import time
import math
import serial
import numpy as np
import cvxpy as cp
from scipy.signal import cont2discrete
import matplotlib.pyplot as plt
from collections import deque
import queue
import pickle
from scipy.spatial import KDTree

shared_data = {
    "encoder_value": None,
    "cte":None,
    "relative_yaw_angle": None,
}
data_lock = threading.Lock()

vidcap = cv2.VideoCapture("output_video.mp4")

# Khởi tạo danh sách lưu trữ dữ liệu để vẽ biểu đồ
# 📌 1. Tải LUT từ file pickle
with open("lookup_table.pkl", "rb") as f:
    lookup_table = pickle.load(f)

# 📌 2. Xây dựng KDTree từ các key trong LUT
lut_keys = np.array(list(lookup_table.keys()))  # Chuyển danh sách key thành mảng NumPy
tree = KDTree(lut_keys)  # Tạo cây KDTree để tăng tốc tìm kiếm
def get_nearest_control(x0):
    """
    Hàm tìm giá trị điều khiển gần nhất từ LUT bằng KDTree.

    Args:
        x0 (numpy.array): Trạng thái đầu vào (vector 4 chiều).

    Returns:
        float: Giá trị điều khiển tối ưu gần nhất (độ).
    """
    key = tuple(np.round(x0, 2))  # Làm tròn đầu vào để khớp key LUT
    dist, idx = tree.query(key)  # Tìm chỉ số của điểm gần nhất
    nearest_key = tuple(lut_keys[idx])  # Lấy key gần nhất
    return - np.rad2deg(lookup_table[nearest_key][0])  # Chuyển từ radian sang độ


step_counter = 0  # Biến đếm thời gian
# Serial port setup
#ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
denta = 0
mid = 80
lane_offset = 0
pre_lane_offset = 0
curvature = 0
cur = np.array([[0], [0], [0], [0], [0]])
anlge_inclination_wrt_x = 0
anlge_inclination_wrt_y =0
pre_inclination_wrt_y = 0
x0 = np.array([[0], [0], [0], [0]])

N = 5
v_k = np.array([[0], [0], [0], [0], [0]])

# Biến toàn cục
A_d, B1_d, B2_d, AX, BU, BV, H, umin, umax, x, u = None, None, None, None, None, None, None, None, None, None, None

C = np.eye(4)  # Quan sát toàn bộ trạng thái
Q = np.diag([0.02, 1, 0.02, 1])  # Nhiễu quá trình
R = np.diag([0.000001, 0.01, 0.000001, 0.01])  # Nhiễu đo lường

# Hiệp phương sai ban đầu và trạng thái ban đầu
P = np.eye(4)*0.0001
x_hat = np.zeros((4, 1))


def kalman_predict(u, w):
    """
    Bước dự đoán của bộ lọc Kalman
    - A, B1, B2: Ma trận trạng thái và điều khiển
    - Q: Ma trận nhiễu quá trình
    - P: Hiệp phương sai hiện tại
    - x_hat: Trạng thái hiện tại
    - u: Đầu vào điều khiển (góc đánh lái)
    - w: Nhiễu bên ngoài
    """
    u = np.atleast_2d(u).reshape(-1, 1)
    w = np.atleast_2d(w).reshape(-1, 1)
    global A_d, B1_d, B2_d,Q,P,x_hat
    x_hat = A_d @ x_hat + B1_d @ u + B2_d @ w
    P = A_d @ P @ A_d.T + Q
    return x_hat, P

def kalman_update(z):
    """
    Bước cập nhật của bộ lọc Kalman
    - C: Ma trận quan sát
    - R: Ma trận nhiễu đo lường
    - P: Hiệp phương sai hiện tại
    - x_hat: Trạng thái hiện tại
    - z: Dữ liệu đo lường thực tế
    """
    global  C, R, P, x_hat
    S = C @ P @ C.T + R
    K = P @ C.T @ np.linalg.inv(S)
    x_hat = x_hat + K @ (z - C @ x_hat)
    P = (np.eye(len(P)) - K @ C) @ P
    return x_hat, P

def kalman_step(u, w, z):
    """
    Cập nhật Kalman một bước
    - u: Đầu vào điều khiển (góc đánh lái)
    - w: Nhiễu bên ngoài (tốc độ gió, địa hình, v.v.)
    - z: Đo lường thực tế từ cảm biến
    """
    global x_hat,P
    x_hat, P = kalman_predict(u, w)
    x_hat, P = kalman_update( z)
    return x_hat, P


def mpc_init(Ts=0.1):
    global A_d, B1_d, B2_d, AX, BU, BV, H, umin, umax, x, u,N

    # --- HỆ THỐNG ---
    m = 2.3
    Lf = 0.12
    Lr = 0.132
    Caf = 70
    Car = 70
    Iz = 0.04
    Vx = 0.4

    # Ma trận động học
    A_c = np.array([[0, 1, 0, 0],
                    [0, -(2 * Caf + 2 * Car) / (m * Vx), (2 * Caf + 2 * Car) / m,
                     (-2 * Caf * Lf + 2 * Car * Lr) / (m * Vx)],
                    [0, 0, 0, 1],
                    [0, (-2 * Caf * Lf + 2 * Car * Lr) / (Iz * Vx), (2 * Caf * Lf - 2 * Car * Lr) / Iz,
                     (-2 * Caf * Lf ** 2 - 2 * Car * Lr ** 2) / (Iz * Vx)]])

    B1_c = np.array([[0],
                     [2 * Caf / m],
                     [0],
                     [2 * Caf * Lf / Iz]])

    B2_c = np.array([[0],
                     [(-2 * Caf * Lf + 2 * Car * Lr) / (m * Vx) - Vx],
                     [0],
                     [(-2 * Caf * Lf ** 2 - 2 * Car * Lr ** 2) / (Iz * Vx)]])

    (A_d, B_d, _, _, _) = cont2discrete((A_c, np.hstack((B1_c, B2_c)), np.eye(4), np.zeros((4, 1))), Ts, method='zoh')
    B1_d = B_d[:, [0]]
    B2_d = B_d[:, [1]]

    # Thông số MPC
    n, m = 4, 1
    Q = np.diag([100, 0, 10, 0])
    QN = Q
    R = np.eye(m) * 6


    AX = np.vstack([np.linalg.matrix_power(A_d, i) for i in range(N + 1)])

    BU = np.zeros(((N + 1) * n, N * m))
    for i in range(1, N + 1):
        for j in range(N):
            if i - j - 1 >= 0:
                BU[i * n:(i + 1) * n, j * m:(j + 1) * m] = (np.linalg.matrix_power(A_d, i - j - 1) @ B1_d).reshape(n, m)

    BV = np.zeros(((N + 1) * n, N * m))
    for i in range(1, N + 1):
        for j in range(i):
            if i - j - 1 >= 0:
                BV[i * n:(i + 1) * n, j * m:(j + 1) * m] = (np.linalg.matrix_power(A_d, i - j - 1) @ B2_d).reshape(n, m)

    QX = np.kron(np.eye(N), Q)
    RU = np.kron(np.eye(N), R)
    QX = np.block([[QX, np.zeros((N * n, n))], [np.zeros((n, N * n)), QN]])
    H = np.block([[QX, np.zeros((QX.shape[0], RU.shape[1]))],
                  [np.zeros((RU.shape[0], QX.shape[1])), RU]])
    H += np.eye(H.shape[0]) * 1e-6

    # Khởi tạo trạng thái và điều khiển




def mpc_control():
    global A_d, B1_d, B2_d, AX, BU, BV, H, umin, umax, N,x0,v_k
    umin, umax = -7 * np.pi / 36, 7 * np.pi / 36
    xk = x0.reshape(-1, 1)
    v_k = cur
    z = cp.Variable(((N + 1) * 4 + N * 1, 1))
    cost = cp.quad_form(z, H)
    constraints = [
        z[: (N + 1) * 4] == AX @ xk + BU @ z[(N + 1) * 4:] + BV @ v_k,
        z[(N + 1) * 4:] >= umin, z[(N + 1) * 4:] <= umax
    ]

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print(f"[❌] MPC không tìm thấy nghiệm. Trạng thái solver: {prob.status}")
        return None
    print(f"hh {z.value[(N + 1) * 4:(N + 1) * 4 + 1].flatten().item()}")
    value_rad = z.value[(N + 1) * 4:(N + 1) * 4 + 1].flatten().item()
    value_deg = np.rad2deg(value_rad)  # Chuyển từ radian sang độ
    return - value_deg


class PID:
    def __init__(self, Kp, Ki, Kd, min_output=-35, max_output=35):
        """
        Khởi tạo PID controller.
        - Kp: Hệ số tỉ lệ (Proportional)
        - Ki: Hệ số tích phân (Integral)
        - Kd: Hệ số đạo hàm (Derivative)
        - min_output: Giá trị giới hạn dưới cho góc lái
        - max_output: Giá trị giới hạn trên cho góc lái
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.min_output = min_output
        self.max_output = max_output

        self.prev_error = 0  # Sai số trước đó
        self.integral = 0  # Tích phân của sai số

    def update(self, error):
        """
        Cập nhật và tính toán điều khiển PID.
        - error: Sai số hiện tại (lateral offset)
        """
        # Tính đạo hàm và tích phân
        derivative = error - self.prev_error
        self.integral += error

        # Tính toán điều khiển PID
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        # Giới hạn output trong khoảng -35 đến 35 độ
        output = max(self.min_output, min(self.max_output, output))

        # Cập nhật sai số và trả về giá trị điều khiển
        self.prev_error = error
        return output


def stanley_control(e, psi, v, k=1.0):
    """
    Hàm điều khiển Stanley để tính toán góc lái.
    - theta_c: Hướng của xe.
    - k: Hệ số điều khiển.
    Returns:
    - delta: Góc lái cần điều khiển (radian).
    """
    cons = 0.001

    # Tính toán góc lái theo phương pháp Stanley
    delta = psi + np.arctan(k * e / (v + cons)) * (180 / np.pi)
    if delta > 35:
        delta = 35
    elif delta < -35:
        delta = -35
    return delta

def send_motor_servo_control(motor_speed, servo_angle):
    try:
        # Gửi dữ liệu xuống STM32
        send_data = f"M+{motor_speed} S{servo_angle} "
        #ser.write(send_data.encode('utf-8'))
    except Exception as e:
        print(f"[Send] Error: {e}")


def calculate_ct_errors(left, right):
    """
    Tính toán cross track error và heading error dựa trên các phương trình bậc hai cho làn trái và phải.

    Parameters:
    - left_fit, right_fit: Các hệ số phương trình bậc hai cho lane trái và phải.
    - image_height: Chiều cao của ảnh (tính theo pixel).
    - image_width: Chiều rộng của ảnh (tính theo pixel).

    Returns:
    - cross_track_error: Lỗi cross track theo pixel.
    - heading_error: Lỗi heading theo độ.
    """

    # Tính toán tọa độ x của lane trái và phải tại y_eval
    left_lane_x = left[0]
    right_lane_x = right[0]

    # Trung tâm của làn đường
    lane_center_x = (left_lane_x + right_lane_x) / 2

    # Vị trí trung tâm của xe, giả sử xe ở chính giữa ảnh (x = 320)
    car_position_x = 320  # 320 cho ảnh có kích thước 480x640

    # Tính toán cross track error theo pixel
    cross_track_error = lane_center_x - car_position_x

    return cross_track_error

# Đọc encoder từ serial
def read_encoder():
    global encoder_value
#    while True:
#        try:
            #data = ser.readline().decode('utf-8').strip()
#            data = "E0"
#            if data.startswith("E"):
 #               # Lấy giá trị encoder
#                encoder_value = int(data[3:])
#                print(f"Encoder Value: {encoder_value}")
#        except Exception as e:
#            print(f"Error reading encoder: {e}")


def convert_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def HSV_color_selection(image):
    # Chuyển ảnh sang HSV
    converted_image = convert_hsv(image)

    # Mask màu trắng
    lower_threshold = np.uint8([0, 0, 220])
    upper_threshold = np.uint8([255, 30, 255])
    white_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)

    # Mask màu vàng
    lower_threshold = np.uint8([0, 10, 10])
    upper_threshold = np.uint8([90, 255, 255])
    yellow_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)

    # Kết hợp mask trắng và vàng
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    img_gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(img_gray, (7, 7), 0)
    return blurred_img


def nothing(x):
    pass


prevLx = []
prevRx = []
def lane_detection():
    prev_time = time.time()
    while True:

        global lane_offset, curvature, mid, angle_inclination_wrt_x, angle_inclination_wrt_y, prevLx, prevRx,x0,pre_inclination_wrt_y,pre_lane_offset,cur,denta,x_hat,P,step_counter
        success, image = vidcap.read()
        frame = cv2.resize(image, (640, 480))

        ## Choosing points for perspective transformation
        height, width = frame.shape[:2]
        tl = (int(width * 0.75), int(height * 0.45))
        tr = (int(width * 0.25), int(height * 0.45))
        bl = (int(0), int(height))
        br = (int(width), int(height))

        cv2.circle(frame, tl, 5, (0, 0, 255), -1)
        cv2.circle(frame, bl, 5, (0, 0, 255), -1)
        cv2.circle(frame, tr, 5, (0, 0, 255), -1)
        cv2.circle(frame, br, 5, (0, 0, 255), -1)

        ## Aplying perspective transformation
        pts1 = np.float32([bl, br, tl, tr])
        pts2 = np.float32([[0, int(height)], [int(width), int(height)], [int(width), 0], [0, 0]])

        # Matrix to warp the image for birdseye window
        matrix = cv2.getPerspectiveTransform(pts1, pts2)

        transformed_frame = cv2.warpPerspective(frame, matrix, (640, 480))

        ### Object Detection
        # Image Thresholding
        mask = HSV_color_selection(transformed_frame)

        # Histogram
        histogram = np.sum(mask[mask.shape[0] // 2:, :], axis=0)
        midpoint = int(histogram.shape[0] / 2)
        left_base = np.argmax(histogram[:midpoint])
        right_base = np.argmax(histogram[midpoint:]) + midpoint

        # Sliding Window
        y = 472
        lx = []
        rx = []
        slope = 0.0

        msk = mask.copy()

        while y > int(height * 0.55):
            ## Left threshold
            img = mask[y - 35:y, left_base - 30:left_base + 30]
            contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] > 50:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    lx.append(left_base - 30 + cx)
                    cv2.circle(msk, (left_base - 30 + cx, y), 2, (0, 0, 0), -1)
                    left_base = left_base - 30 + cx

            ## Right threshold
            img = mask[y - 35:y, right_base - 30:right_base + 30]
            contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] > 50:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    rx.append(right_base - 30 + cx)
                    right_base = right_base - 30 + cx

            cv2.rectangle(msk, (left_base - 30, y), (left_base + 30, y - 35), (255, 0, 0), 2)
            cv2.rectangle(msk, (right_base - 30, y), (right_base + 30, y - 35), (255, 0, 0), 2)
            y -= 35
        print("leng trai", len(lx), "leng phai", len(rx))

        overlay = transformed_frame.copy()

        # Ensure lx and rx are not empty
        if len(lx) == 0:
            lx = prevLx
        else:
            prevLx = lx
        if len(rx) == 0:
            rx = prevRx
        else:
            prevRx = rx

        if len(lx) < 2 and len(rx) > 2:
            y_eval = 430
            right_points = [(rx[i], 470 - i * 35) for i in range(len(rx))]
            right_fit = np.polyfit([p[1] for p in right_points], [p[0] for p in right_points], 2)
            x_right_vals = np.linspace(min([p[1] for p in right_points]), max([p[1] for p in right_points]), 50)
            right_y_vals = right_fit[0] * x_right_vals ** 2 + right_fit[1] * x_right_vals + right_fit[2]
            right_points_on_image = [(int(y), int(x)) for y, x in zip(right_y_vals, x_right_vals)]

            for i in range(len(right_points_on_image) - 1):
                cv2.line(overlay, right_points_on_image[i], right_points_on_image[i + 1], (0, 165, 255),
                         2)  # Cam cho lane phải
            for i in range(5):
                cur[i] =0.4/ (((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.abs(
                    2 * right_fit[0]) * 0.00062857)
                y_eval += 10  # Adjust y_eval as required

            right_curvature = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.abs(
                2 * right_fit[0]) * 0.00062857
            slope = 2 * right_fit[0] * y_eval + right_fit[1]
            anlge_inclination_wrt_y = - np.arctan(slope) * (180 / np.pi)
            print("slope", slope)
            # print("anlge_inclination_wrt_x", anlge_inclination_wrt_x)
            # if (anlge_inclination_wrt_x >= 0):
            #    anlge_inclination_wrt_y = 90 - anlge_inclination_wrt_x

            # elif (anlge_inclination_wrt_x < 0):
            #    anlge_inclination_wrt_y = - (90 - math.fabs(anlge_inclination_wrt_x))

            car_to_right_lane_distance = math.fabs(rx[0] - 320)
            car_to_lane_center = (530 / 2) - car_to_right_lane_distance
            lane_offset = - car_to_lane_center

        elif len(rx) < 2 and len(lx) > 2:
            y_eval = 430
            left_points = [(lx[i], 470 - i * 35) for i in range(len(lx))]
            left_fit = np.polyfit([p[1] for p in left_points], [p[0] for p in left_points], 2)

            x_left_vals = np.linspace(min([p[1] for p in left_points]), max([p[1] for p in left_points]), 50)
            left_y_vals = left_fit[0] * x_left_vals ** 2 + left_fit[1] * x_left_vals + left_fit[2]
            left_points_on_image = [(int(y), int(x)) for y, x in zip(left_y_vals, x_left_vals)]

            for i in range(len(left_points_on_image) - 1):
                cv2.line(overlay, left_points_on_image[i], left_points_on_image[i + 1], (165, 33, 255),
                         2)  # Xanh lá cho lane trái
            for i in range(5):
                cur[i] = 0.4/(((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.abs(
                2 * left_fit[0]) * 0.00062857)
                y_eval += 10  # Adjust y_eval as required
            left_curvature = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.abs(
                2 * left_fit[0]) * 0.00062857

            slope = 2 * left_fit[0] * y_eval + left_fit[1]
            print("slope", slope)
            anlge_inclination_wrt_y = -np.arctan(slope) * (180 / np.pi)
            # print("anlge_inclination_wrt_x",anlge_inclination_wrt_x)
            # if (anlge_inclination_wrt_x >= 0):
            #    anlge_inclination_wrt_y = 90 - anlge_inclination_wrt_x

            # elif (anlge_inclination_wrt_x < 0):
            #    anlge_inclination_wrt_y = - (90 - math.fabs(anlge_inclination_wrt_x))
            car_to_left_lane_distance = math.fabs(lx[0] - 320)
            car_to_lane_center = (530 / 2) - car_to_left_lane_distance
            lane_offset = car_to_lane_center

        elif len(lx) >= 2 and len(rx) >= 2:
            # Ensure both lx and rx have the same length
            min_length = min(len(lx), len(rx))
            print("min_length   ", min_length)

            # Create a copy of the transformed frame

            # Create the top and bottom points for the quadrilateral
            previus_midpoint = (320, 472)

            midpoints = []  # List to store midpoints

            # Vẽ mid point
            for i in range(min_length):
                if min_length > 1:
                    print("i", i)

                    left = (lx[i], 470 - i * 35)
                    right = (rx[i], 470 - i * 35)

                    cv2.line(overlay, left, right, (0, 255, 0), 1)  # Đường màu xanh lá
                    distance = np.sqrt((right[0] - left[0]) ** 2 + (right[1] - left[1]) ** 2)

                    # Tính trung điểm giữa left và right
                    now_midpoint = ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)

                    # Lưu trung điểm vào danh sách
                    midpoints.append(now_midpoint)

                    # Vẽ điểm trung tâm (màu đỏ) lên overlay
                    cv2.circle(overlay, (int(now_midpoint[0]), int(now_midpoint[1])), 5, (0, 0, 255), -1)

                    # Vẽ đường thẳng từ trung điểm trước đó đến trung điểm hiện tại
                    cv2.line(overlay, (int(previus_midpoint[0]), int(previus_midpoint[1])),
                             (int(now_midpoint[0]), int(now_midpoint[1])), (200, 100, 250), 1)
                    # Cập nhật mid point
                    previus_midpoint = now_midpoint

                    # In ra chiều dài
                    print(f"Distance between is {distance:.2f} pixels")

                    # Draw the filled polygon on the transformed frame
                    alpha = 1  # Opacity factor
                    cv2.addWeighted(overlay, alpha, transformed_frame, 1 - alpha, 0, transformed_frame)

            if len(midpoints) >= 3:
                # Chọn 3 điểm đầu tiên
                points = np.array(midpoints[:len(midpoints)], dtype=np.float32)

                # Sử dụng hàm cv2.fitLine để tìm phương trình đường thẳng
                [vx, vy, x0, y0] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
                print("vy", vy[0])
                print("vx", vx[0])
                # Tính toán phương trình đường thẳng từ các tham số tìm được
                slope = vy[0] / vx[0]  # Độ dốc (slope)

                #intercept = y0 - slope * x0  # Giao điểm với trục y (intercept)

                anlge_inclination_wrt_x = -np.arctan(slope) * (180 / np.pi)
                mid = 80
                if (anlge_inclination_wrt_x >= 0):
                    anlge_inclination_wrt_y = 90 - anlge_inclination_wrt_x
                    # angle = mid - anlge_inclination_wrt_y
                    # steering_angle = angle

                elif (anlge_inclination_wrt_x < 0):
                    anlge_inclination_wrt_y = - (90 - math.fabs(anlge_inclination_wrt_x))
                    # angle = mid - anlge_inclination_wrt_y
                    # steering_angle = angle

            elif len(midpoints) == 2:

                # Chọn 2 điểm đầu tiên
                x1, y1 = midpoints[0]
                x2, y2 = midpoints[1]

                # Tính độ dốc (slope)
                slope = (y2 - y1) / (x2 - x1)

                # Tính giao điểm với trục y (intercept)
                #intercept = y1 - slope * x1

                anlge_inclination_wrt_x = -np.arctan(slope) * (180 / np.pi)
                mid = 80
                if (anlge_inclination_wrt_x >= 0):
                    anlge_inclination_wrt_y = 90 - anlge_inclination_wrt_x
                    # angle = mid - anlge_inclination_wrt_y
                    # steering_angle = angle

                elif (anlge_inclination_wrt_x < 0):
                    anlge_inclination_wrt_y = - (90 - math.fabs(anlge_inclination_wrt_x))
                    # angle = mid - anlge_inclination_wrt_y
                    # steering_angle = angle

                # Vẽ đường thẳng từ phương trình
                # left_x = int(150)  # Điểm bắt đầu (x = 0)
                # left_y = int(slope * left_x + intercept)  # Tính toán y tương ứng

                # right_x = int(440)  # Điểm kết thúc (x = chiều rộng ảnh)
                # right_y = int(slope * right_x + intercept)  # Tính toán y tương ứng

                # Vẽ đường thẳng trên ảnh
                # cv2.line(overlay, (int(left_x), int(left_y)), (int(right_x), int(right_y)), (255, 0, 0), 2)

            left_points = [(lx[i], 470 - i * 35) for i in range(min_length)]
            right_points = [(rx[i], 470 - i * 35) for i in range(min_length)]

            lane_offset = calculate_ct_errors(lx, rx)

            # Khớp một đa thức bậc 2 với các điểm lane trái và phải
            left_fit = np.polyfit([p[1] for p in left_points], [p[0] for p in left_points], 2)
            right_fit = np.polyfit([p[1] for p in right_points], [p[0] for p in right_points], 2)

            # Tạo các giá trị x cho đường trái và phải
            x_left_vals = np.linspace(min([p[1] for p in left_points]), max([p[1] for p in left_points]), 50)
            x_right_vals = np.linspace(min([p[1] for p in right_points]), max([p[1] for p in right_points]), 50)

            # Tính toán y từ các giá trị x dựa trên đa thức bậc 2 đã khớp
            left_y_vals = left_fit[0] * x_left_vals ** 2 + left_fit[1] * x_left_vals + left_fit[2]
            right_y_vals = right_fit[0] * x_right_vals ** 2 + right_fit[1] * x_right_vals + right_fit[2]

            # Chuyển các điểm (x, y) thành tọa độ pixel trên ảnh
            left_points_on_image = [(int(y), int(x)) for y, x in zip(left_y_vals, x_left_vals)]
            right_points_on_image = [(int(y), int(x)) for y, x in zip(right_y_vals, x_right_vals)]

            for i in range(len(left_points_on_image) - 1):
                cv2.line(overlay, left_points_on_image[i], left_points_on_image[i + 1], (165, 33, 255),
                         2)  # Xanh lá cho lane trái
                cv2.line(overlay, right_points_on_image[i], right_points_on_image[i + 1], (0, 165, 255),
                         2)  # Cam cho lane phải

            # Calculate the curvature
            y_eval = 470
            for i in range(5):
                cur[i] = 0.4/((((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.abs(
                    2 * right_fit[0]) * 0.00062857 + ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.abs(
                2 * left_fit[0]) * 0.00062857 )/2)
                y_eval += 10  # Adjust y_eval as required
            left_curvature = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.abs(
                2 * left_fit[0]) *  0.000628
            right_curvature = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.abs(
                2 * right_fit[0])*  0.000628

            print("left_curvature", left_curvature)
            print("right_curvature", right_curvature)
            curvature = (left_curvature + right_curvature) / 2
        else:
            lane_offset = 0
            anlge_inclination_wrt_y = 0

        # Khởi tạo PID controller
        # pid_controller = PID(Kp=800.0, Ki=0.1, Kd=-0.05)
        # denta= pid_controller.update(lane_offset*0.000628 )
        x0 = np.array([
            [lane_offset*0.000528],
            [(lane_offset - pre_lane_offset)*0.000528 / 0.1],
            [np.radians(anlge_inclination_wrt_y)],
            [(np.radians(anlge_inclination_wrt_y) - pre_inclination_wrt_y) / 0.1]
        ])
        print("Debug x0:")
        print(f"Lane Offset: {x0[0, 0]}")
        print(f"Lane Offset Rate: {x0[1, 0]}")
        print(f"Angle Inclination (rad): {x0[2, 0]}")
        print(f"Angle Inclination Rate (rad/s): {x0[3, 0]}")
        x_hat,P =kalman_step(denta,cur[0],x0)
        x0 =x_hat.flatten()
        step_counter += 1
        denta =  get_nearest_control(x0)
        #denta = mpc_control()
        pre_lane_offset =lane_offset
        pre_inclination_wrt_y = np.radians(anlge_inclination_wrt_y)
        # Stanley controler
        #denta = stanley_control(lane_offset * 0.000628, anlge_inclination_wrt_y, 1.7, k=4)

        steering_angle = mid + denta
        send_motor_servo_control(0, steering_angle)

        alpha = 1  # Opacity factor
        cv2.addWeighted(overlay, alpha, transformed_frame, 1 - alpha, 0, transformed_frame)
        # Display the transformed frame with the highlighted lane
        # cv2.imshow("Transformed Frame with Highlighted Lane", overlay)
        #   # Calculate the end point of the line based on the angle
        line_length = 100  # Length of the line
        end_x = int(320 + line_length * np.sin(np.radians(denta)))
        end_y = int(480 - line_length * np.cos(np.radians(denta)))

        # Inverse perspective transformation to map the lanes back to the original image
        inv_matrix = cv2.getPerspectiveTransform(pts2, pts1)
        original_perpective_lane_image = cv2.warpPerspective(transformed_frame, inv_matrix, (640, 480))

        # Combine the original frame with the lane image
        result = cv2.addWeighted(frame, 1, original_perpective_lane_image, 0.5, 0)

        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)  # FPS = 1 / thời gian giữa hai khung hình
        prev_time = current_time  # Cập nhật thời gian của khung hình trước

        # Draw a straight line in the center of the frame pointing with the current angle
        cv2.line(result, (320, 480), (end_x, end_y), (255, 0, 0), 2)
        cv2.line(result, (320, 480), (320, 440), (0, 0, 0), 2)
        # Display the curvature, offset, and angle on the frame
        cv2.putText(result, f'Curvature: {curvature:.2f} m', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result, f'Offset: {-lane_offset*0.00528:.6f} m', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result, f'steering_angle: {steering_angle:.2f} deg', (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)
        cv2.putText(result, f'anlge_inclination_wrt_y: {anlge_inclination_wrt_y:.2f} deg', (30, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result, f'anlge_inclination_wrt_x: {anlge_inclination_wrt_x:.2f} deg', (30, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result, f'denta: {denta:.2f} ', (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result, f'FPS: {fps:.2f}', (30, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # cv2.imshow("Original", frame)
        cv2.imshow("Bird's Eye View", transformed_frame)
        # cv2.imshow("Lane Detection - Image Thresholding", mask)
        cv2.imshow("Lane Detection - Sliding Windows", msk)
        cv2.imshow('Lane Detection', result)

        if cv2.waitKey(10) == 27:
            break

    vidcap.release()
    cv2.destroyAllWindows()

def mpc_control_loop():
    global denta
    while True:

        #denta = mpc_control()  # Gọi hàm MPC Control
        time.sleep(0.1)  # Chờ 0.1 giây trước khi gọi lại

# Tạo luồng để điều khiển mpc



# Tạo luồng để đọc encoder
encoder_thread = threading.Thread(target=read_encoder)
encoder_thread.daemon = True  # Đảm bảo luồng này sẽ kết thúc khi chương trình chính kết thúc
encoder_thread.start()

# Tạo luồng để xử lý lane detection và Stanley control
lane_detection_thread = threading.Thread(target=lane_detection)
lane_detection_thread.daemon = True
lane_detection_thread.start()

mpc_thread = threading.Thread(target=mpc_control_loop)
mpc_thread.daemon = True  # Đảm bảo dừng khi chương trình chính kết thúc
mpc_thread.start()
mpc_init(Ts=0.1)

while True:
    # Thực hiện các công việc trong vòng lặp chính nếu cần
    time.sleep(1)