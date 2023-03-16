import numpy as np
import cv2
import time
import onnxruntime

# Input

frame = cv2.imread("test-frame.jpg")
width = frame.shape[0]
height = frame.shape[1]

# Input

# Image normalization constants

mean = np.float32(np.array([0.485, 0.456, 0.406]))
std = np.float32(np.array([0.229, 0.224, 0.225]))
mean = mean / std
std = std * 255.0
mean = -mean
std = 1.0 / std
mean_224 = np.tile(mean, [224, 224, 1])
std_224 = np.tile(std, [224, 224, 1])

# Image normalization constants

# Settings

detection_threshold = 0.6
model_type = 2

res = 224.
mean_res = mean_224
std_res = std_224
if model_type < 0:
    res = 56.
    mean_res = np.tile(mean, [56, 56, 1])
    std_res = np.tile(std, [56, 56, 1])
if model_type < -1:
    res = 112.
    mean_res = np.tile(mean, [112, 112, 1])
    std_res = np.tile(std, [112, 112, 1])

out_res = 27.
if model_type < 0:
    out_res = 6.
if model_type < -1:
    out_res = 13.
out_res_i = int(out_res) + 1

logit_factor = 16.
if model_type < 0:
    logit_factor = 8.
if model_type < -1:
    logit_factor = 16.

# Settings

# Models setup

options = onnxruntime.SessionOptions()
options.inter_op_num_threads = 1
options.intra_op_num_threads = 1
options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
options.log_severity_level = 3
providersList = onnxruntime.capi._pybind_state.get_available_providers()

detection_model = onnxruntime.InferenceSession("models/mnv3_detection_opt.onnx", sess_options=options, providers=providersList)

model_names = [
    "lm_model0_opt.onnx",
    "lm_model1_opt.onnx",
    "lm_model2_opt.onnx",
    "lm_model3_opt.onnx",
    "lm_model4_opt.onnx"
]
model_name = "lm_modelT_opt.onnx"
if model_type >= 0:
    model_name = model_names[model_type]
if model_type == -2:
    model_name = "lm_modelV_opt.onnx"
if model_type == -3:
    model_name = "lm_modelU_opt.onnx"

lm_model = onnxruntime.InferenceSession("models/"+model_name, sess_options=options, providers=providersList)
lm_input_name = lm_model.get_inputs()[0].name 

# Models setup

# Utilities

def visualize(frame, face):
    height = frame.shape[0]
    width = frame.shape[1]

    visual = np.copy(frame)

    for pt_num, (x, y, c) in enumerate(face.lms):
        x = int(x + 0.5)
        y = int(y + 0.5)
        color = (0, 255, 0)

        if pt_num >= 66:
            color = (255, 255, 0)
        if not (x < 0 or y < 0 or x >= height or y >= width):
            visual[int(x), int(y)] = color
        x += 1
        if not (x < 0 or y < 0 or x >= height or y >= width):
            visual[int(x), int(y)] = color
        y += 1
        if not (x < 0 or y < 0 or x >= height or y >= width):
            visual[int(x), int(y)] = color
        x -= 1
        if not (x < 0 or y < 0 or x >= height or y >= width):
            visual[int(x), int(y)] = color

        projected = cv2.projectPoints(face.contour, face.rotation, face.translation, tracker.camera, tracker.dist_coeffs)     
        for [(x,y)] in projected[0]:
            x = int(x + 0.5)
            y = int(y + 0.5)
            if not (x < 0 or y < 0 or x >= height or y >= width):
                frame[int(x), int(y)] = (0, 255, 255)
            x += 1
            if not (x < 0 or y < 0 or x >= height or y >= width):
                frame[int(x), int(y)] = (0, 255, 255)
            y += 1
            if not (x < 0 or y < 0 or x >= height or y >= width):
                frame[int(x), int(y)] = (0, 255, 255)
            x -= 1
            if not (x < 0 or y < 0 or x >= height or y >= width):
                frame[int(x), int(y)] = (0, 255, 255)
    
    cv2.imshow("OpenSeeFace-Lite Visualization", visual)
    cv2.waitKey(0)

def clamp_to_im(pt, w, h):
    x = pt[0]
    y = pt[1]
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x >= w:
        x = w-1
    if y >= h:
        y = h-1
    return (int(x), int(y+1))

def preprocess(im, crop):
    x1, y1, x2, y2 = crop
    im = np.float32(im[y1:y2, x1:x2,::-1]) # Crop and BGR to RGB
    im = cv2.resize(im, (int(res), int(res)), interpolation=cv2.INTER_LINEAR) * std_res + mean_res
    im = np.expand_dims(im, 0)
    im = np.transpose(im, (0,3,1,2))
    return im

def logit_arr(p, factor=16.0):
    p = np.clip(p, 0.0000001, 0.9999999)
    return np.log(p / (1 - p)) / float(factor)

def landmarks(tensor, crop_info):
    crop_x1, crop_y1, scale_x, scale_y, _ = crop_info
    avg_conf = 0
    res_minus_1 = res - 1
    c0, c1, c2 = 66, 132, 198
    if model_type == -1:
        c0, c1, c2 = 30, 60, 90
    t_main = tensor[0:c0].reshape((c0,out_res_i * out_res_i))
    t_m = t_main.argmax(1)
    indices = np.expand_dims(t_m, 1)
    t_conf = np.take_along_axis(t_main, indices, 1).reshape((c0,))
    t_off_x = np.take_along_axis(tensor[c0:c1].reshape((c0,out_res_i * out_res_i)), indices, 1).reshape((c0,))
    t_off_y = np.take_along_axis(tensor[c1:c2].reshape((c0,out_res_i * out_res_i)), indices, 1).reshape((c0,))
    t_off_x = res_minus_1 * logit_arr(t_off_x, logit_factor)
    t_off_y = res_minus_1 * logit_arr(t_off_y, logit_factor)
    t_x = crop_y1 + scale_y * (res_minus_1 * np.floor(t_m / out_res_i) / out_res + t_off_x)
    t_y = crop_x1 + scale_x * (res_minus_1 * np.floor(np.mod(t_m, out_res_i)) / out_res + t_off_y)
    avg_conf = np.average(t_conf)
    lms = np.stack([t_x, t_y, t_conf], 1)
    lms[np.isnan(lms).any(axis=1)] = np.array([0.,0.,0.], dtype=np.float32)
    if model_type == -1:
        lms = lms[[0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,6,7,7,8,8,9,10,10,11,11,12,21,21,21,22,23,23,23,23,23,13,14,14,15,16,16,17,18,18,19,20,20,24,25,25,25,26,26,27,27,27,24,24,28,28,28,26,29,29,29]]
        part_avg = np.mean(np.partition(lms[:,2],3)[0:3])
        if part_avg < 0.65:
            avg_conf = part_avg
    return (avg_conf, np.array(lms))

# Utilties

# Core

def detect_face(frame):
    im = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)[:,:,::-1] * std_224 + mean_224
    im = np.expand_dims(im, 0)
    im = np.transpose(im, (0,3,1,2))

    outputs, maxpool = detection_model.run([], {'input': im})
    outputs = np.array(outputs)
    maxpool = np.array(maxpool)
    outputs[0, 0, outputs[0, 0] != maxpool[0, 0]] = 0
    detections = np.flip(np.argsort(outputs[0,0].flatten()))
    det = detections[0]

    y, x = det // 56, det % 56
    c = outputs[0, 0, y, x]
    r = outputs[0, 1, y, x] * 112.
    x *= 4
    y *= 4
    r *= 1.0
    if c < detection_threshold:
        assert False

    result = np.array((x - r, y - r, 2 * r, 2 * r * 1.0)).astype(np.float32)
    result[[0,2]] *= frame.shape[1] / 224.
    result[[1,3]] *= frame.shape[0] / 224.
    
    return result

def lm(frame, face):
    x,y,w,h = face
    crop_x1 = x - int(w * 0.1)
    crop_y1 = y - int(h * 0.125)
    crop_x2 = x + w + int(w * 0.1)
    crop_y2 = y + h + int(h * 0.125)
    crop_x1, crop_y1 = clamp_to_im((crop_x1, crop_y1), width, height)
    crop_x2, crop_y2 = clamp_to_im((crop_x2, crop_y2), width, height)
    scale_x = float(crop_x2 - crop_x1) / res
    scale_y = float(crop_y2 - crop_y1) / res
    crop_0 = preprocess(frame, (crop_x1, crop_y1, crop_x2, crop_y2))
    crop_info_0 = (crop_x1, crop_y1, scale_x, scale_y, 0.1)

    output = lm_model.run([], {lm_input_name: crop_0})[0]

    # tracker.py:1151
    eye_state = [(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)]

    conf, lms = landmarks(output[0], crop_info_0)

    x1, y1, _ = lms.min(0)
    x2, y2, _ = lms.max(0)
    bb = (x1, y1, x2 - x1, y2 - y1)
    output = (conf, lms, 0, bb)

    # continue from tracker.py:1262

    return output

# Core

# Main

start_time = time.perf_counter()

face = detect_face(frame)
lm(frame, face)

end_time = time.perf_counter()
print("Tracker time used:", end_time - start_time)

# assert len(faces) == 1
# assert faces[0].success

# visualize(frame, faces[0])
