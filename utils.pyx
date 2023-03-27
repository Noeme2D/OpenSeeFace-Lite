# distutils: language=c++

import numpy as np
cimport numpy as np
from libc.string cimport memcpy

# http://makerwannabe.blogspot.com/2013/09/calling-opencv-functions-via-cython.html

ctypedef public np.uint8_t[:,:,:] np_im_t

cdef extern from "opencv2/core/types.hpp" namespace "cv":
    cdef cppclass Size:
        Size(int w, int h) except +

cdef extern from "opencv2/opencv.hpp" namespace "cv":
    cdef int INTER_LINEAR
    void resize(Mat src, Mat dest, Size size, double fx, double fy, int interpol)


cdef extern from "opencv2/core/core.hpp":
    cdef int CV_8UC3

cdef extern from "opencv2/core/core.hpp" namespace "cv":
    cdef cppclass Mat:
        Mat() except +
        void create(int, int, int)
        void* data

# https://github.com/solivr/cython_opencvMat/blob/master/opencv_mat.pyx
cdef extern from "Python.h":
    ctypedef struct PyObject
    object PyMemoryView_FromBuffer(Py_buffer *view)
    int PyBuffer_FillInfo(Py_buffer *view, PyObject *obj, void *buf, Py_ssize_t len, int read_only, int infoflags)
    enum:
        PyBUF_FULL_RO

# Image normalization constants
mean = np.float32(np.array([0.485, 0.456, 0.406]))
std = np.float32(np.array([0.229, 0.224, 0.225]))
mean = mean / std
std = std * 255.0
mean = -mean
std = 1.0 / std
mean_224 = np.tile(mean, [224, 224, 1])
std_224 = np.tile(std, [224, 224, 1])

cdef int res = 224
cdef int res_minus_1 = 223
cdef float out_res = 27
cdef int out_res_i = 28
cdef int logit_factor = 16
cdef int c0 = 66
cdef int c1 = 132
cdef int c2 = 198

cdef int _clamp_to_im(float xf, int w):
    cdef int x = <int>xf
    if x < 0:
        x = 0
    if x >= w:
        x = w-1
    
    return x

def clamp_to_im(pt, w, h):
    return (_clamp_to_im(pt[0], w), _clamp_to_im(pt[1], h) + 1)

cdef _preprocess(np_im_t im, int x1, int y1, int x2, int y2):
    cdef np_im_t cropped = im[y1:y2, x1:x2, ::-1]
    cdef int r = y2-y1
    cdef int c = x2-x1

    cdef np.ndarray[np.uint8_t, ndim=3, mode='c'] cropped_buf = np.ascontiguousarray(cropped, dtype = np.uint8)
    cdef unsigned int* cropped_data = <unsigned int*> cropped_buf.data
    cdef Mat input_mat
    input_mat.create(r, c, CV_8UC3)
    memcpy(input_mat.data, cropped_data, r*c*3)

    cdef Mat output_mat
    resize(input_mat, output_mat, Size(224, 224), 0, 0, INTER_LINEAR)

    cdef Py_buffer output_buf
    PyBuffer_FillInfo(&output_buf, NULL, output_mat.data, 224*224*3, 1, PyBUF_FULL_RO)
    Pydata = PyMemoryView_FromBuffer(&output_buf)
    out = np.asarray(np.ndarray((224, 224, 3), buffer=Pydata, order='c', dtype=np.uint8)) * std_224 + mean_224
    
    reshaped = np.transpose(np.expand_dims(out.copy(), 0), (0,3,1,2))
    return reshaped

def preprocess(im, crop):
    x1, y1, x2, y2 = crop
    return _preprocess(im, x1, y1, x2, y2)

cdef logit_arr(p, float factor):
    p = np.clip(p, 0.0000001, 0.9999999)
    return np.log(p / (1 - p)) / factor

def landmarks(tensor, crop_info):
    crop_x1, crop_y1, scale_x, scale_y, _ = crop_info
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
    lms = np.stack([t_x, t_y, t_conf], 1)
    lms[np.isnan(lms).any(axis=1)] = np.array([0.,0.,0.], dtype=np.float32)
   
    return (np.average(t_conf), np.array(lms))

cdef _matrix_to_quaternion(np.ndarray[np.float32_t, ndim=2] m):
    cdef float t = 0
    cdef float q[4]
    cdef float[:] q_view = q

    if m[2,2] < 0:
        if m[0,0] > m[1,1]:
            t = 1 + m[0,0] - m[1,1] - m[2,2]
            q = [t, m[0,1]+m[1,0], m[2,0]+m[0,2], m[1,2]-m[2,1]]
        else:
            t = 1 - m[0,0] + m[1,1] - m[2,2]
            q = [m[0,1]+m[1,0], t, m[1,2]+m[2,1], m[2,0]-m[0,2]]
    else:
        if m[0,0] < -m[1,1]:
            t = 1 - m[0,0] - m[1,1] + m[2,2]
            q = [m[2,0]+m[0,2], m[1,2]+m[2,1], t, m[0,1]-m[1,0]]
        else:
            t = 1 + m[0,0] + m[1,1] + m[2,2]
            q = [m[1,2]-m[2,1], m[2,0]-m[0,2], m[0,1]-m[1,0], t]
    
    return np.asarray(q_view) * 0.5 / np.sqrt(t)

def matrix_to_quaternion(m):
    return _matrix_to_quaternion(m)
