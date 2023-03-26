# distutils: language=c++

import numpy as np
cimport numpy as np
from libc.string cimport memcpy

cdef extern from "<math.h>":
    float logf(float x)

cdef extern from "<algorithm>" namespace "std":
    float clamp[float](float v, float lo, float h)

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

# Image normalization constants
mean = np.float32(np.array([0.485, 0.456, 0.406]))
std = np.float32(np.array([0.229, 0.224, 0.225]))
mean = mean / std
std = std * 255.0
mean = -mean
std = 1.0 / std
mean_224 = np.tile(mean, [224, 224, 1])
std_224 = np.tile(std, [224, 224, 1])

cdef public tuple[int] clamp_to_im(float x, float y, unsigned int w, unsigned int h):
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x >= w:
        x = w-1
    if y >= h:
        y = h-1
    
    return (int(x), int(y+1))

cdef public float logit_arr(float p, float factor):
    p = clamp[float](p, 0.0000001, 0.9999999)
    return logf(p / (1-p)) / factor

# https://github.com/solivr/cython_opencvMat/blob/master/opencv_mat.pyx
cdef extern from "Python.h":
    ctypedef struct PyObject
    object PyMemoryView_FromBuffer(Py_buffer *view)
    int PyBuffer_FillInfo(Py_buffer *view, PyObject *obj, void *buf, Py_ssize_t len, int read_only, int infoflags)
    enum:
        PyBUF_FULL_RO

cdef public _preprocess(np_im_t im, int x1, int y1, int x2, int y2):
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
    