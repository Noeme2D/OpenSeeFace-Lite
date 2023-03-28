# distutils: language=c++

import numpy as np
cimport numpy as np

cdef extern from "<cmath>":
    float cos(float x)
    float sin(float x)
    cdef float M_PI

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

cdef int MODEL_RES = 224

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

face_3d = np.array([
    [ 0.4551769692672  ,  0.300895790030204, -0.764429433974752],
    [ 0.448998827123556,  0.166995837790733, -0.765143004071253],
    [ 0.437431554952677,  0.022655479179981, -0.739267175112735],
    [ 0.415033422928434, -0.088941454648772, -0.747947437846473],
    [ 0.389123587370091, -0.232380029794684, -0.704788385327458],
    [ 0.334630113904382, -0.361265387599081, -0.615587579236862],
    [ 0.263725112132858, -0.460009725616771, -0.491479221041573],
    [ 0.16241621322721 , -0.558037146073869, -0.339445180872282],
    [ 0.               , -0.621079019321682, -0.287294770748887],
    [-0.16241621322721 , -0.558037146073869, -0.339445180872282],
    [-0.263725112132858, -0.460009725616771, -0.491479221041573],
    [-0.334630113904382, -0.361265387599081, -0.615587579236862],
    [-0.389123587370091, -0.232380029794684, -0.704788385327458],
    [-0.415033422928434, -0.088941454648772, -0.747947437846473],
    [-0.437431554952677,  0.022655479179981, -0.739267175112735],
    [-0.448998827123556,  0.166995837790733, -0.765143004071253],
    [-0.4551769692672  ,  0.300895790030204, -0.764429433974752],
    [ 0.385529968662985,  0.402800553948697, -0.310031082540741],
    [ 0.322196658344302,  0.464439136821772, -0.250558059367669],
    [ 0.25409760441282 ,  0.46420381416882 , -0.208177722146526],
    [ 0.186875436782135,  0.44706071961879 , -0.145299823706503],
    [ 0.120880983543622,  0.423566314072968, -0.110757158774771],
    [-0.120880983543622,  0.423566314072968, -0.110757158774771],
    [-0.186875436782135,  0.44706071961879 , -0.145299823706503],
    [-0.25409760441282 ,  0.46420381416882 , -0.208177722146526],
    [-0.322196658344302,  0.464439136821772, -0.250558059367669],
    [-0.385529968662985,  0.402800553948697, -0.310031082540741],
    [ 0.               ,  0.293332603215811, -0.137582088779393],
    [ 0.               ,  0.194828701837823, -0.069158109325951],
    [ 0.               ,  0.103844017393155, -0.009151819844964],
    [ 0.               ,  0.               ,  0.               ],
    [ 0.080626352317973, -0.041276068128093, -0.134161035564826],
    [ 0.046439347377934, -0.057675223874769, -0.102990627164664],
    [ 0.               , -0.068753126205604, -0.090545348482397],
    [-0.046439347377934, -0.057675223874769, -0.102990627164664],
    [-0.080626352317973, -0.041276068128093, -0.134161035564826],
    [ 0.315905195966084,  0.298337502555443, -0.285107407636464],
    [ 0.275252345439353,  0.312721904921771, -0.244558251170671],
    [ 0.176394511553111,  0.311907184376107, -0.219205360345231],
    [ 0.131229723798772,  0.284447361805627, -0.234239149487417],
    [ 0.184124948330084,  0.260179585304867, -0.226590776513707],
    [ 0.279433549294448,  0.267363071770222, -0.248441437111633],
    [-0.131229723798772,  0.284447361805627, -0.234239149487417],
    [-0.176394511553111,  0.311907184376107, -0.219205360345231],
    [-0.275252345439353,  0.312721904921771, -0.244558251170671],
    [-0.315905195966084,  0.298337502555443, -0.285107407636464],
    [-0.279433549294448,  0.267363071770222, -0.248441437111633],
    [-0.184124948330084,  0.260179585304867, -0.226590776513707],
    [ 0.121155252430729, -0.208988660580347, -0.160606287940521],
    [ 0.041356305910044, -0.194484199722098, -0.096159882202821],
    [ 0.               , -0.205180167345702, -0.083299217789729],
    [-0.041356305910044, -0.194484199722098, -0.096159882202821],
    [-0.121155252430729, -0.208988660580347, -0.160606287940521],
    [-0.132325402795928, -0.290857984604968, -0.187067868218105],
    [-0.064137791831655, -0.325377847425684, -0.158924039726607],
    [ 0.               , -0.343742581679188, -0.113925986025684],
    [ 0.064137791831655, -0.325377847425684, -0.158924039726607],
    [ 0.132325402795928, -0.290857984604968, -0.187067868218105],
    [ 0.181481567104525, -0.243239316141725, -0.231284988892766],
    [ 0.083999507750469, -0.239717753728704, -0.155256465640701],
    [ 0.               , -0.256058040176369, -0.0950619498899  ],
    [-0.083999507750469, -0.239717753728704, -0.155256465640701],
    [-0.181481567104525, -0.243239316141725, -0.231284988892766],
    [-0.074036069749345, -0.250689938345682, -0.177346470406188],
    [ 0.               , -0.264945854681568, -0.112349967428413],
    [ 0.074036069749345, -0.250689938345682, -0.177346470406188],
    # Pupils and eyeball centers
    [ 0.257990002632141,  0.276080012321472, -0.219998998939991],
    [-0.257990002632141,  0.276080012321472, -0.219998998939991],
    [ 0.257990002632141,  0.276080012321472, -0.324570998549461],
    [-0.257990002632141,  0.276080012321472, -0.324570998549461]
], np.float32)
base_scale_v = face_3d[27:30, 1] - face_3d[28:31, 1]
base_scale_h = np.abs(face_3d[[0, 36, 42], 0] - face_3d[[16, 39, 45], 0])

cdef int _clamp_to_im(float xf, int w):
    cdef int x = <int>xf
    if x < 0:
        x = 0
    if x >= w:
        x = w-1
    
    return x

def clamp_to_im(pt, w, h):
    return (_clamp_to_im(pt[0], w), _clamp_to_im(pt[1], h) + 1)

cdef resize_for_model(np_im_t im):
    cdef int r = im.shape[0]
    cdef int c = im.shape[1]
    cdef np.ndarray[np.uint8_t, ndim=3, mode='c'] buf = np.ascontiguousarray(im[:,:,::-1], dtype = np.uint8)

    cdef Mat input_mat
    input_mat.create(r, c, CV_8UC3)
    input_mat.data = <unsigned char *>buf.data

    cdef Mat output_mat
    resize(input_mat, output_mat, Size(MODEL_RES, MODEL_RES), 0, 0, INTER_LINEAR)

    cdef Py_buffer output_buf
    PyBuffer_FillInfo(&output_buf, NULL, output_mat.data, MODEL_RES*MODEL_RES*3, 1, PyBUF_FULL_RO)
    Pydata = PyMemoryView_FromBuffer(&output_buf)
    out = np.asarray(np.ndarray((MODEL_RES, MODEL_RES, 3), buffer=Pydata, order='c', dtype=np.uint8)) * std_224 + mean_224
    
    return out

cdef _preprocess(np_im_t im, int x1, int y1, int x2, int y2):
    cdef np_im_t cropped = im[y1:y2, x1:x2, ::-1]
    cdef int r = y2-y1
    cdef int c = x2-x1

    resized = resize_for_model(cropped)
    
    reshaped = np.transpose(np.expand_dims(resized, 0), (0,3,1,2))
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

cdef float angle(float x1, float y1, float x2, float y2):
    return np.arctan2(y2-y1, x2-x1) % (2*M_PI)

def normalize_pts3d(pts_3d):
    # Calculate angle using nose
    pts_3d[:, 0:2] -= pts_3d[30, 0:2]
    alpha = angle(pts_3d[30][0], pts_3d[30][1], pts_3d[27][0], pts_3d[27][1])
    alpha -= 0.5 * M_PI

    R = [[cos(alpha), -sin(alpha)], [sin(alpha), cos(alpha)]]
    pts_3d[:, 0:2] = (pts_3d - pts_3d[30])[:, 0:2].dot(R) + pts_3d[30, 0:2]

    # Vertical scale
    pts_3d[:, 1] /= np.mean((pts_3d[27:30, 1] - pts_3d[28:31, 1]) / base_scale_v)

    # Horizontal scale
    pts_3d[:, 0] /= np.mean(np.abs(pts_3d[[0, 36, 42], 0] - pts_3d[[16, 39, 45], 0]) / base_scale_h)

    return pts_3d