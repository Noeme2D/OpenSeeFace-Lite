#pragma once

typedef struct {
    int track_fail;
    float translation[3];
    float quaternion[4];
    float eye[2];
    float eye_blink[2];
    float eyebrow_steepness[2];
    float eyebrow_quirk[2];
    float eyebrow_down[2];
    float mouth_corner_down[2];
    float mouth_corner_inout[2];
    float mouth_open;
    float mouth_wide;
} features_t;