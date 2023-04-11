#include "typedefs.h"

#include "osflite.h"

#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define IMAGE_SIZE (640 * 480 * 3)

int main() {
    int err = PyImport_AppendInittab("osflite", PyInit_osflite);
    if (err) {
        printf("PyImport failed.\n");
        return -1;
    }
    Py_Initialize();
    PyImport_ImportModule("osflite");

    FILE *fd = fopen("frame.raw", "rb");
    char *img = malloc(IMAGE_SIZE);
    fread(img, 1, IMAGE_SIZE, fd);
    fclose(fd);

    features_t feature;
    clock_t osf_start, osf_end;
    osf_start = clock();

    run_osf(img, &feature);

    osf_end = clock();
    printf("Time taken by OSF: %f\n", ((float)(osf_end - osf_start)) / CLOCKS_PER_SEC);

    if (feature.track_fail) {
        printf("Face tracking failure.\n");
    } else {
        printf("eye_l: %f\n", feature.eye[0]);
    }

    Py_Finalize();
    free(img);
    return 0;
}