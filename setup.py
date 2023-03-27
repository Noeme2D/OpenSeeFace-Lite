from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize(
        Extension(
            "utils",
            sources = [
                "utils.pyx",
            ],
            include_dirs=[
                np.get_include(),
                "/usr/include/opencv4/"
            ],
            extra_compile_args=[
                "-O3"
            ],
            # -L${libdir} -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_flann
            extra_link_args=[
                "-L/usr/lib",
                "-lopencv_core",
                "-lopencv_imgproc"
            ]
        ),
        compiler_directives={
            'language_level': "3",
            'wraparound': False,
            'boundscheck': False,
            'cdivision': True
        },
        annotate = True,
    )
)
