import onnxruntime as ort

def optimize_onnx(in_filename):
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess_options.optimized_model_filepath = "optimized_" + in_filename

    session = ort.InferenceSession(in_filename, sess_options)

if __name__== "__main__":
    optimize_onnx("detection_model.onnx")
    optimize_onnx("lm_model2.onnx")