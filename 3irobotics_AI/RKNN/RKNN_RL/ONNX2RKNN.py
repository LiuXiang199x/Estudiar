from rknn.api import RKNN

ONNX_MODEL = "test_conv_pool.onnx"
RKNN_MODEL = "test_model.rknn"
 
if __name__ == '__main__':

    # Create RKNN object
	rknn = RKNN()

	# pre-process config
	print("---> condig model")

	# 1806-regnet
	# rknn.config(mean_values=[[123.675, 116.28, 103.53]], std_values=[[58.395, 57.12, 57.375]], reoder_channel="0 1 2")

	# Load models	
	print("---> Loading model")
	ret_RL = rknn.load_onnx(model=ONNX_MODEL)
	if ret_RL != 0:
		print("<======= Load RL model failed! =======>") 
		exit(ret_RL)
	print("========= Loading Done =========")

	# Build models
	print("<======== Building RL model =========>")
	ret_RL = rknn.build(do_quantization=False)
	if ret_RL != 0:
		print("<======= Build RL model failed! =======>")
		exit(ret_RL)
	print("========= Building Done =========")

	# Export rknn model
	print("<========= Export RKNN model =========>")
	ret_RL = rknn.export_rknn(RKNN_MODEL)
	if ret_RL != 0:
		print("<======= Build RL model failed! =======>") 
		exit(ret_RL)
	print("========= Export Done =========")
	
	# Release RKNN Context
	rknn.release()

