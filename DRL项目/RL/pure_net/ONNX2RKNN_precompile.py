import torch
from rknn.api import RKNN

ONNX_MODEL = "rknn_sin_precompile/ckpt.635.rknn"
RKNN_MODEL = "rknn_con_precompile/ckpt.635.rknn"
 
if __name__ == '__main__':

    # Create RKNN object
	rknn = RKNN()

	# pre-process config
	print("---> condig model")

	# Load rknn model
	print('--> Loading RKNN model')
	ret = rknn.load_rknn(ONNX_MODEL)
	if ret != 0:
		print('Load RKNN model failed!')
		exit(ret)
	print('done')

	# Init runtime environment
	print('--> Init runtime environment')

	# Note: you must set rknn2precompile=True when call rknn.init_runtime()
	#       RK3399Pro with android system does not support this function.
	ret = rknn.init_runtime(target='rv1126', rknn2precompile=True)
	if ret != 0:
		print('Init runtime environment failed')
		exit(ret)
	print('done')

	ret = rknn.export_rknn_precompile_model(RKNN_MODEL)

	rknn.release()
