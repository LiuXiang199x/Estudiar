import torch
import numpy
from rknn.api import RKNN


def main():

    # import model
    rknn = RKNN()

    # wihtout quantification
    ret = rknn.load_rknn('test_model.rknn')

    # with quantification
    # rknn.load_rknn('./centerface_quantization_1088_1920.rknn')
    # ret = rknn.init_runtime(target='rk1808', target_sub_class='AICS')
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # get inputs (8,G,G)
    inputs = torch.randn(1, 8, 240, 240)
    map3 = torch.zeros(1, 240, 240)
    map3[0][20][20] = 1.
    inputs[0][3] = map3
    inputs[0][7] = map3
    inputs = numpy.array(inputs)    

    # init environment
    print(">======== Init runtime environment ========<")
    ret = rknn.init_runtime()
    if ret != 0:
        print("<======== Init runtime environment failed! ========>")
        exit(ret)
    print("======== Init environment done =========")

    # Inference
    print(">======== Running RKNN model ==========<")
    outputs = rknn.inference(inputs=[inputs])
    print(outputs)
    print(">======== Inference done =========<")


    rknn.release()

if __name__ == "__main__":
    main()
