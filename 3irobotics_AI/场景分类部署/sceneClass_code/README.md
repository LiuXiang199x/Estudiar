## build

```
mkdir build && cd build
cmake ..
make
make install
```

## install

```
adb push install/sceneClass /userdata/
```

## others
```
replace aa.h with everest/ai.h
Final output of room type  ---> roomType
Public func: runSceneNet(float *Nanodet_res) --> [float_, ..]
```

```
LoadParams.h -> 加载模型权重
Scenen.h -> 主要头文件(继承LoadParams)
replace aa.h with everest/ai.h
调用接口: int SceneNet::runSceneNet(cv::Mat& src, float *Nanodet_res)  --->返回房间类型 int
```

