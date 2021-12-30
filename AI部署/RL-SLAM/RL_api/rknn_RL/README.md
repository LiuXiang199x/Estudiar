## build

```
mkdir build && cd build
cmake ..
make
make install
```

## install

```
adb push install/rknn_mobilenet_demo /userdata/
```

## run
```
adb shell
cd /userdata/rknn_mobilenet_demo/
./rknn_mobilenet_demo mobilenet_v1.rknnn dog_224x224.jpg
```

## about rl_api
## First: input robot's position with function get_target
get_target(int robot_x, int robot_y);

## Second: get taget position(x,y)
int position_x = get_target_x();
int position_y = get_target_y();

