cd到build目录下执行cmake .. && make，然后到bin目录下执行./main，可以看到打印为空，
接着分别按照下面指令去执行，然后查看打印效果，

cmake .. -DWWW1=ON -DWWW2=OFF && make
cmake .. -DWWW1=OFF -DWWW2=ON && make
cmake .. -DWWW1=ON -DWWW2=ON && make
这里有个小坑要注意下：假设有2个options叫A和B，先调用cmake设置了A，下次再调用cmake去设置B，如果没有删除上次执行cmake时产生的缓存文件，那么这次虽然没设置A，也会默认使用A上次的option值。

所以如果option有变化，要么删除上次执行cmake时产生的缓存文件，要么把所有的option都显式的指定其值。

