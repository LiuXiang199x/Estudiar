#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <sys/io.h>

using namespace std;
using namespace cv;




//读取指定文件下的所有图片
vector<Mat> read_images_in_folder(cv::String pattern)
{
	vector<cv::String> fn;
	glob(pattern, fn, false);

	vector<Mat> images;
	// vector<cv::String>&prefix  //
	size_t count = fn.size(); //number of png files in images folder
	for (size_t i = 0; i < count; i++)
	{
	    // prefix.push_back(fn[i].substr(20, 4)); // 此处可以得到文件名的子字符串，可以获取图片前缀
		images.push_back(imread(fn[i])); //直读取图片并返回Mat类型
		//imshow("img", imread(fn[i]));
		//waitKey(1000);
	}
	return images;
}

int main()
{

	cv::String pattern = "/home/agent/val/toilet_room/*.jpg";

	//遍历得到目标文件中所有的.jpg文件
	vector<Mat> images = read_images_in_folder(pattern);
    /*
	for (int i = 0; i < images.size(); i++)
	{
		imshow("img", images[i]);
		waitKey(1000);
	}
    */
   	string pattern_tif = "/home/agent/val/toilet_room/*.jpg";//要遍历文件的路径及文件类型

	vector<cv::String> image_files;
	glob(pattern_tif, image_files,false);//三个参数分别为要遍历的文件夹地址；结果的存储引用；是否递归查找，默认为false
    for (int i = 0; i< image_files.size(); i++)//image_file.size()代表文件中总共的图片个数
    {
        cout << image_files.at(i) << endl;
    }

   
    return 0;
}
