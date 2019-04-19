#include <iostream>
#include <string>
#include <fstream>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include "parse_mnist.h"

using namespace std;
using namespace cv;

int main() {
    string training_images = "../mnist/train-images.idx3-ubyte";
    string training_labels = "../mnist/train-labels.idx1-ubyte";
    string test_images = "../mnist/t10k-images.idx3-ubyte";
    string test_labels = "../mnist/t10k-labels.idx1-ubyte";

    int32_t number = 0;

    parse_mnist training_set(training_images, training_labels);
    vector<Mat> vec1;
    vector<unsigned char> lb1;
    training_set.get_all_images_from_mnist(vec1, lb1);
    cout << vec1.size() << endl << lb1.size() << endl;

    return 0;
}