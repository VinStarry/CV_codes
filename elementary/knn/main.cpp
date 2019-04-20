#include <iostream>
#include "parse_mnist.h"

using namespace std;
using namespace cv;

int main() {
    string training_images = "../mnist/train-images.idx3-ubyte";
    string training_labels = "../mnist/train-labels.idx1-ubyte";
    string test_images = "../mnist/t10k-images.idx3-ubyte";
    string test_labels = "../mnist/t10k-labels.idx1-ubyte";

    /* initilize training data parser */
    parse_mnist train_parse(training_images, training_labels);
    Ptr<ml::TrainData> training_set;
    /* get training image and label sets from files */
    train_parse.get_all_images_from_mnist(training_set);



    return 0;
}