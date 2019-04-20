#include <iostream>
#include "parse_mnist.h"

using namespace std;
using namespace cv;

int main() {
    string training_images = "../mnist/train-images.idx3-ubyte";
    string training_labels = "../mnist/train-labels.idx1-ubyte";
    string test_images = "../mnist/t10k-images.idx3-ubyte";
    string test_labels = "../mnist/t10k-labels.idx1-ubyte";

    /* Initilize training data parser and test data parser */
    parse_mnist train_parse(training_images, training_labels);
    parse_mnist test_parse(test_images, test_labels);
    Ptr<ml::TrainData> training_set;

    /* Get training image and label sets from training files */
    train_parse.get_train_images_with_label_from_mnist(training_set);

    /* Get test image and label sets from test files */
    cv::Mat test_set, test_label_set;
    test_parse.get_test_images_with_label_from_mnist(test_set, test_label_set);

    /* Create a KNN model and train it */
    Ptr<ml::KNearest> knn_model = ml::KNearest::create();

    for (int K_value = 1; K_value <= 30; K_value++) {
        knn_model->setDefaultK(K_value);  // set parameter k for the KNN
        knn_model->setIsClassifier(true);
        knn_model->setAlgorithmType(cv::ml::KNearest::Types::BRUTE_FORCE);
        knn_model->train(training_set, 0); // train this KNN model with training set

        /* Predict result */
        cv::Mat result_set;
        knn_model->findNearest(test_set, knn_model->getDefaultK(), result_set);

        /* Calculate accuracy of the prediction */
        int cnt = 0;
        for (int index = 0; index < test_parse.get_number_of_images(); index++) {
            if (static_cast<int32_t>(result_set.at<float>(index)) == test_label_set.at<int32_t>(index))
                cnt++;
        }

        double accuracy_rate = cnt * 1.0 / test_parse.get_number_of_images();

        cout << "When K is " << knn_model->getDefaultK() << ", Test cases: " << test_parse.get_number_of_images()  << ", Correct cases: " << cnt << ", the accuracy rate is:" << accuracy_rate * 100 << endl;
    }
    return 0;
}