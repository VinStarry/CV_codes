//
// Created by 永鑫   徐 on 2019-04-19.
//

#ifndef KNN_PARSE_MNIST_H
#define KNN_PARSE_MNIST_H

#include <iostream>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <set>
#include <list>

struct image_info {
    image_info() = default;
    ~image_info() = default;

    int32_t image_magic_number;
    int32_t number_of_images;
    int32_t number_of_rows;
    int32_t number_of_cols;
};

struct label_info {
    label_info() = default;
    ~label_info() = default;

    int32_t label_magic_number;
    int32_t number_of_labels;
};

class parse_mnist {
public:
    parse_mnist(std::string &image_file, std::string &label_file):
            image_path(image_file), label_path(label_file), ii(), li(){
        unsigned short v = 0x0102;
        auto *p = reinterpret_cast<unsigned char *>(&v);
        high_endian = (*p == 0x01);
    }

    ~parse_mnist() = default;

    bool get_all_images_from_mnist(std::vector<cv::Mat> &mat_container, std::vector<unsigned char> &lb_container);

    int32_t get_image_magic_number() const { return ii.image_magic_number; }
    int32_t get_label_magic_numberr() const { return li.label_magic_number; }
    int32_t get_number_of_images() const { return ii.number_of_images; }
    int32_t get_number_of_rows() const { return ii.number_of_rows; }
    int32_t get_number_of_cols() const { return ii.number_of_cols; }
    int32_t get_number_of_labels() const { return li.number_of_labels; }

    int32_t swapInt32(int32_t value) {
        return ((value & 0x000000FF) << 24) |
               ((value & 0x0000FF00) << 8) |
               ((value & 0x00FF0000) >> 8) |
               ((value & 0xFF000000) >> 24) ;
    }

private:
    const std::string image_path;
    const std::string label_path;
    image_info ii;
    label_info li;
    bool high_endian;
};

#endif //KNN_PARSE_MNIST_H
