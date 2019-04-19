//
// Created by 永鑫   徐 on 2019-04-19.
//

#include "parse_mnist.h"

bool parse_mnist::get_all_images_from_mnist(std::vector<cv::Mat> &mat_container, std::vector<unsigned char> &lb_container) {
    try {
        if (!mat_container.empty() || !lb_container.empty()) {
            throw "Input Container must be empty!";
        }
        std::fstream image_in, label_in;
        image_in.open(image_path, std::ios::binary | std::ios::in);
        label_in.open(label_path, std::ios::binary | std::ios::in);
        if (!image_in.is_open()) {
            throw "Error opening image data set, please check the path!";
        }
        if (!label_in.is_open()) {
            throw "Error opening label data set, please check the path!";
        }

        image_in.read(reinterpret_cast<char *>(&ii.image_magic_number), sizeof(int));
        image_in.read(reinterpret_cast<char *>(&ii.number_of_images), sizeof(int));
        image_in.read(reinterpret_cast<char *>(&ii.number_of_rows), sizeof(int));
        image_in.read(reinterpret_cast<char *>(&ii.number_of_cols), sizeof(int));
        label_in.read(reinterpret_cast<char *>(&li.label_magic_number), sizeof(int));
        label_in.read(reinterpret_cast<char *>(&li.number_of_labels), sizeof(int));

        if (!high_endian) {
            ii.image_magic_number = swapInt32(ii.image_magic_number);
            ii.number_of_images = swapInt32(ii.number_of_images);
            ii.number_of_rows = swapInt32(ii.number_of_rows);
            ii.number_of_cols = swapInt32(ii.number_of_cols);
            li.label_magic_number = swapInt32(li.label_magic_number);
            li.number_of_labels = swapInt32(li.number_of_labels);
        }

        if (ii.number_of_images != li.number_of_labels) {
            throw "Image and label don't match!";
        }

        mat_container.resize(static_cast<unsigned long>(ii.number_of_images));
        lb_container.resize(static_cast<unsigned long>(li.number_of_labels));


        for (auto i = 0; i < ii.number_of_images; i++) {
            unsigned char block[ii.number_of_cols][ii.number_of_rows];
            unsigned char label;
            image_in.read(reinterpret_cast<char *>(block), sizeof(char) * ii.number_of_cols * ii.number_of_rows);
            label_in.read(reinterpret_cast<char *>(&label), sizeof(char));
            cv::Mat Img(ii.number_of_cols, ii.number_of_rows, CV_8UC1);
            for (int col = 0; col < ii.number_of_cols; col++) {
                for (int row = 0; row < ii.number_of_rows; row++) {
                    Img.at<unsigned char>(col, row) = block[col][row];
                }
            }
            mat_container[i] = Img;
            int32_t reminder = i % 4;
            if (!high_endian)
                lb_container[i / 4 * 4 + 3 - reminder] = label;
            else
                lb_container[i] = label;
        }

        image_in.close();
        label_in.close();
    }
    catch (std::string &str) {
        std::cout << str << std::endl;
        return false;
    }
    return true;
}
