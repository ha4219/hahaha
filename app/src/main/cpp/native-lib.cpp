#include <jni.h>
#include <string>

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

#include <iostream>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <android/log.h>

using namespace std;
using namespace dlib;
using namespace cv;

template <long num_filters, typename SUBNET> using con5d = con<num_filters, 5, 5, 2, 2, SUBNET>;
template <long num_filters, typename SUBNET> using con5  = con<num_filters, 5, 5, 1, 1, SUBNET>;

template <typename SUBNET> using downsampler  = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16, SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5  = relu<affine<con5<45, SUBNET>>>;

using net_type = loss_mmod<con<1, 9, 9, 1, 1, rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

AAssetManager *mgr;

static net_type net;

static shape_predictor sp;
static matrix<rgb_alpha_pixel> glasses, mustache;
static rgb_alpha_pixel a;


extern "C"
JNIEXPORT void JNICALL
Java_com_example_myapplication_MainActivity_ConvertRGBtoGray(JNIEnv *env, jobject instance,
                                                             jlong matAddrInput,
                                                             jlong matAddrResult) {

    // TODO
    Mat &matInput = *(Mat *)matAddrInput;
    Mat &matResult = *(Mat *)matAddrResult;

    cvtColor(matInput, matResult, COLOR_RGBA2GRAY);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_myapplication_MainActivity_processing(JNIEnv *env, jobject instance,
                                                       jlong matAddrInput, jlong matAddrResult) {

    // TODO
    Mat &matInput = *(Mat *)matAddrInput;
    Mat &matResult = *(Mat *)matAddrResult;

    pyramid_up(glasses);
    pyramid_up(mustache);

    /*
     학습되지 않은 이미지를 합성하기 위해
     cv로 이미지를 읽어와서 dlib용 행렬로 변환
    */
    //CV2Matrix(mustache);
    array2d<rgb_pixel> gl;

    // Argument로 들어온 image들에 대해 작업(수염 + 안경)
    matrix<rgb_pixel> img;
    cv_image<rgb_pixel> tmp(matInput);
    assign_image(img, tmp);

    // Image upsampling
    // Face detection acc 올라가는 듯
    pyramid_up(img);

    auto dets = net(img);
    // 강아지 얼굴의 landmark 찾아와서 선으로 이어주는 작업
    // shape_predictor가 찾아줌
    // Landmark 찾기

    for(auto d = dets.begin(); d != dets.end(); ++d){
        auto shape = sp(img, d->rect);

        const rgb_pixel color(0, 255, 0);
        auto top = shape.part(0);
        auto lear = shape.part(1);
        auto leye = shape.part(2);
        auto nose = shape.part(3);
        auto rear = shape.part(4);
        auto reye = shape.part(5);

        // 수염의 양쪽 끝 위치 계산
        auto lmustache = 1.3 * (leye - reye) / 2 + nose;
        auto rmustache = 1.3 * (reye - leye) / 2 + nose;

        // 이미지 상에 안경 그리기
        std::vector<point> from = {2 * point(176, 36), 2 * point(59, 35)}, to = {leye, reye};
        auto tform = find_similarity_transform(from, to);
        for (long r = 0; r < glasses.nr(); ++r) {
            for (long c = 0; c < glasses.nc(); ++c) {
                point p = tform(point(c, r));
                if (get_rect(img).contains(p))
                    assign_pixel((rgb_pixel &) img(p.y(), p.x()), glasses(r, c));
            }
        }

        // 이미지 상에 수염 그리기 >> 코 우측 하단
        auto mrect = get_rect(mustache);
        from = {mrect.tl_corner(), mrect.tr_corner()};
        to = {rmustache, lmustache};
        tform = find_similarity_transform(from, to);
        for (long r = 0; r < mustache.nr(); ++r) {
            for (long c = 0; c < mustache.nc(); ++c) {
                point p = tform(point(c, r));
                if (get_rect(img).contains(p))
                    assign_pixel((rgb_pixel &) img(p.y(), p.x()), mustache(r, c));
            }
        }
    }
    cvtColor(toMat(img), matResult, COLOR_RGB2BGR);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_myapplication_MainActivity_loadModel(JNIEnv *env, jobject instance,
                                                      jobject assetManager) {
    // TODO
    try{
        deserialize("/data/user/0/com.example.myapplication/files/mmod_dog_hipsterizer.dat") >> net >> sp >> glasses >> mustache;
    }catch (exception &e){
        __android_log_print(ANDROID_LOG_DEBUG, "native-lib :: ",
                           "exception thrown! %s",  e.what() );
    }
    __android_log_print(ANDROID_LOG_DEBUG, "native-lib :: ",
                        "test");
}