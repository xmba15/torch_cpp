/**
 * @file    VisualizePoseGroundTruth.cpp
 *
 * @author  btran
 *
 */

#include <iostream>

#include <pangolin/pangolin.h>  // eigen headers need to be put before cv2 eigen headers

#include <opencv2/core/eigen.hpp>

#include "Utils.hpp"

namespace
{
using Pose = Eigen::Isometry3d;
using Poses = std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>;

Eigen::Isometry3d toEigenIsometry3d(const cv::Affine3d& affine3d)
{
    Eigen::Isometry3d output = Eigen::Isometry3d::Identity();

    Eigen::Matrix3d eigenRotationMat;
    cv::cv2eigen(affine3d.rotation(), eigenRotationMat);
    output.linear() = eigenRotationMat;
    output.translation() = Eigen::Vector3d::Map(affine3d.translation().val);

    return output;
}

void drawPoses(const Poses& poses);
}  // namespace

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: [app] [path/to/pose/ground/truth]" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string PATH_TO_POSE_GT = argv[1];
    std::vector<cv::Affine3d> poses = _cv::parseKittiOdometryPosesGT(PATH_TO_POSE_GT);

    Poses eigenPoses;
    std::transform(poses.begin(), poses.end(), std::back_inserter(eigenPoses),
                   [](const auto& pose) { return toEigenIsometry3d(pose); });
    drawPoses(eigenPoses);

    return EXIT_SUCCESS;
}

namespace
{
void drawPoses(const Poses& poses)
{
    if (poses.empty()) {
        return;
    }

    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState sCam(pangolin::ProjectionMatrix(1924, 768, 500, 500, 512, 389, 0.1, 1000),
                                     pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0, -1, 0));

    pangolin::View& dCam =
        pangolin::CreateDisplay().SetBounds(0, 1, 0, 1, -1024. / 768).SetHandler(new pangolin::Handler3D(sCam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        dCam.Activate(sCam);
        glClearColor(1., 1., 1., 1.);
        glLineWidth(2);

        for (const auto& pose : poses) {
            Eigen::Vector3d Ow = pose.translation();
            Eigen::Vector3d Xw = pose * (0.1 * Eigen::Vector3d(1, 0, 0));
            Eigen::Vector3d Yw = pose * (0.1 * Eigen::Vector3d(0, 1, 0));
            Eigen::Vector3d Zw = pose * (0.1 * Eigen::Vector3d(0, 0, 1));
            glBegin(GL_LINES);
            glColor3f(1., 0., 0.);
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glVertex3d(Xw[0], Xw[1], Xw[2]);
            glColor3f(0., 1., 0.);
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glVertex3d(Yw[0], Yw[1], Yw[2]);
            glColor3f(0., 0., 1.);
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glVertex3d(Zw[0], Zw[1], Zw[2]);
            glEnd();
        }

        for (std::size_t i = 0; i < poses.size() - 1; ++i) {
            glColor3f(0.0, 0.0, 0.0);
            glBegin(GL_LINES);
            const Pose& p1 = poses[i];
            const Pose& p2 = poses[i + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }
        pangolin::FinishFrame();
        std::this_thread::sleep_for(std::chrono::microseconds(5000));
    }
}
}  // namespace
