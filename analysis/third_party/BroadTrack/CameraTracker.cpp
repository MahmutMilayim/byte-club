// BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR DOWNLOAD THE SOFTWARE.
// 
// This is a license agreement ("Agreement") between you (called "Licensee" or "You" in this Agreement) and EVS Broadcast Equipment SA. (called "Licensor" in this Agreement).  All rights not specifically granted to you in this Agreement are reserved for Licensor.
// 
// RESERVATION OF OWNERSHIP AND GRANT OF LICENSE:
// Licensor retains exclusive ownership of any copy of the Software (as defined below) licensed under this Agreement and hereby grants to Licensee a personal, non-exclusive, non-transferable license to use the Software for noncommercial research purposes, without the right to sublicense, pursuant to the terms and conditions of this Agreement.  As used in this Agreement, the term "Software" means (i) the actual copy of all or any portion of code for program routines made accessible to Licensee by Licensor pursuant to this Agreement, inclusive of backups, updates, and/or merged copies permitted hereunder or subsequently supplied by Licensor,  including all or any file structures, programming instructions, user interfaces and screen formats and sequences as well as any and all documentation and instructions related to it, and (ii) all or any derivatives and/or modifications created or made by You to any of the items specified in (i).
// CONFIDENTIALITY: Licensee acknowledges that the Software is proprietary to Licensor, and as such, Licensee agrees to receive all such materials in confidence and use the Software only in accordance with the terms of this Agreement.  Licensee agrees to use reasonable effort to protect the Software from unauthorized use, reproduction, distribution, or publication.
// COPYRIGHT: The Software is owned by Licensor and is protected by copyright laws and applicable international treaties and/or conventions.
// PERMITTED USES:  The Software may be used for your own noncommercial internal research purposes. You understand and agree that Licensor is not obligated to implement any suggestions and/or feedback you might provide regarding the Software, but to the extent Licensor does so, you are not entitled to any compensation related thereto.
// DERIVATIVES: You may create derivatives of or make modifications to the Software, however, You agree that all and any such derivatives and modifications will be owned by Licensor and become a part of the Software licensed to You under this Agreement.  You may only use such derivatives and modifications for your own noncommercial internal research purposes, and you may not otherwise use, distribute or copy such derivatives and modifications in violation of this Agreement.
// BACKUPS:  If Licensee is an organization, it may make that number of copies of the Software necessary for internal noncommercial use at a single site within its organization provided that all information appearing in or on the original labels, including the copyright and trademark notices are copied onto the labels of the copies.
// USES NOT PERMITTED:  You may not distribute, copy or use the Software except as explicitly permitted herein. Licensee has not been granted any trademark license as part of this Agreement and may not use the name or mark "EVS" or any renditions thereof without the prior written permission of Licensor.
// You may not sell, rent, lease, sublicense, lend, time-share or transfer, in whole or in part, or provide third parties access to prior or present versions (or any parts thereof) of the Software.
// ASSIGNMENT: You may not assign this Agreement or your rights hereunder without the prior written consent of Licensor. Any attempted assignment without such consent shall be null and void.
// TERM: The term of the license granted by this Agreement is from Licensee's acceptance of this Agreement by downloading the Software or by using the Software until terminated as provided below.
// The Agreement automatically terminates without notice if you fail to comply with any provision of this Agreement.  Licensee may terminate this Agreement by ceasing using the Software.  Upon any termination of this Agreement, Licensee will delete any and all copies of the Software. You agree that all provisions which operate to protect the proprietary rights of Licensor shall remain in force should breach occur and that the obligation of confidentiality described in this Agreement is binding in perpetuity and, as such, survives the term of the Agreement.
// FEE: Provided Licensee abides completely by the terms and conditions of this Agreement, there is no fee due to Licensor for Licensee's use of the Software in accordance with this Agreement.
// DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON-INFRINGEMENT.  LICENSEE BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND RELATED MATERIALS.
// SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is provided as part of this Agreement. 
// EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent permitted under applicable law, Licensor shall not be liable for direct, indirect, special, incidental, or consequential damages or lost profits related to Licensee's use of and/or inability to use the Software, even if Licensor is advised of the possibility of such damage.
// EXPORT REGULATION: Licensee agrees to comply with any and all applicable export control laws, regulations, and/or other laws related to embargoes and sanction programs.
// SEVERABILITY: If any provision(s) of this Agreement shall be held to be invalid, illegal, or unenforceable by a court or other tribunal of competent jurisdiction, the validity, legality and enforceability of the remaining provisions shall not in any way be affected or impaired thereby.
// NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right or remedy under this Agreement shall be construed as a waiver of any future or other exercise of such right or remedy by Licensor.
// GOVERNING LAW: This Agreement shall be construed and enforced in accordance with the laws of Belgium without reference to conflict of laws principles.  You consent to the exclusive jurisdiction of the courts of Liège.
// ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and entire agreement between Licensee and Licensor as to the matter set forth herein and supersedes any previous agreements, understandings, and arrangements between the parties relating hereto.

#include "CameraTracker.h"
#include "Residuals.h"
#include "LineIoUScore.h"
#include <random>

const double MASK_TO_HD_FACTOR = 2.0;

Point3D closestPointOnSegment(const Point3D &P, const Point3D &A, const Point3D &B)
{
    Point3D AB = B - A;
    Point3D AP = P - A;
    double t = AP.dotProduct(AB) / AB.squaredNorm();
    if (t < 0.0)
    {
        return A;
    }
    else if (t > 1.0)
    {
        return B;
    }
    else
    {
        return A + AB * t;
    }
}

double getCurveParameter(const Point3D &point3D, const Polyline3D &polyline3D)
{
    double minSquaredDistance = std::numeric_limits<double>::max();
    double closestCurvePointParameter;
    double t = 0.0;
    for (int i = 0; i < polyline3D.size() - 1; i++)
    {
        auto segmentStart = polyline3D[i];
        auto segmentEnd = polyline3D[i + 1];
        Point3D closestPoint = closestPointOnSegment(point3D, segmentStart, segmentEnd);
        double sqDist = point3D.squaredDistance(closestPoint);
        if (sqDist < minSquaredDistance)
        {
            minSquaredDistance = sqDist;
            closestCurvePointParameter = t + segmentStart.distance(closestPoint);
        }
        t += segmentStart.distance(segmentEnd);
    }
    return closestCurvePointParameter;
}

CameraTracker::CameraTracker() : _pointExtractor(0.5, 10)
{
    _lambda = 0.;
    _soccerPitch3D = SoccerPitch3D();
}

std::tuple<double, Camera> CameraTracker::update(const cv::Mat &semLinesMask,
                                                 const std::vector<std::pair<Point3D, Point2D>> &pitchProjections,
                                                 bool softPosition,
                                                 bool ptz,
                                                 bool lensDistortion,
                                                 int cauchyParameter)
{
    Point2D principalPoint = _camera.getPrincipalPoint();

    _pointExtractor.setMask(semLinesMask);
    std::map<int, std::vector<cv::Point>> soccerLineMatches;
    _pointExtractor.getExtractedPoints(soccerLineMatches);

    ceres::Problem problem;

    std::array<double, 8> cameraData;
    auto angleAxisVector = _camera.getAngleAxis();
    cameraData[0] = angleAxisVector[0];
    cameraData[1] = angleAxisVector[1];
    cameraData[2] = angleAxisVector[2];
    auto positionVector = _camera.getPosition();
    cameraData[3] = positionVector[0];
    cameraData[4] = positionVector[1];
    cameraData[5] = positionVector[2];
    auto focalLength = _camera.getFocalLength();
    cameraData[6] = focalLength;
    auto radialDistortion = _camera.getRadialDistortion();
    if (radialDistortion.size())
    {
        cameraData[7] = radialDistortion[0];
    }
    else
    {
        cameraData[7] = 0.;
    }
    // Fixed radial
    if (!lensDistortion)
    {
        cameraData[7] = 0.;
    }

    problem.AddParameterBlock(&cameraData[0], 8);
    // No radial
    if (!lensDistortion)
    {
        auto *fixedk1 = new ceres::SubsetParameterization(8, {7});
        problem.SetParameterization(&cameraData[0], fixedk1);
    }

    std::map<SoccerPitch3D::LineID, std::vector<double>> curvePointParameterData;

    for (const auto &labeledLine : soccerLineMatches)
    {
        if (labeledLine.first == SoccerPitch3D::LineID::UNDEFINED_LINE)
        {
            continue;
        }
        auto lineId = static_cast<SoccerPitch3D::LineID>(labeledLine.first);
        auto curveSample = labeledLine.second;
        auto polyLine3D = _soccerPitch3D.getPolyline3D(lineId, 1.0); // 0.01, 0.1
        for (const auto &curvePoint : curveSample)
        {
            auto point3D = _soccerPitch3D.getSurface().intersection(
                _camera.getRay(Point2D(curvePoint.x * MASK_TO_HD_FACTOR, curvePoint.y * MASK_TO_HD_FACTOR)));
            auto t = getCurveParameter(point3D, polyLine3D);
            curvePointParameterData[lineId].emplace_back(t);
        }
    }
    for (const auto &labeledLine : soccerLineMatches)
    {
        auto lineId = static_cast<SoccerPitch3D::LineID>(labeledLine.first);
        auto curveSample = labeledLine.second;
        auto polyLine3D = _soccerPitch3D.getPolyline3D(lineId, 1.0);

        for (int i = 0; i < curveSample.size(); i++)
        {
            auto curvePoint = Point2D(curveSample[i].x * MASK_TO_HD_FACTOR, curveSample[i].y * MASK_TO_HD_FACTOR);
            ceres::CostFunction *reprojectionCostFunction =
                CurvePointReprojectionError::createCostFunction(polyLine3D, curvePoint - principalPoint);
            problem.AddResidualBlock(reprojectionCostFunction,
                                     new ceres::CauchyLoss(cauchyParameter),
                                     &cameraData[0],
                                     &curvePointParameterData.at(lineId)[i]);
        }
    }

    for (const auto &pp : pitchProjections)
    {
        ceres::CostFunction *reprojectionCostFunction =
            FixedPointReprojectionError::createCostFunction(pp.first, pp.second - principalPoint);
        problem.AddResidualBlock(reprojectionCostFunction,
                                 new ceres::CauchyLoss(1.0),
                                 &cameraData[0]);
    }

    if (softPosition)
    {
        auto weightedLossTripod = new ceres::ScaledLoss(nullptr, 150, ceres::TAKE_OWNERSHIP);
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<CameraSoftOpticalAxisConstraintResidual, 1, 8>(
                new CameraSoftOpticalAxisConstraintResidual(_tripodCenter, _tripodRadius)),
            weightedLossTripod,
            &cameraData[0]);
    }

    ceres::Solver::Options options;
    options.num_threads = 32;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    angleAxisVector = {cameraData[0], cameraData[1], cameraData[2]};
    Matrix3x3 orientationMatrix;
    ceres::AngleAxisToRotationMatrix(angleAxisVector.data(), orientationMatrix.data());
    positionVector = {cameraData[3], cameraData[4], cameraData[5]};
    focalLength = cameraData[6];

    radialDistortion = {cameraData[7]};

    _camera.setOrientation(orientationMatrix)
        .setPosition(positionVector)
        .setPrincipalPoint(principalPoint)
        .setFocalLength(focalLength)
        .setRadialDistortion(radialDistortion);

    double outScore = this->evaluate(semLinesMask);

    return std::tuple<double, Camera>(outScore, _camera);
}

double CameraTracker::evaluate(const cv::Mat &semLinesMask)
{

    cv::Mat rawLines = semLinesMask.clone();
    rawLines.setTo(255, rawLines >= 150);
    rawLines.setTo(0, rawLines < 150);
    cv::resize(rawLines, rawLines, cv::Size(960, 540), 1);
    Matrix3x3 H = _camera.getGroundPlaneHomography();
    H = Matrix3x3(1.0 / MASK_TO_HD_FACTOR, 0, 0, 0, 1.0 / MASK_TO_HD_FACTOR, 0, 0, 0, 1).multiply(H);
    double outScore = LineIoUScore(rawLines).evaluateFast(
        H,
        _soccerPitch3D.getLength(),
        _soccerPitch3D.getWidth());
    return outScore;
}

Camera CameraTracker::getCamera() const
{
    return _camera;
}

void CameraTracker::setCamera(const Camera &camera)
{
    _camera = camera;
}

void CameraTracker::setTripodInfo(const Point3D &center, double radius)
{
    _tripodCenter = center;
    _tripodRadius = radius;
}

std::tuple<double, double> CameraTracker::evaluateReprojectionError(const std::vector<std::pair<SoccerPitch3D::PointID,
                                                                                                std::vector<Point2D>>> &points,
                                                                    int threshold,
                                                                    std::vector<bool> &outInliers,
                                                                    const Camera &camera)
{
    std::vector<double> errors;
    std::vector<double> inlierErrors;
    auto inlier = outInliers.begin();
    for (auto pointCorrespondence = points.begin(); pointCorrespondence != points.end() && inlier != outInliers.end(); pointCorrespondence++, inlier++)
    {
        Point3D worldPitchPoint = _soccerPitch3D.getPoint3D(pointCorrespondence->first);
        Point2D imagePoint;
        camera.project(worldPitchPoint, imagePoint);
        int len = pointCorrespondence->second.size();
        if (len == 1)
        {
            double pixelDistance = imagePoint.distance(pointCorrespondence->second[0]);
            errors.push_back(pixelDistance);
            if (pixelDistance <= threshold)
            {
                *inlier = true;
                inlierErrors.push_back(pixelDistance);
            }
            else
            {
                *inlier = false;
            }
        }
    }
    double mean = std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
    double inlierError = std::accumulate(inlierErrors.begin(), inlierErrors.end(), 0.0) / inlierErrors.size();

    return std::tuple<double, double>(mean, inlierError);
}

double focalLengthFromTwoPoints(double a, double b, double c, double d)
{

    double c_2 = pow(c, 2);
    double d_2 = pow(d, 2);
    double t1 = 2 * (d_2 * a * b - c_2);
    double t2 = pow(d_2 * (a + b) - 2 * c, 2) - 4 * (d_2 * a * b - c_2) * (d_2 - 1);

    double f = 1;
    if (t2 <= 0)
        return f;
    assert(t2 >= 0);
    double t3 = 2 * c - d_2 * (a + b) + sqrt(t2);

    if (t3 == 0)
        return f;
    assert(t3 != 0);

    double f2 = t1 / t3;
    if (f2 > 1)
        f = sqrt(f2);

    return f;
}

double estimateFocalLengthFromPositionAndTwoPoints(const std::vector<std::pair<Point3D, Point2D>> &points, Point3D position)
{
    Point3D X1 = points[0].first - position;
    X1.scale(1 / X1.norm());
    Point3D X2 = points[1].first - position;
    X2.scale(1 / X2.norm());
    double d = X1.dotProduct(X2) - 1;

    Point2D x1 = points[0].second;
    x1.scale(1. / 1080.);
    Point2D x2 = points[1].second;
    x2.scale(1. / 1080.);

    double a = x1.dotProduct(x1) - 1;
    double b = x2.dotProduct(x2) - 1;
    double c = x1.dotProduct(x2) - 1;

    double f = focalLengthFromTwoPoints(a, b, c, d);
    if (f > 1)
    {
        return f * 1080;
    }
    return f;
}

std::tuple<double, double> CameraTracker::estimatePanTilt(const std::vector<std::pair<Point3D, Point2D>> &detectedPoints,
                                                          double focal,
                                                          Point3D position,
                                                          size_t iwidth,
                                                          size_t iheight)
{
    Point3D mean_optical_axis(0., 0., 0.);
    for (auto &point : detectedPoints)
    {
        Point3D P = point.first - position;
        mean_optical_axis = mean_optical_axis + P;
    }

    mean_optical_axis.scale(1 / mean_optical_axis.norm());
    double curr_pan = atan2(mean_optical_axis[0], -mean_optical_axis[1]);
    double curr_tilt = atan2(-mean_optical_axis[1], mean_optical_axis[2]);

    Camera camera;

    camera.setPanTiltRoll(Vector3x1(curr_pan, curr_tilt, 0.))
        .setPosition(Vector3x1(position.x(), position.y(), position.z()))
        .setPrincipalPoint(Point2D(iwidth / 2., iheight / 2.))
        .setFocalLength(focal);

    for (int i = 0; i < 5; i++)
    {
        std::vector<double> pans;
        std::vector<double> tilts;
        for (auto &point : detectedPoints)
        {
            Point2D projected;
            camera.project(point.first, projected);
            double dx = point.second.x() - projected.x();
            double dy = projected.y() - point.second.y();
            double dpan = atan2(dx, focal);
            double dtilt = atan2(dy, focal);
            pans.push_back(dpan);
            tilts.push_back(dtilt);
        }
        double meanPan = std::accumulate(pans.begin(), pans.end(), 0.0) / pans.size();
        curr_pan -= meanPan;
        double meanTilt = std::accumulate(tilts.begin(), tilts.end(), 0.0) / tilts.size();
        curr_tilt -= meanTilt;

        camera.setPanTiltRoll(Vector3x1(curr_pan, curr_tilt, 0.))
            .setPosition(Vector3x1(position.x(), position.y(), position.z()))
            .setPrincipalPoint(Point2D(iwidth / 2., iheight / 2.))
            .setFocalLength(focal);
    }

    return std::tie(curr_pan, curr_tilt);
}

void CameraTracker::reinit(const std::vector<std::pair<SoccerPitch3D::PointID, std::vector<Point2D>>> &detectedPoints, int threshold, int n_iterations)
{
    std::vector<bool> outInliers(detectedPoints.size(), false);

    double meanReprojectionError, inlierError;
    std::tie(meanReprojectionError, inlierError) = this->evaluateReprojectionError(detectedPoints, threshold, outInliers, _camera);
    if (meanReprojectionError < threshold || detectedPoints.size() < 2)
    {
        return;
    }

    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0, detectedPoints.size() - 1);
    double bestError = meanReprojectionError;
    Camera bestCamera = _camera;
    cv::Size resolution = _camera.getPixelResolution();

    for (int iter = 0; iter < n_iterations; iter++)
    {
        int id1 = distribution(generator);
        int id2 = distribution(generator);

        std::vector<std::pair<Point3D, Point2D>> candidates;
        candidates.push_back(std::make_pair(_soccerPitch3D.getPoint3D(detectedPoints.at(id1).first), detectedPoints.at(id1).second[0]));
        candidates.push_back(std::make_pair(_soccerPitch3D.getPoint3D(detectedPoints.at(id2).first), detectedPoints.at(id2).second[0]));

        double focal_length = estimateFocalLengthFromPositionAndTwoPoints(candidates, _tripodCenter);
        if (focal_length == 1)
        {
            focal_length = _camera.getFocalLength();
        }

        double pan;
        double tilt;
        std::tie(pan, tilt) = this->estimatePanTilt(candidates, focal_length, _tripodCenter, resolution.width, resolution.height);

        Camera hypothesis;
        hypothesis.setPanTiltRoll(Vector3x1(pan, tilt, 0.))
            .setPosition(Vector3x1(_tripodCenter.x(), _tripodCenter.y(), _tripodCenter.z()))
            .setPrincipalPoint(_camera.getPrincipalPoint())
            .setFocalLength(focal_length);

        double currentReprojectionError;
        std::tie(currentReprojectionError, inlierError) = this->evaluateReprojectionError(detectedPoints, threshold, outInliers, hypothesis);
        int inliersCount = std::count(outInliers.begin(), outInliers.end(), true);

        if (currentReprojectionError < bestError)
        {
            bestError = currentReprojectionError;
            bestCamera = hypothesis;
        }

        if ((inliersCount > 0.5 * detectedPoints.size() && inliersCount > 3 && inlierError < bestError))
        {

            std::vector<std::pair<Point3D, Point2D>> inliers;
            for (int i = 0; i < outInliers.size(); i++)
            {
                if (outInliers[i])
                {
                    inliers.push_back(std::make_pair(_soccerPitch3D.getPoint3D(detectedPoints.at(i).first), detectedPoints.at(id1).second[0]));
                }
            }
            std::tie(pan, tilt) = this->estimatePanTilt(inliers, focal_length, _tripodCenter, resolution.width, resolution.height);

            Camera guidedHypothesis;
            guidedHypothesis.setPanTiltRoll(Vector3x1(pan, tilt, 0.))
                .setPosition(Vector3x1(_tripodCenter.x(), _tripodCenter.y(), _tripodCenter.z()))
                .setPrincipalPoint(_camera.getPrincipalPoint())
                .setFocalLength(focal_length);
            double guidedReprojectionError;
            double guidedInlierError;
            std::tie(guidedReprojectionError, guidedInlierError) = this->evaluateReprojectionError(detectedPoints, threshold, outInliers, guidedHypothesis);
            if (guidedInlierError < inlierError)
            {
                bestError = guidedInlierError;
                bestCamera = guidedHypothesis;
            }
            else
            {
                bestError = inlierError;
                bestCamera = hypothesis;
            }

            std::cout << hypothesis.toJSONString() << std::endl;
        }

        if (currentReprojectionError < threshold)
        {
            break;
        }
    }
    _camera = bestCamera;
}