/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include "Frame.h"

#include "G2oTypes.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "ORBextractor.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include "GeometricCamera.h"

#include <thread>
#include <include/CameraModels/Pinhole.h>
#include <include/CameraModels/KannalaBrandt8.h>
#include <time.h>

namespace ORB_SLAM3
{
double EdgeTime = 0;
double EdgeTimeNum = 0;
long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

//For stereo fisheye matching
cv::BFMatcher Frame::BFmatcher = cv::BFMatcher(cv::NORM_HAMMING);

Frame::Frame(): mpcpi(NULL), mpImuPreintegrated(NULL), mpPrevFrame(NULL), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false), mbHasPose(false), mbHasVelocity(false)
{
#ifdef REGISTER_TIMES
    mTimeStereoMatch = 0;
    mTimeORB_Ext = 0;
#endif
}


//Copy Constructor
Frame::Frame(const Frame &frame)
    :mpcpi(frame.mpcpi),mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mK_(Converter::toMatrix3f(frame.mK)), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn), mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mImuCalib(frame.mImuCalib), mnCloseMPs(frame.mnCloseMPs),
     mpImuPreintegrated(frame.mpImuPreintegrated), mpImuPreintegratedFrame(frame.mpImuPreintegratedFrame), mImuBias(frame.mImuBias),
     mnId(frame.mnId), mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors), mNameFile(frame.mNameFile), mnDataset(frame.mnDataset),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2), mpPrevFrame(frame.mpPrevFrame), mpLastKeyFrame(frame.mpLastKeyFrame),
     mbIsSet(frame.mbIsSet), mbImuPreintegrated(frame.mbImuPreintegrated), mpMutexImu(frame.mpMutexImu),
     mpCamera(frame.mpCamera), mpCamera2(frame.mpCamera2), Nleft(frame.Nleft), Nright(frame.Nright),
     monoLeft(frame.monoLeft), monoRight(frame.monoRight), mvLeftToRightMatch(frame.mvLeftToRightMatch),
     mvRightToLeftMatch(frame.mvRightToLeftMatch), mvStereo3Dpoints(frame.mvStereo3Dpoints),
     mTlr(frame.mTlr), mRlr(frame.mRlr), mtlr(frame.mtlr), mTrl(frame.mTrl),
     mTcw(frame.mTcw), mbHasPose(false), mbHasVelocity(false), imgMono(frame.imgMono), EdgeImage(frame.EdgeImage), DTImage(frame.DTImage), mValidEdgePixel(frame.mValidEdgePixel),
     mValidEdgePixelUn(frame.mValidEdgePixelUn), mvpEdgePoints(frame.mvpEdgePoints), status_e(frame.status_e), gradX(frame.gradX), gradY(frame.gradY),
     locationX(frame.locationX), locationY(frame.locationY),mbEdgeOutlier(frame.mbEdgeOutlier)
{
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++){
            mGrid[i][j]=frame.mGrid[i][j];
            if(frame.Nleft > 0){
                mGridRight[i][j] = frame.mGridRight[i][j];
            }
        }

    if(frame.mbHasPose)
        SetPose(frame.GetPose());

    if(frame.HasVelocity())
    {
        SetVelocity(frame.GetVelocity());
    }

    mmProjectPoints = frame.mmProjectPoints;
    mmMatchedInImage = frame.mmMatchedInImage;

#ifdef REGISTER_TIMES
    mTimeStereoMatch = frame.mTimeStereoMatch;
    mTimeORB_Ext = frame.mTimeORB_Ext;
#endif
}


Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera, Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL), mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false),
     mpCamera(pCamera) ,mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft,0,0);
    thread threadRight(&Frame::ExtractORB,this,1,imRight,0,0);
    threadLeft.join();
    threadRight.join();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif

    N = mvKeys.size();
    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartStereoMatches = std::chrono::steady_clock::now();
#endif
    ComputeStereoMatches();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndStereoMatches = std::chrono::steady_clock::now();

    mTimeStereoMatch = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndStereoMatches - time_StartStereoMatches).count();
#endif

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);
    mmProjectPoints.clear();
    mmMatchedInImage.clear();


    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);



        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    if(pPrevF)
    {
        if(pPrevF->HasVelocity())
            SetVelocity(pPrevF->GetVelocity());
    }
    else
    {
        mVw.setZero();
    }

    mpMutexImu = new std::mutex();

    //Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
    monoLeft = -1;
    monoRight = -1;

    AssignFeaturesToGrid();
}

Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera,Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL),mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false),
     mpCamera(pCamera),mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    ExtractORB(0,imGray,0,0);

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif


    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoFromRGBD(imDepth);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));

    mmProjectPoints.clear();
    mmMatchedInImage.clear();

    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    if(pPrevF){
        if(pPrevF->HasVelocity())
            SetVelocity(pPrevF->GetVelocity());
    }
    else{
        mVw.setZero();
    }

    mpMutexImu = new std::mutex();

    //Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
    monoLeft = -1;
    monoRight = -1;

    AssignFeaturesToGrid();
}


Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, GeometricCamera* pCamera, cv::Mat &distCoef, const float &bf, const float &thDepth, Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL),mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(static_cast<Pinhole*>(pCamera)->toK()), mK_(static_cast<Pinhole*>(pCamera)->toK_()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL),mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false), mpCamera(pCamera),
     mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false)
{
    // Frame ID
    mnId=nNextId++;
    imgMono = imGray.clone();
    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    thread threadLeft(&Frame::ExtractORB,this,0,imGray,0,1000);
    thread threadRight(&Frame::ExtractEdgePixels,this,10,10);
    // thread threadRight(&Frame::ExtractEdgePixels_,this,100);
    
    threadLeft.join();
    threadRight.join();
    // ExtractORB(0,imGray,0,1000);
    // ExtractEdge(500);
    // cout<<mValidEdgePixel.size()<<endl;
    mvpEdgePoints = vector<EdgePoint*>((int)mValidEdgePixel.size(), static_cast<EdgePoint*>(NULL));
    UndistortEdgePixels();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif


    N = mvKeys.size();
    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    // Set no stereo information
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);
    mnCloseMPs = 0;

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));

    mmProjectPoints.clear();// = map<long unsigned int, cv::Point2f>(N, static_cast<cv::Point2f>(NULL));
    mmMatchedInImage.clear();

    mvbOutlier = vector<bool>(N,false);
    mbEdgeOutlier = vector<bool>((int)mValidEdgePixel.size(), false);
    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = static_cast<Pinhole*>(mpCamera)->toK().at<float>(0,0);
        fy = static_cast<Pinhole*>(mpCamera)->toK().at<float>(1,1);
        cx = static_cast<Pinhole*>(mpCamera)->toK().at<float>(0,2);
        cy = static_cast<Pinhole*>(mpCamera)->toK().at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }


    mb = mbf/fx;

    //Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
    monoLeft = -1;
    monoRight = -1;

    AssignFeaturesToGrid();

    if(pPrevF)
    {
        if(pPrevF->HasVelocity())
        {
            SetVelocity(pPrevF->GetVelocity());
        }
    }
    else
    {
        mVw.setZero();
    }

    mpMutexImu = new std::mutex();
}


void Frame::AssignFeaturesToGrid()
{
    // Fill matrix with points
    const int nCells = FRAME_GRID_COLS*FRAME_GRID_ROWS;

    int nReserve = 0.5f*N/(nCells);

    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++){
            mGrid[i][j].reserve(nReserve);
            if(Nleft != -1){
                mGridRight[i][j].reserve(nReserve);
            }
        }



    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = (Nleft == -1) ? mvKeysUn[i]
                                                 : (i < Nleft) ? mvKeys[i]
                                                                 : mvKeysRight[i - Nleft];

        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY)){
            if(Nleft == -1 || i < Nleft)
                mGrid[nGridPosX][nGridPosY].push_back(i);
            else
                mGridRight[nGridPosX][nGridPosY].push_back(i - Nleft);
        }
    }
}

void Frame::ExtractORB(int flag, const cv::Mat &im, const int x0, const int x1)
{
    vector<int> vLapping = {x0,x1};
    clock_t s = clock();
    if(flag==0)
        monoLeft = (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors,vLapping);
    else
        monoRight = (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight,vLapping);
    // cout<<"Exact ORB:" << (double)(clock()-s)/CLOCKS_PER_SEC<<endl;
}

bool Frame::isSet() const {
    return mbIsSet;
}

void Frame::SetPose(const Sophus::SE3<float> &Tcw) {
    mTcw = Tcw;

    UpdatePoseMatrices();
    mbIsSet = true;
    mbHasPose = true;
}

void Frame::SetNewBias(const IMU::Bias &b)
{
    mImuBias = b;
    if(mpImuPreintegrated)
        mpImuPreintegrated->SetNewBias(b);
}

void Frame::SetVelocity(Eigen::Vector3f Vwb)
{
    mVw = Vwb;
    mbHasVelocity = true;
}

Eigen::Vector3f Frame::GetVelocity() const
{
    return mVw;
}

void Frame::SetImuPoseVelocity(const Eigen::Matrix3f &Rwb, const Eigen::Vector3f &twb, const Eigen::Vector3f &Vwb)
{
    mVw = Vwb;
    mbHasVelocity = true;

    Sophus::SE3f Twb(Rwb, twb);
    Sophus::SE3f Tbw = Twb.inverse();

    mTcw = mImuCalib.mTcb * Tbw;

    UpdatePoseMatrices();
    mbIsSet = true;
    mbHasPose = true;
}

void Frame::UpdatePoseMatrices()
{
    Sophus::SE3<float> Twc = mTcw.inverse();
    mRwc = Twc.rotationMatrix();
    mOw = Twc.translation();
    mRcw = mTcw.rotationMatrix();
    mtcw = mTcw.translation();
}

Eigen::Matrix<float,3,1> Frame::GetImuPosition() const {
    return mRwc * mImuCalib.mTcb.translation() + mOw;
}

Eigen::Matrix<float,3,3> Frame::GetImuRotation() {
    return mRwc * mImuCalib.mTcb.rotationMatrix();
}

Sophus::SE3<float> Frame::GetImuPose() {
    return mTcw.inverse() * mImuCalib.mTcb;
}

Sophus::SE3f Frame::GetRelativePoseTrl()
{
    return mTrl;
}

Sophus::SE3f Frame::GetRelativePoseTlr()
{
    return mTlr;
}

Eigen::Matrix3f Frame::GetRelativePoseTlr_rotation(){
    return mTlr.rotationMatrix();
}

Eigen::Vector3f Frame::GetRelativePoseTlr_translation() {
    return mTlr.translation();
}


bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    if(Nleft == -1){
        pMP->mbTrackInView = false;
        pMP->mTrackProjX = -1;
        pMP->mTrackProjY = -1;

        // 3D in absolute coordinates
        Eigen::Matrix<float,3,1> P = pMP->GetWorldPos();

        // 3D in camera coordinates
        const Eigen::Matrix<float,3,1> Pc = mRcw * P + mtcw;
        const float Pc_dist = Pc.norm();

        // Check positive depth
        const float &PcZ = Pc(2);
        const float invz = 1.0f/PcZ;
        if(PcZ<0.0f)
            return false;

        const Eigen::Vector2f uv = mpCamera->project(Pc);

        if(uv(0)<mnMinX || uv(0)>mnMaxX)
            return false;
        if(uv(1)<mnMinY || uv(1)>mnMaxY)
            return false;

        pMP->mTrackProjX = uv(0);
        pMP->mTrackProjY = uv(1);

        // Check distance is in the scale invariance region of the MapPoint
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const Eigen::Vector3f PO = P - mOw;
        const float dist = PO.norm();

        if(dist<minDistance || dist>maxDistance)
            return false;

        // Check viewing angle
        Eigen::Vector3f Pn = pMP->GetNormal();

        const float viewCos = PO.dot(Pn)/dist;

        if(viewCos<viewingCosLimit)
            return false;

        // Predict scale in the image
        const int nPredictedLevel = pMP->PredictScale(dist,this);

        // Data used by the tracking
        pMP->mbTrackInView = true;
        pMP->mTrackProjX = uv(0);
        pMP->mTrackProjXR = uv(0) - mbf*invz;

        pMP->mTrackDepth = Pc_dist;

        pMP->mTrackProjY = uv(1);
        pMP->mnTrackScaleLevel= nPredictedLevel;
        pMP->mTrackViewCos = viewCos;

        return true;
    }
    else{
        pMP->mbTrackInView = false;
        pMP->mbTrackInViewR = false;
        pMP -> mnTrackScaleLevel = -1;
        pMP -> mnTrackScaleLevelR = -1;

        pMP->mbTrackInView = isInFrustumChecks(pMP,viewingCosLimit);
        pMP->mbTrackInViewR = isInFrustumChecks(pMP,viewingCosLimit,true);

        return pMP->mbTrackInView || pMP->mbTrackInViewR;
    }
}

bool Frame::ProjectPointDistort(MapPoint* pMP, cv::Point2f &kp, float &u, float &v)
{

    // 3D in absolute coordinates
    Eigen::Vector3f P = pMP->GetWorldPos();

    // 3D in camera coordinates
    const Eigen::Vector3f Pc = mRcw * P + mtcw;
    const float &PcX = Pc(0);
    const float &PcY= Pc(1);
    const float &PcZ = Pc(2);

    // Check positive depth
    if(PcZ<0.0f)
    {
        cout << "Negative depth: " << PcZ << endl;
        return false;
    }

    // Project in image and check it is not outside
    const float invz = 1.0f/PcZ;
    u=fx*PcX*invz+cx;
    v=fy*PcY*invz+cy;

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    float u_distort, v_distort;

    float x = (u - cx) * invfx;
    float y = (v - cy) * invfy;
    float r2 = x * x + y * y;
    float k1 = mDistCoef.at<float>(0);
    float k2 = mDistCoef.at<float>(1);
    float p1 = mDistCoef.at<float>(2);
    float p2 = mDistCoef.at<float>(3);
    float k3 = 0;
    if(mDistCoef.total() == 5)
    {
        k3 = mDistCoef.at<float>(4);
    }

    // Radial distorsion
    float x_distort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
    float y_distort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);

    // Tangential distorsion
    x_distort = x_distort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x));
    y_distort = y_distort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y);

    u_distort = x_distort * fx + cx;
    v_distort = y_distort * fy + cy;


    u = u_distort;
    v = v_distort;

    kp = cv::Point2f(u, v);

    return true;
}

Eigen::Vector3f Frame::inRefCoordinates(Eigen::Vector3f pCw)
{
    return mRcw * pCw + mtcw;
}

vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel, const bool bRight) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    float factorX = r;
    float factorY = r;

    const int nMinCellX = max(0,(int)floor((x-mnMinX-factorX)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
    {
        return vIndices;
    }

    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+factorX)*mfGridElementWidthInv));
    if(nMaxCellX<0)
    {
        return vIndices;
    }

    const int nMinCellY = max(0,(int)floor((y-mnMinY-factorY)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
    {
        return vIndices;
    }

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+factorY)*mfGridElementHeightInv));
    if(nMaxCellY<0)
    {
        return vIndices;
    }

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = (!bRight) ? mGrid[ix][iy] : mGridRight[ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = (Nleft == -1) ? mvKeysUn[vCell[j]]
                                                         : (!bRight) ? mvKeys[vCell[j]]
                                                                     : mvKeysRight[vCell[j]];
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<factorX && fabs(disty)<factorY)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}


void Frame::ComputeBoW()
{
    if(mBowVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

void Frame::UndistortKeyPoints()
{
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(N,2,CV_32F);

    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat, static_cast<Pinhole*>(mpCamera)->toK(),mDistCoef,cv::Mat(),mK);
    mat=mat.reshape(1);


    // Fill undistorted keypoint vector
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }

}

void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if(mDistCoef.at<float>(0)!=0.0)
    {
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,static_cast<Pinhole*>(mpCamera)->toK(),mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        // Undistort corners
        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));
    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

void Frame::ComputeStereoMatches()
{
    mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);

    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;

    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    //Assign keypoints to row table
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

    const int Nr = mvKeysRight.size();

    for(int iR=0; iR<Nr; iR++)
    {
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }

    // Set limits for search
    const float minZ = mb;
    const float minD = 0;
    const float maxD = mbf/minZ;

    // For each left keypoint search a match in the right image
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);

    for(int iL=0; iL<N; iL++)
    {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        const vector<size_t> &vCandidates = vRowIndices[vL];

        if(vCandidates.empty())
            continue;

        const float minU = uL-maxD;
        const float maxU = uL-minD;

        if(maxU<0)
            continue;

        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        const cv::Mat &dL = mDescriptors.row(iL);

        // Compare descriptor to right keypoints
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

            const float &uR = kpR.pt.x;

            if(uR>=minU && uR<=maxU)
            {
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
            const float scaleduL = round(kpL.pt.x*scaleFactor);
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);

            // sliding window search
            const int w = 5;
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);

            int bestDist = INT_MAX;
            int bestincR = 0;
            const int L = 5;
            vector<float> vDists;
            vDists.resize(2*L+1);

            const float iniu = scaleduR0+L-w;
            const float endu = scaleduR0+L+w+1;
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            for(int incR=-L; incR<=+L; incR++)
            {
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);

                float dist = cv::norm(IL,IR,cv::NORM_L1);
                if(dist<bestDist)
                {
                    bestDist =  dist;
                    bestincR = incR;
                }

                vDists[L+incR] = dist;
            }

            if(bestincR==-L || bestincR==L)
                continue;

            // Sub-pixel match (Parabola fitting)
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

            float disparity = (uL-bestuR);

            if(disparity>=minD && disparity<maxD)
            {
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                mvDepth[iL]=mbf/disparity;
                mvuRight[iL] = bestuR;
                vDistIdx.push_back(pair<int,int>(bestDist,iL));
            }
        }
    }

    sort(vDistIdx.begin(),vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size()/2].first;
    const float thDist = 1.5f*1.4f*median;

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}


void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    for(int i=0; i<N; i++)
    {
        const cv::KeyPoint &kp = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        const float d = imDepth.at<float>(v,u);

        if(d>0)
        {
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x-mbf/d;
        }
    }
}

bool Frame::UnprojectStereo(const int &i, Eigen::Vector3f &x3D)
{
    const float z = mvDepth[i];
    if(z>0) {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        Eigen::Vector3f x3Dc(x, y, z);
        x3D = mRwc * x3Dc + mOw;
        return true;
    } else
        return false;
}

bool Frame::imuIsPreintegrated()
{
    unique_lock<std::mutex> lock(*mpMutexImu);
    return mbImuPreintegrated;
}

void Frame::setIntegrated()
{
    unique_lock<std::mutex> lock(*mpMutexImu);
    mbImuPreintegrated = true;
}

Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera, GeometricCamera* pCamera2, Sophus::SE3f& Tlr,Frame* pPrevF, const IMU::Calib &ImuCalib)
        :mpcpi(NULL), mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)),  mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
         mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbImuPreintegrated(false), mpCamera(pCamera), mpCamera2(pCamera2),
         mbHasPose(false), mbHasVelocity(false)

{
    imgLeft = imLeft.clone();
    imgRight = imRight.clone();

    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft,static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[0],static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[1]);
    thread threadRight(&Frame::ExtractORB,this,1,imRight,static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[0],static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[1]);
    threadLeft.join();
    threadRight.join();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif

    Nleft = mvKeys.size();
    Nright = mvKeysRight.size();
    N = Nleft + Nright;

    if(N == 0)
        return;

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf / fx;

    // Sophus/Eigen
    mTlr = Tlr;
    mTrl = mTlr.inverse();
    mRlr = mTlr.rotationMatrix();
    mtlr = mTlr.translation();

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartStereoMatches = std::chrono::steady_clock::now();
#endif
    ComputeStereoFishEyeMatches();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndStereoMatches = std::chrono::steady_clock::now();

    mTimeStereoMatch = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndStereoMatches - time_StartStereoMatches).count();
#endif

    //Put all descriptors in the same matrix
    cv::vconcat(mDescriptors,mDescriptorsRight,mDescriptors);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(nullptr));
    mvbOutlier = vector<bool>(N,false);

    AssignFeaturesToGrid();

    mpMutexImu = new std::mutex();

    UndistortKeyPoints();

}

void Frame::ComputeStereoFishEyeMatches() {
    //Speed it up by matching keypoints in the lapping area
    vector<cv::KeyPoint> stereoLeft(mvKeys.begin() + monoLeft, mvKeys.end());
    vector<cv::KeyPoint> stereoRight(mvKeysRight.begin() + monoRight, mvKeysRight.end());

    cv::Mat stereoDescLeft = mDescriptors.rowRange(monoLeft, mDescriptors.rows);
    cv::Mat stereoDescRight = mDescriptorsRight.rowRange(monoRight, mDescriptorsRight.rows);

    mvLeftToRightMatch = vector<int>(Nleft,-1);
    mvRightToLeftMatch = vector<int>(Nright,-1);
    mvDepth = vector<float>(Nleft,-1.0f);
    mvuRight = vector<float>(Nleft,-1);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(Nleft);
    mnCloseMPs = 0;

    //Perform a brute force between Keypoint in the left and right image
    vector<vector<cv::DMatch>> matches;

    BFmatcher.knnMatch(stereoDescLeft,stereoDescRight,matches,2);

    int nMatches = 0;
    int descMatches = 0;

    //Check matches using Lowe's ratio
    for(vector<vector<cv::DMatch>>::iterator it = matches.begin(); it != matches.end(); ++it){
        if((*it).size() >= 2 && (*it)[0].distance < (*it)[1].distance * 0.7){
            //For every good match, check parallax and reprojection error to discard spurious matches
            Eigen::Vector3f p3D;
            descMatches++;
            float sigma1 = mvLevelSigma2[mvKeys[(*it)[0].queryIdx + monoLeft].octave], sigma2 = mvLevelSigma2[mvKeysRight[(*it)[0].trainIdx + monoRight].octave];
            float depth = static_cast<KannalaBrandt8*>(mpCamera)->TriangulateMatches(mpCamera2,mvKeys[(*it)[0].queryIdx + monoLeft],mvKeysRight[(*it)[0].trainIdx + monoRight],mRlr,mtlr,sigma1,sigma2,p3D);
            if(depth > 0.0001f){
                mvLeftToRightMatch[(*it)[0].queryIdx + monoLeft] = (*it)[0].trainIdx + monoRight;
                mvRightToLeftMatch[(*it)[0].trainIdx + monoRight] = (*it)[0].queryIdx + monoLeft;
                mvStereo3Dpoints[(*it)[0].queryIdx + monoLeft] = p3D;
                mvDepth[(*it)[0].queryIdx + monoLeft] = depth;
                nMatches++;
            }
        }
    }
}

bool Frame::isInFrustumChecks(MapPoint *pMP, float viewingCosLimit, bool bRight) {
    // 3D in absolute coordinates
    Eigen::Vector3f P = pMP->GetWorldPos();

    Eigen::Matrix3f mR;
    Eigen::Vector3f mt, twc;
    if(bRight){
        Eigen::Matrix3f Rrl = mTrl.rotationMatrix();
        Eigen::Vector3f trl = mTrl.translation();
        mR = Rrl * mRcw;
        mt = Rrl * mtcw + trl;
        twc = mRwc * mTlr.translation() + mOw;
    }
    else{
        mR = mRcw;
        mt = mtcw;
        twc = mOw;
    }

    // 3D in camera coordinates
    Eigen::Vector3f Pc = mR * P + mt;
    const float Pc_dist = Pc.norm();
    const float &PcZ = Pc(2);

    // Check positive depth
    if(PcZ<0.0f)
        return false;

    // Project in image and check it is not outside
    Eigen::Vector2f uv;
    if(bRight) uv = mpCamera2->project(Pc);
    else uv = mpCamera->project(Pc);

    if(uv(0)<mnMinX || uv(0)>mnMaxX)
        return false;
    if(uv(1)<mnMinY || uv(1)>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const Eigen::Vector3f PO = P - twc;
    const float dist = PO.norm();

    if(dist<minDistance || dist>maxDistance)
        return false;

    // Check viewing angle
    Eigen::Vector3f Pn = pMP->GetNormal();

    const float viewCos = PO.dot(Pn) / dist;

    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    const int nPredictedLevel = pMP->PredictScale(dist,this);

    if(bRight){
        pMP->mTrackProjXR = uv(0);
        pMP->mTrackProjYR = uv(1);
        pMP->mnTrackScaleLevelR= nPredictedLevel;
        pMP->mTrackViewCosR = viewCos;
        pMP->mTrackDepthR = Pc_dist;
    }
    else{
        pMP->mTrackProjX = uv(0);
        pMP->mTrackProjY = uv(1);
        pMP->mnTrackScaleLevel= nPredictedLevel;
        pMP->mTrackViewCos = viewCos;
        pMP->mTrackDepth = Pc_dist;
    }

    return true;
}

Eigen::Vector3f Frame::UnprojectStereoFishEye(const int &i){
    return mRwc * mvStereo3Dpoints[i] + mOw;
}


/******************************lcx*****************************/
// bool isEdge(uint8_t val)
// {
//     return (val == 1 || val == 255);
// }

void Frame::DetectEdgeImage(cv::Mat img, int TH1, int TH2)
{
    cv::Mat GSimg;
    cv::GaussianBlur(img, GSimg, cv::Size(7,7),2,2, cv::BORDER_REFLECT_101);
    // cv::Sobel(img, gradX, CV_16S, 1,0);
    // cv::Sobel(img, gradY, CV_16S, 0,1);
    cv::Canny(GSimg, EdgeImage, TH1, TH2);
    cv::distanceTransform(255-EdgeImage, DTImage, CV_DIST_L2, CV_DIST_MASK_PRECISE);
    // cv::normalize(DTImage, DTImage, 0, 255, cv::NORM_MINMAX);
}
Eigen::Vector3d Frame::GetGradient(int x, int y)
{
    Eigen::Vector3d gradient;
    double gx = static_cast<double>(-imgMono.at<uint8_t>(y-1,x+1)-2*imgMono.at<uint8_t>(y,x+1)-imgMono.at<uint8_t>(y+1,x+1)
                                + imgMono.at<uint8_t>(y-1,x-1)+2*imgMono.at<uint8_t>(y,x-1)+imgMono.at<uint8_t>(y+1,x-1));
    double gy = static_cast<double>(-imgMono.at<uint8_t>(y+1,x-1)-2*imgMono.at<uint8_t>(y+1,x)-imgMono.at<uint8_t>(y+1,x+1)
                                + imgMono.at<uint8_t>(y-1,x-1)+2*imgMono.at<uint8_t>(y-1,x)+imgMono.at<uint8_t>(y-1,x+1));
    double angle = atan(gy/gx)*180/3.14159265;
    gradient << gx,gy,angle;
    return gradient;
}
Eigen::Vector2d Frame::GetGradient2d(int x, int y)
{
    Eigen::Vector2d gradient;
    double gx = static_cast<double>(-imgMono.at<uint8_t>(y-1,x+1)-2*imgMono.at<uint8_t>(y,x+1)-imgMono.at<uint8_t>(y+1,x+1)
                                + imgMono.at<uint8_t>(y-1,x-1)+2*imgMono.at<uint8_t>(y,x-1)+imgMono.at<uint8_t>(y+1,x-1));
    double gy = static_cast<double>(-imgMono.at<uint8_t>(y+1,x-1)-2*imgMono.at<uint8_t>(y+1,x)-imgMono.at<uint8_t>(y+1,x+1)
                                + imgMono.at<uint8_t>(y-1,x-1)+2*imgMono.at<uint8_t>(y-1,x)+imgMono.at<uint8_t>(y-1,x+1));
    gradient << gx,gy;
    return gradient;
}


bool cmp(const pair<std::vector<cv::Point>, double> a, const pair<std::vector<cv::Point>, double> b)
{
    return a.second > b.second;
}
// void Frame::findEdgePixelInGrid(int threshold)
// {
//     int width = imgMono.cols;
//     int height = imgMono.rows;

//     /*****************************************/
//     int patchsize = 20;
//     int bound = 3;
//     int num_each_patch = 2000 / (patchsize*patchsize);
//     vector<int> EdgeNum = vector<int>(patchsize*patchsize, 0);
//     mGridEdgePixels = vector<vector<cv::Point2f>>(patchsize*patchsize);
//     int patch_w = width/patchsize;
//     int patch_h = height/patchsize;
//     for(size_t col = bound; col < width-bound; col++)
//         for(size_t row = bound; row < height-bound; row++)
//         {
//             int w_pos = static_cast<int>(col/patch_w);
//             int z_pos = static_cast<int>(row/patch_h);
//             int patch_locate = z_pos*patchsize + w_pos;
            
//             const uint8_t EdgePixel = EdgeImage.at<uint8_t>(row, col);
            
//             if(EdgePixel == 1 || EdgePixel == 255)
//             {
//                 mGridEdgePixels[patch_locate].push_back(cv::Point2f(col, row));           
//             }
//         }
//     for(int i=0 ;i < patchsize*patchsize; i++)
//     {
//        int k = 5;
//        vector<cv::Point2f> tempPoint;
//        int intial = rand() % mGridEdgePixels[i].size();
//        tempPoint.push_back(mGridEdgePixels[i][intial]);
//        while(k-- && mGridEdgePixels[i].empty())
//        {   
            
            

//        }

//     }

// }
void Frame::findEdgePixelInGrid(int threshold)
{
    int width = imgMono.cols;
    int height = imgMono.rows;

    /*****************************************/
    int patchsize = 20;
    int bound = 3;
    int num_each_patch = 4000 / (patchsize*patchsize);
    vector<int> EdgeNum = vector<int>(patchsize*patchsize, 0);
    int patch_w = width/patchsize;
    int patch_h = height/patchsize;
    for(size_t col = bound; col < width-bound; col++)
        for(size_t row = bound; row < height-bound; row++)
        {
            int w_pos = static_cast<int>(col/patch_w);
            int z_pos = static_cast<int>(row/patch_h);
            int patch_locate = z_pos*patchsize + w_pos;
            if(EdgeNum[patch_locate] >= num_each_patch || patch_locate > patchsize*patchsize)
                continue;
            
            const uint8_t EdgePixel = EdgeImage.at<uint8_t>(row, col);
            
            if(EdgePixel == 1 || EdgePixel == 255)
            {
                mAllEdgePixels.push_back(cv::Point2f(col, row));
                Eigen::Vector3d g = GetGradient(col, row);
                if(g(0) *g(1) > threshold)
                {
                    mValidEdgePixel.push_back(cv::Point2f(col, row));
                    EdgeNum[patch_locate]++;
                }
                
            }

        }  
}


void Frame::computeEdgePixels()
{
    int width = imgMono.cols;
    int height = imgMono.rows;

    /*****************************************/
    int patchsize = 20;
    int bound = 3;
    int num_each_patch = 10000 / (patchsize*patchsize);
    vector<int> EdgeNum = vector<int>(patchsize*patchsize, 0);
    int patch_w = width/patchsize;
    int patch_h = height/patchsize;
    for(size_t col = bound; col < width-bound; col++)
        for(size_t row = bound; row < height-bound; row++)
        {
            int w_pos = static_cast<int>(col/patch_w);
            int z_pos = static_cast<int>(row/patch_h);
            int patch_locate = z_pos*patchsize + w_pos;
            if(EdgeNum[patch_locate] >= num_each_patch || patch_locate > patchsize*patchsize)
                continue;
            
            const uint8_t EdgePixel = EdgeImage.at<uint8_t>(row, col);
            
            if(EdgePixel == 1 || EdgePixel == 255)
            {
                Eigen::Vector3d g = GetGradient(col, row);
                if(g(0) *g(1) > 5000)
                {
                    mValidEdgePixel.push_back(cv::Point2f(col, row));
                    EdgeNum[patch_locate]++;
                }
                
            }

        }
    /*****************************************/
    // std::vector<std::vector<cv::Point>> contours;
    // cv::Mat edges;
    // cv::Mat kenel = cv::getStructuringElement(cv::MORPH_RECT,  cv::Size(2,2));
    // // cv::dilate(EdgeImage, edges, kenel, cv::Point(-1,-1), 1);
    // cv::findContours(EdgeImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    // vector<pair<std::vector<cv::Point>, double>> EdgeMap;
    // for(const auto& contour: contours)
    // {
    //     double length = cv::arcLength(contour, false);
    //     EdgeMap.push_back(make_pair(contour, length));
    // }
    // sort(EdgeMap.begin(), EdgeMap.end(), cmp);
    // for(int it = 0; it < EdgeMap.size(); it++)
    // {
    //     std::vector<cv::Point> points = EdgeMap[it].first;
    //     for(int k = 0; k < points.size(); k++)
    //     {
    //         mAllEdgePixels.push_back(points[k]);
    //     }
    // }
    // int width = imgMono.cols;
    // int height = imgMono.rows;
    // std::vector<std::vector<cv::Point>> contours;
    // cv::Mat dilateImge;
    // cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
    // cv::dilate(EdgeImage, dilateImge, kernel, cv::Point(-1,-1), 1);
    // cv::findContours(dilateImge, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    // int maxNum = 1000;
    // std::map<double, std::vector<cv::Point>> EdgeMap;
    // for (int i = 0; i < contours.size(); i++) {
    //     double length = cv::arcLength(contours[i], true);
        
    //     EdgeMap.insert(make_pair(1/length, contours[i]));
    // }
    // for(std::map<double, std::vector<cv::Point>>::iterator it = EdgeMap.begin(); it != EdgeMap.end(); it++)
    // {
    //     std::vector<cv::Point> points = it->second;
    //     for(int k = 0; k < points.size(); k++)
    //     {
    //         mValidEdgePixel.push_back(points[k]);
    //         if(mValidEdgePixel.size() == maxNum) return;
    //     }
    // }
}

void Frame::ExtractEdge(int num)
{
    DetectEdgeImage(imgMono, 150, 100);
    // findEdgePixelInGrid(5000);
    // if(mValidEdgePixel.size() < 200)
    // {
    //     mValidEdgePixel.clear();
    //     findEdgePixelInGrid(2000);
    // }
    int width = imgMono.cols;
    int height = imgMono.rows;

    /*****************************************/
    int patchsize = 20;
    int bound = 5;
    vector<int> EdgeNum = vector<int>(patchsize*patchsize, 0);
    int patch_w = width/patchsize;
    int patch_h = height/patchsize;
    for(size_t col = bound; col < width-bound; col++)
        for(size_t row = bound; row < height-bound; row++)
        {
            
            const uint8_t EdgePixel = EdgeImage.at<uint8_t>(row, col);
            
            if(EdgePixel == 1 || EdgePixel == 255)
            {
                Eigen::Vector3d g = GetGradient(col, row);
                mValidEdgePixel.push_back(cv::Point2f(col, row));
                
            }

        }
    // computeNN();
    // int N;
    // if(num > mAllEdgePixels.size()) N = mAllEdgePixels.size();
    // else N = num;
    // for(int i = 0 ; i < N ; i ++)
    //     mValidEdgePixel.push_back(mAllEdgePixels[i]);
}
void Frame::AddEdgePixel(int num)
{
    for(int i = 0 ; i < num ; i ++)
        mValidEdgePixel.push_back(mAllEdgePixels[i]);
}

void Frame::UndistortEdgePixels()
{
    if(mDistCoef.at<float>(0)==0.0)
    {
        mValidEdgePixelUn = mValidEdgePixel;
        return;
    }
    cv::Mat mat(mValidEdgePixel.size(),2,CV_32F);

    for(int i=0;i<mValidEdgePixel.size();i++)
    {
        mat.at<float>(i,0) = mValidEdgePixel[i].x;
        mat.at<float>(i,1) = mValidEdgePixel[i].y;
    }
    mat = mat.reshape(2);
    cv::undistortPoints(mat, mat, static_cast<Pinhole*>(mpCamera)->toK(), mDistCoef, cv::Mat(), mK);
    mat = mat.reshape(1);

    mValidEdgePixelUn.resize(mValidEdgePixel.size());
    for(int i=0; i < mValidEdgePixel.size(); i++)
    {
        cv::Point kp = mValidEdgePixel[i];
        kp.x = mat.at<float>(i,0);
        kp.y = mat.at<float>(i,1);
        mValidEdgePixelUn[i] = kp;
    }
}
void Frame::UndistortMatchedEdgePixels()
{
    if(mDistCoef.at<float>(0)==0.0)
    {
        mMatchedEdgePixelUn = mMatchedEdgePixel;
        return;
    }
    cv::Mat mat(mMatchedEdgePixel.size(),2,CV_32F);
    for(int i=0;i<mMatchedEdgePixel.size();i++)
    {
        mat.at<float>(i,0) = mMatchedEdgePixel[i].x;
        mat.at<float>(i,1) = mMatchedEdgePixel[i].y;
    }
    mat = mat.reshape(2);
    cv::undistortPoints(mat, mat, static_cast<Pinhole*>(mpCamera)->toK(), mDistCoef, cv::Mat(), mK);
    mat = mat.reshape(1);

    mMatchedEdgePixelUn.resize(mMatchedEdgePixel.size());
    for(int i=0; i < mMatchedEdgePixel.size(); i++)
    {
        cv::Point kp = mMatchedEdgePixel[i];
        kp.x = mat.at<float>(i,0);
        kp.y = mat.at<float>(i,1);
        mMatchedEdgePixelUn[i] = kp;
    }
}

cv::Point2f Frame::distortPoint(cv::Point2f p)
{
    cv::Point2f res;
    cv::fisheye::distortPoints(cv::Mat(1,1,CV_32FC2,&p), cv::Mat(1,1,CV_32FC2,&res), static_cast<Pinhole*>(mpCamera)->toK(),mDistCoef);
    return res;

}

void Frame::AddEdgePoint(EdgePoint* pEP, const int &idx)
{
    mvpEdgePoints[idx] = pEP;
}
vector<EdgePoint*> Frame::GetEdgePoints()
{
    return mvpEdgePoints;
}
float calcEuclidean(int x1, int y1, int x2, int y2)
{
    return sqrt(float((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)));
}

// void Frame::computeNN()
// {
//     locationX = cv::Mat(imgMono.rows, imgMono.cols, CV_32SC1);
//     locationY = cv::Mat(imgMono.rows, imgMono.cols, CV_32SC1);
//     for(int col=0 ; col<imgMono.cols; col++)
//         for(int row=0 ; row<imgMono.rows; row++)
//         {
//             cout<<DTImage.at<float>(row, col)<<endl;
//             if(DTImage.at<float>(row, col) == 0)
//             {
//                 locationX.at<int>(row,col) = col;
//                 locationY.at<int>(row,col) = row;
//                 continue;
//             }
//             bool flag = false;
//             int maskSize = 1;
//             float min_dis = 99999;
//             int minDTlocation[2] = {0};
//             while(true)
//             {
//                 int x_min = std::max(0, col-maskSize);
//                 int x_max = std::min(col+maskSize, imgMono.cols-1);
//                 int y_min = std::max(0, row-maskSize);
//                 int y_max = std::min(row+maskSize, imgMono.rows-1);
//                 /*******************************************/
//                 for(int y=y_min;y<=y_max;y++)
//                 {
//                     if(DTImage.at<float>(y,x_min) == 0)
//                     {
//                         float dis = calcEuclidean(x_min,y, col, row);
//                         if(dis<min_dis)
//                         {
//                             minDTlocation[0] = x_min;
//                             minDTlocation[1] = y;
//                             min_dis = dis;
//                             flag = true;
//                         }
//                     }
//                 }
//                 if(min_dis == maskSize)
//                 {
//                     locationX.at<int>(row, col) = minDTlocation[0];
//                     locationY.at<int>(row, col) = minDTlocation[1];
//                     break;
//                 }
//                 /*******************************************/
//                 for(int y=y_min;y<=y_max;y++)
//                 {
//                     if(DTImage.at<float>(y,x_max) == 0)
//                     {
//                         float dis = calcEuclidean(x_max,y, col, row);
//                         if(dis<min_dis)
//                         {
//                             minDTlocation[0] = x_max;
//                             minDTlocation[1] = y;
//                             min_dis = dis;
//                             flag = true;
//                         }
//                     }
//                 }
//                 if(min_dis == maskSize)
//                 {
//                     locationX.at<int>(row, col) = minDTlocation[0];
//                     locationY.at<int>(row, col) = minDTlocation[1];
//                     break;
//                 }         
//                 /*******************************************/       
//                 for(int x=x_min;x<=x_max;x++)
//                 {
//                     if(DTImage.at<float>(y_min,x) == 0)
//                     {
//                         float dis = calcEuclidean(x,y_min, col, row);
//                         if(dis<min_dis)
//                         {
//                             minDTlocation[0] = x;
//                             minDTlocation[1] = y_min;
//                             min_dis = dis;
//                             flag = true;
//                         }
//                     }
//                 }
//                 if(min_dis == maskSize)
//                 {
//                     locationX.at<int>(row, col) = minDTlocation[0];
//                     locationY.at<int>(row, col) = minDTlocation[1];
//                     break;
//                 }       
//                 /***************************************/
//                 for(int x=x_min;x<=x_max;x++)
//                 {
//                     if(DTImage.at<float>(y_max,x) == 0)
//                     {
//                         float dis = calcEuclidean(x,y_max, col, row);
//                         if(dis<min_dis)
//                         {
//                             minDTlocation[0] = x;
//                             minDTlocation[1] = y_max;
//                             min_dis = dis;
//                             flag = true;
//                         }
//                     }
//                 }
//                 if(min_dis == maskSize)
//                 {
//                     locationX.at<int>(row, col) = minDTlocation[0];
//                     locationY.at<int>(row, col) = minDTlocation[1];
//                     break;
//                 }                       

//                 /***************************************/
//                 if(flag)
//                 {
//                     locationX.at<int>(row, col) = minDTlocation[0];
//                     locationY.at<int>(row, col) = minDTlocation[1];
//                     break;                   
//                 }
//                 ++maskSize;
//             }
//         }
// }

Eigen::Vector2d Frame::GetNearestPixel(int col, int row)
{
    // cv::Point2f result;
    if(DTImage.at<float>(row,col) == 0)
        return Eigen::Vector2d{row, col};
    
    bool flag = false;
    int maskSize = 1;
    float min_dis = 99999;
    int minDTlocation[2] = {0};
    while(true)
    {
        int x_min = std::max(0, col-maskSize);
        int x_max = std::min(col+maskSize, imgMono.cols-1);
        int y_min = std::max(0, row-maskSize);
        int y_max = std::min(row+maskSize, imgMono.rows-1);
        /*******************************************/
        for(int y=y_min;y<=y_max;y++)
        {
            if(DTImage.at<float>(y,x_min) == 0)
            {
                float dis = calcEuclidean(x_min,y, col, row);
                if(dis<min_dis)
                {
                    minDTlocation[0] = x_min;
                    minDTlocation[1] = y;
                    min_dis = dis;
                    flag = true;
                }
            }
        }
        if(min_dis == maskSize)
        {
            return Eigen::Vector2d{minDTlocation[0], minDTlocation[1]};
        }
        /*******************************************/
        for(int y=y_min;y<=y_max;y++)
        {
            if(DTImage.at<float>(y,x_max) == 0)
            {
                float dis = calcEuclidean(x_max,y, col, row);
                if(dis<min_dis)
                {
                    minDTlocation[0] = x_max;
                    minDTlocation[1] = y;
                    min_dis = dis;
                    flag = true;
                }
            }
        }
        if(min_dis == maskSize)
        {
            return Eigen::Vector2d{minDTlocation[0], minDTlocation[1]};
        }         
        /*******************************************/       
        for(int x=x_min;x<=x_max;x++)
        {
            if(DTImage.at<float>(y_min,x) == 0)
            {
                float dis = calcEuclidean(x,y_min, col, row);
                if(dis<min_dis)
                {
                    minDTlocation[0] = x;
                    minDTlocation[1] = y_min;
                    min_dis = dis;
                    flag = true;
                }
            }
        }
        if(min_dis == maskSize)
        {
            return Eigen::Vector2d{minDTlocation[0], minDTlocation[1]};
        }       
        /***************************************/
        for(int x=x_min;x<=x_max;x++)
        {
            if(DTImage.at<float>(y_max,x) == 0)
            {
                float dis = calcEuclidean(x,y_max, col, row);
                if(dis<min_dis)
                {
                    minDTlocation[0] = x;
                    minDTlocation[1] = y_max;
                    min_dis = dis;
                    flag = true;
                }
            }
        }
        if(min_dis == maskSize)
        {
            return Eigen::Vector2d{minDTlocation[0], minDTlocation[1]};
        }                       

        /***************************************/
        if(flag)
        {
            return Eigen::Vector2d{minDTlocation[0], minDTlocation[1]};         
        }
        ++maskSize;
    }    
}
Eigen::Vector2d Frame::GetGradientDirection(Eigen::Vector2d g)
{
    double angle = atan2(g(1),g(0))*180/M_PI;
    if(angle<0)angle+=360;
    int direction = (int)((angle+22.5)/45)*45;
    double a = direction/180*M_PI;
    Eigen::Vector2d res;
    double x = cos(a);
    double y = sin(a);
    int area = direction/45%8;
    res << x,y;
    return res;
}

bool cmp_pixel(const Pixel a, const Pixel b)
{
    return a.gradient > b.gradient;
}
void Frame::ExtractEdgePixels(int patchsize, int GridNum)
{
    clock_t s = clock();
    DetectEdgeImage(imgMono, 150, 100);
    int width = imgMono.cols;
    int height = imgMono.rows;
    int bound = 10;
    vector<vector <Pixel>> GridPixels(patchsize*patchsize);
    int patch_w = width/patchsize;
    int patch_h = height/patchsize;
    for(int col = bound ; col < width-bound; col++)
        for(int row=bound; row < height-bound; row++)
        {
            int w_pos = static_cast<int>(col/patch_w);
            int z_pos = static_cast<int>(row/patch_h);
            int patch_locate = z_pos*patchsize + w_pos;
            if(patch_locate >= patchsize*patchsize) continue;
            // if(EdgeNum[patch_locate] >= num_each_patch || patch_locate > patchsize*patchsize)
            //     continue;

            const uint8_t EdgePixel = EdgeImage.at<uint8_t>(row, col);      

            if(EdgePixel == 1 || EdgePixel == 255)   
            {
                Eigen::Vector2d g = GetGradient2d(col, row);
                double angle = atan2(g(1), g(0))*180/M_PI;
                    if(angle < 0) angle+=360;
                Pixel p(col, row, g(0),g(1),angle);
                GridPixels[patch_locate].push_back(p);
            }
        }
        clock_t ss = clock();
        // cout<<"Extract Edge: "<<(double)(ss-s)/CLOCKS_PER_SEC<<endl;
    for(int i=0; i < GridPixels.size() ;i++)
    {
        if(GridPixels[i].size()<GridNum)
        {
            for(auto p: GridPixels[i])
                mValidEdgePixel.push_back(cv::Point(p.x, p.y));
            
            continue;
        }
        sort(GridPixels[i].begin(), GridPixels[i].end(), cmp_pixel);
        vector<Pixel> selectedPixels;
        for(int j=0;j<3;j++)
        {
            selectedPixels.push_back(GridPixels[i][j]);
            GridPixels[i][j].beSelected = true;
        }
        while(selectedPixels.size() < GridNum)
        {
            FindMaxEntropy(GridPixels[i], selectedPixels);
        }
        for(auto sP : selectedPixels)
        {
            mValidEdgePixel.push_back(cv::Point(sP.x, sP.y));
        }
    }
    EdgeTime += (double)(clock()-s)/CLOCKS_PER_SEC;
    EdgeTimeNum++;
    cout<<" Edge Extraction and Selection:  "<<EdgeTime/EdgeTimeNum<<endl;
    
}

void Frame::ExtractEdgePixels_(int G_threshold)
{
    DetectEdgeImage(imgMono, 150, 100);
    int width = imgMono.cols;
    int height = imgMono.rows;
    int bound = 10;
    for(int col = bound ; col < width-bound; col++)
        for(int row=bound; row < height-bound; row++)
        {
            const uint8_t EdgePixel = EdgeImage.at<uint8_t>(row, col);      

            if(EdgePixel == 1 || EdgePixel == 255)   
            {
                Eigen::Vector2d g = GetGradient2d(col, row);
                if(g(0)*g(1) > G_threshold)
                    mValidEdgePixel.push_back(cv::Point(col, row));
            }
        }
}
double calculateDistance(Pixel a, Pixel b)
{
    return sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y));
}
double Frame::calculateEntropy(vector<Pixel>& selectedPoints)
{
    std::vector<int> counts(8, 0);
    for(Pixel sp : selectedPoints)
    {
        int index = static_cast<int>(sp.angle / 45.0);
        counts[index]++;
    }
    double entropy = 0.0;
    for (int count : counts) {
        if (count > 0) {
            double probability = static_cast<double>(count) / selectedPoints.size();
            entropy -= probability * log2(probability);
        }
    }
    return entropy;    
}
void Frame::FindMaxEntropy(std::vector<Pixel>& pixels, std::vector<Pixel>& selectedPoints)
{
    double maxEntropyGain = -1.0;
    Pixel nextP;
    for(Pixel& p : pixels)
    {
        if(p.beSelected)continue;
        bool flag = false;
        for(auto pp: selectedPoints)
        {
            if(calculateDistance(p,pp) < 10)
            {
                flag = true;
                break;
            } 
        }
        if(flag) continue;
        double currentEntropy = calculateEntropy(selectedPoints);
        selectedPoints.push_back(p);
        double entropyGain =calculateEntropy(selectedPoints) - currentEntropy;
        if(entropyGain > maxEntropyGain)
        {
            maxEntropyGain = entropyGain;
            nextP = p;
            
        }
        selectedPoints.pop_back();
    }
    nextP.beSelected = true;
    selectedPoints.push_back(nextP);
}
} //namespace ORB_SLAM
