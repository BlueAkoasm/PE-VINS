#ifndef EDGEPOINT_H
#define EDGEPOINT_H


#include "KeyFrame.h"
#include "Frame.h"
#include "Map.h"
#include "Converter.h"

#include "SerializationUtils.h"

#include<opencv2/core/core.hpp>
#include<mutex>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>

using namespace std;
using namespace cv;

namespace ORB_SLAM3
{
class KeyFrame;
class Map;
class Frame;

class EdgePoint
{
public:
    EdgePoint(const Eigen::Vector3f &Pos);
    EdgePoint(const Eigen::Vector3f &Pos, Map* pMap);
    EdgePoint(const Eigen::Vector3f &Pos, Frame* f, Map* pMap);
    EdgePoint(const Eigen::Vector3f &Pos, KeyFrame *KF, Map* pMap);
    Eigen::Vector3f GetWorldPos();
    Map* GetMap();
    Frame* GetFrame();
    void SetWorldPos(const Eigen::Vector3f &Pos);
    void SetIndex(int idx);
    int GetIndex();
    long unsigned int mnId;
    static long unsigned int nNextId;
    Sophus::SE3f TcwBefOpt;
protected:
    Eigen::Vector3f mWorldPos;
    Map* mpMap;
    Frame*  InitFrame;
    KeyFrame* RefKF;
    int index;
};


}








#endif