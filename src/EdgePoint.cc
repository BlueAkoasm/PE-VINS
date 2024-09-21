#include "EdgePoint.h"

namespace ORB_SLAM3
{
long unsigned int EdgePoint::nNextId=0;
EdgePoint::EdgePoint(const Eigen::Vector3f &Pos)
{
    mWorldPos = Pos;
}
EdgePoint::EdgePoint(const Eigen::Vector3f &Pos, Map* pMap)
{
    mWorldPos = Pos;
    mpMap = pMap;
}
EdgePoint::EdgePoint(const Eigen::Vector3f &Pos, Frame* f, Map* pMap)
{
    mWorldPos = Pos;
    mpMap = pMap;
    InitFrame = f;
    TcwBefOpt = f->GetPose();
}
EdgePoint::EdgePoint(const Eigen::Vector3f &Pos, KeyFrame* KF, Map* pMap)
{
    mWorldPos = Pos;
    mpMap = pMap;
    RefKF = KF;
    mnId = nNextId++;
}
Eigen::Vector3f EdgePoint::GetWorldPos()
{
    return mWorldPos;
}
Map* EdgePoint::GetMap()
{
    return mpMap;
}
void EdgePoint::SetWorldPos(const Eigen::Vector3f &Pos)
{
    mWorldPos = Pos;
}
void EdgePoint::SetIndex(int idx)
{
    index = idx;
}
int EdgePoint::GetIndex()
{
    return index;
}
Frame* EdgePoint::GetFrame()
{
    return InitFrame;
}
}