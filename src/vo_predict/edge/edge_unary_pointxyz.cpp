#include "edge/edge_unary_pointxyz.h"
#include <Eigen/StdVector>
#include <Eigen/Geometry>
#include <math.h>
#include <iostream>
namespace g2o{
void EdgeUnaryPointXYZ::computeError()
{
    const VertexPointXYZ* vertex  =static_cast<const VertexPointXYZ*> ( _vertices[0] );
    
    Eigen::Vector3d posi = vertex->estimate();
    double u = fx_*posi[0]/posi[2]+cx_;
    double v = fy_*posi[1]/posi[2]+cy_;
    double loss = -(fa_*v*v+fb_*v+fc_ - u-(2/1.75)*(v-cy_));
    double scale = 1.75/(v-cy_+0.101023);
    loss = loss*scale;
    double e_loss = loss;
    if(loss<0) loss=0;
    //std::cout<<posi<<" "<<std::endl<<u<<"  "<<v<<" "<<loss<<" "<<e_loss<<std::endl;
    _error(0,0) = e_loss;
}

void EdgeUnaryPointXYZ::linearizeOplus()
{
    ////std::cout<<"test______________"<<std::endl;
    Eigen::Matrix<double, 1, 3> jacobian_l_X;
    const VertexPointXYZ* vertex  =static_cast<const VertexPointXYZ*> ( _vertices[0] );
    
    Eigen::Vector3d posi = vertex->estimate();
    double u = fx_*posi[0]/posi[2]+cx_;
    double v = fy_*posi[1]/posi[2]+cy_;
    double loss = -(fa_*v*v+fb_*v+fc_ - u);
    //std::cout<<" X Y Z" <<posi[0]<<" "<<posi[1]<<" "<<posi[2]<<std::endl;
    //std::cout<<" u v loss" <<u<<" "<<v<<" "<<loss<<std::endl;
    //std::cout<<" fx fy cx cy" <<fx_<<" "<<fy_<<" "<<cx_<<" "<<cy_<<std::endl;
    Eigen::Matrix<double,2,3> jacobian_U_X;
    jacobian_U_X(0,0) = fx_/posi[2];
    jacobian_U_X(0,2) = -fx_*posi[0]/(posi[2]*posi[2]);
    jacobian_U_X(1,0) = fy_/posi[2];
    jacobian_U_X(1,2) = -fy_*posi[1]/(posi[2]*posi[2]);
    jacobian_U_X(0,1) = 0;
    jacobian_U_X(1,0) = 0;

    //std::cout<<"gradient UX"<<jacobian_U_X<<std::endl;
    Eigen::Matrix<double,1,2> jacobian_l_U;
    double scale = 1.75/(v-cy_+0.101023);
    jacobian_l_U(0,1) =(-2*fa_*v+-fb_)*scale + loss*(-scale*scale/1.75);
    jacobian_l_U(0,0) =1*scale;
    //std::cout<<"gradient lU"<<jacobian_l_U<<std::endl;
    jacobian_l_X = jacobian_l_U*jacobian_U_X;
    //std::cout<<"gradient lX"<<jacobian_l_X<<std::endl;
    double jacobian_l = 1;
    if(loss<0) jacobian_l = 0;

    _jacobianOplusXi = jacobian_l*jacobian_l_X;
    /*
    if(loss>=0)
        _jacobianOplusXi = jacobian_l_X;
    else
        jacobian_l_X(0,0) = 0;
        jacobian_l_X(0,1) = 0;
        jacobian_l_X(0,2) = 0;
        _jacobianOplusXi = jacobian_l_X;
        */
    //std::cout<<"gradient "<<_jacobianOplusXi<<std::endl;

}

bool EdgeUnaryPointXYZ::read( std::istream& in )
{
    return true;
}
bool EdgeUnaryPointXYZ::write( std::ostream& out ) const
{
    return true;
}
}//end namespace g2o
