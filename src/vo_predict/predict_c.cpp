#include <Eigen/StdVector>
#include <iostream>
#include <stdint.h>

#include <unordered_set>


#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
//#include "g2o/solvers/eigen/linear_solver_eigen.h"
//#include "g2o/solvers/dense/linear_solver_dense.h"

#include "g2o/solvers/csparse/linear_solver_csparse.h"
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/types/sim3/types_seven_dof_expmap.h"
#include "g2o/types/sim3/sim3.h"
//
using namespace std;

using vvd = std::vector<std::vector<double>>;
class EgoMotionPrediction{
    g2o::SparseOptimizer optimizer;
    public:
    EgoMotionPrediction()
    {
        optimizer.setVerbose(false);
        //std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
        //linearSolver = g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>>();
        typedef g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType> linearSolver;
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<g2o::BlockSolver_6_3>(g2o::make_unique<linearSolver>()));

        optimizer.setAlgorithm(solver);
    }
    vvd predict_group(vvd quats, vvd trans)
    {
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(g2o::SE3Quat());
        vSE3->setId(0);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        const Eigen::Matrix<double,6,6> matLambda = Eigen::Matrix<double,6,6>::Identity();
        for(int i=1; i<6; i++)
        {
            if(i==1 || i==5)
            {
                Eigen::Quaterniond r = Eigen::Quaterniond(quats[i].data());     
                Eigen::Vector3d t = Eigen::Vector3d(trans[i].data());
                g2o::SE3Quat Sji = g2o::SE3Quat(r,t);
                g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
                vSE3->setEstimate(Sji);
                if(i==1)
                vSE3->setId(i);
                else
                vSE3->setId(2);
                vSE3->setFixed(false);
                optimizer.addVertex(vSE3);
            }
        
        }
        // add edge
        for(int i = 0;i<6;i++)    {
            Eigen::Vector3d t = Eigen::Vector3d(trans[i].data());
            Eigen::Quaterniond r = Eigen::Quaterniond(quats[i].data());      
            g2o::SE3Quat Sji = g2o::SE3Quat(r,t);
            g2o::EdgeSE3Expmap* e = new g2o::EdgeSE3Expmap();
            if(i<2)
            {
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(1)));
            }
            else if(i<4)
            {
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(1)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2)));
            }
            else if(i<6)
            {
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2)));
            }
            e->setMeasurement(Sji);
            e->information() = matLambda;
            optimizer.addEdge(e);

        }

        optimizer.setVerbose(true);
        optimizer.initializeOptimization();
        vvd result;
        optimizer.optimize(20);
        for(int i = 1;i<3;i++)    {
            g2o::VertexSE3Expmap* VSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(i));
            g2o::SE3Quat CorrectedSiw =  VSE3->estimate();
         
            Eigen::Matrix<double,4,4> CorrectedSiwM = CorrectedSiw.to_homogeneous_matrix();
            for(int j =0;j<4;j++)
            {
                std::vector<double> re;
                re.push_back(CorrectedSiwM(j,0));
                re.push_back(CorrectedSiwM(j,1));
                re.push_back(CorrectedSiwM(j,2));
                re.push_back(CorrectedSiwM(j,3));
                result.push_back(re);
            }

            
        }
            optimizer.edges().clear();
            optimizer.vertices().clear();
        return result;
    }
    vvd predict_patch(vvd quats, vvd trans)
    {
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(g2o::SE3Quat());
        vSE3->setId(0);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        const Eigen::Matrix<double,6,6> matLambda = Eigen::Matrix<double,6,6>::Identity();
        for(int i=1; i<2; i++)
        {
            Eigen::Quaterniond r = Eigen::Quaterniond(quats[i].data());     
            Eigen::Vector3d t = Eigen::Vector3d(trans[i].data());
            g2o::SE3Quat Sji = g2o::SE3Quat(r,t);
            g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
            vSE3->setEstimate(Sji);
            vSE3->setId(i);
            vSE3->setFixed(false);
            optimizer.addVertex(vSE3);
        
        }
        // add edge
        int patch_size = quats.size()/2;

        for(int i = 0;i<patch_size;i++)    {
            Eigen::Vector3d t = Eigen::Vector3d(trans[i].data());
            Eigen::Quaterniond r = Eigen::Quaterniond(quats[i].data());      
            g2o::SE3Quat Sji = g2o::SE3Quat(r,t);
            g2o::EdgeSE3Expmap* e = new g2o::EdgeSE3Expmap();
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(1)));
            e->setMeasurement(Sji);
            e->information() = matLambda;
            optimizer.addEdge(e);

        }
        for(int i = patch_size;i<2*patch_size;i++)
        {
            Eigen::Vector3d t = Eigen::Vector3d(trans[i].data());
            Eigen::Quaterniond r = Eigen::Quaterniond(quats[i].data());      
            g2o::SE3Quat Sji = g2o::SE3Quat(r,t);
            g2o::EdgeSE3Expmap* e = new g2o::EdgeSE3Expmap();
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(1)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
            e->setMeasurement(Sji);
            e->information() = matLambda;
            optimizer.addEdge(e);

        }

        optimizer.setVerbose(true);
        optimizer.initializeOptimization();
        vvd result;
        optimizer.optimize(20);
        for(int i = 1;i<2;i++)    {
            g2o::VertexSE3Expmap* VSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(i));
            g2o::SE3Quat CorrectedSiw =  VSE3->estimate();
         
            Eigen::Matrix<double,4,4> CorrectedSiwM = CorrectedSiw.to_homogeneous_matrix();
            for(int j =0;j<4;j++)
            {
                std::vector<double> re;
                re.push_back(CorrectedSiwM(j,0));
                re.push_back(CorrectedSiwM(j,1));
                re.push_back(CorrectedSiwM(j,2));
                re.push_back(CorrectedSiwM(j,3));
                result.push_back(re);
            }

            
        }
            optimizer.edges().clear();
            optimizer.vertices().clear();
        return result;
    }

    vvd predict(vvd quats, vvd trans)
    {
        std::vector<double> init_quad,init_tran;
        init_quad.push_back(0);
        init_quad.push_back(0);
        init_quad.push_back(0);
        init_quad.push_back(1);
        init_tran.push_back(0);
        init_tran.push_back(0);
        init_tran.push_back(0);
        Eigen::Quaterniond r = Eigen::Quaterniond(init_quad.data());     
        Eigen::Vector3d t = Eigen::Vector3d(init_tran.data());
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(g2o::SE3Quat(r,t));
        vSE3->setId(0);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        const Eigen::Matrix<double,6,6> matLambda = Eigen::Matrix<double,6,6>::Identity();
        for(int i=1; i<2; i++)
        {
            Eigen::Quaterniond r = Eigen::Quaterniond(quats[i].data());     
            Eigen::Vector3d t = Eigen::Vector3d(trans[i].data());
            g2o::SE3Quat Sji = g2o::SE3Quat(r,t);
            g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
            vSE3->setEstimate(Sji);
            vSE3->setId(i);
            vSE3->setFixed(false);
            optimizer.addVertex(vSE3);
        
        }
        // add edge
        for(int i = 0;i<2;i++)    {
            Eigen::Vector3d t = Eigen::Vector3d(trans[i].data());
            Eigen::Quaterniond r = Eigen::Quaterniond(quats[i].data());      
            g2o::SE3Quat Sji = g2o::SE3Quat(r,t);
            g2o::EdgeSE3Expmap* e = new g2o::EdgeSE3Expmap();
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(1)));
            e->setMeasurement(Sji);
            e->information() = matLambda;
            optimizer.addEdge(e);

        }

        optimizer.setVerbose(true);
        optimizer.initializeOptimization();
        vvd result;
        optimizer.optimize(20);
        for(int i = 1;i<2;i++)    {
            g2o::VertexSE3Expmap* VSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(i));
            g2o::SE3Quat CorrectedSiw =  VSE3->estimate();
         
            Eigen::Matrix<double,4,4> CorrectedSiwM = CorrectedSiw.to_homogeneous_matrix();
            for(int j =0;j<4;j++)
            {
                std::vector<double> re;
                re.push_back(CorrectedSiwM(j,0));
                re.push_back(CorrectedSiwM(j,1));
                re.push_back(CorrectedSiwM(j,2));
                re.push_back(CorrectedSiwM(j,3));
                result.push_back(re);
            }

            
        }
            optimizer.edges().clear();
            optimizer.vertices().clear();
        return result;
    }
};


int main(int argc, char** argv)
{
    EgoMotionPrediction * vo_predict = new EgoMotionPrediction();
    vvd quats;
    vvd trans;
    for(int i =0;i<2;i++)
    {
        std::vector<double> q;
        q.push_back(0);
        q.push_back(0);
        q.push_back(0);
        q.push_back(1);
        quats.push_back(q);
        std::vector<double> t;
        t.push_back(i);
        t.push_back(i);
        t.push_back(i);
        trans.push_back(t);
    }
    vvd res = vo_predict->predict_patch(quats,trans);
    return 0;
}
