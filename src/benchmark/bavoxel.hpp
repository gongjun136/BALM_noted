#ifndef BAVOXEL_HPP
#define BAVOXEL_HPP

#include "tools.hpp"
#include <Eigen/Eigenvalues>
#include <thread>
#include <glog/logging.h>

int layer_limit = 2;
int layer_size[] = {30, 30, 30, 30};
// float eigen_value_array[] = {1.0/4.0, 1.0/4.0, 1.0/4.0};
float eigen_value_array[4] = {1.0 / 16, 1.0 / 16, 1.0 / 16, 1.0 / 16};
int min_ps = 15;
double one_three = (1.0 / 3.0);

double voxel_size = 1;
int life_span = 1000;
// 窗口大小=总帧数
int win_size = 20;

int merge_enable = 1;

// 存储和操作体素地图中的信息，并执行一些基于体素特征的优化存储和操作体素地图中的信息，并执行一些基于体素特征的优化
class VOX_HESS
{
public:
  vector<const PointCluster *> sig_vecs;
  // 当前平面体素内所有帧原始点的统计信息
  // plvec_voxels下标表示第几个平面体素,vector<PointCluster>的下标表示帧id
  vector<const vector<PointCluster> *> plvec_voxels;
  vector<double> coeffs, coeffs_back;

  vector<pcl::PointCloud<PointType>::Ptr> plptrs;

  /**
   * @brief VOX_HESS成员函数,将平面体素插入到当前VOX_HESS中
   *
   * @param vec_orig
   * @param fix
   * @param feat_eigen
   * @param layer
   */
  void push_voxel(const vector<PointCluster> *vec_orig, const PointCluster *fix, double feat_eigen, int layer)
  {
    // 统计平面体素的总点数
    int process_size = 0;
    for (int i = 0; i < win_size; i++)
      if ((*vec_orig)[i].N != 0)
        process_size++;

    // 点数过少就剔除掉
    if (process_size < 2)
      return; // 改

    double coe = 1 - feat_eigen / eigen_value_array[layer];
    coe = coe * coe;
    coe = 1;
    coe = 0;
    for (int j = 0; j < win_size; j++)
      coe += (*vec_orig)[j].N;

    plvec_voxels.push_back(vec_orig);
    sig_vecs.push_back(fix);
    coeffs.push_back(coe);
    pcl::PointCloud<PointType>::Ptr plptr(new pcl::PointCloud<PointType>());
    plptrs.push_back(plptr);
  }

  void acc_evaluate2(const vector<IMUST> &xs, int head, int end, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT, double &residual)
  {
    Hess.setZero();
    JacT.setZero();
    residual = 0;
    vector<PointCluster> sig_tran(win_size);
    const int kk = 0;

    PLV(3)
    viRiTuk(win_size);
    PLM(3)
    viRiTukukT(win_size);

    vector<Eigen::Matrix<double, 3, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 6>>> Auk(win_size);
    Eigen::Matrix3d umumT;

    for (int a = head; a < end; a++)
    {
      const vector<PointCluster> &sig_orig = *plvec_voxels[a];
      double coe = coeffs[a];

      PointCluster sig = *sig_vecs[a];
      for (int i = 0; i < win_size; i++)
        if (sig_orig[i].N != 0)
        {
          sig_tran[i].transform(sig_orig[i], xs[i]);
          sig += sig_tran[i];
        }

      const Eigen::Vector3d &vBar = sig.v / sig.N;
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(sig.P / sig.N - vBar * vBar.transpose());
      const Eigen::Vector3d &lmbd = saes.eigenvalues();
      const Eigen::Matrix3d &U = saes.eigenvectors();
      int NN = sig.N;

      Eigen::Vector3d u[3] = {U.col(0), U.col(1), U.col(2)};

      const Eigen::Vector3d &uk = u[kk];
      Eigen::Matrix3d ukukT = uk * uk.transpose();
      umumT.setZero();
      for (int i = 0; i < 3; i++)
        if (i != kk)
          umumT += 2.0 / (lmbd[kk] - lmbd[i]) * u[i] * u[i].transpose();

      for (int i = 0; i < win_size; i++)
        // for(int i=1; i<win_size; i++)
        if (sig_orig[i].N != 0)
        {
          Eigen::Matrix3d Pi = sig_orig[i].P;
          Eigen::Vector3d vi = sig_orig[i].v;
          Eigen::Matrix3d Ri = xs[i].R;
          double ni = sig_orig[i].N;

          Eigen::Matrix3d vihat;
          vihat << SKEW_SYM_MATRX(vi);
          Eigen::Vector3d RiTuk = Ri.transpose() * uk;
          Eigen::Matrix3d RiTukhat;
          RiTukhat << SKEW_SYM_MATRX(RiTuk);

          Eigen::Vector3d PiRiTuk = Pi * RiTuk;
          viRiTuk[i] = vihat * RiTuk;
          viRiTukukT[i] = viRiTuk[i] * uk.transpose();

          Eigen::Vector3d ti_v = xs[i].p - vBar;
          double ukTti_v = uk.dot(ti_v);

          Eigen::Matrix3d combo1 = hat(PiRiTuk) + vihat * ukTti_v;
          Eigen::Vector3d combo2 = Ri * vi + ni * ti_v;
          Auk[i].block<3, 3>(0, 0) = (Ri * Pi + ti_v * vi.transpose()) * RiTukhat - Ri * combo1;
          Auk[i].block<3, 3>(0, 3) = combo2 * uk.transpose() + combo2.dot(uk) * I33;
          Auk[i] /= NN;

          const Eigen::Matrix<double, 6, 1> &jjt = Auk[i].transpose() * uk;
          JacT.block<6, 1>(6 * i, 0) += coe * jjt;

          const Eigen::Matrix3d &HRt = 2.0 / NN * (1.0 - ni / NN) * viRiTukukT[i];
          Eigen::Matrix<double, 6, 6> Hb = Auk[i].transpose() * umumT * Auk[i];
          Hb.block<3, 3>(0, 0) += 2.0 / NN * (combo1 - RiTukhat * Pi) * RiTukhat - 2.0 / NN / NN * viRiTuk[i] * viRiTuk[i].transpose() - 0.5 * hat(jjt.block<3, 1>(0, 0));
          Hb.block<3, 3>(0, 3) += HRt;
          Hb.block<3, 3>(3, 0) += HRt.transpose();
          Hb.block<3, 3>(3, 3) += 2.0 / NN * (ni - ni * ni / NN) * ukukT;

          Hess.block<6, 6>(6 * i, 6 * i) += coe * Hb;
        }

      for (int i = 0; i < win_size - 1; i++)
        // for(int i=1; i<win_size-1; i++)
        if (sig_orig[i].N != 0)
        {
          double ni = sig_orig[i].N;
          for (int j = i + 1; j < win_size; j++)
            if (sig_orig[j].N != 0)
            {
              double nj = sig_orig[j].N;
              Eigen::Matrix<double, 6, 6> Hb = Auk[i].transpose() * umumT * Auk[j];
              Hb.block<3, 3>(0, 0) += -2.0 / NN / NN * viRiTuk[i] * viRiTuk[j].transpose();
              Hb.block<3, 3>(0, 3) += -2.0 * nj / NN / NN * viRiTukukT[i];
              Hb.block<3, 3>(3, 0) += -2.0 * ni / NN / NN * viRiTukukT[j].transpose();
              Hb.block<3, 3>(3, 3) += -2.0 * ni * nj / NN / NN * ukukT;

              Hess.block<6, 6>(6 * i, 6 * j) += coe * Hb;
            }
        }

      residual += coe * lmbd[kk];
    }

    for (int i = 1; i < win_size; i++)
      for (int j = 0; j < i; j++)
        Hess.block<6, 6>(6 * i, 6 * j) = Hess.block<6, 6>(6 * j, 6 * i).transpose();
  }

  void left_evaluate(const vector<IMUST> &xs, int head, int end, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT, double &residual)
  {
    Hess.setZero();
    JacT.setZero();
    residual = 0;
    // vector<PointCluster> sig_tran(win_size);
    int l = 0;
    Eigen::Matrix<double, 3, 4> Sp;
    Sp.setZero();
    Sp.block<3, 3>(0, 0).setIdentity();
    Eigen::Matrix4d F;
    F.setZero();
    F(3, 3) = 1;

    PLM(4)
    T(win_size);
    for (int i = 0; i < win_size; i++)
      T[i] << xs[i].R, xs[i].p, 0, 0, 0, 1;

    vector<PLM(4) *> Cs;
    for (int a = 0; a < plvec_voxels.size(); a++)
    {
      const vector<PointCluster> &sig_orig = *plvec_voxels[a];
      PLM(4) *Co = new PLM(4)(win_size, Eigen::Matrix4d::Zero());
      for (int i = 0; i < win_size; i++)
        Co->at(i) << sig_orig[i].P, sig_orig[i].v, sig_orig[i].v.transpose(), sig_orig[i].N;
      Cs.push_back(Co);
    }

    double t0 = ros::Time::now().toSec();

    for (int a = head; a < end; a++)
    {
      // const vector<PointCluster> &sig_orig = *plvec_voxels[a];
      double coe = coeffs[a];

      // PLM(4) Co(win_size, Eigen::Matrix4d::Zero());
      Eigen::Matrix4d C;
      C.setZero();
      // for(int i=0; i<win_size; i++)
      // if(sig_orig[i].N != 0)
      // {
      //   Co[i] << sig_orig[i].P, sig_orig[i].v, sig_orig[i].v.transpose(), sig_orig[i].N;
      //   C += T[i] * Co[i] * T[i].transpose();
      // }

      PLM(4) &Co = *Cs[a];
      for (int i = 0; i < win_size; i++)
        if ((int)Co[i](3, 3) > 0)
          C += T[i] * Co[i] * T[i].transpose();

      double NN = C(3, 3);
      C = C / NN;
      // Eigen::Vector4d CF = C.block<4, 1>(0, 3);
      // cout << CF << endl << endl;
      // cout << C*F << endl;
      // exit(0);

      Eigen::Vector3d v_bar = C.block<3, 1>(0, 3);

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(C.block<3, 3>(0, 0) - v_bar * v_bar.transpose());
      Eigen::Vector3d lmbd = saes.eigenvalues();
      Eigen::Matrix3d Uev = saes.eigenvectors();

      residual += coe * lmbd[l];

      Eigen::Vector3d u[3] = {Uev.col(0), Uev.col(1), Uev.col(2)};
      Eigen::Matrix<double, 4, 6> U[3];

      PLV(-1)
      g_kl(3);
      for (int k = 0; k < 3; k++)
      {
        g_kl[k].resize(6 * win_size);
        g_kl[k].setZero();
        U[k].setZero();
        U[k].block<3, 3>(0, 0) = hat(u[k]);
        U[k].block<1, 3>(3, 3) = u[k];
      }

      for (int j = 0; j < win_size; j++)
        for (int k = 0; k < 3; k++)
          if (Co[j](3, 3) > 0.1)
          {
            Eigen::Matrix<double, 3, 4> SpTC = Sp * (T[j] - C * F) * Co[j] * T[j].transpose();
            Eigen::Matrix<double, 1, 6> g1, g2;
            g1 = u[l].transpose() * SpTC * U[k];
            g2 = u[k].transpose() * SpTC * U[l];

            g_kl[k].block<6, 1>(6 * j, 0) = (g1 + g2).transpose() / NN;
          }

      JacT += coe * g_kl[l];

      for (int i = 0; i < win_size; i++)
        if (Co[i](3, 3) > 0.1)
        {
          for (int j = 0; j < win_size; j++)
            if (Co[j](3, 3) > 0.1)
            {
              Eigen::Matrix4d Dij = Co[i] * F * Co[j];
              Eigen::Matrix<double, 6, 6> Hs = -2.0 / NN / NN * U[l].transpose() * T[i] * Dij * T[j].transpose() * U[l];

              if (i == j)
              {
                Hs += 2 / NN * U[l].transpose() * T[j] * Co[j] * T[j].transpose() * U[l];
                Eigen::Vector3d SpTC = Sp * T[j] * Co[j] * (T[j] - C * F).transpose() * Sp.transpose() * u[l];
                Eigen::Matrix3d h1 = hat(SpTC);
                Eigen::Matrix3d h2 = hat(u[l]);

                Hs.block<3, 3>(0, 0) += (h1 * h2 + h2 * h1) / NN;
              }

              Hess.block<6, 6>(6 * i, 6 * j) += coe * Hs;
            }
        }

      for (int k = 0; k < 3; k++)
        if (k != l)
          Hess += coe * 2.0 / (lmbd[l] - lmbd[k]) * g_kl[k] * g_kl[k].transpose();
    }

    double t1 = ros::Time::now().toSec();
    printf("t1: %lf\n", t1 - t0);

    // PLM(6) LL(win_size);
    // Eigen::Matrix3d zero33; zero33.setZero();
    // for(int i=0; i<win_size; i++)
    //   LL[i] << xs[i].R, zero33, hat(xs[i].p) * xs[i].R, xs[i].R;

    // for(int i=0; i<win_size; i++)
    // {
    //   JacT.block<6, 1>(6*i, 0) = LL[i].transpose() * JacT.block<6, 1>(6*i, 0);
    //   for(int j=0; j<win_size; j++)
    //   {
    //     Hess.block<6, 6>(6*i, 6*j) = LL[i].transpose() * Hess.block<6, 6>(6*i, 6*j) * LL[j];
    //   }
    // }

    // Eigen::Matrix3d zero33; zero33.setZero();
    // Eigen::MatrixXd LL(6*win_size, 6*win_size); LL.setZero();
    // for(int i=0; i<win_size; i++)
    // {
    //   LL.block<6, 6>(6*i, 6*i) << xs[i].R, zero33, hat(xs[i].p) * xs[i].R, xs[i].R;
    // }
    // JacT = LL.transpose() * JacT;
    // Hess = LL.transpose() * Hess * LL;
  }

  /**
   * @brief 并行地计算与体素相关的Hessian矩阵、雅可比向量和残差。这是为了在大规模优化问题中提高效率。
   *
   * @param xs 包含位姿等状态信息的向量
   * @param head 用于指定该线程应处理的体素范围
   * @param end 用于指定该线程应处理的体素范围
   * @param Hess 输出的Hessian矩阵
   * @param JacT 输出的雅可比向量
   * @param residual 输出的残差
   */
  void left_evaluate_acc2(const vector<IMUST> &xs, int head, int end, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT, double &residual)
  {
    // 初始化矩阵
    Hess.setZero();
    JacT.setZero();
    residual = 0;
    int l = 0;
    PLM(4)
    T(win_size);
    for (int i = 0; i < win_size; i++)
      T[i] << xs[i].R, xs[i].p, 0, 0, 0, 1;

    vector<PLM(4) *> Cs;
    for (int a = 0; a < plvec_voxels.size(); a++)
    {
      const vector<PointCluster> &sig_orig = *plvec_voxels[a];
      PLM(4) *Co = new PLM(4)(win_size, Eigen::Matrix4d::Zero());
      for (int i = 0; i < win_size; i++)
        Co->at(i) << sig_orig[i].P, sig_orig[i].v, sig_orig[i].v.transpose(), sig_orig[i].N;
      Cs.push_back(Co);
    }

    // 遍历head到end范围内的每一个体素
    for (int a = head; a < end; a++)
    {
      double coe = coeffs[a];
      Eigen::Matrix4d C;
      C.setZero();

      vector<int> Ns(win_size);

      PLM(4) &Co = *Cs[a];
      PLM(4)
      TC(win_size), TCT(win_size);
      for (int j = 0; j < win_size; j++)
        if ((int)Co[j](3, 3) > 0)
        {
          TC[j] = T[j] * Co[j];
          TCT[j] = TC[j] * T[j].transpose();
          C += TCT[j];

          Ns[j] = Co[j](3, 3);
        }

      double NN = C(3, 3);
      C = C / NN;
      Eigen::Vector3d v_bar = C.block<3, 1>(0, 3);

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(C.block<3, 3>(0, 0) - v_bar * v_bar.transpose());
      Eigen::Vector3d lmbd = saes.eigenvalues();
      Eigen::Matrix3d Uev = saes.eigenvectors();

      residual += coe * lmbd[l];

      Eigen::Vector3d u[3] = {Uev.col(0), Uev.col(1), Uev.col(2)};
      Eigen::Matrix<double, 6, 4> U[3];
      PLV(6)
      g_kl[3];
      for (int k = 0; k < 3; k++)
      {
        g_kl[k].resize(win_size);
        U[k].setZero();
        U[k].block<3, 3>(0, 0) = hat(-u[k]);
        U[k].block<3, 1>(3, 3) = u[k];
      }

      PLV(6)
      UlTCF(win_size, Eigen::Matrix<double, 6, 1>::Zero());

      Eigen::VectorXd JacT_iter(6 * win_size);
      for (int i = 0; i < win_size; i++)
        if (Ns[i] != 0)
        {
          Eigen::Matrix<double, 3, 4> temp = T[i].block<3, 4>(0, 0);
          temp.block<3, 1>(0, 3) -= v_bar;
          Eigen::Matrix<double, 4, 3> TC_TCFSp = TC[i] * temp.transpose();
          for (int k = 0; k < 3; k++)
          {
            Eigen::Matrix<double, 6, 1> g1, g2;
            g1 = U[k] * TC_TCFSp * u[l];
            g2 = U[l] * TC_TCFSp * u[k];

            g_kl[k][i] = (g1 + g2) / NN;
          }

          UlTCF[i] = (U[l] * TC[i]).block<6, 1>(0, 3);
          JacT.block<6, 1>(6 * i, 0) += coe * g_kl[l][i];

          // Eigen::Matrix<double, 6, 6> Hb(2.0/NN * U[l] * TCT[i] * U[l].transpose());

          Eigen::Matrix<double, 6, 6> Ha(-2.0 / NN / NN * UlTCF[i] * UlTCF[i].transpose());

          Eigen::Matrix3d Ell = 1.0 / NN * hat(TC_TCFSp.block<3, 3>(0, 0) * u[l]) * hat(u[l]);
          Ha.block<3, 3>(0, 0) += Ell + Ell.transpose();

          for (int k = 0; k < 3; k++)
            if (k != l)
              Ha += 2.0 / (lmbd[l] - lmbd[k]) * g_kl[k][i] * g_kl[k][i].transpose();

          Hess.block<6, 6>(6 * i, 6 * i) += coe * Ha;
        }

      for (int i = 0; i < win_size; i++)
        if (Ns[i] != 0)
        {
          Eigen::Matrix<double, 6, 6> Hb = U[l] * TCT[i] * U[l].transpose();
          Hess.block<6, 6>(6 * i, 6 * i) += 2.0 / NN * coe * Hb;
        }

      for (int i = 0; i < win_size - 1; i++)
        if (Ns[i] != 0)
        {
          for (int j = i + 1; j < win_size; j++)
            if (Ns[j] != 0)
            {
              Eigen::Matrix<double, 6, 6> Ha = -2.0 / NN / NN * UlTCF[i] * UlTCF[j].transpose();

              for (int k = 0; k < 3; k++)
                if (k != l)
                  Ha += 2.0 / (lmbd[l] - lmbd[k]) * g_kl[k][i] * g_kl[k][j].transpose();

              Hess.block<6, 6>(6 * i, 6 * j) += coe * Ha;
            }
        }
    }

    for (int i = 1; i < win_size; i++)
      for (int j = 0; j < i; j++)
        Hess.block<6, 6>(6 * i, 6 * j) = Hess.block<6, 6>(6 * j, 6 * i).transpose();
  }

  void evaluate_only_residual(const vector<IMUST> &xs, double &residual)
  {
    residual = 0;
    vector<PointCluster> sig_tran(win_size);
    int kk = 0; // The kk-th lambda value

    int gps_size = plvec_voxels.size();

    vector<double> ress(gps_size);

    for (int a = 0; a < gps_size; a++)
    {
      const vector<PointCluster> &sig_orig = *plvec_voxels[a];
      PointCluster sig = *sig_vecs[a];

      for (int i = 0; i < win_size; i++)
      {
        sig_tran[i].transform(sig_orig[i], xs[i]);
        sig += sig_tran[i];
      }

      Eigen::Vector3d vBar = sig.v / sig.N;
      Eigen::Matrix3d cmt = sig.P / sig.N - vBar * vBar.transpose();

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cmt);
      Eigen::Vector3d lmbd = saes.eigenvalues();

      residual += coeffs[a] * lmbd[kk];

      ress[a] = lmbd[kk];
    }

    // vector<double> ress_tem = ress;
    // sort(ress_tem.begin(), ress_tem.end());
    // double bound = 0.8;
    // bound = ress_tem[gps_size * bound];
    // coeffs = coeffs_back;

    // for(int a=0; a<gps_size; a++)
    //   if(ress[a] > bound)
    //     coeffs[a] = 0;
  }

  ~VOX_HESS()
  {
    int vsize = sig_vecs.size();
    // for(int i=0; i<vsize; i++)
    // {
    //   delete sig_vecs[i], sig_vecs[i] = nullptr;
    //   delete plvec_voxels[i], plvec_voxels[i] = nullptr;
    // }
  }
};

class VOXEL_MERGE
{
public:
  vector<const PointCluster *> sig_vecs;
  vector<const vector<PointCluster> *> plvec_voxels;

  PLV(3)
  centers, directs, evalues;
  vector<pcl::PointCloud<PointType>::Ptr> plptrs;

  void push_voxel(const vector<PointCluster> *vec_orig, const PointCluster *fix, Eigen::Vector3d &center, Eigen::Vector3d &direct, Eigen::Vector3d &evalue, pcl::PointCloud<PointType>::Ptr plptr = nullptr)
  {
    int process_size = 0;
    for (int i = 0; i < win_size; i++)
      if ((*vec_orig)[i].N != 0)
        process_size++;

    if (process_size < 2)
      return;

    plvec_voxels.push_back(vec_orig);
    sig_vecs.push_back(fix);
    centers.push_back(center);
    directs.push_back(direct);
    evalues.push_back(evalue);
    plptrs.push_back(plptr);
  }

  void reorganize(VOX_HESS &voxhess, pcl::PointCloud<PointType> &pl_send, pcl::PointCloud<PointType> &pl_cent, vector<IMUST> &x_buf)
  {
    static double cos1 = cos(8 / 57.3);
    static double cos2 = cos(80 / 57.3);

    int vsize = centers.size();
    if (vsize <= 0)
      return;

    vector<vector<int>> groups;
    groups.push_back(vector<int>());
    groups[0].push_back(0);
    for (int i = 1; i < vsize; i++)
    {
      Eigen::Vector3d c2 = centers[i];
      Eigen::Vector3d direct2 = directs[i];

      bool match = false;
      if (merge_enable)
      {
        int gsize = groups.size();
        for (int j = 0; j < gsize; j++)
        {
          int surf1 = groups[j][0];

          Eigen::Vector3d c2c = c2 - centers[surf1];
          double c2cd = c2c.norm();
          c2c /= c2cd;
          Eigen::Vector3d direct1 = directs[surf1];

          double dot1 = fabs(direct1.dot(direct2));
          double dot2 = fabs(c2c.dot(direct1));
          double dot3 = fabs(c2c.dot(direct2));

          bool c2flag = (dot2 < cos2 && dot3 < cos2) || (c2cd < 0.1);
          if (dot1 > cos1 && c2flag)
          {
            groups[j].push_back(i);
            match = true;
            break;
          }
        }
      }

      if (!match)
      {
        groups.push_back(vector<int>());
        groups.back().push_back(i);
      }
    }

    int g1size = groups.size();
    // for(int i=0; i<g1size; i++)
    // {
    //   float ref = 255.0*rand()/(RAND_MAX + 1.0f);

    //   int g2size = groups[i].size();
    //   for(int j=0; j<g2size; j++)
    //   {
    //     pcl::PointCloud<PointType>::Ptr plptr = plptrs[groups[i][j]];
    //     for(PointType ap : plptr->points)
    //     {
    //       Eigen::Vector3d pvec(ap.x, ap.y, ap.z);
    //       int pos = ap.intensity;
    //       pvec = x_buf[pos].R * pvec + x_buf[pos].p;
    //       ap.x = pvec[0]; ap.y = pvec[1]; ap.z = pvec[2];
    //       ap.intensity = ref;
    //       pl_send.push_back(ap);
    //     }
    //   }
    // }

    for (int i = 0; i < g1size; i++)
    {
      vector<int> &group = groups[i];
      int g2size = group.size();

      PointCluster *sig_vec = new PointCluster(*sig_vecs[group[0]]);
      vector<PointCluster> *plvec_voxel = new vector<PointCluster>(*plvec_voxels[group[0]]);
      pcl::PointCloud<PointType>::Ptr plptr = plptrs[group[0]];

      for (int j = 1; j < g2size; j++)
      {
        *sig_vec += *sig_vecs[group[j]];
        const vector<PointCluster> &plvec_tem = *plvec_voxels[group[j]];

        for (int k = 0; k < win_size; k++)
          if (plvec_tem[k].N != 0)
            (*plvec_voxel)[k] += plvec_tem[k];

        *plptr += *plptrs[group[j]];
      }

      int process_size = 0;
      for (int j = 0; j < win_size; j++)
        if ((*plvec_voxel)[j].N != 0)
          process_size++;
      if (process_size < 2)
      {
        delete sig_vec;
        delete plvec_voxel;
        continue;
      }

      double coe = 0;
      for (int j = 0; j < win_size; j++)
        coe += (*plvec_voxel)[j].N;

      voxhess.sig_vecs.push_back(sig_vec);
      voxhess.plvec_voxels.push_back(plvec_voxel);
      voxhess.coeffs.push_back(coe);
      voxhess.plptrs.push_back(plptr);
    }
  }
};
enum OCTO_STATE
{
  UNKNOWN = 0,
  MIN_NODE = 1,
  PLANE = 2
};
class OCTO_TREE_NODE
{
public:
  OCTO_STATE octo_state; // 0(unknown), 1(mid node), 2(plane)
  int push_state;
  // layer 节点在八叉树中的深度，从根开始计数
  int layer;
  // vec_orig 存储的是原始的点云数据，vec_tran 存储的是经过旋转和平移变换后的点云数据。下标都为帧数
  vector<PLV(3)> vec_orig, vec_tran;
  // sig_orig 存储原始点云数据的统计信息，sig_tran 存储经过旋转和平移变换后的点云数据的统计信息。下标都为帧数
  vector<PointCluster> sig_orig, sig_tran;
  PointCluster fix_point;
  PLV(3)
  vec_fix;

  OCTO_TREE_NODE *leaves[8];
  // 体素的中心位置
  float voxel_center[3];
  // 体素大小的四分之一，很可能是为了在八叉树中表示子体素的大小或者表示体素中心到边界的距离
  float quater_length;

  Eigen::Vector3d center, direct, value_vector; // temporal
  double decision,                              // 体素的最小特征值与次大特征值的比率
      ref;

  OCTO_TREE_NODE()
  {
    octo_state = OCTO_STATE::UNKNOWN;
    push_state = 0;
    vec_orig.resize(win_size);
    vec_tran.resize(win_size);
    sig_orig.resize(win_size);
    sig_tran.resize(win_size);
    for (int i = 0; i < 8; i++)
      leaves[i] = nullptr;
    ref = 255.0 * rand() / (RAND_MAX + 1.0f);
    layer = 0;
  }

  bool judge_eigen(int win_count)
  {
    // 初始化一个体素
    PointCluster covMat;
    // 遍历所有帧再该体素位置的PointCluster体素,并更新统计信息到covMat
    for (int i = 0; i < win_count; i++)
      covMat += sig_tran[i];

    // 计算特征值和特征向量
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat.cov());
    value_vector = saes.eigenvalues();
    // 计算体素中的数据的中心
    center = covMat.v / covMat.N;
    // 获取最小的特征向量（对应于数据分布的最小变化方向,平面法向量方向）
    direct = saes.eigenvectors().col(0);

    // 计算最小和次大特征值的比率来判断数据分布是否近似于一个平面
    decision = saes.eigenvalues()[0] / saes.eigenvalues()[1];
    // decision = saes.eigenvalues()[0];

    // 先将体素进一步划分为8个小体素（通过对比每个点与中心点的坐标）
    // double eva0 = saes.eigenvalues()[0];
    // 调整数据的中心点,sqrt(eva0)可以被看作是这个方向上的标准偏差
    // 将中心点沿着最小变化方向移动一个距离，这个距离大约等于3倍的标准偏差。这样的移动可能是为了更好地捕捉数据的整体分布
    // center += 3 * sqrt(eva0) * direct;
    // vector<PointCluster> covMats(8);
    // for(int i=0; i<win_count; i++)
    // {
    //   for(Eigen::Vector3d &pvec: vec_tran[i])
    //   {
    //     int xyz[3] = {0, 0, 0};
    //     for(int k=0; k<3; k++)
    //       if(pvec[k] > center[k])
    //         xyz[k] = 1;
    //     int leafnum = 4*xyz[0] + 2*xyz[1] + xyz[2];
    //     covMats[leafnum].push(pvec);
    //   }
    // }

    // 为了判断子体素中的数据是否构成一个平面，设置阈值
    // double ratios[2] = {1.0/(3.0*3.0), 2.0*2.0};
    // 检查子体素是否满足平面阈值
    // int num_all = 0, num_qua = 0;
    // for(int i=0; i<8; i++)
    // {
    //   if(covMats[i].N < 10) continue;
    //   Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMats[i].cov());
    //   double child_eva0 = (saes.eigenvalues()[0]);
    //   if(child_eva0 > ratios[0]*eva0 && child_eva0 < ratios[1]*eva0)
    //     num_qua++;
    //   num_all++;
    // }
    // 计算满足条件的子体素的比例
    // double prop = 1.0 * num_qua / num_all;

    // 根据最小和最大特征值的比率返回是否是平面体素
    return (decision < eigen_value_array[layer]);
    // return (decision < eigen_value_array[layer] && prop > 0.5);
  }

  /**
   * @brief 函数是在 OCTO_TREE_NODE 类中的，它的目的是根据当前体素内的点云数据进行细分，然后将这些数据分配到合适的子体素（子节点）中
   *
   * @param ci 当前窗口的点云索引
   */
  void cut_func(int ci)
  {
    // 获取指定帧 ci 的原始和转换后的点云数据
    PLV(3) &pvec_orig = vec_orig[ci];
    PLV(3) &pvec_tran = vec_tran[ci];

    // 遍历指定帧在当前体素的所有点
    uint a_size = pvec_tran.size();
    for (uint j = 0; j < a_size; j++)
    {
      // 使用三维数组 xyz，函数确定每个点相对于当前体素中心的位置
      int xyz[3] = {0, 0, 0};
      // 如果点的某一坐标大于体素中心的对应坐标，那么该坐标的值设置为1，否则保持为0
      for (uint k = 0; k < 3; k++)
        if (pvec_tran[j][k] > voxel_center[k])
          xyz[k] = 1;
      // 计算出点所在的子体素编号。这实际上是一个简单的三维到一维的映射方法，用于确定点应该放入哪个子节点。
      int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
      // 如果对应的子节点（子体素）是空的（即 leaves[leafnum] == nullptr），函数将为其分配内存并初始化它
      if (leaves[leafnum] == nullptr)
      {
        // 每个体素可以进一步细分为八个等大小的子体素。当你有一个立方体并想要将其细分时，可以想象将其沿每个轴（X、Y和Z轴）的中心分割成两部分
        // 子节点的中心坐标是基于当前体素的中心坐标和 quater_length（四分之一的体素大小）计算得出的
        leaves[leafnum] = new OCTO_TREE_NODE();
        leaves[leafnum]->voxel_center[0] = voxel_center[0] + (2 * xyz[0] - 1) * quater_length;
        leaves[leafnum]->voxel_center[1] = voxel_center[1] + (2 * xyz[1] - 1) * quater_length;
        leaves[leafnum]->voxel_center[2] = voxel_center[2] + (2 * xyz[2] - 1) * quater_length;
        leaves[leafnum]->quater_length = quater_length / 2;
        leaves[leafnum]->layer = layer + 1;
      }

      // 函数将当前遍历到的点添加到对应的子节点的原始和转换后的点云数据中
      leaves[leafnum]->vec_orig[ci].push_back(pvec_orig[j]);
      leaves[leafnum]->vec_tran[ci].push_back(pvec_tran[j]);

      if (leaves[leafnum]->octo_state != 1)
      {
        leaves[leafnum]->sig_orig[ci].push(pvec_orig[j]);
        leaves[leafnum]->sig_tran[ci].push(pvec_tran[j]);
      }
    }

    // 最后，函数使用 swap() 方法清空当前体素的原始和转换后的点云数据，释放内存
    PLV(3)
    ().swap(pvec_orig);
    PLV(3)
    ().swap(pvec_tran);
  }

  /**
   * @brief OCTO_TREE_NODE的函数,将OCTO_TREE_NODE体素再细分，检测更细的平面体素
   *
   * @param win_count 窗口大小=总帧数
   */
  void recut(int win_count)
  {
    // // 首先检查体素的状态是否为 UNKNOWN
    if (octo_state == OCTO_STATE::UNKNOWN)
    {
      // 体素内的点数
      int point_size = 0;
      for (int i = 0; i < win_count; i++)
        point_size += sig_orig[i].N;

      push_state = 0;
      // 检查当前体素的点数是否小于最小点数,小于最小点数则丢弃这个体素
      if (point_size <= min_ps)
      {
        // 设置体素类型为最小体素
        octo_state = OCTO_STATE::MIN_NODE;
        // 清除与该体素相关的所有数据并返回
        vector<PLV(3)>().swap(vec_orig);
        vector<PLV(3)>().swap(vec_tran);
        vector<PointCluster>().swap(sig_orig);
        vector<PointCluster>().swap(sig_tran);
        return;
      }

      if (judge_eigen(win_count))
      {
        if (point_size > layer_size[layer])
        {
          octo_state = OCTO_STATE::PLANE;
          // TODO:如果这里不需要rivz显示的话就可以删除掉
          // vector<PLV(3)>().swap(vec_orig);
          // vector<PLV(3)>().swap(vec_tran);
          // vector<PointCluster>().swap(sig_orig);
          // vector<PointCluster>().swap(sig_tran);
        }

        if (point_size > min_ps)
          push_state = 1;
        return;
      }
      else if (layer == layer_limit)
      {
        // 如果不是平面,且当前体素的层数达到了设定的最大层数，则将当前体素设置为最小节点
        octo_state = OCTO_STATE::MIN_NODE;
        vector<PLV(3)>().swap(vec_orig);
        vector<PLV(3)>().swap(vec_tran);
        vector<PointCluster>().swap(sig_orig);
        vector<PointCluster>().swap(sig_tran);
        return;
      }
      // 如果体素既不是平面特征也没有达到最大层级，函数将对当前体素进行再细分
      // 首先清除与该体素相关的统计信息
      octo_state = OCTO_STATE::MIN_NODE;
      vector<PointCluster>().swap(sig_orig);
      vector<PointCluster>().swap(sig_tran);
      // 对然后调用 cut_func(i) 对每一帧的数据进行再细分
      for (int i = 0; i < win_count; i++)
        cut_func(i);
    }
    // else
    //   cut_func(win_count - 1);

    // 此时是剩下的进行细分后的体素,需要对子体素进行判断,筛选得到平面体素
    for (int i = 0; i < 8; i++)
      if (leaves[i] != nullptr)
        leaves[i]->recut(win_count);
  }

  void to_margi(int mg_size, vector<IMUST> &x_poses, int win_count)
  {
    if (octo_state != 1)
    {
      if (!x_poses.empty())
        for (int i = 0; i < win_count; i++)
        {
          sig_tran[i].transform(sig_orig[i], x_poses[i]);
          plvec_trans(vec_orig[i], vec_tran[i], x_poses[i]);
        }

      if (fix_point.N < 50 && push_state == 1)
        for (int i = 0; i < mg_size; i++)
        {
          fix_point += sig_tran[i];
          vec_fix.insert(vec_fix.end(), vec_tran[i].begin(), vec_tran[i].end());
        }

      for (int i = mg_size; i < win_count; i++)
      {
        sig_orig[i - mg_size] = sig_orig[i];
        sig_tran[i - mg_size] = sig_tran[i];
        vec_orig[i - mg_size].swap(vec_orig[i]);
        vec_tran[i - mg_size].swap(vec_tran[i]);
      }

      for (int i = win_count - mg_size; i < win_count; i++)
      {
        sig_orig[i].clear();
        sig_tran[i].clear();
        vec_orig[i].clear();
        vec_tran[i].clear();
      }
    }
    else
      for (int i = 0; i < 8; i++)
        if (leaves[i] != nullptr)
          leaves[i]->to_margi(mg_size, x_poses, win_count);
  }

  ~OCTO_TREE_NODE()
  {
    for (int i = 0; i < 8; i++)
      if (leaves[i] != nullptr)
        delete leaves[i];
  }

  void tras_display(pcl::PointCloud<PointType> &pl_feat, int win_count)
  {
    if (octo_state != 1)
    {
      if (push_state != 1)
        return;

      PointType ap;
      ap.intensity = ref;

      int tsize = 0;
      for (int i = 0; i < win_count; i++)
        tsize += vec_tran[i].size();
      if (tsize < 100)
        return;

      for (int i = 0; i < win_count; i++)
        for (Eigen::Vector3d pvec : vec_tran[i])
        {
          ap.x = pvec.x();
          ap.y = pvec.y();
          ap.z = pvec.z();
          // ap.normal_x = sqrt(value_vector[1] / value_vector[0]);
          // ap.normal_y = sqrt(value_vector[2] / value_vector[0]);
          // ap.normal_z = sqrt(value_vector[0]);
          // ap.normal_x = voxel_center[0];
          // ap.normal_y = voxel_center[1];
          // ap.normal_z = voxel_center[2];
          // ap.curvature = quater_length * 4;

          pl_feat.push_back(ap);
        }
    }
    else
    {
      // if(layer != layer_limit)
      // {
      //   PointType ap;
      //   ap.x = voxel_center[0];
      //   ap.y = voxel_center[1];
      //   ap.z = voxel_center[2];
      //   pl_cent.push_back(ap);
      // }

      for (int i = 0; i < 8; i++)
        if (leaves[i] != nullptr)
          leaves[i]->tras_display(pl_feat, win_count);
    }
  }

  void tras_merge(VOXEL_MERGE &vlmg, int win_count)
  {
    if (octo_state != 1)
    {
      if (push_state == 1)
      {
        pcl::PointCloud<PointType>::Ptr plptr(new pcl::PointCloud<PointType>());
        for (int i = 0; i < win_count; i++)
        {
          PointType ap;
          ap.intensity = i;
          for (Eigen::Vector3d &pvec : vec_orig[i])
          // for(Eigen::Vector3d &pvec : vec_tran[i])
          {
            ap.x = pvec[0];
            ap.y = pvec[1];
            ap.z = pvec[2];
            plptr->push_back(ap);
          }
        }

        int psize = 0;
        for (int i = 0; i < win_count; i++)
          psize += vec_orig[i].size();

        if (psize > 100)
          vlmg.push_voxel(&sig_orig, &fix_point, center, direct, value_vector, plptr);
      }
    }
    else
    {
      for (int i = 0; i < 8; i++)
        if (leaves[i] != nullptr)
          leaves[i]->tras_merge(vlmg, win_count);
    }
  }

  /**
   * @brief 获取所有的平面体素
   *
   * @param vox_opt 所有的平面体素
   * @param win_count 窗口大小=总帧数
   */
  void tras_opt(VOX_HESS &vox_opt, int win_count)
  {
    // 判断是否为平面体素
    if (octo_state == PLANE)
      // 如果当前体素是平面体素，那么将其原始点云和点云统计信息插入到VOX_HESS对象中
      vox_opt.push_voxel(&sig_orig, &fix_point, decision, layer);
    else
      // 如果不是平面体素，遍历子节点递归到体素为PLANE或者子体素全为nullptr停止
      for (int i = 0; i < 8; i++)
        if (leaves[i] != nullptr)
          leaves[i]->tras_opt(vox_opt, win_count);

    // if (octo_state != 1)
    // {
    //   int points_size = 0;
    //   for (int i = 0; i < win_count; i++)
    //     points_size += sig_orig[i].N;

    //   if (points_size < min_ps)
    //     return;

    //   if (push_state == 1)
    //     vox_opt.push_voxel(&sig_orig, &fix_point, decision, layer);
    // }
    // else
    // {
    //   for (int i = 0; i < 8; i++)
    //     if (leaves[i] != nullptr)
    //       leaves[i]->tras_opt(vox_opt, win_count);
    // }
  }
};

class OCTO_TREE_ROOT : public OCTO_TREE_NODE
{
public:
  bool is2opt;
  int life;
  vector<int> each_num;

  OCTO_TREE_ROOT()
  {
    // octo_state = OCTO_STATE::UNKNOWN;
    is2opt = true;
    life = life_span;
    each_num.resize(win_size);
    for (int i = 0; i < win_size; i++)
      each_num[i] = 0;
  }

  void marginalize(int mg_size, vector<IMUST> &x_poses, int win_count)
  {
    to_margi(mg_size, x_poses, win_count);

    int left_size = 0;
    for (int i = mg_size; i < win_count; i++)
    {
      each_num[i - mg_size] = each_num[i];
      left_size += each_num[i - mg_size];
    }

    if (left_size == 0)
      is2opt = false;

    for (int i = win_count - mg_size; i < win_count; i++)
      each_num[i] = 0;
  }
};

bool iter_stop(Eigen::VectorXd &dx, double thre = 1e-7, int win_size = 0)
{
  // int win_size = dx.rows() / 6;
  if (win_size == 0)
    win_size = dx.rows() / 6;

  double angErr = 0, tranErr = 0;
  for (int i = 0; i < win_size; i++)
  {
    angErr += dx.block<3, 1>(6 * i, 0).norm();
    tranErr += dx.block<3, 1>(6 * i + 3, 0).norm();
  }

  angErr /= win_size;
  tranErr /= win_size;
  return (angErr < thre) && (tranErr < thre);
}

class BALM2
{
public:
  BALM2() {}

  double divide_thread_right(vector<IMUST> &x_stats, VOX_HESS &voxhess, vector<IMUST> &x_ab, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT)
  {
    int thd_num = 4;
    double residual = 0;
    Hess.setZero();
    JacT.setZero();
    PLM(-1)
    hessians(thd_num);
    PLV(-1)
    jacobins(thd_num);

    for (int i = 0; i < thd_num; i++)
    {
      hessians[i].resize(6 * win_size, 6 * win_size);
      jacobins[i].resize(6 * win_size);
    }

    int tthd_num = thd_num;
    vector<double> resis(tthd_num, 0);
    int g_size = voxhess.plvec_voxels.size();
    if (g_size < tthd_num)
      tthd_num = 1;

    vector<thread *> mthreads(tthd_num);
    double part = 1.0 * g_size / tthd_num;
    for (int i = 0; i < tthd_num; i++)
      mthreads[i] = new thread(&VOX_HESS::acc_evaluate2, &voxhess, x_stats, part * i, part * (i + 1), ref(hessians[i]), ref(jacobins[i]), ref(resis[i]));

    for (int i = 0; i < tthd_num; i++)
    {
      mthreads[i]->join();
      Hess += hessians[i];
      JacT += jacobins[i];
      residual += resis[i];
      delete mthreads[i];
    }

    return residual;
  }

  /**
   * @brief 函数的目的是并行地计算与体素相关的Hessian矩阵、雅可比向量和残差
   *
   * @param x_stats 绝对位姿向量
   * @param voxhess 所有平面体素
   * @param x_ab 相对位姿向量
   * @param Hess 输出的Hessian矩阵
   * @param JacT 输出的梯度矩阵(雅可比向量的转置)
   * @return double
   */
  double divide_thread_left(vector<IMUST> &x_stats, VOX_HESS &voxhess, vector<IMUST> &x_ab, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT)
  {
    // 初始化Hessian矩阵,雅可比向量,残差设置为零
    int thd_num = 4;
    double residual = 0;
    Hess.setZero();
    JacT.setZero();

    // 根据thd_num线程数初始化hessians和jacobins向量来存储每个线程的局部Hessian和雅可比值。
    PLM(-1)
    hessians(thd_num);
    PLV(-1)
    jacobins(thd_num);

    // 计算每个线程应处理的体素数量。这是通过将体素总数g_size除以线程数tthd_num来完成的
    for (int i = 0; i < thd_num; i++)
    {
      hessians[i].resize(6 * win_size, 6 * win_size);
      jacobins[i].resize(6 * win_size);
    }

    int tthd_num = thd_num;
    vector<double> resis(tthd_num, 0);
    // 如果体素的数量小于线程的数量，那么只使用一个线程
    int g_size = voxhess.plvec_voxels.size();
    if (g_size < tthd_num)
      tthd_num = 1;

    // 为每个线程创建一个新的线程对象并开始执行
    vector<thread *> mthreads(tthd_num);
    double part = 1.0 * g_size / tthd_num;
    // 每个线程调用VOX_HESS::acc_evaluate2函数来计算其分配的体素的Hessian、雅可比和残差。这是通过给每个线程分配一个体素范围来实现的
    for (int i = 0; i < tthd_num; i++)
      mthreads[i] = new thread(&VOX_HESS::left_evaluate_acc2, &voxhess, x_stats, part * i, part * (i + 1), ref(hessians[i]), ref(jacobins[i]), ref(resis[i]));

    // 将每个线程计算的Hessian和雅可比加到总的Hessian矩阵和雅可比向量中
    for (int i = 0; i < tthd_num; i++)
    {
      mthreads[i]->join();
      Hess += hessians[i];
      JacT += jacobins[i];
      residual += resis[i];
      delete mthreads[i];
    }

    return residual;
  }

  double only_residual(vector<IMUST> &x_stats, VOX_HESS &voxhess, vector<IMUST> &x_ab)
  {
    double residual1 = 0, residual2 = 0;

    voxhess.evaluate_only_residual(x_stats, residual2);
    return (residual1 + residual2);
  }

  /**
   * @brief 执行非线性最优化，调整位姿参数来最小化与特征体素(voxhess)相关的误差
   *
   * @param x_stats
   * @param voxhess
   */
  bool damping_iter(vector<IMUST> &x_stats, VOX_HESS &voxhess, int max_iter = 15, double ep = 1e-6)
  {
    // 初始化一个记录每个位姿状态参与平面数量的向量
    vector<int> planes(x_stats.size(), 0);
    // 遍历所有的体素平面
    for (int i = 0; i < voxhess.plvec_voxels.size(); i++)
    {
      // 遍历当前体素的所有帧,如果N不为0,则次帧有当前平面体素
      for (int j = 0; j < voxhess.plvec_voxels[i]->size(); j++)
        if (voxhess.plvec_voxels[i]->at(j).N != 0)
          planes[j]++;
    }
    // 对参与平面数量进行排序
    sort(planes.begin(), planes.end());
    // 如果最小的平面数量小于20，认为初始误差太大，退出程序
    if (planes[0] < 20)
    {
      printf("Initial error too large.\n");
      printf("Please loose plane determination criteria for more planes.\n");
      printf("The optimization is terminated.\n");
      // exit(0);
    }

    // 定义Levenberg-Marquardt算法中的变量
    double u = 0.01, v = 2;                        // u和v是LM算法的参数，其中u是阻尼系数，v是更新因子
    Eigen::MatrixXd D(6 * win_size, 6 * win_size), // 阻尼矩阵????
        Hess(6 * win_size, 6 * win_size);          // hessian矩阵
    Eigen::VectorXd JacT(6 * win_size),            // 梯度矩阵
        dxi(6 * win_size);                         // 增量

    D.setIdentity();
    double residual1, residual2, q;
    bool is_calc_hess = true;
    // 复制位姿状态用于临时存储
    vector<IMUST> x_stats_temp = x_stats;

    // 初始化与相对位姿有关的向量
    vector<IMUST> x_ab(win_size);
    x_ab[0] = x_stats[0]; // 保存第一帧的位姿:单位阵
    for (int i = 1; i < win_size; i++)
    {
      // 计算相邻帧的相对位姿
      x_ab[i].p = x_stats[i - 1].R.transpose() * (x_stats[i].p - x_stats[i - 1].p);
      x_ab[i].R = x_stats[i - 1].R.transpose() * x_stats[i].R;
    }

    bool isConverged = false;
    // 进行最多10次迭代
    for (int i = 0; i < max_iter; i++)
    {
      // 如果需要计算Hessian矩阵
      if (is_calc_hess)
      {
        // 计算残差和雅可比矩阵
        // residual1 = divide_thread_right(x_stats, voxhess, x_ab, Hess, JacT);
        residual1 = divide_thread_left(x_stats, voxhess, x_ab, Hess, JacT);
      }

      // 更新阻尼矩阵
      D.diagonal() = Hess.diagonal();
      // 计算位姿状态的增量
      dxi = (Hess + u * D).ldlt().solve(-JacT);

      // 应用增量更新位姿状态
      for (int j = 0; j < win_size; j++)
      {
        // right update
        // x_stats_temp[j].R = x_stats[j].R * Exp(dxi.block<3, 1>(DVEL*j, 0));
        // x_stats_temp[j].p = x_stats[j].p + dxi.block<3, 1>(DVEL*j+3, 0);

        // 使用左乘更新规则
        // left update
        Eigen::Matrix3d dR = Exp(dxi.block<3, 1>(DVEL * j, 0));
        x_stats_temp[j].R = dR * x_stats[j].R;
        x_stats_temp[j].p = dR * x_stats[j].p + dxi.block<3, 1>(DVEL * j + 3, 0);
      }
      // 计算预期降低量
      double q1 = 0.5 * dxi.dot(u * D * dxi - JacT);

      // 仅计算新位姿状态的残差
      residual2 = only_residual(x_stats_temp, voxhess, x_ab);

      // 计算实际降低量
      q = (residual1 - residual2);
      // printf("iter%d: (%lf %lf) u: %lf v: %.1lf q: %.3lf %lf %lf\n", i, residual1, residual2, u, v, q / q1, q1, q);
      LOG(INFO) << "iter" << i << ": residual before and after (" << residual1 << "," << residual2 << ") u: "
                << u << " v: " << v << " q: " << q / q1 << " " << q1 << " " << q;

      // 如果实际降低量大于0，则更新位姿状态，调整阻尼因子u和v
      if (q > 0)
      {
        x_stats = x_stats_temp;

        q = q / q1;
        v = 2;
        q = 1 - pow(2 * q - 1, 3);
        u *= (q < one_three ? one_three : q);
        is_calc_hess = true;
      }
      else // 如果实际降低量小于或等于0，增加阻尼因子u并增大v
      {
        u = u * v;
        v = 2 * v;
        is_calc_hess = false;
      }

      // if(iter_stop(dxi2, 1e-4))
      // if(iter_stop(dxi, 1e-6))
      //   break;

      if (fabs(residual1 - residual2) / residual1 < ep)
      {
        isConverged = true;
        break;
      }
    }

    // 更新为相对位姿
    IMUST es0 = x_stats[0];
    for (uint i = 0; i < x_stats.size(); i++)
    {
      x_stats[i].p = es0.R.transpose() * (x_stats[i].p - es0.p);
      x_stats[i].R = es0.R.transpose() * x_stats[i].R;
    }
    return isConverged;
  }
};

/**
 * @brief 将点云数据进行体素化，并建立一个体素地图
 *
 * @param feat_map 映射了体素位置到八叉树结构（OCTO_TREE_ROOT*）。这个结构存储了与每个体素相关的数据。
 * @param pl_feat 要处理的点云
 * @param x_key 表示当前处理点云的位姿状态，包括旋转 R 和平移 p
 * @param fnum 一个整数，通常是指当前帧的索引
 */
void cut_voxel(unordered_map<VOXEL_LOC, OCTO_TREE_ROOT *> &feat_map, pcl::PointCloud<PointType> &pl_feat, const IMUST &x_key, int fnum)
{
  // 用于存储变换后点的体素坐标
  float loc_xyz[3];
  // 遍历点云中的每个点
  for (PointType &p_c : pl_feat.points)
  {
    // 将点的坐标转换为Eigen的Vector3d类型,并转到世界坐标系（第一帧局部坐标系）
    Eigen::Vector3d pvec_orig(p_c.x, p_c.y, p_c.z);
    Eigen::Vector3d pvec_tran = x_key.R * pvec_orig + x_key.p;

    // 计算每个维度上变换后点的体素位置
    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = pvec_tran[j] / voxel_size;
      if (loc_xyz[j] < 0)
      {
        // 当点的坐标为负数时，仅仅除以 voxel_size 会导致体素坐标不准确
        // 通过减去1.0，代码实际上是将这些坐标向下调整一个完整的体素单位，以确保它们被正确地映射到体素网格中
        loc_xyz[j] -= 1.0;
      }
    }

    // 创建体素位置的索引
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    // 尝试在地图中找到这个体素
    auto iter = feat_map.find(position);
    if (iter != feat_map.end())
    {
      // 如果体素已经存在
      // 如果体素状态不是平面，更新原始点和变换后的点
      // if (iter->second->octo_state != 2)
      {
        iter->second->vec_orig[fnum].push_back(pvec_orig);
        iter->second->vec_tran[fnum].push_back(pvec_tran);
      }
      // 如果体素状态不是1，更新信号强度原始点和变换后的点
      // if (iter->second->octo_state != 1)
      {
        iter->second->sig_orig[fnum].push(pvec_orig);
        iter->second->sig_tran[fnum].push(pvec_tran);
      }
      // 设置体素为待优化状态
      iter->second->is2opt = true;
      // 更新体素的生命周期
      iter->second->life = life_span;
      // 增加体素中的点数
      iter->second->each_num[fnum]++;
    }
    else
    {
      // 如果体素不存在
      // 创建一个新的八叉树根节点
      OCTO_TREE_ROOT *ot = new OCTO_TREE_ROOT();
      // 更新原始点和变换后的点
      ot->vec_orig[fnum].push_back(pvec_orig);
      ot->vec_tran[fnum].push_back(pvec_tran);
      // 更新信号强度原始点和变换后的点
      ot->sig_orig[fnum].push(pvec_orig);
      ot->sig_tran[fnum].push(pvec_tran);
      // 更新体素中的点数
      ot->each_num[fnum]++;

      // 设置体素中心和尺寸
      ot->voxel_center[0] = (0.5 + position.x) * voxel_size;
      ot->voxel_center[1] = (0.5 + position.y) * voxel_size;
      ot->voxel_center[2] = (0.5 + position.z) * voxel_size;
      ot->quater_length = voxel_size / 4.0;
      ot->layer = 0;
      feat_map[position] = ot;
    }
  }
}

#endif
