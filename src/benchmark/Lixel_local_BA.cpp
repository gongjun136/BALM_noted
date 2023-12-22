#include "tools.hpp"
#include <ros/ros.h>
#include <Eigen/Eigenvalues>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <geometry_msgs/PoseArray.h>
#include <random>
#include <ctime>
#include <tf/transform_broadcaster.h>
#include "bavoxel.hpp"

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <malloc.h>

using namespace std;

/**
 * @brief 定义一个模板函数，用于发布任何类型的点云数据
 *
 * @tparam T
 * @param pl
 * @param pub
 */
template <typename T>
void pub_pl_func(T &pl, ros::Publisher &pub)
{
  // 设置点云的高度，对于非有序点云，通常为1
  pl.height = 1;
  // 设置点云的宽度，等于点云中点的数量
  pl.width = pl.size();
  // 定义一个ROS的PointCloud2消息，用于发布点云
  sensor_msgs::PointCloud2 output;
  // 将PCL点云转换为ROS消息格式
  pcl::toROSMsg(pl, output);
  // 设置消息的frame_id，这通常是点云数据的参考坐标系
  output.header.frame_id = "camera_init";
  // 设置时间戳为当前时间，确保同步
  output.header.stamp = ros::Time::now();
  // 使用提供的ROS发布器发布点云消息
  pub.publish(output);
}

ros::Publisher pub_path, pub_test, pub_show, pub_cute;

int read_pose(vector<double> &tims, PLM(3) & rots, PLV(3) & poss, string prename)
{
  string readname = prename + "frame_opt.csv";

  // cout << readname << endl;
  ifstream inFile(readname);

  if (!inFile.is_open())
  {
    printf("open fail\n");
    return 0;
  }

  int pose_size = 0;
  string lineStr, str;
  Eigen::Matrix4d aff;
  vector<double> nums;

  int ord = 0;
  while (getline(inFile, lineStr))
  {
    ord++;
    stringstream ss(lineStr);
    while (getline(ss, str, ','))
      nums.push_back(stod(str));

    if (ord == 4)
    {
      for (int j = 0; j < 16; j++)
        aff(j) = nums[j];

      Eigen::Matrix4d affT = aff.transpose();

      rots.push_back(affT.block<3, 3>(0, 0));
      poss.push_back(affT.block<3, 1>(0, 3));
      // tims.push_back(affT(3, 3));
      nums.clear();
      ord = 0;
      pose_size++;
    }
  }

  return pose_size;
}

/**
 * @brief 从给定的路径读取点云数据和位姿信息，并将它们存储在提供的向量中
 *
 * @param x_buf IMUST 类型的向量引用（用于存储位姿信息
 * @param pl_fulls pcl::PointCloud<PointType>::Ptr 类型的向量引用（用于存储点云数据）
 * @param prename 文件前缀路径的字符串
 */
void read_file(vector<IMUST> &x_buf, vector<pcl::PointCloud<PointType>::Ptr> &pl_fulls, string &prename)
{
  // prename = prename + "/datas/benchmark_realworld/";

  // 定义位姿向量
  PLV(3)
  poss;
  PLM(3)
  rots;
  vector<double> tims;
  // 读取位姿信息，返回位姿的数量
  int pose_size = read_pose(tims, rots, poss, prename);

  // 遍历每个位姿对应的pcd点云
  for (int m = 0; m < pose_size; m++)
  {
    // 加载对应的pcd点云
    std::stringstream ss;
    ss << std::setw(8) << std::setfill('0') << m;
    string filename = prename + "pcd/" + ss.str() + ".pcd";

    pcl::PointCloud<PointType>::Ptr pl_ptr(new pcl::PointCloud<PointType>());
    pcl::PointCloud<pcl::PointXYZI> pl_tem;
    pcl::io::loadPCDFile(filename, pl_tem);
    for (pcl::PointXYZI &pp : pl_tem.points)
    {
      PointType ap;
      ap.x = pp.x;
      ap.y = pp.y;
      ap.z = pp.z;
      ap.intensity = pp.intensity;
      pl_ptr->push_back(ap);
    }
    // 将点云插入点云数组
    pl_fulls.push_back(pl_ptr);

    // 将位姿插入点云数组
    IMUST curr;
    curr.R = rots[m];
    curr.p = poss[m];
    // curr.t = tims[m];
    x_buf.push_back(curr);
  }
}

/**
 * @brief 处理和显示点云数据和位姿信息
 *
 * @param x_buf IMUST 类型的位姿向量
 * @param pl_fulls 包含点云指针的向量
 */
void data_show(vector<IMUST> x_buf, vector<pcl::PointCloud<PointType>::Ptr> &pl_fulls)
{
  // 获取第一个IMUST对象作为参考位姿
  IMUST es0 = x_buf[0];
  // 遍历所有IMUST对象，将它们的位置和旋转调整为相对于第一个位姿的相对位置和旋转
  for (uint i = 0; i < x_buf.size(); i++)
  {
    x_buf[i].p = es0.R.transpose() * (x_buf[i].p - es0.p);
    x_buf[i].R = es0.R.transpose() * x_buf[i].R;
  }

  // 定义两个点云，一个用于发送显示，一个用于显示路径
  pcl::PointCloud<PointType> pl_send, pl_path;
  // 获取位姿数量，也是窗口的大小
  int winsize = x_buf.size();
  // 遍历所有位姿和点云数据
  for (int i = 0; i < winsize; i++)
  {
    // 获取当前位姿对应的点云数据
    pcl::PointCloud<PointType> pl_tem = *pl_fulls[i];
    // 对点云数据进行降采样，使用体素网格滤波器，体素大小设置为0.05
    down_sampling_voxel(pl_tem, 0.05);
    // 对点云应用变换，根据当前位姿调整点云的位置和旋转
    pl_transform(pl_tem, x_buf[i]);
    // 将变换后的点云添加到发送缓冲点云中
    pl_send += pl_tem;
    // 每200帧发布一次点云数据，以及最后一帧点云数据
    if ((i % 200 == 0 && i != 0) || i == winsize - 1)
    {
      // 发布点云
      pub_pl_func(pl_send, pub_show);
      pl_send.clear();
      sleep(0.5);
    }

    // 创建一个点，设置为当前位姿的位置
    PointType ap;
    ap.x = x_buf[i].p.x();
    ap.y = x_buf[i].p.y();
    ap.z = x_buf[i].p.z();
    ap.curvature = i;
    // 将点添加到路径点云中
    pl_path.push_back(ap);
  }
  // 发布路径点集
  pub_pl_func(pl_path, pub_path);
}

int main(int argc, char **argv)
{

  google::InitGoogleLogging(argv[0]);
  FLAGS_log_dir = "/home/gj/catkin_ws_BALM2/src/BALM/log";
  FLAGS_alsologtostderr = true;
  // 初始化ROS节点和发布者
  ros::init(argc, argv, "benchmark2");
  ros::NodeHandle n;
  pub_test = n.advertise<sensor_msgs::PointCloud2>("/map_test", 100);
  pub_path = n.advertise<sensor_msgs::PointCloud2>("/map_path", 100);
  pub_show = n.advertise<sensor_msgs::PointCloud2>("/map_show", 100);
  pub_cute = n.advertise<sensor_msgs::PointCloud2>("/map_cute", 100);

  // 定义字符串变量用于存储文件路径前缀
  string prename, ofname;
  // 定义存储IMU和位姿状态的向量
  vector<IMUST> x_buf;
  // 定义存储点云指针的向量
  vector<pcl::PointCloud<PointType>::Ptr> pl_fulls;

  // 从ROS参数服务器获取体素大小参数和获取文件路径参数
  n.param<double>("voxel_size", voxel_size, 1);
  string file_path;
  n.param<string>("file_path", file_path, "");

  // 读取点云文件和位姿数据文件
  read_file(x_buf, pl_fulls, file_path);

  // 初始化位姿状态，设置为缓冲区的第一个元素
  IMUST es0 = x_buf[0];
  // 对所有位姿状态进行调整，使其相对于第一个位姿
  for (uint i = 0; i < x_buf.size(); i++)
  {
    x_buf[i].p = es0.R.transpose() * (x_buf[i].p - es0.p);
    x_buf[i].R = es0.R.transpose() * x_buf[i].R;
  }

  // 获取位姿状态数量，作为窗口的大小
  win_size = x_buf.size();
  printf("The size of poses: %d\n", win_size);
  ros::Duration(1.0).sleep();

  // 调用函数显示初始的点云和位姿
  data_show(x_buf, pl_fulls);
  printf("Check the point cloud with the initial poses.\n");
  printf("If no problem, input '1' to continue or '0' to exit...\n");
  int a;
  cin >> a;
  if (a == 0)
    exit(0);

  // 以下代码块是优化过程的循环，这里设置为只运行一次
  pcl::PointCloud<PointType> pl_full, pl_surf, pl_path, pl_send;
  for (int iterCount = 0; iterCount < 1; iterCount++)
  {
    // 创建用于存储平面点云的地图，键为体素位置，值为八叉树节点指针
    unordered_map<VOXEL_LOC, OCTO_TREE_ROOT *> surf_map;

    // 初始化用于控制优化过程的特征值数组
    eigen_value_array[0] = 1.0 / 16;
    eigen_value_array[1] = 1.0 / 16;
    eigen_value_array[2] = 1.0 / 9;

    // poses = &x_buf;

    // 遍历所有的位姿状态
    for (int i = 0; i < win_size; i++)
    {
      // 处理每个位姿对应的点云数据，主要目的是将输入的点云数据（已经经过旋转和平移变换）分割到具有固定大小的体素中，并更新或创建对应的八叉树结构。
      cut_voxel(surf_map, *pl_fulls[i], x_buf[i], i);
    }
    // 创建用于发送的点云
    pcl::PointCloud<PointType> pl_send;
    // 发布空点云，rviz替代之前的点云
    pub_pl_func(pl_send, pub_show);

    // 创建用于存储点云中心的点云
    pcl::PointCloud<PointType> pl_cent;
    pl_send.clear();
    VOX_HESS voxhess;
    // 遍历所有体素，进行体素细分，保证每个体素要么为平面体素，要么是最小体素
    for (auto iter = surf_map.begin(); iter != surf_map.end() && n.ok(); iter++)
    {
      // 提取平面体素
      iter->second->recut(win_size);
      std::cout << "recut end." << std::endl;
      // 获取所有的平面体素
      iter->second->tras_opt(voxhess, win_size);
      // 展示所有平面体素数据
      iter->second->tras_display(pl_send, win_size);
    }

    // 发布平面体素
    pub_pl_func(pl_send, pub_cute);
    // 输出信息，提示用户检查剪裁后的平面是否足够用于后续优化
    printf("\nThe planes (point association) cut by adaptive voxelization.\n");
    printf("If the planes are too few, the optimization will be degenerated and fail.\n");
    printf("If no problem, input '1' to continue or '0' to exit...\n");
    int a;
    cin >> a;
    if (a == 0)
      exit(0);
    // 清空发送缓冲区的点云数据
    pl_send.clear();
    pub_pl_func(pl_send, pub_cute);

    // 检查优化前的条件，体素的平面数量是否足够
    if (voxhess.plvec_voxels.size() < 3 * x_buf.size())
    {
      printf("Initial error too large.\n");
      printf("Please loose plane determination criteria for more planes.\n");
      printf("The optimization is terminated.\n");
      exit(0);
    }

    // 创建优化对象
    BALM2 opt_lsv;
    // 执行阻尼最小二乘优化
    opt_lsv.damping_iter(x_buf, voxhess);

    // 释放体素地图中的所有资源
    for (auto iter = surf_map.begin(); iter != surf_map.end();)
    {
      delete iter->second;
      surf_map.erase(iter++);
    }
    // 清理完成，清空体素地图
    surf_map.clear();
    // 请求操作系统回收未使用的动态内存
    malloc_trim(0);
  }

  // 显示BA优化后的点云数据
  printf("\nRefined point cloud is publishing...\n");
  malloc_trim(0);
  data_show(x_buf, pl_fulls);
  printf("\nRefined point cloud is published.\n");

  ros::spin();
  return 0;
}
