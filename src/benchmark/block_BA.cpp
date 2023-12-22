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
#include <pcl/filters/uniform_sampling.h>
#include <malloc.h>
#include <unordered_set>

#include <glog/logging.h>

using namespace std;

template <typename T>
void pub_pl_func(T &pl, ros::Publisher &pub)
{
  pl.height = 1;
  pl.width = pl.size();
  sensor_msgs::PointCloud2 output;
  pcl::toROSMsg(pl, output);
  output.header.frame_id = "camera_init";
  output.header.stamp = ros::Time::now();
  std::cout << "pub output cloud size: " << pl.size() << std::endl;

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

void read_file(vector<IMUST> &x_buf, vector<pcl::PointCloud<PointType>::Ptr> &pl_fulls, string &prename)
{
  // prename = prename + "/datas/benchmark_realworld/";

  PLV(3)
  poss;
  PLM(3)
  rots;
  vector<double> tims;
  int pose_size = read_pose(tims, rots, poss, prename);

  for (int m = 0; m < pose_size; m++)
  {
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

    pl_fulls.push_back(pl_ptr);

    IMUST curr;
    curr.R = rots[m];
    curr.p = poss[m];
    // curr.t = tims[m];
    x_buf.push_back(curr);
  }
}

void data_show(vector<IMUST> x_buf, vector<pcl::PointCloud<PointType>::Ptr> &pl_fulls)
{
  IMUST es0 = x_buf[0];
  for (uint i = 0; i < x_buf.size(); i++)
  {
    x_buf[i].p = es0.R.transpose() * (x_buf[i].p - es0.p);
    x_buf[i].R = es0.R.transpose() * x_buf[i].R;
  }

  pcl::PointCloud<PointType> pl_send, pl_path;
  int winsize = x_buf.size();
  for (int i = 0; i < winsize; i++)
  {
    pcl::PointCloud<PointType> pl_tem = *pl_fulls[i];
    down_sampling_voxel(pl_tem, 0.05);
    pl_transform(pl_tem, x_buf[i]);
    pl_send += pl_tem;

    if ((i % 200 == 0 && i != 0) || i == winsize - 1)
    {
      pub_pl_func(pl_send, pub_show);
      pl_send.clear();
      sleep(0.5);
    }

    PointType ap;
    ap.x = x_buf[i].p.x();
    ap.y = x_buf[i].p.y();
    ap.z = x_buf[i].p.z();
    ap.curvature = i;
    pl_path.push_back(ap);
  }

  pub_pl_func(pl_path, pub_path);
}

struct Voxel
{
  int x, y, z;
};

// 计算体素的体积
float calculateVoxelVolume(const pcl::PointXYZ &minPoint, const pcl::PointXYZ &maxPoint)
{
  float length = maxPoint.x - minPoint.x;
  float width = maxPoint.y - minPoint.y;
  float height = maxPoint.z - minPoint.z;

  return length * width * height;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "benchmark2");
  ros::NodeHandle n;
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  FLAGS_log_dir = "/home/gj/catkin_ws_BALM2/src/BALM/log";
  pub_test = n.advertise<sensor_msgs::PointCloud2>("/map_test", 100);
  pub_path = n.advertise<sensor_msgs::PointCloud2>("/map_path", 100);
  pub_show = n.advertise<sensor_msgs::PointCloud2>("/map_show", 100);
  pub_cute = n.advertise<sensor_msgs::PointCloud2>("/map_cute", 100);

  string prename, ofname;
  vector<IMUST> x_buf;
  vector<pcl::PointCloud<PointType>::Ptr> pl_fulls;

  n.param<double>("voxel_size", voxel_size, 1);
  LOG(INFO) << "voxel_size: " << voxel_size;
  string file_path;
  n.param<string>("file_path", file_path, "");

  int max_iter;
  n.param<int>("max_iter", max_iter, 10);
  LOG(INFO) << "max_iter: " << max_iter;

  double initial_block_size; // 初始块大小
  double overlap_size;       // 块之间的重叠为2m
  double z_overlap_size;
  double initial_global_ba_size;
  double initial_z_block_size;
  int min_frame_num;
  n.param<double>("initial_block_size", initial_block_size, 10);
  LOG(INFO) << "initial_block_size: " << initial_block_size;
  n.param<double>("initial_z_block_size", initial_z_block_size, 10);
  LOG(INFO) << "initial_z_block_size: " << initial_z_block_size;

  n.param<double>("overlap_size", overlap_size, 4);
  LOG(INFO) << "overlap_size: " << overlap_size;
    n.param<double>("z_overlap_size", z_overlap_size, 4);
  LOG(INFO) << "z_overlap_size: " << z_overlap_size;
  n.param<double>("initial_global_ba_size", initial_global_ba_size, 300);
  LOG(INFO) << "initial_global_ba_size: " << initial_global_ba_size;
  n.param<int>("min_frame_num", min_frame_num, 150);

  read_file(x_buf, pl_fulls, file_path);

  // 1.遍历所有关键帧，构建最经邻KD树，并得到xyz的范围
  // 初始化xyz的最大值和最小值
  pcl::PointXYZ min_point, max_point;
  min_point.x = min_point.y = min_point.z = std::numeric_limits<float>::max();
  max_point.x = max_point.y = max_point.z = -std::numeric_limits<float>::max();

  pcl::PointCloud<pcl::PointXYZI>::Ptr BACloud(new pcl::PointCloud<pcl::PointXYZI>);
  // 遍历所有雷达点云帧
  for (int i = 0; i < x_buf.size(); ++i)
  {
    // 创建一个XYZI点来表示当前帧
    pcl::PointXYZI frame_point;
    frame_point.x = x_buf[i].p.x();                // 设置x坐标
    frame_point.y = x_buf[i].p.y();                // 设置y坐标
    frame_point.z = x_buf[i].p.z();                // 设置z坐标
    frame_point.intensity = static_cast<float>(i); // 设置关键帧ID和帧数ID

    // 更新xyz的最大值和最小值
    min_point.x = std::min(min_point.x, frame_point.x);
    min_point.y = std::min(min_point.y, frame_point.y);
    min_point.z = std::min(min_point.z, frame_point.z);

    max_point.x = std::max(max_point.x, frame_point.x);
    max_point.y = std::max(max_point.y, frame_point.y);
    max_point.z = std::max(max_point.z, frame_point.z);

    // 将当前帧的点添加到kd树中
    BACloud->push_back(frame_point);
  }
  // 构建kd树，其实分块只需要xyz划分就行，不需要kd树的搜索
  LOG(INFO) << "BA size:" << BACloud->size() << " min-xyz: " << min_point.x << " " << min_point.y << " " << min_point.z
            << " max-xyz: " << max_point.x << " " << max_point.y << " " << max_point.z;

  // 2.给xyz划分block范围并以及BA后可以更新位姿的范围，注释block范围可以重合，更新位姿的范围必须连接且不重合

  double x_range = max_point.x - min_point.x;
  double y_range = max_point.y - min_point.y;
  double z_range = max_point.z - min_point.z;

  // 计算每个轴上的块数量
  int x_blocks = static_cast<int>((x_range - overlap_size) / (initial_block_size - overlap_size)) + 1;
  int y_blocks = static_cast<int>((y_range - overlap_size) / (initial_block_size - overlap_size)) + 1;
  int z_blocks = static_cast<int>((z_range - z_overlap_size) / (initial_z_block_size - z_overlap_size)) + 1;

  LOG(INFO) << "X Block num: " << x_blocks << " Y Block num: " << y_blocks << " Z Block num: " << z_blocks << " sum num:" << x_blocks * y_blocks * z_blocks;

  // 重新计算块大小以确保均匀分割（包括重叠部分）
  double x_block_size = (x_range - overlap_size) / x_blocks + overlap_size;
  double y_block_size = (y_range - overlap_size) / y_blocks + overlap_size;
  double z_block_size = (z_range - z_overlap_size) / z_blocks + z_overlap_size;

  LOG(INFO) << "X-Block Size: " << x_block_size << " Y-Block Size: " << y_block_size << " Z-Block Size: " << z_block_size;

  // 存储块的边界值范围
  // std::map<int, int> block_num_index_map; // 第一个int为xyz对应的键，第二个int为block范围和更新范围的下标
  std::vector<pcl::PointXYZ> block_boundaries_min_v;
  std::vector<pcl::PointXYZ> block_boundaries_max_v;
  std::vector<pcl::PointXYZ> block_update_min_v;
  std::vector<pcl::PointXYZ> block_update_max_v;
  std::vector<pcl::PointXYZ> block_center_min_v;
  std::vector<pcl::PointXYZ> block_center_max_v;
  int count = 0;
  double epsilon = 1e-2; // 数据精度
  for (int x = 0; x < x_blocks; x++)
  {
    for (int y = 0; y < y_blocks; y++)
    {
      for (int z = 0; z < z_blocks; z++)
      {
        // int index = x * 10000 + y * 100 + z;
        // block_num_index_map[index] = count;
        LOG(INFO) << "Block ----" << count;

        count++;
        pcl::PointXYZ block_boundary_min, block_boundary_max;
        pcl::PointXYZ block_update_min, block_update_max;
        pcl::PointXYZ block_center_min, block_center_max;
        block_boundary_min.x = min_point.x + x * (x_block_size - overlap_size);
        block_boundary_min.y = min_point.y + y * (y_block_size - overlap_size);
        block_boundary_min.z = min_point.z + z * (z_block_size - z_overlap_size);
        block_boundary_max.x = block_boundary_min.x + x_block_size;
        block_boundary_max.y = block_boundary_min.y + y_block_size;
        block_boundary_max.z = block_boundary_min.z + z_block_size;
        block_boundaries_min_v.push_back(block_boundary_min);
        block_boundaries_max_v.push_back(block_boundary_max);
        LOG(INFO) << "X Range: (" << block_boundary_min.x << ", " << block_boundary_max.x << ")";
        LOG(INFO) << "Y Range: (" << block_boundary_min.y << ", " << block_boundary_max.y << ")";
        LOG(INFO) << "Z Range: (" << block_boundary_min.z << ", " << block_boundary_max.z << ")";
        block_update_min.x = block_boundary_min.x + overlap_size / 2;
        block_update_max.x = block_boundary_max.x - overlap_size / 2;
        block_update_min.y = block_boundary_min.y + overlap_size / 2;
        block_update_max.y = block_boundary_max.y - overlap_size / 2;
        block_update_min.z = block_boundary_min.z + z_overlap_size / 2;
        block_update_max.z = block_boundary_max.z - z_overlap_size / 2;

        block_center_min.x = block_boundary_min.x + overlap_size;
        block_center_max.x = block_boundary_max.x - overlap_size;
        block_center_min.y = block_boundary_min.y + overlap_size;
        block_center_max.y = block_boundary_max.y - overlap_size;
        block_center_min.z = block_boundary_min.z + z_overlap_size;
        block_center_max.z = block_boundary_max.z - z_overlap_size;
        if (std::abs(block_boundary_min.x - min_point.x) < epsilon)
        {
          block_update_min.x = block_boundary_min.x;
          block_center_min.x = block_boundary_min.x;
        }
        if (std::abs(block_boundary_max.x - max_point.x) < epsilon)
        {
          block_update_max.x = block_boundary_max.x;
          block_center_max.x = block_boundary_max.x;
        }
        if (std::abs(block_boundary_min.y - min_point.y) < epsilon)
        {
          block_update_min.y = block_boundary_min.y;
          block_center_min.y = block_boundary_min.y;
        }
        if (std::abs(block_boundary_max.y - max_point.y) < epsilon)
        {
          block_update_max.y = block_boundary_max.y;
          block_center_max.y = block_boundary_max.y;
        }
        if (std::abs(block_boundary_min.z - min_point.z) < epsilon)
        {
          block_update_min.z = block_boundary_min.z;
          block_center_min.z = block_boundary_min.z;
        }
        if (std::abs(block_boundary_max.z - max_point.z) < epsilon)
        {
          block_update_max.z = block_boundary_max.z;
          block_center_max.z = block_boundary_max.z;
        }
        block_update_min_v.push_back(block_update_min);
        block_update_max_v.push_back(block_update_max);
        block_center_min_v.push_back(block_center_min);
        block_center_max_v.push_back(block_center_max);
        LOG(INFO) << "X Update Range: (" << block_update_min.x << ", " << block_update_max.x << ")";
        LOG(INFO) << "Y Update Range: (" << block_update_min.y << ", " << block_update_max.y << ")";
        LOG(INFO) << "Z Update Range: (" << block_update_min.z << ", " << block_update_max.z << ")";
        LOG(INFO) << "-------------------\n";
      }
    }
  }

  // 3.对每个block进行BA，并更新位姿
  // 遍历每个block，找到对应的帧

  std::vector<std::vector<int>> block_frame_id; // 二维数组，对应每个block的帧数
  block_frame_id.resize(block_boundaries_min_v.size());
  std::vector<int> block_num_v;
  block_num_v.resize(block_boundaries_min_v.size());
  int frame_count2 = 0;
  for (int i = 0; i < block_boundaries_min_v.size(); ++i)
  {
    // 遍历BACloud，找到当前block范围的所有帧
    std::vector<int> cur_block_frame_id;
    // bool isFindBlock=false;
    for (int j = 0; j < BACloud->size(); ++j)
    {
      if (BACloud->points[j].x >= block_boundaries_min_v[i].x && BACloud->points[j].x <= block_boundaries_max_v[i].x && BACloud->points[j].y >= block_boundaries_min_v[i].y && BACloud->points[j].y <= block_boundaries_max_v[i].y && BACloud->points[j].z >= block_boundaries_min_v[i].z && BACloud->points[j].z <= block_boundaries_max_v[i].z)
      {
        int frameNum = static_cast<int>(BACloud->points[j].intensity);
        cur_block_frame_id.push_back(frameNum);
      }
    }
    block_frame_id[i] = cur_block_frame_id;
    block_num_v[i] = cur_block_frame_id.size();
    frame_count2 += cur_block_frame_id.size();
    LOG(INFO) << "Block-" << i << " has frame num: " << cur_block_frame_id.size();
  }

  // 对block_num_v进行排序，并且记录排序前的索引
  std::vector<std::pair<int, int>> indexed_values;
  for (int i = 0; i < block_num_v.size(); ++i)
  {
    indexed_values.push_back({block_num_v[i], i});
  }
  std::sort(indexed_values.begin(), indexed_values.end(), [](const std::pair<int, int> &a, const std::pair<int, int> &b)
            { return (a.first == b.first) ? (a.second < b.second) : (a.first < b.first); });

  std::vector<int> volid_block; // 存储有效的block的数组下标。

  int frame_count = 0;
  for (const auto &pair : indexed_values)
  {
    int index = pair.second; //对应的block在数组中的id
    int frame_num = block_num_v[pair.second];
    if (frame_num == 0)
    {
      continue;
    }
    if (frame_num < min_frame_num)
    {
      LOG(INFO) << "----------------";

      // 搜索领域的帧数
      int z = index % z_blocks;
      int y = (index / z_blocks) % y_blocks;
      int x = index / (y_blocks * z_blocks);
      LOG(INFO) << "Block id-" << index << " pointcloud num is not enough: " << frame_num << " x: " << x << " y: " << y << " z: " << z;
      Voxel current_voxel = {x, y, z};
      Voxel neighbors[6] = {{current_voxel.x + 1, current_voxel.y, current_voxel.z},
                            {current_voxel.x - 1, current_voxel.y, current_voxel.z},
                            {current_voxel.x, current_voxel.y + 1, current_voxel.z},
                            {current_voxel.x, current_voxel.y - 1, current_voxel.z},
                            {current_voxel.x, current_voxel.y, current_voxel.z + 1},
                            {current_voxel.x, current_voxel.y, current_voxel.z - 1}};
      int neighbor_max_frame_num = -std::numeric_limits<int>::max();
      LOG(INFO) << "neighbor_max_frame_num: " << neighbor_max_frame_num;
      int neighbor_max_index = -1;
      int count = 0;
      int which_neighbor = -1;
      for (const auto &neighbor : neighbors)
      {
        int neighbor_index = neighbor.z + neighbor.y * z_blocks + neighbor.x * y_blocks * z_blocks;

        if (neighbor_index < 0 || neighbor.x < 0 || neighbor.y < 0 || neighbor.z < 0 || neighbor.x >= x_blocks || neighbor.y >= y_blocks || neighbor.z >= z_blocks)
        {
          LOG(INFO) << "neighbor index: " << neighbor_index << " x: " << neighbor.x << " y: " << neighbor.y << " z: " << neighbor.z << " is unavaiable.";
          ++count;
          continue;
        }
        int cur_frame_num = block_frame_id[neighbor_index].size();
        LOG(INFO) << "Search neighbor index: " << neighbor_index << " x: " << neighbor.x << " y: " << neighbor.y << " z: " << neighbor.z << " frame num: " << cur_frame_num;
        if (neighbor_max_frame_num < cur_frame_num)
        {
          neighbor_max_frame_num = cur_frame_num;
          LOG(INFO) << "Update neighbor_max_frame_num: " << neighbor_max_frame_num;
          neighbor_max_index = neighbor_index;
          which_neighbor = count;
        }
        ++count;
      }
      if (neighbor_max_index == -1 || which_neighbor == -1)
      {
        LOG(INFO) << "ERROR: impossible has not neighbor !!!!!";
        exit(0);
      }
      block_frame_id[neighbor_max_index].insert(block_frame_id[neighbor_max_index].end(), block_frame_id[index].begin(), block_frame_id[index].end());
      block_frame_id[index].clear();
      // 同一个block删除相同的元素
      std::unordered_set<int> seen;
      auto end = std::remove_if(block_frame_id[neighbor_max_index].begin(), block_frame_id[neighbor_max_index].end(), [&seen](int num)
                                { return !seen.insert(num).second; });
      block_frame_id[neighbor_max_index].erase(end, block_frame_id[neighbor_max_index].end());
      LOG(INFO) << "Update block_frame_id[" << neighbor_max_index << "] to " << block_frame_id[neighbor_max_index].size() << " block_frame_id[" << index << "] to " << block_frame_id[index].size();
      // 更新block的更新范围
      if (which_neighbor == 0)
      {
        LOG(INFO) << "Update min-x from " << block_update_min_v[neighbor_max_index].x << "to: " << block_update_min_v[index].x;
        block_update_min_v[neighbor_max_index].x = block_update_min_v[index].x;
        block_center_min_v[neighbor_max_index].x = block_center_min_v[index].x;
      }
      else if (which_neighbor == 1)
      {
        LOG(INFO) << "Update max-x from " << block_update_max_v[neighbor_max_index].x << "to: " << block_update_max_v[index].x;
        block_update_max_v[neighbor_max_index].x = block_update_max_v[index].x;
        block_center_max_v[neighbor_max_index].x = block_center_max_v[index].x;
      }
      else if (which_neighbor == 2)
      {
        LOG(INFO) << "Update min-y from " << block_update_min_v[neighbor_max_index].y << "to: " << block_update_min_v[index].y;
        block_update_min_v[neighbor_max_index].y = block_update_min_v[index].y;
        block_center_min_v[neighbor_max_index].y = block_center_min_v[index].y;
      }
      else if (which_neighbor == 3)
      {
        LOG(INFO) << "Update max-y from " << block_update_max_v[neighbor_max_index].y << "to: " << block_update_max_v[index].y;
        block_update_max_v[neighbor_max_index].y = block_update_max_v[index].y;
        block_center_max_v[neighbor_max_index].y = block_center_max_v[index].y;
      }
      else if (which_neighbor == 4)
      {
        LOG(INFO) << "Update min-z from " << block_update_min_v[neighbor_max_index].z << "to: " << block_update_min_v[index].z;
        block_update_min_v[neighbor_max_index].z = block_update_min_v[index].z;
        block_center_min_v[neighbor_max_index].z = block_center_min_v[index].z;
      }
      else if (which_neighbor == 5)
      {
        LOG(INFO) << "Update min-z from " << block_update_max_v[neighbor_max_index].z << "to: " << block_update_max_v[index].z;
        block_update_max_v[neighbor_max_index].z = block_update_max_v[index].z;
        block_center_max_v[neighbor_max_index].z = block_center_max_v[index].z;
      }
      else
      {
        LOG(INFO) << "ERROR: which_neighbor if impossible !!!!!";
        exit(0);
      }
    }
  }
  // 排序
  for (auto &row : block_frame_id)
  {
    if (!row.empty())
    {
      std::sort(row.begin(), row.end());
    }
  }

  // 计算block中总共有多少帧,计算总共有多少个block
  std::vector<bool> isCheck(BACloud->size(), false);
  int block_num = 0;
  int frame_count3 = 0;
  for (int i = 0; i < block_frame_id.size(); ++i)
  {
    auto row = block_frame_id[i];
    if (row.size() != 0)
    {
      block_num++;
      LOG(INFO) << "DEBUG block id: " << i << " has frame num: " << row.size();
    }
    for (auto &it : row)
    {
      // int BACloudId = frameNum2BACloudId[it];
      if (!isCheck[it])
      {
        frame_count3++;
        isCheck[it] = true;
      }
    }
  }
  LOG(INFO) << "DEBUG sum frame number: " << frame_count3 << "--" << BACloud->size();
  LOG(INFO) << "DEBUG sum block number: " << block_num;

  for (int i = 0; i < isCheck.size(); ++i)
  {
    if (!isCheck[i])
    {
      LOG(INFO) << "DEBUG frame-" << i << " is false";
    }
  }

  // std::vector<int> global_num_vec;
  // int global_num_x = x_range / initial_global_ba_size;
  // global_num_vec.push_back(global_num_x);
  // int global_num_y = y_range / initial_global_ba_size;
  // global_num_vec.push_back(global_num_y);
  // int global_num_z = z_range / initial_global_ba_size;
  // global_num_vec.push_back(global_num_z);
  // int global_block_num = (global_num_x + 1) * (global_num_y + 1) * (global_num_z + 1);
  // LOG(INFO) << "global_block_num: " << global_block_num;
  // // xyz的方向全局BA的最终尺寸
  // std::vector<double> global_ba_size_vec;
  // double global_ba_size_x = x_range / (global_num_x + 1);
  // global_ba_size_vec.push_back(global_ba_size_x);
  // double global_ba_size_y = y_range / (global_num_y + 1);
  // global_ba_size_vec.push_back(global_ba_size_y);
  // double global_ba_size_z = z_range / (global_num_z + 1);
  // global_ba_size_vec.push_back(global_ba_size_z);

  // // 知道xyz划分的block id
  // std::vector<std::vector<int>> divide_block_x;
  // divide_block_x.resize(global_num_x);
  // std::vector<std::vector<int>> divide_block_y;
  // divide_block_y.resize(global_num_y);
  // std::vector<std::vector<int>> divide_block_z;
  // divide_block_z.resize(global_num_z);

  // std::vector<std::vector<int>> global_divide_block;
  // global_divide_block.resize(global_block_num);
  // for (int gx = 0; gx <= global_num_x; ++gx)
  // {
  //   double g_min_x = min_point.x + gx * global_ba_size_x;
  //   double g_max_x = g_min_x + global_ba_size_x;
  //   for (int gy = 0; gy <= global_num_y; ++gy)
  //   {
  //     double g_min_y = min_point.y + gy * global_ba_size_y;
  //     double g_max_y = g_min_y + global_ba_size_y;
  //     for (int gz = 0; gz <= global_num_z; ++gz)
  //     {
  //       double g_min_z = min_point.z + gz * global_ba_size_z;
  //       double g_max_z = g_min_z + global_ba_size_z;

  //       // 找到边界内部的block和边界处的block，注意在xyz范围边缘的block为边界内部的blcok
  //     }
  //   }
  // }

  // 再排序
  std::vector<size_t> indices(block_frame_id.size());
  std::iota(indices.begin(), indices.end(), 0); // 使用 iota 填充索引数组 [0, 1, 2, ...]
  std::sort(indices.begin(), indices.end(), [&block_frame_id](size_t a, size_t b)
            {
              const auto &a_row = block_frame_id[a];
              const auto &b_row = block_frame_id[b];

              if (a_row.empty() && b_row.empty())
              {
                return false; // 空行相等
              }
              else if (a_row.empty())
              {
                return true; // 空行排在非空行之前
              }
              else if (b_row.empty())
              {
                return false; // 空行排在非空行之后
              }

              // 比较第一个元素
              if (a_row[0] != b_row[0])
              {
                return a_row[0] < b_row[0];
              }

              // 如果第一个元素相同，则比较后面的数
              for (size_t i = 1; i < std::min(a_row.size(), b_row.size()); ++i)
              {
                if (a_row[i] != b_row[i])
                {
                  return a_row[i] < b_row[i];
                }
              }

              // 如果前面的元素都相同，则长度较短的数组排在前面
              return a_row.size() < b_row.size();
            });

  // 对所有的块进行BA
  std::vector<bool> isUpdate(BACloud->size(), false);
  int BAcount = 0;
  double ep;
  vector<IMUST> x_buf_copy;
  for (int id = 0; id < indices.size(); ++id)
  {
    int i = indices[id]; //i
    if (block_frame_id[i].size() == 0)
    {
      continue;
    }
    if (block_frame_id[i].size() < min_frame_num)
    {
      LOG(INFO) << "ERROR: block-" << i << " has not 120 frame !!!!!";
      exit(0);
    }
    LOG(INFO) << "-------------Local BA-" << BAcount << " frame size: " << block_frame_id[i].size() << " block id: " << i;
    vector<IMUST> sub_buf;
    vector<pcl::PointCloud<PointType>::Ptr> sub_pl_fulls;
    for (int j = 0; j < block_frame_id[i].size(); ++j)
    {
      int frameNum = static_cast<int>(block_frame_id[i][j]);
      sub_buf.push_back(x_buf[frameNum]);
      sub_pl_fulls.push_back(pl_fulls[frameNum]);
    }

    vector<IMUST> sub_buf_before = sub_buf;

    if (sub_buf.size() == 0)
    {
      LOG(INFO) << "Local BA don't have cloud-" << i;
      continue;
    }
    win_size = sub_buf.size();
    // 保存初始点云
    pcl::PointCloud<PointType> pc_BA_before;
    for (int l = 0; l < win_size; ++l)
    {
      pcl::PointCloud<PointType> pl_tem = *sub_pl_fulls[l];
      down_sampling_voxel(pl_tem, 0.05);
      pl_transform(pl_tem, sub_buf[l]);
      pc_BA_before += pl_tem;
    }
    pcl::io::savePCDFileBinary("/home/hyshan/dev_sda3/data/BALM/ba_output/pc_BA_before_" + to_string(BAcount) + ".pcd", pc_BA_before);
    pc_BA_before.clear();

    IMUST es0 = sub_buf[0];
    // 多分辨率
    {
      std::vector<double> voxels_size = {4.0, 2.0};
      for (int voxel_id = 0; voxel_id < voxels_size.size(); ++voxel_id)
      {

        // printf("cut_voxel function before.\n");
        // printf("If no problem, input '1' to continue or '0' to exit...\n");
        // int a;
        // cin >> a;
        // if (a == 0)
        //   exit(0);

        voxel_size = voxels_size[voxel_id];

        for (uint i = 0; i < sub_buf.size(); i++)
        {
          sub_buf[i].p = es0.R.transpose() * (sub_buf[i].p - es0.p);
          sub_buf[i].R = es0.R.transpose() * sub_buf[i].R;
        }
        unordered_map<VOXEL_LOC, OCTO_TREE_ROOT *> surf_map;

        eigen_value_array[0] = 1.0 / 16;
        eigen_value_array[1] = 1.0 / 16;
        eigen_value_array[2] = 1.0 / 9;

        // poses = &sub_buf;

        for (int i = 0; i < win_size; i++)
          cut_voxel(surf_map, *sub_pl_fulls[i], sub_buf[i], i);

        // printf("cut_voxel function end.\n");
        // printf("If no problem, input '1' to continue or '0' to exit...\n");
        // cin >> a;
        // if (a == 0)
        //   exit(0);

        VOX_HESS voxhess;
        for (auto iter = surf_map.begin(); iter != surf_map.end() && n.ok(); iter++)
        {
          iter->second->recut(win_size);
          iter->second->tras_opt(voxhess, win_size);
        }

        // printf("recut function end.\n");
        // printf("If no problem, input '1' to continue or '0' to exit...\n");
        // cin >> a;
        // if (a == 0)
        //   exit(0);

        BALM2 opt_lsv;
        ep = 1e-12;
        opt_lsv.damping_iter(sub_buf, voxhess, max_iter, ep);

        for (auto iter = surf_map.begin(); iter != surf_map.end();)
        {
          delete iter->second;
          surf_map.erase(iter++);
        }
        surf_map.clear();

        malloc_trim(0);

        for (uint i = 0; i < sub_buf.size(); i++)
        {
          sub_buf[i].p = es0.R * sub_buf[i].p + es0.p;
          sub_buf[i].R = es0.R * sub_buf[i].R;
        }
        // printf("damping_iter function end.\n");
        // printf("If no problem, input '1' to continue or '0' to exit...\n");
        // cin >> a;
        // if (a == 0)
        //   exit(0);
      }
    }

    malloc_trim(0);
    // 给x_buf的初始值
    IMUST T1 = sub_buf_before.back();
    IMUST T2 = sub_buf.back();
    Eigen::Matrix3d R = T1.R * T2.R.inverse();
    Eigen::Vector3d p = -T1.R * T2.R.inverse() * T2.p + T1.p;
    IMUST T;
    T.R = R;
    T.p = p;
    x_buf_copy.push_back(T);

    // 将BA优化后的点云保存，
    pcl::PointCloud<PointType> pc_BA_after;

    for (int m = 0; m < win_size; ++m)
    {
      pcl::PointCloud<PointType> pl_tem = *sub_pl_fulls[m];
      // down_sampling_voxel(pl_tem, 0.05);
      pcl::UniformSampling<PointType> uniform_sampling;
      int curFrameNum = block_frame_id[i][m];
      pcl::PointXYZI cur_point = BACloud->points[curFrameNum];
      if (cur_point.x >= block_center_min_v[i].x && cur_point.x <= block_center_max_v[i].x && cur_point.y >= block_center_min_v[i].y && cur_point.y <= block_center_max_v[i].y && cur_point.z >= block_center_min_v[i].z && cur_point.z <= block_center_max_v[i].z)
      {
        continue;
      }
      uniform_sampling.setInputCloud(pl_tem.makeShared());
      uniform_sampling.setRadiusSearch(0.1);
      pcl::PointCloud<PointType> cloud_out;
      uniform_sampling.filter(cloud_out);
      pl_transform(cloud_out, sub_buf[m]);
      pc_BA_after += cloud_out;
    }

    pcl::io::savePCDFileBinary("/home/hyshan/dev_sda3/data/BALM/ba_output/pc_BA_after_" + std::to_string(BAcount) + ".pcd", pc_BA_after);
    pc_BA_after.clear();
    BAcount++;
    // sub_buf.clear();
    // sub_pl_fulls.clear();
  }
  x_buf.clear();
  pl_fulls.clear();
  malloc_trim(0);

  // 优化BA后的点云
  // read_file(x_buf, pl_fulls, file_path);
  LOG(INFO) << "BAcount: " << BAcount;
  for (int i = 0; i < BAcount; ++i)
  {

    std::string filename = "/home/hyshan/dev_sda3/data/BALM/ba_output/pc_BA_after_" + std::to_string(i) + ".pcd";
    pcl::PointCloud<PointType>::Ptr cur_pc(new pcl::PointCloud<PointType>());

    if (pcl::io::loadPCDFile<PointType>(filename, *cur_pc) == -1)
    {
      LOG(ERROR) << "Couldn't read file: " << filename;
      return -1;
    }

    pl_fulls.push_back(cur_pc);

    // IMUST cur_pose;
    // cur_pose.R = Eigen::Matrix3d::Identity();
    // cur_pose.p = Eigen::Vector3d(0.0, 0.0, 0.0);
    // x_buf.push_back(cur_pose);
  }
  x_buf = x_buf_copy;

  // 保存初始点云
  win_size = x_buf.size();
  pcl::PointCloud<PointType> pc_BA_before;
  for (int l = 0; l < win_size; ++l)
  {
    pcl::PointCloud<PointType> pl_tem = *pl_fulls[l];
    down_sampling_voxel(pl_tem, 0.2);
    pl_transform(pl_tem, x_buf[l]);
    pc_BA_before += pl_tem;
  }
  pcl::io::savePCDFileBinary("/home/hyshan/dev_sda3/data/BALM/ba_output/map_BA_before.pcd", pc_BA_before);
  pc_BA_before.clear();
  // 对完整地图进行BA

  std::vector<double> voxels_size = {3.0, 2.0};
  min_ps = 50;
  for (int voxel_id = 0; voxel_id < voxels_size.size(); ++voxel_id)
  {
    voxel_size = voxels_size[voxel_id];
    IMUST es0 = x_buf[0];
    for (uint i = 0; i < x_buf.size(); i++)
    {
      x_buf[i].p = es0.R.transpose() * (x_buf[i].p - es0.p);
      x_buf[i].R = es0.R.transpose() * x_buf[i].R;
    }
    unordered_map<VOXEL_LOC, OCTO_TREE_ROOT *> surf_map;

    eigen_value_array[0] = 1.0 / 16;
    eigen_value_array[1] = 1.0 / 16;
    eigen_value_array[2] = 1.0 / 9;

    // poses = &x_buf;

    for (int i = 0; i < win_size; i++)
      cut_voxel(surf_map, *pl_fulls[i], x_buf[i], i);

    pcl::PointCloud<PointType> pl_send;

    pcl::PointCloud<PointType> pl_cent;
    pl_send.clear();
    VOX_HESS voxhess;
    for (auto iter = surf_map.begin(); iter != surf_map.end() && n.ok(); iter++)
    {
      iter->second->recut(win_size);
      iter->second->tras_opt(voxhess, win_size);
      // iter->second->tras_display(pl_send, win_size);
    }
    BALM2 opt_lsv;
    ep = 1e-12;
    opt_lsv.damping_iter(x_buf, voxhess, max_iter, ep);

    for (auto iter = surf_map.begin(); iter != surf_map.end();)
    {
      delete iter->second;
      surf_map.erase(iter++);
    }
    surf_map.clear();

    malloc_trim(0);

    // for (uint i = 0; i < x_buf.size(); i++)
    // {
    //   x_buf[i].p = es0.R * x_buf[i].p + es0.p;
    //   x_buf[i].R = es0.R * x_buf[i].R;
    // }
    es0 = x_buf[0];
    for (uint i = 0; i < x_buf.size(); i++)
    {
      x_buf[i].p = es0.R.transpose() * (x_buf[i].p - es0.p);
      x_buf[i].R = es0.R.transpose() * x_buf[i].R;
    }

    // 将BA优化后的点云保存，
    pcl::PointCloud<PointType> pc_BA_after;
    for (int m = 0; m < win_size; ++m)
    {
      pcl::PointCloud<PointType> pl_tem = *pl_fulls[m];
      // down_sampling_voxel(pl_tem, 0.05);
      pcl::UniformSampling<PointType> uniform_sampling;
      uniform_sampling.setInputCloud(pl_tem.makeShared());
      uniform_sampling.setRadiusSearch(0.2);
      pcl::PointCloud<PointType> cloud_out;
      uniform_sampling.filter(cloud_out);
      pl_transform(cloud_out, x_buf[m]);
      pc_BA_after += cloud_out;
    }

    pcl::io::savePCDFileBinary("/home/hyshan/dev_sda3/data/BALM/ba_output/map_BA_after_opt" + to_string(voxel_id) + ".pcd", pc_BA_after);
    pc_BA_after.clear();
  }

  // ros::spin();
  return 0;
}
