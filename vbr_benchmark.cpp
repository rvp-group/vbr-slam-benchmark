#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <limits>
#include <algorithm>
#include <Eigen/Dense>

// some const
const float MAX_ERROR = 10;
const std::vector<float> PERCENTAGES = {0.01, 0.02, 0.03, 0.05, 0.08, 0.13, 0.21, 0.34, 0.55};
const std::vector<std::string> SEQ_TRAIN_NAMES = {
      "colosseo_train0",
      "campus_train0",
      "campus_train1",
      "pincio_train0",
      "spagna_train0",
      "diag_train0",
      "ciampino_train0",
      "ciampino_train1",
      };
const std::vector<std::string> SEQ_TEST_NAMES = {
      "colosseo_test0",
      "campus_test0",
      "campus_test1",
      "pincio_test0",
      "spagna_test0",
      "diag_test0",
      "ciampino_test0",
      "ciampino_test1",
      };

struct Error
{
  int first_frame;
  double r_err, t_err;
  float len;
  Error(int first_frame, double r_err, double t_err, float len) : first_frame(first_frame), r_err(r_err), t_err(t_err), len(len) {}
};

struct Stats
{
  std::string sequence_name_;
  double r_err, t_err;
  Stats(std::string sequence_name, double r_err, double t_err) : sequence_name_(sequence_name), r_err(r_err), t_err(t_err) {}
};

struct Pose
{
  double timestamp;
  Eigen::Isometry3f transform;
  Pose(double timestamp, Eigen::Isometry3f transform) : timestamp(timestamp), transform(transform) {}
};

inline bool sortComparator(const Stats &stat_l, const Stats &stat_r)
{
  return stat_l.t_err < stat_r.t_err;
}

inline std::vector<Pose> loadPoses(const std::string &file_name)
{
  std::vector<Pose> poses;
  std::ifstream file(file_name);
  if (!file.is_open())
  {
    std::cout << "error: unable to open file " << file_name << std::endl;
    return poses;
  }

  while (!file.eof())
  {
    std::string line;
    std::getline(file, line);

    if (line.empty() || line[0] == '#')
      continue;

    std::istringstream iss(line);
    float t, x, y, z, qx, qy, qz, qw;
    if (!(iss >> t >> x >> y >> z >> qx >> qy >> qz >> qw))
    {
      std::cerr << "error reading line from file: " << line << std::endl;
      continue; // Skip this line if unable to read values
    }

    Eigen::Isometry3f P = Eigen::Isometry3f::Identity();
    // float t, x, y, z, qx, qy, qz, qw;
    // file >> t >> x >> y >> z >> qx >> qy >> qz >> qw;

    P.translation() << x, y, z;

    const Eigen::Quaternionf q(qw, qx, qy, qz);
    P.linear() = q.toRotationMatrix();

    poses.push_back(Pose(t, P));
  }

  file.close();
  return poses;
}

inline std::vector<Pose> matchTimestamps(const std::vector<Pose> &poses_gt, const std::vector<Pose> &poses_es)
{
  std::vector<Pose> poses_matched;
  poses_matched.reserve(poses_gt.size());

  size_t es_index = 0;
  for (const auto &pose_gt : poses_gt)
  {
    // while (es_index < poses_es.size() -  && poses_es[es_index].timestamp < pose_gt.timestamp)
    //   es_index++;

    double min_delta = fabs(poses_es[es_index].timestamp - pose_gt.timestamp);
    while (es_index < poses_es.size() - 1 && fabs(poses_es[es_index + 1].timestamp - pose_gt.timestamp) <= min_delta)
    {
      min_delta = fabs(poses_es[es_index + 1].timestamp - pose_gt.timestamp);
      es_index++;
    }

    poses_matched.push_back(poses_es[es_index]);
  }

  return poses_matched;
}

inline std::vector<float> trajectoryDistances(const std::vector<Pose> &poses)
{
  std::vector<float> dist;
  dist.push_back(0);
  for (size_t i = 1; i < poses.size(); ++i)
  {
    const Eigen::Vector3f t1 = poses[i - 1].transform.translation();
    const Eigen::Vector3f t2 = poses[i].transform.translation();

    dist.push_back(dist[i - 1] + (t1 - t2).norm());
  }
  return dist;
}

inline size_t lastFrameFromSegmentLength(const std::vector<float> &dist, const int &first_frame, const float &len)
{
  for (size_t i = first_frame; i < dist.size(); ++i)
    if (dist[i] > dist[first_frame] + len)
      return i;
  return -1;
}

inline double rotationError(const Eigen::Isometry3f &pose_error)
{
  Eigen::Quaternionf q(pose_error.linear());
  q.normalize();

  const Eigen::Quaternionf q_identity(1.0f, 0.0f, 0.0f, 0.0f);
  const double error_radians = q_identity.angularDistance(q);

  const double error_degrees = error_radians * (180.0f / M_PI);
  return error_degrees;
}

inline double translationError(const Eigen::Isometry3f &pose_error)
{
  const Eigen::Vector3f t = pose_error.translation();
  return t.norm();
}

inline std::vector<Error> computeSequenceErrors(const std::vector<Pose> poses_gt, const std::vector<Pose> &poses_es)
{
  std::vector<Error> err;

  const std::vector<float> dist = trajectoryDistances(poses_gt);
  const float seq_length = dist.back();
  std::cout << "sequence length [m]: " << seq_length << std::endl;

  std::vector<float> lengths;
  for (const float &percentage : PERCENTAGES)
  {
    const float len = seq_length * percentage;
    lengths.push_back(len);
    std::cout << "percentage: " << percentage << ", subsequence length [m]: " << len << std::endl;
  }
  std::cout << std::endl;

  for (size_t first_frame = 0; first_frame < poses_gt.size(); ++first_frame)
  {
    for (size_t i = 0; i < lengths.size(); ++i)
    {

      const float curr_len = lengths[i];
      const int last_frame = lastFrameFromSegmentLength(dist, first_frame, curr_len);

      if (last_frame == -1)
        continue;

      const Eigen::Isometry3f pose_delta_gt = poses_gt[first_frame].transform.inverse() * poses_gt[last_frame].transform;
      const Eigen::Isometry3f pose_delta_es = poses_es[first_frame].transform.inverse() * poses_es[last_frame].transform;
      const Eigen::Isometry3f pose_error = pose_delta_es.inverse() * pose_delta_gt;
      const double r_err = rotationError(pose_error);
      const double t_err = translationError(pose_error);

      err.push_back(Error(first_frame, r_err / curr_len, t_err / curr_len, curr_len));
    }
  }

  return err;
}

inline std::vector<Pose> computeAlignedEstimate(const std::vector<Pose> &poses_gt, const std::vector<Pose> &poses_es)
{
  std::vector<Pose> poses_es_aligned;
  poses_es_aligned.reserve(poses_es.size());

  Eigen::Matrix<float, 3, Eigen::Dynamic> gt_matrix;
  gt_matrix.resize(Eigen::NoChange, poses_gt.size());
  for (size_t i = 0; i < poses_gt.size(); ++i)
    gt_matrix.col(i) = poses_gt[i].transform.translation();

  Eigen::Matrix<float, 3, Eigen::Dynamic> es_matrix;
  es_matrix.resize(Eigen::NoChange, poses_es.size());
  for (size_t i = 0; i < poses_es.size(); ++i)
    es_matrix.col(i) = poses_es[i].transform.translation();

  // last argument to true for monocular (Sim3)
  const Eigen::Matrix4f transform_matrix = Eigen::umeyama(es_matrix, gt_matrix, false);
  Eigen::Isometry3f transform = Eigen::Isometry3f(transform_matrix.block<3, 3>(0, 0));
  transform.translation() = transform_matrix.block<3, 1>(0, 3);

  for (size_t i = 0; i < poses_es.size(); ++i)
    poses_es_aligned.push_back(Pose(poses_es[i].timestamp, transform * poses_es[i].transform));

  return poses_es_aligned;
}

inline Stats computeSequenceRPE(const std::vector<Error> &seq_err, const std::string &sequence_name)
{
  double t_err = 0;
  double r_err = 0;

  for (const Error &error : seq_err)
  {
    t_err += error.t_err;
    r_err += error.r_err;
  }

  const double r_rpe = r_err / double(seq_err.size());
  const double t_rpe = 100 * t_err / double(seq_err.size());
  return Stats(sequence_name, r_rpe, t_rpe);
}

inline Stats computeSequenceATE(const std::vector<Pose> &poses_gt, const std::vector<Pose> &poses_es_aligned, const std::string &sequence_name)
{
  double r_sum = 0;
  double t_sum = 0;

  for (size_t i = 0; i < poses_gt.size(); ++i)
  {
    const Eigen::Isometry3f pose_error = poses_gt[i].transform.inverse() * poses_es_aligned[i].transform;
    const double r_err = rotationError(pose_error);
    const double t_err = translationError(pose_error);

    r_sum += r_err;
    t_sum += t_err;
  }

  const double r_ate_rmse = std::sqrt(r_sum / double(poses_gt.size()));
  const double t_ate_rmse = std::sqrt(t_sum / double(poses_gt.size()));
  return Stats(sequence_name, r_ate_rmse, t_ate_rmse);
}

inline void computeRank(std::vector<Stats> &stats, const std::string &path_to_result_file, const std::string &path_to_rank_file)
{
  // sort(stats.begin(), stats.end(), sortComparator);

  FILE *fp = fopen(path_to_result_file.c_str(), "w");

  double rank = 0;
  for (const Stats stat : stats)
  {
    fprintf(fp, "%f %f\n", stat.t_err, stat.r_err);
    std::cout << stat.sequence_name_ << " " << stat.t_err << " " << stat.r_err;

    if (stat.t_err > MAX_ERROR)
    {
      std::cout << " - exceeded max error" << std::endl;
      continue;
    }

    rank += (double)(MAX_ERROR - stat.t_err);
    std::cout << std::endl;
  }

  fclose(fp);

  fp = fopen(path_to_rank_file.c_str(), "w");
  fprintf(fp, "%f\n", rank);
  fclose(fp);

  std::cout << "rank: " << rank << std::endl
            << std::endl;
}

inline void eval(const std::string &path_to_gt, const std::string &path_to_es, const std::string &eval_type)
{
  const std::string path_to_result = path_to_es + "/results/" + eval_type;
  system(("mkdir -p " + path_to_result).c_str());

  std::vector<std::string> seq_names;
  if (eval_type == "train")
    seq_names = SEQ_TRAIN_NAMES;
  else if (eval_type == "test")
    seq_names = SEQ_TEST_NAMES;
  else
    return;

  std::vector<Stats> rpe_stats;
  std::vector<Stats> ate_stats;
  for (size_t i = 0; i < seq_names.size(); ++i)
  {
    const std::string sequence_name = seq_names[i];
    const std::string path_to_gt_file = path_to_gt + "/" + sequence_name + "_gt.txt";
    const std::string path_to_es_file = path_to_es + "/" + sequence_name + "_es.txt";

    const std::vector<Pose> poses_gt = loadPoses(path_to_gt_file);
    const std::vector<Pose> poses_es_unmatched = loadPoses(path_to_es_file);

    std::cout << "=============================================" << std::endl;
    std::cout << "processing: " << sequence_name << std::endl;
    std::cout << "estimated poses: " << poses_es_unmatched.size() << std::endl;
    std::cout << "gt poses: " << poses_gt.size() << std::endl;

    if (poses_gt.size() == 0 || poses_es_unmatched.size() == 0)
    {
      std::cout << "ERROR: could not read (all) poses of: " << sequence_name << std::endl;
      continue;
    }

    const std::vector<Pose> poses_es = matchTimestamps(poses_gt, poses_es_unmatched);
    std::cout << "matched poses: " << poses_es.size() << std::endl
              << std::endl;

    const std::vector<Error> seq_err = computeSequenceErrors(poses_gt, poses_es);
    const Stats rpe_stat = computeSequenceRPE(seq_err, sequence_name);
    rpe_stats.push_back(rpe_stat);

    const std::vector<Pose> poses_es_aligned = computeAlignedEstimate(poses_gt, poses_es);
    const Stats ate_stat = computeSequenceATE(poses_gt, poses_es_aligned, sequence_name);
    ate_stats.push_back(ate_stat);
  }

  std::cout << eval_type << " stats RPE (sequence, t_err [%], r_err [deg/m]):" << std::endl;
  computeRank(rpe_stats, path_to_result + "/results_rpe.txt", path_to_result + "/rank_rpe.txt");

  std::cout << eval_type << " stats ATE RMSE (sequence, t_err [m], r_err [deg]):" << std::endl;
  computeRank(ate_stats, path_to_result + "/results_ate.txt", path_to_result + "/rank_ate.txt");
}

int main(int argc, char *argv[])
{
  if (argc != 3)
  {
    std::cout << "usage: ./vbr_benchmark path_to_gt path_to_es" << std::endl;
    return 1;
  }

  const std::string &path_to_gt = argv[1];
  const std::string &path_to_es = argv[2];

  eval(path_to_gt, path_to_es, "train");
  eval(path_to_gt, path_to_es, "test");

  return 0;
}