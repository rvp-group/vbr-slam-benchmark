#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <limits>
#include <algorithm>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

const vector<float> percentages_ = {0.01, 0.02, 0.03, 0.05, 0.08, 0.13, 0.21, 0.34, 0.55};

const vector<std::string> sequence_names_ = {
  "campus_train_0",
  "campus_train_1",
  "ciampino_train_0",
  "ciampino_train_1",
  "colosseo_train",
  "piazza_di_spagna_train",
  "pincio_train",
  "diag_train"
};

const float max_error_ = 10;

struct error_ {
  int first_frame_;
  float r_err_;
  float t_err_;
  float len_;
  error_(int first_frame, float r_err, float t_err, float len) :
    first_frame_(first_frame), r_err_(r_err), t_err_(t_err), len_(len) {}
};

struct stats_ {
  string sequence_name_;
  float r_err_;
  float t_err_;
  stats_(string sequence_name, float r_err, float t_err) :
    sequence_name_(sequence_name), r_err_(r_err), t_err_(t_err) {}
};

struct pose_ {
  float timestamp_;
  Isometry3f transform_;
  pose_(float timestamp, Isometry3f transform) :
    timestamp_(timestamp), transform_(transform) {}
};

inline bool sortComparator(const stats_& stat_l, const stats_& stat_r) {
  return stat_l.t_err_ < stat_r.t_err_;
}

inline vector<pose_> loadPoses(const string& file_name) {
  vector<pose_> poses;
  ifstream file(file_name);
  if (!file.is_open()) {
    cerr << "Error: Unable to open file " << file_name << endl;
    return poses;
  }

  while (!file.eof()) {
    Isometry3f P = Isometry3f::Identity();
    float t, x, y, z, qx, qy, qz, qw;
    file >> t >> x >> y >> z >> qx >> qy >> qz >> qw;
    if (file.eof())
      break;

    P.translation() << x, y, z;

    const Quaternionf q(qw, qx, qy, qz);
    P.linear() = q.toRotationMatrix();

    poses.push_back(pose_(t, P));
  }

  file.close();
  return poses;
}

inline vector<pose_> matchTimestamps(const vector<pose_>& poses_gt, const vector<pose_>& poses_es) {
  vector<pose_> poses_matched;
  poses_matched.reserve(poses_gt.size());

  size_t es_index = 0;
  for (const auto& pose_gt : poses_gt) {
    // while (es_index < poses_es.size() -  && poses_es[es_index].timestamp_ < pose_gt.timestamp_)
    //   es_index++;
    
    float min_delta = abs(poses_es[es_index].timestamp_ - pose_gt.timestamp_);
    while (es_index < poses_es.size() - 1 && abs(poses_es[es_index + 1].timestamp_ - pose_gt.timestamp_) <= min_delta) {
      min_delta = abs(poses_es[es_index + 1].timestamp_ - pose_gt.timestamp_);
      es_index++;
    }

    poses_matched.push_back(poses_es[es_index]);
  }

  return poses_matched;
}

inline vector<float> trajectoryDistances(const vector<pose_>& poses) {
  vector<float> dist;
  dist.push_back(0);
  for (int i = 1; i < poses.size(); ++i) {
    const Vector3f t1 = poses[i - 1].transform_.translation();
    const Vector3f t2 = poses[i].transform_.translation();

    dist.push_back(dist[i - 1] + (t1 - t2).norm());
  }
  return dist;
}

inline int lastFrameFromSegmentLength(const vector<float>& dist, const int& first_frame, const float& len) {
  for (int i = first_frame; i < dist.size(); ++i)
    if (dist[i] > dist[first_frame] + len)
      return i;
  return -1;
}

inline float rotationError(const Isometry3f& pose_error) {
  Quaternionf q(pose_error.linear());
  q.normalize();

  const Quaternionf q_identity(1.0f, 0.0f, 0.0f, 0.0f);
  const float error_radians = q_identity.angularDistance(q);

  const float error_degrees = error_radians * (180.0f / M_PI);
  return error_degrees;
}

inline float translationError(const Isometry3f& pose_error) {
    const Vector3f t = pose_error.translation();
    return t.norm();
}

inline vector<error_> computeSequenceErrors(const vector<pose_> poses_gt, const vector<pose_>& poses_es) {
  vector<error_> err;

  const vector<float> dist = trajectoryDistances(poses_gt);
  const float seq_length = dist.back();
  cerr << "Sequence length [m]: " << seq_length << endl;

  vector<float> lengths;
  for (const float& percentage : percentages_) {
    const float len = seq_length * percentage;
    lengths.push_back(len);
    cerr << "Percentage: " << percentage << ", subsequence length [m]: " << len << endl;
  }
  cerr << endl;

  for (int first_frame = 0; first_frame < poses_gt.size(); ++first_frame) {
    for (int i = 0; i < lengths.size(); ++i) {

      const float curr_len = lengths[i];
      const int last_frame = lastFrameFromSegmentLength(dist, first_frame, curr_len);

      if (last_frame == -1)
        continue;

      const Isometry3f pose_delta_gt = poses_gt[first_frame].transform_.inverse() * poses_gt[last_frame].transform_;
      const Isometry3f pose_delta_es = poses_es[first_frame].transform_.inverse() * poses_es[last_frame].transform_;
      const Isometry3f pose_error = pose_delta_es.inverse() * pose_delta_gt;
      const float r_err = rotationError(pose_error);
      const float t_err = translationError(pose_error);

      err.push_back(error_(first_frame, r_err / curr_len, t_err / curr_len, curr_len));
    }
  }

  return err;
}

inline vector<pose_> computeAlignedEstimate(const vector<pose_>& poses_gt, const vector<pose_>& poses_es) {
  vector<pose_> poses_es_aligned;
  poses_es_aligned.reserve(poses_es.size());

  Matrix<float, 3, Dynamic> gt_matrix;
  gt_matrix.resize(NoChange, poses_gt.size());
  for (int i = 0; i < poses_gt.size(); ++i)
    gt_matrix.col(i) = poses_gt[i].transform_.translation();

  Matrix<float, 3, Dynamic> es_matrix;
  es_matrix.resize(NoChange, poses_es.size());
  for (int i = 0; i < poses_es.size(); ++i)
    es_matrix.col(i) = poses_es[i].transform_.translation();
  
  // last argument to true for monocular (Sim3)
  const Eigen::Matrix4f transform_matrix = Eigen::umeyama(es_matrix, gt_matrix, false);
  Eigen::Isometry3f transform = Eigen::Isometry3f(transform_matrix.block<3, 3>(0, 0));
  transform.translation() = transform_matrix.block<3, 1>(0, 3);
  
  for (int i = 0; i < poses_es.size(); ++i)
    poses_es_aligned.push_back(pose_(poses_es[i].timestamp_, transform * poses_es[i].transform_));
    
  return poses_es_aligned;
}

inline stats_ computeSequenceRPE(const vector<error_>& seq_err, const string& sequence_name) {
  float t_err = 0;
  float r_err = 0;

  for (const error_& error: seq_err) {
    t_err += error.t_err_;
    r_err += error.r_err_;
  }

  const float r_rpe = r_err / float(seq_err.size());
  const float t_rpe = 100 * t_err / float(seq_err.size());
  return stats_(sequence_name, r_rpe, t_rpe);
}

inline stats_ computeSequenceATE(const vector<pose_>& poses_gt, const vector<pose_>& poses_es_aligned, const string& sequence_name) {
  float r_sum = 0;
  float t_sum = 0;

  for (int i = 0; i < poses_gt.size(); ++i) {
    const Isometry3f pose_error = poses_gt[i].transform_.inverse() * poses_es_aligned[i].transform_;
    const float r_err = rotationError(pose_error);
    const float t_err = translationError(pose_error);

    r_sum += r_err;
    t_sum += t_err;
  }

  const float r_ate_rmse = std::sqrt(r_sum / float(poses_gt.size()));
  const float t_ate_rmse = std::sqrt(t_sum / float(poses_gt.size()));
  return stats_(sequence_name, r_ate_rmse, t_ate_rmse);
}

inline void computeRank(vector<stats_>& stats, const string& path_to_result_file, const string& path_to_rank_file) {
  sort(stats.begin(), stats.end(), sortComparator);

  FILE* fp = fopen(path_to_result_file.c_str(), "w");

  float rank = 0;
  for (const stats_ stat: stats) {
    fprintf(fp, "%f %f\n", stat.t_err_, stat.r_err_);
    cerr << stat.sequence_name_ << " " << stat.t_err_ << " " << stat.r_err_;

    if (stat.t_err_ > max_error_) {
      cerr << " - exceeded max error" << endl;
      continue;
    }

    rank += max_error_ - stat.t_err_;
    cerr << endl;
  }

  fclose(fp);

  fp = fopen(path_to_rank_file.c_str(), "w");
  fprintf(fp, "%f\n", rank);
  fclose(fp);

  cerr << "Rank: " << rank << endl << endl;
}


inline void eval(const string& path_to_gt, const string& path_to_es) {
  const string path_to_result = path_to_es + "/results";
  system(("mkdir -p " + path_to_result).c_str());

  vector<stats_> rpe_stats;
  vector<stats_> ate_stats;
  for (int i = 0; i < sequence_names_.size(); ++i) {
    const string sequence_name = sequence_names_[i];
    const string path_to_gt_file = path_to_gt + "/" + sequence_name + "_gt.txt";
    const string path_to_es_file = path_to_es + "/" + sequence_name + "_es.txt";

    const vector<pose_> poses_gt = loadPoses(path_to_gt_file);
    const vector<pose_> poses_es_unmatched = loadPoses(path_to_es_file);

    cerr << "Processing: " << sequence_name << endl;
    cerr << "Estimated poses: " << poses_es_unmatched.size() << endl;
    cerr << "Gt poses: " << poses_gt.size() << endl;

    if (poses_gt.size() == 0 || poses_es_unmatched.size() == 0) {
      cerr << "ERROR: Could not read (all) poses of: " << sequence_name << endl;
      continue;
    }

    const vector<pose_> poses_es = matchTimestamps(poses_gt, poses_es_unmatched);
    cerr << "Matched poses: " << poses_es.size() << endl << endl;

    const vector<error_> seq_err = computeSequenceErrors(poses_gt, poses_es);
    const stats_ rpe_stat = computeSequenceRPE(seq_err, sequence_name);
    rpe_stats.push_back(rpe_stat);

    const vector<pose_> poses_es_aligned = computeAlignedEstimate(poses_gt, poses_es);
    const stats_ ate_stat = computeSequenceATE(poses_gt, poses_es_aligned, sequence_name);
    ate_stats.push_back(ate_stat);
  }

  cerr << "Stats RPE (sequence, t_err [%], r_err [deg/m]):" << endl;
  computeRank(rpe_stats, path_to_result + "/results_rpe.txt", path_to_result + "/rank_rpe.txt");

  cerr << "Stats ATE RMSE (sequence, t_err [m], r_err [deg]):" << endl;
  computeRank(ate_stats, path_to_result + "/results_ate.txt", path_to_result + "/rank_ate.txt");
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    cerr << "Usage: ./vbr_benchmark path_to_gt path_to_es" << endl;
    return 1;
  }

  const string path_to_gt = argv[1];
  const string path_to_es = argv[2];

  eval(path_to_gt, path_to_es);

  return 0;
}