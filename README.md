## VBR SLAM Benchmark

Please use this benchmark to validate the training sequences; this is exactly the same running in the website backend [www.rvp-group.net/slam-benchmark](https://www.rvp-group.net/slam-benchmark.html). Follow the below instructions for usage: 
### Compile
```g++ -o vbr_benchmark vbr_benchmark.cpp -I /usr/include/eigen3 -O3 -std=c++17```

### Run 
```./vbr_benchmark /path_to/vbr_gt/ /path_to/method/ [--plot]```

You can enable the `--plot` flag to get aligned trajectory plots and error files for your training estimates.

>[!IMPORTANT]
> When you run this locally, you will not have ground truth data for test sequences, so ignore messages related to missing test data.
> When you submit, make sure to insert both your training and test estimates since we evaluate both. *If you do not submit one of these, the evaluation will not run!* 
