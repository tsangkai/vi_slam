# Visual-Inertial SLAM Simulation Platform

This project is build on [OKVIS: Open Keyframe-based Visual-Inertial SLAM](https://github.com/ethz-asl/okvis).

In addition to simplifying the implementation, I will add different SLAM backend algorithms for completeness.

```
mkdir build
cd build
cmake ../
cmake --build .
./apps/okvis_test ../config/config_fpga_p2_euroc.yaml DATASET_PATH
```

### License ###

The 3-clause BSD license (see file LICENSE) applies.
