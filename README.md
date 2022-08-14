# 📝 torch_cpp

---

## 🎛 Dependencies

---

- [CUDA Toolkits](https://developer.nvidia.com/cuda-toolkit): if you use GPU.

- pytorch c++ API: the easiest way is to reuse build binary provided from [pytorch official website](https://pytorch.org/get-started/locally/). Here is [the sample script to install use the build binary](https://github.com/xmba15/dockerfiles/tree/master/torch_cpp/scripts). For convenience, this repo assumes torch c++ api is installed into _/opt/libtorch_

- other dependencies:

```bash
sudo apt-get install -y --no-install-recommends \
    libopencv-dev
```

## 🔨 How to Build

---

```bash
# build library
make default

# build examples
make apps -j`nproc`
```

## :running: How to Run

---

## :gem: References

---

- [SuperPoint trained by magicleap](https://github.com/magicleap/SuperPointPretrainedNetwork)
- [SuperPoint SLAM](https://github.com/KinglittleQ/SuperPoint_SLAM)
