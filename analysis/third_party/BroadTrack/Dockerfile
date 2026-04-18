FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 as builder

ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        cmake \
        build-essential \
        wget \
        unzip \
        libboost-program-options-dev \
        libboost-filesystem-dev\
        libeigen3-dev \
        libflann-dev \
        libgoogle-glog-dev \
        libgtest-dev \
        libgmock-dev \
        libsqlite3-dev \
        libglew-dev \
        qtbase5-dev \
        libceres-dev \
        rapidjson-dev \
        libopencv-dev

RUN wget https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip -O libtorch.zip &&\
 unzip libtorch.zip &&\
 mv libtorch /opt/

COPY .  /home/broadtrack
WORKDIR /home/broadtrack
RUN mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/broadtrack-install && make -j 32 && make install

FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 as runtime
ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        wget \
        unzip \
        build-essential \
        libboost-program-options1.74.0 \
        libboost-filesystem1.74.0\
        libflann1.9 \
        libceres2 \
        libopencv-dev

COPY --from=builder /broadtrack-install/ /usr/local/
COPY --from=builder /opt/libtorch /opt/libtorch
COPY --from=builder /home/broadtrack/models/tvcalib_model.pt /usr/share/broadtrack/ 
COPY --from=builder /home/broadtrack/models/nbjw_keypoint_model.pt /usr/share/broadtrack/ 


ENV LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH
ENV RPATH=/opt/libtorch/lib

CMD ["/bin/bash"]