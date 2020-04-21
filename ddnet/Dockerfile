FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
ENV LD_LIBRARY_PATH "/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 \
    python3-setuptools \
    python3-pip \
    && pip3 install -U --no-cache-dir pip setuptools wheel \
    && pip3 install --no-cache-dir \
    fire \
    flask \
    logzero \
    numpy \
    scikit-learn \
    scipy \
    tensorflow-gpu==1.15.2 \
    Keras==2.3.1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


VOLUME /model
WORKDIR /root

COPY ["app.py", "ddnet.py", "/root/"]

CMD ["python3", "app.py", "/model/jhmdb_openpose_model.h5", "/model/jhmdb_le.pkl"]