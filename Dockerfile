# Decloud dockerfile
# To build the docker image for cpu, do the following:
#
# docker build --build-arg "BASE_IMAGE=mdl4eo/otbtf3.3.2:cpu-dev" .
#
ARG BASE_IMAGE=mdl4eo/otbtf3.3.2:gpu-dev
FROM $BASE_IMAGE
LABEL description="Decloud docker image"
LABEL maintainer="Remi Cresson [at] inrae [dot] fr"
USER root

# APT Packages
COPY docker/apt_packages.txt /tmp/apt_packages.txt
RUN apt update && sed 's/#.*//' /tmp/apt_packages.txt | xargs apt-get install -y && apt clean
RUN apt upgrade -y

# Pip packages
COPY docker/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt

# Build remote modules
RUN cd /src/otb/otb/Modules/Remote/ && git clone https://gitlab.irstea.fr/remi.cresson/SimpleExtractionTools.git
RUN cd /src/otb/otb/Modules/Remote/ && git clone https://gitlab.irstea.fr/remi.cresson/mlutils.git
COPY . /src/otb/otb/Modules/Remote/decloud/
RUN cd /src/otb/build/OTB/build && cmake /src/otb/otb/ -DModule_SimpleExtractionTools=ON -DModule_MLUtils=ON -DBUILD_TESTING=OFF -DModule_OTBDecloud=ON 
RUN cd /src/otb/build/OTB/build && make -j $(nproc --all) install

# Install decloud
RUN cd /src/otb/otb/Modules/Remote/decloud/ && python3 -m pip install .

USER otbuser
