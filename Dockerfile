FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel

RUN apt-get update && apt-get upgrade -y
RUN apt-get install libgl1-mesa-glx -y
RUN apt-get install libglib2.0-0 -y





ENV NB_USER jovyan
ENV NB_UID 1000
ENV HOME /home/$NB_USER

RUN useradd -M -s /bin/bash -N -u ${NB_UID} ${NB_USER} \
 && mkdir -p ${HOME} \
 && chown -R ${NB_USER}:users ${HOME} \
 && chown -R ${NB_USER}:users /usr/local/bin

ENV PATH "$PATH:/home/jovyan/.local/bin"

USER ${NB_UID}

RUN pip install pandas
RUN pip install notebook
RUN pip install ipykernel
RUN pip install scikit-learn
RUN pip install pandas
RUN pip install tqdm
RUN pip install transformers
RUN pip install bitsandbytes
RUN pip install pytorch-lightning
RUN pip install sentencepiece

RUN pip install iterative-stratification
RUN pip install text_unidecode

RUN apt-get install libgl1-mesa-glx -y
RUN apt-get install libglib2.0-0 -y

RUN pip install ipywidgets
RUN pip install -U protobuf==3.20.1
