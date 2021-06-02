FROM ubuntu:latest
RUN apt-get update
RUN apt-get -y install wget
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b
RUN rm Miniconda3-latest-Linux-x86_64.sh

#add to path
ENV PATH /root/miniconda3/bin:$PATH
#initialise conda
RUN conda init bash

#copy directories
COPY ./model_app /model_app
WORKDIR /model_app
RUN conda env create --name flask_env --file environment.yaml

ENTRYPOINT ["conda", "run", "-n", "flask_env", "python", "./iris_flask.py"]

