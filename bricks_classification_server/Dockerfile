FROM ubuntu:18.04
ADD . /bricks_classifier/

# Install general dependences

RUN apt-get update
RUN apt-get install -y screen
RUN apt-get install -y curl
RUN apt-get install -y unzip
RUN apt-get install -y python3-dev
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip
RUN apt-get install -y cmake
RUN apt-get install -y git

RUN pip3 install torch>=1.9.0
RUN pip3 install torchvision>=0.9.1

# Download third parties

RUN pip3 install gdown
WORKDIR /bricks_classifier/
RUN gdown "https://drive.google.com/uc?id=1uax8egG-tbOy2eIgHDjYNkibCJ0f-utK"
RUN unzip -o brick_classification.zip
RUN rm brick_classification.zip

RUN pip3 install -r requirements.txt

CMD python3 classification_server.py
