FROM pytorch/pytorch:latest
RUN apt-get update
RUN apt-get install -y python3-tk
RUN pip install --upgrade pip 
RUN pip install matplotlib 