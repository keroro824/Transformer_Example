FROM nvcr.io/nvidian/pytorch:20.03-py3
RUN pip install torch torchvision
RUN pip install scipy