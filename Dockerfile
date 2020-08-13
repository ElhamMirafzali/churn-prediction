FROM python:3.8.2-slim-buster

WORKDIR /odin
COPY . ./
ENV TIMEZONE=Iran
RUN  cp --remove-destination /usr/share/zoneinfo/${TIMEZONE} /etc/localtime && \
        echo "${TZ}" > /etc/timezone && \
        pip install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
        pip install <needed libraries>

ENTRYPOINT ["python3.8","your_file.py"]