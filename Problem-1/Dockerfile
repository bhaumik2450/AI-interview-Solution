FROM python:3.8
WORKDIR /workspace
COPY . /workspace
RUN pip install -r requirements.txt
CMD [ "python" , "/workspace/wsgi.py" ]

