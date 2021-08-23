FROM python:3.7-slim

RUN mkdir src
WORKDIR src

RUN pip install jupyter

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
