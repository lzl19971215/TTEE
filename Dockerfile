FROM lzl19971215/ttabsa_tf2.7:v1
MAINTAINER "liziliang <liziliangluck@163.com>"
#ARG HOST_UID
#ENV HOST_UID=${HOST_UID}
#RUN useradd -r -m -d /home/hostuser -s /bin/bash -u ${HOST_UID} hostuser

WORKDIR /workspace
COPY ./sources.list /etc/apt/sources.list
COPY ./docker_init.sh /workspace/docker_init.sh
COPY ./ /workspace/absa/
#RUN echo "hostuser:12345" | chpasswd
RUN /bin/bash docker_init.sh
ENV LC_ALL=C.UTF-8

#USER hostuser
RUN mkdir absa/checkpoint

RUN sed -i "\$aexport \$(cat /proc/1/environ |tr \'\\\\0' '\\\\n' | xargs)" /etc/profile
ENV LC_ALL=C.UTF-8
CMD service ssh restart && /bin/bash
# ENTRYPOINT CMD 运行完毕时退出
