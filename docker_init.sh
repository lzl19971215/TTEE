#!/usr/bin/env bash
set -e
set -x

apt update
apt install -y openssh-server
#mkdir /var/run/sshd
echo 'root:12345' | chpasswd
sed -i '$aPermitRootLogin yes' /etc/ssh/sshd_config
sed -i '$aPubkeyAuthentication yes' /etc/ssh/sshd_config
sed 's@sesesssion\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
service ssh restart



