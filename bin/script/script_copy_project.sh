#!/bin/bash
# use ssh to start this on scheduler #
# change folder path and anaconda path before running #

# add all node list(trainer and sampler)
# cp file to node
for ip in "192.168.100.1" "192.168.100.2" "192.168.100.3" "192.168.100.4"
do
pdsh -R ssh -w $ip -l root rm -rf /root/peng
pdsh -R ssh -w $ip -l root mkdir peng"
pdcp -R ssh -w $ip -l root -r ../../../alphaniao /root/peng/
done


