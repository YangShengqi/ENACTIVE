# run sampler node
for ip in "192.168.100.2" "192.168.100.3" "192.168.100.4"
do
# pdsh -R ssh -w $ip -l root pkill python
pdsh -R ssh -w $ip -l root /root/env/bin/python /root/peng/alphaniao/bin/sideless_scpbt/scpbt_sampler.py &
done