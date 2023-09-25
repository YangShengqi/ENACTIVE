# run trainer node
for ip in "192.168.100.1"
do
# pdsh -R ssh -w $ip -l root pkill python
pdsh -R ssh -w $ip -l root /root/env/bin/python /root/peng/alphaniao/bin/sideless_scpbt/scpbt_trainer.py &
done