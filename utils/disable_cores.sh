num_cores=4
num_cores_ht=$((18+$num_cores))
echo num_cores_ht
for i in `seq $num_cores 1 17`;
	do echo 0 > /sys/devices/system/cpu/cpu$i/online;
done

for i in `seq $num_cores_ht 1 36`;
    do echo 0 > /sys/devices/system/cpu/cpu$i/online;
done

