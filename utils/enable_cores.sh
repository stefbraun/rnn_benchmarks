for i in $(seq 36 $END);
	do echo 1 > /sys/devices/system/cpu/cpu$i/online;
done
