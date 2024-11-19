make -C build_T4 clean && make -C build_T4 bx=16 by=16
# for num in {2..128}
# do 
# echo $num "+++++++++++++++++++++++"
# ./mmpy -n $num
# done

for num in {254,255,256,257,258,1022,1023,1024,1025,1026,2046,2047,2048,2049,2050,4095,4096,4097,8191,8192,8193}
do 
echo $num "+++++++++++++++++++++++"
./mmpy -n $num
done