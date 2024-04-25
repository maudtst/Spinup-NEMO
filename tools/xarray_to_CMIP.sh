for i in "$@"
do
#ls -lh $i
ncrename -v yy,y $i
#ls -lh $i
#ncks -3 -O $i $i
#ls -lh $i
ncrename -v xx,x $i
#ls -lh $i
#ncks -7 - O $i $i
#ls -lh $i
done
