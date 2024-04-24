module unload intel-mpi
module load pytorch-gpu/py3/1.11.0
#$1=Restart_Path
#$2=Radical
#$3=Mask_Path
#$4=Prediction_Path

bash CMIP_to_xarray.sh $1/$2*.nc

python3 main_restart.py --restart_path $1 --radical $2 --mask_file $3 --prediction_path $4

for ind in {0..339}
do
num=$( printf "%04d" $ind )
ncks -A $1'New_'$2'_'$num'.nc' $1$2'_'$num'.nc'
rm -f $1'New_'$2'_'$num'.nc'
echo $num
done

bash xarray_to_CMIP.sh $1/$2*.nc
