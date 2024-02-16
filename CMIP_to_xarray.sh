for i in "$@"; do
  	ncrename -v y,yy $i
	ncrename -v x,xx $i 
done
