mkdir OCE
mkdir OCE/Restart
mkdir ATM
mkdir ATM/Restart
mkdir ICE
mkdir ICE/Restart
mkdir SBG
mkdir SBG/Restart
mkdir SRF
mkdir SRF/Restart
mkdir MBG
mkdir MBG/Restart
mkdir CPL
mkdir CPL/Restart
cp -r ../Exe .
for a in $@
do
mv $a ${a:0:3}'/Restart/'${a#???_}
done
