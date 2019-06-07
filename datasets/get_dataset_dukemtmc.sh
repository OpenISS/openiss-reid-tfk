echo 'Please enter your user name for bitbucket'
read username
wget --user=$username --ask-password https://drive.google.com/uc?id=0B0VOCNYh8HeRdnBPa2ZWaVBYSVk
mv DukeMTMC-reID.zip dukemtmc-reid.zip
unzip dukemtmc-reid.zip
rm -rf dukemtmc-reid.zip
