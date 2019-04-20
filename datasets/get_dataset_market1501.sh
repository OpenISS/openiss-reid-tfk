echo 'Please enter your user name for bitbucket'
read username
wget --user=$username --ask-password https://bitbucket.org/eric_thesis/reid2/downloads/market1501.zip
unzip market1501.zip
rm -rf market1501.zip
