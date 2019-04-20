echo 'Please enter your user name for bitbucket'
read username
wget --user=$username --ask-password https://bitbucket.org/eric_thesis/reid2/downloads/cuhk03-np.zip
unzip cuhk03-np.zip
rm -rf cuhk03-np.zip
