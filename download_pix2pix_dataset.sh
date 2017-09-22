FILE=$1

if [[ $FILE != "cityscapes" && $FILE != "edges2handbags" && $FILE != "edges2shoes" &&  $FILE != "facades" && $FILE != "maps" ]]; then
    echo "Available datasets are: edges2handbags, edges2shoes, maps, cityscapes, facades"
    exit 1
fi

URL=https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/$FILE.tar.gz
TAR_FILE=./datasets/$FILE.tar.gz
TARGET_DIR=./datasets/$FILE/
wget -N $URL -O $TAR_FILE
mkdir $TARGET_DIR
tar -zxvf $TAR_FILE -C ./datasets/
rm $TAR_FILE
