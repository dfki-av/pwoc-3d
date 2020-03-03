#!/bin/bash
wget https://cloud.dfki.de/owncloud/index.php/s/DEqe5SQCxSGWRkQ/download -O pwoc3d-kitti.zip
unzip pwoc3d-kitti.zip -d ./data/
rm pwoc3d-kitti.zip
