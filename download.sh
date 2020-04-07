#!/bin/bash
wget https://cloud.dfki.de/owncloud/index.php/s/DEqe5SQCxSGWRkQ/download -O pwoc3d-kitti.zip
unzip pwoc3d-kitti.zip -d ./data/
rm pwoc3d-kitti.zip
wget https://cloud.dfki.de/owncloud/index.php/s/yjFg74FtrLaxZ8j/download -O pwoc3d-ft3d.zip
unzip pwoc3d-ft3d.zip -d ./data/
rm pwoc3d-ft3d.zip
