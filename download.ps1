Invoke-WebRequest -Uri https://cloud.dfki.de/owncloud/index.php/s/DEqe5SQCxSGWRkQ/download -OutFile pwoc3d-kitti.zip
Expand-Archive pwoc3d-kitti.zip -DestinationPath ./data/
Remove-Item pwoc3d-kitti.zip
Invoke-WebRequest -Uri https://cloud.dfki.de/owncloud/index.php/s/yjFg74FtrLaxZ8j/download -OutFile pwoc3d-ft3d.zip
Expand-Archive pwoc3d-ft3d.zip -DestinationPath ./data/
Remove-Item pwoc3d-ft3d.zip
