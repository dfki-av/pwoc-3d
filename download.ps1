Invoke-WebRequest -Uri https://cloud.dfki.de/owncloud/index.php/s/DEqe5SQCxSGWRkQ/download -OutFile pwoc3d-kitti.zip
Expand-Archive pwoc3d-kitti.zip -DestinationPath ./data/
Remove-Item pwoc3d-kitti.zip
