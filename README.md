# CDS_Depth_Estimation
Depth Estimation from Images using Computer Vision

## Installing Requirements
We used python 3.7.7 for this project.

Our code is built on pytorch version 1.13.1+cu116. We mention torch==1.13.1 in the requirements file but you might need to install a specific cuda version of torch depending on your GPU device type.

```shell
pip install -r requirements.txt
```
## To Run Demo:
### Setting up the Demo
Download the `ckpt` folder from the OneDrive link to get our trained model checkpoints for demo [here](https://sutdapac-my.sharepoint.com/:f:/g/personal/peixuan_lee_mymail_sutd_edu_sg/EpZJNFK_vIpFpSTxINMsr4oBfaxx_g_-J1M-TBHYXP90PA?e=aehY7G)

Place the `ckpt` folder under the `./MIM-Depth-Estimation` directory in the repository.
### Running the Demo
```shell
cd ./MIM-Depth-Estimation
python demo.py
```
After running the commands, you will get the local URL for the demo webpage displayed on the command line. Open the URL to view the demo page.


