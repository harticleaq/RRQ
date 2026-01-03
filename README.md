# RRQ
The code of rrq, we will open all results after our paper accepted!


## Management log
We use Sacred and Omniboard to manage our results, with the data stored in MongoDB.


## Installation
Set up the Sacred:

```bash
pip install sacred
```

Set up StarCraft II and SMAC with the following command:

```bash
bash install_sc2.sh
```
It will download SC2.4.6.2.69232 into the 3rdparty folder and copy the maps necessary to run over. You also need to set the global environment variable:

```bash
export SC2PATH=[Your SC2 Path/StarCraftII]
```

Install Python environment with command:

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
pip install numpy scipy pyyaml pygame pytest probscale imageio snakeviz 
```
