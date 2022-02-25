# Docker image to singularity, and its use on Jean-Zay and HAL

This document explains how to perform the conversion of a docker image to singularity and
how launch it on Jean-Zay.

## Docker to singularity

There are two ways to convert a docker image to singularity :

The first, if your system has a version of singularity, is to use it via the command :

```
singularity build /path/to/singularity/image.sif path/to/docker/image
```

The second, in case singularity is not installed on your system, is to use the docker image `docker2singularity`
which will convert the image as follows:

```
docker run -v /var/run/docker.sock:/var/run/docker.sock 
-v /path/to/output:/output --privileged 
-t --rm quay.io/singularity/docker2singularity docker/image
```

Just change the output path and the docker image used to perform the conversion.

It is possible to perform other types of conversion of the docker image. 
You can get the documentation of the docker image with all the parameters at the following link: 
https://github.com/singularityhub/docker2singularity

## Use on Jean-Zay

The use of singularity on Jean-Zay is detailed on the IDRIS website:
http://www.idris.fr/jean-zay/cpu/jean-zay-utilisation-singularity.html

First, it's necessary to load the singularity module via the command :

```
module load singularity
```

On Jean-Zay, the singularity images must be in the $SINGULARITY_ALLOWED_DIR folder to be used.
You can copy your image to this folder with the command :

```
idrcontmgr cp my-container.sif
```

Once your image has been copied, you can use it via a slurm job as in the following example: 

```
#!/bin/bash
#SBATCH -A xxx@gpu
#SBATCH --partition=gpu_p2
#SBATCH --job-name=job_name # nom du job
#SBATCH --ntasks=1                     # number of MPI tasks
#SBATCH --ntasks-per-node=1            # number of MPI tasks per node
#SBATCH --gres=gpu:8                   # number of GPU to allocate per node
#SBATCH --cpus-per-task=24             # number of core to allocate per task
#SBATCH --qos=qos_gpu-t4
#SBATCH --hint=nomultithread           # allocate physical cores
#SBATCH --distribution=block:block     # pin tasks on contiguous blocks
#SBATCH --time=24:00:00                # wall time
#SBATCH --output=path/to/log.out       # output log
#SBATCH --error=path/to/log.err        # error log

cd ${SLURM_SUBMIT_DIR}
module purge
module load singularity
set -x

# execution
srun singularity exec --nv 
 --bind /path/to/bind1:path/to/bind1, /path/to/bind2:path/to/bind2
 $SINGULARITY_ALLOWED_DIR/singularity_image.tif 
 bash -c 
 "export PYTHONPATH=$PYTHONPATH:/path/to/decloud/repository && 
 python3 ~/decloud/models/train_from_tfrecords.py 
 --training_record /path/to/tfrecord/train 
 --valid_records /path/to/tfrecord/valid 
 --model <MODELNAME> 
 --logdir /path/to/logdir 
 --save_ckpt_dir /path/to/ckpt/dir 
 -bt <BATCHSIZE> -bv <BATCHSIZE> 
 -e nb_epoch
 "
```

The ``--nv`` option allows the use of nvidia cards and therefore the gpu.

In case the program needs to access the ``scratch``, ``work`` disks, it is necessary to mount the paths via the
``--bind`` options.

## Use on HAL

The use of singularity on HAL is detailed on the CNES gitlab:
https://gitlab.cnes.fr/hpc/wikiHPC/-/wikis/Singularity-and-PBS


To use singularity you must load the module in your pbs job with the command  :

```
module load singularity
```

Below is an example of a job pbs for network training :

```
#!/bin/bash
#PBS -N decloud
#PBS -q qgpgpu
#PBS -l select=1:ncpus=8:mem=92G:ngpus=2
#PBS -l walltime=01:00:00
#PBS -o /path/to/output/log/job.o
#PBS -e /path/to/error/log/job.e

cd "${TMPDIR}"
module purge
module load singularity

# execution
singularity exec --nv --bind /path/to/volume:/path/to/volume \
 /path/to/singularity/image/image.sif bash -c \
 "export PYTHONPATH=\$PYTHONPATH:/path/to/decloud && \
 python3 ~/decloud/models/train_from_tfrecords.py \
 --training_record /path/to/TFRecord/train \
 --valid_records /path/to/TFRecord/valid \
 --model <MODELNAME> \
 --logdir /path/to/output/tensorboard \
 --save_ckpt_dir /path/to/save/ckpt \
 --load_ckpt_dir /path/to/ckpt/to/load \
 -bt <BATCHSIZE> -bv <BATCHSIZE> \
 -lr 0.00007 \
 -e <EPOCH>
 "
```

This job launches the network training on 2 gpu. 

We specify the number of CPU, GPU and memory to be allocated with the following instruction:

`#PBS -l select=1:ncpus=8:mem=92G:ngpus=2`

In our case, 8 CPUs, 2 GPUs and 92G of RAM.

Instruction `#PBS -l walltime=01:00:00` indicates the maximum duration of the job.

Finally, we can launch our program with the command : 

`singularity exec --nv --bind /path/to/volume:/path/to/volume \
 /path/to/singularity/image/image.sif bash -c `

We indicate that we want to launch a program in a singularity container with 
Finally, we can launch our program with the command `singularity exec`. The `--nv` option 
indicates the use of GPU, and the `--bind` option the path of volumes to be mounted.
