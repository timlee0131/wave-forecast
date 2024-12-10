#!/usr/bin/env zsh

save_dir="/lustre/smuexa01/client/users/hunjael/results/wave-forecast/${SLURM_JOB_ID}_${dataset}"

srun\
    -N1\
    -G1\
    -p short\
    --no-container-entrypoint\
    --container-image /work/group/humingamelab/sqsh_images/nvidia-pyg.sqsh\
    --container-mounts="${HOME}"/Projects/wave-forecast:/wave-forecast,/work/users/hunjael/data:/data,"${SCRATCH}":/models\
    --container-workdir /wave-forecast\
    --pty bash -i