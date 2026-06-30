

# 3. 网络与分布式训练环境配置
# 注意：修改 /etc/hosts 通常需要 sudo 权限
sh -c 'echo "11.255.255.16 wandb.ai" >> /etc/hosts' 2>/dev/null || echo "Info: Skip hosts update (no sudo)"
sh -c 'echo "11.255.255.16 api.wandb.ai" >> /etc/hosts' 2>/dev/null
sh -c "echo '127.0.0.1 $(hostname)' >> /etc/hosts" 2>/dev/null

export NCCL_IPV6_ADDR=0 
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

source /kpfs-intern/chenyandu/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate /kpfs-intern/chenyandu/miniconda3/envs/lingbotva
which python
python -c "import sys, torch; print(sys.executable); print(torch.__version__)"

NGPU=8 CONFIG_NAME=rmbench_train \
bash script/run_va_posttrain.sh \
      --dataset-path /kpfs-intern/chenyandu/data/rmbench-lingbot \
      --pretrained-model-path /kpfs-intern/chenyandu/models/lingbot-va/lingbot-va-posttrain-robotwin \
      --save-root ./train_out