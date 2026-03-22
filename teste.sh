#!/bin/bash
#PBS -N CharikloRandom
#PBS -q asaza09
#PBS -l walltime=336:00:00
#PBS -l nodes=1:ppn=30
#PBS -o saida_principal.out
#PBS -e erro_principal.err
#PBS -m abe

# --- NOTA SOBRE RECURSOS ---
# O Python multiprocessing roda em memoria compartilhada.
# Estamos pedindo 1 no inteiro com 48 cores (ppn=48).
# Se o job nao entrar nunca (ficar em 'Q' dias), diminua para ppn=40 ou ppn=32.

echo "=========================================="
echo "Job iniciado em: $(date)"
echo "Rodando no host: $(hostname)"
echo "CPUs alocados: $PBS_NP"
echo "=========================================="

cd $PBS_O_WORKDIR

# Carregar o modulo (ajuste se necessario)
module load python/3.6.8-pandas

# Exportar variaveis para garantir que o Numpy nao tente paralelizar por cima
# Queremos que cada processo Python use 1 core, e nao que cada processo use todos.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Iniciando simulacao Python..."

# O script vai ler o PBS_NP automaticamente
python3 teste_lyapunov.py

echo "Job finalizado em: $(date)"