# vllm-power-test

vllm-power-test

venv
pip install datasets
MAX_JOBS=16 pip install -e .

tmux new-session -d -s batch_run "bash -lc 'cd ~/easyhyum/test && python3 ./run_batch_inference.py --data-len=9 --max-tokens=10 --batch-size=9 > run_batch.log 2>&1'"

cd ~/easyhyum/test
source venv/bin/activate
nohup python ./run_batch_inference.py --data-len=9 --max-tokens=10 --batch-size=9 > run_batch.log 2>&1 &
echo $! > run_batch.pid
