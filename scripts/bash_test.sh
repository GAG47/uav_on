conda activate gdino
cd /home/tjn2004/uav/UAV_ON

export GROUNDINGDINO_HOME=/home/tjn2004/GroundingDINO
export PYTHONPATH=/home/tjn2004/GroundingDINO:$PYTHONPATH

export GROUNDINGDINO_CONFIG=/home/tjn2004/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
export GROUNDINGDINO_WEIGHT=/home/tjn2004/GroundingDINO/weights/groundingdino_swint_ogc.pth
export GROUNDINGDINO_DEVICE=cuda
export GROUNDINGDINO_BOX_THRESHOLD=0.25
export GROUNDINGDINO_TEXT_THRESHOLD=0.20

# Hugging Face 本地缓存和离线模式
export HF_HOME=/home/tjn2004/.cache/huggingface
export HF_HUB_CACHE=/home/tjn2004/.cache/huggingface/hub
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

python tools/gdino_server.py




conda activate uavon
cd /home/tjn2004/uav/UAV_ON
PYTHONUNBUFFERED=1 python -u airsim_plugin/AirVLNSimulatorServerTool.py \
  --port=30000 \
  --root_path=/home/tjn2004/uav/TEST_ENVS\




conda activate uavon
cd /home/tjn2004/uav/UAV_ON 
export DASHSCOPE_API_KEY="sk-c107fcf27c3d44358cd304327c5f0698"
export OPENAI_API_KEY="sk-of-ZzsegHwvqOvdKEcnORLFitNmYnZohGjzoaPKtWNjZQUZudJvtljyWUirgEvISbhm"
export OPENAI_BASE_URL="https://api.ofox.ai/v1"
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export USE_GROUNDING_DINO=1
export GDINO_SERVER_URL=http://127.0.0.1:8008/detect
export GROUNDINGDINO_BOX_THRESHOLD=0.25
export GROUNDINGDINO_TEXT_THRESHOLD=0.20
export GROUNDINGDINO_MAX_IMAGES=4
export GROUNDINGDINO_TIMEOUT=20
export GROUNDINGDINO_PROMPT_SOURCE=llm
export GROUNDINGDINO_PROMPT_CACHE=1
export GROUNDINGDINO_MAX_PROMPT_PHRASES=8
export GROUNDINGDINO_MAX_PROMPT_WORDS=8
export TARGET_VERIFIER_ALLOW_DOWN=1
export TARGET_VERIFIER_CROP_PADDING=0.35
export TARGET_VERIFIER_MIN_CROP_SIZE=224
export TARGET_VERIFIER_THRESHOLD=0.58
bash scripts/eval_random_batched.sh 20 42 2