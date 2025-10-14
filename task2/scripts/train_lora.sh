export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="Nechintosh/ghibli"
export OUTPUT_DIR="/content/drive/MyDrive/lab2_outputs/Lora3"

accelerate launch --mixed_precision="no" train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir=$OUTPUT_DIR \
  --dataset_name=$DATASET_NAME \
  --caption_column="caption" \
  --resolution=512 \
  --random_flip \
  --train_batch_size=16 \
  --num_train_epochs=100 \
  --validation_epochs 1 \
  --checkpointing_steps=20000 \
  --learning_rate=1e-04 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --seed=42 \
  --checkpoints_total_limit  \
  --validation_prompt="a girl wearing sunglasses" \
  --resume_from_checkpoint= "/content/drive/MyDrive/lab2_outputs/Lora3/checkpoint-4000"
  