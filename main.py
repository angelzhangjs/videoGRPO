# # check the repo
# import os
# sys.path.append("ltx_video_source")
# from ltx_video.inference import create_ltx_video_pipeline
# from video_grpo import video_rollout, create_sampling_configurations, video_pipeline_configuration
# from dataclasses import dataclass

# # create the pipeline
# pipeline = create_ltx_video_pipeline(
#     ckpt_path="ltx_video_source/configs/ltxv-2b-0.9.8-distilled.yaml",
#     precision="bfloat16",
#     text_encoder_model_name_or_path="google/t5-v1_1-small",
#     sampler="from_checkpoint",
#     device="cuda",
# )

