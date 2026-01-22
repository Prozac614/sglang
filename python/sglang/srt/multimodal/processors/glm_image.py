from typing import List, Union

from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.models.glm_image_ar import GlmImageForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens


class GlmImageProcessor(SGLangBaseProcessor):
    models = [GlmImageForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        # GLM-V specific tokens
        self.IMAGE_TOKEN = "<|image|>"
        self.IMAGE_START_TOKEN = "<|begin_of_image|>"
        self.IMAGE_END_TOKEN = "<|end_of_image|>"

        # Token IDs
        self.IM_TOKEN_ID = hf_config.image_token_id
        self.IM_START_TOKEN_ID = hf_config.image_start_token_id
        self.IM_END_TOKEN_ID = hf_config.image_end_token_id

        # Vision config
        self.IMAGE_FACTOR = 16
        self.MIN_PIXELS = 224 * 224
        self.MAX_PIXELS = 2048 * 2048

        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.IMAGE_TOKEN,
            image_token_id=self.IM_TOKEN_ID,
        ).build(_processor)

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
        )

        mm_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        input_ids = input_ids.flatten()
        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index_glm4v(
            input_ids=input_ids.unsqueeze(0),
            hf_config=self.hf_config,
            image_grid_thw=getattr(ret, "image_grid_thw", None),
            attention_mask=getattr(ret, "attention_mask", None),
        )
        mrope_positions = mrope_positions.squeeze(1)

        mm_inputs = {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
            "im_token_id": self.mm_tokens.image_token_id,
            "video_token_id": self.mm_tokens.video_token_id,
            "mrope_positions": mrope_positions,
            "mrope_position_delta": mrope_position_delta,
        }
        return mm_inputs
