# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

import os

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import hydra

from verl.trainer.sft_trainer import run_sft


def _apply_qwen3_vl_freeze_patch():
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

    original_build_module = FSDPEngine._build_module

    def _build_module_with_freeze(self):
        module = original_build_module(self)

        if hasattr(module, "visual"):
            # Freeze visual backbone blocks
            if hasattr(module.visual, "blocks"):
                for p in module.visual.blocks.parameters():
                    p.requires_grad = False

            # Keep DeepStack merger trainable
            if hasattr(module.visual, "deepstack_merger_list"):
                for p in module.visual.deepstack_merger_list.parameters():
                    p.requires_grad = True

            # Keep visual projection merger trainable
            if hasattr(module.visual, "merger"):
                for p in module.visual.merger.parameters():
                    p.requires_grad = True

        # Freeze text embeddings
        if hasattr(module, "get_input_embeddings"):
            for p in module.get_input_embeddings().parameters():
                p.requires_grad = False

        def _count_trainable(mod):
            return sum(p.numel() for p in mod.parameters() if p.requires_grad)

        if hasattr(module, "visual"):
            if hasattr(module.visual, "blocks"):
                print("[freeze_check] visual.blocks trainable:", _count_trainable(module.visual.blocks))
            if hasattr(module.visual, "deepstack_merger_list"):
                print(
                    "[freeze_check] visual.deepstack_merger_list trainable:",
                    _count_trainable(module.visual.deepstack_merger_list),
                )
            if hasattr(module.visual, "merger"):
                print("[freeze_check] visual.merger trainable:", _count_trainable(module.visual.merger))

        if hasattr(module, "get_input_embeddings"):
            print("[freeze_check] text embeddings trainable:", _count_trainable(module.get_input_embeddings()))

        print("[freeze_check] total trainable:", _count_trainable(module))

        return module

    FSDPEngine._build_module = _build_module_with_freeze


@hydra.main(config_path="config", config_name="sft_trainer_engine", version_base=None)
def main(config):
    _apply_qwen3_vl_freeze_patch()
    run_sft(config)


if __name__ == "__main__":
    main()
