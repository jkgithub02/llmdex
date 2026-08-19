"""R2.2 - parameter counts read from the safetensors headers.

The Hub's own total is a sum of stored element counts, which is not a parameter
count for any checkpoint that packs weights below one byte: NVIDIA's NVFP4
Nemotron reports 17.82B for a model the card calls 30B. `derive.param_counts`
refuses that number, correctly, and until now that was the end of it.

The headers say enough to count properly, because they name every tensor and
give its dtype and shape. Three facts come out of that which a summed total
cannot carry: a scale tensor is not a parameter, a 4-bit weight is stored two to
a byte (provably -- the stored last dimension is exactly half the logical one),
and a routed expert bank is separable from the rest, so active parameters are
counted rather than modelled with an assumed FFN shape.
"""

import json
import struct

import pytest

from app.features.models.tensors import count_parameters, parse_header


def tensor(dtype: str, shape: list[int]) -> dict:
    return {"dtype": dtype, "shape": shape, "data_offsets": [0, 0]}


class TestCountParameters:
    def test_no_headers_is_no_number(self):
        assert count_parameters([], bits=None, experts_per_tok=None).total is None

    def test_unpacks_a_four_bit_container_and_skips_its_scales(self):
        # The proof this is right is in the shapes: the stored tensor is
        # [10, 5] where the model's own dims make it [10, 10].
        header = {
            "backbone.embeddings.weight": tensor("BF16", [100, 10]),
            "backbone.layers.0.mixer.experts.0.up_proj.weight": tensor("U8", [10, 5]),
            "backbone.layers.0.mixer.experts.0.up_proj.weight_scale": tensor("F8_E4M3", [10, 1]),
            "backbone.layers.0.mixer.experts.0.up_proj.weight_scale_2": tensor("F32", []),
            "backbone.layers.0.mixer.in_proj.input_scale": tensor("F32", [1]),
            "__metadata__": {"format": "pt"},
        }

        tally = count_parameters([header], bits=4, experts_per_tok=None)

        assert tally.total == 1000 + 100
        assert tally.unreliable_reason is None

    def test_an_eight_bit_checkpoint_is_counted_one_to_a_byte(self):
        header = {"model.layers.0.mlp.up_proj.weight": tensor("I8", [10, 10])}

        assert count_parameters([header], bits=None, experts_per_tok=None).total == 100

    def test_holds_a_declared_draft_head_out_of_the_total(self):
        # R2.2 - the speculative-decoding head ships in the same files and is not
        # part of the model you prompt. Counted, named, and not added in.
        header = {
            "backbone.layers.0.mixer.up_proj.weight": tensor("BF16", [10, 10]),
            "mtp.layers.0.mixer.up_proj.weight": tensor("BF16", [10, 5]),
        }

        tally = count_parameters([header], bits=None, experts_per_tok=None, has_draft_head=True)

        assert tally.total == 100
        assert tally.auxiliary == 50
        assert tally.auxiliary_module == "mtp"

    def test_counts_a_draft_head_in_when_the_config_declares_none(self):
        # Without `num_nextn_predict_layers` there is no evidence the module is a
        # draft head, and dropping weights on a name alone would be a guess.
        header = {
            "backbone.layers.0.mixer.up_proj.weight": tensor("BF16", [10, 10]),
            "mtp.layers.0.mixer.up_proj.weight": tensor("BF16", [10, 5]),
        }

        tally = count_parameters([header], bits=None, experts_per_tok=None, has_draft_head=False)

        assert tally.total == 150
        assert tally.auxiliary is None

    def test_active_comes_from_the_expert_tensors_not_an_assumed_ffn(self):
        # Nemotron's experts are two matrices, not the three `param_counts`
        # assumes. Counting them removes the assumption entirely.
        header = {"backbone.embeddings.weight": tensor("BF16", [5, 10])}
        for index in range(4):
            header[f"backbone.layers.1.mixer.experts.{index}.up_proj.weight"] = tensor(
                "BF16", [10, 5]
            )
            header[f"backbone.layers.1.mixer.experts.{index}.down_proj.weight"] = tensor(
                "BF16", [10, 5]
            )

        tally = count_parameters([header], bits=None, experts_per_tok=1)

        assert tally.total == 50 + 400
        # Three of four experts stay dormant.
        assert tally.active == 450 - 300

    def test_a_shared_expert_is_never_dormant(self):
        header = {
            "backbone.layers.1.mixer.shared_experts.up_proj.weight": tensor("BF16", [10, 10]),
            "backbone.layers.1.mixer.experts.0.up_proj.weight": tensor("BF16", [10, 10]),
            "backbone.layers.1.mixer.experts.1.up_proj.weight": tensor("BF16", [10, 10]),
        }

        tally = count_parameters([header], bits=None, experts_per_tok=1)

        assert tally.total == 300
        assert tally.active == 200

    def test_refuses_active_when_the_experts_are_not_the_same_size(self):
        # R2.7 - unequal experts mean the tensors do not describe one expert bank,
        # so no active count is reported and the reason says which model it was.
        header = {
            "backbone.layers.1.mixer.experts.0.up_proj.weight": tensor("BF16", [10, 10]),
            "backbone.layers.1.mixer.experts.1.up_proj.weight": tensor("BF16", [10, 5]),
        }

        tally = count_parameters([header], bits=None, experts_per_tok=1)

        assert tally.total == 150
        assert tally.active is None
        assert "same size" in tally.unreliable_reason

    def test_no_active_count_without_a_stated_experts_per_tok(self):
        header = {"backbone.layers.1.mixer.experts.0.up_proj.weight": tensor("BF16", [10, 10])}

        assert count_parameters([header], bits=None, experts_per_tok=None).active is None

    def test_a_dense_model_is_asked_no_active_question(self):
        header = {"model.layers.0.mlp.up_proj.weight": tensor("BF16", [10, 10])}

        tally = count_parameters([header], bits=None, experts_per_tok=None)

        assert tally.total == 100
        assert tally.active is None
        assert tally.unreliable_reason is None

    def test_refuses_active_when_the_config_routes_experts_the_files_do_not_show(self):
        """A fused expert bank has no `experts.<n>.` tensors. Answering "all of
        them fire" would be a plausible wrong number for an MoE."""
        header = {"model.layers.0.mlp.gate_up_proj": tensor("BF16", [128, 10, 10])}

        tally = count_parameters([header], bits=None, experts_per_tok=8)

        assert tally.total == 12800
        assert tally.active is None
        assert "no per-expert tensors" in tally.unreliable_reason


class TestParseHeader:
    def test_reads_the_length_prefixed_json_a_safetensors_file_opens_with(self):
        body = json.dumps({"a.weight": tensor("BF16", [2, 3])}).encode()
        blob = struct.pack("<Q", len(body)) + body

        assert parse_header(blob)["a.weight"]["shape"] == [2, 3]

    def test_refuses_a_truncated_header_rather_than_counting_half_of_it(self):
        body = json.dumps({"a.weight": tensor("BF16", [2, 3])}).encode()
        blob = struct.pack("<Q", len(body) + 500) + body

        with pytest.raises(ValueError, match="truncated"):
            parse_header(blob)
