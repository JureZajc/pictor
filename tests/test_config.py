import json

from pictor.config import Config


def test_painting_is_off_until_the_user_turns_it_on():
    config = Config()
    assert config.image_generator == "off"
    assert (config.image_model, config.image_quantize, config.image_steps) == (
        "flux2-klein-4b",
        8,
        4,
    )
    assert config.image_max_side == 1024 and config.image_small_text_encoder is False


def test_settings_survive_a_save_and_unknown_keys_are_ignored():
    Config(model="gemma4:12b", image_generator="mflux", image_quantize=4).save()
    raw = json.loads(Config.path().read_text())
    raw["from_a_newer_version"] = True
    Config.path().write_text(json.dumps(raw))
    loaded = Config.load()
    assert (loaded.model, loaded.image_generator, loaded.image_quantize) == (
        "gemma4:12b",
        "mflux",
        4,
    )


def test_a_hand_edited_file_cannot_break_the_image_generator():
    Config.path().write_text(
        json.dumps(
            {
                "image_generator": "dall-e",
                "image_model": "  ",
                "image_quantize": 7,
                "image_steps": "many",
                "image_max_side": 99999,
                "image_small_text_encoder": "yes",
                "think": "sometimes",
            }
        )
    )
    loaded = Config.load()
    assert loaded == Config()  # every bad value fell back to its default

    Config.path().write_text(json.dumps({"image_steps": True, "image_max_side": 768}))
    assert Config.load().image_steps == 4 and Config.load().image_max_side == 768
