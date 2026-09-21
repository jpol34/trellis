def test_settings_import():
    from trellis.settings import settings

    assert settings.trellis_device == "cpu"
