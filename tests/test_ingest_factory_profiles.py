from src.ingest.factory import _organizer_compatible_enabled


def _config(profile: str) -> dict:
    return {
        "profile": profile,
        "models": {
            "organizer_compatible_visual": {
                "enabled": True,
                "enabled_profiles": ["accelerated"],
            }
        },
    }


def test_organizer_compatible_channel_isolated_from_laptop_default() -> None:
    assert not _organizer_compatible_enabled(_config("laptop"))
    assert _organizer_compatible_enabled(_config("accelerated"))
    assert _organizer_compatible_enabled(_config("laptop"), explicit=True)
