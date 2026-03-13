from jax_plugins.mps.sysinfo import format_info, get_info


def test_get_info():
    info = get_info()
    expected_keys = {
        "jax",
        "jaxlib",
        "jax-mps",
        "Python",
        "macOS",
        "Chip",
        "Memory",
        "Backend",
    }
    assert set(info.keys()) == expected_keys
    # All values should be non-empty strings
    for key, value in info.items():
        assert isinstance(value, str), f"{key} is not a string"
        assert value, f"{key} is empty"


def test_format_info():
    output = format_info()
    lines = output.strip().split("\n")
    assert len(lines) == 8
    for line in lines:
        assert ":" in line
